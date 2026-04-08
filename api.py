"""
FastAPI backend for StartupOps AI Simulator.

Provides REST endpoints for:
- Running simulations (auto/manual)
- Parsing emails with LLM
- Managing scenarios
"""
from __future__ import annotations

import json
import traceback
from typing import Any, Dict, List, Optional

from enum import Enum

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from env.core import StartupOpsEnv
from env.generator import ManualInputState
from env.grader import grade_episode
from env.llm_parser import parse_email
from env.models import Email, ParsedEmail, Sentiment, Urgency
from env.scenarios import get_scenario, list_scenarios
from agents.baseline import BaselineAgent


# -----------------------------------------------------------------------------
# FastAPI App
# -----------------------------------------------------------------------------
app = FastAPI(
    title="StartupOps AI Simulator API",
    description="AI-powered startup operations simulation with LLM parsing",
    version="1.0.0",
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------------------------------------------------------
# Pydantic Models for API
# -----------------------------------------------------------------------------
class EmailInput(BaseModel):
    text: str
    sender: str = "unknown@email.com"
    subject: str = "No subject"
    thread_id: str = "default"


class TaskInput(BaseModel):
    name: str
    hours_required: float = 8.0
    deadline: int = 5
    priority: str = "medium"
    effort: int = 3
    impact: float = 1.0


class SimulationRequest(BaseModel):
    mode: str = Field(..., description="'auto' or 'manual'")
    scenario: Optional[str] = Field(None, description="Scenario name for auto mode")
    emails: Optional[List[EmailInput]] = Field(None, description="Manual emails")
    tasks: Optional[List[TaskInput]] = Field(None, description="Manual tasks")
    seed: int = Field(42, description="Random seed for determinism")
    max_steps: int = Field(50, description="Maximum simulation steps")


class ParsedEmailOutput(BaseModel):
    text: str
    urgency: str
    sentiment: str
    requires_action: bool
    thread_id: str
    escalation_level: int


class SimulationOutput(BaseModel):
    summary: str
    emails: List[ParsedEmailOutput]
    actions: List[Dict[str, Any]]
    result: str
    confidence: str
    total_reward: float
    scores: Dict[str, float]


class ScenarioInfo(BaseModel):
    name: str
    description: str
    email_count: int
    task_count: int
    negotiation_count: int


class ActionType(str, Enum):
    """Valid action types for the environment."""
    reply_email = "reply_email"
    ignore_email = "ignore_email"
    assign_task = "assign_task"
    accept_offer = "accept_offer"
    reject_offer = "reject_offer"
    negotiate = "negotiate"
    wait = "wait"


class StepRequest(BaseModel):
    """Request model for /step endpoint."""
    action: str = Field(..., description="Action type to perform")
    target_id: Optional[str] = Field(None, description="Target ID for the action (if applicable)")


class StepResponse(BaseModel):
    """Response model for /step endpoint."""
    observation: Dict[str, Any]
    reward: float
    done: bool
    info: Dict[str, Any]


class StateResponse(BaseModel):
    """Response model for /state endpoint."""
    budget: float
    satisfaction: float
    team_hours: float
    revenue: float
    num_emails: int
    num_tasks: int
    num_negotiations: int
    missed_tasks: int
    step: int
    email_ids: List[str]
    unassigned_task_ids: List[str]
    negotiation_ids: List[str]
    high_urgency_emails: List[str]
    overdue_tasks: List[str]
    task_deadlines: Dict[str, int]
    negotiation_offers: Dict[str, float]
    email_urgencies: Dict[str, str]


class ResetResponse(BaseModel):
    """Response model for /reset endpoint."""
    observation: Dict[str, Any]
    info: Dict[str, Any]


# -----------------------------------------------------------------------------
# Base Config
# -----------------------------------------------------------------------------
def get_base_config(seed: int = 42, max_steps: int = 50) -> Dict[str, Any]:
    """Get base environment configuration."""
    return {
        "seed": seed,
        "max_steps": max_steps,
        "initial_budget": 100_000,
        "initial_satisfaction": 0.8,
        "initial_team_hours": 100.0,
        "email_gen_prob": 0.0,
        "task_gen_prob": 0.0,
        "negotiation_gen_prob": 0.0,
        "max_emails": 20,
        "max_tasks": 20,
        "max_negotiations": 10,
        "reply_satisfaction_boost": 0.1,
        "high_urgency_ignore_penalty": -2.0,
        "accept_offer_budget_cost": 1000.0,
        "negotiate_adjustment": 0.9,
        "task_miss_penalty": -5.0,
        "step_survival_reward": 0.1,
    }


# -----------------------------------------------------------------------------
# API Endpoints
# -----------------------------------------------------------------------------

@app.get("/")
async def root():
    """API health check."""
    return {"status": "ok", "service": "StartupOps AI Simulator"}


@app.get("/scenarios", response_model=List[ScenarioInfo])
async def get_scenarios():
    """List all available scenarios."""
    scenarios = []
    for name in list_scenarios():
        scenario = get_scenario(name)
        scenarios.append(ScenarioInfo(
            name=name,
            description=scenario.get("description", ""),
            email_count=len(scenario.get("emails", [])),
            task_count=len(scenario.get("tasks", [])),
            negotiation_count=len(scenario.get("negotiations", [])),
        ))
    return scenarios


class ParseEmailRequest(BaseModel):
    text: str
    sender: str = "unknown@email.com"
    subject: str = "No subject"
    context: Optional[List[str]] = None


@app.post("/parse-email")
async def parse_single_email(request: ParseEmailRequest):
    """Parse a single email with LLM (or fallback)."""
    try:
        result: ParsedEmail = parse_email(request.text, request.context or [])

        return {
            "urgency": result.urgency.value,
            "sentiment": result.sentiment.value,
            "requires_action": result.requires_action,
            "confidence": result.confidence,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Parse error: {str(e)}")


@app.post("/run-simulation", response_model=SimulationOutput)
async def run_simulation(request: SimulationRequest):
    """
    Run a simulation (auto or manual mode).

    Auto mode: Uses predefined scenario
    Manual mode: Uses user-provided emails/tasks
    """
    try:
        config = get_base_config(request.seed, request.max_steps)

        # Setup based on mode
        if request.mode == "auto":
            if not request.scenario:
                raise HTTPException(status_code=400, detail="Scenario required for auto mode")
            config["scenario"] = request.scenario
            config["mode"] = "auto"
        elif request.mode == "manual":
            config["mode"] = "manual"

            # Convert manual inputs
            manual_emails = []
            if request.emails:
                for i, email_input in enumerate(request.emails):
                    manual_emails.append({
                        "text": email_input.text,
                        "sender": email_input.sender,
                        "subject": email_input.subject,
                        "thread_id": email_input.thread_id,
                        "timestamp": i,
                    })

            manual_tasks = []
            if request.tasks:
                for task_input in request.tasks:
                    manual_tasks.append({
                        "name": task_input.name,
                        "hours_required": task_input.hours_required,
                        "deadline": task_input.deadline,
                        "priority": task_input.priority,
                        "effort": task_input.effort,
                        "impact": task_input.impact,
                    })

            config["manual_inputs"] = ManualInputState(
                emails=manual_emails,
                tasks=manual_tasks,
            )
        else:
            raise HTTPException(status_code=400, detail="Mode must be 'auto' or 'manual'")

        # Create and run environment
        env = StartupOpsEnv(config)
        agent = BaselineAgent()

        obs = env.reset()
        done = False
        actions_taken = []

        while not done:
            action = agent.act(obs)
            obs, reward, done, info = env.step(action)
            actions_taken.append({
                "step": len(actions_taken) + 1,
                "action": action,
                "reward": reward,
            })

        # Grade results
        totals = env.get_totals()
        grades = grade_episode(
            logs=env.logs,
            total_emails_created=totals["total_emails_created"],
            total_tasks_created=totals["total_tasks_created"],
            total_negotiations_created=totals["total_negotiations_created"],
            state=env.state,
        )

        # Build parsed email outputs
        parsed_emails = []
        for email in env.state.emails:
            parsed_emails.append(ParsedEmailOutput(
                text=email.text,
                urgency=email.urgency.value,
                sentiment=email.sentiment.value,
                requires_action=email.requires_action,
                thread_id=email.thread_id,
                escalation_level=email.escalation_level,
            ))

        # Calculate confidence based on parser used
        confidence = "High" if grades["overall_score"] > 0.7 else "Medium" if grades["overall_score"] > 0.4 else "Low"

        return SimulationOutput(
            summary=grades["summary"],
            emails=parsed_emails,
            actions=actions_taken,
            result="completed",
            confidence=confidence,
            total_reward=grades["total_reward"],
            scores={
                "email_score": grades["email_score"],
                "task_score": grades["task_score"],
                "negotiation_score": grades["negotiation_score"],
                "overall_score": grades["overall_score"],
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        traceback_str = traceback.format_exc()
        raise HTTPException(status_code=500, detail=f"Simulation error: {str(e)}\n{traceback_str}")


# -----------------------------------------------------------------------------
# OpenEnv Standard Endpoints (/step, /reset, /state)
# -----------------------------------------------------------------------------

# Global environment instance for OpenEnv endpoints
_env_instance: Optional[StartupOpsEnv] = None


def get_env() -> StartupOpsEnv:
    """Get or create the environment instance."""
    global _env_instance
    if _env_instance is None:
        config = get_base_config()
        config["scenario"] = "investor_pressure"
        config["mode"] = "auto"
        _env_instance = StartupOpsEnv(config)
    return _env_instance


def obs_to_dict(obs) -> Dict[str, Any]:
    """Convert observation to dictionary."""
    return {
        "budget": obs.budget,
        "satisfaction": obs.satisfaction,
        "team_hours": obs.team_hours,
        "revenue": obs.revenue,
        "num_emails": obs.num_emails,
        "num_tasks": obs.num_tasks,
        "num_negotiations": obs.num_negotiations,
        "missed_tasks": obs.missed_tasks,
        "step": obs.step,
        "email_ids": obs.email_ids,
        "unassigned_task_ids": obs.unassigned_task_ids,
        "negotiation_ids": obs.negotiation_ids,
        "high_urgency_emails": obs.high_urgency_emails,
        "overdue_tasks": obs.overdue_tasks,
        "task_deadlines": obs.task_deadlines,
        "negotiation_offers": obs.negotiation_offers,
        "email_urgencies": obs.email_urgencies,
    }


class EnvConfigRequest(BaseModel):
    """Configuration for environment initialization."""
    seed: int = Field(42, description="Random seed for determinism")
    max_steps: int = Field(50, description="Maximum steps per episode")
    scenario: Optional[str] = Field(None, description="Scenario name to load")


@app.post("/reset", response_model=ResetResponse)
async def reset_endpoint(config: Optional[EnvConfigRequest] = None):
    """
    Reset the environment to initial state.

    Returns initial observation and info dict.
    """
    global _env_instance

    try:
        base_config = get_base_config()

        # Set defaults for OpenEnv compatibility
        base_config["seed"] = 42
        base_config["max_steps"] = 50
        base_config["scenario"] = "investor_pressure"
        base_config["mode"] = "auto"

        if config:
            base_config["seed"] = config.seed
            base_config["max_steps"] = config.max_steps
            if config.scenario:
                base_config["scenario"] = config.scenario

        _env_instance = StartupOpsEnv(base_config)
        obs = _env_instance.reset()

        return ResetResponse(
            observation=obs_to_dict(obs),
            info={"seed": base_config["seed"], "max_steps": base_config["max_steps"], "scenario": base_config["scenario"]}
        )
    except Exception as e:
        import traceback
        traceback_str = traceback.format_exc()
        raise HTTPException(status_code=500, detail=f"Reset error: {str(e)}\n{traceback_str}")


@app.post("/step", response_model=StepResponse)
async def step_endpoint(request: StepRequest):
    """
    Execute one step in the environment.

    Takes an action and returns observation, reward, done flag, and info.
    """
    env = get_env()

    try:
        action = {"type": request.action}
        if request.target_id:
            action["target_id"] = request.target_id

        obs, reward, done, info = env.step(action)

        return StepResponse(
            observation=obs_to_dict(obs),
            reward=reward,
            done=done,
            info=info
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Step error: {str(e)}")


@app.get("/state", response_model=StateResponse)
async def state_endpoint():
    """
    Get current environment state as observation.

    Returns the current observation without taking any action.
    """
    env = get_env()

    try:
        obs = env._get_obs()
        return StateResponse(
            budget=obs.budget,
            satisfaction=obs.satisfaction,
            team_hours=obs.team_hours,
            revenue=obs.revenue,
            num_emails=obs.num_emails,
            num_tasks=obs.num_tasks,
            num_negotiations=obs.num_negotiations,
            missed_tasks=obs.missed_tasks,
            step=obs.step,
            email_ids=obs.email_ids,
            unassigned_task_ids=obs.unassigned_task_ids,
            negotiation_ids=obs.negotiation_ids,
            high_urgency_emails=obs.high_urgency_emails,
            overdue_tasks=obs.overdue_tasks,
            task_deadlines=obs.task_deadlines,
            negotiation_offers=obs.negotiation_offers,
            email_urgencies=obs.email_urgencies,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"State error: {str(e)}")


# -----------------------------------------------------------------------------
# Run Server
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
