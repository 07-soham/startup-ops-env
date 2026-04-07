"""
inference.py — OpenEnv-compliant inference script for StartupOpsEnv.

Usage:
    python inference.py                    # Run with default config
    python inference.py --seed 42          # Run with specific seed
    python inference.py --scenario investor_pressure  # Run specific scenario

Output Format:
    [START]
    {"episode": 0, "step": 0, ...}
    [STEP]
    {"step": 1, "action": "...", "observation": {...}, "reward": 0.0, "done": false}
    [STEP]
    ...
    [END]
    {"total_reward": ..., "email_score": ..., "task_score": ..., "negotiation_score": ..., "overall_score": ...}
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict

from agents.baseline import BaselineAgent
from env.core import StartupOpsEnv
from env.grader import grade_episode


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run StartupOpsEnv inference for OpenEnv validation."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=int(os.environ.get("SEED", 42)),
        help="Random seed for determinism (default: from SEED env var or 42)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=int(os.environ.get("MAX_STEPS", 50)),
        help="Maximum steps to run (default: from MAX_STEPS env var or 50)",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default=os.environ.get("SCENARIO", "investor_pressure"),
        help="Scenario to run (default: from SCENARIO env var or investor_pressure)",
    )
    parser.add_argument(
        "--api-base-url",
        type=str,
        default=os.environ.get("API_BASE_URL", ""),
        help="Base URL for API (optional)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=os.environ.get("MODEL_NAME", ""),
        help="Model name (optional)",
    )
    return parser.parse_args()


def format_observation(obs) -> Dict[str, Any]:
    """Convert observation to serializable dict."""
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


def run_inference(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run inference with the given configuration.

    Returns:
        Dict with scores and results.
    """
    # Create environment
    env = StartupOpsEnv(config)
    agent = BaselineAgent()

    # Reset environment
    obs = env.reset()
    done = False
    total_reward = 0.0
    step_count = 0

    # Print START marker with initial state
    print("[START]")
    print(json.dumps({
        "episode": 0,
        "step": 0,
        "observation": format_observation(obs),
    }))
    sys.stdout.flush()

    # Run episode
    while not done:
        # Agent selects action
        action = agent.act(obs)

        # Environment steps
        obs, reward, done, info = env.step(action)
        total_reward += reward
        step_count += 1

        # Print STEP marker with step data
        print("[STEP]")
        print(json.dumps({
            "step": step_count,
            "action": action,
            "observation": format_observation(obs),
            "reward": reward,
            "done": done,
            "info": info,
        }))
        sys.stdout.flush()

    # Get final grades
    totals = env.get_totals()
    grades = grade_episode(
        logs=env.logs,
        total_emails_created=totals["total_emails_created"],
        total_tasks_created=totals["total_tasks_created"],
        total_negotiations_created=totals["total_negotiations_created"],
        state=env.state,
    )

    # Print END marker with final results
    print("[END]")
    print(json.dumps({
        "total_reward": total_reward,
        "email_score": grades["email_score"],
        "task_score": grades["task_score"],
        "negotiation_score": grades["negotiation_score"],
        "overall_score": grades["overall_score"],
    }))
    sys.stdout.flush()

    return {
        "total_reward": total_reward,
        "email_score": grades["email_score"],
        "task_score": grades["task_score"],
        "negotiation_score": grades["negotiation_score"],
        "overall_score": grades["overall_score"],
        "steps": step_count,
    }


def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Build configuration
    config = {
        "seed": args.seed,
        "max_steps": args.steps,
        "initial_budget": 100_000.0,
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
        "scenario": args.scenario,
        "mode": "auto",
    }

    # Optional: Store API config for reference
    if args.api_base_url:
        config["api_base_url"] = args.api_base_url
    if args.model_name:
        config["model_name"] = args.model_name

    try:
        result = run_inference(config)
        return 0
    except Exception as e:
        print(f"Error during inference: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
