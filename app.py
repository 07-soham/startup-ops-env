"""
StartupOps AI Simulator - Gradio UI

Interactive web interface for running simulations with:
- Scenario selection
- Manual mode with custom emails/tasks
- Real-time step visualization
- Episode grading and metrics
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr

from agents.baseline import BaselineAgent
from env.core import StartupOpsEnv
from env.generator import ManualInputState
from env.grader import grade_episode
from env.scenarios import get_scenario, list_scenarios


def run_simulation_auto(scenario: str, seed: int, max_steps: int) -> Tuple[str, str, str]:
    """Run auto mode simulation."""
    config = {
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
        "scenario": scenario,
        "mode": "auto",
    }

    env = StartupOpsEnv(config)
    agent = BaselineAgent()

    obs = env.reset()
    done = False
    logs = []

    while not done:
        action = agent.act(obs)
        obs, reward, done, info = env.step(action)
        logs.append({
            "step": len(logs) + 1,
            "action": action,
            "reward": reward,
            "budget": obs.budget,
            "satisfaction": obs.satisfaction,
        })

    totals = env.get_totals()
    grades = grade_episode(
        logs=env.logs,
        total_emails_created=totals["total_emails_created"],
        total_tasks_created=totals["total_tasks_created"],
        total_negotiations_created=totals["total_negotiations_created"],
        state=env.state,
    )

    summary = grades["summary"]
    logs_text = json.dumps(logs, indent=2)
    scores = json.dumps(grades, indent=2)

    return summary, logs_text, scores


def run_simulation_manual(
    emails_text: str,
    tasks_text: str,
    seed: int,
    max_steps: int,
) -> Tuple[str, str, str]:
    """Run manual mode simulation."""
    # Parse manual inputs
    manual_emails = []
    manual_tasks = []

    if emails_text.strip():
        for i, line in enumerate(emails_text.strip().split("\n")):
            if line.strip():
                manual_emails.append({
                    "text": line.strip(),
                    "sender": "user@manual.com",
                    "subject": f"Manual Email {i+1}",
                    "thread_id": f"manual_{i}",
                    "timestamp": i,
                })

    if tasks_text.strip():
        for i, line in enumerate(tasks_text.strip().split("\n")):
            if line.strip():
                parts = line.strip().split(",")
                name = parts[0].strip() if parts else f"Task {i+1}"
                hours = float(parts[1].strip()) if len(parts) > 1 else 8.0
                deadline = int(parts[2].strip()) if len(parts) > 2 else 5
                priority = parts[3].strip() if len(parts) > 3 else "medium"

                manual_tasks.append({
                    "name": name,
                    "hours_required": hours,
                    "deadline": deadline,
                    "priority": priority,
                    "effort": 3,
                    "impact": 1.0,
                })

    config = {
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
        "mode": "manual",
        "manual_inputs": ManualInputState(emails=manual_emails, tasks=manual_tasks),
    }

    env = StartupOpsEnv(config)
    agent = BaselineAgent()

    obs = env.reset()
    done = False
    logs = []

    while not done:
        action = agent.act(obs)
        obs, reward, done, info = env.step(action)
        logs.append({
            "step": len(logs) + 1,
            "action": action,
            "reward": reward,
            "budget": obs.budget,
            "satisfaction": obs.satisfaction,
        })

    totals = env.get_totals()
    grades = grade_episode(
        logs=env.logs,
        total_emails_created=totals["total_emails_created"],
        total_tasks_created=totals["total_tasks_created"],
        total_negotiations_created=totals["total_negotiations_created"],
        state=env.state,
    )

    summary = grades["summary"]
    logs_text = json.dumps(logs, indent=2)
    scores = json.dumps(grades, indent=2)

    return summary, logs_text, scores


def get_scenario_info(scenario: str) -> str:
    """Get scenario description."""
    s = get_scenario(scenario)
    return f"""
**{s['name']}**: {s['description']}

- Emails: {len(s.get('emails', []))}
- Tasks: {len(s.get('tasks', []))}
- Negotiations: {len(s.get('negotiations', []))}
"""


# Build the UI
with gr.Blocks(title="StartupOps AI Simulator") as demo:
    gr.Markdown("""
    # StartupOps AI Simulator

    A deterministic RL environment simulating startup operations with AI-powered email parsing.
    """)

    with gr.Tab("Auto Mode (Scenarios)"):
        gr.Markdown("Run a predefined scenario with the baseline agent.")

        scenarios = list_scenarios()

        with gr.Row():
            with gr.Column(scale=1):
                scenario_dropdown = gr.Dropdown(
                    choices=scenarios,
                    value=scenarios[0] if scenarios else None,
                    label="Select Scenario",
                )
                scenario_info = gr.Markdown()

                seed_auto = gr.Number(value=42, label="Seed", precision=0)
                steps_auto = gr.Slider(10, 100, value=50, step=10, label="Max Steps")

                run_auto_btn = gr.Button("Run Simulation", variant="primary")

            with gr.Column(scale=2):
                with gr.Tab("Summary"):
                    auto_summary = gr.Textbox(label="Results", lines=15)
                with gr.Tab("Step Logs"):
                    auto_logs = gr.Textbox(label="Logs", lines=20)
                with gr.Tab("Scores"):
                    auto_scores = gr.Textbox(label="Detailed Scores", lines=20)

        scenario_dropdown.change(fn=get_scenario_info, inputs=scenario_dropdown, outputs=scenario_info)
        run_auto_btn.click(
            fn=run_simulation_auto,
            inputs=[scenario_dropdown, seed_auto, steps_auto],
            outputs=[auto_summary, auto_logs, auto_scores],
        )

    with gr.Tab("Manual Mode"):
        gr.Markdown("Provide your own emails and tasks.")

        with gr.Row():
            with gr.Column(scale=1):
                manual_emails = gr.Textbox(
                    label="Emails (one per line)",
                    lines=5,
                    placeholder="We need this ASAP!\nFollowing up on my previous email...",
                )
                manual_tasks = gr.Textbox(
                    label="Tasks (format: name, hours, deadline, priority)",
                    lines=5,
                    placeholder="Fix critical bug, 4, 2, high\nUpdate documentation, 2, 5, medium",
                )

                seed_manual = gr.Number(value=42, label="Seed")
                steps_manual = gr.Slider(10, 100, value=50, step=10, label="Max Steps")

                run_manual_btn = gr.Button("Run Simulation", variant="primary")

            with gr.Column(scale=2):
                gr.Markdown("### Results")
                manual_summary = gr.Textbox(label="Summary", lines=10)
                manual_logs = gr.Textbox(label="Step Logs", lines=10)
                manual_scores = gr.Textbox(label="Detailed Scores", lines=10)

        run_manual_btn.click(
            fn=run_simulation_manual,
            inputs=[manual_emails, manual_tasks, seed_manual, steps_manual],
            outputs=[manual_summary, manual_logs, manual_scores],
        )

    with gr.Tab("About"):
        gr.Markdown("""
        ## Actions

        | Action | Target | Effect |
        |--------|--------|--------|
        | reply_email | email_N | Removes email, boosts satisfaction |
        | ignore_email | email_N | Penalty if urgency=high |
        | assign_task | task_N | Consumes team_hours, marks task done |
        | accept_offer | negotiation_N | Spends budget cost, adds revenue |
        | reject_offer | negotiation_N | Discards negotiation |
        | negotiate | negotiation_N | Lowers offer price by 10% |
        | wait | — | No-op; dynamics still advance |

        ## Scoring

        - **Email Score**: % of emails replied to
        - **Task Score**: % of tasks not missed
        - **Negotiation Score**: % of handled negotiations accepted
        - **Overall Score**: Weighted 30% email / 40% task / 30% negotiation
        """)

    # Initialize scenario info
    if scenarios:
        demo.load(
            fn=lambda: get_scenario_info(scenarios[0]),
            outputs=scenario_info,
        )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
