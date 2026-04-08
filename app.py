"""
StartupOps AI Simulator — Unified Gradio UI + FastAPI backend.

Serves:
  • Gradio interactive simulation UI at  /
  • OpenEnv REST API endpoints at        /reset  /step  /state  etc.
  • FastAPI docs at                       /docs

Run:
    python app.py
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr

from agents.baseline import BaselineAgent
from env.core import StartupOpsEnv
from env.generator import ManualInputState
from env.grader import grade_episode
from env.scenarios import get_scenario, list_scenarios

# ---------------------------------------------------------------------------
# Import the FastAPI app so its OpenEnv endpoints are mounted alongside Gradio
# ---------------------------------------------------------------------------
from api import app as fastapi_app  # noqa: F401  (side-effect: registers routes)


# ---------------------------------------------------------------------------
# Simulation helpers
# ---------------------------------------------------------------------------

def _base_config(seed: int = 42, max_steps: int = 50) -> Dict[str, Any]:
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


def _run_episode(env: StartupOpsEnv) -> Tuple[str, str, str]:
    """Run a full episode with the baseline agent, return (summary, logs, scores)."""
    agent = BaselineAgent()
    obs = env.reset()
    done = False
    step_logs: List[Dict] = []

    while not done:
        action = agent.act(obs)
        obs, reward, done, info = env.step(action)
        step_logs.append({
            "step": len(step_logs) + 1,
            "action": action,
            "reward": round(reward, 4),
            "budget": round(obs.budget, 2),
            "satisfaction": round(obs.satisfaction, 4),
            "info": info,
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
    logs_text = json.dumps(step_logs, indent=2)
    scores_text = json.dumps(
        {
            "email_score": round(grades["email_score"], 4),
            "task_score": round(grades["task_score"], 4),
            "negotiation_score": round(grades["negotiation_score"], 4),
            "overall_score": round(grades["overall_score"], 4),
            "total_reward": round(grades["total_reward"], 4),
        },
        indent=2,
    )

    return summary, logs_text, scores_text


def run_simulation_auto(scenario: str, seed: int, max_steps: int) -> Tuple[str, str, str]:
    """Auto mode: run a predefined scenario with the baseline agent."""
    config = _base_config(int(seed), int(max_steps))
    config["scenario"] = scenario
    config["mode"] = "auto"
    env = StartupOpsEnv(config)
    return _run_episode(env)


def run_simulation_manual(
    emails_text: str,
    tasks_text: str,
    seed: int,
    max_steps: int,
) -> Tuple[str, str, str]:
    """Manual mode: user provides raw emails and tasks."""
    manual_emails = []
    manual_tasks = []

    for i, line in enumerate((emails_text or "").strip().splitlines()):
        line = line.strip()
        if line:
            manual_emails.append({
                "text": line,
                "sender": "user@manual.com",
                "subject": f"Email {i + 1}",
                "thread_id": f"manual_{i}",
                "timestamp": i,
            })

    for i, line in enumerate((tasks_text or "").strip().splitlines()):
        parts = [p.strip() for p in line.split(",")]
        if not parts or not parts[0]:
            continue
        manual_tasks.append({
            "name": parts[0],
            "hours_required": float(parts[1]) if len(parts) > 1 else 8.0,
            "deadline": int(parts[2]) if len(parts) > 2 else 5,
            "priority": parts[3] if len(parts) > 3 else "medium",
            "effort": 3,
            "impact": 1.0,
        })

    config = _base_config(int(seed), int(max_steps))
    config["mode"] = "manual"
    config["manual_inputs"] = ManualInputState(emails=manual_emails, tasks=manual_tasks)
    env = StartupOpsEnv(config)
    return _run_episode(env)


def get_scenario_info(scenario: str) -> str:
    s = get_scenario(scenario)
    return (
        f"**{s['name']}** — {s['description']}\n\n"
        f"- 📧 Emails: **{len(s.get('emails', []))}**\n"
        f"- ✅ Tasks: **{len(s.get('tasks', []))}**\n"
        f"- 🤝 Negotiations: **{len(s.get('negotiations', []))}**"
    )


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

THEME = gr.themes.Soft(
    primary_hue="violet",
    secondary_hue="purple",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui"],
)

CSS = """
.gradio-container { max-width: 1100px !important; margin: auto; }
.tab-nav button { font-size: 1rem; font-weight: 600; }
.summary-box textarea { font-size: 0.9rem; line-height: 1.6; }
.score-box textarea { font-family: monospace; font-size: 0.85rem; }
footer { display: none !important; }
"""

scenarios = list_scenarios()

with gr.Blocks(theme=THEME, css=CSS, title="StartupOps AI Simulator") as demo:

    # ---- Header ------------------------------------------------------------
    gr.Markdown(
        """
        # 🚀 StartupOps AI Simulator
        ### AI-powered Reinforcement Learning environment for startup operations
        Manage **emails**, **tasks**, and **deal negotiations** — powered by a deterministic RL engine.
        """
    )

    # ---- Tabs --------------------------------------------------------------
    with gr.Tabs():

        # ── Auto Mode ───────────────────────────────────────────────────────
        with gr.Tab("🤖 Auto Mode"):
            gr.Markdown("Run a **predefined scenario** automatically using the heuristic baseline agent.")

            with gr.Row():
                with gr.Column(scale=1):
                    scenario_dd = gr.Dropdown(
                        choices=scenarios,
                        value=scenarios[0] if scenarios else None,
                        label="📋 Scenario",
                        interactive=True,
                    )
                    scenario_info_md = gr.Markdown(
                        value=get_scenario_info(scenarios[0]) if scenarios else "No scenarios available"
                    )
                    seed_auto = gr.Number(value=42, label="🎲 Seed", precision=0)
                    steps_auto = gr.Slider(10, 100, value=50, step=10, label="⏱ Max Steps")
                    run_auto_btn = gr.Button("▶ Run Simulation", variant="primary", size="lg")

                with gr.Column(scale=2):
                    auto_summary = gr.Textbox(
                        label="📊 Episode Summary",
                        lines=8,
                        elem_classes=["summary-box"],
                        show_copy_button=True,
                    )
                    with gr.Row():
                        auto_scores = gr.Textbox(
                            label="🏆 Scores",
                            lines=9,
                            elem_classes=["score-box"],
                            show_copy_button=True,
                        )
                        auto_logs = gr.Textbox(
                            label="📜 Step Logs",
                            lines=9,
                            elem_classes=["score-box"],
                            show_copy_button=True,
                        )

            scenario_dd.change(fn=get_scenario_info, inputs=scenario_dd, outputs=scenario_info_md)
            run_auto_btn.click(
                fn=run_simulation_auto,
                inputs=[scenario_dd, seed_auto, steps_auto],
                outputs=[auto_summary, auto_logs, auto_scores],
            )

        # ── Manual Mode ─────────────────────────────────────────────────────
        with gr.Tab("✍ Manual Mode"):
            gr.Markdown(
                "Provide your **own emails and tasks** — the baseline agent will handle them.\n\n"
                "Tasks format: `Task name, hours, deadline_steps, priority`"
            )

            with gr.Row():
                with gr.Column(scale=1):
                    manual_emails_box = gr.Textbox(
                        label="📧 Emails (one per line)",
                        lines=5,
                        placeholder="We need this fixed ASAP!\nFollowing up on the proposal...\nInvestor wants a call this week.",
                    )
                    manual_tasks_box = gr.Textbox(
                        label="✅ Tasks (name, hours, deadline, priority)",
                        lines=5,
                        placeholder="Fix critical bug, 4, 2, high\nUpdate investor deck, 3, 5, high\nWrite docs, 2, 8, medium",
                    )
                    seed_manual = gr.Number(value=42, label="🎲 Seed", precision=0)
                    steps_manual = gr.Slider(10, 100, value=50, step=10, label="⏱ Max Steps")
                    run_manual_btn = gr.Button("▶ Run Simulation", variant="primary", size="lg")

                with gr.Column(scale=2):
                    manual_summary = gr.Textbox(
                        label="📊 Episode Summary",
                        lines=8,
                        elem_classes=["summary-box"],
                        show_copy_button=True,
                    )
                    with gr.Row():
                        manual_scores = gr.Textbox(
                            label="🏆 Scores",
                            lines=9,
                            elem_classes=["score-box"],
                            show_copy_button=True,
                        )
                        manual_logs = gr.Textbox(
                            label="📜 Step Logs",
                            lines=9,
                            elem_classes=["score-box"],
                            show_copy_button=True,
                        )

            run_manual_btn.click(
                fn=run_simulation_manual,
                inputs=[manual_emails_box, manual_tasks_box, seed_manual, steps_manual],
                outputs=[manual_summary, manual_logs, manual_scores],
            )

        # ── About ───────────────────────────────────────────────────────────
        with gr.Tab("ℹ About"):
            gr.Markdown(
                """
                ## Actions Available

                | Action | Effect |
                |--------|--------|
                | `reply_email` | Removes email from inbox, boosts satisfaction |
                | `ignore_email` | Penalty −2 if urgency = high |
                | `assign_task` | Consumes team hours, marks task assigned |
                | `accept_offer` | Spends budget, adds revenue |
                | `reject_offer` | Discards negotiation |
                | `negotiate` | Lowers offer price by 10% |
                | `wait` | No-op — dynamics still advance |

                ## Scoring Weights

                | Component | Weight |
                |-----------|--------|
                | Email Score (% replied) | 30% |
                | Task Score (% not missed) | 40% |
                | Negotiation Score (% accepted) | 30% |

                ## OpenEnv REST API

                The API endpoints are live alongside this UI:

                ```
                POST /reset   → Reset environment, returns initial observation
                POST /step    → Execute one action, returns obs + reward + done
                GET  /state   → Returns current observation snapshot
                GET  /docs    → Interactive Swagger docs
                ```
                """
            )


# ---------------------------------------------------------------------------
# Mount FastAPI routes onto Gradio app and launch
# ---------------------------------------------------------------------------

# Mount the FastAPI OpenEnv routes so /reset, /step, /state etc. still work
app = gr.mount_gradio_app(fastapi_app, demo, path="/")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
