"""
app.py — Polished Gradio UI for StartupOps AI Simulator.

A hackathon-winning interface with:
- Auto Simulation mode
- Interactive Manual mode (step-by-step)
- Agent Comparison mode
- Scoring Dashboard

Entry-point for both local runs and Hugging Face Spaces deployment.
"""
from __future__ import annotations

import json
import random
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import gradio as gr
import yaml

from env.core import StartupOpsEnv
from env.grader import grade_episode
from env.models import Observation
from agents.baseline import BaselineAgent


# -----------------------------------------------------------------------------
# Global State
# -----------------------------------------------------------------------------
_env: Optional[StartupOpsEnv] = None
_current_obs: Optional[Observation] = None

_CONFIG_PATH = Path(__file__).parent / "configs" / "config.yaml"


def _load_config() -> dict:
    with open(_CONFIG_PATH, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _empty_figure(title: str = ""):
    """Return a blank matplotlib Figure used as a safe fallback."""
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.set_title(title)
    ax.text(
        0.5, 0.5, "No data",
        ha="center", va="center", transform=ax.transAxes,
    )
    return fig


def _format_state(obs: Optional[Observation]) -> str:
    """Format the current observation into a readable state string."""
    if obs is None:
        return "Environment not initialized. Please click 'Reset Environment'."

    return f"""
📊 Current State:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🕐 Step:           {obs.step}
💰 Budget:         ${obs.budget:,.2f}
😊 Satisfaction:    {obs.satisfaction:.2%}
👥 Team Hours:      {obs.team_hours:.1f}
💵 Revenue:         ${obs.revenue:,.2f}

📧 Emails:          {obs.num_emails}
📋 Tasks:           {obs.num_tasks}
🤝 Negotiations:    {obs.num_negotiations}
❌ Missed Tasks:    {obs.missed_tasks}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""


# -----------------------------------------------------------------------------
# Random Agent (for comparison)
# -----------------------------------------------------------------------------
class RandomAgent:
    """
    Simple random policy agent for comparison.
    Randomly chooses action type and target_id.
    """

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def act(self, obs: Observation) -> Dict[str, Any]:
        """Choose a random valid action."""
        action_types = ["reply_email", "ignore_email", "assign_task",
                        "accept_offer", "reject_offer", "negotiate", "wait"]

        action_type = self.rng.choice(action_types)
        target_id = None

        if action_type == "reply_email" and obs.email_ids:
            target_id = self.rng.choice(obs.email_ids)
        elif action_type == "ignore_email" and obs.email_ids:
            target_id = self.rng.choice(obs.email_ids)
        elif action_type == "assign_task" and obs.unassigned_task_ids:
            target_id = self.rng.choice(obs.unassigned_task_ids)
        elif action_type in ["accept_offer", "reject_offer", "negotiate"]:
            if obs.negotiation_ids:
                target_id = self.rng.choice(obs.negotiation_ids)
            else:
                action_type = "wait"

        if target_id is None and action_type not in ["wait"]:
            action_type = "wait"

        result = {"type": action_type}
        if target_id:
            result["target_id"] = target_id
        return result


# -----------------------------------------------------------------------------
# Environment Functions
# -----------------------------------------------------------------------------
def reset_env() -> Tuple[str, str]:
    """
    Reset the environment and return initial state.

    Returns:
        state_str: Formatted state display
        status: Status message
    """
    global _env, _current_obs
    try:
        config = _load_config()
        _env = StartupOpsEnv(config)
        _current_obs = _env.reset()

        return _format_state(_current_obs), "Environment reset successfully!"
    except Exception as e:
        return f"Error: {str(e)}", f"Reset failed: {str(e)}"


def step_manual(action_type: str, target_id: str, value: float) -> Tuple[str, str, str, str]:
    """
    Execute one manual step in the environment.

    Args:
        action_type: Type of action to take
        target_id: Target ID (if applicable)
        value: Value slider input (0-100)

    Returns:
        state_str: Formatted state display
        status: Status message
        reward: Reward from the step
        done: Done status
    """
    global _env, _current_obs

    try:
        if _env is None or _current_obs is None:
            return (
                "⚠️ Environment not initialized! Please click 'Reset Environment' first.",
                "Waiting for reset...",
                "N/A",
                "false"
            )

        # Build action dict
        action = {"type": action_type}

        # Add target_id if provided and not empty
        if target_id and target_id.strip():
            action["target_id"] = target_id.strip()

        # Add value if applicable (scaled to 0-1)
        if action_type in ["negotiate", "assign_task"]:
            action["value"] = value / 100.0

        # Execute step
        _current_obs, reward, done, info = _env.step(action)

        # Build status message
        valid_str = "✅ Valid" if info.get("valid_action", True) else "❌ Invalid"
        status = f"{valid_str} action: {action_type}"
        if target_id:
            status += f" on {target_id}"

        # Add event info
        new_events = info.get("new_emails", 0) + info.get("new_tasks", 0) + info.get("new_negotiations", 0)
        if new_events > 0:
            status += f" | New events: {new_events}"

        if info.get("missed_tasks_this_step", 0) > 0:
            status += f" | ⚠️ {info['missed_tasks_this_step']} task(s) missed!"

        return (
            _format_state(_current_obs),
            status,
            f"{reward:+.2f}",
            "true" if done else "false"
        )

    except Exception as e:
        error_msg = f"Step error: {str(e)}"
        return (
            _format_state(_current_obs) if _current_obs else "Error state",
            error_msg,
            "0.00",
            "false"
        )


def get_available_targets(action_type: str) -> List[str]:
    """Get available target IDs for a given action type."""
    global _current_obs

    if _current_obs is None:
        return []

    if action_type in ["reply_email", "ignore_email"]:
        return _current_obs.email_ids
    elif action_type == "assign_task":
        return _current_obs.unassigned_task_ids
    elif action_type in ["accept_offer", "reject_offer", "negotiate"]:
        return _current_obs.negotiation_ids
    return []


# -----------------------------------------------------------------------------
# Auto Simulation
# -----------------------------------------------------------------------------
def run_auto() -> Tuple[str, str, Any, Any, Any]:
    """
    Run a full episode with the BaselineAgent.

    Returns:
        summary: Episode summary text
        logs_json: JSON logs
        fig_reward: Reward plot
        fig_budget: Budget plot
        fig_efficiency: Efficiency plot
    """
    try:
        config = _load_config()
        env = StartupOpsEnv(config)
        agent = BaselineAgent()

        obs = env.reset()
        done = False

        while not done:
            action = agent.act(obs)
            obs, _, done, _ = env.step(action)

        # Grade the episode
        totals = env.get_totals()
        grades = grade_episode(
            logs=env.logs,
            total_emails_created=totals["total_emails_created"],
            total_tasks_created=totals["total_tasks_created"],
            total_negotiations_created=totals["total_negotiations_created"],
            state=env.state,
        )

        summary = grades["summary"]
        logs_json = json.dumps(env.logs, indent=2)

        # Extract series for plots
        steps = [log["step"] for log in env.logs]
        rewards = [log["reward"] for log in env.logs]
        budgets = [log["budget"] for log in env.logs]
        efficiencies = [log["efficiency"] for log in env.logs]

        # Plot 1 - Reward vs Steps
        fig1, ax1 = plt.subplots(figsize=(7, 3))
        ax1.plot(steps, rewards)
        ax1.set_title("Reward vs Steps")
        ax1.set_xlabel("Step")
        ax1.set_ylabel("Reward")
        ax1.grid(True, alpha=0.3)
        fig1.tight_layout()

        # Plot 2 - Budget vs Steps
        fig2, ax2 = plt.subplots(figsize=(7, 3))
        ax2.plot(steps, budgets)
        ax2.set_title("Budget vs Steps")
        ax2.set_xlabel("Step")
        ax2.set_ylabel("Budget ($)")
        ax2.grid(True, alpha=0.3)
        fig2.tight_layout()

        # Plot 3 - Efficiency vs Steps
        fig3, ax3 = plt.subplots(figsize=(7, 3))
        ax3.plot(steps, efficiencies)
        ax3.set_title("Efficiency vs Steps")
        ax3.set_xlabel("Step")
        ax3.set_ylabel("Efficiency")
        ax3.set_ylim(0.0, 1.05)
        ax3.grid(True, alpha=0.3)
        fig3.tight_layout()

        return summary, logs_json, fig1, fig2, fig3

    except Exception:
        err_msg = f"Simulation error:\n\n{traceback.format_exc()}"
        return (
            err_msg,
            "[]",
            _empty_figure("Reward vs Steps"),
            _empty_figure("Budget vs Steps"),
            _empty_figure("Efficiency vs Steps"),
        )


# -----------------------------------------------------------------------------
# Agent Comparison
# -----------------------------------------------------------------------------
def run_comparison() -> Tuple[str, str, Any, Any]:
    """
    Compare BaselineAgent vs RandomAgent on the same seed.

    Returns:
        summary: Comparison summary
        comparison_table: JSON comparison table
        fig_reward: Reward comparison plot
        fig_efficiency: Efficiency comparison plot
    """
    try:
        config = _load_config()
        seed = config.get("seed", 42)

        # Run Baseline Agent
        env1 = StartupOpsEnv(config)
        agent1 = BaselineAgent()
        obs1 = env1.reset()
        done1 = False

        while not done1:
            action = agent1.act(obs1)
            obs1, _, done1, _ = env1.step(action)

        # Run Random Agent with same seed
        env2 = StartupOpsEnv(config)
        agent2 = RandomAgent(seed=seed)
        obs2 = env2.reset()
        done2 = False

        while not done2:
            action = agent2.act(obs2)
            obs2, _, done2, _ = env2.step(action)

        # Extract results
        totals1 = env1.get_totals()
        totals2 = env2.get_totals()

        grades1 = grade_episode(
            logs=env1.logs,
            total_emails_created=totals1["total_emails_created"],
            total_tasks_created=totals1["total_tasks_created"],
            total_negotiations_created=totals1["total_negotiations_created"],
            state=env1.state,
        )

        grades2 = grade_episode(
            logs=env2.logs,
            total_emails_created=totals2["total_emails_created"],
            total_tasks_created=totals2["total_tasks_created"],
            total_negotiations_created=totals2["total_negotiations_created"],
            state=env2.state,
        )

        # Build comparison
        steps1 = [log["step"] for log in env1.logs]
        rewards1 = [log["reward"] for log in env1.logs]
        efficiencies1 = [log["efficiency"] for log in env1.logs]

        steps2 = [log["step"] for log in env2.logs]
        rewards2 = [log["reward"] for log in env2.logs]
        efficiencies2 = [log["efficiency"] for log in env2.logs]

        # Calculate cumulative rewards
        cumreward1 = [sum(rewards1[:i+1]) for i in range(len(rewards1))]
        cumreward2 = [sum(rewards2[:i+1]) for i in range(len(rewards2))]

        comparison_data = {
            "baseline": {
                "total_reward": grades1["total_reward"],
                "overall_score": grades1["overall_score"],
                "email_score": grades1["email_score"],
                "task_score": grades1["task_score"],
                "negotiation_score": grades1["negotiation_score"],
                "steps": len(env1.logs),
            },
            "random": {
                "total_reward": grades2["total_reward"],
                "overall_score": grades2["overall_score"],
                "email_score": grades2["email_score"],
                "task_score": grades2["task_score"],
                "negotiation_score": grades2["negotiation_score"],
                "steps": len(env2.logs),
            },
        }

        # Build summary
        summary = f"""
⚔️ Agent Comparison Results:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🏆 BASELINE AGENT:
   Total Reward:     {grades1['total_reward']:.2f}
   Overall Score:    {grades1['overall_score']:.2%}
   Steps Taken:      {len(env1.logs)}

🎲 RANDOM AGENT:
   Total Reward:     {grades2['total_reward']:.2f}
   Overall Score:    {grades2['overall_score']:.2%}
   Steps Taken:      {len(env2.logs)}

📊 PERFORMANCE DIFFERENCE:
   Reward Delta:     {grades1['total_reward'] - grades2['total_reward']:+.2f}
   Score Delta:      {(grades1['overall_score'] - grades2['overall_score']) * 100:+.2f}%
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

        # Plot 1 - Reward Curves
        fig1, ax1 = plt.subplots(figsize=(7, 3))
        ax1.plot(steps1, cumreward1, label="Baseline Agent", linewidth=2)
        ax1.plot(steps2, cumreward2, label="Random Agent", linewidth=2)
        ax1.set_title("Cumulative Reward Comparison")
        ax1.set_xlabel("Step")
        ax1.set_ylabel("Cumulative Reward")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        fig1.tight_layout()

        # Plot 2 - Efficiency Curves
        fig2, ax2 = plt.subplots(figsize=(7, 3))
        ax2.plot(steps1, efficiencies1, label="Baseline Agent", linewidth=2)
        ax2.plot(steps2, efficiencies2, label="Random Agent", linewidth=2)
        ax2.set_title("Efficiency Comparison")
        ax2.set_xlabel("Step")
        ax2.set_ylabel("Efficiency")
        ax2.set_ylim(0, 1.05)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        fig2.tight_layout()

        return summary, json.dumps(comparison_data, indent=2), fig1, fig2

    except Exception:
        err_msg = f"Comparison error:\n\n{traceback.format_exc()}"
        return (
            err_msg,
            "{}",
            _empty_figure("Reward Comparison"),
            _empty_figure("Efficiency Comparison"),
        )


# -----------------------------------------------------------------------------
# Scoring Dashboard
# -----------------------------------------------------------------------------
def evaluate_performance() -> str:
    """
    Run a full episode and return detailed scores.

    Returns:
        scores: Formatted scoring output
    """
    try:
        config = _load_config()
        env = StartupOpsEnv(config)
        agent = BaselineAgent()

        obs = env.reset()
        done = False

        while not done:
            action = agent.act(obs)
            obs, _, done, _ = env.step(action)

        totals = env.get_totals()
        grades = grade_episode(
            logs=env.logs,
            total_emails_created=totals["total_emails_created"],
            total_tasks_created=totals["total_tasks_created"],
            total_negotiations_created=totals["total_negotiations_created"],
            state=env.state,
        )

        scores = f"""
🏆 SCORING DASHBOARD:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📧 Email Score:           {grades['email_score']:.2%}
   └─ Replied: {env.state.replied_emails} / {totals['total_emails_created']}

✅ Task Score:             {grades['task_score']:.2%}
   └─ Completed: {totals['total_tasks_created'] - env.state.missed_tasks} / {totals['total_tasks_created']}

🤝 Negotiation Score:     {grades['negotiation_score']:.2%}
   └─ Accepted: {env.state.accepted_negotiations} / {env.state.accepted_negotiations + env.state.rejected_negotiations}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎯 OVERALL SCORE:         {grades['overall_score']:.2%}

💰 Total Reward:          {grades['total_reward']:.2f}
📊 Total Steps:           {len(env.logs)}

📈 Final Stats:
   • Budget:              ${env.state.budget:,.2f}
   • Revenue:             ${env.state.revenue:,.2f}
   • Satisfaction:        {env.state.satisfaction:.2%}
   • Team Hours Remaining: {env.state.team_hours:.1f}
   • Missed Tasks:        {env.state.missed_tasks}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

        return scores

    except Exception as e:
        return f"Evaluation error:\n\n{traceback.format_exc()}"


# -----------------------------------------------------------------------------
# Gradio UI
# -----------------------------------------------------------------------------
with gr.Blocks(
    title="StartupOps AI Simulator"
) as demo:

    # Header
    gr.Markdown("""
    # 🚀 StartupOps AI Simulator
    ### AI vs Human Decision Making — Train, Compare, and Evaluate Agents
    """)

    # -----------------------------------------------------------------------------
    # TABS
    # -----------------------------------------------------------------------------
    with gr.Tabs():

        # ======================================================================
        # TAB 1: Auto Simulation
        # ======================================================================
        with gr.TabItem("🤖 Auto Simulation"):
            gr.Markdown("Run the Baseline Agent through a full episode automatically.")

            run_btn = gr.Button("▶ Run AI Simulation", variant="primary", size="lg")

            with gr.Row():
                summary_out = gr.Textbox(
                    label="📋 Episode Summary",
                    lines=16,
                    max_lines=20,
                    interactive=False,
                )
                logs_out = gr.Code(
                    label="📝 Step Logs (JSON)",
                    language="json",
                    lines=16,
                )

            with gr.Row():
                plot_reward = gr.Plot(label="Reward vs Steps")
                plot_budget = gr.Plot(label="Budget vs Steps")
                plot_efficiency = gr.Plot(label="Efficiency vs Steps")

            run_btn.click(
                fn=run_auto,
                inputs=[],
                outputs=[summary_out, logs_out, plot_reward, plot_budget, plot_efficiency],
            )

        # ======================================================================
        # TAB 2: Manual Mode
        # ======================================================================
        with gr.TabItem("🎮 Manual Mode"):
            gr.Markdown("Take control! Step through the environment manually.")

            with gr.Row():
                with gr.Column(scale=1):
                    reset_btn = gr.Button("🔄 Reset Environment", variant="secondary")

                    gr.Markdown("### 🎮 Action Controls")

                    action_dropdown = gr.Dropdown(
                        choices=[
                            ("Reply to Email", "reply_email"),
                            ("Ignore Email", "ignore_email"),
                            ("Assign Task", "assign_task"),
                            ("Accept Offer", "accept_offer"),
                            ("Reject Offer", "reject_offer"),
                            ("Negotiate", "negotiate"),
                            ("Wait", "wait"),
                        ],
                        label="Action Type",
                        value="wait",
                    )

                    target_dropdown = gr.Dropdown(
                        choices=[],
                        label="Target ID (optional)",
                        value="",
                        allow_custom_value=True,
                    )

                    value_slider = gr.Slider(
                        minimum=0,
                        maximum=100,
                        value=50,
                        label="Value (0-100)",
                        info="Used for negotiate/assign_task actions",
                    )

                    step_btn = gr.Button("➡️ Step", variant="primary")

                with gr.Column(scale=2):
                    state_display = gr.Textbox(
                        label="📊 Current State",
                        lines=14,
                        interactive=False,
                        value="Click 'Reset Environment' to start!",
                    )

                    with gr.Row():
                        status_display = gr.Textbox(
                            label="📝 Status",
                            lines=2,
                            interactive=False,
                            value="Waiting...",
                        )
                        reward_display = gr.Textbox(
                            label="💰 Reward",
                            lines=2,
                            interactive=False,
                            value="N/A",
                        )
                        done_display = gr.Textbox(
                            label="✅ Done",
                            lines=2,
                            interactive=False,
                            value="false",
                        )

            # Reset environment
            reset_btn.click(
                fn=reset_env,
                inputs=[],
                outputs=[state_display, status_display],
            )

            # Update target dropdown based on action type
            def update_targets(action_type):
                targets = get_available_targets(action_type)
                return gr.Dropdown(choices=[(t, t) for t in targets], value="" if not targets else targets[0])

            action_dropdown.change(
                fn=update_targets,
                inputs=[action_dropdown],
                outputs=[target_dropdown],
            )

            # Step
            step_btn.click(
                fn=step_manual,
                inputs=[action_dropdown, target_dropdown, value_slider],
                outputs=[state_display, status_display, reward_display, done_display],
            )

        # ======================================================================
        # TAB 3: Agent Comparison
        # ======================================================================
        with gr.TabItem("⚔️ Agent Comparison"):
            gr.Markdown("Compare Baseline Agent vs Random Agent on the same seed.")

            compare_btn = gr.Button("⚔️ Run Comparison", variant="primary", size="lg")

            with gr.Row():
                compare_summary = gr.Textbox(
                    label="📊 Comparison Summary",
                    lines=16,
                    interactive=False,
                )
                compare_table = gr.Code(
                    label="📋 Comparison Table (JSON)",
                    language="json",
                    lines=16,
                )

            with gr.Row():
                compare_reward_plot = gr.Plot(label="Reward Curves")
                compare_efficiency_plot = gr.Plot(label="Efficiency Curves")

            compare_btn.click(
                fn=run_comparison,
                inputs=[],
                outputs=[compare_summary, compare_table, compare_reward_plot, compare_efficiency_plot],
            )

        # ======================================================================
        # TAB 4: Scoring Dashboard
        # ======================================================================
        with gr.TabItem("🏆 Scoring Dashboard"):
            gr.Markdown("Evaluate agent performance with detailed scoring metrics.")

            eval_btn = gr.Button("🏆 Evaluate Performance", variant="primary", size="lg")

            scores_output = gr.Textbox(
                label="📊 Performance Scores",
                lines=25,
                interactive=False,
            )

            eval_btn.click(
                fn=evaluate_performance,
                inputs=[],
                outputs=[scores_output],
            )


# -----------------------------------------------------------------------------
# Launch
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    demo.launch(
        theme=gr.themes.Soft(),
        css="""
        .tab { font-size: 16px; font-weight: 600; }
        .output-box { background-color: #f5f5f5; border-radius: 8px; }
        """
    )
