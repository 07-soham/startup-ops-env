"""
Reward function for StartupOpsEnv.

All reward computation is centralised here and called from core.py.
Returns a plain float — no side-effects.
"""
from __future__ import annotations

from typing import Any, Dict

from .models import EnvState


def calculate_reward(
    state: EnvState,
    action_reward: float,
    missed_tasks: int,
    config: Dict[str, Any],
) -> float:
    """
    Compute the total scalar reward for a single environment step.

    Components
    ----------
    action_reward
        Reward (positive or negative) produced directly by the action
        taken this step (passed in from core.py).
    task_miss_penalty
        Applied per task that expired without being assigned.
    step_survival_reward
        Small positive reward for each step the episode continues —
        encourages the agent to keep the startup alive.
    satisfaction_bonus / penalty
        +0.5  if satisfaction > 0.8
        -0.5  if satisfaction < 0.3
    budget_health_bonus / penalty
        +0.2  if budget > 80 % of initial
        -1.0  if budget < 20 % of initial

    Parameters
    ----------
    state : EnvState
        Current (post-action, post-dynamics) state.
    action_reward : float
        Direct reward from the action applied this step.
    missed_tasks : int
        Number of tasks that newly expired this step.
    config : dict
        Environment configuration dict.

    Returns
    -------
    float
        Total reward for this step, rounded to 4 decimal places.
    """
    reward: float = action_reward

    # ---- Task miss penalties ------------------------------------------
    reward += missed_tasks * config["task_miss_penalty"]

    # ---- Survival bonus -----------------------------------------------
    reward += config["step_survival_reward"]

    # ---- Satisfaction bonus / penalty ---------------------------------
    if state.satisfaction > 0.8:
        reward += 0.5
    elif state.satisfaction < 0.3:
        reward -= 0.5

    # ---- Budget health bonus / penalty --------------------------------
    initial_budget: float = config["initial_budget"]
    if state.budget > initial_budget * 0.8:
        reward += 0.2
    elif state.budget < initial_budget * 0.2:
        reward -= 1.0

    return round(reward, 4)
