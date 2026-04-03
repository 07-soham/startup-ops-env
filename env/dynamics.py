"""
State-transition dynamics for StartupOpsEnv.

Called once per step *after* the agent's action has been applied.
Mutates EnvState in-place and returns summary counts.
"""
from __future__ import annotations

from typing import Tuple

from .models import EnvState


def step_dynamics(state: EnvState) -> Tuple[int, int]:
    """
    Apply one step of world dynamics to *state* (mutated in-place).

    Rules
    -----
    Tasks
        • Decrement ``deadline`` by 1 for every non-missed task.
        • If ``deadline`` reaches 0 and the task was **not** assigned
          → mark ``missed = True``, increment ``state.missed_tasks``.
        • If ``deadline`` reaches 0 and the task **was** assigned
          → the work is done; remove it silently.
        • Remove all completed / missed tasks from the list.

    Negotiations
        • Decrement ``deadline`` by 1 for every negotiation.
        • Remove any negotiation whose ``deadline`` has reached 0.

    Parameters
    ----------
    state : EnvState
        Live environment state (mutated in-place).

    Returns
    -------
    (missed_task_count, expired_negotiation_count) : Tuple[int, int]
    """
    missed_count: int = 0

    # ------------------------------------------------------------------
    # Tasks
    # ------------------------------------------------------------------
    for task in state.tasks:
        if task.missed:
            continue
        task.deadline -= 1
        if task.deadline <= 0 and not task.assigned:
            task.missed = True
            state.missed_tasks += 1
            missed_count += 1

    # Retain only active tasks (not missed, not completed)
    state.tasks = [
        t for t in state.tasks
        if not t.missed and not (t.assigned and t.deadline <= 0)
    ]

    # ------------------------------------------------------------------
    # Negotiations
    # ------------------------------------------------------------------
    for neg in state.negotiations:
        neg.deadline -= 1

    expired_count: int = sum(1 for n in state.negotiations if n.deadline <= 0)
    state.negotiations = [n for n in state.negotiations if n.deadline > 0]

    return missed_count, expired_count
