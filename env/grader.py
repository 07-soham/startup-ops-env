"""
Episode grader for StartupOpsEnv.

Reads self.logs and the final EnvState to produce structured scores
across three domains: email handling, task management, negotiations.
"""
from __future__ import annotations

from typing import Any, Dict, List

from .models import EnvState


def safe_score(x) -> float:
    """Clamp x to the strictly-open interval (0, 1) required by the validator.

    * None  → 0.5  (no data, neutral default)
    * ≤ 0   → 0.01 (floor)
    * ≥ 1   → 0.99 (ceiling)
    * else  → float(x) as-is
    """
    if x is None:
        return 0.5
    x = float(x)
    if x <= 0:
        return 0.01
    if x >= 1:
        return 0.99
    return x


def grade_episode(
    logs: List[Dict[str, Any]],
    total_emails_created: int,
    total_tasks_created: int,
    total_negotiations_created: int,
    state: EnvState,
) -> Dict[str, Any]:
    """
    Grade a completed episode.

    Scoring breakdown
    -----------------
    email_score
        Fraction of all emails that were replied to (``replied_emails``
        / ``total_emails_created``).  Capped at 1.0.

    task_score
        Fraction of tasks that were **not** missed
        (1 - missed / total_tasks_created).  Clamped to [0, 1].

    negotiation_score
        Fraction of *handled* negotiations that were accepted
        (accepted / (accepted + rejected)).  Unhandled / expired
        negotiations are excluded — the agent is not penalised twice
        (dynamics already does that via reward signals).

    overall_score
        Weighted combination: 30 % email, 40 % task, 30 % negotiation.

    Parameters
    ----------
    logs : list of dicts
        ``env.logs`` — one entry per step.
    total_emails_created : int
        Lifetime count of emails generated.
    total_tasks_created : int
        Lifetime count of tasks generated.
    total_negotiations_created : int
        Lifetime count of negotiations generated.
    state : EnvState
        Final state after the episode ended.

    Returns
    -------
    dict with keys:
        email_score, task_score, negotiation_score,
        overall_score, total_reward, summary
    """
    if not logs:
        # Return neutral valid scores within (0, 1) open interval
        return {
            "email_score": float(0.5),
            "task_score": float(0.5),
            "negotiation_score": float(0.5),
            "overall_score": float(0.5),
            "total_reward": float(0.0),
            "summary": "No steps recorded.",
        }

    total_reward: float = sum(log["reward"] for log in logs)

    # ------------------------------------------------------------------
    # email_score
    # ------------------------------------------------------------------
    if total_emails_created == 0:
        email_score: float = 0.5
    else:
        email_score = min(1.0, state.replied_emails / total_emails_created)
    email_score = safe_score(email_score)

    # ------------------------------------------------------------------
    # task_score
    # ------------------------------------------------------------------
    if total_tasks_created == 0:
        task_score: float = 0.5
    else:
        task_score = max(0.0, 1.0 - state.missed_tasks / total_tasks_created)
    task_score = safe_score(task_score)

    # ------------------------------------------------------------------
    # negotiation_score
    # ------------------------------------------------------------------
    handled_negs: int = state.accepted_negotiations + state.rejected_negotiations
    if total_negotiations_created == 0:
        negotiation_score: float = 0.5
    else:
        negotiation_score = state.accepted_negotiations / max(1, handled_negs)
    negotiation_score = safe_score(negotiation_score)

    # ------------------------------------------------------------------
    # overall_score  (simple average, then safe_score)
    # ------------------------------------------------------------------
    overall_score: float = safe_score(
        (email_score + task_score + negotiation_score) / 3
    )

    # ------------------------------------------------------------------
    # Human-readable summary
    # ------------------------------------------------------------------
    tasks_not_missed = total_tasks_created - state.missed_tasks
    summary = (
        f"Episode complete - {len(logs)} steps.\n"
        f"\n"
        f"  Email Score       : {email_score:.2%}  "
        f"({state.replied_emails} / {total_emails_created} replied)\n"
        f"  Task Score        : {task_score:.2%}  "
        f"({tasks_not_missed} / {total_tasks_created} not missed)\n"
        f"  Negotiation Score : {negotiation_score:.2%}  "
        f"({state.accepted_negotiations} / {max(1, handled_negs)} accepted)\n"
        f"\n"
        f"  Overall Score     : {overall_score:.2%}\n"
        f"  Total Reward      : {total_reward:.2f}\n"
        f"\n"
        f"  Final Budget      : ${state.budget:,.2f}\n"
        f"  Revenue Generated : ${state.revenue:,.2f}\n"
        f"  Satisfaction      : {state.satisfaction:.2%}\n"
        f"  Missed Tasks      : {state.missed_tasks}\n"
    )

    # ------------------------------------------------------------------
    # Return all keys including summary for callers that need it
    # ------------------------------------------------------------------
    return {
        "email_score": float(round(email_score, 4)),
        "task_score": float(round(task_score, 4)),
        "negotiation_score": float(round(negotiation_score, 4)),
        "overall_score": float(round(overall_score, 4)),
        "total_reward": float(round(total_reward, 4)),
        "summary": summary,
    }
