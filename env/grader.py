"""
Episode grader for StartupOpsEnv.

Reads self.logs and the final EnvState to produce structured scores
across three domains: email handling, task management, negotiations.
"""
from __future__ import annotations

from typing import Any, Dict, List

from .models import EnvState


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
        return {
            "email_score": 0.0,
            "task_score": 0.0,
            "negotiation_score": 0.0,
            "overall_score": 0.0,
            "total_reward": 0.0,
            "summary": "No steps recorded.",
        }

    total_reward: float = sum(log["reward"] for log in logs)

    # ------------------------------------------------------------------
    # email_score
    # ------------------------------------------------------------------
    email_score: float = min(
        1.0,
        state.replied_emails / max(1, total_emails_created),
    )

    # ------------------------------------------------------------------
    # task_score
    # ------------------------------------------------------------------
    task_score: float = max(
        0.0,
        1.0 - state.missed_tasks / max(1, total_tasks_created),
    )

    # ------------------------------------------------------------------
    # negotiation_score
    # ------------------------------------------------------------------
    handled_negs: int = state.accepted_negotiations + state.rejected_negotiations
    negotiation_score: float = state.accepted_negotiations / max(1, handled_negs)

    # ------------------------------------------------------------------
    # overall_score  (weighted average)
    # ------------------------------------------------------------------
    overall_score: float = (
        email_score * 0.30
        + task_score * 0.40
        + negotiation_score * 0.30
    )

    # ------------------------------------------------------------------
    # Human-readable summary
    # ------------------------------------------------------------------
    tasks_not_missed = total_tasks_created - state.missed_tasks
    summary = (
        f"Episode complete — {len(logs)} steps.\n"
        f"\n"
        f"📧  Email Score      : {email_score:.2%}  "
        f"({state.replied_emails} / {total_emails_created} replied)\n"
        f"✅  Task Score       : {task_score:.2%}  "
        f"({tasks_not_missed} / {total_tasks_created} not missed)\n"
        f"🤝  Negotiation Score: {negotiation_score:.2%}  "
        f"({state.accepted_negotiations} / {max(1, handled_negs)} accepted)\n"
        f"\n"
        f"🏆  Overall Score    : {overall_score:.2%}\n"
        f"💰  Total Reward     : {total_reward:.2f}\n"
        f"\n"
        f"📊  Final Budget     : ${state.budget:,.2f}\n"
        f"📈  Revenue Generated: ${state.revenue:,.2f}\n"
        f"😊  Satisfaction     : {state.satisfaction:.2%}\n"
        f"❌  Missed Tasks     : {state.missed_tasks}\n"
    )

    return {
        "email_score": round(email_score, 4),
        "task_score": round(task_score, 4),
        "negotiation_score": round(negotiation_score, 4),
        "overall_score": round(overall_score, 4),
        "total_reward": round(total_reward, 4),
        "summary": summary,
    }
