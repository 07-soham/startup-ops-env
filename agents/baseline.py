"""
Baseline heuristic agent for StartupOpsEnv.

Uses ALL observation fields — including the new sentiment, priority,
impact, quality, and min_price data — to make smarter decisions.

Priority order
--------------
1. Reply to high-urgency AND action-required emails first.
2. Assign the highest-impact task that is also overdue.
3. Assign the highest-impact task within normal deadline.
4. Accept the highest-quality negotiation deal (if budget healthy).
5. Skip negotiations where quality is too low (< 0.7).
6. Reply any remaining emails.
7. Wait — used only when there is literally nothing to do.
"""
from __future__ import annotations

from typing import Any, Dict

from env.models import Observation

_MIN_QUALITY_THRESHOLD = 0.7       # refuse deals below this quality
_BUDGET_SAFETY_FLOOR   = 50_000.0  # won't accept offers below this budget


class BaselineAgent:
    """
    Rule-based baseline agent.

    Reads an ``Observation`` Pydantic model and returns an action dict
    with keys ``"type"`` and optionally ``"target_id"``.

    Never crashes — every code-path returns a valid action dict.
    """

    def act(self, obs: Observation) -> Dict[str, Any]:
        """
        Choose the best action given the current observation.

        Parameters
        ----------
        obs : Observation
            Current environment observation (derived from EnvState via
            _get_obs).

        Returns
        -------
        dict
            e.g. ``{"type": "assign_task", "target_id": "task_3"}``
        """

        # ------------------------------------------------------------------
        # 1. High-urgency emails that require action → immediate reply
        # ------------------------------------------------------------------
        urgent_actionable = [
            eid for eid in obs.high_urgency_emails
            if eid in obs.action_required_emails
        ]
        if urgent_actionable:
            return {"type": "reply_email", "target_id": urgent_actionable[0]}

        # ------------------------------------------------------------------
        # 2. Overdue tasks (deadline ≤ 2) — pick highest impact
        # ------------------------------------------------------------------
        if obs.overdue_tasks and obs.team_hours > 0:
            target = max(
                obs.overdue_tasks,
                key=lambda tid: obs.task_impacts.get(tid, 1.0),
            )
            return {"type": "assign_task", "target_id": target}

        # ------------------------------------------------------------------
        # 3. Regular unassigned tasks — prioritise by impact × (1/deadline)
        #    so urgent high-impact tasks rank first
        # ------------------------------------------------------------------
        if obs.unassigned_task_ids and obs.team_hours > 0:
            def task_score(tid: str) -> float:
                impact   = obs.task_impacts.get(tid, 1.0)
                deadline = obs.task_deadlines.get(tid, 99)
                return impact / max(1, deadline)

            target = max(obs.unassigned_task_ids, key=task_score)
            return {"type": "assign_task", "target_id": target}

        # ------------------------------------------------------------------
        # 4. Negotiations — accept the highest-quality deal if budget allows
        #    and quality clears the minimum threshold
        # ------------------------------------------------------------------
        if obs.negotiation_ids and obs.budget > _BUDGET_SAFETY_FLOOR:
            # Filter to quality-worthy deals
            good_deals = [
                nid for nid in obs.negotiation_ids
                if obs.negotiation_qualities.get(nid, 0.0) >= _MIN_QUALITY_THRESHOLD
            ]
            if good_deals:
                best = max(
                    good_deals,
                    key=lambda nid: obs.negotiation_qualities.get(nid, 0.0),
                )
                return {"type": "accept_offer", "target_id": best}

            # Low-quality deals → reject rather than accept or counter-offer
            worst = min(
                obs.negotiation_ids,
                key=lambda nid: obs.negotiation_qualities.get(nid, 0.0),
            )
            return {"type": "reject_offer", "target_id": worst}

        # ------------------------------------------------------------------
        # 5. Remaining action-required emails
        # ------------------------------------------------------------------
        actionable_emails = [
            eid for eid in obs.email_ids
            if eid in obs.action_required_emails
        ]
        if actionable_emails:
            return {"type": "reply_email", "target_id": actionable_emails[0]}

        # ------------------------------------------------------------------
        # 6. Any remaining email (informational)
        # ------------------------------------------------------------------
        if obs.email_ids:
            return {"type": "reply_email", "target_id": obs.email_ids[0]}

        # ------------------------------------------------------------------
        # 7. Nothing to do → wait
        # ------------------------------------------------------------------
        return {"type": "wait"}
