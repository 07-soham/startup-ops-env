"""
StartupOpsEnv — core environment.

Implements the full RL loop:
  reset() → Observation
  step(action) → (Observation, float, bool, dict)

Design contracts:
  • Internal state is an EnvState Pydantic model.
  • Observation is derived via _get_obs() and returned as an
    Observation Pydantic model.
  • self.logs: List[dict] records one entry per step.
  • Episode ends when self.time_step >= config["max_steps"].
  • Invalid actions return current obs + penalty = -1.0, no crash.
  • All randomness flows through self.rng = random.Random(seed).
"""
from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

from .dynamics import step_dynamics
from .generator import EventGenerator, generate_initial_state
from .models import EnvState, Negotiation, Observation, Task, Email, Urgency
from .reward import calculate_reward


class StartupOpsEnv:
    """
    Startup Operations Reinforcement Learning Environment.

    The agent manages a simulated startup across three domains:

    Emails
        ``reply_email``   — removes the email, boosts satisfaction.
        ``ignore_email``  — if urgency=high, applies a penalty.

    Tasks
        ``assign_task``   — consumes team_hours, marks task assigned.
        Unassigned tasks that reach deadline=0 are automatically missed.

    Negotiations
        ``accept_offer``  — spends budget cost, adds revenue.
        ``reject_offer``  — discards the negotiation.
        ``negotiate``     — lowers offer_price by negotiate_adjustment factor.

    Utility
        ``wait``          — no-op; dynamics still advance.

    Parameters
    ----------
    config : dict
        Loaded from configs/config.yaml via yaml.safe_load.
    """

    # ------------------------------------------------------------------
    # Construction / Reset
    # ------------------------------------------------------------------

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.rng: random.Random = random.Random(config["seed"])
        self.logs: List[Dict[str, Any]] = []

        # Counters and state are set up in _setup_episode()
        self._email_counter: int = 0
        self._task_counter: int = 0
        self._negotiation_counter: int = 0
        self._total_emails_created: int = 0
        self._total_tasks_created: int = 0
        self._total_negotiations_created: int = 0

        self.time_step: int = 0
        self.state: EnvState = EnvState(
            budget=0.0, satisfaction=0.0, team_hours=0.0, revenue=0.0
        )  # placeholder; overwritten by _setup_episode()
        self._setup_episode()
        self.generator = EventGenerator(self.config, self.rng)

    def reset(self) -> Observation:
        """Reset the environment to its initial state and return first obs."""
        self.rng = random.Random(self.config["seed"])
        self.logs = []
        self.time_step = 0
        self._setup_episode()
        self.generator = EventGenerator(self.config, self.rng)
        return self._get_obs()

    def _setup_episode(self) -> None:
        """
        Initialise (or re-initialise) the episode state.

        Calls ``generate_initial_state`` to populate the inbox, task
        board, and negotiation pipeline according to difficulty level,
        then seeds all ID/lifetime counters from the returned counts.
        """
        difficulty: str = self.config.get("difficulty", "medium")
        emails, tasks, negotiations, ne, nt, nn = generate_initial_state(
            config=self.config,
            rng=self.rng,
            difficulty=difficulty,
        )

        # ID counters continue from initial batch so per-step IDs never clash
        self._email_counter        = ne
        self._task_counter         = nt
        self._negotiation_counter  = nn
        self._total_emails_created        = ne
        self._total_tasks_created         = nt
        self._total_negotiations_created  = nn

        self.state = EnvState(
            budget=self.config["initial_budget"],
            satisfaction=self.config["initial_satisfaction"],
            team_hours=self.config["initial_team_hours"],
            revenue=0.0,
            emails=emails,
            tasks=tasks,
            negotiations=negotiations,
        )

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(
        self, action: Dict[str, Any]
    ) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        """
        Advance the environment by one step.

        Parameters
        ----------
        action : dict
            Must contain key ``"type"``.  Optional key ``"target_id"``.

        Returns
        -------
        obs : Observation
        reward : float
        done : bool
        info : dict
        """
        action_type: str = action.get("type", "")
        target_id: Optional[str] = action.get("target_id", None)

        action_reward: float = 0.0
        valid: bool = True

        # ==================================================================
        # Action dispatch — strict validation, no crashes
        # ==================================================================

        if action_type == "reply_email":
            email = self._find_email(target_id)
            if email is None:
                action_reward = -1.0
                valid = False
            else:
                self.state.emails.remove(email)
                self.state.satisfaction = min(
                    1.0,
                    self.state.satisfaction + self.config["reply_satisfaction_boost"],
                )
                self.state.replied_emails += 1
                action_reward = 1.0

        elif action_type == "ignore_email":
            email = self._find_email(target_id)
            if email is None:
                action_reward = -1.0
                valid = False
            else:
                if email.urgency == Urgency.high:
                    action_reward = self.config["high_urgency_ignore_penalty"]
                    self.state.satisfaction = max(
                        0.0, self.state.satisfaction - 0.1
                    )
                # Low/medium ignore has no reward effect
                self.state.emails.remove(email)

        elif action_type == "assign_task":
            task = self._find_task(target_id)
            if task is None or task.assigned:
                action_reward = -1.0
                valid = False
            else:
                if self.state.team_hours >= task.hours_required:
                    self.state.team_hours -= task.hours_required
                    task.assigned = True
                    # Higher-impact tasks yield proportionally more reward
                    action_reward = round(2.0 * task.impact, 4)
                else:
                    # Not enough hours — valid action structure but can't execute
                    action_reward = -0.5

        elif action_type == "accept_offer":
            neg = self._find_negotiation(target_id)
            if neg is None:
                action_reward = -1.0
                valid = False
            else:
                self.state.budget -= self.config["accept_offer_budget_cost"]
                self.state.revenue += neg.offer_price
                self.state.accepted_negotiations += 1
                self.state.negotiations.remove(neg)
                # Higher-quality deals yield proportionally more reward
                action_reward = round(3.0 * neg.quality, 4)

        elif action_type == "reject_offer":
            neg = self._find_negotiation(target_id)
            if neg is None:
                action_reward = -1.0
                valid = False
            else:
                self.state.negotiations.remove(neg)
                self.state.rejected_negotiations += 1
                action_reward = 0.0

        elif action_type == "negotiate":
            neg = self._find_negotiation(target_id)
            if neg is None:
                action_reward = -1.0
                valid = False
            else:
                new_price = round(
                    neg.offer_price * self.config["negotiate_adjustment"], 2
                )
                if new_price < neg.min_price:
                    # Negotiated below client's floor → deal collapses
                    self.state.negotiations.remove(neg)
                    self.state.collapsed_negotiations += 1
                    action_reward = -1.5
                else:
                    neg.offer_price = new_price
                    action_reward = -0.5   # small cost for counter-offering

        elif action_type == "wait":
            action_reward = 0.0

        else:
            # Unknown or missing action type
            action_reward = -1.0
            valid = False

        # ==================================================================
        # Generate new events (before dynamics so deadlines start fair)
        # ==================================================================
        new_emails, new_tasks, new_negotiations = self.generator.generate_events(
            self._email_counter,
            self._task_counter,
            self._negotiation_counter,
        )

        self._email_counter += len(new_emails)
        self._task_counter += len(new_tasks)
        self._negotiation_counter += len(new_negotiations)
        self._total_emails_created += len(new_emails)
        self._total_tasks_created += len(new_tasks)
        self._total_negotiations_created += len(new_negotiations)

        # Enforce inbox / board / pipeline capacity limits
        for e in new_emails:
            if len(self.state.emails) < self.config["max_emails"]:
                self.state.emails.append(e)
        for t in new_tasks:
            if len(self.state.tasks) < self.config["max_tasks"]:
                self.state.tasks.append(t)
        for n in new_negotiations:
            if len(self.state.negotiations) < self.config["max_negotiations"]:
                self.state.negotiations.append(n)

        # ==================================================================
        # Dynamics — deadline counting, miss detection, expiry
        # ==================================================================
        missed_tasks, _expired_negs = step_dynamics(self.state)

        # ==================================================================
        # Reward — entirely computed in reward.py
        # ==================================================================
        reward = calculate_reward(
            state=self.state,
            action_reward=action_reward,
            missed_tasks=missed_tasks,
            config=self.config,
        )

        # ==================================================================
        # Advance clock
        # ==================================================================
        self.time_step += 1
        self.state.step = self.time_step

        # ==================================================================
        # Log this step
        # ==================================================================
        efficiency = self._compute_efficiency()
        self.logs.append(
            {
                "step": self.time_step,
                "action": action_type,
                "reward": reward,
                "budget": round(self.state.budget, 2),
                "efficiency": round(efficiency, 4),
            }
        )

        # ==================================================================
        # Termination
        # ==================================================================
        done: bool = self.time_step >= self.config["max_steps"]

        obs = self._get_obs()
        info: Dict[str, Any] = {
            "valid_action": valid,
            "new_emails": len(new_emails),
            "new_tasks": len(new_tasks),
            "new_negotiations": len(new_negotiations),
            "missed_tasks_this_step": missed_tasks,
        }

        return obs, reward, done, info

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_email(self, email_id: Optional[str]) -> Optional[Email]:
        if email_id is None:
            return None
        for e in self.state.emails:
            if e.id == email_id:
                return e
        return None

    def _find_task(self, task_id: Optional[str]) -> Optional[Task]:
        if task_id is None:
            return None
        for t in self.state.tasks:
            if t.id == task_id:
                return t
        return None

    def _find_negotiation(self, neg_id: Optional[str]) -> Optional[Negotiation]:
        if neg_id is None:
            return None
        for n in self.state.negotiations:
            if n.id == neg_id:
                return n
        return None

    def _compute_efficiency(self) -> float:
        """
        Scalar in [0, 1] measuring how well the agent has used its
        opportunities relative to adverse outcomes.

        efficiency = (positive_outcomes / total_outcomes) * satisfaction
        """
        positive = self.state.replied_emails + self.state.accepted_negotiations
        negative = self.state.missed_tasks
        total = positive + negative
        if total == 0:
            return round(self.state.satisfaction, 4)
        return round((positive / total) * self.state.satisfaction, 4)

    def _get_obs(self) -> Observation:
        """Derive an Observation snapshot from the current EnvState."""
        s = self.state
        return Observation(
            budget=s.budget,
            satisfaction=s.satisfaction,
            team_hours=s.team_hours,
            revenue=s.revenue,
            num_emails=len(s.emails),
            num_tasks=len(s.tasks),
            num_negotiations=len(s.negotiations),
            missed_tasks=s.missed_tasks,
            step=s.step,
            # ---- ID lists ------------------------------------------------
            email_ids=[e.id for e in s.emails],
            unassigned_task_ids=[
                t.id for t in s.tasks if not t.assigned and not t.missed
            ],
            negotiation_ids=[n.id for n in s.negotiations],
            # ---- Derived helpers -----------------------------------------
            high_urgency_emails=[
                e.id for e in s.emails if e.urgency == Urgency.high
            ],
            action_required_emails=[
                e.id for e in s.emails if e.requires_action
            ],
            overdue_tasks=[
                t.id for t in s.tasks if t.deadline <= 2 and not t.assigned
            ],
            # ---- Basic lookup maps ---------------------------------------
            task_deadlines={t.id: t.deadline for t in s.tasks},
            negotiation_offers={n.id: n.offer_price for n in s.negotiations},
            email_urgencies={e.id: e.urgency.value for e in s.emails},
            # ---- Extended lookup maps (new fields) -----------------------
            email_sentiments={e.id: e.sentiment.value for e in s.emails},
            task_priorities={t.id: t.priority.value for t in s.tasks},
            task_impacts={t.id: t.impact for t in s.tasks},
            negotiation_min_prices={n.id: n.min_price for n in s.negotiations},
            negotiation_qualities={n.id: n.quality for n in s.negotiations},
        )

    # ------------------------------------------------------------------
    # Public accessors for grader / UI
    # ------------------------------------------------------------------

    def get_totals(self) -> Dict[str, int]:
        """Return lifetime creation counts for the grader."""
        return {
            "total_emails_created": self._total_emails_created,
            "total_tasks_created": self._total_tasks_created,
            "total_negotiations_created": self._total_negotiations_created,
        }
