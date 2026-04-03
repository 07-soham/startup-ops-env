"""
Event generator for StartupOpsEnv.

Two entry-points:

1. generate_initial_state(config, rng, difficulty)
   Called once at episode start.  Populates the environment with a
   difficulty-scaled batch of emails, tasks, and negotiations.

2. EventGenerator.generate_events(...)
   Called every step.  Probabilistically produces new events using
   config probabilities.

All randomness routes through the supplied ``rng`` (random.Random)
instance — no global random calls — ensuring full determinism.
"""
from __future__ import annotations

import random
from typing import Any, Dict, List, Tuple

from .models import Email, Negotiation, Priority, Sentiment, Task, Urgency

# ---------------------------------------------------------------------------
# Static content pools
# ---------------------------------------------------------------------------

_SENDERS = [
    "alice@corp.com",
    "bob@venture.com",
    "charlie@partner.org",
    "diana@client.net",
    "eve@investor.io",
    "frank@advisor.co",
    "grace@startup.io",
]

_SUBJECTS = [
    "Partnership opportunity",
    "Follow-up on proposal",
    "Urgent contract review",
    "Project status update",
    "Investment interest",
    "Due diligence request",
    "Support escalation",
    "Q3 roadmap discussion",
]

_TASK_NAMES = [
    "MVP Development Sprint",
    "Market Research Report",
    "Product Demo Preparation",
    "Investor Pitch Deck",
    "Legal Entity Review",
    "Financial Audit",
    "Team Hiring Campaign",
    "Customer Onboarding Flow",
    "PR & Media Campaign",
    "Platform Load Testing",
    "Compliance Documentation",
    "API Integration",
]

_CLIENTS = [
    "Acme Corp",
    "Beta Ventures",
    "Gamma Inc",
    "Delta Partners",
    "Epsilon LLC",
    "Zeta Capital",
    "Eta Solutions",
]

_URGENCIES    = [Urgency.low,    Urgency.medium,    Urgency.high]
_SENTIMENTS   = [Sentiment.positive, Sentiment.neutral, Sentiment.negative]
_PRIORITIES   = [Priority.low,   Priority.medium,   Priority.high]

# ---------------------------------------------------------------------------
# Difficulty configuration
# ---------------------------------------------------------------------------

# Each value is (min_count, max_count) for that entity type at episode start.
DIFFICULTY_CONFIG: Dict[str, Dict[str, Tuple[int, int]]] = {
    "easy":   {"emails": (2, 3), "tasks": (2, 3), "deals": (1, 2)},
    "medium": {"emails": (3, 5), "tasks": (3, 5), "deals": (1, 3)},
    "hard":   {"emails": (5, 7), "tasks": (5, 7), "deals": (2, 4)},
}


# ---------------------------------------------------------------------------
# Module-level helper: initial state generator
# ---------------------------------------------------------------------------

def generate_initial_state(
    config: Dict[str, Any],
    rng: random.Random,
    difficulty: str = "medium",
) -> Tuple[List[Email], List[Task], List[Negotiation], int, int, int]:
    """
    Generate the opening batch of emails, tasks, and negotiations.

    Called once at the start of each episode (inside ``reset()``).
    Uses difficulty level to determine how many of each entity to
    create, giving the agent a non-trivial initial workload.

    ID counters start at 1 for the first item of each type so that
    per-step ``EventGenerator`` can continue from where this leaves off.

    Parameters
    ----------
    config : dict
        Full environment configuration.
    rng : random.Random
        Seeded RNG shared with the rest of the environment.
    difficulty : str
        One of ``"easy"``, ``"medium"``, ``"hard"``.

    Returns
    -------
    (emails, tasks, negotiations, n_emails, n_tasks, n_deals)
        Pydantic model lists + their counts (used to seed ID counters).
    """
    cfg = DIFFICULTY_CONFIG.get(difficulty, DIFFICULTY_CONFIG["medium"])

    n_emails = rng.randint(*cfg["emails"])
    n_tasks  = rng.randint(*cfg["tasks"])
    n_deals  = rng.randint(*cfg["deals"])

    # ---- Emails ----------------------------------------------------------
    emails: List[Email] = []
    for i in range(n_emails):
        urgency   = _URGENCIES[rng.randint(0, len(_URGENCIES) - 1)]
        # Weighted: positive 30 %, neutral 40 %, negative 30 %
        sentiment = rng.choices(_SENTIMENTS, weights=[3, 4, 3], k=1)[0]
        sender    = _SENDERS[rng.randint(0, len(_SENDERS) - 1)]
        subject   = _SUBJECTS[rng.randint(0, len(_SUBJECTS) - 1)]
        emails.append(
            Email(
                id=f"email_{i + 1}",
                sender=sender,
                subject=subject,
                urgency=urgency,
                sentiment=sentiment,
                requires_action=True,
            )
        )

    # ---- Tasks -----------------------------------------------------------
    tasks: List[Task] = []
    for i in range(n_tasks):
        name     = _TASK_NAMES[rng.randint(0, len(_TASK_NAMES) - 1)]
        hours    = round(rng.uniform(4.0, 24.0), 1)
        deadline = rng.randint(2, 10)
        priority = _PRIORITIES[rng.randint(0, len(_PRIORITIES) - 1)]
        effort   = rng.randint(1, 5)
        impact   = round(rng.uniform(0.5, 2.0), 2)
        tasks.append(
            Task(
                id=f"task_{i + 1}",
                name=name,
                hours_required=hours,
                deadline=deadline,
                priority=priority,
                effort=effort,
                impact=impact,
            )
        )

    # ---- Negotiations ----------------------------------------------------
    negotiations: List[Negotiation] = []
    for i in range(n_deals):
        client      = _CLIENTS[rng.randint(0, len(_CLIENTS) - 1)]
        offer_price = round(rng.uniform(500.0, 20_000.0), 2)
        # min_price is always 60–90 % of offer so it never exceeds offer
        min_price   = round(offer_price * rng.uniform(0.6, 0.9), 2)
        quality     = round(rng.uniform(0.5, 1.5), 2)
        deadline    = rng.randint(2, 8)
        negotiations.append(
            Negotiation(
                id=f"negotiation_{i + 1}",
                client=client,
                offer_price=offer_price,
                min_price=min_price,
                quality=quality,
                deadline=deadline,
            )
        )

    return emails, tasks, negotiations, n_emails, n_tasks, n_deals


# ---------------------------------------------------------------------------
# Per-step event generator
# ---------------------------------------------------------------------------

class EventGenerator:
    """
    Generates new events each step of the episode.

    All randomness is routed through the supplied ``rng`` instance,
    which is seeded once at environment reset for reproducibility.
    """

    def __init__(self, config: Dict[str, Any], rng: random.Random) -> None:
        self.config = config
        self.rng = rng

    def generate_events(
        self,
        email_counter: int,
        task_counter: int,
        negotiation_counter: int,
    ) -> Tuple[List[Email], List[Task], List[Negotiation]]:
        """
        Probabilistically create new events for this step.

        IDs continue from the passed-in counters so that IDs always
        take the strictly-increasing form ``email_N``, ``task_N``,
        ``negotiation_N``.

        Returns
        -------
        (new_emails, new_tasks, new_negotiations)
        """
        new_emails: List[Email] = []
        new_tasks: List[Task] = []
        new_negotiations: List[Negotiation] = []

        # ---- Email -------------------------------------------------------
        if self.rng.random() < self.config["email_gen_prob"]:
            eid      = email_counter + 1
            urgency  = _URGENCIES[self.rng.randint(0, len(_URGENCIES) - 1)]
            sentiment = self.rng.choices(_SENTIMENTS, weights=[3, 4, 3], k=1)[0]
            sender   = _SENDERS[self.rng.randint(0, len(_SENDERS) - 1)]
            subject  = _SUBJECTS[self.rng.randint(0, len(_SUBJECTS) - 1)]
            # Negative-sentiment emails always require action
            req_action = True if sentiment == Sentiment.negative else True
            new_emails.append(
                Email(
                    id=f"email_{eid}",
                    sender=sender,
                    subject=subject,
                    urgency=urgency,
                    sentiment=sentiment,
                    requires_action=req_action,
                )
            )

        # ---- Task --------------------------------------------------------
        if self.rng.random() < self.config["task_gen_prob"]:
            tid      = task_counter + 1
            name     = _TASK_NAMES[self.rng.randint(0, len(_TASK_NAMES) - 1)]
            hours    = round(self.rng.uniform(4.0, 24.0), 1)
            deadline = self.rng.randint(3, 8)
            priority = _PRIORITIES[self.rng.randint(0, len(_PRIORITIES) - 1)]
            effort   = self.rng.randint(1, 5)
            impact   = round(self.rng.uniform(0.5, 2.0), 2)
            new_tasks.append(
                Task(
                    id=f"task_{tid}",
                    name=name,
                    hours_required=hours,
                    deadline=deadline,
                    priority=priority,
                    effort=effort,
                    impact=impact,
                )
            )

        # ---- Negotiation -------------------------------------------------
        if self.rng.random() < self.config["negotiation_gen_prob"]:
            nid         = negotiation_counter + 1
            client      = _CLIENTS[self.rng.randint(0, len(_CLIENTS) - 1)]
            offer_price = round(self.rng.uniform(5_000.0, 50_000.0), 2)
            min_price   = round(offer_price * self.rng.uniform(0.6, 0.9), 2)
            quality     = round(self.rng.uniform(0.5, 1.5), 2)
            deadline    = self.rng.randint(3, 6)
            new_negotiations.append(
                Negotiation(
                    id=f"negotiation_{nid}",
                    client=client,
                    offer_price=offer_price,
                    min_price=min_price,
                    quality=quality,
                    deadline=deadline,
                )
            )

        return new_emails, new_tasks, new_negotiations
