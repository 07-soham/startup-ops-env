"""
Event generator for StartupOpsEnv.

Supports TWO MODES:
1. AUTO MODE (default): Scenario-based generation with seed
2. MANUAL MODE: User-provided inputs with LLM parsing

All randomness routes through the supplied ``rng`` for determinism.
"""
from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

from .llm_parser import apply_urgency_override, check_escalation_triggers, parse_email
from .models import Email, Negotiation, ParsedEmail, Priority, Scenario, Sentiment, Task, Urgency
from .scenarios import get_scenario, list_scenarios

# ---------------------------------------------------------------------------
# Static content pools (for per-step generation)
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

_URGENCIES = [Urgency.low, Urgency.medium, Urgency.high]
_SENTIMENTS = [Sentiment.positive, Sentiment.neutral, Sentiment.negative]
_PRIORITIES = [Priority.low, Priority.medium, Priority.high]

# ---------------------------------------------------------------------------
# Difficulty configuration
# ---------------------------------------------------------------------------

DIFFICULTY_CONFIG: Dict[str, Dict[str, Tuple[int, int]]] = {
    "easy": {"emails": (2, 3), "tasks": (2, 3), "deals": (1, 2)},
    "medium": {"emails": (3, 5), "tasks": (3, 5), "deals": (1, 3)},
    "hard": {"emails": (5, 7), "tasks": (5, 7), "deals": (2, 4)},
}


# ---------------------------------------------------------------------------
# Manual mode state holder
# ---------------------------------------------------------------------------

class ManualInputState:
    """
    Holds user-provided manual inputs for deterministic replay.
    """

    def __init__(self, emails: Optional[List[Dict]] = None, tasks: Optional[List[Dict]] = None):
        self.emails = emails or []
        self.tasks = tasks or []
        self._email_index = 0
        self._task_index = 0

    def get_next_emails(self, count: int) -> List[Dict]:
        """Get next batch of emails."""
        emails = self.emails[self._email_index:self._email_index + count]
        self._email_index += len(emails)
        return emails

    def get_next_tasks(self, count: int) -> List[Dict]:
        """Get next batch of tasks."""
        tasks = self.tasks[self._task_index:self._task_index + count]
        self._task_index += len(tasks)
        return tasks

    def has_more(self) -> bool:
        """Check if there are more manual inputs."""
        return self._email_index < len(self.emails) or self._task_index < len(self.tasks)


# ---------------------------------------------------------------------------
# Scenario-based initial state generator (AUTO MODE)
# ---------------------------------------------------------------------------

def generate_initial_state(
    config: Dict[str, Any],
    rng: random.Random,
    difficulty: str = "medium",
    scenario_name: Optional[str] = None,
    manual_inputs: Optional[ManualInputState] = None,
    mode: str = "auto",
) -> Tuple[List[Email], List[Task], List[Negotiation], int, int, int]:
    """
    Generate the opening batch of emails, tasks, and negotiations.

    Supports:
    - AUTO mode: Uses scenario or random generation based on seed
    - MANUAL mode: Uses user-provided inputs with LLM parsing

    Args:
        config: Environment configuration
        rng: Seeded RNG for determinism
        difficulty: One of "easy", "medium", "hard"
        scenario_name: Name of scenario to use (auto mode)
        manual_inputs: ManualInputState with user emails/tasks
        mode: "auto" or "manual"

    Returns:
        (emails, tasks, negotiations, n_emails, n_tasks, n_deals)
    """
    if mode == "manual" and manual_inputs is not None:
        return _generate_manual_state(manual_inputs, rng)

    if scenario_name:
        return _generate_scenario_state(scenario_name, rng)

    return _generate_random_state(difficulty, rng)


def _generate_manual_state(
    manual_inputs: ManualInputState,
    rng: random.Random,
) -> Tuple[List[Email], List[Task], List[Negotiation], int, int, int]:
    """Generate state from manual user inputs."""
    emails: List[Email] = []
    tasks: List[Task] = []
    negotiations: List[Negotiation] = []

    # Track threads for context
    thread_emails: Dict[str, List[Dict]] = {}

    # Process manual emails
    email_data_list = manual_inputs.emails
    for i, email_data in enumerate(email_data_list):
        thread_id = email_data.get("thread_id", f"manual_{i}")
        timestamp = email_data.get("timestamp", i)
        text = email_data.get("text", "")

        # Get thread history for context
        thread_history = [
            e.get("text", "") for e in thread_emails.get(thread_id, [])
        ]

        # Parse with LLM (or fallback)
        parsed = parse_email(text, thread_history if thread_history else None)

        # Check escalation
        thread_so_far = thread_emails.get(thread_id, [])
        escalation_level = check_escalation_triggers(text, thread_so_far, 0)

        # Apply urgency override
        final_urgency = apply_urgency_override(parsed.urgency, escalation_level)

        email = Email(
            id=f"email_{i + 1}",
            text=text,
            sender=email_data.get("sender", "unknown@email.com"),
            subject=email_data.get("subject", "No subject"),
            thread_id=thread_id,
            timestamp=timestamp,
            escalation_level=escalation_level,
            urgency=final_urgency,
            sentiment=parsed.sentiment,
            requires_action=parsed.requires_action,
        )
        emails.append(email)

        # Track for thread context
        if thread_id not in thread_emails:
            thread_emails[thread_id] = []
        thread_emails[thread_id].append(email_data)

    # Process manual tasks
    task_data_list = manual_inputs.tasks
    for i, task_data in enumerate(task_data_list):
        priority_str = task_data.get("priority", "medium")
        priority = Priority.medium
        if priority_str == "high":
            priority = Priority.high
        elif priority_str == "low":
            priority = Priority.low

        tasks.append(
            Task(
                id=f"task_{i + 1}",
                name=task_data.get("name", f"Task {i + 1}"),
                hours_required=task_data.get("hours_required", 8.0),
                deadline=task_data.get("deadline", 5),
                priority=priority,
                effort=task_data.get("effort", 3),
                impact=task_data.get("impact", 1.0),
            )
        )

    # Generate some random negotiations (not typically manual)
    n_deals = min(2, len(email_data_list))  # Small number based on emails
    for i in range(n_deals):
        # B1 FIX: use seeded rng instead of global random.choice for determinism
        client = rng.choice(_CLIENTS) if _CLIENTS else "Client"
        offer_price = round(rng.uniform(5_000.0, 50_000.0), 2)
        min_price = round(offer_price * rng.uniform(0.6, 0.9), 2)
        quality = round(rng.uniform(0.5, 1.5), 2)
        deadline = rng.randint(3, 8)
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

    return emails, tasks, negotiations, len(emails), len(tasks), len(negotiations)


def _generate_scenario_state(
    scenario_name: str,
    rng: random.Random,
) -> Tuple[List[Email], List[Task], List[Negotiation], int, int, int]:
    """Generate state from a predefined scenario."""
    scenario = get_scenario(scenario_name)

    emails: List[Email] = []
    tasks: List[Task] = []
    negotiations: List[Negotiation] = []

    # Track threads for escalation detection
    thread_emails: Dict[str, List[Dict]] = {}

    # Process scenario emails
    email_data_list = scenario.get("emails", [])
    for i, email_data in enumerate(email_data_list):
        thread_id = email_data.get("thread_id", f"scenario_{i}")
        text = email_data.get("text", "")
        timestamp = i

        # Get thread history
        thread_history = [e.get("text", "") for e in thread_emails.get(thread_id, [])]

        # Parse (may use LLM or fallback)
        parsed = parse_email(text, thread_history if thread_history else None)

        # Check escalation in thread
        thread_so_far = thread_emails.get(thread_id, [])
        escalation_level = check_escalation_triggers(text, thread_so_far, 0)

        # Apply urgency override
        base_urgency_str = email_data.get("urgency", "medium")
        base_urgency = Urgency.medium
        if base_urgency_str == "high":
            base_urgency = Urgency.high
        elif base_urgency_str == "low":
            base_urgency = Urgency.low

        final_urgency = apply_urgency_override(base_urgency, escalation_level)

        sentiment_str = email_data.get("sentiment", "neutral")
        sentiment = Sentiment.neutral
        if sentiment_str == "positive":
            sentiment = Sentiment.positive
        elif sentiment_str == "negative":
            sentiment = Sentiment.negative

        email = Email(
            id=f"email_{i + 1}",
            text=text,
            sender=email_data.get("sender", "unknown@email.com"),
            subject=email_data.get("subject", "No subject"),
            thread_id=thread_id,
            timestamp=timestamp,
            escalation_level=escalation_level,
            urgency=final_urgency,
            sentiment=sentiment,
            requires_action=email_data.get("requires_action", True),
        )
        emails.append(email)

        # Track for thread
        if thread_id not in thread_emails:
            thread_emails[thread_id] = []
        thread_emails[thread_id].append(email_data)

    # Process scenario tasks
    task_data_list = scenario.get("tasks", [])
    for i, task_data in enumerate(task_data_list):
        priority_str = task_data.get("priority", "medium")
        priority = Priority.medium
        if priority_str == "high":
            priority = Priority.high
        elif priority_str == "low":
            priority = Priority.low

        tasks.append(
            Task(
                id=f"task_{i + 1}",
                name=task_data.get("name", f"Task {i + 1}"),
                hours_required=task_data.get("hours_required", 8.0),
                deadline=task_data.get("deadline", 5),
                priority=priority,
                effort=task_data.get("effort", 3),
                impact=task_data.get("impact", 1.0),
            )
        )

    # Process scenario negotiations
    neg_data_list = scenario.get("negotiations", [])
    for i, neg_data in enumerate(neg_data_list):
        negotiations.append(
            Negotiation(
                id=f"negotiation_{i + 1}",
                client=neg_data.get("client", "Client"),
                offer_price=neg_data.get("offer_price", 10000.0),
                min_price=neg_data.get("min_price", 7000.0),
                quality=neg_data.get("quality", 1.0),
                deadline=neg_data.get("deadline", 5),
            )
        )

    return emails, tasks, negotiations, len(emails), len(tasks), len(negotiations)


def _generate_random_state(
    difficulty: str,
    rng: random.Random,
) -> Tuple[List[Email], List[Task], List[Negotiation], int, int, int]:
    """Generate random state (original behavior preserved)."""
    cfg = DIFFICULTY_CONFIG.get(difficulty, DIFFICULTY_CONFIG["medium"])

    n_emails = rng.randint(*cfg["emails"])
    n_tasks = rng.randint(*cfg["tasks"])
    n_deals = rng.randint(*cfg["deals"])

    emails: List[Email] = []
    for i in range(n_emails):
        urgency = _URGENCIES[rng.randint(0, len(_URGENCIES) - 1)]
        sentiment = rng.choices(_SENTIMENTS, weights=[3, 4, 3], k=1)[0]
        sender = _SENDERS[rng.randint(0, len(_SENDERS) - 1)]
        subject = _SUBJECTS[rng.randint(0, len(_SUBJECTS) - 1)]
        emails.append(
            Email(
                id=f"email_{i + 1}",
                text=f"Email from {sender} about {subject}",
                sender=sender,
                subject=subject,
                thread_id=f"random_{i}",
                timestamp=i,
                escalation_level=0,
                urgency=urgency,
                sentiment=sentiment,
                requires_action=True,
            )
        )

    tasks: List[Task] = []
    for i in range(n_tasks):
        name = _TASK_NAMES[rng.randint(0, len(_TASK_NAMES) - 1)]
        hours = round(rng.uniform(4.0, 24.0), 1)
        deadline = rng.randint(2, 10)
        priority = _PRIORITIES[rng.randint(0, len(_PRIORITIES) - 1)]
        effort = rng.randint(1, 5)
        impact = round(rng.uniform(0.5, 2.0), 2)
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

    negotiations: List[Negotiation] = []
    for i in range(n_deals):
        client = _CLIENTS[rng.randint(0, len(_CLIENTS) - 1)]
        offer_price = round(rng.uniform(500.0, 20_000.0), 2)
        min_price = round(offer_price * rng.uniform(0.6, 0.9), 2)
        quality = round(rng.uniform(0.5, 1.5), 2)
        deadline = rng.randint(2, 8)
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
    Supports both AUTO and MANUAL modes.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        rng: random.Random,
        mode: str = "auto",
        manual_inputs: Optional[ManualInputState] = None,
    ) -> None:
        self.config = config
        self.rng = rng
        self.mode = mode
        self.manual_inputs = manual_inputs
        self._manual_email_index = 0

    def generate_events(
        self,
        email_counter: int,
        task_counter: int,
        negotiation_counter: int,
        current_step: int = 0,
    ) -> Tuple[List[Email], List[Task], List[Negotiation]]:
        """
        Generate new events for this step.

        In MANUAL mode: returns next batch of manual inputs if available
        In AUTO mode: probabilistically generates events
        """
        if self.mode == "manual" and self.manual_inputs and self.manual_inputs.has_more():
            return self._generate_manual_events(email_counter, current_step)

        return self._generate_auto_events(email_counter, task_counter, negotiation_counter, current_step)

    def _generate_manual_events(
        self,
        email_counter: int,
        current_step: int,
    ) -> Tuple[List[Email], List[Task], List[Negotiation]]:
        """Generate events from manual inputs."""
        new_emails: List[Email] = []
        new_tasks: List[Task] = []
        new_negotiations: List[Negotiation] = []

        # Get next batch of emails (up to 2 per step)
        email_batch = self.manual_inputs.get_next_emails(2) if self.manual_inputs else []

        for email_data in email_batch:
            self._manual_email_index += 1
            eid = email_counter + self._manual_email_index

            text = email_data.get("text", "")
            thread_id = email_data.get("thread_id", f"manual_{eid}")

            # Parse
            parsed = parse_email(text, None)

            email = Email(
                id=f"email_{eid}",
                text=text,
                sender=email_data.get("sender", "unknown@email.com"),
                subject=email_data.get("subject", "No subject"),
                thread_id=thread_id,
                timestamp=current_step,
                escalation_level=0,
                urgency=parsed.urgency,
                sentiment=parsed.sentiment,
                requires_action=parsed.requires_action,
            )
            new_emails.append(email)

        return new_emails, new_tasks, new_negotiations

    def _generate_auto_events(
        self,
        email_counter: int,
        task_counter: int,
        negotiation_counter: int,
        current_step: int = 0,
    ) -> Tuple[List[Email], List[Task], List[Negotiation]]:
        """Generate events probabilistically (original behavior)."""
        new_emails: List[Email] = []
        new_tasks: List[Task] = []
        new_negotiations: List[Negotiation] = []

        # Email
        if self.rng.random() < self.config.get("email_gen_prob", 0.3):
            eid = email_counter + 1
            urgency = _URGENCIES[self.rng.randint(0, len(_URGENCIES) - 1)]
            sentiment = self.rng.choices(_SENTIMENTS, weights=[3, 4, 3], k=1)[0]
            sender = _SENDERS[self.rng.randint(0, len(_SENDERS) - 1)]
            subject = _SUBJECTS[self.rng.randint(0, len(_SUBJECTS) - 1)]
            # B2 FIX: informational positive/neutral emails may not require action
            req_action = (sentiment == Sentiment.negative) or (urgency == Urgency.high)

            new_emails.append(
                Email(
                    id=f"email_{eid}",
                    text=f"Email from {sender}: {subject}",
                    sender=sender,
                    subject=subject,
                    thread_id=f"auto_{eid}",
                    timestamp=current_step,  # B3 FIX: use real step, not hardcoded 0
                    escalation_level=0,
                    urgency=urgency,
                    sentiment=sentiment,
                    requires_action=req_action,
                )
            )

        # Task
        if self.rng.random() < self.config.get("task_gen_prob", 0.2):
            tid = task_counter + 1
            name = _TASK_NAMES[self.rng.randint(0, len(_TASK_NAMES) - 1)]
            hours = round(self.rng.uniform(4.0, 24.0), 1)
            deadline = self.rng.randint(3, 8)
            priority = _PRIORITIES[self.rng.randint(0, len(_PRIORITIES) - 1)]
            effort = self.rng.randint(1, 5)
            impact = round(self.rng.uniform(0.5, 2.0), 2)
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

        # Negotiation
        if self.rng.random() < self.config.get("negotiation_gen_prob", 0.15):
            nid = negotiation_counter + 1
            client = _CLIENTS[self.rng.randint(0, len(_CLIENTS) - 1)]
            offer_price = round(self.rng.uniform(5_000.0, 50_000.0), 2)
            min_price = round(offer_price * self.rng.uniform(0.6, 0.9), 2)
            quality = round(self.rng.uniform(0.5, 1.5), 2)
            deadline = self.rng.randint(3, 6)
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
