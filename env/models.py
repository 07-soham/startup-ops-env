"""
Pydantic models for StartupOpsEnv.

All internal state is represented as structured Python objects —
never raw dicts.
"""
from __future__ import annotations

from enum import Enum
from typing import Dict, List

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Urgency(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"


class Sentiment(str, Enum):
    positive = "positive"
    neutral = "neutral"
    negative = "negative"


class Priority(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"


# ---------------------------------------------------------------------------
# Domain objects
# ---------------------------------------------------------------------------

class Email(BaseModel):
    """Represents an incoming email that needs to be handled."""

    model_config = ConfigDict(validate_assignment=False)

    id: str
    sender: str
    subject: str
    urgency: Urgency
    sentiment: Sentiment = Sentiment.neutral   # tone of the email
    requires_action: bool = True               # False = informational only


class Task(BaseModel):
    """Represents an operational task with a deadline and hour cost."""

    model_config = ConfigDict(validate_assignment=False)

    id: str
    name: str
    hours_required: float
    deadline: int               # steps remaining until it expires
    assigned: bool = False      # True once assign_task is called
    missed: bool = False        # True if deadline expired without assignment

    priority: Priority = Priority.medium   # scheduling urgency signal
    effort: int = 1                        # relative effort score (1–5)
    impact: float = 1.0                    # reward multiplier when completed


class Negotiation(BaseModel):
    """Represents an ongoing deal negotiation with a client."""

    model_config = ConfigDict(validate_assignment=False)

    id: str
    client: str
    offer_price: float
    deadline: int               # steps before the offer lapses

    min_price: float = 0.0     # client's price floor — negotiating below this
                                # causes the deal to collapse
    quality: float = 1.0       # deal quality — multiplies accept reward


# ---------------------------------------------------------------------------
# Environment state
# ---------------------------------------------------------------------------

class EnvState(BaseModel):
    """Complete mutable state of the environment."""

    model_config = ConfigDict(validate_assignment=False)

    budget: float
    satisfaction: float
    team_hours: float
    revenue: float

    emails: List[Email] = Field(default_factory=list)
    tasks: List[Task] = Field(default_factory=list)
    negotiations: List[Negotiation] = Field(default_factory=list)

    # Counters used by grader
    missed_tasks: int = 0
    replied_emails: int = 0
    accepted_negotiations: int = 0
    rejected_negotiations: int = 0
    collapsed_negotiations: int = 0   # deals lost by negotiating below min_price

    step: int = 0


# ---------------------------------------------------------------------------
# Observation (returned to agent each step)
# ---------------------------------------------------------------------------

class Observation(BaseModel):
    """
    Read-only snapshot derived from EnvState via _get_obs().
    Contains all information an agent needs to decide its next action.
    """

    model_config = ConfigDict(validate_assignment=False)

    # Scalar resources
    budget: float
    satisfaction: float
    team_hours: float
    revenue: float

    # Counts
    num_emails: int
    num_tasks: int
    num_negotiations: int
    missed_tasks: int
    step: int

    # ID lists
    email_ids: List[str] = Field(default_factory=list)
    unassigned_task_ids: List[str] = Field(default_factory=list)
    negotiation_ids: List[str] = Field(default_factory=list)

    # Derived helpers
    high_urgency_emails: List[str] = Field(default_factory=list)
    action_required_emails: List[str] = Field(default_factory=list)   # requires_action=True
    overdue_tasks: List[str] = Field(default_factory=list)            # deadline <= 2

    # Lookup maps — basic
    task_deadlines: Dict[str, int] = Field(default_factory=dict)
    negotiation_offers: Dict[str, float] = Field(default_factory=dict)
    email_urgencies: Dict[str, str] = Field(default_factory=dict)

    # Lookup maps — extended (from new fields)
    email_sentiments: Dict[str, str] = Field(default_factory=dict)
    task_priorities: Dict[str, str] = Field(default_factory=dict)
    task_impacts: Dict[str, float] = Field(default_factory=dict)
    negotiation_min_prices: Dict[str, float] = Field(default_factory=dict)
    negotiation_qualities: Dict[str, float] = Field(default_factory=dict)
