"""
Predefined scenarios for StartupOpsEnv.

Each scenario contains realistic email threads, tasks, and negotiations
that simulate real startup situations.
"""
from __future__ import annotations

from typing import Dict, List


# ---------------------------------------------------------------------------
# Scenario: Investor Pressure
# ---------------------------------------------------------------------------
INVESTOR_PRESSURE = {
    "name": "investor_pressure",
    "description": "Lead investor is concerned about burn rate and wants updates",
    "emails": [
        {
            "sender": "sarah@bigvc.com",
            "subject": "Monthly update needed",
            "text": "Hi team, We're approaching the end of Q2 and I need the monthly metrics report. Can you send over burn rate, runway, and active user numbers by EOD? Thanks, Sarah",
            "thread_id": "investor_sarah",
            "urgency": "medium",
            "sentiment": "neutral",
            "requires_action": True,
        },
        {
            "sender": "sarah@bigvc.com",
            "subject": "Re: Monthly update needed",
            "text": "Following up on my previous email. Still waiting for those numbers. The board meeting is tomorrow and I need to be prepared. Please send ASAP.",
            "thread_id": "investor_sarah",
            "urgency": "medium",
            "sentiment": "neutral",
            "requires_action": True,
        },
        {
            "sender": "sarah@bigvc.com",
            "subject": "URGENT: Board meeting in 2 hours",
            "text": "I still haven't received the metrics. This is unacceptable. I'm escalating this to the board. We need to discuss your reporting processes urgently.",
            "thread_id": "investor_sarah",
            "urgency": "high",
            "sentiment": "negative",
            "requires_action": True,
        },
        {
            "sender": "mike@angel.co",
            "subject": "Intro to potential customer",
            "text": "Hey! Just met someone at TechCrunch Disrupt who needs exactly what you're building. Want an intro? They're a Fortune 500 looking for a pilot.",
            "thread_id": "intro_mike",
            "urgency": "low",
            "sentiment": "positive",
            "requires_action": True,
        },
    ],
    "tasks": [
        {
            "name": "Prepare Board Metrics Deck",
            "hours_required": 8.0,
            "deadline": 3,
            "priority": "high",
            "effort": 4,
            "impact": 2.0,
        },
        {
            "name": "Update Financial Model",
            "hours_required": 6.0,
            "deadline": 5,
            "priority": "high",
            "effort": 3,
            "impact": 1.5,
        },
    ],
    "negotiations": [
        {
            "client": "Enterprise Solutions Inc",
            "offer_price": 50000.0,
            "min_price": 35000.0,
            "quality": 1.2,
            "deadline": 7,
        },
    ],
}


# ---------------------------------------------------------------------------
# Scenario: Vendor Delay
# ---------------------------------------------------------------------------
VENDOR_DELAY = {
    "name": "vendor_delay",
    "description": "Critical vendor is delaying delivery, blocking product launch",
    "emails": [
        {
            "sender": "support@cloudhost.com",
            "subject": "Scheduled maintenance notification",
            "text": "Dear customer, We will be performing routine maintenance this weekend. No action required. Services may experience brief interruptions.",
            "thread_id": "vendor_cloudhost",
            "urgency": "low",
            "sentiment": "neutral",
            "requires_action": False,
        },
        {
            "sender": "jake@devshop.io",
            "subject": "Delay in API integration",
            "text": "Hey, hitting some snags with the third-party API integration. Need another week to sort it out. Will update you by Friday.",
            "thread_id": "vendor_devshop",
            "urgency": "medium",
            "sentiment": "neutral",
            "requires_action": True,
        },
        {
            "sender": "jake@devshop.io",
            "subject": "Re: Delay in API integration",
            "text": "Still debugging. The API documentation was outdated and we're dealing with rate limits. This is pushing our timeline.",
            "thread_id": "vendor_devshop",
            "urgency": "medium",
            "sentiment": "negative",
            "requires_action": True,
        },
        {
            "sender": "jake@devshop.io",
            "subject": "Still waiting on your feedback",
            "text": "It's been 3 days since my last update. We need your go-ahead on the workaround approach. Every day of delay costs us both money. Still waiting...",
            "thread_id": "vendor_devshop",
            "urgency": "high",
            "sentiment": "negative",
            "requires_action": True,
        },
        {
            "sender": "ceo@customer.com",
            "subject": "Launch timeline question",
            "text": "Our CEO is asking when the integration will be live. We have a marketing campaign scheduled. Any updates?",
            "thread_id": "customer_ceo",
            "urgency": "high",
            "sentiment": "neutral",
            "requires_action": True,
        },
    ],
    "tasks": [
        {
            "name": "Review API Workaround",
            "hours_required": 4.0,
            "deadline": 2,
            "priority": "high",
            "effort": 3,
            "impact": 1.8,
        },
        {
            "name": "Update Customer Timeline",
            "hours_required": 2.0,
            "deadline": 3,
            "priority": "high",
            "effort": 2,
            "impact": 1.5,
        },
        {
            "name": "Find Backup Vendor",
            "hours_required": 6.0,
            "deadline": 5,
            "priority": "medium",
            "effort": 4,
            "impact": 1.5,
        },
    ],
    "negotiations": [
        {
            "client": "DevShop.io",
            "offer_price": 25000.0,
            "min_price": 20000.0,
            "quality": 0.9,
            "deadline": 5,
        },
        {
            "client": "CloudHost Premium",
            "offer_price": 15000.0,
            "min_price": 12000.0,
            "quality": 1.0,
            "deadline": 10,
        },
    ],
}


# ---------------------------------------------------------------------------
# Scenario: Customer Churn
# ---------------------------------------------------------------------------
CUSTOMER_CHURN = {
    "name": "customer_churn",
    "description": "Key enterprise customer is considering leaving",
    "emails": [
        {
            "sender": "cto@enterprise.com",
            "subject": "Performance concerns",
            "text": "We've been experiencing frequent downtime over the past month. Our team is losing confidence in the platform. Can you explain what's happening?",
            "thread_id": "enterprise_cto",
            "urgency": "high",
            "sentiment": "negative",
            "requires_action": True,
        },
        {
            "sender": "cto@enterprise.com",
            "subject": "Re: Performance concerns",
            "text": "Still waiting for your response. This is affecting our operations. We need a call ASAP to discuss our contract.",
            "thread_id": "enterprise_cto",
            "urgency": "high",
            "sentiment": "negative",
            "requires_action": True,
        },
        {
            "sender": "ops@enterprise.com",
            "subject": "Support ticket #4521",
            "text": "Ticket #4521 has been open for 5 days with no resolution. Our users are complaining daily. This is unacceptable for an enterprise account.",
            "thread_id": "enterprise_ops",
            "urgency": "high",
            "sentiment": "negative",
            "requires_action": True,
        },
        {
            "sender": "success@startup.io",
            "subject": "Weekly check-in",
            "text": "Hope you're doing well! Just checking in to see if you need any training for the new features we released. Happy to help!",
            "thread_id": "success_outbound",
            "urgency": "low",
            "sentiment": "positive",
            "requires_action": False,
        },
        {
            "sender": "billing@enterprise.com",
            "subject": "Invoice question",
            "text": "Can you clarify line item 3 on last month's invoice? The usage charges seem higher than expected.",
            "thread_id": "enterprise_billing",
            "urgency": "medium",
            "sentiment": "neutral",
            "requires_action": True,
        },
    ],
    "tasks": [
        {
            "name": "Emergency Performance Audit",
            "hours_required": 10.0,
            "deadline": 2,
            "priority": "high",
            "effort": 5,
            "impact": 2.5,
        },
        {
            "name": "Executive Response to CTO",
            "hours_required": 2.0,
            "deadline": 1,
            "priority": "high",
            "effort": 2,
            "impact": 2.0,
        },
        {
            "name": "Resolve Support Ticket #4521",
            "hours_required": 4.0,
            "deadline": 2,
            "priority": "high",
            "effort": 3,
            "impact": 1.8,
        },
    ],
    "negotiations": [
        {
            "client": "Enterprise Co",
            "offer_price": 100000.0,
            "min_price": 80000.0,
            "quality": 1.5,
            "deadline": 10,
        },
    ],
}


# ---------------------------------------------------------------------------
# Scenario: Hiring Crunch
# ---------------------------------------------------------------------------
HIRING_CRUNCH = {
    "name": "hiring_crunch",
    "description": "Key engineer quit, critical features blocked, need to hire fast",
    "emails": [
        {
            "sender": "alex@talent.io",
            "subject": "Found a great candidate",
            "text": "Hi! I found a senior backend engineer with 8 years experience. They're interviewing with 3 other companies. Want me to set up a call?",
            "thread_id": "recruiter_alex",
            "urgency": "medium",
            "sentiment": "positive",
            "requires_action": True,
        },
        {
            "sender": "alex@talent.io",
            "subject": "Re: Found a great candidate",
            "text": "Following up - the candidate has an offer from a competitor expiring tomorrow. If you're interested, we need to move fast.",
            "thread_id": "recruiter_alex",
            "urgency": "high",
            "sentiment": "neutral",
            "requires_action": True,
        },
        {
            "sender": "team@startup.io",
            "subject": "Sprint planning update",
            "text": "With Jordan's departure, we're reassessing Q3 priorities. The authentication refactor is blocked. Need leadership input.",
            "thread_id": "internal_team",
            "urgency": "medium",
            "sentiment": "neutral",
            "requires_action": True,
        },
        {
            "sender": "candidate@email.com",
            "subject": "Interview follow-up",
            "text": "Hi, I interviewed last week for the PM role. Just wondering about next steps? I have another offer I'm considering.",
            "thread_id": "candidate_pm",
            "urgency": "medium",
            "sentiment": "neutral",
            "requires_action": True,
        },
        {
            "sender": "contractor@dev.io",
            "subject": "Available for contract work",
            "text": "Heard you're looking for temporary help. I'm available for 20hrs/week for the next month. Rate is $150/hr. Let me know if interested.",
            "thread_id": "contractor_outbound",
            "urgency": "low",
            "sentiment": "neutral",
            "requires_action": True,
        },
    ],
    "tasks": [
        {
            "name": "Interview Senior Backend Candidate",
            "hours_required": 3.0,
            "deadline": 1,
            "priority": "high",
            "effort": 3,
            "impact": 2.0,
        },
        {
            "name": "Review PM Candidate Application",
            "hours_required": 2.0,
            "deadline": 2,
            "priority": "medium",
            "effort": 2,
            "impact": 1.5,
        },
        {
            "name": "Auth Refactor Sprint Planning",
            "hours_required": 4.0,
            "deadline": 3,
            "priority": "high",
            "effort": 3,
            "impact": 1.8,
        },
    ],
    "negotiations": [
        {
            "client": "Contractor Dev.io",
            "offer_price": 12000.0,
            "min_price": 8000.0,
            "quality": 0.8,
            "deadline": 5,
        },
        {
            "client": "Talent Agency",
            "offer_price": 8000.0,
            "min_price": 6000.0,
            "quality": 0.7,
            "deadline": 3,
        },
    ],
}


# ---------------------------------------------------------------------------
# Scenario Registry
# ---------------------------------------------------------------------------

SCENARIOS: Dict[str, Dict] = {
    "investor_pressure": INVESTOR_PRESSURE,
    "vendor_delay": VENDOR_DELAY,
    "customer_churn": CUSTOMER_CHURN,
    "hiring_crunch": HIRING_CRUNCH,
}


def get_scenario(name: str) -> Dict:
    """Get a scenario by name. Returns investor_pressure if not found."""
    return SCENARIOS.get(name, INVESTOR_PRESSURE)


def list_scenarios() -> List[str]:
    """List all available scenario names."""
    return list(SCENARIOS.keys())
