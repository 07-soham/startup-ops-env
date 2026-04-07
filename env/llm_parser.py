"""
LLM-based email parser for StartupOpsEnv.

Parses raw email text to structured attributes (urgency, sentiment, requires_action)
using an LLM with fallback to deterministic keyword parsing.

Design constraints:
- temperature = 0 for determinism
- Uses thread history as context
- Validates JSON output
- Retries on failure
- Falls back to keyword parser if LLM unavailable
"""
from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Optional

from .models import ParsedEmail, Sentiment, Urgency


# ---------------------------------------------------------------------------
# Keyword-based fallback parser (deterministic)
# ---------------------------------------------------------------------------

_URGENT_KEYWORDS = [
    "urgent", "asap", "immediately", "critical", "emergency",
    "deadline", "expiring", "today", "eod", "end of day",
    "blocking", "blocked", "can't proceed", "waiting",
]

_NEGATIVE_KEYWORDS = [
    "unacceptable", "frustrated", "disappointed", "angry",
    "terrible", "awful", "worst", "hate", "useless",
    "waste", "failed", "failure", "broken", "complaint",
]

_POSITIVE_KEYWORDS = [
    "great", "excellent", "amazing", "love", "perfect",
    "thanks", "appreciate", "pleased", "happy", "excited",
    "awesome", "fantastic", "wonderful", "outstanding",
]

_ESCALATION_KEYWORDS = [
    "following up", "still waiting", "repeated", "multiple times",
    "escalating", "escalate", "manager", "ceo", "board",
    "legal", "lawyer", "complaint", "unacceptable",
]


def _keyword_parse(text: str) -> ParsedEmail:
    """
    Deterministic keyword-based parser.
    Used as fallback when LLM is unavailable.
    """
    text_lower = text.lower()

    # Count keyword matches
    urgent_matches = sum(1 for kw in _URGENT_KEYWORDS if kw in text_lower)
    negative_matches = sum(1 for kw in _NEGATIVE_KEYWORDS if kw in text_lower)
    positive_matches = sum(1 for kw in _POSITIVE_KEYWORDS if kw in text_lower)
    escalation_matches = sum(1 for kw in _ESCALATION_KEYWORDS if kw in text_lower)

    # Determine urgency
    if urgent_matches >= 2 or escalation_matches >= 2:
        urgency = Urgency.high
    elif urgent_matches >= 1 or "urgent" in text_lower:
        urgency = Urgency.medium
    else:
        urgency = Urgency.low

    # Determine sentiment
    if negative_matches > positive_matches:
        sentiment = Sentiment.negative
    elif positive_matches > negative_matches:
        sentiment = Sentiment.positive
    else:
        sentiment = Sentiment.neutral

    # Determine if action is required
    requires_action = True
    info_patterns = [
        r"no action required",
        r"for your information",
        r"fyi",
        r"just a heads up",
        r"routine maintenance",
        r"automatic",
    ]
    for pattern in info_patterns:
        if re.search(pattern, text_lower):
            requires_action = False
            break

    return ParsedEmail(
        urgency=urgency,
        sentiment=sentiment,
        requires_action=requires_action,
        confidence=0.7,  # Lower confidence for keyword method
    )


# ---------------------------------------------------------------------------
# LLM Parser
# ---------------------------------------------------------------------------

class LLMParser:
    """
    LLM-based email parser with keyword fallback.

    Uses Anthropic's Claude API or OpenAI's GPT API if available,
    otherwise falls back to deterministic keyword parsing.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        provider: Optional[str] = None,  # "anthropic", "openai", or None (auto)
    ):
        self.anthropic_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.openai_key = os.environ.get("OPENAI_API_KEY")
        self.provider = provider
        self.client = None

        # Determine which provider to use
        self.provider = self._select_provider()
        self.use_llm = self.provider is not None

        if self.use_llm:
            self._init_client()

    def _select_provider(self) -> Optional[str]:
        """Select LLM provider based on available API keys."""
        # If explicitly specified, use that
        if self.provider == "anthropic" and self.anthropic_key:
            return "anthropic"
        if self.provider == "openai" and self.openai_key:
            return "openai"

        # Auto-select: prefer Anthropic, fallback to OpenAI
        if self.anthropic_key:
            try:
                import anthropic
                return "anthropic"
            except ImportError:
                pass

        if self.openai_key:
            try:
                import openai
                return "openai"
            except ImportError:
                pass

        return None

    def _init_client(self):
        """Initialize the LLM client."""
        if self.provider == "anthropic":
            try:
                import anthropic
                self.client = anthropic.Anthropic(api_key=self.anthropic_key)
            except ImportError:
                self.use_llm = False
        elif self.provider == "openai":
            try:
                import openai
                self.client = openai.OpenAI(api_key=self.openai_key)
            except ImportError:
                self.use_llm = False

    def _build_prompt(self, email_text: str, thread_history: List[str]) -> str:
        """Build the prompt for the LLM."""
        context = ""
        if thread_history:
            context = "\n\nPrevious emails in this thread:\n"
            for i, prev in enumerate(thread_history[-2:], 1):  # Last 2 emails
                context += f"{i}. {prev[:500]}...\n"

        prompt = f"""You are an email parsing assistant for a startup operations environment.

Analyze the following email and extract three attributes:
1. urgency: "low", "medium", or "high" based on time sensitivity and importance
2. sentiment: "positive", "neutral", or "negative" based on tone
3. requires_action: true if the email needs a response or action, false if informational only

Current email to analyze:
---
{email_text}
---{context}

Respond ONLY with a JSON object in this exact format:
{{"urgency": "low|medium|high", "sentiment": "positive|neutral|negative", "requires_action": true|false}}

Rules:
- Urgency is HIGH if: deadlines mentioned, repeated follow-ups, escalation language, "ASAP", "urgent"
- Urgency is MEDIUM if: time-sensitive but not critical, scheduled meetings, requests
- Urgency is LOW if: informational, no deadline, casual check-ins
- Sentiment is NEGATIVE if: complaints, frustration, criticism, threats
- Sentiment is POSITIVE if: praise, excitement, gratitude
- requires_action is FALSE if: automated notifications, FYI, "no action needed"
"""
        return prompt

    def _parse_llm_response(self, response_text: str) -> ParsedEmail:
        """Parse and validate LLM response."""
        try:
            # Extract JSON from response
            # Handle both plain JSON and markdown code blocks
            json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = response_text.strip()

            data = json.loads(json_str)

            # Validate fields
            urgency_str = data.get("urgency", "low").lower()
            sentiment_str = data.get("sentiment", "neutral").lower()
            requires_action = data.get("requires_action", True)

            # Map to enums
            urgency = Urgency.low
            if urgency_str == "high":
                urgency = Urgency.high
            elif urgency_str == "medium":
                urgency = Urgency.medium

            sentiment = Sentiment.neutral
            if sentiment_str == "positive":
                sentiment = Sentiment.positive
            elif sentiment_str == "negative":
                sentiment = Sentiment.negative

            return ParsedEmail(
                urgency=urgency,
                sentiment=sentiment,
                requires_action=requires_action,
                confidence=0.95,
            )
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            # Fallback to keyword parser on parse error
            return _keyword_parse(response_text)

    def _call_anthropic(self, prompt: str) -> str:
        """Call Anthropic Claude API."""
        response = self.client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=150,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    def _call_openai(self, prompt: str) -> str:
        """Call OpenAI GPT API."""
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            max_tokens=150,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content or ""

    def parse(
        self,
        email_text: str,
        thread_history: Optional[List[str]] = None,
        max_retries: int = 2,
    ) -> ParsedEmail:
        """
        Parse an email using LLM or fallback to keywords.

        Args:
            email_text: The email content to parse
            thread_history: List of previous emails in the thread (for context)
            max_retries: Number of retries on LLM failure

        Returns:
            ParsedEmail with urgency, sentiment, requires_action
        """
        if not self.use_llm or self.client is None:
            return _keyword_parse(email_text)

        thread_history = thread_history or []
        prompt = self._build_prompt(email_text, thread_history)

        for attempt in range(max_retries + 1):
            try:
                if self.provider == "anthropic":
                    response_text = self._call_anthropic(prompt)
                elif self.provider == "openai":
                    response_text = self._call_openai(prompt)
                else:
                    return _keyword_parse(email_text)

                return self._parse_llm_response(response_text)

            except Exception:
                if attempt == max_retries:
                    # Fall back to keyword parser on final failure
                    return _keyword_parse(email_text)
                continue

        return _keyword_parse(email_text)


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------

_parser_instance: Optional[LLMParser] = None


def get_parser(api_key: Optional[str] = None) -> LLMParser:
    """Get or create singleton parser instance."""
    global _parser_instance
    if _parser_instance is None:
        _parser_instance = LLMParser(api_key=api_key)
    return _parser_instance


def parse_email(
    email_text: str,
    thread_history: Optional[List[str]] = None,
    api_key: Optional[str] = None,
) -> ParsedEmail:
    """
    Parse an email to structured attributes.

    Uses LLM if available, otherwise falls back to keyword parsing.
    Deterministic and reproducible.
    """
    parser = get_parser(api_key=api_key)
    return parser.parse(email_text, thread_history)


# ---------------------------------------------------------------------------
# Escalation detection
# ---------------------------------------------------------------------------

def check_escalation_triggers(
    email_text: str,
    thread_emails: List[Dict],
    current_level: int = 0,
) -> int:
    """
    Check if email should be escalated based on content and thread history.

    Args:
        email_text: Current email text
        thread_emails: All emails in the thread
        current_level: Current escalation level

    Returns:
        New escalation level (may be same or higher)
    """
    text_lower = email_text.lower()
    triggers = 0

    # Check for escalation keywords
    for keyword in _ESCALATION_KEYWORDS:
        if keyword in text_lower:
            triggers += 1

    # Check for negative sentiment
    negative_count = sum(1 for kw in _NEGATIVE_KEYWORDS if kw in text_lower)
    if negative_count >= 2:
        triggers += 1

    # Check for repeated follow-ups (multiple emails in thread)
    if len(thread_emails) >= 3:
        triggers += 1

    # Check if previous emails were unanswered
    if len(thread_emails) >= 2:
        # Check time gap (simulated by email count for now)
        triggers += 1

    # Calculate new escalation level
    if triggers >= 3 and current_level < 2:
        return current_level + 1
    elif triggers >= 2 and current_level < 1:
        return current_level + 1

    return current_level


def apply_urgency_override(
    base_urgency: Urgency,
    escalation_level: int,
) -> Urgency:
    """
    Apply urgency overrides based on escalation level.

    Rules:
    - level 0: Keep LLM/keyword output
    - level 1: Minimum "medium"
    - level >=2: Force "high"
    """
    if escalation_level >= 2:
        return Urgency.high
    elif escalation_level == 1:
        if base_urgency == Urgency.low:
            return Urgency.medium
        return base_urgency
    return base_urgency
