"""
Comprehensive tests for StartupOpsEnv.

Tests cover:
- Determinism (auto + manual modes)
- Context-aware parsing
- Thread consistency
- Escalation logic
- Urgency overrides
- Manual input handling
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from env.core import StartupOpsEnv
from env.generator import ManualInputState, generate_initial_state
from env.llm_parser import (
    apply_urgency_override,
    check_escalation_triggers,
    parse_email,
)
from env.models import Email, ParsedEmail, Priority, Sentiment, Task, Urgency
from env.scenarios import get_scenario, list_scenarios


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def base_config():
    """Minimal config for testing."""
    return {
        "seed": 42,
        "max_steps": 50,
        "initial_budget": 100_000,
        "initial_satisfaction": 0.8,
        "initial_team_hours": 100.0,
        "email_gen_prob": 0.0,  # Disable auto-gen for tests
        "task_gen_prob": 0.0,
        "negotiation_gen_prob": 0.0,
        "max_emails": 20,
        "max_tasks": 20,
        "max_negotiations": 10,
        "reply_satisfaction_boost": 0.1,
        "high_urgency_ignore_penalty": -2.0,
        "accept_offer_budget_cost": 1000.0,
        "negotiate_adjustment": 0.9,
        "task_miss_penalty": -5.0,
        "step_survival_reward": 0.1,
    }


@pytest.fixture
def rng():
    """Seeded RNG for determinism tests."""
    return random.Random(42)


# -----------------------------------------------------------------------------
# Determinism Tests
# -----------------------------------------------------------------------------

class TestDeterminism:
    """Ensure same seed produces same state."""

    def test_auto_mode_determinism(self, base_config, rng):
        """Same seed should produce identical state in auto mode."""
        rng1 = random.Random(42)
        rng2 = random.Random(42)

        emails1, tasks1, negs1, n1, t1, d1 = generate_initial_state(
            base_config, rng1, difficulty="medium", mode="auto"
        )
        emails2, tasks2, negs2, n2, t2, d2 = generate_initial_state(
            base_config, rng2, difficulty="medium", mode="auto"
        )

        assert len(emails1) == len(emails2)
        assert len(tasks1) == len(tasks2)
        assert len(negs1) == len(negs2)

        for e1, e2 in zip(emails1, emails2):
            assert e1.sender == e2.sender
            assert e1.urgency == e2.urgency
            assert e1.sentiment == e2.sentiment

    def test_scenario_determinism(self, base_config, rng):
        """Same scenario should produce consistent state."""
        rng1 = random.Random(42)
        rng2 = random.Random(99)  # Different seed, but scenario should dominate

        emails1, tasks1, negs1, _, _, _ = generate_initial_state(
            base_config, rng1, scenario_name="investor_pressure", mode="auto"
        )
        emails2, tasks2, negs2, _, _, _ = generate_initial_state(
            base_config, rng2, scenario_name="investor_pressure", mode="auto"
        )

        # Same scenario should have same structure
        assert len(emails1) == len(emails2)
        assert len(tasks1) == len(tasks2)

    def test_manual_mode_determinism(self, base_config, rng):
        """Same manual inputs should produce same state."""
        manual_inputs = ManualInputState(
            emails=[
                {
                    "sender": "test@test.com",
                    "subject": "Test",
                    "text": "This is urgent!",
                    "thread_id": "test_thread",
                    "timestamp": 0,
                }
            ],
            tasks=[
                {
                    "name": "Test Task",
                    "hours_required": 4.0,
                    "deadline": 3,
                    "priority": "high",
                    "effort": 3,
                    "impact": 1.5,
                }
            ],
        )

        rng1 = random.Random(42)
        rng2 = random.Random(42)

        emails1, tasks1, negs1, _, _, _ = generate_initial_state(
            base_config, rng1, manual_inputs=manual_inputs, mode="manual"
        )
        emails2, tasks2, negs2, _, _, _ = generate_initial_state(
            base_config, rng2, manual_inputs=manual_inputs, mode="manual"
        )

        assert len(emails1) == len(emails2)
        assert len(tasks1) == len(tasks2)


# -----------------------------------------------------------------------------
# Scenario Tests
# -----------------------------------------------------------------------------

class TestScenarios:
    """Test scenario library."""

    def test_all_scenarios_exist(self):
        """All expected scenarios should be available."""
        scenarios = list_scenarios()
        expected = ["investor_pressure", "vendor_delay", "customer_churn", "hiring_crunch"]
        for exp in expected:
            assert exp in scenarios

    def test_scenario_structure(self):
        """Scenarios should have required fields."""
        for name in list_scenarios():
            scenario = get_scenario(name)
            assert "name" in scenario
            assert "description" in scenario
            assert "emails" in scenario
            assert "tasks" in scenario
            assert "negotiations" in scenario

    def test_investor_pressure_emails(self, base_config, rng):
        """Investor pressure scenario should have follow-up emails."""
        emails, tasks, _, _, _, _ = generate_initial_state(
            base_config, rng, scenario_name="investor_pressure", mode="auto"
        )

        # Should have multiple emails
        assert len(emails) >= 3

        # Should have thread groupings
        thread_ids = {e.thread_id for e in emails}
        assert len(thread_ids) >= 1

        # Should have at least one high urgency
        high_urgency = [e for e in emails if e.urgency == Urgency.high]
        assert len(high_urgency) >= 1

    def test_vendor_delay_escalation(self, base_config, rng):
        """Vendor delay should show escalation in thread."""
        emails, tasks, _, _, _, _ = generate_initial_state(
            base_config, rng, scenario_name="vendor_delay", mode="auto"
        )

        # Find thread with multiple emails
        from collections import Counter
        thread_counts = Counter(e.thread_id for e in emails)
        multi_threads = [t for t, c in thread_counts.items() if c >= 3]

        assert len(multi_threads) >= 1, "Should have thread with follow-ups"


# -----------------------------------------------------------------------------
# Thread and Context Tests
# -----------------------------------------------------------------------------

class TestThreadConsistency:
    """Test thread grouping and context."""

    def test_emails_grouped_by_thread(self, base_config, rng):
        """Emails with same thread_id should be grouped."""
        manual_inputs = ManualInputState(
            emails=[
                {"sender": "a@test.com", "subject": "Thread A", "text": "First", "thread_id": "thread_1"},
                {"sender": "b@test.com", "subject": "Thread B", "text": "Other", "thread_id": "thread_2"},
                {"sender": "a@test.com", "subject": "Re: Thread A", "text": "Second", "thread_id": "thread_1"},
            ]
        )

        emails, _, _, _, _, _ = generate_initial_state(
            base_config, rng, manual_inputs=manual_inputs, mode="manual"
        )

        thread_1_emails = [e for e in emails if e.thread_id == "thread_1"]
        assert len(thread_1_emails) == 2

    def test_timestamps_sequential(self, base_config, rng):
        """Timestamps should increase sequentially."""
        manual_inputs = ManualInputState(
            emails=[
                {"sender": "a@test.com", "subject": "1", "text": "First", "timestamp": 0},
                {"sender": "a@test.com", "subject": "2", "text": "Second", "timestamp": 1},
                {"sender": "a@test.com", "subject": "3", "text": "Third", "timestamp": 2},
            ]
        )

        emails, _, _, _, _, _ = generate_initial_state(
            base_config, rng, manual_inputs=manual_inputs, mode="manual"
        )

        timestamps = [e.timestamp for e in emails]
        assert timestamps == sorted(timestamps)


# -----------------------------------------------------------------------------
# Escalation Tests
# -----------------------------------------------------------------------------

class TestEscalation:
    """Test escalation logic."""

    def test_follow_up_escalation(self):
        """Repeated follow-ups should escalate."""
        text = "Still waiting for your response. This is the third time I've asked."
        thread = [
            {"text": "First email"},
            {"text": "Second email - no response"},
        ]

        level = check_escalation_triggers(text, thread, current_level=0)
        assert level >= 1

    def test_negative_sentiment_escalation(self):
        """Negative sentiment should trigger escalation."""
        text = "This is completely unacceptable. I'm escalating this to your CEO."
        thread = []

        level = check_escalation_triggers(text, thread, current_level=0)
        assert level >= 1

    def test_escalation_growth(self):
        """Multiple triggers should increase escalation level."""
        text = "Following up on my previous emails. Still waiting for a response. This is urgent!"
        thread = [{"text": "First"}, {"text": "Second"}, {"text": "Third"}]

        level = check_escalation_triggers(text, thread, current_level=0)
        assert level >= 1


# -----------------------------------------------------------------------------
# Urgency Override Tests
# -----------------------------------------------------------------------------

class TestUrgencyOverride:
    """Test urgency escalation overrides."""

    def test_level_0_no_override(self):
        """Level 0 should preserve base urgency."""
        result = apply_urgency_override(Urgency.low, escalation_level=0)
        assert result == Urgency.low

    def test_level_1_boosts_low(self):
        """Level 1 should boost low to medium."""
        result = apply_urgency_override(Urgency.low, escalation_level=1)
        assert result == Urgency.medium

    def test_level_1_preserves_high(self):
        """Level 1 should preserve high urgency."""
        result = apply_urgency_override(Urgency.high, escalation_level=1)
        assert result == Urgency.high

    def test_level_2_forces_high(self):
        """Level 2 should force high urgency regardless of base."""
        result = apply_urgency_override(Urgency.low, escalation_level=2)
        assert result == Urgency.high

        result = apply_urgency_override(Urgency.medium, escalation_level=2)
        assert result == Urgency.high


# -----------------------------------------------------------------------------
# Parser Tests
# -----------------------------------------------------------------------------

class TestLLMParser:
    """Test LLM and keyword parser."""

    def test_keyword_parser_urgency(self):
        """Keyword parser should detect urgency."""
        text = "This is urgent! We need this ASAP!"
        result = parse_email(text)

        assert result.urgency in [Urgency.medium, Urgency.high]

    def test_keyword_parser_sentiment(self):
        """Keyword parser should detect sentiment."""
        negative = parse_email("This is terrible and frustrating!")
        assert negative.sentiment == Sentiment.negative

        positive = parse_email("This is great and amazing!")
        assert positive.sentiment == Sentiment.positive

    def test_parser_with_context(self):
        """Parser should use thread context."""
        current = "Following up on my previous email."
        history = ["Original request sent last week."]

        result = parse_email(current, history)
        assert isinstance(result, ParsedEmail)

    def test_parser_confidence(self):
        """Parser should return confidence score."""
        result = parse_email("Test email")
        assert 0.0 <= result.confidence <= 1.0


# -----------------------------------------------------------------------------
# Environment Integration Tests
# -----------------------------------------------------------------------------

class TestEnvironmentIntegration:
    """Test full environment with new features."""

    def test_env_with_scenario(self, base_config):
        """Environment should load scenario correctly."""
        base_config["scenario"] = "investor_pressure"
        base_config["mode"] = "auto"

        env = StartupOpsEnv(base_config)
        obs = env.reset()

        assert obs.num_emails > 0
        assert len(obs.email_thread_ids) > 0

    def test_env_step_stability(self, base_config):
        """Environment steps should be stable."""
        env = StartupOpsEnv(base_config)
        obs = env.reset()

        initial_emails = obs.num_emails

        # Take a wait action
        obs2, reward, done, info = env.step({"type": "wait"})

        # Should have valid observation
        assert obs2 is not None
        assert isinstance(reward, float)

    def test_email_reply(self, base_config):
        """Replying to email should work."""
        base_config["scenario"] = "investor_pressure"
        base_config["mode"] = "auto"

        env = StartupOpsEnv(base_config)
        obs = env.reset()

        if obs.email_ids:
            email_id = obs.email_ids[0]
            obs2, reward, done, info = env.step({
                "type": "reply_email",
                "target_id": email_id
            })

            assert obs2.num_emails == obs.num_emails - 1

    def test_thread_info_in_observation(self, base_config):
        """Observation should include thread info."""
        base_config["scenario"] = "investor_pressure"
        base_config["mode"] = "auto"

        env = StartupOpsEnv(base_config)
        obs = env.reset()

        # Should have thread and escalation info
        assert hasattr(obs, "email_thread_ids")
        assert hasattr(obs, "email_escalation_levels")


# -----------------------------------------------------------------------------
# Manual Input Tests
# -----------------------------------------------------------------------------

class TestManualInputs:
    """Test manual input mode."""

    def test_manual_email_parsing(self, base_config, rng):
        """Manual emails should be parsed correctly."""
        manual_inputs = ManualInputState(
            emails=[
                {
                    "sender": "boss@company.com",
                    "subject": "Urgent: Q3 planning",
                    "text": "We need to finalize Q3 planning by end of day. This is critical for our funding round.",
                }
            ]
        )

        emails, _, _, _, _, _ = generate_initial_state(
            base_config, rng, manual_inputs=manual_inputs, mode="manual"
        )

        assert len(emails) == 1
        assert emails[0].urgency in [Urgency.medium, Urgency.high]

    def test_manual_task_creation(self, base_config, rng):
        """Manual tasks should be created with correct attributes."""
        manual_inputs = ManualInputState(
            tasks=[
                {
                    "name": "Critical Bug Fix",
                    "hours_required": 6.0,
                    "deadline": 2,
                    "priority": "high",
                    "effort": 4,
                    "impact": 2.0,
                }
            ]
        )

        _, tasks, _, _, _, _ = generate_initial_state(
            base_config, rng, manual_inputs=manual_inputs, mode="manual"
        )

        assert len(tasks) == 1
        assert tasks[0].priority == Priority.high
        assert tasks[0].impact == 2.0

    def test_empty_manual_inputs(self, base_config, rng):
        """Empty manual inputs should still work."""
        manual_inputs = ManualInputState(emails=[], tasks=[])

        emails, tasks, negs, _, _, _ = generate_initial_state(
            base_config, rng, manual_inputs=manual_inputs, mode="manual"
        )

        # Should still create some default content
        assert len(emails) == 0
        assert len(tasks) == 0


# -----------------------------------------------------------------------------
# Logging Tests
# -----------------------------------------------------------------------------

class TestLogging:
    """Test enhanced logging."""

    def test_step_logging(self, base_config):
        """Each step should be logged."""
        base_config["scenario"] = "investor_pressure"

        env = StartupOpsEnv(base_config)
        env.reset()

        # Take a step
        env.step({"type": "wait"})

        # Should have log entry
        assert len(env.logs) >= 1
        assert "step" in env.logs[0]

    def test_log_contains_action(self, base_config):
        """Logs should contain action info."""
        env = StartupOpsEnv(base_config)
        env.reset()

        env.step({"type": "wait"})

        assert env.logs[0]["action"] == "wait"


# -----------------------------------------------------------------------------
# Variation Tests
# -----------------------------------------------------------------------------

class TestVariation:
    """Test that different seeds produce different states."""

    def test_different_seeds_different_states(self, base_config):
        """Different seeds should produce different random states."""
        rng1 = random.Random(1)
        rng2 = random.Random(2)

        emails1, _, _, _, _, _ = generate_initial_state(
            base_config, rng1, difficulty="medium", mode="auto"
        )
        emails2, _, _, _, _, _ = generate_initial_state(
            base_config, rng2, difficulty="medium", mode="auto"
        )

        # Different seeds likely produce different results
        # (Not guaranteed but highly probable)
        senders1 = [e.sender for e in emails1]
        senders2 = [e.sender for e in emails2]

        assert senders1 != senders2 or len(emails1) != len(emails2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
