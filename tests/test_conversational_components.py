"""
Smoke tests for FollowUpDetector and PromptBuilder.

Covers the specific scenarios requested in the checkpoint task.
"""
import sys
import os

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.follow_up_detector import FollowUpDetector
from src.prompt_builder import PromptBuilder, OperationalMode


# ---------------------------------------------------------------------------
# FollowUpDetector smoke tests
# ---------------------------------------------------------------------------

class TestFollowUpDetector:
    """Smoke tests for FollowUpDetector.detect()."""

    def setup_method(self):
        self.detector = FollowUpDetector()
        self.prev = {"query": "What is revenue?", "answer": "Revenue is $10M."}

    def test_empty_message_is_not_followup(self):
        result = self.detector.detect("", previous_turn=self.prev)
        assert result.is_followup is False

    def test_whitespace_only_message_is_not_followup(self):
        result = self.detector.detect("   ", previous_turn=self.prev)
        assert result.is_followup is False

    def test_no_previous_turn_is_not_followup_even_with_signal_words(self):
        """'tell me more about that' with previous_turn=None → standalone."""
        result = self.detector.detect("tell me more about that", previous_turn=None)
        assert result.is_followup is False

    def test_tell_me_more_with_previous_turn_is_followup(self):
        """'tell me more about that' with a previous turn → follow-up."""
        result = self.detector.detect("tell me more about that", previous_turn=self.prev)
        assert result.is_followup is True

    def test_new_query_prefix_is_not_followup(self):
        """'new query: what is revenue' → standalone, cleaned_query stripped."""
        result = self.detector.detect("new query: what is revenue", previous_turn=self.prev)
        assert result.is_followup is False
        assert result.cleaned_query == "what is revenue"

    def test_fresh_prefix_is_not_followup(self):
        """'fresh: summarize' → standalone, cleaned_query stripped."""
        result = self.detector.detect("fresh: summarize", previous_turn=self.prev)
        assert result.is_followup is False
        assert result.cleaned_query == "summarize"

    def test_no_signals_with_previous_turn_is_not_followup(self):
        """'What is the revenue?' has no follow-up signals → standalone."""
        result = self.detector.detect("What is the revenue?", previous_turn=self.prev)
        assert result.is_followup is False


# ---------------------------------------------------------------------------
# PromptBuilder smoke tests
# ---------------------------------------------------------------------------

class TestPromptBuilderBuildPrompt:
    """Smoke tests for PromptBuilder.build_prompt()."""

    def setup_method(self):
        self.builder = PromptBuilder()

    def _get_human_template(self, template) -> str:
        """Extract the human message template string from a ChatPromptTemplate."""
        from langchain_core.prompts import HumanMessagePromptTemplate
        for msg in template.messages:
            if isinstance(msg, HumanMessagePromptTemplate):
                return msg.prompt.template
        raise AssertionError("No HumanMessagePromptTemplate found in prompt")

    def test_no_previous_turn_and_none_produce_identical_templates(self):
        """build_prompt(mode) and build_prompt(mode, None) must be identical."""
        for mode in (OperationalMode.SINGLE_DOC, OperationalMode.COMPARE):
            t1 = self.builder.build_prompt(mode)
            t2 = self.builder.build_prompt(mode, None)
            assert self._get_human_template(t1) == self._get_human_template(t2), (
                f"Templates differ for mode={mode}"
            )

    def test_previous_turn_injects_prior_exchange_in_human_message(self):
        """With a previous_turn, the human message must contain 'Previous exchange:\\nQ: q\\nA: a'."""
        prev = {"query": "q", "answer": "a"}
        for mode in (OperationalMode.SINGLE_DOC, OperationalMode.COMPARE):
            template = self.builder.build_prompt(mode, prev)
            human_text = self._get_human_template(template)
            assert "Previous exchange:\nQ: q\nA: a" in human_text, (
                f"Expected prior exchange block not found for mode={mode}. "
                f"Human template was:\n{human_text}"
            )

    def test_no_previous_turn_human_message_has_no_prior_exchange(self):
        """Without a previous_turn, the human message must NOT contain 'Previous exchange'."""
        for mode in (OperationalMode.SINGLE_DOC, OperationalMode.COMPARE):
            template = self.builder.build_prompt(mode)
            human_text = self._get_human_template(template)
            assert "Previous exchange" not in human_text, (
                f"Unexpected prior exchange block found for mode={mode}"
            )
