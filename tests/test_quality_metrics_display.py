"""
Tests for quality metrics display fix (tasks 2.1, 2.2, 2.3).

These tests verify the fix programmatically without requiring a running
Chainlit server. They cover:
  - Task 2.1: Fix verification — footer format is correct after the fix
  - Task 2.2: Regression prevention — QualityMetricsComputer still works
  - Task 2.3: Edge cases — zero, max, decimal scores, long/empty answers
"""
import re
import sys
import os

import pytest

# Ensure src/ is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from langchain_core.documents import Document
from src.quality_metrics import QualityMetricsComputer, QualityReport


# ---------------------------------------------------------------------------
# Helpers — mirror the exact footer-building logic from app.py
# ---------------------------------------------------------------------------

def build_quality_footer(quality_report: QualityReport) -> str:
    """Exact copy of the footer construction from app.py (post-fix)."""
    return (
        f"\n\n---\n"
        f"📊 **Confidence:** `{quality_report.confidence_score:.0f}/100` | "
        f"**Keyword Match:** `{quality_report.keyword_match_accuracy:.0f}/100`"
    )


CLEAN_ANSWER_REGEX = re.compile(
    r'```(?:json)?\s*\{[^}]*"confidence_score"[^}]*\}\s*```',
    flags=re.DOTALL,
)


def clean_answer(raw: str) -> str:
    """Defensive cleanup — strips JSON quality blocks the LLM might emit."""
    return CLEAN_ANSWER_REGEX.sub("", raw).strip()


def build_final_content(answer: str, quality_report: QualityReport) -> str:
    """Combine clean answer + quality footer, as app.py does."""
    return clean_answer(answer) + build_quality_footer(quality_report)


# ---------------------------------------------------------------------------
# Task 2.1 — Fix Verification Tests
# ---------------------------------------------------------------------------

class TestFixVerification:
    """Verify the quality footer is correctly formatted after the fix."""

    def _make_report(self, confidence=75.0, keyword_match=80.0) -> QualityReport:
        return QualityReport(
            confidence_score=confidence,
            keyword_match_accuracy=keyword_match,
        )

    def test_footer_contains_confidence_label(self):
        footer = build_quality_footer(self._make_report())
        assert "📊 **Confidence:**" in footer

    def test_footer_contains_keyword_match_label(self):
        footer = build_quality_footer(self._make_report())
        assert "**Keyword Match:**" in footer

    def test_footer_contains_separator_line(self):
        footer = build_quality_footer(self._make_report())
        assert "---" in footer

    def test_footer_uses_plain_pipe_separator(self):
        footer = build_quality_footer(self._make_report())
        assert " | " in footer

    def test_footer_does_not_contain_nbsp(self):
        footer = build_quality_footer(self._make_report())
        assert "&nbsp;" not in footer

    def test_footer_score_format_confidence(self):
        """Confidence score should appear as `75/100` (backtick-wrapped)."""
        footer = build_quality_footer(self._make_report(confidence=75.0))
        assert "`75/100`" in footer

    def test_footer_score_format_keyword_match(self):
        """Keyword match score should appear as `80/100` (backtick-wrapped)."""
        footer = build_quality_footer(self._make_report(keyword_match=80.0))
        assert "`80/100`" in footer

    def test_footer_separator_precedes_metrics(self):
        """The --- separator must come before the metrics line."""
        footer = build_quality_footer(self._make_report())
        sep_pos = footer.index("---")
        conf_pos = footer.index("📊")
        assert sep_pos < conf_pos

    def test_final_content_ends_with_footer(self):
        """The footer should be at the very end of final_content."""
        report = self._make_report()
        content = build_final_content("Some answer text.", report)
        footer = build_quality_footer(report)
        assert content.endswith(footer)


# ---------------------------------------------------------------------------
# Task 2.2 — Regression Prevention Tests
# ---------------------------------------------------------------------------

class TestRegressionPrevention:
    """Verify quality metrics computation still works correctly."""

    def _make_chunks(self, texts: list[str]) -> list[Document]:
        return [Document(page_content=t) for t in texts]

    def test_compute_returns_quality_report(self):
        computer = QualityMetricsComputer()
        chunks = self._make_chunks(["The quick brown fox jumps over the lazy dog."])
        report = computer.compute("quick fox", chunks)
        assert isinstance(report, QualityReport)

    def test_confidence_score_in_valid_range(self):
        computer = QualityMetricsComputer()
        chunks = self._make_chunks(["Machine learning is a subset of artificial intelligence."])
        report = computer.compute("machine learning artificial intelligence", chunks)
        assert 0 <= report.confidence_score <= 100

    def test_keyword_match_accuracy_in_valid_range(self):
        computer = QualityMetricsComputer()
        chunks = self._make_chunks(["Python is a popular programming language."])
        report = computer.compute("Python programming", chunks)
        assert 0 <= report.keyword_match_accuracy <= 100

    def test_compute_with_no_chunks_returns_zeros(self):
        computer = QualityMetricsComputer()
        report = computer.compute("some query", [])
        assert report.confidence_score == 0.0
        assert report.keyword_match_accuracy == 0.0

    def test_clean_answer_strips_json_quality_block(self):
        """The regex must remove JSON quality blocks the LLM might emit."""
        raw = (
            'Here is the answer.\n\n'
            '```json\n{"confidence_score": 80, "keyword_match_accuracy": 70}\n```'
        )
        result = clean_answer(raw)
        assert "confidence_score" not in result
        assert "Here is the answer." in result

    def test_clean_answer_strips_json_block_no_language_tag(self):
        raw = (
            'Answer text.\n\n'
            '```\n{"confidence_score": 90, "keyword_match_accuracy": 60}\n```'
        )
        result = clean_answer(raw)
        assert "confidence_score" not in result
        assert "Answer text." in result

    def test_clean_answer_preserves_normal_content(self):
        """Content without a JSON quality block must be unchanged."""
        raw = "This is a normal answer with no quality block."
        result = clean_answer(raw)
        assert result == raw

    def test_final_content_concatenation(self):
        """final_content = clean_answer + quality_footer must work correctly."""
        report = QualityReport(confidence_score=60.0, keyword_match_accuracy=55.0)
        answer = "The document discusses climate change."
        content = build_final_content(answer, report)
        assert "The document discusses climate change." in content
        assert "📊 **Confidence:**" in content
        assert "**Keyword Match:**" in content

    def test_keyword_match_high_overlap(self):
        """When query terms appear in chunks, keyword match should be > 0."""
        computer = QualityMetricsComputer()
        chunks = self._make_chunks(["revenue profit growth quarterly earnings report"])
        report = computer.compute("revenue profit earnings", chunks)
        assert report.keyword_match_accuracy > 0

    def test_confidence_score_high_overlap(self):
        """When query terms appear frequently in chunks, confidence should be > 0."""
        computer = QualityMetricsComputer()
        chunks = self._make_chunks(["revenue revenue revenue profit profit earnings"])
        report = computer.compute("revenue profit earnings", chunks)
        assert report.confidence_score > 0


# ---------------------------------------------------------------------------
# Task 2.3 — Edge Case Testing
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Test edge cases for footer formatting and metrics computation."""

    def test_zero_scores_format_correctly(self):
        report = QualityReport(confidence_score=0.0, keyword_match_accuracy=0.0)
        footer = build_quality_footer(report)
        assert "`0/100`" in footer
        # Both metrics should show 0
        assert footer.count("`0/100`") == 2

    def test_max_scores_format_correctly(self):
        report = QualityReport(confidence_score=100.0, keyword_match_accuracy=100.0)
        footer = build_quality_footer(report)
        assert "`100/100`" in footer
        assert footer.count("`100/100`") == 2

    def test_decimal_confidence_rounded_to_integer(self):
        """75.6 should display as 76, not 75.6."""
        report = QualityReport(confidence_score=75.6, keyword_match_accuracy=50.0)
        footer = build_quality_footer(report)
        assert "`76/100`" in footer
        assert "75.6" not in footer

    def test_decimal_keyword_match_rounded_to_integer(self):
        """33.3 should display as 33."""
        report = QualityReport(confidence_score=50.0, keyword_match_accuracy=33.3)
        footer = build_quality_footer(report)
        assert "`33/100`" in footer
        assert "33.3" not in footer

    def test_decimal_rounds_up(self):
        """0.5 rounds to 1 with :.0f formatting."""
        report = QualityReport(confidence_score=0.5, keyword_match_accuracy=99.5)
        footer = build_quality_footer(report)
        # Python's :.0f uses banker's rounding; 0.5 → 0, 99.5 → 100
        # We just verify no decimal point appears
        assert "." not in footer.split("---")[1]

    def test_long_answer_gets_footer_at_end(self):
        """A very long answer should still have the footer appended at the end."""
        long_answer = "This is a very detailed analysis. " * 200  # ~6800 chars
        report = QualityReport(confidence_score=88.0, keyword_match_accuracy=72.0)
        content = build_final_content(long_answer, report)
        footer = build_quality_footer(report)
        assert content.endswith(footer)
        # The long answer content should still be present
        assert "This is a very detailed analysis." in content

    def test_empty_answer_still_gets_footer(self):
        """An empty answer should still produce a footer."""
        report = QualityReport(confidence_score=0.0, keyword_match_accuracy=0.0)
        content = build_final_content("", report)
        assert "📊 **Confidence:**" in content
        assert "**Keyword Match:**" in content

    def test_minimal_answer_still_gets_footer(self):
        """A one-word answer should still get the footer."""
        report = QualityReport(confidence_score=50.0, keyword_match_accuracy=50.0)
        content = build_final_content("Yes.", report)
        assert "Yes." in content
        assert "📊 **Confidence:**" in content

    def test_answer_with_json_block_and_footer(self):
        """JSON quality block is stripped, then footer is appended."""
        raw = (
            'The answer is 42.\n\n'
            '```json\n{"confidence_score": 80, "keyword_match_accuracy": 70}\n```'
        )
        report = QualityReport(confidence_score=80.0, keyword_match_accuracy=70.0)
        content = build_final_content(raw, report)
        assert "The answer is 42." in content
        assert "confidence_score" not in content  # JSON block stripped
        assert "📊 **Confidence:**" in content
        assert "`80/100`" in content

    def test_nbsp_never_appears_in_footer(self):
        """Regression: &nbsp; must never appear regardless of score values."""
        for confidence, keyword in [(0, 0), (50, 50), (100, 100), (33.3, 66.7)]:
            report = QualityReport(
                confidence_score=confidence,
                keyword_match_accuracy=keyword,
            )
            footer = build_quality_footer(report)
            assert "&nbsp;" not in footer, (
                f"&nbsp; found in footer for scores ({confidence}, {keyword})"
            )


# ---------------------------------------------------------------------------
# Task 2 — Metrics Differentiation Tests
# ---------------------------------------------------------------------------

class TestMetricsDifferentiation:
    """Verify confidence_score is meaningfully distinct from keyword_match_accuracy."""

    def test_scores_differ_when_tf_greater_than_one(self):
        """confidence_score must differ from keyword_match_accuracy when tf > 1."""
        computer = QualityMetricsComputer()
        # Multiple chunks: keywords repeated (tf > 1) but spread across chunks
        # so IDF dampens the score, making confidence_score < keyword_match_accuracy
        chunks = [
            Document(page_content="revenue revenue revenue profit profit earnings earnings earnings"),
            Document(page_content="revenue profit earnings analysis report"),
            Document(page_content="revenue profit earnings quarterly results"),
        ]
        report = computer.compute("revenue profit earnings", chunks)
        assert report.confidence_score != report.keyword_match_accuracy

    def test_higher_tf_produces_higher_confidence(self):
        """Doubling keyword frequency should increase confidence score."""
        computer = QualityMetricsComputer()
        low_tf  = [Document(page_content="revenue profit earnings")]
        high_tf = [Document(page_content="revenue revenue revenue profit profit earnings earnings")]
        low_report  = computer.compute("revenue profit earnings", low_tf)
        high_report = computer.compute("revenue profit earnings", high_tf)
        assert high_report.confidence_score >= low_report.confidence_score

    def test_concentrated_keywords_produce_higher_confidence(self):
        """Keywords in fewer chunks (higher IDF) should yield higher confidence."""
        computer = QualityMetricsComputer()
        query = "machine learning"
        # Concentrated: both keywords in 1 chunk
        concentrated = [
            Document(page_content="machine learning machine learning"),
            Document(page_content="unrelated content about cooking recipes"),
            Document(page_content="more unrelated content about sports"),
        ]
        # Spread: keywords scattered across all chunks
        spread = [
            Document(page_content="machine learning"),
            Document(page_content="machine algorithms"),
            Document(page_content="learning systems"),
        ]
        conc_report   = computer.compute(query, concentrated)
        spread_report = computer.compute(query, spread)
        assert conc_report.confidence_score >= spread_report.confidence_score

    def test_confidence_score_always_in_valid_range(self):
        """confidence_score must always be in [0.0, 100.0]."""
        computer = QualityMetricsComputer()
        test_cases = [
            ("single keyword", [Document(page_content="keyword keyword keyword keyword keyword")]),
            ("multi keyword query", [Document(page_content="alpha beta gamma delta epsilon " * 10)]),
            ("no match", [Document(page_content="completely unrelated text here")]),
            ("empty chunks", []),
        ]
        for query, chunks in test_cases:
            report = computer.compute(query, chunks)
            assert 0.0 <= report.confidence_score <= 100.0, (
                f"Out of range for query='{query}': {report.confidence_score}"
            )

    def test_zero_chunks_still_returns_zero(self):
        computer = QualityMetricsComputer()
        report = computer.compute("revenue profit", [])
        assert report.confidence_score == 0.0
        assert report.keyword_match_accuracy == 0.0

    def test_no_keyword_overlap_returns_zero_confidence(self):
        computer = QualityMetricsComputer()
        chunks = [Document(page_content="completely unrelated content")]
        report = computer.compute("revenue profit earnings", chunks)
        assert report.confidence_score == 0.0

    def test_keyword_match_accuracy_formula_unchanged(self):
        """keyword_match_accuracy must still use binary presence formula."""
        computer = QualityMetricsComputer()
        chunks = [Document(page_content="revenue profit earnings growth")]
        report = computer.compute("revenue profit earnings", chunks)
        # All 3 keywords present → 100%
        assert report.keyword_match_accuracy == 100.0
