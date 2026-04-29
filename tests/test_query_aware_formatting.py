"""
Tests for query-aware response formatting in src/prompt_builder.py.

Task 1: Bug condition exploration tests — these assert the EXPECTED (fixed) behavior,
so they FAIL on unfixed code. Failure confirms the bug exists.
"""
from src.prompt_builder import COMPARE_SYSTEM_PROMPT, SINGLE_DOC_SYSTEM_PROMPT, OperationalMode, PromptBuilder


# ---------------------------------------------------------------------------
# Task 1: Bug condition exploration tests
# These tests encode the EXPECTED behavior after the fix.
# They FAIL on unfixed code — that failure is proof the bug exists.
# ---------------------------------------------------------------------------

def test_bug_condition_no_unconditional_mandate():
    """Bug condition: COMPARE_SYSTEM_PROMPT must NOT contain the hardcoded three-section mandate."""
    assert "always produce all three sections" not in COMPARE_SYSTEM_PROMPT


def test_bug_condition_has_query_classification():
    """Bug condition: COMPARE_SYSTEM_PROMPT must contain query-type classification guidance.

    The unfixed prompt only mentions 'factual' in the citation rule ('Cite every factual claim'),
    not as a query-type classification. The fixed prompt must contain 'synthesis' or 'analytical'
    as explicit query-type labels, or contain 'factual' in a query-classification context
    (i.e., alongside 'synthesis' or 'analytical').
    """
    prompt_lower = COMPARE_SYSTEM_PROMPT.lower()
    # 'synthesis' and 'analytical' are unambiguous query-type classification words;
    # 'factual' alone is insufficient because the unfixed prompt uses it in the citation rule.
    has_synthesis = "synthesis" in prompt_lower
    has_analytical = "analytical" in prompt_lower
    assert has_synthesis or has_analytical, \
        "COMPARE_SYSTEM_PROMPT must contain query-type classification guidance " \
        "('synthesis' or 'analytical' as query-type labels)"


def test_bug_condition_prompt_builder_no_mandate():
    """Bug condition: PromptBuilder.get_system_prompt(COMPARE) must not return the hardcoded mandate."""
    pb = PromptBuilder()
    prompt = pb.get_system_prompt(OperationalMode.COMPARE)
    assert "always produce all three sections" not in prompt


# ---------------------------------------------------------------------------
# Task 2: Preservation property tests
# These assert behaviors that must survive the fix.
# They PASS on unfixed code — establishes the baseline.
# ---------------------------------------------------------------------------

def test_preservation_citation_format_in_compare_prompt():
    """Preservation: [Source: filename.pdf, Page N] citation instruction must remain in COMPARE_SYSTEM_PROMPT."""
    assert "Source:" in COMPARE_SYSTEM_PROMPT
    assert "Page" in COMPARE_SYSTEM_PROMPT

def test_preservation_citation_format_in_single_doc_prompt():
    """Preservation: [Source: filename.pdf, Page N] citation instruction must remain in SINGLE_DOC_SYSTEM_PROMPT."""
    assert "Source:" in SINGLE_DOC_SYSTEM_PROMPT
    assert "Page" in SINGLE_DOC_SYSTEM_PROMPT

def test_preservation_document_coverage_rule():
    """Preservation: COMPARE_SYSTEM_PROMPT must instruct addressing every document."""
    prompt_lower = COMPARE_SYSTEM_PROMPT.lower()
    assert "every document" in prompt_lower or "all" in prompt_lower or "each document" in prompt_lower

def test_preservation_single_doc_not_found_refusal():
    """Preservation: SINGLE_DOC_SYSTEM_PROMPT must state when answer is not found in the document."""
    assert "not found in the provided document" in SINGLE_DOC_SYSTEM_PROMPT

def test_preservation_followup_injection():
    """Preservation: build_prompt(COMPARE, previous_turn=...) must inject the prior exchange into the human message."""
    pb = PromptBuilder()
    previous_turn = {"query": "What is the topic?", "answer": "The topic is AI."}
    template = pb.build_prompt(OperationalMode.COMPARE, previous_turn=previous_turn)
    # The human message template should contain the previous exchange
    human_message = template.messages[1]
    human_content = human_message.prompt.template if hasattr(human_message, 'prompt') else str(human_message)
    assert "Previous exchange" in human_content
    assert "What is the topic?" in human_content
    assert "The topic is AI." in human_content

def test_preservation_compare_prompt_has_comparison_table_instruction():
    """Preservation: COMPARE_SYSTEM_PROMPT must still instruct a Comparison Table for comparison queries."""
    # After the fix, the table instruction must still exist — just conditionally for comparison queries
    assert "Comparison Table" in COMPARE_SYSTEM_PROMPT or "comparison table" in COMPARE_SYSTEM_PROMPT.lower()

def test_preservation_compare_prompt_has_key_differences_instruction():
    """Preservation: COMPARE_SYSTEM_PROMPT must still instruct Key Differences for comparison queries."""
    assert "Key Differences" in COMPARE_SYSTEM_PROMPT or "key differences" in COMPARE_SYSTEM_PROMPT.lower()
