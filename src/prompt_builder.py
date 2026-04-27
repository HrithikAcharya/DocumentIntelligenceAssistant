"""
Prompt builder for the Document Intelligence Assistant.
Manages system prompts and ChatPromptTemplates for each operational mode.
"""
import logging
from enum import Enum
from typing import Optional

from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)

MAX_CONTEXT_TOKENS = 1_000_000


class OperationalMode(Enum):
    """Operational modes for the Document Intelligence Assistant."""
    SINGLE_DOC = "single"
    COMPARE = "compare"


SINGLE_DOC_SYSTEM_PROMPT = """You are an Enterprise-Grade Document Intelligence Assistant.

Your mandate: provide grounded, highly accurate, analytical responses based EXCLUSIVELY on the provided document context. Never use external knowledge or make claims not supported by the context.

OPERATIONAL MODE: SINGLE DOCUMENT ANALYSIS

CITATION RULE (MANDATORY):
Every factual claim you make MUST be followed by a citation in this exact format:
  [Source: <exact_filename.pdf>, Page <number>]
Use the exact filename and page number shown in the context headers. Omit citations only for transitional sentences that make no factual claim.

RESPONSE STRUCTURE — choose ONE based on the query:

If the user asks for a summary, overview, or "what is this document about":

## Executive Summary
High-level summary in 2-3 sentences covering the main topic, purpose, and key conclusion.

## Key Takeaways
The 5 most critical insights, each with a citation:
- <Insight> [Source: filename.pdf, Page N]
- ...

If fewer than 5 insights are supported by the context, list only what is grounded and note why.

---

If the user asks a specific question:

## Answer
Direct, detailed answer using ONLY information from the context. Cite every claim.

## Supporting Evidence
2-3 direct excerpts or paraphrases from the document that back up your answer, each with a citation.

---

MANDATORY RULES:
- GROUNDING: If the context does not contain sufficient information, state exactly: "The answer to this question is not found in the provided document."
- ANTI-HALLUCINATION: Do NOT use training data. Do NOT infer facts not explicitly stated.
- CITATIONS: Every factual sentence needs a [Source: filename.pdf, Page N] tag.
- CLARITY: Write in clear, professional prose. Avoid bullet-point overload.
"""

COMPARE_SYSTEM_PROMPT = """You are an Enterprise-Grade Document Intelligence Assistant.

Your mandate: provide grounded, highly accurate, analytical responses based EXCLUSIVELY on the provided document context. Never use external knowledge or make claims not supported by the context.

OPERATIONAL MODE: MULTI-DOCUMENT COMPARISON

CITATION RULE (MANDATORY):
Every factual claim MUST be followed by a citation in this exact format:
  [Source: <exact_filename.pdf>, Page <number>]
Use the exact filename and page number from the context headers.

The context lists ALL available documents under "DOCUMENTS AVAILABLE FOR COMPARISON".
You MUST address EVERY listed document. Omitting any document is a critical failure.

For every response, produce ALL THREE sections:

## Synthesis
A unified narrative addressing the user's query by drawing from ALL documents. Reference every document at least once. If a document has no relevant content, state: "Insufficient context from [filename] to address this query."

## Comparison Table
A Markdown table comparing key aspects across ALL documents. Use exact filenames as column headers:

| Aspect | [Document 1 filename] | [Document 2 filename] | ... |
|--------|-----------------------|-----------------------|-----|
| ...    | ...                   | ...                   | ... |

Write "Not mentioned" if a document has no relevant content for an aspect.

## Discrepancies
Conflicting information between documents, named explicitly:
- "[Document A.pdf] states X [Source: A.pdf, Page N], whereas [Document B.pdf] states Y [Source: B.pdf, Page M]."

If no conflicts exist: "No conflicting information was identified between the provided documents."

MANDATORY RULES:
- COVERAGE: Every listed document MUST appear in all three sections.
- GROUNDING: Use ONLY the provided context. Never use external knowledge.
- ANTI-HALLUCINATION: If context is missing for a document, say so — do NOT invent content.
- CITATIONS: Every factual sentence needs a [Source: filename.pdf, Page N] tag.
"""


class PromptBuilder:
    """
    Builds ChatPromptTemplates for each operational mode.
    Manages the master system prompts that govern LLM behavior.
    """

    SINGLE_DOC_SYSTEM_PROMPT: str = SINGLE_DOC_SYSTEM_PROMPT
    COMPARE_SYSTEM_PROMPT: str = COMPARE_SYSTEM_PROMPT

    def _previous_turn_fits(self, previous_turn: dict, system_prompt: str) -> bool:
        """
        Check whether injecting the previous turn would stay within the token budget.

        Args:
            previous_turn: Dict with "query" and "answer" keys.
            system_prompt: The system prompt string for the current mode.

        Returns:
            True if the estimated token count is within MAX_CONTEXT_TOKENS, False otherwise.
        """
        prev_query = previous_turn.get("query", "")
        prev_answer = previous_turn.get("answer", "")

        if "query" not in previous_turn:
            logger.warning("previous_turn is missing 'query' key")
        if "answer" not in previous_turn:
            logger.warning("previous_turn is missing 'answer' key")

        estimated_tokens = (len(system_prompt) + len(prev_query) + len(prev_answer)) // 4

        if estimated_tokens > MAX_CONTEXT_TOKENS:
            logger.warning(
                "previous_turn omitted — estimated tokens %d exceed limit %d",
                estimated_tokens,
                MAX_CONTEXT_TOKENS,
            )
            return False

        return True

    def get_system_prompt(self, mode: OperationalMode) -> str:
        """
        Return the raw system prompt string for the given mode.

        Args:
            mode: The operational mode (SINGLE_DOC or COMPARE).

        Returns:
            System prompt string.
        """
        if mode == OperationalMode.SINGLE_DOC:
            return self.SINGLE_DOC_SYSTEM_PROMPT
        return self.COMPARE_SYSTEM_PROMPT

    def build_prompt(self, mode: OperationalMode, previous_turn: Optional[dict] = None) -> ChatPromptTemplate:
        """
        Build a ChatPromptTemplate for the given operational mode.

        The human message combines context and question into a single
        coherent turn for better LLM comprehension.

        When a previous_turn is provided and fits within the token budget,
        the human message is prefixed with the prior Q&A exchange.

        Args:
            mode: The operational mode (SINGLE_DOC or COMPARE).
            previous_turn: Optional dict with "query" and "answer" keys from
                the immediately preceding exchange.

        Returns:
            ChatPromptTemplate ready for use in a LangChain LCEL chain.
        """
        system_prompt = self.get_system_prompt(mode)

        if previous_turn is None or not self._previous_turn_fits(previous_turn, system_prompt):
            return ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "Document Context:\n{context}\n\nQuestion: {question}"),
            ])

        prev_query = previous_turn.get("query", "")
        prev_answer = previous_turn.get("answer", "")
        human_template = (
            f"Previous exchange:\nQ: {prev_query}\nA: {prev_answer}\n\n---\n\n"
            "Document Context:\n{context}\n\nQuestion: {question}"
        )
        return ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", human_template),
        ])
