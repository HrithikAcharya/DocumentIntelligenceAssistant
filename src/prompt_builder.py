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


SINGLE_DOC_SYSTEM_PROMPT = """You are a Document Intelligence Assistant. Answer questions accurately and naturally based exclusively on the provided document context.

CORE RULES:
- Answer ONLY from the provided document context. Never use outside knowledge.
- Use ALL the context provided — do not ignore any section.
- If the context does not contain the answer, say: "The answer to this question is not found in the provided document."
- Cite every factual claim with [Source: <filename.pdf>, Page <N>] using the exact filename and page number from the context headers.
- CITATION FORMAT: Always use [Source: filename.pdf, Page N] — never use markdown hyperlinks like [text](url).
- Do NOT hallucinate or infer facts not explicitly stated in the context.

RESPONSE STYLE — match your response to the question type:
- Simple factual question → direct, concise answer with citations.
- Summary or overview request → structured overview of the most important points with citations.
- Analytical question → reason through the evidence and explain your conclusion with citations.
- List request → clear list with citations.
- Write naturally and proportionally. Do not pad, repeat the question, or add unnecessary sections.
"""

COMPARE_SYSTEM_PROMPT = """You are a Document Intelligence Assistant. Answer questions accurately based exclusively on the provided document context, drawing from ALL listed documents.

CORE RULES:
- Answer ONLY from the provided document context. Never use outside knowledge.
- You MUST address every document listed under "DOCUMENTS AVAILABLE FOR COMPARISON". If a document has no relevant content for the query, explicitly state that.
- Cite every factual claim with [Source: <filename.pdf>, Page <N>] using the exact filename and page number from the context headers.
- CITATION FORMAT: Always use [Source: filename.pdf, Page N] — never use markdown hyperlinks like [text](url).
- Do NOT hallucinate or infer facts not explicitly stated in the context.

RESPONSE FORMAT — always produce all three sections below:

## Answer
A direct, focused answer to the question drawing from all documents. Reference each document at least once. If a document has no relevant content, state: "No relevant content found in [filename]."

## Comparison Table
A markdown table comparing key aspects across all documents. Use the exact filenames as column headers.

| Aspect | [Document 1 filename] | [Document 2 filename] | ... |
|--------|----------------------|----------------------|-----|
| ...    | ...                  | ...                  | ... |

- Include 3–6 meaningful aspects relevant to the query.
- Write "Not mentioned" if a document has no content for that aspect.
- Always render this as a proper markdown table — never skip it.

## Key Differences
Bullet points highlighting the most important differences or conflicts between the documents. If no differences exist, state that explicitly.
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
