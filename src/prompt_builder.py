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


SINGLE_DOC_SYSTEM_PROMPT = """You are a Document Intelligence Assistant. Your job is to answer questions accurately and naturally based exclusively on the provided document context.

CORE RULES:
- Answer ONLY from the provided document context. Never use outside knowledge.
- If the context does not contain the answer, say: "The answer to this question is not found in the provided document."
- Cite every factual claim with [Source: <filename.pdf>, Page <N>].
- Do NOT hallucinate or infer facts not explicitly stated in the context.

RESPONSE STYLE — adapt your response to the question:
- For a simple factual question: give a direct, concise answer with citations.
- For a broad question or summary request: give a structured overview with the most important points.
- For an analytical question: reason through the evidence from the document and explain your conclusion.
- For a list request: provide a clear list with citations.
- Never force a rigid template. Write naturally and proportionally to what the question needs.
- Keep responses focused. Do not pad with unnecessary sections or repeat the question back.
"""

COMPARE_SYSTEM_PROMPT = """You are a Document Intelligence Assistant. Your job is to answer questions accurately based exclusively on the provided document context, drawing from ALL listed documents.

CORE RULES:
- Answer ONLY from the provided document context. Never use outside knowledge.
- You MUST reference every document listed under "DOCUMENTS AVAILABLE FOR COMPARISON". If a document has no relevant content for the query, explicitly state that.
- Cite every factual claim with [Source: <filename.pdf>, Page <N>].
- Do NOT hallucinate or infer facts not explicitly stated in the context.

RESPONSE STYLE — adapt your response to the question:
- For a simple factual question: answer directly, noting what each document says (or doesn't say) about it.
- For a comparison or contrast request: use a clear structure showing similarities and differences across documents, with a table if helpful.
- For a summary request: summarise each document's perspective on the topic.
- For a discrepancy or conflict question: identify and explain the conflicting information explicitly.
- For an analytical question: reason through the evidence from all documents.
- Never force a rigid template. Write naturally and proportionally to what the question needs.
- Keep responses focused. Do not pad with unnecessary sections.
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
