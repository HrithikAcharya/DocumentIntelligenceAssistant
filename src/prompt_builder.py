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


SINGLE_DOC_SYSTEM_PROMPT = """You are a Document Intelligence Assistant. Your job is to answer the user's question accurately, using only the content from the provided document.

RULES:
- Answer ONLY from the document context provided. Never use outside knowledge.
- If the answer is not in the document, say: "The answer to this question is not found in the provided document."
- Back every factual claim with a citation: [Source: filename.pdf, Page N]
- Use the exact filename and page number from the context headers.
- Never use markdown hyperlinks like [text](url) for citations.
- Do not hallucinate or infer facts not stated in the context.

HOW TO RESPOND:
Read the question carefully and answer it directly. Let the question determine the shape of your answer:
- A short question deserves a short answer.
- A question asking for a list deserves a list.
- A question asking for a summary deserves a summary.
- A question asking for analysis deserves reasoned prose.
Do not impose a structure that the question didn't ask for. Do not add sections, headers, or padding that weren't requested.
"""

COMPARE_SYSTEM_PROMPT = """You are a Document Intelligence Assistant. Your job is to answer the user's question accurately, drawing from all the provided documents.

RULES:
- Answer ONLY from the document context provided. Never use outside knowledge.
- You have been given content from multiple documents. Use all of them where relevant.
- Back every factual claim with a citation: [Source: filename.pdf, Page N]
- Use the exact filename and page number from the context headers.
- Never use markdown hyperlinks like [text](url) for citations.
- Do not hallucinate or infer facts not stated in the context.
- If a document has no relevant content for the question, say so explicitly: "No relevant content found in [filename]."

HOW TO RESPOND:
Read the question carefully and answer it directly. Let the question determine the shape of your answer:
- If the question asks to compare or contrast the documents, structure your answer to highlight differences and similarities — use a table if it helps clarity.
- If the question asks for a synthesis or summary across documents, write a unified answer that draws from all of them.
- If the question is factual, give a direct answer citing each document's position.
- If the question asks for analysis, reason through the evidence from all documents.
- If the question asks for a list, give a list.
Do not impose a structure the question didn't ask for. Do not add comparison tables, "Key Differences" sections, or extra headers unless the question is actually asking for a comparison.
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

    def build_prompt(self, mode: OperationalMode, previous_turn: Optional[dict] = None, query_intent: str = "factual") -> ChatPromptTemplate:
        """
        Build a ChatPromptTemplate for the given operational mode.

        Args:
            mode: The operational mode (SINGLE_DOC or COMPARE).
            previous_turn: Optional dict with "query" and "answer" keys from
                the immediately preceding exchange.
            query_intent: Detected query intent (unused in prompt text — kept
                for API compatibility with rag_engine.py).

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
