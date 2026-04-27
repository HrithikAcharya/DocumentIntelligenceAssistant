"""
Follow-up detector for the Document Intelligence Assistant.
Classifies user messages as follow-ups to the previous exchange or standalone queries.
"""
import re
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# Fresh prefix patterns — force standalone regardless of message content
_FRESH_PREFIXES = re.compile(
    r'^(?:new\s+query|fresh)\s*:\s*',
    re.IGNORECASE,
)

# Single-word follow-up pronouns — matched as whole words to avoid false positives
# e.g. "Italy" should NOT trigger on "it"
_PRONOUN_SIGNALS = re.compile(
    r'\b(?:that|it|this|those|them|they)\b',
    re.IGNORECASE,
)

# Multi-word follow-up phrases — matched as substrings (case-insensitive)
_PHRASE_SIGNALS = [
    "tell me more",
    "expand on",
    "elaborate",
    "explain further",
    "what about",
    "what did you mean",
    "based on that",
    "in addition to that",
    "more about",
    "go deeper",
    "dig into",
]


@dataclass
class DetectionResult:
    """Result of follow-up detection."""
    is_followup: bool    # True if message is a follow-up to previous_turn
    cleaned_query: str   # Message with fresh prefix stripped (or original)


class FollowUpDetector:
    """
    Classifies a user message as a follow-up to the previous exchange
    or as a standalone query.

    Detection is purely lexical — no LLM call is made.
    Completes in well under 50 ms for any message up to 2 000 characters.
    """

    def detect(
        self,
        message: str,
        previous_turn: Optional[dict],
    ) -> DetectionResult:
        """
        Classify a user message.

        Args:
            message: The raw user message text.
            previous_turn: The previous conversation turn dict
                           {"query": str, "answer": str}, or None.

        Returns:
            DetectionResult with is_followup flag and cleaned query text.
        """
        # Handle empty input gracefully
        if not message or not message.strip():
            return DetectionResult(is_followup=False, cleaned_query=message or "")

        stripped = message.lstrip()

        # Step 1: Check for fresh prefix — forces standalone regardless of content
        fresh_match = _FRESH_PREFIXES.match(stripped)
        if fresh_match:
            cleaned = stripped[fresh_match.end():].strip()
            logger.debug("Fresh prefix detected — treating as standalone query.")
            return DetectionResult(is_followup=False, cleaned_query=cleaned)

        # Step 2: No previous turn — always standalone
        if previous_turn is None:
            return DetectionResult(is_followup=False, cleaned_query=message)

        # Step 3: Scan for follow-up signals
        lower = message.lower()

        # Check single-word pronoun signals (word-boundary matched)
        if _PRONOUN_SIGNALS.search(message):
            logger.debug("Follow-up pronoun signal detected.")
            return DetectionResult(is_followup=True, cleaned_query=message)

        # Check multi-word phrase signals (substring match)
        for phrase in _PHRASE_SIGNALS:
            if phrase in lower:
                logger.debug("Follow-up phrase signal detected: '%s'", phrase)
                return DetectionResult(is_followup=True, cleaned_query=message)

        # No signals found — standalone query
        return DetectionResult(is_followup=False, cleaned_query=message)
