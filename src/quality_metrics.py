"""
Quality metrics computation for the Document Intelligence Assistant.
Computes Confidence Score and Keyword Match Accuracy without extra API calls.

Free-tier optimization:
- Confidence Score is computed using BM25-style term frequency scoring
  against retrieved chunks — NO additional embedding API calls needed.
- Keyword Match Accuracy uses simple token overlap.
Both scores are in [0, 100].
"""
import json
import logging
import math
import re
import string
from dataclasses import dataclass
from typing import Optional

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# Common English stopwords to filter from keyword matching
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "need", "dare",
    "ought", "used", "it", "its", "this", "that", "these", "those", "i",
    "me", "my", "we", "our", "you", "your", "he", "she", "they", "them",
    "what", "which", "who", "whom", "how", "when", "where", "why", "not",
    "no", "nor", "so", "yet", "both", "either", "neither", "each", "few",
    "more", "most", "other", "some", "such", "than", "too", "very", "just",
    "about", "above", "after", "before", "between", "into", "through",
    "during", "including", "until", "against", "among", "throughout",
    "despite", "towards", "upon", "concerning", "as", "if", "while",
}


def _tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, split into tokens, remove stopwords."""
    translator = str.maketrans("", "", string.punctuation)
    tokens = text.lower().translate(translator).split()
    return [t for t in tokens if t and t not in STOPWORDS]


@dataclass
class QualityReport:
    """Quality metrics for a RAG response."""

    confidence_score: float        # 0.0–100.0
    keyword_match_accuracy: float  # 0.0–100.0

    def to_json(self) -> str:
        """
        Serialize to JSON string with exactly the keys 'confidence_score'
        and 'keyword_match_accuracy'.
        """
        return json.dumps({
            "confidence_score": round(self.confidence_score, 2),
            "keyword_match_accuracy": round(self.keyword_match_accuracy, 2),
        })


class QualityMetricsComputer:
    """
    Computes response quality metrics locally — zero extra API calls.

    - Confidence Score: TF-IDF-style term overlap between query keywords
      and the top retrieved chunk, scaled to [0, 100].
    - Keyword Match Accuracy: percentage of query keywords found in the
      combined retrieved chunk text, in [0, 100].

    The embeddings parameter is accepted for interface compatibility but
    is NOT called during metric computation, preserving free-tier quota.
    """

    def __init__(self, embeddings=None) -> None:
        """
        Args:
            embeddings: Accepted for interface compatibility. Not used
                        during metric computation (saves API calls).
        """
        self.embeddings = embeddings  # kept for future optional use

    def keyword_match_accuracy(self, query: str, chunks: list[Document]) -> float:
        """
        Compute the percentage of query keywords found in retrieved chunks.

        Args:
            query: The user's query string.
            chunks: Retrieved document chunks.

        Returns:
            Float in [0.0, 100.0].
        """
        if not query or not chunks:
            return 0.0

        keywords = _tokenize(query)
        if not keywords:
            return 0.0

        combined_text = " ".join(chunk.page_content.lower() for chunk in chunks)
        matched = sum(1 for kw in keywords if kw in combined_text)
        return min(100.0, max(0.0, (matched / len(keywords)) * 100.0))

    def confidence_score(self, query: str, chunks: list[Document]) -> float:
        """
        Compute a confidence score using TF-IDF-style term overlap between
        the query and ALL retrieved chunks (not just the top one).

        Scoring against all chunks prevents zero scores when the first
        chunk was selected for source-diversity rather than pure relevance.

        Args:
            query: The user's query string.
            chunks: All retrieved document chunks.

        Returns:
            Float in [0.0, 100.0].
        """
        if not query or not chunks:
            return 0.0

        query_keywords = _tokenize(query)
        if not query_keywords:
            return 0.0

        # Build a merged TF map across ALL chunks (take max TF per term)
        # Also track how many distinct chunks contain each term (for IDF)
        chunk_hit: dict[str, int] = {}
        best_tf: dict[str, int] = {}
        for chunk in chunks:
            chunk_tokens_set = set(_tokenize(chunk.page_content))   # distinct terms in this chunk
            chunk_tf: dict[str, int] = {}
            for token in _tokenize(chunk.page_content):             # full list for TF count
                chunk_tf[token] = chunk_tf.get(token, 0) + 1
            for term, freq in chunk_tf.items():
                best_tf[term] = max(best_tf.get(term, 0), freq)
            for term in chunk_tokens_set:
                chunk_hit[term] = chunk_hit.get(term, 0) + 1

        if not best_tf:
            return 0.0

        # TF-IDF-inspired weighted score
        num_chunks = len(chunks)
        matched = 0
        raw_score = 0.0
        max_possible = 0.0

        for kw in query_keywords:
            if kw in best_tf:
                matched += 1
                tf_s = math.log(1 + best_tf[kw])
                idf = math.log(1 + num_chunks / (1 + chunk_hit.get(kw, 0)))
                raw_score    += tf_s * idf
                # max IDF assumes keyword appears in exactly 1 chunk
                max_possible += tf_s * math.log(1 + num_chunks / 2)

        if matched == 0:
            return 0.0

        if max_possible == 0.0:
            return 0.0

        match_rate = matched / len(query_keywords)

        if num_chunks == 1:
            # With a single chunk IDF always cancels to 1.0, making confidence == keyword_match.
            # Instead, measure topical density: fraction of chunk tokens that are query keywords.
            # This produces a genuinely different signal — a large chunk with few keyword
            # occurrences scores lower than a focused chunk where keywords dominate.
            all_chunk_tokens = _tokenize(chunks[0].page_content)
            total_tokens = max(len(all_chunk_tokens), 1)
            keyword_hits = sum(all_chunk_tokens.count(kw) for kw in query_keywords if kw in best_tf)
            density = min(1.0, keyword_hits / total_tokens)
            # Scale density to [0, 100] — a chunk that is 10%+ query keywords is very focused
            # (density of 0.10 → score of 100 after scaling by 10x, capped at 1.0)
            normalised = min(1.0, density * 10.0) * match_rate
        else:
            normalised = (raw_score / max_possible) * match_rate

        return round(min(1.0, normalised) * 100.0, 2)

    def compute(
        self,
        query: str,
        retrieved_chunks: list[Document],
        query_embedding: Optional[list[float]] = None,  # kept for compat, unused
    ) -> QualityReport:
        """
        Compute both quality metrics and return a QualityReport.
        No API calls are made.

        Args:
            query: The user's query string.
            retrieved_chunks: Chunks retrieved from the vector store.
            query_embedding: Ignored. Kept for interface compatibility.

        Returns:
            QualityReport with confidence_score and keyword_match_accuracy.
        """
        if not retrieved_chunks:
            logger.warning("No chunks provided for quality metrics computation.")
            return QualityReport(confidence_score=0.0, keyword_match_accuracy=0.0)

        kma = self.keyword_match_accuracy(query, retrieved_chunks)
        cs = self.confidence_score(query, retrieved_chunks)

        logger.debug(
            "Quality metrics — confidence_score: %.1f, keyword_match_accuracy: %.1f",
            cs, kma
        )
        return QualityReport(confidence_score=cs, keyword_match_accuracy=kma)
