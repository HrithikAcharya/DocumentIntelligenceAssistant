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

SUPPORT_THRESHOLD = 0.5


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
    answer_faithfulness_score: float = 0.0  # 0.0–100.0
    retrieval_quality: float = 0.0          # 0.0–100.0
    citation_accuracy: float = 0.0          # 0.0–100.0

    def to_json(self) -> str:
        """
        Serialize to JSON string with all quality metric keys.
        """
        return json.dumps({
            "confidence_score": round(self.confidence_score, 2),
            "keyword_match_accuracy": round(self.keyword_match_accuracy, 2),
            "answer_faithfulness_score": round(self.answer_faithfulness_score, 2),
            "retrieval_quality": round(self.retrieval_quality, 2),
            "citation_accuracy": round(self.citation_accuracy, 2),
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

    def retrieval_quality(self, query: str, chunks: list[Document]) -> float:
        """
        Compute retrieval quality as the average per-chunk keyword coverage.

        For each retrieved chunk, calculate what fraction of query keywords
        appear in that chunk, then average across all chunks. This measures
        how consistently relevant the retrieved chunks are — a high score
        means most chunks individually contain query keywords, not just the
        combined corpus.

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

        per_chunk_scores = []
        for chunk in chunks:
            chunk_text = chunk.page_content.lower()
            matched = sum(1 for kw in keywords if kw in chunk_text)
            per_chunk_scores.append(matched / len(keywords))

        return round((sum(per_chunk_scores) / len(per_chunk_scores)) * 100.0, 2)

    def citation_accuracy(self, answer: str, chunks: list[Document]) -> float:
        """
        Compute citation accuracy as the fraction of citations in the answer
        that reference sources and pages actually present in the retrieved chunks.

        A citation [Source: file.pdf, Page N] is "valid" if:
        - The filename matches a source in the retrieved chunks, AND
        - The page number is within a reasonable range of pages in those chunks
          (exact match preferred; ±1 page tolerance for chunk boundary cases).

        Args:
            answer: The LLM-generated answer text.
            chunks: Retrieved document chunks.

        Returns:
            Float in [0.0, 100.0]. Returns 100.0 if no citations are present
            (no citations to be wrong about).
        """
        if not answer:
            return 100.0  # no answer, no citations to check

        # Extract all citations from the answer
        citation_matches = re.findall(
            r'\[Source:\s*(.+?\.pdf),\s*Page\s*(\d+)\]',
            answer,
            re.IGNORECASE
        )

        if not citation_matches:
            return 100.0  # no citations present — nothing to penalise

        if not chunks:
            return 0.0  # citations present but no chunks to validate against

        # Build a set of (source, page) pairs from retrieved chunks
        valid_sources: set[str] = set()
        valid_pages: dict[str, set[int]] = {}  # source -> set of page numbers

        for chunk in chunks:
            src = chunk.metadata.get("source", "").lower().strip()
            page = chunk.metadata.get("page", None)
            if src:
                valid_sources.add(src)
                if page is not None:
                    valid_pages.setdefault(src, set()).add(int(page))

        valid_count = 0
        for filename, page_str in citation_matches:
            cited_src = filename.lower().strip()
            cited_page = int(page_str)

            # Check source exists
            if cited_src not in valid_sources:
                continue

            # Check page is valid (exact or ±1 tolerance for chunk boundaries)
            chunk_pages = valid_pages.get(cited_src, set())
            if not chunk_pages:
                # Source exists but no page metadata — give benefit of the doubt
                valid_count += 1
                continue

            if any(abs(cited_page - p) <= 1 for p in chunk_pages):
                valid_count += 1

        return round((valid_count / len(citation_matches)) * 100.0, 2)

    def answer_faithfulness_score(self, answer: str, chunks: list[Document]) -> float:
        """
        Compute the fraction of answer sentences whose meaningful tokens are
        sufficiently supported by the retrieved chunk corpus.

        A sentence is "supported" if at least SUPPORT_THRESHOLD (50%) of its
        meaningful tokens appear in the combined chunk text.

        Args:
            answer: The LLM-generated answer text.
            chunks: Retrieved document chunks.

        Returns:
            Float in [0.0, 100.0].
        """
        if not answer or not chunks:
            return 0.0

        # Build corpus from all retrieved chunks (lowercased)
        corpus_text = " ".join(chunk.page_content.lower() for chunk in chunks)

        # Split answer into sentences on . ! ? boundaries
        sentences = [s.strip() for s in re.split(r'[.!?]', answer) if s.strip()]
        if not sentences:
            return 0.0

        supported = 0
        scoreable = 0

        for sentence in sentences:
            tokens = _tokenize(sentence)
            if not tokens:
                continue  # skip stopword-only sentences
            scoreable += 1
            matched = sum(1 for t in tokens if t in corpus_text)
            if matched / len(tokens) >= SUPPORT_THRESHOLD:
                supported += 1

        if scoreable == 0:
            return 0.0

        return round((supported / scoreable) * 100.0, 2)

    def compute(
        self,
        query: str,
        retrieved_chunks: list[Document],
        answer: str = "",
        query_embedding: Optional[list[float]] = None,  # kept for compat, unused
    ) -> QualityReport:
        """
        Compute both quality metrics and return a QualityReport.
        No API calls are made.

        Args:
            query: The user's query string.
            retrieved_chunks: Chunks retrieved from the vector store.
            answer: The LLM-generated answer text (for faithfulness scoring).
            query_embedding: Ignored. Kept for interface compatibility.

        Returns:
            QualityReport with confidence_score, keyword_match_accuracy,
            and answer_faithfulness_score.
        """
        if not retrieved_chunks:
            logger.warning("No chunks provided for quality metrics computation.")
            return QualityReport(confidence_score=0.0, keyword_match_accuracy=0.0, answer_faithfulness_score=0.0)

        kma = self.keyword_match_accuracy(query, retrieved_chunks)
        cs = self.confidence_score(query, retrieved_chunks)
        afs = self.answer_faithfulness_score(answer, retrieved_chunks)
        rq = self.retrieval_quality(query, retrieved_chunks)
        ca = self.citation_accuracy(answer, retrieved_chunks)

        logger.debug(
            "Quality metrics — confidence: %.1f, kma: %.1f, faithfulness: %.1f, retrieval: %.1f, citation: %.1f",
            cs, kma, afs, rq, ca
        )
        return QualityReport(
            confidence_score=cs,
            keyword_match_accuracy=kma,
            answer_faithfulness_score=afs,
            retrieval_quality=rq,
            citation_accuracy=ca,
        )
