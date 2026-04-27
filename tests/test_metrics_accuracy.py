"""
Accuracy tests for all 5 quality metrics.
Tests cover realistic scenarios to validate correctness and identify weaknesses.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from langchain_core.documents import Document
from src.quality_metrics import QualityMetricsComputer

c = QualityMetricsComputer()


def compute(query, chunks, answer=""):
    return c.compute(query, chunks, answer=answer)


# ---------------------------------------------------------------------------
# Keyword Match Accuracy
# ---------------------------------------------------------------------------

class TestKeywordMatchAccuracy:

    def test_all_keywords_present(self):
        chunks = [Document(page_content="revenue profit earnings quarterly report")]
        r = compute("revenue profit earnings", chunks)
        assert r.keyword_match_accuracy == 100.0

    def test_no_keywords_present(self):
        chunks = [Document(page_content="climate change global warming temperature")]
        r = compute("revenue profit earnings", chunks)
        assert r.keyword_match_accuracy == 0.0

    def test_partial_keywords(self):
        chunks = [Document(page_content="revenue report analysis")]
        r = compute("revenue profit earnings", chunks)
        # Only "revenue" matches out of 3 keywords → ~33%
        assert 30 <= r.keyword_match_accuracy <= 40

    def test_stopwords_ignored(self):
        # "what is the" are all stopwords — no meaningful keywords
        chunks = [Document(page_content="some random text")]
        r = compute("what is the", chunks)
        assert r.keyword_match_accuracy == 0.0

    def test_case_insensitive(self):
        chunks = [Document(page_content="REVENUE PROFIT EARNINGS")]
        r = compute("revenue profit earnings", chunks)
        assert r.keyword_match_accuracy == 100.0


# ---------------------------------------------------------------------------
# Retrieval Quality
# ---------------------------------------------------------------------------

class TestRetrievalQuality:

    def test_all_chunks_relevant(self):
        chunks = [
            Document(page_content="revenue profit earnings quarterly"),
            Document(page_content="revenue profit earnings annual"),
        ]
        r = compute("revenue profit earnings", chunks)
        assert r.retrieval_quality == 100.0

    def test_only_one_chunk_relevant(self):
        chunks = [
            Document(page_content="revenue profit financial results"),
            Document(page_content="climate change global warming"),
            Document(page_content="sports football basketball"),
            Document(page_content="cooking recipes ingredients"),
            Document(page_content="travel destinations tourism"),
        ]
        r = compute("revenue profit", chunks)
        # Only 1 of 5 chunks has both keywords → 20%
        assert r.retrieval_quality <= 25

    def test_rq_differs_from_kma_when_keywords_in_one_chunk(self):
        """KMA=100 (keywords exist somewhere) but RQ should be low (only in 1 chunk)."""
        chunks = [
            Document(page_content="revenue profit financial results"),
            Document(page_content="unrelated content about cooking"),
            Document(page_content="more unrelated content about sports"),
        ]
        r = compute("revenue profit", chunks)
        assert r.keyword_match_accuracy == 100.0
        assert r.retrieval_quality < r.keyword_match_accuracy

    def test_empty_chunks(self):
        r = compute("revenue", [])
        assert r.retrieval_quality == 0.0


# ---------------------------------------------------------------------------
# Answer Faithfulness Score
# ---------------------------------------------------------------------------

class TestAnswerFaithfulness:

    def test_answer_copies_chunk_text(self):
        chunks = [Document(page_content="revenue profit quarterly earnings report financial results")]
        r = compute("revenue profit", chunks,
                    answer="Revenue profit quarterly earnings report financial results.")
        assert r.answer_faithfulness_score >= 80.0

    def test_completely_off_topic_answer(self):
        chunks = [Document(page_content="revenue profit quarterly earnings report")]
        r = compute("revenue profit", chunks,
                    answer="The weather is sunny today. Climate change affects agriculture. Birds migrate south.")
        assert r.answer_faithfulness_score <= 20.0

    def test_empty_answer(self):
        chunks = [Document(page_content="revenue report")]
        r = compute("revenue", chunks, answer="")
        assert r.answer_faithfulness_score == 0.0

    def test_empty_chunks(self):
        r = compute("revenue", [], answer="Revenue was high.")
        assert r.answer_faithfulness_score == 0.0

    def test_stopword_only_sentences_excluded(self):
        """Sentences with only stopwords should not count against faithfulness."""
        chunks = [Document(page_content="revenue profit earnings")]
        # "It is the" → all stopwords, should be excluded from scoring
        r = compute("revenue", chunks, answer="Revenue was high. It is the.")
        # Only "Revenue was high" is scoreable — revenue appears in chunk → supported
        assert r.answer_faithfulness_score >= 50.0

    def test_paraphrased_answer_moderate_faithfulness(self):
        chunks = [Document(page_content="the company revenue increased significantly during fiscal year 2024 showing strong growth trajectory")]
        r = compute("revenue growth", chunks,
                    answer="Company revenue grew strongly in 2024.")
        # "revenue" and "2024" appear in chunk; "grew" doesn't but "growth" does
        assert r.answer_faithfulness_score >= 40.0

    def test_mixed_answer_partial_faithfulness(self):
        chunks = [Document(page_content="revenue profit earnings quarterly report")]
        r = compute("revenue profit", chunks,
                    answer="Revenue profit earnings quarterly report. The moon is made of cheese and unicorns fly.")
        # First sentence: fully supported. Second: not supported.
        assert 40.0 <= r.answer_faithfulness_score <= 60.0


# ---------------------------------------------------------------------------
# Citation Accuracy
# ---------------------------------------------------------------------------

class TestCitationAccuracy:

    def test_correct_citation(self):
        chunks = [Document(page_content="revenue report", metadata={"source": "a.pdf", "page": 5})]
        r = compute("revenue", chunks,
                    answer="Revenue [Source: a.pdf, Page 5].")
        assert r.citation_accuracy == 100.0

    def test_wrong_page_number(self):
        chunks = [Document(page_content="revenue report", metadata={"source": "a.pdf", "page": 5})]
        r = compute("revenue", chunks,
                    answer="Revenue [Source: a.pdf, Page 99].")
        assert r.citation_accuracy == 0.0

    def test_page_off_by_one_tolerated(self):
        """±1 page tolerance for chunk boundary cases."""
        chunks = [Document(page_content="revenue report", metadata={"source": "a.pdf", "page": 5})]
        r = compute("revenue", chunks,
                    answer="Revenue [Source: a.pdf, Page 6].")
        assert r.citation_accuracy == 100.0

    def test_wrong_filename(self):
        chunks = [Document(page_content="revenue report", metadata={"source": "a.pdf", "page": 1})]
        r = compute("revenue", chunks,
                    answer="Revenue [Source: b.pdf, Page 1].")
        assert r.citation_accuracy == 0.0

    def test_no_citations_returns_100(self):
        """No citations = nothing to be wrong about."""
        chunks = [Document(page_content="revenue report", metadata={"source": "a.pdf", "page": 1})]
        r = compute("revenue", chunks, answer="Revenue was high.")
        assert r.citation_accuracy == 100.0

    def test_mixed_citations_partial_accuracy(self):
        chunks = [Document(page_content="revenue report", metadata={"source": "a.pdf", "page": 1})]
        r = compute("revenue", chunks,
                    answer="Revenue [Source: a.pdf, Page 1]. Also [Source: b.pdf, Page 99].")
        # 1 valid out of 2 → 50%
        assert r.citation_accuracy == 50.0

    def test_empty_answer_returns_100(self):
        chunks = [Document(page_content="revenue report", metadata={"source": "a.pdf", "page": 1})]
        r = compute("revenue", chunks, answer="")
        assert r.citation_accuracy == 100.0

    def test_no_chunks_with_citations_returns_0(self):
        r = compute("revenue", [],
                    answer="Revenue [Source: a.pdf, Page 1].")
        assert r.citation_accuracy == 0.0

    def test_case_insensitive_filename(self):
        """Citation filename matching should be case-insensitive."""
        chunks = [Document(page_content="revenue report", metadata={"source": "Report.PDF", "page": 1})]
        r = compute("revenue", chunks,
                    answer="Revenue [Source: report.pdf, Page 1].")
        assert r.citation_accuracy == 100.0


# ---------------------------------------------------------------------------
# Cross-metric independence
# ---------------------------------------------------------------------------

class TestMetricIndependence:

    def test_kma_independent_of_answer(self):
        """Changing the answer should not affect keyword_match_accuracy."""
        chunks = [Document(page_content="revenue profit earnings")]
        r1 = compute("revenue profit", chunks, answer="Revenue was high.")
        r2 = compute("revenue profit", chunks, answer="The sky is blue.")
        assert r1.keyword_match_accuracy == r2.keyword_match_accuracy

    def test_faithfulness_independent_of_query(self):
        """Changing the query should not affect answer_faithfulness_score."""
        chunks = [Document(page_content="revenue profit earnings quarterly")]
        answer = "Revenue profit earnings quarterly."
        r1 = compute("revenue profit", chunks, answer=answer)
        r2 = compute("climate change", chunks, answer=answer)
        assert r1.answer_faithfulness_score == r2.answer_faithfulness_score

    def test_citation_accuracy_independent_of_query(self):
        """Changing the query should not affect citation_accuracy."""
        chunks = [Document(page_content="revenue report", metadata={"source": "a.pdf", "page": 1})]
        answer = "Revenue [Source: a.pdf, Page 1]."
        r1 = compute("revenue", chunks, answer=answer)
        r2 = compute("climate change", chunks, answer=answer)
        assert r1.citation_accuracy == r2.citation_accuracy

    def test_all_metrics_in_range(self):
        """All metrics must always be in [0, 100]."""
        chunks = [Document(page_content="revenue profit earnings", metadata={"source": "a.pdf", "page": 1})]
        r = compute("revenue profit", chunks,
                    answer="Revenue profit [Source: a.pdf, Page 1].")
        assert 0 <= r.confidence_score <= 100
        assert 0 <= r.keyword_match_accuracy <= 100
        assert 0 <= r.answer_faithfulness_score <= 100
        assert 0 <= r.retrieval_quality <= 100
        assert 0 <= r.citation_accuracy <= 100
