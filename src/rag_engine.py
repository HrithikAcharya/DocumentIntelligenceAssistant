"""
RAG Engine for the Document Intelligence Assistant.
Orchestrates retrieval, prompt construction, LLM generation, and quality metrics.
"""
import logging
from dataclasses import dataclass, field
from typing import Optional

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from src.cache import ResponseCache
from src.citation_parser import Citation, CitationParser
from src.prompt_builder import OperationalMode, PromptBuilder
from src.quality_metrics import QualityMetricsComputer, QualityReport
from src.rate_limiter import QuotaExhaustedError, RateLimiter

logger = logging.getLogger(__name__)

# Approximate tokens per character ratio for token estimation
# (conservative estimate: 1 token ≈ 4 characters)
CHARS_PER_TOKEN = 4
# Gemini 1.5 Flash context window limit
MAX_CONTEXT_TOKENS = 1_048_576


@dataclass
class RAGResponse:
    """Complete response from the RAG pipeline."""
    answer: str
    citations: list[Citation]
    quality_report: QualityReport
    retrieved_chunks: list[Document]
    prompt_token_count: int


class RAGEngine:
    """
    Orchestrates the full RAG pipeline:
    cache check → retrieval → prompt construction → rate limiting →
    LLM generation → citation extraction → quality metrics → cache store.
    """

    def __init__(
        self,
        vector_store,
        llm,
        embeddings,
        prompt_builder: PromptBuilder,
        cache: ResponseCache,
        rate_limiter: RateLimiter,
        top_k: int = 5,
        max_retries: int = 5,
        initial_retry_delay: float = 1.0,
    ) -> None:
        """
        Args:
            vector_store: FAISS vector store instance.
            llm: ChatGoogleGenerativeAI instance with streaming=True.
            embeddings: GoogleGenerativeAIEmbeddings for quality metrics.
            prompt_builder: PromptBuilder for mode-specific prompts.
            cache: ResponseCache for avoiding redundant API calls.
            rate_limiter: RateLimiter for free-tier compliance.
            top_k: Number of chunks to retrieve (default 5).
            max_retries: Max retry attempts on rate limit errors (default 5).
            initial_retry_delay: Initial backoff delay in seconds (default 1.0).
        """
        self.vector_store = vector_store
        self.llm = llm
        self.embeddings = embeddings
        self.prompt_builder = prompt_builder
        self.cache = cache
        self.rate_limiter = rate_limiter
        self.top_k = top_k
        self.max_retries = max_retries
        self.initial_retry_delay = initial_retry_delay
        self._citation_parser = CitationParser()
        self._quality_computer = QualityMetricsComputer(embeddings)

    def _retrieve(self, query: str) -> list[Document]:
        """
        Fetch top-K semantically relevant chunks from the FAISS vector store.

        Strategy:
        - Single document: uses MMR (Maximal Marginal Relevance) to fetch
          diverse, non-redundant chunks so overlapping passages don't crowd
          out independent evidence.
        - Multiple documents: applies per-source diversity capping so every
          document is represented regardless of semantic similarity to the query.

        Args:
            query: The user's query string.

        Returns:
            List of retrieved Document chunks ordered by relevance.
        """
        try:
            # Discover unique sources in the store
            try:
                all_docs = list(self.vector_store.docstore._dict.values())
                unique_sources = set(
                    d.metadata.get("source", "unknown") for d in all_docs
                )
            except Exception:
                unique_sources = set()

            num_sources = max(len(unique_sources), 1)

            if num_sources <= 1:
                # Single document — use MMR for intra-document diversity
                # (avoids returning near-duplicate overlapping chunks)
                results = self.vector_store.max_marginal_relevance_search(
                    query,
                    k=self.top_k,
                    fetch_k=min(self.top_k * 4, len(all_docs) if all_docs else self.top_k * 4),
                    lambda_mult=0.7,  # 0=max diversity, 1=max relevance; 0.7 balances both
                )
                logger.info(
                    "MMR retrieval (single doc): %d chunks from %d candidates",
                    len(results), min(self.top_k * 4, len(all_docs) if all_docs else self.top_k * 4)
                )
            else:
                # Multiple documents — fetch broad candidate pool then cap per source
                candidate_k = min(self.top_k * 4, len(all_docs) if all_docs else self.top_k * 4)
                candidates = self.vector_store.similarity_search(query, k=candidate_k)

                per_source_limit = max(2, self.top_k // num_sources)
                from collections import defaultdict
                seen: dict[str, int] = defaultdict(int)
                results = []
                for chunk in candidates:
                    src = chunk.metadata.get("source", "unknown")
                    if seen[src] < per_source_limit:
                        results.append(chunk)
                        seen[src] += 1
                    if len(results) >= self.top_k:
                        break

                logger.info(
                    "Diversity retrieval (%d docs): %d chunks, cap %d/source. Distribution: %s",
                    num_sources, len(results), per_source_limit,
                    dict(seen),
                )

            # Log relevance scores for the top chunks (for debugging)
            try:
                scored = self.vector_store.similarity_search_with_relevance_scores(query, k=3)
                top_scores = [(round(score, 3), doc.metadata.get("source","?"), doc.metadata.get("page","?"))
                              for doc, score in scored]
                logger.info("Top-3 relevance scores: %s", top_scores)
            except Exception:
                pass  # Scoring is diagnostic only; don't break retrieval

            logger.debug("Retrieved %d chunks for query.", len(results))
            return results

        except Exception as e:
            logger.error("Retrieval failed: %s", e)
            return []

    def _retrieve_balanced(self, query: str, doc_filenames: dict) -> list[Document]:
        """
        Retrieve chunks ensuring balanced representation across all documents.

        Scans the FAISS docstore directly to get chunks per document,
        guaranteeing every document is represented in the context.
        Uses the first N chunks per document (ordered by page number)
        to avoid re-embedding overhead.
        """
        if not doc_filenames:
            return self._retrieve(query)

        filenames = list(doc_filenames.values())
        if len(filenames) <= 1:
            return self._retrieve(query)

        chunks_per_doc = max(3, self.top_k // max(len(doc_filenames), 1))

        # Scan docstore to get ALL chunks grouped by source
        by_source: dict[str, list[Document]] = {}
        try:
            all_docs = list(self.vector_store.docstore._dict.values())
            for doc in all_docs:
                src = doc.metadata.get("source", "unknown")
                by_source.setdefault(src, []).append(doc)
            logger.info(
                "Docstore scan: %d total chunks, sources: %s",
                len(all_docs), list(by_source.keys())
            )
        except Exception as e:
            logger.warning("Docstore scan failed: %s", e)
            return self._retrieve(query)

        # Also do a global similarity search to get the most relevant chunks
        try:
            similar = self.vector_store.similarity_search(query, k=min(50, len(all_docs)))
            for doc in similar:
                src = doc.metadata.get("source", "unknown")
                # Mark similarity-retrieved chunks so we prefer them
                doc.metadata["_sim_retrieved"] = True
                if src in by_source:
                    # Move to front of the list for this source
                    existing = [d for d in by_source[src] if d.page_content != doc.page_content]
                    by_source[src] = [doc] + existing
        except Exception:
            pass  # Fall back to page-order selection

        # Select top chunks_per_doc from each document
        selected = []
        for filename in filenames:
            doc_chunks = by_source.get(filename, [])
            if not doc_chunks:
                logger.warning("No chunks in docstore for: %s", filename)
                continue
            # Sort by page number for coherent context
            doc_chunks_sorted = sorted(
                doc_chunks[:chunks_per_doc * 3],
                key=lambda d: (
                    0 if d.metadata.get("_sim_retrieved") else 1,
                    d.metadata.get("page", 999)
                )
            )
            top = doc_chunks_sorted[:chunks_per_doc]
            selected.extend(top)
            logger.info("Selected %d chunks from %s", len(top), filename)

        logger.info(
            "Balanced retrieval: %d chunks from %d/%d documents",
            len(selected), len([f for f in filenames if f in by_source]), len(filenames)
        )
        return selected

    def _estimate_prompt_tokens(
        self, system_prompt: str, chunks: list[Document], query: str
    ) -> int:
        """
        Estimate the total token count (input + expected output) for rate limiting.

        Uses a conservative character-to-token ratio (4 chars ≈ 1 token) for
        input, then adds an estimated output budget. The output estimate is
        1.5x the input to account for the LLM's response tokens, which the
        API counts against the TPM limit but which we cannot know in advance.

        Args:
            system_prompt: The mode-specific system prompt string.
            chunks: Retrieved document chunks.
            query: The user's query string.

        Returns:
            Estimated total token count (input + output buffer).
        """
        context_text = "\n\n".join(chunk.page_content for chunk in chunks)
        input_chars = len(system_prompt) + len(context_text) + len(query)
        input_tokens = input_chars // CHARS_PER_TOKEN
        # Add 50% buffer for expected output tokens (LLM response)
        total_estimated = int(input_tokens * 1.5)
        logger.info(
            "Estimated tokens — input: %d, total with output buffer: %d",
            input_tokens, total_estimated
        )
        return total_estimated

    def _build_context(self, chunks: list[Document], doc_filenames: dict = None) -> str:
        """
        Format retrieved chunks into a context string for the prompt.

        In Compare mode, prepends a document inventory so the LLM knows
        exactly which documents are available, preventing it from ignoring
        documents that happen to have fewer retrieved chunks.
        """
        parts = []

        # In Compare mode, add a document inventory header
        if doc_filenames and len(doc_filenames) > 1:
            filenames = list(doc_filenames.values())
            inventory = "DOCUMENTS AVAILABLE FOR COMPARISON:\n"
            for i, fname in enumerate(filenames, 1):
                inventory += f"  {i}. {fname}\n"
            inventory += "\nYou MUST reference ALL of the above documents in your response.\n"
            inventory += "If a document has no relevant content for this query, explicitly state that.\n"
            parts.append(inventory)

        # Group chunks by source for cleaner context
        by_source: dict[str, list[Document]] = {}
        for chunk in chunks:
            src = chunk.metadata.get("source", "unknown")
            by_source.setdefault(src, []).append(chunk)

        for source, source_chunks in by_source.items():
            parts.append(f"=== FROM: {source} ===")
            for chunk in source_chunks:
                page = chunk.metadata.get("page", "?")
                parts.append(f"[Source: {source}, Page {page}]\n{chunk.page_content}")

        return "\n\n---\n\n".join(parts)

    async def query(
        self,
        query: str,
        mode: OperationalMode,
        doc_ids: list[str],
        callback_handler=None,
        doc_filenames: dict = None,
        previous_turn: Optional[dict] = None,
    ) -> RAGResponse:
        """
        Execute the full RAG pipeline for a user query.

        In Compare mode, uses balanced per-document retrieval to ensure
        all uploaded documents contribute to the context.

        Args:
            query: The user's query string.
            mode: Current operational mode (SINGLE_DOC or COMPARE).
            doc_ids: List of document IDs for cache keying.
            callback_handler: Optional LangChain callback handler for streaming.
            doc_filenames: dict mapping doc_id -> filename (used for balanced retrieval).
            previous_turn: Optional dict with "query" and "answer" keys from
                the immediately preceding exchange, for conversational context.

        Returns:
            RAGResponse with answer, citations, quality report, and metadata.

        Raises:
            QuotaExhaustedError: If the daily API quota is exhausted.
        """
        # Check cache first
        cache_key = self.cache.make_key(query, doc_ids)
        cached = self.cache.get(cache_key)
        if cached is not None:
            logger.info("Cache hit for query.")
            return cached

        # Retrieve relevant chunks — balanced across docs in Compare mode
        logger.info(
            "Query mode: %s | doc_filenames: %s | len: %d",
            mode, list(doc_filenames.values()) if doc_filenames else None,
            len(doc_filenames) if doc_filenames else 0
        )
        if mode == OperationalMode.COMPARE and doc_filenames and len(doc_filenames) > 1:
            logger.info("Using balanced retrieval")
            chunks = self._retrieve_balanced(query, doc_filenames)
        else:
            logger.info("Using standard retrieval")
            chunks = self._retrieve(query)

        if not chunks:
            logger.warning("No chunks retrieved. Returning empty-context response.")
            empty_report = QualityReport(confidence_score=0.0, keyword_match_accuracy=0.0)
            return RAGResponse(
                answer="The answer to this question is not found in the provided documents.",
                citations=[],
                quality_report=empty_report,
                retrieved_chunks=[],
                prompt_token_count=0,
            )

        # Build prompt and estimate tokens
        system_prompt = self.prompt_builder.get_system_prompt(mode)
        estimated_tokens = self._estimate_prompt_tokens(system_prompt, chunks, query)

        # Enforce rate limits proactively
        await self.rate_limiter.check_and_wait(estimated_tokens)

        # Build context string — include document inventory for Compare mode
        context = self._build_context(chunks, doc_filenames if mode == OperationalMode.COMPARE else None)

        # Log context sources for debugging
        context_sources = set(c.metadata.get("source", "?") for c in chunks)
        logger.info("Context sources being sent to LLM: %s", context_sources)
        logger.info("Total chunks in context: %d", len(chunks))

        # Build prompt template
        prompt_template = self.prompt_builder.build_prompt(mode, previous_turn)

        # Build LCEL chain
        chain = prompt_template | self.llm | StrOutputParser()

        # Prepare callbacks
        callbacks = []
        if callback_handler is not None:
            callbacks.append(callback_handler)

        # Execute chain with exponential backoff
        async def _invoke():
            return await chain.ainvoke(
                {"context": context, "question": query},
                config={"callbacks": callbacks, "run_name": "doc-intelligence-rag", "tags": [mode.value]},
            )

        answer = await self.rate_limiter.execute_with_backoff(
            _invoke,
            max_retries=self.max_retries,
            initial_delay=self.initial_retry_delay,
        )

        # Record the completed request
        self.rate_limiter.record_request(estimated_tokens)

        # Extract citations
        citations = self._citation_parser.extract_citations(answer)

        # Compute quality metrics
        quality_report = self._quality_computer.compute(query, chunks, answer=answer)

        # Build response
        response = RAGResponse(
            answer=answer,
            citations=citations,
            quality_report=quality_report,
            retrieved_chunks=chunks,
            prompt_token_count=estimated_tokens,
        )

        # Store in cache
        self.cache.set(cache_key, response)

        return response
