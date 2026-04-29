"""
Document ingestion pipeline for the Document Intelligence Assistant.
Handles PDF validation, text extraction, chunking, and FAISS vector store creation.

Uses local HuggingFace embeddings (all-MiniLM-L6-v2) — no API calls,
no quota limits, runs entirely on CPU.
"""
import hashlib
import logging
from typing import Optional

import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class DocumentIngestor:
    """
    Full PDF ingestion pipeline: validate → extract → chunk → embed → index.
    
    Produces a FAISS vector store from one or more PDF files, preserving
    page-level metadata on every chunk for citation deep linking.
    """

    def __init__(
        self,
        embeddings,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ) -> None:
        """
        Args:
            embeddings: GoogleGenerativeAIEmbeddings instance for vectorization.
            chunk_size: Maximum characters per chunk (default 500).
            chunk_overlap: Character overlap between consecutive chunks (default 50).
        """
        self.embeddings = embeddings
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            add_start_index=True,
        )

    def validate_pdf(self, file_bytes: bytes) -> bool:
        """
        Validate that bytes represent a parseable PDF document.
        
        Returns True iff the bytes start with %PDF and PyMuPDF can open them
        without raising an exception.
        
        Args:
            file_bytes: Raw bytes of the uploaded file.
            
        Returns:
            True if valid PDF, False otherwise.
        """
        if not file_bytes or not file_bytes.startswith(b"%PDF"):
            return False
        try:
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            doc.close()
            return True
        except Exception:
            return False

    def extract_pages(self, file_bytes: bytes, filename: str) -> list[Document]:
        """
        Extract text from each page of a PDF, returning one Document per page.
        
        Args:
            file_bytes: Raw bytes of the PDF file.
            filename: Original filename, used for citation metadata.
            
        Returns:
            List of LangChain Documents with metadata:
            {source, page (1-indexed), doc_id}
            
        Raises:
            ValueError: If the PDF has 0 readable pages or cannot be opened.
        """
        doc_id = hashlib.sha256(filename.encode("utf-8")).hexdigest()
        documents = []

        try:
            pdf = fitz.open(stream=file_bytes, filetype="pdf")
        except Exception as e:
            raise ValueError(
                f"The uploaded file '{filename}' appears to be corrupted and could not be read. "
                "Please try a different file."
            ) from e

        try:
            for page_num in range(len(pdf)):
                page = pdf[page_num]
                text = page.get_text()
                if text.strip():
                    documents.append(
                        Document(
                            page_content=text,
                            metadata={
                                "source": filename,
                                "page": page_num + 1,  # 1-indexed
                                "doc_id": doc_id,
                            },
                        )
                    )
        finally:
            pdf.close()

        if not documents:
            raise ValueError(
                f"The uploaded PDF '{filename}' contains no readable text. "
                "Please try a different file."
            )

        logger.info("Extracted %d pages from '%s'", len(documents), filename)
        return documents

    def chunk_documents(self, documents: list[Document]) -> list[Document]:
        """
        Split documents into chunks, preserving source metadata.
        
        Args:
            documents: List of page-level Documents from extract_pages().
            
        Returns:
            List of chunk Documents with metadata:
            {source, page, doc_id, chunk_index}
        """
        all_chunks = []
        for doc in documents:
            chunks = self._splitter.split_documents([doc])
            for idx, chunk in enumerate(chunks):
                # Preserve all original metadata and add chunk_index
                chunk.metadata["chunk_index"] = idx
                all_chunks.append(chunk)

        logger.info("Created %d chunks from %d pages", len(all_chunks), len(documents))
        return all_chunks

    def build_vector_store(self, chunks: list[Document]) -> FAISS:
        """
        Embed chunks using local HuggingFace embeddings and build a FAISS index.
        No API calls — runs entirely on CPU with no quota limits.

        Args:
            chunks: List of chunk Documents from chunk_documents().

        Returns:
            FAISS vector store instance.
        """
        logger.info("Building FAISS index for %d chunks (local embeddings)...", len(chunks))
        vector_store = FAISS.from_documents(chunks, self.embeddings)
        logger.info("FAISS index built successfully.")
        return vector_store

    def ingest(self, files: list[tuple[str, bytes]]) -> FAISS:
        """
        Full ingestion pipeline: validate → extract → chunk → embed → index.
        
        Args:
            files: List of (filename, file_bytes) tuples.
            
        Returns:
            FAISS vector store containing all document chunks.
            
        Raises:
            ValueError: If any file is invalid, corrupted, or has no readable text.
        """
        all_chunks = []

        for filename, file_bytes in files:
            logger.info("Ingesting '%s'...", filename)

            # Validate
            if not self.validate_pdf(file_bytes):
                raise ValueError(
                    f"The file '{filename}' is not a valid PDF. "
                    "Only PDF files are accepted."
                )

            # Extract pages
            pages = self.extract_pages(file_bytes, filename)

            # Chunk
            chunks = self.chunk_documents(pages)
            all_chunks.extend(chunks)

        if not all_chunks:
            raise ValueError("No readable content found in the uploaded files.")

        # Build vector store
        return self.build_vector_store(all_chunks)

    def ingest_incremental(
        self,
        filename: str,
        file_bytes: bytes,
        existing_store: FAISS,
    ) -> FAISS:
        """
        Incrementally add a single new document to an existing FAISS vector store.

        Only the new document is embedded — the existing store is not rebuilt.
        Uses FAISS.merge_from() to combine the new index into the existing one
        in-place, which is significantly faster than rebuilding from scratch.

        Args:
            filename: Name of the new PDF file.
            file_bytes: Raw bytes of the new PDF file.
            existing_store: The current FAISS vector store to merge into.

        Returns:
            The updated FAISS vector store (same object, mutated in-place).

        Raises:
            ValueError: If the file is invalid, corrupted, or has no readable text.
        """
        logger.info("Incremental ingest of '%s' into existing store...", filename)

        # Validate
        if not self.validate_pdf(file_bytes):
            raise ValueError(
                f"The file '{filename}' is not a valid PDF. "
                "Only PDF files are accepted."
            )

        # Extract and chunk only the new document
        pages = self.extract_pages(file_bytes, filename)
        chunks = self.chunk_documents(pages)

        # Build a small FAISS index for just the new document
        logger.info("Embedding %d chunks for '%s'...", len(chunks), filename)
        new_store = FAISS.from_documents(chunks, self.embeddings)

        # Merge new index into existing store (no re-embedding of existing docs)
        existing_store.merge_from(new_store)
        logger.info(
            "Merged '%s' into existing store. Total chunks: %d",
            filename,
            len(existing_store.docstore._dict),
        )
        return existing_store
