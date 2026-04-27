"""
In-memory response cache for the Document Intelligence Assistant.
Keyed by SHA-256 hash of query + document set to avoid redundant LLM API calls.
"""
import hashlib
from typing import Any, Optional


class ResponseCache:
    """
    In-memory cache keyed by SHA-256(query + sorted_doc_ids).
    Stores RAGResponse objects to avoid redundant LLM API calls for
    identical queries against the same document set.
    Persists for the duration of the application session.
    """

    def __init__(self) -> None:
        self._store: dict[str, Any] = {}

    def make_key(self, query: str, doc_ids: list[str]) -> str:
        """
        Create a deterministic cache key from query and document IDs.

        Args:
            query: The user's query string.
            doc_ids: List of document identifiers (SHA-256 of filenames).

        Returns:
            SHA-256 hex digest of query + sorted(doc_ids) concatenation.
        """
        key_material = query + "".join(sorted(doc_ids))
        return hashlib.sha256(key_material.encode("utf-8")).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve a cached response.

        Args:
            key: Cache key from make_key().

        Returns:
            Cached RAGResponse or None on cache miss.
        """
        return self._store.get(key)

    def set(self, key: str, response: Any) -> None:
        """
        Store a response in the cache.

        Args:
            key: Cache key from make_key().
            response: RAGResponse to cache.
        """
        self._store[key] = response

    def invalidate_all(self) -> None:
        """
        Clear all cache entries.
        Called when the user uploads a new document or switches modes.
        """
        self._store.clear()

    def __len__(self) -> int:
        return len(self._store)
