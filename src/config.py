"""
Application configuration for Document Intelligence Assistant.
Loads settings from environment variables with sensible defaults.
"""
import os
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv

# Load .env file if present
load_dotenv()


@dataclass
class AppConfig:
    """Centralised configuration loaded from environment variables."""

    google_api_key: str
    langsmith_api_key: Optional[str]
    langsmith_project: str
    model_name: str
    embedding_model: str
    top_k: int
    chunk_size: int
    chunk_overlap: int
    max_retries: int
    initial_retry_delay: float

    @classmethod
    def from_env(cls) -> "AppConfig":
        """Create AppConfig from environment variables with defaults."""
        google_api_key = os.environ.get("GOOGLE_API_KEY", "")
        if not google_api_key:
            raise ValueError(
                "GOOGLE_API_KEY environment variable is required. "
                "Please set it in your .env file or environment."
            )

        return cls(
            google_api_key=google_api_key,
            langsmith_api_key=os.environ.get("LANGCHAIN_API_KEY") or None,
            langsmith_project=os.environ.get("LANGCHAIN_PROJECT", "doc-intelligence"),
            model_name=os.environ.get("GEMINI_MODEL", "models/gemini-2.5-flash"),
            embedding_model="local:all-MiniLM-L6-v2",  # local, no API quota
            top_k=int(os.environ.get("TOP_K", "4")),
            chunk_size=int(os.environ.get("CHUNK_SIZE", "600")),
            chunk_overlap=int(os.environ.get("CHUNK_OVERLAP", "60")),
            max_retries=int(os.environ.get("MAX_RETRIES", "5")),
            initial_retry_delay=float(os.environ.get("INITIAL_RETRY_DELAY", "1.0")),
        )
