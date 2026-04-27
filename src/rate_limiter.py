"""
Proactive rate limiter for Google Gemini free-tier API compliance.
Enforces limits BEFORE each API call using a sliding-window approach.

Free-tier limits (as of 2025):
  gemini-2.0-flash-lite (LLM):
    - 30 requests per minute (RPM)
    - 1,000,000 tokens per minute (TPM)
    - 1,500 requests per day

  Embeddings: handled locally via HuggingFace — no API quota.
"""
import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class QuotaExhaustedError(Exception):
    """Raised when the daily API quota has been fully exhausted."""
    pass


@dataclass
class RateLimitConfig:
    """Configuration for Gemini free-tier rate limits."""
    requests_per_minute: int = 30        # gemini-2.0-flash-lite free tier
    tokens_per_minute: int = 1_000_000
    requests_per_day: int = 1_500


class RateLimiter:
    """
    Proactive sliding-window rate limiter for Gemini free-tier compliance.
    
    Tracks request timestamps and token counts in a 60-second sliding window.
    Checks limits BEFORE each API call and waits if necessary, rather than
    relying solely on reactive 429 error handling.
    """

    def __init__(self, config: RateLimitConfig = None) -> None:
        self.config = config or RateLimitConfig()
        # Sliding window: deque of (timestamp, tokens_used) tuples
        self._minute_window: deque = deque()
        # Daily counter: list of timestamps for today's requests
        self._daily_timestamps: list = []
        self._daily_start: float = time.time()

    def _purge_old_entries(self) -> None:
        """Remove entries older than 60 seconds from the sliding window."""
        cutoff = time.time() - 60.0
        while self._minute_window and self._minute_window[0][0] < cutoff:
            self._minute_window.popleft()

    def _purge_daily_entries(self) -> None:
        """Remove entries older than 24 hours from the daily counter."""
        cutoff = time.time() - 86400.0
        self._daily_timestamps = [ts for ts in self._daily_timestamps if ts >= cutoff]

    @property
    def daily_requests_remaining(self) -> int:
        """Returns remaining daily request budget."""
        self._purge_daily_entries()
        used = len(self._daily_timestamps)
        return max(0, self.config.requests_per_day - used)

    async def check_and_wait(self, estimated_tokens: int) -> None:
        """
        Proactively check rate limits and wait if necessary before an API call.
        
        Args:
            estimated_tokens: Estimated token count for the upcoming request.
            
        Raises:
            QuotaExhaustedError: If the daily request limit has been reached.
        """
        # Check daily limit first
        self._purge_daily_entries()
        if len(self._daily_timestamps) >= self.config.requests_per_day:
            raise QuotaExhaustedError(
                f"Daily API quota ({self.config.requests_per_day} requests) has been reached. "
                "The service will resume tomorrow."
            )

        # Check per-minute limits with sliding window
        while True:
            self._purge_old_entries()
            current_rpm = len(self._minute_window)
            current_tpm = sum(tokens for _, tokens in self._minute_window)

            rpm_ok = current_rpm < self.config.requests_per_minute
            tpm_ok = (current_tpm + estimated_tokens) <= self.config.tokens_per_minute

            if rpm_ok and tpm_ok:
                break

            # Calculate wait time needed
            if self._minute_window:
                oldest_ts = self._minute_window[0][0]
                wait_time = max(0.1, (oldest_ts + 60.0) - time.time())
            else:
                wait_time = 1.0

            if not rpm_ok:
                logger.info(
                    "RPM limit reached (%d/%d). Waiting %.1fs before retrying.",
                    current_rpm, self.config.requests_per_minute, wait_time
                )
            else:
                logger.info(
                    "TPM limit would be exceeded (%d + %d > %d). Waiting %.1fs.",
                    current_tpm, estimated_tokens, self.config.tokens_per_minute, wait_time
                )

            await asyncio.sleep(wait_time)

    def record_request(self, tokens_used: int) -> None:
        """
        Record a completed API request for sliding-window tracking.
        
        Args:
            tokens_used: Actual token count used by the completed request.
        """
        now = time.time()
        self._minute_window.append((now, tokens_used))
        self._daily_timestamps.append(now)
        logger.debug(
            "Request recorded. RPM: %d/%d, Daily: %d/%d",
            len(self._minute_window),
            self.config.requests_per_minute,
            len(self._daily_timestamps),
            self.config.requests_per_day,
        )

    async def execute_with_backoff(self, coro_factory, max_retries: int = 5, initial_delay: float = 1.0):
        """
        Execute a coroutine with exponential backoff on rate limit errors.
        Respects the retry_delay hint from the API response when available.
        """
        import re as _re
        last_exception = None
        for attempt in range(max_retries + 1):
            try:
                return await coro_factory()
            except Exception as e:
                error_str = str(e).lower()
                is_rate_limit = (
                    "429" in str(e) or
                    "resource exhausted" in error_str or
                    "quota" in error_str or
                    "rate limit" in error_str
                )
                if not is_rate_limit or attempt == max_retries:
                    raise

                last_exception = e
                # Respect the retry_delay hint from the API if present
                match = _re.search(r'retry_delay\s*\{\s*seconds:\s*(\d+)', str(e))
                if match:
                    delay = int(match.group(1)) + 2  # add 2s buffer
                else:
                    delay = initial_delay * (2 ** attempt)

                logger.warning(
                    "Rate limit error on attempt %d/%d. Retrying in %.0fs. Error: %s",
                    attempt + 1, max_retries, delay, str(e)[:120]
                )
                await asyncio.sleep(delay)

        raise last_exception
