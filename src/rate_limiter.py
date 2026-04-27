"""
Proactive rate limiter for Google Gemini free-tier API compliance.
Enforces limits BEFORE each API call using a sliding-window approach.

Free-tier limits for gemini-2.5-flash (as of 2025):
  - 15 requests per minute (RPM)
  - 1,000,000 tokens per minute (TPM)
  - 500 requests per day (RPD)

The limiter operates at 80% of each limit as a safety margin to avoid
hitting the API boundary and receiving 429 errors.
"""
import asyncio
import logging
import random
import re
import time
from collections import deque
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Safety margin — operate at this fraction of the declared limit
_SAFETY_MARGIN = 0.80


class QuotaExhaustedError(Exception):
    """Raised when the daily API quota has been fully exhausted."""
    pass


@dataclass
class RateLimitConfig:
    """Configuration for Gemini free-tier rate limits."""
    requests_per_minute: int = 15          # gemini-2.5-flash free tier
    tokens_per_minute: int = 1_000_000    # gemini-2.5-flash free tier
    requests_per_day: int = 500            # gemini-2.5-flash free tier

    @property
    def effective_rpm(self) -> int:
        """RPM limit with safety margin applied."""
        return max(1, int(self.requests_per_minute * _SAFETY_MARGIN))

    @property
    def effective_tpm(self) -> int:
        """TPM limit with safety margin applied."""
        return int(self.tokens_per_minute * _SAFETY_MARGIN)


class RateLimiter:
    """
    Proactive sliding-window rate limiter for Gemini free-tier compliance.

    Tracks request timestamps and token counts in a 60-second sliding window.
    Checks limits BEFORE each API call and waits if necessary, rather than
    relying solely on reactive 429 error handling.

    Operates at 80% of declared limits to provide a safety buffer.
    """

    def __init__(self, config: RateLimitConfig = None) -> None:
        self.config = config or RateLimitConfig()
        # Sliding window: deque of (timestamp, tokens_used) tuples
        self._minute_window: deque = deque()
        # Daily counter: list of timestamps for today's requests
        self._daily_timestamps: list = []

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
                "The quota resets every 24 hours."
            )

        # Check per-minute limits with sliding window
        max_wait_iterations = 120  # safety cap: never wait more than ~2 minutes total
        iterations = 0
        while iterations < max_wait_iterations:
            self._purge_old_entries()
            current_rpm = len(self._minute_window)
            current_tpm = sum(tokens for _, tokens in self._minute_window)

            rpm_ok = current_rpm < self.config.effective_rpm
            tpm_ok = (current_tpm + estimated_tokens) <= self.config.effective_tpm

            if rpm_ok and tpm_ok:
                break

            # Calculate precise wait time until the oldest entry expires
            if self._minute_window:
                oldest_ts = self._minute_window[0][0]
                wait_time = max(0.5, (oldest_ts + 61.0) - time.time())  # +1s buffer
            else:
                wait_time = 1.0

            # Cap individual wait at 65 seconds (one full window + buffer)
            wait_time = min(wait_time, 65.0)

            if not rpm_ok:
                logger.info(
                    "RPM limit reached (%d/%d effective). Waiting %.1fs.",
                    current_rpm, self.config.effective_rpm, wait_time
                )
            else:
                logger.info(
                    "TPM limit would be exceeded (%d + %d > %d effective). Waiting %.1fs.",
                    current_tpm, estimated_tokens, self.config.effective_tpm, wait_time
                )

            await asyncio.sleep(wait_time)
            iterations += 1

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
            "Request recorded. RPM: %d/%d (effective %d), Daily: %d/%d",
            len(self._minute_window),
            self.config.requests_per_minute,
            self.config.effective_rpm,
            len(self._daily_timestamps),
            self.config.requests_per_day,
        )

    @staticmethod
    def _parse_retry_delay(error_str: str) -> int | None:
        """
        Extract the retry delay in seconds from a 429 error message.

        Handles multiple formats returned by the Gemini API:
          - retry_delay { seconds: 48 }
          - retryDelay: "48s"
          - Retry-After: 48
          - retry after 48 seconds
        """
        patterns = [
            r'retry_delay\s*\{\s*seconds:\s*(\d+)',   # proto format
            r'retryDelay["\s:]+(\d+)',                  # JSON camelCase
            r'retry.after[:\s]+(\d+)',                  # HTTP header style
            r'retry\s+after\s+(\d+)',                   # plain English
            r'"retryDelay"\s*:\s*"(\d+)s"',            # JSON string with unit
        ]
        for pattern in patterns:
            match = re.search(pattern, error_str, re.IGNORECASE)
            if match:
                return int(match.group(1))
        return None

    async def execute_with_backoff(
        self,
        coro_factory,
        max_retries: int = 5,
        initial_delay: float = 2.0,
    ):
        """
        Execute a coroutine with exponential backoff + jitter on rate limit errors.

        Strategy:
        - If the API provides a retry-delay hint, use it (+ 3s buffer)
        - Otherwise use exponential backoff: initial_delay * 2^attempt + jitter
        - Jitter is ±20% of the computed delay to avoid thundering herd
        - Minimum delay between retries: 5 seconds
        - Maximum delay between retries: 120 seconds

        Args:
            coro_factory: Callable that returns a coroutine to execute.
            max_retries: Maximum number of retry attempts (default 5).
            initial_delay: Base delay in seconds for exponential backoff (default 2.0).

        Returns:
            The result of the coroutine on success.

        Raises:
            The last exception if all retries are exhausted.
            QuotaExhaustedError if the daily quota is detected as exhausted.
        """
        last_exception = None

        for attempt in range(max_retries + 1):
            try:
                return await coro_factory()

            except Exception as e:
                error_str = str(e)
                error_lower = error_str.lower()

                is_rate_limit = (
                    "429" in error_str
                    or "resource exhausted" in error_lower
                    or "rate limit" in error_lower
                    or "quota exceeded" in error_lower
                    or "too many requests" in error_lower
                )

                # Distinguish daily quota exhaustion from per-minute rate limit
                is_daily_exhausted = (
                    "daily" in error_lower
                    or "per day" in error_lower
                    or ("quota" in error_lower and "minute" not in error_lower)
                )

                if is_daily_exhausted and is_rate_limit:
                    raise QuotaExhaustedError(
                        "Daily API quota exhausted. The quota resets every 24 hours."
                    ) from e

                if not is_rate_limit or attempt == max_retries:
                    raise

                last_exception = e

                # Try to get the API-suggested retry delay
                api_delay = self._parse_retry_delay(error_str)

                if api_delay is not None:
                    # Use API hint + 3s buffer
                    base_delay = api_delay + 3
                    logger.info("API retry hint: %ds. Using %ds.", api_delay, base_delay)
                else:
                    # Exponential backoff: 2, 4, 8, 16, 32 seconds
                    base_delay = min(120.0, initial_delay * (2 ** attempt))

                # Add ±20% jitter to avoid thundering herd
                jitter = base_delay * 0.2 * (random.random() * 2 - 1)
                delay = max(5.0, base_delay + jitter)

                logger.warning(
                    "Rate limit on attempt %d/%d. Retrying in %.0fs. Error: %s",
                    attempt + 1, max_retries, delay, error_str[:150]
                )
                await asyncio.sleep(delay)

        raise last_exception
