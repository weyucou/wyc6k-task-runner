import asyncio
import logging
import time
from collections import deque
from threading import Lock

logger = logging.getLogger(__name__)

SECONDS_PER_MINUTE = 60


class RateLimiter:
    """Thread-safe rate limiter using sliding window algorithm.

    Tracks API call timestamps and enforces requests-per-minute limits
    by introducing wait times when necessary.
    """

    def __init__(self, rpm: int) -> None:
        """Initialize rate limiter.

        Args:
            rpm: Maximum requests per minute. 0 means unlimited.
        """
        self.rpm = rpm
        self.window_seconds = SECONDS_PER_MINUTE
        self._timestamps: deque[float] = deque()
        self._lock = Lock()

    def _cleanup_old_timestamps(self, now: float) -> None:
        """Remove timestamps outside the sliding window."""
        cutoff = now - self.window_seconds
        while self._timestamps and self._timestamps[0] < cutoff:
            self._timestamps.popleft()

    def get_wait_time(self) -> float:
        """Calculate wait time before next request is allowed.

        Returns:
            Seconds to wait. 0 if request can proceed immediately.
        """
        if self.rpm <= 0:
            return 0.0

        now = time.monotonic()

        with self._lock:
            self._cleanup_old_timestamps(now)

            if len(self._timestamps) < self.rpm:
                return 0.0

            # Calculate when the oldest request will expire
            oldest = self._timestamps[0]
            wait_time = (oldest + self.window_seconds) - now
            return max(0.0, wait_time)

    def acquire(self) -> float:
        """Acquire permission to make a request, blocking if necessary.

        Returns:
            Actual wait time in seconds (0 if no wait was needed).
        """
        wait_time = self.get_wait_time()

        if wait_time > 0:
            logger.debug(f"Rate limit: waiting {wait_time:.2f}s before request")
            time.sleep(wait_time)

        now = time.monotonic()
        with self._lock:
            self._cleanup_old_timestamps(now)
            self._timestamps.append(now)

        return wait_time

    async def acquire_async(self) -> float:
        """Async version of acquire.

        Returns:
            Actual wait time in seconds (0 if no wait was needed).
        """
        wait_time = self.get_wait_time()

        if wait_time > 0:
            logger.debug(f"Rate limit: waiting {wait_time:.2f}s before request")
            await asyncio.sleep(wait_time)

        now = time.monotonic()
        with self._lock:
            self._cleanup_old_timestamps(now)
            self._timestamps.append(now)

        return wait_time

    def reset(self) -> None:
        """Clear all tracked timestamps."""
        with self._lock:
            self._timestamps.clear()

    @property
    def current_count(self) -> int:
        """Get current request count in the sliding window."""
        now = time.monotonic()
        with self._lock:
            self._cleanup_old_timestamps(now)
            return len(self._timestamps)


class RateLimiterRegistry:
    """Registry to manage rate limiters per agent."""

    def __init__(self) -> None:
        self._limiters: dict[str, RateLimiter] = {}
        self._lock = Lock()

    def get_or_create(self, agent_id: str, rpm: int) -> RateLimiter:
        """Get existing or create new rate limiter for an agent.

        Args:
            agent_id: The agent's name or identifier.
            rpm: Requests per minute limit.

        Returns:
            RateLimiter instance for the agent.
        """
        with self._lock:
            limiter = self._limiters.get(agent_id)

            if limiter is None or limiter.rpm != rpm:
                limiter = RateLimiter(rpm)
                self._limiters[agent_id] = limiter

            return limiter

    def remove(self, agent_id: str) -> None:
        """Remove rate limiter for an agent."""
        with self._lock:
            self._limiters.pop(agent_id, None)

    def clear(self) -> None:
        """Clear all rate limiters."""
        with self._lock:
            self._limiters.clear()


# Global registry instance
rate_limiter_registry = RateLimiterRegistry()
