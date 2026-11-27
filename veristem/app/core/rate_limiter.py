import time
from collections import defaultdict, deque
from threading import Lock

from app.core.exceptions import RateLimitExceededError


class SlidingWindowRateLimiter:
    """Thread-safe in-process sliding window rate limiter keyed by client IP or API key."""

    def __init__(self, max_requests: int = 60, window_seconds: int = 60) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._buckets: dict[str, deque[float]] = defaultdict(deque)
        self._lock = Lock()

    def check(self, key: str) -> None:
        now = time.monotonic()
        cutoff = now - self.window_seconds
        with self._lock:
            bucket = self._buckets[key]
            while bucket and bucket[0] < cutoff:
                bucket.popleft()
            if len(bucket) >= self.max_requests:
                raise RateLimitExceededError(
                    f"Rate limit of {self.max_requests} requests per {self.window_seconds}s exceeded"
                )
            bucket.append(now)
