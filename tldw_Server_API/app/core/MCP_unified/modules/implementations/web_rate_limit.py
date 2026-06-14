"""Per-domain rate limiting for the read-only web MCP tools.

A small in-memory sliding-window limiter keyed by destination host. It throttles
how often a single ``web.fetch`` (and, transitively, ``web.research``) instance
will hit one domain, for politeness and basic abuse protection. The actual
SSRF/egress safety net lives in the outbound policy; this is a separate, softer
control.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict, deque
from collections.abc import Callable

# Generous defaults: enough headroom for normal research/redirect flows, low
# enough to curb a tool hammering a single host.
_DEFAULT_MAX_REQUESTS = 60
_DEFAULT_WINDOW_SECONDS = 60.0
# Cap how many distinct domains we track so a long-lived limiter cannot grow
# without bound; the least-recently-touched domain is evicted when exceeded.
_MAX_TRACKED_DOMAINS = 1024


class DomainRateLimiter:
    """Thread-safe sliding-window request limiter keyed by domain.

    ``max_requests <= 0`` (or ``None``) disables limiting entirely (always
    allows), which is the explicit way to opt out.
    """

    def __init__(
        self,
        *,
        max_requests: int | None = _DEFAULT_MAX_REQUESTS,
        window_seconds: float = _DEFAULT_WINDOW_SECONDS,
        clock: Callable[[], float] | None = None,
        max_tracked_domains: int = _MAX_TRACKED_DOMAINS,
    ) -> None:
        if max_requests is not None and max_requests > 0:
            self._max_requests = max_requests
        else:
            self._max_requests = 0
        self._window_seconds = max(0.0, window_seconds)
        self._clock = clock or time.monotonic
        self._max_tracked_domains = max(1, max_tracked_domains)
        self._lock = threading.Lock()
        # OrderedDict gives O(1) LRU eviction: accessed domains move to the end,
        # and the least-recently-used (front) entry is dropped when over the cap.
        self._hits: OrderedDict[str, deque[float]] = OrderedDict()

    @property
    def enabled(self) -> bool:
        return self._max_requests > 0

    def try_acquire(self, domain: str) -> bool:
        """Record a request to ``domain``; return ``False`` if it exceeds the limit."""
        if not self.enabled:
            return True
        key = (domain or "unknown").lower()
        now = self._clock()
        cutoff = now - self._window_seconds
        with self._lock:
            bucket = self._hits.get(key)
            if bucket is None:
                bucket = deque()
                self._hits[key] = bucket
                self._evict_if_needed()
            else:
                # Mark this domain as most-recently-used for LRU eviction.
                self._hits.move_to_end(key)
            while bucket and bucket[0] <= cutoff:
                bucket.popleft()
            if len(bucket) >= self._max_requests:
                return False
            bucket.append(now)
            return True

    def _evict_if_needed(self) -> None:
        """Drop the least-recently-used domain when over the cap (caller holds lock)."""
        if len(self._hits) > self._max_tracked_domains:
            self._hits.popitem(last=False)


__all__ = ["DomainRateLimiter"]
