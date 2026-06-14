"""In-memory TTL + LRU response cache for the ``web.fetch`` MCP tool.

Caches successful fetch results keyed by the request inputs (url, format,
max_bytes) so repeated fetches of the same resource — common inside a
``web.research`` bundle or across closely-spaced calls — skip the network. It is
opt-in: ``WebFetchModule`` only uses it when a cache instance is supplied.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from typing import Any

_DEFAULT_TTL_SECONDS = 300.0
_DEFAULT_MAX_ENTRIES = 256


def make_cache_key(url: str, fmt: str, max_bytes: int) -> tuple[str, str, int]:
    """Build the cache key from the request inputs that determine the output."""
    return (url, fmt, int(max_bytes))


class ResponseCache:
    """Thread-safe in-memory TTL cache with LRU eviction.

    ``ttl_seconds <= 0`` or ``max_entries <= 0`` disables caching (``enabled`` is
    ``False``); ``get`` always misses and ``put`` is a no-op.
    """

    def __init__(
        self,
        *,
        ttl_seconds: float = _DEFAULT_TTL_SECONDS,
        max_entries: int = _DEFAULT_MAX_ENTRIES,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._ttl_seconds = ttl_seconds
        self._max_entries = max_entries
        self._clock = clock or time.monotonic
        self._lock = threading.Lock()
        # key -> (expires_at, value); ordered for LRU eviction.
        self._entries: OrderedDict[Any, tuple[float, Any]] = OrderedDict()

    @property
    def enabled(self) -> bool:
        return self._ttl_seconds > 0 and self._max_entries > 0

    def get(self, key: Any) -> Any | None:
        """Return the cached value for ``key`` or ``None`` if missing/expired."""
        if not self.enabled:
            return None
        now = self._clock()
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            expires_at, value = entry
            if expires_at <= now:
                del self._entries[key]
                return None
            self._entries.move_to_end(key)
            return value

    def put(self, key: Any, value: Any) -> None:
        """Store ``value`` for ``key`` with the configured TTL, evicting LRU as needed."""
        if not self.enabled:
            return
        expires_at = self._clock() + self._ttl_seconds
        with self._lock:
            self._entries[key] = (expires_at, value)
            self._entries.move_to_end(key)
            while len(self._entries) > self._max_entries:
                self._entries.popitem(last=False)


__all__ = ["ResponseCache", "make_cache_key"]
