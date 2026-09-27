"""Thread-safe TTL cache for graph query results."""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from collections import OrderedDict
from typing import Any


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "")
    try:
        return int(raw)
    except (ValueError, TypeError):
        return default


class GraphCache:
    """Simple in-memory TTL cache keyed on a hash of query parameters.

    Config via environment variables:
      - ``NOTES_GRAPH_CACHE_TTL`` – seconds before expiry (default 20)
      - ``NOTES_GRAPH_CACHE_MAX_KEYS`` – max cached entries (default 1000)
    """

    def __init__(
        self,
        ttl_seconds: int | None = None,
        max_keys: int | None = None,
    ) -> None:
        self._ttl = ttl_seconds if ttl_seconds is not None else _env_int("NOTES_GRAPH_CACHE_TTL", 20)
        self._max_keys = max_keys if max_keys is not None else _env_int("NOTES_GRAPH_CACHE_MAX_KEYS", 1000)
        self._store: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    @staticmethod
    def make_cache_key(user_id: str, query_params: dict) -> str:
        """SHA-256[:32] of *user_id* + deterministic JSON of *query_params*."""
        raw = user_id + "|" + json.dumps(query_params, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()[:32]

    @staticmethod
    def make_revision_key(
        *,
        user_id: str,
        dataset_id: str,
        graph_revision: int,
        parser_version: int,
        query_params: dict,
    ) -> str:
        """Build a cache key that cannot survive a graph-visible mutation."""

        return GraphCache.make_cache_key(
            user_id,
            {
                "dataset_id": dataset_id,
                "graph_revision": graph_revision,
                "parser_version": parser_version,
                "query": query_params,
            },
        )

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def get(self, key: str) -> Any | None:
        """Return cached value or ``None`` on miss / expiry."""
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self._misses += 1
                return None
            ts, value = entry
            if time.monotonic() - ts > self._ttl:
                del self._store[key]
                self._misses += 1
                return None
            # Move to end (most-recently-used)
            self._store.move_to_end(key)
            self._hits += 1
            return value

    def put(self, key: str, value: Any) -> None:
        """Insert or replace a cache entry."""
        with self._lock:
            now = time.monotonic()
            if key in self._store:
                self._store.move_to_end(key)
            self._store[key] = (now, value)
            # Evict oldest if over capacity
            while len(self._store) > self._max_keys:
                self._store.popitem(last=False)

    def stats(self) -> dict:
        """Return cache statistics snapshot."""
        with self._lock:
            return {
                "size": len(self._store),
                "max_keys": self._max_keys,
                "ttl_seconds": self._ttl,
                "hits": self._hits,
                "misses": self._misses,
            }
