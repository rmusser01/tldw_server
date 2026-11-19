from __future__ import annotations

import threading
import time
from collections import deque


class TTLReceiptStore:
    """In-memory TTL dedupe store for webhook/event receipts."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._seen: dict[str, float] = {}

    def clear(self) -> None:
        with self._lock:
            self._seen.clear()

    def seen_or_store(self, key: str, ttl_seconds: int, now: float | None = None) -> bool:
        ts = now if now is not None else time.time()
        expiry = ts + max(1, ttl_seconds)
        with self._lock:
            stale_keys = [k for k, expires_at in self._seen.items() if expires_at <= ts]
            for stale_key in stale_keys:
                self._seen.pop(stale_key, None)
            existing = self._seen.get(key)
            if existing and existing > ts:
                return True
            self._seen[key] = expiry
            return False


class SlidingWindowLimiter:
    """Thread-safe per-key sliding-window limiter."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._windows: dict[str, deque[float]] = {}

    def clear(self) -> None:
        with self._lock:
            self._windows.clear()

    def allow(self, key: str, limit_per_minute: int, now: float | None = None) -> tuple[bool, int]:
        ts = now if now is not None else time.time()
        limit = max(1, int(limit_per_minute))
        cutoff = ts - 60.0
        with self._lock:
            window = self._windows.setdefault(key, deque())
            while window and window[0] <= cutoff:
                window.popleft()
            if len(window) >= limit:
                retry_after = int(max(1, 60 - (ts - window[0])))
                return False, retry_after
            window.append(ts)
            return True, 0
