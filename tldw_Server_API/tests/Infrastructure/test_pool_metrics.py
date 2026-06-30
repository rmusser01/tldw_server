"""Tests for database connection pool metrics exporter."""
from __future__ import annotations

import queue

import pytest

from tldw_Server_API.app.core.Infrastructure import pool_metrics as pool_metrics_module
from tldw_Server_API.app.core.Infrastructure.pool_metrics import collect_pool_metrics


# ---------------------------------------------------------------------------
# Fake pool classes for testing
# ---------------------------------------------------------------------------

class FakeAsyncpgPool:
    """Mimics the asyncpg Pool interface."""

    def get_size(self) -> int:
        return 10

    def get_idle_size(self) -> int:
        return 7

    def get_min_size(self) -> int:
        return 2

    def get_max_size(self) -> int:
        return 20


class FakeQueuePool:
    """Mimics a generic pool backed by a Queue."""

    def __init__(self, size: int = 5) -> None:
        self._pool: queue.Queue = queue.Queue()
        for _ in range(size):
            self._pool.put(object())


class FakeUnknownPool:
    """A pool type that exposes no known attributes."""
    pass


class FakeBrokenAsyncpgPool:
    """Mimics a closed pool whose metric accessors raise."""

    def get_size(self) -> int:
        raise RuntimeError("pool is closed")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCollectPoolMetrics:
    def test_asyncpg_pool(self) -> None:
        pool = FakeAsyncpgPool()
        metrics = collect_pool_metrics(pool)
        assert metrics["available"] is True
        assert metrics["pool_type"] == "FakeAsyncpgPool"
        assert metrics["size"] == 10
        assert metrics["free_size"] == 7
        assert metrics["min_size"] == 2
        assert metrics["max_size"] == 20

    def test_queue_pool(self) -> None:
        pool = FakeQueuePool(size=3)
        metrics = collect_pool_metrics(pool)
        assert metrics["available"] is True
        assert metrics["pool_type"] == "FakeQueuePool"
        assert metrics["size"] == 3

    def test_unknown_pool(self) -> None:
        pool = FakeUnknownPool()
        metrics = collect_pool_metrics(pool)
        assert metrics["available"] is False
        assert metrics["pool_type"] == "FakeUnknownPool"

    def test_none_pool(self) -> None:
        """Even None should not raise -- returns unavailable."""
        metrics = collect_pool_metrics(None)
        assert metrics["available"] is False

    def test_broken_pool_accessor_returns_unavailable(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        captured: list[tuple[tuple[object, ...], dict[str, object]]] = []
        monkeypatch.setattr(
            pool_metrics_module.logger,
            "debug",
            lambda *args, **kwargs: captured.append((args, kwargs)),
        )

        metrics = collect_pool_metrics(FakeBrokenAsyncpgPool())

        assert metrics["available"] is False
        assert metrics["pool_type"] == "FakeBrokenAsyncpgPool"
        assert metrics["error"] == "RuntimeError"
        assert captured
        assert "FakeBrokenAsyncpgPool" in repr(captured)
        assert "RuntimeError" in repr(captured)
