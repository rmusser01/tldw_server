from __future__ import annotations

import types
import pytest

from tldw_Server_API.app.core.Infrastructure import redis_factory as rf
from tldw_Server_API.app.core.Metrics.metrics_manager import get_metrics_registry


def _metric_entries(registry, name):
    seq = registry.values.get(name)
    if not seq:
        return []
    return list(seq)


@pytest.mark.asyncio
async def test_async_client_success_records_metrics(monkeypatch):
    class FakeAsyncRedis:
        async def ping(self):
            return True

        async def close(self):
            return None

    fake_client = FakeAsyncRedis()
    monkeypatch.setattr(
        rf,
        "aioredis",
        types.SimpleNamespace(from_url=lambda *args, **kwargs: fake_client),
        raising=False,
    )

    registry = get_metrics_registry()
    before_attempts = len(_metric_entries(registry, "infra_redis_connection_attempts_total"))
    before_duration = len(
        _metric_entries(registry, "infra_redis_connection_duration_seconds")
    )

    client = await rf.create_async_redis_client(context="tests-async-success", fallback_to_fake=False)

    assert client is fake_client

    attempts = _metric_entries(registry, "infra_redis_connection_attempts_total")
    durations = _metric_entries(registry, "infra_redis_connection_duration_seconds")
    assert len(attempts) == before_attempts + 1
    assert len(durations) == before_duration + 1
    assert attempts[-1].labels == {
        "mode": "async",
        "context": "tests-async-success",
        "outcome": "real",
    }
    assert durations[-1].labels == {
        "mode": "async",
        "context": "tests-async-success",
        "outcome": "real",
    }


@pytest.mark.asyncio
async def test_async_client_fallback_records_metrics(monkeypatch):
    class FailingAsyncRedis:
        async def ping(self):
            raise ConnectionError("boom")

        async def close(self):
            return None

    failing_client = FailingAsyncRedis()

    async def fake_from_url(*args, **kwargs):
        return failing_client

    monkeypatch.setattr(
        rf,
        "aioredis",
        types.SimpleNamespace(from_url=fake_from_url),
        raising=False,
    )

    registry = get_metrics_registry()
    before_attempts = len(_metric_entries(registry, "infra_redis_connection_attempts_total"))
    before_duration = len(
        _metric_entries(registry, "infra_redis_connection_duration_seconds")
    )
    before_fallbacks = len(_metric_entries(registry, "infra_redis_fallback_total"))

    client = await rf.create_async_redis_client(
        context="tests-async-fallback",
        fallback_to_fake=True,
    )

    assert isinstance(client, rf.InMemoryAsyncRedis)

    attempts = _metric_entries(registry, "infra_redis_connection_attempts_total")
    durations = _metric_entries(registry, "infra_redis_connection_duration_seconds")
    fallbacks = _metric_entries(registry, "infra_redis_fallback_total")
    assert len(attempts) == before_attempts + 1
    assert len(durations) == before_duration + 1
    assert len(fallbacks) == before_fallbacks + 1
    assert attempts[-1].labels == {
        "mode": "async",
        "context": "tests-async-fallback",
        "outcome": "stub",
    }
    assert fallbacks[-1].labels["mode"] == "async"
    assert fallbacks[-1].labels["context"] == "tests-async-fallback"
    assert fallbacks[-1].labels["reason"] == "ConnectionError"


def test_sync_client_error_records_metrics(monkeypatch):
    class FailingSyncRedis:
        def ping(self):
            raise ConnectionError("nope")

        def close(self):
            return None

    def fake_from_url(*args, **kwargs):
        return FailingSyncRedis()

    monkeypatch.setattr(
        rf,
        "redis",
        types.SimpleNamespace(from_url=fake_from_url),
        raising=False,
    )

    registry = get_metrics_registry()
    before_attempts = len(_metric_entries(registry, "infra_redis_connection_attempts_total"))
    before_duration = len(
        _metric_entries(registry, "infra_redis_connection_duration_seconds")
    )
    before_errors = len(_metric_entries(registry, "infra_redis_connection_errors_total"))

    with pytest.raises(ConnectionError):
        rf.create_sync_redis_client(context="tests-sync-error", fallback_to_fake=False)

    attempts = _metric_entries(registry, "infra_redis_connection_attempts_total")
    durations = _metric_entries(registry, "infra_redis_connection_duration_seconds")
    errors = _metric_entries(registry, "infra_redis_connection_errors_total")
    assert len(attempts) == before_attempts + 1
    assert len(durations) == before_duration + 1
    assert len(errors) == before_errors + 1
    assert attempts[-1].labels == {
        "mode": "sync",
        "context": "tests-sync-error",
        "outcome": "error",
    }
    assert errors[-1].labels == {
        "mode": "sync",
        "context": "tests-sync-error",
        "error": "ConnectionError",
    }
