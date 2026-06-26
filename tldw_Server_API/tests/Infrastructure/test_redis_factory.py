from __future__ import annotations

from pathlib import Path
import time
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.Infrastructure import redis_factory as rf


def test_sync_stub_core_commands():
    client = rf.InMemorySyncRedis()

    client.set("k1", "v1")
    assert client.get("k1") == "v1"

    client.setex("k2", 5, "v2")
    ttl = client.ttl("k2")
    assert ttl >= 0

    assert client.sadd("set:1", "a") == 1
    assert client.sadd("set:1", "a") == 0
    assert client.smembers("set:1") == {"a"}
    assert client.srem("set:1", "a") == 1

    client.zadd("z:1", {"alpha": 1.0, "beta": 2.0})
    assert client.zrange("z:1", 0, -1) == ["alpha", "beta"]
    assert client.zscore("z:1", "alpha") == 1.0
    assert client.zincrby("z:1", 2.5, "alpha") == 3.5

    assert client.hset("h:1", {"field": "1"}) == 1
    assert client.hget("h:1", "field") == "1"
    assert client.hincrby("h:1", "field", 2) == 3
    assert client.hgetall("h:1")["field"] == "3"

    cursor, keys = client.scan(0, match="k*", count=10)
    assert cursor == 0
    assert "k1" in keys
    assert set(client.keys("k*")) >= {"k1", "k2"}

    info = client.info("memory")
    assert "used_memory" in info
    assert client.dbsize() >= 4

    assert client.delete("k1", "k2") == 2


def test_redis_factory_readme_documents_fail_closed_defaults():
    readme_path = Path(rf.__file__).with_name("README.md")
    readme = readme_path.read_text(encoding="utf-8")

    assert "fallback_to_fake=False" in readme
    assert "fallback_to_fake=True" in readme
    assert "create_async_redis_client(context=\"demo\", fallback_to_fake=True)" in readme
    assert "create_sync_redis_client(context=\"demo-sync\", fallback_to_fake=True)" in readme


@pytest.mark.asyncio
async def test_async_stub_streams_and_scripts():
    client = rf.InMemoryAsyncRedis()

    await client.xadd("stream:1", {"field": "a"})
    await client.xadd("stream:1", {"field": "b"})
    assert await client.xlen("stream:1") == 2

    await client.xgroup_create("stream:1", "group:1")
    first_batch = await client.xreadgroup(
        "group:1",
        "consumer:1",
        {"stream:1": ">"},
        count=1,
    )
    assert first_batch
    assert first_batch[0][0] == "stream:1"
    assert len(first_batch[0][1]) == 1

    second_batch = await client.xreadgroup(
        "group:1",
        "consumer:1",
        {"stream:1": ">"},
        count=1,
    )
    assert second_batch
    assert len(second_batch[0][1]) == 1

    script = "redis.call('ZRANGE', KEYS[1], 0, -1); redis.call('ZREMRANGEBYSCORE', KEYS[1], 0, 0)"
    sha = await client.script_load(script)
    result = await client.evalsha(sha, 1, "rate:key", 1, 60, 1000.0)
    assert result == [1, 0]

    eval_result = await client.eval(script, 1, "rate:key", 1, 60, time.time())
    assert isinstance(eval_result, list)


@pytest.mark.asyncio
async def test_async_factory_falls_back_when_redis_package_missing(monkeypatch):
    monkeypatch.setattr(rf, "aioredis", None)
    monkeypatch.setattr(rf, "_import_error", ImportError("redis missing"))

    client = await rf.create_async_redis_client(
        fallback_to_fake=True,
        context="missing_package",
    )

    assert await client.ping() is True
    await client.set("missing:async", "ok")
    assert await client.get("missing:async") == "ok"


@pytest.mark.asyncio
async def test_async_factory_raises_when_redis_package_missing_and_fallback_disabled(monkeypatch):
    monkeypatch.setattr(rf, "aioredis", None)
    monkeypatch.setattr(rf, "_import_error", ImportError("redis missing"))

    with pytest.raises(RuntimeError, match="redis\\[asyncio\\] is required"):
        await rf.create_async_redis_client(
            fallback_to_fake=False,
            context="missing_package",
        )


@pytest.mark.asyncio
async def test_async_factory_raises_when_redis_package_missing_by_default(monkeypatch):
    monkeypatch.setattr(rf, "aioredis", None)
    monkeypatch.setattr(rf, "_import_error", ImportError("redis missing"))

    with pytest.raises(RuntimeError, match="redis\\[asyncio\\] is required"):
        await rf.create_async_redis_client(context="missing_package")


def test_sync_factory_falls_back_when_redis_package_missing(monkeypatch):
    monkeypatch.setattr(rf, "redis", None)
    monkeypatch.setattr(rf, "_import_error", ImportError("redis missing"))

    client = rf.create_sync_redis_client(
        fallback_to_fake=True,
        context="missing_package",
    )

    assert client.ping() is True
    client.set("missing:sync", "ok")
    assert client.get("missing:sync") == "ok"


def test_sync_factory_raises_when_redis_package_missing_by_default(monkeypatch):
    monkeypatch.setattr(rf, "redis", None)
    monkeypatch.setattr(rf, "_import_error", ImportError("redis missing"))

    with pytest.raises(RuntimeError, match="redis client is required"):
        rf.create_sync_redis_client(context="missing_package")


@pytest.mark.asyncio
async def test_async_factory_redacts_redis_url_credentials_in_warning(monkeypatch):
    class _FailingAsyncRedis:
        async def ping(self):
            raise OSError("connection refused")

        async def close(self):
            return None

    captured: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def _capture_warning(*args, **kwargs):
        captured.append((args, kwargs))

    monkeypatch.setattr(
        rf,
        "aioredis",
        SimpleNamespace(from_url=lambda *args, **kwargs: _FailingAsyncRedis()),
    )
    monkeypatch.setattr(rf.logger, "warning", _capture_warning)

    client = await rf.create_async_redis_client(
        preferred_url="redis://:super-secret@example.com:6379/0",
        fallback_to_fake=True,
        context="redaction",
    )

    assert await client.ping() is True
    warning_payload = repr(captured)
    assert "super-secret" not in warning_payload
    assert "redis://***:***@example.com:6379/0" in warning_payload


@pytest.mark.asyncio
async def test_async_factory_redacts_username_without_password_in_warning(monkeypatch):
    class _FailingAsyncRedis:
        async def ping(self):
            raise OSError("connection refused")

        async def close(self):
            return None

    captured: list[tuple[tuple[object, ...], dict[str, object]]] = []

    monkeypatch.setattr(
        rf,
        "aioredis",
        SimpleNamespace(from_url=lambda *args, **kwargs: _FailingAsyncRedis()),
    )
    monkeypatch.setattr(
        rf.logger,
        "warning",
        lambda *args, **kwargs: captured.append((args, kwargs)),
    )

    client = await rf.create_async_redis_client(
        preferred_url="redis://token-user@example.com:6379/0",
        fallback_to_fake=True,
        context="redaction",
    )

    assert await client.ping() is True
    warning_payload = repr(captured)
    assert "token-user" not in warning_payload
    assert "redis://***@example.com:6379/0" in warning_payload


@pytest.mark.asyncio
async def test_async_factory_handles_malformed_redis_url_in_warning(monkeypatch):
    class _FailingAsyncRedis:
        async def ping(self):
            raise OSError("connection refused")

        async def close(self):
            return None

    captured: list[tuple[tuple[object, ...], dict[str, object]]] = []

    monkeypatch.setattr(
        rf,
        "aioredis",
        SimpleNamespace(from_url=lambda *args, **kwargs: _FailingAsyncRedis()),
    )
    monkeypatch.setattr(
        rf.logger,
        "warning",
        lambda *args, **kwargs: captured.append((args, kwargs)),
    )

    client = await rf.create_async_redis_client(
        preferred_url="redis://:secret@example.com:not-a-port/0",
        fallback_to_fake=True,
        context="redaction",
    )

    assert await client.ping() is True
    warning_payload = repr(captured)
    assert "secret" not in warning_payload
    assert "<invalid-redis-url>" in warning_payload
