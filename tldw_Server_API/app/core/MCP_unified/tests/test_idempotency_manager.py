"""Fault-injection tests for bounded MCP idempotency ownership."""

from __future__ import annotations

import asyncio
import importlib
import inspect
import re
from collections.abc import Callable
from dataclasses import FrozenInstanceError
from typing import Any

import pytest
from loguru import logger

from tldw_Server_API.app.core.MCP_unified.execution_outcomes import (
    ExpectedToolFailure,
    ExpectedToolFailureReason,
)
from tldw_Server_API.app.core.MCP_unified.protocol import MCPProtocol, RequestContext
from tldw_Server_API.app.core.MCP_unified.protocol_types import (
    AuthenticatedExecutionScope,
    InvalidParamsException,
)
from tldw_Server_API.app.core.MCP_unified.tool_execution.canonical import (
    canonical_json_bytes,
)
from tldw_Server_API.app.core.MCP_unified.tool_execution.idempotency import (
    IdempotencyManager,
    RedisError,
)
from tldw_Server_API.app.core.MCP_unified.tool_execution.models import (
    IdempotencyExecutionPolicy,
)
from tldw_Server_API.app.core.MCP_unified.tool_execution.runtime import (
    ToolExecutionRuntime,
)


def _policy(*, max_result_bytes: int = 4_096) -> IdempotencyExecutionPolicy:
    return IdempotencyExecutionPolicy(
        inject_argument=False,
        ttl_seconds=30,
        contention_wait_seconds=1,
        finalize_seconds=1,
        lock_ttl_seconds=7,
        max_entries=16,
        max_result_bytes=max_result_bytes,
    )


def _local_manager(
    *,
    on_degraded: Callable[[str, str], None] | None = None,
) -> IdempotencyManager:
    kwargs: dict[str, Any] = {}
    if on_degraded is not None:
        kwargs["on_degraded"] = on_degraded
    manager = IdempotencyManager(**kwargs)
    manager._redis_attempted = True
    manager._redis_ready = False
    return manager


def _remote_manager(
    redis: _FakeRedis,
    *,
    on_degraded: Callable[[str, str], None] | None = None,
) -> IdempotencyManager:
    kwargs: dict[str, Any] = {}
    if on_degraded is not None:
        kwargs["on_degraded"] = on_degraded
    manager = IdempotencyManager(**kwargs)
    manager._redis_client = redis
    manager._redis_attempted = True
    manager._redis_ready = True
    return manager


def _result_key(cache_key: str) -> str:
    return f"mcp:idemp:result:{cache_key}"


def _lock_key(cache_key: str) -> str:
    return f"mcp:idemp:lock:{cache_key}"


def _binding_key(cache_key: str) -> str:
    return f"mcp:idemp:args:{cache_key}"


class _FakeRedis:
    """Small byte-oriented Redis double with explicit fault injection."""

    def __init__(self) -> None:
        self.values: dict[str, bytes] = {}
        self.calls: list[tuple[str, str]] = []
        self.set_expirations: list[tuple[str, int | None]] = []
        self.expirations: list[tuple[str, int]] = []
        self.lock_contended = asyncio.Event()
        self.lock_denials = 0
        self.binding_write_then_raise = False
        self.lock_write_then_raise = False
        self.result_write_then_raise = False
        self.release_delete_then_raise = False
        self.release_returns_false = False
        self.fail_poll_read = False
        self.fail_pre_owner_read = False
        self.block_poll_read = False
        self.poll_read_release = asyncio.Event()
        self.block_cleanup = False
        self.cleanup_entered = asyncio.Event()
        self.cleanup_release = asyncio.Event()

    @staticmethod
    def _bytes(value: Any) -> bytes:
        if isinstance(value, bytes):
            return value
        return str(value).encode("utf-8")

    async def get(self, key: str) -> bytes | None:
        self.calls.append(("get", key))
        if self.fail_pre_owner_read and key.startswith("mcp:idemp:result:"):
            self.fail_pre_owner_read = False
            raise RedisError("pre-owner credential=TOP_SECRET")
        if self.block_poll_read and key.startswith("mcp:idemp:result:") and self.lock_denials:
            await self.poll_read_release.wait()
        if self.fail_poll_read and key.startswith("mcp:idemp:result:") and self.lock_denials:
            raise RedisError("poll credential=TOP_SECRET")
        return self.values.get(key)

    async def set(
        self,
        key: str,
        value: Any,
        *,
        nx: bool = False,
        ex: int | None = None,
    ) -> bool:
        self.calls.append(("set", key))
        self.set_expirations.append((key, ex))
        encoded = self._bytes(value)
        if nx and key in self.values:
            if key.startswith("mcp:idemp:lock:"):
                self.lock_denials += 1
                self.lock_contended.set()
            return False
        if nx:
            self.values[key] = encoded
            if key.startswith("mcp:idemp:args:") and self.binding_write_then_raise:
                raise RedisError("binding credential=TOP_SECRET")
            if key.startswith("mcp:idemp:lock:") and self.lock_write_then_raise:
                raise RedisError("lock credential=TOP_SECRET")
            return True
        self.values[key] = encoded
        if key.startswith("mcp:idemp:result:") and self.result_write_then_raise:
            raise RedisError("result credential=TOP_SECRET")
        return True

    async def expire(self, key: str, ttl: int) -> bool:
        self.calls.append(("expire", key))
        self.expirations.append((key, ttl))
        if self.block_cleanup:
            self.cleanup_entered.set()
            await self.cleanup_release.wait()
        return key in self.values

    async def eval(self, script: str, key_count: int, *values: Any) -> int:
        del script
        keys = [str(value) for value in values[:key_count]]
        for key in keys:
            self.calls.append(("eval", key))

        if key_count == 2 and len(values) == 4:
            binding_key, result_key, arguments_hash, ttl = values
            encoded_hash = self._bytes(arguments_hash)
            existing = self.values.get(str(binding_key))
            if existing is not None:
                if existing != encoded_hash:
                    return -1
                self.expirations.append((str(binding_key), int(ttl)))
                return 1
            if str(result_key) in self.values:
                return -2
            self.values[str(binding_key)] = encoded_hash
            self.set_expirations.append((str(binding_key), int(ttl)))
            if self.binding_write_then_raise:
                raise RedisError("binding credential=TOP_SECRET")
            return 2

        if key_count == 2 and len(values) == 5:
            binding_key, result_key, arguments_hash, encoded, ttl = values
            if self.values.get(str(binding_key)) != self._bytes(arguments_hash):
                return 0
            self.values[str(result_key)] = self._bytes(encoded)
            self.set_expirations.append((str(result_key), int(ttl)))
            self.expirations.append((str(binding_key), int(ttl)))
            if self.result_write_then_raise:
                raise RedisError("result credential=TOP_SECRET")
            return 1

        if key_count == 1 and len(values) == 3:
            binding_key, arguments_hash, ttl = values
            if self.values.get(str(binding_key)) != self._bytes(arguments_hash):
                return 0
            self.expirations.append((str(binding_key), int(ttl)))
            if self.block_cleanup:
                self.cleanup_entered.set()
                await self.cleanup_release.wait()
            return 1

        if key_count != 1 or len(values) != 2:
            raise AssertionError("Unexpected Redis Lua operation")
        key, token = (str(values[0]), values[1])
        if self.block_cleanup:
            self.cleanup_entered.set()
            await self.cleanup_release.wait()
        deleted = 0
        if self.values.get(key) == self._bytes(token):
            del self.values[key]
            deleted = 1
        if self.release_delete_then_raise:
            raise RedisError("release credential=TOP_SECRET")
        if self.release_returns_false:
            return 0
        return deleted


@pytest.mark.unit
def test_run_result_and_executor_are_frozen_narrow_types() -> None:
    models = importlib.import_module(
        "tldw_Server_API.app.core.MCP_unified.tool_execution.models",
    )
    dependencies = importlib.import_module(
        "tldw_Server_API.app.core.MCP_unified.tool_execution.dependencies",
    )
    result_type = getattr(models, "IdempotencyRunResult", None)
    executor_type = getattr(dependencies, "IdempotencyExecutor", None)

    assert result_type is not None
    assert executor_type is not None
    result = result_type(payload={"ok": True}, from_cache=False, persistence="local")
    with pytest.raises(FrozenInstanceError):
        result.from_cache = True

    source = inspect.getsource(ToolExecutionRuntime.execute_prepared_tool_call)
    assert source.count("self.idempotency.execute(") == 1
    assert ".bind_arguments(" not in source
    assert "self.idempotency.run(" not in source


@pytest.mark.unit
def test_pending_local_owner_lock_is_not_pruned_before_acquire() -> None:
    manager = _local_manager()
    pending_lock = manager._get_local_lock("pending-owner")

    with manager._local_guard:
        manager._prune_local_locks_locked()

    assert manager._local_locks["pending-owner"] is pending_lock


@pytest.mark.unit
@pytest.mark.asyncio
async def test_local_and_redis_replay_call_callback_once_and_return_fresh_copies() -> None:
    for manager in (_local_manager(), _remote_manager(_FakeRedis())):
        calls = 0
        original = {"content": [{"type": "json", "json": {"items": [1]}}]}

        async def _execute(payload: dict[str, Any] = original) -> dict[str, Any]:
            nonlocal calls
            calls += 1
            return payload

        first = await manager.execute("replay", "args-a", _execute, policy=_policy())
        first.payload["content"][0]["json"]["items"].append(99)
        second = await manager.execute("replay", "args-a", _execute, policy=_policy())
        second.payload["content"][0]["json"]["items"].append(2)
        third = await manager.execute("replay", "args-a", _execute, policy=_policy())

        assert calls == 1
        assert first.from_cache is False
        assert second.from_cache is True
        assert third.from_cache is True
        assert third.payload == {"content": [{"json": {"items": [1]}, "type": "json"}]}
        assert first.payload is original
        assert second.payload is not third.payload


@pytest.mark.unit
@pytest.mark.asyncio
async def test_concurrent_local_waiter_times_out_without_cancelling_owner() -> None:
    manager = _local_manager()
    owner_started = asyncio.Event()
    finish_owner = asyncio.Event()
    calls = 0

    async def _execute() -> dict[str, Any]:
        nonlocal calls
        calls += 1
        owner_started.set()
        await finish_owner.wait()
        return {"value": "owner"}

    owner = asyncio.create_task(manager.execute("local-wait", "args", _execute, policy=_policy()))
    await asyncio.wait_for(owner_started.wait(), timeout=0.5)
    started_at = asyncio.get_running_loop().time()
    with pytest.raises(ExpectedToolFailure) as caught:
        await manager.execute("local-wait", "args", _execute, policy=_policy())
    elapsed = asyncio.get_running_loop().time() - started_at

    assert caught.value.reason is ExpectedToolFailureReason.IDEMPOTENCY_IN_PROGRESS
    assert 0.8 <= elapsed <= 1.8
    assert not owner.done()
    finish_owner.set()
    assert (await asyncio.wait_for(owner, timeout=0.5)).payload == {"value": "owner"}
    assert calls == 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_concurrent_redis_waiter_replays_result_before_deadline() -> None:
    redis = _FakeRedis()
    owner_manager = _remote_manager(redis)
    waiter_manager = _remote_manager(redis)
    owner_started = asyncio.Event()
    finish_owner = asyncio.Event()
    calls = 0

    async def _execute() -> dict[str, Any]:
        nonlocal calls
        calls += 1
        owner_started.set()
        await finish_owner.wait()
        return {"value": [1]}

    owner = asyncio.create_task(owner_manager.execute("redis-wait", "args", _execute, policy=_policy()))
    await asyncio.wait_for(owner_started.wait(), timeout=0.5)
    waiter = asyncio.create_task(waiter_manager.execute("redis-wait", "args", _execute, policy=_policy()))
    await asyncio.wait_for(redis.lock_contended.wait(), timeout=0.5)
    finish_owner.set()
    owner_result, waiter_result = await asyncio.gather(owner, waiter)

    assert calls == 1
    assert owner_result.from_cache is False
    assert waiter_result.from_cache is True
    assert waiter_result.payload == owner_result.payload
    assert waiter_result.payload is not owner_result.payload


@pytest.mark.unit
@pytest.mark.asyncio
async def test_concurrent_redis_waiter_times_out_without_dispatch() -> None:
    redis = _FakeRedis()
    redis.values[_lock_key("redis-timeout")] = b"other-owner"
    manager = _remote_manager(redis)
    calls = 0

    async def _execute() -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return {"unexpected": True}

    started_at = asyncio.get_running_loop().time()
    with pytest.raises(ExpectedToolFailure) as caught:
        await manager.execute("redis-timeout", "args", _execute, policy=_policy())
    elapsed = asyncio.get_running_loop().time() - started_at

    assert caught.value.reason is ExpectedToolFailureReason.IDEMPOTENCY_IN_PROGRESS
    assert 0.8 <= elapsed <= 1.8
    assert calls == 0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_redis_poll_error_after_lock_denial_fails_closed() -> None:
    redis = _FakeRedis()
    redis.values[_lock_key("redis-poll-error")] = b"other-owner"
    redis.fail_poll_read = True
    stages: list[tuple[str, str]] = []
    manager = _remote_manager(
        redis,
        on_degraded=lambda stage, error_type: stages.append((stage, error_type)),
    )
    calls = 0

    async def _execute() -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return {"unexpected": True}

    with pytest.raises(ExpectedToolFailure) as caught:
        await manager.execute("redis-poll-error", "args", _execute, policy=_policy())

    assert caught.value.reason is ExpectedToolFailureReason.IDEMPOTENCY_UNAVAILABLE
    assert calls == 0
    assert stages == [("redis_result_read", "RedisError")]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_redis_poll_await_uses_the_same_contention_deadline() -> None:
    redis = _FakeRedis()
    redis.values[_lock_key("redis-poll-deadline")] = b"other-owner"
    redis.block_poll_read = True
    manager = _remote_manager(redis)
    calls = 0

    async def _execute() -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return {"unexpected": True}

    started_at = asyncio.get_running_loop().time()
    with pytest.raises(ExpectedToolFailure) as caught:
        await asyncio.wait_for(
            manager.execute("redis-poll-deadline", "args", _execute, policy=_policy()),
            timeout=2,
        )
    elapsed = asyncio.get_running_loop().time() - started_at

    assert caught.value.reason is ExpectedToolFailureReason.IDEMPOTENCY_UNAVAILABLE
    assert 0.8 <= elapsed <= 1.8
    assert calls == 0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_redis_result_read_failure_before_set_nx_falls_back_once_locally() -> None:
    redis = _FakeRedis()
    redis.fail_pre_owner_read = True
    manager = _remote_manager(redis)
    calls = 0

    async def _execute() -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return {"fallback": True}

    result = await manager.execute("pre-owner-fallback", "args", _execute, policy=_policy())

    assert result.payload == {"fallback": True}
    assert result.persistence == "local"
    assert calls == 1
    assert all(key != _lock_key("pre-owner-fallback") for _operation, key in redis.calls)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_callback_private_pre_owner_sentinel_never_triggers_backend_fallback() -> None:
    idempotency_module = importlib.import_module(
        "tldw_Server_API.app.core.MCP_unified.tool_execution.idempotency",
    )
    sentinel_type = idempotency_module._PreOwnerRedisFailure
    sentinel = sentinel_type("callback-owned exception")
    manager = _remote_manager(_FakeRedis())
    calls = 0

    async def _execute() -> dict[str, Any]:
        nonlocal calls
        calls += 1
        raise sentinel

    with pytest.raises(sentinel_type) as caught:
        await manager.execute("private-pre-owner-sentinel", "args", _execute, policy=_policy())

    assert calls == 1
    assert caught.value is sentinel


@pytest.mark.unit
@pytest.mark.asyncio
async def test_redis_lock_write_then_raise_is_ambiguous_and_never_falls_back() -> None:
    redis = _FakeRedis()
    redis.lock_write_then_raise = True
    stages: list[tuple[str, str]] = []
    manager = _remote_manager(
        redis,
        on_degraded=lambda stage, error_type: stages.append((stage, error_type)),
    )
    calls = 0

    async def _execute() -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return {"unexpected": True}

    with pytest.raises(ExpectedToolFailure) as caught:
        await manager.execute("lock-ambiguous", "args", _execute, policy=_policy())

    assert caught.value.reason is ExpectedToolFailureReason.IDEMPOTENCY_UNAVAILABLE
    assert calls == 0
    assert _lock_key("lock-ambiguous") in redis.values
    assert "lock-ambiguous" not in manager._local_cache
    assert stages == [("redis_lock_acquire", "RedisError")]

    redis.lock_write_then_raise = False
    with pytest.raises(ExpectedToolFailure) as retry_caught:
        await manager.execute("lock-ambiguous", "args", _execute, policy=_policy())
    assert retry_caught.value.reason is ExpectedToolFailureReason.IDEMPOTENCY_UNAVAILABLE
    assert calls == 0


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fault",
    ["result_write_then_raise", "release_delete_then_raise", "release_returns_false"],
)
async def test_redis_post_callback_fault_returns_original_and_keeps_local_replay(fault: str) -> None:
    redis = _FakeRedis()
    setattr(redis, fault, True)
    stages: list[tuple[str, str]] = []
    manager = _remote_manager(
        redis,
        on_degraded=lambda stage, error_type: stages.append((stage, error_type)),
    )
    calls = 0
    original = {"content": [{"type": "text", "text": "paid-success"}]}

    async def _execute() -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return original

    first = await manager.execute("post-success-fault", "args", _execute, policy=_policy())
    replay = await manager.execute("post-success-fault", "args", _execute, policy=_policy())

    assert calls == 1
    assert first.payload is original
    expected_persistence = "local" if fault == "result_write_then_raise" else "durable"
    assert first.persistence == expected_persistence
    assert replay.from_cache is True
    assert replay.payload == original
    assert replay.payload is not original
    assert manager.remote_degraded is True
    if fault == "result_write_then_raise":
        assert stages == [("redis_result_write", "RedisError")]
    elif fault == "release_delete_then_raise":
        assert stages == [("redis_release", "RedisError")]
    else:
        assert stages == [("redis_release", "RuntimeError")]

    if fault != "result_write_then_raise":
        result_key = _result_key("post-success-fault")
        lock_key = _lock_key("post-success-fault")
        assert redis.values[result_key] == canonical_json_bytes(
            original,
            max_bytes=_policy().max_result_bytes,
        )
        lock_sets_before = redis.calls.count(("set", lock_key))
        fresh_replay = await _remote_manager(redis).execute(
            "post-success-fault",
            "args",
            _execute,
            policy=_policy(),
        )
        assert calls == 1
        assert fresh_replay.from_cache is True
        assert fresh_replay.persistence == "durable"
        assert redis.calls.count(("set", lock_key)) == lock_sets_before


@pytest.mark.unit
@pytest.mark.asyncio
async def test_argument_binding_write_then_raise_fails_closed_without_callback() -> None:
    redis = _FakeRedis()
    redis.binding_write_then_raise = True
    stages: list[tuple[str, str]] = []
    manager = _remote_manager(
        redis,
        on_degraded=lambda stage, error_type: stages.append((stage, error_type)),
    )
    calls = 0

    async def _execute() -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return {"unexpected": True}

    with pytest.raises(ExpectedToolFailure) as caught:
        await manager.execute("binding-ambiguous", "args", _execute, policy=_policy())

    assert caught.value.reason is ExpectedToolFailureReason.IDEMPOTENCY_UNAVAILABLE
    assert calls == 0
    assert _binding_key("binding-ambiguous") in redis.values
    assert "binding-ambiguous" not in manager._local_bindings
    assert stages == [("redis_binding", "RedisError")]

    redis.binding_write_then_raise = False
    with pytest.raises(ExpectedToolFailure) as retry_caught:
        await manager.execute("binding-ambiguous", "args", _execute, policy=_policy())
    assert retry_caught.value.reason is ExpectedToolFailureReason.IDEMPOTENCY_UNAVAILABLE
    assert calls == 0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_orphan_remote_result_cannot_be_rebound_or_replayed() -> None:
    redis = _FakeRedis()
    key = "orphan-result"
    redis.values[_result_key(key)] = canonical_json_bytes(
        {"from": "old-arguments"},
        max_bytes=_policy().max_result_bytes,
    )
    stages: list[tuple[str, str]] = []
    manager = _remote_manager(
        redis,
        on_degraded=lambda stage, error_type: stages.append((stage, error_type)),
    )
    calls = 0

    async def _execute() -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return {"unexpected": True}

    with pytest.raises(ExpectedToolFailure) as caught:
        await manager.execute(key, "new-arguments", _execute, policy=_policy())

    assert caught.value.reason is ExpectedToolFailureReason.IDEMPOTENCY_UNAVAILABLE
    assert calls == 0
    assert _binding_key(key) not in redis.values
    assert key not in manager._local_bindings
    assert manager.remote_degraded is True
    assert stages == [("redis_binding", "RuntimeError")]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_stale_remote_owner_cannot_publish_after_argument_rebind() -> None:
    redis = _FakeRedis()
    key = "stale-owner"
    stages: list[tuple[str, str]] = []
    manager = _remote_manager(
        redis,
        on_degraded=lambda stage, error_type: stages.append((stage, error_type)),
    )
    original = {"content": [{"type": "text", "text": "completed"}]}
    calls = 0

    async def _execute() -> dict[str, Any]:
        nonlocal calls
        calls += 1
        redis.values[_binding_key(key)] = b"replacement-arguments"
        return original

    result = await manager.execute(key, "owner-arguments", _execute, policy=_policy())

    assert calls == 1
    assert result.payload is original
    assert result.persistence == "local"
    assert _result_key(key) not in redis.values
    assert stages == [("redis_result_write", "RuntimeError")]


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("payload_factory", "max_result_bytes", "expected_stage"),
    [
        (lambda: {"value": object()}, 4_096, "serialization"),
        (lambda: {"value": float("nan")}, 4_096, "serialization"),
        (lambda: {"value": "x" * 1_000}, 64, "result_size"),
    ],
)
@pytest.mark.parametrize("backend", ["local", "redis"])
async def test_uncacheable_success_is_returned_unchanged_without_persistence(
    payload_factory: Callable[[], dict[str, Any]],
    max_result_bytes: int,
    expected_stage: str,
    backend: str,
) -> None:
    stages: list[tuple[str, str]] = []
    redis = _FakeRedis()
    manager = (
        _local_manager(on_degraded=lambda stage, error_type: stages.append((stage, error_type)))
        if backend == "local"
        else _remote_manager(
            redis,
            on_degraded=lambda stage, error_type: stages.append((stage, error_type)),
        )
    )
    original = payload_factory()

    async def _execute() -> dict[str, Any]:
        return original

    result = await manager.execute(
        f"uncacheable-{backend}-{expected_stage}",
        "args",
        _execute,
        policy=_policy(max_result_bytes=max_result_bytes),
    )

    assert result.payload is original
    assert result.from_cache is False
    assert result.persistence == "none"
    assert f"uncacheable-{backend}-{expected_stage}" not in manager._local_cache
    assert _result_key(f"uncacheable-{backend}-{expected_stage}") not in redis.values
    assert [stage for stage, _ in stages] == [expected_stage]
    assert manager.remote_degraded is False


@pytest.mark.unit
@pytest.mark.asyncio
async def test_remote_uncacheable_success_survives_local_binding_refresh_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    redis = _FakeRedis()
    stages: list[tuple[str, str]] = []
    manager = _remote_manager(
        redis,
        on_degraded=lambda stage, error_type: stages.append((stage, error_type)),
    )
    original_put = manager._put_local_binding_locked
    fail_refresh = False
    original = {"value": object()}

    def _put_binding(*args: Any, **kwargs: Any) -> None:
        if fail_refresh:
            raise RuntimeError("private local binding detail")
        original_put(*args, **kwargs)

    monkeypatch.setattr(manager, "_put_local_binding_locked", _put_binding)

    async def _execute() -> dict[str, Any]:
        nonlocal fail_refresh
        fail_refresh = True
        return original

    result = await manager.execute("uncacheable-refresh", "args", _execute, policy=_policy())

    assert result.payload is original
    assert result.persistence == "none"
    assert stages == [
        ("serialization", "TypeError"),
        ("local_commit", "RuntimeError"),
    ]
    assert (_binding_key("uncacheable-refresh"), _policy().ttl_seconds) in redis.expirations
    assert _lock_key("uncacheable-refresh") not in redis.values


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["local", "redis"])
@pytest.mark.parametrize("expected", [True, False], ids=["expected-failure", "unexpected-failure"])
async def test_callback_failures_cache_no_result_and_retain_argument_binding(
    backend: str,
    expected: bool,
) -> None:
    redis = _FakeRedis()
    manager = _local_manager() if backend == "local" else _remote_manager(redis)
    key = f"callback-failure-{backend}-{expected}"
    calls = 0

    async def _fail() -> dict[str, Any]:
        nonlocal calls
        calls += 1
        if expected:
            raise ExpectedToolFailure(ExpectedToolFailureReason.DEPENDENCY_UNAVAILABLE)
        raise RuntimeError("private callback detail")

    failure_type = ExpectedToolFailure if expected else RuntimeError
    with pytest.raises(failure_type):
        await manager.execute(key, "args-a", _fail, policy=_policy())

    assert key not in manager._local_cache
    assert _result_key(key) not in redis.values
    assert calls == 1

    async def _must_not_run() -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return {"unexpected": True}

    with pytest.raises(InvalidParamsException):
        await manager.execute(key, "args-b", _must_not_run, policy=_policy())
    assert calls == 1
    if backend == "local":
        assert manager._local_bindings[key][1] == "args-a"
    else:
        assert redis.values[_binding_key(key)] == b"args-a"
        assert (_binding_key(key), _policy().ttl_seconds) in redis.set_expirations
        assert (_lock_key(key), _policy().lock_ttl_seconds) in redis.set_expirations
        assert (_binding_key(key), _policy().ttl_seconds) in redis.expirations


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "callback_exception",
    [ValueError("callback failure"), asyncio.CancelledError("callback cancellation")],
    ids=["failure", "cancellation"],
)
async def test_local_callback_outcome_survives_binding_refresh_failure(
    callback_exception: BaseException,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stages: list[tuple[str, str]] = []
    manager = _local_manager(
        on_degraded=lambda stage, error_type: stages.append((stage, error_type)),
    )
    original_put = manager._put_local_binding_locked
    fail_refresh = False

    def _put_binding(*args: Any, **kwargs: Any) -> None:
        if fail_refresh:
            raise RuntimeError("private local binding detail")
        original_put(*args, **kwargs)

    monkeypatch.setattr(manager, "_put_local_binding_locked", _put_binding)

    async def _execute() -> dict[str, Any]:
        nonlocal fail_refresh
        fail_refresh = True
        raise callback_exception

    with pytest.raises(type(callback_exception)) as caught:
        await manager.execute("local-refresh-outcome", "args", _execute, policy=_policy())

    assert caught.value is callback_exception
    assert stages == [("local_commit", "RuntimeError")]
    assert "local-refresh-outcome" not in manager._local_cache


@pytest.mark.unit
@pytest.mark.asyncio
async def test_remote_callback_failure_survives_local_binding_refresh_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    redis = _FakeRedis()
    stages: list[tuple[str, str]] = []
    manager = _remote_manager(
        redis,
        on_degraded=lambda stage, error_type: stages.append((stage, error_type)),
    )
    original_put = manager._put_local_binding_locked
    fail_refresh = False
    callback_exception = ValueError("callback failure")

    def _put_binding(*args: Any, **kwargs: Any) -> None:
        if fail_refresh:
            raise RuntimeError("private local binding detail")
        original_put(*args, **kwargs)

    monkeypatch.setattr(manager, "_put_local_binding_locked", _put_binding)

    async def _execute() -> dict[str, Any]:
        nonlocal fail_refresh
        fail_refresh = True
        raise callback_exception

    with pytest.raises(ValueError) as caught:
        await manager.execute("remote-refresh-outcome", "args", _execute, policy=_policy())

    assert caught.value is callback_exception
    assert stages == [("local_commit", "RuntimeError")]
    assert (_binding_key("remote-refresh-outcome"), _policy().ttl_seconds) in redis.expirations
    assert _lock_key("remote-refresh-outcome") not in redis.values


@pytest.mark.unit
@pytest.mark.asyncio
async def test_callback_failure_with_ambiguous_release_blocks_local_retry() -> None:
    redis = _FakeRedis()
    redis.release_delete_then_raise = True
    manager = _remote_manager(redis)
    calls = 0

    async def _fail() -> dict[str, Any]:
        nonlocal calls
        calls += 1
        raise RuntimeError("private callback detail")

    with pytest.raises(RuntimeError):
        await manager.execute("failed-release", "args", _fail, policy=_policy())

    redis.release_delete_then_raise = False
    with pytest.raises(ExpectedToolFailure) as caught:
        await manager.execute("failed-release", "args", _fail, policy=_policy())

    assert caught.value.reason is ExpectedToolFailureReason.IDEMPOTENCY_UNAVAILABLE
    assert calls == 1
    assert "failed-release" not in manager._local_cache


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["local", "redis"])
async def test_cancellation_before_success_propagates_without_cache_or_fallback(backend: str) -> None:
    redis = _FakeRedis()
    manager = _local_manager() if backend == "local" else _remote_manager(redis)
    started = asyncio.Event()
    never_finish = asyncio.Event()
    calls = 0
    key = f"cancel-{backend}"

    async def _execute() -> dict[str, Any]:
        nonlocal calls
        calls += 1
        started.set()
        await never_finish.wait()
        return {"too_late": True}

    task = asyncio.create_task(manager.execute(key, "args", _execute, policy=_policy()))
    await asyncio.wait_for(started.wait(), timeout=0.5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert calls == 1
    assert key not in manager._local_cache
    assert _result_key(key) not in redis.values
    if backend == "redis":
        assert (_binding_key(key), _policy().ttl_seconds) in redis.set_expirations
        assert (_lock_key(key), _policy().lock_ttl_seconds) in redis.set_expirations


@pytest.mark.unit
@pytest.mark.asyncio
async def test_remote_cancellation_propagates_before_blocking_cleanup() -> None:
    redis = _FakeRedis()
    redis.block_cleanup = True
    manager = _remote_manager(redis)
    started = asyncio.Event()
    never_finish = asyncio.Event()
    key = "cancel-before-cleanup"

    async def _execute() -> dict[str, Any]:
        started.set()
        await never_finish.wait()
        return {"too_late": True}

    task = asyncio.create_task(manager.execute(key, "args", _execute, policy=_policy()))
    await asyncio.wait_for(started.wait(), timeout=0.5)
    cleanup_probe = asyncio.create_task(redis.cleanup_entered.wait())
    task.cancel()
    done, _pending = await asyncio.wait(
        {task, cleanup_probe},
        timeout=0.5,
        return_when=asyncio.FIRST_COMPLETED,
    )
    completed_before_cleanup = task in done
    redis.cleanup_release.set()
    if not task.done():
        with pytest.raises(asyncio.CancelledError):
            await task
    cleanup_probe.cancel()
    await asyncio.gather(cleanup_probe, return_exceptions=True)

    assert completed_before_cleanup
    assert not redis.cleanup_entered.is_set()
    assert _binding_key(key) in redis.values
    assert _lock_key(key) in redis.values
    assert _result_key(key) not in redis.values


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["local", "redis"])
async def test_fresh_success_survives_local_replay_install_failure(
    backend: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    redis = _FakeRedis()
    stages: list[tuple[str, str]] = []
    manager = (
        _local_manager(on_degraded=lambda stage, error_type: stages.append((stage, error_type)))
        if backend == "local"
        else _remote_manager(
            redis,
            on_degraded=lambda stage, error_type: stages.append((stage, error_type)),
        )
    )
    original = {"content": [{"type": "text", "text": "committed externally"}]}
    calls = 0

    def _fail_local_commit(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("private local commit detail")

    monkeypatch.setattr(manager, "_put_local_replay_locked", _fail_local_commit)

    async def _execute() -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return original

    result = await manager.execute(f"local-commit-{backend}", "args", _execute, policy=_policy())

    assert calls == 1
    assert result.payload is original
    assert result.persistence == "none"
    assert _result_key(f"local-commit-{backend}") not in redis.values
    assert stages == [("local_commit", "RuntimeError")]
    assert manager.remote_degraded is False


@pytest.mark.unit
@pytest.mark.asyncio
async def test_verified_remote_replay_survives_local_install_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    redis = _FakeRedis()
    key = "remote-replay-local-commit"
    payload = {"content": [{"type": "text", "text": "durable"}]}
    redis.values[_binding_key(key)] = b"args"
    redis.values[_result_key(key)] = canonical_json_bytes(
        payload,
        max_bytes=_policy().max_result_bytes,
    )
    stages: list[tuple[str, str]] = []
    manager = _remote_manager(
        redis,
        on_degraded=lambda stage, error_type: stages.append((stage, error_type)),
    )
    calls = 0

    def _fail_local_commit(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("private local commit detail")

    monkeypatch.setattr(manager, "_put_local_replay_locked", _fail_local_commit)

    async def _execute() -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return {"unexpected": True}

    replay = await manager.execute(key, "args", _execute, policy=_policy())

    assert calls == 0
    assert replay.payload == payload
    assert replay.payload is not payload
    assert replay.from_cache is True
    assert replay.persistence == "durable"
    assert stages == [("local_commit", "RuntimeError")]
    assert manager.remote_degraded is False


@pytest.mark.unit
@pytest.mark.asyncio
async def test_scoped_keys_partition_all_manager_state_and_personal_key_is_exact() -> None:
    protocol = MCPProtocol()
    manager = _local_manager()
    org_id = 101_001
    team_id = 202_002
    org_context = RequestContext(
        request_id="org",
        user_id="user-1",
        server_auth_scope=AuthenticatedExecutionScope(active_org_id=org_id),
    )
    team_context = RequestContext(
        request_id="team",
        user_id="user-1",
        server_auth_scope=AuthenticatedExecutionScope(active_team_id=team_id),
    )
    personal_context = RequestContext(request_id="personal", user_id="user-1")
    org_key = protocol._make_idempotency_cache_key(org_context, "module-a", "tool.write", "key-1")
    team_key = protocol._make_idempotency_cache_key(team_context, "module-a", "tool.write", "key-1")
    personal_key = protocol._make_idempotency_cache_key(
        personal_context,
        "module-a",
        "tool.write",
        "key-1",
    )
    calls = 0

    async def _execute() -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return {"call": calls}

    org_result = await manager.execute(org_key, "same-args", _execute, policy=_policy())
    team_result = await manager.execute(team_key, "same-args", _execute, policy=_policy())
    org_replay = await manager.execute(org_key, "same-args", _execute, policy=_policy())

    assert calls == 2
    assert org_result.payload == org_replay.payload == {"call": 1}
    assert team_result.payload == {"call": 2}
    assert personal_key == "user:user-1|module:module-a|tool:tool.write|key:key-1"
    assert org_key != team_key
    assert re.search(r"scope:sha256:[0-9a-f]{64}", org_key)
    assert re.search(r"scope:sha256:[0-9a-f]{64}", team_key)
    for state_key in set(manager._local_cache) | set(manager._local_bindings) | set(manager._local_locks):
        assert str(org_id) not in state_key
        assert str(team_id) not in state_key

    redis = _FakeRedis()
    remote = _remote_manager(redis)
    await remote.execute(org_key, "same-args", _execute, policy=_policy())
    await remote.execute(team_key, "same-args", _execute, policy=_policy())
    redis_keys = {key for _operation, key in redis.calls}
    for redis_key in redis_keys:
        assert str(org_id) not in redis_key
        assert str(team_id) not in redis_key
    for scoped_key in (org_key, team_key):
        assert _result_key(scoped_key) in redis_keys
        assert _lock_key(scoped_key) in redis_keys
        assert _binding_key(scoped_key) in redis_keys


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("encoded", [b"not-json", b'{"b":1,"a":2}'])
async def test_corrupt_or_noncanonical_remote_result_fails_closed_without_dispatch(encoded: bytes) -> None:
    redis = _FakeRedis()
    key = "corrupt-result"
    redis.values[_binding_key(key)] = b"args"
    redis.values[_result_key(key)] = encoded
    manager = _remote_manager(redis)
    calls = 0

    async def _execute() -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return {"unexpected": True}

    with pytest.raises(ExpectedToolFailure) as caught:
        await manager.execute(key, "args", _execute, policy=_policy())

    assert caught.value.reason is ExpectedToolFailureReason.IDEMPOTENCY_UNAVAILABLE
    assert calls == 0
    assert key not in manager._local_cache

    with pytest.raises(ExpectedToolFailure) as retry_caught:
        await manager.execute(key, "args", _execute, policy=_policy())
    assert retry_caught.value.reason is ExpectedToolFailureReason.IDEMPOTENCY_UNAVAILABLE
    assert calls == 0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_redis_factory_disables_response_decoding(monkeypatch: pytest.MonkeyPatch) -> None:
    idempotency_module = importlib.import_module(
        "tldw_Server_API.app.core.MCP_unified.tool_execution.idempotency",
    )
    redis = _FakeRedis()
    factory_kwargs: dict[str, Any] = {}

    class _Config:
        def get_redis_connection_params(self) -> dict[str, str]:
            return {"url": "redis://localhost:6379/0"}

    async def _factory(**kwargs: Any) -> _FakeRedis:
        factory_kwargs.update(kwargs)
        return redis

    monkeypatch.setattr(idempotency_module, "get_config", lambda: _Config())
    manager = IdempotencyManager(redis_client_factory=_factory)

    await manager.execute("decode-false", "args", lambda: _async_payload({"ok": True}), policy=_policy())

    assert factory_kwargs["decode_responses"] is False


@pytest.mark.unit
@pytest.mark.asyncio
async def test_cancelled_redis_factory_attempt_can_retry_without_false_degradation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    idempotency_module = importlib.import_module(
        "tldw_Server_API.app.core.MCP_unified.tool_execution.idempotency",
    )
    redis = _FakeRedis()
    first_started = asyncio.Event()
    never_finishes = asyncio.Event()
    attempts = 0

    class _Config:
        def get_redis_connection_params(self) -> dict[str, str]:
            return {"url": "redis://localhost:6379/0"}

    async def _factory(**_kwargs: Any) -> _FakeRedis:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            first_started.set()
            await never_finishes.wait()
        return redis

    monkeypatch.setattr(idempotency_module, "get_config", lambda: _Config())
    manager = IdempotencyManager(redis_client_factory=_factory)
    first_attempt = asyncio.create_task(manager._ensure_redis())
    await asyncio.wait_for(first_started.wait(), timeout=1)

    first_attempt.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first_attempt

    assert manager._redis_attempted is False
    assert manager._redis_ready is False
    assert manager._redis_client is None
    assert manager.remote_degraded is False
    assert await manager._ensure_redis() is True
    assert attempts == 2
    assert manager._redis_client is redis


@pytest.mark.unit
@pytest.mark.asyncio
async def test_post_mutation_local_cache_failure_rolls_back_before_redispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = _local_manager()
    original_put = manager._put_local_replay_locked
    fail_once = True
    calls = 0

    def _put_then_fail(*args: Any, **kwargs: Any) -> None:
        nonlocal fail_once
        original_put(*args, **kwargs)
        if fail_once:
            fail_once = False
            raise RuntimeError("post-mutation local cache failure")

    monkeypatch.setattr(manager, "_put_local_replay_locked", _put_then_fail)

    async def _execute() -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return {"call": calls}

    first = await manager.execute("local-rollback", "args", _execute, policy=_policy())

    assert first.persistence == "none"
    assert "local-rollback" not in manager._local_cache
    assert "local-rollback" in manager._local_bindings

    second = await manager.execute("local-rollback", "args", _execute, policy=_policy())
    assert calls == 2
    assert second.from_cache is False
    assert second.persistence == "local"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_post_mutation_include_binding_failure_restores_prior_lru_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = _local_manager()
    await manager.execute("prior-a", "args-a", lambda: _async_payload({"a": 1}), policy=_policy())
    await manager.execute("prior-b", "args-b", lambda: _async_payload({"b": 2}), policy=_policy())
    cache_before = list(manager._local_cache.items())
    bindings_before = list(manager._local_bindings.items())
    original_put = manager._put_local_replay_locked

    def _put_then_fail(*args: Any, **kwargs: Any) -> None:
        original_put(*args, **kwargs)
        raise RuntimeError("post-mutation include-binding failure")

    monkeypatch.setattr(manager, "_put_local_replay_locked", _put_then_fail)
    template = {"new": True}
    encoded = canonical_json_bytes(template, max_bytes=_policy().max_result_bytes)

    committed = manager._try_commit_local_replay(
        "new-key",
        "new-args",
        template,
        encoded,
        policy=_policy(),
        include_binding=True,
    )

    assert committed is False
    assert list(manager._local_cache.items()) == cache_before
    assert list(manager._local_bindings.items()) == bindings_before


@pytest.mark.unit
@pytest.mark.asyncio
async def test_hostile_degraded_observer_cannot_replace_outcome_or_leak_name() -> None:
    hostile_error = type("Observer=TOP_SECRET", (BaseException,), {})

    def _hostile_observer(_stage: str, _error_type: str) -> None:
        raise hostile_error()

    messages: list[str] = []
    sink_id = logger.add(lambda message: messages.append(str(message)), format="{message}")
    original = {"value": object()}
    try:
        result = await _local_manager(on_degraded=_hostile_observer).execute(
            "hostile-observer",
            "args",
            lambda: _async_payload(original),
            policy=_policy(),
        )
    finally:
        logger.remove(sink_id)

    assert result.payload is original
    assert result.persistence == "none"
    assert all("TOP_SECRET" not in message for message in messages)
    assert any(
        "degraded observer failed error_type=Exception" in message
        for message in messages
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_hostile_exception_attribute_hook_cannot_replace_degraded_outcome() -> None:
    class _HostileObserverError(BaseException):
        def __getattribute__(self, name: str) -> Any:
            if name == "__class__":
                raise RuntimeError("observer credential=TOP_SECRET")
            return super().__getattribute__(name)

    def _hostile_observer(_stage: str, _error_type: str) -> None:
        raise _HostileObserverError()

    messages: list[str] = []
    sink_id = logger.add(lambda message: messages.append(str(message)), format="{message}")
    original = {"value": object()}
    try:
        try:
            result = await _local_manager(on_degraded=_hostile_observer).execute(
                "hostile-observer-attribute-hook",
                "args",
                lambda: _async_payload(original),
                policy=_policy(),
            )
        except RuntimeError:
            raise AssertionError("hostile observer replaced the degraded outcome") from None
    finally:
        logger.remove(sink_id)

    assert result.payload is original
    assert result.persistence == "none"
    assert all("TOP_SECRET" not in message for message in messages)
    assert any(
        "degraded observer failed error_type=_HostileObserverError" in message
        for message in messages
    )


async def _async_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return payload


@pytest.mark.unit
@pytest.mark.asyncio
async def test_degraded_stages_are_bounded_safe_and_remote_health_is_specific() -> None:
    stages: list[tuple[str, str]] = []
    messages: list[str] = []
    sink_id = logger.add(lambda message: messages.append(str(message)), format="{message}")
    try:
        local = _local_manager(on_degraded=lambda stage, error_type: stages.append((stage, error_type)))
        invalid = {"value": object()}
        local_result = await local.execute(
            "local-degraded",
            "args",
            lambda: _async_payload(invalid),
            policy=_policy(),
        )

        redis = _FakeRedis()
        redis.result_write_then_raise = True
        remote = _remote_manager(
            redis,
            on_degraded=lambda stage, error_type: stages.append((stage, error_type)),
        )
        remote_result = await remote.execute(
            "remote-degraded",
            "args",
            lambda: _async_payload({"ok": True}),
            policy=_policy(),
        )
    finally:
        logger.remove(sink_id)

    assert local_result.persistence == "none"
    assert local.remote_degraded is False
    assert remote_result.persistence == "local"
    assert remote.remote_degraded is True
    assert [stage for stage, _ in stages] == ["serialization", "redis_result_write"]
    assert all(error_type in {"TypeError", "RedisError"} for _, error_type in stages)
    assert all("TOP_SECRET" not in message for message in messages)
    assert any("stage=serialization" in message and "error_type=TypeError" in message for message in messages)
    assert any(
        "stage=redis_result_write" in message and "error_type=RedisError" in message
        for message in messages
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_shutdown_is_available_without_task6_finalizers() -> None:
    manager = _local_manager()

    assert await manager.shutdown() is None
