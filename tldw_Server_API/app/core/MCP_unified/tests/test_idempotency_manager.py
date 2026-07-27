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
from tldw_Server_API.app.core.MCP_unified.server import MCPServer
from tldw_Server_API.app.core.MCP_unified.tool_execution.canonical import (
    canonical_json_bytes,
)
from tldw_Server_API.app.core.MCP_unified.tool_execution.idempotency import (
    IdempotencyManager,
    RedisError,
)
from tldw_Server_API.app.core.MCP_unified.tool_execution.models import (
    IdempotencyExecutionPolicy,
    IdempotencyRunResult,
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
        self.replay_reads: list[tuple[str, str, bytes, int]] = []
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
        self.replace_binding_after_result_read = False
        self.remove_binding_during_replay_read = False
        self.expire_result_during_replay_read = False
        self.binding_refresh_returns_false = False
        self.binding_refresh_raises = False
        self.binding_refresh_response: Any | None = None
        self.result_store_calls = 0
        self.lock_release_calls = 0

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
        value = self.values.get(key)
        if self.expire_result_during_replay_read and key.startswith("mcp:idemp:result:"):
            self.expire_result_during_replay_read = False
            self.values.pop(key, None)
        if self.replace_binding_after_result_read and key.startswith("mcp:idemp:result:"):
            self.replace_binding_after_result_read = False
            cache_key = key.removeprefix("mcp:idemp:result:")
            self.values[_binding_key(cache_key)] = b"different-arguments"
        return value

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

    async def eval(self, script: str, key_count: int, *values: Any) -> Any:
        keys = [str(value) for value in values[:key_count]]
        for key in keys:
            self.calls.append(("eval", key))

        if key_count == 2 and len(values) == 4 and "return {1, result}" in script:
            binding_key, result_key, arguments_hash, ttl = values
            binding_key = str(binding_key)
            result_key = str(result_key)
            encoded_hash = self._bytes(arguments_hash)
            self.replay_reads.append((binding_key, result_key, encoded_hash, int(ttl)))
            if self.fail_pre_owner_read:
                self.fail_pre_owner_read = False
                raise RedisError("pre-owner credential=TOP_SECRET")
            if self.block_poll_read and self.lock_denials:
                await self.poll_read_release.wait()
            if self.fail_poll_read and self.lock_denials:
                raise RedisError("poll credential=TOP_SECRET")
            if self.replace_binding_after_result_read:
                self.replace_binding_after_result_read = False
                self.values[binding_key] = b"different-arguments"
            if self.remove_binding_during_replay_read:
                self.remove_binding_during_replay_read = False
                self.values.pop(binding_key, None)
            binding = self.values.get(binding_key)
            if binding is None:
                return [-2]
            if binding != encoded_hash:
                return [-1]
            if self.expire_result_during_replay_read:
                self.expire_result_during_replay_read = False
                self.values.pop(result_key, None)
            result = self.values.get(result_key)
            if result is None:
                return [0]
            self.expirations.append((binding_key, int(ttl)))
            self.expirations.append((result_key, int(ttl)))
            return [1, result]

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

        if key_count == 3 and len(values) == 7:
            self.result_store_calls += 1
            binding_key, result_key, lock_key, arguments_hash, encoded, ttl, token = values
            if self.values.get(str(binding_key)) != self._bytes(arguments_hash):
                return 0
            if self.values.get(str(lock_key)) != self._bytes(token):
                return -1
            existing = self.values.get(str(result_key))
            if existing is not None and existing != self._bytes(encoded):
                return -2
            self.values[str(result_key)] = self._bytes(encoded)
            self.set_expirations.append((str(result_key), int(ttl)))
            self.expirations.append((str(binding_key), int(ttl)))
            if self.result_write_then_raise:
                raise RedisError("result credential=TOP_SECRET")
            return 1

        if key_count == 1 and len(values) == 3:
            binding_key, arguments_hash, ttl = values
            if self.binding_refresh_raises:
                raise RedisError("refresh credential=TOP_SECRET")
            if self.binding_refresh_response is not None:
                return self.binding_refresh_response
            if self.binding_refresh_returns_false:
                return 0
            if self.values.get(str(binding_key)) != self._bytes(arguments_hash):
                return 0
            self.expirations.append((str(binding_key), int(ttl)))
            if self.block_cleanup:
                self.cleanup_entered.set()
                await self.cleanup_release.wait()
            return 1

        if key_count != 1 or len(values) != 2:
            raise AssertionError("Unexpected Redis Lua operation")
        self.lock_release_calls += 1
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


class _BlockingResultRedis(_FakeRedis):
    """Redis double that can hold a result write across cancellation."""

    def __init__(self, *, ignore_cancellation: bool = False) -> None:
        super().__init__()
        self.ignore_cancellation = ignore_cancellation
        self.write_started = asyncio.Event()
        self.allow_write = asyncio.Event()
        self.write_cancellations = 0

    async def eval(self, script: str, key_count: int, *values: Any) -> Any:
        if key_count == 3 and len(values) == 7:
            self.write_started.set()
            while not self.allow_write.is_set():
                try:
                    await self.allow_write.wait()
                except asyncio.CancelledError:
                    self.write_cancellations += 1
                    if not self.ignore_cancellation:
                        raise
        return await super().eval(script, key_count, *values)


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
    expected_remote_degraded = fault != "release_returns_false"
    assert manager.remote_degraded is expected_remote_degraded
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
async def test_refresh_ownership_loss_evicts_uncertain_write_replay_key_locally() -> None:
    redis = _FakeRedis()
    redis.result_write_then_raise = True
    redis.binding_refresh_returns_false = True
    stages: list[tuple[str, str]] = []
    manager = _remote_manager(
        redis,
        on_degraded=lambda stage, error_type: stages.append((stage, error_type)),
    )
    payload = {"content": [{"type": "text", "text": "committed-local"}]}
    calls = 0

    async def _execute() -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return payload

    first = await manager.execute("uncertain-local", "args", _execute, policy=_policy())
    with pytest.raises(ExpectedToolFailure) as retry_caught:
        await manager.execute("uncertain-local", "args", _execute, policy=_policy())
    redis.result_write_then_raise = False
    redis.binding_refresh_returns_false = False
    unrelated = await manager.execute(
        "unrelated-after-refresh-loss",
        "other-args",
        lambda: _async_payload({"unrelated": True}),
        policy=_policy(),
    )

    assert first.persistence == "none"
    assert retry_caught.value.reason is ExpectedToolFailureReason.IDEMPOTENCY_UNAVAILABLE
    assert calls == 1
    assert "uncertain-local" not in manager._local_cache
    assert "uncertain-local" in manager._remote_uncertain
    assert unrelated.persistence == "durable"
    assert manager._redis_ready is True
    assert manager.remote_degraded is False
    assert stages == [("redis_binding", "RuntimeError")]


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("refresh_fault", ["transport", "malformed"])
async def test_refresh_protocol_failure_preserves_local_replay_and_degrades_globally(
    refresh_fault: str,
) -> None:
    redis = _FakeRedis()
    redis.result_write_then_raise = True
    if refresh_fault == "transport":
        redis.binding_refresh_raises = True
    else:
        redis.binding_refresh_response = "invalid"
    manager = _remote_manager(redis)
    payload = {"content": [{"type": "text", "text": "committed-local"}]}
    calls = 0

    async def _execute() -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return payload

    first = await manager.execute("refresh-failure", "args", _execute, policy=_policy())
    replay = await manager.execute("refresh-failure", "args", _execute, policy=_policy())

    assert first.persistence == "local"
    assert replay.from_cache is True
    assert replay.payload == payload
    assert calls == 1
    assert "refresh-failure" in manager._local_cache
    assert "refresh-failure" in manager._remote_uncertain
    assert manager._redis_ready is False
    assert manager.remote_degraded is True


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
    assert result.persistence == "none"
    assert _result_key(key) not in redis.values
    assert stages == [("redis_result_write", "RuntimeError")]


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "semantic_outcome",
    ["binding_lost", "ownership_lost", "result_conflict"],
)
async def test_semantic_result_rejection_is_key_local_and_evicts_stale_replay(
    semantic_outcome: str,
) -> None:
    redis = _FakeRedis()
    key = f"semantic-{semantic_outcome}"
    stages: list[tuple[str, str]] = []
    manager = _remote_manager(
        redis,
        on_degraded=lambda stage, error_type: stages.append((stage, error_type)),
    )
    stale_payload = {"content": [{"type": "text", "text": "stale"}]}
    newer_payload = {"content": [{"type": "text", "text": "newer"}]}
    newer_encoded = canonical_json_bytes(
        newer_payload,
        max_bytes=_policy().max_result_bytes,
    )
    calls = 0

    async def _execute() -> dict[str, Any]:
        nonlocal calls
        calls += 1
        if semantic_outcome == "binding_lost":
            redis.values[_binding_key(key)] = b"new-binding"
        elif semantic_outcome == "ownership_lost":
            redis.values[_lock_key(key)] = b"new-owner-token"
        else:
            redis.values[_result_key(key)] = newer_encoded
        return stale_payload

    result = await manager.execute(key, "args", _execute, policy=_policy())

    with pytest.raises(ExpectedToolFailure) as retry_caught:
        await manager.execute(key, "args", _execute, policy=_policy())

    unrelated = await manager.execute(
        f"unrelated-{semantic_outcome}",
        "other-args",
        lambda: _async_payload({"unrelated": True}),
        policy=_policy(),
    )

    assert result.payload is stale_payload
    assert result.persistence == "none"
    assert retry_caught.value.reason is ExpectedToolFailureReason.IDEMPOTENCY_UNAVAILABLE
    assert calls == 1
    assert key not in manager._local_cache
    assert key in manager._remote_uncertain
    assert unrelated.persistence == "durable"
    assert manager._redis_ready is True
    assert manager.remote_degraded is False
    assert stages[0] == ("redis_result_write", "RuntimeError")
    assert all(stage in {"redis_result_write", "redis_release"} for stage, _ in stages)


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
async def test_oversized_success_cancellation_retains_remote_cleanup_owner() -> None:
    redis = _FakeRedis()
    redis.block_cleanup = True
    manager = _remote_manager(redis)
    key = "oversized-cancel-cleanup"
    payload = {"value": "x" * 1_000}
    calls = 0

    async def _execute() -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return payload

    task = asyncio.create_task(
        manager.execute(
            key,
            "args",
            _execute,
            policy=_policy(max_result_bytes=64),
        )
    )
    await asyncio.wait_for(redis.cleanup_entered.wait(), timeout=0.5)
    task.cancel()
    await asyncio.sleep(0)

    assert task.done() is False
    assert len(manager._finalizers) == 1
    redis.cleanup_release.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=0.5)

    assert calls == 1
    assert _lock_key(key) not in redis.values
    assert manager._finalizers == set()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_uncacheable_success_observer_cancellation_releases_remote_lock() -> None:
    redis = _FakeRedis()
    cancellation = asyncio.CancelledError("degraded observer cancellation")

    def _cancel_degraded_observer(_stage: str, _error_type: str) -> None:
        raise cancellation

    manager = _remote_manager(redis, on_degraded=_cancel_degraded_observer)
    key = "uncacheable-observer-cancel"
    calls = 0

    async def _execute() -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return {"value": "x" * 1_000}

    with pytest.raises(asyncio.CancelledError) as exc_info:
        await manager.execute(
            key,
            "args",
            _execute,
            policy=_policy(max_result_bytes=64),
        )

    assert exc_info.value is cancellation
    assert calls == 1
    assert _lock_key(key) not in redis.values
    assert manager._finalizers == set()


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
async def test_remote_replay_atomically_renews_result_and_binding_ttl() -> None:
    redis = _FakeRedis()
    key = "remote-replay-ttl"
    payload = {"content": [{"type": "text", "text": "durable"}]}
    encoded = canonical_json_bytes(payload, max_bytes=_policy().max_result_bytes)
    redis.values[_binding_key(key)] = b"args"
    redis.values[_result_key(key)] = encoded
    manager = _remote_manager(redis)
    calls = 0

    async def _execute() -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return {"unexpected": True}

    replay = await manager.execute(key, "args", _execute, policy=_policy())

    assert calls == 0
    assert replay.payload == payload
    assert replay.persistence == "durable"
    assert redis.values[_result_key(key)] == encoded
    assert redis.replay_reads == [
        (
            _binding_key(key),
            _result_key(key),
            b"args",
            _policy().ttl_seconds,
        )
    ]
    assert ("get", _result_key(key)) not in redis.calls
    assert redis.expirations[-2:] == [
        (_binding_key(key), _policy().ttl_seconds),
        (_result_key(key), _policy().ttl_seconds),
    ]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_expired_remote_result_during_atomic_read_proceeds_without_degradation() -> None:
    redis = _FakeRedis()
    key = "remote-replay-expired"
    redis.values[_binding_key(key)] = b"args"
    redis.values[_result_key(key)] = canonical_json_bytes(
        {"stale": True},
        max_bytes=_policy().max_result_bytes,
    )
    redis.expire_result_during_replay_read = True
    stages: list[tuple[str, str]] = []
    manager = _remote_manager(
        redis,
        on_degraded=lambda stage, error_type: stages.append((stage, error_type)),
    )
    calls = 0
    original = {"fresh": True}

    async def _execute() -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return original

    result = await manager.execute(key, "args", _execute, policy=_policy())

    assert calls == 1
    assert result.payload is original
    assert result.from_cache is False
    assert result.persistence == "durable"
    assert manager._redis_ready is True
    assert manager.remote_degraded is False
    assert stages == []
    assert ("get", _result_key(key)) not in redis.calls
    assert redis.values[_result_key(key)] == canonical_json_bytes(
        original,
        max_bytes=_policy().max_result_bytes,
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_remote_replay_binding_change_during_ttl_renewal_fails_closed() -> None:
    redis = _FakeRedis()
    key = "remote-replay-binding-change"
    payload = {"content": [{"type": "text", "text": "durable"}]}
    redis.values[_binding_key(key)] = b"args"
    redis.values[_result_key(key)] = canonical_json_bytes(
        payload,
        max_bytes=_policy().max_result_bytes,
    )
    redis.replace_binding_after_result_read = True
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

    with pytest.raises(InvalidParamsException):
        await manager.execute(key, "args", _execute, policy=_policy())

    assert calls == 0
    assert key not in manager._local_cache
    assert manager._redis_ready is True
    assert manager.remote_degraded is False
    assert stages == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_remote_replay_missing_binding_fails_key_only_without_degradation() -> None:
    redis = _FakeRedis()
    key = "remote-replay-binding-missing"
    redis.values[_binding_key(key)] = b"args"
    redis.values[_result_key(key)] = canonical_json_bytes(
        {"content": [{"type": "text", "text": "durable"}]},
        max_bytes=_policy().max_result_bytes,
    )
    redis.remove_binding_during_replay_read = True
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
        await manager.execute(key, "args", _execute, policy=_policy())

    assert caught.value.reason is ExpectedToolFailureReason.IDEMPOTENCY_UNAVAILABLE
    assert calls == 0
    assert key not in manager._local_cache
    assert key in manager._remote_uncertain
    assert manager._redis_ready is True
    assert manager.remote_degraded is False
    assert stages == []


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
async def test_concurrent_execute_waits_for_shared_redis_initialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    idempotency_module = importlib.import_module(
        "tldw_Server_API.app.core.MCP_unified.tool_execution.idempotency",
    )
    redis = _FakeRedis()
    factory_started = asyncio.Event()
    release_factory = asyncio.Event()
    callback_started = asyncio.Event()
    factory_calls = 0
    callback_calls = 0

    class _Config:
        def get_redis_connection_params(self) -> dict[str, str]:
            return {"url": "redis://localhost:6379/0"}

    async def _factory(**_kwargs: Any) -> _FakeRedis:
        nonlocal factory_calls
        factory_calls += 1
        factory_started.set()
        await release_factory.wait()
        return redis

    async def _execute() -> dict[str, Any]:
        nonlocal callback_calls
        callback_calls += 1
        callback_started.set()
        return {"call": callback_calls}

    monkeypatch.setattr(idempotency_module, "get_config", lambda: _Config())
    manager = IdempotencyManager(redis_client_factory=_factory)
    first = asyncio.create_task(manager.execute("shared-init", "args", _execute, policy=_policy()))
    await asyncio.wait_for(factory_started.wait(), timeout=1)
    second = asyncio.create_task(manager.execute("shared-init", "args", _execute, policy=_policy()))
    try:
        await asyncio.wait_for(callback_started.wait(), timeout=0.1)
        callback_ran_before_factory = True
    except TimeoutError:
        callback_ran_before_factory = False

    release_factory.set()
    first_result, second_result = await asyncio.gather(first, second)

    assert callback_ran_before_factory is False
    assert factory_calls == 1
    assert callback_calls == 1
    assert first_result.payload == second_result.payload == {"call": 1}
    assert _result_key("shared-init") in redis.values


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
async def test_degraded_observer_cancellation_propagates_exact_instance() -> None:
    cancellation = asyncio.CancelledError("observer cancellation")

    def _cancelling_observer(_stage: str, _error_type: str) -> None:
        raise cancellation

    manager = _local_manager(on_degraded=_cancelling_observer)
    messages: list[str] = []
    sink_id = logger.add(lambda message: messages.append(str(message)), format="{message}")
    calls = 0

    async def _execute() -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return {"value": object()}

    try:
        with pytest.raises(asyncio.CancelledError) as caught:
            await manager.execute(
                "degraded-observer-cancellation",
                "args",
                _execute,
                policy=_policy(),
            )
    finally:
        logger.remove(sink_id)

    assert caught.value is cancellation
    assert calls == 1
    assert "degraded-observer-cancellation" not in manager._local_cache
    assert "degraded-observer-cancellation" in manager._local_bindings
    assert all("degraded observer failed" not in message for message in messages)


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


@pytest.mark.unit
@pytest.mark.asyncio
async def test_hostile_exception_metaclass_cannot_replace_degraded_outcome() -> None:
    class _HostileExceptionMeta(type):
        def __getattribute__(cls, name: str) -> Any:
            if name == "__name__":
                raise RuntimeError("metaclass credential=TOP_SECRET")
            return super().__getattribute__(name)

    class _HostileObserverError(BaseException, metaclass=_HostileExceptionMeta):
        pass

    def _hostile_observer(_stage: str, _error_type: str) -> None:
        raise _HostileObserverError()

    messages: list[str] = []
    sink_id = logger.add(lambda message: messages.append(str(message)), format="{message}")
    original = {"value": object()}
    try:
        try:
            result = await _local_manager(on_degraded=_hostile_observer).execute(
                "hostile-observer-metaclass",
                "args",
                lambda: _async_payload(original),
                policy=_policy(),
            )
        except RuntimeError:
            raise AssertionError("hostile metaclass replaced the degraded outcome") from None
    finally:
        logger.remove(sink_id)

    assert result.payload is original
    assert result.persistence == "none"
    assert all("TOP_SECRET" not in message for message in messages)
    assert any(
        "degraded observer failed error_type=Exception" in message
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
async def test_caller_cancellation_waits_for_exact_remote_commit() -> None:
    redis = _BlockingResultRedis()
    manager = _remote_manager(redis)
    payload = {"content": [{"type": "text", "text": "exact-result"}]}
    expected = canonical_json_bytes(payload, max_bytes=_policy().max_result_bytes)
    calls = 0

    async def _execute() -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return payload

    task = asyncio.create_task(
        manager.execute("cancel-after-local", "args", _execute, policy=_policy())
    )
    await asyncio.wait_for(redis.write_started.wait(), timeout=0.5)
    task.cancel()
    await asyncio.sleep(0)
    cancellation_was_deferred = not task.done()
    redis.allow_write.set()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=0.5)

    replay = await manager.execute(
        "cancel-after-local",
        "args",
        _execute,
        policy=_policy(),
    )
    assert cancellation_was_deferred is True
    assert redis.values[_result_key("cancel-after-local")] == expected
    assert replay.payload == payload
    assert replay.from_cache is True
    assert calls == 1
    assert manager._finalizers == set()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_remote_finalize_timeout_keeps_local_replay_and_drains_cancelled_task() -> None:
    redis = _BlockingResultRedis()
    stages: list[tuple[str, str]] = []
    manager = _remote_manager(
        redis,
        on_degraded=lambda stage, error_type: stages.append((stage, error_type)),
    )
    payload = {"content": [{"type": "text", "text": "local-after-timeout"}]}
    calls = 0

    async def _execute() -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return payload

    result = await asyncio.wait_for(
        manager.execute("finalize-timeout", "args", _execute, policy=_policy()),
        timeout=2.5,
    )
    replay = await manager.execute(
        "finalize-timeout",
        "args",
        _execute,
        policy=_policy(),
    )

    assert result.payload is payload
    assert result.persistence == "local"
    assert replay.payload == payload
    assert replay.from_cache is True
    assert calls == 1
    assert redis.write_cancellations == 1
    assert manager._finalizers == set()
    assert manager.remote_degraded is True
    assert stages == [("finalize_timeout", "TimeoutError")]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_cancellation_resistant_finalizer_stays_owned_and_shutdown_drains_exact_bytes() -> None:
    redis = _BlockingResultRedis(ignore_cancellation=True)
    stages: list[tuple[str, str]] = []
    messages: list[str] = []
    manager = _remote_manager(
        redis,
        on_degraded=lambda stage, error_type: stages.append((stage, error_type)),
    )
    payload = {"content": [{"type": "text", "text": "captured-before-mutation"}]}
    expected = canonical_json_bytes(payload, max_bytes=_policy().max_result_bytes)

    sink_id = logger.add(lambda message: messages.append(str(message)), format="{message}")
    task = asyncio.create_task(
        manager.execute(
            "stuck-finalizer-private-key",
            "args",
            lambda: _async_payload(payload),
            policy=_policy(),
        )
    )
    try:
        done, _pending = await asyncio.wait({task}, timeout=3.0)
        returned_before_release = task in done
        if not returned_before_release:
            redis.allow_write.set()
            await task

        result = task.result()
        payload["content"][0]["text"] = "mutated-after-return"
        retained_before_shutdown = len(manager._finalizers)
        shutdown_task = asyncio.create_task(manager.shutdown())
        await asyncio.sleep(0)
        redis.allow_write.set()
        await asyncio.wait_for(shutdown_task, timeout=1.0)
    finally:
        redis.allow_write.set()
        await asyncio.gather(task, return_exceptions=True)
        logger.remove(sink_id)

    assert returned_before_release is True
    assert result.persistence == "local"
    assert retained_before_shutdown == 1
    assert redis.values[_result_key("stuck-finalizer-private-key")] == expected
    assert manager._finalizers == set()
    assert manager.remote_degraded is True
    assert stages == [
        ("finalize_timeout", "TimeoutError"),
        ("finalizer_stuck", "TimeoutError"),
    ]
    assert all("stuck-finalizer-private-key" not in message for message in messages)
    assert all("captured-before-mutation" not in message for message in messages)
    assert all(
        "stage=finalize_timeout error_type=TimeoutError" in message
        or "stage=finalizer_stuck error_type=TimeoutError" in message
        for message in messages
        if "MCP idempotency degraded stage=" in message
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_stale_finalizer_cannot_overwrite_new_owner_payload() -> None:
    redis = _BlockingResultRedis(ignore_cancellation=True)
    manager = _remote_manager(redis)
    key = "stale-finalizer-owner"
    stale_payload = {"content": [{"type": "text", "text": "stale"}]}
    newer_payload = {"content": [{"type": "text", "text": "newer"}]}
    newer_encoded = canonical_json_bytes(
        newer_payload,
        max_bytes=_policy().max_result_bytes,
    )

    execution = asyncio.create_task(
        manager.execute(
            key,
            "args",
            lambda: _async_payload(stale_payload),
            policy=_policy(),
        )
    )
    await asyncio.wait_for(redis.write_started.wait(), timeout=0.5)
    result = await asyncio.wait_for(execution, timeout=2.5)

    assert result.payload is stale_payload
    assert result.persistence == "local"
    assert len(manager._finalizers) == 1

    redis.values[_lock_key(key)] = b"new-owner-token"
    redis.values[_result_key(key)] = newer_encoded
    redis.allow_write.set()
    await asyncio.wait_for(manager.shutdown(), timeout=1.0)

    assert redis.values[_result_key(key)] == newer_encoded
    assert redis.values[_lock_key(key)] == b"new-owner-token"
    assert manager._finalizers == set()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_normal_manager_shutdown_leaves_no_finalizer_pending() -> None:
    manager = _remote_manager(_FakeRedis())

    await manager.execute(
        "normal-shutdown",
        "args",
        lambda: _async_payload({"ok": True}),
        policy=_policy(),
    )
    await manager.shutdown()

    assert manager._finalizers == set()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_shutdown_waits_for_admitted_execution_to_create_and_drain_finalizer() -> None:
    redis = _BlockingResultRedis()
    manager = _remote_manager(redis)
    callback_started = asyncio.Event()
    allow_callback = asyncio.Event()
    original_payload = {"content": [{"type": "text", "text": "original"}]}
    later_callback_calls = 0

    async def _execute_original() -> dict[str, Any]:
        callback_started.set()
        await allow_callback.wait()
        return original_payload

    async def _execute_later() -> dict[str, Any]:
        nonlocal later_callback_calls
        later_callback_calls += 1
        return {"unexpected": True}

    execution = asyncio.create_task(
        manager.execute("shutdown-race", "args", _execute_original, policy=_policy())
    )
    await asyncio.wait_for(callback_started.wait(), timeout=0.5)
    shutdown = asyncio.create_task(manager.shutdown())
    await asyncio.sleep(0)
    shutdown_returned_before_creator = shutdown.done()
    later_execution = asyncio.create_task(
        manager.execute("after-closing", "args", _execute_later, policy=_policy())
    )
    await asyncio.sleep(0)

    allow_callback.set()
    await asyncio.wait_for(redis.write_started.wait(), timeout=0.5)
    redis.allow_write.set()
    execution_result, later_result = await asyncio.gather(
        execution,
        later_execution,
        return_exceptions=True,
    )
    await asyncio.wait_for(shutdown, timeout=1.0)
    await asyncio.sleep(0)

    assert shutdown_returned_before_creator is False
    assert isinstance(execution_result, IdempotencyRunResult)
    assert execution_result.payload is original_payload
    assert isinstance(later_result, ExpectedToolFailure)
    assert later_result.reason is ExpectedToolFailureReason.IDEMPOTENCY_UNAVAILABLE
    assert later_callback_calls == 0
    assert manager._inflight_executions == 0
    assert manager._finalizers == set()

    with pytest.raises(ExpectedToolFailure) as post_shutdown:
        await manager.execute("after-return", "args", _execute_later, policy=_policy())
    assert post_shutdown.value.reason is ExpectedToolFailureReason.IDEMPOTENCY_UNAVAILABLE
    assert later_callback_calls == 0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_shutdown_waits_for_late_admitted_success_to_persist_and_release() -> None:
    redis = _FakeRedis()
    manager = _remote_manager(redis)
    callback_started = asyncio.Event()
    allow_callback = asyncio.Event()
    payload = {"content": [{"type": "text", "text": "late-success"}]}
    key = "late-owner-shutdown"
    manager._finalize_bound = lambda _policy: 0.01

    async def _execute() -> dict[str, Any]:
        callback_started.set()
        await allow_callback.wait()
        return payload

    execution = asyncio.create_task(
        manager.execute(key, "args", _execute, policy=_policy())
    )
    await asyncio.wait_for(callback_started.wait(), timeout=0.5)
    shutdown = asyncio.create_task(manager.shutdown())

    try:
        await asyncio.sleep(0.03)
        assert shutdown.done() is False

        allow_callback.set()
        result = await asyncio.wait_for(execution, timeout=0.5)
        await asyncio.wait_for(shutdown, timeout=0.5)

        assert result.payload is payload
        assert result.persistence == "durable"
        assert redis.values[_result_key(key)] == canonical_json_bytes(
            payload,
            max_bytes=_policy().max_result_bytes,
        )
        assert _lock_key(key) not in redis.values
        assert redis.result_store_calls == 1
        assert redis.lock_release_calls == 1
        assert manager._inflight_executions == 0
        assert manager._finalizers == set()
    finally:
        allow_callback.set()
        await asyncio.gather(execution, shutdown, return_exceptions=True)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_shutdown_retains_late_execution_past_hard_deadline_until_durable() -> None:
    redis = _FakeRedis()
    stages: list[tuple[str, str]] = []
    manager = _remote_manager(
        redis,
        on_degraded=lambda stage, error_type: stages.append((stage, error_type)),
    )
    manager._shutdown_execution_bound = lambda: 0.01
    callback_started = asyncio.Event()
    release_callback = asyncio.Event()
    payload = {"content": [{"type": "text", "text": "retained-success"}]}
    key = "retained-owner-shutdown"
    callback_calls = 0
    callback_cancellations = 0

    async def _execute() -> dict[str, Any]:
        nonlocal callback_calls, callback_cancellations
        callback_calls += 1
        callback_started.set()
        while not release_callback.is_set():
            try:
                await release_callback.wait()
            except asyncio.CancelledError:
                callback_cancellations += 1
        return payload

    execution = asyncio.create_task(
        manager.execute(key, "args", _execute, policy=_policy())
    )
    await asyncio.wait_for(callback_started.wait(), timeout=0.5)
    shutdown = asyncio.create_task(manager.shutdown())

    try:
        await asyncio.wait_for(asyncio.shield(shutdown), timeout=0.5)

        assert execution.done() is False
        assert manager._admitted_execution_tasks == {execution: 1}
        assert manager._finalizers == set()
        assert manager._redis_ready is True
        assert manager.remote_degraded is False
        assert stages == [("shutdown_execution_timeout", "TimeoutError")]

        release_callback.set()
        result = await asyncio.wait_for(execution, timeout=0.5)
        await asyncio.sleep(0)

        replay_calls = 0

        async def _unexpected_replay_callback() -> dict[str, Any]:
            nonlocal replay_calls
            replay_calls += 1
            return {"unexpected": True}

        replay = await _remote_manager(redis).execute(
            key,
            "args",
            _unexpected_replay_callback,
            policy=_policy(),
        )

        assert result.payload is payload
        assert result.persistence == "durable"
        assert replay.from_cache is True
        assert replay.persistence == "durable"
        assert replay.payload == payload
        assert callback_calls == 1
        assert callback_cancellations == 0
        assert replay_calls == 0
        assert redis.values[_result_key(key)] == canonical_json_bytes(
            payload,
            max_bytes=_policy().max_result_bytes,
        )
        assert _lock_key(key) not in redis.values
        assert redis.result_store_calls == 1
        assert redis.lock_release_calls == 1
        assert manager._admitted_execution_tasks == {}
        assert manager._finalizers == set()
        assert manager._redis_ready is True
        assert manager.remote_degraded is False
    finally:
        release_callback.set()
        await asyncio.gather(execution, shutdown, return_exceptions=True)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_nested_same_task_execution_tracking_uses_refcounts() -> None:
    manager = _local_manager()
    observed_refcounts: list[int] = []

    async def _execute_inner() -> dict[str, Any]:
        task = asyncio.current_task()
        assert task is not None
        observed_refcounts.append(manager._admitted_execution_tasks[task])
        return {"inner": True}

    async def _execute_outer() -> dict[str, Any]:
        task = asyncio.current_task()
        assert task is not None
        observed_refcounts.append(manager._admitted_execution_tasks[task])
        await manager.execute("nested-inner", "inner-args", _execute_inner, policy=_policy())
        observed_refcounts.append(manager._admitted_execution_tasks[task])
        return {"outer": True}

    await manager.execute("nested-outer", "outer-args", _execute_outer, policy=_policy())

    assert observed_refcounts == [1, 2, 1]
    assert manager._admitted_execution_tasks == {}
    assert manager._inflight_executions == 0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_public_server_shutdown_is_single_flight_for_staggered_callers() -> None:
    redis = _BlockingResultRedis(ignore_cancellation=True)
    server = MCPServer()
    manager = server.protocol._idempotency
    manager._redis_client = redis
    manager._redis_attempted = True
    manager._redis_ready = True
    manager._finalize_bound = lambda _policy: 0.01
    payload = {"content": [{"type": "text", "text": "shared-shutdown"}]}
    expected = canonical_json_bytes(payload, max_bytes=_policy().max_result_bytes)
    key = "shared-resource-shutdown"
    protocol_started = asyncio.Event()
    modules_stopped = asyncio.Event()
    protocol_shutdown_calls = 0
    module_shutdown_calls = 0

    original_protocol_shutdown = server.protocol.shutdown

    async def _shutdown_protocol() -> None:
        nonlocal protocol_shutdown_calls
        protocol_shutdown_calls += 1
        protocol_started.set()
        await original_protocol_shutdown()

    async def _shutdown_modules() -> None:
        nonlocal module_shutdown_calls
        module_shutdown_calls += 1
        modules_stopped.set()

    server.protocol.shutdown = _shutdown_protocol
    server.module_registry.shutdown_all = _shutdown_modules
    execution = asyncio.create_task(
        manager.execute(
            key,
            "args",
            lambda: _async_payload(payload),
            policy=_policy(),
        )
    )
    await asyncio.wait_for(redis.write_started.wait(), timeout=0.5)
    result = await asyncio.wait_for(execution, timeout=0.5)
    cancellation_count_before_shutdown = redis.write_cancellations
    first = asyncio.create_task(server.shutdown())
    await asyncio.wait_for(protocol_started.wait(), timeout=0.5)
    second = asyncio.create_task(server.shutdown())
    first.cancel("first staggered cancellation")
    await asyncio.sleep(0)
    first_survived_cancellation = not first.done()

    try:
        with pytest.raises(asyncio.CancelledError) as caught:
            await asyncio.wait_for(first, timeout=0.5)
        await asyncio.wait_for(second, timeout=0.5)

        assert caught.value.args == ("first staggered cancellation",)
        assert first_survived_cancellation is True
        assert protocol_shutdown_calls == 1
        assert redis.write_cancellations == cancellation_count_before_shutdown + 1
        assert modules_stopped.is_set() is False
        resource_task = server._resource_shutdown_task
        assert resource_task is not None
        assert resource_task.done() is True

        redis.allow_write.set()
        await asyncio.wait_for(modules_stopped.wait(), timeout=0.5)
        await asyncio.sleep(0)

        assert result.payload is payload
        assert result.persistence == "local"
        assert redis.values[_result_key(key)] == expected
        assert _lock_key(key) not in redis.values
        assert redis.result_store_calls == 1
        assert redis.lock_release_calls == 1
        assert module_shutdown_calls == 1
        assert manager._admitted_execution_tasks == {}
        assert manager._finalizers == set()
        assert server._module_shutdown_task is None

        await server.shutdown()
        assert server._resource_shutdown_task is resource_task
        assert protocol_shutdown_calls == 1
        assert module_shutdown_calls == 1
    finally:
        redis.allow_write.set()
        await asyncio.gather(execution, first, second, return_exceptions=True)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_server_defers_single_module_shutdown_until_retained_owner_finishes() -> None:
    redis = _FakeRedis()
    server = MCPServer()
    manager = server.protocol._idempotency
    manager._redis_client = redis
    manager._redis_attempted = True
    manager._redis_ready = True
    manager._shutdown_execution_bound = lambda: 0.01
    callback_started = asyncio.Event()
    release_callback = asyncio.Event()
    modules_stopped = asyncio.Event()
    payload = {"content": [{"type": "text", "text": "server-retained-success"}]}
    key = "server-retained-owner"
    callback_calls = 0
    module_shutdown_calls = 0

    async def _execute() -> dict[str, Any]:
        nonlocal callback_calls
        callback_calls += 1
        if modules_stopped.is_set():
            raise RuntimeError("module teardown overtook callback")
        callback_started.set()
        await release_callback.wait()
        if modules_stopped.is_set():
            raise RuntimeError("module teardown overtook callback")
        return payload

    async def _shutdown_modules() -> None:
        nonlocal module_shutdown_calls
        module_shutdown_calls += 1
        modules_stopped.set()

    server.module_registry.shutdown_all = _shutdown_modules
    execution = asyncio.create_task(
        manager.execute(key, "args", _execute, policy=_policy())
    )
    await asyncio.wait_for(callback_started.wait(), timeout=0.5)
    shutdown_one = asyncio.create_task(server.shutdown())
    shutdown_two = asyncio.create_task(server.shutdown())

    try:
        await asyncio.wait_for(
            asyncio.gather(shutdown_one, shutdown_two),
            timeout=0.5,
        )

        assert modules_stopped.is_set() is False
        assert module_shutdown_calls == 0
        deferred_task = server._module_shutdown_task
        assert deferred_task is not None
        assert deferred_task.done() is False
        assert manager.has_pending_shutdown_work is True

        release_callback.set()
        result = await asyncio.wait_for(execution, timeout=0.5)
        await asyncio.wait_for(modules_stopped.wait(), timeout=0.5)
        await asyncio.sleep(0)

        assert result.payload is payload
        assert result.persistence == "durable"
        assert callback_calls == 1
        assert module_shutdown_calls == 1
        assert redis.values[_result_key(key)] == canonical_json_bytes(
            payload,
            max_bytes=_policy().max_result_bytes,
        )
        assert _lock_key(key) not in redis.values
        assert redis.result_store_calls == 1
        assert redis.lock_release_calls == 1
        assert manager.has_pending_shutdown_work is False
        assert manager._admitted_execution_tasks == {}
        assert manager._finalizers == set()
        assert deferred_task.done() is True
        assert server._module_shutdown_task is None
        assert server._module_shutdown_complete is True
        assert server.initialized is False
    finally:
        release_callback.set()
        await asyncio.gather(
            execution,
            shutdown_one,
            shutdown_two,
            return_exceptions=True,
        )
