"""Bounded idempotent execution with sticky local or Redis ownership."""

from __future__ import annotations

import asyncio
import copy
import secrets
import threading
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Literal, cast

from loguru import logger

from ..config import get_config
from ..execution_outcomes import ExpectedToolFailure, ExpectedToolFailureReason
from ..protocol_types import InvalidParamsException
from .canonical import (
    IDEMPOTENCY_RESULT_HARD_MAX_BYTES,
    CanonicalJsonTooLarge,
    JsonValue,
    canonical_json_bytes,
    decode_canonical_json_object,
)
from .models import IdempotencyExecutionPolicy, IdempotencyRunResult

try:  # pragma: no cover - optional dependency
    from redis.exceptions import RedisError
except ImportError:  # pragma: no cover - redis not installed

    class RedisError(Exception):
        """Fallback RedisError when redis-py is unavailable."""


_DEGRADED_STAGES = frozenset(
    {
        "serialization",
        "result_size",
        "local_commit",
        "redis_connect",
        "redis_binding",
        "redis_result_read",
        "redis_lock_acquire",
        "redis_result_write",
        "redis_release",
        "finalize_timeout",
        "finalizer_stuck",
    }
)
_REDIS_POLL_INTERVAL_SECONDS = 0.05
_REDIS_BIND_ARGUMENTS_SCRIPT = """
local binding = redis.call('GET', KEYS[1])
if binding then
    if binding ~= ARGV[1] then return -1 end
    redis.call('EXPIRE', KEYS[1], ARGV[2])
    return 1
end
if redis.call('EXISTS', KEYS[2]) == 1 then return -2 end
redis.call('SET', KEYS[1], ARGV[1], 'EX', ARGV[2])
return 2
"""
_REDIS_REFRESH_BINDING_SCRIPT = """
if redis.call('GET', KEYS[1]) ~= ARGV[1] then return 0 end
redis.call('EXPIRE', KEYS[1], ARGV[2])
return 1
"""
_REDIS_STORE_RESULT_SCRIPT = """
if redis.call('GET', KEYS[1]) ~= ARGV[1] then return 0 end
redis.call('SET', KEYS[2], ARGV[2], 'EX', ARGV[3])
redis.call('EXPIRE', KEYS[1], ARGV[3])
return 1
"""


async def _no_redis_client_factory(**_kwargs: Any) -> None:
    """Fallback Redis factory used when embedders do not provide Redis support."""
    return None


def _noop_degraded(_stage: str, _error_type: str) -> None:
    """Default degradation observer."""


def _safe_error_type(exc: BaseException) -> str:
    name = exc.__class__.__name__
    if (
        1 <= len(name) <= 64
        and name.isascii()
        and (name[0].isalpha() or name[0] == "_")
        and all(character.isalnum() or character == "_" for character in name)
    ):
        return name
    return "Exception"


@dataclass(frozen=True, slots=True)
class _LocalReplayEntry:
    timestamp: float
    arguments_hash: str
    template: dict[str, JsonValue]
    canonical_bytes: bytes


@dataclass(frozen=True, slots=True)
class _RemoteUncertainEntry:
    expires_at: float


class _PreOwnerRedisFailure(Exception):
    """Legacy private sentinel retained only to prove it cannot select fallback."""


@dataclass(frozen=True, slots=True)
class _RemoteReplayRead:
    replay: IdempotencyRunResult | None
    remote_available: bool = True
    local_committed: bool = False


@dataclass(frozen=True, slots=True)
class _RemotePreOwnerState:
    client: Any
    result_key: str
    binding_key: str
    lock_key: str
    replay: IdempotencyRunResult | None


class IdempotencyManager:
    """Execute a callback once per bounded idempotency ownership attempt."""

    def __init__(
        self,
        redis_client_factory: Callable[..., Any] | None = None,
        *,
        on_degraded: Callable[[str, str], None] | None = None,
    ) -> None:
        self._local_cache: OrderedDict[str, _LocalReplayEntry] = OrderedDict()
        self._local_bindings: OrderedDict[str, tuple[float, str]] = OrderedDict()
        self._remote_uncertain: OrderedDict[str, _RemoteUncertainEntry] = OrderedDict()
        self._local_locks: dict[str, asyncio.Lock] = {}
        self._local_lock_users: dict[str, int] = {}
        self._local_guard = threading.RLock()
        self._redis_client: Any | None = None
        self._redis_ready = False
        self._redis_attempted = False
        self._redis_guard = asyncio.Lock()
        self._redis_client_factory = redis_client_factory or _no_redis_client_factory
        self._on_degraded = on_degraded or _noop_degraded
        self._remote_degraded = False

    @property
    def remote_degraded(self) -> bool:
        """Whether a configured remote idempotency backend has degraded."""
        with self._local_guard:
            return self._remote_degraded

    def _record_degraded(self, stage: str, exc: BaseException, *, remote: bool) -> None:
        if stage not in _DEGRADED_STAGES:  # pragma: no cover - internal invariant
            stage = "serialization"
        error_type = _safe_error_type(exc)
        if remote:
            with self._local_guard:
                self._remote_degraded = True
        logger.warning(
            "MCP idempotency degraded stage={stage} error_type={error_type}",
            stage=stage,
            error_type=error_type,
        )
        try:
            self._on_degraded(stage, error_type)
        except Exception as observer_exc:  # noqa: BLE001 - metrics cannot replace execution outcomes.
            logger.debug(
                "MCP idempotency degraded observer failed error_type={error_type}",
                error_type=observer_exc.__class__.__name__,
            )

    def _mark_remote_failure(self, stage: str, exc: BaseException) -> None:
        self._redis_ready = False
        self._record_degraded(stage, exc, remote=True)

    async def _ensure_redis(self) -> bool:
        if self._redis_attempted:
            return self._redis_ready
        async with self._redis_guard:
            if self._redis_attempted:
                return self._redis_ready
            self._redis_attempted = True
            try:
                params = dict(get_config().get_redis_connection_params() or {})
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - backend discovery is an explicit fallback phase.
                self._mark_remote_failure("redis_connect", exc)
                return False
            if not params:
                self._redis_ready = False
                return False

            url = params.pop("url", None)
            try:
                client = await self._redis_client_factory(
                    preferred_url=url,
                    decode_responses=False,
                    fallback_to_fake=False,
                    context="mcp_idempotency",
                    redis_kwargs=params,
                )
                if client is None:
                    raise RuntimeError("Redis client unavailable")
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - no remote ownership was attempted.
                self._redis_client = None
                self._mark_remote_failure("redis_connect", exc)
                return False

            self._redis_client = client
            self._redis_ready = True
            return True

    @staticmethod
    def _expired(timestamp: float, ttl_seconds: int, now: float) -> bool:
        return now - timestamp > ttl_seconds

    def _prune_local_locks_locked(self) -> None:
        active_keys = set(self._local_cache) | set(self._local_bindings)
        for key, lock in list(self._local_locks.items()):
            if (
                key not in active_keys
                and self._local_lock_users.get(key, 0) == 0
                and not lock.locked()
            ):
                del self._local_locks[key]

    def _get_local_replay_locked(
        self,
        cache_key: str,
        arguments_hash: str,
        *,
        ttl_seconds: int,
    ) -> IdempotencyRunResult | None:
        entry = self._local_cache.get(cache_key)
        if entry is None:
            return None
        if self._expired(entry.timestamp, ttl_seconds, time.monotonic()):
            del self._local_cache[cache_key]
            self._prune_local_locks_locked()
            return None
        if entry.arguments_hash != arguments_hash:
            raise InvalidParamsException(
                "Idempotency key was already used with different arguments"
            )
        self._local_cache.move_to_end(cache_key)
        return IdempotencyRunResult(
            payload=copy.deepcopy(entry.template),
            from_cache=True,
            persistence="local",
        )

    def _check_local_binding_locked(
        self,
        cache_key: str,
        arguments_hash: str,
        *,
        ttl_seconds: int,
    ) -> None:
        binding = self._local_bindings.get(cache_key)
        if binding is None:
            return
        timestamp, bound_hash = binding
        if self._expired(timestamp, ttl_seconds, time.monotonic()):
            del self._local_bindings[cache_key]
            self._prune_local_locks_locked()
            return
        if bound_hash != arguments_hash:
            raise InvalidParamsException(
                "Idempotency key was already used with different arguments"
            )
        self._local_bindings.move_to_end(cache_key)

    def _check_remote_uncertain_locked(self, cache_key: str) -> None:
        entry = self._remote_uncertain.get(cache_key)
        if entry is None:
            return
        if time.monotonic() >= entry.expires_at:
            del self._remote_uncertain[cache_key]
            return
        self._remote_uncertain.move_to_end(cache_key)
        raise ExpectedToolFailure(ExpectedToolFailureReason.IDEMPOTENCY_UNAVAILABLE)

    def _put_remote_uncertain_locked(
        self,
        cache_key: str,
        *,
        policy: IdempotencyExecutionPolicy,
    ) -> None:
        retention_seconds = max(policy.ttl_seconds, policy.lock_ttl_seconds)
        self._remote_uncertain[cache_key] = _RemoteUncertainEntry(
            expires_at=time.monotonic() + retention_seconds,
        )
        self._remote_uncertain.move_to_end(cache_key)
        while len(self._remote_uncertain) > max(1, policy.max_entries):
            self._remote_uncertain.popitem(last=False)

    def _block_remote_key(
        self,
        cache_key: str,
        *,
        policy: IdempotencyExecutionPolicy,
    ) -> None:
        with self._local_guard:
            self._put_remote_uncertain_locked(cache_key, policy=policy)

    def _put_local_binding_locked(
        self,
        cache_key: str,
        arguments_hash: str,
        *,
        ttl_seconds: int,
        max_entries: int,
    ) -> None:
        now = time.monotonic()
        self._local_bindings[cache_key] = (now, arguments_hash)
        self._local_bindings.move_to_end(cache_key)
        for key, (timestamp, _) in list(self._local_bindings.items()):
            if self._expired(timestamp, ttl_seconds, now):
                del self._local_bindings[key]
        while len(self._local_bindings) > max_entries:
            self._local_bindings.popitem(last=False)
        self._prune_local_locks_locked()

    def _refresh_local_binding(
        self,
        cache_key: str,
        arguments_hash: str,
        *,
        policy: IdempotencyExecutionPolicy,
    ) -> None:
        with self._local_guard:
            self._put_local_binding_locked(
                cache_key,
                arguments_hash,
                ttl_seconds=policy.ttl_seconds,
                max_entries=policy.max_entries,
            )

    def _put_local_replay_locked(
        self,
        cache_key: str,
        arguments_hash: str,
        template: dict[str, JsonValue],
        canonical_bytes: bytes,
        *,
        ttl_seconds: int,
        max_entries: int,
    ) -> None:
        now = time.monotonic()
        self._local_cache[cache_key] = _LocalReplayEntry(
            timestamp=now,
            arguments_hash=arguments_hash,
            template=template,
            canonical_bytes=canonical_bytes,
        )
        self._local_cache.move_to_end(cache_key)
        for key, entry in list(self._local_cache.items()):
            if self._expired(entry.timestamp, ttl_seconds, now):
                del self._local_cache[key]
        while len(self._local_cache) > max_entries:
            self._local_cache.popitem(last=False)
        self._prune_local_locks_locked()

    def _try_commit_local_replay(
        self,
        cache_key: str,
        arguments_hash: str,
        template: dict[str, JsonValue],
        canonical_bytes: bytes,
        *,
        policy: IdempotencyExecutionPolicy,
        include_binding: bool = False,
    ) -> bool:
        try:
            with self._local_guard:
                if include_binding:
                    self._put_local_binding_locked(
                        cache_key,
                        arguments_hash,
                        ttl_seconds=policy.ttl_seconds,
                        max_entries=policy.max_entries,
                    )
                self._put_local_replay_locked(
                    cache_key,
                    arguments_hash,
                    template,
                    canonical_bytes,
                    ttl_seconds=policy.ttl_seconds,
                    max_entries=policy.max_entries,
                )
        except Exception as exc:  # noqa: BLE001 - a valid callback result stays authoritative.
            self._record_degraded("local_commit", exc, remote=False)
            return False
        return True

    def _get_local_lock(self, cache_key: str) -> asyncio.Lock:
        with self._local_guard:
            lock = self._local_locks.get(cache_key)
            if lock is None:
                lock = asyncio.Lock()
                self._local_locks[cache_key] = lock
            self._local_lock_users[cache_key] = self._local_lock_users.get(cache_key, 0) + 1
            return lock

    def _release_local_lock_reference(self, cache_key: str) -> None:
        with self._local_guard:
            users = self._local_lock_users.get(cache_key, 0)
            if users <= 1:
                self._local_lock_users.pop(cache_key, None)
            else:
                self._local_lock_users[cache_key] = users - 1
            self._prune_local_locks_locked()

    @staticmethod
    def _remaining(deadline: float) -> float:
        return deadline - asyncio.get_running_loop().time()

    async def _call_before_deadline(
        self,
        operation: Callable[[], Awaitable[Any]],
        deadline: float,
    ) -> Any:
        remaining = self._remaining(deadline)
        if remaining <= 0:
            raise TimeoutError
        return await asyncio.wait_for(operation(), timeout=remaining)

    async def _acquire_local_lock(self, lock: asyncio.Lock, deadline: float) -> None:
        remaining = self._remaining(deadline)
        if remaining <= 0:
            raise ExpectedToolFailure(ExpectedToolFailureReason.IDEMPOTENCY_IN_PROGRESS)
        try:
            await asyncio.wait_for(lock.acquire(), timeout=remaining)
        except TimeoutError as exc:
            raise ExpectedToolFailure(
                ExpectedToolFailureReason.IDEMPOTENCY_IN_PROGRESS
            ) from exc

    @staticmethod
    def _result_limit(policy: IdempotencyExecutionPolicy) -> int:
        configured = policy.max_result_bytes
        if type(configured) is not int or configured < 1:
            return 1
        return min(configured, IDEMPOTENCY_RESULT_HARD_MAX_BYTES)

    def _canonicalize_success(
        self,
        payload: dict[str, Any],
        *,
        policy: IdempotencyExecutionPolicy,
    ) -> tuple[bytes, dict[str, JsonValue]] | None:
        max_bytes = self._result_limit(policy)
        try:
            encoded = canonical_json_bytes(
                cast(JsonValue, payload),
                max_bytes=max_bytes,
            )
            template = decode_canonical_json_object(encoded, max_bytes=max_bytes)
        except CanonicalJsonTooLarge as exc:
            self._record_degraded("result_size", exc, remote=False)
            return None
        except Exception as exc:  # noqa: BLE001 - valid success remains the caller outcome.
            self._record_degraded("serialization", exc, remote=False)
            return None
        return encoded, template

    async def _execute_local(
        self,
        cache_key: str,
        arguments_hash: str,
        execute_fn: Callable[[], Awaitable[dict[str, Any]]],
        *,
        policy: IdempotencyExecutionPolicy,
        deadline: float,
    ) -> IdempotencyRunResult:
        lock = self._get_local_lock(cache_key)
        acquired = False
        try:
            await self._acquire_local_lock(lock, deadline)
            acquired = True
            with self._local_guard:
                replay = self._get_local_replay_locked(
                    cache_key,
                    arguments_hash,
                    ttl_seconds=policy.ttl_seconds,
                )
                if replay is not None:
                    return replay
                self._check_local_binding_locked(
                    cache_key,
                    arguments_hash,
                    ttl_seconds=policy.ttl_seconds,
                )
                self._put_local_binding_locked(
                    cache_key,
                    arguments_hash,
                    ttl_seconds=policy.ttl_seconds,
                    max_entries=policy.max_entries,
                )

            try:
                payload = await execute_fn()
            except asyncio.CancelledError:
                self._refresh_local_binding(
                    cache_key,
                    arguments_hash,
                    policy=policy,
                )
                raise
            except Exception:
                self._refresh_local_binding(
                    cache_key,
                    arguments_hash,
                    policy=policy,
                )
                raise
            canonical = self._canonicalize_success(payload, policy=policy)
            if canonical is None:
                return IdempotencyRunResult(
                    payload=cast(dict[str, JsonValue], payload),
                    from_cache=False,
                    persistence="none",
                )
            encoded, template = canonical
            if not self._try_commit_local_replay(
                cache_key,
                arguments_hash,
                template,
                encoded,
                policy=policy,
            ):
                return IdempotencyRunResult(
                    payload=cast(dict[str, JsonValue], payload),
                    from_cache=False,
                    persistence="none",
                )
            return IdempotencyRunResult(
                payload=cast(dict[str, JsonValue], payload),
                from_cache=False,
                persistence="local",
            )
        finally:
            if acquired:
                lock.release()
            self._release_local_lock_reference(cache_key)

    async def _redis_bind_arguments(
        self,
        client: Any,
        binding_key: str,
        result_key: str,
        arguments_hash: str,
        *,
        binding_ttl_seconds: int,
        deadline: float,
    ) -> Literal["bound", "created", "mismatch", "orphan"]:
        status = await self._call_before_deadline(
            lambda: client.eval(
                _REDIS_BIND_ARGUMENTS_SCRIPT,
                2,
                binding_key,
                result_key,
                arguments_hash.encode("utf-8"),
                binding_ttl_seconds,
            ),
            deadline,
        )
        if status == 1:
            return "bound"
        if status == 2:
            return "created"
        if status == -1:
            return "mismatch"
        if status == -2:
            return "orphan"
        raise RuntimeError("Redis argument binding returned an invalid state")

    def _decode_remote_replay(
        self,
        raw: Any,
        *,
        policy: IdempotencyExecutionPolicy,
    ) -> tuple[bytes, dict[str, JsonValue]]:
        if type(raw) is not bytes:
            raise TypeError("Redis idempotency result must be bytes")
        max_bytes = self._result_limit(policy)
        template = decode_canonical_json_object(raw, max_bytes=max_bytes)
        canonical = canonical_json_bytes(template, max_bytes=max_bytes)
        if canonical != raw:
            raise ValueError("Redis idempotency result is not canonical")
        return raw, template

    async def _read_remote_replay(
        self,
        client: Any,
        result_key: str,
        binding_key: str,
        cache_key: str,
        arguments_hash: str,
        *,
        policy: IdempotencyExecutionPolicy,
        pre_owner: bool,
        deadline: float | None,
    ) -> _RemoteReplayRead:
        try:
            if deadline is None:
                raw = await client.get(result_key)
            else:
                raw = await self._call_before_deadline(
                    lambda: client.get(result_key),
                    deadline,
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - Redis clients expose multiple error families.
            self._mark_remote_failure("redis_result_read", exc)
            if pre_owner:
                return _RemoteReplayRead(replay=None, remote_available=False)
            self._block_remote_key(cache_key, policy=policy)
            raise ExpectedToolFailure(
                ExpectedToolFailureReason.IDEMPOTENCY_UNAVAILABLE
            ) from exc
        if raw is None:
            return _RemoteReplayRead(replay=None)
        try:
            encoded, template = self._decode_remote_replay(raw, policy=policy)
        except Exception as exc:  # noqa: BLE001 - corrupt remote bytes must fail closed.
            self._mark_remote_failure("redis_result_read", exc)
            self._block_remote_key(cache_key, policy=policy)
            raise ExpectedToolFailure(
                ExpectedToolFailureReason.IDEMPOTENCY_UNAVAILABLE
            ) from exc
        local_committed = self._try_commit_local_replay(
            cache_key,
            arguments_hash,
            template,
            encoded,
            policy=policy,
            include_binding=True,
        )
        binding_refreshed = await self._refresh_remote_binding(
            client,
            binding_key,
            cache_key,
            arguments_hash,
            policy=policy,
            deadline=deadline,
        )
        persistence: Literal["durable", "local", "none"] = "durable"
        if not binding_refreshed and local_committed:
            persistence = "local"
        return _RemoteReplayRead(
            replay=IdempotencyRunResult(
                payload=copy.deepcopy(template),
                from_cache=True,
                persistence=persistence,
            ),
            local_committed=local_committed,
        )

    async def _release_remote_lock(self, client: Any, lock_key: str, token: bytes) -> bool:
        lua_script = (
            "if redis.call('get', KEYS[1]) == ARGV[1] "
            "then return redis.call('del', KEYS[1]) end"
        )
        try:
            released = await client.eval(lua_script, 1, lock_key, token)
            if released != 1:
                raise RuntimeError("Redis idempotency lock ownership was lost")
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - success/failure outcomes remain authoritative.
            self._mark_remote_failure("redis_release", exc)
            return False
        return True

    async def _refresh_remote_binding(
        self,
        client: Any,
        binding_key: str,
        cache_key: str,
        arguments_hash: str,
        *,
        policy: IdempotencyExecutionPolicy,
        deadline: float | None = None,
    ) -> bool:
        self._refresh_local_binding(
            cache_key,
            arguments_hash,
            policy=policy,
        )
        try:
            def operation() -> Awaitable[Any]:
                return client.eval(
                    _REDIS_REFRESH_BINDING_SCRIPT,
                    1,
                    binding_key,
                    arguments_hash.encode("utf-8"),
                    policy.ttl_seconds,
                )

            if deadline is None:
                refreshed = await operation()
            else:
                refreshed = await self._call_before_deadline(
                    operation,
                    deadline,
                )
            if refreshed != 1:
                raise RuntimeError("Redis argument binding ownership was lost")
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - binding health cannot replace callback outcomes.
            self._mark_remote_failure("redis_binding", exc)
            self._block_remote_key(cache_key, policy=policy)
            return False
        return True

    async def _store_remote_result(
        self,
        client: Any,
        binding_key: str,
        result_key: str,
        cache_key: str,
        arguments_hash: str,
        encoded: bytes,
        *,
        policy: IdempotencyExecutionPolicy,
    ) -> Literal["durable", "uncertain", "binding_lost"]:
        try:
            stored = await client.eval(
                _REDIS_STORE_RESULT_SCRIPT,
                2,
                binding_key,
                result_key,
                arguments_hash.encode("utf-8"),
                encoded,
                policy.ttl_seconds,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - local commit prevents redispatch.
            self._mark_remote_failure("redis_result_write", exc)
            return "uncertain"
        if stored != 1:
            exc = RuntimeError("Redis argument binding changed before result commit")
            self._mark_remote_failure("redis_result_write", exc)
            self._block_remote_key(cache_key, policy=policy)
            return "binding_lost"
        return "durable"

    async def _release_remote_without_replay(
        self,
        client: Any,
        lock_key: str,
        token: bytes,
        cache_key: str,
        *,
        policy: IdempotencyExecutionPolicy,
    ) -> None:
        if not await self._release_remote_lock(client, lock_key, token):
            self._block_remote_key(cache_key, policy=policy)

    async def _execute_remote_owner(
        self,
        client: Any,
        result_key: str,
        binding_key: str,
        lock_key: str,
        token: bytes,
        cache_key: str,
        arguments_hash: str,
        execute_fn: Callable[[], Awaitable[dict[str, Any]]],
        *,
        policy: IdempotencyExecutionPolicy,
    ) -> IdempotencyRunResult:
        try:
            replay_read = await self._read_remote_replay(
                client,
                result_key,
                binding_key,
                cache_key,
                arguments_hash,
                policy=policy,
                pre_owner=False,
                deadline=None,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            await self._release_remote_without_replay(
                client,
                lock_key,
                token,
                cache_key,
                policy=policy,
            )
            raise
        replay = replay_read.replay
        if replay is not None:
            released = await self._release_remote_lock(client, lock_key, token)
            if released:
                return replay
            persistence = (
                "local" if replay_read.local_committed else replay.persistence
            )
            return IdempotencyRunResult(
                payload=replay.payload,
                from_cache=True,
                persistence=persistence,
            )

        try:
            payload = await execute_fn()
        except asyncio.CancelledError:
            raise
        except Exception:
            await self._refresh_remote_binding(
                client,
                binding_key,
                cache_key,
                arguments_hash,
                policy=policy,
            )
            await self._release_remote_without_replay(
                client,
                lock_key,
                token,
                cache_key,
                policy=policy,
            )
            raise

        canonical = self._canonicalize_success(payload, policy=policy)
        persistence: Literal["durable", "local", "none"] = "none"
        if canonical is not None:
            encoded, template = canonical
            if not self._try_commit_local_replay(
                cache_key,
                arguments_hash,
                template,
                encoded,
                policy=policy,
            ):
                await self._refresh_remote_binding(
                    client,
                    binding_key,
                    cache_key,
                    arguments_hash,
                    policy=policy,
                )
                await self._release_remote_without_replay(
                    client,
                    lock_key,
                    token,
                    cache_key,
                    policy=policy,
                )
                return IdempotencyRunResult(
                    payload=cast(dict[str, JsonValue], payload),
                    from_cache=False,
                    persistence="none",
                )
            persistence = "local"
            result_state = await self._store_remote_result(
                client,
                binding_key,
                result_key,
                cache_key,
                arguments_hash,
                encoded,
                policy=policy,
            )
            if result_state == "durable":
                persistence = "durable"
            elif result_state == "uncertain":
                await self._refresh_remote_binding(
                    client,
                    binding_key,
                    cache_key,
                    arguments_hash,
                    policy=policy,
                )
        else:
            await self._refresh_remote_binding(
                client,
                binding_key,
                cache_key,
                arguments_hash,
                policy=policy,
            )

        released = await self._release_remote_lock(client, lock_key, token)
        if not released and canonical is None:
            self._block_remote_key(cache_key, policy=policy)
        if not released and persistence == "durable":
            persistence = "local"
        return IdempotencyRunResult(
            payload=cast(dict[str, JsonValue], payload),
            from_cache=False,
            persistence=persistence,
        )

    async def _prepare_redis(
        self,
        cache_key: str,
        arguments_hash: str,
        *,
        policy: IdempotencyExecutionPolicy,
        deadline: float,
    ) -> _RemotePreOwnerState | None:
        client = self._redis_client
        if client is None:
            return None

        binding_key = f"mcp:idemp:args:{cache_key}"
        result_key = f"mcp:idemp:result:{cache_key}"
        lock_key = f"mcp:idemp:lock:{cache_key}"
        try:
            binding_state = await self._redis_bind_arguments(
                client,
                binding_key,
                result_key,
                arguments_hash,
                binding_ttl_seconds=policy.lock_ttl_seconds,
                deadline=deadline,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - ambiguous binding writes fail closed.
            self._mark_remote_failure("redis_binding", exc)
            self._block_remote_key(cache_key, policy=policy)
            raise ExpectedToolFailure(
                ExpectedToolFailureReason.IDEMPOTENCY_UNAVAILABLE
            ) from exc
        if binding_state == "mismatch":
            raise InvalidParamsException(
                "Idempotency key was already used with different arguments"
            )
        if binding_state == "orphan":
            exc = RuntimeError("Redis result exists without an argument binding")
            self._mark_remote_failure("redis_binding", exc)
            self._block_remote_key(cache_key, policy=policy)
            raise ExpectedToolFailure(
                ExpectedToolFailureReason.IDEMPOTENCY_UNAVAILABLE
            ) from exc
        with self._local_guard:
            self._put_local_binding_locked(
                cache_key,
                arguments_hash,
                ttl_seconds=policy.ttl_seconds,
                max_entries=policy.max_entries,
            )

        replay_read = await self._read_remote_replay(
            client,
            result_key,
            binding_key,
            cache_key,
            arguments_hash,
            policy=policy,
            pre_owner=True,
            deadline=deadline,
        )
        if not replay_read.remote_available:
            return None
        return _RemotePreOwnerState(
            client=client,
            result_key=result_key,
            binding_key=binding_key,
            lock_key=lock_key,
            replay=replay_read.replay,
        )

    async def _execute_redis(
        self,
        state: _RemotePreOwnerState,
        cache_key: str,
        arguments_hash: str,
        execute_fn: Callable[[], Awaitable[dict[str, Any]]],
        *,
        policy: IdempotencyExecutionPolicy,
        deadline: float,
    ) -> IdempotencyRunResult:
        client = state.client
        if state.replay is not None:
            return state.replay

        while True:
            remaining = self._remaining(deadline)
            if remaining <= 0:
                raise ExpectedToolFailure(
                    ExpectedToolFailureReason.IDEMPOTENCY_IN_PROGRESS
                )
            token = secrets.token_urlsafe(16).encode("ascii")
            try:
                acquired = await self._call_before_deadline(
                    lambda token=token: client.set(
                        state.lock_key,
                        token,
                        nx=True,
                        ex=policy.lock_ttl_seconds,
                    ),
                    deadline,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - SET NX ownership is ambiguous.
                self._mark_remote_failure("redis_lock_acquire", exc)
                self._block_remote_key(cache_key, policy=policy)
                raise ExpectedToolFailure(
                    ExpectedToolFailureReason.IDEMPOTENCY_UNAVAILABLE
                ) from exc
            if acquired:
                return await self._execute_remote_owner(
                    client,
                    state.result_key,
                    state.binding_key,
                    state.lock_key,
                    token,
                    cache_key,
                    arguments_hash,
                    execute_fn,
                    policy=policy,
                )

            replay_read = await self._read_remote_replay(
                client,
                state.result_key,
                state.binding_key,
                cache_key,
                arguments_hash,
                policy=policy,
                pre_owner=False,
                deadline=deadline,
            )
            if replay_read.replay is not None:
                return replay_read.replay
            remaining = self._remaining(deadline)
            if remaining <= 0:
                raise ExpectedToolFailure(
                    ExpectedToolFailureReason.IDEMPOTENCY_IN_PROGRESS
                )
            await asyncio.sleep(min(_REDIS_POLL_INTERVAL_SECONDS, remaining))

    async def execute(
        self,
        cache_key: str,
        arguments_hash: str,
        execute_fn: Callable[[], Awaitable[dict[str, Any]]],
        *,
        policy: IdempotencyExecutionPolicy,
    ) -> IdempotencyRunResult:
        """Bind arguments, acquire ownership, and return a fresh replay or success."""
        with self._local_guard:
            replay = self._get_local_replay_locked(
                cache_key,
                arguments_hash,
                ttl_seconds=policy.ttl_seconds,
            )
            if replay is not None:
                return replay
            self._check_local_binding_locked(
                cache_key,
                arguments_hash,
                ttl_seconds=policy.ttl_seconds,
            )
            self._check_remote_uncertain_locked(cache_key)

        deadline = asyncio.get_running_loop().time() + policy.contention_wait_seconds
        if await self._ensure_redis():
            remote_state = await self._prepare_redis(
                cache_key,
                arguments_hash,
                policy=policy,
                deadline=deadline,
            )
            if remote_state is not None:
                return await self._execute_redis(
                    remote_state,
                    cache_key,
                    arguments_hash,
                    execute_fn,
                    policy=policy,
                    deadline=deadline,
                )
        return await self._execute_local(
            cache_key,
            arguments_hash,
            execute_fn,
            policy=policy,
            deadline=deadline,
        )

    async def shutdown(self) -> None:
        """Task 5 lifecycle hook; Task 6 owns finalizer draining semantics."""
        return None
