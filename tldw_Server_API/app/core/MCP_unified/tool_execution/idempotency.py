"""Idempotency execution with Redis backing and local lock fallback."""

import asyncio
import contextlib
import json
import secrets
import time
from collections import OrderedDict
from typing import Any, Callable, Optional

from loguru import logger

from ..auth.rate_limiter import RateLimitExceeded
from ..config import get_config
from ..protocol_types import InvalidParamsException

try:  # pragma: no cover - optional dependency
    from redis.exceptions import RedisError
except ImportError:  # pragma: no cover - redis not installed
    class RedisError(Exception):
        """Fallback RedisError when redis-py is unavailable."""
        pass


# Redis exceptions can include connection URLs and credentials; keep logs sanitized.
def _redact_redis_error(exc: Exception) -> str:
    return f"{exc.__class__.__name__}: Redis connection error - details redacted"


_MCP_PROTOCOL_NONCRITICAL_EXCEPTIONS = (
    asyncio.CancelledError,
    asyncio.TimeoutError,
    AssertionError,
    AttributeError,
    ConnectionError,
    FileNotFoundError,
    ImportError,
    IndexError,
    KeyError,
    LookupError,
    OSError,
    PermissionError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
    UnicodeDecodeError,
    json.JSONDecodeError,
    RedisError,
    RateLimitExceeded,
    InvalidParamsException,
)


async def _no_redis_client_factory(**_kwargs: Any) -> None:
    """Fallback Redis factory used when embedders do not provide Redis support."""
    return None


class IdempotencyManager:
    """Idempotency manager with Redis backing and local lock fallback."""

    def __init__(self, redis_client_factory: Optional[Callable[..., Any]] = None) -> None:
        self._local_cache: OrderedDict[str, tuple[float, dict[str, Any]]] = OrderedDict()
        self._local_bindings: OrderedDict[str, tuple[float, str]] = OrderedDict()
        self._local_locks: dict[str, asyncio.Lock] = {}
        self._local_guard = asyncio.Lock()
        self._redis_client: Any | None = None
        self._redis_ready = False
        self._redis_attempted = False
        self._redis_guard = asyncio.Lock()
        self._redis_client_factory = redis_client_factory or _no_redis_client_factory

    def _prune_local_locks(self) -> None:
        """Drop stale local locks once their cache/binding entries are gone."""
        active_keys = set(self._local_cache.keys()) | set(self._local_bindings.keys())
        stale_keys = [
            key
            for key, lock in self._local_locks.items()
            if key not in active_keys and not lock.locked()
        ]
        for key in stale_keys:
            with contextlib.suppress(_MCP_PROTOCOL_NONCRITICAL_EXCEPTIONS):
                del self._local_locks[key]

    async def _ensure_redis(self) -> bool:
        if self._redis_attempted:
            return self._redis_ready
        async with self._redis_guard:
            if self._redis_attempted:
                return self._redis_ready
            self._redis_attempted = True
            cfg = get_config()
            params = cfg.get_redis_connection_params()
            if not params:
                self._redis_ready = False
                return False
            url = params.pop("url", None)
            try:
                self._redis_client = await self._redis_client_factory(
                    preferred_url=url,
                    decode_responses=True,
                    fallback_to_fake=False,
                    context="mcp_idempotency",
                    redis_kwargs=params,
                )
                if self._redis_client is None:
                    logger.warning(
                        "MCP idempotency Redis unavailable; falling back to local locks. "
                        "Error: Redis client factory returned None for context=mcp_idempotency",
                    )
                    self._redis_client = None
                    self._redis_ready = False
                    return False
                self._redis_ready = True
            except _MCP_PROTOCOL_NONCRITICAL_EXCEPTIONS as exc:
                logger.warning(
                    "MCP idempotency Redis unavailable; falling back to local locks. Error: {}",
                    _redact_redis_error(exc),
                )
                self._redis_client = None
                self._redis_ready = False
            return self._redis_ready

    def _local_get(self, cache_key: str, ttl: int) -> Optional[dict[str, Any]]:
        item = self._local_cache.get(cache_key)
        if not item:
            return None
        ts, payload = item
        if time.time() - ts > ttl:
            with contextlib.suppress(_MCP_PROTOCOL_NONCRITICAL_EXCEPTIONS):
                del self._local_cache[cache_key]
            self._prune_local_locks()
            return None
        with contextlib.suppress(_MCP_PROTOCOL_NONCRITICAL_EXCEPTIONS):
            self._local_cache.move_to_end(cache_key)
        return payload

    def _local_put(self, cache_key: str, payload: dict[str, Any], ttl: int, max_size: int) -> None:
        now = time.time()
        self._local_cache[cache_key] = (now, payload)
        with contextlib.suppress(_MCP_PROTOCOL_NONCRITICAL_EXCEPTIONS):
            self._local_cache.move_to_end(cache_key)
        # Evict expired entries opportunistically
        expired = [k for k, (ts, _) in self._local_cache.items() if now - ts > ttl]
        for k in expired:
            with contextlib.suppress(_MCP_PROTOCOL_NONCRITICAL_EXCEPTIONS):
                del self._local_cache[k]
        # Enforce max size (LRU)
        while len(self._local_cache) > max_size:
            try:
                self._local_cache.popitem(last=False)
            except _MCP_PROTOCOL_NONCRITICAL_EXCEPTIONS:
                break
        self._prune_local_locks()

    def _local_get_binding(self, binding_key: str, ttl: int) -> Optional[str]:
        item = self._local_bindings.get(binding_key)
        if not item:
            return None
        ts, arguments_hash = item
        if time.time() - ts > ttl:
            with contextlib.suppress(_MCP_PROTOCOL_NONCRITICAL_EXCEPTIONS):
                del self._local_bindings[binding_key]
            self._prune_local_locks()
            return None
        with contextlib.suppress(_MCP_PROTOCOL_NONCRITICAL_EXCEPTIONS):
            self._local_bindings.move_to_end(binding_key)
        return arguments_hash

    def _local_put_binding(self, binding_key: str, arguments_hash: str, ttl: int, max_size: int) -> None:
        now = time.time()
        self._local_bindings[binding_key] = (now, arguments_hash)
        with contextlib.suppress(_MCP_PROTOCOL_NONCRITICAL_EXCEPTIONS):
            self._local_bindings.move_to_end(binding_key)
        expired = [k for k, (ts, _) in self._local_bindings.items() if now - ts > ttl]
        for k in expired:
            with contextlib.suppress(_MCP_PROTOCOL_NONCRITICAL_EXCEPTIONS):
                del self._local_bindings[k]
        while len(self._local_bindings) > max_size:
            try:
                self._local_bindings.popitem(last=False)
            except _MCP_PROTOCOL_NONCRITICAL_EXCEPTIONS:
                break
        self._prune_local_locks()

    async def _get_local_lock(self, cache_key: str) -> asyncio.Lock:
        async with self._local_guard:
            lock = self._local_locks.get(cache_key)
            if lock is None:
                lock = asyncio.Lock()
                self._local_locks[cache_key] = lock
            return lock

    async def _redis_get(self, client: Any, key: str) -> Optional[dict[str, Any]]:
        raw = await client.get(key)
        if not raw:
            return None
        try:
            return json.loads(raw)
        except _MCP_PROTOCOL_NONCRITICAL_EXCEPTIONS:
            return None

    async def _redis_set(self, client: Any, key: str, payload: dict[str, Any], ttl: int) -> None:
        data = json.dumps(payload, separators=(",", ":"), default=str)
        await client.set(key, data, ex=ttl)

    async def _redis_try_acquire(self, client: Any, key: str, token: str, ttl: int) -> bool:
        resp = await client.set(key, token, nx=True, ex=ttl)
        return bool(resp)

    async def _redis_release(self, client: Any, key: str, token: str) -> None:
        lua_script = (
            "if redis.call('get', KEYS[1]) == ARGV[1] "
            "then return redis.call('del', KEYS[1]) end"
        )
        with contextlib.suppress(_MCP_PROTOCOL_NONCRITICAL_EXCEPTIONS):
            await client.eval(lua_script, 1, key, token)

    async def _redis_bind_arguments(self, client: Any, key: str, arguments_hash: str, ttl: int) -> bool:
        binding_key = f"mcp:idemp:args:{key}"
        created = await client.set(binding_key, arguments_hash, nx=True, ex=ttl)
        if created:
            return True
        existing = await client.get(binding_key)
        if existing is None:
            # Key may have expired between checks; retry once.
            created = await client.set(binding_key, arguments_hash, nx=True, ex=ttl)
            if created:
                return True
            existing = await client.get(binding_key)
        if existing is None:
            return True
        if existing == arguments_hash:
            with contextlib.suppress(_MCP_PROTOCOL_NONCRITICAL_EXCEPTIONS):
                await client.expire(binding_key, ttl)
            return True
        return False

    async def _run_local(
        self,
        cache_key: str,
        execute_fn: Callable[[], Any],
        *,
        ttl: int,
        max_size: int,
    ) -> tuple[dict[str, Any], bool]:
        async with self._local_guard:
            cached = self._local_get(cache_key, ttl)
        if cached is not None:
            return cached, True

        lock = await self._get_local_lock(cache_key)
        try:
            async with lock:
                async with self._local_guard:
                    cached = self._local_get(cache_key, ttl)
                if cached is not None:
                    return cached, True
                result = await execute_fn()
                async with self._local_guard:
                    self._local_put(cache_key, result, ttl, max_size)
                return result, False
        finally:
            async with self._local_guard:
                self._prune_local_locks()

    async def _run_redis(
        self,
        cache_key: str,
        execute_fn: Callable[[], Any],
        *,
        ttl: int,
        lock_ttl: int,
    ) -> tuple[dict[str, Any], bool]:
        client = self._redis_client
        if client is None:
            raise RuntimeError("Redis client not initialized")
        result_key = f"mcp:idemp:result:{cache_key}"
        lock_key = f"mcp:idemp:lock:{cache_key}"
        cached = await self._redis_get(client, result_key)
        if cached is not None:
            return cached, True

        poll_interval = 0.2
        while True:
            token = secrets.token_urlsafe(16)
            acquired = await self._redis_try_acquire(client, lock_key, token, lock_ttl)
            if acquired:
                try:
                    result = await execute_fn()
                    await self._redis_set(client, result_key, result, ttl)
                finally:
                    await self._redis_release(client, lock_key, token)
                return result, False

            cached = await self._redis_get(client, result_key)
            if cached is not None:
                return cached, True
            await asyncio.sleep(poll_interval)

    async def run(
        self,
        cache_key: str,
        execute_fn: Callable[[], Any],
        *,
        ttl: int,
        max_size: int,
        lock_ttl: int,
    ) -> tuple[dict[str, Any], bool]:
        if await self._ensure_redis():
            try:
                return await self._run_redis(
                    cache_key,
                    execute_fn,
                    ttl=ttl,
                    lock_ttl=lock_ttl,
                )
            except RedisError as exc:
                logger.warning(
                    "MCP idempotency Redis path failed; falling back to local locks. Error: {}",
                    _redact_redis_error(exc),
                )
                self._redis_ready = False
        return await self._run_local(cache_key, execute_fn, ttl=ttl, max_size=max_size)

    async def bind_arguments(
        self,
        cache_key: str,
        arguments_hash: str,
        *,
        ttl: int,
        max_size: int,
    ) -> bool:
        if await self._ensure_redis():
            try:
                client = self._redis_client
                if client is not None:
                    return await self._redis_bind_arguments(client, cache_key, arguments_hash, ttl)
            except RedisError as exc:
                logger.warning(
                    "MCP idempotency binding Redis path failed; falling back to local cache. Error: {}",
                    _redact_redis_error(exc),
                )
                self._redis_ready = False

        async with self._local_guard:
            existing = self._local_get_binding(cache_key, ttl)
            if existing is None:
                self._local_put_binding(cache_key, arguments_hash, ttl, max_size)
                return True
            if existing != arguments_hash:
                return False
            self._local_put_binding(cache_key, arguments_hash, ttl, max_size)
            return True
