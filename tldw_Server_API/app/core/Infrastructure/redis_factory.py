from __future__ import annotations

import asyncio
import contextlib
import fnmatch
import hashlib
import inspect
import os
import threading
import time
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from loguru import logger

from tldw_Server_API.app.core.config import settings

try:  # pragma: no cover - import guard
    import redis  # type: ignore
    import redis.asyncio as aioredis  # type: ignore
except ImportError as exc:  # pragma: no cover
    redis = None  # type: ignore
    aioredis = None  # type: ignore
    _import_error = exc
else:
    _import_error = None

if redis is not None:
    _REDIS_CLIENT_EXCEPTIONS: tuple[type[Exception], ...] = (redis.RedisError,)  # type: ignore[attr-defined]
else:
    _REDIS_CLIENT_EXCEPTIONS = ()

_REDIS_NONCRITICAL_EXCEPTIONS: tuple[type[Exception], ...] = (
    AttributeError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
) + _REDIS_CLIENT_EXCEPTIONS

_ASYNC_STUB_CACHE: dict[str, InMemoryAsyncRedis] = {}

try:  # pragma: no cover - optional metrics dependency
    from tldw_Server_API.app.core.Metrics.metrics_manager import (
        get_metrics_registry as _get_metrics_registry,
    )
except ImportError:  # pragma: no cover - metrics optional during early startup
    _get_metrics_registry = None  # type: ignore[assignment]

DEFAULT_REDIS_URL = "redis://127.0.0.1:6379"


def _settings_lookup(*keys: str) -> str | None:
    """Look up configuration values with environment variables overriding file settings.

    Order of precedence:
      1) Settings file values (e.g., from config.txt / .env)
      2) Environment variables (if set and non-empty)
    """
    for key in keys:
        # Fall back to settings-backed values
        try:
            value = settings.get(key)  # type: ignore[attr-defined]
            if isinstance(value, str) and value.strip():
                return value.strip()
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
            pass
        # Prefer explicit environment overrides when present
        env = os.getenv(key)
        if env is not None and str(env).strip():
            return str(env).strip()
    return None


def _resolve_url(preferred: str | None = None) -> str:
    if preferred and str(preferred).strip():
        return str(preferred).strip()
    url = _settings_lookup("EMBEDDINGS_REDIS_URL", "REDIS_URL")
    return url or DEFAULT_REDIS_URL


def _redact_redis_url(url: str) -> str:
    """Return *url* with any username/password userinfo redacted."""
    text = str(url or "")
    try:
        parsed = urlsplit(text)
        password = parsed.password
        username = parsed.username
        if password is None and username is None:
            return text

        host = parsed.hostname or ""
        port = parsed.port
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        if port is not None:
            host = f"{host}:{port}"

        if password is not None:
            redacted_username = "" if username is None else "***"
            userinfo = f"{redacted_username}:***"
        else:
            userinfo = "***" if username is not None else ""

        netloc = f"{userinfo}@{host}" if userinfo or password is not None else host
        return urlunsplit((parsed.scheme, netloc, parsed.path, parsed.query, parsed.fragment))
    except ValueError:
        return "<invalid-redis-url>"


def _metrics_registry():
    if _get_metrics_registry is None:
        return None
    try:
        return _get_metrics_registry()
    except _REDIS_NONCRITICAL_EXCEPTIONS:
        return None


def _record_connection_metrics(
    *,
    mode: str,
    context: str,
    outcome: str,
    start_time: float,
    error: BaseException | None = None,
):
    registry = _metrics_registry()
    if registry is None:
        return

    elapsed = max(time.perf_counter() - start_time, 0.0)
    labels = {"mode": mode, "context": context, "outcome": outcome}
    try:
        registry.increment("infra_redis_connection_attempts_total", 1, labels)
        registry.observe("infra_redis_connection_duration_seconds", elapsed, labels)
        if outcome == "stub":
            reason = type(error).__name__ if error else "fallback"
            registry.increment(
                "infra_redis_fallback_total",
                1,
                {"mode": mode, "context": context, "reason": reason},
            )
        elif outcome == "error":
            reason = type(error).__name__ if error else "unknown"
            registry.increment(
                "infra_redis_connection_errors_total",
                1,
                {"mode": mode, "context": context, "error": reason},
            )
    except _REDIS_NONCRITICAL_EXCEPTIONS as metric_exc:
        logger.debug(
            "Failed to record Redis infrastructure metrics: {err}",
            err=metric_exc,
        )


async def create_async_redis_client(
    *,
    preferred_url: str | None = None,
    decode_responses: bool = True,
    fallback_to_fake: bool = False,
    context: str = "default",
    redis_kwargs: dict | None = None,
):
    """
    Instantiate an asyncio Redis client. Falls back to an in-memory stub when allowed.

    Args:
        preferred_url: Explicit URL to prioritize (e.g., embeddings queue).
        decode_responses: Whether to decode bytes into str.
        fallback_to_fake: If True, transparently fallback to an in-memory stub when
            the real server is unreachable. Production-sensitive callers should
            leave this disabled unless they explicitly support per-process memory.
        context: Human-readable label for logging (helps trace callers).

    Returns:
        An asyncio Redis client implementing the standard redis-py API.
    """

    url = _resolve_url(preferred_url)
    context_label = (context or "default").strip() or "default"
    options = dict(redis_kwargs or {})
    if "decode_responses" not in options:
        options["decode_responses"] = decode_responses
    start_time = time.perf_counter()
    decode_option = options.get("decode_responses", decode_responses)

    if aioredis is None:
        exc = RuntimeError("redis[asyncio] is required but not installed")
        if not fallback_to_fake:
            _record_connection_metrics(
                mode="async",
                context=context_label,
                outcome="error",
                start_time=start_time,
                error=exc,
            )
            raise exc from _import_error
        cache_key = f"{url}::{context_label}::{decode_option}"
        fake_client = _ASYNC_STUB_CACHE.get(cache_key)
        if fake_client is None:
            fake_client = InMemoryAsyncRedis(decode_responses=decode_option)
            _ASYNC_STUB_CACHE[cache_key] = fake_client
        fake_client._tldw_is_stub = True
        await fake_client.ping()
        _record_connection_metrics(
            mode="async",
            context=context_label,
            outcome="stub",
            start_time=start_time,
            error=exc,
        )
        return fake_client

    client = None
    try:
        candidate = aioredis.from_url(url, **options)
        if inspect.isawaitable(candidate):  # redis<5 compatibility
            candidate = await candidate
        client = candidate
        ping = getattr(client, "ping", None)
        if ping is None:
            _record_connection_metrics(
                mode="async",
                context=context_label,
                outcome="stub",
                start_time=start_time,
            )
            return client
        result = ping()
        if inspect.isawaitable(result):
            await result
    except _REDIS_NONCRITICAL_EXCEPTIONS as exc:
        if client is not None:
            with contextlib.suppress(_REDIS_NONCRITICAL_EXCEPTIONS):
                await client.close()
        if not fallback_to_fake:
            _record_connection_metrics(
                mode="async",
                context=context_label,
                outcome="error",
                start_time=start_time,
                error=exc,
            )
            raise
        logger.warning(
            "Redis unavailable at {url} for {context}; using in-memory stub. Error: {err}",
            url=_redact_redis_url(url),
            context=context_label,
            err=exc,
        )
        cache_key = f"{url}::{context_label}::{decode_option}"
        fake_client = _ASYNC_STUB_CACHE.get(cache_key)
        if fake_client is None:
            fake_client = InMemoryAsyncRedis(decode_responses=decode_option)
            _ASYNC_STUB_CACHE[cache_key] = fake_client
        fake_client._tldw_is_stub = True
        await fake_client.ping()
        _record_connection_metrics(
            mode="async",
            context=context_label,
            outcome="stub",
            start_time=start_time,
            error=exc,
        )
        return fake_client

    _record_connection_metrics(
        mode="async",
        context=context_label,
        outcome="real",
        start_time=start_time,
    )
    return client


def create_sync_redis_client(
    *,
    preferred_url: str | None = None,
    decode_responses: bool = True,
    fallback_to_fake: bool = False,
    context: str = "default",
    redis_kwargs: dict | None = None,
):
    """
    Instantiate a synchronous Redis client with optional in-memory fallback.
    """

    url = _resolve_url(preferred_url)
    context_label = (context or "default").strip() or "default"
    options = dict(redis_kwargs or {})
    if "decode_responses" not in options:
        options["decode_responses"] = decode_responses
    start_time = time.perf_counter()

    if redis is None:
        exc = RuntimeError("redis client is required but not installed")
        if not fallback_to_fake:
            _record_connection_metrics(
                mode="sync",
                context=context_label,
                outcome="error",
                start_time=start_time,
                error=exc,
            )
            raise exc from _import_error
        fake_client = InMemorySyncRedis(
            decode_responses=options.get("decode_responses", True)
        )
        fake_client._tldw_is_stub = True
        fake_client.ping()
        _record_connection_metrics(
            mode="sync",
            context=context_label,
            outcome="stub",
            start_time=start_time,
            error=exc,
        )
        return fake_client

    client = None

    try:
        client = redis.from_url(url, **options)
        client.ping()
    except _REDIS_NONCRITICAL_EXCEPTIONS as exc:
        if client is not None:
            with contextlib.suppress(_REDIS_NONCRITICAL_EXCEPTIONS):
                client.close()
        if not fallback_to_fake:
            _record_connection_metrics(
                mode="sync",
                context=context_label,
                outcome="error",
                start_time=start_time,
                error=exc,
            )
            raise
        logger.warning(
            "Redis unavailable at {url} for {context}; using in-memory stub. Error: {err}",
            url=_redact_redis_url(url),
            context=context_label,
            err=exc,
        )
        fake_client = InMemorySyncRedis(
            decode_responses=options.get("decode_responses", True)
        )
        fake_client._tldw_is_stub = True
        fake_client.ping()
        _record_connection_metrics(
            mode="sync",
            context=context_label,
            outcome="stub",
            start_time=start_time,
            error=exc,
        )
        return fake_client

    _record_connection_metrics(
        mode="sync",
        context=context_label,
        outcome="real",
        start_time=start_time,
    )
    return client


class _InMemoryRedisCore:
    """Stateful in-memory substitute implementing minimal Redis behaviors."""

    def __init__(self, decode_responses: bool = True):
        self.decode_responses = decode_responses
        self._strings: dict[str, str] = {}
        self._sets: dict[str, set] = {}
        self._sorted_sets: dict[str, dict[str, float]] = {}
        self._hashes: dict[str, dict[str, str]] = {}
        self._streams: dict[str, list[tuple[str, dict[str, str]]]] = {}
        self._stream_counters: dict[str, int] = {}
        self._groups: dict[str, set] = {}
        self._group_last_ids: dict[tuple[str, str], str] = {}
        self._expiry: dict[str, float] = {}
        self._scripts: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Basic utilities
    # ------------------------------------------------------------------
    def ping(self) -> bool:
        return True

    def close(self) -> None:
        pass

    def _now(self) -> float:
        return time.time()

    def _delete_internal(self, key: str) -> None:
        self._strings.pop(key, None)
        self._sets.pop(key, None)
        self._sorted_sets.pop(key, None)
        self._hashes.pop(key, None)
        self._streams.pop(key, None)
        self._stream_counters.pop(key, None)
        self._groups.pop(key, None)
        self._expiry.pop(key, None)

    def _check_expiry(self, key: str) -> None:
        expires_at = self._expiry.get(key)
        if expires_at is not None and expires_at <= self._now():
            self._delete_internal(key)

    def _stream_id_tuple(self, entry_id: str) -> tuple[int, int]:
        if entry_id in ("-", None):
            return (0, 0)
        if entry_id == "+":
            return (2**63 - 1, 2**63 - 1)
        text = str(entry_id)
        if text == "0":
            return (0, 0)
        parts = text.split("-", 1)
        try:
            ts = int(parts[0])
        except (TypeError, ValueError):
            ts = 0
        seq = 0
        if len(parts) > 1:
            try:
                seq = int(parts[1])
            except (TypeError, ValueError):
                seq = 0
        return (ts, seq)

    def _stream_id_gt(self, entry_id: str, other_id: str) -> bool:
        return self._stream_id_tuple(entry_id) > self._stream_id_tuple(other_id)

    # ------------------------------------------------------------------
    # Stream operations
    # ------------------------------------------------------------------
    def xlen(self, name: str) -> int:
        return len(self._streams.get(name, []))

    def xadd(self, name: str, fields: dict[str, Any]) -> str:
        stream = self._streams.setdefault(name, [])
        counter = self._stream_counters.get(name, 0) + 1
        self._stream_counters[name] = counter
        entry_id = f"{int(self._now() * 1000)}-{counter}"
        stream.append((entry_id, {str(k): str(v) for k, v in fields.items()}))
        return entry_id

    def xrange(
        self,
        name: str,
        min: str = "-",  # noqa: A002 - match redis signature
        max: str = "+",
        count: int | None = None,
        **kwargs: Any,
    ) -> list[tuple[str, dict[str, str]]]:
        minimum = kwargs.get("minimum", min)
        maximum = kwargs.get("maximum", max)
        stream = list(self._streams.get(name, []))
        if minimum not in ("-", None) or maximum not in ("+", None):
            def _within(entry_id: str) -> bool:
                ts = entry_id.split("-", 1)[0]
                if minimum not in ("-", None) and ts < str(minimum):
                    return False
                return not (maximum not in ("+", None) and ts > str(maximum))
            stream = [item for item in stream if _within(item[0])]
        if count is not None and count >= 0:
            stream = stream[:count]
        return [(entry_id, dict(data)) for entry_id, data in stream]

    def xrevrange(
        self,
        name: str,
        max: str = "+",
        min: str = "-",
        count: int | None = None,
        **kwargs: Any,
    ) -> list[tuple[str, dict[str, str]]]:
        maximum = kwargs.get("maximum", max)
        minimum = kwargs.get("minimum", min)
        stream = list(self._streams.get(name, []))
        stream.reverse()
        if minimum not in ("-", None) or maximum not in ("+", None):
            def _within(entry_id: str) -> bool:
                ts = entry_id.split("-", 1)[0]
                if maximum not in ("+", None) and ts > str(maximum):
                    return False
                return not (minimum not in ("-", None) and ts < str(minimum))
            stream = [item for item in stream if _within(item[0])]
        if count is not None and count >= 0:
            stream = stream[:count]
        return [(entry_id, dict(data)) for entry_id, data in stream]

    def xdel(self, name: str, entry_id: str) -> int:
        stream = self._streams.get(name, [])
        before = len(stream)
        stream[:] = [item for item in stream if item[0] != entry_id]
        return before - len(stream)

    def xgroup_create(self, name: str, group: str, id: str = "0") -> None:
        self._groups.setdefault(name, set()).add(group)
        if id == "$":
            stream = self._streams.get(name, [])
            last_id = stream[-1][0] if stream else "0-0"
        else:
            last_id = "0-0" if id in ("0", "0-0") else str(id)
        self._group_last_ids[(name, group)] = last_id

    def xreadgroup(
        self,
        groupname: str,
        consumername: str,
        streams: dict[str, str],
        count: int | None = None,
        block: int | None = None,
        noack: bool = False,
    ) -> list[tuple[str, list[tuple[str, dict[str, str]]]]]:
        _ = (consumername, block, noack)
        results: list[tuple[str, list[tuple[str, dict[str, str]]]]] = []
        for stream_name, start_id in streams.items():
            if groupname not in self._groups.get(stream_name, set()):
                raise RuntimeError("NOGROUP")
            last_id = self._group_last_ids.get((stream_name, groupname), "0-0")
            compare_id = last_id if start_id == ">" else (start_id or "0-0")
            entries: list[tuple[str, dict[str, str]]] = []
            for entry_id, data in self._streams.get(stream_name, []):
                if self._stream_id_gt(entry_id, compare_id):
                    entries.append((entry_id, dict(data)))
            if count is not None and count >= 0:
                entries = entries[:count]
            if entries:
                results.append((stream_name, entries))
                self._group_last_ids[(stream_name, groupname)] = entries[-1][0]
        return results

    # ------------------------------------------------------------------
    # Set / sorted set operations
    # ------------------------------------------------------------------
    def sadd(self, key: str, member: str) -> int:
        st = self._sets.setdefault(key, set())
        before = len(st)
        st.add(str(member))
        return 1 if len(st) > before else 0

    def srem(self, key: str, member: str) -> int:
        st = self._sets.get(key)
        if st is None:
            return 0
        try:
            st.remove(str(member))
            return 1
        except KeyError:
            return 0

    def smembers(self, key: str) -> set:
        return set(self._sets.get(key, set()))

    def zadd(self, key: str, mapping: dict[str, float]) -> None:
        zset = self._sorted_sets.setdefault(key, {})
        for member, score in mapping.items():
            zset[str(member)] = float(score)

    def zrem(self, key: str, member: str) -> int:
        zset = self._sorted_sets.get(key)
        if not zset:
            return 0
        member = str(member)
        if member in zset:
            try:
                del zset[member]
            except KeyError:
                return 0
            if not zset:
                self._sorted_sets.pop(key, None)
            return 1
        return 0

    def zremrangebyscore(self, key: str, minimum: float, maximum: float) -> int:
        zset = self._sorted_sets.get(key)
        if not zset:
            return 0
        removed = 0
        for member in list(zset.keys()):
            score = zset[member]
            if minimum <= score <= maximum:
                del zset[member]
                removed += 1
        if not zset:
            self._sorted_sets.pop(key, None)
        return removed

    def zcard(self, key: str) -> int:
        return len(self._sorted_sets.get(key, {}))

    def zrange(self, key: str, start: int, stop: int) -> list[str]:
        """Return members in score order (ascending) from start to stop inclusive.

        This is a minimal emulation adequate for tests that need to list ZSET members.
        """
        z = self._sorted_sets.get(key, {})
        if not z:
            return []
        # Sort members by score ascending, then by member name to stabilize
        items = sorted(z.items(), key=lambda kv: (kv[1], kv[0]))
        members = [m for m, _ in items]
        n = len(members)
        # Normalize negative indices like Redis
        if start < 0:
            start = n + start
        if stop < 0:
            stop = n + stop
        # Clamp
        start = max(0, start)
        stop = min(n - 1, stop) if n > 0 else -1
        if start > stop or stop < 0:
            return []
        return members[start: stop + 1]

    def zscore(self, key: str, member: str) -> float | None:
        z = self._sorted_sets.get(key, {})
        if not z:
            return None
        return float(z.get(str(member))) if str(member) in z else None

    def zincrby(self, key: str, amount: float, member: str) -> float:
        zset = self._sorted_sets.setdefault(key, {})
        member = str(member)
        current = float(zset.get(member, 0.0))
        current += float(amount)
        zset[member] = current
        return current

    # ------------------------------------------------------------------
    # Hash operations
    # ------------------------------------------------------------------
    def hset(self, key: str, mapping: dict[str, Any]) -> int:
        target = self._hashes.setdefault(key, {})
        created = 0
        for field, value in mapping.items():
            field = str(field)
            if field not in target:
                created += 1
            target[field] = str(value)
        return created

    def hget(self, key: str, field: str) -> str | None:
        return self._hashes.get(key, {}).get(str(field))

    def hgetall(self, key: str) -> dict[str, str]:
        return dict(self._hashes.get(key, {}))

    def hincrby(self, key: str, field: str, amount: int = 1) -> int:
        target = self._hashes.setdefault(key, {})
        current = int(target.get(str(field), "0"))
        current += int(amount)
        target[str(field)] = str(current)
        return current

    # ------------------------------------------------------------------
    # String operations
    # ------------------------------------------------------------------
    def set(self, key: str, value: Any, ex: int | None = None) -> None:
        self._strings[key] = str(value)
        if ex is not None:
            self._expiry[key] = self._now() + int(ex)

    def setex(self, key: str, seconds: int, value: Any) -> None:
        self.set(key, value, ex=int(seconds))

    def get(self, key: str) -> str | None:
        self._check_expiry(key)
        return self._strings.get(key)

    def delete(self, *keys: str) -> int:
        deleted = 0
        for key in keys:
            existed = (
                key in self._strings
                or key in self._sets
                or key in self._sorted_sets
                or key in self._hashes
                or key in self._streams
            )
            if existed:
                deleted += 1
            self._delete_internal(key)
        return deleted

    def expire(self, key: str, seconds: int) -> None:
        if key in self._strings or key in self._sets or key in self._sorted_sets or key in self._hashes:
            self._expiry[key] = self._now() + int(seconds)

    def ttl(self, key: str) -> int:
        self._check_expiry(key)
        expires_at = self._expiry.get(key)
        if expires_at is None:
            return -1
        remaining = int(round(expires_at - self._now()))
        return remaining if remaining >= 0 else -2

    def incr(self, key: str) -> int:
        return self.incrby(key, 1)

    def incrby(self, key: str, amount: int) -> int:
        self._check_expiry(key)
        current = int(self._strings.get(key, "0"))
        current += int(amount)
        self._strings[key] = str(current)
        return current

    # ------------------------------------------------------------------
    # Misc helpers
    # ------------------------------------------------------------------
    def scan(self, cursor: int, match: str | None, count: int | None) -> tuple[int, list[str]]:
        keys = set(self._strings.keys()) | set(self._sets.keys()) | set(self._sorted_sets.keys()) | set(self._hashes.keys()) | set(self._streams.keys())
        if match:
            pattern = match
            keys = {k for k in keys if fnmatch.fnmatch(k, pattern)}
        result = sorted(keys)
        if count is not None and count >= 0:
            result = result[:count]
        return 0, result

    def keys(self, pattern: str) -> list[str]:
        _, result = self.scan(0, match=pattern, count=None)
        return result

    def dbsize(self) -> int:
        keys = set(self._strings.keys()) | set(self._sets.keys()) | set(self._sorted_sets.keys()) | set(self._hashes.keys()) | set(self._streams.keys())
        return len(keys)

    def info(self, section: str | None = None) -> dict[str, Any]:
        if section not in (None, "memory"):
            return {}
        used = self.dbsize()
        return {
            "used_memory": used,
            "used_memory_human": f"{used} keys",
            "maxmemory": 0,
        }

    def script_load(self, script: str) -> str:
        sha = hashlib.sha1(script.encode("utf-8"), usedforsecurity=False).hexdigest()
        self._scripts[sha] = script
        return sha

    def evalsha(self, sha: str, num_keys: int, *args) -> list[Any]:
        script = self._scripts.get(sha)
        if script is None:
            raise RuntimeError("NOSCRIPT")
        return self.eval(script, num_keys, *args)

    def eval(self, script: str, num_keys: int, *args) -> list[Any]:
        # Heuristic: handle the rate limiter script used by DistributedRateLimiter
        if "ZRANGE" in script and "ZREMRANGEBYSCORE" in script:
            if num_keys < 1 or len(args) < 3:
                raise RuntimeError("Invalid arguments for rate limiter script")
            redis_key = args[0]
            limit = int(args[1])
            window = int(args[2])
            current_time = float(args[3]) if len(args) > 3 else self._now()
            return self._eval_rate_limiter(redis_key, limit, window, current_time)
        raise RuntimeError("Unsupported script")

    def _eval_rate_limiter(self, redis_key: str, limit: int, window: int, current_time: float) -> list[Any]:
        zset = self._sorted_sets.setdefault(redis_key, {})
        cutoff = current_time - window
        for member in list(zset.keys()):
            if zset[member] <= cutoff:
                del zset[member]
        if len(zset) < limit:
            member_id = f"{current_time}:{len(zset)+1}"
            zset[member_id] = current_time
            return [1, 0]
        oldest = min(zset.values()) if zset else current_time
        retry_after = int(max(0, oldest + window - current_time)) or window
        return [0, retry_after]


class InMemoryAsyncRedis:
    """Asyncio-friendly wrapper around the in-memory Redis core."""

    def __init__(self, decode_responses: bool = True):
        self._core = _InMemoryRedisCore(decode_responses=decode_responses)
        self._lock = asyncio.Lock()

    async def ping(self):
        return self._core.ping()

    async def close(self):
        self._core.close()

    async def xlen(self, name: str) -> int:
        async with self._lock:
            return self._core.xlen(name)

    async def xadd(self, name: str, fields: dict[str, Any]):
        async with self._lock:
            return self._core.xadd(name, fields)

    async def xrange(self, name: str, *args, **kwargs):
        async with self._lock:
            return self._core.xrange(name, *args, **kwargs)

    async def xrevrange(self, name: str, *args, **kwargs):
        async with self._lock:
            return self._core.xrevrange(name, *args, **kwargs)

    async def xdel(self, name: str, entry_id: str) -> int:
        async with self._lock:
            return self._core.xdel(name, entry_id)

    async def xgroup_create(self, name: str, group: str, id: str = "0") -> None:
        async with self._lock:
            self._core.xgroup_create(name, group, id)

    async def xreadgroup(
        self,
        groupname: str,
        consumername: str,
        streams: dict[str, str],
        count: int | None = None,
        block: int | None = None,
        noack: bool = False,
    ):
        async with self._lock:
            return self._core.xreadgroup(
                groupname,
                consumername,
                streams,
                count=count,
                block=block,
                noack=noack,
            )

    async def sadd(self, key: str, member: str) -> int:
        async with self._lock:
            return self._core.sadd(key, member)

    async def srem(self, key: str, member: str) -> int:
        async with self._lock:
            return self._core.srem(key, member)

    async def smembers(self, key: str):
        async with self._lock:
            return self._core.smembers(key)

    async def zadd(self, key: str, mapping: dict[str, float]) -> None:
        async with self._lock:
            self._core.zadd(key, mapping)

    async def zrem(self, key: str, member: str) -> int:
        async with self._lock:
            return self._core.zrem(key, member)

    async def zremrangebyscore(self, key: str, minimum: float, maximum: float) -> int:
        async with self._lock:
            return self._core.zremrangebyscore(key, float(minimum), float(maximum))

    async def zcard(self, key: str) -> int:
        async with self._lock:
            return self._core.zcard(key)

    async def zrange(self, key: str, start: int, stop: int) -> list[str]:
        async with self._lock:
            return self._core.zrange(key, start, stop)

    async def zscore(self, key: str, member: str) -> float | None:
        async with self._lock:
            return self._core.zscore(key, member)

    async def zincrby(self, key: str, amount: float, member: str) -> float:
        async with self._lock:
            return self._core.zincrby(key, amount, member)

    async def hset(self, key: str, mapping: dict[str, Any]) -> int:
        async with self._lock:
            return self._core.hset(key, mapping)

    async def hgetall(self, key: str) -> dict[str, str]:
        async with self._lock:
            return self._core.hgetall(key)

    async def hget(self, key: str, field: str) -> str | None:
        async with self._lock:
            return self._core.hget(key, field)

    async def hincrby(self, key: str, field: str, amount: int = 1) -> int:
        async with self._lock:
            return self._core.hincrby(key, field, amount)

    async def set(self, key: str, value: Any, ex: int | None = None) -> None:
        async with self._lock:
            self._core.set(key, value, ex=ex)

    async def setex(self, key: str, seconds: int, value: Any) -> None:
        async with self._lock:
            self._core.setex(key, seconds, value)

    async def get(self, key: str) -> str | None:
        async with self._lock:
            return self._core.get(key)

    async def delete(self, *keys: str) -> int:
        async with self._lock:
            return self._core.delete(*keys)

    async def expire(self, key: str, seconds: int) -> None:
        async with self._lock:
            self._core.expire(key, seconds)

    async def ttl(self, key: str) -> int:
        async with self._lock:
            return self._core.ttl(key)

    async def incr(self, key: str) -> int:
        async with self._lock:
            return self._core.incr(key)

    async def incrby(self, key: str, amount: int) -> int:
        async with self._lock:
            return self._core.incrby(key, amount)

    async def scan(self, cursor: int = 0, match: str | None = None, count: int | None = None):
        async with self._lock:
            return self._core.scan(cursor, match, count)

    async def keys(self, pattern: str) -> list[str]:
        async with self._lock:
            return self._core.keys(pattern)

    async def dbsize(self) -> int:
        async with self._lock:
            return self._core.dbsize()

    async def info(self, section: str | None = None) -> dict[str, Any]:
        async with self._lock:
            return self._core.info(section)

    async def script_load(self, script: str) -> str:
        async with self._lock:
            return self._core.script_load(script)

    async def evalsha(self, sha: str, num_keys: int, *args) -> list[Any]:
        async with self._lock:
            return self._core.evalsha(sha, num_keys, *args)

    async def eval(self, script: str, num_keys: int, *args) -> list[Any]:
        async with self._lock:
            return self._core.eval(script, num_keys, *args)

    def pipeline(self):
        return InMemoryAsyncPipeline(self)


class InMemoryAsyncPipeline:
    def __init__(self, redis_client: InMemoryAsyncRedis):
        self._redis = redis_client
        self._ops: list[tuple[str, tuple[Any, ...]]] = []

    def incr(self, key: str, amount: int = 1):
        self._ops.append(("incrby", (key, amount)))
        return self

    def incrby(self, key: str, amount: int):
        self._ops.append(("incrby", (key, amount)))
        return self

    def expire(self, key: str, seconds: int):
        self._ops.append(("expire", (key, seconds)))
        return self

    async def execute(self):
        results = []
        for name, args in self._ops:
            method = getattr(self._redis, name)
            result = await method(*args)
            results.append(result)
        self._ops.clear()
        return results


class InMemorySyncRedis:
    """Synchronous wrapper around the in-memory Redis core."""

    def __init__(self, decode_responses: bool = True):
        self._core = _InMemoryRedisCore(decode_responses=decode_responses)
        self._lock = threading.Lock()

    def ping(self):
        with self._lock:
            return self._core.ping()

    def close(self):
        with self._lock:
            self._core.close()

    def pipeline(self):
        return InMemorySyncPipeline(self)

    def xlen(self, name: str) -> int:
        with self._lock:
            return self._core.xlen(name)

    def xadd(self, name: str, fields: dict[str, Any]):
        with self._lock:
            return self._core.xadd(name, fields)

    def xrange(self, name: str, *args, **kwargs):
        with self._lock:
            return self._core.xrange(name, *args, **kwargs)

    def xrevrange(self, name: str, *args, **kwargs):
        with self._lock:
            return self._core.xrevrange(name, *args, **kwargs)

    def xdel(self, name: str, entry_id: str) -> int:
        with self._lock:
            return self._core.xdel(name, entry_id)

    def xgroup_create(self, name: str, group: str, id: str = "0") -> None:
        with self._lock:
            self._core.xgroup_create(name, group, id)

    def xreadgroup(
        self,
        groupname: str,
        consumername: str,
        streams: dict[str, str],
        count: int | None = None,
        block: int | None = None,
        noack: bool = False,
    ):
        with self._lock:
            return self._core.xreadgroup(
                groupname,
                consumername,
                streams,
                count=count,
                block=block,
                noack=noack,
            )

    def sadd(self, key: str, member: str) -> int:
        with self._lock:
            return self._core.sadd(key, member)

    def srem(self, key: str, member: str) -> int:
        with self._lock:
            return self._core.srem(key, member)

    def smembers(self, key: str):
        with self._lock:
            return self._core.smembers(key)

    def zadd(self, key: str, mapping: dict[str, float]) -> None:
        with self._lock:
            self._core.zadd(key, mapping)

    def zremrangebyscore(self, key: str, minimum: float, maximum: float) -> int:
        with self._lock:
            return self._core.zremrangebyscore(key, float(minimum), float(maximum))

    def zincrby(self, key: str, amount: float, member: str) -> float:
        with self._lock:
            return self._core.zincrby(key, amount, member)

    def hset(self, key: str, mapping: dict[str, Any]) -> int:
        with self._lock:
            return self._core.hset(key, mapping)

    def hget(self, key: str, field: str) -> str | None:
        with self._lock:
            return self._core.hget(key, field)

    def hgetall(self, key: str) -> dict[str, str]:
        with self._lock:
            return self._core.hgetall(key)

    def hincrby(self, key: str, field: str, amount: int = 1) -> int:
        with self._lock:
            return self._core.hincrby(key, field, amount)

    def setex(self, key: str, seconds: int, value: Any) -> None:
        with self._lock:
            self._core.setex(key, seconds, value)

    def ttl(self, key: str) -> int:
        with self._lock:
            return self._core.ttl(key)

    def scan(self, cursor: int = 0, match: str | None = None, count: int | None = None):
        with self._lock:
            return self._core.scan(cursor, match, count)

    def keys(self, pattern: str) -> list[str]:
        with self._lock:
            return self._core.keys(pattern)

    def dbsize(self) -> int:
        with self._lock:
            return self._core.dbsize()

    def info(self, section: str | None = None) -> dict[str, Any]:
        with self._lock:
            return self._core.info(section)

    def script_load(self, script: str) -> str:
        with self._lock:
            return self._core.script_load(script)

    def evalsha(self, sha: str, num_keys: int, *args) -> list[Any]:
        with self._lock:
            return self._core.evalsha(sha, num_keys, *args)

    def eval(self, script: str, num_keys: int, *args) -> list[Any]:
        with self._lock:
            return self._core.eval(script, num_keys, *args)

    def publish(self, channel: str, message: str) -> int:
        return 0

    # Expose subset of operations needed by the codebase
    def incr(self, key: str) -> int:
        with self._lock:
            return self._core.incr(key)

    def incrby(self, key: str, amount: int) -> int:
        with self._lock:
            return self._core.incrby(key, amount)

    def expire(self, key: str, seconds: int):
        with self._lock:
            self._core.expire(key, seconds)

    def get(self, key: str):
        with self._lock:
            return self._core.get(key)

    # Minimal ZSET API for tests
    def zcard(self, key: str) -> int:
        with self._lock:
            return self._core.zcard(key)

    def zrange(self, key: str, start: int, stop: int) -> list[str]:
        with self._lock:
            return self._core.zrange(key, start, stop)

    def zscore(self, key: str, member: str) -> float | None:
        with self._lock:
            return self._core.zscore(key, member)

    def zrem(self, key: str, member: str) -> int:
        """Remove a member from a sorted set; returns number of removed members (0 or 1).

        Matches the signature/behavior of the in-memory core and async wrapper.
        """
        with self._lock:
            return self._core.zrem(key, member)

    def set(self, key: str, value: Any, ex: int | None = None):
        with self._lock:
            self._core.set(key, value, ex=ex)

    def delete(self, *keys: str) -> int:
        with self._lock:
            return self._core.delete(*keys)


class InMemorySyncPipeline:
    def __init__(self, redis_client: InMemorySyncRedis):
        self._redis = redis_client
        self._ops: list[tuple[str, tuple[Any, ...]]] = []

    def incr(self, key: str, amount: int = 1):
        self._ops.append(("incrby", (key, amount)))
        return self

    def incrby(self, key: str, amount: int):
        self._ops.append(("incrby", (key, amount)))
        return self

    def expire(self, key: str, seconds: int):
        self._ops.append(("expire", (key, seconds)))
        return self

    def execute(self):
        results = []
        for name, args in self._ops:
            method = getattr(self._redis, name)
            results.append(method(*args))
        self._ops.clear()
        return results


async def ensure_async_client_closed(client) -> None:
    """
    Best-effort close for asyncio Redis clients (real or fake).
    """
    if client is None:
        return
    try:
        close = getattr(client, "close", None)
        if close is None:
            return
        result = close()
        if inspect.isawaitable(result):
            await result
    except _REDIS_NONCRITICAL_EXCEPTIONS:
        pass


def ensure_sync_client_closed(client) -> None:
    """
    Best-effort close for synchronous Redis clients (real or fake).
    """
    if client is None:
        return
    try:
        close = getattr(client, "close", None)
        if close:
            close()
    except _REDIS_NONCRITICAL_EXCEPTIONS:
        pass
