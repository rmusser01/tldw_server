"""
MCP Protocol implementation for unified module

Implements JSON-RPC 2.0 with enhanced error handling and request routing.
"""

import asyncio
import hashlib
import hmac
import json
import secrets
import uuid
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from enum import IntEnum
from pathlib import Path
from typing import Any, Callable, Literal, Optional, Union

from pydantic import BaseModel, Field

try:
    from pydantic import field_validator, model_validator  # v2
except ImportError:  # Fallback for v1
    from pydantic import validator as field_validator  # type: ignore
    try:
        from pydantic import root_validator as model_validator  # type: ignore
    except ImportError:
        model_validator = None  # type: ignore
import contextlib
import inspect
import re
import time
from collections import OrderedDict

from loguru import logger

from tldw_Server_API.app.core.Infrastructure.redis_factory import create_async_redis_client
from tldw_Server_API.app.core.Metrics.telemetry import get_telemetry_manager
from tldw_Server_API.app.core.testing import is_truthy

from .auth.authnz_rbac import Action, Resource, get_rbac_policy
from .auth.rate_limiter import RateLimitExceeded, get_rate_limiter
from .config import get_config
from .modules.base import BaseModule
from .modules.registry import get_module_registry
from .monitoring.metrics import get_metrics_collector

try:  # pragma: no cover - optional dependency
    from redis.exceptions import RedisError
except ImportError:  # pragma: no cover - redis not installed
    class RedisError(Exception):
        """Fallback RedisError when redis-py is unavailable."""
        pass


# Redis exceptions can include connection URLs and credentials; keep logs sanitized.
def _redact_redis_error(exc: Exception) -> str:
    return f"{exc.__class__.__name__}: Redis connection error - details redacted"


# JSON-RPC 2.0 Error Codes
class ErrorCode(IntEnum):
    """Standard JSON-RPC 2.0 error codes"""
    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603

    # Custom error codes (must be -32000 to -32099)
    AUTHENTICATION_ERROR = -32000
    AUTHORIZATION_ERROR = -32001
    RATE_LIMIT_ERROR = -32002
    MODULE_ERROR = -32003
    TIMEOUT_ERROR = -32004


class InvalidParamsException(Exception):
    """Raised when tool parameters fail validation or validators are missing for write tools."""
    pass


class GovernanceDeniedError(PermissionError):
    """Permission error carrying structured governance decision details."""

    def __init__(self, message: str, governance: Optional[dict[str, Any]] = None):
        super().__init__(message)
        self.governance = governance or {}


class ApprovalRequiredError(PermissionError):
    """Permission error carrying structured MCP Hub approval request details."""

    def __init__(self, message: str, approval: Optional[dict[str, Any]] = None):
        super().__init__(message)
        self.approval = approval or {}


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


class MCPRequest(BaseModel):
    """MCP request following JSON-RPC 2.0 specification"""
    jsonrpc: Literal["2.0"] = Field(default="2.0")
    method: str = Field(..., min_length=1, max_length=100)
    params: Optional[dict[str, Any]] = None
    id: Optional[Union[str, int]] = None

    @field_validator("method")
    @classmethod
    def validate_method(cls, v):
        """Validate method name"""
        # Prevent potential injection attacks
        if any(char in v for char in ["'", '"', ';', '--', '/*', '*/']):
            raise ValueError("Invalid characters in method name")
        return v

    @field_validator("params")
    @classmethod
    def validate_params(cls, v):
        """Validate and sanitize parameters"""
        if v is not None and not isinstance(v, dict):
            raise ValueError("Params must be a dictionary")
        return v


class MCPError(BaseModel):
    """MCP error structure"""
    code: int
    message: str
    data: Optional[Any] = None


class MCPResponse(BaseModel):
    """MCP response following JSON-RPC 2.0 specification"""
    jsonrpc: Literal["2.0"] = Field(default="2.0")
    result: Optional[Any] = None
    error: Optional[MCPError] = None
    id: Optional[Union[str, int]] = None

    if model_validator is not None:
        @model_validator(mode="after")
        def _validate_error_result(self):
            """Ensure either result or error is set, not both"""
            if self.error is not None and self.result is not None:
                raise ValueError("Response cannot have both result and error")
            return self


class RequestContext:
    """Context for request processing"""
    def __init__(
        self,
        request_id: str,
        user_id: Optional[str] = None,
        client_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None
    ):
        self.request_id = request_id
        self.user_id = user_id
        self.client_id = client_id
        self.session_id = session_id
        self.metadata = metadata or {}
        self.start_time = datetime.now(timezone.utc)
        # Derive per-user db paths (read-only) if possible
        self.db_paths: dict[str, str] = {}
        try:
            if self.user_id is not None:
                # Attempt to parse an integer user id (expected by DatabasePaths)
                uid_int = int(str(self.user_id))
                from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
                paths = DatabasePaths.get_all_user_db_paths(uid_int)
                # Convert Paths to strings for downstream use
                self.db_paths = {k: str(v) for k, v in paths.items()}
        except _MCP_PROTOCOL_NONCRITICAL_EXCEPTIONS as _e:
            # Non-fatal: leave db_paths empty when user id is not numeric or any failure occurs
            pass
        # Build a bound logger for this request
        self.logger = logger.bind(
            request_id=request_id,
            user_id=user_id,
            client_id=client_id,
            session_id=session_id,
        )


@dataclass(frozen=True, slots=True)
class PreparedToolCall:
    """Prepared tool execution context reused by nested tool orchestration."""

    tool_name: str
    tool_args: Any
    module: BaseModule
    module_id: Optional[str]
    tool_def: Optional[dict[str, Any]]
    is_write: Optional[bool]
    normalized_idempotency_key: Optional[str]
    idempotency_cache_key: Optional[str]
    arguments_hash: Optional[str]
    context_fingerprint: str
    integrity_tag: str
    context: RequestContext


class IdempotencyManager:
    """Idempotency manager with Redis backing and local lock fallback."""

    def __init__(self) -> None:
        self._local_cache: OrderedDict[str, tuple[float, dict[str, Any]]] = OrderedDict()
        self._local_bindings: OrderedDict[str, tuple[float, str]] = OrderedDict()
        self._local_locks: dict[str, asyncio.Lock] = {}
        self._local_guard = asyncio.Lock()
        self._redis_client: Any | None = None
        self._redis_ready = False
        self._redis_attempted = False
        self._redis_guard = asyncio.Lock()

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
                self._redis_client = await create_async_redis_client(
                    preferred_url=url,
                    decode_responses=True,
                    fallback_to_fake=False,
                    context="mcp_idempotency",
                    redis_kwargs=params,
                )
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


class MCPProtocol:
    """
    MCP Protocol handler with enhanced security and error handling.

    Features:
    - JSON-RPC 2.0 compliance
    - Request validation and sanitization
    - Authentication and authorization
    - Rate limiting
    - Request routing
    - Error handling with proper codes
    - Request tracing
    """

    def __init__(self):
        self.module_registry = get_module_registry()
        self.rbac_policy = get_rbac_policy()
        self.rate_limiter = get_rate_limiter()
        self.protocol_version = "2024-11-05"
        self.metrics = get_metrics_collector()
        # Strict tool name validation regex
        self._tool_name_re = re.compile(r'^[A-Za-z0-9_.:-]{1,100}$')
        # Idempotency manager for write-capable tools
        self._idempotency = IdempotencyManager()
        # Integrity secret for prepared tool call execution
        self._prepared_call_secret = secrets.token_bytes(32)
        # Governance preflight state
        self._governance_service: Any | None = None
        self._governance_store: Any | None = None
        self._governance_lock = asyncio.Lock()

        # Method handlers — telemetry is accessed via a property so that
        # a shutdown/re-init cycle is picked up automatically.
        self.handlers: dict[str, Callable] = {
            "initialize": self._handle_initialize,
            "ping": self._handle_ping,
            "tools/list": self._handle_tools_list,
            "tools/call": self._handle_tools_call,
            "resources/list": self._handle_resources_list,
            "resources/read": self._handle_resources_read,
            "prompts/list": self._handle_prompts_list,
            "prompts/get": self._handle_prompts_get,
            "modules/list": self._handle_modules_list,
            "modules/health": self._handle_modules_health,
        }

        logger.info("MCP Protocol handler initialized")

    @property
    def telemetry(self):
        """Always return the *current* global telemetry manager so that a
        shutdown/re-init cycle is picked up automatically."""
        return get_telemetry_manager()

    async def _rbac_check(self, user_id: Optional[str], resource: Resource, action: Action, resource_id: Optional[str] = None) -> bool:
        if not user_id:
            return False
        fn = getattr(self.rbac_policy, "check_permission", None)
        if not fn:
            return False
        try:
            if inspect.iscoroutinefunction(fn):
                return await fn(user_id, resource, action, resource_id)
            return fn(user_id, resource, action, resource_id)
        except _MCP_PROTOCOL_NONCRITICAL_EXCEPTIONS:
            return False

    def _scoped_permissions(self, context: RequestContext) -> list[str]:
        metadata = getattr(context, "metadata", {})
        if not isinstance(metadata, dict):
            return []
        raw = metadata.get("permissions") or []
        if isinstance(raw, str):
            return [raw]
        if isinstance(raw, list):
            return [str(item) for item in raw if isinstance(item, str)]
        return []

    def _mcp_scopes(self, context: RequestContext) -> list[str]:
        scopes: list[str] = []
        for scope in self._scoped_permissions(context):
            try:
                if scope.lower().startswith("mcp:"):
                    scopes.append(scope)
            except _MCP_PROTOCOL_NONCRITICAL_EXCEPTIONS:
                continue
        return scopes

    def _api_key_scopes(self, context: RequestContext) -> Optional[set[str]]:
        """Return normalized API key scopes when present on the request context."""
        metadata = getattr(context, "metadata", {})
        if not isinstance(metadata, dict):
            return None
        raw = metadata.get("api_key_scopes")
        if raw is None:
            return None
        try:
            from tldw_Server_API.app.core.AuthNZ.api_key_manager import normalize_scope
        except _MCP_PROTOCOL_NONCRITICAL_EXCEPTIONS:
            normalize_scope = None  # type: ignore

        if normalize_scope is not None:
            try:
                return set(normalize_scope(raw))
            except _MCP_PROTOCOL_NONCRITICAL_EXCEPTIONS:
                pass

        if isinstance(raw, str):
            return {raw.strip().lower()} if raw.strip() else set()
        if isinstance(raw, list):
            return {str(item).strip().lower() for item in raw if str(item).strip()}
        return set()

    def _api_key_scope_level(self, context: RequestContext) -> Optional[str]:
        scopes = self._api_key_scopes(context)
        if not scopes:
            return None
        if "admin" in scopes or "service" in scopes:
            return "admin"
        if "write" in scopes:
            return "write"
        if "read" in scopes:
            return "read"
        return None

    def _api_key_allows(self, context: RequestContext, *, is_write: Optional[bool] = None) -> bool:
        """Gate MCP operations by API key scopes when present."""
        level = self._api_key_scope_level(context)
        if level is None:
            return True
        if level == "admin":
            return True
        if is_write is None:
            return level in {"read", "write"}
        if is_write:
            return level == "write"
        return level in {"read", "write"}

    def _scope_matches(self, scope: str, resource_kind: str, identifier: Optional[str]) -> bool:
        scope = scope.strip().lower()
        if not scope.startswith("mcp:"):
            return False
        parts = scope.split(":")
        if len(parts) == 2 and parts[1] == "*":
            return True
        if len(parts) < 3:
            return False
        kind = parts[1]
        value = ":".join(parts[2:])
        if kind == "*":
            return True
        if kind != resource_kind:
            return False
        if value in {"*", ""}:
            return True
        if identifier is None:
            return False
        return value == identifier.lower()

    def _scope_allows(self, context: RequestContext, resource_kind: str, identifier: Optional[str]) -> bool:
        scopes = self._mcp_scopes(context)
        if not scopes:
            return True
        identifier_norm = identifier.lower() if isinstance(identifier, str) else None
        if identifier_norm is None:
            # Allow listing/browsing when any scoped permission exists for this resource kind.
            for scope in scopes:
                try:
                    parts = scope.strip().lower().split(":")
                except _MCP_PROTOCOL_NONCRITICAL_EXCEPTIONS:
                    continue
                if len(parts) >= 2 and parts[0] == "mcp":
                    if parts[1] == "*" or parts[1] == resource_kind:
                        return True
        return any(self._scope_matches(scope, resource_kind, identifier_norm) for scope in scopes)

    async def _has_module_permission(self, context: RequestContext, module_id: Optional[str]) -> bool:
        module_id_norm = module_id or ""
        if not await self._rbac_check(context.user_id, Resource.MODULE, Action.READ, module_id_norm):
            return False
        return self._scope_allows(context, Resource.MODULE.value, module_id_norm or None)

    async def _has_tool_permission(self, context: RequestContext, tool_name: str, *, is_write: Optional[bool] = None) -> bool:
        if not await self._rbac_check(context.user_id, Resource.TOOL, Action.EXECUTE, tool_name):
            return False
        if not self._scope_allows(context, Resource.TOOL.value, tool_name):
            return False
        return self._api_key_allows(context, is_write=is_write)

    async def _has_resource_permission(self, context: RequestContext, resource_uri: str, module_id: Optional[str]) -> bool:
        if await self._rbac_check(context.user_id, Resource.RESOURCE, Action.READ, resource_uri):
            return self._scope_allows(context, Resource.RESOURCE.value, resource_uri)
        if await self._has_module_permission(context, module_id):
            return self._scope_allows(context, Resource.RESOURCE.value, resource_uri)
        return False

    async def _has_prompt_permission(self, context: RequestContext, prompt_name: str, module_id: Optional[str]) -> bool:
        if await self._rbac_check(context.user_id, Resource.PROMPT, Action.READ, prompt_name):
            return self._scope_allows(context, Resource.PROMPT.value, prompt_name)
        if await self._has_module_permission(context, module_id):
            return self._scope_allows(context, Resource.PROMPT.value, prompt_name)
        return False

    @staticmethod
    def _hash_arguments(arguments: dict[str, Any]) -> Optional[str]:
        try:
            payload = json.dumps(arguments or {}, sort_keys=True, default=str).encode("utf-8")
            import hashlib
            return hashlib.sha256(payload).hexdigest()
        except _MCP_PROTOCOL_NONCRITICAL_EXCEPTIONS:
            return None

    async def _resolve_tool_definition(
        self,
        module: BaseModule,
        tool_name: str,
    ) -> Optional[dict[str, Any]]:
        """Resolve a tool definition for a module/tool pair."""
        try:
            get_def = getattr(module, "get_tool_def", None)
            if callable(get_def):
                tool_def = await get_def(tool_name)  # type: ignore[misc]
                if isinstance(tool_def, dict):
                    return tool_def
            tool_defs = await module.get_tools()
            for candidate in tool_defs:
                if isinstance(candidate, dict) and candidate.get("name") == tool_name:
                    return candidate
        except _MCP_PROTOCOL_NONCRITICAL_EXCEPTIONS:
            return None
        return None

    def _classify_write_tool_call(
        self,
        module: BaseModule,
        tool_name: str,
        tool_args: Any,
        tool_def: Optional[dict[str, Any]],
    ) -> Optional[bool]:
        """Best-effort write classification using per-call module hook."""
        try:
            normalized_args = tool_args if isinstance(tool_args, dict) else {}
            return module.is_write_tool_call(tool_name, normalized_args, tool_def=tool_def)
        except _MCP_PROTOCOL_NONCRITICAL_EXCEPTIONS:
            return None

    def _resolve_write_classification(
        self,
        module: BaseModule,
        tool_name: str,
        tool_args: Any,
        tool_def: Optional[dict[str, Any]],
        *,
        fallback_to_name_heuristic: bool,
    ) -> bool:
        """Resolve write classification with optional legacy fallback."""
        is_write = self._classify_write_tool_call(module, tool_name, tool_args, tool_def)
        if is_write is not None:
            return bool(is_write)
        if fallback_to_name_heuristic:
            return bool(re.search(r"(ingest|update|delete|create|import)", str(tool_name).lower()))
        return False

    @staticmethod
    def _strip_forbidden_tool_argument_overrides(tool_args: dict[str, Any]) -> dict[str, Any]:
        """Remove tool argument fields that could override request ownership/db scope."""
        forbidden = {"user_id", "db_path", "db_paths", "chacha_db", "media_db", "prompts_db"}
        sanitized = dict(tool_args)
        for key in forbidden:
            sanitized.pop(key, None)
        return sanitized

    def _harden_and_sanitize_tool_arguments(
        self,
        module: BaseModule,
        tool_args: Any,
    ) -> Any:
        """Normalize tool arguments before policy and execution checks."""
        if not isinstance(tool_args, dict):
            return tool_args
        hardened_args = self._strip_forbidden_tool_argument_overrides(tool_args)
        try:
            return module.sanitize_input(hardened_args)
        except _MCP_PROTOCOL_NONCRITICAL_EXCEPTIONS as san_err:
            raise InvalidParamsException(f"Invalid arguments: {str(san_err)}") from san_err

    def _prepared_tool_call_payload(
        self,
        *,
        tool_name: str,
        module_id: Optional[str],
        is_write: Optional[bool],
        idempotency_cache_key: Optional[str],
        arguments_hash: Optional[str],
        context_fingerprint: str,
    ) -> bytes:
        payload = {
            "tool_name": str(tool_name),
            "module_id": str(module_id or ""),
            "is_write": bool(is_write),
            "idempotency_cache_key": str(idempotency_cache_key or ""),
            "arguments_hash": str(arguments_hash or ""),
            "context_fingerprint": str(context_fingerprint or ""),
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")

    def _build_prepared_tool_call_integrity_tag(
        self,
        *,
        tool_name: str,
        module_id: Optional[str],
        is_write: Optional[bool],
        idempotency_cache_key: Optional[str],
        arguments_hash: Optional[str],
        context_fingerprint: str,
    ) -> str:
        payload = self._prepared_tool_call_payload(
            tool_name=tool_name,
            module_id=module_id,
            is_write=is_write,
            idempotency_cache_key=idempotency_cache_key,
            arguments_hash=arguments_hash,
            context_fingerprint=context_fingerprint,
        )
        return hmac.new(self._prepared_call_secret, payload, digestmod="sha256").hexdigest()

    def _verify_prepared_tool_call_integrity(
        self,
        prepared: PreparedToolCall,
    ) -> None:
        if not isinstance(prepared.tool_name, str) or not self._tool_name_re.match(prepared.tool_name):
            raise InvalidParamsException("Prepared tool call integrity check failed: invalid tool name")

        expected_hash = self._hash_arguments(prepared.tool_args if isinstance(prepared.tool_args, dict) else {})
        if expected_hash != prepared.arguments_hash:
            raise InvalidParamsException("Prepared tool call integrity check failed: argument fingerprint mismatch")

        expected_context_fingerprint = self._fingerprint_request_context(prepared.context)
        if expected_context_fingerprint != prepared.context_fingerprint:
            raise InvalidParamsException("Prepared tool call integrity check failed: context fingerprint mismatch")

        expected_write = self._resolve_write_classification(
            prepared.module,
            prepared.tool_name,
            prepared.tool_args,
            prepared.tool_def,
            fallback_to_name_heuristic=True,
        )
        if bool(expected_write) != bool(prepared.is_write):
            raise InvalidParamsException("Prepared tool call integrity check failed: write classification mismatch")

        expected_tag = self._build_prepared_tool_call_integrity_tag(
            tool_name=prepared.tool_name,
            module_id=prepared.module_id,
            is_write=prepared.is_write,
            idempotency_cache_key=prepared.idempotency_cache_key,
            arguments_hash=prepared.arguments_hash,
            context_fingerprint=prepared.context_fingerprint,
        )
        if not hmac.compare_digest(prepared.integrity_tag, expected_tag):
            raise InvalidParamsException("Prepared tool call integrity check failed: signature mismatch")

    def _fingerprint_request_context(self, context: RequestContext) -> str:
        payload = {
            "request_id": str(context.request_id or ""),
            "user_id": str(context.user_id or ""),
            "client_id": str(context.client_id or ""),
            "session_id": str(context.session_id or ""),
            "metadata": self._context_json_safe(getattr(context, "metadata", {})),
            "db_paths": self._context_json_safe(getattr(context, "db_paths", {})),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _context_json_safe(self, value: Any) -> Any:
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {str(key): self._context_json_safe(item) for key, item in sorted(value.items(), key=lambda kv: str(kv[0]))}
        if isinstance(value, (list, tuple, set)):
            return [self._context_json_safe(item) for item in value]
        if is_dataclass(value):
            return self._context_json_safe(asdict(value))
        return str(value)

    @staticmethod
    def _normalize_idempotency_key(
        params: dict[str, Any],
        idempotency_key: str | None = None,
    ) -> Optional[str]:
        """Normalize idempotency key from explicit argument or request params."""
        raw_idempotency_key = idempotency_key
        if raw_idempotency_key is None:
            raw_idempotency_key = params.get("idempotencyKey")
            if raw_idempotency_key is None:
                raw_idempotency_key = params.get("idempotency_key")
        if raw_idempotency_key is None:
            arguments = params.get("arguments")
            if isinstance(arguments, dict):
                raw_idempotency_key = arguments.get("idempotencyKey")
                if raw_idempotency_key is None:
                    raw_idempotency_key = arguments.get("idempotency_key")

        if raw_idempotency_key is None:
            return None
        if not isinstance(raw_idempotency_key, str):
            raise InvalidParamsException("idempotencyKey must be a string")

        normalized = raw_idempotency_key.strip()
        if not normalized:
            raise InvalidParamsException("idempotencyKey must not be empty")
        return normalized

    def _audit_tool_event(
        self,
        context: RequestContext,
        tool_name: str,
        module_id: Optional[str],
        status: str,
        duration_ms: float,
        arguments_hash: Optional[str],
        error: Optional[Exception] = None,
    ) -> None:
        try:
            log = logger.bind(
                audit=True,
                request_id=context.request_id,
                user_id=context.user_id,
                client_id=context.client_id,
                session_id=context.session_id,
                tool=tool_name,
                module=module_id or "unknown",
                duration_ms=round(duration_ms, 2),
                arguments_hash=arguments_hash,
                status=status,
            )
            if error:
                log.error("MCP tool execution failed", error_type=error.__class__.__name__, error_message=str(error)[:200])
            else:
                log.info("MCP tool executed")
        except _MCP_PROTOCOL_NONCRITICAL_EXCEPTIONS:
            pass

    @staticmethod
    def _governance_preflight_bypassed(tool_name: str, context: RequestContext) -> bool:
        if str(tool_name or "").startswith("governance."):
            return True

        metadata = getattr(context, "metadata", None)
        if not isinstance(metadata, dict):
            return False

        raw = metadata.get("governance_bypass")
        if isinstance(raw, bool):
            return raw
        if isinstance(raw, (int, float)):
            return bool(raw)
        if isinstance(raw, str):
            return is_truthy(raw)
        return False

    @staticmethod
    def _governance_summary(tool_name: str, tool_args: dict[str, Any]) -> str:
        rendered_args = ""
        try:
            rendered_args = json.dumps(tool_args or {}, sort_keys=True, default=str)
        except _MCP_PROTOCOL_NONCRITICAL_EXCEPTIONS:
            rendered_args = str(tool_args)
        if len(rendered_args) > 1200:
            rendered_args = rendered_args[:1200]
        return f"tool={tool_name}; args={rendered_args}"

    @staticmethod
    def _resolve_governance_category(tool_name: str, tool_def: Optional[dict[str, Any]]) -> str:
        try:
            if isinstance(tool_def, dict):
                meta = tool_def.get("metadata")
                if isinstance(meta, dict):
                    category = str(meta.get("category") or "").strip().lower()
                    if category:
                        return category
        except _MCP_PROTOCOL_NONCRITICAL_EXCEPTIONS:
            pass

        if isinstance(tool_name, str) and "." in tool_name:
            prefix = tool_name.split(".", 1)[0].strip().lower()
            if prefix:
                return prefix
        return "general"

    @staticmethod
    def _resolve_governance_rollout_mode(metadata: Optional[dict[str, Any]] = None) -> str:
        """Resolve governance rollout mode from metadata override and server config."""
        raw_mode = None
        if isinstance(metadata, dict):
            raw_mode = metadata.get("governance_rollout_mode")

        try:
            from tldw_Server_API.app.core import config as app_config

            return app_config.resolve_governance_rollout_mode(
                str(raw_mode) if raw_mode is not None else None
            )
        except _MCP_PROTOCOL_NONCRITICAL_EXCEPTIONS as exc:
            logger.debug(f"Unable to resolve governance rollout mode from config: {exc}")
            candidate = str(raw_mode or "").strip().lower()
            return candidate if candidate in {"off", "shadow", "enforce"} else "off"

    def _record_governance_check(
        self,
        *,
        surface: str,
        category: str,
        status: str,
        rollout_mode: str,
    ) -> None:
        """Emit one governance check metric entry, failing open on metric errors."""
        with contextlib.suppress(_MCP_PROTOCOL_NONCRITICAL_EXCEPTIONS):
            self.metrics.record_governance_check(
                surface=surface,
                category=category,
                status=status,
                rollout_mode=rollout_mode,
            )

    @classmethod
    def _serialize_governance_decision(cls, decision: Any) -> dict[str, Any]:
        if decision is None:
            return {}
        if isinstance(decision, dict):
            return {str(k): v for k, v in decision.items()}
        if is_dataclass(decision):
            return cls._serialize_governance_decision(asdict(decision))
        dump = getattr(decision, "model_dump", None)
        if callable(dump):
            try:
                dumped = dump()
                if isinstance(dumped, dict):
                    return {str(k): v for k, v in dumped.items()}
            except _MCP_PROTOCOL_NONCRITICAL_EXCEPTIONS:
                pass
        payload: dict[str, Any] = {}
        for key in ("action", "status", "category", "category_source", "fallback_reason", "matched_rules"):
            value = getattr(decision, key, None)
            if value is not None:
                payload[key] = value
        return payload

    async def _ensure_governance_service(self) -> Any | None:
        if self._governance_service is not None:
            return self._governance_service

        async with self._governance_lock:
            if self._governance_service is not None:
                return self._governance_service
            try:
                from tldw_Server_API.app.core.Governance.service import GovernanceService
                from tldw_Server_API.app.core.Governance.store import GovernanceStore
            except _MCP_PROTOCOL_NONCRITICAL_EXCEPTIONS as exc:
                logger.debug(f"MCP governance preflight unavailable (import failure): {exc}")
                return None

            try:
                cfg = get_config()
                configured_path = getattr(cfg, "governance_db_path", None)
                sqlite_path = str(configured_path or "Databases/governance.db")
                db_path = Path(sqlite_path).expanduser()
                db_path.parent.mkdir(parents=True, exist_ok=True)

                self._governance_store = GovernanceStore(sqlite_path=str(db_path))
                await self._governance_store.ensure_schema()
                self._governance_service = GovernanceService(store=self._governance_store)
                return self._governance_service
            except _MCP_PROTOCOL_NONCRITICAL_EXCEPTIONS as exc:
                logger.debug(f"MCP governance preflight disabled (service init failure): {exc}")
                self._governance_service = None
                self._governance_store = None
                return None

    async def _run_governance_preflight(
        self,
        *,
        tool_name: str,
        tool_args: dict[str, Any],
        tool_def: Optional[dict[str, Any]],
        context: RequestContext,
    ) -> Optional[dict[str, Any]]:
        if self._governance_preflight_bypassed(tool_name, context):
            return None

        metadata = context.metadata if isinstance(getattr(context, "metadata", None), dict) else {}
        rollout_mode = self._resolve_governance_rollout_mode(metadata)
        category = self._resolve_governance_category(tool_name, tool_def)

        if rollout_mode == "off":
            self._record_governance_check(
                surface="mcp_tool",
                category=category,
                status="unknown",
                rollout_mode=rollout_mode,
            )
            return {"status": "unknown", "rollout_mode": rollout_mode}

        service = await self._ensure_governance_service()
        if service is None:
            self._record_governance_check(
                surface="mcp_tool",
                category=category,
                status="error",
                rollout_mode=rollout_mode,
            )
            return None

        try:
            decision = await service.validate_change(
                surface="mcp_tool",
                summary=self._governance_summary(tool_name, tool_args),
                category=category,
                metadata=metadata,
            )
            payload = self._serialize_governance_decision(decision)
            payload.setdefault("rollout_mode", rollout_mode)
            if isinstance(context.metadata, dict):
                context.metadata["governance_preflight"] = payload
            action = str(payload.get("action") or payload.get("status") or "").strip().lower() or "unknown"
            self._record_governance_check(
                surface="mcp_tool",
                category=category,
                status=action,
                rollout_mode=rollout_mode,
            )
            if action == "deny" and rollout_mode == "enforce":
                raise GovernanceDeniedError(
                    "Permission denied by governance policy",
                    governance=payload,
                )
            return payload
        except GovernanceDeniedError:
            raise
        except _MCP_PROTOCOL_NONCRITICAL_EXCEPTIONS as exc:
            self._record_governance_check(
                surface="mcp_tool",
                category=category,
                status="error",
                rollout_mode=rollout_mode,
            )
            try:
                context.logger.debug(f"Governance preflight failed open: {exc}")
            except _MCP_PROTOCOL_NONCRITICAL_EXCEPTIONS:
                pass
            return None

    async def process_request(
        self,
        request: Union[dict[str, Any], list[dict[str, Any]], MCPRequest],
        context: Optional[RequestContext] = None
    ) -> Union[MCPResponse, list[MCPResponse], None]:
        """
        Process an MCP request and return response.

        Args:
            request: MCP request (dict or MCPRequest object)
            context: Request context with user/session info

        Returns:
            MCP response
        """
        # Support batch requests
        if isinstance(request, list):
            if not request:
                return self._error_response(
                    ErrorCode.INVALID_REQUEST,
                    "Invalid request: empty batch",
                    None,
                )
            responses: list[MCPResponse] = []
            for item in request:
                try:
                    resp = await self.process_request(item, context)
                    # Notifications return None; do not include in batch response
                    if isinstance(resp, MCPResponse):
                        responses.append(resp)
                except _MCP_PROTOCOL_NONCRITICAL_EXCEPTIONS as e:
                    # If parsing fails at top-level, try to include an error response for that item
                    try:
                        req_id = item.get("id") if isinstance(item, dict) else None
                    except _MCP_PROTOCOL_NONCRITICAL_EXCEPTIONS:
                        req_id = None
                    responses.append(self._error_response(ErrorCode.INVALID_REQUEST, str(e), req_id))
            # Per JSON-RPC, if the batch is empty or only notifications, return no response
            return responses if responses else None

        # Parse single request if dict
        if isinstance(request, dict):
            try:
                request = MCPRequest(**request)
            except _MCP_PROTOCOL_NONCRITICAL_EXCEPTIONS as e:
                req_id = request.get("id") if isinstance(request, dict) else None
                return self._error_response(
                    ErrorCode.INVALID_REQUEST,
                    f"Invalid request format: {str(e)}",
                    req_id
                )

        # Create context if not provided
        if context is None:
            context = RequestContext(
                request_id=str(uuid.uuid4()),
                client_id="unknown"
            )

        # Bound logger for this request
        log = context.logger
        # Log request (without params) and ensure secrets get redacted in any error paths
        log.info(
            f"MCP request: method={request.method}, user={context.user_id}, client={context.client_id}",
            extra={"audit": True}
        )

        start_ts = time.time()
        try:
            # Check rate limit (skip when ingress RG already enforced)
            skip_rate_limit = False
            try:
                if context.metadata and context.metadata.get("rg_ingress_enforced"):
                    skip_rate_limit = True
            except _MCP_PROTOCOL_NONCRITICAL_EXCEPTIONS as exc:
                log.debug(
                    "Failed to read rg_ingress_enforced from metadata; rate limit will be enforced",
                    error=str(exc),
                )
                skip_rate_limit = False
            if not skip_rate_limit:
                if context.user_id:
                    await self.rate_limiter.check_rate_limit(f"user:{context.user_id}")
                elif context.client_id:
                    await self.rate_limiter.check_rate_limit(f"client:{context.client_id}")

            # Validate JSON-RPC version
            if request.jsonrpc != "2.0":
                return self._error_response(
                    ErrorCode.INVALID_REQUEST,
                    "Invalid JSON-RPC version",
                    request.id
                )

            # If this is a tools/call, validate tool name early (before RBAC)
            try:
                if request.method == "tools/call":
                    _p = request.params or {}
                    _name = _p.get("name") if isinstance(_p, dict) else None
                    if not _name:
                        return self._error_response(
                            ErrorCode.INVALID_PARAMS,
                            "Tool name is required",
                            request.id,
                        )
                    if not isinstance(_name, str):
                        # Non-string name → invalid params
                        return self._error_response(
                            ErrorCode.INVALID_PARAMS,
                            "Invalid tool name",
                            request.id,
                        )
                    if not self._tool_name_re.match(_name):
                        # Regex violation treated as internal error per legacy expectation
                        return self._error_response(
                            ErrorCode.INTERNAL_ERROR,
                            "Invalid tool name",
                            request.id,
                        )
            except _MCP_PROTOCOL_NONCRITICAL_EXCEPTIONS:
                # Uniformly surface as INVALID_PARAMS for caller clarity
                return self._error_response(ErrorCode.INVALID_PARAMS, "Invalid tool name", request.id)

            # Find handler
            handler = self.handlers.get(request.method)
            if not handler:
                return self._error_response(
                    ErrorCode.METHOD_NOT_FOUND,
                    f"Method not found: {request.method}",
                    request.id
                )

            # Check authorization
            if not await self._check_authorization(request, context):
                # Provide a short hint for common denied operations
                hint_data = None
                try:
                    if request.method == "tools/call":
                        tool = (request.params or {}).get("name")
                        if tool:
                            hint_data = {
                                "hint": (
                                    f"Permission denied. Ask an admin to grant tools.execute:{tool} "
                                    f"or tools.execute:* to your role (Admin → Access Control)."
                                )
                            }
                except _MCP_PROTOCOL_NONCRITICAL_EXCEPTIONS:
                    hint_data = None

                return self._error_response(
                    ErrorCode.AUTHORIZATION_ERROR,
                    "Insufficient permissions",
                    request.id,
                    data=hint_data
                )

            # Execute handler within OTEL span
            start_exec = time.time()
            with self.telemetry.trace_context(
                "mcp.request",
                {
                    "mcp.method": request.method,
                    "mcp.request_id": str(request.id) if request.id is not None else "notification",
                    "mcp.user_id": str(context.user_id or ""),
                    "mcp.client_id": str(context.client_id or ""),
                    "mcp.session_id": str(context.session_id or ""),
                },
            ) as span:
                try:
                    result = await handler(request.params or {}, context)
                    span.set_attribute("mcp.status", "success")
                except _MCP_PROTOCOL_NONCRITICAL_EXCEPTIONS as _span_e:
                    sanitized = self._sanitize_exception_for_telemetry(_span_e)
                    span.set_attribute("mcp.status", "failure")
                    span.set_attribute("mcp.error_type", sanitized.__class__.__name__)
                    span.set_attribute("mcp.error_message", str(sanitized)[:200])
                    raise
                except Exception as _span_e:
                    sanitized = self._sanitize_exception_for_telemetry(_span_e)
                    span.set_attribute("mcp.status", "failure")
                    span.set_attribute("mcp.error_type", sanitized.__class__.__name__)
                    span.set_attribute("mcp.error_message", str(sanitized)[:200])
                    raise
                finally:
                    span.set_attribute("mcp.duration_ms", max(0.0, (time.time() - start_exec) * 1000.0))

            # Log success and record metrics
            elapsed = (datetime.now(timezone.utc) - context.start_time).total_seconds()
            log.info(
                f"MCP request completed: method={request.method}, "
                f"elapsed={elapsed:.3f}s",
                extra={"audit": True}
            )
            with contextlib.suppress(_MCP_PROTOCOL_NONCRITICAL_EXCEPTIONS):
                self.metrics.record_request(method=request.method, duration=elapsed, status="success")

            # Notification: do not return a response
            if request.id is None:
                return None
            # Return success response for standard requests
            return MCPResponse(result=result, id=request.id)

        except RateLimitExceeded:
            # Record rate limit hit and re-raise for caller-specific mapping
            try:
                key_type = "user" if context.user_id else ("client" if context.client_id else "anonymous")
                self.metrics.record_rate_limit_hit(key_type=key_type)
            except _MCP_PROTOCOL_NONCRITICAL_EXCEPTIONS:
                pass
            raise
        except InvalidParamsException as ive:
            # Notification: do not return a response
            if isinstance(request, MCPRequest) and request.id is None:
                return None
            return self._error_response(ErrorCode.INVALID_PARAMS, str(ive), request.id if isinstance(request, MCPRequest) else None)
        except PermissionError as perr:
            # Map policy/permission errors to AUTHORIZATION_ERROR
            if isinstance(request, MCPRequest) and request.id is None:
                return None
            # Redact any secrets in message (defensive)
            msg = self._mask_secrets(str(perr))
            error_data = None
            if isinstance(perr, GovernanceDeniedError):
                error_data = {"governance": dict(perr.governance or {})}
            elif isinstance(perr, ApprovalRequiredError):
                error_data = {"approval": dict(perr.approval or {})}
            return self._error_response(
                ErrorCode.AUTHORIZATION_ERROR,
                msg,
                request.id if isinstance(request, MCPRequest) else None,
                data=error_data,
            )
        except _MCP_PROTOCOL_NONCRITICAL_EXCEPTIONS as e:
            # Log error
            masked = self._mask_secrets(str(e))
            secret_redacted = bool(getattr(e, "_mcp_masked_secret", False)) or masked != str(e)
            if secret_redacted:
                log.error(
                    f"MCP request failed: method={request.method}, error={masked}",
                    extra={"audit": True},
                )
            else:
                log.exception(
                    f"MCP request failed: method={request.method}, error={masked}",
                    extra={"audit": True},
                )
            try:
                elapsed = max(0.0, time.time() - start_ts)
                self.metrics.record_request(method=request.method, duration=elapsed, status="failure")
            except _MCP_PROTOCOL_NONCRITICAL_EXCEPTIONS:
                pass

            # Notification: do not return a response
            if isinstance(request, MCPRequest) and request.id is None:
                return None
            # Return error response with reduced leakage when not in debug mode
            try:
                cfg = get_config()
                debug_mode = bool(getattr(cfg, "debug_mode", False))
            except _MCP_PROTOCOL_NONCRITICAL_EXCEPTIONS:
                debug_mode = False
            msg = masked if debug_mode and not secret_redacted else "Internal error"
            return self._error_response(
                ErrorCode.INTERNAL_ERROR,
                msg,
                request.id if isinstance(request, MCPRequest) else None,
            )
        except Exception as e:
            masked = self._mask_secrets(str(e))
            secret_redacted = bool(getattr(e, "_mcp_masked_secret", False)) or masked != str(e)
            if secret_redacted:
                log.error(
                    f"MCP request failed: method={request.method}, error={masked}",
                    extra={"audit": True},
                )
            else:
                log.exception(
                    f"MCP request failed: method={request.method}, error={masked}",
                    extra={"audit": True},
                )
            with contextlib.suppress(_MCP_PROTOCOL_NONCRITICAL_EXCEPTIONS):
                elapsed = max(0.0, time.time() - start_ts)
                self.metrics.record_request(method=request.method, duration=elapsed, status="failure")
            if isinstance(request, MCPRequest) and request.id is None:
                return None
            return self._error_response(
                ErrorCode.INTERNAL_ERROR,
                "Internal error",
                request.id if isinstance(request, MCPRequest) else None,
            )

    def _mask_secrets(self, text: str) -> str:
        """Best-effort masking of bearer/API keys in strings."""
        try:
            if not text:
                return text
            import re as _re
            # Mask Bearer tokens
            text = _re.sub(r"(Bearer)\s+[A-Za-z0-9._\-~+/=]+", r"\1 ****", text, flags=_re.IGNORECASE)
            # Mask common token fields
            patterns = [
                r"(api[_-]?key)\s*[:=]\s*([^\s,;]+)",
                r"(token)\s*[:=]\s*([^\s,;]+)",
                r"(access[_-]?token)\s*[:=]\s*([^\s,;]+)",
                r"(refresh[_-]?token)\s*[:=]\s*([^\s,;]+)",
            ]
            for p in patterns:
                text = _re.sub(p, lambda m: f"{m.group(1)}=****", text, flags=_re.IGNORECASE)
            return text
        except _MCP_PROTOCOL_NONCRITICAL_EXCEPTIONS:
            return text

    def _sanitize_exception_for_telemetry(self, exc: Exception) -> Exception:
        """Redact secret-bearing exception messages before tracing records them."""
        original = str(exc)
        masked = self._mask_secrets(original)
        if masked == original:
            with contextlib.suppress(Exception):
                setattr(exc, "_mcp_masked_secret", False)
            return exc
        try:
            sanitized = exc.__class__(masked)
        except Exception:
            sanitized = RuntimeError(masked)
        with contextlib.suppress(Exception):
            setattr(sanitized, "_mcp_masked_secret", True)
        if hasattr(sanitized, "__dict__") and hasattr(exc, "__dict__"):
            with contextlib.suppress(Exception):
                for attr in ("errno", "code", "name", "lineno"):
                    if hasattr(exc, attr):
                        setattr(sanitized, attr, getattr(exc, attr))
        with contextlib.suppress(Exception):
            sanitized.args = (masked,)
        with contextlib.suppress(Exception):
            setattr(sanitized, "_mcp_masked_secret", True)
        return sanitized

    def _error_response(
        self,
        code: ErrorCode,
        message: str,
        request_id: Optional[Union[str, int]] = None,
        data: Optional[Any] = None
    ) -> MCPResponse:
        """Create an error response"""
        data = self._attach_error_hint(code, message, data)
        return MCPResponse(
            error=MCPError(
                code=code,
                message=message,
                data=data
            ),
            id=request_id
        )

    def _attach_error_hint(
        self,
        code: ErrorCode,
        message: str,
        data: Optional[Any]
    ) -> Optional[Any]:
        """Attach a structured hint for common error scenarios."""
        if data is not None:
            return data

        hint: Optional[str] = None
        lowered = message.lower()

        if code == ErrorCode.INVALID_PARAMS:
            prefix = "missing required parameter:"
            if lowered.startswith(prefix):
                # Extract the parameter name from original message
                try:
                    missing = message.split(":", 1)[1].strip().strip("'\"")
                except _MCP_PROTOCOL_NONCRITICAL_EXCEPTIONS:
                    missing = None
                if missing:
                    hint = f"Add '{missing}' to the tool arguments payload before retrying."
            elif "invalid parameters for tool" in lowered:
                hint = "Verify the tool arguments match the schema published by /mcp/tools."
        elif code == ErrorCode.AUTHORIZATION_ERROR and "write tools are disabled" in lowered:
            hint = "Enable write tools (set MCP_DISABLE_WRITE_TOOLS=0) or switch to a read-only operation."

        if hint:
            return {"hint": hint}
        return None

    async def _check_authorization(
        self,
        request: MCPRequest,
        context: RequestContext
    ) -> bool:
        """Check if user is authorized for method"""
        # Public methods that don't require auth
        public_methods = ["initialize", "ping"]
        method = request.method
        if method in public_methods:
            return True

        # Admin override (e.g., endpoint-level admin guard) for certain methods
        try:
            if isinstance(getattr(context, "metadata", None), dict):
                if context.metadata.get("admin_override") is True and request.method in {"modules/health"}:
                    return True
        except _MCP_PROTOCOL_NONCRITICAL_EXCEPTIONS:
            pass
        # No user context means no auth
        if not context.user_id:
            return False

        # tools/list: allow any authenticated user (deny if unauthenticated)
        if method == "tools/list":
            if not context.user_id:
                return False
            if not self._scope_allows(context, Resource.TOOL.value, None):
                return False
            return self._api_key_allows(context, is_write=None)

        # Map methods to resources and actions
        method_permissions = {
            # tools/list handled above
            "tools/call": (Resource.TOOL, Action.EXECUTE),
            "resources/list": (Resource.RESOURCE, Action.READ),
            "resources/read": (Resource.RESOURCE, Action.READ),
            "prompts/list": (Resource.PROMPT, Action.READ),
            "prompts/get": (Resource.PROMPT, Action.READ),
            "modules/list": (Resource.MODULE, Action.READ),
            "modules/health": (Resource.MODULE, Action.READ),
        }

        if method in method_permissions:
            resource, action = method_permissions[method]
            fn = getattr(self.rbac_policy, 'check_permission', None)
            if fn is None:
                return False
            # Provide resource_id (e.g., tool name) when applicable
            resource_id = None
            try:
                if resource == Resource.TOOL and action == Action.EXECUTE:
                    params = request.params or {}
                    name = params.get("name") if isinstance(params, dict) else None
                    if isinstance(name, str) and name:
                        resource_id = name
            except _MCP_PROTOCOL_NONCRITICAL_EXCEPTIONS:
                resource_id = None
            if inspect.iscoroutinefunction(fn):
                allowed = await fn(context.user_id, resource, action, resource_id)
            else:
                allowed = fn(context.user_id, resource, action, resource_id)
            if not allowed:
                return False
            if not self._scope_allows(context, resource.value, resource_id):
                return False
            # Apply API key scope gating for read-style methods
            if method != "tools/call":
                return self._api_key_allows(context, is_write=None)
            # For tools/call, evaluate write vs read-only tool when possible
            tool_name = resource_id if isinstance(resource_id, str) else None
            tool_def = None
            module = None
            is_write = None
            try:
                tool_args = params.get("arguments", {}) if isinstance(params, dict) else {}
                if tool_name:
                    module = await self.module_registry.find_module_for_tool(tool_name)
                if module is not None and tool_name:
                    tool_def = await self._resolve_tool_definition(module, tool_name)
                    tool_args = self._harden_and_sanitize_tool_arguments(module, tool_args)
                    is_write = self._resolve_write_classification(
                        module,
                        tool_name,
                        tool_args,
                        tool_def,
                        fallback_to_name_heuristic=True,
                    )
                elif tool_name:
                    is_write = bool(re.search(r"(ingest|update|delete|create|import)", tool_name.lower()))
            except _MCP_PROTOCOL_NONCRITICAL_EXCEPTIONS:
                is_write = None
            return self._api_key_allows(context, is_write=is_write)

        # Unknown method - deny by default
        return False

    # Protocol method handlers

    async def _handle_initialize(
        self,
        params: dict[str, Any],
        context: RequestContext
    ) -> dict[str, Any]:
        """Handle initialize request"""
        client_info = params.get("clientInfo", {})

        logger.info(f"Client initializing: {client_info}")

        # Get server capabilities
        modules = await self.module_registry.get_all_modules()

        capabilities = {
            "tools": {"available": bool(modules)},
            "resources": {"available": bool(modules)},
            "prompts": {"available": bool(modules)}
        }

        return {
            "protocolVersion": self.protocol_version,
            "capabilities": capabilities,
            "serverInfo": {
                "name": "tldw-mcp-unified",
                "version": "3.0.0"
            }
        }

    async def _handle_ping(
        self,
        params: dict[str, Any],
        context: RequestContext
    ) -> dict[str, Any]:
        """Handle ping request"""
        return {"pong": True, "timestamp": datetime.now(timezone.utc).isoformat()}

    async def _resolve_catalog_tool_names(
        self,
        params: dict[str, Any],
        context: RequestContext
    ) -> Optional[set[str]]:
        """Resolve catalog parameter into a set of tool names for filtering."""
        strict = False
        if isinstance(params, dict):
            raw_strict = params.get("catalog_strict")
            if isinstance(raw_strict, bool):
                strict = raw_strict
            elif isinstance(raw_strict, (int, float)):
                strict = bool(raw_strict)
            elif isinstance(raw_strict, str):
                strict = is_truthy(raw_strict)
        catalog_name = None
        catalog_id = None
        if isinstance(params, dict):
            catalog_name = params.get("catalog")
            catalog_id = params.get("catalog_id")
        if catalog_name is None and catalog_id is None:
            return None
        try:
            from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
            pool = await get_db_pool()
        except _MCP_PROTOCOL_NONCRITICAL_EXCEPTIONS as exc:
            context.logger.debug(f"Catalog lookup unavailable: {exc}")
            return None

        resolved_id: Optional[int] = None
        if catalog_id is not None:
            try:
                resolved_id = int(catalog_id)
            except _MCP_PROTOCOL_NONCRITICAL_EXCEPTIONS:
                resolved_id = None

        if resolved_id is None and isinstance(catalog_name, str) and catalog_name.strip():
            name = catalog_name.strip()
            meta = getattr(context, "metadata", {}) or {}
            team_id = meta.get("team_id")
            org_id = meta.get("org_id")

            row = None
            try:
                if team_id is not None:
                    row = await pool.fetchone(
                        "SELECT id FROM tool_catalogs WHERE name = ? AND team_id = ?",
                        name,
                        team_id,
                    )
                if row is None and org_id is not None:
                    row = await pool.fetchone(
                        "SELECT id FROM tool_catalogs WHERE name = ? AND org_id = ? AND team_id IS NULL",
                        name,
                        org_id,
                    )
                if row is None:
                    row = await pool.fetchone(
                        "SELECT id FROM tool_catalogs WHERE name = ? AND org_id IS NULL AND team_id IS NULL",
                        name,
                    )
                if row and row.get("id") is not None:
                    resolved_id = int(row.get("id"))
            except _MCP_PROTOCOL_NONCRITICAL_EXCEPTIONS as exc:
                context.logger.debug(f"Catalog lookup failed: {exc}")

        if resolved_id is None:
            return set() if strict else None

        try:
            rows = await pool.fetchall(
                "SELECT tool_name FROM tool_catalog_entries WHERE catalog_id = ?",
                resolved_id,
            )
        except _MCP_PROTOCOL_NONCRITICAL_EXCEPTIONS as exc:
            context.logger.debug(f"Catalog entries lookup failed: {exc}")
            return None

        names: set[str] = set()
        for r in rows:
            try:
                val = r["tool_name"] if isinstance(r, dict) else r[0]
            except _MCP_PROTOCOL_NONCRITICAL_EXCEPTIONS:
                val = None
            if isinstance(val, str):
                names.add(val)
        if names:
            return names
        return set() if strict else None

    async def _handle_tools_list(
        self,
        params: dict[str, Any],
        context: RequestContext
    ) -> dict[str, Any]:
        """List available tools"""
        tools = []
        catalog_filter = await self._resolve_catalog_tool_names(params, context)
        modules = await self.module_registry.get_all_modules()
        module_filter = None
        if isinstance(params, dict):
            module_filter = params.get("module")
        allowed_modules: Optional[set[str]] = None
        if isinstance(module_filter, str) and module_filter.strip():
            allowed_modules = {module_filter.strip()}
        elif isinstance(module_filter, list):
            allowed_modules = {str(m).strip() for m in module_filter if str(m).strip()}

        for module_id, module in modules.items():
            if allowed_modules is not None and module_id not in allowed_modules:
                continue
            if catalog_filter is not None:
                context.logger.info(
                    "Catalog filter applied",
                    catalog=catalog_filter,
                    module_count=len(modules),
                )
            try:
                if not await self._has_module_permission(context, module_id):
                    continue
                module_tools = await module.get_tools()

                for tool in module_tools:
                    tool_copy = tool.copy()
                    tool_copy["module"] = module_id
                    name = tool_copy.get("name")
                    # Scoped tool permissions: when scopes are present, list only matching tools
                    if self._mcp_scopes(context) and isinstance(name, str):
                        if not self._scope_allows(context, Resource.TOOL.value, name):
                            continue
                    # Catalog filter: include only when in selected catalog
                    if catalog_filter is not None and isinstance(name, str):
                        meta = tool_copy.get("metadata") if isinstance(tool_copy, dict) else None
                        exempt = isinstance(meta, dict) and bool(meta.get("catalog_exempt"))
                        if name not in catalog_filter and not exempt:
                            continue
                    is_write = None
                    try:
                        if isinstance(tool_copy, dict):
                            is_write = module.is_write_tool_def(tool_copy)
                    except _MCP_PROTOCOL_NONCRITICAL_EXCEPTIONS:
                        is_write = None
                    can_execute = await self._has_tool_permission(context, name, is_write=is_write) if name else False
                    tool_copy["canExecute"] = can_execute
                    tools.append(tool_copy)
            except _MCP_PROTOCOL_NONCRITICAL_EXCEPTIONS as e:
                context.logger.exception(f"Error getting tools from module {module_id}: {e}")

        return {"tools": tools}

    def _extract_allowed_tools(self, context: RequestContext) -> list[str] | None:
        """Extract allowed-tools list from request context metadata."""
        try:
            metadata = context.metadata or {}
            allowed = metadata.get("allowed_tools")
        except _MCP_PROTOCOL_NONCRITICAL_EXCEPTIONS:
            return None

        if allowed is None:
            return None
        if isinstance(allowed, list):
            cleaned = [str(item).strip() for item in allowed if str(item).strip()]
            return cleaned or None
        if isinstance(allowed, str):
            try:
                parsed = json.loads(allowed)
                if isinstance(parsed, list):
                    cleaned = [str(item).strip() for item in parsed if str(item).strip()]
                    return cleaned or None
            except json.JSONDecodeError:
                pass
            cleaned = [part.strip() for part in allowed.split(",") if part.strip()]
            return cleaned or None
        return None

    def _extract_tool_command(self, tool_args: Any) -> str | None:
        """Extract command-like string from tool arguments for pattern matching."""
        if not isinstance(tool_args, dict):
            return None
        for key in ("command", "cmd", "args", "arguments"):
            if key not in tool_args:
                continue
            value = tool_args.get(key)
            if isinstance(value, str):
                return value
            if isinstance(value, list):
                return " ".join(str(part) for part in value)
        return None

    def _matches_allowed_tool_pattern(self, tool_name: str, tool_args: Any, pattern: str) -> bool:
        """Check if tool invocation matches an allowed-tools pattern."""
        pattern = str(pattern or "").strip()
        if not pattern:
            return False
        if "(" not in pattern:
            return tool_name == pattern
        if not pattern.endswith(")"):
            return False

        base_name, cmd_pattern = pattern.split("(", 1)
        cmd_pattern = cmd_pattern[:-1]
        base_name = base_name.strip()
        if tool_name != base_name:
            return False

        command = self._extract_tool_command(tool_args)
        if command is None:
            return False

        regex_pattern = re.escape(cmd_pattern)
        regex_pattern = regex_pattern.replace(r"\*", ".*")
        try:
            return bool(re.match(f"^{regex_pattern}$", command.strip()))
        except re.error:
            return False

    def _is_tool_allowed_by_context(self, tool_name: str, tool_args: Any, context: RequestContext) -> bool:
        """Return True when tool usage is allowed by context metadata."""
        allowed_tools = self._extract_allowed_tools(context)
        if not allowed_tools:
            return True
        return any(self._matches_allowed_tool_pattern(tool_name, tool_args, pattern) for pattern in allowed_tools)

    async def _resolve_effective_tool_policy(self, context: RequestContext) -> dict[str, Any] | None:
        metadata = getattr(context, "metadata", None)
        if not isinstance(metadata, dict):
            return None
        if not is_truthy(metadata.get("mcp_policy_context_enabled")):
            return None
        cached = metadata.get("_mcp_effective_tool_policy")
        if isinstance(cached, dict):
            return cached
        try:
            from tldw_Server_API.app.services.mcp_hub_policy_resolver import (
                get_mcp_hub_policy_resolver,
            )

            resolver = await get_mcp_hub_policy_resolver()
            policy = await resolver.resolve_for_context(
                user_id=context.user_id,
                metadata=metadata,
            )
        except _MCP_PROTOCOL_NONCRITICAL_EXCEPTIONS as exc:
            logger.warning("Failed to resolve MCP Hub effective policy: {}", exc)
            policy = {
                "enabled": True,
                "allowed_tools": [],
                "denied_tools": [],
                "capabilities": [],
                "sources": [],
                "resolution_error": "policy_resolution_failed",
            }
        if policy is not None:
            metadata["_mcp_effective_tool_policy"] = policy
        return policy

    def _is_tool_allowed_by_effective_policy(
        self,
        tool_name: str,
        tool_args: Any,
        policy: dict[str, Any] | None,
    ) -> bool:
        if not isinstance(policy, dict) or not bool(policy.get("enabled", False)):
            return True
        if str(policy.get("resolution_error") or "").strip():
            return False
        denied_tools = [
            str(pattern).strip()
            for pattern in (policy.get("denied_tools") or [])
            if str(pattern).strip()
        ]
        if any(self._matches_allowed_tool_pattern(tool_name, tool_args, pattern) for pattern in denied_tools):
            return False
        allowed_tools = [
            str(pattern).strip()
            for pattern in (policy.get("allowed_tools") or [])
            if str(pattern).strip()
        ]
        if not allowed_tools:
            return True
        return any(self._matches_allowed_tool_pattern(tool_name, tool_args, pattern) for pattern in allowed_tools)

    async def _evaluate_runtime_approval(
        self,
        *,
        effective_policy: dict[str, Any] | None,
        tool_name: str,
        tool_args: Any,
        context: RequestContext,
        tool_def: dict[str, Any] | None,
        is_write: bool | None,
        within_effective_policy: bool,
        force_approval: bool = False,
        approval_reason: str | None = None,
        scope_payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        policy = dict(effective_policy or {})
        if not bool(policy.get("enabled", False)):
            return {"status": "allow", "reason": "policy_disabled"}
        if str(policy.get("resolution_error") or "").strip():
            return {"status": "deny", "reason": "policy_unavailable"}
        try:
            from tldw_Server_API.app.services.mcp_hub_approval_service import (
                get_mcp_hub_approval_service,
            )

            approval_service = await get_mcp_hub_approval_service()
            return await approval_service.evaluate_tool_call(
                effective_policy=policy,
                tool_name=tool_name,
                tool_args=tool_args,
                context=context,
                tool_def=tool_def,
                is_write=is_write,
                within_effective_policy=within_effective_policy,
                force_approval=force_approval,
                approval_reason=approval_reason,
                scope_payload=scope_payload,
            )
        except _MCP_PROTOCOL_NONCRITICAL_EXCEPTIONS as exc:
            logger.debug("Failed to evaluate MCP Hub runtime approval: {}", exc)
            if policy.get("approval_policy_id") is not None or policy.get("approval_mode"):
                return {"status": "deny", "reason": "approval_unavailable"}
            return {"status": "allow" if within_effective_policy else "deny", "reason": "approval_not_configured"}

    async def _evaluate_path_scope(
        self,
        *,
        effective_policy: dict[str, Any] | None,
        tool_name: str,
        tool_args: Any,
        context: RequestContext,
        tool_def: dict[str, Any] | None,
    ) -> dict[str, Any]:
        policy = dict(effective_policy or {})
        policy_document = dict(policy.get("policy_document") or {})
        path_scope_mode = str(policy_document.get("path_scope_mode") or "").strip()
        if not bool(policy.get("enabled", False)) or not path_scope_mode or path_scope_mode == "none":
            return {
                "enabled": False,
                "within_scope": True,
                "reason": None,
                "force_approval": False,
                "normalized_paths": [],
                "scope_payload": None,
            }
        if str(policy.get("resolution_error") or "").strip():
            return {
                "enabled": True,
                "within_scope": False,
                "reason": "policy_unavailable",
                "force_approval": False,
                "normalized_paths": [],
                "scope_payload": {"path_scope_mode": path_scope_mode, "reason": "policy_unavailable"},
            }
        try:
            from tldw_Server_API.app.services.mcp_hub_path_enforcement_service import (
                get_mcp_hub_path_enforcement_service,
            )

            path_service = await get_mcp_hub_path_enforcement_service()
            return await path_service.evaluate_tool_call(
                effective_policy=policy,
                context=context,
                tool_name=tool_name,
                tool_args=tool_args,
                tool_def=tool_def,
            )
        except _MCP_PROTOCOL_NONCRITICAL_EXCEPTIONS as exc:
            logger.debug("Failed to evaluate MCP Hub path scope: {}", exc)
            return {
                "enabled": True,
                "within_scope": False,
                "reason": "path_scope_unavailable",
                "force_approval": True,
                "normalized_paths": [],
                "scope_payload": {"path_scope_mode": path_scope_mode, "reason": "path_scope_unavailable"},
            }

    async def _evaluate_external_access(
        self,
        *,
        effective_policy: dict[str, Any] | None,
        tool_name: str,
        context: RequestContext,
    ) -> dict[str, Any]:
        deny_only_reasons = {
            "external_access_unavailable",
            "external_server_not_bound",
            "invalid_external_tool_name",
            "required_slot_not_granted",
            "required_slot_secret_missing",
        }
        if not str(tool_name or "").startswith("ext."):
            return {
                "enabled": False,
                "within_scope": True,
                "reason": None,
                "scope_payload": None,
                "hard_deny": False,
            }
        policy = dict(effective_policy or {})
        if not bool(policy.get("enabled", False)):
            return {
                "enabled": False,
                "within_scope": True,
                "reason": None,
                "scope_payload": None,
                "hard_deny": False,
            }
        sources = policy.get("sources")
        if not isinstance(sources, list):
            return {
                "enabled": True,
                "within_scope": False,
                "reason": "external_access_unavailable",
                "scope_payload": {
                    "server_id": tool_name.split(".", 2)[1],
                    "reason": "external_access_unavailable",
                    "blocked_reason": "external_access_unavailable",
                    "requested_slots": [],
                    "missing_bound_slots": [],
                    "missing_secret_slots": [],
                },
                "hard_deny": True,
            }
        parts = str(tool_name or "").split(".", 2)
        if len(parts) != 3 or not parts[1]:
            return {
                "enabled": True,
                "within_scope": False,
                "reason": "invalid_external_tool_name",
                "scope_payload": {
                    "reason": "invalid_external_tool_name",
                    "blocked_reason": "invalid_external_tool_name",
                    "requested_slots": [],
                    "missing_bound_slots": [],
                    "missing_secret_slots": [],
                },
                "hard_deny": True,
            }
        server_id = parts[1]
        metadata = context.metadata if isinstance(getattr(context, "metadata", None), dict) else {}
        cached = metadata.get("_mcp_effective_external_access")
        if not isinstance(cached, dict):
            try:
                from tldw_Server_API.app.services.mcp_hub_external_access_resolver import (
                    get_mcp_hub_external_access_resolver,
                )

                resolver = await get_mcp_hub_external_access_resolver()
                cached = await resolver.resolve_for_sources(
                    sources=[dict(item) for item in sources if isinstance(item, dict)],
                    effective_policy=policy,
                )
                metadata["_mcp_effective_external_access"] = cached
            except _MCP_PROTOCOL_NONCRITICAL_EXCEPTIONS as exc:
                logger.debug("Failed to evaluate MCP Hub external access: {}", exc)
                return {
                    "enabled": True,
                    "within_scope": False,
                    "reason": "external_access_unavailable",
                    "scope_payload": {
                        "server_id": server_id,
                        "reason": "external_access_unavailable",
                        "blocked_reason": "external_access_unavailable",
                        "requested_slots": [],
                        "missing_bound_slots": [],
                        "missing_secret_slots": [],
                    },
                    "hard_deny": True,
                }

        rows = cached.get("servers") if isinstance(cached, dict) else None
        server_row = next(
            (
                row for row in rows
                if isinstance(row, dict) and str(row.get("server_id") or "") == server_id
            ),
            None,
        ) if isinstance(rows, list) else None
        if not isinstance(server_row, dict):
            return {
                "enabled": True,
                "within_scope": False,
                "reason": "external_server_not_bound",
                "scope_payload": {
                    "server_id": server_id,
                    "reason": "external_server_not_bound",
                    "blocked_reason": "external_server_not_bound",
                    "requested_slots": [],
                    "missing_bound_slots": [],
                    "missing_secret_slots": [],
                },
                "hard_deny": True,
            }
        runtime_executable = bool(server_row.get("runtime_executable"))
        reason = str(server_row.get("blocked_reason") or "").strip() or None
        requested_slots = [
            str(slot).strip()
            for slot in (server_row.get("requested_slots") or [])
            if str(slot).strip()
        ]
        bound_slots = [
            str(slot).strip()
            for slot in (server_row.get("bound_slots") or [])
            if str(slot).strip()
        ]
        missing_bound_slots = [
            str(slot).strip()
            for slot in (server_row.get("missing_bound_slots") or [])
            if str(slot).strip()
        ]
        missing_secret_slots = [
            str(slot).strip()
            for slot in (server_row.get("missing_secret_slots") or [])
            if str(slot).strip()
        ]
        scope_payload = {
            "server_id": server_id,
            "server_name": str(server_row.get("server_name") or "").strip() or None,
            "reason": reason or ("external_server_allowed" if runtime_executable else "external_server_not_bound"),
            "blocked_reason": reason or ("external_server_allowed" if runtime_executable else "external_server_not_bound"),
            "requested_slots": requested_slots,
            "bound_slots": bound_slots,
            "missing_bound_slots": missing_bound_slots,
            "missing_secret_slots": missing_secret_slots,
        }
        if not runtime_executable:
            return {
                "enabled": True,
                "within_scope": False,
                "reason": reason or "external_server_not_bound",
                "scope_payload": scope_payload,
                "hard_deny": (reason or "external_server_not_bound") in deny_only_reasons,
            }
        return {
            "enabled": True,
            "within_scope": True,
            "reason": None,
            "scope_payload": scope_payload,
            "hard_deny": False,
        }

    async def _handle_tools_call(
        self,
        params: dict[str, Any],
        context: RequestContext
    ) -> dict[str, Any]:
        """Execute a tool."""
        prepared = await self.prepare_tool_call(params=params, context=context)
        return await self.execute_prepared_tool_call(prepared)

    async def prepare_tool_call(
        self,
        params: dict[str, Any],
        context: RequestContext,
        idempotency_key: str | None = None,
    ) -> PreparedToolCall:
        """Prepare a tool invocation through protocol policy, validation, and governance checks."""
        tool_name = params.get("name")
        tool_args = params.get("arguments", {})
        normalized_idempotency_key = self._normalize_idempotency_key(params, idempotency_key=idempotency_key)

        if not tool_name:
            raise InvalidParamsException("Tool name is required")

        # Strictly validate tool name
        if not self._tool_name_re.match(tool_name):
            raise InvalidParamsException("Invalid tool name")

        # Enforce allowed-tools constraints from context metadata (skill execution)
        if not self._is_tool_allowed_by_context(tool_name, tool_args, context):
            raise PermissionError(f"Tool '{tool_name}' not allowed by execution context")
        effective_policy = await self._resolve_effective_tool_policy(context)
        within_effective_policy = self._is_tool_allowed_by_effective_policy(tool_name, tool_args, effective_policy)
        external_access_result = await self._evaluate_external_access(
            effective_policy=effective_policy,
            tool_name=tool_name,
            context=context,
        )
        external_block_reason = str(external_access_result.get("reason") or "").strip()
        if bool(external_access_result.get("hard_deny")) or external_block_reason in {
            "required_slot_not_granted",
            "required_slot_secret_missing",
        }:
            external_scope = (
                dict(external_access_result.get("scope_payload") or {})
                if isinstance(external_access_result.get("scope_payload"), dict)
                else {}
            )
            blocked_reason = external_block_reason or "external_access_denied"
            raise GovernanceDeniedError(
                "Blocked external credential use",
                governance={
                    "action": "deny",
                    "status": "deny",
                    "reason_code": blocked_reason,
                    "external_access": external_scope,
                },
            )

        # Find module for tool
        module = await self.module_registry.find_module_for_tool(tool_name)
        if not module:
            raise InvalidParamsException(f"Tool not found: {tool_name}")

        module_id = self.module_registry.get_module_id_for_tool(tool_name) or getattr(module, "name", None)

        # Look up tool definition early for scope gating and validation
        tool_def = await self._resolve_tool_definition(module, tool_name)
        tool_args = self._harden_and_sanitize_tool_arguments(module, tool_args)

        # Determine write-capable status from sanitized arguments.
        is_write = self._resolve_write_classification(
            module,
            tool_name,
            tool_args,
            tool_def,
            fallback_to_name_heuristic=True,
        )

        module_allowed = await self._has_module_permission(context, module_id)
        tool_allowed = await self._has_tool_permission(context, tool_name, is_write=is_write)

        if not module_allowed and not tool_allowed:
            raise PermissionError(f"Permission denied for module: {module_id}")

        if not tool_allowed:
            raise PermissionError(f"Permission denied for tool: {tool_name}")

        # Protocol-level pre-execution validation for write-capable tools
        # Ensures that modules validate arguments even if they forgot to call
        # validate_tool_arguments inside execute_tool.
        # Look up tool definition from module cache where possible
        if tool_def is None:
            tool_def = await self._resolve_tool_definition(module, tool_name)

        idempotency_cache_key = None
        try:
            # Lightweight inputSchema validation (config-gated)
            cfg = get_config()
            if cfg.validate_input_schema and isinstance(tool_def, dict):
                schema = tool_def.get("inputSchema") or {}
                try:
                    self._validate_input_schema(schema, tool_args)
                except InvalidParamsException:
                    with contextlib.suppress(_MCP_PROTOCOL_NONCRITICAL_EXCEPTIONS):
                        self.metrics.record_tool_invalid_params(getattr(module, "name", "unknown"), str(tool_name))
                    raise

            # Optional policy: disable write-capable tools entirely
            if is_write:
                if get_config().disable_write_tools:
                    raise PermissionError("Write tools are disabled by server policy")
                # Check module overrides validator
                if module.__class__.validate_tool_arguments is BaseModule.validate_tool_arguments:
                    with contextlib.suppress(_MCP_PROTOCOL_NONCRITICAL_EXCEPTIONS):
                        self.metrics.record_tool_validator_missing(getattr(module, "name", "unknown"), str(tool_name))
                    raise ValueError(
                        "Write-capable tool requires module.validate_tool_arguments override"
                    )
                # Run validator
                try:
                    module.validate_tool_arguments(tool_name, tool_args)
                except _MCP_PROTOCOL_NONCRITICAL_EXCEPTIONS as ve:
                    with contextlib.suppress(_MCP_PROTOCOL_NONCRITICAL_EXCEPTIONS):
                        self.metrics.record_tool_invalid_params(getattr(module, "name", "unknown"), str(tool_name))
                    raise ValueError(f"Invalid parameters for tool {tool_name}: {ve}") from ve

                if normalized_idempotency_key:
                    idempotency_cache_key = self._make_idempotency_cache_key(
                        context,
                        module_id or getattr(module, "name", "unknown"),
                        tool_name,
                        normalized_idempotency_key,
                    )
        except ValueError as ve:
            # Surface as JSON-RPC INVALID_PARAMS at the protocol layer
            # by raising a sentinel exception handled by process_request
            raise InvalidParamsException(str(ve)) from ve

        path_scope_result = await self._evaluate_path_scope(
            effective_policy=effective_policy,
            tool_name=tool_name,
            tool_args=tool_args,
            context=context,
            tool_def=tool_def if isinstance(tool_def, dict) else None,
        )
        within_resolved_scope = bool(path_scope_result.get("within_scope", True)) and bool(
            external_access_result.get("within_scope", True)
        )
        approval_reason = str(path_scope_result.get("reason") or "").strip() or None
        if approval_reason is None:
            approval_reason = str(external_access_result.get("reason") or "").strip() or None
        scope_payload: dict[str, Any] | None = None
        for payload in (
            path_scope_result.get("scope_payload"),
            external_access_result.get("scope_payload"),
        ):
            if isinstance(payload, dict):
                scope_payload = dict(scope_payload or {})
                scope_payload.update(payload)

        path_scope_block_reason = str(path_scope_result.get("reason") or "").strip()
        if path_scope_block_reason and not bool(path_scope_result.get("within_scope", True)):
            requires_approval = bool(path_scope_result.get("force_approval", False))
            if not requires_approval or path_scope_block_reason == "workspace_unresolvable_for_trust_source":
                raise GovernanceDeniedError(
                    "Blocked path-scoped tool use",
                    governance={
                        "action": "deny",
                        "status": "deny",
                        "reason_code": path_scope_block_reason,
                        "path_scope": dict(scope_payload or {}),
                    },
                )

        approval_result = await self._evaluate_runtime_approval(
            effective_policy=effective_policy,
            tool_name=tool_name,
            tool_args=tool_args,
            context=context,
            tool_def=tool_def if isinstance(tool_def, dict) else None,
            is_write=is_write,
            within_effective_policy=within_effective_policy and within_resolved_scope,
            force_approval=bool(path_scope_result.get("force_approval", False)),
            approval_reason=approval_reason,
            scope_payload=scope_payload,
        )
        approval_status = str(approval_result.get("status") or "allow").strip().lower()
        if approval_status == "approval_required":
            raise ApprovalRequiredError(
                "Approval required by MCP Hub policy",
                approval=approval_result.get("approval") if isinstance(approval_result.get("approval"), dict) else None,
            )
        if approval_status != "allow":
            raise PermissionError(f"Tool '{tool_name}' not allowed by MCP Hub policy")

        args_hash = self._hash_arguments(tool_args if isinstance(tool_args, dict) else {})
        await self._run_governance_preflight(
            tool_name=tool_name,
            tool_args=tool_args if isinstance(tool_args, dict) else {},
            tool_def=tool_def if isinstance(tool_def, dict) else None,
            context=context,
        )
        context_fingerprint = self._fingerprint_request_context(context)
        integrity_tag = self._build_prepared_tool_call_integrity_tag(
            tool_name=tool_name,
            module_id=module_id,
            is_write=is_write,
            idempotency_cache_key=idempotency_cache_key,
            arguments_hash=args_hash,
            context_fingerprint=context_fingerprint,
        )

        return PreparedToolCall(
            tool_name=tool_name,
            tool_args=tool_args,
            module=module,
            module_id=module_id,
            tool_def=tool_def if isinstance(tool_def, dict) else None,
            is_write=is_write,
            normalized_idempotency_key=normalized_idempotency_key,
            idempotency_cache_key=idempotency_cache_key,
            arguments_hash=args_hash,
            context_fingerprint=context_fingerprint,
            integrity_tag=integrity_tag,
            context=context,
        )

    async def execute_prepared_tool_call(self, prepared: PreparedToolCall) -> dict[str, Any]:
        """Execute a previously prepared tool invocation."""
        self._verify_prepared_tool_call_integrity(prepared)
        tool_name = prepared.tool_name
        tool_args = prepared.tool_args
        module = prepared.module
        module_id = prepared.module_id
        tool_def = prepared.tool_def
        is_write = prepared.is_write
        normalized_idempotency_key = prepared.normalized_idempotency_key
        idempotency_cache_key = prepared.idempotency_cache_key
        args_hash = prepared.arguments_hash
        context = prepared.context

        async def _execute_tool_call() -> dict[str, Any]:
            # Optional per-tool/category rate limits (ingestion vs read)
            try:
                # Categorization for per-category rate limiting
                cfg = get_config()
                category = None
                # 1) Prefer tool metadata.category if available
                try:
                    meta = (tool_def or {}).get("metadata") or {}
                    cat = str(meta.get("category") or "").lower()
                    if cat in {"ingestion", "management", "read"}:
                        category = cat
                except _MCP_PROTOCOL_NONCRITICAL_EXCEPTIONS:
                    category = None
                # 2) Config-driven mapping
                if not category:
                    try:
                        if isinstance(cfg.tool_category_map, dict) and tool_name in cfg.tool_category_map:
                            category = str(cfg.tool_category_map.get(tool_name))
                    except _MCP_PROTOCOL_NONCRITICAL_EXCEPTIONS:
                        category = None
                # 3) Heuristic fallback
                if not category:
                    ingestion_tools = {'ingest_media', 'update_media', 'delete_media'}
                    category = 'ingestion' if tool_name in ingestion_tools else 'read'
                key_owner = f"user:{context.user_id}" if context.user_id else (f"client:{context.client_id}" if context.client_id else "anon")
                rl_key = f"{key_owner}:tool:{tool_name}:cat:{category}"
                await self.rate_limiter.check_rate_limit(rl_key, category=category)
            except RateLimitExceeded:
                raise
            except _MCP_PROTOCOL_NONCRITICAL_EXCEPTIONS:
                # Best-effort; do not block on limiter errors
                pass

            # Execute tool with circuit breaker (pass context through)
            t0 = time.time()

            try:
                # Trace the tool call with OTEL
                with self.telemetry.trace_context(
                    "mcp.tool_call",
                    {
                        "mcp.tool": tool_name,
                        "mcp.module": getattr(module, "name", "unknown"),
                        "mcp.user_id": str(context.user_id or ""),
                        "mcp.client_id": str(context.client_id or ""),
                    },
                ) as span:
                    try:
                        execution_args = tool_args
                        tool_schema_props = (
                            (tool_def or {}).get("inputSchema", {}).get("properties", {})
                            if isinstance(tool_def, dict)
                            else {}
                        )
                        if (
                            normalized_idempotency_key
                            and isinstance(tool_args, dict)
                            and isinstance(tool_schema_props, dict)
                            and "idempotencyKey" in tool_schema_props
                            and "idempotencyKey" not in tool_args
                        ):
                            execution_args = dict(tool_args)
                            execution_args["idempotencyKey"] = normalized_idempotency_key
                        result = await module.execute_with_circuit_breaker(
                            module.execute_tool,
                            tool_name,
                            execution_args,
                            context
                        )
                        span.set_attribute("mcp.status", "success")
                    except InvalidParamsException as _tool_e:
                        span.set_attribute("mcp.status", "failure")
                        span.set_attribute("mcp.error_type", _tool_e.__class__.__name__)
                        span.set_attribute("mcp.error_message", str(_tool_e)[:200])
                        with contextlib.suppress(_MCP_PROTOCOL_NONCRITICAL_EXCEPTIONS):
                            self.metrics.record_tool_invalid_params(getattr(module, "name", "unknown"), str(tool_name))
                        raise
                    except (TypeError, ValueError) as _tool_e:
                        # Module argument validators often raise ValueError/TypeError.
                        # Normalize those to INVALID_PARAMS so HTTP callers receive 400.
                        span.set_attribute("mcp.status", "failure")
                        span.set_attribute("mcp.error_type", _tool_e.__class__.__name__)
                        span.set_attribute("mcp.error_message", str(_tool_e)[:200])
                        with contextlib.suppress(_MCP_PROTOCOL_NONCRITICAL_EXCEPTIONS):
                            self.metrics.record_tool_invalid_params(getattr(module, "name", "unknown"), str(tool_name))
                        raise InvalidParamsException(str(_tool_e)) from _tool_e
                    except _MCP_PROTOCOL_NONCRITICAL_EXCEPTIONS as _tool_e:
                        span.set_attribute("mcp.status", "failure")
                        span.set_attribute("mcp.error_type", _tool_e.__class__.__name__)
                        span.set_attribute("mcp.error_message", str(_tool_e)[:200])
                        raise
                    finally:
                        span.set_attribute("mcp.duration_ms", max(0.0, (time.time() - t0) * 1000.0))

                # Format result
                if isinstance(result, str):
                    content = [{"type": "text", "text": result}]
                elif isinstance(result, list):
                    content = result
                elif isinstance(result, dict):
                    # Preserve structured tool results as JSON content instead of stringifying.
                    content = [{"type": "json", "json": result}]
                else:
                    content = [{"type": "text", "text": str(result)}]

                module_name = module_id or getattr(module, "name", None)
                # Record module operation metrics
                try:
                    duration = max(0.0, time.time() - t0)
                    self.metrics.record_module_operation(module=module_name or "unknown", operation="tools_call", duration=duration, success=True)
                except _MCP_PROTOCOL_NONCRITICAL_EXCEPTIONS:
                    pass
                self._audit_tool_event(
                    context,
                    tool_name,
                    module_name,
                    status="success",
                    duration_ms=max(0.0, (time.time() - t0) * 1000.0),
                    arguments_hash=args_hash,
                )
                response_payload = {"content": content, "module": module_name, "tool": tool_name}
                return response_payload

            except _MCP_PROTOCOL_NONCRITICAL_EXCEPTIONS as e:
                sanitized_error = self._mask_secrets(str(e))
                context.logger.exception(f"Tool execution failed: {tool_name} - {sanitized_error}")
                try:
                    duration = max(0.0, time.time() - t0)
                    self.metrics.record_module_operation(module=getattr(module, "name", "unknown"), operation="tools_call", duration=duration, success=False)
                except _MCP_PROTOCOL_NONCRITICAL_EXCEPTIONS:
                    pass
                self._audit_tool_event(
                    context,
                    tool_name,
                    module_id or getattr(module, "name", None),
                    status="failure",
                    duration_ms=max(0.0, (time.time() - t0) * 1000.0),
                    arguments_hash=args_hash,
                    error=e,
                )
                raise

        if is_write and idempotency_cache_key:
            cfg = get_config()
            ttl = max(1, int(getattr(cfg, "idempotency_ttl_seconds", 300)))
            max_size = max(1, int(getattr(cfg, "idempotency_cache_size", 512)))
            if args_hash is None:
                raise InvalidParamsException("Unable to fingerprint tool arguments for idempotency")
            arguments_bound = await self._idempotency.bind_arguments(
                idempotency_cache_key,
                args_hash,
                ttl=ttl,
                max_size=max_size,
            )
            if not arguments_bound:
                raise InvalidParamsException("Idempotency key was already used with different arguments")
            module_timeout = int(getattr(getattr(module, "config", None), "timeout_seconds", cfg.module_timeout))
            lock_ttl = max(ttl, module_timeout * 2)
            payload, from_cache = await self._idempotency.run(
                idempotency_cache_key,
                _execute_tool_call,
                ttl=ttl,
                max_size=max_size,
                lock_ttl=lock_ttl,
            )
            try:
                if from_cache:
                    self.metrics.record_idempotency_hit(module_id or getattr(module, "name", "unknown"), str(tool_name))
                else:
                    self.metrics.record_idempotency_miss(module_id or getattr(module, "name", "unknown"), str(tool_name))
            except _MCP_PROTOCOL_NONCRITICAL_EXCEPTIONS:
                pass
            return payload

        return await _execute_tool_call()

    # -------------------------
    # Idempotency cache helpers
    # -------------------------
    def _make_idempotency_cache_key(self, context: RequestContext, module_name: str, tool_name: str, idempotency_key: str) -> str:
        owner = f"user:{context.user_id}" if context.user_id else (f"client:{context.client_id}" if context.client_id else "anon")
        return f"{owner}|module:{module_name}|tool:{tool_name}|key:{idempotency_key}"

    def _validate_input_schema(self, schema: dict[str, Any], args: dict[str, Any]) -> None:
        """Quick JSON Schema checks: required keys, primitive types, unknown fields.
        Only applies when schema.type == object.
        """
        try:
            if not isinstance(schema, dict):
                return
            if schema.get("type") != "object":
                return
            if not isinstance(args, dict):
                raise InvalidParamsException("Arguments must be an object")
            props = schema.get("properties") or {}
            required = schema.get("required") or []
            addl = schema.get("additionalProperties", True)

            # Required
            for key in required:
                if key not in args or args.get(key) is None:
                    raise InvalidParamsException(f"Missing required parameter: {key}")

            # Unknown fields
            if addl is False:
                unknown = [k for k in args if k not in props]
                if unknown:
                    raise InvalidParamsException(f"Unknown parameters: {', '.join(unknown)}")

            # Primitive type checks
            def _type_ok(expected: str, value: Any) -> bool:
                mapping = {
                    "string": str,
                    "number": (int, float),
                    "integer": int,
                    "boolean": bool,
                    "object": dict,
                    "array": list,
                }
                py = mapping.get(expected)
                if py is None:
                    return True
                # number should not reject ints; python isinstance(True, int) caveat
                if expected in {"number", "integer"} and isinstance(value, bool):
                    return False
                return isinstance(value, py)

            for k, v in args.items():
                if k in props:
                    p = props.get(k) or {}
                    t = p.get("type")
                    if isinstance(t, str) and not _type_ok(t, v):
                        raise InvalidParamsException(f"Invalid type for '{k}': expected {t}")
        except InvalidParamsException:
            raise
        except _MCP_PROTOCOL_NONCRITICAL_EXCEPTIONS:
            # Be forgiving on schema format errors
            return

    async def _handle_resources_list(
        self,
        params: dict[str, Any],
        context: RequestContext
    ) -> dict[str, Any]:
        """List available resources"""
        resources = []
        modules = await self.module_registry.get_all_modules()
        catalog_filter = await self._resolve_catalog_tool_names(params, context)
        module_tool_names: dict[str, set[str]] = {}

        for module_id, module in modules.items():
            try:
                if catalog_filter is not None:
                    context.logger.info(f"Catalog filter applied: {sorted(catalog_filter)}")
                if not await self._has_module_permission(context, module_id):
                    continue
                if catalog_filter is not None:
                    cached_names = module_tool_names.get(module_id)
                    if cached_names is None:
                        try:
                            module_tools = await module.get_tools()
                            cached_names = {
                                str(tool.get("name"))
                                for tool in module_tools
                                if isinstance(tool, dict) and isinstance(tool.get("name"), str)
                            }
                        except _MCP_PROTOCOL_NONCRITICAL_EXCEPTIONS:
                            cached_names = set()
                        module_tool_names[module_id] = cached_names
                    if not cached_names.intersection(catalog_filter):
                        continue
                module_resources = await module.get_resources()

                for resource in module_resources:
                    uri = resource.get("uri") if isinstance(resource, dict) else None
                    if uri and not await self._has_resource_permission(context, uri, module_id):
                        continue
                    resource_copy = resource.copy() if isinstance(resource, dict) else resource
                    if isinstance(resource_copy, dict):
                        resource_copy["module"] = module_id
                    resources.append(resource_copy)

            except _MCP_PROTOCOL_NONCRITICAL_EXCEPTIONS as e:
                context.logger.exception(f"Error getting resources from module {module_id}: {e}")

        return {"resources": resources}

    async def _handle_resources_read(
        self,
        params: dict[str, Any],
        context: RequestContext
    ) -> dict[str, Any]:
        """Read a resource"""
        uri = params.get("uri")
        if not uri:
            raise InvalidParamsException("Resource URI is required")

        # Find module for resource
        module = await self.module_registry.find_module_for_resource(uri)
        if not module:
            raise InvalidParamsException(f"Resource not found: {uri}")
        module_id = self.module_registry.get_module_id_for_resource(uri) or getattr(module, "name", None)

        if not await self._has_resource_permission(context, uri, module_id):
            raise PermissionError(f"Permission denied for resource: {uri}")

        # Read resource (pass context when supported)
        read_fn = module.read_resource
        try:
            params = inspect.signature(read_fn).parameters
        except (TypeError, ValueError):
            params = {}
        if "context" in params:
            content = await read_fn(uri, context=context)
        else:
            content = await read_fn(uri)

        return {"contents": [content]}

    async def _handle_prompts_list(
        self,
        params: dict[str, Any],
        context: RequestContext
    ) -> dict[str, Any]:
        """List available prompts"""
        prompts = []
        modules = await self.module_registry.get_all_modules()

        for module_id, module in modules.items():
            try:
                if not await self._has_module_permission(context, module_id):
                    continue
                module_prompts = await module.get_prompts()

                for prompt in module_prompts:
                    name = prompt.get("name") if isinstance(prompt, dict) else None
                    if name and not await self._has_prompt_permission(context, name, module_id):
                        continue
                    prompt_copy = prompt.copy() if isinstance(prompt, dict) else prompt
                    if isinstance(prompt_copy, dict):
                        prompt_copy["module"] = module_id
                    prompts.append(prompt_copy)

            except _MCP_PROTOCOL_NONCRITICAL_EXCEPTIONS as e:
                context.logger.exception(f"Error getting prompts from module {module_id}: {e}")

        return {"prompts": prompts}

    async def _handle_prompts_get(
        self,
        params: dict[str, Any],
        context: RequestContext
    ) -> dict[str, Any]:
        """Get a specific prompt"""
        name = params.get("name")
        if not name:
            raise InvalidParamsException("Prompt name is required")

        arguments = params.get("arguments", {})

        # Find module for prompt
        module = await self.module_registry.find_module_for_prompt(name)
        if not module:
            raise InvalidParamsException(f"Prompt not found: {name}")
        module_id = self.module_registry.get_module_id_for_prompt(name) or getattr(module, "name", None)

        if not await self._has_prompt_permission(context, name, module_id):
            raise PermissionError(f"Permission denied for prompt: {name}")

        # Get prompt
        prompt = await module.get_prompt(name, arguments)

        return prompt

    async def _handle_modules_list(
        self,
        params: dict[str, Any],
        context: RequestContext
    ) -> dict[str, Any]:
        """List registered modules"""
        registrations = await self.module_registry.list_registrations()
        filtered: list[dict[str, Any]] = []
        for entry in registrations:
            module_id = entry.get("module_id") if isinstance(entry, dict) else None
            try:
                if await self._has_module_permission(context, module_id):
                    filtered.append(entry)
            except _MCP_PROTOCOL_NONCRITICAL_EXCEPTIONS:
                continue
        return {"modules": filtered}

    async def _handle_modules_health(
        self,
        params: dict[str, Any],
        context: RequestContext
    ) -> dict[str, Any]:
        """Get module health status"""
        health_results = await self.module_registry.check_all_health()

        # Convert to serializable format
        health_data = {}
        for module_id, health in health_results.items():
            last_check_iso = None
            try:
                if getattr(health, "last_check", None):
                    last_check_iso = health.last_check.isoformat()
            except _MCP_PROTOCOL_NONCRITICAL_EXCEPTIONS:
                last_check_iso = None
            health_data[module_id] = {
                "status": health.status.value if getattr(health, "status", None) else "unknown",
                "message": getattr(health, "message", ""),
                "checks": getattr(health, "checks", {}),
                "last_check": last_check_iso,
            }

        return {"health": health_data}


# Convenience function
async def process_mcp_request(
    request: Union[dict[str, Any], MCPRequest],
    context: Optional[RequestContext] = None
) -> MCPResponse:
    """Process an MCP request"""
    protocol = MCPProtocol()
    return await protocol.process_request(request, context)
