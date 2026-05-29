"""Runtime dependency protocols for the MCP Unified package boundary.

These protocols describe the services MCP Unified needs without importing a
host application. Embedders can satisfy them with local implementations, while
`tldw_server` re-exports the same contracts through its compatibility shims.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from .policy import (
    ApprovalEvaluator,
    EffectivePolicyResolver,
    ExternalAccessEvaluator,
    PathScopeEnforcer,
)


class ModuleRegistry(Protocol):
    """Registry operations needed by protocol and server request routing."""

    async def start_health_monitoring(self) -> Any: ...

    async def register_module(
        self,
        module_id: str,
        module_type: type[Any],
        config: Any,
    ) -> None: ...

    async def get_module(self, module_id: str) -> Any | None: ...

    async def get_all_modules(self) -> dict[str, Any]: ...

    async def find_module_for_tool(self, tool_name: str) -> Any | None: ...

    def get_module_id_for_tool(self, tool_name: str) -> str | None: ...

    async def find_module_for_resource(self, uri: str) -> Any | None: ...

    def get_module_id_for_resource(self, uri: str) -> str | None: ...

    async def find_module_for_prompt(self, name: str) -> Any | None: ...

    def get_module_id_for_prompt(self, name: str) -> str | None: ...

    async def check_all_health(self) -> dict[str, Any]: ...

    async def get_module_status(self, module_id: str) -> dict[str, Any] | None: ...

    async def list_registrations(self) -> list[dict[str, Any]]: ...

    async def shutdown_all(self) -> None: ...


class RbacPolicy(Protocol):
    """RBAC permission checker used for MCP resource and tool access.

    Hosts may provide synchronous in-memory checks or asynchronous database-backed
    checks; protocol consumers normalize either result shape.
    """

    def check_permission(
        self,
        user_id: str | None,
        resource: Any,
        action: Any,
        resource_id: str | None = None,
    ) -> bool | Awaitable[bool]: ...


class RateLimiter(Protocol):
    """Rate-limit gate for MCP protocol operations."""

    async def check_rate_limit(self, key: str, *, category: str = "default") -> None: ...


class MetricsCollector(Protocol):
    """Metrics sink used by MCP protocol and server lifecycle paths."""

    async def start_collection(self) -> None: ...

    async def stop_collection(self) -> None: ...

    def record_request(
        self,
        method: str,
        duration: float,
        status: str = "success",
        labels: dict[str, str] | None = None,
    ) -> None: ...

    def record_module_operation(
        self,
        module: str,
        operation: str,
        duration: float,
        success: bool,
    ) -> None: ...

    def update_connection_count(self, connection_type: str, count: int) -> None: ...

    def record_connection_error(self, connection_type: str, error: str) -> None: ...

    def record_ws_session_closure(self, reason: str) -> None: ...

    def record_ws_rejection(self, reason: str, ip_bucket: str = "unknown") -> None: ...

    def record_rate_limit_hit(self, key_type: str = "user") -> None: ...

    def record_idempotency_hit(self, module: str, tool: str) -> None: ...

    def record_idempotency_miss(self, module: str, tool: str) -> None: ...

    def record_governance_check(self, *args: Any, **kwargs: Any) -> None: ...

    def record_tool_invalid_params(self, module: str, tool: str) -> None: ...

    def record_tool_validator_missing(self, module: str, tool: str) -> None: ...


class TelemetryProvider(Protocol):
    """Tracing provider that yields span-like context managers."""

    def trace_context(
        self,
        operation_name: str,
        attributes: dict[str, Any] | None = None,
    ) -> Any: ...


class DatabasePathResolver(Protocol):
    """Resolver for per-user database paths visible to MCP modules."""

    def resolve_user_db_paths(self, user_id: str | int | None) -> dict[str, str]: ...


class ApiKeyScopeNormalizer(Protocol):
    """Normalizer for API-key scope payloads from a host auth layer."""

    def normalize(self, raw_scopes: Any) -> set[str]: ...


class RedisClientFactory(Protocol):
    """Factory for creating Redis-compatible async clients."""

    def __call__(self, **kwargs: Any) -> Awaitable[Any]: ...


class CircuitBreakerFactory(Protocol):
    """Factory for creating module circuit breakers from neutral configs."""

    def __call__(self, *, name: str, config: Any) -> Any: ...


@dataclass(slots=True)
class AuthenticatedIdentity:
    """Host-authenticated identity projected into MCP request metadata."""

    user_id: str
    roles: list[str] = field(default_factory=list)
    permissions: list[str] = field(default_factory=list)


class ServerAuthProvider(Protocol):
    """Host authentication operations needed by MCPServer transports."""

    def get_mcp_jwt_manager(self) -> Any: ...

    def is_authnz_access_token(self, token: str) -> bool: ...

    async def authenticate_authnz_websocket_token(
        self,
        token: str,
        *,
        websocket: Any,
    ) -> AuthenticatedIdentity | None: ...

    async def validate_api_key(
        self,
        api_key: str,
        *,
        ip_address: str | None = None,
    ) -> dict[str, Any] | None: ...

    def normalize_api_key_permissions(self, info: dict[str, Any] | None) -> list[str]: ...


class LifecycleGuard(Protocol):
    """Host lifecycle guard for MCP transport startup and shutdown drains."""

    def assert_may_start_work(self, app: Any, family: str) -> None: ...

    def register_shutdown_transport_family(
        self,
        family: str,
        *,
        active_count: Callable[[], int],
        drain: Callable[[float | None], Awaitable[None]],
    ) -> None: ...


class PermissionSeeder(Protocol):
    """Host hook for seeding compatibility permissions during startup."""

    async def seed_default_tool_permissions(self) -> None: ...


class ModuleConfigProvider(Protocol):
    """Host defaults used while building module configuration."""

    def default_media_db_path(self) -> str: ...


class PolicyContextProvider(Protocol):
    """Host feature flag provider for MCP policy-context metadata."""

    def is_policy_context_enabled(self) -> bool: ...


class EnvironmentFlagsProvider(Protocol):
    """Host environment/test-mode helper facade used by MCPServer."""

    def env_flag_enabled(self, name: str) -> bool: ...

    def is_test_mode(self) -> bool: ...

    def is_explicit_pytest_runtime(self) -> bool: ...

    def is_truthy(self, value: Any) -> bool: ...


class WebSocketCloseTarget(Protocol):
    """Minimal websocket close capability exposed through stream wrappers."""

    async def close(self, code: int = 1000, reason: str = "") -> Any: ...


class WebSocketStream(Protocol):
    """Host-neutral websocket stream operations used by MCPServer."""

    ws: WebSocketCloseTarget

    async def start(self) -> None: ...

    async def stop(self) -> None: ...

    def mark_activity(self) -> None: ...

    async def send_json(self, payload: dict[str, Any]) -> None: ...


class WebSocketStreamFactory(Protocol):
    """Factory for host websocket stream lifecycle wrappers."""

    def __call__(
        self,
        websocket: Any,
        *,
        heartbeat_interval_s: float | None,
        idle_timeout_s: float | None,
        close_on_done: bool,
        labels: dict[str, str],
    ) -> WebSocketStream: ...


@dataclass(slots=True)
class MCPRuntimeDependencies:
    """Concrete dependency bundle passed into MCP runtime components."""

    module_registry: ModuleRegistry
    rbac_policy: RbacPolicy
    rate_limiter: RateLimiter
    metrics_collector: MetricsCollector
    telemetry_provider: TelemetryProvider
    database_path_resolver: DatabasePathResolver
    api_key_scope_normalizer: ApiKeyScopeNormalizer
    effective_policy_resolver: EffectivePolicyResolver
    approval_evaluator: ApprovalEvaluator
    path_scope_enforcer: PathScopeEnforcer
    external_access_evaluator: ExternalAccessEvaluator
    redis_client_factory: RedisClientFactory
    circuit_breaker_factory: CircuitBreakerFactory
    auth_provider: ServerAuthProvider
    lifecycle_guard: LifecycleGuard
    permission_seeder: PermissionSeeder
    module_config_provider: ModuleConfigProvider
    policy_context_provider: PolicyContextProvider
    environment_flags_provider: EnvironmentFlagsProvider
    websocket_stream_factory: WebSocketStreamFactory
