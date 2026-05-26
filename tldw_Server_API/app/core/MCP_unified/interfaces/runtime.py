from __future__ import annotations

from collections.abc import Awaitable
from dataclasses import dataclass
from typing import Any, Protocol

from .policy import (
    ApprovalEvaluator,
    EffectivePolicyResolver,
    ExternalAccessEvaluator,
    PathScopeEnforcer,
)


class ModuleRegistry(Protocol):
    async def start_health_monitoring(self) -> Any: ...

    async def register_module(
        self,
        module_id: str,
        module_class: Any,
        config: Any | None = None,
    ) -> Any: ...

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
    def check_permission(
        self,
        user_id: str | None,
        resource: Any,
        action: Any,
        resource_id: str | None = None,
    ) -> bool | Awaitable[bool]: ...


class RateLimiter(Protocol):
    async def check_rate_limit(self, key: str, *, category: str = "default") -> None: ...


class MetricsCollector(Protocol):
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
    def trace_context(
        self,
        operation_name: str,
        attributes: dict[str, Any] | None = None,
    ) -> Any: ...


class DatabasePathResolver(Protocol):
    def resolve_user_db_paths(self, user_id: str | int | None) -> dict[str, str]: ...


class ApiKeyScopeNormalizer(Protocol):
    def normalize(self, raw_scopes: Any) -> set[str]: ...


class RedisClientFactory(Protocol):
    def __call__(self, **kwargs: Any) -> Awaitable[Any]: ...


class CircuitBreakerFactory(Protocol):
    def __call__(self, *, name: str, config: Any) -> Any: ...


@dataclass(slots=True)
class MCPRuntimeDependencies:
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
