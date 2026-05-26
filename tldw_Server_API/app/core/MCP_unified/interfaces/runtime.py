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
    async def find_module_for_tool(self, tool_name: str) -> Any | None: ...

    def get_module_id_for_tool(self, tool_name: str) -> str | None: ...


class RbacPolicy(Protocol):
    async def check_permission(
        self,
        user_id: str | None,
        resource: Any,
        action: Any,
        resource_id: str | None = None,
    ) -> bool: ...


class RateLimiter(Protocol):
    async def check_rate_limit(self, key: str, *, category: str = "default") -> None: ...


class MetricsCollector(Protocol):
    def record_request(
        self,
        method: str,
        duration: float,
        status: str = "success",
        labels: dict[str, str] | None = None,
    ) -> None: ...


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
