from __future__ import annotations

from typing import Any

from tldw_Server_API.app.core.Infrastructure.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
)
from tldw_Server_API.app.core.Infrastructure.redis_factory import create_async_redis_client
from tldw_Server_API.app.core.MCP_unified.auth.authnz_rbac import get_rbac_policy
from tldw_Server_API.app.core.MCP_unified.auth.rate_limiter import get_rate_limiter
from tldw_Server_API.app.core.MCP_unified.interfaces.runtime import MCPRuntimeDependencies
from tldw_Server_API.app.core.MCP_unified.modules.registry import get_module_registry
from tldw_Server_API.app.core.MCP_unified.monitoring.metrics import get_metrics_collector
from tldw_Server_API.app.core.Metrics.telemetry import get_telemetry_manager


class TldwDatabasePathResolver:
    def resolve_user_db_paths(self, user_id: str | int | None) -> dict[str, str]:
        if user_id is None:
            return {}
        try:
            normalized_user_id = int(str(user_id))
        except (TypeError, ValueError):
            return {}

        from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths

        paths = DatabasePaths.get_all_user_db_paths(normalized_user_id)
        return {key: str(value) for key, value in paths.items()}


class TldwApiKeyScopeNormalizer:
    def normalize(self, raw_scopes: Any) -> set[str]:
        try:
            from tldw_Server_API.app.core.AuthNZ.api_key_manager import normalize_scope

            return set(normalize_scope(raw_scopes))
        except Exception:
            return set()


def create_tldw_circuit_breaker(*, name: str, config: CircuitBreakerConfig) -> CircuitBreaker:
    return CircuitBreaker(name=name, config=config)


def build_default_runtime_dependencies() -> MCPRuntimeDependencies:
    from tldw_Server_API.app.core.MCP_unified.adapters.tldw_policy import (
        TldwApprovalEvaluator,
        TldwEffectivePolicyResolver,
        TldwExternalAccessEvaluator,
        TldwPathScopeEnforcer,
    )

    return MCPRuntimeDependencies(
        module_registry=get_module_registry(),
        rbac_policy=get_rbac_policy(),
        rate_limiter=get_rate_limiter(),
        metrics_collector=get_metrics_collector(),
        telemetry_provider=get_telemetry_manager(),
        database_path_resolver=TldwDatabasePathResolver(),
        api_key_scope_normalizer=TldwApiKeyScopeNormalizer(),
        effective_policy_resolver=TldwEffectivePolicyResolver(),
        approval_evaluator=TldwApprovalEvaluator(),
        path_scope_enforcer=TldwPathScopeEnforcer(),
        external_access_evaluator=TldwExternalAccessEvaluator(),
        redis_client_factory=create_async_redis_client,
        circuit_breaker_factory=create_tldw_circuit_breaker,
    )
