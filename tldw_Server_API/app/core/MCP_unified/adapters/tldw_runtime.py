"""Default host adapters for running MCP Unified inside tldw_server.

The interfaces package defines the extraction boundary in host-neutral terms.
This module binds those contracts back to the current tldw_server services so
the in-repo server keeps its legacy behavior while embedders can provide their
own dependency bundle.
"""

from __future__ import annotations

import json
from typing import Any

from loguru import logger

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
    """Resolve per-user database paths through the tldw_server DB path helper."""

    def resolve_user_db_paths(self, user_id: str | int | None) -> dict[str, str]:
        """Return string paths for the user's MCP-visible databases.

        Invalid user identifiers or host DB path failures fall back to an empty
        mapping so standalone callers can continue without tldw_server storage.
        """
        if user_id is None:
            return {}
        try:
            normalized_user_id = int(str(user_id))
        except (TypeError, ValueError):
            return {}

        try:
            from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths

            paths = DatabasePaths.get_all_user_db_paths(normalized_user_id)
            return {key: str(value) for key, value in paths.items()}
        except Exception as exc:
            logger.debug(
                "MCP user database path resolution failed; returning empty paths: {}",
                exc.__class__.__name__,
            )
            return {}


class TldwApiKeyScopeNormalizer:
    """Normalize tldw_server API-key scope payloads for MCP authorization."""

    def normalize(self, raw_scopes: Any) -> set[str]:
        """Return normalized scope strings, falling back to local parsing."""
        try:
            from tldw_Server_API.app.core.AuthNZ.api_key_manager import normalize_scope

            return set(normalize_scope(raw_scopes))
        except Exception as exc:
            logger.debug(
                "MCP API key scope normalizer failed; using local fallback: {}",
                exc.__class__.__name__,
            )
            return self._manual_normalize(raw_scopes)

    @staticmethod
    def _manual_normalize(raw_scopes: Any) -> set[str]:
        """Normalize simple string and collection payloads without AuthNZ imports."""
        if isinstance(raw_scopes, str):
            stripped = raw_scopes.strip()
            if stripped.startswith("["):
                try:
                    parsed = json.loads(stripped)
                except json.JSONDecodeError:
                    pass
                else:
                    if isinstance(parsed, list):
                        return {
                            item.strip().lower()
                            for item in parsed
                            if isinstance(item, str) and item.strip()
                        }
            return {stripped.lower()} if stripped else set()
        if isinstance(raw_scopes, (list, tuple, set)):
            return {
                item.strip().lower()
                for item in raw_scopes
                if isinstance(item, str) and item.strip()
            }
        return set()


def _to_tldw_circuit_breaker_config(config: Any) -> CircuitBreakerConfig:
    """Convert a host-neutral circuit-breaker config to tldw_server config."""
    if isinstance(config, CircuitBreakerConfig):
        return config
    return CircuitBreakerConfig(
        failure_threshold=config.failure_threshold,
        recovery_timeout=config.recovery_timeout,
        backoff_factor=config.backoff_factor,
        max_recovery_timeout=config.max_recovery_timeout,
        half_open_max_calls=config.half_open_max_calls,
        success_threshold=config.success_threshold,
        category=config.category,
        service=config.service,
    )


def create_tldw_circuit_breaker(*, name: str, config: Any) -> CircuitBreaker:
    """Create a tldw_server circuit breaker for an MCP module."""
    return CircuitBreaker(name=name, config=_to_tldw_circuit_breaker_config(config))


def build_default_runtime_dependencies() -> MCPRuntimeDependencies:
    """Build the default dependency bundle for the in-repo MCP server."""
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
