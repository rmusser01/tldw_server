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
from tldw_Server_API.app.core.MCP_unified.auth.jwt_manager import get_jwt_manager
from tldw_Server_API.app.core.MCP_unified.auth.rate_limiter import get_rate_limiter
from tldw_Server_API.app.core.MCP_unified.interfaces.runtime import (
    AuthenticatedIdentity,
    MCPRuntimeDependencies,
    WebSocketStream,
)
from tldw_Server_API.app.core.MCP_unified.modules.registry import get_module_registry
from tldw_Server_API.app.core.MCP_unified.monitoring.metrics import get_metrics_collector
from tldw_Server_API.app.core.Metrics.telemetry import get_telemetry_manager

_TLDW_RUNTIME_ADAPTER_EXCEPTIONS = (
    AssertionError,
    AttributeError,
    ConnectionError,
    FileNotFoundError,
    ImportError,
    KeyError,
    LookupError,
    OSError,
    PermissionError,
    RuntimeError,
    TimeoutError,
    TypeError,
    UnicodeError,
    ValueError,
    json.JSONDecodeError,
)


class TldwTelemetryProvider:
    """Proxy the current tldw_server telemetry manager for MCP protocol spans."""

    @staticmethod
    def _current() -> Any:
        """Return the current host telemetry manager."""
        return get_telemetry_manager()

    def trace_context(
        self,
        operation_name: str,
        attributes: dict[str, Any] | None = None,
    ) -> Any:
        """Open a trace context through the current host telemetry manager."""
        return self._current().trace_context(operation_name, attributes)

    def __getattr__(self, name: str) -> Any:
        """Forward compatibility methods to the current host telemetry manager."""
        return getattr(self._current(), name)


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
        except _TLDW_RUNTIME_ADAPTER_EXCEPTIONS as exc:
            logger.debug(
                "MCP user database path resolution failed; returning empty paths: {}",
                exc.__class__.__name__,
            )
            return {}


class TldwToolCatalogProvider:
    """Resolve host-managed tool catalog filters through the AuthNZ database."""

    async def resolve_tool_names(
        self,
        *,
        catalog_name: str | None,
        catalog_id: Any,
        metadata: dict[str, Any],
        strict: bool,
    ) -> set[str] | None:
        """Return tool names for a catalog identifier or name, preserving legacy fallback semantics."""
        try:
            from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
            from tldw_Server_API.app.services.admin_tool_catalog_service import (
                resolve_tool_catalog_filter_names,
            )

            pool = await get_db_pool()
        except _TLDW_RUNTIME_ADAPTER_EXCEPTIONS as exc:
            logger.debug(
                "MCP catalog lookup unavailable; returning fallback: {}",
                exc.__class__.__name__,
            )
            return set() if strict else None

        try:
            return await resolve_tool_catalog_filter_names(
                pool,
                catalog_name=catalog_name,
                catalog_id=catalog_id,
                metadata=metadata,
                strict=strict,
            )
        except _TLDW_RUNTIME_ADAPTER_EXCEPTIONS as exc:
            logger.debug(
                "MCP catalog lookup failed; returning fallback: {}",
                exc.__class__.__name__,
            )
            return set() if strict else None


class TldwApiKeyScopeNormalizer:
    """Normalize tldw_server API-key scope payloads for MCP authorization."""

    def normalize(self, raw_scopes: Any) -> set[str]:
        """Return normalized scope strings, falling back to local parsing."""
        try:
            from tldw_Server_API.app.core.AuthNZ.api_key_manager import normalize_scope

            return set(normalize_scope(raw_scopes))
        except _TLDW_RUNTIME_ADAPTER_EXCEPTIONS as exc:
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


class TldwServerAuthProvider:
    """Authenticate MCP transport callers through current tldw_server services."""

    def __init__(self) -> None:
        self._jwt_manager = get_jwt_manager()
        self._scope_normalizer = TldwApiKeyScopeNormalizer()

    def get_mcp_jwt_manager(self) -> Any:
        """Return the in-repo MCP JWT manager used for legacy MCP tokens."""
        return self._jwt_manager

    def is_authnz_access_token(self, token: str) -> bool:
        """Return True when the token verifies as an AuthNZ access token."""
        try:
            from tldw_Server_API.app.core.AuthNZ.exceptions import (
                InvalidTokenError,
                TokenExpiredError,
            )
            from tldw_Server_API.app.core.AuthNZ.jwt_service import get_jwt_service

            jwt_service = get_jwt_service()
            jwt_service.decode_access_token(token)
            return True
        except TokenExpiredError:
            return True
        except InvalidTokenError:
            return False
        except _TLDW_RUNTIME_ADAPTER_EXCEPTIONS as exc:
            logger.debug(
                "MCP AuthNZ token detection failed closed: {}",
                exc.__class__.__name__,
            )
            return False

    async def authenticate_authnz_websocket_token(
        self,
        token: str,
        *,
        websocket: Any,
    ) -> AuthenticatedIdentity | None:
        """Authenticate an AuthNZ websocket bearer token and project identity data."""
        try:
            from starlette.requests import Request

            from tldw_Server_API.app.core.AuthNZ.exceptions import (
                InvalidTokenError,
                TokenExpiredError,
            )
            from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import (
                verify_jwt_and_fetch_user,
            )
        except _TLDW_RUNTIME_ADAPTER_EXCEPTIONS as exc:
            logger.debug(
                "MCP AuthNZ websocket dependencies unavailable: {}",
                exc.__class__.__name__,
            )
            return None

        try:
            scope = {
                "type": "http",
                "method": "GET",
                "path": "/api/v1/mcp/ws",
                "headers": [
                    (key.encode("latin-1"), value.encode("latin-1"))
                    for key, value in websocket.headers.items()
                ],
            }
            client = websocket.client
            if isinstance(client, (list, tuple)) and len(client) >= 2:
                scope["client"] = (client[0], client[1])
            elif client is not None and getattr(client, "host", None) is not None:
                scope["client"] = (client.host, getattr(client, "port", 0))
        except _TLDW_RUNTIME_ADAPTER_EXCEPTIONS as exc:
            logger.debug(
                "MCP AuthNZ websocket scope projection failed: {}",
                exc.__class__.__name__,
            )
            return None

        try:
            user = await verify_jwt_and_fetch_user(Request(scope), token)
        except (InvalidTokenError, TokenExpiredError) as exc:
            logger.debug(
                "MCP AuthNZ websocket token authentication failed closed: {}",
                exc.__class__.__name__,
            )
            return None
        except _TLDW_RUNTIME_ADAPTER_EXCEPTIONS as exc:
            logger.debug(
                "MCP AuthNZ websocket user lookup failed closed: {}",
                exc.__class__.__name__,
            )
            return None
        user_id = str(getattr(user, "id", None) or "")
        if not user_id:
            return None
        return AuthenticatedIdentity(
            user_id=user_id,
            roles=list(getattr(user, "roles", []) or []),
            permissions=list(getattr(user, "permissions", []) or []),
        )

    async def validate_api_key(
        self,
        api_key: str,
        *,
        ip_address: str | None = None,
    ) -> dict[str, Any] | None:
        """Validate a tldw_server API key for MCP transport authentication."""
        from tldw_Server_API.app.core.AuthNZ.api_key_manager import get_api_key_manager

        manager = await get_api_key_manager()
        return await manager.validate_api_key(api_key, ip_address=ip_address)

    def normalize_api_key_permissions(self, info: dict[str, Any] | None) -> list[str]:
        """Normalize API-key scope payloads into MCP permission strings."""
        if not info:
            return []
        raw_scopes = info.get("scopes")
        if raw_scopes is None:
            raw_scopes = info.get("scope")
        scopes = self._scope_normalizer.normalize(raw_scopes)
        return sorted(scopes) if scopes else []


class TldwLifecycleGuard:
    """Bridge MCP server lifecycle hooks to tldw_server app lifecycle services."""

    def assert_may_start_work(self, app: Any, family: str) -> None:
        """Raise when the host app is draining and new work should not start."""
        from tldw_Server_API.app.services.app_lifecycle import assert_may_start_work

        assert_may_start_work(app, family)

    def register_shutdown_transport_family(
        self,
        family: str,
        *,
        active_count: Any,
        drain: Any,
    ) -> None:
        """Register a transport family with the host shutdown registry."""
        from tldw_Server_API.app.services.shutdown_transport_registry import (
            register_shutdown_transport_family,
        )

        register_shutdown_transport_family(
            family,
            active_count=active_count,
            drain=drain,
        )


class TldwPermissionSeeder:
    """Seed MCP compatibility permissions through the current AuthNZ database."""

    async def seed_default_tool_permissions(self) -> None:
        """Ensure the legacy wildcard tool execution permission exists."""
        from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
        from tldw_Server_API.app.services.admin_roles_permissions_service import (
            ensure_permission,
        )

        pool = await get_db_pool()
        async with pool.acquire() as conn:
            await ensure_permission(
                conn,
                "tools.execute:*",
                "Wildcard tool execution",
                category="tools",
            )


class TldwModuleConfigProvider:
    """Provide tldw_server defaults for MCP module configuration."""

    def default_media_db_path(self) -> str:
        """Return the single-user media database path used by legacy module defaults."""
        from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths

        return str(DatabasePaths.get_media_db_path(DatabasePaths.get_single_user_id()))


class TldwPolicyContextProvider:
    """Expose host MCP Hub policy-context feature flags to MCPServer."""

    def is_policy_context_enabled(self) -> bool:
        """Return whether host MCP Hub policy-context metadata should be attached."""
        from tldw_Server_API.app.core.feature_flags import (
            is_mcp_hub_policy_enforcement_enabled,
        )

        return is_mcp_hub_policy_enforcement_enabled()


class TldwEnvironmentFlagsProvider:
    """Expose tldw_server environment and test-mode helpers to MCPServer."""

    def env_flag_enabled(self, name: str) -> bool:
        """Return whether a named environment flag is enabled by host rules."""
        from tldw_Server_API.app.core.testing import env_flag_enabled

        return env_flag_enabled(name)

    def is_test_mode(self) -> bool:
        """Return whether the host process is running in test mode."""
        from tldw_Server_API.app.core.testing import is_test_mode

        return is_test_mode()

    def is_explicit_pytest_runtime(self) -> bool:
        """Return whether the host process is an explicit pytest runtime."""
        from tldw_Server_API.app.core.testing import is_explicit_pytest_runtime

        return is_explicit_pytest_runtime()

    def is_truthy(self, value: Any) -> bool:
        """Normalize host truthy environment-style values."""
        from tldw_Server_API.app.core.testing import is_truthy

        return is_truthy(value)


class TldwWebSocketStreamFactory:
    """Create tldw_server WebSocketStream wrappers for MCP websocket sessions."""

    def __call__(
        self,
        websocket: Any,
        *,
        heartbeat_interval_s: float | None,
        idle_timeout_s: float | None,
        close_on_done: bool,
        labels: dict[str, str],
    ) -> WebSocketStream:
        """Return a host WebSocketStream configured for MCP transport lifecycle."""
        from tldw_Server_API.app.core.Streaming.streams import WebSocketStream

        return WebSocketStream(
            websocket,
            heartbeat_interval_s=heartbeat_interval_s,
            idle_timeout_s=idle_timeout_s,
            close_on_done=close_on_done,
            labels=labels,
        )


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
        telemetry_provider=TldwTelemetryProvider(),
        database_path_resolver=TldwDatabasePathResolver(),
        tool_catalog_provider=TldwToolCatalogProvider(),
        api_key_scope_normalizer=TldwApiKeyScopeNormalizer(),
        effective_policy_resolver=TldwEffectivePolicyResolver(),
        approval_evaluator=TldwApprovalEvaluator(),
        path_scope_enforcer=TldwPathScopeEnforcer(),
        external_access_evaluator=TldwExternalAccessEvaluator(),
        redis_client_factory=create_async_redis_client,
        circuit_breaker_factory=create_tldw_circuit_breaker,
        auth_provider=TldwServerAuthProvider(),
        lifecycle_guard=TldwLifecycleGuard(),
        permission_seeder=TldwPermissionSeeder(),
        module_config_provider=TldwModuleConfigProvider(),
        policy_context_provider=TldwPolicyContextProvider(),
        environment_flags_provider=TldwEnvironmentFlagsProvider(),
        websocket_stream_factory=TldwWebSocketStreamFactory(),
    )
