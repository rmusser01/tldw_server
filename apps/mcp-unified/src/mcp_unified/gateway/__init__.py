"""Standalone MCP gateway entrypoint helpers."""

from typing import TYPE_CHECKING, Any

from .bootstrap import (
    GatewayProfileBootstrap,
    bootstrap_profile_gateway,
    build_profile_gateway_runtime,
)
from .config import (
    GatewayAdminAuthBootstrapConfig,
    GatewayConfigFormat,
    GatewayExternalRuntimeBootstrapConfig,
    GatewayProfileBootstrapConfig,
    GatewayProfileStoreConfig,
    GatewayProfileStoreKind,
    GatewayToolUseReportingConfig,
    GatewayToolUseReportingStoreConfig,
    GatewayToolUseReportingStoreKind,
    bootstrap_profile_gateway_from_config,
    build_gateway_tool_use_event_store,
    build_gateway_tool_use_recorder,
    credential_grant_manager_from_storage,
    load_gateway_profile_bootstrap_config,
)
from .credential_grants import (
    GatewayCredentialGrantManagementError,
    GatewayCredentialGrantManager,
)
from .lifecycle import GatewayExternalRuntimeLifecycleConfig
from .policy_explain import (
    GatewayPolicyExplainError,
    GatewayPolicyExplainService,
    PolicyExplainRequest,
    ProfileToolPreviewRequest,
)
from .profile_governance import (
    AllowPermissionChangeGovernor,
    PermissionChangeDecision,
    PermissionChangeGovernor,
    PermissionChangeRequest,
)
from .profile_runtime import ProfileAwareGatewayRuntime
from .profiles import (
    GatewayProfileManagementError,
    GatewayProfileManager,
    GatewayProfileStoreMetadata,
)
from .protocol_cancellation import GatewayCancellationToken
from .protocol_connection import GatewayProtocolConnection
from .protocol_errors import (
    GatewayApplicationError,
    GatewayInvalidApplicationResult,
    GatewayResourceNotFound,
    GatewayResultTooLarge,
    GatewayToolExecutionError,
)
from .protocol_limits import GatewayLimits
from .protocol_profiles import (
    CURRENT_PROTOCOL_VERSION,
    PREFERRED_LEGACY_PROTOCOL_VERSION,
    PROTOCOL_PROFILES,
    SUPPORTED_LEGACY_PROTOCOL_VERSIONS,
    SUPPORTED_MODERN_PROTOCOL_VERSIONS,
    SUPPORTED_PROTOCOL_VERSIONS,
    GatewayProtocolProfile,
)
from .protocol_stdio import (
    GatewayAsyncByteReader,
    GatewayAsyncByteWriter,
    GatewayProtocolStdioServer,
    serve_stdio,
)
from .runtime import (
    GatewayCoreRuntime,
    GatewayJSONValue,
    GatewayPolicyDenied,
    GatewayRequestContext,
    GatewayResourceTemplateRuntime,
    GatewayRuntime,
)
from .stdio import GatewayStdioServer, handle_stdio_line
from .tool_use_reporting import ToolUseReportingGatewayRuntime

if TYPE_CHECKING:
    from .admin_auth import GatewayAdminAuthConfig, GatewayAdminAuthError
    from .external_runtime import (
        GatewayExternalRuntimeError,
        GatewayExternalRuntimeManager,
    )
    from .external_runtime_adapter import ExternalRuntimeGatewayRuntime
    from .fastapi import create_gateway_app, create_gateway_router

__all__ = [
    "CURRENT_PROTOCOL_VERSION",
    "PREFERRED_LEGACY_PROTOCOL_VERSION",
    "PROTOCOL_PROFILES",
    "SUPPORTED_LEGACY_PROTOCOL_VERSIONS",
    "SUPPORTED_MODERN_PROTOCOL_VERSIONS",
    "SUPPORTED_PROTOCOL_VERSIONS",
    "GatewayApplicationError",
    "GatewayAsyncByteReader",
    "GatewayAsyncByteWriter",
    "GatewayCancellationToken",
    "GatewayCoreRuntime",
    "GatewayInvalidApplicationResult",
    "GatewayJSONValue",
    "GatewayLimits",
    "GatewayPolicyDenied",
    "GatewayProtocolConnection",
    "GatewayProtocolProfile",
    "GatewayProtocolStdioServer",
    "GatewayResourceNotFound",
    "GatewayResourceTemplateRuntime",
    "GatewayResultTooLarge",
    "GatewayAdminAuthBootstrapConfig",
    "GatewayAdminAuthConfig",
    "GatewayAdminAuthError",
    "GatewayConfigFormat",
    "GatewayCredentialGrantManagementError",
    "GatewayCredentialGrantManager",
    "GatewayExternalRuntimeBootstrapConfig",
    "GatewayExternalRuntimeLifecycleConfig",
    "GatewayExternalRuntimeError",
    "GatewayExternalRuntimeManager",
    "ExternalRuntimeGatewayRuntime",
    "AllowPermissionChangeGovernor",
    "GatewayPolicyExplainError",
    "GatewayPolicyExplainService",
    "GatewayProfileBootstrap",
    "GatewayProfileBootstrapConfig",
    "GatewayProfileManagementError",
    "GatewayProfileManager",
    "GatewayProfileStoreMetadata",
    "GatewayRequestContext",
    "GatewayRuntime",
    "GatewayStdioServer",
    "GatewayToolExecutionError",
    "GatewayProfileStoreConfig",
    "GatewayProfileStoreKind",
    "GatewayToolUseReportingConfig",
    "GatewayToolUseReportingStoreConfig",
    "GatewayToolUseReportingStoreKind",
    "PermissionChangeDecision",
    "PermissionChangeGovernor",
    "PermissionChangeRequest",
    "PolicyExplainRequest",
    "ProfileToolPreviewRequest",
    "ToolUseReportingGatewayRuntime",
    "ProfileAwareGatewayRuntime",
    "bootstrap_profile_gateway",
    "bootstrap_profile_gateway_from_config",
    "build_gateway_tool_use_event_store",
    "build_gateway_tool_use_recorder",
    "build_profile_gateway_runtime",
    "create_gateway_app",
    "create_gateway_router",
    "credential_grant_manager_from_storage",
    "handle_stdio_line",
    "load_gateway_profile_bootstrap_config",
    "serve_stdio",
]


def __getattr__(name: str) -> Any:
    """Lazily expose FastAPI helpers so stdio imports do not require FastAPI."""

    if name in {"create_gateway_app", "create_gateway_router"}:
        from .fastapi import create_gateway_app, create_gateway_router

        return {
            "create_gateway_app": create_gateway_app,
            "create_gateway_router": create_gateway_router,
        }[name]
    if name in {"GatewayAdminAuthConfig", "GatewayAdminAuthError"}:
        from .admin_auth import GatewayAdminAuthConfig, GatewayAdminAuthError

        return {
            "GatewayAdminAuthConfig": GatewayAdminAuthConfig,
            "GatewayAdminAuthError": GatewayAdminAuthError,
        }[name]
    if name in {"GatewayExternalRuntimeError", "GatewayExternalRuntimeManager"}:
        from .external_runtime import (
            GatewayExternalRuntimeError,
            GatewayExternalRuntimeManager,
        )

        return {
            "GatewayExternalRuntimeError": GatewayExternalRuntimeError,
            "GatewayExternalRuntimeManager": GatewayExternalRuntimeManager,
        }[name]
    if name == "ExternalRuntimeGatewayRuntime":
        from .external_runtime_adapter import ExternalRuntimeGatewayRuntime

        return ExternalRuntimeGatewayRuntime
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
