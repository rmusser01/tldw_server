"""Compatibility wrapper for external MCP federation config schemas."""

from __future__ import annotations

from mcp_unified.federation.config_schema import (
    ExternalAuthConfig,
    ExternalAuthMode,
    ExternalCircuitBreakerConfig,
    ExternalMCPServerConfig,
    ExternalRetryConfig,
    ExternalServerRegistryConfig,
    ExternalServerRegistryPartialLoadError,
    ExternalStdioConfig,
    ExternalTimeoutConfig,
    ExternalToolPolicy,
    ExternalTransportType,
    ExternalWebSocketConfig,
    parse_external_server_registry,
)
from mcp_unified.federation.config_schema import (
    load_external_server_registry as _load_external_server_registry,
)

_DEFAULT_CONFIG_PATH = "tldw_Server_API/Config_Files/mcp_external_servers.yaml"


def load_external_server_registry(
    config_path: str | None = None,
) -> ExternalServerRegistryConfig:
    """Load the host external registry using the historical default config path."""

    return _load_external_server_registry(
        config_path,
        default_config_path=_DEFAULT_CONFIG_PATH,
    )


__all__ = [
    "ExternalAuthConfig",
    "ExternalAuthMode",
    "ExternalCircuitBreakerConfig",
    "ExternalMCPServerConfig",
    "ExternalRetryConfig",
    "ExternalServerRegistryConfig",
    "ExternalServerRegistryPartialLoadError",
    "ExternalStdioConfig",
    "ExternalTimeoutConfig",
    "ExternalToolPolicy",
    "ExternalTransportType",
    "ExternalWebSocketConfig",
    "load_external_server_registry",
    "parse_external_server_registry",
]
