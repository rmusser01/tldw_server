"""Standalone non-spawning external federation shell."""

from . import catalog_loader, config_schema
from .catalog_loader import get_catalog_entry, list_catalog_entries, load_mcp_catalog
from .config_schema import (
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
    load_external_server_registry,
    parse_external_server_registry,
)
from .installers import ExternalServerInstaller, NullExternalServerInstaller
from .manager import ExternalFederationManager
from .models import (
    BrokeredExternalCredential,
    ExternalToolCallResult,
    ExternalToolDefinition,
    FederatedToolResult,
    FederationPolicyDenied,
    MCPAuthType,
    MCPCatalogEntry,
    VirtualExternalTool,
)
from .transports import ExternalFederationTransport, FakeExternalTransport

__all__ = [
    "BrokeredExternalCredential",
    "ExternalAuthConfig",
    "ExternalAuthMode",
    "ExternalCircuitBreakerConfig",
    "ExternalFederationManager",
    "ExternalFederationTransport",
    "ExternalMCPServerConfig",
    "ExternalRetryConfig",
    "ExternalServerInstaller",
    "ExternalServerRegistryConfig",
    "ExternalServerRegistryPartialLoadError",
    "ExternalStdioConfig",
    "ExternalTimeoutConfig",
    "ExternalToolPolicy",
    "ExternalToolCallResult",
    "ExternalToolDefinition",
    "ExternalTransportType",
    "ExternalWebSocketConfig",
    "FakeExternalTransport",
    "FederatedToolResult",
    "FederationPolicyDenied",
    "MCPAuthType",
    "MCPCatalogEntry",
    "NullExternalServerInstaller",
    "VirtualExternalTool",
    "catalog_loader",
    "config_schema",
    "get_catalog_entry",
    "list_catalog_entries",
    "load_external_server_registry",
    "load_mcp_catalog",
    "parse_external_server_registry",
]
