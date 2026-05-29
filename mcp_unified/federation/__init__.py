"""Standalone non-spawning external federation shell."""

from . import catalog_loader
from .catalog_loader import get_catalog_entry, list_catalog_entries, load_mcp_catalog
from .manager import ExternalFederationManager
from .models import (
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
    "ExternalFederationManager",
    "ExternalFederationTransport",
    "ExternalToolCallResult",
    "ExternalToolDefinition",
    "FakeExternalTransport",
    "FederatedToolResult",
    "FederationPolicyDenied",
    "MCPAuthType",
    "MCPCatalogEntry",
    "VirtualExternalTool",
    "catalog_loader",
    "get_catalog_entry",
    "list_catalog_entries",
    "load_mcp_catalog",
]
