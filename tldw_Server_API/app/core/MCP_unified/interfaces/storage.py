"""Compatibility re-exports for MCP Unified storage interfaces."""

from mcp_unified.interfaces.storage import AuditStore, ExternalRegistryStore, ProfileStore

__all__ = ["AuditStore", "ExternalRegistryStore", "ProfileStore"]
