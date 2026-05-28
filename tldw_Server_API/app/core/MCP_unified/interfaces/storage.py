"""Compatibility re-exports for MCP Unified storage interfaces."""

from mcp_unified.interfaces.storage import (
    ApprovalPolicyStore,
    AuditStore,
    CredentialGrantStore,
    ExternalRegistryStore,
    ProfileAssignmentStore,
    ProfileStore,
)

__all__ = [
    "ApprovalPolicyStore",
    "AuditStore",
    "CredentialGrantStore",
    "ExternalRegistryStore",
    "ProfileAssignmentStore",
    "ProfileStore",
]
