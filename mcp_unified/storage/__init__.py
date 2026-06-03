"""Storage payload models for MCP Unified standalone stores."""

from typing import TYPE_CHECKING, Any

from .models import (
    ApprovalPolicyDocument,
    AuditEvent,
    CredentialGrant,
    ExternalServerDefinition,
    ProfileAssignment,
)

if TYPE_CHECKING:
    from .sqlite import SQLiteMCPStore

__all__ = [
    "ApprovalPolicyDocument",
    "AuditEvent",
    "CredentialGrant",
    "ExternalServerDefinition",
    "ProfileAssignment",
    "SQLiteMCPStore",
]


def __getattr__(name: str) -> Any:
    """Lazily expose SQLite storage so core imports do not require SQLAlchemy."""

    if name == "SQLiteMCPStore":
        try:
            from .sqlite import SQLiteMCPStore
        except ModuleNotFoundError as exc:
            if exc.name == "sqlalchemy":
                raise ImportError(
                    "SQLiteMCPStore requires the mcp-unified sqlite extra. "
                    "Install mcp-unified[sqlite] or mcp-unified[gateway]."
                ) from exc
            raise
        return SQLiteMCPStore
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
