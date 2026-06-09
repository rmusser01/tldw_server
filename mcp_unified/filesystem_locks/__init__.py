"""Filesystem lock lease managers for MCP Unified."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .memory import InMemoryFilesystemLockManager, create_filesystem_lock_manager
from .models import (
    FilesystemLockConflict,
    FilesystemLockLease,
    FilesystemLockManager,
    FilesystemLockMissing,
)

if TYPE_CHECKING:
    from .sqlite import SQLiteFilesystemLockManager

__all__ = [
    "FilesystemLockConflict",
    "FilesystemLockLease",
    "FilesystemLockManager",
    "FilesystemLockMissing",
    "InMemoryFilesystemLockManager",
    "SQLiteFilesystemLockManager",
    "create_filesystem_lock_manager",
]


def __getattr__(name: str) -> Any:
    """Lazily expose SQLite locks so core imports do not require SQLAlchemy."""

    if name == "SQLiteFilesystemLockManager":
        try:
            from .sqlite import SQLiteFilesystemLockManager
        except ModuleNotFoundError as exc:
            if exc.name == "sqlalchemy":
                raise ImportError(
                    "SQLiteFilesystemLockManager requires the mcp-unified sqlite extra. "
                    "Install mcp-unified[sqlite] or mcp-unified[gateway]."
                ) from exc
            raise
        return SQLiteFilesystemLockManager
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
