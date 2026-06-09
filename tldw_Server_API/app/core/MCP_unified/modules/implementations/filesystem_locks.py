"""Compatibility exports for MCP filesystem lock leases."""

from __future__ import annotations

from mcp_unified.filesystem_locks import (
    FilesystemLockConflict,
    FilesystemLockLease,
    FilesystemLockManager,
    FilesystemLockMissing,
    InMemoryFilesystemLockManager,
    create_filesystem_lock_manager,
)
# Compatibility export for older tests/callers that monkeypatch this module.
from mcp_unified.filesystem_locks.memory import time as time  # noqa: F401


__all__ = [
    "FilesystemLockConflict",
    "FilesystemLockLease",
    "FilesystemLockManager",
    "FilesystemLockMissing",
    "InMemoryFilesystemLockManager",
    "create_filesystem_lock_manager",
    "time",
]
