from __future__ import annotations

import pytest


def test_package_and_compatibility_imports_expose_same_lock_types() -> None:
    from mcp_unified.filesystem_locks import (
        FilesystemLockConflict,
        FilesystemLockLease,
        FilesystemLockManager,
        FilesystemLockMissing,
        InMemoryFilesystemLockManager,
        create_filesystem_lock_manager,
    )
    from tldw_Server_API.app.core.MCP_unified.modules.implementations.filesystem_locks import (
        FilesystemLockConflict as CompatConflict,
        FilesystemLockLease as CompatLease,
        FilesystemLockManager as CompatManagerProtocol,
        FilesystemLockMissing as CompatMissing,
        InMemoryFilesystemLockManager as CompatManager,
        create_filesystem_lock_manager as compat_create_filesystem_lock_manager,
    )

    assert CompatConflict is FilesystemLockConflict  # nosec B101
    assert CompatLease is FilesystemLockLease  # nosec B101
    assert CompatManagerProtocol is FilesystemLockManager  # nosec B101
    assert CompatMissing is FilesystemLockMissing  # nosec B101
    assert CompatManager is InMemoryFilesystemLockManager  # nosec B101
    assert compat_create_filesystem_lock_manager is create_filesystem_lock_manager  # nosec B101


def test_memory_lock_manager_acquire_conflict_renew_validate_release() -> None:
    from mcp_unified.filesystem_locks import (
        FilesystemLockConflict,
        FilesystemLockMissing,
        InMemoryFilesystemLockManager,
    )

    manager = InMemoryFilesystemLockManager()

    lease, renewed = manager.acquire(
        workspace_key="ws",
        path="docs/story.txt",
        owner="agent-a",
        ttl_seconds=60,
    )

    assert renewed is False  # nosec B101
    assert lease.workspace_key == "ws"  # nosec B101
    assert lease.path == "docs/story.txt"  # nosec B101
    assert lease.owner == "agent-a"  # nosec B101
    assert manager.validate(workspace_key="ws", path="docs/story.txt", lease_id=lease.lease_id) is lease  # nosec B101

    with pytest.raises(FilesystemLockConflict) as conflict:
        manager.acquire(
            workspace_key="ws",
            path="docs/story.txt",
            owner="agent-b",
            ttl_seconds=60,
        )
    assert conflict.value.lease is lease  # nosec B101

    renewed_lease, renewed = manager.acquire(
        workspace_key="ws",
        path="docs/story.txt",
        owner="agent-a",
        ttl_seconds=120,
        lease_id=lease.lease_id,
    )

    assert renewed is True  # nosec B101
    assert renewed_lease.lease_id == lease.lease_id  # nosec B101
    assert renewed_lease.ttl_seconds == 120  # nosec B101

    released = manager.release(workspace_key="ws", path="docs/story.txt", lease_id=lease.lease_id)
    assert released == renewed_lease  # nosec B101

    with pytest.raises(FilesystemLockMissing):
        manager.validate(workspace_key="ws", path="docs/story.txt", lease_id=lease.lease_id)


def test_filesystem_lock_manager_factory_backend_selection() -> None:
    from mcp_unified.filesystem_locks import InMemoryFilesystemLockManager, create_filesystem_lock_manager

    assert isinstance(create_filesystem_lock_manager(), InMemoryFilesystemLockManager)  # nosec B101
    assert isinstance(create_filesystem_lock_manager({}), InMemoryFilesystemLockManager)  # nosec B101
    assert isinstance(  # nosec B101
        create_filesystem_lock_manager({"lock_manager_backend": "memory"}),
        InMemoryFilesystemLockManager,
    )
    assert isinstance(  # nosec B101
        create_filesystem_lock_manager({"lock_manager_backend": "in_memory"}),
        InMemoryFilesystemLockManager,
    )

    with pytest.raises(ValueError, match="unsupported filesystem lock_manager_backend"):
        create_filesystem_lock_manager({"lock_manager_backend": ""})
