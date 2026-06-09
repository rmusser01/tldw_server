from __future__ import annotations

from pathlib import Path

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


def test_sqlite_lock_manager_coordinates_two_instances(tmp_path: Path) -> None:
    from mcp_unified.filesystem_locks import FilesystemLockConflict, SQLiteFilesystemLockManager

    db_path = tmp_path / "locks.db"
    first = SQLiteFilesystemLockManager(db_path)
    second = SQLiteFilesystemLockManager(db_path)
    try:
        lease, renewed = first.acquire(
            workspace_key="ws",
            path="docs/story.txt",
            owner="agent-a",
            ttl_seconds=60,
        )

        assert renewed is False  # nosec B101

        with pytest.raises(FilesystemLockConflict) as conflict:
            second.acquire(
                workspace_key="ws",
                path="docs/story.txt",
                owner="agent-b",
                ttl_seconds=60,
            )

        assert conflict.value.lease.lease_id == lease.lease_id  # nosec B101
    finally:
        first.close()
        second.close()


def test_sqlite_lock_manager_renews_matching_active_token(tmp_path: Path) -> None:
    from mcp_unified.filesystem_locks import SQLiteFilesystemLockManager

    db_path = tmp_path / "locks.db"
    manager = SQLiteFilesystemLockManager(db_path)
    try:
        lease, acquired_renewed = manager.acquire(
            workspace_key="ws",
            path="docs/story.txt",
            owner="agent-a",
            ttl_seconds=60,
        )

        renewed_lease, renewed = manager.acquire(
            workspace_key="ws",
            path="docs/story.txt",
            owner="agent-a",
            ttl_seconds=120,
            lease_id=lease.lease_id,
        )

        assert acquired_renewed is False  # nosec B101
        assert renewed is True  # nosec B101
        assert renewed_lease.lease_id == lease.lease_id  # nosec B101
        assert renewed_lease.ttl_seconds == 120  # nosec B101
        assert (  # nosec B101
            manager.validate(workspace_key="ws", path="docs/story.txt", lease_id=lease.lease_id)
            == renewed_lease
        )
    finally:
        manager.close()


def test_sqlite_lock_manager_wrong_active_token_renew_raises_conflict(tmp_path: Path) -> None:
    from mcp_unified.filesystem_locks import FilesystemLockConflict, SQLiteFilesystemLockManager

    db_path = tmp_path / "locks.db"
    manager = SQLiteFilesystemLockManager(db_path)
    try:
        lease, _ = manager.acquire(
            workspace_key="ws",
            path="docs/story.txt",
            owner="agent-a",
            ttl_seconds=60,
        )

        with pytest.raises(FilesystemLockConflict) as conflict:
            manager.acquire(
                workspace_key="ws",
                path="docs/story.txt",
                owner="agent-b",
                ttl_seconds=120,
                lease_id="wrong-token",
            )

        assert conflict.value.lease.lease_id == lease.lease_id  # nosec B101
    finally:
        manager.close()


def test_sqlite_lock_manager_releases_matching_active_token(tmp_path: Path) -> None:
    from mcp_unified.filesystem_locks import (
        FilesystemLockMissing,
        SQLiteFilesystemLockManager,
    )

    db_path = tmp_path / "locks.db"
    manager = SQLiteFilesystemLockManager(db_path)
    try:
        lease, _ = manager.acquire(
            workspace_key="ws",
            path="docs/story.txt",
            owner="agent-a",
            ttl_seconds=60,
        )

        released = manager.release(
            workspace_key="ws",
            path="docs/story.txt",
            lease_id=lease.lease_id,
        )

        assert released == lease  # nosec B101
        with pytest.raises(FilesystemLockMissing):
            manager.validate(workspace_key="ws", path="docs/story.txt", lease_id=lease.lease_id)
    finally:
        manager.close()


def test_sqlite_lock_manager_expired_row_does_not_block_new_acquire(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mcp_unified.filesystem_locks import SQLiteFilesystemLockManager
    import mcp_unified.filesystem_locks.sqlite as sqlite_locks

    db_path = tmp_path / "locks.db"
    manager = SQLiteFilesystemLockManager(db_path)
    try:
        monkeypatch.setattr(sqlite_locks.time, "time", lambda: 1_000.0)
        expired, _ = manager.acquire(
            workspace_key="ws",
            path="docs/story.txt",
            owner="agent-a",
            ttl_seconds=1,
        )

        monkeypatch.setattr(sqlite_locks.time, "time", lambda: 1_002.0)
        lease, renewed = manager.acquire(
            workspace_key="ws",
            path="docs/story.txt",
            owner="agent-b",
            ttl_seconds=60,
        )

        assert renewed is False  # nosec B101
        assert lease.lease_id != expired.lease_id  # nosec B101
        assert lease.owner == "agent-b"  # nosec B101
    finally:
        manager.close()


def test_sqlite_lock_manager_batches_expired_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mcp_unified.filesystem_locks import FilesystemLockMissing, SQLiteFilesystemLockManager
    import mcp_unified.filesystem_locks.sqlite as sqlite_locks

    db_path = tmp_path / "locks.db"
    manager = SQLiteFilesystemLockManager(db_path, cleanup_interval=100)
    try:
        monkeypatch.setattr(sqlite_locks.time, "time", lambda: 1_000.0)
        first, _ = manager.acquire(
            workspace_key="ws",
            path="docs/one.txt",
            owner="agent-a",
            ttl_seconds=1,
        )
        second, _ = manager.acquire(
            workspace_key="ws",
            path="docs/two.txt",
            owner="agent-a",
            ttl_seconds=1,
        )

        def _fail_per_row_cleanup(*args: object, **kwargs: object) -> None:
            raise AssertionError("cleanup should not delete expired rows one at a time")

        monkeypatch.setattr(manager, "_delete_key", _fail_per_row_cleanup)
        manager._cleanup_interval = 1
        monkeypatch.setattr(sqlite_locks.time, "time", lambda: 1_002.0)

        lease, renewed = manager.acquire(
            workspace_key="ws",
            path="docs/three.txt",
            owner="agent-b",
            ttl_seconds=60,
        )

        assert renewed is False  # nosec B101
        assert lease.path == "docs/three.txt"  # nosec B101
        with pytest.raises(FilesystemLockMissing):
            manager.validate(workspace_key="ws", path="docs/one.txt", lease_id=first.lease_id)
        with pytest.raises(FilesystemLockMissing):
            manager.validate(workspace_key="ws", path="docs/two.txt", lease_id=second.lease_id)
    finally:
        manager.close()


def test_sqlite_lock_manager_expired_token_renew_raises_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mcp_unified.filesystem_locks import FilesystemLockMissing, SQLiteFilesystemLockManager
    import mcp_unified.filesystem_locks.sqlite as sqlite_locks

    db_path = tmp_path / "locks.db"
    manager = SQLiteFilesystemLockManager(db_path)
    try:
        monkeypatch.setattr(sqlite_locks.time, "time", lambda: 1_000.0)
        lease, _ = manager.acquire(
            workspace_key="ws",
            path="docs/story.txt",
            owner="agent-a",
            ttl_seconds=1,
        )

        monkeypatch.setattr(sqlite_locks.time, "time", lambda: 1_002.0)
        with pytest.raises(FilesystemLockMissing):
            manager.acquire(
                workspace_key="ws",
                path="docs/story.txt",
                owner="agent-a",
                ttl_seconds=60,
                lease_id=lease.lease_id,
            )
    finally:
        manager.close()


def test_sqlite_lock_manager_wrong_token_release_raises_conflict(tmp_path: Path) -> None:
    from mcp_unified.filesystem_locks import FilesystemLockConflict, SQLiteFilesystemLockManager

    db_path = tmp_path / "locks.db"
    manager = SQLiteFilesystemLockManager(db_path)
    try:
        lease, _ = manager.acquire(
            workspace_key="ws",
            path="docs/story.txt",
            owner="agent-a",
            ttl_seconds=60,
        )

        with pytest.raises(FilesystemLockConflict) as conflict:
            manager.release(
                workspace_key="ws",
                path="docs/story.txt",
                lease_id="wrong-token",
            )

        assert conflict.value.lease.lease_id == lease.lease_id  # nosec B101
    finally:
        manager.close()


def test_sqlite_lock_manager_missing_and_expired_release_return_none(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mcp_unified.filesystem_locks import SQLiteFilesystemLockManager
    import mcp_unified.filesystem_locks.sqlite as sqlite_locks

    db_path = tmp_path / "locks.db"
    manager = SQLiteFilesystemLockManager(db_path)
    try:
        assert (  # nosec B101
            manager.release(workspace_key="ws", path="missing.txt", lease_id="missing") is None
        )

        monkeypatch.setattr(sqlite_locks.time, "time", lambda: 1_000.0)
        lease, _ = manager.acquire(
            workspace_key="ws",
            path="docs/story.txt",
            owner="agent-a",
            ttl_seconds=1,
        )

        monkeypatch.setattr(sqlite_locks.time, "time", lambda: 1_002.0)
        assert (  # nosec B101
            manager.release(workspace_key="ws", path="docs/story.txt", lease_id=lease.lease_id)
            is None
        )
    finally:
        manager.close()


def test_sqlite_lock_manager_validate_classifies_matching_wrong_and_missing_tokens(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mcp_unified.filesystem_locks import (
        FilesystemLockConflict,
        FilesystemLockMissing,
        SQLiteFilesystemLockManager,
    )
    import mcp_unified.filesystem_locks.sqlite as sqlite_locks

    db_path = tmp_path / "locks.db"
    manager = SQLiteFilesystemLockManager(db_path)
    try:
        monkeypatch.setattr(sqlite_locks.time, "time", lambda: 1_000.0)
        lease, _ = manager.acquire(
            workspace_key="ws",
            path="docs/story.txt",
            owner="agent-a",
            ttl_seconds=1,
        )

        assert (  # nosec B101
            manager.validate(workspace_key="ws", path="docs/story.txt", lease_id=lease.lease_id)
            == lease
        )

        with pytest.raises(FilesystemLockConflict) as conflict:
            manager.validate(
                workspace_key="ws",
                path="docs/story.txt",
                lease_id="wrong-token",
            )
        assert conflict.value.lease == lease  # nosec B101

        with pytest.raises(FilesystemLockMissing):
            manager.validate(workspace_key="ws", path="missing.txt", lease_id=lease.lease_id)

        monkeypatch.setattr(sqlite_locks.time, "time", lambda: 1_002.0)
        with pytest.raises(FilesystemLockMissing):
            manager.validate(workspace_key="ws", path="docs/story.txt", lease_id=lease.lease_id)
    finally:
        manager.close()


def test_filesystem_lock_manager_factory_sqlite_backend_selection(tmp_path: Path) -> None:
    from mcp_unified.filesystem_locks import SQLiteFilesystemLockManager, create_filesystem_lock_manager

    manager = create_filesystem_lock_manager(
        {
            "lock_manager_backend": "sqlite",
            "lock_manager_sqlite_path": str(tmp_path / "locks.db"),
        }
    )
    try:
        assert isinstance(manager, SQLiteFilesystemLockManager)  # nosec B101
    finally:
        manager.close()


def test_sqlite_lock_manager_normalizes_configured_path_whitespace(tmp_path: Path) -> None:
    from mcp_unified.filesystem_locks import (
        SQLiteFilesystemLockManager,
        create_filesystem_lock_manager,
    )

    db_path = tmp_path / "locks.db"
    direct = SQLiteFilesystemLockManager(f"{db_path} ")
    try:
        assert direct.path == str(db_path)  # nosec B101
    finally:
        direct.close()

    from_factory = create_filesystem_lock_manager(
        {
            "lock_manager_backend": "sqlite",
            "lock_manager_sqlite_path": f" {db_path} ",
        }
    )
    try:
        assert isinstance(from_factory, SQLiteFilesystemLockManager)  # nosec B101
        assert from_factory.path == str(db_path)  # nosec B101
    finally:
        from_factory.close()


def test_filesystem_lock_manager_factory_sqlite_backend_requires_path() -> None:
    from mcp_unified.filesystem_locks import create_filesystem_lock_manager

    with pytest.raises(ValueError, match="lock_manager_sqlite_path"):
        create_filesystem_lock_manager({"lock_manager_backend": "sqlite"})
