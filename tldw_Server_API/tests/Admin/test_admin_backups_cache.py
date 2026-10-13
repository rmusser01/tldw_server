"""Unit tests for the backups list TTL cache and page-scoped stat work.

Covers the admin perf A-S4 remediation for ``GET /admin/backups``:

1. Repeated ``list_backup_items`` calls within ``BACKUP_SCAN_CACHE_TTL_SEC``
   skip the filesystem scan entirely.
2. Backup write paths (create/restore) invalidate the cache so the next list
   call re-scans.
3. Pagination keeps returning a truthful total over the full tree, newest
   first, while only building ``BackupFile`` objects for the requested page.

These tests call the service directly (no HTTP), so the module-level cache is
reset around every test to avoid leaking rows between per-test tmp backup
roots.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from tldw_Server_API.app.services import admin_data_ops_service as svc


@pytest.fixture(autouse=True)
def _fresh_backup_cache():
    svc._reset_backup_cache_for_tests()
    yield
    svc._reset_backup_cache_for_tests()


@pytest.fixture
def backup_root(tmp_path, monkeypatch) -> Path:
    root = tmp_path / "backups"
    monkeypatch.setenv("TLDW_DB_BACKUP_PATH", str(root))
    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path / "user_dbs"))
    return root


class _ScandirSpy:
    """Wrap os.scandir and count invocations."""

    def __init__(self, original):
        self._original = original
        self.calls = 0

    def __call__(self, path="."):
        self.calls += 1
        return self._original(path)


def _count_scandir(monkeypatch) -> _ScandirSpy:
    spy = _ScandirSpy(os.scandir)
    monkeypatch.setattr(os, "scandir", spy)
    return spy


def _dataset_dir(root: Path, dataset: str, user_id: int | None) -> Path:
    if user_id is None:
        return root / dataset
    return root / f"user_{user_id}" / dataset


def _write_backup_file(directory: Path, name: str, mtime: float) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_bytes(b"backup-payload")
    os.utime(path, (mtime, mtime))
    return path


def test_second_call_within_ttl_skips_scan(backup_root, monkeypatch):
    spy = _count_scandir(monkeypatch)
    media_dir = _dataset_dir(backup_root, "media", 1)
    _write_backup_file(media_dir, "a.db", mtime=1_000.0)
    _write_backup_file(media_dir, "b.db", mtime=2_000.0)

    first, total_first = svc.list_backup_items(dataset="media", user_id=1, limit=10, offset=0)
    assert total_first == 2
    assert [item.filename for item in first] == ["b.db", "a.db"]
    scans_after_first = spy.calls
    assert scans_after_first >= 1

    second, total_second = svc.list_backup_items(dataset="media", user_id=1, limit=10, offset=0)
    assert total_second == 2
    assert [item.filename for item in second] == [item.filename for item in first]
    assert spy.calls == scans_after_first, "second call within TTL must not re-scan"


def test_cache_invalidated_on_backup_write(backup_root, monkeypatch):
    spy = _count_scandir(monkeypatch)
    media_dir = _dataset_dir(backup_root, "media", 1)
    _write_backup_file(media_dir, "old.db", mtime=1_000.0)

    _, total = svc.list_backup_items(dataset="media", user_id=1, limit=10, offset=0)
    assert total == 1
    scans_after_first = spy.calls

    def _fake_create(db_path, backup_dir, dataset):
        path = os.path.join(str(backup_dir), "created-backup.db")
        with open(path, "wb") as handle:
            handle.write(b"backup-bytes")
        return f"Backup created: {path}"

    monkeypatch.setattr(svc, "create_backup", _fake_create)

    created = svc.create_backup_snapshot(
        dataset="media", user_id=1, backup_type="full", max_backups=None
    )
    assert created.filename == "created-backup.db"

    items, total = svc.list_backup_items(dataset="media", user_id=1, limit=10, offset=0)
    assert spy.calls > scans_after_first, "write must invalidate the cache and force a re-scan"
    assert total == 2
    assert any(item.filename == "created-backup.db" for item in items)


def test_cache_invalidated_on_backup_restore(backup_root, monkeypatch):
    spy = _count_scandir(monkeypatch)
    media_dir = _dataset_dir(backup_root, "media", 1)
    _write_backup_file(media_dir, "backup.db", mtime=1_000.0)

    _, total = svc.list_backup_items(dataset="media", user_id=1, limit=10, offset=0)
    assert total == 1
    scans_after_first = spy.calls

    monkeypatch.setattr(
        svc, "restore_single_db_backup", lambda *args, **kwargs: "Database restored OK"
    )
    result = svc.restore_backup_snapshot(dataset="media", user_id=1, backup_id="backup.db")
    assert result.startswith("Database restored")

    svc.list_backup_items(dataset="media", user_id=1, limit=10, offset=0)
    assert spy.calls > scans_after_first, "restore must invalidate the cache and force a re-scan"


def test_pagination_returns_page_and_truthful_total(backup_root):
    media_dir = _dataset_dir(backup_root, "media", 1)
    for index in range(25):
        _write_backup_file(media_dir, f"backup_{index:02d}.db", mtime=1_000_000.0 + index)

    page, total = svc.list_backup_items(dataset="media", user_id=1, limit=10, offset=20)

    assert total == 25
    assert len(page) == 5
    # Newest first: offset 20 of 25 desc lands on the five oldest, newest-first.
    assert [item.filename for item in page] == [
        "backup_04.db",
        "backup_03.db",
        "backup_02.db",
        "backup_01.db",
        "backup_00.db",
    ]


def test_stats_only_page_files(backup_root, monkeypatch):
    """Only the requested page pays full BackupFile construction/stat-detail cost.

    Interpretation note: ``os.DirEntry.stat`` is a built-in method that cannot
    be monkeypatched, and its results are cached per entry, so an exact syscall
    count is not directly observable. The side channel is ``BackupFile``
    construction: constructing one requires the per-file stat detail (size +
    mtime), so a fresh scan of 25 files serving a 10-item page must construct
    exactly 10 objects - the page - not 25 - the whole tree.
    """
    media_dir = _dataset_dir(backup_root, "media", 1)
    for index in range(25):
        _write_backup_file(media_dir, f"backup_{index:02d}.db", mtime=1_000_000.0 + index)

    constructions: list[str] = []

    class _CountingBackupFile(svc.BackupFile):
        # Override __init__ (not __post_init__: BackupFile does not define
        # one, so the dataclass-generated __init__ would never call it).
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            constructions.append(self.filename)

    monkeypatch.setattr(svc, "BackupFile", _CountingBackupFile)
    spy = _count_scandir(monkeypatch)

    page, total = svc.list_backup_items(dataset="media", user_id=1, limit=10, offset=0)

    assert total == 25
    assert spy.calls >= 1, "scan must have run (cache was fresh)"
    assert constructions == [f"backup_{index:02d}.db" for index in range(24, 14, -1)]
    assert len(constructions) == 10
    assert [item.filename for item in page] == constructions
