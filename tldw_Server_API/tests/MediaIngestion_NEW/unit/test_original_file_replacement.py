"""Original replacement cleanup keeps the current binary and retries partial failures."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from loguru import logger

from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.DB_Management.media_db.api import create_media_database
from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase
from tldw_Server_API.app.core.DB_Management.media_db.repositories.media_files_repository import MediaFilesRepository
from tldw_Server_API.app.core.Ingestion_Media_Processing.persistence import cleanup_superseded_original_files
from tldw_Server_API.app.core.Storage.filesystem_storage import FileSystemStorage

Replacement = tuple[MediaDatabase, FileSystemStorage, int]

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


@pytest.fixture
def replacement(tmp_path: Path) -> Iterator[Replacement]:
    """Provide a real per-test database and filesystem for replacement behavior."""
    db = create_media_database(db_path=str(tmp_path / "media.db"), client_id="replacement-test")
    media_id, _, _ = db.add_media_with_keywords(title="PDF", content="plaintext", media_type="pdf")
    storage = FileSystemStorage(base_path=tmp_path / "storage")
    yield db, storage, media_id
    db.close_connection()


async def _store_original(replacement: Replacement, filename: str) -> str:
    """Store filename bytes and register the original, returning its storage key."""
    db, storage, media_id = replacement
    path = await storage.store("1", media_id, filename, filename.encode())
    db.insert_media_file(media_id, "original", path)
    return path


@pytest.mark.parametrize("protected_by", [None, "original", "thumbnail"])
async def test_cleanup_groups_legacy_paths_and_protects_retained_references(
    replacement: Replacement, monkeypatch: pytest.MonkeyPatch, protected_by: str | None
) -> None:
    """Retire duplicate legacy rows once while preserving paths still needed by an artifact."""
    db, storage, media_id = replacement
    old_path = await _store_original(replacement, "old.pdf")
    old_id = db.get_media_file(media_id)["id"]
    db.insert_media_file(media_id, "original", old_path)
    db.soft_delete_media_file(old_id)
    new_path = await _store_original(replacement, "new.pdf")
    if protected_by:
        db.insert_media_file(media_id, protected_by, old_path)

    deletes = []
    delete = storage.delete

    async def record_delete(path: str) -> bool:
        """Record physical deletions while preserving the storage backend behavior."""
        deletes.append(path)
        return await delete(path)

    monkeypatch.setattr(storage, "delete", record_delete)
    assert await cleanup_superseded_original_files(db, storage, media_id) == []
    files = db.get_media_files(media_id, include_deleted=True)
    assert len([row for row in files if row["file_type"] == "original"]) == 1
    assert (storage.base_path / old_path).exists() is bool(protected_by)
    assert deletes.count(old_path) == (0 if protected_by else 1)
    expected_current = old_path if protected_by == "original" else new_path
    assert db.get_media_file(media_id)["storage_path"] == expected_current


@pytest.mark.parametrize("failure_stage", ["blob", "row"])
async def test_cleanup_failure_keeps_retry_record_and_current_original(
    replacement: Replacement, monkeypatch: pytest.MonkeyPatch, failure_stage: str | None
) -> None:
    """A blob or row failure leaves the latest file usable and permits a later retry."""
    db, storage, media_id = replacement
    old_path = await _store_original(replacement, "old.pdf")
    new_path = await _store_original(replacement, "new.pdf")

    async def fail_blob(_path: str) -> bool:
        """Simulate unavailable storage during retirement."""
        raise OSError("storage unavailable")

    def fail_row(*_args: Any, **_kwargs: Any) -> None:
        """Simulate a database failure after the blob has been removed."""
        raise RuntimeError("database unavailable")

    with monkeypatch.context() as patch:
        if failure_stage == "blob":
            patch.setattr(storage, "delete", fail_blob)
        else:
            patch.setattr(db, "soft_delete_media_file", fail_row)
        assert await cleanup_superseded_original_files(db, storage, media_id)

    assert len(db.get_media_files(media_id, include_deleted=True)) == 2
    assert (storage.base_path / new_path).read_bytes() == b"new.pdf"
    assert await cleanup_superseded_original_files(db, storage, media_id) == []
    assert [row["storage_path"] for row in db.get_media_files(media_id, include_deleted=True)] == [new_path]
    assert not (storage.base_path / old_path).exists()


async def test_overlapping_cleaners_are_idempotent(replacement: Replacement, monkeypatch: pytest.MonkeyPatch) -> None:
    """Two cleaners deleting the same old file leave one intact current original."""
    db, storage, media_id = replacement
    old_path = await _store_original(replacement, "old.pdf")
    new_path = await _store_original(replacement, "new.pdf")
    delete = storage.delete
    both_started = asyncio.Event()
    calls = 0

    async def synchronized_delete(path: str) -> bool:
        """Allow both cleaners to reach deletion before either removes the file."""
        nonlocal calls
        calls += 1
        if calls == 2:
            both_started.set()
        await asyncio.wait_for(both_started.wait(), timeout=5)
        return await delete(path)

    monkeypatch.setattr(storage, "delete", synchronized_delete)
    warnings = await asyncio.gather(
        cleanup_superseded_original_files(db, storage, media_id),
        cleanup_superseded_original_files(db, storage, media_id),
    )
    assert warnings == [[], []]
    assert not (storage.base_path / old_path).exists()
    assert (storage.base_path / new_path).exists()
    assert len(db.get_media_files(media_id, include_deleted=True)) == 1


async def test_cleanup_snapshot_cannot_delete_a_later_registration(
    replacement: Replacement, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An upload registered during cleanup survives the earlier cleanup attempt."""
    db, storage, media_id = replacement
    await _store_original(replacement, "old.pdf")
    middle_path = await _store_original(replacement, "middle.pdf")
    delete = storage.delete
    entered = asyncio.Event()
    resume = asyncio.Event()

    async def paused_delete(path: str) -> bool:
        """Hold storage deletion while another upload registers its original."""
        entered.set()
        await asyncio.wait_for(resume.wait(), timeout=5)
        return await delete(path)

    with monkeypatch.context() as patch:
        patch.setattr(storage, "delete", paused_delete)
        cleanup = asyncio.create_task(cleanup_superseded_original_files(db, storage, media_id))
        try:
            await asyncio.wait_for(entered.wait(), timeout=5)
            newest_path = await _store_original(replacement, "newest.pdf")
        finally:
            resume.set()
            warnings = await cleanup

    assert warnings == []
    assert (storage.base_path / middle_path).exists()
    assert (storage.base_path / newest_path).exists()
    assert await cleanup_superseded_original_files(db, storage, media_id) == []
    assert not (storage.base_path / middle_path).exists()
    assert db.get_media_file(media_id)["storage_path"] == newest_path


async def test_cleanup_cancellation_preserves_current_and_retry_record(
    replacement: Replacement, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cancellation propagates while preserving the current file and a retryable record."""
    db, storage, media_id = replacement
    await _store_original(replacement, "old.pdf")
    current_path = await _store_original(replacement, "new.pdf")

    async def cancel_delete(_path: str) -> bool:
        """Cancel retirement before the old file is removed."""
        raise asyncio.CancelledError

    with monkeypatch.context() as patch:
        patch.setattr(storage, "delete", cancel_delete)
        with pytest.raises(asyncio.CancelledError):
            await cleanup_superseded_original_files(db, storage, media_id)

    assert (storage.base_path / current_path).exists()
    assert len(db.get_media_files(media_id, include_deleted=True)) == 2
    assert await cleanup_superseded_original_files(db, storage, media_id) == []


async def test_cleanup_without_active_original_preserves_soft_deleted_file(replacement: Replacement) -> None:
    """Without a committed replacement, a recoverable original remains on disk."""
    db, storage, media_id = replacement
    path = await _store_original(replacement, "deleted.pdf")
    db.soft_delete_media_file(db.get_media_file(media_id)["id"])
    assert await cleanup_superseded_original_files(db, storage, media_id) == []
    assert (storage.base_path / path).exists()


async def test_replacement_removes_chatbook_original_from_its_user_data_root(
    replacement: Replacement, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Retirement removes imported originals from the authenticated user's data root."""
    db, storage, media_id = replacement
    user_root = tmp_path / "user-data"
    imported_path = f"imported_media/media_{media_id}/old.pdf"
    imported_file = user_root / imported_path
    imported_file.parent.mkdir(parents=True)
    imported_file.write_bytes(b"imported original")
    db.insert_media_file(media_id, "original", imported_path)
    current_path = await _store_original(replacement, "current.pdf")
    monkeypatch.setattr(DatabasePaths, "resolve_user_base_directory", lambda _user_id: user_root)

    assert await cleanup_superseded_original_files(db, storage, media_id, user_id="1") == []
    assert not imported_file.exists()
    assert (storage.base_path / current_path).exists()
    assert len(db.get_media_files(media_id, include_deleted=True)) == 1


@pytest.mark.parametrize("path", ["imported_media/media_999/old.pdf", "imported_media/media_1/../../secret.pdf"])
async def test_cleanup_rejects_imported_paths_outside_the_media_directory(replacement: Replacement, path: str) -> None:
    """Malformed imported paths warn and retain their records without touching the current file."""
    db, storage, media_id = replacement
    db.insert_media_file(media_id, "original", path)
    current_path = await _store_original(replacement, "current.pdf")
    assert await cleanup_superseded_original_files(db, storage, media_id, user_id="1")
    assert len(db.get_media_files(media_id, include_deleted=True)) == 2
    assert (storage.base_path / current_path).exists()


@pytest.mark.parametrize("deleted", [False, True])
async def test_replacement_preserves_a_path_referenced_by_another_media(
    replacement: Replacement, deleted: bool
) -> None:
    """Retiring one registration preserves another media item's shared or recoverable file."""
    db, storage, media_id = replacement
    shared_path = await _store_original(replacement, "shared.pdf")
    other_id, _, _ = db.add_media_with_keywords(title="Other", content="other", media_type="pdf")
    db.insert_media_file(other_id, "original", shared_path)
    if deleted:
        db.soft_delete_media_file(db.get_media_file(other_id)["id"])
    current_path = await _store_original(replacement, "current.pdf")

    assert await cleanup_superseded_original_files(db, storage, media_id) == []
    assert (storage.base_path / shared_path).read_bytes() == b"shared.pdf"
    assert [row["storage_path"] for row in db.get_media_files(media_id)] == [current_path]
    assert db.get_media_file(other_id, include_deleted=True)["storage_path"] == shared_path


async def test_concurrent_retirement_of_shared_original_removes_the_blob(
    replacement: Replacement, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two media items retiring one shared original leave neither an orphan nor stale rows."""
    db, storage, media_id = replacement
    shared_path = await _store_original(replacement, "shared.pdf")
    other_id, _, _ = db.add_media_with_keywords(title="Other", content="other", media_type="pdf")
    db.insert_media_file(other_id, "original", shared_path)
    current_path = await _store_original(replacement, "current.pdf")
    other_path = await _store_original((db, storage, other_id), "other.pdf")
    original = MediaFilesRepository.has_retained_references
    both_checked = threading.Barrier(2)

    def synchronized_check(self: MediaFilesRepository, path: str, excluded_ids: set[int]) -> bool:
        """Let both cleaners inspect references before either can remove its registrations."""
        result = original(self, path, excluded_ids)
        both_checked.wait(timeout=5)
        return result

    monkeypatch.setattr(MediaFilesRepository, "has_retained_references", synchronized_check)
    warnings = await asyncio.gather(
        cleanup_superseded_original_files(db, storage, media_id),
        cleanup_superseded_original_files(db, storage, other_id),
    )
    assert warnings == [[], []]
    assert [row["storage_path"] for row in db.get_media_files(media_id)] == [current_path]
    assert [row["storage_path"] for row in db.get_media_files(other_id)] == [other_path]
    assert (storage.base_path / current_path).exists()
    assert (storage.base_path / other_path).exists()
    assert not (storage.base_path / shared_path).exists()


async def test_older_cleaner_resumes_after_newer_cleaner_retires_its_current_row(
    replacement: Replacement, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A paused cleaner can retire only its original candidate IDs, never the newest upload."""
    db, storage, media_id = replacement
    old_path = await _store_original(replacement, "old.pdf")
    middle_path = await _store_original(replacement, "middle.pdf")
    delete = storage.delete
    entered = asyncio.Event()
    resume = asyncio.Event()
    first_call = True

    async def pause_first_delete(path: str) -> bool:
        """Pause only the older cleaner while allowing the newer cleaner to finish."""
        nonlocal first_call
        if first_call:
            first_call = False
            entered.set()
            await asyncio.wait_for(resume.wait(), timeout=5)
        return await delete(path)

    monkeypatch.setattr(storage, "delete", pause_first_delete)
    older_cleanup = asyncio.create_task(cleanup_superseded_original_files(db, storage, media_id))
    try:
        await asyncio.wait_for(entered.wait(), timeout=5)
        latest_path = await _store_original(replacement, "latest.pdf")
        assert await cleanup_superseded_original_files(db, storage, media_id) == []
        assert not (storage.base_path / middle_path).exists()
    finally:
        resume.set()
        assert await older_cleanup == []
    assert not (storage.base_path / old_path).exists()
    assert (storage.base_path / latest_path).read_bytes() == b"latest.pdf"
    assert [row["storage_path"] for row in db.get_media_files(media_id)] == [latest_path]


@pytest.mark.parametrize("operation", ["get_media_files", "soft_delete_media_file"])
async def test_slow_cleanup_database_work_does_not_block_the_event_loop(
    replacement: Replacement, monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    """The event loop can unblock slow database work before its worker timeout expires."""
    db, storage, media_id = replacement
    await _store_original(replacement, "old.pdf")
    current_path = await _store_original(replacement, "current.pdf")
    original = getattr(db, operation)
    loop = asyncio.get_running_loop()
    entered = asyncio.Event()
    release = threading.Event()

    def blocked_operation(*args: Any, **kwargs: Any) -> Any:
        """Model a blocking driver until the event loop gives it permission to finish."""
        loop.call_soon_threadsafe(entered.set)
        if not release.wait(timeout=2):
            raise TimeoutError("The event loop could not unblock database cleanup")
        return original(*args, **kwargs)

    monkeypatch.setattr(db, operation, blocked_operation)
    cleanup = asyncio.create_task(cleanup_superseded_original_files(db, storage, media_id))
    try:
        await asyncio.wait_for(entered.wait(), timeout=5)
    finally:
        release.set()
    assert await cleanup == []
    assert (storage.base_path / current_path).exists()


async def test_cleanup_warning_preserves_the_failure_traceback(
    replacement: Replacement,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A retirement warning retains the exception and stack needed to diagnose the failure."""
    db, storage, media_id = replacement
    await _store_original(replacement, "old.pdf")
    await _store_original(replacement, "current.pdf")
    failure = OSError("storage unavailable")
    records: list[dict[str, Any]] = []

    async def fail_delete(path: str) -> bool:
        """Raise the storage failure whose traceback the cleanup warning must retain."""
        raise failure

    monkeypatch.setattr(storage, "delete", fail_delete)
    sink = logger.add(lambda message: records.append(message.record), level="WARNING")
    try:
        assert await cleanup_superseded_original_files(db, storage, media_id)
    finally:
        logger.remove(sink)
    assert any(
        record["exception"] and record["exception"].value is failure and record["exception"].traceback is not None
        for record in records
    )
