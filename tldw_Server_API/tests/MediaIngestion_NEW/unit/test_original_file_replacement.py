"""Original replacement cleanup keeps the current binary and retries partial failures."""

from __future__ import annotations

import asyncio

import pytest

from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.DB_Management.media_db.api import create_media_database
from tldw_Server_API.app.core.Ingestion_Media_Processing.persistence import _cleanup_superseded_original_files
from tldw_Server_API.app.core.Storage.filesystem_storage import FileSystemStorage

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


@pytest.fixture
def replacement(tmp_path):
    db = create_media_database(db_path=str(tmp_path / "media.db"), client_id="replacement-test")
    media_id, _, _ = db.add_media_with_keywords(title="PDF", content="plaintext", media_type="pdf")
    storage = FileSystemStorage(base_path=tmp_path / "storage")
    yield db, storage, media_id
    db.close_connection()


async def _store_original(replacement, filename):
    db, storage, media_id = replacement
    path = await storage.store("1", media_id, filename, filename.encode())
    db.insert_media_file(media_id, "original", path)
    return path


@pytest.mark.parametrize("protected_by", [None, "original", "thumbnail"])
async def test_cleanup_groups_legacy_paths_and_protects_retained_references(replacement, monkeypatch, protected_by):
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

    async def record_delete(path):
        deletes.append(path)
        return await delete(path)

    monkeypatch.setattr(storage, "delete", record_delete)
    assert await _cleanup_superseded_original_files(db, storage, media_id) == []
    files = db.get_media_files(media_id, include_deleted=True)
    assert len([row for row in files if row["file_type"] == "original"]) == 1
    assert (storage.base_path / old_path).exists() is bool(protected_by)
    assert deletes.count(old_path) == (0 if protected_by else 1)
    expected_current = old_path if protected_by == "original" else new_path
    assert db.get_media_file(media_id)["storage_path"] == expected_current


@pytest.mark.parametrize("failure_stage", ["blob", "row"])
async def test_cleanup_failure_keeps_retry_record_and_current_original(replacement, monkeypatch, failure_stage):
    db, storage, media_id = replacement
    old_path = await _store_original(replacement, "old.pdf")
    new_path = await _store_original(replacement, "new.pdf")

    async def fail_blob(_path):
        raise OSError("storage unavailable")

    def fail_row(*_args, **_kwargs):
        raise RuntimeError("database unavailable")

    with monkeypatch.context() as patch:
        if failure_stage == "blob":
            patch.setattr(storage, "delete", fail_blob)
        else:
            patch.setattr(db, "soft_delete_media_file", fail_row)
        assert await _cleanup_superseded_original_files(db, storage, media_id)

    assert len(db.get_media_files(media_id, include_deleted=True)) == 2
    assert (storage.base_path / new_path).read_bytes() == b"new.pdf"
    assert await _cleanup_superseded_original_files(db, storage, media_id) == []
    assert [row["storage_path"] for row in db.get_media_files(media_id, include_deleted=True)] == [new_path]
    assert not (storage.base_path / old_path).exists()


async def test_overlapping_cleaners_are_idempotent(replacement, monkeypatch):
    db, storage, media_id = replacement
    old_path = await _store_original(replacement, "old.pdf")
    new_path = await _store_original(replacement, "new.pdf")
    delete = storage.delete
    both_started = asyncio.Event()
    calls = 0

    async def synchronized_delete(path):
        nonlocal calls
        calls += 1
        if calls == 2:
            both_started.set()
        await asyncio.wait_for(both_started.wait(), timeout=5)
        return await delete(path)

    monkeypatch.setattr(storage, "delete", synchronized_delete)
    warnings = await asyncio.gather(
        _cleanup_superseded_original_files(db, storage, media_id),
        _cleanup_superseded_original_files(db, storage, media_id),
    )
    assert warnings == [[], []]
    assert not (storage.base_path / old_path).exists()
    assert (storage.base_path / new_path).exists()
    assert len(db.get_media_files(media_id, include_deleted=True)) == 1


async def test_cleanup_snapshot_cannot_delete_a_later_registration(replacement, monkeypatch):
    db, storage, media_id = replacement
    await _store_original(replacement, "old.pdf")
    middle_path = await _store_original(replacement, "middle.pdf")
    delete = storage.delete
    entered = asyncio.Event()
    resume = asyncio.Event()

    async def paused_delete(path):
        entered.set()
        await asyncio.wait_for(resume.wait(), timeout=5)
        return await delete(path)

    with monkeypatch.context() as patch:
        patch.setattr(storage, "delete", paused_delete)
        cleanup = asyncio.create_task(_cleanup_superseded_original_files(db, storage, media_id))
        try:
            await asyncio.wait_for(entered.wait(), timeout=5)
            newest_path = await _store_original(replacement, "newest.pdf")
        finally:
            resume.set()
            warnings = await cleanup

    assert warnings == []
    assert (storage.base_path / middle_path).exists()
    assert (storage.base_path / newest_path).exists()
    assert await _cleanup_superseded_original_files(db, storage, media_id) == []
    assert not (storage.base_path / middle_path).exists()
    assert db.get_media_file(media_id)["storage_path"] == newest_path


async def test_cleanup_cancellation_preserves_current_and_retry_record(replacement, monkeypatch):
    db, storage, media_id = replacement
    await _store_original(replacement, "old.pdf")
    current_path = await _store_original(replacement, "new.pdf")

    async def cancel_delete(_path):
        raise asyncio.CancelledError

    with monkeypatch.context() as patch:
        patch.setattr(storage, "delete", cancel_delete)
        with pytest.raises(asyncio.CancelledError):
            await _cleanup_superseded_original_files(db, storage, media_id)

    assert (storage.base_path / current_path).exists()
    assert len(db.get_media_files(media_id, include_deleted=True)) == 2
    assert await _cleanup_superseded_original_files(db, storage, media_id) == []


async def test_cleanup_without_active_original_preserves_soft_deleted_file(replacement):
    db, storage, media_id = replacement
    path = await _store_original(replacement, "deleted.pdf")
    db.soft_delete_media_file(db.get_media_file(media_id)["id"])
    assert await _cleanup_superseded_original_files(db, storage, media_id) == []
    assert (storage.base_path / path).exists()


async def test_replacement_removes_chatbook_original_from_its_user_data_root(replacement, tmp_path, monkeypatch):
    db, storage, media_id = replacement
    user_root = tmp_path / "user-data"
    imported_path = f"imported_media/media_{media_id}/old.pdf"
    imported_file = user_root / imported_path
    imported_file.parent.mkdir(parents=True)
    imported_file.write_bytes(b"imported original")
    db.insert_media_file(media_id, "original", imported_path)
    current_path = await _store_original(replacement, "current.pdf")
    monkeypatch.setattr(DatabasePaths, "resolve_user_base_directory", lambda _user_id: user_root)

    assert await _cleanup_superseded_original_files(db, storage, media_id, user_id="1") == []
    assert not imported_file.exists()
    assert (storage.base_path / current_path).exists()
    assert len(db.get_media_files(media_id, include_deleted=True)) == 1


@pytest.mark.parametrize("path", ["imported_media/media_999/old.pdf", "imported_media/media_1/../../secret.pdf"])
async def test_cleanup_rejects_imported_paths_outside_the_media_directory(replacement, path):
    db, storage, media_id = replacement
    db.insert_media_file(media_id, "original", path)
    current_path = await _store_original(replacement, "current.pdf")
    assert await _cleanup_superseded_original_files(db, storage, media_id, user_id="1")
    assert len(db.get_media_files(media_id, include_deleted=True)) == 2
    assert (storage.base_path / current_path).exists()
