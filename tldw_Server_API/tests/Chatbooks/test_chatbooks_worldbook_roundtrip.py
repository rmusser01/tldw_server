"""Real archive imports preserve world-book entries and honor conflict choices."""

import asyncio
from contextlib import ExitStack
from pathlib import Path

import pytest

from tldw_Server_API.app.core.Character_Chat.world_book_manager import WorldBookService
from tldw_Server_API.app.core.Chatbooks.chatbook_models import (
    ChatbookVersion,
    ConflictResolution,
    ContentType,
    ImportJob,
    ImportStatus,
)
from tldw_Server_API.app.core.Chatbooks.chatbook_service import ChatbookService
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.Jobs.manager import JobManager

pytestmark = pytest.mark.integration


@pytest.fixture
async def world_archive(tmp_path, monkeypatch):
    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path / "users"))
    monkeypatch.delenv("JOBS_DB_URL", raising=False)
    monkeypatch.setenv("JOBS_DB_PATH", str(tmp_path / "jobs.db"))
    with ExitStack() as stack:
        source = CharactersRAGDB(tmp_path / "source.db", client_id="1")
        destination = CharactersRAGDB(DatabasePaths.get_chacha_db_path(2), client_id="2")
        stack.callback(source.close_all_connections)
        stack.callback(destination.close_all_connections)
        world = WorldBookService(source)
        book = world.create_world_book("Rowan world", "Exact portable facts")
        world.add_entry(book, ["Rowan"], "The telescope is Selene.")
        note = source.add_note("Rowan Note", "Keep this successful item.")
        service = ChatbookService(user_id="1", user_id_int=1, db=source)
        okay, message, path = await service.create_chatbook(
            name="Rowan world backup",
            description="Real world-book and Note archive",
            content_selections={ContentType.WORLD_BOOK: [str(book)], ContentType.NOTE: [note]},
            format_version=ChatbookVersion.V1_1,
            async_mode=False,
        )
        assert okay, message  # nosec B101 -- Pytest fixture archive guard.
        restored = ChatbookService(user_id="2", user_id_int=2, db=destination)
        target = restored.import_dir / "world.chatbook"
        target.write_bytes(Path(path).read_bytes())
        yield destination, restored, target


@pytest.mark.parametrize("mode", ["fresh", "skip", "rename", "prefix"])
async def test_public_worldbook_import_honors_entries_and_conflict_name(world_archive, mode):
    db, service, path = world_archive
    world = WorldBookService(db)
    existing = None
    if mode in {"skip", "rename"}:
        existing = world.create_world_book("Rowan world", "Destination original")
        world.add_entry(existing, ["Keep"], "Keep destination facts.")
    okay, message, result = await service.import_chatbook(
        str(path),
        conflict_resolution=ConflictResolution.RENAME if mode == "rename" else ConflictResolution.SKIP,
        prefix_imported=mode == "prefix",
        async_mode=False,
    )
    assert okay, message  # nosec B101 -- Pytest behavior assertion.
    assert not any("Error importing world book" in row for row in result["warnings"])  # nosec B101 -- No hidden selected-item failure.
    books = world.list_world_books()
    assert len(books) == (2 if mode == "rename" else 1)  # nosec B101 -- Pytest behavior assertion.
    if existing is not None:
        assert world.get_entries(existing)[0]["content"] == "Keep destination facts."  # nosec B101 -- Retained destination content.
    if mode == "skip":
        return
    imported = next(row for row in books if row["id"] != existing)
    if mode == "prefix":
        assert imported["name"] == "[Imported] Rowan world"  # nosec B101 -- Pytest behavior assertion.
    elif mode == "rename":
        assert imported["name"] != "Rowan world"  # nosec B101 -- Pytest behavior assertion.
    entries = world.get_entries(imported["id"])
    assert len(entries) == 1 and entries[0]["content"] == "The telescope is Selene."  # nosec B101 -- Exact restored entry.


async def test_partial_archive_failure_is_not_reported_as_complete(world_archive, monkeypatch):
    db, service, path = world_archive

    def refuse_world_write(*_args, **_kwargs):
        raise RuntimeError("Synthetic world-book storage failure")

    monkeypatch.setattr(WorldBookService, "import_world_book", refuse_world_write)
    okay, _message, result = await service.import_chatbook(
        str(path), conflict_resolution=ConflictResolution.SKIP, prefix_imported=False, async_mode=False
    )
    assert not okay  # nosec B101 -- A failed selected item must not report complete.
    assert db.list_notes()[0]["content"] == "Keep this successful item."  # nosec B101 -- Partial writes remain explicit.
    assert result["failed_items"] == 1 and result["imported_items"]["note"] == 1  # nosec B101 -- Retain failure and partial success.


@pytest.mark.parametrize("lane", ["legacy", "sdk", "core", "core-refusal"])
async def test_failed_import_jobs_retain_partial_counts_and_diagnostics(world_archive, monkeypatch, lane):
    db, service, path = world_archive
    from tldw_Server_API.app.core.Chatbooks.services import jobs_worker
    from tldw_Server_API.app.services import core_jobs_worker

    def refuse_world_write(*_args, **_kwargs):
        raise RuntimeError("Synthetic world-book storage failure")

    monkeypatch.setattr(WorldBookService, "import_world_book", refuse_world_write)
    job = ImportJob(
        job_id="partial-world-import",
        user_id="2",
        status=ImportStatus.PENDING,
        chatbook_path=str(path),
        warnings=["Queued archive warning"],
        metadata={"source_format": "chatbook"},
    )
    service._save_import_job(job)
    if lane == "legacy":
        await service._import_chatbook_async(job.job_id, str(path), None, ConflictResolution.SKIP, False, True, True)
    elif lane == "sdk":
        with pytest.raises(jobs_worker.ChatbooksJobError) as error:
            await jobs_worker._handle_import(service, {"file_token": str(path)}, job.job_id)
        assert error.value.retryable is False  # nosec B101 -- Storage failure stays terminal until legitimate user retry.
    else:
        stop = asyncio.Event()
        manager = JobManager()
        queued = manager.create_job(
            domain="chatbooks",
            queue="default",
            job_type="chatbook_import",
            owner_user_id="2",
            payload={
                "action": "import",
                "chatbooks_job_id": job.job_id,
                "file_token": str(path),
                "conflict_resolution": "merge" if lane == "core-refusal" else "skip",
            },
        )

        async def stop_when_terminal():
            try:
                while manager.get_job(queued["id"])["status"] not in {"failed", "completed", "cancelled"}:
                    await asyncio.sleep(0.01)
            finally:
                stop.set()

        watcher = asyncio.create_task(stop_when_terminal())
        try:
            await asyncio.wait_for(core_jobs_worker.run_chatbooks_core_jobs_worker(stop), timeout=5)
        finally:
            watcher.cancel()
            await asyncio.gather(watcher, return_exceptions=True)
        assert manager.get_job(queued["id"])["status"] == "failed"  # nosec B101 -- Real durable dispatcher outcome.
    saved = service._get_import_job(job.job_id)
    assert saved.status is ImportStatus.FAILED  # nosec B101 -- No false completed status.
    if lane == "core-refusal":
        assert "not supported yet" in saved.error_message  # nosec B101 -- Refusal must not read an unbound prior result.
        assert saved.metadata == {"source_format": "chatbook"}  # nosec B101 -- No unrelated partial metadata.
        assert saved.processed_items == saved.successful_items == saved.failed_items == 0  # nosec B101 -- No import ran.
        return
    assert (saved.total_items, saved.processed_items, saved.successful_items, saved.failed_items) == (2, 2, 1, 1)  # nosec B101 -- Exact partial counters.
    assert saved.metadata["imported_items"] == {"note": 1}  # nosec B101 -- Preserve successful partial write.
    assert saved.metadata["source_format"] == "chatbook"  # nosec B101 -- Preserve queued metadata.
    assert "Queued archive warning" in saved.warnings  # nosec B101 -- Preserve queued diagnostics.
    assert any("Error importing world book" in row for row in saved.warnings)  # nosec B101 -- Retain the selected failure.
    assert "_account_restore_payload" not in saved.metadata  # nosec B101 -- Private account state is never job metadata.


@pytest.mark.parametrize("cancelled", [False, True])
def test_failed_result_recorder_preserves_cancellation_and_excludes_private_payload(cancelled):
    from tldw_Server_API.app.core.Chatbooks.chatbook_service import record_failed_import_result

    job = ImportJob(
        job_id="safe-result",
        user_id="2",
        status=ImportStatus.CANCELLED if cancelled else ImportStatus.IN_PROGRESS,
        chatbook_path="local",
        warnings=["Queued warning"],
        metadata={"source_format": "chatbook"},
    )
    before = job.to_dict()
    record_failed_import_result(
        job,
        {
            "total_items": 2,
            "processed_items": 2,
            "successful_items": 1,
            "failed_items": 1,
            "imported_items": {"note": 1},
            "warnings": ["Selected item failed"],
            "_account_restore_payload": {"private": "synthetic-private-value"},
            "metadata": {"also_private": "synthetic-private-value"},
        },
    )
    if cancelled:
        assert job.to_dict() == before  # nosec B101 -- Late failure cannot overwrite cancellation.
    else:
        assert job.metadata == {"source_format": "chatbook", "imported_items": {"note": 1}}  # nosec B101 -- Only public result keys persist.
        assert job.warnings == ["Queued warning", "Selected item failed"]  # nosec B101 -- Retain both diagnostic sources.
