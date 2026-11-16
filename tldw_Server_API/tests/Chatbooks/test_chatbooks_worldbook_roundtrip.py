"""Real archive imports preserve world-book entries and honor conflict choices."""

import asyncio
from contextlib import ExitStack, contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

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
            "id_maps": {"notes": {"source-note": "restored-note"}},
            "warnings": ["Selected item failed"],
            "_account_restore_payload": {"private": "synthetic-private-value"},
            "metadata": {"also_private": "synthetic-private-value"},
        },
    )
    for key in ("status", "started_at", "completed_at", "error_message", "progress_percentage"):
        assert job.to_dict()[key] == before[key]  # nosec B101 -- Receipts cannot change the terminal lifecycle.
    assert (job.total_items, job.processed_items, job.successful_items, job.failed_items) == (2, 2, 1, 1)  # nosec B101 -- Already-restored items remain visible after cancellation.
    assert job.metadata == {"source_format": "chatbook", "imported_items": {"note": 1}, "id_maps": {"notes": {"source-note": "restored-note"}}}  # nosec B101 -- Only public result keys persist.
    assert job.warnings == ["Queued warning", "Selected item failed"]  # nosec B101 -- Retain both diagnostic sources.


@pytest.mark.parametrize("late_status", [ImportStatus.PENDING, ImportStatus.IN_PROGRESS, ImportStatus.FAILED, ImportStatus.COMPLETED])
async def test_cancelled_import_rejects_stale_worker_save(world_archive, late_status):
    """Cancellation committed after a worker read wins over that worker's late save."""
    _, service, path = world_archive
    job = ImportJob(
        job_id="cancel-before-late-save", user_id="2", status=ImportStatus.IN_PROGRESS,
        chatbook_path=str(path), warnings=["Cancellation retained"], metadata={"source_format": "chatbook"},
    )
    service._save_import_job(job)
    stale = service._get_import_job(job.job_id)
    cancelled = service._get_import_job(job.job_id)
    cancelled.status = ImportStatus.CANCELLED
    service._save_import_job(cancelled)
    before = service._get_import_job(job.job_id).to_dict()

    stale.status = late_status
    stale.total_items = stale.processed_items = stale.successful_items = 4
    stale.warnings = ["Late worker warning"]
    stale.metadata = {"unrelated_stale": "No public result receipt"}
    service._save_import_job(stale)

    assert service._get_import_job(job.job_id).to_dict() == before  # nosec B101 -- The database must reject a stale lifecycle overwrite atomically.


@pytest.mark.parametrize("late_status", [ImportStatus.FAILED, ImportStatus.COMPLETED])
async def test_cancelled_import_merges_late_terminal_public_receipt(world_archive, late_status):
    """A cancellation between the final worker read/save retains both public receipts."""
    from tldw_Server_API.app.core.Chatbooks.chatbook_service import record_failed_import_result

    _, service, path = world_archive
    job = ImportJob(
        job_id="cancel-before-result-save", user_id="2", status=ImportStatus.IN_PROGRESS,
        chatbook_path=str(path), metadata={"source_format": "chatbook"},
    )
    service._save_import_job(job)
    stale = service._get_import_job(job.job_id)
    cancelled = service._get_import_job(job.job_id)
    cancelled.status = ImportStatus.CANCELLED
    cancelled.warnings = ["Cancellation observed"]
    cancelled.successful_items = 3
    cancelled.metadata.update({
        "cancel_context": "Keep current metadata",
        "imported_items": {"note": 3},
        "id_maps": {"notes": {"earlier": "note-1", "ambiguous": "note-old"}},
    })
    service._save_import_job(cancelled)
    before = service._get_import_job(job.job_id).to_dict()
    record_failed_import_result(stale, {
        "total_items": 5, "processed_items": 5, "successful_items": 2, "failed_items": 1,
        "warnings": ["Late item warning"], "imported_items": {"media": 2},
        "id_maps": {"notes": {"later": "note-2", "ambiguous": "note-new"}},
        "inventory_summary": {"media": 2}, "skipped_non_restorable": {"external": 2},
        "_account_restore_payload": {"private": "synthetic-private-value"},
        "metadata": {"private": "synthetic-private-value"},
    })
    stale.status = late_status
    stale.metadata["unrelated_stale"] = "Do not copy stale metadata"
    stale.error_message = "Late worker outcome"
    stale.completed_at = datetime.now(timezone.utc)
    service._save_import_job(stale)
    saved = service._get_import_job(job.job_id)
    for key in ("status", "started_at", "completed_at", "error_message", "progress_percentage"):
        assert saved.to_dict()[key] == before[key]  # nosec B101 -- Cancellation owns the lifecycle even after late result persistence.
    assert saved.warnings == ["Cancellation observed", "Late item warning"]  # nosec B101 -- Neither current nor late warnings are lost.
    assert (saved.total_items, saved.processed_items, saved.successful_items, saved.failed_items) == (5, 5, 2, 1)  # nosec B101 -- Persist prepared result counters without inventing aggregation.
    assert saved.metadata == {  # nosec B101 -- Only public receipt fields enrich current metadata; ambiguous aliases stay unresolved.
        "source_format": "chatbook", "cancel_context": "Keep current metadata",
        "imported_items": {"note": 3, "media": 2},
        "id_maps": {"notes": {"earlier": "note-1", "later": "note-2", "ambiguous": None}},
        "inventory_summary": {"media": 2}, "skipped_non_restorable": {"external": 2},
    }


@pytest.mark.parametrize("backend_name", ["SQLITE", "POSTGRESQL"])
def test_late_terminal_receipt_uses_owned_backend_lock(backend_name):
    """The fallback joins a transaction and locks an owned cancelled PG row before merging."""
    import json

    from tldw_Server_API.app.core.DB_Management.backends.base import BackendType
    from tldw_Server_API.app.core.DB_Management.chacha.chatbook_job_store import save_import_job

    calls = []
    locked = False

    @contextmanager
    def transaction(*, preserve_existing):
        nonlocal locked
        assert preserve_existing  # nosec B101 -- Never settle a caller's pre-existing transaction.
        locked = True
        try:
            yield
        finally:
            locked = False

    parameter_logging = []

    def execute(query, params, *, commit=False, log_params=True):
        calls.append((query, params, commit, locked))
        parameter_logging.append(log_params)
        if query.startswith("SELECT"):
            return SimpleNamespace(fetchone=lambda: {"warnings": "[]", "metadata": '{"current":true}'})
        return SimpleNamespace(rowcount=0)

    values = ("job", "owner", "failed", "archive", "created", None, None, "late", 30, 2, 2, 1, 1, 0, "[]", "[]", '{"imported_items":{"note":1}}')
    save_import_job(SimpleNamespace(backend_type=BackendType[backend_name], transaction=transaction, execute_query=execute), values)
    assert len(calls) == 3  # nosec B101 -- Only a rejected terminal save enters the receipt fallback.
    assert calls[1][0].endswith(" FOR UPDATE") == (backend_name == "POSTGRESQL")  # nosec B101 -- PostgreSQL requires a row lock; SQLite's transaction owns its write lock.
    assert calls[1][1] == ("job", "owner", "cancelled") and calls[1][3]  # nosec B101 -- Select and merge are scoped and transactional.
    assert calls[2][1][-3:] == ("job", "owner", "cancelled") and calls[2][2:] == (False, True)  # nosec B101 -- Preserve owner/status and caller commit semantics.
    assert json.loads(calls[2][1][6]) == {"current": True, "imported_items": {"note": 1}}  # nosec B101 -- Lock-protected current metadata survives.
    assert parameter_logging == [False, True, False]  # nosec B101 -- Both payload-bearing writes redact parameters; the scoped identity read is unchanged.


@pytest.mark.parametrize("selected_status", [ImportStatus.PENDING, ImportStatus.VALIDATING])
@pytest.mark.parametrize("cancel_after_selection", [False, True])
async def test_stale_import_cleanup_respects_cancellation_after_selection(
    world_archive, monkeypatch, selected_status, cancel_after_selection,
):
    """Cleanup may fail a stale job only while its selected state still holds."""
    _, service, path = world_archive
    job = ImportJob(
        job_id="cancel-before-orphan-update", user_id="2", status=selected_status,
        chatbook_path=str(path), created_at=datetime.now(timezone.utc) - timedelta(days=2),
        successful_items=1, metadata={"id_maps": {"notes": {"source": "restored"}}},
    )
    service._save_import_job(job)
    before = None
    remove_file = service._try_delete_import_file

    def cancel_then_remove(file_path):
        nonlocal before
        current = service._get_import_job(job.job_id)
        current.status = ImportStatus.CANCELLED
        service._save_import_job(current)
        before = service._get_import_job(job.job_id).to_dict()
        return remove_file(file_path)

    if cancel_after_selection:
        monkeypatch.setattr(service, "_try_delete_import_file", cancel_then_remove)
    service.cleanup_import_orphans(age_threshold_hours=24)
    saved = service._get_import_job(job.job_id)
    if cancel_after_selection:
        assert before is not None and saved.to_dict() == before  # nosec B101 -- A selected stale row cannot overwrite a later cancellation.
    else:
        assert saved.status is ImportStatus.FAILED  # nosec B101 -- Ordinary stale cleanup still completes.
        assert saved.error_message == "Marked failed by orphan cleanup (stale job)"  # nosec B101 -- Preserve actionable cleanup outcome.
        assert saved.metadata == job.metadata and saved.successful_items == 1  # nosec B101 -- Cleanup preserves existing restore receipts.
