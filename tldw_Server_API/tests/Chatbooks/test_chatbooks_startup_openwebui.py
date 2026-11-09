"""The normal API-startup worker must honor advertised non-archive Jobs."""

import asyncio
import base64
import json
from uuid import uuid4

import pytest

from tldw_Server_API.app.core.Chatbooks.chatbook_models import ImportJob, ImportStatus
from tldw_Server_API.app.core.Chatbooks.chatbook_service import ChatbookService
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.services import core_jobs_worker
from tldw_Server_API.tests.Chatbooks.test_openwebui_db_import_adapter import _write_openwebui_db
from tldw_Server_API.tests.Chatbooks.test_openwebui_hydration_service import _write_openwebui_hydration_db

pytestmark = [pytest.mark.asyncio, pytest.mark.integration]
LEASE_RECEIPT = "source-lease"
QUESTION = "What is Rowan Observatory's telescope called?"
ANSWER = "The telescope is Selene."
IMAGE = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+ip1sAAAAASUVORK5CYII=")


@pytest.fixture()
def service(tmp_path, monkeypatch):
    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path / "users"))
    monkeypatch.setenv("OPENWEBUI_HYDRATION_MEDIA_STORAGE_PATH", str(tmp_path / "hydration-files"))
    db = CharactersRAGDB(db_path=str(tmp_path / "chacha.db"), client_id="101")
    db.add_character_card({"name": "Rowan import assistant"})
    result = ChatbookService(user_id="101", user_id_int=101, db=db)
    result.temp_dir = tmp_path / "uploads"
    result.import_dir = tmp_path / "imports"
    result.temp_dir.mkdir()
    result.import_dir.mkdir()
    try:
        yield result
    finally:
        if result._media_db is not None:
            result._media_db.close_connection()
        db.close_all_connections()


async def run_startup(service, payload, monkeypatch, *, job_type="import", cancel_after=None, core_status="processing"):
    stop = asyncio.Event()
    outcomes = []
    reads = 0
    job = {
        "id": 42,
        "uuid": "startup-source-job",
        "owner_user_id": "101",
        "job_type": job_type,
        "payload": payload,
        "lease_id": LEASE_RECEIPT,
    }

    class Jobs:
        def acquire_next_job(self, **_kwargs):
            return job

        def get_job(self, _job_id):
            nonlocal reads
            reads += 1
            if core_status != "processing":
                stop.set()
            return {
                "status": core_status,
                "cancel_requested_at": "now" if cancel_after is not None and reads >= cancel_after else None,
            }

        def complete_job(self, job_id, **kwargs):
            outcomes.append(("completed", job_id, kwargs))
            stop.set()

        def fail_job(self, job_id, **kwargs):
            outcomes.append(("failed", job_id, kwargs))
            stop.set()

        def finalize_cancelled(self, job_id, **kwargs):
            outcomes.append(("cancelled", job_id, kwargs))
            stop.set()

        def renew_job_lease(self, *_args, **_kwargs):
            return None

    monkeypatch.setattr(core_jobs_worker, "JobManager", Jobs)
    monkeypatch.setattr(core_jobs_worker, "_build_chacha_db_for_user", lambda _owner: service.db)
    monkeypatch.setattr(core_jobs_worker, "ChatbookService", lambda *_args, **_kwargs: service)
    monkeypatch.setenv("JOBS_POLL_INTERVAL_SECONDS", "0.01")
    await asyncio.wait_for(core_jobs_worker.run_chatbooks_core_jobs_worker(stop), timeout=5)
    return outcomes


def source_chat():
    return {
        "title": "S02 startup Rowan",
        "history": {
            "currentId": "a",
            "messages": {
                "q": {"id": "q", "role": "user", "content": QUESTION, "timestamp": 1700000000, "childrenIds": ["a"]},
                "a": {"id": "a", "role": "assistant", "content": ANSWER, "timestamp": 1700000001, "parentId": "q"},
            },
        },
    }


@pytest.mark.parametrize("source_format", ["openwebui_json", "openwebui_db"])
async def test_startup_import_routes_actual_source_format(service, monkeypatch, source_format):
    source = service.temp_dir / ("rowan.json" if source_format == "openwebui_json" else "webui.db")
    if source_format == "openwebui_json":
        source.write_text(json.dumps([{"id": "source-rowan", "chat": source_chat()}]))
    else:
        _write_openwebui_db(
            source,
            users=[{"id": "ow-user", "name": "Alice", "email": "alice@example.test"}],
            chats=[{"id": "source-rowan", "user_id": "ow-user", "title": "S02 startup Rowan", "chat": source_chat()}],
        )
    imported = ImportJob(
        job_id=str(uuid4()),
        user_id="101",
        status=ImportStatus.PENDING,
        chatbook_path=str(source),
        metadata={"source_format": source_format},
    )
    service._save_import_job(imported)
    outcomes = await run_startup(
        service,
        {
            "action": "import",
            "chatbooks_job_id": imported.job_id,
            "file_token": str(source),
            "source_format": source_format,
            "selected_openwebui_user_id": "ow-user",
            "conflict_resolution": "skip",
            "prefix_imported": False,
            "import_media": False,
            "import_embeddings": False,
        },
        monkeypatch,
    )
    assert [row[0] for row in outcomes] == ["completed"]  # nosec B101 -- Actual source must reach its parser.
    assert service._get_import_job(imported.job_id).status is ImportStatus.COMPLETED  # nosec B101 -- Public child outcome.
    final = service._get_import_job(imported.job_id)
    assert (
        final.total_items,
        final.processed_items,
        final.successful_items,
        final.failed_items,
        final.skipped_items,
    ) == (1, 1, 1, 0, 0)  # nosec B101 -- Completed progress uses chats, not zero or message counts.
    conversations = service.db.search_conversations_by_title("S02 startup Rowan")
    assert len(conversations) == 1  # nosec B101 -- One source, one destination.
    messages = service.db.get_messages_for_conversation(conversations[0]["id"])
    assert [row["content"] for row in messages] == [QUESTION, ANSWER]  # nosec B101 -- Exact source data survives.
    assert not source.exists()  # nosec B101 -- Original worker cleans the admitted temporary upload.
    assert {key: outcomes[0][2][key] for key in ("worker_id", "lease_id", "completion_token")} == {  # nosec B101 -- Verify exact lease authority.
        "worker_id": "cb-core-worker",
        "lease_id": LEASE_RECEIPT,
        "completion_token": LEASE_RECEIPT,
    }  # nosec B101 -- Completion retains lease authority.


async def test_startup_runs_real_openwebui_attachment_job(service, tmp_path, monkeypatch):
    root = tmp_path / "openwebui"
    (root / "uploads").mkdir(parents=True)
    _write_openwebui_hydration_db(root)
    (root / "uploads/file-image_image.png").write_bytes(IMAGE)
    monkeypatch.setenv("INGESTION_SOURCE_ALLOWED_ROOTS", str(root))
    character = service.db.list_character_cards()[0]
    conversation = service.db.add_conversation({"title": "S02 hydration", "character_id": character["id"]})
    message = service.db.add_message({"conversation_id": conversation, "sender": "user", "content": QUESTION})
    service.db.add_message_metadata(message, extra={"openwebui_import": {"attachment_refs": [{"id": "file-image"}]}})
    outcomes = await run_startup(
        service,
        {
            "openwebui_data_root": str(root),
            "scope": {"conversation_ids": [conversation], "source_user_id": "ow-user"},
            "process_supported_files": False,
        },
        monkeypatch,
        job_type="openwebui_attachment_hydration",
    )
    assert [row[0] for row in outcomes] == ["completed"], outcomes  # nosec B101 -- Normal startup recognizes the real job type.
    images = service.db.get_message_images(message)
    assert len(images) == 1 and images[0]["image_data"] == IMAGE  # nosec B101 -- Actual storage, no handler stub.
    assert outcomes[0][2]["result"]["hydrated_images"] == 1  # nosec B101 -- Truthful completion data.


@pytest.mark.parametrize("retryable,backoff", [(True, 7), (False, 1)])
async def test_startup_delegated_failures_keep_typed_retry_and_authority(service, monkeypatch, retryable, backoff):
    from tldw_Server_API.app.core.Chatbooks.services import jobs_worker

    async def reject(_service, _payload, _job_id):
        raise jobs_worker.ChatbooksJobError("controlled claim refusal", retryable=retryable, backoff_seconds=backoff)

    monkeypatch.setattr(jobs_worker, "_handle_import", reject)
    outcomes = await run_startup(
        service, {"action": "import", "source_format": "openwebui_json", "chatbooks_job_id": "controlled"}, monkeypatch
    )
    assert [row[0] for row in outcomes] == ["failed"]  # nosec B101 -- No false completion after refusal.
    actual = outcomes[0][2]
    assert actual["retryable"] is retryable and actual["backoff_seconds"] == backoff  # nosec B101 -- Preserve granted retry policy.
    assert {key: actual[key] for key in ("worker_id", "lease_id", "completion_token")} == {  # nosec B101 -- Verify exact lease authority.
        "worker_id": "cb-core-worker",
        "lease_id": LEASE_RECEIPT,
        "completion_token": LEASE_RECEIPT,
    }  # nosec B101 -- Failure remains lease fenced.


@pytest.mark.parametrize("cancel_after", [1, 2])
async def test_startup_delegated_cancellation_never_completes(service, monkeypatch, cancel_after):
    from tldw_Server_API.app.core.Chatbooks.services import jobs_worker

    imported = ImportJob(job_id=str(uuid4()), user_id="101", status=ImportStatus.PENDING, chatbook_path="controlled")
    service._save_import_job(imported)
    calls = []

    async def finish(_service, _payload, _job_id):
        calls.append(True)
        return {"openwebui_result": {"imported_chats": 1}}

    monkeypatch.setattr(jobs_worker, "_handle_import", finish)
    outcomes = await run_startup(
        service,
        {"action": "import", "source_format": "openwebui_json", "chatbooks_job_id": imported.job_id},
        monkeypatch,
        cancel_after=cancel_after,
    )
    assert calls == ([] if cancel_after == 1 else [True])  # nosec B101 -- Pre-cancel performs no import; mid-cancel does not rerun it.
    assert [row[0] for row in outcomes] == ["cancelled"]  # nosec B101 -- No false completion after cancellation.
    assert service._get_import_job(imported.job_id).status is ImportStatus.CANCELLED  # nosec B101 -- Public child agrees.
    assert {key: outcomes[0][2][key] for key in ("worker_id", "lease_id", "expected_uuid")} == {
        "worker_id": "cb-core-worker",
        "lease_id": LEASE_RECEIPT,
        "expected_uuid": "startup-source-job",
    }  # nosec B101 -- Cancellation is exact-job fenced.


@pytest.mark.parametrize("source_format", ["openwebui_json", "openwebui_db"])
async def test_startup_partial_openwebui_import_is_truthfully_failed(service, monkeypatch, source_format):
    invalid = {
        "title": "S02 unsupported chat",
        "history": {
            "currentId": "tool",
            "messages": {"tool": {"id": "tool", "role": "tool", "content": "not an importable user or assistant turn"}},
        },
    }
    source = service.temp_dir / ("partial.json" if source_format == "openwebui_json" else "partial.db")
    if source_format == "openwebui_json":
        source.write_text(json.dumps([{"id": "valid", "chat": source_chat()}, {"id": "invalid", "chat": invalid}]))
    else:
        _write_openwebui_db(
            source,
            users=[{"id": "ow-user", "name": "Alice", "email": "alice@example.test"}],
            chats=[
                {"id": "valid", "user_id": "ow-user", "title": "S02 startup Rowan", "chat": source_chat()},
                {"id": "invalid", "user_id": "ow-user", "title": "S02 unsupported chat", "chat": invalid},
            ],
        )
    imported = ImportJob(
        job_id=str(uuid4()),
        user_id="101",
        status=ImportStatus.PENDING,
        chatbook_path=str(source),
        metadata={"source_format": source_format},
    )
    service._save_import_job(imported)
    outcomes = await run_startup(
        service,
        {
            "action": "import",
            "chatbooks_job_id": imported.job_id,
            "file_token": str(source),
            "source_format": source_format,
            "selected_openwebui_user_id": "ow-user",
            "conflict_resolution": "skip",
        },
        monkeypatch,
    )
    assert [row[0] for row in outcomes] == ["failed"]  # nosec B101 -- Partial failure is never a false completed job.
    final = service._get_import_job(imported.job_id)
    assert final.status is ImportStatus.FAILED  # nosec B101 -- Public child agrees with Jobs failure.
    assert (
        final.total_items,
        final.processed_items,
        final.successful_items,
        final.failed_items,
        final.skipped_items,
    ) == (2, 2, 1, 1, 0)  # nosec B101 -- Preserve honest partial counts.
    assert final.metadata["imported_items"] == {"conversation": 1}  # nosec B101 -- Disclose retained successful content.
    assert final.warnings and final.error_message  # nosec B101 -- Explain recoverable failure.
    assert len(service.db.search_conversations_by_title("S02 startup Rowan")) == 1  # nosec B101 -- Successful source survives.
    assert service.db.search_conversations_by_title("S02 unsupported chat") == []  # nosec B101 -- Failed chat leaves no shell.
    corrected = source_chat()
    corrected["title"] = "S02 recovered chat"
    if source_format == "openwebui_json":
        source.write_text(json.dumps([{"id": "valid", "chat": source_chat()}, {"id": "invalid", "chat": corrected}]))
    else:
        _write_openwebui_db(
            source,
            users=[{"id": "ow-user", "name": "Alice", "email": "alice@example.test"}],
            chats=[
                {"id": "valid", "user_id": "ow-user", "title": "S02 startup Rowan", "chat": source_chat()},
                {"id": "invalid", "user_id": "ow-user", "title": "S02 recovered chat", "chat": corrected},
            ],
        )
    retry = ImportJob(
        job_id=str(uuid4()),
        user_id="101",
        status=ImportStatus.PENDING,
        chatbook_path=str(source),
        metadata={"source_format": source_format},
    )
    service._save_import_job(retry)
    repaired = await run_startup(
        service,
        {
            "action": "import",
            "chatbooks_job_id": retry.job_id,
            "file_token": str(source),
            "source_format": source_format,
            "selected_openwebui_user_id": "ow-user",
            "conflict_resolution": "skip",
        },
        monkeypatch,
    )
    assert [row[0] for row in repaired] == ["completed"]  # nosec B101 -- A new, legitimate Retry can finish.
    completed = service._get_import_job(retry.job_id)
    assert (
        completed.total_items,
        completed.processed_items,
        completed.successful_items,
        completed.failed_items,
        completed.skipped_items,
    ) == (2, 2, 1, 0, 1)  # nosec B101 -- Existing success is skipped, not duplicated.
    assert service._get_import_job(imported.job_id).status is ImportStatus.FAILED  # nosec B101 -- Original failure remains in history.
    assert len(service.db.search_conversations_by_title("S02 startup Rowan")) == 1  # nosec B101 -- Original chat remains singular.
    assert len(service.db.search_conversations_by_title("S02 recovered chat")) == 1  # nosec B101 -- Recovery adds exactly the missing chat.


async def test_startup_child_cancellation_precedes_core_cancel_flag(service, monkeypatch):
    imported = ImportJob(job_id=str(uuid4()), user_id="101", status=ImportStatus.CANCELLED, chatbook_path="unused")
    service._save_import_job(imported)
    outcomes = await run_startup(
        service,
        {"action": "import", "source_format": "openwebui_json", "chatbooks_job_id": imported.job_id},
        monkeypatch,
    )
    assert [row[0] for row in outcomes] == ["cancelled"]  # nosec B101 -- Child cancellation cannot become core completion.
    assert service._get_import_job(imported.job_id).status is ImportStatus.CANCELLED  # nosec B101 -- Preserve real stored terminal state.


async def test_startup_requeued_job_is_not_dispatched_or_finalized(service, monkeypatch):
    from tldw_Server_API.app.core.Chatbooks.services import jobs_worker

    calls = []

    async def forbidden(*_args):
        calls.append(True)
        return {}

    monkeypatch.setattr(jobs_worker, "_handle_import", forbidden)
    outcomes = await run_startup(
        service,
        {"action": "import", "source_format": "openwebui_json", "chatbooks_job_id": "unused"},
        monkeypatch,
        core_status="queued",
    )
    assert calls == [] and outcomes == []  # nosec B101 -- Requeued lease belongs to the next delivery, not this worker.
