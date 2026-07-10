import asyncio
import json
import zipfile
from pathlib import Path

import pytest

from tldw_Server_API.app.core.Chatbooks.chatbook_models import (
    ExportJob,
    ExportStatus,
    ImportJob,
    ImportStatus,
)
from tldw_Server_API.app.core.Chatbooks.chatbook_service import ChatbookService
from tldw_Server_API.app.services import core_jobs_worker

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_completed_import_restores_chatbook_media_and_embeddings(tmp_path, monkeypatch):
    archive_path = tmp_path / "full-account.chatbook"
    archive_path.write_bytes(b"temporary archive")
    stop_event = asyncio.Event()
    import_job = ImportJob(
        job_id="import-job-1",
        user_id="1",
        status=ImportStatus.PENDING,
        chatbook_path=str(archive_path),
        warnings=["queued with archive warning"],
        metadata={"source_format": "chatbook", "audit_id": "audit-1"},
    )
    import_call = None
    public_job_status = "processing"
    failures = []

    class FakeJobManager:
        def acquire_next_job(self, **_kwargs):
            return {
                "id": 42,
                "owner_user_id": "1",
                "payload": {
                    "action": "import",
                    "chatbooks_job_id": import_job.job_id,
                    "file_token": str(archive_path),
                    "source_format": "chatbook",
                    "content_selections": None,
                    "import_media": True,
                    "import_embeddings": True,
                },
                "lease_id": "lease-1",
            }

        def get_job(self, _job_id):
            return {"status": public_job_status}

        def complete_job(self, *_args, **_kwargs):
            nonlocal public_job_status
            public_job_status = "completed"
            stop_event.set()

        def fail_job(self, *_args, **kwargs):
            nonlocal public_job_status
            public_job_status = "failed"
            failures.append(kwargs)
            stop_event.set()

        def finalize_cancelled(self, *_args, **_kwargs):
            stop_event.set()

        def renew_job_lease(self, *_args, **_kwargs):
            return None

    class FakeChatbookService:
        def __init__(self, *_args, **_kwargs):
            return None

        def _get_import_job(self, _job_id):
            return import_job

        def _save_import_job(self, job):
            assert job is import_job

        def _import_chatbook_sync(
            self,
            file_path,
            content_selections,
            conflict_resolution,
            prefix_imported,
            import_media,
            import_embeddings,
        ):
            nonlocal import_call
            import_call = {
                "file_path": file_path,
                "content_selections": content_selections,
                "conflict_resolution": conflict_resolution,
                "prefix_imported": prefix_imported,
                "import_media": import_media,
                "import_embeddings": import_embeddings,
            }
            return True, "ok", {
                "imported_items": {"media": 1, "embedding": 1},
                "warnings": ["restore warning"],
            }

        def _resolve_import_archive_path(self, _file_ref):
            return Path(archive_path)

    monkeypatch.setattr(core_jobs_worker, "JobManager", FakeJobManager)
    monkeypatch.setattr(core_jobs_worker, "_build_chacha_db_for_user", lambda _owner: object())
    monkeypatch.setattr(core_jobs_worker, "ChatbookService", FakeChatbookService)
    monkeypatch.setenv("JOBS_POLL_INTERVAL_SECONDS", "0.01")

    await asyncio.wait_for(core_jobs_worker.run_chatbooks_core_jobs_worker(stop_event), timeout=2)

    assert failures == []
    assert import_call is not None
    assert import_call["import_media"] is True
    assert import_call["import_embeddings"] is True
    assert import_job.status is ImportStatus.COMPLETED
    assert import_job.progress_percentage == 100
    assert import_job.total_items == 2
    assert import_job.processed_items == 2
    assert import_job.warnings == ["queued with archive warning", "restore warning"]
    assert import_job.metadata["source_format"] == "chatbook"
    assert import_job.metadata["audit_id"] == "audit-1"
    assert import_job.metadata["imported_items"] == {"media": 1, "embedding": 1}
    assert public_job_status == "completed"


@pytest.mark.asyncio
async def test_completed_export_persists_progress_counts_size_and_redacted_metadata(
    tmp_path, monkeypatch
):
    archive_path = tmp_path / "full-account.chatbook"
    with zipfile.ZipFile(archive_path, "w") as zf:
        zf.writestr(
            "manifest.json",
            json.dumps(
                {
                    "user_id": "redacted-user-hash",
                    "source_instance": {"host": "must-not-leak"},
                    "account_inventory_summary": {
                        "counts": {"notes": 2, "media_records": 1},
                        "archive_size_bytes": 4096,
                        "post_write_verification": True,
                        "pointer_only_count": 0,
                        "warning_count": 1,
                        "sensitive_category_count": 2,
                    },
                }
            ),
        )

    stop_event = asyncio.Event()
    export_job = ExportJob(
        job_id="export-job-1",
        user_id="1",
        status=ExportStatus.PENDING,
        chatbook_name="Full account",
        metadata={"selection_mode": "full_account", "audit_id": "audit-2"},
    )
    public_job_status = "processing"
    failures = []

    class FakeJobManager:
        def acquire_next_job(self, **_kwargs):
            return {
                "id": 43,
                "owner_user_id": "1",
                "payload": {
                    "action": "export",
                    "chatbooks_job_id": export_job.job_id,
                    "name": export_job.chatbook_name,
                    "description": "full account export",
                    "selection_mode": "full_account",
                    "content_selections": None,
                },
                "lease_id": "lease-2",
            }

        def get_job(self, _job_id):
            return {"status": public_job_status}

        def complete_job(self, *_args, **_kwargs):
            nonlocal public_job_status
            public_job_status = "completed"
            stop_event.set()

        def fail_job(self, *_args, **kwargs):
            nonlocal public_job_status
            public_job_status = "failed"
            failures.append(kwargs)
            stop_event.set()

        def finalize_cancelled(self, *_args, **_kwargs):
            stop_event.set()

        def renew_job_lease(self, *_args, **_kwargs):
            return None

    class FakeChatbookService:
        def __init__(self, *_args, **_kwargs):
            self.export_dir = tmp_path

        def _get_export_job(self, _job_id):
            return export_job

        def _save_export_job(self, job):
            assert job is export_job

        async def _create_chatbook_sync_wrapper(self, **_kwargs):
            return True, "ok", str(archive_path)

        def build_export_job_metadata(self, output_path):
            return ChatbookService.build_export_job_metadata(output_path)

        def _get_export_expiry(self, now_utc):
            return now_utc

        def _get_download_expiry(self, _now_utc, expires_at):
            return expires_at

        def _build_download_url(self, job_id, _expires_at):
            return f"/api/v1/chatbooks/download/{job_id}"

    monkeypatch.setattr(core_jobs_worker, "JobManager", FakeJobManager)
    monkeypatch.setattr(core_jobs_worker, "_build_chacha_db_for_user", lambda _owner: object())
    monkeypatch.setattr(core_jobs_worker, "ChatbookService", FakeChatbookService)
    monkeypatch.setenv("JOBS_POLL_INTERVAL_SECONDS", "0.01")

    await asyncio.wait_for(core_jobs_worker.run_chatbooks_core_jobs_worker(stop_event), timeout=2)

    assert failures == []
    assert export_job.status is ExportStatus.COMPLETED
    assert export_job.progress_percentage == 100
    assert export_job.total_items == 3
    assert export_job.processed_items == 3
    assert export_job.file_size_bytes == archive_path.stat().st_size
    assert export_job.metadata["selection_mode"] == "full_account"
    assert export_job.metadata["audit_id"] == "audit-2"
    assert export_job.metadata["post_write_verification"] is True
    assert export_job.metadata["account_inventory_summary"]["counts"] == {
        "notes": 2,
        "media_records": 1,
    }
    assert "must-not-leak" not in json.dumps(export_job.metadata)
    assert public_job_status == "completed"
