import asyncio
from pathlib import Path

import pytest

from tldw_Server_API.app.core.Chatbooks.chatbook_models import ImportJob, ImportStatus
from tldw_Server_API.app.services import core_jobs_worker

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_live_worker_restores_chatbook_media_and_embeddings(tmp_path, monkeypatch):
    archive_path = tmp_path / "full-account.chatbook"
    archive_path.write_bytes(b"temporary archive")
    stop_event = asyncio.Event()
    import_job = ImportJob(
        job_id="import-job-1",
        user_id="1",
        status=ImportStatus.PENDING,
        chatbook_path=str(archive_path),
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
            return True, "ok", {"imported_items": {"media": 1, "embedding": 1}}

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
    assert public_job_status == "completed"
