"""Exercise cancellation and lease cleanup in the core Chatbooks worker."""

import asyncio
import time
from datetime import timedelta
from pathlib import Path

import pytest

from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Jobs.migrations import ensure_jobs_tables


@pytest.mark.asyncio
async def test_core_worker_honors_mid_processing_cancellation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # Prepare isolated jobs DB
    jobs_db = tmp_path / "jobs.db"
    ensure_jobs_tables(jobs_db)
    jm = JobManager(jobs_db)

    # Build a fake ChatbookService with minimal surface
    from tldw_Server_API.app.core.Chatbooks.chatbook_models import ExportStatus

    class FakeExportJob:
        def __init__(self, jid: str):
            self.job_id = jid
            self.status = ExportStatus.PENDING
            self.started_at = None
            self.completed_at = None
            self.output_path = None
            self.file_size_bytes = None
            self.expires_at = None
            self.download_url = None
            self.error_message = None

    class FakeChatbookService:
        def __init__(self, user_id, db, **kwargs):
            self._jobs = {}

        def _get_export_job(self, jid: str):
            return self._jobs.get(jid)

        def _save_export_job(self, ej):
            self._jobs[ej.job_id] = ej

        def _build_download_url(self, job_id, _exp):
            return f"http://test/{job_id}"

        async def _create_chatbook_sync_wrapper(self, **kwargs):
            # Simulate long work
            await asyncio.sleep(0.5)
            return True, None, "/tmp/fake.zip"  # nosec B108

    # Patch the worker to use our FakeChatbookService and a JM bound to our DB
    import tldw_Server_API.app.services.core_jobs_worker as worker

    class JMProxy:
        def __init__(self):
            self._jm = jm

        def __getattr__(self, name):
            return getattr(self._jm, name)

    monkeypatch.setattr(worker, "JobManager", JMProxy)
    monkeypatch.setattr(worker, "ChatbookService", FakeChatbookService)

    # Seed a Chatbooks export job state the worker expects
    svc = FakeChatbookService("1", None)
    ej = FakeExportJob("cb-1")
    svc._save_export_job(ej)

    # Ensure the worker built in module will find the ej when it constructs svc anew
    # by monkeypatching _get_export_job to always refer to our seeded state
    def _fake_get_export_job(self, jid):
        # Avoid recursion by reading from the seeded instance dictionary directly
        return svc._jobs.get(jid)

    monkeypatch.setattr(FakeChatbookService, "_get_export_job", _fake_get_export_job)

    # Create a job and start the worker
    job = jm.create_job(
        domain="chatbooks",
        queue="default",
        job_type="export",
        payload={"action": "export", "chatbooks_job_id": "cb-1", "user_id": "1"},
        owner_user_id="1",
    )

    stop_event = asyncio.Event()
    run_task = asyncio.create_task(worker.run_chatbooks_core_jobs_worker(stop_event))

    # Allow worker to acquire and start processing
    await asyncio.sleep(0.2)
    # Request cancellation mid-flight
    jm.cancel_job(int(job["id"]))

    # Wait for worker to process
    await asyncio.sleep(0.6)
    stop_event.set()
    # Prefer graceful stop; if still running after timeout, cancel
    try:
        await asyncio.wait_for(run_task, timeout=2.0)
    except asyncio.TimeoutError:
        run_task.cancel()
        try:
            await run_task
        except asyncio.CancelledError:
            pass

    # Assert job was terminally cancelled
    jr = jm.get_job(int(job["id"]))
    assert jr["status"] == "cancelled"
    # Assert Chatbooks export job reflected CANCELLED (allow brief propagation)
    deadline = asyncio.get_event_loop().time() + 3.0
    while True:
        ej2 = svc._get_export_job("cb-1")
        if ej2 and getattr(ej2.status, "name", str(ej2.status)) == "CANCELLED":
            break
        if asyncio.get_event_loop().time() > deadline:
            break
        await asyncio.sleep(0.05)
    assert ej2.status.name == "CANCELLED"


@pytest.mark.asyncio
async def test_core_worker_allows_supported_import_media_and_embeddings(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # Prepare isolated jobs DB
    jobs_db = tmp_path / "jobs.db"
    ensure_jobs_tables(jobs_db)
    jm = JobManager(jobs_db)

    from tldw_Server_API.app.core.Chatbooks.chatbook_models import ImportStatus

    class FakeImportJob:
        def __init__(self, jid: str):
            self.job_id = jid
            self.status = ImportStatus.PENDING
            self.started_at = None
            self.completed_at = None
            self.error_message = None
            self.warnings = []
            self.metadata = {}

    class FakeChatbookService:
        jobs = {}
        called_args = None

        def __init__(self, user_id, db, **kwargs):
            pass

        def _get_import_job(self, jid: str):
            return FakeChatbookService.jobs.get(jid)

        def _save_import_job(self, ij):
            FakeChatbookService.jobs[ij.job_id] = ij

        def _import_chatbook_sync(
            self,
            file_path,
            content_selections,
            conflict_resolution,
            prefix_imported,
            import_media,
            import_embeddings,
        ):
            FakeChatbookService.called_args = {
                "file_path": file_path,
                "content_selections": content_selections,
                "conflict_resolution": conflict_resolution,
                "prefix_imported": prefix_imported,
                "import_media": import_media,
                "import_embeddings": import_embeddings,
            }
            return True, "ok", {"imported_items": {"media": 1, "embedding": 1}, "warnings": []}

        async def finalize_account_restore(
            self,
            ok: bool,
            message: str,
            result: dict[str, object],
        ) -> tuple[bool, str, dict[str, object]]:
            return ok, message, result

        def _resolve_import_archive_path(self, file_ref):
            return Path(file_ref)

    import tldw_Server_API.app.services.core_jobs_worker as worker

    class JMProxy:
        def __init__(self):
            self._jm = jm

        def __getattr__(self, name):
            return getattr(self._jm, name)

    monkeypatch.setattr(worker, "JobManager", JMProxy)
    monkeypatch.setattr(worker, "ChatbookService", FakeChatbookService)
    monkeypatch.setattr(worker, "_build_chacha_db_for_user", lambda _user_id: None)

    # Seed import job state for worker
    FakeChatbookService.jobs["cb-imp-1"] = FakeImportJob("cb-imp-1")

    # Create dummy archive file for cleanup
    file_path = tmp_path / "import.chatbook"
    file_path.write_text("dummy", encoding="utf-8")

    job = jm.create_job(
        domain="chatbooks",
        queue="default",
        job_type="import",
        payload={
            "action": "import",
            "chatbooks_job_id": "cb-imp-1",
            "file_token": str(file_path),
            "source_format": "chatbook",
            "content_selections": None,
            "import_media": True,
            "import_embeddings": True,
        },
        owner_user_id="1",
    )

    stop_event = asyncio.Event()
    run_task = asyncio.create_task(worker.run_chatbooks_core_jobs_worker(stop_event))

    await asyncio.sleep(0.3)
    stop_event.set()
    try:
        await asyncio.wait_for(run_task, timeout=2.0)
    except asyncio.TimeoutError:
        run_task.cancel()
        try:
            await run_task
        except asyncio.CancelledError:
            pass

    jr = jm.get_job(int(job["id"]))
    assert jr["status"] == "completed"
    assert FakeChatbookService.called_args is not None
    assert FakeChatbookService.called_args["import_media"] is True
    assert FakeChatbookService.called_args["import_embeddings"] is True
    ij = FakeChatbookService.jobs["cb-imp-1"]
    assert ij.status == ImportStatus.COMPLETED
    assert ij.error_message is None


@pytest.mark.asyncio
async def test_core_worker_cleans_export_file_and_stops_renew_after_midflight_cancel(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    jobs_db = tmp_path / "jobs.db"
    ensure_jobs_tables(jobs_db)
    jm = JobManager(jobs_db)

    from tldw_Server_API.app.core.Chatbooks.chatbook_models import ExportStatus

    export_dir = tmp_path / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)
    export_path = export_dir / "cancelled.chatbook"

    class FakeExportJob:
        def __init__(self, jid: str):
            self.job_id = jid
            self.status = ExportStatus.PENDING
            self.started_at = None
            self.completed_at = None
            self.output_path = None
            self.file_size_bytes = None
            self.expires_at = None
            self.download_url = None
            self.error_message = None
            self.metadata = {}

    class FakeChatbookService:
        jobs = {}

        def __init__(self, user_id, db, **kwargs):
            self.export_dir = export_dir

        def _get_export_job(self, jid: str):
            return FakeChatbookService.jobs.get(jid)

        def _save_export_job(self, ej):
            FakeChatbookService.jobs[ej.job_id] = ej

        def _get_export_expiry(self, now_utc):
            return now_utc + timedelta(days=1)

        def _get_download_expiry(self, _now_utc, expires_at):
            return expires_at

        def _build_download_url(self, job_id, _exp):
            return f"http://test/{job_id}"

        async def _create_chatbook_sync_wrapper(self, **kwargs):
            export_path.write_text("payload", encoding="utf-8")
            await asyncio.sleep(0.4)
            return True, None, str(export_path)

    import tldw_Server_API.app.services.core_jobs_worker as worker

    renew_calls = {"count": 0}
    original_renew = jm.renew_job_lease

    def _counted_renew(*args, **kwargs):
        renew_calls["count"] += 1
        return original_renew(*args, **kwargs)

    monkeypatch.setattr(jm, "renew_job_lease", _counted_renew)

    class JMProxy:
        def __init__(self):
            self._jm = jm

        def __getattr__(self, name):
            return getattr(self._jm, name)

    monkeypatch.setattr(worker, "JobManager", JMProxy)
    monkeypatch.setattr(worker, "ChatbookService", FakeChatbookService)
    monkeypatch.setattr(worker, "_build_chacha_db_for_user", lambda _user_id: None)
    monkeypatch.setenv("JOBS_LEASE_RENEW_SECONDS", "1")
    monkeypatch.setenv("JOBS_LEASE_RENEW_JITTER_SECONDS", "0")

    FakeChatbookService.jobs["cb-clean-1"] = FakeExportJob("cb-clean-1")

    job = jm.create_job(
        domain="chatbooks",
        queue="default",
        job_type="export",
        payload={"action": "export", "chatbooks_job_id": "cb-clean-1", "user_id": "1"},
        owner_user_id="1",
    )

    stop_event = asyncio.Event()
    run_task = asyncio.create_task(worker.run_chatbooks_core_jobs_worker(stop_event))

    try:
        await asyncio.sleep(0.15)
        jm.cancel_job(int(job["id"]))

        deadline = time.time() + 3.0
        while time.time() < deadline:
            ej = FakeChatbookService.jobs.get("cb-clean-1")
            if (
                ej is not None
                and ej.status == ExportStatus.CANCELLED
                and ej.completed_at is not None
                and not export_path.exists()
            ):
                break
            await asyncio.sleep(0.05)
        jr = jm.get_job(int(job["id"]))
        assert jr and jr["status"] == "cancelled"
        ej = FakeChatbookService.jobs["cb-clean-1"]
        assert ej.status == ExportStatus.CANCELLED
        assert ej.completed_at is not None
        assert not export_path.exists()
        await asyncio.sleep(0.2)
        renew_count_after_cancel = renew_calls["count"]
        await asyncio.sleep(1.2)
        assert renew_calls["count"] == renew_count_after_cancel
    finally:
        stop_event.set()
        try:
            await asyncio.wait_for(run_task, timeout=2.0)
        except asyncio.TimeoutError:
            run_task.cancel()
            try:
                await run_task
            except asyncio.CancelledError:
                pass

    ej = FakeChatbookService.jobs["cb-clean-1"]
    assert ej.status == ExportStatus.CANCELLED


@pytest.mark.asyncio
async def test_core_worker_stops_renew_after_successful_export(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    jobs_db = tmp_path / "jobs.db"
    ensure_jobs_tables(jobs_db)
    jm = JobManager(jobs_db)

    from tldw_Server_API.app.core.Chatbooks.chatbook_models import ExportStatus

    export_dir = tmp_path / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)
    export_path = export_dir / "completed.chatbook"

    class FakeExportJob:
        def __init__(self, jid: str):
            self.job_id = jid
            self.status = ExportStatus.PENDING
            self.started_at = None
            self.completed_at = None
            self.output_path = None
            self.file_size_bytes = None
            self.expires_at = None
            self.download_url = None
            self.error_message = None
            self.metadata = {}

    class FakeChatbookService:
        jobs = {}

        def __init__(self, user_id, db, **kwargs):
            self.export_dir = export_dir

        def _get_export_job(self, jid: str):
            return FakeChatbookService.jobs.get(jid)

        def _save_export_job(self, ej):
            FakeChatbookService.jobs[ej.job_id] = ej

        def _get_export_expiry(self, now_utc):
            return now_utc + timedelta(days=1)

        def _get_download_expiry(self, _now_utc, expires_at):
            return expires_at

        def _build_download_url(self, job_id, _exp):
            return f"http://test/{job_id}"

        async def _create_chatbook_sync_wrapper(self, **kwargs):
            export_path.write_text("payload", encoding="utf-8")
            await asyncio.sleep(0.2)
            return True, None, str(export_path)

        def build_export_job_metadata(self, _file_path: str) -> dict[str, object]:
            return {}

    import tldw_Server_API.app.services.core_jobs_worker as worker

    renew_calls = {"count": 0}
    original_renew = jm.renew_job_lease

    def _counted_renew(*args, **kwargs):
        renew_calls["count"] += 1
        return original_renew(*args, **kwargs)

    monkeypatch.setattr(jm, "renew_job_lease", _counted_renew)

    class JMProxy:
        def __init__(self):
            self._jm = jm

        def __getattr__(self, name):
            return getattr(self._jm, name)

    monkeypatch.setattr(worker, "JobManager", JMProxy)
    monkeypatch.setattr(worker, "ChatbookService", FakeChatbookService)
    monkeypatch.setattr(worker, "_build_chacha_db_for_user", lambda _user_id: None)
    monkeypatch.setenv("JOBS_LEASE_RENEW_SECONDS", "1")
    monkeypatch.setenv("JOBS_LEASE_RENEW_JITTER_SECONDS", "0")

    FakeChatbookService.jobs["cb-ok-1"] = FakeExportJob("cb-ok-1")

    job = jm.create_job(
        domain="chatbooks",
        queue="default",
        job_type="export",
        payload={"action": "export", "chatbooks_job_id": "cb-ok-1", "user_id": "1"},
        owner_user_id="1",
    )

    stop_event = asyncio.Event()
    run_task = asyncio.create_task(worker.run_chatbooks_core_jobs_worker(stop_event))

    try:
        deadline = time.time() + 3.0
        while time.time() < deadline:
            jr = jm.get_job(int(job["id"]))
            if jr and jr.get("status") == "completed":
                break
            await asyncio.sleep(0.05)
        jr = jm.get_job(int(job["id"]))
        assert jr and jr["status"] == "completed"
        assert export_path.exists()

        await asyncio.sleep(0.2)
        renew_count_after_complete = renew_calls["count"]
        await asyncio.sleep(1.2)
        assert renew_calls["count"] == renew_count_after_complete
    finally:
        stop_event.set()
        try:
            await asyncio.wait_for(run_task, timeout=2.0)
        except asyncio.TimeoutError:
            run_task.cancel()
            try:
                await run_task
            except asyncio.CancelledError:
                pass

    ej = FakeChatbookService.jobs["cb-ok-1"]
    assert ej.status == ExportStatus.COMPLETED


@pytest.mark.asyncio
async def test_core_worker_propagates_task_cancellation_while_idle(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    jobs_db = tmp_path / "jobs.db"
    ensure_jobs_tables(jobs_db)
    jm = JobManager(jobs_db)

    import tldw_Server_API.app.services.core_jobs_worker as worker

    class JMProxy:
        def __init__(self):
            self._jm = jm

        def __getattr__(self, name):
            return getattr(self._jm, name)

    monkeypatch.setattr(worker, "JobManager", JMProxy)
    monkeypatch.setenv("JOBS_POLL_INTERVAL_SECONDS", "30")

    run_task = asyncio.create_task(worker.run_chatbooks_core_jobs_worker())
    await asyncio.sleep(0.1)
    run_task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(run_task, timeout=1.0)
