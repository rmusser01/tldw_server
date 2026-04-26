import pytest


pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_media_ingest_worker_honors_cancel_before_processing(monkeypatch, tmp_path):
    monkeypatch.setenv("JOBS_DB_PATH", str(tmp_path / "jobs.db"))
    monkeypatch.delenv("JOBS_DB_URL", raising=False)

    from tldw_Server_API.app.core.Jobs.manager import JobManager
    import tldw_Server_API.app.services.media_ingest_jobs_worker as worker

    jm = JobManager()
    payload = {
        "batch_id": "batch-1",
        "media_type": "document",
        "source": str(tmp_path / "cancelled.txt"),
        "source_kind": "file",
        "input_ref": "cancelled.txt",
        "temp_dir": str(tmp_path / "staging"),
        "cleanup_temp_dir": False,
        "options": {"media_type": "document"},
    }
    row = jm.create_job(
        domain="media_ingest",
        queue="default",
        job_type="media_ingest_item",
        payload=payload,
        owner_user_id="1",
    )
    job_id = int(row.get("id"))
    jm.cancel_job(job_id, reason="test cancel")

    def _boom(*_args, **_kwargs):
        raise AssertionError("processing called despite cancellation")

    monkeypatch.setattr(worker, "process_batch_media", _boom, raising=True)
    monkeypatch.setattr(worker, "process_document_like_item", _boom, raising=True)

    job = jm.get_job(job_id)
    progress = worker._ProgressState()
    result = await worker._handle_job(job, jm, progress)

    assert result == {}
    updated = jm.get_job(job_id)
    assert updated is not None
    assert updated.get("status") == "cancelled"


@pytest.mark.asyncio
async def test_media_ingest_worker_updates_progress_fields(monkeypatch, tmp_path):
    monkeypatch.setenv("JOBS_DB_PATH", str(tmp_path / "jobs.db"))
    monkeypatch.delenv("JOBS_DB_URL", raising=False)

    from tldw_Server_API.app.core.Jobs.manager import JobManager
    import tldw_Server_API.app.services.media_ingest_jobs_worker as worker

    class _DummyDB:
        def __init__(self, path: str):
            self.db_path_str = path
            self.client_id = "media_ingest_test"

        def close_connection(self):
            return None

    def _fake_create_db(_user_id: str):
        return _DummyDB(str(tmp_path / "media.db"))

    async def _fake_process_document_like_item(**_kwargs):
        return {
            "status": "Success",
            "db_id": 123,
            "media_uuid": "media-uuid-123",
            "warnings": None,
            "db_message": "Media added to database.",
        }

    monkeypatch.setattr(worker, "_create_db", _fake_create_db, raising=True)
    monkeypatch.setattr(worker, "process_document_like_item", _fake_process_document_like_item, raising=True)
    monkeypatch.setattr(worker, "prepare_chunking_options_dict", lambda _form: None, raising=True)

    jm = JobManager()
    payload = {
        "batch_id": "batch-2",
        "media_type": "document",
        "source": str(tmp_path / "doc.txt"),
        "source_kind": "file",
        "input_ref": "doc.txt",
        "options": {"media_type": "document"},
    }
    row = jm.create_job(
        domain="media_ingest",
        queue="default",
        job_type="media_ingest_item",
        payload=payload,
        owner_user_id="1",
    )

    job = jm.get_job(int(row.get("id")))
    progress = worker._ProgressState()
    result = await worker._handle_job(job, jm, progress)

    assert result.get("status") == "Success"
    assert result.get("db_message") == "Media added to database."
    updated = jm.get_job(int(row.get("id")))
    assert updated is not None
    assert updated.get("progress_message") == "completed"
    assert float(updated.get("progress_percent") or 0.0) >= 100.0


@pytest.mark.asyncio
async def test_media_ingest_worker_returns_existing_media_id_for_skipped_dedupe_result(monkeypatch, tmp_path):
    monkeypatch.setenv("JOBS_DB_PATH", str(tmp_path / "jobs.db"))
    monkeypatch.delenv("JOBS_DB_URL", raising=False)

    from tldw_Server_API.app.core.Jobs.manager import JobManager
    import tldw_Server_API.app.services.media_ingest_jobs_worker as worker

    class _DummyDB:
        def __init__(self, path: str):
            self.db_path_str = path
            self.client_id = "media_ingest_test"

        def close_connection(self):
            return None

    def _fake_create_db(_user_id: str):
        return _DummyDB(str(tmp_path / "media.db"))

    async def _fake_process_batch_media(**_kwargs):
        return [
            {
                "status": "Skipped",
                "db_id": 321,
                "media_uuid": "existing-media-uuid",
                "db_message": "Media already exists.",
                "warnings": None,
            }
        ]

    monkeypatch.setattr(worker, "_create_db", _fake_create_db, raising=True)
    monkeypatch.setattr(worker, "process_batch_media", _fake_process_batch_media, raising=True)
    monkeypatch.setattr(worker, "prepare_chunking_options_dict", lambda _form: None, raising=True)

    jm = JobManager()
    payload = {
        "batch_id": "batch-skip",
        "media_type": "video",
        "source": "https://example.com/video?id=123",
        "source_kind": "url",
        "input_ref": "https://example.com/video?id=123",
        "options": {"media_type": "video", "transcription_model": "whisper-test"},
    }
    row = jm.create_job(
        domain="media_ingest",
        queue="default",
        job_type="media_ingest_item",
        payload=payload,
        owner_user_id="1",
    )

    job = jm.get_job(int(row.get("id")))
    progress = worker._ProgressState()
    result = await worker._handle_job(job, jm, progress)

    assert result == {
        "status": "Skipped",
        "media_id": 321,
        "media_uuid": "existing-media-uuid",
        "error": None,
        "warnings": None,
        "db_message": "Media already exists.",
    }


@pytest.mark.asyncio
async def test_media_ingest_heavy_worker_uses_configured_queue(monkeypatch):
    import tldw_Server_API.app.services.media_ingest_jobs_worker as worker

    monkeypatch.setenv("MEDIA_INGEST_JOBS_HEAVY_QUEUE", "media-heavy-q")
    called = {}

    async def _fake_run_media_ingest_jobs_worker(
        stop_event,
        *,
        queue=None,
        worker_id=None,
    ):
        called["stop_event"] = stop_event
        called["queue"] = queue
        called["worker_id"] = worker_id

    monkeypatch.setattr(
        worker,
        "run_media_ingest_jobs_worker",
        _fake_run_media_ingest_jobs_worker,
        raising=True,
    )

    await worker.run_media_ingest_heavy_jobs_worker(None)
    assert called["queue"] == "media-heavy-q"
    assert called["worker_id"] == "media-ingest-worker-media-heavy-q"


@pytest.mark.asyncio
async def test_media_ingest_worker_clears_stale_acquire_gate_on_start(monkeypatch):
    import tldw_Server_API.app.services.media_ingest_jobs_worker as worker
    from tldw_Server_API.app.core.Jobs.manager import JobManager

    class _DummySDK:
        def __init__(self, jm, cfg):
            self.jm = jm
            self.cfg = cfg

        def stop(self):
            return None

        async def run(self, **_kwargs):
            assert JobManager._ACQUIRE_GATE_ENABLED is False

    monkeypatch.setattr(worker, "WorkerSDK", _DummySDK, raising=True)

    try:
        JobManager.set_acquire_gate(True)
        await worker.run_media_ingest_jobs_worker()
    finally:
        JobManager.set_acquire_gate(False)


@pytest.mark.asyncio
async def test_media_ingest_schedule_embeddings_marks_media_processed(monkeypatch):
    import tldw_Server_API.app.services.media_ingest_jobs_worker as worker

    class _DummyDB:
        def __init__(self) -> None:
            self.errors: list[tuple[int, str]] = []

        def mark_embeddings_error(self, media_id: int, detail: str) -> None:
            self.errors.append((media_id, detail))

    captured: dict[str, object] = {}

    async def _fake_get_media_content(media_id: int, db):  # noqa: ARG001
        return {"media_item": {"title": f"Doc {media_id}"}, "content": {"content": "hello world"}}

    async def _fake_generate_embeddings_for_media(**kwargs):
        captured["kwargs"] = dict(kwargs)
        return {"status": "success", "embedding_count": 1, "chunks_processed": 1}

    def _fake_mark_media_as_processed(*, db_instance, media_id):  # noqa: ANN001
        captured["processed"] = (db_instance, media_id)

    monkeypatch.setattr(worker, "mark_media_as_processed", _fake_mark_media_as_processed, raising=True)

    import tldw_Server_API.app.api.v1.endpoints.media_embeddings as media_embeddings

    monkeypatch.setattr(media_embeddings, "get_media_content", _fake_get_media_content, raising=True)
    monkeypatch.setattr(
        media_embeddings,
        "generate_embeddings_for_media",
        _fake_generate_embeddings_for_media,
        raising=True,
    )

    db = _DummyDB()
    await worker._schedule_embeddings(
        media_id=55,
        user_id="77",
        db=db,
        form_data=type(
            "FormData",
            (),
            {
                "embedding_model": None,
                "embedding_provider": None,
                "chunk_size": 64,
                "overlap": 16,
            },
        )(),
    )

    assert captured["processed"] == (db, 55)
    assert captured["kwargs"]["user_id"] == "77"
    assert db.errors == []


@pytest.mark.asyncio
async def test_media_ingest_schedule_embeddings_marks_error_on_failure(monkeypatch):
    import tldw_Server_API.app.services.media_ingest_jobs_worker as worker

    class _DummyDB:
        def __init__(self) -> None:
            self.errors: list[tuple[int, str]] = []

        def mark_embeddings_error(self, media_id: int, detail: str) -> None:
            self.errors.append((media_id, detail))

    async def _fake_get_media_content(media_id: int, db):  # noqa: ARG001
        return {"media_item": {"title": f"Doc {media_id}"}, "content": {"content": "hello world"}}

    async def _fake_generate_embeddings_for_media(**_kwargs):
        return {"status": "error", "error": "embedding backend unavailable"}

    def _fail_if_processed(**_kwargs):  # noqa: ANN001
        raise AssertionError("mark_media_as_processed should not run for failed embeddings")

    monkeypatch.setattr(worker, "mark_media_as_processed", _fail_if_processed, raising=True)

    import tldw_Server_API.app.api.v1.endpoints.media_embeddings as media_embeddings

    monkeypatch.setattr(media_embeddings, "get_media_content", _fake_get_media_content, raising=True)
    monkeypatch.setattr(
        media_embeddings,
        "generate_embeddings_for_media",
        _fake_generate_embeddings_for_media,
        raising=True,
    )

    db = _DummyDB()
    await worker._schedule_embeddings(
        media_id=91,
        user_id="12",
        db=db,
        form_data=type(
            "FormData",
            (),
            {
                "embedding_model": None,
                "embedding_provider": None,
                "chunk_size": 64,
                "overlap": 16,
            },
        )(),
    )

    assert db.errors == [(91, "embedding backend unavailable")]


@pytest.mark.asyncio
async def test_media_ingest_schedule_embeddings_retries_conflict_without_marking_error(monkeypatch):
    import tldw_Server_API.app.services.media_ingest_jobs_worker as worker
    from tldw_Server_API.app.core.DB_Management.media_db.errors import ConflictError

    class _DummyDB:
        def __init__(self) -> None:
            self.errors: list[tuple[int, str]] = []

        def mark_embeddings_error(self, media_id: int, detail: str) -> None:
            self.errors.append((media_id, detail))

    attempts = {"processed": 0}
    captured: dict[str, object] = {}

    async def _fake_get_media_content(media_id: int, db):  # noqa: ARG001
        return {"media_item": {"title": f"Doc {media_id}"}, "content": {"content": "hello world"}}

    async def _fake_generate_embeddings_for_media(**kwargs):
        captured["kwargs"] = dict(kwargs)
        return {"status": "success", "embedding_count": 1, "chunks_processed": 1}

    def _flaky_mark_media_as_processed(*, db_instance, media_id):  # noqa: ANN001
        attempts["processed"] += 1
        if attempts["processed"] == 1:
            raise ConflictError("Media", media_id)
        captured["processed"] = (db_instance, media_id)

    monkeypatch.setattr(worker, "mark_media_as_processed", _flaky_mark_media_as_processed, raising=True)

    import tldw_Server_API.app.api.v1.endpoints.media_embeddings as media_embeddings

    monkeypatch.setattr(media_embeddings, "get_media_content", _fake_get_media_content, raising=True)
    monkeypatch.setattr(
        media_embeddings,
        "generate_embeddings_for_media",
        _fake_generate_embeddings_for_media,
        raising=True,
    )

    db = _DummyDB()
    await worker._schedule_embeddings(
        media_id=73,
        user_id="22",
        db=db,
        form_data=type(
            "FormData",
            (),
            {
                "embedding_model": None,
                "embedding_provider": None,
                "chunk_size": 64,
                "overlap": 16,
            },
        )(),
    )

    assert attempts["processed"] == 2
    assert captured["processed"] == (db, 73)
    assert captured["kwargs"]["user_id"] == "22"
    assert db.errors == []
