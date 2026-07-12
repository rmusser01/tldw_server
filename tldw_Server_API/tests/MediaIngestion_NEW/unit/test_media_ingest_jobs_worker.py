from pathlib import Path

import pytest


pytestmark = pytest.mark.unit


class _CollectionUpdateRecorder:
    def __init__(self) -> None:
        self.status_updates: list[dict] = []
        self.resolutions: list[dict] = []

    def update_media_collection_item_status(self, item_id: int, **kwargs):
        self.status_updates.append({"item_id": item_id, **kwargs})
        return {"id": item_id, **kwargs}

    def resolve_media_collection_item(self, item_id: int, **kwargs):
        self.resolutions.append({"item_id": item_id, **kwargs})
        return {"id": item_id, **kwargs}

    def close(self):
        return None


def test_media_ingest_worker_defaults_to_responsive_idle_backoff(monkeypatch):
    import tldw_Server_API.app.services.media_ingest_jobs_worker as worker

    monkeypatch.delenv("MEDIA_INGEST_JOBS_BACKOFF_BASE_SECONDS", raising=False)
    monkeypatch.delenv("MEDIA_INGEST_JOBS_BACKOFF_MAX_SECONDS", raising=False)

    config = worker._build_worker_config(worker_id="media-worker", queue="default")

    assert config.backoff_base_seconds == 2
    assert config.backoff_max_seconds == 2


def _install_fake_collections_db(monkeypatch, worker, recorder: _CollectionUpdateRecorder):
    class _FakeCollectionsDatabase:
        @classmethod
        def for_user(cls, user_id):  # noqa: ARG003
            return recorder

    monkeypatch.setattr(worker, "CollectionsDatabase", _FakeCollectionsDatabase, raising=False)


def test_truncate_collection_error_redacts_sensitive_details():
    import tldw_Server_API.app.services.media_ingest_jobs_worker as worker

    raw = (
        "failed token=super-secret at /private/tmp/video.mp4 "
        "url=https://example.com/watch?api_key=secret-value "
        "hash=0123456789abcdef0123456789abcdef"
    )

    sanitized = worker._truncate_collection_error(raw)

    assert sanitized is not None
    assert "super-secret" not in sanitized
    assert "secret-value" not in sanitized
    assert "/private/tmp/video.mp4" not in sanitized
    assert "0123456789abcdef0123456789abcdef" not in sanitized
    assert "[redacted]" in sanitized
    assert "[redacted-path]" in sanitized
    assert "[redacted-hex]" in sanitized


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
async def test_media_ingest_worker_marks_planned_collection_item_completed(monkeypatch, tmp_path):
    monkeypatch.setenv("JOBS_DB_PATH", str(tmp_path / "jobs.db"))
    monkeypatch.delenv("JOBS_DB_URL", raising=False)

    from tldw_Server_API.app.core.Jobs.manager import JobManager
    import tldw_Server_API.app.services.media_ingest_jobs_worker as worker

    recorder = _CollectionUpdateRecorder()
    _install_fake_collections_db(monkeypatch, worker, recorder)

    class _DummyDB:
        db_path_str = str(tmp_path / "media.db")
        client_id = "media_ingest_test"

        def close_connection(self):
            return None

    async def _fake_process_batch_media(**_kwargs):
        return [
            {
                "status": "Success",
                "db_id": 456,
                "media_uuid": "media-uuid-456",
                "warnings": None,
                "db_message": "Media added to database.",
            }
        ]

    async def _fake_chunking_resolver(*_args, **_kwargs):
        return None, None

    monkeypatch.setattr(worker, "_create_db", lambda _user_id: _DummyDB(), raising=True)
    monkeypatch.setattr(worker, "process_batch_media", _fake_process_batch_media, raising=True)
    monkeypatch.setattr(
        worker,
        "async_resolve_chunking_options_and_plan",
        _fake_chunking_resolver,
        raising=True,
    )

    jm = JobManager()
    row = jm.create_job(
        domain="media_ingest",
        queue="default",
        job_type="media_ingest_item",
        payload={
            "batch_id": "batch-conf-complete",
            "media_type": "video",
            "source": "https://example.com/talk-1",
            "source_kind": "url",
            "input_ref": "https://example.com/talk-1",
            "collection_id": "42",
            "planned_item_id": "101",
            "idempotency_key": "conference-42-101-0",
            "options": {"media_type": "video"},
        },
        owner_user_id="1",
    )

    result = await worker._handle_job(
        jm.get_job(int(row.get("id"))),
        jm,
        worker._ProgressState(),
    )

    assert result["media_id"] == 456
    assert recorder.status_updates == [
        {
            "item_id": 101,
            "status": "processing",
            "latest_job_id": str(row.get("id")),
        }
    ]
    assert recorder.resolutions == [
        {
            "item_id": 101,
            "media_id": 456,
            "status": "completed",
            "latest_job_id": str(row.get("id")),
        }
    ]


@pytest.mark.asyncio
async def test_media_ingest_worker_marks_duplicate_result_skipped_existing(monkeypatch, tmp_path):
    monkeypatch.setenv("JOBS_DB_PATH", str(tmp_path / "jobs.db"))
    monkeypatch.delenv("JOBS_DB_URL", raising=False)

    from tldw_Server_API.app.core.Jobs.manager import JobManager
    import tldw_Server_API.app.services.media_ingest_jobs_worker as worker

    recorder = _CollectionUpdateRecorder()
    _install_fake_collections_db(monkeypatch, worker, recorder)

    class _DummyDB:
        db_path_str = str(tmp_path / "media.db")
        client_id = "media_ingest_test"

        def close_connection(self):
            return None

    async def _fake_process_batch_media(**_kwargs):
        return [
            {
                "status": "Skipped",
                "db_id": 789,
                "media_uuid": "existing-media-uuid",
                "warnings": None,
                "db_message": "Media already exists.",
            }
        ]

    async def _fake_chunking_resolver(*_args, **_kwargs):
        return None, None

    monkeypatch.setattr(worker, "_create_db", lambda _user_id: _DummyDB(), raising=True)
    monkeypatch.setattr(worker, "process_batch_media", _fake_process_batch_media, raising=True)
    monkeypatch.setattr(
        worker,
        "async_resolve_chunking_options_and_plan",
        _fake_chunking_resolver,
        raising=True,
    )

    jm = JobManager()
    row = jm.create_job(
        domain="media_ingest",
        queue="default",
        job_type="media_ingest_item",
        payload={
            "batch_id": "batch-conf-skip",
            "media_type": "video",
            "source": "https://example.com/talk-2",
            "source_kind": "url",
            "input_ref": "https://example.com/talk-2",
            "collection_id": "42",
            "planned_item_id": "102",
            "options": {"media_type": "video"},
        },
        owner_user_id="1",
    )

    await worker._handle_job(jm.get_job(int(row.get("id"))), jm, worker._ProgressState())

    assert recorder.resolutions == [
        {
            "item_id": 102,
            "media_id": 789,
            "status": "skipped_existing",
            "latest_job_id": str(row.get("id")),
        }
    ]


@pytest.mark.asyncio
async def test_media_ingest_worker_marks_missing_media_id_failed(monkeypatch, tmp_path):
    monkeypatch.setenv("JOBS_DB_PATH", str(tmp_path / "jobs.db"))
    monkeypatch.delenv("JOBS_DB_URL", raising=False)

    from tldw_Server_API.app.core.Jobs.manager import JobManager
    import tldw_Server_API.app.services.media_ingest_jobs_worker as worker

    recorder = _CollectionUpdateRecorder()
    _install_fake_collections_db(monkeypatch, worker, recorder)

    class _DummyDB:
        db_path_str = str(tmp_path / "media.db")
        client_id = "media_ingest_test"

        def close_connection(self):
            return None

    async def _fake_process_batch_media(**_kwargs):
        return [
            {
                "status": "Success",
                "media_uuid": None,
                "warnings": None,
                "db_message": "Media was accepted without a row id.",
            }
        ]

    async def _fake_chunking_resolver(*_args, **_kwargs):
        return None, None

    monkeypatch.setattr(worker, "_create_db", lambda _user_id: _DummyDB(), raising=True)
    monkeypatch.setattr(worker, "process_batch_media", _fake_process_batch_media, raising=True)
    monkeypatch.setattr(
        worker,
        "async_resolve_chunking_options_and_plan",
        _fake_chunking_resolver,
        raising=True,
    )

    jm = JobManager()
    row = jm.create_job(
        domain="media_ingest",
        queue="default",
        job_type="media_ingest_item",
        payload={
            "batch_id": "batch-conf-no-media-id",
            "media_type": "video",
            "source": "https://example.com/talk-no-media-id",
            "source_kind": "url",
            "input_ref": "https://example.com/talk-no-media-id",
            "collection_id": "42",
            "planned_item_id": "104",
            "options": {"media_type": "video"},
        },
        owner_user_id="1",
    )

    await worker._handle_job(jm.get_job(int(row.get("id"))), jm, worker._ProgressState())

    assert recorder.resolutions == []
    assert recorder.status_updates[-1] == {
        "item_id": 104,
        "status": "failed",
        "latest_job_id": str(row.get("id")),
        "error_summary": "No media id returned",
    }


@pytest.mark.asyncio
async def test_workspace_source_ingest_job_reports_existing_media_readiness(monkeypatch, tmp_path):
    monkeypatch.setenv("JOBS_DB_PATH", str(tmp_path / "jobs.db"))
    monkeypatch.delenv("JOBS_DB_URL", raising=False)

    from tldw_Server_API.app.core.Jobs.manager import JobManager
    import tldw_Server_API.app.services.media_ingest_jobs_worker as worker

    class _WorkspaceMediaDB:
        def __init__(self) -> None:
            self.closed = False

        def get_media_by_id(
            self,
            media_id: int,
            *,
            include_deleted: bool = False,
            include_trash: bool = False,
        ):
            _ = (include_deleted, include_trash)
            assert media_id == 123
            return {
                "id": 123,
                "title": "Ready workspace source",
                "content": "Evidence text",
                "chunking_status": "completed",
                "vector_processing": 1,
                "summary": "Short summary",
            }

        def close_connection(self) -> None:
            self.closed = True

    media_db = _WorkspaceMediaDB()
    monkeypatch.setattr(worker, "_create_db", lambda _user_id: media_db, raising=True)

    def _boom(*_args, **_kwargs):
        raise AssertionError("workspace source readiness jobs must not re-ingest media")

    monkeypatch.setattr(worker, "process_batch_media", _boom, raising=True)
    monkeypatch.setattr(worker, "process_document_like_item", _boom, raising=True)

    jm = JobManager()
    row = jm.create_job(
        domain="media_ingest",
        queue="default",
        job_type="workspace_source_ingest",
        payload={
            "workspace_id": "ws-ready",
            "workspace_source_id": "src-ready",
            "source_id": "src-ready",
            "media_id": 123,
            "source_type": "pdf",
            "title": "Ready workspace source",
            "requested_stages": ["ingestion", "extraction", "chunking", "indexing"],
        },
        owner_user_id="1",
    )

    job = jm.get_job(int(row.get("id")))
    progress = worker._ProgressState()
    result = await worker._handle_job(job, jm, progress)

    assert result == {
        "status": "ready",
        "workspace_id": "ws-ready",
        "workspace_source_id": "src-ready",
        "media_id": 123,
        "state": "queryable",
        "readiness": {
            "metadata_ready": True,
            "text_extracted": True,
            "fts_ready": True,
            "vector_ready": True,
            "citation_ready": True,
            "summary_ready": True,
            "tool_accessible": True,
        },
    }
    updated = jm.get_job(int(row.get("id")))
    assert updated is not None
    assert updated.get("progress_message") == "completed"
    assert float(updated.get("progress_percent") or 0.0) == 100.0
    assert media_db.closed is True


@pytest.mark.unit
async def test_workspace_source_ingest_job_missing_media_sets_stable_failure_code(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("JOBS_DB_PATH", str(tmp_path / "jobs.db"))
    monkeypatch.delenv("JOBS_DB_URL", raising=False)

    from tldw_Server_API.app.core.Jobs.manager import JobManager
    import tldw_Server_API.app.services.media_ingest_jobs_worker as worker

    class _MissingWorkspaceMediaDB:
        def __init__(self) -> None:
            self.closed = False

        def get_media_by_id(
            self,
            media_id: int,
            *,
            include_deleted: bool = False,
            include_trash: bool = False,
        ) -> dict[str, object] | None:
            _ = (include_deleted, include_trash)
            assert media_id == 404
            return None

        def close_connection(self) -> None:
            self.closed = True

    media_db = _MissingWorkspaceMediaDB()
    monkeypatch.setattr(worker, "_create_db", lambda _user_id: media_db, raising=True)

    jm = JobManager()
    row = jm.create_job(
        domain="media_ingest",
        queue="default",
        job_type="workspace_source_ingest",
        payload={
            "workspace_id": "ws-missing",
            "workspace_source_id": "src-missing",
            "media_id": 404,
            "source_type": "pdf",
            "title": "Missing workspace source",
        },
        owner_user_id="1",
    )

    job = jm.get_job(int(row.get("id")))
    with pytest.raises(worker.MediaIngestJobError) as exc_info:
        await worker._handle_job(job, jm, worker._ProgressState())

    assert str(exc_info.value) == "workspace source media item not found"
    assert exc_info.value.retryable is False
    assert exc_info.value.failure_code == "workspace_source_media_not_found"
    assert media_db.closed is True


@pytest.mark.unit
def test_media_ingest_job_error_exposes_optional_failure_code() -> None:
    import tldw_Server_API.app.services.media_ingest_jobs_worker as worker

    exc = worker.MediaIngestJobError("fallback failure")

    assert exc.failure_code is None


@pytest.mark.asyncio
async def test_media_ingest_worker_marks_planned_collection_item_failed_on_exception(monkeypatch, tmp_path):
    monkeypatch.setenv("JOBS_DB_PATH", str(tmp_path / "jobs.db"))
    monkeypatch.delenv("JOBS_DB_URL", raising=False)

    from tldw_Server_API.app.core.Jobs.manager import JobManager
    import tldw_Server_API.app.services.media_ingest_jobs_worker as worker

    recorder = _CollectionUpdateRecorder()
    _install_fake_collections_db(monkeypatch, worker, recorder)

    class _DummyDB:
        db_path_str = str(tmp_path / "media.db")
        client_id = "media_ingest_test"

        def close_connection(self):
            return None

    async def _fake_process_batch_media(**_kwargs):
        raise RuntimeError("private video")

    async def _fake_chunking_resolver(*_args, **_kwargs):
        return None, None

    monkeypatch.setattr(worker, "_create_db", lambda _user_id: _DummyDB(), raising=True)
    monkeypatch.setattr(worker, "process_batch_media", _fake_process_batch_media, raising=True)
    monkeypatch.setattr(
        worker,
        "async_resolve_chunking_options_and_plan",
        _fake_chunking_resolver,
        raising=True,
    )

    jm = JobManager()
    row = jm.create_job(
        domain="media_ingest",
        queue="default",
        job_type="media_ingest_item",
        payload={
            "batch_id": "batch-conf-fail",
            "media_type": "video",
            "source": "https://example.com/private-talk",
            "source_kind": "url",
            "input_ref": "https://example.com/private-talk",
            "collection_id": "42",
            "planned_item_id": "103",
            "options": {"media_type": "video"},
        },
        owner_user_id="1",
    )

    with pytest.raises(RuntimeError, match="private video"):
        await worker._handle_job(jm.get_job(int(row.get("id"))), jm, worker._ProgressState())

    assert recorder.status_updates[-1] == {
        "item_id": 103,
        "status": "failed",
        "latest_job_id": str(row.get("id")),
        "error_summary": "private video",
    }


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
async def test_media_ingest_worker_returns_auto_chunking_plan(monkeypatch, tmp_path):
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

    seen: dict[str, object] = {}

    async def _fake_process_document_like_item(**kwargs):
        seen["chunk_options"] = kwargs.get("chunk_options")
        return {
            "status": "Success",
            "db_id": 234,
            "media_uuid": "media-uuid-234",
            "warnings": None,
            "db_message": "Media added to database.",
            "content": "# Intro\n\nChunkable body.",
            "metadata": {
                "chunking_plan": {
                    "mode": "auto",
                    "goal": "navigation_summary",
                    "used_llm": False,
                    "method": "structure_aware",
                    "max_size": 1400,
                    "overlap": 100,
                    "fallback_reason": None,
                    "derived_views": ["section_titles", "outline"],
                    "profile": {"media_type": "document"},
                }
            },
        }

    monkeypatch.setattr(worker, "_create_db", _fake_create_db, raising=True)
    monkeypatch.setattr(
        worker,
        "process_document_like_item",
        _fake_process_document_like_item,
        raising=True,
    )

    jm = JobManager()
    payload = {
        "batch_id": "batch-auto",
        "media_type": "document",
        "source": str(tmp_path / "doc.md"),
        "source_kind": "file",
        "input_ref": "doc.md",
        "options": {
            "media_type": "document",
            "perform_chunking": True,
            "chunking_mode": "auto",
            "auto_chunking_goal": "qa_search",
            "auto_chunking_use_llm": True,
            "chunk_method": "words",
            "chunk_size": 333,
            "chunk_overlap": 1,
        },
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

    assert seen["chunk_options"]["method"] == "semantic"
    assert seen["chunk_options"]["max_size"] == 700
    assert result["chunking_plan"]["mode"] == "auto"
    assert result["chunking_plan"]["goal"] == "navigation_summary"
    assert result["chunking_plan"]["method"] == "structure_aware"
    assert result["chunking_plan"]["max_size"] == 1400
    assert result["chunking_plan"]["used_llm"] is False
    assert "ai_assist_unavailable" not in str(result["chunking_plan"]["fallback_reason"])


@pytest.mark.asyncio
async def test_media_ingest_worker_uses_async_auto_chunking_resolver(monkeypatch, tmp_path):
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

    async_resolver_calls = []
    seen: dict[str, object] = {}

    async def _fake_async_resolver(*args, **kwargs):
        async_resolver_calls.append((args, kwargs))
        return (
            {
                "method": "semantic",
                "max_size": 820,
                "overlap": 82,
                "adaptive": False,
                "multi_level": False,
                "language": None,
            },
            {
                "mode": "auto",
                "goal": "balanced",
                "used_llm": True,
                "method": "semantic",
                "max_size": 820,
                "overlap": 82,
                "template_name": None,
                "derived_views": ["topic_sections"],
                "fallback_reason": None,
                "rationale": "Assistant selected topic shifts.",
                "profile": {"media_type": "document"},
                "provider": "openai",
                "model": "gpt-test",
            },
        )

    async def _fake_process_document_like_item(**kwargs):
        seen["chunk_options"] = kwargs.get("chunk_options")
        return {
            "status": "Success",
            "db_id": 235,
            "media_uuid": "media-uuid-235",
            "warnings": None,
            "db_message": "Media added to database.",
            "metadata": {},
        }

    monkeypatch.setattr(worker, "_create_db", lambda _user_id: _DummyDB(str(tmp_path / "media.db")), raising=True)
    monkeypatch.setattr(worker, "async_resolve_chunking_options_and_plan", _fake_async_resolver, raising=True)
    monkeypatch.setattr(worker, "process_document_like_item", _fake_process_document_like_item, raising=True)

    jm = JobManager()
    row = jm.create_job(
        domain="media_ingest",
        queue="default",
        job_type="media_ingest_item",
        payload={
            "batch_id": "batch-auto-async",
            "media_type": "document",
            "source": str(tmp_path / "doc.md"),
            "source_kind": "file",
            "input_ref": "doc.md",
            "options": {
                "media_type": "document",
                "perform_chunking": True,
                "chunking_mode": "auto",
                "auto_chunking_use_llm": True,
            },
        },
        owner_user_id="1",
    )

    result = await worker._handle_job(jm.get_job(int(row.get("id"))), jm, worker._ProgressState())

    assert async_resolver_calls
    assert seen["chunk_options"]["method"] == "semantic"
    assert seen["chunk_options"]["max_size"] == 820
    assert result["chunking_plan"]["used_llm"] is True
    assert result["chunking_plan"]["provider"] == "openai"


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


def _playlist_preflight_data(raw_items):
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Video.playlist_preflight import (
        PlaylistPreflightData,
        normalize_preflight_items,
    )

    items = normalize_preflight_items(raw_items)
    return PlaylistPreflightData(
        source_url="https://www.youtube.com/playlist?list=PLworker",
        source_kind="youtube_playlist",
        playlist_id="PLworker",
        playlist_title="Worker playlist",
        video_id=None,
        item_count=len(items),
        selected_count=sum(item.selected for item in items),
        duplicate_count=sum(item.duplicate_status != "new" for item in items),
        warnings=[],
        items=items,
    )


def _seed_playlist_preflight_job(jm):
    from datetime import timedelta
    from unittest.mock import patch
    from uuid import uuid4

    import tldw_Server_API.app.core.Ingestion_Media_Processing.Video.playlist_ingest_store as store_module
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Video.playlist_ingest_store import (
        PlaylistIngestStore,
    )

    store = PlaylistIngestStore(jm)
    preflight_id = str(uuid4())
    job = jm.create_job(
        domain="media_ingest",
        queue="default",
        job_type="playlist_preflight",
        payload={
            "preflight_id": preflight_id,
            "source_url": "https://attacker.invalid/ignored?token=secret",
            "max_items": 10,
            "timeout_seconds": 5,
        },
        owner_user_id="7",
    )
    with patch.object(store_module, "uuid4", return_value=preflight_id):
        record = store.create_preflight(
            "7",
            source_url="https://www.youtube.com/playlist?list=PLworker",
            source_kind="youtube_playlist",
            expires_at=store._now() + timedelta(hours=1),
            playlist_id="PLworker",
            job_id=int(job["id"]),
        )
    claimed = jm.acquire_next_job(
        domain="media_ingest",
        queue="default",
        lease_seconds=120,
        worker_id="playlist-preflight-test-worker",
    )
    assert claimed is not None
    return store, record, claimed


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("field", "value"),
    [("lease_id", None), ("lease_id", 123), ("worker_id", " ")],
)
async def test_playlist_preflight_worker_rejects_missing_or_malformed_lease_before_extraction(
    monkeypatch,
    tmp_path,
    field,
    value,
):
    monkeypatch.setenv("JOBS_DB_PATH", str(tmp_path / "jobs.db"))
    monkeypatch.delenv("JOBS_DB_URL", raising=False)

    import tldw_Server_API.app.services.media_ingest_jobs_worker as worker
    from tldw_Server_API.app.core.Jobs.manager import JobManager

    jm = JobManager()
    store, preflight, job = _seed_playlist_preflight_job(jm)
    malformed_job = dict(job)
    malformed_job[field] = value
    runner_called = False

    async def fake_runner(_url, **_kwargs):
        nonlocal runner_called
        runner_called = True
        return _playlist_preflight_data([{"source_url": "https://youtu.be/lease123"}])

    monkeypatch.setattr(worker, "run_playlist_preflight_process", fake_runner, raising=True)

    with pytest.raises(worker.MediaIngestJobError) as exc_info:
        await worker._handle_job(malformed_job, jm, worker._ProgressState())

    assert str(exc_info.value) == "playlist_preflight_lease_required"
    assert runner_called is False
    assert store.get_preflight("7", preflight.preflight_id).status == "pending"
    assert list(store.list_preflight_items("7", preflight.preflight_id, limit=10)) == []


@pytest.mark.asyncio
async def test_playlist_preflight_stale_worker_does_not_mutate_or_block_active_reclaim(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("JOBS_DB_PATH", str(tmp_path / "jobs.db"))
    monkeypatch.delenv("JOBS_DB_URL", raising=False)

    import tldw_Server_API.app.services.media_ingest_jobs_worker as worker
    from tldw_Server_API.app.core.Jobs.manager import JobManager

    jm = JobManager()
    store, preflight, stale_job = _seed_playlist_preflight_job(jm)
    connection = jm._connect()
    try:
        connection.execute(
            "UPDATE jobs SET leased_until = DATETIME('now', '-1 second') WHERE id = ?",
            (int(stale_job["id"]),),
        )
        connection.commit()
    finally:
        connection.close()
    active_job = jm.acquire_next_job(
        domain="media_ingest",
        queue="default",
        lease_seconds=120,
        worker_id="playlist-preflight-reclaimer",
    )
    assert active_job is not None
    extracted = _playlist_preflight_data([{"source_url": "https://youtu.be/reclaimed123", "title": "Reclaimed"}])
    runner_lease_ids = []

    async def fake_runner(_url, **_kwargs):
        runner_lease_ids.append(active_job["lease_id"])
        return extracted

    class _OwnerMediaDB:
        def get_media_by_urls(self, _urls, **_kwargs):
            return []

        def close_connection(self):
            return None

    monkeypatch.setattr(worker, "run_playlist_preflight_process", fake_runner, raising=True)
    monkeypatch.setattr(worker, "_create_db", lambda _user_id: _OwnerMediaDB(), raising=True)

    with pytest.raises(worker.MediaIngestJobError) as exc_info:
        await worker._handle_job(stale_job, jm, worker._ProgressState())

    assert str(exc_info.value) == "playlist_preflight_lease_lost"
    assert runner_lease_ids == []
    assert store.get_preflight("7", preflight.preflight_id).status == "pending"
    assert list(store.list_preflight_items("7", preflight.preflight_id, limit=10)) == []

    result = await worker._handle_job(active_job, jm, worker._ProgressState())
    assert result["status"] == "ready"
    assert runner_lease_ids == [active_job["lease_id"]]
    assert store.get_preflight("7", preflight.preflight_id).status == "ready"


@pytest.mark.asyncio
async def test_playlist_preflight_worker_marks_owner_library_and_snapshot_duplicates(monkeypatch, tmp_path):
    monkeypatch.setenv("JOBS_DB_PATH", str(tmp_path / "jobs.db"))
    monkeypatch.delenv("JOBS_DB_URL", raising=False)

    from tldw_Server_API.app.core.Jobs.manager import JobManager
    import tldw_Server_API.app.services.media_ingest_jobs_worker as worker

    jm = JobManager()
    store, preflight, job = _seed_playlist_preflight_job(jm)
    extracted = _playlist_preflight_data(
        [
            {"source_url": "https://youtu.be/abc123", "title": "Existing"},
            {"source_url": "https://www.youtube.com/watch?v=abc123", "title": "Repeated"},
            {"source_url": "https://youtu.be/def456", "title": "New"},
            {"title": "Private or deleted"},
        ]
    )
    runner_calls = []

    async def fake_runner(url, **kwargs):
        runner_calls.append((url, kwargs))
        return extracted

    class _OwnerMediaDB:
        def __init__(self):
            self.closed = False

        def get_media_by_urls(self, urls, **_kwargs):
            assert urls == [
                "https://www.youtube.com/watch?v=abc123",
                "https://www.youtube.com/watch?v=def456",
            ]
            return [{"id": 91, "url": "https://www.youtube.com/watch?v=abc123"}]

        def close_connection(self):
            self.closed = True

    media_db = _OwnerMediaDB()
    monkeypatch.setattr(worker, "run_playlist_preflight_process", fake_runner, raising=True)
    monkeypatch.setattr(worker, "_create_db", lambda user_id: media_db if user_id == "7" else None, raising=True)

    result = await worker._handle_job(jm.get_job(int(job["id"])), jm, worker._ProgressState())

    assert runner_calls[0][0] == preflight.source_url
    assert result["status"] == "ready"
    ready = store.get_preflight("7", preflight.preflight_id)
    assert ready.status == "ready"
    assert ready.summary == {
        "loaded_count": 4,
        "ingestible_count": 3,
        "unavailable_count": 1,
        "duplicate_count": 2,
        "selected_count": 1,
        "warnings": [],
    }
    items = list(store.list_preflight_items("7", preflight.preflight_id, limit=10))
    assert [item.duplicate_status for item in items] == [
        "duplicate_existing",
        "duplicate_in_batch",
        "new",
        "unknown",
    ]
    assert items[0].selected_by_default is False
    assert items[1].selected_by_default is False
    assert items[1].duplicate_of_occurrence_id == items[0].occurrence_id
    assert items[2].selected_by_default is True
    assert items[3].availability == "unavailable"
    assert items[3].selected_by_default is False
    assert len({item.occurrence_id for item in items}) == 4
    assert all(item.occurrence_id not in {"abc123", "def456"} for item in items)
    assert media_db.closed is True


@pytest.mark.asyncio
async def test_playlist_preflight_worker_library_lookup_failure_records_unknown_warning(monkeypatch, tmp_path):
    monkeypatch.setenv("JOBS_DB_PATH", str(tmp_path / "jobs.db"))
    monkeypatch.delenv("JOBS_DB_URL", raising=False)

    from tldw_Server_API.app.core.Jobs.manager import JobManager
    import tldw_Server_API.app.services.media_ingest_jobs_worker as worker

    jm = JobManager()
    store, preflight, job = _seed_playlist_preflight_job(jm)
    extracted = _playlist_preflight_data(
        [
            {"source_url": "https://youtu.be/abc123", "title": "Unresolved"},
            {"source_url": "https://youtu.be/abc123", "title": "Known repeat"},
        ]
    )

    async def fake_runner(_url, **_kwargs):
        return extracted

    class _FailingMediaDB:
        def get_media_by_urls(self, _urls, **_kwargs):
            raise RuntimeError("database path /private/secret owner token=do-not-leak")

        def close_connection(self):
            return None

    monkeypatch.setattr(worker, "run_playlist_preflight_process", fake_runner, raising=True)
    monkeypatch.setattr(worker, "_create_db", lambda _user_id: _FailingMediaDB(), raising=True)

    result = await worker._handle_job(jm.get_job(int(job["id"])), jm, worker._ProgressState())

    assert result["status"] == "ready"
    ready = store.get_preflight("7", preflight.preflight_id)
    assert ready.summary["warnings"] == [{"code": "library_lookup_failed"}]
    assert ready.summary["selected_count"] == 0
    items = list(store.list_preflight_items("7", preflight.preflight_id, limit=10))
    assert [item.duplicate_status for item in items] == ["unknown", "duplicate_in_batch"]
    assert all(item.selected_by_default is False for item in items)


@pytest.mark.asyncio
async def test_playlist_preflight_worker_excludes_explicit_unavailable_entries_from_enrichment(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("JOBS_DB_PATH", str(tmp_path / "jobs.db"))
    monkeypatch.delenv("JOBS_DB_URL", raising=False)

    import tldw_Server_API.app.services.media_ingest_jobs_worker as worker
    from tldw_Server_API.app.core.Jobs.manager import JobManager

    jm = JobManager()
    store, preflight, job = _seed_playlist_preflight_job(jm)
    extracted = _playlist_preflight_data(
        [
            {
                "source_url": "https://youtu.be/private123",
                "title": "Private entry with an ID",
                "availability": "private",
            },
            {
                "id": "deleted123",
                "webpage_url": "https://www.youtube.com/watch?v=deleted123",
                "title": "[Deleted video]",
            },
            {"source_url": "https://youtu.be/public123", "title": "Public entry"},
        ]
    )

    async def fake_runner(_url, **_kwargs):
        return extracted

    class _OwnerMediaDB:
        def __init__(self):
            self.lookup_urls = None

        def get_media_by_urls(self, urls, **_kwargs):
            self.lookup_urls = list(urls)
            return []

        def close_connection(self):
            return None

    media_db = _OwnerMediaDB()
    monkeypatch.setattr(worker, "run_playlist_preflight_process", fake_runner, raising=True)
    monkeypatch.setattr(worker, "_create_db", lambda _user_id: media_db, raising=True)

    result = await worker._handle_job(jm.get_job(int(job["id"])), jm, worker._ProgressState())

    assert media_db.lookup_urls == ["https://www.youtube.com/watch?v=public123"]
    assert result["loaded_count"] == 3
    assert result["ingestible_count"] == 1
    assert result["unavailable_count"] == 2
    assert result["duplicate_count"] == 0
    assert result["selected_count"] == 1
    items = list(store.list_preflight_items("7", preflight.preflight_id, limit=10))
    assert [item.ordinal for item in items] == [1, 2, 3]
    assert items[0].availability == "private"
    assert items[0].source_url is None
    assert items[0].duplicate_status == "unknown"
    assert items[0].selected_by_default is False
    assert items[1].availability == "deleted"
    assert items[1].source_url is None
    assert items[1].duplicate_status == "unknown"
    assert items[1].selected_by_default is False
    assert items[2].availability == "available"
    assert items[2].duplicate_status == "new"
    assert items[2].selected_by_default is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error_code",
    ["playlist_too_large", "playlist_preflight_result_too_large"],
)
async def test_playlist_preflight_worker_configured_limit_failure_blocks_without_partial_snapshot(
    monkeypatch,
    tmp_path,
    error_code,
):
    monkeypatch.setenv("JOBS_DB_PATH", str(tmp_path / "jobs.db"))
    monkeypatch.delenv("JOBS_DB_URL", raising=False)

    import tldw_Server_API.app.services.media_ingest_jobs_worker as worker
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Video.playlist_preflight import (
        PlaylistPreflightProcessError,
    )
    from tldw_Server_API.app.core.Jobs.manager import JobManager

    jm = JobManager()
    store, preflight, job = _seed_playlist_preflight_job(jm)

    async def too_large(_url, **_kwargs):
        raise PlaylistPreflightProcessError(error_code)

    monkeypatch.setattr(worker, "run_playlist_preflight_process", too_large, raising=True)

    with pytest.raises(worker.MediaIngestJobError, match=error_code):
        await worker._handle_job(jm.get_job(int(job["id"])), jm, worker._ProgressState())

    blocked = store.get_preflight("7", preflight.preflight_id)
    assert blocked.status == "blocked"
    assert blocked.error == {"code": error_code}
    assert list(store.list_preflight_items("7", preflight.preflight_id, limit=10)) == []


@pytest.mark.asyncio
async def test_playlist_preflight_worker_cancellation_during_library_lookup_never_marks_ready(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("JOBS_DB_PATH", str(tmp_path / "jobs.db"))
    monkeypatch.delenv("JOBS_DB_URL", raising=False)

    import tldw_Server_API.app.services.media_ingest_jobs_worker as worker
    from tldw_Server_API.app.core.Jobs.manager import JobManager

    jm = JobManager()
    store, preflight, job = _seed_playlist_preflight_job(jm)
    extracted = _playlist_preflight_data([{"source_url": "https://youtu.be/race123", "title": "Race"}])

    async def fake_runner(_url, **_kwargs):
        return extracted

    class _CancellingMediaDB:
        def get_media_by_urls(self, _urls, **_kwargs):
            assert jm.cancel_job(int(job["id"]), reason="test cancellation race") is True
            return []

        def close_connection(self):
            return None

    monkeypatch.setattr(worker, "run_playlist_preflight_process", fake_runner, raising=True)
    monkeypatch.setattr(worker, "_create_db", lambda _user_id: _CancellingMediaDB(), raising=True)

    with pytest.raises(worker.MediaIngestJobError) as exc_info:
        await worker._handle_job(job, jm, worker._ProgressState())

    assert str(exc_info.value) == "playlist_preflight_cancelled"
    blocked = store.get_preflight("7", preflight.preflight_id)
    assert blocked.status == "blocked"
    assert blocked.error == {"code": "playlist_preflight_cancelled"}
    assert list(store.list_preflight_items("7", preflight.preflight_id, limit=10)) == []
    assert jm.get_job(int(job["id"]))["status"] == "cancelled"


@pytest.mark.asyncio
async def test_playlist_preflight_atomic_ready_guard_preserves_terminal_cancellation(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("JOBS_DB_PATH", str(tmp_path / "jobs.db"))
    monkeypatch.delenv("JOBS_DB_URL", raising=False)

    import tldw_Server_API.app.services.media_ingest_jobs_worker as worker
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Video.playlist_ingest_store import (
        PlaylistIngestStore,
    )
    from tldw_Server_API.app.core.Jobs.manager import JobManager

    jm = JobManager()
    store, preflight, job = _seed_playlist_preflight_job(jm)
    extracted = _playlist_preflight_data([{"source_url": "https://youtu.be/cancel-ready", "title": "Cancelled"}])

    async def fake_runner(_url, **_kwargs):
        return extracted

    class _OwnerMediaDB:
        def get_media_by_urls(self, _urls, **_kwargs):
            return []

        def close_connection(self):
            return None

    original_replace = PlaylistIngestStore.replace_preflight_snapshot

    def cancel_at_ready(self, owner_user_id, preflight_id, *, status, **kwargs):
        if status == "ready":
            assert jm.cancel_job(int(job["id"]), reason="atomic ready race") is True
        return original_replace(
            self,
            owner_user_id,
            preflight_id,
            status=status,
            **kwargs,
        )

    monkeypatch.setattr(worker, "run_playlist_preflight_process", fake_runner, raising=True)
    monkeypatch.setattr(worker, "_create_db", lambda _user_id: _OwnerMediaDB(), raising=True)
    monkeypatch.setattr(PlaylistIngestStore, "replace_preflight_snapshot", cancel_at_ready, raising=True)

    with pytest.raises(worker.MediaIngestJobError) as exc_info:
        await worker._handle_job(job, jm, worker._ProgressState())

    assert str(exc_info.value) == "playlist_preflight_cancelled"
    blocked = store.get_preflight("7", preflight.preflight_id)
    assert blocked.status == "blocked"
    assert blocked.error == {"code": "playlist_preflight_cancelled"}
    assert list(store.list_preflight_items("7", preflight.preflight_id, limit=10)) == []
    assert jm.get_job(int(job["id"]))["status"] == "cancelled"


@pytest.mark.asyncio
async def test_playlist_preflight_worker_unexpected_extraction_failure_blocks_with_safe_error(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("JOBS_DB_PATH", str(tmp_path / "jobs.db"))
    monkeypatch.delenv("JOBS_DB_URL", raising=False)

    import tldw_Server_API.app.services.media_ingest_jobs_worker as worker
    from tldw_Server_API.app.core.Jobs.manager import JobManager

    jm = JobManager()
    store, preflight, job = _seed_playlist_preflight_job(jm)

    async def unexpected_failure(_url, **_kwargs):
        raise RuntimeError("private URL token=do-not-expose")

    monkeypatch.setattr(worker, "run_playlist_preflight_process", unexpected_failure, raising=True)

    with pytest.raises(worker.MediaIngestJobError) as exc_info:
        await worker._handle_job(jm.get_job(int(job["id"])), jm, worker._ProgressState())

    assert str(exc_info.value) == "playlist_preflight_failed"
    blocked = store.get_preflight("7", preflight.preflight_id)
    assert blocked.status == "blocked"
    assert blocked.error == {"code": "playlist_preflight_failed"}
    assert "do-not-expose" not in str(exc_info.value)
