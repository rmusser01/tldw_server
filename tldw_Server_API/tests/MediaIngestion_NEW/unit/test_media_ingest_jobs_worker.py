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
