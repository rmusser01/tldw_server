from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any

import pytest

import tldw_Server_API.app.core.Metrics.metrics_manager as metrics_manager
from tldw_Server_API.app.core.Ingestion_Media_Processing import (
    persistence as ingestion_persistence,
)


pytestmark = pytest.mark.unit


class _MetricsCapture:
    def __init__(self) -> None:
        self.increment_calls: list[tuple[str, float, dict[str, Any] | None]] = []

    def increment(
        self,
        metric_name: str,
        value: float = 1,
        labels: dict[str, Any] | None = None,
    ) -> None:
        self.increment_calls.append((metric_name, value, labels))

    def observe(self, *_args: Any, **_kwargs: Any) -> None:
        return None


class _RepoBackedWorkerDB:
    instances: list["_RepoBackedWorkerDB"] = []

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.backend = "sqlite"
        self.closed = False
        self.upsert_calls: list[dict[str, Any]] = []
        type(self).instances.append(self)

    def close_connection(self) -> None:
        self.closed = True

    def upsert_email_message_graph(self, **kwargs: Any) -> dict[str, Any]:
        self.upsert_calls.append(kwargs)
        return {"email_message_id": 901}


class _FakeMediaRepository:
    calls: list[dict[str, Any]] = []

    def add_media_with_keywords(self, **kwargs: Any) -> tuple[int, str, str]:
        type(self).calls.append(kwargs)
        idx = len(type(self).calls)
        return idx, f"uuid-{idx}", f"Media '{kwargs.get('title')}' added."


def _fake_create_media_database(*_args: Any, **_kwargs: Any) -> _RepoBackedWorkerDB:
    return _RepoBackedWorkerDB()


class _ChunkCountDb:
    def __init__(self, count: int) -> None:
        self.count = count
        self.closed = False

    def get_unvectorized_chunk_count(self, media_id: int) -> int:
        assert media_id > 0
        return self.count

    def close_connection(self) -> None:
        self.closed = True


class _HelperSessionDb:
    def __init__(self) -> None:
        self.closed = False

    def close_connection(self) -> None:
        self.closed = True


def test_with_media_db_session_closes_connection_on_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub_db = _HelperSessionDb()

    monkeypatch.setattr(
        ingestion_persistence,
        "create_media_database",
        lambda client_id, *, db_path=None, **_kwargs: stub_db,
    )

    with pytest.raises(RuntimeError, match="boom"):
        ingestion_persistence._with_media_db_session(
            db_path=":memory:",
            client_id="test-client",
            operation=lambda _db: (_ for _ in ()).throw(RuntimeError("boom")),
        )

    assert stub_db.closed is True


@pytest.mark.asyncio
async def test_chunk_consistency_warn_policy_adds_warning_and_metric(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metrics = _MetricsCapture()
    monkeypatch.setattr(ingestion_persistence, "get_metrics_registry", lambda: metrics)
    monkeypatch.setenv("MEDIA_CHUNK_CONSISTENCY_POLICY", "warn")

    async def _fake_fetch_count(**_kwargs: Any) -> int:
        return 2

    monkeypatch.setattr(
        ingestion_persistence,
        "_fetch_unvectorized_chunk_count",
        _fake_fetch_count,
    )

    result = {
        "status": "Success",
        "warnings": None,
        "error": None,
        "db_id": 11,
        "db_message": "Media 'clip' added.",
        "media_uuid": "uuid-11",
    }
    await ingestion_persistence._enforce_chunk_consistency_after_persist(
        result=result,
        form_data=SimpleNamespace(chunk_consistency_policy=None),
        media_type="audio",
        path_kind="upload",
        processor="audio_primary_persist",
        expected_chunk_count=3,
        db_message=result["db_message"],
        media_id=result["db_id"],
        db_path=":memory:",
        client_id="test-client",
        loop=asyncio.get_running_loop(),
    )

    assert result["status"] == "Success"
    warnings = result.get("warnings") or []
    assert any("Chunk consistency warning" in msg for msg in warnings)
    assert (
        "ingestion_validation_failures_total",
        1,
        {"reason": "chunk_consistency", "path_kind": "upload"},
    ) in metrics.increment_calls


@pytest.mark.asyncio
async def test_fetch_unvectorized_chunk_count_uses_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub_db = _ChunkCountDb(4)

    monkeypatch.setattr(
        ingestion_persistence,
        "create_media_database",
        lambda client_id, *, db_path=None, **_kwargs: stub_db,
    )

    count = await ingestion_persistence._fetch_unvectorized_chunk_count(
        db_path=":memory:",
        client_id="test-client",
        media_id=11,
        loop=asyncio.get_running_loop(),
    )

    assert count == 4
    assert stub_db.closed is True


@pytest.mark.asyncio
async def test_chunk_consistency_error_policy_marks_error_but_preserves_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MEDIA_CHUNK_CONSISTENCY_POLICY", "error")

    async def _fake_fetch_count(**_kwargs: Any) -> int:
        return 1

    monkeypatch.setattr(
        ingestion_persistence,
        "_fetch_unvectorized_chunk_count",
        _fake_fetch_count,
    )

    result = {
        "status": "Success",
        "warnings": None,
        "error": None,
        "db_id": 22,
        "db_message": "Media 'doc' added.",
        "media_uuid": "uuid-22",
    }
    await ingestion_persistence._enforce_chunk_consistency_after_persist(
        result=result,
        form_data=SimpleNamespace(chunk_consistency_policy=None),
        media_type="document",
        path_kind="url",
        processor="document_persist",
        expected_chunk_count=4,
        db_message=result["db_message"],
        media_id=result["db_id"],
        db_path=":memory:",
        client_id="test-client",
        loop=asyncio.get_running_loop(),
    )

    assert result["status"] == "Error"
    assert "Chunk consistency validation failed" in str(result.get("error", ""))
    assert "Chunk consistency validation failed" in str(result.get("db_message", ""))
    assert result["db_id"] == 22
    assert result["media_uuid"] == "uuid-22"


@pytest.mark.asyncio
async def test_chunk_consistency_skips_non_persisting_messages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MEDIA_CHUNK_CONSISTENCY_POLICY", "warn")

    async def _should_not_run(**_kwargs: Any) -> int:
        raise AssertionError("chunk count lookup should be skipped")

    monkeypatch.setattr(
        ingestion_persistence,
        "_fetch_unvectorized_chunk_count",
        _should_not_run,
    )

    result = {
        "status": "Success",
        "warnings": None,
        "error": None,
        "db_id": 3,
        "db_message": "Media 'x' already exists. Overwrite not enabled.",
        "media_uuid": "uuid-3",
    }
    await ingestion_persistence._enforce_chunk_consistency_after_persist(
        result=result,
        form_data=SimpleNamespace(chunk_consistency_policy=None),
        media_type="audio",
        path_kind="upload",
        processor="audio_primary_persist",
        expected_chunk_count=5,
        db_message=result["db_message"],
        media_id=result["db_id"],
        db_path=":memory:",
        client_id="test-client",
        loop=asyncio.get_running_loop(),
    )

    assert result["status"] == "Success"
    assert result.get("warnings") is None


@pytest.mark.asyncio
async def test_persist_primary_av_item_invokes_chunk_consistency_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int | None] = []
    _RepoBackedWorkerDB.instances = []
    _FakeMediaRepository.calls = []

    async def _fake_enforce(**kwargs: Any) -> None:
        calls.append(kwargs.get("expected_chunk_count"))

    async def _fake_persist_claims(**_kwargs: Any) -> None:
        return None

    monkeypatch.setattr(
        ingestion_persistence,
        "_enforce_chunk_consistency_after_persist",
        _fake_enforce,
    )
    monkeypatch.setattr(ingestion_persistence, "persist_claims_if_applicable", _fake_persist_claims)
    monkeypatch.setattr(
        ingestion_persistence,
        "create_media_database",
        _fake_create_media_database,
    )
    monkeypatch.setattr(
        ingestion_persistence,
        "get_media_repository",
        lambda db: _FakeMediaRepository(),
        raising=False,
    )

    process_result = {
        "status": "Success",
        "input_ref": "clip.mp3",
        "processing_source": "clip.mp3",
        "metadata": {},
        "content": "hello world. second sentence.",
        "transcript": "hello world. second sentence.",
        "summary": None,
        "analysis": None,
        "analysis_details": {},
        "warnings": None,
        "error": None,
    }

    await ingestion_persistence.persist_primary_av_item(
        process_result=process_result,
        form_data=SimpleNamespace(
            keywords=[],
            custom_prompt=None,
            overwrite_existing=True,
            transcription_model=None,
            chunk_consistency_policy="warn",
        ),
        media_type="audio",
        original_input_ref="clip.mp3",
        chunk_options={"method": "sentences", "max_size": 500, "overlap": 0},
        path_kind="upload",
        db_path=":memory:",
        client_id="test-client",
        loop=asyncio.get_running_loop(),
        claims_context=None,
    )

    assert process_result.get("db_id") == 1
    assert len(calls) == 1
    assert calls[0] == 1
    assert len(_FakeMediaRepository.calls) == 1
    assert _FakeMediaRepository.calls[0]["title"] == "clip"
    assert _FakeMediaRepository.calls[0]["media_type"] == "audio"
    assert _RepoBackedWorkerDB.instances
    assert _RepoBackedWorkerDB.instances[0].closed is True


@pytest.mark.asyncio
async def test_persist_primary_av_item_upserts_normalized_transcript_via_extracted_helper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metrics_manager._metrics_registry = None
    registry = metrics_manager.get_metrics_registry()
    _RepoBackedWorkerDB.instances = []
    _FakeMediaRepository.calls = []
    transcript_calls: list[dict[str, Any]] = []

    async def _fake_enforce(**_kwargs: Any) -> None:
        return None

    async def _fake_persist_claims(**_kwargs: Any) -> None:
        return None

    monkeypatch.setattr(
        ingestion_persistence,
        "_enforce_chunk_consistency_after_persist",
        _fake_enforce,
    )
    monkeypatch.setattr(ingestion_persistence, "persist_claims_if_applicable", _fake_persist_claims)
    monkeypatch.setattr(
        ingestion_persistence,
        "create_media_database",
        _fake_create_media_database,
    )
    monkeypatch.setattr(
        ingestion_persistence,
        "get_media_repository",
        lambda db: _FakeMediaRepository(),
        raising=False,
    )

    from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio import (
        Audio_Transcription_Lib,
        stt_provider_adapter,
    )

    monkeypatch.setattr(
        stt_provider_adapter,
        "get_stt_provider_registry",
        lambda: SimpleNamespace(
            resolve_provider_for_model=lambda _model: ("fake-provider", "resolved-model", None)
        ),
    )
    monkeypatch.setattr(
        Audio_Transcription_Lib,
        "to_normalized_stt_artifact",
        lambda text, segments, language, provider, model: {
            "text": text,
            "segments": segments,
            "language": language,
            "metadata": {"provider": provider, "model": model},
        },
    )

    def _fake_upsert_transcript(**kwargs: Any) -> dict[str, Any]:
        transcript_calls.append(kwargs)
        return {"id": len(transcript_calls)}

    monkeypatch.setattr(
        ingestion_persistence,
        "upsert_transcript",
        _fake_upsert_transcript,
        raising=False,
    )

    process_result = {
        "status": "Success",
        "input_ref": "clip.mp3",
        "processing_source": "clip.mp3",
        "metadata": {"model": "base-model"},
        "content": "hello world",
        "transcript": "hello world",
        "segments": [{"text": "hello world", "start": 0.0, "end": 1.0}],
        "summary": None,
        "analysis": None,
        "analysis_details": {"transcription_language": "en"},
        "warnings": None,
        "error": None,
    }

    await ingestion_persistence.persist_primary_av_item(
        process_result=process_result,
        form_data=SimpleNamespace(
            keywords=[],
            custom_prompt=None,
            overwrite_existing=True,
            transcription_model=None,
            chunk_consistency_policy="warn",
        ),
        media_type="audio",
        original_input_ref="clip.mp3",
        chunk_options=None,
        path_kind="upload",
        db_path=":memory:",
        client_id="test-client",
        loop=asyncio.get_running_loop(),
        claims_context=None,
    )

    try:
        assert len(transcript_calls) == 1
        assert transcript_calls[0]["media_id"] == 1
        assert transcript_calls[0]["whisper_model"] == "resolved-model"
        assert transcript_calls[0].get("idempotency_key") is None
        assert json.loads(transcript_calls[0]["transcription"]) == {
            "text": "hello world",
            "segments": [{"text": "hello world", "start": 0.0, "end": 1.0}],
            "language": "en",
            "metadata": {"provider": "fake-provider", "model": "resolved-model"},
        }
        assert registry.get_cumulative_counter(
            "audio_stt_run_writes_total",
            {"provider": "other", "write_result": "created"},
        ) == 1
        assert process_result["normalized_stt"]["metadata"]["model"] == "resolved-model"
        assert len(_RepoBackedWorkerDB.instances) == 2
        assert all(instance.closed for instance in _RepoBackedWorkerDB.instances)
    finally:
        metrics_manager._metrics_registry = None


@pytest.mark.asyncio
async def test_persist_primary_av_item_recomputes_chunks_when_requested_but_chunk_options_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int | None] = []
    _RepoBackedWorkerDB.instances = []
    _FakeMediaRepository.calls = []

    async def _fake_enforce(**kwargs: Any) -> None:
        calls.append(kwargs.get("expected_chunk_count"))

    async def _fake_persist_claims(**_kwargs: Any) -> None:
        return None

    monkeypatch.setattr(
        ingestion_persistence,
        "_enforce_chunk_consistency_after_persist",
        _fake_enforce,
    )
    monkeypatch.setattr(
        ingestion_persistence,
        "persist_claims_if_applicable",
        _fake_persist_claims,
    )
    monkeypatch.setattr(
        ingestion_persistence,
        "create_media_database",
        _fake_create_media_database,
    )
    monkeypatch.setattr(
        ingestion_persistence,
        "get_media_repository",
        lambda db: _FakeMediaRepository(),
        raising=False,
    )

    process_result = {
        "status": "Success",
        "input_ref": "clip.mp3",
        "processing_source": "clip.mp3",
        "metadata": {"model": "parakeet-mlx"},
        "content": "hello world. second sentence.",
        "transcript": "hello world. second sentence.",
        "summary": None,
        "analysis": None,
        "analysis_details": {},
        "warnings": None,
        "error": None,
    }

    await ingestion_persistence.persist_primary_av_item(
        process_result=process_result,
        form_data=SimpleNamespace(
            keywords=[],
            custom_prompt=None,
            overwrite_existing=True,
            transcription_model="parakeet-mlx",
            chunk_consistency_policy="warn",
            perform_chunking=True,
            media_type="audio",
            transcription_language="en",
        ),
        media_type="audio",
        original_input_ref="clip.mp3",
        chunk_options=None,
        path_kind="upload",
        db_path=":memory:",
        client_id="test-client",
        loop=asyncio.get_running_loop(),
        claims_context=None,
    )

    assert process_result.get("db_id") == 1
    assert len(_FakeMediaRepository.calls) == 1
    persisted_chunks = _FakeMediaRepository.calls[0]["chunks"]
    assert isinstance(persisted_chunks, list)
    assert len(persisted_chunks) == 1
    assert calls == [1]


@pytest.mark.asyncio
async def test_persist_doc_item_and_children_routes_email_parent_and_child_through_media_repository(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _RepoBackedWorkerDB.instances = []
    _FakeMediaRepository.calls = []

    async def _fake_enforce(**_kwargs: Any) -> None:
        return None

    async def _fake_persist_claims(**_kwargs: Any) -> None:
        return None

    monkeypatch.setattr(
        ingestion_persistence,
        "_enforce_chunk_consistency_after_persist",
        _fake_enforce,
    )
    monkeypatch.setattr(ingestion_persistence, "persist_claims_if_applicable", _fake_persist_claims)
    monkeypatch.setattr(ingestion_persistence, "_is_email_native_persist_enabled", lambda: True)
    monkeypatch.setattr(
        ingestion_persistence,
        "create_media_database",
        _fake_create_media_database,
    )
    monkeypatch.setattr(
        ingestion_persistence,
        "get_media_repository",
        lambda db: _FakeMediaRepository(),
        raising=False,
    )

    final_result = {
        "status": "Success",
        "content": "Parent email body",
        "summary": None,
        "analysis": None,
        "metadata": {"email": {"message_id": "msg-1"}, "author": "Alice"},
        "children": [
            {
                "status": "Success",
                "content": "Attachment body",
                "metadata": {"title": "Attachment A", "filename": "a.txt"},
            }
        ],
        "warnings": None,
        "error": None,
    }

    await ingestion_persistence.persist_doc_item_and_children(
        final_result=final_result,
        form_data=SimpleNamespace(
            keywords=[],
            custom_prompt=None,
            overwrite_existing=False,
            title=None,
            author=None,
            ingest_attachments=True,
            accept_archives=False,
            accept_mbox=False,
            accept_pst=False,
        ),
        media_type="email",
        item_input_ref="mail.eml",
        processing_filename=None,
        chunk_options=None,
        path_kind="upload",
        db_path=":memory:",
        client_id="test-client",
        loop=asyncio.get_running_loop(),
        claims_context=None,
    )

    assert final_result["db_id"] == 1
    assert final_result["email_message_id"] == 901
    assert len(_FakeMediaRepository.calls) == 2
    assert _FakeMediaRepository.calls[0]["title"] == "mail"
    assert _FakeMediaRepository.calls[0]["media_type"] == "email"
    assert _FakeMediaRepository.calls[1]["title"] == "Attachment A"
    assert "child_db_results" in final_result
    assert len(final_result["child_db_results"]) == 1
    assert len(_RepoBackedWorkerDB.instances) == 2
    assert _RepoBackedWorkerDB.instances[0].upsert_calls
    assert _RepoBackedWorkerDB.instances[1].upsert_calls
    assert all(db.closed for db in _RepoBackedWorkerDB.instances)


@pytest.mark.asyncio
async def test_persist_doc_item_and_children_routes_archive_children_through_media_repository(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _RepoBackedWorkerDB.instances = []
    _FakeMediaRepository.calls = []

    async def _fake_enforce(**_kwargs: Any) -> None:
        return None

    async def _fake_persist_claims(**_kwargs: Any) -> None:
        return None

    monkeypatch.setattr(
        ingestion_persistence,
        "_enforce_chunk_consistency_after_persist",
        _fake_enforce,
    )
    monkeypatch.setattr(ingestion_persistence, "persist_claims_if_applicable", _fake_persist_claims)
    monkeypatch.setattr(ingestion_persistence, "_is_email_native_persist_enabled", lambda: True)
    monkeypatch.setattr(
        ingestion_persistence,
        "create_media_database",
        _fake_create_media_database,
    )
    monkeypatch.setattr(
        ingestion_persistence,
        "get_media_repository",
        lambda db: _FakeMediaRepository(),
        raising=False,
    )

    final_result = {
        "status": "Success",
        "content": "",
        "summary": None,
        "analysis": None,
        "metadata": {},
        "children": [
            {
                "status": "Success",
                "content": "Archived email body",
                "metadata": {"title": "Archive Child", "filename": "child.eml"},
            }
        ],
        "warnings": None,
        "error": None,
    }

    await ingestion_persistence.persist_doc_item_and_children(
        final_result=final_result,
        form_data=SimpleNamespace(
            keywords=[],
            custom_prompt=None,
            overwrite_existing=False,
            title=None,
            author=None,
            ingest_attachments=False,
            accept_archives=True,
            accept_mbox=False,
            accept_pst=False,
        ),
        media_type="email",
        item_input_ref="bundle.zip",
        processing_filename="bundle.zip",
        chunk_options=None,
        path_kind="upload",
        db_path=":memory:",
        client_id="test-client",
        loop=asyncio.get_running_loop(),
        claims_context=None,
    )

    assert final_result["db_message"] == "Persisted archive children."
    assert len(_FakeMediaRepository.calls) == 1
    assert _FakeMediaRepository.calls[0]["title"] == "Archive Child"
    assert _FakeMediaRepository.calls[0]["media_type"] == "email"
    assert len(final_result["child_db_results"]) == 1
    assert len(_RepoBackedWorkerDB.instances) == 1
    assert _RepoBackedWorkerDB.instances[0].upsert_calls
    assert _RepoBackedWorkerDB.instances[0].closed is True
