import asyncio
import json
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.config import settings
from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase
from tldw_Server_API.app.core.TTS import tts_jobs_worker
from tldw_Server_API.app.core.TTS.tts_exceptions import (
    TTSNetworkError,
    TTSRateLimitError,
)
from tldw_Server_API.app.core.TTS.tts_jobs_worker import _handle_tts_job


@pytest.fixture(autouse=True)
def _resolve_owner_credentials(monkeypatch):
    """Keep worker output/history units focused on behavior after resolution."""

    async def _resolve_tts_byok(*, current_user, **_kwargs):
        return int(current_user.id), None, None

    monkeypatch.setattr(tts_jobs_worker, "_resolve_tts_byok", _resolve_tts_byok)


def _failure_job(*, job_id: int, text: str) -> dict[str, object]:
    return {
        "id": job_id,
        "job_type": "tts_longform",
        "owner_user_id": "1",
        "payload": {
            "speech_request": {
                "model": "tts-1",
                "input": text,
                "voice": "alloy",
                "response_format": "mp3",
                "stream": False,
            },
            "provider_hint": "openai",
        },
    }


def _patch_failure_runtime(monkeypatch, service) -> None:
    async def _get_service():
        return service

    monkeypatch.setattr(tts_jobs_worker, "get_tts_service_v2", _get_service)
    monkeypatch.setattr(
        tts_jobs_worker,
        "JobManager",
        lambda: SimpleNamespace(
            renew_job_lease=lambda *_args, **_kwargs: True,
            update_job_progress=lambda *_args, **_kwargs: True,
        ),
    )
    monkeypatch.setattr(tts_jobs_worker, "emit_job_event", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(settings, "TTS_HISTORY_ENABLED", False, raising=False)


@pytest.mark.unit
async def test_tts_jobs_worker_provider_failure_after_dispatch_is_terminal(
    monkeypatch,
) -> None:
    """An ambiguous provider failure cannot automatically replay synthesis."""

    sentinel = "provider-dispatch-secret-sentinel"
    dispatches = 0

    class _Service:
        def generate_speech(self, *_args, **_kwargs):
            nonlocal dispatches
            dispatches += 1
            raise TTSRateLimitError(
                sentinel,
                provider="openai",
                details={"retry_after": 37},
            )

    _patch_failure_runtime(monkeypatch, _Service())

    with pytest.raises(tts_jobs_worker.TTSJobError) as exc_info:
        await _handle_tts_job(
            _failure_job(job_id=61, text="provider dispatch failure")
        )

    assert dispatches == 1
    assert exc_info.value.retryable is False
    assert exc_info.value.failure_code == "provider_unavailable"
    assert str(exc_info.value) == "TTS provider request failed"
    assert not hasattr(exc_info.value, "backoff_seconds")
    assert sentinel not in repr(exc_info.value)


@pytest.mark.unit
async def test_tts_jobs_worker_partial_stream_failure_is_terminal(monkeypatch) -> None:
    """A failure after the first audio chunk cannot replay a partial synthesis."""

    sentinel = "partial-stream-secret-sentinel"
    dispatches = 0
    first_chunk = asyncio.Event()

    class _Service:
        def generate_speech(self, *_args, **_kwargs):
            nonlocal dispatches
            dispatches += 1

            async def _chunks():
                first_chunk.set()
                yield b"partial-audio"
                raise TTSNetworkError(sentinel, provider="openai")

            return _chunks()

    _patch_failure_runtime(monkeypatch, _Service())

    with pytest.raises(tts_jobs_worker.TTSJobError) as exc_info:
        await _handle_tts_job(
            _failure_job(job_id=62, text="partial stream failure")
        )

    assert first_chunk.is_set()
    assert dispatches == 1
    assert exc_info.value.retryable is False
    assert exc_info.value.failure_code == "provider_unavailable"
    assert str(exc_info.value) == "TTS provider request failed"
    assert sentinel not in repr(exc_info.value)


@pytest.mark.unit
async def test_tts_jobs_worker_disables_in_handler_replay_without_mutating_payload(
    monkeypatch,
) -> None:
    """Long-form jobs permit one provider dispatch across retry/fallback controls."""

    sentinel = "in-handler-replay-secret-sentinel"
    dispatches = 0

    class _Service:
        def generate_speech(self, request, *, fallback, **_kwargs):
            async def _chunks():
                nonlocal dispatches
                attempts = int((request.extra_params or {}).get("segment_retry_max", 2))
                for _ in range(attempts):
                    dispatches += 1
                if fallback:
                    dispatches += 1
                raise TTSNetworkError(sentinel, provider="openai")
                yield b"unreachable"

            return _chunks()

    _patch_failure_runtime(monkeypatch, _Service())
    job = _failure_job(job_id=64, text="disable in-handler replay")
    speech_request = job["payload"]["speech_request"]
    speech_request["stream"] = True
    speech_request["extra_params"] = {
        "segment_retry_max": 4,
        "persisted_marker": "unchanged",
    }
    original_job = json.loads(json.dumps(job))

    with pytest.raises(tts_jobs_worker.TTSJobError) as exc_info:
        await _handle_tts_job(job)

    assert dispatches == 1
    assert exc_info.value.retryable is False
    assert job == original_job


@pytest.mark.unit
@pytest.mark.parametrize("failure_stage", ["file_write", "artifact_write"])
async def test_tts_jobs_worker_output_failure_after_synthesis_is_terminal(
    monkeypatch,
    tmp_path,
    failure_stage: str,
) -> None:
    """Completed synthesis is never replayed when durable output persistence fails."""

    sentinel = f"{failure_stage}-secret-sentinel"
    synthesis_completed = asyncio.Event()
    dispatches = 0

    class _Service:
        def generate_speech(self, *_args, **_kwargs):
            nonlocal dispatches
            dispatches += 1

            async def _chunks():
                yield b"completed-audio"
                synthesis_completed.set()

            return _chunks()

    class _Collections:
        def resolve_output_storage_path(self, name):
            return name

        def create_output_artifact(self, **_kwargs):
            if failure_stage == "artifact_write":
                raise RuntimeError(sentinel)
            pytest.fail("Artifact registration must not run after a file-write failure")

        def __enter__(self):
            return self

        def __exit__(self, _exc_type, _exc, _tb):
            return False

    _patch_failure_runtime(monkeypatch, _Service())
    monkeypatch.setattr(
        tts_jobs_worker.DatabasePaths,
        "get_user_outputs_dir",
        lambda _user_id: tmp_path,
    )
    monkeypatch.setattr(
        tts_jobs_worker.CollectionsDatabase,
        "for_user",
        lambda user_id: _Collections(),
    )
    if failure_stage == "file_write":
        def _fail_write(_path, _data):
            raise OSError(sentinel)

        monkeypatch.setattr(
            type(tmp_path),
            "write_bytes",
            _fail_write,
        )

    with pytest.raises(tts_jobs_worker.TTSJobError) as exc_info:
        await _handle_tts_job(
            _failure_job(job_id=63, text=f"{failure_stage} failure")
        )

    assert synthesis_completed.is_set()
    assert dispatches == 1
    assert exc_info.value.retryable is False
    assert exc_info.value.failure_code == "tts_output_persistence_failed"
    assert str(exc_info.value) == "write_failed"
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    assert sentinel not in repr(exc_info.value)


@pytest.mark.unit
async def test_standalone_worker_waits_for_override_bootstrap(monkeypatch):
    """Standalone dispatch cannot start before the first policy load succeeds."""
    refresh_started = asyncio.Event()
    allow_refresh = asyncio.Event()
    run_started = asyncio.Event()
    lifecycle_calls: list[str] = []

    async def _refresh(*, force=False):
        assert force is True
        lifecycle_calls.append("refresh")
        refresh_started.set()
        await allow_refresh.wait()

    def _start() -> None:
        lifecycle_calls.append("start")

    async def _run() -> None:
        lifecycle_calls.append("run")
        run_started.set()

    async def _shutdown() -> None:
        lifecycle_calls.append("shutdown")

    monkeypatch.setattr(tts_jobs_worker, "refresh_llm_provider_overrides", _refresh)
    monkeypatch.setattr(
        tts_jobs_worker,
        "start_llm_provider_override_refresh_service",
        _start,
    )
    monkeypatch.setattr(tts_jobs_worker, "run_tts_jobs_worker", _run)
    monkeypatch.setattr(
        tts_jobs_worker,
        "shutdown_llm_provider_override_recovery",
        _shutdown,
    )

    worker_task = asyncio.create_task(tts_jobs_worker.main())
    await refresh_started.wait()
    assert not run_started.is_set()
    allow_refresh.set()
    await worker_task

    assert lifecycle_calls == ["refresh", "start", "run", "shutdown"]


@pytest.mark.unit
def test_open_media_db_for_history_uses_media_db_api_factory(tmp_path, monkeypatch):
    captured = {}
    sentinel = object()
    db_path = tmp_path / "tts-history.sqlite3"

    monkeypatch.setattr(
        tts_jobs_worker.DatabasePaths,
        "get_media_db_path",
        lambda user_id: db_path,
    )

    def _fake_create_media_database(client_id, **kwargs):
        captured["client_id"] = client_id
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(tts_jobs_worker, "create_media_database", _fake_create_media_database)

    result = tts_jobs_worker._open_media_db_for_history("17")

    assert result is sentinel
    assert captured == {
        "client_id": "tts_jobs_worker",
        "db_path": str(db_path),
    }


@pytest.mark.unit
async def test_tts_jobs_worker_writes_output(tmp_path, monkeypatch):
    progress_calls = []

    class DummyJM:
        def renew_job_lease(self, *_args, **_kwargs):
            return True

        def update_job_progress(self, job_id, *, progress_percent=None, progress_message=None):
            progress_calls.append((job_id, progress_percent, progress_message))
            return True

    class DummyService:
        def generate_speech(self, *args, **kwargs):
            async def _gen():
                yield b"\x00\x01"
                yield b"\x02\x03"
            return _gen()

    async def _get_service():
        return DummyService()

    monkeypatch.setattr(
        "tldw_Server_API.app.core.TTS.tts_jobs_worker.get_tts_service_v2",
        _get_service,
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.TTS.tts_jobs_worker.JobManager",
        lambda: DummyJM(),
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.TTS.tts_jobs_worker.emit_job_event",
        lambda *args, **kwargs: None,
    )

    monkeypatch.setattr(
        "tldw_Server_API.app.core.TTS.tts_jobs_worker.DatabasePaths.get_user_outputs_dir",
        lambda user_id: tmp_path,
    )

    class DummyCDB:
        def resolve_output_storage_path(self, name):
            return name

        def create_output_artifact(self, **kwargs):
            return SimpleNamespace(
                id=123,
                storage_path=kwargs.get("storage_path"),
                format=kwargs.get("format_"),
            )

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(
        "tldw_Server_API.app.core.TTS.tts_jobs_worker.CollectionsDatabase.for_user",
        lambda user_id: DummyCDB(),
    )

    job = {
        "id": 55,
        "job_type": "tts_longform",
        "owner_user_id": "1",
        "payload": {
            "user_id": "1",
            "speech_request": {
                "model": "kokoro",
                "input": "hello",
                "voice": "af_heart",
                "response_format": "mp3",
                "stream": False,
            },
        },
    }

    result = await _handle_tts_job(job)
    assert result["output_id"] == 123
    assert (tmp_path / "tts_job_55.mp3").exists()
    assert progress_calls


@pytest.mark.unit
async def test_tts_jobs_worker_writes_history_with_artifact_ids(tmp_path, monkeypatch):
    class DummyService:
        def generate_speech(self, *args, **kwargs):
            async def _gen():
                yield b"\x10\x11"
            return _gen()

    async def _get_service():
        return DummyService()

    monkeypatch.setattr(
        "tldw_Server_API.app.core.TTS.tts_jobs_worker.get_tts_service_v2",
        _get_service,
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.TTS.tts_jobs_worker.JobManager",
        lambda: SimpleNamespace(
            renew_job_lease=lambda *args, **kwargs: True,
            update_job_progress=lambda *args, **kwargs: True,
        ),
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.TTS.tts_jobs_worker.emit_job_event",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.TTS.tts_jobs_worker.DatabasePaths.get_user_outputs_dir",
        lambda user_id: tmp_path,
    )

    db_path = tmp_path / "Media_DB_v2.db"
    monkeypatch.setattr(
        "tldw_Server_API.app.core.TTS.tts_jobs_worker.DatabasePaths.get_media_db_path",
        lambda user_id: db_path,
    )

    class DummyCDB:
        def resolve_output_storage_path(self, name):
            return name

        def create_output_artifact(self, **kwargs):
            return SimpleNamespace(
                id=987,
                storage_path=kwargs.get("storage_path"),
                format=kwargs.get("format_"),
            )

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(
        "tldw_Server_API.app.core.TTS.tts_jobs_worker.CollectionsDatabase.for_user",
        lambda user_id: DummyCDB(),
    )
    monkeypatch.setattr(settings, "TTS_HISTORY_ENABLED", True, raising=False)
    monkeypatch.setattr(settings, "TTS_HISTORY_STORE_TEXT", True, raising=False)
    monkeypatch.setattr(settings, "TTS_HISTORY_STORE_FAILED", True, raising=False)
    monkeypatch.setattr(settings, "TTS_HISTORY_HASH_KEY", "unit-stage2-history-key", raising=False)

    job = {
        "id": 56,
        "job_type": "tts_longform",
        "request_id": "job-req-stage2",
        "owner_user_id": "1",
        "payload": {
            "user_id": "1",
            "speech_request": {
                "model": "kokoro",
                "input": "artifact id history test",
                "voice": "af_heart",
                "response_format": "mp3",
                "stream": False,
            },
        },
    }

    result = await _handle_tts_job(job)
    assert result["output_id"] == 987

    media_db = MediaDatabase(db_path=str(db_path), client_id="tts_jobs_worker_history_assert")
    try:
        row = media_db.execute_query(
            "SELECT job_id, output_id, artifact_ids FROM tts_history WHERE user_id = ? ORDER BY id DESC LIMIT 1",
            ("1",),
        ).fetchone()
    finally:
        media_db.close_connection()

    assert row is not None
    assert int(row["job_id"]) == 56
    assert int(row["output_id"]) == 987
    artifact_ids = json.loads(row["artifact_ids"])
    assert artifact_ids == ["output:987"]


@pytest.mark.unit
async def test_tts_jobs_worker_history_write_failure_logs_job_and_request_id(tmp_path, monkeypatch):
    class DummyService:
        def generate_speech(self, *args, **kwargs):
            async def _gen():
                yield b"\xAA\xBB"
            return _gen()

    async def _get_service():
        return DummyService()

    monkeypatch.setattr(
        "tldw_Server_API.app.core.TTS.tts_jobs_worker.get_tts_service_v2",
        _get_service,
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.TTS.tts_jobs_worker.JobManager",
        lambda: SimpleNamespace(
            renew_job_lease=lambda *args, **kwargs: True,
            update_job_progress=lambda *args, **kwargs: True,
        ),
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.TTS.tts_jobs_worker.emit_job_event",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.TTS.tts_jobs_worker.DatabasePaths.get_user_outputs_dir",
        lambda user_id: tmp_path,
    )

    class DummyCDB:
        def resolve_output_storage_path(self, name):
            return name

        def create_output_artifact(self, **kwargs):
            return SimpleNamespace(
                id=654,
                storage_path=kwargs.get("storage_path"),
                format=kwargs.get("format_"),
            )

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(
        "tldw_Server_API.app.core.TTS.tts_jobs_worker.CollectionsDatabase.for_user",
        lambda user_id: DummyCDB(),
    )

    class FailingHistoryDB:
        def create_tts_history_entry(self, **kwargs):
            raise RuntimeError("history insert failed")

        def close_connection(self):
            return None

    monkeypatch.setattr(
        "tldw_Server_API.app.core.TTS.tts_jobs_worker._open_media_db_for_history",
        lambda user_id: FailingHistoryDB(),
    )
    monkeypatch.setattr(settings, "TTS_HISTORY_ENABLED", True, raising=False)
    monkeypatch.setattr(settings, "TTS_HISTORY_STORE_TEXT", True, raising=False)
    monkeypatch.setattr(settings, "TTS_HISTORY_STORE_FAILED", True, raising=False)
    monkeypatch.setattr(settings, "TTS_HISTORY_HASH_KEY", "unit-stage2-log-key", raising=False)

    debug_lines: list[str] = []
    bound_contexts: list[dict[str, str]] = []

    def _capture_debug(message, *args, **kwargs):
        try:
            rendered = str(message).format(*args)
        except Exception:
            rendered = f"{message} {args}"
        debug_lines.append(rendered)

    def _capture_bind(**context):
        bound_contexts.append(context)
        return SimpleNamespace(debug=_capture_debug)

    monkeypatch.setattr(
        "tldw_Server_API.app.core.TTS.tts_jobs_worker.logger.bind",
        _capture_bind,
    )

    job = {
        "id": 57,
        "job_type": "tts_longform",
        "request_id": "job-req-57",
        "owner_user_id": "1",
        "payload": {
            "user_id": "1",
            "speech_request": {
                "model": "kokoro",
                "input": "log correlation test",
                "voice": "af_heart",
                "response_format": "mp3",
                "stream": False,
            },
        },
    }

    result = await _handle_tts_job(job)
    assert result["output_id"] == 654
    assert any(
        "failed to write history record" in line and "job_id=57" in line and "request_id=job-req-57" in line
        for line in debug_lines
    )
    assert {"error_type": "RuntimeError"} in bound_contexts
