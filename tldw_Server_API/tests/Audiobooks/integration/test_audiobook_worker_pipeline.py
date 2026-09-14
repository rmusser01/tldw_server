import asyncio
import io
import json
import os
import shutil
import wave
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.endpoints.audio import audiobooks as audiobooks_endpoints
from tldw_Server_API.app.api.v1.endpoints.audio.audiobooks import router as audiobooks_router
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.DB_Management.Collections_DB import CollectionsDatabase
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.TTS.audio_converter import AudioConverter
from tldw_Server_API.app.services import audiobook_jobs_worker

pytestmark = pytest.mark.integration


def _alignment_payload_for(text: str) -> dict:
    words = text.split()
    payload_words = []
    start = 0
    for word in words:
        end = start + 500
        payload_words.append(
            {
                "word": word,
                "start_ms": start,
                "end_ms": end,
                "char_start": None,
                "char_end": None,
            }
        )
        start = end
    return {
        "engine": "kokoro",
        "sample_rate": 24000,
        "words": payload_words,
    }


def _wav_bytes(duration_seconds: float = 0.2, sample_rate: int = 16000) -> bytes:
    frames = int(duration_seconds * sample_rate)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"\x00\x00" * frames)
    return buffer.getvalue()


def test_route_metadata_merge_marks_conflicting_actual_routes_as_mixed():
    current = {
        "requested_backend": "gateway:company",
        "actual_backend": "gateway:company",
        "requested_model": "Vendor/Exact-TTS",
        "actual_model": "Vendor/Exact-TTS",
        "fallback_used": False,
        "conversion_used": False,
    }
    update = {
        "requested_backend": "gateway:company",
        "actual_backend": "gateway:backup",
        "requested_model": "Vendor/Exact-TTS",
        "actual_model": "Vendor/Backup-TTS",
        "fallback_used": True,
        "conversion_used": True,
    }

    audiobook_jobs_worker._merge_route_metadata(current, update)

    assert current["requested_backend"] == "gateway:company"
    assert current["requested_model"] == "Vendor/Exact-TTS"
    assert current["actual_backend"] == "mixed"
    assert current["actual_model"] == "mixed"
    assert current["mixed_route"] is True
    assert current["fallback_used"] is True
    assert current["conversion_used"] is True


def test_gateway_text_validation_does_not_apply_a_legacy_provider_normalizer():
    text = "Visit https://example.test/docs or email narrator@example.test."

    sanitized = audiobook_jobs_worker._validate_text(
        text,
        provider=None,
        model="Vendor/Exact-TTS",
        backend="gateway:company",
    )

    assert sanitized == text


def test_chapter_only_gateway_selects_provider_neutral_text_validation():
    text = "Visit https://example.test/docs or email narrator@example.test."
    backend = audiobook_jobs_worker._explicit_text_validation_backend(
        None,
        [{"chapter_id": "ch_001", "tts_backend": "gateway:chapter"}],
    )

    sanitized = audiobook_jobs_worker._validate_text(
        text,
        provider=None,
        model="Vendor/Exact-TTS",
        backend=backend,
    )

    assert backend == "gateway:chapter"
    assert sanitized == text


def test_hybrid_legacy_and_gateway_route_aggregate_is_marked_mixed():
    route = {
        "requested_backend": "gateway:chapter",
        "actual_backend": "gateway:chapter",
        "requested_model": "Vendor/Exact-TTS",
        "actual_model": "Vendor/Exact-TTS",
        "fallback_used": False,
        "conversion_used": False,
    }

    audiobook_jobs_worker._mark_route_metadata_mixed(route)

    assert route["requested_backend"] == "mixed"
    assert route["actual_backend"] == "mixed"
    assert route["requested_model"] == "mixed"
    assert route["actual_model"] == "mixed"
    assert route["mixed_route"] is True


@pytest.fixture()
def user_base_dir(tmp_path, monkeypatch):
    base_dir = tmp_path / "user_dbs"
    monkeypatch.setenv("USER_DB_BASE_DIR", str(base_dir))
    return base_dir


@pytest.fixture()
def jobs_db_path(tmp_path, monkeypatch):
    db_path = tmp_path / "jobs.db"
    monkeypatch.setenv("JOBS_DB_PATH", str(db_path))
    try:
        with audiobooks_endpoints._job_manager_lock:
            audiobooks_endpoints._job_manager_cache.clear()
    except Exception:
        _ = None
    yield db_path
    try:
        with audiobooks_endpoints._job_manager_lock:
            audiobooks_endpoints._job_manager_cache.clear()
    except Exception:
        _ = None


@pytest.fixture()
def fake_tts(monkeypatch):
    async def _fake_generate(*_args, **_kwargs):
        text = _kwargs.get("text") or ""
        return b"FAKEAUDIO", _alignment_payload_for(text)

    monkeypatch.setattr(audiobook_jobs_worker, "_generate_tts_audio", _fake_generate)


def _build_job_payload(project_id: str) -> dict:
    return {
        "project_title": "Test Book",
        "source": {"input_type": "txt", "raw_text": "Hello world."},
        "chapters": [
            {"chapter_id": "ch_001", "include": True, "voice": "af_heart", "speed": 1.0}
        ],
        "output": {"merge": False, "per_chapter": True, "formats": ["mp3"]},
        "subtitles": {"formats": ["srt"], "mode": "sentence", "variant": "wide"},
        "metadata": {"tts_model": "kokoro"},
        "project_id": project_id,
    }


def _build_batch_payload(project_id: str) -> dict:
    return {
        "project_title": "Batch Book",
        "items": [
            {
                "source": {"input_type": "txt", "raw_text": "Hello world."},
                "chapters": [{"chapter_id": "ch_001", "include": True, "voice": "af_heart", "speed": 1.0}],
                "metadata": {"title": "Item One"},
            },
            {
                "source": {"input_type": "txt", "raw_text": "Goodbye world."},
                "chapters": [{"chapter_id": "ch_001", "include": True, "voice": "am_adam", "speed": 0.98}],
                "metadata": {"title": "Item Two"},
            },
        ],
        "output": {"merge": False, "per_chapter": True, "formats": ["mp3"]},
        "subtitles": {"formats": ["srt"], "mode": "sentence", "variant": "wide"},
        "metadata": {"tts_model": "kokoro"},
        "project_id": project_id,
    }


@pytest.mark.asyncio
async def test_generate_gateway_audio_rebuilds_request_for_current_owner(monkeypatch):
    captured: dict = {}

    class DummyService:
        def generate_speech(self, request, **kwargs):
            captured["request"] = request
            captured["kwargs"] = kwargs
            request._tts_metadata = {
                "requested_backend": "gateway:company",
                "provider": "gateway:company",
                "model": "Vendor/Exact-TTS",
                "fallback_used": False,
                "conversion_used": False,
            }

            async def _stream():
                yield b"audio"

            return _stream()

    async def _get_service():
        return DummyService()

    monkeypatch.setattr(audiobook_jobs_worker, "get_tts_service_v2", _get_service)

    audio, alignment, route = await audiobook_jobs_worker._generate_tts_audio(
        text="Read me",
        model="Vendor/Exact-TTS",
        provider=None,
        backend="gateway:company",
        allow_fallback=False,
        voice="narrator",
        speed=None,
        response_format="mp3",
        user_id=23,
    )

    assert audio == b"audio"
    assert alignment is None
    assert route["actual_backend"] == "gateway:company"
    assert route["actual_model"] == "Vendor/Exact-TTS"
    assert captured["request"].backend == "gateway:company"
    assert captured["request"].allow_fallback is False
    assert captured["request"].model == "Vendor/Exact-TTS"
    assert captured["kwargs"] == {
        "provider": None,
        "fallback": False,
        "user_id": 23,
    }


def test_audiobook_worker_creates_outputs(user_base_dir, jobs_db_path, fake_tts):
    user_id = 1
    project_id = "abk_test01"
    payload = _build_job_payload(project_id)

    job_manager = JobManager()
    job = job_manager.create_job(
        domain="audiobooks",
        queue="default",
        job_type="audiobook_generate",
        payload=payload,
        owner_user_id=str(user_id),
        priority=5,
    )

    acquired = job_manager.acquire_next_job(
        domain="audiobooks",
        queue="default",
        lease_seconds=60,
        worker_id="test-worker",
    )
    assert acquired is not None

    asyncio.run(
        audiobook_jobs_worker.process_audiobook_job(
            acquired,
            job_manager=job_manager,
            worker_id="test-worker",
        )
    )

    job_row = job_manager.get_job(int(job["id"]))
    assert job_row is not None
    assert job_row.get("status") == "completed"

    collections_db = CollectionsDatabase(user_id)
    outputs, _total = collections_db.list_output_artifacts(job_id=int(job["id"]), limit=50, offset=0)
    types = {row.type for row in outputs}
    assert "audiobook_audio" in types
    assert "audiobook_subtitle" in types
    assert "audiobook_alignment" in types
    for row in outputs:
        meta = json.loads(row.metadata_json or "{}")
        assert "chapter_index" in meta
        assert isinstance(meta.get("byte_size"), int)

    projects = collections_db.list_audiobook_projects(limit=10, offset=0)
    assert len(projects) == 1
    project = projects[0]
    assert project.status == "completed"
    chapters = collections_db.list_audiobook_chapters(project_id=project.id, limit=10, offset=0)
    assert len(chapters) == 1
    assert "requested_backend" not in json.loads(chapters[0].metadata_json or "{}")
    artifacts = collections_db.list_audiobook_artifacts(project_id=project.id, limit=20, offset=0)
    assert len(artifacts) == len(outputs)
    artifact_types = {row.artifact_type for row in artifacts}
    assert {"audio", "subtitle", "alignment"}.issubset(artifact_types)

    outputs_dir = DatabasePaths.get_user_outputs_dir(user_id)
    for row in outputs:
        path = outputs_dir / row.storage_path
        assert path.exists()
        if row.type == "audiobook_subtitle":
            assert "Hello world" in path.read_text(encoding="utf-8")
        if row.type == "audiobook_alignment":
            payload = json.loads(path.read_text(encoding="utf-8"))
            assert payload.get("engine") == "kokoro"


def test_audiobook_gateway_worker_uses_owner_credentials_and_persists_route_metadata(
    user_base_dir,
    jobs_db_path,
    monkeypatch,
):
    calls: list[dict] = []
    events: list[tuple[str, dict]] = []

    async def _fake_generate(**kwargs):
        calls.append(kwargs)
        return (
            b"GATEWAY-AUDIO",
            None,
            {
                "requested_backend": "gateway:company",
                "actual_backend": "gateway:backup",
                "requested_model": "Vendor/Exact-TTS",
                "actual_model": "Vendor/Backup-TTS",
                "fallback_used": True,
                "conversion_used": False,
            },
        )

    monkeypatch.setattr(audiobook_jobs_worker, "_generate_tts_audio", _fake_generate)
    monkeypatch.setattr(
        audiobook_jobs_worker,
        "emit_job_event",
        lambda event_type, *, job, attrs: events.append((event_type, attrs)),
    )

    payload = _build_job_payload("abk_gateway")
    payload.pop("subtitles", None)
    payload["tts_backend"] = "gateway:company"
    payload["tts_allow_fallback"] = True
    payload["tts_model"] = "Vendor/Exact-TTS"
    payload["tts_voice"] = "narrator"
    payload["chapters"][0]["voice"] = "narrator"

    job_manager = JobManager()
    job = job_manager.create_job(
        domain="audiobooks",
        queue="default",
        job_type="audiobook_generate",
        payload=payload,
        owner_user_id="17",
        priority=5,
    )
    acquired = job_manager.acquire_next_job(
        domain="audiobooks",
        queue="default",
        lease_seconds=60,
        worker_id="test-worker",
    )
    assert acquired is not None

    asyncio.run(
        audiobook_jobs_worker.process_audiobook_job(
            acquired,
            job_manager=job_manager,
            worker_id="test-worker",
        )
    )

    assert calls
    assert calls[0]["backend"] == "gateway:company"
    assert calls[0]["allow_fallback"] is True
    assert calls[0]["provider"] is None
    assert calls[0]["user_id"] == 17

    collections_db = CollectionsDatabase(17)
    outputs, _ = collections_db.list_output_artifacts(job_id=int(job["id"]), limit=50, offset=0)
    assert outputs
    output_metadata = json.loads(outputs[0].metadata_json or "{}")
    assert output_metadata["requested_backend"] == "gateway:company"
    assert output_metadata["actual_backend"] == "gateway:backup"
    assert output_metadata["requested_model"] == "Vendor/Exact-TTS"
    assert output_metadata["actual_model"] == "Vendor/Backup-TTS"
    assert output_metadata["fallback_used"] is True

    project = collections_db.list_audiobook_projects(limit=10, offset=0)[0]
    chapter = collections_db.list_audiobook_chapters(project_id=project.id, limit=10, offset=0)[0]
    chapter_metadata = json.loads(chapter.metadata_json or "{}")
    assert chapter_metadata["actual_backend"] == "gateway:backup"
    assert chapter_metadata["fallback_used"] is True

    completed = job_manager.get_job(int(job["id"]))
    result = completed.get("result") or completed.get("result_json")
    if isinstance(result, str):
        result = json.loads(result)
    assert result["requested_backend"] == "gateway:company"
    assert result["actual_backend"] == "gateway:backup"
    assert result["fallback_used"] is True
    assert any(
        event_type == "audiobook.tts.route"
        and attrs.get("actual_backend") == "gateway:backup"
        for event_type, attrs in events
    )
    serialized = json.dumps({"events": events, "result": result, "metadata": output_metadata}).casefold()
    for forbidden in ("api_key", "authorization", "base_url", "headers", "credential_scope"):
        assert forbidden not in serialized


def test_hybrid_audiobook_merged_artifact_and_job_result_use_mixed_route(
    user_base_dir,
    jobs_db_path,
    monkeypatch,
):
    calls: list[dict] = []

    async def _fake_generate(**kwargs):
        calls.append(kwargs)
        if kwargs.get("backend"):
            route = {
                "requested_backend": "gateway:chapter",
                "actual_backend": "gateway:chapter",
                "requested_model": "Vendor/Exact-TTS",
                "actual_model": "Vendor/Exact-TTS",
                "fallback_used": False,
                "conversion_used": False,
            }
        else:
            route = {}
        return _wav_bytes(), None, route

    async def _fake_concat(_paths, output_path, _format):
        output_path.write_bytes(_wav_bytes(duration_seconds=0.4))
        return True

    monkeypatch.setattr(audiobook_jobs_worker, "_generate_tts_audio", _fake_generate)
    monkeypatch.setattr(AudioConverter, "concat_audio_files", _fake_concat)

    payload = {
        "project_title": "Hybrid Book",
        "source": {
            "input_type": "txt",
            "raw_text": (
                "[[chapter:id=ch_001]]\nLegacy chapter.\n"
                "[[chapter:id=ch_002]]\nGateway chapter."
            ),
        },
        "chapters": [
            {"chapter_id": "ch_001", "include": True, "voice": "narrator"},
            {
                "chapter_id": "ch_002",
                "include": True,
                "voice": "narrator",
                "tts_backend": "gateway:chapter",
                "tts_allow_fallback": False,
            },
        ],
        "output": {"merge": True, "per_chapter": True, "formats": ["wav"]},
        "tts_model": "Vendor/Exact-TTS",
        "project_id": "abk_hybrid_route",
    }
    job_manager = JobManager()
    job = job_manager.create_job(
        domain="audiobooks",
        queue="default",
        job_type="audiobook_generate",
        payload=payload,
        owner_user_id="21",
        priority=5,
    )
    acquired = job_manager.acquire_next_job(
        domain="audiobooks",
        queue="default",
        lease_seconds=60,
        worker_id="test-worker",
    )
    assert acquired is not None

    asyncio.run(
        audiobook_jobs_worker.process_audiobook_job(
            acquired,
            job_manager=job_manager,
            worker_id="test-worker",
        )
    )

    assert [call.get("backend") for call in calls] == [None, "gateway:chapter"]
    collections_db = CollectionsDatabase(21)
    outputs, _ = collections_db.list_output_artifacts(
        job_id=int(job["id"]),
        limit=50,
        offset=0,
    )
    merged_metadata = next(
        metadata
        for row in outputs
        if (metadata := json.loads(row.metadata_json or "{}")).get("scope") == "merged"
    )
    assert merged_metadata["actual_backend"] == "mixed"
    assert merged_metadata["mixed_route"] is True

    completed = job_manager.get_job(int(job["id"]))
    result = completed.get("result") or completed.get("result_json")
    if isinstance(result, str):
        result = json.loads(result)
    assert result["actual_backend"] == "mixed"
    assert result["mixed_route"] is True


def test_audiobook_output_usage_decrements_on_delete(user_base_dir, jobs_db_path, fake_tts):
    user_id = 1
    project_id = "abk_quota_usage"
    payload = _build_job_payload(project_id)

    job_manager = JobManager()
    job = job_manager.create_job(
        domain="audiobooks",
        queue="default",
        job_type="audiobook_generate",
        payload=payload,
        owner_user_id=str(user_id),
        priority=5,
    )

    acquired = job_manager.acquire_next_job(
        domain="audiobooks",
        queue="default",
        lease_seconds=60,
        worker_id="test-worker",
    )
    assert acquired is not None

    asyncio.run(
        audiobook_jobs_worker.process_audiobook_job(
            acquired,
            job_manager=job_manager,
            worker_id="test-worker",
        )
    )

    collections_db = CollectionsDatabase(user_id)
    used_before = collections_db.get_audiobook_output_usage()
    assert used_before is not None
    outputs, _total = collections_db.list_output_artifacts(job_id=int(job["id"]), limit=50, offset=0)
    assert outputs
    first = outputs[0]
    meta = json.loads(first.metadata_json or "{}")
    size_bytes = int(meta.get("byte_size") or 0)
    assert size_bytes > 0

    ok = collections_db.delete_output_artifact(first.id, hard=True)
    assert ok is True

    used_after = collections_db.get_audiobook_output_usage()
    assert used_after is not None
    assert used_after == max(0, used_before - size_bytes)


def test_audiobook_worker_processes_batch_items(user_base_dir, jobs_db_path, fake_tts):
    user_id = 1
    project_id = "abk_batch01"
    payload = _build_batch_payload(project_id)

    job_manager = JobManager()
    job = job_manager.create_job(
        domain="audiobooks",
        queue="default",
        job_type="audiobook_generate",
        payload=payload,
        owner_user_id=str(user_id),
        priority=5,
    )

    acquired = job_manager.acquire_next_job(
        domain="audiobooks",
        queue="default",
        lease_seconds=60,
        worker_id="test-worker",
    )
    assert acquired is not None

    asyncio.run(
        audiobook_jobs_worker.process_audiobook_job(
            acquired,
            job_manager=job_manager,
            worker_id="test-worker",
        )
    )

    job_row = job_manager.get_job(int(job["id"]))
    assert job_row is not None
    assert job_row.get("status") == "completed"

    collections_db = CollectionsDatabase(user_id)
    outputs, _total = collections_db.list_output_artifacts(job_id=int(job["id"]), limit=50, offset=0)
    audio_rows = [row for row in outputs if row.type == "audiobook_audio"]
    subtitle_rows = [row for row in outputs if row.type == "audiobook_subtitle"]
    alignment_rows = [row for row in outputs if row.type == "audiobook_alignment"]
    assert len(audio_rows) == 2
    assert len(subtitle_rows) == 2
    assert len(alignment_rows) == 2

    titles = [row.title for row in audio_rows]
    assert len(set(titles)) == len(titles)

    for row in outputs:
        meta = json.loads(row.metadata_json or "{}")
        assert "item_index" in meta
        assert isinstance(meta.get("byte_size"), int)

    projects = collections_db.list_audiobook_projects(limit=10, offset=0)
    assert len(projects) == 1
    project = projects[0]
    chapters = collections_db.list_audiobook_chapters(project_id=project.id, limit=10, offset=0)
    assert len(chapters) == 2
    artifacts = collections_db.list_audiobook_artifacts(project_id=project.id, limit=20, offset=0)
    assert len(artifacts) == len(outputs)


def test_audiobook_worker_enforces_artifact_quota(user_base_dir, jobs_db_path, monkeypatch):
    user_id = 1
    project_id = "abk_quota01"
    payload = _build_job_payload(project_id)

    monkeypatch.setenv("AUDIOBOOK_ARTIFACT_QUOTA_MB", "0.001")

    async def _fake_generate(*_args, **_kwargs):
        text = _kwargs.get("text") or ""
        return b"A" * 2048, _alignment_payload_for(text)

    monkeypatch.setattr(audiobook_jobs_worker, "_generate_tts_audio", _fake_generate)

    job_manager = JobManager()
    job = job_manager.create_job(
        domain="audiobooks",
        queue="default",
        job_type="audiobook_generate",
        payload=payload,
        owner_user_id=str(user_id),
        priority=5,
    )

    acquired = job_manager.acquire_next_job(
        domain="audiobooks",
        queue="default",
        lease_seconds=60,
        worker_id="test-worker",
    )
    assert acquired is not None

    asyncio.run(
        audiobook_jobs_worker.process_audiobook_job(
            acquired,
            job_manager=job_manager,
            worker_id="test-worker",
        )
    )

    job_row = job_manager.get_job(int(job["id"]))
    assert job_row is not None
    assert job_row.get("status") == "failed"
    error_text = " ".join(
        str(job_row.get(key) or "")
        for key in ("error_message", "last_error", "error_code", "error")
    )
    assert "audiobook_artifact_quota_exceeded" in error_text


def test_audiobook_worker_rejects_subtitles_for_non_kokoro(
    user_base_dir,
    jobs_db_path,
    fake_tts,
):
    user_id = 1
    project_id = "abk_non_kokoro_subtitles"
    payload = _build_job_payload(project_id)
    payload["tts_provider"] = "openai"
    payload["tts_model"] = "tts-1"

    job_manager = JobManager()
    job = job_manager.create_job(
        domain="audiobooks",
        queue="default",
        job_type="audiobook_generate",
        payload=payload,
        owner_user_id=str(user_id),
        priority=5,
    )

    acquired = job_manager.acquire_next_job(
        domain="audiobooks",
        queue="default",
        lease_seconds=60,
        worker_id="test-worker",
    )
    assert acquired is not None

    asyncio.run(
        audiobook_jobs_worker.process_audiobook_job(
            acquired,
            job_manager=job_manager,
            worker_id="test-worker",
        )
    )

    job_row = job_manager.get_job(int(job["id"]))
    assert job_row is not None
    assert job_row.get("status") == "failed"
    error_text = " ".join(
        str(job_row.get(key) or "")
        for key in ("error_message", "last_error", "error_code", "error")
    )
    assert "subtitles_not_supported_for_provider" in error_text


def test_audiobook_worker_allows_non_kokoro_item_with_null_subtitles(
    user_base_dir,
    jobs_db_path,
    fake_tts,
):
    user_id = 1
    project_id = "abk_batch_non_kokoro_null_subs"
    payload = _build_batch_payload(project_id)
    payload["items"][1]["tts_provider"] = "openai"
    payload["items"][1]["tts_model"] = "tts-1"
    payload["items"][1]["subtitles"] = None

    job_manager = JobManager()
    job = job_manager.create_job(
        domain="audiobooks",
        queue="default",
        job_type="audiobook_generate",
        payload=payload,
        owner_user_id=str(user_id),
        priority=5,
    )

    acquired = job_manager.acquire_next_job(
        domain="audiobooks",
        queue="default",
        lease_seconds=60,
        worker_id="test-worker",
    )
    assert acquired is not None

    asyncio.run(
        audiobook_jobs_worker.process_audiobook_job(
            acquired,
            job_manager=job_manager,
            worker_id="test-worker",
        )
    )

    job_row = job_manager.get_job(int(job["id"]))
    assert job_row is not None
    assert job_row.get("status") == "completed"

    collections_db = CollectionsDatabase(user_id)
    outputs, _total = collections_db.list_output_artifacts(job_id=int(job["id"]), limit=50, offset=0)
    openai_subtitles = []
    for row in outputs:
        meta = json.loads(row.metadata_json or "{}")
        if meta.get("tts_provider") == "openai" and row.type in {"audiobook_subtitle", "audiobook_alignment"}:
            openai_subtitles.append(row)
    assert not openai_subtitles


@pytest.fixture()
def client_user_only(monkeypatch, user_base_dir):
    monkeypatch.setenv("MINIMAL_TEST_APP", "0")
    monkeypatch.setenv("ULTRA_MINIMAL_APP", "0")
    monkeypatch.setenv("ROUTES_STABLE_ONLY", "false")
    existing_enable = (os.getenv("ROUTES_ENABLE") or "").strip()
    enable_parts = [p for p in existing_enable.replace(" ", ",").split(",") if p]
    if "audiobooks" not in [p.lower() for p in enable_parts]:
        enable_parts.append("audiobooks")
    monkeypatch.setenv("ROUTES_ENABLE", ",".join(enable_parts))

    app = FastAPI()
    app.include_router(audiobooks_router, prefix="/api/v1")

    async def override_user():
        return User(id=1, username="tester", email="t@e.com", is_active=True, is_admin=True)

    app.dependency_overrides[get_request_user] = override_user
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.clear()


def test_audiobook_artifacts_endpoint_lists_outputs(
    client_user_only,
    jobs_db_path,
    fake_tts,
):
    create_payload = {
        "project_title": "Test Book",
        "source": {"input_type": "txt", "raw_text": "Hello world."},
        "chapters": [
            {"chapter_id": "ch_001", "include": True, "voice": "af_heart", "speed": 1.0}
        ],
        "output": {"merge": False, "per_chapter": True, "formats": ["mp3"]},
        "subtitles": {"formats": ["srt"], "mode": "sentence", "variant": "wide"},
    }
    resp = client_user_only.post("/api/v1/audiobooks/jobs", json=create_payload)
    assert resp.status_code == 200
    data = resp.json()
    job_id = data["job_id"]

    job_manager = JobManager()
    acquired = job_manager.acquire_next_job(
        domain="audiobooks",
        queue="default",
        lease_seconds=60,
        worker_id="test-worker",
    )
    assert acquired is not None

    asyncio.run(
        audiobook_jobs_worker.process_audiobook_job(
            acquired,
            job_manager=job_manager,
            worker_id="test-worker",
        )
    )

    artifacts_resp = client_user_only.get(f"/api/v1/audiobooks/jobs/{job_id}/artifacts")
    assert artifacts_resp.status_code == 200
    artifacts = artifacts_resp.json()["artifacts"]
    assert artifacts
    first = artifacts[0]
    assert first["output_id"]
    assert first["download_url"].endswith(f"/api/v1/outputs/{first['output_id']}/download")


def test_audiobook_job_status_includes_chapter_progress(
    client_user_only,
    jobs_db_path,
    fake_tts,
):
    create_payload = {
        "project_title": "Test Book",
        "source": {"input_type": "txt", "raw_text": "Hello world."},
        "chapters": [
            {"chapter_id": "ch_001", "include": True, "voice": "af_heart", "speed": 1.0}
        ],
        "output": {"merge": False, "per_chapter": True, "formats": ["mp3"]},
        "subtitles": {"formats": ["srt"], "mode": "sentence", "variant": "wide"},
    }
    resp = client_user_only.post("/api/v1/audiobooks/jobs", json=create_payload)
    assert resp.status_code == 200
    job_id = resp.json()["job_id"]

    job_manager = JobManager()
    acquired = job_manager.acquire_next_job(
        domain="audiobooks",
        queue="default",
        lease_seconds=60,
        worker_id="test-worker",
    )
    assert acquired is not None

    asyncio.run(
        audiobook_jobs_worker.process_audiobook_job(
            acquired,
            job_manager=job_manager,
            worker_id="test-worker",
        )
    )

    status_resp = client_user_only.get(f"/api/v1/audiobooks/jobs/{job_id}")
    assert status_resp.status_code == 200
    progress = status_resp.json().get("progress") or {}
    assert progress.get("chapter_index") is not None
    assert progress.get("chapters_total") is not None


def test_audiobook_job_status_includes_batch_item_progress(
    client_user_only,
    jobs_db_path,
    fake_tts,
):
    create_payload = {
        "project_title": "Batch Book",
        "items": [
            {
                "source": {"input_type": "txt", "raw_text": "Hello world."},
                "chapters": [
                    {"chapter_id": "ch_001", "include": True, "voice": "af_heart", "speed": 1.0}
                ],
                "metadata": {"title": "Item One"},
            },
            {
                "source": {"input_type": "txt", "raw_text": "Goodbye world."},
                "chapters": [
                    {"chapter_id": "ch_001", "include": True, "voice": "am_adam", "speed": 0.98}
                ],
                "metadata": {"title": "Item Two"},
            },
        ],
        "output": {"merge": False, "per_chapter": True, "formats": ["mp3"]},
        "subtitles": {"formats": ["srt"], "mode": "sentence", "variant": "wide"},
    }
    resp = client_user_only.post("/api/v1/audiobooks/jobs", json=create_payload)
    assert resp.status_code == 200
    job_id = resp.json()["job_id"]

    job_manager = JobManager()
    acquired = job_manager.acquire_next_job(
        domain="audiobooks",
        queue="default",
        lease_seconds=60,
        worker_id="test-worker",
    )
    assert acquired is not None

    asyncio.run(
        audiobook_jobs_worker.process_audiobook_job(
            acquired,
            job_manager=job_manager,
            worker_id="test-worker",
        )
    )

    status_resp = client_user_only.get(f"/api/v1/audiobooks/jobs/{job_id}")
    assert status_resp.status_code == 200
    progress = status_resp.json().get("progress") or {}
    assert progress.get("item_index") == 1
    assert progress.get("items_total") == 2


def test_audiobook_worker_chunks_large_chapters(
    user_base_dir,
    jobs_db_path,
    monkeypatch,
):
    monkeypatch.setenv("AUDIOBOOK_CHAPTER_MAX_CHARS", "12")
    call_count = {"count": 0}

    async def _fake_generate(*_args, **_kwargs):
        call_count["count"] += 1
        text = _kwargs.get("text") or ""
        return b"FAKEAUDIO", _alignment_payload_for(text)

    async def _fake_concat(input_paths, output_path, target_format, **_kwargs):
        Path(output_path).write_bytes(b"MERGED")
        return True

    monkeypatch.setattr(audiobook_jobs_worker, "_generate_tts_audio", _fake_generate)
    monkeypatch.setattr(
        audiobook_jobs_worker.AudioConverter,
        "concat_audio_files",
        staticmethod(_fake_concat),
    )

    user_id = 1
    project_id = "abk_chunk01"
    payload = _build_job_payload(project_id)
    payload["source"]["raw_text"] = "Hello world\nAgain here"

    job_manager = JobManager()
    job = job_manager.create_job(
        domain="audiobooks",
        queue="default",
        job_type="audiobook_generate",
        payload=payload,
        owner_user_id=str(user_id),
        priority=5,
    )

    acquired = job_manager.acquire_next_job(
        domain="audiobooks",
        queue="default",
        lease_seconds=60,
        worker_id="test-worker",
    )
    assert acquired is not None

    asyncio.run(
        audiobook_jobs_worker.process_audiobook_job(
            acquired,
            job_manager=job_manager,
            worker_id="test-worker",
        )
    )

    assert call_count["count"] > 1

    collections_db = CollectionsDatabase(user_id)
    outputs, _total = collections_db.list_output_artifacts(job_id=int(job["id"]), limit=50, offset=0)
    alignment_rows = [row for row in outputs if row.type == "audiobook_alignment"]
    assert alignment_rows
    outputs_dir = DatabasePaths.get_user_outputs_dir(user_id)
    payload = json.loads((outputs_dir / alignment_rows[0].storage_path).read_text(encoding="utf-8"))
    words = payload.get("words") or []
    assert len(words) == 4
    assert words[2]["start_ms"] == 1000


def test_audiobook_worker_applies_voice_profile(
    user_base_dir,
    jobs_db_path,
    monkeypatch,
):
    captured = {}

    async def _fake_generate(*_args, **_kwargs):
        captured["voice"] = _kwargs.get("voice")
        captured["speed"] = _kwargs.get("speed")
        text = _kwargs.get("text") or ""
        return b"FAKEAUDIO", _alignment_payload_for(text)

    monkeypatch.setattr(audiobook_jobs_worker, "_generate_tts_audio", _fake_generate)

    user_id = 1
    collections_db = CollectionsDatabase(user_id)
    profile = collections_db.create_voice_profile(
        profile_id="vp_test_profile",
        name="Test Profile",
        default_voice="af_heart",
        default_speed=1.1,
        chapter_overrides_json=json.dumps(
            [{"chapter_id": "ch_001", "voice": "am_adam", "speed": 0.9}]
        ),
    )

    project_id = "abk_voice01"
    payload = _build_job_payload(project_id)
    payload["voice_profile_id"] = profile.profile_id
    payload["chapters"] = [{"chapter_id": "ch_001", "include": True}]

    job_manager = JobManager()
    job_manager.create_job(
        domain="audiobooks",
        queue="default",
        job_type="audiobook_generate",
        payload=payload,
        owner_user_id=str(user_id),
        priority=5,
    )

    acquired = job_manager.acquire_next_job(
        domain="audiobooks",
        queue="default",
        lease_seconds=60,
        worker_id="test-worker",
    )
    assert acquired is not None

    asyncio.run(
        audiobook_jobs_worker.process_audiobook_job(
            acquired,
            job_manager=job_manager,
            worker_id="test-worker",
        )
    )

    assert captured.get("voice") == "am_adam"
    assert captured.get("speed") == 0.9


def test_audiobook_worker_applies_item_voice_profile_overrides(
    user_base_dir,
    jobs_db_path,
    monkeypatch,
):
    captured: list[dict] = []

    async def _fake_generate(*_args, **_kwargs):
        captured.append(
            {
                "voice": _kwargs.get("voice"),
                "speed": _kwargs.get("speed"),
            }
        )
        text = _kwargs.get("text") or ""
        return b"FAKEAUDIO", _alignment_payload_for(text)

    monkeypatch.setattr(audiobook_jobs_worker, "_generate_tts_audio", _fake_generate)

    user_id = 1
    collections_db = CollectionsDatabase(user_id)
    default_profile = collections_db.create_voice_profile(
        profile_id="vp_default",
        name="Default Profile",
        default_voice="af_heart",
        default_speed=1.0,
        chapter_overrides_json=None,
    )
    item_profile = collections_db.create_voice_profile(
        profile_id="vp_item",
        name="Item Profile",
        default_voice="am_adam",
        default_speed=0.9,
        chapter_overrides_json=None,
    )

    project_id = "abk_voice_batch01"
    payload = {
        "project_title": "Batch Voices",
        "project_id": project_id,
        "voice_profile_id": default_profile.profile_id,
        "items": [
            {
                "source": {"input_type": "txt", "raw_text": "Hello world."},
                "chapters": [{"chapter_id": "ch_001", "include": True}],
                "metadata": {"title": "Item One"},
            },
            {
                "source": {"input_type": "txt", "raw_text": "Goodbye world."},
                "chapters": [{"chapter_id": "ch_001", "include": True}],
                "voice_profile_id": item_profile.profile_id,
                "metadata": {"title": "Item Two"},
            },
        ],
        "output": {"merge": False, "per_chapter": True, "formats": ["mp3"]},
        "subtitles": {"formats": ["srt"], "mode": "sentence", "variant": "wide"},
        "metadata": {"tts_model": "kokoro"},
    }

    job_manager = JobManager()
    job_manager.create_job(
        domain="audiobooks",
        queue="default",
        job_type="audiobook_generate",
        payload=payload,
        owner_user_id=str(user_id),
        priority=5,
    )

    acquired = job_manager.acquire_next_job(
        domain="audiobooks",
        queue="default",
        lease_seconds=60,
        worker_id="test-worker",
    )
    assert acquired is not None

    asyncio.run(
        audiobook_jobs_worker.process_audiobook_job(
            acquired,
            job_manager=job_manager,
            worker_id="test-worker",
        )
    )

    assert len(captured) == 2
    assert captured[0]["voice"] == "af_heart"
    assert captured[0]["speed"] == 1.0
    assert captured[1]["voice"] == "am_adam"
    assert captured[1]["speed"] == 0.9


def test_audiobook_project_endpoints(
    client_user_only,
    jobs_db_path,
    fake_tts,
):
    create_payload = {
        "project_title": "Project API Book",
        "source": {"input_type": "txt", "raw_text": "Hello world."},
        "chapters": [
            {"chapter_id": "ch_001", "include": True, "voice": "af_heart", "speed": 1.0}
        ],
        "output": {"merge": False, "per_chapter": True, "formats": ["mp3"]},
        "subtitles": {"formats": ["srt"], "mode": "sentence", "variant": "wide"},
        "queue": {"priority": 4, "batch_group": "batch_project_01"},
    }
    resp = client_user_only.post("/api/v1/audiobooks/jobs", json=create_payload)
    assert resp.status_code == 200
    data = resp.json()
    project_id = data["project_id"]

    job_manager = JobManager()
    acquired = job_manager.acquire_next_job(
        domain="audiobooks",
        queue="default",
        lease_seconds=60,
        worker_id="test-worker",
    )
    assert acquired is not None

    asyncio.run(
        audiobook_jobs_worker.process_audiobook_job(
            acquired,
            job_manager=job_manager,
            worker_id="test-worker",
        )
    )

    list_resp = client_user_only.get("/api/v1/audiobooks/projects")
    assert list_resp.status_code == 200
    projects = list_resp.json()["projects"]
    assert any(project.get("project_id") == project_id for project in projects)

    detail_resp = client_user_only.get(f"/api/v1/audiobooks/projects/{project_id}")
    assert detail_resp.status_code == 200
    project = detail_resp.json()["project"]
    assert project["project_id"] == project_id
    assert project["status"] == "completed"
    settings = project.get("settings") or {}
    queue_settings = settings.get("queue") or {}
    assert queue_settings.get("priority") == 4
    assert queue_settings.get("batch_group") == "batch_project_01"

    chapters_resp = client_user_only.get(f"/api/v1/audiobooks/projects/{project_id}/chapters")
    assert chapters_resp.status_code == 200
    chapters = chapters_resp.json()["chapters"]
    assert len(chapters) == 1
    assert chapters[0]["chapter_index"] == 0

    artifacts_resp = client_user_only.get(f"/api/v1/audiobooks/projects/{project_id}/artifacts")
    assert artifacts_resp.status_code == 200
    artifacts = artifacts_resp.json()["artifacts"]
    assert artifacts

    project_db_id = project["project_db_id"]
    numeric_resp = client_user_only.get(f"/api/v1/audiobooks/projects/{project_db_id}")
    assert numeric_resp.status_code == 200


def test_audiobook_project_pagination(
    client_user_only,
    jobs_db_path,
    fake_tts,
):
    job_manager = JobManager()
    project_ids = []
    for idx in range(2):
        create_payload = {
            "project_title": f"Paginated Book {idx}",
            "source": {"input_type": "txt", "raw_text": "Hello world."},
            "chapters": [
                {"chapter_id": "ch_001", "include": True, "voice": "af_heart", "speed": 1.0}
            ],
            "output": {"merge": False, "per_chapter": True, "formats": ["mp3"]},
            "subtitles": {"formats": ["srt"], "mode": "sentence", "variant": "wide"},
        }
        resp = client_user_only.post("/api/v1/audiobooks/jobs", json=create_payload)
        assert resp.status_code == 200
        data = resp.json()
        project_ids.append(data["project_id"])

        acquired = job_manager.acquire_next_job(
            domain="audiobooks",
            queue="default",
            lease_seconds=60,
            worker_id="test-worker",
        )
        assert acquired is not None
        asyncio.run(
            audiobook_jobs_worker.process_audiobook_job(
                acquired,
                job_manager=job_manager,
                worker_id="test-worker",
            )
        )

    page1 = client_user_only.get("/api/v1/audiobooks/projects?limit=1&offset=0")
    assert page1.status_code == 200
    page2 = client_user_only.get("/api/v1/audiobooks/projects?limit=1&offset=1")
    assert page2.status_code == 200
    page1_payload = page1.json()
    page2_payload = page2.json()
    projects_page1 = page1_payload["projects"]
    projects_page2 = page2_payload["projects"]
    assert len(projects_page1) == 1
    assert len(projects_page2) == 1
    assert projects_page1[0]["project_id"] != projects_page2[0]["project_id"]
    assert page1_payload["total"] == 2
    assert page1_payload["limit"] == 1
    assert page1_payload["offset"] == 0
    assert page1_payload["has_more"] is True
    assert page1_payload["next_offset"] == 1
    assert page1_payload["pagination"] == {
        "mode": "offset",
        "total": 2,
        "limit": 1,
        "offset": 0,
        "has_more": True,
        "next_offset": 1,
    }
    assert page2_payload["total"] == 2
    assert page2_payload["limit"] == 1
    assert page2_payload["offset"] == 1
    assert page2_payload["has_more"] is False
    assert page2_payload["next_offset"] is None
    assert page2_payload["pagination"] == {
        "mode": "offset",
        "total": 2,
        "limit": 1,
        "offset": 1,
        "has_more": False,
        "next_offset": None,
    }

    project_id = project_ids[0]
    art_page1 = client_user_only.get(
        f"/api/v1/audiobooks/projects/{project_id}/artifacts?limit=1&offset=0"
    )
    assert art_page1.status_code == 200
    art_page2 = client_user_only.get(
        f"/api/v1/audiobooks/projects/{project_id}/artifacts?limit=1&offset=1"
    )
    assert art_page2.status_code == 200
    artifacts1 = art_page1.json()["artifacts"]
    artifacts2 = art_page2.json()["artifacts"]
    assert len(artifacts1) == 1
    assert len(artifacts2) == 1
    assert artifacts1[0]["output_id"] != artifacts2[0]["output_id"]


def test_audiobook_worker_converts_additional_formats(
    user_base_dir,
    jobs_db_path,
    fake_tts,
    monkeypatch,
):
    async def _fake_convert_format(input_path, output_path, target_format, **_kwargs):
        Path(output_path).write_bytes(b"CONVERTED")
        return True

    monkeypatch.setattr(
        audiobook_jobs_worker.AudioConverter,
        "convert_format",
        staticmethod(_fake_convert_format),
    )

    user_id = 1
    project_id = "abk_convert01"
    payload = _build_job_payload(project_id)
    payload["output"]["formats"] = ["wav", "mp3"]

    job_manager = JobManager()
    job = job_manager.create_job(
        domain="audiobooks",
        queue="default",
        job_type="audiobook_generate",
        payload=payload,
        owner_user_id=str(user_id),
        priority=5,
    )

    acquired = job_manager.acquire_next_job(
        domain="audiobooks",
        queue="default",
        lease_seconds=60,
        worker_id="test-worker",
    )
    assert acquired is not None

    asyncio.run(
        audiobook_jobs_worker.process_audiobook_job(
            acquired,
            job_manager=job_manager,
            worker_id="test-worker",
        )
    )

    collections_db = CollectionsDatabase(user_id)
    outputs, _total = collections_db.list_output_artifacts(job_id=int(job["id"]), limit=50, offset=0)
    audio_rows = [row for row in outputs if row.type == "audiobook_audio"]
    formats = {row.format for row in audio_rows}
    assert "wav" in formats
    assert "mp3" in formats

    converted_rows = [row for row in audio_rows if row.format == "mp3"]
    assert converted_rows
    metadata = json.loads(converted_rows[0].metadata_json or "{}")
    assert metadata.get("converted_from") == "wav"

    outputs_dir = DatabasePaths.get_user_outputs_dir(user_id)
    for row in converted_rows:
        assert (outputs_dir / row.storage_path).exists()


def test_audiobook_worker_converts_with_ffmpeg(
    user_base_dir,
    jobs_db_path,
    monkeypatch,
    tmp_path,
):
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg not available for conversion test")

    wav_bytes = _wav_bytes()

    async def _fake_generate(*_args, **_kwargs):
        text = _kwargs.get("text") or ""
        return wav_bytes, _alignment_payload_for(text)

    monkeypatch.setattr(audiobook_jobs_worker, "_generate_tts_audio", _fake_generate)

    pre_in = tmp_path / "probe.wav"
    pre_out = tmp_path / "probe.flac"
    pre_in.write_bytes(wav_bytes)
    if not asyncio.run(AudioConverter.convert_format(pre_in, pre_out, "flac")):
        pytest.skip("ffmpeg conversion unavailable for flac")

    user_id = 1
    project_id = "abk_ffmpeg01"
    payload = _build_job_payload(project_id)
    payload["output"]["formats"] = ["wav", "flac"]

    job_manager = JobManager()
    job = job_manager.create_job(
        domain="audiobooks",
        queue="default",
        job_type="audiobook_generate",
        payload=payload,
        owner_user_id=str(user_id),
        priority=5,
    )

    acquired = job_manager.acquire_next_job(
        domain="audiobooks",
        queue="default",
        lease_seconds=60,
        worker_id="test-worker",
    )
    assert acquired is not None

    asyncio.run(
        audiobook_jobs_worker.process_audiobook_job(
            acquired,
            job_manager=job_manager,
            worker_id="test-worker",
        )
    )

    collections_db = CollectionsDatabase(user_id)
    outputs, _total = collections_db.list_output_artifacts(job_id=int(job["id"]), limit=50, offset=0)
    audio_rows = [row for row in outputs if row.type == "audiobook_audio"]
    formats = {row.format for row in audio_rows}
    assert "wav" in formats
    assert "flac" in formats


def test_audiobook_worker_creates_merged_and_m4b(
    user_base_dir,
    jobs_db_path,
    fake_tts,
    monkeypatch,
):
    async def _fake_concat(input_paths, output_path, target_format, **_kwargs):
        Path(output_path).write_bytes(b"MERGED")
        return True

    async def _fake_m4b(input_paths, output_path, chapter_titles, metadata=None):
        Path(output_path).write_bytes(b"M4B")
        return True

    monkeypatch.setattr(
        audiobook_jobs_worker.AudioConverter,
        "concat_audio_files",
        staticmethod(_fake_concat),
    )
    monkeypatch.setattr(
        audiobook_jobs_worker.AudioConverter,
        "package_m4b_with_chapters",
        staticmethod(_fake_m4b),
    )

    user_id = 1
    project_id = "abk_merge01"
    payload = _build_job_payload(project_id)
    payload["output"] = {"merge": True, "per_chapter": True, "formats": ["mp3", "m4b"]}

    job_manager = JobManager()
    job = job_manager.create_job(
        domain="audiobooks",
        queue="default",
        job_type="audiobook_generate",
        payload=payload,
        owner_user_id=str(user_id),
        priority=5,
    )

    acquired = job_manager.acquire_next_job(
        domain="audiobooks",
        queue="default",
        lease_seconds=60,
        worker_id="test-worker",
    )
    assert acquired is not None

    asyncio.run(
        audiobook_jobs_worker.process_audiobook_job(
            acquired,
            job_manager=job_manager,
            worker_id="test-worker",
        )
    )

    collections_db = CollectionsDatabase(user_id)
    outputs, _total = collections_db.list_output_artifacts(job_id=int(job["id"]), limit=50, offset=0)
    merged_audio = [
        row for row in outputs
        if row.type == "audiobook_audio"
        and json.loads(row.metadata_json or "{}").get("scope") == "merged"
    ]
    packaged = [row for row in outputs if row.type == "audiobook_package"]
    assert merged_audio
    assert packaged


def test_audiobook_worker_fails_on_m4b_packaging_error(
    user_base_dir,
    jobs_db_path,
    fake_tts,
    monkeypatch,
):
    async def _fake_m4b(*_args, **_kwargs):
        return False

    monkeypatch.setattr(
        audiobook_jobs_worker.AudioConverter,
        "package_m4b_with_chapters",
        staticmethod(_fake_m4b),
    )

    user_id = 1
    project_id = "abk_m4b_fail"
    payload = _build_job_payload(project_id)
    payload["output"] = {"merge": True, "per_chapter": False, "formats": ["m4b"]}

    job_manager = JobManager()
    job = job_manager.create_job(
        domain="audiobooks",
        queue="default",
        job_type="audiobook_generate",
        payload=payload,
        owner_user_id=str(user_id),
        priority=5,
    )

    acquired = job_manager.acquire_next_job(
        domain="audiobooks",
        queue="default",
        lease_seconds=60,
        worker_id="test-worker",
    )
    assert acquired is not None

    asyncio.run(
        audiobook_jobs_worker.process_audiobook_job(
            acquired,
            job_manager=job_manager,
            worker_id="test-worker",
        )
    )

    job_row = job_manager.get_job(int(job["id"]))
    assert job_row is not None
    assert job_row.get("status") == "failed"
    error_text = " ".join(
        str(job_row.get(key) or "")
        for key in ("error_message", "last_error", "error_code", "error")
    )
    assert "m4b_packaging_failed" in error_text

    collections_db = CollectionsDatabase(user_id)
    projects = collections_db.list_audiobook_projects(limit=10, offset=0)
    assert projects
    assert projects[0].status == "failed"


def test_audiobook_worker_time_stretch_scales_alignment(
    user_base_dir,
    jobs_db_path,
    fake_tts,
    monkeypatch,
):
    monkeypatch.setenv("AUDIOBOOK_TIME_STRETCH_MAX_RATIO", "1.2")
    called = {}

    async def _fake_time_stretch(input_path, output_path, speed_ratio):
        called["ratio"] = speed_ratio
        output_path.write_bytes(input_path.read_bytes())
        return True

    monkeypatch.setattr(
        audiobook_jobs_worker.AudioConverter,
        "time_stretch",
        staticmethod(_fake_time_stretch),
    )

    user_id = 1
    project_id = "abk_stretch01"
    payload = _build_job_payload(project_id)
    payload["chapters"][0]["speed"] = 1.1

    job_manager = JobManager()
    job = job_manager.create_job(
        domain="audiobooks",
        queue="default",
        job_type="audiobook_generate",
        payload=payload,
        owner_user_id=str(user_id),
        priority=5,
    )

    acquired = job_manager.acquire_next_job(
        domain="audiobooks",
        queue="default",
        lease_seconds=60,
        worker_id="test-worker",
    )
    assert acquired is not None

    asyncio.run(
        audiobook_jobs_worker.process_audiobook_job(
            acquired,
            job_manager=job_manager,
            worker_id="test-worker",
        )
    )

    assert called.get("ratio") == pytest.approx(1.1, rel=1e-6)

    collections_db = CollectionsDatabase(user_id)
    outputs, _total = collections_db.list_output_artifacts(job_id=int(job["id"]), limit=50, offset=0)
    alignment_rows = [row for row in outputs if row.type == "audiobook_alignment"]
    assert alignment_rows

    outputs_dir = DatabasePaths.get_user_outputs_dir(user_id)
    payload = json.loads((outputs_dir / alignment_rows[0].storage_path).read_text(encoding="utf-8"))
    words = payload.get("words") or []
    assert words
    assert words[0]["end_ms"] == 455
    if len(words) > 1:
        assert words[1]["end_ms"] == 909
