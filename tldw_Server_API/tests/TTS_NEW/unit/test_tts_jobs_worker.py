import json
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from tldw_Server_API.app.api.v1.endpoints.audio import audio_tts
from tldw_Server_API.app.api.v1.schemas.audio_schemas import OpenAISpeechRequest
from tldw_Server_API.app.core.config import settings
from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase
from tldw_Server_API.app.core.TTS import tts_jobs_worker
from tldw_Server_API.app.core.TTS.gateway_config import normalize_gateway_specs
from tldw_Server_API.app.core.TTS.gateway_preflight import (
    preflight_gateway_speech,
    reject_gateway_persistence_authority,
)
from tldw_Server_API.app.core.TTS.tts_jobs_worker import _handle_tts_job


def _gateway_specs(
    *,
    allow_discovered_models: bool = False,
    allowed_request_options: list[str] | None = None,
):
    return normalize_gateway_specs(
        {},
        {
            "company": {
                "enabled": True,
                "allow_user_api_key": True,
                "base_url": "https://speech.example.test/v1",
                "speech_path": "audio/speech",
                "default_model": "Vendor/Exact-TTS",
                "default_voice": "narrator",
                "allowed_models": None if allow_discovered_models else ["Vendor/Exact-TTS"],
                "allow_discovered_models": allow_discovered_models,
                "allowed_request_options": allowed_request_options or [],
                "capability_defaults": {
                    "formats": ["mp3"],
                    "supports_speed": True,
                },
                "fallback": {
                    "on": ["timeout"],
                    "max_attempts": 2,
                    "targets": [],
                },
            }
        },
    )


@pytest.mark.unit
def test_gateway_job_preflight_resolves_only_configured_identity() -> None:
    result = preflight_gateway_speech(
        backend="company",
        model="Vendor/Exact-TTS",
        voice=None,
        voice_supplied=False,
        response_format="mp3",
        allow_fallback=True,
        supplied_fields=frozenset(),
        gateway_specs=_gateway_specs(),
    )

    assert result.backend == "gateway:company"
    assert result.model == "Vendor/Exact-TTS"
    assert result.voice == "narrator"
    assert result.allow_fallback is False


@pytest.mark.unit
def test_gateway_job_preflight_rejects_discovery_only_model_without_network() -> None:
    with pytest.raises(ValueError, match="statically configured"):
        preflight_gateway_speech(
            backend="gateway:company",
            model="Vendor/Discovered-Only",
            voice="narrator",
            voice_supplied=True,
            response_format="mp3",
            allow_fallback=False,
            supplied_fields=frozenset({"voice"}),
            gateway_specs=_gateway_specs(allow_discovered_models=True),
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    "unsafe_value",
    [
        {"provider_overrides": {"api_key": "secret"}},
        {"metadata": {"admin_url": "https://attacker.invalid"}},
        {"metadata": {"request_headers": {"x-secret": "secret"}}},
        {"metadata": {"credential_metadata": {"revision": 1}}},
        {"metadata": {"model_path": "/private/model"}},
        {"metadata": {"auth": "bearer secret"}},
    ],
)
def test_gateway_persistence_rejects_authority_key_variants(unsafe_value) -> None:
    with pytest.raises(ValueError, match="credential or route authority"):
        reject_gateway_persistence_authority(unsafe_value)


@pytest.mark.unit
def test_gateway_persistence_allows_benign_authority_substrings() -> None:
    reject_gateway_persistence_authority(
        {
            "metadata": {
                "secretary": "narrator",
                "passwordless": True,
                "bearerTokenization": "phonetic",
            }
        }
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("allowed_pointer", "extra_params"),
    [
        (
            "/provider_overrides/apiKey",
            {"provider_overrides": {"apiKey": "distinctive-private-secret"}},
        ),
        ("/clientSecret", {"clientSecret": "distinctive-private-secret"}),
        ("/dbPassword", {"dbPassword": "distinctive-private-secret"}),
        ("/authBearer", {"authBearer": "distinctive-private-secret"}),
        ("/secretKey", {"secretKey": "distinctive-private-secret"}),
        ("/secretValue", {"secretValue": "distinctive-private-secret"}),
        ("/passwordHash", {"passwordHash": "distinctive-private-secret"}),
        ("/accessKey", {"accessKey": "distinctive-private-secret"}),
        ("/baseUri", {"baseUri": "https://attacker.invalid/v1"}),
        ("/endpointUri", {"endpointUri": "https://attacker.invalid/v1"}),
        ("/host", {"host": "attacker.invalid"}),
        ("/origin", {"origin": "https://attacker.invalid"}),
        ("/upstreamHost", {"upstreamHost": "attacker.invalid"}),
        ("/serviceOrigin", {"serviceOrigin": "https://attacker.invalid"}),
        ("/apiEndpoint", {"apiEndpoint": "https://attacker.invalid/v1"}),
    ],
)
async def test_gateway_speech_job_rejects_allowlisted_authority_extra_params(
    monkeypatch,
    allowed_pointer,
    extra_params,
) -> None:
    specs = _gateway_specs(
        allowed_request_options=[allowed_pointer],
    )

    def _shim(name: str):
        if name == "_sanitize_speech_request":
            return lambda *_args, **_kwargs: None
        if name == "_resolve_tts_byok":
            raise AssertionError("explicit gateway enqueue must not resolve credentials")
        raise KeyError(name)

    monkeypatch.setattr(audio_tts, "_audio_shim_attr", _shim)
    monkeypatch.setattr(
        audio_tts,
        "get_tts_config_manager",
        lambda: SimpleNamespace(get_gateway_specs=lambda: specs),
    )

    with pytest.raises(HTTPException) as exc_info:
        await audio_tts.create_speech_job(
            request_data=OpenAISpeechRequest(
                backend="company",
                model="Vendor/Exact-TTS",
                input="Do not persist authority.",
                extra_params=extra_params,
            ),
            request=Request(
                {"type": "http", "method": "POST", "path": "/", "headers": []}
            ),
            current_user=SimpleNamespace(id="7"),
            jm=SimpleNamespace(
                create_job=lambda **_kwargs: (_ for _ in ()).throw(
                    AssertionError("unsafe job must not be created")
                )
            ),
        )

    assert exc_info.value.status_code == 422


@pytest.mark.unit
async def test_gateway_speech_job_enqueue_is_config_only_and_secret_free(monkeypatch) -> None:
    class DummyJM:
        payload = None

        def create_job(self, **kwargs):
            self.payload = kwargs["payload"]
            return {"id": 91, "status": "queued"}

    def _shim(name: str):
        if name == "_sanitize_speech_request":
            return lambda *_args, **_kwargs: None
        if name == "_resolve_tts_byok":
            raise AssertionError("explicit gateway enqueue must not resolve credentials")
        raise KeyError(name)

    monkeypatch.setattr(audio_tts, "_audio_shim_attr", _shim)
    monkeypatch.setattr(
        audio_tts,
        "get_tts_config_manager",
        lambda: SimpleNamespace(get_gateway_specs=_gateway_specs),
    )
    manager = DummyJM()
    request = Request(
        {"type": "http", "method": "POST", "path": "/", "headers": []}
    )

    await audio_tts.create_speech_job(
        request_data=OpenAISpeechRequest(
            backend="company",
            model="Vendor/Exact-TTS",
            input="Queue this.",
        ),
        request=request,
        current_user=SimpleNamespace(id="7"),
        jm=manager,
    )

    assert manager.payload == {
        "speech_request": {
            "backend": "gateway:company",
            "model": "Vendor/Exact-TTS",
            "input": "Queue this.",
            "voice": "narrator",
            "allow_fallback": False,
            "stream": False,
        },
        "user_id": 7,
    }
    serialized = json.dumps(manager.payload).casefold()
    for forbidden in (
        "api_key",
        "authorization",
        "provider_overrides",
        "base_url",
        "speech_path",
        "headers",
        "credential_scope",
        "credential_source",
        "credential_revision",
    ):
        assert forbidden not in serialized


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
        lambda: SimpleNamespace(update_job_progress=lambda *args, **kwargs: True),
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
        lambda: SimpleNamespace(update_job_progress=lambda *args, **kwargs: True),
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

    def _capture_debug(message, *args, **kwargs):
        try:
            rendered = str(message).format(*args)
        except Exception:
            rendered = f"{message} {args}"
        debug_lines.append(rendered)

    monkeypatch.setattr(
        "tldw_Server_API.app.core.TTS.tts_jobs_worker.logger.debug",
        _capture_debug,
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


@pytest.mark.unit
async def test_gateway_tts_worker_resolves_current_owner_and_records_actual_route(
    tmp_path,
    monkeypatch,
):
    service_calls: list[dict] = []
    history_calls: list[dict] = []
    artifact_calls: list[dict] = []

    class DummyService:
        def generate_speech(self, request, **kwargs):
            service_calls.append({"request": request, "kwargs": kwargs})
            request._tts_metadata = {
                "requested_backend": "gateway:company",
                "actual_backend": "gateway:backup",
                "actual_provider": "gateway:backup",
                "model": "Vendor/Backup-TTS",
                "voice": "backup-narrator",
                "format": "mp3",
                "fallback_used": True,
                "conversion_used": True,
            }

            async def _gen():
                yield b"gateway-audio"

            return _gen()

    async def _get_service():
        return DummyService()

    class DummyHistoryDB:
        def create_tts_history_entry(self, **kwargs):
            history_calls.append(kwargs)

        def close_connection(self):
            return None

    class DummyCDB:
        def resolve_output_storage_path(self, name):
            return name

        def create_output_artifact(self, **kwargs):
            artifact_calls.append(kwargs)
            return SimpleNamespace(id=432, storage_path=kwargs["storage_path"], format=kwargs["format_"])

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(tts_jobs_worker, "get_tts_service_v2", _get_service)
    monkeypatch.setattr(tts_jobs_worker, "JobManager", lambda: SimpleNamespace(update_job_progress=lambda *a, **k: True))
    monkeypatch.setattr(tts_jobs_worker, "emit_job_event", lambda *args, **kwargs: None)
    monkeypatch.setattr(tts_jobs_worker.DatabasePaths, "get_user_outputs_dir", lambda user_id: tmp_path)
    monkeypatch.setattr(tts_jobs_worker.CollectionsDatabase, "for_user", lambda user_id: DummyCDB())
    monkeypatch.setattr(tts_jobs_worker, "_open_media_db_for_history", lambda user_id: DummyHistoryDB())
    monkeypatch.setattr(settings, "TTS_HISTORY_ENABLED", True, raising=False)
    monkeypatch.setattr(settings, "TTS_HISTORY_STORE_TEXT", True, raising=False)
    monkeypatch.setattr(settings, "TTS_HISTORY_STORE_FAILED", True, raising=False)
    monkeypatch.setattr(settings, "TTS_HISTORY_HASH_KEY", "stage4-gateway-history", raising=False)

    result = await _handle_tts_job(
        {
            "id": 58,
            "job_type": "tts_longform",
            "owner_user_id": "19",
            "payload": {
                "user_id": "99",
                "provider_hint": "malicious-legacy-provider",
                "provider_overrides": {"api_key": "must-never-be-reused"},
                "speech_request": {
                    "backend": "gateway:company",
                    "model": "Vendor/Exact-TTS",
                    "input": "Fresh credentials only",
                    "voice": "narrator",
                    "allow_fallback": True,
                    "response_format": "mp3",
                    "stream": False,
                },
            },
        }
    )

    assert service_calls[0]["kwargs"] == {
        "provider": None,
        "fallback": False,
        "provider_overrides": None,
        "user_id": 19,
    }
    history = history_calls[0]
    assert history["provider"] == "gateway:backup"
    assert history["model"] == "Vendor/Backup-TTS"
    assert history["params_json"]["requested_backend"] == "gateway:company"
    assert history["params_json"]["fallback_used"] is True
    assert history["params_json"]["conversion_used"] is True
    metadata = json.loads(artifact_calls[0]["metadata_json"])
    assert metadata["requested_backend"] == "gateway:company"
    assert metadata["actual_backend"] == "gateway:backup"
    assert metadata["fallback_used"] is True
    assert result["requested_backend"] == "gateway:company"
    assert result["actual_backend"] == "gateway:backup"
    assert result["fallback_used"] is True
    serialized = json.dumps({"history": history, "metadata": metadata, "result": result}).casefold()
    assert "must-never-be-reused" not in serialized
    assert "api_key" not in serialized
