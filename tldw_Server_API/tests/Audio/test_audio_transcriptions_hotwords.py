import io
import os as real_os
import time
from unittest.mock import call

import numpy as np
import pytest
import soundfile as sf
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.endpoints.audio.audio import router as audio_router

TEST_API_KEY = "test-api-key-1234567890"


class _LoggerStub:
    def __init__(self) -> None:
        self.debugs: list[str] = []
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def debug(self, message, *args, **kwargs) -> None:
        self.debugs.append(self._render(message, args))

    def error(self, message, *args, **kwargs) -> None:
        self.errors.append(self._render(message, args))

    def warning(self, message, *args, **kwargs) -> None:
        self.warnings.append(self._render(message, args))

    @staticmethod
    def _render(message, args) -> str:
        try:
            return str(message).format(*args)
        except (IndexError, KeyError, ValueError):
            return str(message)


class _DurationFallbackSoundFile:
    def info(self, _path):
        raise RuntimeError("soundfile info leaked /private/audio.wav")

    def read(self, _path):
        return [0.0] * 1600, 16000


class _UnreadableSoundFile:
    def info(self, _path):
        raise RuntimeError("soundfile info leaked /private/audio.wav")

    def read(self, _path):
        raise RuntimeError("soundfile read leaked /private/audio.wav")


class _LongDurationInfo:
    frames = 16000 * 121
    samplerate = 16000


class _LongDurationSoundFile:
    def info(self, _path):
        return _LongDurationInfo()


class _FailingBillingEnforcer:
    async def check_limit(self, *_args, **_kwargs):
        raise RuntimeError("billing enforcer leaked /private/audio.wav")


class _FailingUsageLog:
    def log_event(self, *_args, **_kwargs):
        raise RuntimeError("usage log leaked /private/audio.wav")


class _FailingTranscriptionAdapter:
    def transcribe_batch(self, *_args, **_kwargs):
        raise RuntimeError("adapter failure leaked /private/audio.wav")


class _FailingTranscriptionRegistry:
    def resolve_provider_for_model(self, _model):
        return "vibevoice", "vibevoice-asr", None

    def get_adapter(self, _provider):
        return _FailingTranscriptionAdapter()


class _ErrorSentinelTranscriptionAdapter:
    def transcribe_batch(self, *_args, **_kwargs):
        return {
            "text": "[Error] provider leaked /private/audio.wav",
            "language": "en",
            "segments": [],
            "diarization": {"enabled": False, "speakers": None},
            "usage": {"duration_ms": None, "tokens": None},
            "metadata": {"provider": "vibevoice", "model": "vibevoice-asr"},
        }


class _ErrorSentinelTranscriptionRegistry:
    def resolve_provider_for_model(self, _model):
        return "vibevoice", "vibevoice-asr", None

    def get_adapter(self, _provider):
        return _ErrorSentinelTranscriptionAdapter()


class _WhisperTranscriptionAdapter:
    def transcribe_batch(self, *_args, **kwargs):
        return {
            "text": "stub whisper transcript",
            "language": kwargs.get("language") or "en",
            "segments": [],
            "diarization": {"enabled": False, "speakers": None},
            "usage": {"duration_ms": None, "tokens": None},
            "metadata": {"provider": "faster-whisper", "model": kwargs.get("model") or "base"},
        }


class _WhisperTranscriptionRegistry:
    def resolve_provider_for_model(self, _model):
        return "faster-whisper", "base", None

    def get_adapter(self, _provider):
        return _WhisperTranscriptionAdapter()


class _FailingWhisperTranscriptionAdapter:
    def transcribe_batch(self, *_args, **_kwargs):
        raise RuntimeError("whisper adapter leaked /private/audio.wav")


class _FailingWhisperTranscriptionRegistry:
    def resolve_provider_for_model(self, _model):
        return "faster-whisper", "base", None

    def get_adapter(self, _provider):
        return _FailingWhisperTranscriptionAdapter()


def _make_wav_bytes(duration_sec: float = 0.1, sr: int = 16000) -> bytes:
    buf = io.BytesIO()
    data = np.zeros(int(sr * duration_sec), dtype=np.float32)
    sf.write(buf, data, sr, format="WAV")
    return buf.getvalue()


def _setup_stubbed_audio_app(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[FastAPI, dict[str, list[str] | None]]:
    monkeypatch.setenv("TEST_MODE", "true")
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("SINGLE_USER_API_KEY", TEST_API_KEY)
    monkeypatch.setenv("SINGLE_USER_FIXED_ID", "1")

    from types import SimpleNamespace
    from tldw_Server_API.app.api.v1.API_Deps.auth_deps import (
        get_auth_principal,
        get_db_transaction,
    )
    from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
    from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
    import tldw_Server_API.app.api.v1.endpoints.audio.audio as audio_ep
    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib as atlib
    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.stt_provider_adapter as stt_adapter

    captured: dict[str, list[str] | None] = {"hotwords": None}

    async def _fake_get_request_user() -> User:
        return User(id=1, username="single_user")

    async def _fake_get_auth_principal() -> AuthPrincipal:
        return AuthPrincipal(
            kind="user",
            user_id=1,
            api_key_id=None,
            username="single_user",
            email=None,
            subject="single_user",
            token_type=None,
            jti=None,
            roles=["admin"],
            permissions=["*"],
            is_admin=True,
            org_ids=[],
            team_ids=[],
        )

    async def _fake_get_db_transaction():
        yield SimpleNamespace()

    async def _allow_job(*_args, **_kwargs):
        return True, None

    async def _noop_async(*_args, **_kwargs):
        return None

    class _StubAdapter:
        def transcribe_batch(
            self,
            audio_path,
            *,
            model=None,
            language=None,
            task="transcribe",
            word_timestamps=False,
            prompt=None,
            hotwords=None,
            base_dir=None,
        ):
            captured["hotwords"] = hotwords
            return {
                "text": "stub transcript",
                "language": language or "en",
                "segments": [
                    {
                        "start_seconds": 0.0,
                        "end_seconds": 0.0,
                        "Text": "stub transcript",
                    }
                ],
                "diarization": {"enabled": False, "speakers": None},
                "usage": {"duration_ms": None, "tokens": None},
                "metadata": {"provider": "vibevoice", "model": model or "vibevoice-asr"},
            }

    class _StubRegistry:
        def resolve_provider_for_model(self, _model):
            return "vibevoice", "vibevoice-asr", None

        def get_adapter(self, _provider):
            return _StubAdapter()

    monkeypatch.setattr(audio_ep, "can_start_job", _allow_job)
    monkeypatch.setattr(audio_ep, "increment_jobs_started", _noop_async)
    monkeypatch.setattr(audio_ep, "finish_job", _noop_async)
    monkeypatch.setattr(audio_ep, "check_daily_minutes_allow", _allow_job)
    monkeypatch.setattr(audio_ep, "add_daily_minutes", _noop_async)
    monkeypatch.setattr(stt_adapter, "get_stt_provider_registry", lambda: _StubRegistry())
    monkeypatch.setattr(atlib, "convert_to_wav", lambda path, *args, **kwargs: path)

    app = FastAPI()
    app.dependency_overrides[get_request_user] = _fake_get_request_user
    app.dependency_overrides[get_auth_principal] = _fake_get_auth_principal
    app.dependency_overrides[get_db_transaction] = _fake_get_db_transaction
    app.include_router(audio_router, prefix="/api/v1/audio")
    return app, captured


@pytest.mark.unit
def test_audio_transcriptions_sanitizes_shim_lookup_logs(monkeypatch):
    import builtins
    import tldw_Server_API.app.api.v1.endpoints.audio.audio_transcriptions as audio_tx

    class _KwargLoggerStub:
        def __init__(self) -> None:
            self.debug_calls = []

        def debug(self, message, *args, **kwargs) -> None:
            self.debug_calls.append(call(message, *args, **kwargs))

    original_import = builtins.__import__
    scenarios = [
        (
            ImportError("package lookup leaked /private/audio-package.py"),
            ArithmeticError("module lookup leaked /private/audio-module.py"),
            [
                call("audio_transcriptions shim package lookup failed"),
                call("audio_transcriptions shim module lookup raised unexpected error"),
            ],
        ),
        (
            ArithmeticError("package lookup leaked /private/audio-package.py"),
            ImportError("module lookup leaked /private/audio-module.py"),
            [
                call("audio_transcriptions shim package lookup raised unexpected error"),
                call("audio_transcriptions shim module lookup failed"),
            ],
        ),
    ]

    for package_error, module_error, expected_calls in scenarios:
        logger_stub = _KwargLoggerStub()
        monkeypatch.setattr(audio_tx, "logger", logger_stub)

        def _raising_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "tldw_Server_API.app.api.v1.endpoints" and "audio" in fromlist:
                raise package_error
            if name == "tldw_Server_API.app.api.v1.endpoints.audio" and "audio" in fromlist:
                raise module_error
            return original_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", _raising_import)

        assert audio_tx._audio_shim_attr("sf") is audio_tx.sf
        assert logger_stub.debug_calls == expected_calls


@pytest.mark.unit
def test_audio_transcriptions_hotwords_csv(monkeypatch, bypass_api_limits):
    app, captured = _setup_stubbed_audio_app(monkeypatch)

    with bypass_api_limits(app), TestClient(app) as client:
        wav_bytes = _make_wav_bytes()
        headers = {"X-API-KEY": TEST_API_KEY}
        files = {"file": ("sample.wav", io.BytesIO(wav_bytes), "audio/wav")}
        data = {
            "model": "vibevoice-asr",
            "response_format": "json",
            "hotwords": "alpha, beta ,gamma",
        }
        resp = client.post(
            "/api/v1/audio/transcriptions",
            headers=headers,
            files=files,
            data=data,
        )
        if resp.status_code == 404:
            pytest.skip("audio/transcriptions endpoint not mounted in this build")
        assert resp.status_code == 200, resp.text
        assert captured["hotwords"] == ["alpha", "beta", "gamma"]


@pytest.mark.unit
def test_audio_transcriptions_hotwords_json(monkeypatch, bypass_api_limits):
    app, captured = _setup_stubbed_audio_app(monkeypatch)

    with bypass_api_limits(app), TestClient(app) as client:
        wav_bytes = _make_wav_bytes()
        headers = {"X-API-KEY": TEST_API_KEY}
        files = {"file": ("sample.wav", io.BytesIO(wav_bytes), "audio/wav")}
        data = {
            "model": "vibevoice-asr",
            "response_format": "json",
            "hotwords": "[\"alpha\", \"beta\", \" \"]",
        }
        resp = client.post(
            "/api/v1/audio/transcriptions",
            headers=headers,
            files=files,
            data=data,
        )
        if resp.status_code == 404:
            pytest.skip("audio/transcriptions endpoint not mounted in this build")
        assert resp.status_code == 200, resp.text
        assert captured["hotwords"] == ["alpha", "beta"]


@pytest.mark.unit
def test_audio_transcriptions_sanitizes_test_mode_canonical_path_log(
    monkeypatch,
    bypass_api_limits,
):
    app, _captured = _setup_stubbed_audio_app(monkeypatch)
    import tldw_Server_API.app.api.v1.endpoints.audio.audio_transcriptions as audio_tx

    logger_stub = _LoggerStub()
    monkeypatch.setattr(audio_tx, "logger", logger_stub)

    with bypass_api_limits(app), TestClient(app) as client:
        wav_bytes = _make_wav_bytes()
        headers = {"X-API-KEY": TEST_API_KEY}
        files = {"file": ("sample.wav", io.BytesIO(wav_bytes), "audio/wav")}
        data = {
            "model": "vibevoice-asr",
            "response_format": "json",
        }
        resp = client.post(
            "/api/v1/audio/transcriptions",
            headers=headers,
            files=files,
            data=data,
        )
        if resp.status_code == 404:
            pytest.skip("audio/transcriptions endpoint not mounted in this build")
        assert resp.status_code == 200, resp.text

    canonical_path_logs = [
        msg
        for msg in logger_stub.debugs
        if msg.startswith("TEST_MODE: canonical audio path resolved")
    ]
    assert canonical_path_logs == ["TEST_MODE: canonical audio path resolved"]


@pytest.mark.unit
def test_audio_transcriptions_sanitizes_billing_recheck_fail_open_log(
    monkeypatch,
    bypass_api_limits,
):
    app, _captured = _setup_stubbed_audio_app(monkeypatch)
    from tldw_Server_API.app.api.v1.API_Deps.billing_deps import get_billing_org_id
    import tldw_Server_API.app.api.v1.endpoints.audio as audio_pkg
    import tldw_Server_API.app.api.v1.endpoints.audio.audio_transcriptions as audio_tx

    logger_stub = _LoggerStub()
    app.dependency_overrides[get_billing_org_id] = lambda: 123
    monkeypatch.setattr(audio_tx, "logger", logger_stub)
    monkeypatch.setattr(audio_pkg, "sf", _LongDurationSoundFile())
    monkeypatch.setattr(audio_tx, "enforcement_enabled", lambda: True)
    monkeypatch.setattr(audio_tx, "get_billing_enforcer", lambda: _FailingBillingEnforcer())

    with bypass_api_limits(app), TestClient(app) as client:
        wav_bytes = _make_wav_bytes()
        headers = {"X-API-KEY": TEST_API_KEY}
        files = {"file": ("sample.wav", io.BytesIO(wav_bytes), "audio/wav")}
        data = {
            "model": "vibevoice-asr",
            "response_format": "json",
        }
        resp = client.post(
            "/api/v1/audio/transcriptions",
            headers=headers,
            files=files,
            data=data,
        )
        if resp.status_code == 404:
            pytest.skip("audio/transcriptions endpoint not mounted in this build")
        assert resp.status_code == 200, resp.text

    billing_logs = [
        msg
        for msg in logger_stub.debugs
        if msg.startswith("Billing secondary minutes check failed")
    ]
    assert billing_logs == ["Billing secondary minutes check failed; allowing by default"]


@pytest.mark.unit
def test_audio_transcriptions_sanitizes_custom_vocabulary_failure_log(
    monkeypatch,
    bypass_api_limits,
):
    app, _captured = _setup_stubbed_audio_app(monkeypatch)
    import tldw_Server_API.app.api.v1.endpoints.audio.audio_transcriptions as audio_tx
    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Custom_Vocabulary as cv

    logger_stub = _LoggerStub()
    monkeypatch.setattr(audio_tx, "logger", logger_stub)

    def _raise_custom_vocabulary_failure(_text):
        raise RuntimeError("custom vocabulary leaked /private/audio.wav")

    monkeypatch.setattr(cv, "postprocess_text_if_enabled", _raise_custom_vocabulary_failure)

    with bypass_api_limits(app), TestClient(app) as client:
        wav_bytes = _make_wav_bytes()
        headers = {"X-API-KEY": TEST_API_KEY}
        files = {"file": ("sample.wav", io.BytesIO(wav_bytes), "audio/wav")}
        data = {
            "model": "vibevoice-asr",
            "response_format": "json",
        }
        resp = client.post(
            "/api/v1/audio/transcriptions",
            headers=headers,
            files=files,
            data=data,
        )
        if resp.status_code == 404:
            pytest.skip("audio/transcriptions endpoint not mounted in this build")
        assert resp.status_code == 200, resp.text

    custom_vocabulary_logs = [
        msg
        for msg in logger_stub.debugs
        if msg.startswith("Custom vocabulary postprocessing failed")
    ]
    assert custom_vocabulary_logs == [
        "Custom vocabulary postprocessing failed; continuing without it"
    ]


@pytest.mark.unit
def test_audio_transcriptions_sanitizes_auto_segmentation_failure_log(
    monkeypatch,
    bypass_api_limits,
):
    app, _captured = _setup_stubbed_audio_app(monkeypatch)
    import tldw_Server_API.app.api.v1.endpoints.audio.audio_transcriptions as audio_tx

    logger_stub = _LoggerStub()
    monkeypatch.setattr(audio_tx, "logger", logger_stub)

    async def _raise_tree_segmentation_failure(*_args, **_kwargs):
        raise RuntimeError("tree segmentation leaked /private/audio.wav")

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Transcript_TreeSegmentation.TreeSegmenter.create_async",
        _raise_tree_segmentation_failure,
    )

    with bypass_api_limits(app), TestClient(app) as client:
        wav_bytes = _make_wav_bytes()
        headers = {"X-API-KEY": TEST_API_KEY}
        files = {"file": ("sample.wav", io.BytesIO(wav_bytes), "audio/wav")}
        data = {
            "model": "vibevoice-asr",
            "response_format": "json",
            "segment": "true",
        }
        resp = client.post(
            "/api/v1/audio/transcriptions",
            headers=headers,
            files=files,
            data=data,
        )
        if resp.status_code == 404:
            pytest.skip("audio/transcriptions endpoint not mounted in this build")
        assert resp.status_code == 200, resp.text
        assert "segmentation" not in resp.json()

    segmentation_logs = [
        msg
        for msg in logger_stub.warnings
        if msg.startswith("Auto-segmentation failed")
    ]
    assert segmentation_logs == ["Auto-segmentation failed; continuing without it"]


@pytest.mark.unit
def test_audio_transcriptions_sanitizes_heartbeat_interval_failure_log(
    monkeypatch,
    bypass_api_limits,
):
    app, _captured = _setup_stubbed_audio_app(monkeypatch)
    import tldw_Server_API.app.api.v1.endpoints.audio.audio_transcriptions as audio_tx

    logger_stub = _LoggerStub()
    monkeypatch.setattr(audio_tx, "logger", logger_stub)

    original_shim_attr = audio_tx._audio_shim_attr

    def _shim_attr(name):
        if name == "get_job_heartbeat_interval_seconds":
            def _raise_interval_failure():
                raise RuntimeError("heartbeat interval leaked /private/rg.sock")

            return _raise_interval_failure
        return original_shim_attr(name)

    monkeypatch.setattr(audio_tx, "_audio_shim_attr", _shim_attr)

    with bypass_api_limits(app), TestClient(app) as client:
        wav_bytes = _make_wav_bytes()
        headers = {"X-API-KEY": TEST_API_KEY}
        files = {"file": ("sample.wav", io.BytesIO(wav_bytes), "audio/wav")}
        data = {
            "model": "vibevoice-asr",
            "response_format": "json",
        }
        resp = client.post(
            "/api/v1/audio/transcriptions",
            headers=headers,
            files=files,
            data=data,
        )
        if resp.status_code == 404:
            pytest.skip("audio/transcriptions endpoint not mounted in this build")
        assert resp.status_code == 200, resp.text

    heartbeat_logs = [
        msg
        for msg in logger_stub.debugs
        if "job heartbeat" in msg or "get_job_heartbeat_interval_seconds" in msg
    ]
    assert heartbeat_logs == [
        "audio.transcriptions job heartbeat interval lookup failed; skipping job heartbeat"
    ]


@pytest.mark.unit
def test_audio_transcriptions_sanitizes_heartbeat_task_start_failure_log(
    monkeypatch,
    bypass_api_limits,
):
    app, _captured = _setup_stubbed_audio_app(monkeypatch)
    import tldw_Server_API.app.api.v1.endpoints.audio.audio_transcriptions as audio_tx

    logger_stub = _LoggerStub()
    monkeypatch.setattr(audio_tx, "logger", logger_stub)

    original_shim_attr = audio_tx._audio_shim_attr

    def _shim_attr(name):
        if name == "get_job_heartbeat_interval_seconds":
            return lambda: 10
        return original_shim_attr(name)

    def _raise_create_task(coro):
        coro.close()
        raise RuntimeError("heartbeat task leaked /private/rg-task")

    monkeypatch.setattr(audio_tx, "_audio_shim_attr", _shim_attr)
    monkeypatch.setattr(audio_tx.asyncio, "create_task", _raise_create_task)

    with bypass_api_limits(app), TestClient(app) as client:
        wav_bytes = _make_wav_bytes()
        headers = {"X-API-KEY": TEST_API_KEY}
        files = {"file": ("sample.wav", io.BytesIO(wav_bytes), "audio/wav")}
        data = {
            "model": "vibevoice-asr",
            "response_format": "json",
        }
        resp = client.post(
            "/api/v1/audio/transcriptions",
            headers=headers,
            files=files,
            data=data,
        )
        if resp.status_code == 404:
            pytest.skip("audio/transcriptions endpoint not mounted in this build")
        assert resp.status_code == 200, resp.text

    heartbeat_logs = [
        msg
        for msg in logger_stub.debugs
        if "job heartbeat task" in msg or "failed to start job heartbeat" in msg
    ]
    assert heartbeat_logs == [
        "audio.transcriptions failed to start job heartbeat task; continuing without heartbeat"
    ]


@pytest.mark.unit
def test_audio_transcriptions_sanitizes_heartbeat_jobs_failure_log(
    monkeypatch,
    bypass_api_limits,
):
    app, _captured = _setup_stubbed_audio_app(monkeypatch)
    import tldw_Server_API.app.api.v1.endpoints.audio.audio_transcriptions as audio_tx

    logger_stub = _LoggerStub()
    monkeypatch.setattr(audio_tx, "logger", logger_stub)

    original_shim_attr = audio_tx._audio_shim_attr
    heartbeat_calls = 0
    heartbeat_observed = audio_tx.asyncio.Event()

    async def _slow_increment_jobs_started(*_args, **_kwargs):
        await audio_tx.asyncio.wait_for(heartbeat_observed.wait(), timeout=1.0)

    async def _failing_heartbeat_jobs(*_args, **_kwargs):
        nonlocal heartbeat_calls
        heartbeat_calls += 1
        heartbeat_observed.set()
        if heartbeat_calls == 1:
            raise RuntimeError("heartbeat job leaked /private/rg-heartbeat")
        raise audio_tx.asyncio.CancelledError()

    def _shim_attr(name):
        if name == "get_job_heartbeat_interval_seconds":
            return lambda: 0.001
        if name == "increment_jobs_started":
            return _slow_increment_jobs_started
        if name == "heartbeat_jobs":
            return _failing_heartbeat_jobs
        return original_shim_attr(name)

    monkeypatch.setattr(audio_tx, "_audio_shim_attr", _shim_attr)

    with bypass_api_limits(app), TestClient(app) as client:
        wav_bytes = _make_wav_bytes()
        headers = {"X-API-KEY": TEST_API_KEY}
        files = {"file": ("sample.wav", io.BytesIO(wav_bytes), "audio/wav")}
        data = {
            "model": "vibevoice-asr",
            "response_format": "json",
        }
        resp = client.post(
            "/api/v1/audio/transcriptions",
            headers=headers,
            files=files,
            data=data,
        )
        if resp.status_code == 404:
            pytest.skip("audio/transcriptions endpoint not mounted in this build")
        assert resp.status_code == 200, resp.text

        deadline = time.monotonic() + 1.0
        heartbeat_logs = []
        while time.monotonic() < deadline:
            heartbeat_logs = [
                msg
                for msg in logger_stub.debugs
                if msg.startswith("audio.transcriptions heartbeat_jobs failed")
                or msg.startswith("audio.transcriptions job heartbeat failed")
            ]
            if heartbeat_logs:
                break
            time.sleep(0.01)

    assert heartbeat_logs == ["audio.transcriptions job heartbeat failed; continuing"]


@pytest.mark.unit
def test_audio_transcriptions_sanitizes_whisper_preflight_failure_log(
    monkeypatch,
    bypass_api_limits,
):
    app, _captured = _setup_stubbed_audio_app(monkeypatch)
    import tldw_Server_API.app.api.v1.endpoints.audio.audio_transcriptions as audio_tx
    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Files as audio_files
    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.stt_provider_adapter as stt_adapter

    logger_stub = _LoggerStub()
    monkeypatch.setattr(audio_tx, "logger", logger_stub)
    monkeypatch.setattr(stt_adapter, "get_stt_provider_registry", lambda: _WhisperTranscriptionRegistry())

    def _raise_model_status_failure(_model):
        raise RuntimeError("whisper preflight leaked /private/models/base")

    monkeypatch.setattr(audio_files, "check_transcription_model_status", _raise_model_status_failure)

    with bypass_api_limits(app), TestClient(app) as client:
        wav_bytes = _make_wav_bytes()
        headers = {"X-API-KEY": TEST_API_KEY}
        files = {"file": ("sample.wav", io.BytesIO(wav_bytes), "audio/wav")}
        data = {
            "model": "base",
            "response_format": "json",
        }
        resp = client.post(
            "/api/v1/audio/transcriptions",
            headers=headers,
            files=files,
            data=data,
        )
        if resp.status_code == 404:
            pytest.skip("audio/transcriptions endpoint not mounted in this build")
        assert resp.status_code == 200, resp.text

    preflight_logs = [
        msg
        for msg in logger_stub.debugs
        if msg.startswith("Whisper model preflight check failed")
    ]
    assert preflight_logs == ["Whisper model preflight check failed; proceeding without it"]


@pytest.mark.unit
def test_audio_transcriptions_sanitizes_whisper_adapter_failure_log(
    monkeypatch,
    bypass_api_limits,
):
    app, _captured = _setup_stubbed_audio_app(monkeypatch)
    import tldw_Server_API.app.api.v1.endpoints.audio.audio_transcriptions as audio_tx
    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Files as audio_files
    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.stt_provider_adapter as stt_adapter

    logger_stub = _LoggerStub()
    monkeypatch.setattr(audio_tx, "logger", logger_stub)
    monkeypatch.setattr(stt_adapter, "get_stt_provider_registry", lambda: _FailingWhisperTranscriptionRegistry())
    monkeypatch.setattr(
        audio_files,
        "check_transcription_model_status",
        lambda _model: {"available": True, "model": "base", "message": "cached"},
    )

    with bypass_api_limits(app), TestClient(app) as client:
        wav_bytes = _make_wav_bytes()
        headers = {"X-API-KEY": TEST_API_KEY}
        files = {"file": ("sample.wav", io.BytesIO(wav_bytes), "audio/wav")}
        data = {
            "model": "base",
            "response_format": "json",
        }
        resp = client.post(
            "/api/v1/audio/transcriptions",
            headers=headers,
            files=files,
            data=data,
        )
        if resp.status_code == 404:
            pytest.skip("audio/transcriptions endpoint not mounted in this build")
        assert resp.status_code == 500
        assert resp.json()["detail"]["message"] == "Whisper transcription failed"

    whisper_logs = [
        msg
        for msg in logger_stub.errors
        if msg.startswith("Whisper transcription failed")
    ]
    assert whisper_logs == ["Whisper transcription failed"]


@pytest.mark.unit
def test_audio_transcriptions_sanitizes_outer_failure_log(
    monkeypatch,
    bypass_api_limits,
):
    app, _captured = _setup_stubbed_audio_app(monkeypatch)
    import tldw_Server_API.app.api.v1.endpoints.audio.audio_transcriptions as audio_tx

    logger_stub = _LoggerStub()
    monkeypatch.setattr(audio_tx, "logger", logger_stub)

    def _raise_suffix_failure(*_args, **_kwargs):
        raise RuntimeError("suffix resolver leaked /private/upload.wav")

    monkeypatch.setattr(audio_tx, "_resolve_audio_upload_suffix", _raise_suffix_failure)

    with bypass_api_limits(app), TestClient(app) as client:
        wav_bytes = _make_wav_bytes()
        headers = {"X-API-KEY": TEST_API_KEY}
        files = {"file": ("sample.wav", io.BytesIO(wav_bytes), "audio/wav")}
        data = {
            "model": "vibevoice-asr",
            "response_format": "json",
        }
        resp = client.post(
            "/api/v1/audio/transcriptions",
            headers=headers,
            files=files,
            data=data,
        )
        if resp.status_code == 404:
            pytest.skip("audio/transcriptions endpoint not mounted in this build")
        assert resp.status_code == 500
        assert resp.json()["detail"]["message"] == "Transcription failed"

    outer_logs = [
        msg
        for msg in logger_stub.errors
        if msg.startswith("Error during transcription")
    ]
    assert outer_logs == ["Error during transcription"]


@pytest.mark.unit
def test_audio_transcriptions_sanitizes_temp_cleanup_failure_log(
    monkeypatch,
    bypass_api_limits,
):
    app, _captured = _setup_stubbed_audio_app(monkeypatch)
    import tldw_Server_API.app.api.v1.endpoints.audio.audio_transcriptions as audio_tx

    logger_stub = _LoggerStub()
    monkeypatch.setattr(audio_tx, "logger", logger_stub)

    class _OSStub:
        path = real_os.path

        @staticmethod
        def remove(_path):
            raise OSError("temp cleanup leaked /private/upload.wav")

    monkeypatch.setattr(audio_tx, "os", _OSStub)

    with bypass_api_limits(app), TestClient(app) as client:
        wav_bytes = _make_wav_bytes()
        headers = {"X-API-KEY": TEST_API_KEY}
        files = {"file": ("sample.wav", io.BytesIO(wav_bytes), "audio/wav")}
        data = {
            "model": "vibevoice-asr",
            "response_format": "json",
        }
        resp = client.post(
            "/api/v1/audio/transcriptions",
            headers=headers,
            files=files,
            data=data,
        )
        if resp.status_code == 404:
            pytest.skip("audio/transcriptions endpoint not mounted in this build")
        assert resp.status_code == 200, resp.text

    cleanup_logs = [
        msg
        for msg in logger_stub.warnings
        if msg.startswith("Failed to remove temp audio file")
    ]
    assert cleanup_logs == ["Failed to remove temp audio file"]


@pytest.mark.unit
def test_audio_transcriptions_sanitizes_temp_cleanup_metrics_failure_log(
    monkeypatch,
    bypass_api_limits,
):
    app, _captured = _setup_stubbed_audio_app(monkeypatch)
    import tldw_Server_API.app.api.v1.endpoints.audio.audio_transcriptions as audio_tx

    logger_stub = _LoggerStub()
    monkeypatch.setattr(audio_tx, "logger", logger_stub)

    class _OSStub:
        path = real_os.path

        @staticmethod
        def remove(_path):
            raise OSError("temp cleanup leaked /private/upload.wav")

    def _raise_metrics_failure(*_args, **_kwargs):
        raise RuntimeError("metrics cleanup leaked /private/metrics.sock")

    monkeypatch.setattr(audio_tx, "os", _OSStub)
    monkeypatch.setattr(audio_tx, "increment_counter", _raise_metrics_failure)

    with bypass_api_limits(app), TestClient(app) as client:
        wav_bytes = _make_wav_bytes()
        headers = {"X-API-KEY": TEST_API_KEY}
        files = {"file": ("sample.wav", io.BytesIO(wav_bytes), "audio/wav")}
        data = {
            "model": "vibevoice-asr",
            "response_format": "json",
        }
        resp = client.post(
            "/api/v1/audio/transcriptions",
            headers=headers,
            files=files,
            data=data,
        )
        if resp.status_code == 404:
            pytest.skip("audio/transcriptions endpoint not mounted in this build")
        assert resp.status_code == 200, resp.text

    metrics_logs = [
        msg
        for msg in logger_stub.debugs
        if msg.startswith("metrics increment failed (audio tempfile_remove_failed)")
    ]
    assert metrics_logs == ["metrics increment failed (audio tempfile_remove_failed)"]


@pytest.mark.unit
def test_audio_transcriptions_sanitizes_canonical_cleanup_failure_log(
    monkeypatch,
    bypass_api_limits,
):
    app, _captured = _setup_stubbed_audio_app(monkeypatch)
    import tldw_Server_API.app.api.v1.endpoints.audio.audio_transcriptions as audio_tx
    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib as atlib

    logger_stub = _LoggerStub()
    monkeypatch.setattr(audio_tx, "logger", logger_stub)
    created_canonical_paths: list[str] = []

    def _copy_to_canonical(path, *_args, **_kwargs):
        canonical_path = f"{path}.canonical.wav"
        with open(path, "rb") as src, open(canonical_path, "wb") as dst:
            dst.write(src.read())
        created_canonical_paths.append(canonical_path)
        return canonical_path

    class _OSStub:
        path = real_os.path

        @staticmethod
        def remove(path):
            if str(path).endswith(".canonical.wav"):
                raise OSError("canonical cleanup leaked /private/canonical.wav")
            real_os.remove(path)

    monkeypatch.setattr(atlib, "convert_to_wav", _copy_to_canonical)
    monkeypatch.setattr(audio_tx, "os", _OSStub)

    try:
        with bypass_api_limits(app), TestClient(app) as client:
            wav_bytes = _make_wav_bytes()
            headers = {"X-API-KEY": TEST_API_KEY}
            files = {"file": ("sample.wav", io.BytesIO(wav_bytes), "audio/wav")}
            data = {
                "model": "vibevoice-asr",
                "response_format": "json",
            }
            resp = client.post(
                "/api/v1/audio/transcriptions",
                headers=headers,
                files=files,
                data=data,
            )
            if resp.status_code == 404:
                pytest.skip("audio/transcriptions endpoint not mounted in this build")
            assert resp.status_code == 200, resp.text
    finally:
        for canonical_path in created_canonical_paths:
            if real_os.path.exists(canonical_path):
                real_os.remove(canonical_path)

    cleanup_logs = [
        msg
        for msg in logger_stub.warnings
        if msg.startswith("Failed to remove canonical audio file")
    ]
    assert cleanup_logs == ["Failed to remove canonical audio file"]


@pytest.mark.unit
def test_audio_transcriptions_sanitizes_malformed_timestamp_granularity_log(
    monkeypatch,
    bypass_api_limits,
):
    app, _captured = _setup_stubbed_audio_app(monkeypatch)
    import tldw_Server_API.app.api.v1.endpoints.audio.audio_transcriptions as audio_tx

    logger_stub = _LoggerStub()
    monkeypatch.setattr(audio_tx, "logger", logger_stub)

    with bypass_api_limits(app), TestClient(app) as client:
        wav_bytes = _make_wav_bytes()
        headers = {"X-API-KEY": TEST_API_KEY}
        files = {"file": ("sample.wav", io.BytesIO(wav_bytes), "audio/wav")}
        data = {
            "model": "vibevoice-asr",
            "response_format": "json",
            "timestamp_granularities": '["segment",',
        }
        resp = client.post(
            "/api/v1/audio/transcriptions",
            headers=headers,
            files=files,
            data=data,
        )
        if resp.status_code == 404:
            pytest.skip("audio/transcriptions endpoint not mounted in this build")
        assert resp.status_code == 200, resp.text

    parse_logs = [
        msg
        for msg in logger_stub.debugs
        if msg.startswith("Failed to parse timestamp_granularities")
    ]
    assert parse_logs == ["Failed to parse timestamp_granularities; defaulting to 'segment'"]


@pytest.mark.unit
def test_audio_transcriptions_sanitizes_malformed_hotwords_json_log(
    monkeypatch,
    bypass_api_limits,
):
    app, captured = _setup_stubbed_audio_app(monkeypatch)
    import tldw_Server_API.app.api.v1.endpoints.audio.audio_transcriptions as audio_tx

    logger_stub = _LoggerStub()
    monkeypatch.setattr(audio_tx, "logger", logger_stub)

    with bypass_api_limits(app), TestClient(app) as client:
        wav_bytes = _make_wav_bytes()
        headers = {"X-API-KEY": TEST_API_KEY}
        files = {"file": ("sample.wav", io.BytesIO(wav_bytes), "audio/wav")}
        data = {
            "model": "vibevoice-asr",
            "response_format": "json",
            "hotwords": '["alpha",',
        }
        resp = client.post(
            "/api/v1/audio/transcriptions",
            headers=headers,
            files=files,
            data=data,
        )
        if resp.status_code == 404:
            pytest.skip("audio/transcriptions endpoint not mounted in this build")
        assert resp.status_code == 200, resp.text
        assert captured["hotwords"] == ['["alpha"']

    parse_logs = [
        msg
        for msg in logger_stub.debugs
        if msg.startswith("Failed to parse hotwords JSON")
    ]
    assert parse_logs == ["Failed to parse hotwords JSON; falling back to CSV parsing"]


@pytest.mark.unit
def test_audio_transcriptions_sanitizes_soundfile_info_fallback_log(
    monkeypatch,
    bypass_api_limits,
):
    app, _captured = _setup_stubbed_audio_app(monkeypatch)
    import tldw_Server_API.app.api.v1.endpoints.audio as audio_pkg
    import tldw_Server_API.app.api.v1.endpoints.audio.audio_transcriptions as audio_tx

    logger_stub = _LoggerStub()
    monkeypatch.setattr(audio_tx, "logger", logger_stub)
    monkeypatch.setattr(audio_pkg, "sf", _DurationFallbackSoundFile())

    with bypass_api_limits(app), TestClient(app) as client:
        wav_bytes = _make_wav_bytes()
        headers = {"X-API-KEY": TEST_API_KEY}
        files = {"file": ("sample.wav", io.BytesIO(wav_bytes), "audio/wav")}
        data = {
            "model": "vibevoice-asr",
            "response_format": "json",
        }
        resp = client.post(
            "/api/v1/audio/transcriptions",
            headers=headers,
            files=files,
            data=data,
        )
        if resp.status_code == 404:
            pytest.skip("audio/transcriptions endpoint not mounted in this build")
        assert resp.status_code == 200, resp.text

    duration_logs = [
        msg
        for msg in logger_stub.debugs
        if msg.startswith("soundfile.info failed")
    ]
    assert duration_logs == ["soundfile.info failed; falling back to read for duration"]


@pytest.mark.unit
def test_audio_transcriptions_sanitizes_soundfile_read_fallback_log(
    monkeypatch,
    bypass_api_limits,
):
    app, _captured = _setup_stubbed_audio_app(monkeypatch)
    import tldw_Server_API.app.api.v1.endpoints.audio as audio_pkg
    import tldw_Server_API.app.api.v1.endpoints.audio.audio_transcriptions as audio_tx

    logger_stub = _LoggerStub()
    monkeypatch.setattr(audio_tx, "logger", logger_stub)
    monkeypatch.setattr(audio_pkg, "sf", _UnreadableSoundFile())

    with bypass_api_limits(app), TestClient(app) as client:
        wav_bytes = _make_wav_bytes()
        headers = {"X-API-KEY": TEST_API_KEY}
        files = {"file": ("sample.wav", io.BytesIO(wav_bytes), "audio/wav")}
        data = {
            "model": "vibevoice-asr",
            "response_format": "json",
        }
        resp = client.post(
            "/api/v1/audio/transcriptions",
            headers=headers,
            files=files,
            data=data,
        )
        if resp.status_code == 404:
            pytest.skip("audio/transcriptions endpoint not mounted in this build")
        assert resp.status_code == 200, resp.text

    duration_logs = [
        msg
        for msg in logger_stub.debugs
        if msg.startswith("Failed to compute audio duration")
    ]
    assert duration_logs == ["Failed to compute audio duration; defaulting to 0"]


@pytest.mark.unit
async def test_create_translation_sanitizes_usage_log_failure(monkeypatch):
    import tldw_Server_API.app.api.v1.endpoints.audio.audio_transcriptions as audio_tx

    logger_stub = _LoggerStub()
    monkeypatch.setattr(audio_tx, "logger", logger_stub)

    async def _fake_create_transcription(**kwargs):
        return {"delegated": kwargs["task"]}

    monkeypatch.setattr(audio_tx, "create_transcription", _fake_create_transcription)

    result = await audio_tx.create_translation(
        request=object(),
        file=object(),
        model="whisper-1",
        prompt=None,
        response_format="json",
        temperature=0.0,
        current_user=object(),
        principal=object(),
        db=object(),
        usage_log=_FailingUsageLog(),
        billing_org_id=None,
    )

    assert result == {"delegated": "translate"}
    assert logger_stub.debugs == ["usage_log audio.translations failed"]


@pytest.mark.unit
def test_audio_transcriptions_sanitizes_adapter_failure_log(
    monkeypatch,
    bypass_api_limits,
):
    app, _captured = _setup_stubbed_audio_app(monkeypatch)
    import tldw_Server_API.app.api.v1.endpoints.audio.audio_transcriptions as audio_tx
    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.stt_provider_adapter as stt_adapter

    logger_stub = _LoggerStub()
    monkeypatch.setattr(audio_tx, "logger", logger_stub)
    monkeypatch.setattr(
        stt_adapter,
        "get_stt_provider_registry",
        lambda: _FailingTranscriptionRegistry(),
    )

    with bypass_api_limits(app), TestClient(app) as client:
        wav_bytes = _make_wav_bytes()
        headers = {"X-API-KEY": TEST_API_KEY}
        files = {"file": ("sample.wav", io.BytesIO(wav_bytes), "audio/wav")}
        data = {
            "model": "vibevoice-asr",
            "response_format": "json",
        }
        resp = client.post(
            "/api/v1/audio/transcriptions",
            headers=headers,
            files=files,
            data=data,
        )
        if resp.status_code == 404:
            pytest.skip("audio/transcriptions endpoint not mounted in this build")
        assert resp.status_code == 500, resp.text

    failure_logs = [
        msg
        for msg in logger_stub.errors
        if msg.startswith("Transcription failed")
    ]
    assert failure_logs == ["Transcription failed for STT provider"]


@pytest.mark.unit
def test_audio_transcriptions_sanitizes_error_sentinel_log(
    monkeypatch,
    bypass_api_limits,
):
    app, _captured = _setup_stubbed_audio_app(monkeypatch)
    import tldw_Server_API.app.api.v1.endpoints.audio.audio_transcriptions as audio_tx
    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.stt_provider_adapter as stt_adapter

    logger_stub = _LoggerStub()
    monkeypatch.setattr(audio_tx, "logger", logger_stub)
    monkeypatch.setattr(
        stt_adapter,
        "get_stt_provider_registry",
        lambda: _ErrorSentinelTranscriptionRegistry(),
    )

    with bypass_api_limits(app), TestClient(app) as client:
        wav_bytes = _make_wav_bytes()
        headers = {"X-API-KEY": TEST_API_KEY}
        files = {"file": ("sample.wav", io.BytesIO(wav_bytes), "audio/wav")}
        data = {
            "model": "vibevoice-asr",
            "response_format": "json",
        }
        resp = client.post(
            "/api/v1/audio/transcriptions",
            headers=headers,
            files=files,
            data=data,
        )
        if resp.status_code == 404:
            pytest.skip("audio/transcriptions endpoint not mounted in this build")
        assert resp.status_code == 500, resp.text

    failure_logs = [
        msg
        for msg in logger_stub.errors
        if msg.startswith("Transcription")
    ]
    assert failure_logs == ["Transcription returned error sentinel"]
