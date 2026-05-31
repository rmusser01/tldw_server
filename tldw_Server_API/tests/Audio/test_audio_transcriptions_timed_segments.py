import io
import importlib
import os
import sys
import types

import numpy as np
import pytest
import soundfile as sf
from fastapi import FastAPI
from fastapi.testclient import TestClient
from typing import Any

TEST_API_KEY = os.getenv("TLDW_TEST_AUDIO_API_KEY", "unit-test-auth-token")


def _make_wav_bytes(duration_sec: float = 0.1, sr: int = 16000) -> bytes:
    """Return a short silent WAV payload for upload tests."""
    buf = io.BytesIO()
    data = np.zeros(int(sr * duration_sec), dtype=np.float32)
    sf.write(buf, data, sr, format="WAV")
    return buf.getvalue()


def _setup_stubbed_audio_app(monkeypatch: pytest.MonkeyPatch) -> FastAPI:
    """Create a FastAPI app wired to stubbed transcription dependencies."""
    monkeypatch.setenv("TEST_MODE", "true")
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("SINGLE_USER_API_KEY", TEST_API_KEY)
    monkeypatch.setenv("SINGLE_USER_FIXED_ID", "1")

    # Keep the test import path independent from optional redis dependency.
    if "redis" not in sys.modules:
        class _RedisError(Exception):
            pass

        class _RedisConnectionError(_RedisError):
            pass

        redis_mod = types.ModuleType("redis")
        redis_asyncio_mod = types.ModuleType("redis.asyncio")
        redis_exceptions_mod = types.ModuleType("redis.exceptions")
        redis_asyncio_mod.Redis = object  # type: ignore[attr-defined]
        redis_asyncio_mod.from_url = lambda *_args, **_kwargs: None  # type: ignore[attr-defined]
        redis_exceptions_mod.RedisError = _RedisError  # type: ignore[attr-defined]
        redis_exceptions_mod.ConnectionError = _RedisConnectionError  # type: ignore[attr-defined]
        redis_mod.RedisError = _RedisError  # type: ignore[attr-defined]
        redis_mod.asyncio = redis_asyncio_mod  # type: ignore[attr-defined]
        redis_mod.exceptions = redis_exceptions_mod  # type: ignore[attr-defined]
        sys.modules["redis"] = redis_mod
        sys.modules["redis.asyncio"] = redis_asyncio_mod
        sys.modules["redis.exceptions"] = redis_exceptions_mod

    if "argon2" not in sys.modules:
        class _FakeVerificationError(Exception):
            pass

        class _FakeVerifyMismatchError(_FakeVerificationError):
            pass

        class _FakePasswordHasher:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                _ = (args, kwargs)

            def hash(self, password: str) -> str:
                return f"fake-hash::{password}"

            def verify(self, password_hash: str, password: str) -> bool:
                return password_hash == f"fake-hash::{password}"

            def check_needs_rehash(self, _password_hash: str) -> bool:
                return False

        argon2_mod = types.ModuleType("argon2")
        argon2_exc_mod = types.ModuleType("argon2.exceptions")
        argon2_mod.PasswordHasher = _FakePasswordHasher  # type: ignore[attr-defined]
        argon2_exc_mod.VerificationError = _FakeVerificationError  # type: ignore[attr-defined]
        argon2_exc_mod.VerifyMismatchError = _FakeVerifyMismatchError  # type: ignore[attr-defined]
        argon2_mod.exceptions = argon2_exc_mod  # type: ignore[attr-defined]
        sys.modules["argon2"] = argon2_mod
        sys.modules["argon2.exceptions"] = argon2_exc_mod

    if "passlib" not in sys.modules:
        class _FakeCryptContext:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                _ = (args, kwargs)

            def hash(self, password: str) -> str:
                return f"fake-hash::{password}"

            def verify(self, _password: str, _password_hash: str) -> bool:
                return True

        passlib_mod = types.ModuleType("passlib")
        passlib_context_mod = types.ModuleType("passlib.context")
        passlib_context_mod.CryptContext = _FakeCryptContext  # type: ignore[attr-defined]
        passlib_mod.context = passlib_context_mod  # type: ignore[attr-defined]
        sys.modules["passlib"] = passlib_mod
        sys.modules["passlib.context"] = passlib_context_mod

    if "PIL" not in sys.modules:
        pil_mod = types.ModuleType("PIL")
        pil_image_mod = types.ModuleType("PIL.Image")
        pil_image_mod.Image = type("Image", (), {})  # type: ignore[attr-defined]
        pil_mod.Image = pil_image_mod  # type: ignore[attr-defined]
        sys.modules["PIL"] = pil_mod
        sys.modules["PIL.Image"] = pil_image_mod

    if "jinja2" not in sys.modules:
        class _FakeTemplate:
            def __init__(self, template: str) -> None:
                self._template = template

            def render(self, **_kwargs: Any) -> str:
                return self._template

        class _FakeSandboxedEnvironment:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                _ = (args, kwargs)

            def from_string(self, template: str) -> _FakeTemplate:
                return _FakeTemplate(template)

        jinja2_mod = types.ModuleType("jinja2")
        jinja2_sandbox_mod = types.ModuleType("jinja2.sandbox")
        jinja2_sandbox_mod.SandboxedEnvironment = _FakeSandboxedEnvironment  # type: ignore[attr-defined]
        jinja2_mod.sandbox = jinja2_sandbox_mod  # type: ignore[attr-defined]
        jinja2_mod.__spec__ = importlib.machinery.ModuleSpec("jinja2", loader=None)
        jinja2_sandbox_mod.__spec__ = importlib.machinery.ModuleSpec("jinja2.sandbox", loader=None)
        sys.modules["jinja2"] = jinja2_mod
        sys.modules["jinja2.sandbox"] = jinja2_sandbox_mod

    atlib_mod_name = (
        "tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib"
    )
    if atlib_mod_name not in sys.modules:
        fake_atlib = types.ModuleType(atlib_mod_name)

        class _FakeConversionError(Exception):
            pass

        def _convert_to_wav(path: str, *_args: Any, **_kwargs: Any) -> str:
            return path

        def _is_transcription_error_message(_text: Any) -> bool:
            return False

        def _validate_whisper_model_identifier(model_id: str) -> str:
            return model_id

        fake_atlib.ConversionError = _FakeConversionError  # type: ignore[attr-defined]
        fake_atlib.convert_to_wav = _convert_to_wav  # type: ignore[attr-defined]
        fake_atlib.is_transcription_error_message = _is_transcription_error_message  # type: ignore[attr-defined]
        fake_atlib.validate_whisper_model_identifier = _validate_whisper_model_identifier  # type: ignore[attr-defined]
        fake_atlib.__spec__ = importlib.machinery.ModuleSpec(atlib_mod_name, loader=None)
        sys.modules[atlib_mod_name] = fake_atlib

    audio_files_mod_name = (
        "tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Files"
    )
    if audio_files_mod_name not in sys.modules:
        fake_audio_files = types.ModuleType(audio_files_mod_name)

        def _check_transcription_model_status(model_name: str) -> dict[str, Any]:
            return {"available": True, "model": model_name}

        fake_audio_files.check_transcription_model_status = _check_transcription_model_status  # type: ignore[attr-defined]
        fake_audio_files.__spec__ = importlib.machinery.ModuleSpec(audio_files_mod_name, loader=None)
        sys.modules[audio_files_mod_name] = fake_audio_files

    from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
    import tldw_Server_API.app.api.v1.endpoints.audio.audio_transcriptions as audio_ep
    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.stt_provider_adapter as stt_adapter

    async def _fake_get_request_user() -> User:
        return User(id=1, username="single_user")

    async def _allow_job(*_args: Any, **_kwargs: Any) -> tuple[bool, None]:
        """Allow all quota/job checks in this test harness."""
        return True, None

    async def _noop_async(*_args: Any, **_kwargs: Any) -> None:
        """No-op async hook used to bypass job side effects in tests."""
        return None

    async def _get_limits_for_user(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        """Return permissive in-memory quota limits for endpoint tests."""
        return {
            "daily_minutes": 30.0,
            "concurrent_streams": 1,
            "concurrent_jobs": 1,
            "max_file_size_mb": 25,
        }

    def _heartbeat_interval_seconds() -> int:
        """Disable heartbeat loop in unit tests."""
        return 0

    class _StubAdapter:
        """Minimal STT adapter that returns fixed timed segments."""

        def transcribe_batch(
            self,
            _audio_path: str,
            *,
            model: str | None = None,
            language: str | None = None,
            task: str = "transcribe",
            word_timestamps: bool = False,
            prompt: str | None = None,
            hotwords: str | None = None,
            base_dir: str | None = None,
        ) -> dict[str, Any]:
            _ = (task, word_timestamps, prompt, hotwords, base_dir)
            return {
                "text": "first line second line",
                "language": language or "en",
                "segments": [
                    {
                        "start_seconds": 1.0,
                        "end_seconds": 2.5,
                        "Text": "first line",
                        "words": [
                            {"start": 1.0, "end": 1.4, "word": "first"},
                            {"start": 1.4, "end": 2.5, "word": "line"},
                        ],
                    },
                    {
                        "start_seconds": 2.5,
                        "end_seconds": 4.0,
                        "Text": "second line",
                        "words": [
                            {"start": 2.5, "end": 3.2, "word": "second"},
                            {"start": 3.2, "end": 4.0, "word": "line"},
                        ],
                    },
                ],
                "diarization": {"enabled": False, "speakers": None},
                "usage": {"duration_ms": None, "tokens": None},
                "metadata": {"provider": "parakeet", "model": model or "parakeet-mlx"},
            }

    class _StubRegistry:
        """Registry stub that always routes to Parakeet MLX."""

        def resolve_provider_for_model(self, _model: str) -> tuple[str, str, str]:
            return "parakeet", "parakeet-mlx", "mlx"

        def get_adapter(self, _provider: str) -> _StubAdapter:
            return _StubAdapter()

    shim_values: dict[str, Any] = {
        "can_start_job": _allow_job,
        "increment_jobs_started": _noop_async,
        "finish_job": _noop_async,
        "check_daily_minutes_allow": _allow_job,
        "add_daily_minutes": _noop_async,
        "get_limits_for_user": _get_limits_for_user,
        "get_job_heartbeat_interval_seconds": _heartbeat_interval_seconds,
        "heartbeat_jobs": _noop_async,
        "sf": sf,
    }
    monkeypatch.setattr(audio_ep, "_audio_shim_attr", lambda name: shim_values[name])
    monkeypatch.setattr(stt_adapter, "get_stt_provider_registry", lambda: _StubRegistry())

    audio_router = audio_ep.router
    app = FastAPI()
    app.dependency_overrides[get_request_user] = _fake_get_request_user
    app.include_router(audio_router, prefix="/api/v1/audio")
    return app


@pytest.mark.unit
def test_setup_stubbed_audio_app_has_type_hints() -> None:
    annotations = _setup_stubbed_audio_app.__annotations__
    assert annotations.get("monkeypatch") is pytest.MonkeyPatch
    assert annotations.get("return") is FastAPI


@pytest.mark.unit
def test_audio_transcriptions_uses_provider_timed_segments_for_srt(monkeypatch, bypass_api_limits):
    app = _setup_stubbed_audio_app(monkeypatch)

    with bypass_api_limits(app), TestClient(app) as client:
        wav_bytes = _make_wav_bytes()
        headers = {"X-API-KEY": TEST_API_KEY}
        files = {"file": ("sample.wav", io.BytesIO(wav_bytes), "audio/wav")}
        data = {
            "model": "parakeet-mlx",
            "response_format": "srt",
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
        assert resp.headers["content-type"].startswith("application/x-subrip")
        assert "00:00:01,000 --> 00:00:02,500" in resp.text
        assert "00:00:02,500 --> 00:00:04,000" in resp.text
        assert "00:00:00,000 --> 00:00:10,000" not in resp.text


@pytest.mark.unit
def test_audio_transcriptions_uses_provider_timed_segments_for_vtt(monkeypatch, bypass_api_limits):
    app = _setup_stubbed_audio_app(monkeypatch)

    with bypass_api_limits(app), TestClient(app) as client:
        wav_bytes = _make_wav_bytes()
        headers = {"X-API-KEY": TEST_API_KEY}
        files = {"file": ("sample.wav", io.BytesIO(wav_bytes), "audio/wav")}
        data = {
            "model": "parakeet-mlx",
            "response_format": "vtt",
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
        assert resp.text.startswith("WEBVTT")
        assert "00:00:01.000 --> 00:00:02.500" in resp.text
        assert "00:00:02.500 --> 00:00:04.000" in resp.text


@pytest.mark.unit
def test_audio_transcriptions_verbose_json_exposes_timed_segments_for_parakeet(monkeypatch, bypass_api_limits):
    app = _setup_stubbed_audio_app(monkeypatch)

    with bypass_api_limits(app), TestClient(app) as client:
        wav_bytes = _make_wav_bytes()
        headers = {"X-API-KEY": TEST_API_KEY}
        files = {"file": ("sample.wav", io.BytesIO(wav_bytes), "audio/wav")}
        data = {
            "model": "parakeet-mlx",
            "response_format": "verbose_json",
            "timestamp_granularities": "segment,word",
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
        payload = resp.json()
        assert payload["segments"][0]["start"] == pytest.approx(1.0)
        assert payload["segments"][0]["end"] == pytest.approx(2.5)
        assert payload["segments"][0]["text"] == "first line"
        assert payload["segments"][0]["words"][0]["word"] == "first"
        assert payload["segments"][1]["start"] == pytest.approx(2.5)
