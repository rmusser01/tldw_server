import io
import sys
from types import ModuleType

import numpy as np
import pytest
import soundfile as sf
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.endpoints.audio.audio_transcriptions import router as audio_transcriptions_router
from tldw_Server_API.app.api.v1.endpoints.audio.audio_transcriptions import (
    _normalize_language_for_provider,
    _normalize_language_tag,
)

TEST_API_KEY = "test-api-key-1234567890"


def _make_wav_bytes(duration_sec: float = 0.1, sr: int = 16000) -> bytes:
    buf = io.BytesIO()
    data = np.zeros(int(sr * duration_sec), dtype=np.float32)
    sf.write(buf, data, sr, format="WAV")
    return buf.getvalue()


@pytest.mark.unit
def test_normalize_language_tag_accepts_bcp47_and_underscores():
    assert _normalize_language_tag(" en-US ") == "en-US"
    assert _normalize_language_tag("pt_BR") == "pt-BR"
    assert _normalize_language_tag("  ") is None
    assert _normalize_language_tag(None) is None


@pytest.mark.unit
@pytest.mark.parametrize(
    ("provider", "language", "expected"),
    [
        ("faster-whisper", "en-US", "en"),
        ("parakeet", "de-DE", "de"),
        ("canary", "fr_FR", "fr"),
        ("qwen2audio", "es-MX", "es"),
        ("qwen3-asr", "en-US", "en-US"),
        ("vibevoice", "pt-BR", "pt-BR"),
        ("external", "zh-Hans-CN", "zh-Hans-CN"),
        ("unknown-provider", "en-US", "en-US"),
    ],
)
def test_normalize_language_for_provider(provider, language, expected):
    assert _normalize_language_for_provider(provider, language) == expected


@pytest.mark.unit
def test_audio_transcriptions_default_flow_normalizes_en_us_for_whisper(monkeypatch, bypass_api_limits):
    monkeypatch.setenv("TEST_MODE", "true")
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("SINGLE_USER_API_KEY", TEST_API_KEY)
    monkeypatch.setenv("SINGLE_USER_TEST_API_KEY", TEST_API_KEY)
    monkeypatch.setenv("SINGLE_USER_FIXED_ID", "1")

    from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
    import tldw_Server_API.app.api.v1.endpoints.audio.audio_transcriptions as audio_tx

    fake_atlib = ModuleType("Audio_Transcription_Lib")

    class _FakeConversionError(Exception):
        pass

    def _fake_parse_transcription_model(model_name: str):
        return "whisper", (model_name or "").strip(), None

    fake_atlib.ConversionError = _FakeConversionError
    fake_atlib.convert_to_wav = lambda path, *args, **kwargs: path
    fake_atlib.is_transcription_error_message = lambda _text: False
    fake_atlib.validate_whisper_model_identifier = lambda value: value
    fake_atlib.parse_transcription_model = _fake_parse_transcription_model
    monkeypatch.setitem(
        sys.modules,
        "tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib",
        fake_atlib,
    )

    fake_audio_files = ModuleType("Audio_Files")
    fake_audio_files.check_transcription_model_status = lambda _model_name: {"available": True}
    monkeypatch.setitem(
        sys.modules,
        "tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Files",
        fake_audio_files,
    )

    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.stt_provider_adapter as stt_adapter

    async def _fake_get_request_user() -> User:
        return User(id=1, username="single_user")

    async def _allow_job(*_args, **_kwargs):
        return True, None

    async def _noop_async(*_args, **_kwargs):
        return None

    async def _get_limits_for_user(_user_id: int):
        return {
            "daily_minutes": 30.0,
            "concurrent_streams": 1,
            "concurrent_jobs": 1,
            "max_file_size_mb": 25,
        }

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
            assert language == "en"
            return {
                "text": "normalized language transcript",
                "language": language or "en",
                "segments": [{"start_seconds": 0.0, "end_seconds": 0.1, "Text": "normalized language transcript"}],
                "diarization": {"enabled": False, "speakers": None},
                "usage": {"duration_ms": None, "tokens": None},
                "metadata": {"provider": "faster-whisper", "model": model or "large-v3"},
            }

    class _StubRegistry:
        def resolve_provider_for_model(self, _model):
            return "faster-whisper", "large-v3", None

        def list_capabilities(self, include_disabled=True):
            assert include_disabled is True
            return [{"provider": "faster-whisper", "availability": "enabled", "capabilities": {"batch": True}}]

        def get_adapter(self, _provider):
            return _StubAdapter()

    quota_fakes = {
        "can_start_job": _allow_job,
        "increment_jobs_started": _noop_async,
        "finish_job": _noop_async,
        "consume_daily_minutes": _allow_job,
        "get_limits_for_user": _get_limits_for_user,
    }
    for name, fake in quota_fakes.items():
        monkeypatch.setattr(f"tldw_Server_API.app.core.Usage.audio_quota.{name}", fake)
    monkeypatch.setattr(stt_adapter, "get_stt_provider_registry", lambda: _StubRegistry())
    monkeypatch.setattr(stt_adapter, "resolve_default_transcription_model", lambda _fallback: "whisper-1")

    app = FastAPI()
    app.dependency_overrides[get_request_user] = _fake_get_request_user
    app.include_router(audio_transcriptions_router, prefix="/api/v1/audio")

    with bypass_api_limits(app), TestClient(app) as client:
        wav_bytes = _make_wav_bytes()
        headers = {"X-API-KEY": TEST_API_KEY}
        files = {"file": ("sample.wav", io.BytesIO(wav_bytes), "audio/wav")}
        data = {
            "response_format": "json",
            "language": "en-US",
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
        assert resp.json().get("text") == "normalized language transcript"


@pytest.mark.unit
def test_audio_transcriptions_allows_whisper_first_use_download_when_model_not_cached(
    monkeypatch,
    bypass_api_limits,
):
    monkeypatch.setenv("TEST_MODE", "true")
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("SINGLE_USER_API_KEY", TEST_API_KEY)
    monkeypatch.setenv("SINGLE_USER_TEST_API_KEY", TEST_API_KEY)
    monkeypatch.setenv("SINGLE_USER_FIXED_ID", "1")

    from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
    import tldw_Server_API.app.api.v1.endpoints.audio.audio_transcriptions as audio_tx

    fake_atlib = ModuleType("Audio_Transcription_Lib")

    class _FakeConversionError(Exception):
        pass

    def _fake_parse_transcription_model(model_name: str):
        return "whisper", (model_name or "").strip(), None

    fake_atlib.ConversionError = _FakeConversionError
    fake_atlib.convert_to_wav = lambda path, *args, **kwargs: path
    fake_atlib.is_transcription_error_message = lambda _text: False
    fake_atlib.validate_whisper_model_identifier = lambda value: value
    fake_atlib.parse_transcription_model = _fake_parse_transcription_model
    monkeypatch.setitem(
        sys.modules,
        "tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib",
        fake_atlib,
    )

    fake_audio_files = ModuleType("Audio_Files")
    fake_audio_files.check_transcription_model_status = lambda _model_name: {
        "available": False,
        "usable": False,
        "on_demand": True,
        "model": "large-v3",
        "provider": "whisper",
        "message": "Model large-v3 is not cached locally yet and will download on first use.",
    }
    monkeypatch.setitem(
        sys.modules,
        "tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Files",
        fake_audio_files,
    )

    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.stt_provider_adapter as stt_adapter

    async def _fake_get_request_user() -> User:
        return User(id=1, username="single_user")

    async def _allow_job(*_args, **_kwargs):
        return True, None

    async def _noop_async(*_args, **_kwargs):
        return None

    async def _get_limits_for_user(_user_id: int):
        return {
            "daily_minutes": 30.0,
            "concurrent_streams": 1,
            "concurrent_jobs": 1,
            "max_file_size_mb": 25,
        }

    captured: dict[str, object] = {"called": False, "model": None}

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
            captured["called"] = True
            captured["model"] = model
            return {
                "text": "first use download transcript",
                "language": language or "en",
                "segments": [{"start_seconds": 0.0, "end_seconds": 0.1, "Text": "first use download transcript"}],
                "diarization": {"enabled": False, "speakers": None},
                "usage": {"duration_ms": None, "tokens": None},
                "metadata": {"provider": "faster-whisper", "model": model or "large-v3"},
            }

    class _StubRegistry:
        def resolve_provider_for_model(self, _model):
            return "faster-whisper", "large-v3", None

        def list_capabilities(self, include_disabled=True):
            assert include_disabled is True
            return [{"provider": "faster-whisper", "availability": "enabled", "capabilities": {"batch": True}}]

        def get_adapter(self, _provider):
            return _StubAdapter()

    quota_fakes = {
        "can_start_job": _allow_job,
        "increment_jobs_started": _noop_async,
        "finish_job": _noop_async,
        "consume_daily_minutes": _allow_job,
        "get_limits_for_user": _get_limits_for_user,
    }
    for name, fake in quota_fakes.items():
        monkeypatch.setattr(f"tldw_Server_API.app.core.Usage.audio_quota.{name}", fake)
    monkeypatch.setattr(stt_adapter, "get_stt_provider_registry", lambda: _StubRegistry())

    app = FastAPI()
    app.dependency_overrides[get_request_user] = _fake_get_request_user
    app.include_router(audio_transcriptions_router, prefix="/api/v1/audio")

    with bypass_api_limits(app), TestClient(app) as client:
        wav_bytes = _make_wav_bytes()
        headers = {"X-API-KEY": TEST_API_KEY}
        files = {"file": ("sample.wav", io.BytesIO(wav_bytes), "audio/wav")}
        data = {
            "model": "whisper-1",
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
        assert resp.json().get("text") == "first use download transcript"
        assert captured["called"] is True
        assert captured["model"] == "large-v3"
