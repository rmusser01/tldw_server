import io
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf
from fastapi import status
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.endpoints.audio.audio import router as audio_router

TEST_API_KEY = "test-api-key-1234567890"


def _make_wav_bytes(duration_sec: float = 0.1, sr: int = 16000) -> bytes:
    buf = io.BytesIO()
    data = np.zeros(int(sr * duration_sec), dtype=np.float32)
    sf.write(buf, data, sr, format="WAV")
    return buf.getvalue()


def test_audio_transcriptions_uses_adapter_base_dir(
    monkeypatch: pytest.MonkeyPatch,
    bypass_api_limits: Any,
) -> None:
    """Successful uploads pass adapter base_dir and force canonical WAV conversion."""

    monkeypatch.setenv("TEST_MODE", "true")
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("SINGLE_USER_API_KEY", TEST_API_KEY)
    monkeypatch.setenv("SINGLE_USER_TEST_API_KEY", TEST_API_KEY)
    monkeypatch.setenv("SINGLE_USER_FIXED_ID", "1")

    from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_request_user, User
    import tldw_Server_API.app.api.v1.endpoints.audio.audio as audio_ep
    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib as atlib

    async def _fake_get_request_user() -> User:
        return User(id=1, username="single_user")

    async def _allow_job(*_args: object, **_kwargs: object) -> tuple[bool, None]:
        return True, None

    async def _noop_async(*_args: object, **_kwargs: object) -> None:
        return None

    class _StubAdapter:
        def transcribe_batch(
            self,
            audio_path: str,
            *,
            model: str | None = None,
            language: str | None = None,
            task: str = "transcribe",
            word_timestamps: bool = False,
            prompt: str | None = None,
            hotwords: list[str] | None = None,
            base_dir: Path | None = None,
        ) -> dict[str, Any]:
            assert hotwords is None
            assert base_dir is not None
            assert base_dir == Path(audio_path).parent
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
                "metadata": {"provider": "external", "model": model or "external:stub"},
            }

    class _StubRegistry:
        def resolve_provider_for_model(self, _model: object) -> tuple[str, str, None]:
            return "external", "external:stub", None

        def get_adapter(self, _provider: object) -> _StubAdapter:
            return _StubAdapter()

    monkeypatch.setattr(audio_ep, "can_start_job", _allow_job)
    monkeypatch.setattr(audio_ep, "increment_jobs_started", _noop_async)
    monkeypatch.setattr(audio_ep, "finish_job", _noop_async)
    monkeypatch.setattr(audio_ep, "check_daily_minutes_allow", _allow_job)
    monkeypatch.setattr(audio_ep, "add_daily_minutes", _noop_async)
    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.stt_provider_adapter as stt_adapter

    captured_conversion: dict[str, Any] = {}

    def _capture_convert_to_wav(path: str, *args: object, **kwargs: object) -> str:
        captured_conversion["path"] = path
        captured_conversion["overwrite"] = kwargs.get("overwrite")
        return path

    monkeypatch.setattr(stt_adapter, "get_stt_provider_registry", lambda: _StubRegistry())
    monkeypatch.setattr(atlib, "convert_to_wav", _capture_convert_to_wav)

    app = FastAPI()
    app.dependency_overrides[get_request_user] = _fake_get_request_user
    app.include_router(audio_router, prefix="/api/v1/audio")

    with bypass_api_limits(app), TestClient(app) as client:
        wav_bytes = _make_wav_bytes()
        headers = {"X-API-KEY": TEST_API_KEY}
        files = {"file": ("sample.wav", io.BytesIO(wav_bytes), "audio/wav")}
        data = {"model": "external:stub", "response_format": "json"}
        resp = client.post(
            "/api/v1/audio/transcriptions",
            headers=headers,
            files=files,
            data=data,
        )
        if resp.status_code == 404:
            pytest.skip("audio/transcriptions endpoint not mounted in this build")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["text"] == "stub transcript"
        assert Path(captured_conversion["path"]).suffix == ".wav"
        assert captured_conversion["overwrite"] is True


def test_audio_cpp_transcription_uses_real_registry_without_fallback(
    monkeypatch: pytest.MonkeyPatch,
    bypass_api_limits: Any,
) -> None:
    """The ordinary endpoint routes an audio.cpp selector through its native adapter."""
    monkeypatch.setenv("TEST_MODE", "true")
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("SINGLE_USER_API_KEY", TEST_API_KEY)
    monkeypatch.setenv("SINGLE_USER_TEST_API_KEY", TEST_API_KEY)
    monkeypatch.setenv("SINGLE_USER_FIXED_ID", "1")
    monkeypatch.setenv("STT_AUDIO_CPP_ENABLED", "true")
    monkeypatch.setenv("STT_AUDIO_CPP_BASE_URL", "http://127.0.0.1:18080")
    monkeypatch.setenv("STT_AUDIO_CPP_DEFAULT_MODEL", "unused-default")
    monkeypatch.setenv("STT_AUDIO_CPP_TIMEOUT_SECONDS", "17.25")

    import tldw_Server_API.app.api.v1.endpoints.audio.audio as audio_ep
    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.stt_provider_adapter as stt_adapter
    from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio import (
        Audio_Transcription_AudioCpp as audio_cpp,
    )

    async def _fake_get_request_user() -> User:
        return User(id=1, username="single_user")

    async def _allow_job(*_args: object, **_kwargs: object) -> tuple[bool, None]:
        return True, None

    async def _noop_async(*_args: object, **_kwargs: object) -> None:
        return None

    registry = stt_adapter.SttProviderRegistry()
    adapter_lookups: list[str] = []
    get_adapter = registry.get_adapter

    def _trace_adapter_lookup(provider: str) -> Any:
        adapter_lookups.append(provider)
        return get_adapter(provider)

    captured: dict[str, object] = {}

    def _fake_transcribe_audio_cpp(
        audio_path: str,
        *,
        route: Any,
        model_id: str,
        **_kwargs: object,
    ) -> Any:
        captured["audio_path"] = audio_path
        captured["provider"] = route.provider
        captured["model_id"] = model_id
        return stt_adapter.SttTranscriptionOutcome(
            artifact={
                "text": "audio.cpp endpoint transcript",
                "segments": [],
                "metadata": {
                    "provider": "audio-cpp",
                    "contract": "audio_cpp_http_v1",
                    "model_id": model_id,
                    "model_family": "whisper",
                    "model_mode": "offline",
                    "server_backend": "cpu",
                },
            },
            actual_execution=stt_adapter.actual_execution_from_route(
                route,
                device=None,
            ),
        )

    monkeypatch.setattr(audio_ep, "can_start_job", _allow_job)
    monkeypatch.setattr(audio_ep, "increment_jobs_started", _noop_async)
    monkeypatch.setattr(audio_ep, "finish_job", _noop_async)
    monkeypatch.setattr(audio_ep, "check_daily_minutes_allow", _allow_job)
    monkeypatch.setattr(audio_ep, "add_daily_minutes", _noop_async)
    monkeypatch.setattr(registry, "get_adapter", _trace_adapter_lookup)
    monkeypatch.setattr(
        stt_adapter,
        "get_stt_provider_registry",
        lambda: registry,
    )
    monkeypatch.setattr(
        audio_cpp,
        "transcribe_audio_cpp",
        _fake_transcribe_audio_cpp,
    )

    app = FastAPI()
    app.dependency_overrides[get_request_user] = _fake_get_request_user
    app.include_router(audio_router, prefix="/api/v1/audio")

    with bypass_api_limits(app), TestClient(app) as client:
        response = client.post(
            "/api/v1/audio/transcriptions",
            headers={"X-API-KEY": TEST_API_KEY},
            files={
                "file": (
                    "sample.wav",
                    io.BytesIO(_make_wav_bytes()),
                    "audio/wav",
                )
            },
            data={
                "model": "audio-cpp:whisper-small",
                "response_format": "json",
            },
        )

    assert response.status_code == status.HTTP_200_OK, response.text
    assert response.json()["text"] == "audio.cpp endpoint transcript"
    assert adapter_lookups == ["audio-cpp"]
    assert captured["provider"] == "audio-cpp"
    assert captured["model_id"] == "whisper-small"
    assert Path(str(captured["audio_path"])).suffix == ".wav"


def test_audio_transcriptions_derives_suffix_for_extensionless_upload(
    monkeypatch,
    bypass_api_limits,
):
    monkeypatch.setenv("TEST_MODE", "true")
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("SINGLE_USER_API_KEY", TEST_API_KEY)
    monkeypatch.setenv("SINGLE_USER_TEST_API_KEY", TEST_API_KEY)
    monkeypatch.setenv("SINGLE_USER_FIXED_ID", "1")

    from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_request_user, User
    import tldw_Server_API.app.api.v1.endpoints.audio.audio as audio_ep
    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib as atlib
    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.stt_provider_adapter as stt_adapter

    async def _fake_get_request_user() -> User:
        return User(id=1, username="single_user")

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
            assert prompt is None
            assert hotwords is None
            assert base_dir == Path(audio_path).parent
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
                "metadata": {"provider": "external", "model": model or "external:stub"},
            }

    class _StubRegistry:
        def resolve_provider_for_model(self, _model):
            return "external", "external:stub", None

        def get_adapter(self, _provider):
            return _StubAdapter()

    monkeypatch.setattr(audio_ep, "can_start_job", _allow_job)
    monkeypatch.setattr(audio_ep, "increment_jobs_started", _noop_async)
    monkeypatch.setattr(audio_ep, "finish_job", _noop_async)
    monkeypatch.setattr(audio_ep, "check_daily_minutes_allow", _allow_job)
    monkeypatch.setattr(audio_ep, "add_daily_minutes", _noop_async)
    monkeypatch.setattr(stt_adapter, "get_stt_provider_registry", lambda: _StubRegistry())

    captured = {}

    def _capture_convert_to_wav(path, *args, **kwargs):
        captured["staged_path"] = Path(path)
        return path

    monkeypatch.setattr(atlib, "convert_to_wav", _capture_convert_to_wav)

    app = FastAPI()
    app.dependency_overrides[get_request_user] = _fake_get_request_user
    app.include_router(audio_router, prefix="/api/v1/audio")

    with bypass_api_limits(app), TestClient(app) as client:
        wav_bytes = _make_wav_bytes()
        headers = {"X-API-KEY": TEST_API_KEY}
        files = {"file": ("recording", io.BytesIO(wav_bytes), "audio/wav")}
        data = {"model": "external:stub", "response_format": "json"}
        resp = client.post(
            "/api/v1/audio/transcriptions",
            headers=headers,
            files=files,
            data=data,
        )
        if resp.status_code == 404:
            pytest.skip("audio/transcriptions endpoint not mounted in this build")
        assert resp.status_code == 200, resp.text
        assert captured["staged_path"].suffix == ".wav"


def test_audio_transcriptions_returns_503_when_provider_disabled(
    monkeypatch,
    bypass_api_limits,
):
    monkeypatch.setenv("TEST_MODE", "true")
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("SINGLE_USER_API_KEY", TEST_API_KEY)
    monkeypatch.setenv("SINGLE_USER_TEST_API_KEY", TEST_API_KEY)
    monkeypatch.setenv("SINGLE_USER_FIXED_ID", "1")

    from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
    import tldw_Server_API.app.api.v1.endpoints.audio.audio as audio_ep
    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib as atlib
    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.stt_provider_adapter as stt_adapter

    async def _fake_get_request_user() -> User:
        return User(id=1, username="single_user")

    async def _allow_job(*_args, **_kwargs):
        return True, None

    async def _noop_async(*_args, **_kwargs):
        return None

    class _StubRegistry:
        def resolve_provider_for_model(self, _model):
            return "external", "external:stub", None

        def list_capabilities(self, include_disabled=True):
            assert include_disabled is True
            return [{"provider": "external", "availability": "disabled", "capabilities": None}]

        def get_adapter(self, _provider):
            pytest.fail("get_adapter should not be called for disabled providers")

    monkeypatch.setattr(audio_ep, "can_start_job", _allow_job)
    monkeypatch.setattr(audio_ep, "increment_jobs_started", _noop_async)
    monkeypatch.setattr(audio_ep, "finish_job", _noop_async)
    monkeypatch.setattr(audio_ep, "check_daily_minutes_allow", _allow_job)
    monkeypatch.setattr(audio_ep, "add_daily_minutes", _noop_async)
    monkeypatch.setattr(stt_adapter, "get_stt_provider_registry", lambda: _StubRegistry())
    monkeypatch.setattr(atlib, "convert_to_wav", lambda path, *args, **kwargs: path)

    app = FastAPI()
    app.dependency_overrides[get_request_user] = _fake_get_request_user
    app.include_router(audio_router, prefix="/api/v1/audio")

    with bypass_api_limits(app), TestClient(app) as client:
        wav_bytes = _make_wav_bytes()
        headers = {"X-API-KEY": TEST_API_KEY}
        files = {"file": ("sample.wav", io.BytesIO(wav_bytes), "audio/wav")}
        data = {"model": "external:stub", "response_format": "json"}
        resp = client.post(
            "/api/v1/audio/transcriptions",
            headers=headers,
            files=files,
            data=data,
        )
        if resp.status_code == 404:
            pytest.skip("audio/transcriptions endpoint not mounted in this build")
        assert resp.status_code == 503, resp.text
        detail = resp.json().get("detail") or {}
        assert detail.get("status") == "provider_unavailable"
        assert detail.get("provider") == "external"
        assert detail.get("availability") == "disabled"


def test_audio_transcriptions_prevents_cross_provider_fallback(
    monkeypatch,
    bypass_api_limits,
):
    monkeypatch.setenv("TEST_MODE", "true")
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("SINGLE_USER_API_KEY", TEST_API_KEY)
    monkeypatch.setenv("SINGLE_USER_TEST_API_KEY", TEST_API_KEY)
    monkeypatch.setenv("SINGLE_USER_FIXED_ID", "1")

    from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
    import tldw_Server_API.app.api.v1.endpoints.audio.audio as audio_ep
    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib as atlib
    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.stt_provider_adapter as stt_adapter

    async def _fake_get_request_user() -> User:
        return User(id=1, username="single_user")

    async def _allow_job(*_args, **_kwargs):
        return True, None

    async def _noop_async(*_args, **_kwargs):
        return None

    class _AdapterName:
        value = "faster-whisper"

    class _MismatchedAdapter:
        name = _AdapterName()

        def transcribe_batch(self, *args, **kwargs):
            pytest.fail("transcribe_batch should not be called for mismatched provider fallback")

    class _StubRegistry:
        def resolve_provider_for_model(self, _model):
            return "external", "external:stub", None

        def list_capabilities(self, include_disabled=True):
            assert include_disabled is True
            return [{"provider": "external", "availability": "enabled", "capabilities": {"streaming": True}}]

        def get_adapter(self, _provider):
            return _MismatchedAdapter()

    monkeypatch.setattr(audio_ep, "can_start_job", _allow_job)
    monkeypatch.setattr(audio_ep, "increment_jobs_started", _noop_async)
    monkeypatch.setattr(audio_ep, "finish_job", _noop_async)
    monkeypatch.setattr(audio_ep, "check_daily_minutes_allow", _allow_job)
    monkeypatch.setattr(audio_ep, "add_daily_minutes", _noop_async)
    monkeypatch.setattr(stt_adapter, "get_stt_provider_registry", lambda: _StubRegistry())
    monkeypatch.setattr(atlib, "convert_to_wav", lambda path, *args, **kwargs: path)

    app = FastAPI()
    app.dependency_overrides[get_request_user] = _fake_get_request_user
    app.include_router(audio_router, prefix="/api/v1/audio")

    with bypass_api_limits(app), TestClient(app) as client:
        wav_bytes = _make_wav_bytes()
        headers = {"X-API-KEY": TEST_API_KEY}
        files = {"file": ("sample.wav", io.BytesIO(wav_bytes), "audio/wav")}
        data = {"model": "external:stub", "response_format": "json"}
        resp = client.post(
            "/api/v1/audio/transcriptions",
            headers=headers,
            files=files,
            data=data,
        )
        if resp.status_code == 404:
            pytest.skip("audio/transcriptions endpoint not mounted in this build")
        assert resp.status_code == 503, resp.text
        detail = resp.json().get("detail") or {}
        assert detail.get("status") == "provider_unavailable"
        assert detail.get("provider") == "external"
        assert detail.get("resolved_provider") == "faster-whisper"


def test_audio_transcriptions_rejects_upload_when_wav_conversion_fails(
    monkeypatch: pytest.MonkeyPatch,
    bypass_api_limits: Any,
) -> None:
    """Reject compressed uploads instead of falling back when WAV conversion fails."""
    monkeypatch.setenv("TEST_MODE", "true")
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("SINGLE_USER_API_KEY", TEST_API_KEY)
    monkeypatch.setenv("SINGLE_USER_TEST_API_KEY", TEST_API_KEY)
    monkeypatch.setenv("SINGLE_USER_FIXED_ID", "1")

    from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
    import tldw_Server_API.app.api.v1.endpoints.audio.audio as audio_ep
    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib as atlib
    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.stt_provider_adapter as stt_adapter

    async def _fake_get_request_user() -> User:
        return User(id=1, username="single_user")

    async def _allow_job(*_args: object, **_kwargs: object) -> tuple[bool, None]:
        return True, None

    async def _noop_async(*_args: object, **_kwargs: object) -> None:
        return None

    class _StubAdapter:
        def transcribe_batch(self, *args: object, **kwargs: object) -> None:
            pytest.fail("adapter should not receive original upload after conversion failure")

    class _StubRegistry:
        def resolve_provider_for_model(self, _model: object) -> tuple[str, str, None]:
            return "external", "external:stub", None

        def get_adapter(self, _provider: object) -> _StubAdapter:
            return _StubAdapter()

    def _fail_convert(*_args: object, **_kwargs: object) -> None:
        raise atlib.ConversionError("ffmpeg failed")

    monkeypatch.setattr(audio_ep, "can_start_job", _allow_job)
    monkeypatch.setattr(audio_ep, "increment_jobs_started", _noop_async)
    monkeypatch.setattr(audio_ep, "finish_job", _noop_async)
    monkeypatch.setattr(audio_ep, "check_daily_minutes_allow", _allow_job)
    monkeypatch.setattr(audio_ep, "add_daily_minutes", _noop_async)
    monkeypatch.setattr(stt_adapter, "get_stt_provider_registry", lambda: _StubRegistry())
    monkeypatch.setattr(atlib, "convert_to_wav", _fail_convert)

    app = FastAPI()
    app.dependency_overrides[get_request_user] = _fake_get_request_user
    app.include_router(audio_router, prefix="/api/v1/audio")

    with bypass_api_limits(app), TestClient(app) as client:
        headers = {"X-API-KEY": TEST_API_KEY}
        files = {"file": ("sample.mp3", io.BytesIO(b"not a decodable mp3"), "audio/mpeg")}
        data = {"model": "external:stub", "response_format": "json"}
        resp = client.post(
            "/api/v1/audio/transcriptions",
            headers=headers,
            files=files,
            data=data,
        )
        if resp.status_code == 404:
            pytest.skip("audio/transcriptions endpoint not mounted in this build")
        assert resp.status_code == status.HTTP_400_BAD_REQUEST, resp.text
        detail = resp.json().get("detail") or {}
        assert detail.get("status") == "invalid_audio"


def test_audio_transcriptions_rejects_spoofed_wav_output(
    monkeypatch: pytest.MonkeyPatch,
    bypass_api_limits: Any,
) -> None:
    """Reject a .wav conversion output when its bytes are not a WAV container."""
    monkeypatch.setenv("TEST_MODE", "true")
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("SINGLE_USER_API_KEY", TEST_API_KEY)
    monkeypatch.setenv("SINGLE_USER_TEST_API_KEY", TEST_API_KEY)
    monkeypatch.setenv("SINGLE_USER_FIXED_ID", "1")

    from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
    import tldw_Server_API.app.api.v1.endpoints.audio.audio as audio_ep
    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib as atlib
    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.stt_provider_adapter as stt_adapter

    async def _fake_get_request_user() -> User:
        return User(id=1, username="single_user")

    async def _allow_job(*_args: object, **_kwargs: object) -> tuple[bool, None]:
        return True, None

    async def _noop_async(*_args: object, **_kwargs: object) -> None:
        return None

    class _StubAdapter:
        def transcribe_batch(self, *args: object, **kwargs: object) -> dict[str, Any]:
            return {"text": "should not transcribe"}

    class _StubRegistry:
        def resolve_provider_for_model(self, _model: object) -> tuple[str, str, None]:
            return "external", "external:stub", None

        def get_adapter(self, _provider: object) -> _StubAdapter:
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
    app.include_router(audio_router, prefix="/api/v1/audio")

    with bypass_api_limits(app), TestClient(app) as client:
        headers = {"X-API-KEY": TEST_API_KEY}
        files = {"file": ("sample.wav", io.BytesIO(b"not a wav"), "audio/wav")}
        data = {"model": "external:stub", "response_format": "json"}
        resp = client.post(
            "/api/v1/audio/transcriptions",
            headers=headers,
            files=files,
            data=data,
        )
        if resp.status_code == 404:
            pytest.skip("audio/transcriptions endpoint not mounted in this build")
        np.testing.assert_equal(
            resp.status_code,
            status.HTTP_400_BAD_REQUEST,
            err_msg=resp.text,
        )
        detail = resp.json().get("detail") or {}
        np.testing.assert_equal(detail.get("status"), "invalid_audio")


@pytest.mark.parametrize("conversion_output_kind", ["empty", "non_wav"])
def test_audio_transcriptions_rejects_unusable_conversion_output(
    monkeypatch: pytest.MonkeyPatch,
    bypass_api_limits: Any,
    tmp_path: Path,
    conversion_output_kind: str,
) -> None:
    """Reject conversion outputs that are empty, missing, or not canonical WAV files."""
    monkeypatch.setenv("TEST_MODE", "true")
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("SINGLE_USER_API_KEY", TEST_API_KEY)
    monkeypatch.setenv("SINGLE_USER_TEST_API_KEY", TEST_API_KEY)
    monkeypatch.setenv("SINGLE_USER_FIXED_ID", "1")

    from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
    import tldw_Server_API.app.api.v1.endpoints.audio.audio as audio_ep
    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib as atlib
    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.stt_provider_adapter as stt_adapter

    async def _fake_get_request_user() -> User:
        return User(id=1, username="single_user")

    async def _allow_job(*_args: object, **_kwargs: object) -> tuple[bool, None]:
        return True, None

    async def _noop_async(*_args: object, **_kwargs: object) -> None:
        return None

    class _StubAdapter:
        def transcribe_batch(self, *args: object, **kwargs: object) -> None:
            pytest.fail("adapter should not receive a non-WAV conversion output")

    class _StubRegistry:
        def resolve_provider_for_model(self, _model: object) -> tuple[str, str, None]:
            return "external", "external:stub", None

        def get_adapter(self, _provider: object) -> _StubAdapter:
            return _StubAdapter()

    if conversion_output_kind == "empty":
        conversion_output = None
    else:
        bad_output = tmp_path / "converted.mp3"
        bad_output.write_bytes(b"still compressed")
        conversion_output = str(bad_output)

    monkeypatch.setattr(audio_ep, "can_start_job", _allow_job)
    monkeypatch.setattr(audio_ep, "increment_jobs_started", _noop_async)
    monkeypatch.setattr(audio_ep, "finish_job", _noop_async)
    monkeypatch.setattr(audio_ep, "check_daily_minutes_allow", _allow_job)
    monkeypatch.setattr(audio_ep, "add_daily_minutes", _noop_async)
    monkeypatch.setattr(stt_adapter, "get_stt_provider_registry", lambda: _StubRegistry())
    monkeypatch.setattr(atlib, "convert_to_wav", lambda *args, **kwargs: conversion_output)

    app = FastAPI()
    app.dependency_overrides[get_request_user] = _fake_get_request_user
    app.include_router(audio_router, prefix="/api/v1/audio")

    with bypass_api_limits(app), TestClient(app) as client:
        headers = {"X-API-KEY": TEST_API_KEY}
        files = {"file": ("sample.webm", io.BytesIO(b"not webm"), "audio/webm")}
        data = {"model": "external:stub", "response_format": "json"}
        resp = client.post(
            "/api/v1/audio/transcriptions",
            headers=headers,
            files=files,
            data=data,
        )
        if resp.status_code == 404:
            pytest.skip("audio/transcriptions endpoint not mounted in this build")
        assert resp.status_code == status.HTTP_400_BAD_REQUEST, resp.text
        detail = resp.json().get("detail") or {}
        assert detail.get("status") == "invalid_audio"
