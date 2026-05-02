"""Unit tests for OpenAI OAuth retry behavior in audio TTS endpoints."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException
from starlette.requests import Request

import tldw_Server_API.app.api.v1.endpoints.audio.audio_tts as audio_tts
from tldw_Server_API.app.api.v1.schemas.audio_schemas import OpenAISpeechRequest
from tldw_Server_API.app.core.AuthNZ.exceptions import StorageError
from tldw_Server_API.app.core.TTS.tts_exceptions import TTSAuthenticationError


class _DummyByokResolution:
    def __init__(self, *, api_key: str, auth_source: str = "oauth"):
        self.api_key = api_key
        self.auth_source = auth_source
        self.touch_calls = 0

    async def touch_last_used(self):
        self.touch_calls += 1


class _FailingTouchByokResolution(_DummyByokResolution):
    async def touch_last_used(self):
        raise RuntimeError("byok touch leaked /private/tts-byok.json")


class _AuthRetryTTSService:
    def __init__(self, failures_before_success: int):
        self.failures_before_success = failures_before_success
        self.calls = 0

    def generate_speech(self, *args, **kwargs):  # noqa: ARG002
        self.calls += 1
        call_idx = self.calls

        async def _gen():
            if call_idx <= self.failures_before_success:
                raise TTSAuthenticationError("oauth access token invalid")
            yield b"recovered audio"

        return _gen()


def _make_request(path: str = "/api/v1/audio/speech", headers: list[tuple[str, str]] | None = None) -> Request:
    scope = {
        "type": "http",
        "method": "POST",
        "path": path,
        "headers": [
            (name.lower().encode("latin-1"), value.encode("latin-1"))
            for name, value in (headers or [])
        ],
        "query_string": b"",
        "server": ("testserver", 80),
        "client": ("testclient", 12345),
    }

    async def _receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    return Request(scope, _receive)


def _request_data() -> OpenAISpeechRequest:
    return OpenAISpeechRequest(
        input="hello world",
        model="tts-1",
        voice="alloy",
        stream=False,
        response_format="mp3",
    )


def _patch_audio_shim(monkeypatch, resolve_tts_byok):
    async def _unused_save_and_register_tts_audio(**kwargs):  # pragma: no cover - defensive fallback
        _ = kwargs
        return {"id": None}

    shim_map = {
        "_sanitize_speech_request": lambda *args, **kwargs: "openai",
        "_resolve_tts_byok": resolve_tts_byok,
        "save_and_register_tts_audio": _unused_save_and_register_tts_audio,
    }

    def _shim_attr(name: str):
        if name not in shim_map:
            raise NameError(name)
        return shim_map[name]

    monkeypatch.setattr(audio_tts, "_audio_shim_attr", _shim_attr, raising=True)


class _SuccessfulTTSService:
    def generate_speech(self, *args, **kwargs):  # noqa: ARG002
        async def _gen():
            yield b"generated audio"

        return _gen()


class _FailingUsageLog:
    def log_event(self, *args, **kwargs):  # noqa: ARG002
        raise RuntimeError("usage logger leaked /private/tts-usage.json")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_list_tts_providers_failure_log_is_sanitized(monkeypatch):
    class _FailingTTSService:
        async def get_capabilities(self):
            raise RuntimeError("providers backend leaked /private/tts-providers.json")

        async def list_voices(self):  # pragma: no cover - get_capabilities fails first
            return {}

    fake_logger = MagicMock()
    monkeypatch.setattr(audio_tts, "logger", fake_logger)

    with pytest.raises(HTTPException) as exc_info:
        await audio_tts.list_tts_providers(
            _make_request("/api/v1/audio/providers"),
            tts_service=_FailingTTSService(),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail["message"] == "Failed to list providers"
    fake_logger.error.assert_called_once_with("Error listing TTS providers")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_list_tts_voices_failure_log_is_sanitized(monkeypatch):
    class _FailingTTSService:
        async def list_voices(self):
            raise RuntimeError("voices backend leaked /private/tts-voices.json")

    fake_logger = MagicMock()
    monkeypatch.setattr(audio_tts, "logger", fake_logger)

    with pytest.raises(HTTPException) as exc_info:
        await audio_tts.list_tts_voices(
            _make_request("/api/v1/audio/voices/catalog"),
            tts_service=_FailingTTSService(),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail["message"] == "Failed to list voices"
    fake_logger.error.assert_called_once_with("Error listing TTS voices")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_reset_tts_metrics_failure_log_is_sanitized(monkeypatch):
    class _FailingTTSService:
        def reset_metrics(self):
            raise RuntimeError("metrics backend leaked /private/tts-metrics.json")

    fake_logger = MagicMock()
    monkeypatch.setattr(audio_tts, "logger", fake_logger)

    with pytest.raises(HTTPException) as exc_info:
        await audio_tts.reset_tts_metrics(
            _make_request("/api/v1/audio/reset-metrics"),
            tts_service=_FailingTTSService(),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail["message"] == "Failed to reset metrics"
    fake_logger.error.assert_called_once_with("Error resetting metrics")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_audio_speech_invalid_voice_to_voice_header_log_is_sanitized(monkeypatch):
    async def _resolve_tts_byok(*args, **kwargs):
        _ = args, kwargs
        return (1, {}, None)

    _patch_audio_shim(monkeypatch, _resolve_tts_byok)
    fake_logger = MagicMock()
    monkeypatch.setattr(audio_tts, "logger", fake_logger)

    response = await audio_tts.create_speech(
        _request_data(),
        _make_request(headers=[("x-voice-to-voice-start", "bad /private/tts-header.txt")]),
        tts_service=_SuccessfulTTSService(),
        current_user=SimpleNamespace(id=1),
        media_db=None,
        usage_log=SimpleNamespace(log_event=lambda *args, **kwargs: None),
    )

    assert response.status_code == 200
    fake_logger.debug.assert_called_once_with("Invalid X-Voice-To-Voice-Start header")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_audio_speech_request_state_failure_log_is_sanitized(monkeypatch):
    class _StateFailingRequest:
        headers = {}

        @property
        def state(self):
            raise RuntimeError("request state leaked /private/tts-state.json")

    async def _resolve_tts_byok(*args, **kwargs):
        _ = args, kwargs
        return (1, {}, None)

    _patch_audio_shim(monkeypatch, _resolve_tts_byok)
    fake_logger = MagicMock()
    monkeypatch.setattr(audio_tts, "logger", fake_logger)

    response = await audio_tts.create_speech(
        _request_data(),
        _StateFailingRequest(),
        tts_service=_SuccessfulTTSService(),
        current_user=SimpleNamespace(id=1),
        media_db=None,
        usage_log=SimpleNamespace(log_event=lambda *args, **kwargs: None),
    )

    assert response.status_code == 200
    fake_logger.debug.assert_called_once_with("Failed to read voice_to_voice_start from request.state")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_audio_speech_usage_log_failure_log_is_sanitized(monkeypatch):
    async def _resolve_tts_byok(*args, **kwargs):
        _ = args, kwargs
        return (1, {}, None)

    _patch_audio_shim(monkeypatch, _resolve_tts_byok)
    fake_logger = MagicMock()
    monkeypatch.setattr(audio_tts, "logger", fake_logger)

    response = await audio_tts.create_speech(
        _request_data(),
        _make_request(),
        tts_service=_SuccessfulTTSService(),
        current_user=SimpleNamespace(id=1),
        media_db=None,
        usage_log=_FailingUsageLog(),
    )

    assert response.status_code == 200
    fake_logger.debug.assert_called_once_with("usage_log audio.tts failed")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_audio_metadata_usage_log_failure_log_is_sanitized(monkeypatch):
    async def _resolve_tts_byok(*args, **kwargs):
        _ = args, kwargs
        return (1, {}, None)

    _patch_audio_shim(monkeypatch, _resolve_tts_byok)
    fake_logger = MagicMock()
    monkeypatch.setattr(audio_tts, "logger", fake_logger)

    response = await audio_tts.create_speech_metadata(
        _request_data(),
        _make_request(path="/api/v1/audio/speech/metadata"),
        tts_service=_SuccessfulTTSService(),
        current_user=SimpleNamespace(id=1),
        usage_log=_FailingUsageLog(),
    )

    assert response.status_code == 204
    fake_logger.debug.assert_called_once_with("usage_log audio.tts.metadata failed")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_audio_speech_history_hash_failure_log_is_sanitized(monkeypatch):
    async def _resolve_tts_byok(*args, **kwargs):
        _ = args, kwargs
        return (1, {}, None)

    def _fail_text_hash(*args, **kwargs):
        _ = args, kwargs
        raise RuntimeError("history hash leaked /private/tts-history-hash.json")

    _patch_audio_shim(monkeypatch, _resolve_tts_byok)
    fake_logger = MagicMock()
    monkeypatch.setattr(audio_tts, "logger", fake_logger)
    monkeypatch.setattr(audio_tts, "_tts_history_config", lambda: {"enabled": True, "store_text": True, "store_failed": True, "hash_key": None})
    monkeypatch.setattr(audio_tts, "compute_tts_history_text_hash", _fail_text_hash)

    response = await audio_tts.create_speech(
        _request_data(),
        _make_request(),
        tts_service=_SuccessfulTTSService(),
        current_user=SimpleNamespace(id=1),
        media_db=SimpleNamespace(create_tts_history_entry=lambda **kwargs: None),
        usage_log=SimpleNamespace(log_event=lambda *args, **kwargs: None),
    )

    assert response.status_code == 200
    fake_logger.debug.assert_called_once_with("TTS history: failed to compute text hash")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_audio_speech_history_write_failure_log_is_sanitized(monkeypatch):
    async def _resolve_tts_byok(*args, **kwargs):
        _ = args, kwargs
        return (1, {}, None)

    class _FailingMediaDB:
        def create_tts_history_entry(self, **kwargs):  # noqa: ARG002
            raise RuntimeError("history write leaked /private/tts-history-write.json")

    _patch_audio_shim(monkeypatch, _resolve_tts_byok)
    fake_logger = MagicMock()
    monkeypatch.setattr(audio_tts, "logger", fake_logger)
    monkeypatch.setattr(audio_tts, "_tts_history_config", lambda: {"enabled": True, "store_text": True, "store_failed": True, "hash_key": None})
    monkeypatch.setattr(audio_tts, "compute_tts_history_text_hash", lambda *args, **kwargs: "safe-hash")

    response = await audio_tts.create_speech(
        _request_data(),
        _make_request(),
        tts_service=_SuccessfulTTSService(),
        current_user=SimpleNamespace(id=1),
        media_db=_FailingMediaDB(),
        usage_log=SimpleNamespace(log_event=lambda *args, **kwargs: None),
    )

    assert response.status_code == 200
    fake_logger.debug.assert_called_once_with("TTS history: failed to write record")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_audio_speech_byok_touch_failure_log_is_sanitized(monkeypatch):
    async def _resolve_tts_byok(*args, **kwargs):
        _ = args, kwargs
        return (1, {}, _FailingTouchByokResolution(api_key="byok-key", auth_source="byok"))

    _patch_audio_shim(monkeypatch, _resolve_tts_byok)
    fake_logger = MagicMock()
    monkeypatch.setattr(audio_tts, "logger", fake_logger)

    response = await audio_tts.create_speech(
        _request_data(),
        _make_request(),
        tts_service=_SuccessfulTTSService(),
        current_user=SimpleNamespace(id=1),
        media_db=None,
        usage_log=SimpleNamespace(log_event=lambda *args, **kwargs: None),
    )

    assert response.status_code == 200
    fake_logger.debug.assert_called_once_with("Failed to update BYOK last_used timestamp")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_audio_speech_streaming_byok_touch_failure_log_is_sanitized(monkeypatch):
    async def _resolve_tts_byok(*args, **kwargs):
        _ = args, kwargs
        return (1, {}, _FailingTouchByokResolution(api_key="byok-key", auth_source="byok"))

    request_data = _request_data()
    request_data.stream = True

    _patch_audio_shim(monkeypatch, _resolve_tts_byok)
    fake_logger = MagicMock()
    monkeypatch.setattr(audio_tts, "logger", fake_logger)

    response = await audio_tts.create_speech(
        request_data,
        _make_request(),
        tts_service=_SuccessfulTTSService(),
        current_user=SimpleNamespace(id=1),
        media_db=None,
        usage_log=SimpleNamespace(log_event=lambda *args, **kwargs: None),
    )
    chunks = []
    async for chunk in response.body_iterator:
        chunks.append(chunk)

    assert response.status_code == 200
    assert chunks == [b"generated audio"]
    fake_logger.debug.assert_called_once_with("Failed to update BYOK last_used timestamp")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_audio_metadata_byok_touch_failure_log_is_sanitized(monkeypatch):
    async def _resolve_tts_byok(*args, **kwargs):
        _ = args, kwargs
        return (1, {}, _FailingTouchByokResolution(api_key="byok-key", auth_source="byok"))

    _patch_audio_shim(monkeypatch, _resolve_tts_byok)
    fake_logger = MagicMock()
    monkeypatch.setattr(audio_tts, "logger", fake_logger)

    response = await audio_tts.create_speech_metadata(
        _request_data(),
        _make_request(path="/api/v1/audio/speech/metadata"),
        tts_service=_SuccessfulTTSService(),
        current_user=SimpleNamespace(id=1),
        usage_log=SimpleNamespace(log_event=lambda *args, **kwargs: None),
    )

    assert response.status_code == 204
    fake_logger.debug.assert_called_once_with("Failed to update BYOK last_used timestamp")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_audio_speech_alignment_header_failure_log_is_sanitized(monkeypatch):
    async def _resolve_tts_byok(*args, **kwargs):
        _ = args, kwargs
        return (1, {}, None)

    request_data = _request_data()
    object.__setattr__(request_data, "_tts_metadata", {"alignment": {"bad": object()}})

    _patch_audio_shim(monkeypatch, _resolve_tts_byok)
    fake_logger = MagicMock()
    monkeypatch.setattr(audio_tts, "logger", fake_logger)

    response = await audio_tts.create_speech(
        request_data,
        _make_request(),
        tts_service=_SuccessfulTTSService(),
        current_user=SimpleNamespace(id=1),
        media_db=None,
        usage_log=SimpleNamespace(log_event=lambda *args, **kwargs: None),
    )

    assert response.status_code == 200
    assert "X-TTS-Alignment" not in response.headers
    fake_logger.debug.assert_called_once_with("Failed to encode alignment metadata header")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_audio_speech_openai_oauth_auth_failure_retries_once(monkeypatch):
    force_flags: list[bool] = []

    async def _resolve_tts_byok(*args, **kwargs):
        forced = bool(kwargs.get("force_oauth_refresh", False))
        force_flags.append(forced)
        resolution = _DummyByokResolution(
            api_key="oauth-refreshed-key" if forced else "oauth-initial-key",
            auth_source="oauth",
        )
        return (1, {"api_key": resolution.api_key}, resolution)

    _patch_audio_shim(monkeypatch, _resolve_tts_byok)

    tts_service = _AuthRetryTTSService(failures_before_success=1)
    response = await audio_tts.create_speech(
        _request_data(),
        _make_request(),
        tts_service=tts_service,
        current_user=SimpleNamespace(id=1),
        media_db=None,
        usage_log=SimpleNamespace(log_event=lambda *args, **kwargs: None),
    )

    assert response.status_code == 200
    assert response.body == b"recovered audio"
    assert tts_service.calls == 2
    assert force_flags[:2] == [False, True]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_audio_speech_download_link_sanitizes_storage_error(monkeypatch):
    async def _resolve_tts_byok(*args, **kwargs):
        _ = args, kwargs
        return (1, {}, None)

    async def _raise_storage_error(**kwargs):
        _ = kwargs
        raise StorageError("storage backend exploded at /private/generated/audio.mp3")

    shim_map = {
        "_sanitize_speech_request": lambda *args, **kwargs: "kitten_tts",
        "_resolve_tts_byok": _resolve_tts_byok,
        "save_and_register_tts_audio": _raise_storage_error,
    }

    def _shim_attr(name: str):
        if name not in shim_map:
            raise NameError(name)
        return shim_map[name]

    monkeypatch.setattr(audio_tts, "_audio_shim_attr", _shim_attr, raising=True)
    request_data = _request_data()
    request_data.return_download_link = True

    with pytest.raises(HTTPException) as exc_info:
        await audio_tts.create_speech(
            request_data,
            _make_request(),
            tts_service=_SuccessfulTTSService(),
            current_user=SimpleNamespace(id=1),
            media_db=None,
            usage_log=SimpleNamespace(log_event=lambda *args, **kwargs: None),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to store generated speech audio"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_audio_speech_openai_oauth_second_auth_failure_propagates_original_auth_error(monkeypatch):
    async def _resolve_tts_byok(*args, **kwargs):
        forced = bool(kwargs.get("force_oauth_refresh", False))
        resolution = _DummyByokResolution(
            api_key="oauth-refreshed-key" if forced else "oauth-initial-key",
            auth_source="oauth",
        )
        return (1, {"api_key": resolution.api_key}, resolution)

    _patch_audio_shim(monkeypatch, _resolve_tts_byok)

    with pytest.raises(HTTPException) as exc:
        await audio_tts.create_speech(
            _request_data(),
            _make_request(),
            tts_service=_AuthRetryTTSService(failures_before_success=2),
            current_user=SimpleNamespace(id=1),
            media_db=None,
            usage_log=SimpleNamespace(log_event=lambda *args, **kwargs: None),
        )

    assert exc.value.status_code == 502
    detail = exc.value.detail or {}
    assert detail.get("message") == "TTS provider authentication failed"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_audio_metadata_openai_oauth_auth_failure_retries_once(monkeypatch):
    force_flags: list[bool] = []

    async def _resolve_tts_byok(*args, **kwargs):
        forced = bool(kwargs.get("force_oauth_refresh", False))
        force_flags.append(forced)
        resolution = _DummyByokResolution(
            api_key="oauth-refreshed-key" if forced else "oauth-initial-key",
            auth_source="oauth",
        )
        return (1, {"api_key": resolution.api_key}, resolution)

    _patch_audio_shim(monkeypatch, _resolve_tts_byok)

    response = await audio_tts.create_speech_metadata(
        _request_data(),
        _make_request(path="/api/v1/audio/speech/metadata"),
        tts_service=_AuthRetryTTSService(failures_before_success=1),
        current_user=SimpleNamespace(id=1),
        usage_log=SimpleNamespace(log_event=lambda *args, **kwargs: None),
    )

    assert response.status_code == 204
    assert force_flags[:2] == [False, True]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_audio_metadata_openai_oauth_second_auth_failure_propagates_original_auth_error(monkeypatch):
    async def _resolve_tts_byok(*args, **kwargs):
        forced = bool(kwargs.get("force_oauth_refresh", False))
        resolution = _DummyByokResolution(
            api_key="oauth-refreshed-key" if forced else "oauth-initial-key",
            auth_source="oauth",
        )
        return (1, {"api_key": resolution.api_key}, resolution)

    _patch_audio_shim(monkeypatch, _resolve_tts_byok)

    with pytest.raises(HTTPException) as exc:
        await audio_tts.create_speech_metadata(
            _request_data(),
            _make_request(path="/api/v1/audio/speech/metadata"),
            tts_service=_AuthRetryTTSService(failures_before_success=2),
            current_user=SimpleNamespace(id=1),
            usage_log=SimpleNamespace(log_event=lambda *args, **kwargs: None),
        )

    assert exc.value.status_code == 502
    detail = exc.value.detail or {}
    assert detail.get("message") == "TTS provider authentication failed"
