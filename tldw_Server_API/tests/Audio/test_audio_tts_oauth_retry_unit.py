"""Unit tests for OpenAI OAuth retry behavior in audio TTS endpoints."""

from __future__ import annotations

import asyncio
from types import MappingProxyType, SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException
from starlette.requests import Request

import tldw_Server_API.app.api.v1.endpoints.audio.audio_tts as audio_tts
from tldw_Server_API.app.api.v1.endpoints.audio import audio as audio_endpoint
from tldw_Server_API.app.api.v1.schemas.audio_schemas import OpenAISpeechRequest
from tldw_Server_API.app.core.Audio import tts_service as tts_service_module
from tldw_Server_API.app.core.AuthNZ.byok_runtime import (
    ByokResolutionError,
    ResolvedByokCredentials,
)
from tldw_Server_API.app.core.AuthNZ.exceptions import StorageError
from tldw_Server_API.app.core.TTS.tts_exceptions import TTSAuthenticationError

_STALE_OAUTH_KEY = "tts-oauth-stale-secret-must-not-leak"
_REFRESHED_OAUTH_KEY = "tts-oauth-refreshed-secret-must-not-leak"
_SECOND_REFRESH_OAUTH_KEY = "tts-oauth-second-refresh-must-not-occur"
_STALE_OAUTH_GENERATION = "tts-oauth-generation-stale"
_REFRESHED_OAUTH_GENERATION = "tts-oauth-generation-refreshed"


def _oauth_resolution(
    *,
    api_key: str,
    generation: Any,
    provider: str = "openai",
) -> ResolvedByokCredentials:
    return ResolvedByokCredentials(
        provider=provider,
        api_key=api_key,
        app_config={},
        credential_fields={},
        source="user",
        allowlisted=True,
        auth_source="oauth",
        _credential_generation=generation,
    )


class _NoOverrideSnapshot:
    def enforce(self, _model: str | None) -> None:
        return None

    def ensure_healthy(self) -> None:
        return None

    def server_fallback(self, base_fallback=None):
        return base_fallback


class _CoalescingOAuthResolver:
    """Model the generation-aware coalescing contract at the endpoint seam."""

    def __init__(self) -> None:
        self.current_key = _STALE_OAUTH_KEY
        self.current_generation = _STALE_OAUTH_GENERATION
        self.token_exchange_count = 0
        self.initial_resolutions: list[ResolvedByokCredentials] = []
        self.rejected_resolutions: list[ResolvedByokCredentials | None] = []
        self._lock = asyncio.Lock()

    async def __call__(self, *_args, **kwargs):
        if not kwargs.get("force_oauth_refresh", False):
            resolution = _oauth_resolution(
                api_key=self.current_key,
                generation=self.current_generation,
            )
            self.initial_resolutions.append(resolution)
            return (1, {"api_key": resolution.api_key}, resolution)

        rejected = kwargs.get("rejected_credentials")
        self.rejected_resolutions.append(rejected)
        async with self._lock:
            rejected_generation = getattr(
                rejected,
                "_credential_generation",
                None,
            )
            if (
                rejected_generation is None
                or rejected_generation == self.current_generation
            ):
                self.token_exchange_count += 1
                if self.token_exchange_count == 1:
                    self.current_key = _REFRESHED_OAUTH_KEY
                    self.current_generation = _REFRESHED_OAUTH_GENERATION
                else:
                    self.current_key = _SECOND_REFRESH_OAUTH_KEY
                    self.current_generation = "tts-oauth-generation-second-refresh"
            resolution = _oauth_resolution(
                api_key=self.current_key,
                generation=self.current_generation,
            )
        return (1, {"api_key": resolution.api_key}, resolution)


class _ConcurrentAuthRetryTTSService:
    """Gate both stale requests at their first chunk before returning 401."""

    def __init__(self) -> None:
        self._stale_arrivals = 0
        self._both_stale_arrived = asyncio.Event()
        self.success_keys: list[str] = []

    def generate_speech(self, *args, **kwargs):  # noqa: ARG002
        overrides = kwargs.get("provider_overrides") or {}
        api_key = overrides.get("api_key")

        async def _gen():
            if api_key == _STALE_OAUTH_KEY:
                self._stale_arrivals += 1
                if self._stale_arrivals == 2:
                    self._both_stale_arrived.set()
                await self._both_stale_arrived.wait()
                raise TTSAuthenticationError("expired OAuth token")
            self.success_keys.append(api_key)
            yield b"recovered audio"

        return _gen()


def _assert_concurrent_refresh_coalesced(
    resolver: _CoalescingOAuthResolver,
    service: _ConcurrentAuthRetryTTSService,
) -> None:
    assert resolver.token_exchange_count == 1
    assert len(resolver.initial_resolutions) == 2
    assert len(resolver.rejected_resolutions) == 2
    assert {id(item) for item in resolver.rejected_resolutions} == {
        id(item) for item in resolver.initial_resolutions
    }
    assert all(
        item._credential_generation == _STALE_OAUTH_GENERATION
        for item in resolver.rejected_resolutions
    )
    assert service.success_keys == [
        _REFRESHED_OAUTH_KEY,
        _REFRESHED_OAUTH_KEY,
    ]


def _assert_oauth_secrets_not_serialized(
    payload: object,
    fake_logger: MagicMock,
) -> None:
    serialized = repr((payload, fake_logger.method_calls))
    assert _STALE_OAUTH_KEY not in serialized
    assert _REFRESHED_OAUTH_KEY not in serialized
    assert _SECOND_REFRESH_OAUTH_KEY not in serialized


def _patch_tts_helper_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        tts_service_module,
        "capture_provider_override_call_snapshot",
        lambda _provider: _NoOverrideSnapshot(),
    )
    monkeypatch.setattr(
        tts_service_module,
        "_capture_tts_provider_config",
        lambda _provider: {},
    )
    monkeypatch.setattr(
        tts_service_module,
        "load_server_config_snapshot",
        lambda: {},
    )
    monkeypatch.setattr(
        tts_service_module,
        "resolve_static_server_fallback_from_snapshot",
        lambda *_args: SimpleNamespace(
            api_key=None,
            app_config={},
            credential_fields={},
            auth_source=None,
        ),
    )


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
    fake_logger.debug.assert_called_once()
    log_args = fake_logger.debug.call_args.args
    assert log_args[0] == "TTS history: failed to write record request_id={}"
    assert isinstance(log_args[1], str)
    assert "/private/" not in repr(fake_logger.debug.call_args)


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
async def test_audio_metadata_oauth_retry_touches_only_refreshed_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stale = _DummyByokResolution(api_key="oauth-initial-key")
    refreshed = _DummyByokResolution(api_key="oauth-refreshed-key")
    force_flags: list[bool] = []

    async def _resolve_tts_byok(*_args, **kwargs):
        forced = bool(kwargs.get("force_oauth_refresh", False))
        force_flags.append(forced)
        resolution = refreshed if forced else stale
        return (1, {"api_key": resolution.api_key}, resolution)

    class _MetadataAuthRetryTTSService:
        def __init__(self) -> None:
            self.calls = 0

        def generate_speech(self, request_data: OpenAISpeechRequest, **_kwargs: Any):
            self.calls += 1
            call_number = self.calls

            async def _generate():
                if call_number == 1:
                    raise TTSAuthenticationError("oauth access token invalid")
                object.__setattr__(
                    request_data,
                    "_tts_metadata",
                    MappingProxyType({"alignment": {"words": []}}),
                )
                if False:  # pragma: no cover - make this an async generator
                    yield b""

            return _generate()

    _patch_audio_shim(monkeypatch, _resolve_tts_byok)
    response = await audio_tts.create_speech_metadata(
        _request_data(),
        _make_request(path="/api/v1/audio/speech/metadata"),
        tts_service=_MetadataAuthRetryTTSService(),
        current_user=SimpleNamespace(id=1),
        usage_log=SimpleNamespace(log_event=lambda *args, **kwargs: None),
    )

    assert response.status_code == 200
    assert force_flags == [False, True]
    assert stale.touch_calls == 0
    assert refreshed.touch_calls == 1


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


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.concurrent
async def test_audio_speech_streaming_concurrent_oauth_401s_coalesce_refresh_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sibling first-chunk 401s adopt one published OAuth generation."""
    resolver = _CoalescingOAuthResolver()
    service = _ConcurrentAuthRetryTTSService()
    fake_logger = MagicMock()
    _patch_audio_shim(monkeypatch, resolver)
    monkeypatch.setattr(audio_tts, "logger", fake_logger)

    async def _request_speech():
        request_data = _request_data()
        request_data.stream = True
        return await audio_tts.create_speech(
            request_data,
            _make_request(),
            tts_service=service,
            current_user=SimpleNamespace(id=1),
            media_db=None,
            usage_log=SimpleNamespace(log_event=lambda *args, **kwargs: None),
        )

    responses = await asyncio.wait_for(
        asyncio.gather(_request_speech(), _request_speech()),
        timeout=10,
    )
    response_payloads: list[bytes] = []
    for response in responses:
        chunks = [chunk async for chunk in response.body_iterator]
        response_payloads.append(b"".join(chunks))

    assert [response.status_code for response in responses] == [200, 200]
    assert response_payloads == [b"recovered audio", b"recovered audio"]
    _assert_concurrent_refresh_coalesced(resolver, service)
    _assert_oauth_secrets_not_serialized(
        [
            (response.status_code, dict(response.headers), payload)
            for response, payload in zip(responses, response_payloads)
        ],
        fake_logger,
    )


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.concurrent
async def test_audio_metadata_concurrent_oauth_401s_coalesce_refresh_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sibling metadata 401s adopt one published OAuth generation."""
    resolver = _CoalescingOAuthResolver()
    service = _ConcurrentAuthRetryTTSService()
    fake_logger = MagicMock()
    _patch_audio_shim(monkeypatch, resolver)
    monkeypatch.setattr(audio_tts, "logger", fake_logger)

    async def _request_metadata():
        return await audio_tts.create_speech_metadata(
            _request_data(),
            _make_request(path="/api/v1/audio/speech/metadata"),
            tts_service=service,
            current_user=SimpleNamespace(id=1),
            usage_log=SimpleNamespace(log_event=lambda *args, **kwargs: None),
        )

    responses = await asyncio.wait_for(
        asyncio.gather(_request_metadata(), _request_metadata()),
        timeout=10,
    )

    assert [response.status_code for response in responses] == [204, 204]
    _assert_concurrent_refresh_coalesced(resolver, service)
    _assert_oauth_secrets_not_serialized(
        [(response.status_code, dict(response.headers)) for response in responses],
        fake_logger,
    )


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("force_oauth_refresh", "expected_rejected_generation"),
    ((False, None), (True, _STALE_OAUTH_GENERATION)),
)
async def test_tts_byok_forwards_rejected_generation_only_for_forced_refresh(
    monkeypatch: pytest.MonkeyPatch,
    force_oauth_refresh: bool,
    expected_rejected_generation: str | None,
) -> None:
    """Only a forced refresh may forward the rejected credential generation."""
    _patch_tts_helper_config(monkeypatch)
    rejected = _oauth_resolution(
        api_key=_STALE_OAUTH_KEY,
        generation=_STALE_OAUTH_GENERATION,
    )
    resolver_calls: list[dict[str, Any]] = []

    async def _resolve_credentials(
        provider: str,
        **kwargs: Any,
    ) -> ResolvedByokCredentials:
        assert provider == "openai"
        resolver_calls.append(kwargs)
        return _oauth_resolution(
            api_key=_REFRESHED_OAUTH_KEY,
            generation=_REFRESHED_OAUTH_GENERATION,
        )

    await tts_service_module._resolve_tts_byok(
        provider_hint="openai",
        model="tts-1",
        current_user=SimpleNamespace(id=1),
        request=_make_request(),
        force_oauth_refresh=force_oauth_refresh,
        rejected_credentials=rejected,
        credential_resolver=_resolve_credentials,
    )

    assert len(resolver_calls) == 1
    if expected_rejected_generation is None:
        assert "rejected_credential_generation" not in resolver_calls[0]
    else:
        assert resolver_calls[0]["rejected_credential_generation"] == (
            expected_rejected_generation
        )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_audio_wrapper_forwards_rejected_credentials_and_injected_resolver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The compatibility wrapper preserves the rejected snapshot and test seam."""
    rejected = _oauth_resolution(
        api_key=_STALE_OAUTH_KEY,
        generation=_STALE_OAUTH_GENERATION,
    )
    captured: dict[str, Any] = {}

    async def _injected_resolver(*_args, **_kwargs):
        raise AssertionError("the core seam should receive, not invoke, this resolver")

    async def _capture_core_resolution(**kwargs):
        captured.update(kwargs)
        return (1, {"api_key": _REFRESHED_OAUTH_KEY}, rejected)

    monkeypatch.setattr(
        audio_endpoint,
        "resolve_byok_credentials",
        _injected_resolver,
    )
    monkeypatch.setattr(
        tts_service_module,
        "_resolve_tts_byok",
        _capture_core_resolution,
    )

    await audio_endpoint._resolve_tts_byok(
        provider_hint="openai",
        model="tts-1",
        current_user=SimpleNamespace(id=1),
        request=_make_request(),
        force_oauth_refresh=True,
        rejected_credentials=rejected,
    )

    assert captured["rejected_credentials"] is rejected
    assert captured["credential_resolver"] is _injected_resolver


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "rejected",
    (
        None,
        SimpleNamespace(
            provider="openai",
            _credential_generation=_STALE_OAUTH_GENERATION,
        ),
        _oauth_resolution(api_key=_STALE_OAUTH_KEY, generation=None),
        _oauth_resolution(api_key=_STALE_OAUTH_KEY, generation=" "),
        _oauth_resolution(api_key=_STALE_OAUTH_KEY, generation=object()),
        _oauth_resolution(
            api_key=_STALE_OAUTH_KEY,
            generation=_STALE_OAUTH_GENERATION,
            provider="anthropic",
        ),
    ),
    ids=(
        "missing-snapshot",
        "wrong-type",
        "missing-generation",
        "blank-generation",
        "generation-wrong-type",
        "provider-mismatch",
    ),
)
async def test_tts_byok_invalid_rejected_generation_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    rejected: object,
) -> None:
    """Invalid rejected metadata never reaches the credential resolver."""
    _patch_tts_helper_config(monkeypatch)
    resolver_called = False

    async def _unexpected_resolver(
        provider: str,
        **kwargs: Any,
    ) -> ResolvedByokCredentials:
        nonlocal resolver_called
        resolver_called = True
        return _oauth_resolution(
            api_key=_REFRESHED_OAUTH_KEY,
            generation=_REFRESHED_OAUTH_GENERATION,
        )

    with pytest.raises(ByokResolutionError) as exc_info:
        await tts_service_module._resolve_tts_byok(
            provider_hint="openai",
            model="tts-1",
            current_user=SimpleNamespace(id=1),
            request=_make_request(),
            force_oauth_refresh=True,
            rejected_credentials=rejected,
            credential_resolver=_unexpected_resolver,
        )

    assert exc_info.value.code == "invalid_provider_credentials"
    assert resolver_called is False
