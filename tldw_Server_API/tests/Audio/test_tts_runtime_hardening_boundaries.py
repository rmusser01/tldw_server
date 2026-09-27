"""High-risk TTS runtime boundary regressions."""

from __future__ import annotations

import asyncio
import traceback
from collections.abc import AsyncGenerator
from types import SimpleNamespace

import httpx
import pytest
from loguru import logger

from tldw_Server_API.app.api.v1.schemas.audio_schemas import OpenAISpeechRequest
from tldw_Server_API.app.core.Audio import tts_service
from tldw_Server_API.app.core.AuthNZ.byok_runtime import (
    ByokResolutionError,
    ByokResolutionStatus,
    ResolvedByokCredentials,
)
from tldw_Server_API.app.core.TTS.adapter_registry import (
    TTSAdapterRegistry,
    TTSProvider,
)
from tldw_Server_API.app.core.TTS.adapters import elevenlabs_adapter as elevenlabs_adapter_module
from tldw_Server_API.app.core.TTS.adapters import openai_adapter as openai_adapter_module
from tldw_Server_API.app.core.TTS.adapters import qwen3_runtime_remote as qwen_remote_module
from tldw_Server_API.app.core.TTS.adapters.base import AudioFormat, ProviderStatus, TTSRequest
from tldw_Server_API.app.core.TTS.adapters.elevenlabs_adapter import (
    ElevenLabsAdapter,
    ElevenLabsTTSAdapter,
)
from tldw_Server_API.app.core.TTS.adapters.fish_s2_adapter import FishS2Adapter
from tldw_Server_API.app.core.TTS.adapters.openai_adapter import (
    OpenAIAdapter,
    OpenAITTSAdapter,
)
from tldw_Server_API.app.core.TTS.adapters.qwen3_runtime_remote import RemoteQwenRuntime
from tldw_Server_API.app.core.TTS.backends import fish_s2_native_http as fish_native_module
from tldw_Server_API.app.core.TTS.backends.fish_s2_commercial_api import (
    FishS2CommercialApiBackend,
)
from tldw_Server_API.app.core.TTS.backends.fish_s2_native_http import (
    FishS2NativeHttpBackend,
)
from tldw_Server_API.app.core.TTS.tts_exceptions import (
    TTSError,
    TTSGenerationError,
    TTSNetworkError,
    TTSProviderError,
    TTSProviderInitializationError,
    TTSProviderNotConfiguredError,
    TTSRateLimitError,
    TTSTimeoutError,
)
from tldw_Server_API.app.core.TTS.tts_resource_manager import HTTPConnectionPool
from tldw_Server_API.app.core.TTS.tts_service_v2 import TTSServiceV2


class _NoOverrideSnapshot:
    def enforce(self, _model: str | None) -> None:
        return None

    def ensure_healthy(self) -> None:
        return None

    def server_fallback(self, base_fallback=None):
        return base_fallback


def _request_context() -> SimpleNamespace:
    return SimpleNamespace(state=SimpleNamespace())


def _assert_detached_exception(exc: BaseException, *sentinels: str) -> None:
    """Assert a public error does not retain its sensitive source exception."""

    assert exc.__cause__ is None
    assert exc.__context__ is None
    rendered = "".join(traceback.format_exception(exc))
    for sentinel in sentinels:
        assert sentinel not in rendered


@pytest.mark.asyncio
async def test_real_tts_registry_keeps_authoritative_key_endpoint_and_headers_atomic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A later live config B cannot splice into already-resolved snapshots."""

    registry = TTSAdapterRegistry(
        {
            "providers": {
                "openai": {
                    "enabled": True,
                    "api_key": "live-key-before",
                    "base_url": "https://live-before.example/v1/audio/speech",
                    "org_id": "live-org-before",
                    "project_id": "live-project-before",
                }
            }
        }
    )
    snapshots_ready = asyncio.Event()
    release_dispatch = asyncio.Event()

    async def _skip_remote_initialization(self) -> bool:
        self._initialized = True
        self._status = ProviderStatus.AVAILABLE
        return True

    monkeypatch.setattr(OpenAIAdapter, "ensure_initialized", _skip_remote_initialization)
    monkeypatch.setenv("OPENAI_API_KEY", "later-env-key-b")

    async def _dispatch(overrides: dict[str, object]):
        snapshots_ready.set()
        await release_dispatch.wait()
        return await registry.create_adapter_with_overrides(TTSProvider.OPENAI, overrides)

    endpoint_a = "https://generation-a.example/v1/audio/speech"
    task_a = asyncio.create_task(
        _dispatch(
            {
                "credentials_resolved": True,
                "api_key": "generation-key-a",
                "openai_api_key": "generation-key-a",
                "openai_base_url": endpoint_a,
                "organization": "generation-org-a",
                "project": "generation-project-a",
            }
        )
    )
    task_without_endpoint = asyncio.create_task(
        _dispatch(
            {
                "credentials_resolved": True,
                "api_key": "generation-key-without-endpoint",
                "openai_api_key": "generation-key-without-endpoint",
            }
        )
    )
    await snapshots_ready.wait()
    registry.config["providers"]["openai"] = {
        "enabled": True,
        "api_key": "later-live-key-b",
        "base_url": "https://later-live-b.example/v1/audio/speech",
        "org_id": "later-live-org-b",
        "project_id": "later-live-project-b",
    }
    release_dispatch.set()

    adapter_a, adapter_without_endpoint = await asyncio.gather(
        task_a,
        task_without_endpoint,
    )

    assert isinstance(adapter_a, OpenAIAdapter)
    assert isinstance(adapter_without_endpoint, OpenAIAdapter)
    assert (adapter_a.api_key, adapter_a.base_url) == ("generation-key-a", endpoint_a)
    assert adapter_a._request_headers() == {
        "Authorization": "Bearer generation-key-a",
        "Content-Type": "application/json",
        "OpenAI-Organization": "generation-org-a",
        "OpenAI-Project": "generation-project-a",
    }
    assert adapter_without_endpoint.api_key == "generation-key-without-endpoint"
    assert adapter_without_endpoint.base_url == "https://api.openai.com/v1/audio/speech"


@pytest.mark.asyncio
async def test_authoritative_keyless_tts_snapshot_never_uses_cached_adapter_or_fallback() -> None:
    """A marker-only resolved snapshot is distinct from absent legacy overrides."""

    captured: dict[str, object] = {}

    class _Service(TTSServiceV2):
        async def _ensure_factory(self):
            return SimpleNamespace()

        def _convert_request(self, _request):
            return SimpleNamespace(extra_params={})

        def _resolve_observability_context(self, _request, *, explicit_request_id=None):
            return explicit_request_id or "request-id", None

        async def _prepare_generate_speech_request(self, **kwargs):
            captured["fallback"] = kwargs["fallback"]
            captured["provider_overrides"] = kwargs["provider_overrides"]
            raise TTSProviderNotConfiguredError("authoritative snapshot has no credentials")

    service = object.__new__(_Service)
    service._factory = None
    service._stream_errors_as_audio = False
    request = OpenAISpeechRequest(
        input="fail closed",
        model="tts-1",
        voice="alloy",
        response_format="mp3",
    )

    with pytest.raises(TTSProviderNotConfiguredError):
        async for _chunk in service.generate_speech(
            request,
            provider="openai",
            fallback=True,
            provider_overrides={"credentials_resolved": True},
        ):
            pass

    assert captured == {
        "fallback": False,
        "provider_overrides": {"credentials_resolved": True},
    }


def test_elevenlabs_registered_wrapper_honors_authoritative_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The registered compatibility wrapper cannot revive env/live credentials."""
    monkeypatch.setenv("ELEVENLABS_API_KEY", "stale-env-key")

    with pytest.raises(TTSProviderNotConfiguredError):
        ElevenLabsTTSAdapter({"credentials_resolved": True})

    endpoint = "https://generation-a.example/v1"
    adapter = ElevenLabsTTSAdapter(
        {
            "credentials_resolved": True,
            "api_key": "generic-key-b",
            "elevenlabs_api_key": "generation-key-a",
            "base_url": "https://live-endpoint-b.example/v1",
            "elevenlabs_base_url": endpoint,
        }
    )

    assert adapter.api_key == "generation-key-a"
    assert adapter.base_url == endpoint
    assert adapter.config["credentials_resolved"] is True


def test_openai_registered_wrapper_rejects_keyless_authoritative_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The registered OpenAI wrapper cannot revive a process env key."""
    monkeypatch.setenv("OPENAI_API_KEY", "stale-env-key")

    with pytest.raises(TTSProviderNotConfiguredError):
        OpenAITTSAdapter({"credentials_resolved": True})


@pytest.mark.parametrize(
    ("backend", "requires_api_key"),
    [
        (None, False),
        ("native_http", False),
        ("local_runtime", False),
        ("commercial_api", True),
        ("hosted", True),
        ("fish_audio", True),
    ],
)
def test_fish_s2_requires_api_key_only_for_hosted_backends(
    backend: str | None,
    requires_api_key: bool,
) -> None:
    """Fish credential gating mirrors the adapter's backend dispatch aliases."""
    provider_config = {} if backend is None else {"backend": backend}

    assert (
        tts_service._tts_provider_requires_api_key("fish_s2", provider_config)
        is requires_api_key
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("resolver_result", ("malformed", "wrong_provider"))
async def test_tts_credential_resolver_output_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    resolver_result: str,
) -> None:
    """TTS cannot project an unvalidated credential adapter result."""

    async def _resolver(_provider: str, **_kwargs):
        if resolver_result == "malformed":
            return object()
        return ResolvedByokCredentials(
            provider="elevenlabs",
            api_key="wrong-provider-key",
            app_config={},
            credential_fields={},
            source="user",
            allowlisted=True,
            status=ByokResolutionStatus.RESOLVED,
        )

    monkeypatch.setattr(tts_service, "load_server_config_snapshot", lambda: {})
    monkeypatch.setattr(
        tts_service,
        "capture_provider_override_call_snapshot",
        lambda _provider: _NoOverrideSnapshot(),
    )
    monkeypatch.setattr(
        tts_service,
        "get_tts_config",
        lambda: SimpleNamespace(providers={"openai": {"enabled": True}}),
    )

    with pytest.raises(ByokResolutionError) as exc_info:
        await tts_service._resolve_tts_byok(
            provider_hint="openai",
            model="tts-1",
            current_user=SimpleNamespace(id=7),
            request=_request_context(),
            credential_resolver=_resolver,
        )

    assert exc_info.value.code == "invalid_provider_credentials"
    assert exc_info.value.__context__ is None


@pytest.mark.asyncio
async def test_fish_s2_keyless_native_snapshot_dispatches_native_without_commercial_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit native_http is keyless; commercial_api remains key-required."""

    native_url = "http://127.0.0.1:18081"

    async def _resolver(provider: str, **_kwargs) -> ResolvedByokCredentials:
        return ResolvedByokCredentials(
            provider=provider,
            api_key=None,
            app_config=None,
            credential_fields={},
            source="none",
            allowlisted=True,
            status=ByokResolutionStatus.ABSENT,
        )

    monkeypatch.setattr(tts_service, "load_server_config_snapshot", lambda: {})
    monkeypatch.setattr(tts_service, "resolve_byok_credentials", _resolver)
    monkeypatch.setattr(
        tts_service,
        "capture_provider_override_call_snapshot",
        lambda _provider: _NoOverrideSnapshot(),
    )
    monkeypatch.setattr(
        tts_service,
        "get_tts_config",
        lambda: SimpleNamespace(
            providers={
                "fish_s2": {
                    "enabled": True,
                    "backend": "native_http",
                    "base_url": native_url,
                    "timeout": 15,
                }
            }
        ),
    )

    async def _native_healthy(_self) -> bool:
        return True

    async def _commercial_must_not_run(_self) -> bool:
        raise AssertionError("commercial Fish backend was selected")

    monkeypatch.setattr(FishS2NativeHttpBackend, "health_check", _native_healthy)
    monkeypatch.setattr(FishS2CommercialApiBackend, "health_check", _commercial_must_not_run)

    _user_id, overrides, _resolution = await tts_service._resolve_tts_byok(
        provider_hint="fish_s2",
        model="fish-s2-pro",
        current_user=SimpleNamespace(id=7),
        request=_request_context(),
    )
    registry = TTSAdapterRegistry(
        {
            "providers": {
                "fish_s2": {
                    "enabled": True,
                    "backend": "commercial_api",
                    "base_url": "https://later-commercial.example",
                    "api_key": "later-commercial-key",
                }
            }
        }
    )
    adapter = await registry.create_adapter_with_overrides(TTSProvider.FISH_S2, overrides)

    assert isinstance(adapter, FishS2Adapter)
    assert adapter.backend_name == "native_http"
    assert isinstance(adapter._backend, FishS2NativeHttpBackend)
    assert adapter._backend.base_url == native_url
    assert adapter._backend.api_key is None


@pytest.mark.asyncio
async def test_remote_tts_calls_mark_credential_urls_sensitive_and_openai_never_logs_them(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every remote TTS runtime enters the sensitive HTTP observability boundary."""

    sentinel_url = "https://user:secret@tts-sentinel.invalid/audio?token=private"
    captured: dict[str, dict[str, object]] = {}
    log_records: list[dict[str, object]] = []

    class _Response:
        status_code = 200
        content = b"audio"
        headers: dict[str, str] = {}

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {}

    async def _openai_post(**kwargs):
        captured["openai"] = kwargs
        return _Response()

    async def _qwen_post(**kwargs):
        captured["qwen"] = kwargs
        return _Response()

    async def _elevenlabs_fetch(**kwargs):
        captured["elevenlabs"] = kwargs
        return _Response()

    async def _fish_fetch(**kwargs):
        captured["fish"] = kwargs
        return _Response()

    monkeypatch.setattr(openai_adapter_module, "apost", _openai_post)
    monkeypatch.setattr(qwen_remote_module, "apost", _qwen_post)
    monkeypatch.setattr(elevenlabs_adapter_module, "afetch", _elevenlabs_fetch)
    monkeypatch.setattr(fish_native_module, "afetch", _fish_fetch)
    sink_id = logger.add(lambda message: log_records.append(dict(message.record)), level="DEBUG")
    try:
        openai = OpenAIAdapter(
            {"openai_api_key": "openai-key", "openai_base_url": sentinel_url}
        )
        await openai._generate_complete(openai._request_headers(), {"input": "hello"})

        qwen = RemoteQwenRuntime({"base_url": sentinel_url, "api_key": "qwen-key"})
        await qwen._generate_complete({"Authorization": "Bearer qwen-key"}, {"input": "hello"})

        elevenlabs = ElevenLabsAdapter(
            {"elevenlabs_api_key": "eleven-key", "elevenlabs_base_url": sentinel_url}
        )
        await elevenlabs._generate_complete_elevenlabs(
            text="hello",
            voice_id="voice",
            model_id="eleven_multilingual_v2",
            request=TTSRequest(text="hello", format=AudioFormat.MP3),
        )

        fish = FishS2NativeHttpBackend({"base_url": sentinel_url})
        await fish.synthesize(
            text="hello",
            response_format="wav",
            streaming=False,
            reference_id=None,
            extra_params=None,
        )
    finally:
        logger.remove(sink_id)

    assert set(captured) == {"openai", "qwen", "elevenlabs", "fish"}
    assert all(call.get("sensitive_observability") is True for call in captured.values())
    assert sentinel_url not in repr(log_records)
    assert "tts-sentinel.invalid" not in repr(log_records)


@pytest.mark.asyncio
async def test_remote_tts_transport_errors_never_retain_credential_derived_urls() -> None:
    """Adapter error normalization must not undo sensitive HTTP redaction."""

    sentinel_url = "https://user:secret@tts-error.invalid/audio?token=private"
    cases: list[tuple[str, object]] = []

    openai = OpenAIAdapter({"openai_api_key": "openai-key"})
    with pytest.raises(TTSNetworkError) as openai_error:
        await openai._raise_normalized_request_error(httpx.ConnectError(sentinel_url))
    cases.append(("openai", openai_error.value))

    qwen = RemoteQwenRuntime({"base_url": "https://qwen.example", "api_key": "qwen-key"})
    with pytest.raises(TTSNetworkError) as qwen_error:
        await qwen._raise_remote_error(httpx.ConnectError(sentinel_url))
    cases.append(("qwen", qwen_error.value))

    fish_native = FishS2NativeHttpBackend({"base_url": "https://fish-native.example"})
    with pytest.raises(TTSNetworkError) as fish_native_error:
        fish_native._raise_transport_error(OSError(sentinel_url))
    cases.append(("fish_native", fish_native_error.value))

    fish_commercial = FishS2CommercialApiBackend(
        {"base_url": "https://fish-commercial.example", "api_key": "fish-key"}
    )
    with pytest.raises(TTSNetworkError) as fish_commercial_error:
        fish_commercial._raise_transport_error(OSError(sentinel_url))
    cases.append(("fish_commercial", fish_commercial_error.value))

    for provider, error in cases:
        rendered = f"{error!s} {getattr(error, 'details', {})!r}"
        assert sentinel_url not in rendered, provider
        assert "tts-error.invalid" not in rendered, provider
        assert getattr(error, "details", {}).get("error_type"), provider


@pytest.mark.asyncio
async def test_remote_tts_http_errors_never_retain_upstream_response_bodies() -> None:
    """Provider response bodies are untrusted and must not survive error mapping."""

    sentinel_body = "UPSTREAM_TTS_SECRET https://user:secret@body.invalid/path"
    request = httpx.Request("POST", "https://transport.invalid/audio")
    cases: list[tuple[str, object]] = []

    def _http_error(status_code: int) -> httpx.HTTPStatusError:
        response = httpx.Response(status_code, request=request, text=sentinel_body)
        return httpx.HTTPStatusError(
            str(status_code),
            request=request,
            response=response,
        )

    openai = OpenAIAdapter({"openai_api_key": "openai-key"})
    with pytest.raises(TTSProviderError) as openai_error:
        await openai._handle_http_status_error(_http_error(500))
    cases.append(("openai", openai_error.value))

    qwen = RemoteQwenRuntime({"base_url": "https://qwen.example", "api_key": "qwen-key"})
    with pytest.raises(TTSProviderError) as qwen_error:
        await qwen._handle_http_status_error(_http_error(500))
    cases.append(("qwen", qwen_error.value))

    elevenlabs = ElevenLabsAdapter({"elevenlabs_api_key": "eleven-key"})
    with pytest.raises(TTSProviderError) as elevenlabs_error:
        elevenlabs._raise_mapped_http_error(_http_error(400))
    cases.append(("elevenlabs", elevenlabs_error.value))

    for provider, backend in (
        ("fish_native", FishS2NativeHttpBackend({"base_url": "https://fish-native.example"})),
        (
            "fish_commercial",
            FishS2CommercialApiBackend(
                {"base_url": "https://fish-commercial.example", "api_key": "fish-key"}
            ),
        ),
    ):
        with pytest.raises(TTSProviderError) as fish_error:
            backend._raise_for_response_error(
                SimpleNamespace(status_code=500, text=sentinel_body, headers={})
            )
        cases.append((provider, fish_error.value))

    for provider, error in cases:
        rendered = f"{error!s} {getattr(error, 'details', {})!r}"
        assert sentinel_body not in rendered, provider
        assert "body.invalid" not in rendered, provider


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_elevenlabs_transport_failures_are_bounded_for_both_paths(
    monkeypatch: pytest.MonkeyPatch,
    streaming: bool,
) -> None:
    """ElevenLabs streaming and complete paths map transport failures identically."""

    sentinel_url = "https://user:secret@eleven-error.invalid/path?token=private"
    adapter = ElevenLabsAdapter({"elevenlabs_api_key": "eleven-key"})
    request = TTSRequest(
        text="hello",
        voice="rachel",
        format=AudioFormat.MP3,
        stream=False,
    )

    if streaming:
        async def _raise_stream(**_kwargs):
            raise httpx.ConnectError(sentinel_url)
            yield b"unreachable"

        monkeypatch.setattr(elevenlabs_adapter_module, "astream_bytes", _raise_stream)
        with pytest.raises(TTSNetworkError) as exc_info:
            async for _chunk in adapter._stream_audio_elevenlabs(
                text="hello",
                voice_id="voice",
                model_id="eleven_multilingual_v2",
                request=request,
            ):
                pass
    else:
        async def _raise_fetch(**_kwargs):
            raise httpx.ConnectError(sentinel_url)

        monkeypatch.setattr(elevenlabs_adapter_module, "afetch", _raise_fetch)
        with pytest.raises(TTSNetworkError) as exc_info:
            await adapter._generate_complete_elevenlabs(
                text="hello",
                voice_id="voice",
                model_id="eleven_multilingual_v2",
                request=request,
            )

    rendered = f"{exc_info.value!s} {exc_info.value.details!r}"
    assert sentinel_url not in rendered
    assert "eleven-error.invalid" not in rendered
    assert exc_info.value.details["error_type"] == "ConnectError"
    _assert_detached_exception(exc_info.value, sentinel_url, "eleven-error.invalid")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("transport_kind", "expected_error"),
    [
        ("network", TTSNetworkError),
        ("timeout", TTSTimeoutError),
    ],
)
async def test_elevenlabs_public_generate_preserves_transport_error_type(
    monkeypatch: pytest.MonkeyPatch,
    transport_kind: str,
    expected_error: type[Exception],
) -> None:
    """The public non-stream adapter preserves network/timeout classification."""

    sentinel_url = f"https://user:secret@eleven-{transport_kind}.invalid/path"
    adapter = ElevenLabsAdapter({"elevenlabs_api_key": "eleven-key"})

    async def _ready() -> bool:
        return True

    async def _fail_fetch(**_kwargs):
        if transport_kind == "timeout":
            raise httpx.ReadTimeout(sentinel_url)
        raise httpx.ConnectError(sentinel_url)

    monkeypatch.setattr(adapter, "ensure_initialized", _ready)
    monkeypatch.setattr(elevenlabs_adapter_module, "afetch", _fail_fetch)

    with pytest.raises(expected_error) as exc_info:
        await adapter.generate(
            TTSRequest(
                text="hello",
                voice="rachel",
                format=AudioFormat.MP3,
                stream=False,
            )
        )

    _assert_detached_exception(exc_info.value, sentinel_url)


@pytest.mark.asyncio
async def test_elevenlabs_malformed_http_detail_preserves_rate_limit_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-object provider detail cannot break typed HTTP normalization."""

    request = httpx.Request("POST", "https://eleven.example/v1/text-to-speech")
    response = httpx.Response(
        429,
        request=request,
        headers={"retry-after": "7"},
        json={"detail": "temporarily unavailable"},
    )
    adapter = ElevenLabsTTSAdapter({"api_key": "eleven-key"})

    async def _ready() -> bool:
        return True

    async def _response(**_kwargs):
        return response

    monkeypatch.setattr(adapter, "ensure_initialized", _ready)
    monkeypatch.setattr(elevenlabs_adapter_module, "afetch", _response)

    with pytest.raises(TTSRateLimitError) as exc_info:
        await adapter.generate(
            TTSRequest(
                text="hello",
                voice="rachel",
                model="eleven_multilingual_v2",
                format=AudioFormat.MP3,
                stream=False,
            )
        )

    assert exc_info.value.retry_after == 7


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["openai", "elevenlabs"])
@pytest.mark.parametrize(
    ("source_type", "expected_type"),
    [
        (TTSError, TTSProviderError),
        (TTSProviderError, TTSProviderError),
        (TTSNetworkError, TTSNetworkError),
    ],
)
async def test_remote_tts_mapper_replaces_unsafe_typed_errors_without_mutation(
    provider: str,
    source_type: type[TTSError],
    expected_type: type[TTSError],
) -> None:
    """Already-typed provider errors are inputs, never trusted output objects."""

    sentinel_url = f"https://user:secret@{provider}-typed.invalid/path?token=private"
    raw_cause = RuntimeError(sentinel_url)
    source = source_type(
        sentinel_url,
        provider=provider,
        details={"original_error": sentinel_url},
    )
    try:
        raise source from raw_cause
    except source_type:
        pass
    source_traceback = source.__traceback__

    with pytest.raises(expected_type) as exc_info:
        if provider == "openai":
            adapter = OpenAIAdapter({"openai_api_key": "openai-key"})
            await adapter._raise_normalized_request_error(source)
        else:
            adapter = ElevenLabsAdapter({"elevenlabs_api_key": "eleven-key"})
            adapter._raise_transport_error(source)

    bounded = exc_info.value
    assert bounded is not source
    assert sentinel_url not in f"{bounded!s} {bounded.details!r}"
    _assert_detached_exception(bounded, sentinel_url)
    assert source.__cause__ is raw_cause
    assert source.__traceback__ is source_traceback
    assert sentinel_url in f"{source!s} {source.details!r}"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "entrypoint",
    ["openai_wrapper", "elevenlabs_base", "elevenlabs_wrapper"],
)
@pytest.mark.parametrize("termination", ["aclose", "cancel"])
async def test_remote_tts_stream_deterministically_closes_inner_iterator(
    monkeypatch: pytest.MonkeyPatch,
    entrypoint: str,
    termination: str,
) -> None:
    """Early close and cancellation synchronously hand off inner HTTP cleanup."""

    inner_closed = asyncio.Event()
    inner_streams: list[AsyncGenerator[bytes, None]] = []

    async def _tracked_inner_stream():
        try:
            yield b"chunk"
            await asyncio.Event().wait()
        finally:
            inner_closed.set()

    def _inner_stream(**_kwargs):
        stream = _tracked_inner_stream()
        inner_streams.append(stream)
        return stream

    if entrypoint == "openai_wrapper":
        monkeypatch.setattr(openai_adapter_module, "astream_bytes", _inner_stream)
        adapter = OpenAITTSAdapter({"api_key": "openai-key"})
        stream = adapter.generate_stream(
            TTSRequest(
                text="hello",
                voice="alloy",
                model="tts-1",
                format=AudioFormat.MP3,
                stream=True,
            )
        )
    else:
        monkeypatch.setattr(elevenlabs_adapter_module, "astream_bytes", _inner_stream)
        if entrypoint == "elevenlabs_wrapper":
            adapter = ElevenLabsTTSAdapter({"api_key": "eleven-key"})
            stream = adapter.generate_stream(
                TTSRequest(
                    text="hello",
                    voice="rachel",
                    model="eleven_multilingual_v2",
                    format=AudioFormat.MP3,
                    stream=True,
                )
            )
        else:
            adapter = ElevenLabsAdapter({"elevenlabs_api_key": "eleven-key"})
            stream = adapter._stream_audio_elevenlabs(
                text="hello",
                voice_id="voice",
                model_id="eleven_multilingual_v2",
                request=TTSRequest(text="hello", format=AudioFormat.MP3),
            )

    closed_immediately = False
    try:
        assert await stream.__anext__() == b"chunk"
        if termination == "cancel":
            with pytest.raises(asyncio.CancelledError):
                await stream.athrow(asyncio.CancelledError())
        else:
            await stream.aclose()
        closed_immediately = inner_closed.is_set()
    finally:
        await stream.aclose()
        for inner_stream in inner_streams:
            await inner_stream.aclose()

    assert closed_immediately


@pytest.mark.asyncio
async def test_openai_wrapper_stream_transport_failure_is_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The production wrapper cannot rewrap a raw credential-derived URL."""

    sentinel_url = "https://user:secret@openai-stream.invalid/path?token=private"
    adapter = OpenAITTSAdapter({"openai_api_key": "openai-key"})

    async def _raise_stream(**_kwargs):
        raise httpx.ConnectError(sentinel_url)
        yield b"unreachable"

    monkeypatch.setattr(openai_adapter_module, "astream_bytes", _raise_stream)
    request = TTSRequest(
        text="hello",
        voice="alloy",
        model="tts-1",
        format=AudioFormat.MP3,
        stream=True,
    )

    with pytest.raises(TTSNetworkError) as exc_info:
        async for _chunk in adapter.generate_stream(request):
            pass

    rendered = f"{exc_info.value!s} {exc_info.value.details!r}"
    assert sentinel_url not in rendered
    assert "openai-stream.invalid" not in rendered
    assert exc_info.value.details["error_type"] == "ConnectError"
    _assert_detached_exception(exc_info.value, sentinel_url, "openai-stream.invalid")


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_openai_http_status_failure_detaches_request_headers_and_body(
    monkeypatch: pytest.MonkeyPatch,
    streaming: bool,
) -> None:
    """The real wrapper boundary must discard the full raw HTTP exchange."""

    sentinel_url = "https://user:secret@openai-status.invalid/v1/audio?token=private"
    sentinel_body = "RAW_OPENAI_STATUS_BODY_SECRET"
    raw_request = httpx.Request(
        "POST",
        sentinel_url,
        headers={"Authorization": "Bearer RAW_OPENAI_AUTH_SECRET"},
    )
    raw_response = httpx.Response(500, request=raw_request, text=sentinel_body)
    raw_error = httpx.HTTPStatusError(
        "provider failed",
        request=raw_request,
        response=raw_response,
    )
    adapter = OpenAITTSAdapter(
        {"api_key": "openai-key", "base_url": "https://openai.example/v1"}
    )

    async def _ready() -> bool:
        return True

    async def _raise_post(**_kwargs):
        raise raw_error

    async def _raise_stream(**_kwargs):
        raise raw_error
        yield b"unreachable"

    monkeypatch.setattr(adapter, "ensure_initialized", _ready)
    monkeypatch.setattr(openai_adapter_module, "apost", _raise_post)
    monkeypatch.setattr(openai_adapter_module, "astream_bytes", _raise_stream)
    request = TTSRequest(
        text="hello",
        voice="alloy",
        model="tts-1",
        format=AudioFormat.MP3,
        stream=streaming,
    )

    with pytest.raises((TTSProviderError, TTSGenerationError)) as exc_info:
        if streaming:
            async for _chunk in adapter.generate_stream(request):
                pass
        else:
            await adapter.generate(request)

    _assert_detached_exception(
        exc_info.value,
        sentinel_url,
        "openai-status.invalid",
        sentinel_body,
        "RAW_OPENAI_AUTH_SECRET",
    )
    rendered = f"{exc_info.value!s} {exc_info.value.details!r}"
    assert sentinel_body not in rendered
    assert "RAW_OPENAI_AUTH_SECRET" not in rendered


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["openai", "elevenlabs"])
async def test_concurrent_remote_tts_failures_detach_each_request_exception_graph(
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
) -> None:
    """Concurrent adapter failures must not retain or cross request endpoints."""

    entered = 0
    both_entered = asyncio.Event()
    release = asyncio.Event()
    hosts = [f"{provider}-error-alpha.invalid", f"{provider}-error-beta.invalid"]

    async def _raise_after_barrier(**kwargs):
        nonlocal entered
        entered += 1
        if entered == 2:
            both_entered.set()
        await release.wait()
        raise httpx.ConnectError(str(kwargs["url"]))
        yield b"unreachable"

    if provider == "openai":
        monkeypatch.setattr(openai_adapter_module, "astream_bytes", _raise_after_barrier)
        adapters = [
            OpenAITTSAdapter(
                {"api_key": f"key-{index}", "base_url": f"https://user:secret@{host}/v1"}
            )
            for index, host in enumerate(hosts)
        ]
        voice, model = "alloy", "tts-1"
    else:
        monkeypatch.setattr(elevenlabs_adapter_module, "astream_bytes", _raise_after_barrier)
        adapters = [
            ElevenLabsTTSAdapter(
                {"api_key": f"key-{index}", "base_url": f"https://user:secret@{host}/v1"}
            )
            for index, host in enumerate(hosts)
        ]
        voice, model = "rachel", "eleven_multilingual_v2"

    async def _consume(adapter) -> None:
        request = TTSRequest(
            text="hello",
            voice=voice,
            model=model,
            format=AudioFormat.MP3,
            stream=True,
        )
        async for _chunk in adapter.generate_stream(request):
            pass

    tasks = [asyncio.create_task(_consume(adapter)) for adapter in adapters]
    errors: list[object] = []
    try:
        await asyncio.wait_for(both_entered.wait(), timeout=5.0)
        release.set()
        errors = list(
            await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=5.0,
            )
        )
    finally:
        release.set()
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    assert all(isinstance(error, TTSNetworkError) for error in errors)
    assert len({id(error) for error in errors}) == len(errors)
    for error in errors:
        assert isinstance(error, TTSNetworkError)
        _assert_detached_exception(error, *hosts)
        assert all(host not in f"{error!s} {error.details!r}" for host in hosts)


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["openai", "elevenlabs"])
async def test_remote_tts_error_mapper_detaches_an_active_caller_exception(
    provider: str,
) -> None:
    """A retry caller's active exception cannot become the new public context."""

    caller_sentinel = f"ACTIVE_CALLER_{provider.upper()}_SECRET"
    transport_url = f"https://user:secret@{provider}-mapper.invalid/path?token=private"
    try:
        raise RuntimeError(caller_sentinel)
    except RuntimeError:
        if provider == "openai":
            adapter = OpenAIAdapter({"openai_api_key": "openai-key"})
            with pytest.raises(TTSNetworkError) as exc_info:
                await adapter._raise_normalized_request_error(
                    httpx.ConnectError(transport_url)
                )
        else:
            adapter = ElevenLabsAdapter({"elevenlabs_api_key": "eleven-key"})
            with pytest.raises(TTSNetworkError) as exc_info:
                adapter._raise_transport_error(httpx.ConnectError(transport_url))

    _assert_detached_exception(
        exc_info.value,
        caller_sentinel,
        transport_url,
        f"{provider}-mapper.invalid",
    )


@pytest.mark.asyncio
async def test_elevenlabs_initialization_failure_does_not_retain_transport_details(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Initialization errors retain only a bounded exception type."""

    sentinel_url = "https://user:secret@eleven-init.invalid/path?token=private"

    class _ResourceManager:
        async def get_http_client(self, **_kwargs):
            raise httpx.ConnectError(sentinel_url)

    async def _resource_manager():
        return _ResourceManager()

    monkeypatch.setattr(elevenlabs_adapter_module, "get_resource_manager", _resource_manager)
    adapter = ElevenLabsAdapter({"elevenlabs_api_key": "eleven-key"})

    with pytest.raises(TTSProviderInitializationError) as exc_info:
        await adapter.initialize()

    rendered = f"{exc_info.value!s} {exc_info.value.details!r}"
    assert sentinel_url not in rendered
    assert "eleven-init.invalid" not in rendered
    assert exc_info.value.details["error_type"] == "ConnectError"
    _assert_detached_exception(exc_info.value, sentinel_url, "eleven-init.invalid")


@pytest.mark.asyncio
async def test_elevenlabs_generation_wrapper_does_not_retain_adapter_details(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unexpected adapter failures cannot persist credential-derived details."""

    sentinel_url = "https://user:secret@eleven-generate.invalid/path?token=private"
    adapter = ElevenLabsAdapter({"elevenlabs_api_key": "eleven-key"})

    async def _ready() -> bool:
        return True

    async def _fail_generation(**_kwargs):
        raise OSError(sentinel_url)

    monkeypatch.setattr(adapter, "ensure_initialized", _ready)
    monkeypatch.setattr(adapter, "_generate_complete_elevenlabs", _fail_generation)
    request = TTSRequest(
        text="hello",
        voice="rachel",
        format=AudioFormat.MP3,
        stream=False,
    )

    with pytest.raises(TTSGenerationError) as exc_info:
        await adapter.generate(request)

    rendered = f"{exc_info.value!s} {exc_info.value.details!r}"
    assert sentinel_url not in rendered
    assert "eleven-generate.invalid" not in rendered
    assert exc_info.value.details["error_type"] == "OSError"
    _assert_detached_exception(exc_info.value, sentinel_url, "eleven-generate.invalid")


@pytest.mark.asyncio
async def test_tts_connection_stats_never_retain_credential_derived_base_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Operator-facing resource stats contain provider identity, never endpoints."""

    from tldw_Server_API.app.core import http_client

    sentinel_url = "https://user:secret@tts-stats.invalid/v1?token=private"

    class _Client:
        async def aclose(self) -> None:
            return None

    monkeypatch.setattr(http_client, "create_async_client", lambda **_kwargs: _Client())
    pool = HTTPConnectionPool()
    try:
        await pool.get_client("openai", base_url=sentinel_url)
        stats = pool.get_stats()
    finally:
        await pool.close_all()

    assert stats["openai"]["metadata"] == {"provider": "openai"}
    assert sentinel_url not in repr(stats)
    assert "tts-stats.invalid" not in repr(stats)


@pytest.mark.asyncio
async def test_tts_http_pool_construction_failure_is_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Client construction cannot retain a credential-derived endpoint."""

    from tldw_Server_API.app.core import http_client

    sentinel_url = "https://user:secret@tts-pool.invalid/v1?token=private"

    def _fail_client(**_kwargs):
        raise ValueError(sentinel_url)

    monkeypatch.setattr(http_client, "create_async_client", _fail_client)
    pool = HTTPConnectionPool()

    with pytest.raises(TTSNetworkError) as exc_info:
        await pool.get_client("openai", base_url=sentinel_url)

    rendered = f"{exc_info.value!s} {exc_info.value.details!r}"
    assert sentinel_url not in rendered
    assert "tts-pool.invalid" not in rendered
    assert exc_info.value.details["error_type"] == "ValueError"
    _assert_detached_exception(exc_info.value, sentinel_url, "tts-pool.invalid")
