import asyncio
import json
import traceback
from unittest.mock import AsyncMock, MagicMock

import pytest
from loguru import logger

from tldw_Server_API.app.api.v1.schemas.audio_schemas import OpenAISpeechRequest
from tldw_Server_API.app.core.TTS.adapter_registry import TTSProvider
from tldw_Server_API.app.core.TTS.adapters.base import (
    AudioFormat,
    ProviderStatus,
    TTSAdapter,
    TTSCapabilities,
    TTSRequest,
    TTSResponse,
)
from tldw_Server_API.app.core.TTS.circuit_breaker import CircuitBreakerManager
from tldw_Server_API.app.core.TTS.tts_exceptions import (
    TTSAllProvidersFailedError,
    TTSAuthenticationError,
    TTSCircuitOpenError,
    TTSFallbackExhaustedError,
    TTSGenerationError,
    TTSGPUError,
    TTSInvalidVoiceReferenceError,
    TTSModelLoadError,
    TTSNetworkError,
    TTSProviderError,
    TTSProviderInitializationError,
    TTSProviderUnavailableError,
    TTSQuotaExceededError,
    TTSTimeoutError,
)
from tldw_Server_API.app.core.TTS.tts_service_v2 import (
    TTSServiceV2,
    _safe_tts_exception,
    _safe_tts_failure,
)


class FailingAdapter(TTSAdapter):
    def __init__(self, provider_name: str = "failing"):
        super().__init__({})
        self._status = ProviderStatus.AVAILABLE
        self._initialized = True
        self.provider_id = provider_name

    async def initialize(self) -> bool:
        self._initialized = True
        self._status = ProviderStatus.AVAILABLE
        return True

    async def generate(self, request: TTSRequest) -> TTSResponse:
        raise TTSProviderError("simulated failure", provider=self.provider_id)

    async def get_capabilities(self) -> TTSCapabilities:
        from tldw_Server_API.app.core.TTS.adapters.base import VoiceInfo
        return TTSCapabilities(
            provider_name=self.provider_id,
            supports_streaming=True,
            supports_voice_cloning=False,
            supported_languages={"en"},
            supported_formats={AudioFormat.MP3},
            max_text_length=5000,
            supported_voices=[VoiceInfo(id="v1", name="V1")],
        )


class AuthFailingStreamAdapter(TTSAdapter):
    PROVIDER_KEY = "openai"

    def __init__(self, provider_name: str = "openai"):
        super().__init__({})
        self._status = ProviderStatus.AVAILABLE
        self._initialized = True
        self.provider_id = provider_name

    async def initialize(self) -> bool:
        self._initialized = True
        self._status = ProviderStatus.AVAILABLE
        return True

    async def generate(self, request: TTSRequest) -> TTSResponse:
        async def _failing_stream():
            if False:
                yield b""
            raise TTSAuthenticationError("invalid api key", provider=self.provider_id)

        return TTSResponse(audio_stream=_failing_stream(), format=request.format)

    async def get_capabilities(self) -> TTSCapabilities:
        from tldw_Server_API.app.core.TTS.adapters.base import VoiceInfo
        return TTSCapabilities(
            provider_name=self.provider_id,
            supports_streaming=True,
            supports_voice_cloning=False,
            supported_languages={"en"},
            supported_formats={AudioFormat.MP3},
            max_text_length=5000,
            supported_voices=[VoiceInfo(id="v1", name="V1")],
        )


class AuthFailingAdapter(TTSAdapter):
    PROVIDER_KEY = "openai"

    def __init__(self, provider_name: str = "openai"):
        super().__init__({})
        self._status = ProviderStatus.AVAILABLE
        self._initialized = True
        self.provider_id = provider_name

    async def initialize(self) -> bool:
        self._initialized = True
        self._status = ProviderStatus.AVAILABLE
        return True

    async def generate(self, request: TTSRequest) -> TTSResponse:
        raise TTSAuthenticationError("invalid api key", provider=self.provider_id)

    async def get_capabilities(self) -> TTSCapabilities:
        from tldw_Server_API.app.core.TTS.adapters.base import VoiceInfo
        return TTSCapabilities(
            provider_name=self.provider_id,
            supports_streaming=True,
            supports_voice_cloning=False,
            supported_languages={"en"},
            supported_formats={AudioFormat.MP3},
            max_text_length=5000,
            supported_voices=[VoiceInfo(id="v1", name="V1")],
        )


class MetricsStub:
    def __init__(self):
        self.registered = []
        self.gauges = []
        self.increments = []
        self.observations = []

    def register_metric(self, *args, **kwargs):
        self.registered.append((args, kwargs))

    def set_gauge(self, *args, **kwargs):
        self.gauges.append((args, kwargs))

    def increment(self, *args, **kwargs):
        self.increments.append((args, kwargs))

    def observe(self, *args, **kwargs):
        self.observations.append((args, kwargs))


class SentinelFailureAdapter(TTSAdapter):
    """Adapter that gates, then raises a caller-provided provider failure."""

    def __init__(
        self,
        error: Exception,
        *,
        provider_key: str = "openai",
        started: asyncio.Event | None = None,
        release: asyncio.Event | None = None,
    ):
        super().__init__({})
        self.PROVIDER_KEY = provider_key
        self._error = error
        self._status = ProviderStatus.AVAILABLE
        self._initialized = True
        self._started = started
        self._release = release

    async def initialize(self) -> bool:
        return True

    async def generate(self, request: TTSRequest) -> TTSResponse:
        if self._started is not None:
            self._started.set()
        if self._release is not None:
            await self._release.wait()
        raise self._error

    async def get_capabilities(self) -> TTSCapabilities:
        from tldw_Server_API.app.core.TTS.adapters.base import VoiceInfo

        return TTSCapabilities(
            provider_name=self.provider_key,
            supports_streaming=True,
            supports_voice_cloning=False,
            supported_languages={"en"},
            supported_formats={AudioFormat.MP3, AudioFormat.PCM},
            max_text_length=5000,
            supported_voices=[VoiceInfo(id="v1", name="V1")],
        )


class CancellableStreamAdapter(TTSAdapter):
    """Adapter whose owned stream remains open until its consumer is cancelled."""

    PROVIDER_KEY = "openai"

    def __init__(
        self,
        *,
        started: asyncio.Event,
        release: asyncio.Event,
        closed: asyncio.Event,
    ):
        super().__init__({})
        self._status = ProviderStatus.AVAILABLE
        self._initialized = True
        self._started = started
        self._release = release
        self._closed = closed

    async def initialize(self) -> bool:
        return True

    async def generate(self, request: TTSRequest) -> TTSResponse:
        async def _stream():
            self._started.set()
            try:
                await self._release.wait()
                if False:
                    yield b""
            finally:
                self._closed.set()

        return TTSResponse(audio_stream=_stream(), format=request.format)

    async def get_capabilities(self) -> TTSCapabilities:
        from tldw_Server_API.app.core.TTS.adapters.base import VoiceInfo

        return TTSCapabilities(
            provider_name=self.provider_key,
            supports_streaming=True,
            supports_voice_cloning=False,
            supported_languages={"en"},
            supported_formats={AudioFormat.MP3},
            max_text_length=5000,
            supported_voices=[VoiceInfo(id="v1", name="V1")],
        )


def _service_for_adapter(
    adapter: TTSAdapter,
    *,
    max_concurrent_generations: int = 1,
) -> TTSServiceV2:
    factory = MagicMock()
    factory.get_adapter_by_model = AsyncMock(return_value=adapter)
    registry = MagicMock()
    registry.config = {
        "performance": {
            "max_concurrent_generations": max_concurrent_generations,
            "stream_errors_as_audio": True,
        }
    }
    factory.registry = registry
    service = TTSServiceV2(factory)
    service.metrics = MetricsStub()
    return service


def _assert_detached_exception_graph(error: Exception, *sentinels: str) -> None:
    """Assert no original failure is reachable through the public exception graph."""
    assert error.__cause__ is None
    assert error.__context__ is None

    seen: set[int] = set()
    pending = [error]
    rendered: list[str] = []
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        rendered.extend((str(current), repr(current), repr(vars(current))))
        for linked in (current.__cause__, current.__context__):
            if linked is not None:
                pending.append(linked)

    rendered.append("".join(traceback.format_exception(error)))
    graph_text = "\n".join(rendered)
    for sentinel in sentinels:
        assert sentinel not in graph_text


@pytest.mark.asyncio
async def test_stream_errors_as_audio_true_yields_error_bytes():
    # Factory/registry mock with compat flag enabled
    factory = MagicMock()
    factory.get_adapter_by_model = AsyncMock(return_value=FailingAdapter("mock"))
    registry = MagicMock()
    registry.config = {"performance": {"max_concurrent_generations": 1, "stream_errors_as_audio": True}}
    factory.registry = registry

    svc = TTSServiceV2(factory)
    svc.metrics = MetricsStub()

    req = OpenAISpeechRequest(input="Hello", model="mock", voice="v1", response_format="mp3")
    # Disable fallback so we hit direct error path
    chunks = []
    async for chunk in svc.generate_speech(req, fallback=False):
        chunks.append(chunk)

    assert len(chunks) >= 1
    joined = b"".join(chunks)
    assert joined.startswith(b"ERROR:")


@pytest.mark.asyncio
async def test_stream_errors_as_audio_false_raises_exception():
    factory = MagicMock()
    factory.get_adapter_by_model = AsyncMock(return_value=FailingAdapter("mock"))
    registry = MagicMock()
    registry.config = {"performance": {"max_concurrent_generations": 1, "stream_errors_as_audio": False}}
    factory.registry = registry

    svc = TTSServiceV2(factory)
    svc.metrics = MetricsStub()

    req = OpenAISpeechRequest(input="Hello", model="mock", voice="v1", response_format="mp3")

    with pytest.raises(TTSProviderError):
        # Consume the async generator to trigger exception
        async for _ in svc.generate_speech(req, fallback=False):
            pass


@pytest.mark.asyncio
async def test_streaming_auth_failure_records_authentication_breaker_category():
    factory = MagicMock()
    factory.get_adapter_by_model = AsyncMock(return_value=AuthFailingStreamAdapter("openai"))
    registry = MagicMock()
    registry.config = {"performance": {"max_concurrent_generations": 1, "stream_errors_as_audio": False}}
    factory.registry = registry

    circuit_manager = CircuitBreakerManager({})
    svc = TTSServiceV2(factory, circuit_manager=circuit_manager)
    svc.metrics = MetricsStub()

    req = OpenAISpeechRequest(
        input="Hello",
        model="tts-1",
        voice="v1",
        response_format="mp3",
        stream=True,
    )

    with pytest.raises(TTSAuthenticationError):
        async for _ in svc.generate_speech(req, fallback=False):
            pass

    breaker = await circuit_manager.get_breaker("openai")
    detailed = breaker.get_detailed_status()
    assert detailed["stats"]["failure_count"] == 1
    assert detailed["error_analysis"]["error_categories"]["authentication"] == 1


@pytest.mark.asyncio
async def test_non_streaming_auth_failure_records_authentication_breaker_category():
    factory = MagicMock()
    factory.get_adapter_by_model = AsyncMock(return_value=AuthFailingAdapter("openai"))
    registry = MagicMock()
    registry.config = {"performance": {"max_concurrent_generations": 1, "stream_errors_as_audio": False}}
    factory.registry = registry

    circuit_manager = CircuitBreakerManager({})
    svc = TTSServiceV2(factory, circuit_manager=circuit_manager)
    svc.metrics = MetricsStub()

    req = OpenAISpeechRequest(
        input="Hello",
        model="tts-1",
        voice="v1",
        response_format="mp3",
        stream=False,
    )

    with pytest.raises(TTSAuthenticationError):
        async for _ in svc.generate_speech(req, fallback=False):
            pass

    breaker = await circuit_manager.get_breaker("openai")
    detailed = breaker.get_detailed_status()
    assert detailed["stats"]["failure_count"] == 1
    assert detailed["error_analysis"]["error_categories"]["authentication"] == 1


def test_tts_service_default_stream_errors_as_audio_false(monkeypatch):


    """
    When no environment override or registry config is present,
    TTSServiceV2 should default to _stream_errors_as_audio == False so
    errors propagate as HTTP errors instead of embedded audio bytes.
    """
    # Ensure no env override is present
    monkeypatch.delenv("TTS_STREAM_ERRORS_AS_AUDIO", raising=False)

    # Factory without a registry/config so the service falls back to defaults
    factory = MagicMock()

    svc = TTSServiceV2(factory)
    assert svc._stream_errors_as_audio is False


@pytest.mark.parametrize(
    ("error_class", "message", "error_code", "category"),
    [
        (
            TTSInvalidVoiceReferenceError,
            "TTS request validation failed",
            "tts_validation_failed",
            "validation",
        ),
        (
            TTSAuthenticationError,
            "TTS provider authentication failed",
            "tts_provider_authentication_failed",
            "authentication",
        ),
        (
            TTSNetworkError,
            "TTS provider request failed",
            "tts_provider_network_failed",
            "network",
        ),
        (
            TTSProviderUnavailableError,
            "TTS provider unavailable",
            "tts_provider_unavailable",
            "provider_unavailable",
        ),
        (
            TTSProviderInitializationError,
            "TTS provider unavailable",
            "tts_provider_unavailable",
            "configuration",
        ),
        (
            TTSModelLoadError,
            "TTS model unavailable",
            "tts_model_unavailable",
            "model",
        ),
        (
            TTSQuotaExceededError,
            "TTS quota exceeded",
            "tts_provider_quota_exceeded",
            "quota",
        ),
        (
            TTSCircuitOpenError,
            "TTS provider request failed",
            "tts_provider_request_failed",
            "provider_error",
        ),
        (
            TTSAllProvidersFailedError,
            "TTS generation failed",
            "tts_generation_failed",
            "unknown",
        ),
        (
            TTSFallbackExhaustedError,
            "TTS generation failed",
            "tts_generation_failed",
            "unknown",
        ),
        (
            TTSGPUError,
            "TTS resource unavailable",
            "tts_resource_unavailable",
            "resource",
        ),
    ],
)
def test_safe_failure_metadata_matches_preserved_tts_subtype(
    error_class,
    message,
    error_code,
    category,
):
    raw_error = error_class(
        "https://provider.invalid?api_key=raw-secret",
        provider="openai",
    )

    failure = _safe_tts_failure(raw_error)
    safe_error = _safe_tts_exception(raw_error, "openai")

    assert type(safe_error) is error_class
    assert failure == {
        "message": message,
        "error_code": error_code,
        "error_type": error_class.__name__,
        "category": category,
    }
    assert safe_error.details == failure


class NetworkFailingAdapter(TTSAdapter):
    """Adapter that always fails with a retryable network-style error."""

    def __init__(self, provider_name: str = "openai"):
        super().__init__({})
        self._status = ProviderStatus.AVAILABLE
        self._initialized = True
        self.provider_id = provider_name

    async def initialize(self) -> bool:
        self._initialized = True
        self._status = ProviderStatus.AVAILABLE
        return True

    async def generate(self, request: TTSRequest) -> TTSResponse:
        raise TTSNetworkError("simulated network failure", provider=self.provider_id)

    async def get_capabilities(self) -> TTSCapabilities:
        from tldw_Server_API.app.core.TTS.adapters.base import VoiceInfo
        return TTSCapabilities(
            provider_name=self.provider_id,
            supports_streaming=True,
            supports_voice_cloning=False,
            supported_languages={"en"},
            supported_formats={AudioFormat.MP3},
            max_text_length=5000,
            supported_voices=[VoiceInfo(id="v1", name="V1")],
        )


class TimeoutFailingAdapter(NetworkFailingAdapter):
    """Adapter that always fails with a retryable timeout error."""

    async def generate(self, request: TTSRequest) -> TTSResponse:
        raise TTSTimeoutError("simulated timeout", provider=self.provider_id)


class FallbackSuccessAdapter(TTSAdapter):
    """Fallback adapter that returns successful audio."""

    def __init__(self, provider_name: str = "elevenlabs"):
        super().__init__({})
        self._status = ProviderStatus.AVAILABLE
        self._initialized = True
        self.provider_id = provider_name

    async def initialize(self) -> bool:
        self._initialized = True
        self._status = ProviderStatus.AVAILABLE
        return True

    async def generate(self, request: TTSRequest) -> TTSResponse:
        return TTSResponse(audio_data=b"fallback-audio", format=request.format)

    async def get_capabilities(self) -> TTSCapabilities:
        from tldw_Server_API.app.core.TTS.adapters.base import VoiceInfo
        return TTSCapabilities(
            provider_name=self.provider_id,
            supports_streaming=True,
            supports_voice_cloning=False,
            supported_languages={"en"},
            supported_formats={AudioFormat.MP3},
            max_text_length=5000,
            supported_voices=[VoiceInfo(id="v1", name="V1")],
        )


class EmptyResponseAdapter(FallbackSuccessAdapter):
    """Adapter that owns a valid request but returns no audio payload."""

    async def generate(self, request: TTSRequest) -> TTSResponse:
        return TTSResponse(format=request.format)


class MalformedProviderAdapter:
    """Fallback candidate whose provider identity cannot be resolved."""

    def __init__(self, sentinel: str):
        self._sentinel = sentinel

    @property
    def provider_key(self) -> str:
        raise RuntimeError(self._sentinel)


@pytest.mark.asyncio
async def test_network_error_triggers_fallback_and_metrics_increment():
    """TTSError subclasses like TTSNetworkError should trigger fallback and increment metrics."""

    primary_adapter = NetworkFailingAdapter("openai")
    fallback_adapter = FallbackSuccessAdapter("elevenlabs")

    class DummyRegistry:
        def __init__(self):
            # Minimal adapter specs mapping so _get_fallback_adapter can see these providers
            self._adapter_specs = {
                TTSProvider.OPENAI: object(),
                TTSProvider.ELEVENLABS: object(),
            }

        async def get_adapter(self, provider_enum: TTSProvider) -> TTSAdapter:
            if provider_enum == TTSProvider.OPENAI:
                return primary_adapter
            if provider_enum == TTSProvider.ELEVENLABS:
                return fallback_adapter
            raise TTSProviderError("provider not configured", provider=str(provider_enum.value))

    class DummyFactory:
        def __init__(self):
            self.registry = DummyRegistry()

        async def get_adapter_by_model(self, model: str) -> TTSAdapter:
            # Always return primary adapter for the initial model
            return primary_adapter

        async def get_best_adapter(self, *_, **__) -> TTSAdapter:
            # Fallback adapter chosen by _get_fallback_adapter
            return fallback_adapter

    factory = DummyFactory()
    svc = TTSServiceV2(factory)
    metrics = MetricsStub()
    svc.metrics = metrics

    req = OpenAISpeechRequest(
        input="Hello",
        model="tts-1",
        voice="alloy",
        response_format="mp3",
        stream=False,
    )

    chunks = []
    async for c in svc.generate_speech(req, fallback=True):
        chunks.append(c)

    joined = b"".join(chunks)
    assert b"fallback-audio" in joined

    # Ensure at least one fallback attempt metric was recorded
    fallback_metrics = [
        (args, kwargs)
        for args, kwargs in metrics.increments
        if args and args[0] == "tts_fallback_attempts"
    ]
    assert fallback_metrics, "Expected tts_fallback_attempts to be incremented on network error"


@pytest.mark.asyncio
async def test_timeout_error_triggers_fallback_and_metrics_increment():
    """TTSTimeoutError should also trigger fallback and increment metrics."""

    primary_adapter = TimeoutFailingAdapter("openai")
    fallback_adapter = FallbackSuccessAdapter("elevenlabs")

    class DummyRegistry:
        def __init__(self):
            self._adapter_specs = {
                TTSProvider.OPENAI: object(),
                TTSProvider.ELEVENLABS: object(),
            }

        async def get_adapter(self, provider_enum: TTSProvider) -> TTSAdapter:
            if provider_enum == TTSProvider.OPENAI:
                return primary_adapter
            if provider_enum == TTSProvider.ELEVENLABS:
                return fallback_adapter
            raise TTSProviderError("provider not configured", provider=str(provider_enum.value))

    class DummyFactory:
        def __init__(self):
            self.registry = DummyRegistry()

        async def get_adapter_by_model(self, model: str) -> TTSAdapter:
            return primary_adapter

        async def get_best_adapter(self, *_, **__) -> TTSAdapter:
            return fallback_adapter

    factory = DummyFactory()
    svc = TTSServiceV2(factory)
    metrics = MetricsStub()
    svc.metrics = metrics

    req = OpenAISpeechRequest(
        input="Hello",
        model="tts-1",
        voice="alloy",
        response_format="mp3",
        stream=False,
    )

    chunks = []
    async for c in svc.generate_speech(req, fallback=True):
        chunks.append(c)

    joined = b"".join(chunks)
    assert b"fallback-audio" in joined

    fallback_metrics = [
        (args, kwargs)
        for args, kwargs in metrics.increments
        if args and args[0] == "tts_fallback_attempts"
    ]
    assert fallback_metrics, "Expected tts_fallback_attempts to be incremented on timeout error"


@pytest.mark.asyncio
async def test_public_cancellation_propagates_and_cleans_owned_stream_without_fallback():
    started = asyncio.Event()
    release = asyncio.Event()
    closed = asyncio.Event()
    adapter = CancellableStreamAdapter(started=started, release=release, closed=closed)
    service = _service_for_adapter(adapter)
    service._stream_errors_as_audio = False
    service._get_fallback_adapter = AsyncMock(return_value=None)
    metric_spy = MagicMock(wraps=service._record_tts_metrics)
    service._record_tts_metrics = metric_spy
    request = OpenAISpeechRequest(
        input="Hello",
        model="tts-1",
        voice="v1",
        response_format="mp3",
        stream=True,
    )

    async def _consume() -> None:
        async for _chunk in service.generate_speech(request, fallback=True):
            pass

    task = asyncio.create_task(_consume())
    try:
        await asyncio.wait_for(started.wait(), timeout=2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.wait_for(closed.wait(), timeout=2)
    finally:
        release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    assert service._active_request_counts.get("openai", 0) == 0
    service._get_fallback_adapter.assert_not_awaited()
    assert not any(
        call.kwargs.get("success") is False
        for call in metric_spy.call_args_list
    )


@pytest.mark.asyncio
async def test_public_empty_primary_response_emits_one_failure_and_cleans_active_slot():
    service = _service_for_adapter(EmptyResponseAdapter("openai"))
    metric_spy = MagicMock(wraps=service._record_tts_metrics)
    service._record_tts_metrics = metric_spy
    request = OpenAISpeechRequest(
        input="Hello",
        model="tts-1",
        voice="v1",
        response_format="mp3",
        stream=False,
    )

    chunks = [
        chunk async for chunk in service.generate_speech(request, fallback=False)
    ]

    assert chunks == [b"ERROR: TTS generation failed"]
    terminal_metrics = [
        call.kwargs.get("success") for call in metric_spy.call_args_list
    ]
    assert terminal_metrics == [False]
    assert service._active_request_counts == {}


@pytest.mark.asyncio
async def test_public_empty_fallback_response_emits_one_terminal_and_cleans_slots():
    primary = SentinelFailureAdapter(
        TTSNetworkError("primary raw secret", provider="openai"),
        provider_key="openai",
    )
    empty_fallback = EmptyResponseAdapter("elevenlabs")
    service = _service_for_adapter(primary)
    service._get_fallback_adapter = AsyncMock(return_value=empty_fallback)
    metric_spy = MagicMock(wraps=service._record_tts_metrics)
    service._record_tts_metrics = metric_spy
    request = OpenAISpeechRequest(
        input="Hello",
        model="tts-1",
        voice="v1",
        response_format="mp3",
        stream=False,
    )

    chunks = [
        chunk async for chunk in service.generate_speech(request, fallback=True)
    ]

    assert chunks == [b"ERROR: TTS generation failed"]
    assert not any(
        call.kwargs.get("success") is True for call in metric_spy.call_args_list
    )
    assert service._active_request_counts == {}
    service._get_fallback_adapter.assert_awaited_once()


@pytest.mark.asyncio
async def test_public_retryable_fallback_reaches_third_real_adapter_without_error_audio():
    primary = SentinelFailureAdapter(
        TTSNetworkError("primary raw secret", provider="openai"),
        provider_key="openai",
    )
    first_fallback = SentinelFailureAdapter(
        TTSTimeoutError("fallback raw secret", provider="elevenlabs"),
        provider_key="elevenlabs",
    )
    final_fallback = FallbackSuccessAdapter("fish_s2")
    service = _service_for_adapter(primary)
    service._get_fallback_adapter = AsyncMock(
        side_effect=[first_fallback, final_fallback]
    )
    request = OpenAISpeechRequest(
        input="Hello",
        model="tts-1",
        voice="v1",
        response_format="mp3",
        stream=False,
    )

    audio = b"".join(
        [chunk async for chunk in service.generate_speech(request, fallback=True)]
    )

    assert audio == b"fallback-audio"
    assert service._get_fallback_adapter.await_count == 2


@pytest.mark.asyncio
async def test_public_three_adapter_failure_emits_one_terminal_error():
    primary = SentinelFailureAdapter(
        TTSNetworkError("primary raw secret", provider="openai"),
        provider_key="openai",
    )
    first_fallback = SentinelFailureAdapter(
        TTSNetworkError("first fallback raw secret", provider="elevenlabs"),
        provider_key="elevenlabs",
    )
    final_fallback = SentinelFailureAdapter(
        TTSTimeoutError("final fallback raw secret", provider="fish_s2"),
        provider_key="fish_s2",
    )
    service = _service_for_adapter(primary)
    service._get_fallback_adapter = AsyncMock(
        side_effect=[first_fallback, final_fallback]
    )
    request = OpenAISpeechRequest(
        input="Hello",
        model="tts-1",
        voice="v1",
        response_format="mp3",
        stream=False,
    )

    chunks = [
        chunk async for chunk in service.generate_speech(request, fallback=True)
    ]

    assert chunks == [b"ERROR: All providers failed"]
    assert service._get_fallback_adapter.await_count == 2


@pytest.mark.asyncio
async def test_partial_chunk_metadata_uses_bounded_provider_failure_fields():
    url_sentinel = "https://tts.invalid/v1?api_key=url-secret"
    body_sentinel = "response-body-secret"
    header_sentinel = "Bearer header-secret"
    provider_error = TTSProviderError(
        f"POST {url_sentinel} failed: {body_sentinel}; Authorization={header_sentinel}",
        provider="openai",
        error_code=header_sentinel,
        details={"response_body": body_sentinel, "url": url_sentinel},
    )
    service = TTSServiceV2()
    adapter = MagicMock(spec=TTSAdapter)
    calls = 0

    async def _generate(request: TTSRequest) -> TTSResponse:
        nonlocal calls
        calls += 1
        if calls == 1:
            return TTSResponse(
                audio_data=b"\x00\x00" * 200,
                format=AudioFormat.PCM,
                sample_rate=24000,
                provider="openai",
            )
        raise provider_error

    adapter.generate = AsyncMock(side_effect=_generate)
    adapter.get_capabilities = AsyncMock(
        return_value=TTSCapabilities(
            provider_name="openai",
            supports_streaming=False,
            supported_languages={"en"},
            supported_formats={AudioFormat.PCM},
            max_text_length=5000,
            supported_voices=[],
        )
    )
    adapter.convert_audio_format.return_value = None
    request = TTSRequest(
        text="Sentence one is long enough to split. Sentence two is also long enough to split.",
        format=AudioFormat.PCM,
        stream=False,
        extra_params={
            "chunking": True,
            "chunk_max_chars": 30,
            "segment_retry_max": 0,
            "segment_allow_partial": True,
            "segment_silence_on_fail": True,
        },
    )

    response = await service._generate_chunked_response(
        adapter=adapter,
        request=request,
        provider_key="openai",
        target_chars=20,
        max_chars=30,
        min_chars=10,
        crossfade_ms=0,
    )

    assert response is not None
    failed = next(segment for segment in response.metadata["segments"] if segment["status"] == "failed")
    assert failed["error"] == "TTS provider request failed"
    assert failed["error_type"] == "TTSProviderError"
    assert failed["error_code"] == "tts_provider_request_failed"
    assert failed["category"] == "provider_error"
    assert "status_code" not in failed
    serialized = json.dumps(response.metadata, sort_keys=True)
    for sentinel in (url_sentinel, body_sentinel, header_sentinel):
        assert sentinel not in serialized


@pytest.mark.asyncio
async def test_primary_provider_failure_redacts_audio_logs_and_metric_error():
    url_sentinel = "https://tts.invalid/generate?key=url-secret"
    body_sentinel = "provider-body-secret"
    header_sentinel = "Bearer provider-header-secret"
    error = TTSProviderError(
        f"POST {url_sentinel}: {body_sentinel}; {header_sentinel}",
        provider="openai",
        error_code=header_sentinel,
        details={"body": body_sentinel},
    )
    service = _service_for_adapter(SentinelFailureAdapter(error))
    metric_spy = MagicMock(wraps=service._record_tts_metrics)
    service._record_tts_metrics = metric_spy
    messages: list[str] = []
    sink_id = logger.add(lambda message: messages.append(str(message)), format="{message}")
    request = OpenAISpeechRequest(
        input="Hello",
        model="tts-1",
        voice="v1",
        response_format="mp3",
    )

    try:
        audio = b"".join([chunk async for chunk in service.generate_speech(request, fallback=False)])
    finally:
        logger.remove(sink_id)

    assert audio == b"ERROR: TTS provider request failed"
    failure_call = next(call for call in metric_spy.call_args_list if call.kwargs.get("success") is False)
    assert failure_call.kwargs["error"] == "tts_provider_request_failed"
    artifacts = audio.decode() + "\n" + "".join(messages) + "\n" + repr(metric_spy.call_args_list)
    for sentinel in (url_sentinel, body_sentinel, header_sentinel):
        assert sentinel not in artifacts


@pytest.mark.asyncio
async def test_concurrent_provider_failures_keep_bounded_failures_request_local():
    auth_url = "https://auth.invalid?api_key=auth-url-secret"
    auth_body = "auth-response-body-secret"
    auth_header = "Bearer auth-header-secret"
    runtime_url = "https://runtime.invalid?api_key=runtime-url-secret"
    runtime_body = "runtime-response-body-secret"
    runtime_header = "Bearer runtime-header-secret"
    auth_started = asyncio.Event()
    runtime_started = asyncio.Event()
    release = asyncio.Event()
    adapters = {
        "auth-model": SentinelFailureAdapter(
            TTSAuthenticationError(
                f"POST {auth_url}: {auth_body}; {auth_header}",
                provider="openai",
                error_code=auth_header,
                details={"body": auth_body},
            ),
            provider_key="openai",
            started=auth_started,
            release=release,
        ),
        "runtime-model": SentinelFailureAdapter(
            RuntimeError(f"POST {runtime_url}: {runtime_body}; {runtime_header}"),
            provider_key="elevenlabs",
            started=runtime_started,
            release=release,
        ),
    }
    factory = MagicMock()
    factory.get_adapter_by_model = AsyncMock(side_effect=lambda model: adapters[model])
    registry = MagicMock()
    registry.config = {
        "performance": {
            "max_concurrent_generations": 2,
            "stream_errors_as_audio": True,
        }
    }
    factory.registry = registry
    service = TTSServiceV2(factory)
    service.metrics = MetricsStub()
    metric_spy = MagicMock(wraps=service._record_tts_metrics)
    service._record_tts_metrics = metric_spy
    messages: list[str] = []
    sink_id = logger.add(lambda message: messages.append(str(message)), format="{message}")

    async def _collect(model: str) -> bytes:
        request = OpenAISpeechRequest(
            input=f"Hello from {model}",
            model=model,
            voice="v1",
            response_format="mp3",
        )
        return b"".join([chunk async for chunk in service.generate_speech(request, fallback=False)])

    auth_task = asyncio.create_task(_collect("auth-model"))
    runtime_task = asyncio.create_task(_collect("runtime-model"))
    try:
        await asyncio.wait_for(auth_started.wait(), timeout=1)
        await asyncio.wait_for(runtime_started.wait(), timeout=1)
        release.set()
        auth_audio, runtime_audio = await asyncio.gather(auth_task, runtime_task)
    finally:
        release.set()
        logger.remove(sink_id)

    assert auth_audio == b"ERROR: TTS provider authentication failed"
    assert runtime_audio == b"ERROR: TTS generation failed"
    error_codes = {
        call.kwargs["error"]
        for call in metric_spy.call_args_list
        if call.kwargs.get("success") is False
    }
    assert error_codes == {
        "tts_provider_authentication_failed",
        "tts_generation_failed",
    }
    artifacts = (
        auth_audio.decode()
        + runtime_audio.decode()
        + "\n"
        + "".join(messages)
        + "\n"
        + repr(metric_spy.call_args_list)
    )
    for sentinel in (
        auth_url,
        auth_body,
        auth_header,
        runtime_url,
        runtime_body,
        runtime_header,
    ):
        assert sentinel not in artifacts


@pytest.mark.asyncio
async def test_concurrent_provider_failures_raise_distinct_detached_errors():
    """Raw request failures cannot become context on concurrent public errors."""

    sentinels = (
        "https://auth-detach.invalid?token=auth-secret",
        "https://runtime-detach.invalid?token=runtime-secret",
    )
    started = [asyncio.Event(), asyncio.Event()]
    release = asyncio.Event()
    adapters = {
        "auth-model": SentinelFailureAdapter(
            TTSAuthenticationError(sentinels[0], provider="openai"),
            provider_key="openai",
            started=started[0],
            release=release,
        ),
        "runtime-model": SentinelFailureAdapter(
            RuntimeError(sentinels[1]),
            provider_key="elevenlabs",
            started=started[1],
            release=release,
        ),
    }
    factory = MagicMock()
    factory.get_adapter_by_model = AsyncMock(side_effect=lambda model: adapters[model])
    registry = MagicMock()
    registry.config = {
        "performance": {
            "max_concurrent_generations": 2,
            "stream_errors_as_audio": False,
        }
    }
    factory.registry = registry
    service = TTSServiceV2(factory)
    service.metrics = MetricsStub()

    async def _consume(model: str) -> None:
        request = OpenAISpeechRequest(
            input=f"Hello from {model}",
            model=model,
            voice="v1",
            response_format="mp3",
        )
        async for _chunk in service.generate_speech(request, fallback=False):
            pass

    tasks = [
        asyncio.create_task(_consume("auth-model")),
        asyncio.create_task(_consume("runtime-model")),
    ]
    try:
        await asyncio.wait_for(
            asyncio.gather(*(event.wait() for event in started)),
            timeout=2,
        )
        release.set()
        errors = await asyncio.gather(*tasks, return_exceptions=True)
    finally:
        release.set()
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    assert isinstance(errors[0], TTSAuthenticationError)
    assert isinstance(errors[1], TTSGenerationError)
    assert errors[0] is not errors[1]
    for error in errors:
        assert isinstance(error, Exception)
        assert error.__cause__ is None
        assert error.__context__ is None
        rendered = "".join(traceback.format_exception(error))
        assert all(sentinel not in rendered for sentinel in sentinels)


@pytest.mark.asyncio
async def test_legacy_fallback_boundary_detaches_full_failure_graph():
    primary_sentinel = "https://primary.invalid?api_key=primary-secret"
    fallback_sentinel = "https://fallback.invalid?api_key=fallback-secret"
    service = TTSServiceV2()
    service.generate = AsyncMock(
        side_effect=[
            TTSGenerationError(
                primary_sentinel,
                provider="openai",
                details={"url": primary_sentinel},
            ),
            TTSNetworkError(
                fallback_sentinel,
                provider="elevenlabs",
                details={"url": fallback_sentinel},
            ),
        ]
    )
    request = TTSRequest(
        text="Hello",
        voice="v1",
        model="tts-1",
        format=AudioFormat.MP3,
        stream=False,
    )

    with pytest.raises(TTSNetworkError) as exc_info:
        await service.generate_with_fallback(request, ["elevenlabs"])

    raised = exc_info.value
    assert raised.details == {
        "message": "TTS provider request failed",
        "error_code": "tts_provider_network_failed",
        "error_type": "TTSNetworkError",
        "category": "network",
    }
    _assert_detached_exception_graph(
        raised,
        primary_sentinel,
        fallback_sentinel,
    )


@pytest.mark.asyncio
async def test_legacy_fallback_boundary_detaches_active_caller_context():
    primary_sentinel = "https://primary.invalid?api_key=primary-secret"
    caller_sentinel = "https://caller.invalid?api_key=caller-secret"
    service = TTSServiceV2()
    service.generate = AsyncMock(
        side_effect=TTSGenerationError(primary_sentinel, provider="openai")
    )
    request = TTSRequest(
        text="Hello",
        voice="v1",
        model="tts-1",
        format=AudioFormat.MP3,
        stream=False,
    )

    try:
        raise RuntimeError(caller_sentinel)
    except RuntimeError:
        with pytest.raises(TTSGenerationError) as exc_info:
            await service.generate_with_fallback(request, [])

    raised = exc_info.value
    _assert_detached_exception_graph(
        raised,
        primary_sentinel,
        caller_sentinel,
    )


@pytest.mark.asyncio
async def test_adapter_boundary_failure_raises_detached_typed_error_without_audio():
    url_sentinel = "https://fallback.invalid?api_key=fallback-url-secret"
    body_sentinel = "fallback-response-body-secret"
    header_sentinel = "Bearer fallback-header-secret"
    adapter = SentinelFailureAdapter(
        TTSProviderError(
            f"POST {url_sentinel}: {body_sentinel}; {header_sentinel}",
            provider="openai",
            error_code=header_sentinel,
            details={"body": body_sentinel},
        )
    )
    service = TTSServiceV2()
    service.metrics = MetricsStub()
    service._stream_errors_as_audio = True
    request = TTSRequest(
        text="Hello",
        voice="v1",
        model="tts-1",
        format=AudioFormat.MP3,
        stream=False,
    )
    chunks: list[bytes] = []

    with pytest.raises(TTSProviderError) as exc_info:
        async for chunk in service._generate_with_adapter(adapter, request):
            chunks.append(chunk)

    assert chunks == []
    raised = exc_info.value
    assert isinstance(raised, Exception)
    details = getattr(raised, "details", {})
    assert details["message"] == "All providers failed"
    assert details["error_type"] == "TTSProviderError"
    assert details["error_code"] == "tts_provider_request_failed"
    assert "status_code" not in details
    assert raised.__cause__ is None
    assert raised.__context__ is None
    rendered = "".join(traceback.format_exception(raised))
    artifacts = b"".join(chunks).decode() + "\n" + json.dumps(details, sort_keys=True)
    for sentinel in (url_sentinel, body_sentinel, header_sentinel):
        assert sentinel not in artifacts
        assert sentinel not in rendered


@pytest.mark.asyncio
async def test_final_fallback_failure_uses_bounded_event_and_audio_metadata():
    first_sentinel = "https://fallback-one.invalid?api_key=first-secret"
    final_body_sentinel = "final-response-body-secret"
    final_header_sentinel = "Bearer final-header-secret"
    first_adapter = SentinelFailureAdapter(RuntimeError("unused"), provider_key="openai")
    final_adapter = SentinelFailureAdapter(RuntimeError("unused"), provider_key="elevenlabs")
    service = TTSServiceV2()
    service.metrics = MetricsStub()
    service._stream_errors_as_audio = True
    service._get_fallback_adapter = AsyncMock(side_effect=[first_adapter, final_adapter])

    async def _generate(adapter, request, **kwargs):
        if False:
            yield b""
        if adapter is first_adapter:
            raise TTSProviderError(
                f"POST {first_sentinel}",
                provider="openai",
                details={"url": first_sentinel},
            )
        raise RuntimeError(f"{final_body_sentinel}; {final_header_sentinel}")

    service._generate_with_adapter = _generate
    event_spy = MagicMock(wraps=service._record_fallback_event)
    service._record_fallback_event = event_spy
    messages: list[str] = []
    sink_id = logger.add(lambda message: messages.append(str(message)), format="{message}")
    request = TTSRequest(text="Hello", voice="v1", format=AudioFormat.MP3, stream=False)

    try:
        audio = b"".join(
            [
                chunk
                async for chunk in service._try_fallback_providers(
                    request,
                    exclude_providers=[],
                    failed_provider="primary",
                )
            ]
        )
    finally:
        logger.remove(sink_id)

    assert audio == b"ERROR: All providers failed"
    final_event = next(
        call
        for call in event_spy.call_args_list
        if call.kwargs.get("from_provider") == "openai"
        and call.kwargs.get("outcome") == "failed"
    )
    event_error = final_event.kwargs["error"]
    details = getattr(event_error, "details", {})
    assert details["message"] == "TTS generation failed"
    assert details["error_code"] == "tts_generation_failed"
    assert details["error_type"] == "TTSGenerationError"
    assert "status_code" not in details
    artifacts = audio.decode() + "\n" + json.dumps(details, sort_keys=True) + "\n" + "".join(messages)
    for sentinel in (first_sentinel, final_body_sentinel, final_header_sentinel):
        assert sentinel not in artifacts


@pytest.mark.asyncio
async def test_unexpected_fallback_failure_redacts_audio_and_logs():
    url_sentinel = "https://unexpected.invalid?api_key=unexpected-secret"
    body_sentinel = "unexpected-response-body-secret"
    header_sentinel = "Bearer unexpected-header-secret"
    adapter = SentinelFailureAdapter(RuntimeError("unused"), provider_key="openai")
    service = TTSServiceV2()
    service.metrics = MetricsStub()
    service._stream_errors_as_audio = True
    service._get_fallback_adapter = AsyncMock(return_value=adapter)

    async def _generate(*args, **kwargs):
        if False:
            yield b""
        raise RuntimeError(f"POST {url_sentinel}: {body_sentinel}; {header_sentinel}")

    service._generate_with_adapter = _generate
    messages: list[str] = []
    sink_id = logger.add(lambda message: messages.append(str(message)), format="{message}")
    request = TTSRequest(text="Hello", voice="v1", format=AudioFormat.MP3, stream=False)

    try:
        audio = b"".join(
            [
                chunk
                async for chunk in service._try_fallback_providers(
                    request,
                    exclude_providers=[],
                    failed_provider="primary",
                )
            ]
        )
    finally:
        logger.remove(sink_id)

    assert audio == b"ERROR: TTS generation failed"
    artifacts = audio.decode() + "\n" + "".join(messages)
    for sentinel in (url_sentinel, body_sentinel, header_sentinel):
        assert sentinel not in artifacts


@pytest.mark.asyncio
async def test_initial_fallback_selector_failure_emits_one_bounded_terminal():
    sentinel = "https://selector.invalid?api_key=selector-secret"
    service = TTSServiceV2()
    service.metrics = MetricsStub()
    service._stream_errors_as_audio = True
    service._get_fallback_adapter = AsyncMock(side_effect=RuntimeError(sentinel))
    request = TTSRequest(
        text="Hello",
        voice="v1",
        model="tts-1",
        format=AudioFormat.MP3,
        stream=False,
    )
    messages: list[str] = []
    sink_id = logger.add(lambda message: messages.append(str(message)), format="{message}")

    try:
        chunks = [
            chunk
            async for chunk in service._try_fallback_providers(
                request,
                exclude_providers=[],
                failed_provider="primary",
            )
        ]
    finally:
        logger.remove(sink_id)

    assert chunks == [b"ERROR: All providers failed"]
    assert sentinel not in "".join(messages)
    service._get_fallback_adapter.assert_awaited_once()


@pytest.mark.asyncio
async def test_second_fallback_selector_failure_raises_detached_terminal():
    first_sentinel = "https://first.invalid?api_key=first-secret"
    selector_sentinel = "https://selector.invalid?api_key=selector-secret"
    first_adapter = SentinelFailureAdapter(
        TTSNetworkError(first_sentinel, provider="openai"),
        provider_key="openai",
    )
    service = TTSServiceV2()
    service.metrics = MetricsStub()
    service._stream_errors_as_audio = False
    service._get_fallback_adapter = AsyncMock(
        side_effect=[first_adapter, RuntimeError(selector_sentinel)]
    )
    request = TTSRequest(
        text="Hello",
        voice="v1",
        model="tts-1",
        format=AudioFormat.MP3,
        stream=False,
    )

    with pytest.raises(TTSGenerationError) as exc_info:
        async for _chunk in service._try_fallback_providers(
            request,
            exclude_providers=[],
            failed_provider="primary",
        ):
            pass

    _assert_detached_exception_graph(
        exc_info.value,
        first_sentinel,
        selector_sentinel,
    )
    assert service._get_fallback_adapter.await_count == 2


@pytest.mark.asyncio
async def test_malformed_final_fallback_emits_one_bounded_terminal():
    malformed_sentinel = "https://malformed.invalid?api_key=malformed-secret"
    first_adapter = SentinelFailureAdapter(
        TTSNetworkError("first raw secret", provider="openai"),
        provider_key="openai",
    )
    service = TTSServiceV2()
    service.metrics = MetricsStub()
    service._stream_errors_as_audio = True
    service._get_fallback_adapter = AsyncMock(
        side_effect=[first_adapter, MalformedProviderAdapter(malformed_sentinel)]
    )
    request = TTSRequest(
        text="Hello",
        voice="v1",
        model="tts-1",
        format=AudioFormat.MP3,
        stream=False,
    )
    messages: list[str] = []
    sink_id = logger.add(lambda message: messages.append(str(message)), format="{message}")

    try:
        chunks = [
            chunk
            async for chunk in service._try_fallback_providers(
                request,
                exclude_providers=[],
                failed_provider="primary",
            )
        ]
    finally:
        logger.remove(sink_id)

    assert chunks == [b"ERROR: All providers failed"]
    assert malformed_sentinel not in "".join(messages)
    assert service._get_fallback_adapter.await_count == 2


@pytest.mark.asyncio
async def test_retryable_fallback_exhaustion_raises_detached_terminal_error():
    sentinel = "https://fallback.invalid?api_key=raw-secret"
    adapter = SentinelFailureAdapter(
        TTSNetworkError(sentinel, provider="openai"),
        provider_key="openai",
    )
    service = TTSServiceV2()
    service.metrics = MetricsStub()
    service._stream_errors_as_audio = False
    service._get_fallback_adapter = AsyncMock(side_effect=[adapter, None])
    request = TTSRequest(
        text="Hello",
        voice="v1",
        model="tts-1",
        format=AudioFormat.MP3,
        stream=False,
    )

    with pytest.raises(TTSFallbackExhaustedError) as exc_info:
        async for _chunk in service._try_fallback_providers(
            request,
            exclude_providers=[],
            failed_provider="primary",
        ):
            pass

    raised = exc_info.value
    assert raised.__cause__ is None
    assert raised.__context__ is None
    assert sentinel not in "".join(traceback.format_exception(raised))
    assert service._get_fallback_adapter.await_count == 2


@pytest.mark.asyncio
async def test_unavailable_fallback_detaches_active_caller_context():
    sentinel = "https://caller.invalid?api_key=caller-secret"
    service = TTSServiceV2()
    service.metrics = MetricsStub()
    service._stream_errors_as_audio = False
    service._get_fallback_adapter = AsyncMock(return_value=None)
    request = TTSRequest(
        text="Hello",
        voice="v1",
        model="tts-1",
        format=AudioFormat.MP3,
        stream=False,
    )

    try:
        raise RuntimeError(sentinel)
    except RuntimeError:
        with pytest.raises(TTSFallbackExhaustedError) as exc_info:
            async for _chunk in service._try_fallback_providers(
                request,
                exclude_providers=[],
                failed_provider="primary",
            ):
                pass

    raised = exc_info.value
    assert raised.__cause__ is None
    assert raised.__context__ is None
    assert sentinel not in "".join(traceback.format_exception(raised))
