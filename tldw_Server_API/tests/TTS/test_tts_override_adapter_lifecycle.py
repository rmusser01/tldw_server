"""Speech requests own adapters created for credential/config overrides."""

import asyncio
import io
from types import SimpleNamespace
from typing import Any

import pytest

from tldw_Server_API.app.api.v1.schemas.audio_schemas import OpenAISpeechRequest
from tldw_Server_API.app.core.TTS import tts_service_v2
from tldw_Server_API.app.core.TTS.adapter_registry import TTSAdapterFactory, TTSProvider
from tldw_Server_API.app.core.TTS.adapters.base import (
    AudioFormat,
    TTSAdapter,
    TTSCapabilities,
    TTSRequest,
    TTSResponse,
)
from tldw_Server_API.app.core.TTS.tts_exceptions import TTSGenerationError, TTSValidationError

pytestmark = pytest.mark.unit


@pytest.fixture
def speech_runtime(monkeypatch: pytest.MonkeyPatch) -> Any:
    adapters = []
    events = []
    waiting = asyncio.Event()
    initialized = asyncio.Event()

    class Adapter(TTSAdapter):
        PROVIDER_KEY = "mock"

        def __init__(self, config: dict[str, Any] | None = None) -> None:
            super().__init__(config)
            self.resource = io.BytesIO()
            self.mode = (config or {}).get("mode", "complete")
            adapters.append(self)

        async def initialize(self) -> bool:
            initialized.set()
            if self.mode == "init_wait":
                await asyncio.Event().wait()
            return self.mode != "init_false"

        async def ensure_initialized(self) -> bool:
            if self.mode == "init_raise":
                raise RuntimeError("Initialization failed")
            return await super().ensure_initialized()

        async def get_capabilities(self) -> TTSCapabilities:
            return TTSCapabilities(
                provider_name="mock",
                supported_languages={"en"},
                supported_voices=[],
                supported_formats={AudioFormat.MP3},
                max_text_length=500,
                supports_streaming=True,
            )

        async def generate(self, request: TTSRequest) -> TTSResponse:
            if self.mode == "failure":
                raise TTSGenerationError("Synthesis failed", provider="mock")

            async def stream():
                try:
                    yield b"speech"
                    if self.mode == "wait":
                        waiting.set()
                        await asyncio.Event().wait()
                finally:
                    events.append("stream_closed")

            return TTSResponse(audio_stream=stream(), format=AudioFormat.MP3, provider="mock")

        async def _cleanup_resources(self) -> None:
            events.append("adapter_closed")
            self.resource.close()

    async def resource_manager():
        return SimpleNamespace(touch_model=lambda *args: None)

    monkeypatch.setattr(tts_service_v2, "get_resource_manager", resource_manager)
    factory = TTSAdapterFactory({"providers": {"mock": {"enabled": True}}})
    factory.registry.register_adapter(TTSProvider.MOCK, Adapter)
    service = tts_service_v2.TTSServiceV2(factory=factory)
    return SimpleNamespace(
        service=service,
        adapters=adapters,
        events=events,
        waiting=waiting,
        initialized=initialized,
        adapter_type=Adapter,
    )


def speech(service: tts_service_v2.TTSServiceV2, overrides: dict[str, Any] | None, provider: str = "mock"):
    return service.generate_speech(
        OpenAISpeechRequest(input="Hello", model="mock", voice="voice", response_format="mp3", stream=True),
        provider=provider,
        provider_overrides=overrides,
        fallback=False,
    )


@pytest.mark.asyncio
async def test_override_adapter_closes_after_complete_stream(speech_runtime: Any) -> None:
    chunks = [chunk async for chunk in speech(speech_runtime.service, {"mode": "complete"})]
    assert chunks == [b"speech"]
    assert speech_runtime.adapters[0].resource.closed
    assert speech_runtime.events == ["stream_closed", "adapter_closed"]


@pytest.mark.asyncio
async def test_service_created_omnivoice_overrides_close_without_caller_overrides(
    speech_runtime: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = speech_runtime.service
    monkeypatch.setattr(speech_runtime.adapter_type, "PROVIDER_KEY", "omnivoice")
    service.factory.registry.config["providers"]["omnivoice"] = {"enabled": True}
    service.factory.registry.register_adapter(TTSProvider.OMNIVOICE, speech_runtime.adapter_type)
    monkeypatch.setattr(service, "_get_or_create_omnivoice_supervisor", lambda: object())
    assert [chunk async for chunk in speech(service, None, provider="omnivoice")] == [b"speech"]
    assert speech_runtime.adapters[0].resource.closed


@pytest.mark.asyncio
async def test_override_adapter_closes_on_generation_failure(speech_runtime: Any) -> None:
    with pytest.raises(TTSGenerationError):
        [chunk async for chunk in speech(speech_runtime.service, {"mode": "failure"})]
    assert speech_runtime.adapters[0].resource.closed
    assert speech_runtime.events == ["adapter_closed"]


@pytest.mark.asyncio
async def test_override_adapter_closes_when_consumer_stops_stream(speech_runtime: Any) -> None:
    stream = speech(speech_runtime.service, {"mode": "wait"})
    assert await stream.__anext__() == b"speech"
    await stream.aclose()
    assert speech_runtime.adapters[0].resource.closed
    assert speech_runtime.events == ["stream_closed", "adapter_closed"]


@pytest.mark.asyncio
async def test_override_adapter_closes_on_cancelled_generation(speech_runtime: Any) -> None:
    async def consume() -> None:
        [chunk async for chunk in speech(speech_runtime.service, {"mode": "wait"})]

    task = asyncio.create_task(consume())
    try:
        await asyncio.wait_for(speech_runtime.waiting.wait(), 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert speech_runtime.adapters[0].resource.closed
        assert speech_runtime.events == ["stream_closed", "adapter_closed"]
        assert speech_runtime.service._active_request_counts == {}
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_override_adapter_closes_when_provider_validation_rejects(
    speech_runtime: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    validate = tts_service_v2.validate_tts_request

    def provider_validation(request: Any, *, provider: str | None, config: Any) -> None:
        if provider == "mock":
            raise TTSValidationError("Provider-specific rejection", provider="mock")
        validate(request, provider=provider, config=config)

    monkeypatch.setattr(tts_service_v2, "validate_tts_request", provider_validation)
    with pytest.raises(TTSValidationError):
        [chunk async for chunk in speech(speech_runtime.service, {"mode": "complete"})]
    assert speech_runtime.adapters[0].resource.closed


@pytest.mark.asyncio
async def test_override_adapter_closes_when_cancelled_waiting_for_active_slot(speech_runtime: Any) -> None:
    service = speech_runtime.service
    service._active_request_counts["mock"] = 1
    await service._active_requests_lock.acquire()

    async def consume() -> None:
        [chunk async for chunk in speech(service, {"mode": "complete"})]

    task = asyncio.create_task(consume())
    try:
        await asyncio.wait_for(speech_runtime.initialized.wait(), 2)
        task.cancel()
    finally:
        service._active_requests_lock.release()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert speech_runtime.adapters[0].resource.closed
    assert service._active_request_counts == {"mock": 1}


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["init_false", "init_raise"])
async def test_registry_closes_override_adapter_when_initialization_fails(speech_runtime: Any, mode: str) -> None:
    adapter = await speech_runtime.service.factory.registry.create_adapter_with_overrides(
        TTSProvider.MOCK, {"mode": mode}
    )
    assert adapter is None
    assert speech_runtime.adapters[0].resource.closed
    assert speech_runtime.events == ["adapter_closed"]


@pytest.mark.asyncio
async def test_registry_closes_override_adapter_when_initialization_is_cancelled(speech_runtime: Any) -> None:
    task = asyncio.create_task(
        speech_runtime.service.factory.registry.create_adapter_with_overrides(TTSProvider.MOCK, {"mode": "init_wait"})
    )
    try:
        await asyncio.wait_for(speech_runtime.initialized.wait(), 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert speech_runtime.adapters[0].resource.closed
        assert speech_runtime.events == ["adapter_closed"]
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_cached_adapter_remains_open_after_generation(
    speech_runtime: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    cached = speech_runtime.adapter_type({})
    await cached.ensure_initialized()

    async def get_cached(provider: TTSProvider) -> TTSAdapter:
        assert provider == TTSProvider.MOCK
        return cached

    monkeypatch.setattr(speech_runtime.service.factory.registry, "get_adapter", get_cached)
    try:
        assert [chunk async for chunk in speech(speech_runtime.service, None)] == [b"speech"]
        assert not cached.resource.closed
        assert speech_runtime.events == ["stream_closed"]
    finally:
        await cached.close()


@pytest.mark.asyncio
async def test_cached_omnivoice_fallback_remains_open(speech_runtime: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    service = speech_runtime.service
    monkeypatch.setattr(speech_runtime.adapter_type, "PROVIDER_KEY", "omnivoice")
    cached = speech_runtime.adapter_type({})
    await cached.ensure_initialized()
    service.factory.registry._adapters["omnivoice"] = cached

    async def missing_adapter(*args: Any, **kwargs: Any) -> None:
        return None

    async def fallback_adapter(*args: Any, **kwargs: Any) -> TTSAdapter:
        return cached

    monkeypatch.setattr(service, "_get_adapter", missing_adapter)
    monkeypatch.setattr(service, "_get_fallback_adapter", fallback_adapter)
    try:
        chunks = [
            chunk
            async for chunk in service.generate_speech(
                OpenAISpeechRequest(input="Hello", model="mock", voice="voice", stream=True), fallback=True
            )
        ]
        assert chunks == [b"speech"]
        assert not cached.resource.closed
    finally:
        await cached.close()
