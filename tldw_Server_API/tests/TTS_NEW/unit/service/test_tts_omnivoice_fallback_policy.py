import base64
import wave
from io import BytesIO

import pytest

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
from tldw_Server_API.app.core.TTS.tts_exceptions import TTSProviderError
from tldw_Server_API.app.core.TTS.tts_service_v2 import TTSServiceV2


pytestmark = pytest.mark.unit


def _make_reference_wav(duration_seconds: float, *, sample_rate: int = 24000) -> bytes:
    frame_count = max(1, int(duration_seconds * sample_rate))
    buffer = BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"\x00\x01" * frame_count)
    return buffer.getvalue()


class _FailingOmniVoiceAdapter(TTSAdapter):
    def __init__(self) -> None:
        super().__init__({})
        self._status = ProviderStatus.AVAILABLE
        self._initialized = True
        self.provider_id = "omnivoice"
        self.calls = 0

    async def initialize(self) -> bool:
        return True

    async def generate(self, request: TTSRequest) -> TTSResponse:
        self.calls += 1
        raise TTSProviderError("simulated retryable omnivoice failure", provider="omnivoice")

    async def get_capabilities(self) -> TTSCapabilities:
        return TTSCapabilities(
            provider_name="omnivoice",
            supported_languages={"en"},
            supported_voices=[],
            supported_formats={AudioFormat.WAV},
            max_text_length=5000,
            supports_streaming=False,
            supports_voice_cloning=True,
        )


class _SuccessfulOpenAIAdapter(TTSAdapter):
    def __init__(self) -> None:
        super().__init__({})
        self._status = ProviderStatus.AVAILABLE
        self._initialized = True
        self.provider_id = "openai"
        self.calls = 0

    async def initialize(self) -> bool:
        return True

    async def generate(self, request: TTSRequest) -> TTSResponse:
        self.calls += 1
        return TTSResponse(audio_data=b"fallback-audio", format=request.format, sample_rate=24000)

    async def get_capabilities(self) -> TTSCapabilities:
        return TTSCapabilities(
            provider_name="openai",
            supported_languages={"en"},
            supported_voices=[],
            supported_formats={AudioFormat.WAV},
            max_text_length=4096,
            supports_streaming=False,
            supports_voice_cloning=False,
        )


class _FailingOpenAIAdapter(_SuccessfulOpenAIAdapter):
    def __init__(self) -> None:
        super().__init__()
        self.provider_id = "openai"

    async def generate(self, request: TTSRequest) -> TTSResponse:
        self.calls += 1
        raise TTSProviderError("simulated retryable openai failure", provider="openai")


class _Registry:
    def __init__(self, omnivoice: _FailingOmniVoiceAdapter, openai: _SuccessfulOpenAIAdapter) -> None:
        self._omnivoice = omnivoice
        self._openai = openai
        self._adapter_specs = {
            TTSProvider.OMNIVOICE: object(),
            TTSProvider.OPENAI: object(),
        }

    async def create_adapter_with_overrides(self, provider_enum: TTSProvider, overrides):
        if provider_enum == TTSProvider.OMNIVOICE:
            return self._omnivoice
        raise TTSProviderError("provider not configured", provider=provider_enum.value)

    async def get_adapter(self, provider_enum: TTSProvider):
        if provider_enum == TTSProvider.OPENAI:
            return self._openai
        if provider_enum == TTSProvider.OMNIVOICE:
            return self._omnivoice
        raise TTSProviderError("provider not configured", provider=provider_enum.value)


class _Factory:
    def __init__(self) -> None:
        self.fake_omnivoice = _FailingOmniVoiceAdapter()
        self.fake_openai = _SuccessfulOpenAIAdapter()
        self.registry = _Registry(self.fake_omnivoice, self.fake_openai)

    def get_provider_for_model(self, model: str) -> TTSProvider:
        return TTSProvider.OMNIVOICE

    async def get_adapter_by_model(self, model: str):
        return self.fake_omnivoice

    async def get_best_adapter(self, *_, **__):
        return self.fake_openai


def _service_with_failing_omnivoice_and_successful_openai() -> TTSServiceV2:
    factory = _Factory()
    service = TTSServiceV2(factory)
    service._build_omnivoice_adapter_overrides = lambda overrides=None: {}
    service.fake_omnivoice = factory.fake_omnivoice
    service.fake_openai = factory.fake_openai
    return service


def _service_with_failing_openai_and_successful_fallback() -> TTSServiceV2:
    failing_openai = _FailingOpenAIAdapter()
    fallback_adapter = _SuccessfulOpenAIAdapter()
    fallback_adapter.provider_id = "elevenlabs"

    class _ExplicitOpenAIRegistry:
        def __init__(self) -> None:
            self._adapter_specs = {
                TTSProvider.OPENAI: object(),
                TTSProvider.ELEVENLABS: object(),
            }

        async def get_adapter(self, provider_enum: TTSProvider):
            if provider_enum == TTSProvider.OPENAI:
                return failing_openai
            if provider_enum == TTSProvider.ELEVENLABS:
                return fallback_adapter
            raise TTSProviderError("provider not configured", provider=provider_enum.value)

    class _ExplicitOpenAIFactory:
        def __init__(self) -> None:
            self.registry = _ExplicitOpenAIRegistry()

        def get_provider_for_model(self, model: str) -> TTSProvider:
            return TTSProvider.OPENAI

        async def get_adapter_by_model(self, model: str):
            return failing_openai

        async def get_best_adapter(self, *_, **__):
            return fallback_adapter

    service = TTSServiceV2(_ExplicitOpenAIFactory())
    service.fake_openai = failing_openai
    service.fake_fallback = fallback_adapter
    return service


async def _collect(request: OpenAISpeechRequest, service: TTSServiceV2) -> bytes:
    chunks = [chunk async for chunk in service.generate_speech(request, fallback=True)]
    return b"".join(chunks)


@pytest.mark.asyncio
async def test_explicit_omnivoice_instruct_request_does_not_fallback():
    service = _service_with_failing_omnivoice_and_successful_openai()
    request = OpenAISpeechRequest(
        model="tts-1",
        input="hello",
        voice="auto",
        response_format="wav",
        stream=False,
        extra_params={"instruct": "calm narrator"},
    )

    with pytest.raises(TTSProviderError):
        await _collect(request, service)

    assert service.fake_openai.calls == 0


@pytest.mark.asyncio
async def test_omnivoice_direct_voice_reference_does_not_fallback():
    service = _service_with_failing_omnivoice_and_successful_openai()
    request = OpenAISpeechRequest(
        model="tts-1",
        input="hello",
        voice="clone",
        voice_reference=base64.b64encode(_make_reference_wav(3.5)).decode("ascii"),
        response_format="wav",
        stream=False,
        extra_params={"reference_text": "reference transcript"},
    )

    with pytest.raises(TTSProviderError):
        await _collect(request, service)

    assert service.fake_openai.calls == 0


@pytest.mark.asyncio
async def test_omnivoice_custom_voice_does_not_fallback():
    service = _service_with_failing_omnivoice_and_successful_openai()
    request = OpenAISpeechRequest(
        model="tts-1",
        input="hello",
        voice="custom:voice-1",
        voice_reference=base64.b64encode(_make_reference_wav(3.5)).decode("ascii"),
        response_format="wav",
        stream=False,
        extra_params={"reference_text": "reference transcript"},
    )

    with pytest.raises(TTSProviderError):
        await _collect(request, service)

    assert service.fake_openai.calls == 0


@pytest.mark.asyncio
async def test_omnivoice_generation_parameter_does_not_fallback():
    service = _service_with_failing_omnivoice_and_successful_openai()
    request = OpenAISpeechRequest(
        model="tts-1",
        input="hello",
        voice="auto",
        response_format="wav",
        stream=False,
        extra_params={"num_step": 8},
    )

    with pytest.raises(TTSProviderError):
        await _collect(request, service)

    assert service.fake_openai.calls == 0


@pytest.mark.asyncio
async def test_implicit_omnivoice_priority_without_semantics_can_fallback():
    service = _service_with_failing_omnivoice_and_successful_openai()
    request = OpenAISpeechRequest(
        model="tts-1",
        input="hello",
        voice="auto",
        response_format="wav",
        stream=False,
    )

    audio = await _collect(request, service)

    assert audio == b"fallback-audio"
    assert service.fake_openai.calls == 1


@pytest.mark.asyncio
async def test_explicit_non_omnivoice_custom_voice_keeps_fallback_enabled():
    service = _service_with_failing_openai_and_successful_fallback()
    request = OpenAISpeechRequest(
        model="tts-1",
        input="hello",
        voice="custom:voice-1",
        voice_reference=base64.b64encode(_make_reference_wav(3.5)).decode("ascii"),
        response_format="wav",
        stream=False,
        extra_params={"reference_text": "reference transcript"},
    )

    audio = b"".join(
        [
            chunk
            async for chunk in service.generate_speech(
                request,
                provider="openai",
                fallback=True,
            )
        ]
    )

    assert audio == b"fallback-audio"
    assert service.fake_openai.calls == 1
    assert service.fake_fallback.calls == 1
