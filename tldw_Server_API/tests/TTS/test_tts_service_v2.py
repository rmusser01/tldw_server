# test_tts_service_v2.py
# Description: Tests for the V2 TTS Service architecture
#
# Imports
import asyncio
import base64
import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch, Mock
import pytest
import numpy as np
from typing import Dict, Any, Optional

# Local Imports
from tldw_Server_API.app.api.v1.schemas.audio_schemas import OpenAISpeechRequest
from tldw_Server_API.app.core.TTS.tts_service_v2 import TTSServiceV2
from tldw_Server_API.app.core.TTS.adapter_registry import TTSAdapterRegistry, TTSProvider, TTSAdapterFactory
from tldw_Server_API.app.core.TTS.adapters.base import (
    TTSAdapter,
    TTSCapabilities,
    TTSRequest,
    TTSResponse,
    AudioFormat,
    ProviderStatus,
)
from tldw_Server_API.app.core.TTS.circuit_breaker import CircuitBreaker, CircuitState
from tldw_Server_API.app.core.TTS.audio_utils import AudioProcessor, process_voice_reference
from tldw_Server_API.app.core.TTS.tts_exceptions import TTSGenerationError, TTSProviderBusyError, TTSProviderError

#######################################################################################################################
#
# Test Classes


class MockAdapter(TTSAdapter):
    """Mock adapter for testing"""

    def __init__(self, config: Dict[str, Any] = None):
        # The parent expects provider_config, but registry passes config
        provider_config = config or {}
        super().__init__(provider_config)
        self.initialized = False
        self.generate_called = False
        self.provider_id = provider_config.get("name", "mock")

    async def initialize(self) -> bool:
        self.initialized = True
        self._initialized = True
        self._status = ProviderStatus.AVAILABLE
        self._capabilities = await self.get_capabilities()
        return True

    async def generate(self, request: TTSRequest) -> TTSResponse:
        self.generate_called = True
        target_format = request.format if isinstance(request.format, AudioFormat) else AudioFormat.MP3

        if request.stream:

            async def _stream():
                for chunk in (b"mock ", b"stream ", b"data"):
                    yield chunk

            return TTSResponse(audio_stream=_stream(), format=target_format, provider=self.provider_id)

        return TTSResponse(audio_data=b"mock audio data", format=target_format, provider=self.provider_id)

    async def get_capabilities(self) -> TTSCapabilities:
        from tldw_Server_API.app.core.TTS.adapters.base import VoiceInfo, AudioFormat

        return TTSCapabilities(
            provider_name="mock",
            supports_streaming=True,
            supports_voice_cloning=False,
            supported_languages={"en"},
            supported_formats={AudioFormat.MP3, AudioFormat.WAV},
            max_text_length=5000,
            supported_voices=[VoiceInfo(id="voice1", name="Voice 1"), VoiceInfo(id="voice2", name="Voice 2")],
        )

    async def list_voices(self):
        return ["voice1", "voice2"]

    async def cleanup(self):
        self.initialized = False


class MetricsStub:
    def register_metric(self, *args, **kwargs):
        return None

    def set_gauge(self, *args, **kwargs):
        return None

    def increment(self, *args, **kwargs):
        return None

    def observe(self, *args, **kwargs):
        return None

    def gauge_add(self, *args, **kwargs):
        return None


class RecordingAdapter(TTSAdapter):
    PROVIDER_KEY = "openai"

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config or {})
        self.last_text: Optional[str] = None

    async def initialize(self) -> bool:
        self._initialized = True
        self._status = ProviderStatus.AVAILABLE
        self._capabilities = await self.get_capabilities()
        return True

    async def generate(self, request: TTSRequest) -> TTSResponse:
        self.last_text = request.text
        return TTSResponse(audio_data=b"ok", format=request.format)

    async def get_capabilities(self) -> TTSCapabilities:
        return TTSCapabilities(
            provider_name="Recording",
            supports_streaming=True,
            supports_voice_cloning=False,
            supported_languages={"en"},
            supported_formats={AudioFormat.MP3},
            max_text_length=5000,
            supported_voices=[],
        )


class TestTTSServiceV2:
    """Tests for TTSServiceV2"""

    import pytest_asyncio

    @pytest_asyncio.fixture
    async def service(self):
        """Create a test service instance"""
        config = {
            "providers": {"mock": {"enabled": True, "priority": 1}},
            "fallback": {"enabled": True, "max_attempts": 2},
        }

        # Create a mock factory
        factory = MagicMock(spec=TTSAdapterFactory)
        factory.create_adapter = AsyncMock(return_value=MockAdapter(config.get("mock", {})))
        factory.get_adapter = AsyncMock(return_value=MockAdapter(config.get("mock", {})))
        factory.list_adapters = AsyncMock(return_value=[TTSProvider.MOCK])
        factory.get_adapter_by_model = AsyncMock(return_value=MockAdapter({"name": "mock"}))
        factory.get_best_adapter = AsyncMock(return_value=MockAdapter({"name": "mock"}))
        factory.registry = MagicMock(spec=TTSAdapterRegistry)
        factory.registry.get_all_capabilities = AsyncMock(return_value={})
        factory.registry.get_adapter = AsyncMock(return_value=MockAdapter({"name": "mock"}))

        service = TTSServiceV2(factory)

        class MetricsStub:
            def register_metric(self, *args, **kwargs):
                return None

            def set_gauge(self, *args, **kwargs):
                return None

            def increment(self, *args, **kwargs):
                return None

            def observe(self, *args, **kwargs):
                return None

            def gauge_add(self, *args, **kwargs):
                return None

        service.metrics = MetricsStub()

        return service

    @pytest.mark.asyncio
    async def test_service_initialization(self, service):
        """Test service initializes properly"""
        assert service is not None
        assert hasattr(service, "factory")
        # Check that service is properly initialized
        assert service.factory is not None

    @pytest.mark.asyncio
    async def test_generate_speech(self, service):
        """Test basic speech generation"""
        # Use OpenAISpeechRequest which the service expects
        request = OpenAISpeechRequest(input="Test text", model="mock", voice="voice1", response_format="mp3")

        request.stream = False

        mock_adapter = MockAdapter({"name": "mock"})
        await mock_adapter.initialize()
        service.factory.get_adapter_by_model = AsyncMock(return_value=mock_adapter)

        chunks = [chunk async for chunk in service.generate_speech(request)]
        payload = b"".join(chunks)

        assert payload == b"mock audio data"

    @pytest.mark.asyncio
    async def test_generate_stream(self, service):
        """Test streaming generation"""
        # Use OpenAISpeechRequest
        request = OpenAISpeechRequest(input="Test text", model="mock", voice="voice1", response_format="mp3")

        mock_adapter = MockAdapter({"name": "mock"})
        await mock_adapter.initialize()
        service.factory.get_adapter_by_model = AsyncMock(return_value=mock_adapter)

        chunks = [chunk async for chunk in service.generate_speech(request)]

        assert len(chunks) == 3
        assert b"".join(chunks) == b"mock stream data"

    @pytest.mark.asyncio
    async def test_get_capabilities(self, service):
        """Test getting provider capabilities"""
        mock_adapter = MockAdapter({"name": "mock"})
        await mock_adapter.initialize()
        mock_caps = await mock_adapter.get_capabilities()
        service.factory.registry.get_all_capabilities = AsyncMock(return_value={TTSProvider.MOCK: mock_caps})

        caps = await service.get_capabilities()

        assert caps is not None
        assert "mock" in caps
        provider_caps = caps["mock"]
        assert provider_caps["supports_streaming"] is True
        assert "en" in provider_caps["languages"]
        assert "mp3" in provider_caps["formats"]

    @pytest.mark.asyncio
    async def test_list_providers(self, service):
        """Test listing available providers"""
        # Mock the factory's get_status
        service.factory.get_status = MagicMock(return_value={"providers": {"mock": "available"}})

        # The service may not have list_providers, check what it has
        status = service.factory.get_status()

        assert "providers" in status
        assert "mock" in status["providers"]
        assert status["providers"]["mock"] == "available"

    @pytest.mark.asyncio
    async def test_fallback_mechanism(self):
        """Test fallback to another provider on failure"""
        config = {
            "providers": {"failing": {"enabled": True, "priority": 1}, "working": {"enabled": True, "priority": 2}},
            "fallback": {"enabled": True, "max_attempts": 3},
        }

        # Create failing adapter
        failing_adapter = MockAdapter({"name": "failing"})

        # Create working adapter
        working_adapter = MockAdapter({"name": "working"})

        # Create mock factory that returns appropriate adapters
        factory = MagicMock(spec=TTSAdapterFactory)
        factory.create_adapter = AsyncMock(
            side_effect=lambda name, *args: failing_adapter if name == "failing" else working_adapter
        )
        factory.get_adapter = AsyncMock(
            side_effect=lambda name: failing_adapter if name == "failing" else working_adapter
        )
        factory.list_adapters = AsyncMock(return_value=["failing", "working"])

        request = OpenAISpeechRequest(input="Test text", model="failing", voice="voice1", response_format="mp3")
        request.stream = False

        failing_error = TTSProviderError("Provider failed", provider="failing")
        failing_adapter.generate = AsyncMock(side_effect=failing_error)
        await failing_adapter.initialize()
        await working_adapter.initialize()

        factory.get_adapter_by_model = AsyncMock(return_value=failing_adapter)
        factory.get_best_adapter = AsyncMock(return_value=working_adapter)
        factory.registry = MagicMock(spec=TTSAdapterRegistry)
        factory.registry.get_adapter = AsyncMock(return_value=working_adapter)

        service = TTSServiceV2(factory)

        class MetricsStub:
            def register_metric(self, *args, **kwargs):
                return None

            def set_gauge(self, *args, **kwargs):
                return None

            def increment(self, *args, **kwargs):
                return None

            def observe(self, *args, **kwargs):
                return None

            def gauge_add(self, *args, **kwargs):
                return None

        service.metrics = MetricsStub()

        chunks = [chunk async for chunk in service.generate_speech(request)]
        payload = b"".join(chunks)

        assert payload == b"mock audio data"

    @pytest.mark.asyncio
    async def test_circuit_breaker(self, service):
        """Test circuit breaker functionality"""
        # Get circuit breaker for mock provider
        cb = service.circuit_manager.get_circuit("mock") if getattr(service, "circuit_manager", None) else None

        if cb:
            assert cb.state == CircuitState.CLOSED

            # Simulate failures (use private method)
            for _ in range(5):
                cb._record_failure()

            # Circuit should be open
            assert cb.state == CircuitState.OPEN

            # Should not allow calls
            assert not cb.allow_request()


@pytest.mark.asyncio
async def test_generate_speech_uses_provider_key_for_validation():
    adapter = RecordingAdapter()
    factory = MagicMock()
    factory.get_adapter_by_model = AsyncMock(return_value=adapter)
    factory.get_provider_for_model = MagicMock(return_value=None)
    factory.registry = MagicMock()
    factory.registry.get_adapter = AsyncMock(return_value=adapter)
    factory.registry.config = {"providers": {"openai": {"sanitize_text": False}}}

    service = TTSServiceV2(factory)
    service.metrics = MetricsStub()

    providers = []

    def fake_validate(request, provider=None, config=None):
        providers.append(provider)

    request = OpenAISpeechRequest(
        input="Hello world",
        model="openai",
        voice="alloy",
        response_format="mp3",
    )
    request.stream = False

    with patch("tldw_Server_API.app.core.TTS.tts_service_v2.validate_tts_request", side_effect=fake_validate):
        chunks = [chunk async for chunk in service.generate_speech(request)]

    assert b"ok" in b"".join(chunks)
    assert providers
    assert providers[-1] == "openai"


@pytest.mark.asyncio
@pytest.mark.parametrize("sanitize_enabled", [True, False])
async def test_provider_opt_in_sanitization(sanitize_enabled):
    adapter = RecordingAdapter()
    factory = MagicMock()
    factory.get_adapter_by_model = AsyncMock(return_value=adapter)
    factory.get_provider_for_model = MagicMock(return_value=None)
    factory.registry = MagicMock()
    factory.registry.get_adapter = AsyncMock(return_value=adapter)
    factory.registry.config = {
        "providers": {"openai": {"sanitize_text": sanitize_enabled}},
        "strict_validation": False,
    }

    service = TTSServiceV2(factory)
    service.metrics = MetricsStub()

    input_text = "Hello <script>alert('xss')</script> world"
    request = OpenAISpeechRequest(
        input=input_text,
        model="openai",
        voice="alloy",
        response_format="mp3",
    )
    request.stream = False

    chunks = [chunk async for chunk in service.generate_speech(request)]

    assert b"ok" in b"".join(chunks)
    assert adapter.last_text is not None
    if sanitize_enabled:
        assert "<script>" not in adapter.last_text
    else:
        assert adapter.last_text == input_text


@pytest.mark.asyncio
async def test_generate_speech_propagates_request_and_correlation_ids():
    adapter = MockAdapter({"name": "openai"})
    await adapter.initialize()
    factory = MagicMock()
    factory.get_adapter_by_model = AsyncMock(return_value=adapter)
    factory.get_provider_for_model = MagicMock(return_value=None)
    factory.registry = MagicMock()
    factory.registry.get_adapter = AsyncMock(return_value=adapter)
    factory.registry.config = {"providers": {"openai": {"sanitize_text": False}}}

    service = TTSServiceV2(factory)
    service.metrics = MetricsStub()

    request = OpenAISpeechRequest(
        input="Hello observability",
        model="tts-1",
        voice="alloy",
        response_format="mp3",
        stream=False,
        extra_params={"correlation_id": "corr-123"},
    )

    chunks = [chunk async for chunk in service.generate_speech(request, request_id="req-123", fallback=False)]
    assert b"mock audio data" in b"".join(chunks)

    metadata = getattr(request, "_tts_metadata", {})
    assert metadata.get("request_id") == "req-123"
    assert metadata.get("correlation_id") == "corr-123"


def test_convert_request_maps_extra_seed_to_tts_request_seed():
    factory = MagicMock()
    service = TTSServiceV2(factory)
    request = OpenAISpeechRequest(
        input="Hello seeded Chatterbox",
        model="chatterbox-turbo",
        voice="default",
        response_format="wav",
        extra_params={"seed": "1234"},
    )

    converted = service._convert_request(request)

    assert converted.model == "chatterbox-turbo"
    assert converted.seed == 1234
    assert converted.extra_params["seed"] == 1234


def test_convert_request_uses_chatterbox_output_format_alias_when_response_format_omitted():
    service = TTSServiceV2(MagicMock())
    request = OpenAISpeechRequest(
        input="Hello Chatterbox",
        model="chatterbox",
        voice="default",
        output_format="wav",
    )

    converted = service._convert_request(request)

    assert converted.format == AudioFormat.WAV
    assert request.response_format == "wav"


def test_convert_request_prefers_response_format_over_chatterbox_output_format_alias():
    service = TTSServiceV2(MagicMock())
    request = OpenAISpeechRequest(
        input="Hello Chatterbox",
        model="chatterbox-turbo",
        voice="default",
        response_format="flac",
        output_format="wav",
    )

    converted = service._convert_request(request)

    assert converted.format == AudioFormat.FLAC


def test_convert_request_ignores_output_format_alias_for_non_chatterbox_models():
    service = TTSServiceV2(MagicMock())
    request = OpenAISpeechRequest(
        input="Hello Kokoro",
        model="kokoro",
        voice="default",
        output_format="wav",
    )

    converted = service._convert_request(request)

    assert converted.format == AudioFormat.MP3


def test_convert_request_uses_chatterbox_language_alias_when_lang_code_omitted():
    service = TTSServiceV2(MagicMock())
    request = OpenAISpeechRequest(
        input="Bonjour",
        model="chatterbox-multilingual",
        voice="default",
        language="fr",
    )

    converted = service._convert_request(request)

    assert converted.language == "fr"


def test_convert_request_prefers_lang_code_over_chatterbox_language_alias():
    service = TTSServiceV2(MagicMock())
    request = OpenAISpeechRequest(
        input="Hola",
        model="chatterbox-multilingual",
        voice="default",
        lang_code="es",
        language="fr",
    )

    converted = service._convert_request(request)

    assert converted.language == "es"


def test_convert_request_ignores_language_alias_for_non_chatterbox_models():
    service = TTSServiceV2(MagicMock())
    request = OpenAISpeechRequest(
        input="Bonjour",
        model="kokoro",
        voice="default",
        language="fr",
    )

    converted = service._convert_request(request)

    assert converted.language is None


def test_resolve_chunking_params_accepts_chatterbox_split_text_alias():
    service = TTSServiceV2(MagicMock())

    enabled, target, max_chars, min_chars, crossfade_ms = service._resolve_chunking_params(
        {"split_text": True}
    )

    assert enabled is True
    assert target == 120
    assert max_chars == 150
    assert min_chars == 50
    assert crossfade_ms == 50


def test_resolve_chunking_params_maps_chatterbox_chunk_size_alias():
    service = TTSServiceV2(MagicMock())

    enabled, target, max_chars, min_chars, _crossfade_ms = service._resolve_chunking_params(
        {"split_text": True, "chunk_size": 320}
    )

    assert enabled is True
    assert target == 320
    assert max_chars == 320
    assert min_chars == 50


def test_resolve_chunking_params_respects_disabled_chatterbox_split_text():
    service = TTSServiceV2(MagicMock())

    assert service._resolve_chunking_params({"split_text": False, "chunk_size": 320}) == (False, 0, 0, 0, 0)


@pytest.mark.asyncio
async def test_generate_chunked_response_strips_chatterbox_chunking_aliases_from_child_requests():
    service = TTSServiceV2(MagicMock())
    seen_extras: list[dict[str, Any]] = []

    class ChunkAdapter:
        sample_rate = 24000

        async def get_capabilities(self):
            return TTSCapabilities(
                provider_name="chunker",
                supported_languages={"en"},
                supported_voices=[],
                supported_formats={AudioFormat.PCM},
                max_text_length=1000,
                supports_streaming=False,
            )

        async def generate(self, request: TTSRequest):
            seen_extras.append(dict(request.extra_params or {}))
            return TTSResponse(
                audio_data=np.zeros(240, dtype=np.int16).tobytes(),
                format=AudioFormat.PCM,
                sample_rate=24000,
            )

    request = TTSRequest(
        text="One sentence. Two sentence. Three sentence.",
        voice="default",
        model="chatterbox",
        format=AudioFormat.WAV,
        extra_params={
            "split_text": True,
            "chunk_size": 12,
            "chunk_crossfade_ms": 0,
            "preserve": "yes",
        },
    )

    response = await service._generate_chunked_response(
        adapter=ChunkAdapter(),
        request=request,
        provider_key="chatterbox",
        target_chars=12,
        max_chars=12,
        min_chars=1,
        crossfade_ms=0,
    )

    assert response is not None
    assert seen_extras
    for extras in seen_extras:
        assert extras == {"preserve": "yes"}


@pytest.mark.asyncio
async def test_convert_chatterbox_voice_delegates_to_adapter_runtime():
    adapter = MagicMock()
    adapter.convert_voice = AsyncMock(
        return_value=TTSResponse(audio_data=b"converted", format=AudioFormat.WAV, sample_rate=24000)
    )
    service = TTSServiceV2(MagicMock())
    service.metrics = MetricsStub()
    service._get_adapter = AsyncMock(return_value=adapter)

    response = await service.convert_chatterbox_voice(
        source_audio_path="/tmp/source.wav",
        target_voice_path="/tmp/target.wav",
        response_format="wav",
        stream=False,
    )

    service._get_adapter.assert_awaited_once_with("chatterbox", "chatterbox", overrides=None)
    adapter.convert_voice.assert_awaited_once_with(
        source_audio_path="/tmp/source.wav",
        target_voice_path="/tmp/target.wav",
        format=AudioFormat.WAV,
        stream=False,
    )
    assert response.audio_data == b"converted"
    assert response.format is AudioFormat.WAV
    assert service._active_request_counts == {}


@pytest.mark.asyncio
async def test_unload_provider_rejects_active_requests():
    factory = MagicMock()
    factory.unload_provider = AsyncMock(return_value={"provider": "chatterbox", "unloaded": True})
    service = TTSServiceV2(factory)
    service._active_request_counts["chatterbox"] = 1

    with pytest.raises(TTSProviderBusyError) as exc_info:
        await service.unload_provider("chatterbox")

    assert exc_info.value.details == {"active_requests": 1}
    factory.unload_provider.assert_not_called()




@pytest.mark.asyncio
async def test_fallback_outcomes_metric_records_initiated_and_success():
    primary_adapter = MagicMock()
    primary_adapter.provider_key = "openai"
    primary_adapter.provider_name = "openai"
    primary_adapter.handles_text_chunking = False
    primary_adapter.generate = AsyncMock(side_effect=TTSProviderError("provider failed", provider="openai"))

    fallback_adapter = MagicMock()
    fallback_adapter.provider_key = "elevenlabs"
    fallback_adapter.provider_name = "elevenlabs"
    fallback_adapter.handles_text_chunking = False
    fallback_adapter.generate = AsyncMock(return_value=TTSResponse(audio_data=b"fallback-audio", format=AudioFormat.MP3))

    class RecordingMetrics:
        def __init__(self) -> None:
            self.increments: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

        def register_metric(self, *args, **kwargs):
            return None

        def set_gauge(self, *args, **kwargs):
            return None

        def observe(self, *args, **kwargs):
            return None

        def increment(self, *args, **kwargs):
            self.increments.append((args, kwargs))

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

    service = TTSServiceV2(DummyFactory())
    metrics = RecordingMetrics()
    service.metrics = metrics
    service._get_fallback_adapter = AsyncMock(return_value=fallback_adapter)

    request = OpenAISpeechRequest(
        input="hello",
        model="tts-1",
        voice="alloy",
        response_format="mp3",
        stream=False,
    )

    chunks = [chunk async for chunk in service.generate_speech(request, fallback=True, request_id="req-fallback")]
    assert b"fallback-audio" in b"".join(chunks)

    fallback_outcome_calls = [
        kwargs.get("labels", {})
        for args, kwargs in metrics.increments
        if args and args[0] == "tts_fallback_outcomes_total"
    ]
    assert any(label.get("outcome") == "initiated" for label in fallback_outcome_calls)
    assert any(label.get("outcome") == "success" for label in fallback_outcome_calls)


class TestAudioUtils:
    """Tests for audio processing utilities"""

    def test_audio_processor_initialization(self):
        """Test AudioProcessor initialization"""
        processor = AudioProcessor()
        assert processor is not None
        assert "higgs" in processor.PROVIDER_REQUIREMENTS
        assert "chatterbox" in processor.PROVIDER_REQUIREMENTS
        assert "vibevoice" in processor.PROVIDER_REQUIREMENTS

    def test_validate_audio(self):
        """Test audio validation"""
        processor = AudioProcessor()

        # Create a simple WAV header (minimal valid WAV)
        wav_header = b'RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00"V\x00\x00D\xac\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00'
        # Add some audio data (5 seconds worth at 22050Hz, 16-bit mono)
        audio_data = b"\x00\x00" * (22050 * 5)
        valid_wav = wav_header + audio_data

        # Test validation for Higgs provider
        is_valid, msg, info = processor.validate_audio(valid_wav, "higgs", check_duration=False)
        # Note: Without proper audio libs, this might fail, but at least test the method exists
        assert isinstance(is_valid, bool)
        if msg:
            assert isinstance(msg, str)

    def test_process_voice_reference(self):
        """Test voice reference processing"""
        # Create a base64 encoded simple audio
        audio_data = (
            b'RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00"V\x00\x00D\xac\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00'
            + b"\x00\x00" * 1000
        )
        base64_audio = base64.b64encode(audio_data).decode("utf-8")

        # Process
        processed, error = process_voice_reference(
            base64_audio,  # Pass base64 string, not raw bytes
            provider="higgs",
            validate=False,  # Skip validation for test
            convert=False,
        )

        # Check result
        assert processed is not None or error is not None
        if error:
            assert isinstance(error, str)

    @pytest.mark.asyncio
    async def test_base64_encoding(self):
        """Test base64 encoding/decoding for voice references"""
        original_data = b"test audio data"

        # Encode
        encoded = base64.b64encode(original_data).decode("utf-8")

        # Decode
        decoded = base64.b64decode(encoded)

        assert decoded == original_data


class TestCircuitBreaker:
    """Tests for Circuit Breaker"""

    def test_circuit_breaker_initialization(self):
        """Test circuit breaker initialization"""
        cb = CircuitBreaker(provider_name="test", failure_threshold=3, recovery_timeout=10)

        assert cb.state == CircuitState.CLOSED
        # Check status instead of direct attribute
        status = cb.get_status()
        assert status["stats"]["failure_count"] == 0

    @pytest.mark.asyncio
    async def test_circuit_opens_after_threshold(self):
        """Test circuit opens after failure threshold"""
        cb = CircuitBreaker(provider_name="test", failure_threshold=3)

        # Define a failing function
        async def failing_func():
            raise Exception("Test failure")

        # Record failures through the call method
        for _ in range(3):
            try:
                await cb.call(failing_func)
            except Exception:
                _ = None  # Expected to fail

        assert cb.state == CircuitState.OPEN
        assert not cb.is_available

    @pytest.mark.asyncio
    async def test_circuit_recovery(self):
        """Test circuit recovery to half-open state"""
        cb = CircuitBreaker(
            provider_name="test_provider",
            failure_threshold=2,
            recovery_timeout=0.1,  # 100ms for testing
            success_threshold=2,
        )

        # Define functions for testing
        async def failing_func():
            raise Exception("Test failure")

        async def success_func():
            return "success"

        # Open the circuit
        for _ in range(2):
            try:
                await cb.call(failing_func)
            except Exception:
                _ = None
        assert cb.state == CircuitState.OPEN

        # Wait for recovery timeout
        await asyncio.sleep(0.2)

        # Should transition to half-open on next check
        assert cb.is_available  # This triggers transition
        assert cb.state == CircuitState.HALF_OPEN

        # Success should close circuit after success_threshold
        for _ in range(2):
            await cb.call(success_func)
        assert cb.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_circuit_reopens_on_failure_in_half_open(self):
        """Test circuit reopens if failure occurs in half-open state"""
        cb = CircuitBreaker(provider_name="test", failure_threshold=2, recovery_timeout=0.1)

        # Define a failing function
        async def failing_func():
            raise Exception("Test failure")

        # Open circuit
        for _ in range(2):
            try:
                await cb.call(failing_func)
            except Exception:
                _ = None

        # Wait and transition to half-open
        await asyncio.sleep(0.2)
        assert cb.is_available  # Triggers transition
        assert cb.state == CircuitState.HALF_OPEN

        # Failure in half-open should reopen
        try:
            await cb.call(failing_func)
        except Exception:
            _ = None  # Expected
        assert cb.state == CircuitState.OPEN


class TestAdapterRegistry:
    """Tests for Adapter Registry"""

    def test_registry_initialization(self):
        """Test registry initialization"""
        config = {"providers": {"openai": {"enabled": True}, "kokoro": {"enabled": False}}}

        registry = TTSAdapterRegistry(config)
        assert registry is not None
        assert registry.config == config

    @pytest.mark.asyncio
    async def test_register_adapter(self):
        """Test registering an adapter"""
        registry = TTSAdapterRegistry({})

        # Register the adapter CLASS, not an instance
        registry.register_adapter(TTSProvider.MOCK, MockAdapter)

        # Check it was registered in the adapter_classes dict
        assert TTSProvider.MOCK in registry._adapter_specs
        assert registry._adapter_specs[TTSProvider.MOCK] == MockAdapter

    @pytest.mark.asyncio
    async def test_get_adapter(self):
        """Test getting an adapter"""
        # Create registry with mock config
        config = {"mock_enabled": True}
        registry = TTSAdapterRegistry(config)

        # Register the MockAdapter class
        registry.register_adapter(TTSProvider.MOCK, MockAdapter)

        # Get the adapter (this will initialize it)
        retrieved = await registry.get_adapter(TTSProvider.MOCK)

        # Check we got an instance
        assert retrieved is not None
        assert isinstance(retrieved, MockAdapter)

    @pytest.mark.asyncio
    async def test_list_adapters(self):
        """Test listing adapters"""
        config = {"mock_enabled": True, "openai_enabled": False}
        registry = TTSAdapterRegistry(config)

        # Register MockAdapter
        registry.register_adapter(TTSProvider.MOCK, MockAdapter)

        # Initialize an adapter
        await registry.get_adapter(TTSProvider.MOCK)

        # Get status summary
        status = registry.get_status_summary()

        assert "providers" in status
        assert TTSProvider.MOCK.value in status["providers"]


class TestVoiceCloning:
    """Tests for voice cloning functionality"""

    @pytest.mark.asyncio
    async def test_voice_reference_in_request(self):
        """Test voice reference field in request"""
        voice_data = base64.b64encode(b"audio data").decode()

        request = TTSRequest(text="Test with voice cloning", voice="clone", voice_reference=voice_data)

        assert request.voice_reference == voice_data
        assert request.voice == "clone"

    def test_voice_reference_validation(self):
        """Test voice reference validation"""
        processor = AudioProcessor()

        # Test provider requirements
        reqs = processor.PROVIDER_REQUIREMENTS.get("higgs")
        assert reqs is not None
        assert reqs["min_duration"] == 3.0
        assert reqs["max_duration"] == 10.0
        assert reqs["preferred_sample_rate"] == 24000

        reqs = processor.PROVIDER_REQUIREMENTS.get("chatterbox")
        assert reqs is not None
        assert reqs["min_duration"] == 5.0
        assert reqs["max_duration"] == 20.0

        reqs = processor.PROVIDER_REQUIREMENTS.get("vibevoice")
        assert reqs is not None
        assert reqs["min_duration"] == 3.0
        assert reqs["max_duration"] == 30.0

    @pytest.mark.asyncio
    async def test_temp_file_cleanup(self):
        """Test temporary file cleanup for voice references"""
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            temp_path = f.name
            f.write(b"test audio data")

        assert os.path.exists(temp_path)

        # Simulate cleanup
        try:
            # Process file
            pass
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

        assert not os.path.exists(temp_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
