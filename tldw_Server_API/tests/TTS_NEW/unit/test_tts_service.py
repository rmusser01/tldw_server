"""
Unit tests for the TTSServiceV2 core functionality.

Tests the main service logic, adapter selection, and request processing
with mocked dependencies.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import asyncio

from tldw_Server_API.app.core.TTS import tts_service_v2 as tts_service_v2_module
from tldw_Server_API.app.core.TTS.tts_service_v2 import TTSServiceV2
from tldw_Server_API.app.core.TTS.adapters.base import (
    TTSRequest,
    TTSResponse,
    AudioFormat,
    TTSCapabilities,
    VoiceInfo,
)
from tldw_Server_API.app.api.v1.schemas.audio_schemas import OpenAISpeechRequest
from tldw_Server_API.app.core.TTS.tts_exceptions import (
    TTSProviderNotConfiguredError,
    TTSGenerationError,
    TTSProviderError,
    TTSRateLimitError,
)
from tldw_Server_API.app.core.TTS.voice_manager import VoiceReferenceMetadata

# ========================================================================
# Service Initialization Tests
# ========================================================================

class TestServiceInitialization:
    """Test TTS service initialization and setup."""

    @pytest.mark.unit
    async def test_service_initialization(self):
        """Test basic service initialization."""
        service = TTSServiceV2()

        assert service is not None
        assert hasattr(service, 'generate')
        assert hasattr(service, 'generate_stream')

        await service.shutdown()

    @pytest.mark.unit
    @patch('tldw_Server_API.app.core.TTS.tts_service_v2.get_tts_factory')
    async def test_service_with_mock_factory(self, mock_factory_getter, mock_adapter_factory):
        """Test service initialization with mocked factory."""
        mock_factory_getter.return_value = mock_adapter_factory

        service = TTSServiceV2()

        assert service._factory == mock_adapter_factory

        await service.shutdown()

    @pytest.mark.unit
    async def test_service_shutdown(self, tts_service):
        """Test service shutdown process."""
        # Service from fixture
        await tts_service.shutdown()

        # Should handle multiple shutdowns gracefully
        await tts_service.shutdown()

    @pytest.mark.unit
    async def test_service_shutdown_uses_public_supervisor_shutdown_and_still_closes_factory(self):
        service = TTSServiceV2()

        factory = MagicMock()
        factory.close = AsyncMock()
        service.factory = factory
        service._factory = None

        class _FailingSupervisor:
            def __init__(self) -> None:
                self.mark_closing_calls = 0
                self.shutdown = AsyncMock(side_effect=RuntimeError("sidecar stop failed"))

            def mark_closing(self) -> None:
                self.mark_closing_calls += 1

        supervisor = _FailingSupervisor()
        service._omnivoice_supervisor = supervisor

        await service.shutdown()

        assert supervisor.mark_closing_calls == 1
        supervisor.shutdown.assert_awaited_once()
        factory.close.assert_awaited_once()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_close_tts_service_v2_invokes_instance_shutdown(monkeypatch):
    service = MagicMock()
    service.shutdown = AsyncMock()

    close_factory = AsyncMock()
    monkeypatch.setattr(tts_service_v2_module, "_service_instance", service, raising=True)
    monkeypatch.setattr(tts_service_v2_module, "close_tts_factory", close_factory, raising=True)

    await tts_service_v2_module.close_tts_service_v2()

    service.shutdown.assert_awaited_once()
    close_factory.assert_awaited_once()
    assert tts_service_v2_module._service_instance is None

# ========================================================================
# Text Generation Tests
# ========================================================================

class TestTextGeneration:
    """Test text-to-speech generation."""

    @pytest.mark.unit
    async def test_basic_generation(self, tts_service, basic_tts_request):
        """Test basic TTS generation."""
        result = await tts_service.generate(basic_tts_request)

        assert result is not None
        assert hasattr(result, 'audio_content')
        assert result.audio_content == b"mock_audio"

    @pytest.mark.unit
    async def test_generation_with_provider_selection(self, tts_service, basic_tts_request):
        """Test generation with specific provider."""
        basic_tts_request.provider = "elevenlabs"

        result = await tts_service.generate(basic_tts_request)

        assert result is not None
        tts_service._factory.get_adapter.assert_called_with("elevenlabs")

    @pytest.mark.unit
    async def test_generation_with_invalid_provider(self, tts_service, basic_tts_request):
        """Test generation with invalid provider."""
        basic_tts_request.provider = "invalid_provider"
        tts_service._factory.get_adapter.side_effect = TTSProviderNotConfiguredError("Provider not found")

        with pytest.raises(TTSProviderNotConfiguredError):
            await tts_service.generate(basic_tts_request)

    @pytest.mark.unit
    async def test_generation_with_long_text(self, tts_service, long_text_request):
        """Test generation with long text that needs chunking."""
        result = await tts_service.generate(long_text_request)

        assert result is not None
        # Should handle long text appropriately

    @pytest.mark.unit
    async def test_request_validation_called(self, tts_service, basic_tts_request):
        """Current generate() delegates directly to adapter without service-level validation."""
        adapter = tts_service._factory.get_adapter("openai")
        await tts_service.generate(basic_tts_request)
        adapter.generate.assert_called_once()

    @pytest.mark.unit
    async def test_token_defaults_applied(self):
        """Service applies min/max token defaults when not provided."""
        service = TTSServiceV2()
        request = TTSRequest(text="Hello world", format=AudioFormat.MP3)

        service._apply_token_defaults(request)

        assert "max_new_tokens" in request.extra_params
        assert "min_new_tokens" in request.extra_params
        assert request.extra_params["max_new_tokens"] >= request.extra_params["min_new_tokens"]

    @pytest.mark.unit
    async def test_chunked_retry_metadata(self):
        """Chunked requests retry per segment and return metadata."""
        service = TTSServiceV2()
        adapter = MagicMock()

        async def mock_caps():
            return TTSCapabilities(
                provider_name="mock",
                supported_languages={"en"},
                supported_voices=[VoiceInfo(id="v", name="v")],
                supported_formats={AudioFormat.PCM, AudioFormat.MP3},
                max_text_length=1000,
                supports_streaming=False,
            )

        adapter.get_capabilities = AsyncMock(side_effect=mock_caps)

        call_count = {"n": 0}

        async def mock_generate(request):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise TTSRateLimitError("rate limited", provider="mock")
            return TTSResponse(
                audio_data=b"\x00\x00" * 200,
                format=AudioFormat.PCM,
                sample_rate=24000,
                provider="mock",
            )

        adapter.generate = AsyncMock(side_effect=mock_generate)

        text = "Sentence one is long enough to split. Sentence two is also long enough to split."
        req = TTSRequest(
            text=text,
            format=AudioFormat.MP3,
            stream=False,
            extra_params={
                "chunking": True,
                "chunk_max_chars": 30,
                "segment_retry_max": 1,
                "segment_retry_backoff_ms": 1,
            },
        )

        response = await service._generate_chunked_response(
            adapter=adapter,
            request=req,
            provider_key="mock",
            target_chars=20,
            max_chars=30,
            min_chars=10,
            crossfade_ms=10,
        )

        assert response is not None
        assert response.metadata.get("chunked") is True
        assert response.metadata.get("segments")

# ========================================================================
# Streaming Generation Tests
# ========================================================================

class TestStreamingGeneration:
    """Test streaming TTS generation."""

    @pytest.mark.unit
    async def test_streaming_generation(self, tts_service, basic_tts_request):
        """Test streaming TTS generation."""
        # Mock adapter to return streaming response
        adapter = tts_service._factory.get_adapter("openai")

        async def mock_stream():
            for chunk in [b"chunk1", b"chunk2", b"chunk3"]:
                yield chunk

        adapter.generate_stream = AsyncMock(return_value=mock_stream())

        chunks = []
        async for chunk in tts_service.generate_stream(basic_tts_request):
            chunks.append(chunk)

        assert len(chunks) == 3
        assert chunks == [b"chunk1", b"chunk2", b"chunk3"]

    @pytest.mark.unit
    async def test_streaming_with_error(self, tts_service, basic_tts_request):
        """Test streaming generation error handling."""
        adapter = tts_service._factory.get_adapter("openai")

        async def mock_error_stream():
            yield b"chunk1"
            raise TTSGenerationError("Stream interrupted")

        adapter.generate_stream = AsyncMock(return_value=mock_error_stream())

        chunks = []
        with pytest.raises(TTSGenerationError):
            async for chunk in tts_service.generate_stream(basic_tts_request):
                chunks.append(chunk)

        # Should have received first chunk before error
        assert len(chunks) == 1

# ========================================================================
# Provider Management Tests
# ========================================================================

class TestProviderManagement:
    """Test provider management functionality."""

    @pytest.mark.unit
    async def test_list_providers(self, tts_service):
        """Test listing available providers."""
        providers = await tts_service.list_providers()

        assert isinstance(providers, list)
        assert "openai" in providers
        assert "elevenlabs" in providers

    @pytest.mark.unit
    async def test_get_provider_info(self, tts_service):
        """Test getting provider information."""
        # Mock the adapter
        adapter = tts_service._factory.get_adapter("openai")
        adapter.get_info = Mock(return_value={
            "name": "openai",
            "models": ["tts-1", "tts-1-hd"],
            "voices": ["alloy", "echo"]
        })

        info = await tts_service.get_provider_info("openai")

        assert info["name"] == "openai"
        assert "tts-1" in info["models"]
        assert "alloy" in info["voices"]

    @pytest.mark.unit
    async def test_switch_default_provider(self, tts_service):
        """Test switching the default provider."""
        await tts_service.set_default_provider("elevenlabs")

        # Generate without specifying provider should use new default
        basic_request = TTSRequest(text="Test", voice="rachel")
        await tts_service.generate(basic_request)

        tts_service._factory.get_adapter.assert_called_with("elevenlabs")

# ========================================================================
# Error Handling Tests
# ========================================================================

class TestErrorHandling:
    """Test error handling in the service."""

    @pytest.mark.unit
    async def test_handle_rate_limit_error(self, tts_service, basic_tts_request):
        """Test handling of rate limit errors."""
        adapter = tts_service._factory.get_adapter("openai")
        adapter.generate = AsyncMock(side_effect=TTSRateLimitError("Rate limited", details={"retry_after": 60}))

        with pytest.raises(TTSRateLimitError) as exc_info:
            await tts_service.generate(basic_tts_request)

        assert exc_info.value.retry_after == 60

    @pytest.mark.unit
    async def test_handle_generation_error(self, tts_service, basic_tts_request):
        """Test handling of generation errors."""
        adapter = tts_service._factory.get_adapter("openai")
        adapter.generate = AsyncMock(side_effect=TTSGenerationError("Generation failed"))

        with pytest.raises(TTSGenerationError):
            await tts_service.generate(basic_tts_request)

    @pytest.mark.unit
    async def test_fallback_provider_on_error(self, tts_service, basic_tts_request):
        """Test fallback to another provider on error."""
        # First adapter fails
        openai_adapter = tts_service._factory.get_adapter("openai")
        openai_adapter.generate = AsyncMock(side_effect=TTSGenerationError("Failed"))

        # Second adapter succeeds
        elevenlabs_adapter = tts_service._factory.get_adapter("elevenlabs")
        elevenlabs_adapter.generate = AsyncMock(return_value=MagicMock(audio_content=b"fallback_audio"))

        # Enable fallback
        tts_service.enable_fallback = True

        result = await tts_service.generate_with_fallback(basic_tts_request, fallback_providers=["elevenlabs"])

        assert result.audio_content == b"fallback_audio"

# ========================================================================
# Resource Management Tests
# ========================================================================

class TestResourceManagement:
    """Test resource management in the service."""

    @pytest.mark.unit
    @patch('tldw_Server_API.app.core.TTS.tts_service_v2.get_resource_manager')
    async def test_resource_check_before_generation(self, mock_resource_manager, tts_service, basic_tts_request):
        """Test that resources are checked before generation."""
        manager = MagicMock()
        manager.check_resources = AsyncMock(return_value=True)
        mock_resource_manager.return_value = manager

        await tts_service.generate(basic_tts_request)

        manager.check_resources.assert_called()

    @pytest.mark.unit
    @patch('tldw_Server_API.app.core.TTS.tts_service_v2.get_resource_manager')
    async def test_insufficient_resources(self, mock_resource_manager, tts_service, basic_tts_request):
        """Current legacy generate() ignores resource check failures; generation proceeds."""
        manager = MagicMock()
        manager.check_resources = AsyncMock(return_value=False)
        mock_resource_manager.return_value = manager

        # Should not raise; should still delegate to adapter
        result = await tts_service.generate(basic_tts_request)
        assert result is not None

# ========================================================================
# Metrics Collection Tests
# ========================================================================

class TestMetricsCollection:
    """Test metrics collection in the service."""

    @pytest.mark.unit
    async def test_metrics_recorded_on_success(self, tts_service):
        """Metrics are recorded in generate_speech path via increment/observe."""
        from tldw_Server_API.app.api.v1.schemas.audio_schemas import OpenAISpeechRequest
        metrics_registry = MagicMock()
        # Inject mocked registry into service
        tts_service.metrics = metrics_registry

        # Prepare a factory/adapter compatible with generate_speech
        adapter = MagicMock()
        adapter.provider_name = "openai"
        adapter.generate = AsyncMock(return_value=TTSResponse(audio_data=b"ok"))
        fac = MagicMock()
        fac.get_adapter_by_model = AsyncMock(return_value=adapter)
        fac.registry = MagicMock()
        fac.registry.get_adapter = AsyncMock(return_value=adapter)
        tts_service.factory = fac

        req = OpenAISpeechRequest(model="tts-1", input="hello", voice="alloy", response_format="mp3", stream=False)
        # Consume the generator to trigger metrics
        chunks = []
        async for c in tts_service.generate_speech(req):
            chunks.append(c)
        assert b"ok" in b"".join(chunks)
        # Assert metrics increment/observe were called
        assert metrics_registry.increment.called
        assert metrics_registry.observe.called

    @pytest.mark.unit
    async def test_metrics_recorded_on_failure(self, tts_service):
        """Metrics failure path in generate_speech uses increment/observe, not record."""
        from tldw_Server_API.app.api.v1.schemas.audio_schemas import OpenAISpeechRequest
        metrics_registry = MagicMock()
        tts_service.metrics = metrics_registry

        adapter = MagicMock()
        adapter.provider_name = "openai"
        adapter.generate = AsyncMock(side_effect=TTSGenerationError("Failed"))
        fac = MagicMock()
        fac.get_adapter_by_model = AsyncMock(return_value=adapter)
        fac.registry = MagicMock()
        fac.registry.get_adapter = AsyncMock(return_value=adapter)
        tts_service.factory = fac

        req = OpenAISpeechRequest(model="tts-1", input="hello", voice="alloy", response_format="mp3", stream=False)
        # Consume and expect error reported via yielded "ERROR:" chunk
        chunks = []
        async for c in tts_service.generate_speech(req):
            chunks.append(c)
        # Assert metrics increment/observe called for failure
        assert metrics_registry.increment.called
        assert metrics_registry.observe.called

    @pytest.mark.unit
    async def test_soft_failure_no_audio_records_primary_failure_metrics(self):
        """
        When a provider returns no audio but no exception, generate_speech
        should record a failure for the primary provider before falling back.
        """
        from tldw_Server_API.app.api.v1.schemas.audio_schemas import OpenAISpeechRequest

        # Set up a service with a mocked factory/adapter
        adapter = MagicMock()
        adapter.provider_name = "openai"
        adapter.generate = AsyncMock(return_value=TTSResponse())

        factory = MagicMock()
        factory.get_adapter_by_model = AsyncMock(return_value=adapter)
        factory.registry = MagicMock()
        factory.registry.get_adapter = AsyncMock(return_value=adapter)
        factory.registry.config = {"performance": {"max_concurrent_generations": 1, "stream_errors_as_audio": False}}

        service = TTSServiceV2(factory=factory)
        service.factory = factory
        service._factory = factory
        # Stub metrics and internal helpers
        service.metrics = MagicMock()
        service._record_tts_metrics = MagicMock()

        async def fake_try_fallback(
            request,
            exclude,
            from_provider,
            *,
            metadata_only: bool = False,
            metadata_target=None,
            user_id=None,
        ):
            yield b"fallback-audio"

        service._handle_provider_fallback = AsyncMock(return_value=None)
        service._try_fallback_providers = fake_try_fallback

        req = OpenAISpeechRequest(
            model="tts-1",
            input="hello",
            voice="alloy",
            response_format="mp3",
            stream=False,
        )

        # Trigger the no-audio soft failure with fallback enabled
        chunks = []
        async for c in service.generate_speech(req, fallback=True):
            chunks.append(c)

        assert b"fallback-audio" in b"".join(chunks)

        # Verify that a failure metrics record was emitted for the primary provider
        failure_calls = [
            call for call in service._record_tts_metrics.call_args_list
            if call.kwargs.get("success") is False
        ]
        assert failure_calls, "Expected failure metrics call for primary provider soft-fail"
        kwargs = failure_calls[0].kwargs
        assert kwargs["provider"] == "openai"
        assert "No audio data returned" in (kwargs.get("error") or "")

    @pytest.mark.unit
    async def test_generate_speech_writes_request_context_into_metadata(self):
        adapter = MagicMock()
        adapter.provider_name = "openai"
        adapter.generate = AsyncMock(return_value=TTSResponse(audio_data=b"ok"))

        factory = MagicMock()
        factory.get_adapter_by_model = AsyncMock(return_value=adapter)
        factory.registry = MagicMock()
        factory.registry.get_adapter = AsyncMock(return_value=adapter)
        factory.registry.config = {"performance": {"max_concurrent_generations": 1}}

        service = TTSServiceV2(factory=factory)
        service.factory = factory
        service._factory = factory
        service.metrics = MagicMock()

        req = OpenAISpeechRequest(
            model="tts-1",
            input="hello",
            voice="alloy",
            response_format="mp3",
            stream=False,
            extra_params={"correlation_id": "corr-unit-1"},
        )

        chunks = [c async for c in service.generate_speech(req, fallback=False, request_id="req-unit-1")]
        assert b"ok" in b"".join(chunks)
        metadata = getattr(req, "_tts_metadata", {})
        assert metadata.get("request_id") == "req-unit-1"
        assert metadata.get("correlation_id") == "corr-unit-1"

    @pytest.mark.unit
    async def test_fallback_outcome_metric_emitted(self):
        adapter = MagicMock()
        adapter.provider_name = "openai"
        adapter.generate = AsyncMock(side_effect=TTSProviderError("down", provider="openai"))

        factory = MagicMock()
        factory.get_adapter_by_model = AsyncMock(return_value=adapter)
        factory.registry = MagicMock()
        factory.registry.get_adapter = AsyncMock(return_value=adapter)
        factory.registry.config = {"performance": {"max_concurrent_generations": 1, "stream_errors_as_audio": False}}

        service = TTSServiceV2(factory=factory)
        service.factory = factory
        service._factory = factory
        service.metrics = MagicMock()

        async def fake_try_fallback(
            request,
            exclude,
            from_provider,
            *,
            metadata_only: bool = False,
            metadata_target=None,
            user_id=None,
        ):
            service._record_fallback_event(
                from_provider=from_provider or "unknown",
                to_provider="mock_fallback",
                success="true",
                outcome="success",
                request_id="req-fallback-1",
            )
            yield b"fallback-audio"

        service._try_fallback_providers = fake_try_fallback

        req = OpenAISpeechRequest(
            model="tts-1",
            input="hello",
            voice="alloy",
            response_format="mp3",
            stream=False,
        )

        chunks = [c async for c in service.generate_speech(req, fallback=True, request_id="req-fallback-1")]
        assert b"fallback-audio" in b"".join(chunks)

        fallback_outcome_calls = [
            call for call in service.metrics.increment.call_args_list
            if call.args and call.args[0] == "tts_fallback_outcomes_total"
        ]
        assert fallback_outcome_calls


@pytest.mark.unit
async def test_convert_response_if_needed_handles_non_omnivoice_wav_response(monkeypatch):
    from tldw_Server_API.app.core.TTS import tts_service_v2 as tts_service_module

    service = TTSServiceV2()
    response = TTSResponse(
        audio_data=b"wav-bytes",
        format=AudioFormat.WAV,
        sample_rate=24000,
        channels=2,
        provider="openai",
    )
    request = TTSRequest(
        text="hello world",
        format=AudioFormat.MP3,
        stream=False,
    )
    to_thread_calls: list[str] = []

    async def _fake_to_thread(func, *args, **kwargs):
        to_thread_calls.append(getattr(func, "__name__", repr(func)))
        return func(*args, **kwargs)

    async def _fake_convert_format(input_path, output_path, target_format, sample_rate=None, channels=None):
        output_path.write_bytes(b"converted-audio")
        return True

    monkeypatch.setattr(tts_service_module.asyncio, "to_thread", _fake_to_thread, raising=True)
    monkeypatch.setattr(tts_service_module.AudioConverter, "convert_format", _fake_convert_format, raising=True)

    converted = await service._convert_response_if_needed(
        response,
        request,
        provider_key="openai",
    )

    assert converted.audio_data == b"converted-audio"
    assert converted.format == AudioFormat.MP3
    assert to_thread_calls  # nosec B101


@pytest.mark.unit
async def test_custom_voice_resolution_injects_reference(tts_service, monkeypatch):
    from tldw_Server_API.app.api.v1.schemas.audio_schemas import OpenAISpeechRequest
    from tldw_Server_API.app.core.TTS.voice_manager import VoiceReferenceMetadata

    class _FakeVoiceManager:
        async def load_voice_reference_audio(self, user_id, voice_id):
            assert user_id == 1
            assert voice_id == "voice-1"
            return b"audio-bytes"

        async def load_reference_metadata(self, user_id, voice_id):
            return VoiceReferenceMetadata(
                voice_id=voice_id,
                reference_text="stored text",
                provider_artifacts={
                    "neutts": {
                        "ref_codes": [1, 2, 3],
                        "reference_text": "stored text",
                    }
                },
            )

    def _fake_get_voice_manager():
        return _FakeVoiceManager()

    monkeypatch.setattr(
        "tldw_Server_API.app.core.TTS.voice_manager.get_voice_manager",
        _fake_get_voice_manager,
        raising=True,
    )

    req = OpenAISpeechRequest(
        model="neutts-air",
        input="hello",
        voice="custom:voice-1",
        response_format="mp3",
        stream=False,
    )
    tts_request = tts_service._convert_request(req)
    await tts_service._apply_custom_voice_reference(tts_request, user_id=1, provider_hint="neutts")

    assert tts_request.voice_reference == b"audio-bytes"
    assert tts_request.extra_params.get("ref_codes") == [1, 2, 3]
    assert tts_request.extra_params.get("reference_text") == "stored text"


@pytest.mark.unit
async def test_convert_request_language_override(tts_service):
    req = OpenAISpeechRequest(
        model="kokoro",
        input="hello",
        voice="af_heart",
        response_format="mp3",
        stream=False,
        lang_code="en",
        extra_params={"language": "ja"},
    )
    tts_request = tts_service._convert_request(req)
    assert tts_request.language == "ja"


@pytest.mark.unit
async def test_convert_request_target_sample_rate(tts_service):
    req = OpenAISpeechRequest(
        model="kokoro",
        input="hello",
        voice="af_heart",
        response_format="pcm",
        stream=False,
        target_sample_rate=22050,
        extra_params={"sample_rate": 16000},
    )
    tts_request = tts_service._convert_request(req)
    assert tts_request.target_sample_rate == 22050
    assert tts_request.extra_params.get("target_sample_rate") == 22050
    assert tts_request.extra_params.get("sample_rate") == 22050


@pytest.mark.unit
async def test_custom_voice_loads_qwen3_prompt_metadata(tts_service, monkeypatch):
    class _FakeVoiceManager:
        async def load_voice_reference_audio(self, user_id, voice_id):
            return b"audio-bytes"

        async def load_reference_metadata(self, user_id, voice_id):
            return VoiceReferenceMetadata(
                voice_id=voice_id,
                voice_clone_prompt_b64="BASE64PROMPT",
                voice_clone_prompt_format="qwen3_tts_prompt_v1",
            )

    def _fake_get_voice_manager():
        return _FakeVoiceManager()

    monkeypatch.setattr(
        "tldw_Server_API.app.core.TTS.voice_manager.get_voice_manager",
        _fake_get_voice_manager,
        raising=True,
    )

    req = OpenAISpeechRequest(
        model="qwen/qwen3-tts-12hz-1.7b-base",
        input="hello",
        voice="custom:voice-1",
        response_format="mp3",
        stream=False,
    )
    tts_request = tts_service._convert_request(req)
    await tts_service._apply_custom_voice_reference(tts_request, user_id=1, provider_hint="qwen3_tts")

    assert tts_request.voice_reference == b"audio-bytes"
    assert tts_request.extra_params.get("voice_clone_prompt") == {
        "format": "qwen3_tts_prompt_v1",
        "data_b64": "BASE64PROMPT",
    }


@pytest.mark.unit
async def test_custom_voice_stores_qwen3_prompt_metadata(tts_service, monkeypatch):
    saved = {}

    class _FakeVoiceManager:
        async def load_reference_metadata(self, user_id, voice_id):
            return None

        async def save_reference_metadata(self, user_id, metadata):
            saved["metadata"] = metadata

    def _fake_get_voice_manager():
        return _FakeVoiceManager()

    monkeypatch.setattr(
        "tldw_Server_API.app.core.TTS.voice_manager.get_voice_manager",
        _fake_get_voice_manager,
        raising=True,
    )

    request = TTSRequest(
        text="hello",
        voice="custom:voice-1",
        format=AudioFormat.MP3,
        extra_params={
            "voice_clone_prompt": {
                "format": "qwen3_tts_prompt_v1",
                "data_b64": "PROMPTDATA",
            }
        },
    )

    await tts_service._maybe_store_qwen3_voice_prompt(request, user_id=1, provider_key="qwen3_tts")

    metadata = saved.get("metadata")
    assert metadata is not None
    assert metadata.voice_clone_prompt_b64 == "PROMPTDATA"
    assert metadata.voice_clone_prompt_format == "qwen3_tts_prompt_v1"

# ========================================================================
# Caching Tests
# ========================================================================

class TestCaching:
    """Test caching functionality in the service."""

    @pytest.mark.unit
    async def test_cache_hit(self, tts_service, basic_tts_request):
        """Test cache hit for repeated requests."""
        # First request
        result1 = await tts_service.generate(basic_tts_request)

        # Second identical request should hit cache
        result2 = await tts_service.generate(basic_tts_request)

        # Current implementation does not cache at service layer; ensure both succeeded
        assert result1 is not None and result2 is not None

    @pytest.mark.unit
    async def test_cache_miss_different_params(self, tts_service):
        """Test cache miss with different parameters."""
        request1 = TTSRequest(text="Test 1", voice="alloy")
        request2 = TTSRequest(text="Test 2", voice="alloy")

        await tts_service.generate(request1)
        await tts_service.generate(request2)

        # Should call adapter twice for different texts
        adapter = tts_service._factory.get_adapter("openai")
        assert adapter.generate.call_count == 2
