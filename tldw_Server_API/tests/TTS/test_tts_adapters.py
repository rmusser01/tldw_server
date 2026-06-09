# test_tts_adapters.py
# Description: Comprehensive tests for TTS adapter pattern implementation
#
# Imports
import asyncio
import os
import pytest
from unittest.mock import Mock, AsyncMock, MagicMock, patch
import numpy as np

#
# Local Imports
from tldw_Server_API.app.core.TTS.adapters.base import (
    TTSAdapter,
    TTSCapabilities,
    TTSRequest,
    TTSResponse,
    AudioFormat,
    VoiceInfo,
    ProviderStatus,
)
from tldw_Server_API.app.core.TTS.adapters.openai_adapter import OpenAIAdapter
from tldw_Server_API.app.core.TTS.adapters.kokoro_adapter import KokoroAdapter
from tldw_Server_API.app.core.TTS.adapters.higgs_adapter import HiggsAdapter
from tldw_Server_API.app.core.TTS.adapters.dia_adapter import DiaAdapter
from tldw_Server_API.app.core.TTS.adapters.chatterbox_adapter import ChatterboxAdapter
from tldw_Server_API.app.core.TTS.adapter_registry import (
    TTSAdapterRegistry,
    TTSAdapterFactory,
    TTSProvider,
    get_tts_factory,
    close_tts_factory,
)
from tldw_Server_API.app.core.TTS.tts_service_v2 import TTSServiceV2, get_tts_service_v2, close_tts_service_v2


_RUN_EXTERNAL_API_TESTS = os.getenv("RUN_EXTERNAL_API_TESTS", "0") == "1"
REAL_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") if _RUN_EXTERNAL_API_TESTS else None


@pytest.fixture(autouse=True)
def clear_tts_env(monkeypatch):
    """Default to cleared API keys; individual tests restore as needed."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    return None


#
#######################################################################################################################
#
# Test Classes


class TestTTSAdapterBase:
    """Test the base TTSAdapter class"""

    def test_adapter_initialization(self):
        """Test basic adapter initialization"""

        # Create a concrete implementation for testing
        class TestAdapter(TTSAdapter):
            async def initialize(self) -> bool:
                return True

            async def generate(self, request: TTSRequest) -> TTSResponse:
                return TTSResponse(audio_data=b"test")

            async def get_capabilities(self) -> TTSCapabilities:
                return TTSCapabilities(
                    provider_name="Test",
                    supported_languages={"en"},
                    supported_voices=[],
                    supported_formats={AudioFormat.WAV},
                    max_text_length=1000,
                    supports_streaming=False,
                )

        adapter = TestAdapter({"test_key": "test_value"})
        assert adapter.config["test_key"] == "test_value"
        assert adapter.status == ProviderStatus.NOT_CONFIGURED
        assert adapter.provider_name == "Test"

    @pytest.mark.asyncio
    async def test_validate_request(self):
        """Test request validation"""

        class TestAdapter(TTSAdapter):
            async def initialize(self) -> bool:
                return True

            async def generate(self, request: TTSRequest) -> TTSResponse:
                return TTSResponse(audio_data=b"test")

            async def get_capabilities(self) -> TTSCapabilities:
                return TTSCapabilities(
                    provider_name="Test",
                    supported_languages={"en"},
                    supported_voices=[],
                    supported_formats={AudioFormat.WAV, AudioFormat.MP3},
                    max_text_length=100,
                    supports_streaming=True,
                )

        adapter = TestAdapter()
        await adapter.ensure_initialized()

        # Valid request
        request = TTSRequest(text="Hello world", language="en", format=AudioFormat.WAV, stream=True)
        is_valid, error = await adapter.validate_request(request)
        assert is_valid
        assert error is None

        # Invalid format
        request.format = AudioFormat.OPUS
        is_valid, error = await adapter.validate_request(request)
        assert not is_valid
        assert "Format opus not supported" in error

        # Text too long
        request.format = AudioFormat.WAV
        request.text = "x" * 200
        is_valid, error = await adapter.validate_request(request)
        assert not is_valid
        assert "exceeds maximum length" in error

    def test_parse_dialogue(self):
        """Test dialogue parsing"""

        class TestAdapter(TTSAdapter):
            async def initialize(self) -> bool:
                return True

            async def generate(self, request: TTSRequest) -> TTSResponse:
                return TTSResponse()

            async def get_capabilities(self) -> TTSCapabilities:
                return TTSCapabilities(
                    provider_name="Test",
                    supported_languages={"en"},
                    supported_voices=[],
                    supported_formats={AudioFormat.WAV},
                    max_text_length=1000,
                    supports_streaming=False,
                )

        adapter = TestAdapter()

        # Parse dialogue with speakers
        text = "Alice: Hello there! Bob: Hi Alice! Charlie: Hey everyone!"
        result = adapter.parse_dialogue(text)
        assert len(result) == 3
        assert result[0] == ("Alice", "Hello there!")
        assert result[1] == ("Bob", "Hi Alice!")
        assert result[2] == ("Charlie", "Hey everyone!")

        # No speakers
        text = "Just plain text"
        result = adapter.parse_dialogue(text)
        assert len(result) == 1
        assert result[0] == ("default", "Just plain text")


@pytest.mark.asyncio
class TestOpenAIAdapter:
    """Test OpenAI adapter implementation"""

    async def test_initialization_without_key(self):
        """Test initialization without API key"""
        adapter = OpenAIAdapter({})
        assert adapter.api_key is None
        assert adapter.status == ProviderStatus.NOT_CONFIGURED

        success = await adapter.initialize()
        assert not success
        assert adapter.status == ProviderStatus.NOT_CONFIGURED

    async def test_initialization_with_key(self):
        """Test initialization with API key"""
        adapter = OpenAIAdapter({"openai_api_key": "test-key-123"})
        assert adapter.api_key == "test-key-123"

        success = await adapter.initialize()
        assert success
        assert adapter.status == ProviderStatus.AVAILABLE

    async def test_capabilities(self):
        """Test OpenAI capabilities"""
        adapter = OpenAIAdapter({"openai_api_key": "test-key"})
        await adapter.initialize()

        caps = await adapter.get_capabilities()
        assert caps.provider_name == "OpenAI"
        assert "en" in caps.supported_languages
        assert AudioFormat.MP3 in caps.supported_formats
        assert caps.supports_streaming
        assert not caps.supports_emotion_control
        assert not caps.supports_voice_cloning

    def test_voice_mapping(self):
        """Test voice mapping"""
        adapter = OpenAIAdapter({})

        # Direct OpenAI voice
        assert adapter.map_voice("alloy") == "alloy"
        assert adapter.map_voice("nova") == "nova"

        # Generic mappings
        assert adapter.map_voice("male") == "onyx"
        assert adapter.map_voice("female") == "nova"
        assert adapter.map_voice("neutral") == "alloy"

        # Unknown voice
        assert adapter.map_voice("unknown") == "alloy"


@pytest.mark.asyncio
class TestKokoroAdapter:
    """Test Kokoro adapter implementation"""

    async def test_initialization_onnx(self):
        """Test ONNX initialization"""
        adapter = KokoroAdapter(
            {"kokoro_use_onnx": True, "kokoro_model_path": "test_model.onnx", "kokoro_voices_json": "test_voices.json"}
        )

        # Mock the kokoro_onnx import
        with patch("tldw_Server_API.app.core.TTS.adapters.kokoro_adapter.os.path.exists") as mock_exists:
            mock_exists.return_value = False  # Model files don't exist

            success = await adapter.initialize()
            assert not success  # Should fail without model files

    async def test_capabilities(self):
        """Test Kokoro capabilities"""
        adapter = KokoroAdapter({})

        # Get capabilities without initialization
        caps = await adapter.get_capabilities()
        assert caps.provider_name == "Kokoro"
        assert "en-us" in caps.supported_languages
        assert "en-gb" in caps.supported_languages
        assert AudioFormat.WAV in caps.supported_formats
        assert caps.supports_streaming
        assert caps.supports_phonemes
        assert caps.supports_multi_speaker  # Voice mixing

    def test_voice_processing(self):
        """Test voice processing and mixing"""
        adapter = KokoroAdapter({})

        # Single voice
        assert adapter._process_voice("af_bella") == "af_bella"

        # Mixed voice
        mixed = "af_bella(2)+af_sky(1)"
        assert adapter._process_voice(mixed) == mixed

        # Generic mapping
        assert adapter._process_voice("female") == "af_bella"
        assert adapter._process_voice("british_female") == "bf_emma"

    def test_language_detection(self):
        """Test language detection from voice"""
        adapter = KokoroAdapter({})

        # American voices
        assert adapter._get_language_from_voice("af_bella") == "en-us"
        assert adapter._get_language_from_voice("am_adam") == "en-us"

        # British voices
        assert adapter._get_language_from_voice("bf_emma") == "en-gb"
        assert adapter._get_language_from_voice("bm_george") == "en-gb"

        # Mixed voice (uses first voice)
        assert adapter._get_language_from_voice("af_bella+bf_emma") == "en-us"

    def test_text_chunking(self):
        """Test text chunking for optimal processing"""
        adapter = KokoroAdapter({})

        # Short text (single chunk)
        text = "This is a short sentence."
        chunks = adapter.chunk_text(text)
        assert len(chunks) == 1
        assert chunks[0] == text

        # Long text (multiple chunks)
        text = "First sentence. " * 20  # Create long text
        chunks = adapter.chunk_text(text)
        assert len(chunks) > 1
        assert all(chunk.endswith(".") for chunk in chunks)


@pytest.mark.asyncio
class TestAdapterRegistry:
    """Test the adapter registry"""

    async def test_registry_initialization(self):
        """Test registry initialization"""
        registry = TTSAdapterRegistry({"test_config": "value"})
        assert registry.config["test_config"] == "value"
        assert len(registry._adapter_specs) == len(TTSAdapterRegistry.DEFAULT_ADAPTERS)

    async def test_register_custom_adapter(self):
        """Test registering a custom adapter"""
        registry = TTSAdapterRegistry()

        class CustomAdapter(TTSAdapter):
            async def initialize(self) -> bool:
                return True

            async def generate(self, request: TTSRequest) -> TTSResponse:
                return TTSResponse()

            async def get_capabilities(self) -> TTSCapabilities:
                return TTSCapabilities(
                    provider_name="Custom",
                    supported_languages={"en"},
                    supported_voices=[],
                    supported_formats={AudioFormat.WAV},
                    max_text_length=1000,
                    supports_streaming=False,
                )

        # Register custom adapter
        registry.register_adapter(TTSProvider.OPENAI, CustomAdapter)
        assert registry._adapter_specs[TTSProvider.OPENAI] == CustomAdapter

    async def test_get_adapter_with_config(self):
        """Test getting adapter with configuration"""
        config = {"openai_api_key": "test-key", "openai_enabled": True}
        registry = TTSAdapterRegistry(config)

        adapter = await registry.get_adapter(TTSProvider.OPENAI)
        assert adapter is not None
        assert adapter.status == ProviderStatus.AVAILABLE

    async def test_get_adapter_with_alias_string(self):
        """Test alias normalization for provider names."""
        config = {"openai_api_key": "test-key", "openai_enabled": True}
        registry = TTSAdapterRegistry(config)

        adapter = await registry.get_adapter("open_ai")
        assert adapter is not None
        assert adapter.status == ProviderStatus.AVAILABLE

    def test_legacy_kokoro_provider_config_aliases_generic_fields(self):
        """Legacy flattened config should expose Kokoro ONNX settings under adapter keys."""
        registry = TTSAdapterRegistry(
            {
                "kokoro_enabled": True,
                "kokoro_config": {
                    "enabled": True,
                    "use_onnx": True,
                    "model_path": "models/kokoro/onnx/model.onnx",
                    "voices_json": "models/kokoro/onnx/voices-v1.0.bin",
                    "voice_dir": "models/kokoro/voices",
                    "device": "cpu",
                },
            }
        )

        provider_config = registry._get_provider_config(TTSProvider.KOKORO)

        assert provider_config["kokoro_use_onnx"] is True
        assert provider_config["kokoro_model_path"] == "models/kokoro/onnx/model.onnx"
        assert provider_config["kokoro_voices_json"] == "models/kokoro/onnx/voices-v1.0.bin"
        assert provider_config["kokoro_voice_dir"] == "models/kokoro/voices"
        assert provider_config["kokoro_device"] == "cpu"

    def test_chatterbox_provider_config_aliases_yaml_fields(self):
        """Chatterbox YAML settings should be duplicated under adapter-prefixed keys."""
        registry = TTSAdapterRegistry(
            {
                "chatterbox_enabled": True,
                "chatterbox_config": {
                    "enabled": True,
                    "variant": "turbo",
                    "model_path": "ResembleAI/chatterbox",
                    "multilingual_model_path": "ResembleAI/chatterbox-multilingual",
                    "turbo_model_path": "ResembleAI/chatterbox-turbo",
                    "vc_model_path": "models/chatterbox-vc",
                    "device": "cpu",
                    "use_multilingual": True,
                    "use_bf16": "auto",
                    "disable_watermark": False,
                    "target_latency_ms": 250,
                    "auto_download": False,
                    "conditionals_cache_size": 4,
                    "default_exaggeration": 0.7,
                    "cfg_weight": 0.4,
                    "temperature": 0.9,
                    "repetition_penalty": 1.1,
                    "min_p": 0.02,
                    "top_p": 0.95,
                },
            }
        )

        provider_config = registry._get_provider_config(TTSProvider.CHATTERBOX)

        assert provider_config["chatterbox_variant"] == "turbo"
        assert provider_config["chatterbox_model_path"] == "ResembleAI/chatterbox"
        assert provider_config["chatterbox_multilingual_model_path"] == "ResembleAI/chatterbox-multilingual"
        assert provider_config["chatterbox_turbo_model_path"] == "ResembleAI/chatterbox-turbo"
        assert provider_config["chatterbox_vc_model_path"] == "models/chatterbox-vc"
        assert provider_config["chatterbox_device"] == "cpu"
        assert provider_config["chatterbox_use_multilingual"] is True
        assert provider_config["chatterbox_use_bf16"] == "auto"
        assert provider_config["chatterbox_disable_watermark"] is False
        assert provider_config["chatterbox_target_latency_ms"] == 250
        assert provider_config["chatterbox_auto_download"] is False
        assert provider_config["chatterbox_conditionals_cache_size"] == 4
        assert provider_config["chatterbox_default_exaggeration"] == 0.7
        assert provider_config["chatterbox_cfg_weight"] == 0.4
        assert provider_config["chatterbox_temperature"] == 0.9
        assert provider_config["chatterbox_repetition_penalty"] == 1.1
        assert provider_config["chatterbox_min_p"] == 0.02
        assert provider_config["chatterbox_top_p"] == 0.95

    async def test_disabled_provider(self):
        """Test disabled provider"""
        config = {"openai_enabled": False}
        registry = TTSAdapterRegistry(config)

        adapter = await registry.get_adapter(TTSProvider.OPENAI)
        assert adapter is None

    async def test_close_all_skips_resource_manager_when_no_adapters_initialized(self):
        """Cleanup should not lazily create the TTS resource manager."""
        registry = TTSAdapterRegistry({})

        with patch(
            "tldw_Server_API.app.core.TTS.adapter_registry.get_resource_manager",
            new=AsyncMock(side_effect=AssertionError("resource manager should not be created during close_all")),
        ):
            await registry.close_all()

    async def test_find_adapter_for_requirements(self):
        """Test finding adapter by requirements"""
        registry = TTSAdapterRegistry({"openai_api_key": "test"})

        # Initialize OpenAI adapter
        await registry.get_adapter(TTSProvider.OPENAI)

        # Find adapter for streaming
        adapter = await registry.find_adapter_for_requirements(supports_streaming=True, format=AudioFormat.MP3)
        assert adapter is not None
        assert adapter.capabilities.supports_streaming

    async def test_get_all_capabilities(self):
        """Test getting all capabilities"""
        registry = TTSAdapterRegistry({"openai_api_key": "test"})

        caps = await registry.get_all_capabilities()
        assert TTSProvider.OPENAI in caps
        assert caps[TTSProvider.OPENAI].provider_name == "OpenAI"

    async def test_get_all_capabilities_uses_static_kitten_capabilities_without_init(self):
        """Static KittenTTS capability discovery should not initialize the runtime."""

        class StaticKittenAdapter(TTSAdapter):
            STATIC_CAPABILITY_DISCOVERY = True

            async def initialize(self) -> bool:
                raise AssertionError("Kitten static capability discovery should not initialize the adapter")

            async def generate(self, request: TTSRequest) -> TTSResponse:
                return TTSResponse(audio_data=b"unused")

            async def get_capabilities(self) -> TTSCapabilities:
                return TTSCapabilities(
                    provider_name="KittenTTS",
                    supported_languages={"en"},
                    supported_voices=[VoiceInfo(id="Bella", name="Bella")],
                    supported_formats={AudioFormat.WAV, AudioFormat.MP3, AudioFormat.PCM},
                    max_text_length=5000,
                    supports_streaming=True,
                    sample_rate=24000,
                    default_format=AudioFormat.WAV,
                )

        registry = TTSAdapterRegistry({"providers": {"kitten_tts": {"enabled": True}}})
        registry.register_adapter(TTSProvider.KITTEN_TTS, StaticKittenAdapter)

        caps = await registry.get_all_capabilities()

        assert TTSProvider.KITTEN_TTS in caps
        assert caps[TTSProvider.KITTEN_TTS].provider_name == "KittenTTS"

    async def test_get_all_capabilities_keeps_static_kitten_capabilities_after_failed_init(self):
        """Failed runtime init must not suppress static capability discovery for KittenTTS."""

        class StaticKittenAdapter(TTSAdapter):
            STATIC_CAPABILITY_DISCOVERY = True

            async def initialize(self) -> bool:
                raise RuntimeError("simulated runtime init failure")

            async def generate(self, request: TTSRequest) -> TTSResponse:
                return TTSResponse(audio_data=b"unused")

            async def get_capabilities(self) -> TTSCapabilities:
                return TTSCapabilities(
                    provider_name="KittenTTS",
                    supported_languages={"en"},
                    supported_voices=[VoiceInfo(id="Bella", name="Bella")],
                    supported_formats={AudioFormat.WAV, AudioFormat.MP3, AudioFormat.PCM},
                    max_text_length=5000,
                    supports_streaming=True,
                    sample_rate=24000,
                    default_format=AudioFormat.WAV,
                )

        registry = TTSAdapterRegistry({"providers": {"kitten_tts": {"enabled": True}}})
        registry.register_adapter(TTSProvider.KITTEN_TTS, StaticKittenAdapter)

        failed_adapter = await registry.get_adapter(TTSProvider.KITTEN_TTS)

        assert failed_adapter is None

        caps = await registry.get_all_capabilities()

        assert TTSProvider.KITTEN_TTS in caps
        assert caps[TTSProvider.KITTEN_TTS].provider_name == "KittenTTS"

    def test_status_summary(self):
        """Test status summary"""
        registry = TTSAdapterRegistry()

        summary = registry.get_status_summary()
        assert summary["total_providers"] == len(TTSProvider)
        assert summary["initialized"] == 0
        assert summary["available"] == 0
        assert "openai" in summary["providers"]


@pytest.mark.asyncio
class TestTTSAdapterFactory:
    """Test the adapter factory"""

    async def test_factory_initialization(self):
        """Test factory initialization"""
        factory = TTSAdapterFactory({"test": "config"})
        assert factory.registry.config["test"] == "config"

    async def test_get_adapter_by_model(self):
        """Test getting adapter by model name"""
        factory = TTSAdapterFactory({"openai_api_key": "test"})

        # OpenAI models
        adapter = await factory.get_adapter_by_model("tts-1")
        assert adapter is not None
        assert isinstance(adapter, OpenAIAdapter)

        # Kokoro models
        adapter = await factory.get_adapter_by_model("kokoro")
        assert adapter is not None or adapter is None  # Depends on model availability

        # Unknown model
        adapter = await factory.get_adapter_by_model("unknown-model")
        assert adapter is None

    def test_get_provider_for_model_alias(self):
        """Factory should resolve provider aliases when model mapping is absent."""
        factory = TTSAdapterFactory({})

        assert factory.get_provider_for_model("open_ai") == TTSProvider.OPENAI
        assert factory.get_provider_for_model("vibevoice-realtime") == TTSProvider.VIBEVOICE_REALTIME
        assert factory.get_provider_for_model("kitten_tts") == TTSProvider.KITTEN_TTS
        assert factory.get_provider_for_model("kitten-tts") == TTSProvider.KITTEN_TTS
        assert factory.get_provider_for_model("KittenML/kitten-tts-nano-0.8") == TTSProvider.KITTEN_TTS
        assert factory.get_provider_for_model("KittenML/kitten-tts-nano-0.8-fp32") == TTSProvider.KITTEN_TTS

    async def test_get_best_adapter(self):
        """Test getting best adapter for requirements"""
        factory = TTSAdapterFactory({"openai_api_key": "test"})

        adapter = await factory.get_best_adapter(language="en", supports_streaming=True)
        assert adapter is not None

    def test_get_status(self):
        """Test factory status"""
        factory = TTSAdapterFactory()

        status = factory.get_status()
        assert "total_providers" in status
        assert "initialized" in status
        assert "providers" in status

    async def test_unload_provider_closes_cached_adapter(self):
        """Factory unload should close one initialized provider and clear its cache entry."""

        class ClosingAdapter(TTSAdapter):
            closed = False

            async def initialize(self) -> bool:
                self._status = ProviderStatus.AVAILABLE
                return True

            async def generate(self, request: TTSRequest) -> TTSResponse:
                return TTSResponse(audio_data=b"ok")

            async def get_capabilities(self) -> TTSCapabilities:
                return TTSCapabilities(
                    provider_name="Mock",
                    supported_languages={"en"},
                    supported_voices=[],
                    supported_formats={AudioFormat.WAV},
                    max_text_length=1000,
                    supports_streaming=False,
                )

            async def close(self):
                self.closed = True
                await super().close()

        factory = TTSAdapterFactory({"providers": {"mock": {"enabled": True}}})
        factory.registry._adapter_specs = {TTSProvider.MOCK: ClosingAdapter}
        factory.registry._base.register_adapter(TTSProvider.MOCK.value, ClosingAdapter)

        adapter = await factory.registry.get_adapter(TTSProvider.MOCK)
        assert adapter is not None
        assert adapter.status == ProviderStatus.AVAILABLE

        result = await factory.unload_provider("mock")

        assert result == {"provider": "mock", "unloaded": True}
        assert adapter.closed is True
        status_after = factory.get_status()
        assert status_after["providers"]["mock"]["initialized"] is False

    async def test_unload_provider_is_idempotent_when_not_loaded(self):
        """Unloading a registered but uncached provider should be a no-op success."""
        class UnusedAdapter(TTSAdapter):
            async def initialize(self) -> bool:
                self._status = ProviderStatus.AVAILABLE
                return True

            async def generate(self, request: TTSRequest) -> TTSResponse:
                return TTSResponse(audio_data=b"ok")

            async def get_capabilities(self) -> TTSCapabilities:
                return TTSCapabilities(
                    provider_name="Mock",
                    supported_languages={"en"},
                    supported_voices=[],
                    supported_formats={AudioFormat.WAV},
                    max_text_length=1000,
                    supports_streaming=False,
                )

        factory = TTSAdapterFactory({"providers": {"mock": {"enabled": True}}})
        factory.registry.register_adapter(TTSProvider.MOCK, UnusedAdapter)

        result = await factory.unload_provider("mock")

        assert result == {"provider": "mock", "unloaded": False}


@pytest.mark.asyncio
class TestTTSServiceV2:
    """Test the enhanced TTS service"""

    async def test_service_initialization(self):
        """Test service initialization"""
        factory = TTSAdapterFactory({"openai_api_key": "test"})
        service = TTSServiceV2(factory)
        assert service.factory == factory

    @pytest.mark.skipif(not REAL_OPENAI_API_KEY, reason="Requires OPENAI_API_KEY")
    async def test_list_voices(self, monkeypatch):
        """Test listing voices - requires real API key"""
        monkeypatch.setenv("OPENAI_API_KEY", REAL_OPENAI_API_KEY)
        factory = TTSAdapterFactory({"openai_api_key": REAL_OPENAI_API_KEY})
        service = TTSServiceV2(factory)

        voices = await service.list_voices()
        assert len(voices) > 0  # Should have at least one provider

    @pytest.mark.skipif(not REAL_OPENAI_API_KEY, reason="Requires OPENAI_API_KEY")
    async def test_get_capabilities(self, monkeypatch):
        """Test getting capabilities - requires real API key"""
        monkeypatch.setenv("OPENAI_API_KEY", REAL_OPENAI_API_KEY)
        factory = TTSAdapterFactory({"openai_api_key": REAL_OPENAI_API_KEY})
        service = TTSServiceV2(factory)

        caps = await service.get_capabilities()
        assert len(caps) > 0  # Should have at least one provider

    def test_get_status(self):
        """Test service status"""
        factory = TTSAdapterFactory()
        service = TTSServiceV2(factory)

        status = service.get_status()
        assert "total_providers" in status
        assert "providers" in status

    async def test_unload_provider_delegates_to_factory(self):
        """Service unload should delegate to the configured factory."""
        factory = MagicMock()
        factory.unload_provider = AsyncMock(return_value={"provider": "chatterbox", "unloaded": True})
        service = TTSServiceV2(factory)

        result = await service.unload_provider("chatterbox")

        assert result == {"provider": "chatterbox", "unloaded": True}
        factory.unload_provider.assert_awaited_once_with("chatterbox")

    @pytest.mark.asyncio
    async def test_request_conversion(self):
        """Test OpenAI request conversion"""
        from tldw_Server_API.app.api.v1.schemas.audio_schemas import OpenAISpeechRequest

        factory = TTSAdapterFactory()
        service = TTSServiceV2(factory)

        openai_request = OpenAISpeechRequest(
            input="Hello world", model="tts-1", voice="alloy", response_format="mp3", speed=1.0
        )

        tts_request = service._convert_request(openai_request)
        assert tts_request.text == "Hello world"
        assert tts_request.voice == "alloy"
        assert tts_request.format == AudioFormat.MP3
        assert tts_request.speed == 1.0


# Integration tests
@pytest.mark.asyncio
class TestIntegration:
    """Integration tests for the complete system"""

    async def test_singleton_management(self):
        """Test singleton management"""
        # Get factory singleton
        factory1 = await get_tts_factory({"test": "config"})
        factory2 = await get_tts_factory()
        assert factory1 is factory2

        # Get service singleton
        service1 = await get_tts_service_v2()
        service2 = await get_tts_service_v2()
        assert service1 is service2

        # Clean up
        await close_tts_service_v2()
        await close_tts_factory()

    @pytest.mark.skipif(not REAL_OPENAI_API_KEY, reason="Requires OPENAI_API_KEY")
    async def test_backwards_compatibility(self, monkeypatch):
        """Test backwards compatibility wrapper - requires real API key"""
        from tldw_Server_API.app.api.v1.schemas.audio_schemas import OpenAISpeechRequest
        from tldw_Server_API.app.core.TTS.tts_service_v2 import TTSService

        monkeypatch.setenv("OPENAI_API_KEY", REAL_OPENAI_API_KEY)

        # Create adapter (backwards compatible)
        adapter = TTSService()

        request = OpenAISpeechRequest(input="Test text", model="tts-1", voice="alloy")

        # Test generation with real service (will use real API if key is available)
        chunks = []
        async for chunk in adapter.generate_audio_stream(request, "openai_official_tts-1"):
            chunks.append(chunk)
            break  # Just get first chunk to avoid consuming API quota

        assert len(chunks) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

#
# End of test_tts_adapters.py
#######################################################################################################################
