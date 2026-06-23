# test_elevenlabs_adapter_mock.py
# Description: Mock/Unit tests for ElevenLabs TTS adapter
#
# Imports
import pytest
pytestmark = pytest.mark.unit
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import httpx
#
# Local Imports
from tldw_Server_API.app.core.TTS.adapters.elevenlabs_adapter import ElevenLabsAdapter
from tldw_Server_API.app.core.TTS.adapters.base import (
    TTSRequest,
    TTSResponse,
    AudioFormat,
    ProviderStatus
)
from tldw_Server_API.app.core.TTS.tts_exceptions import (
    TTSProviderNotConfiguredError,
    TTSAuthenticationError,
    TTSRateLimitError,
    TTSGenerationError
)
#
#######################################################################################################################
#
# Mock Tests for ElevenLabs Adapter

@pytest.mark.asyncio
class TestElevenLabsAdapterMock:
    """Mock/Unit tests for ElevenLabs adapter"""

    @pytest.fixture(autouse=True)
    def _mock_user_voice_fetch(self, monkeypatch):
        monkeypatch.setattr(
            ElevenLabsAdapter,
            "_fetch_user_voices",
            AsyncMock(return_value=None),
        )

    async def test_initialization_without_api_key(self):
        """Test initialization without API key"""
        adapter = ElevenLabsAdapter({})
        assert adapter.api_key is None

        # Should not initialize without key
        success = await adapter.initialize()
        assert not success
        assert adapter._status == ProviderStatus.NOT_CONFIGURED

    async def test_initialization_with_api_key(self):
        """Test initialization with API key"""
        adapter = ElevenLabsAdapter({"elevenlabs_api_key": "test-key-123"})
        assert adapter.api_key == "test-key-123"

        success = await adapter.initialize()
        assert success
        assert adapter._status == ProviderStatus.AVAILABLE

    async def test_capabilities_reporting(self):
        """Test capabilities are correctly reported"""
        adapter = ElevenLabsAdapter({"elevenlabs_api_key": "test-key"})
        await adapter.initialize()

        caps = await adapter.get_capabilities()

        assert caps.provider_name == "ElevenLabs"
        assert caps.supports_streaming is True
        assert caps.supports_voice_cloning is True
        assert caps.supports_emotion_control is True
        assert caps.max_text_length == 5000
        assert AudioFormat.MP3 in caps.supported_formats
        assert AudioFormat.PCM in caps.supported_formats
        assert AudioFormat.WAV in caps.supported_formats
        assert AudioFormat.ULAW not in caps.supported_formats

    async def test_voice_mapping(self):
        """Test voice mapping functionality"""
        adapter = ElevenLabsAdapter({})

        # Test generic mappings
        assert adapter.map_voice("female") == "rachel"
        assert adapter.map_voice("male") == "drew"
        assert adapter.map_voice("british") == "dave"
        assert adapter.map_voice("irish") == "fin"
        assert adapter.map_voice("young_female") == "bella"
        assert adapter.map_voice("young_male") == "antoni"

        # Test passthrough for unknown voices
        assert adapter.map_voice("custom-voice") == "custom-voice"

    async def test_voice_id_detection(self):
        """Test voice ID detection"""
        adapter = ElevenLabsAdapter({"elevenlabs_api_key": "test-key"})
        await adapter.initialize()

        # Test that 20-character alphanumeric strings are treated as voice IDs
        voice_id = "21m00Tcm4TlvDq8ikWAM"
        assert adapter._get_voice_id(voice_id) == voice_id

        # Test that voice names are mapped to IDs
        assert adapter._get_voice_id("rachel") == "21m00Tcm4TlvDq8ikWAM"
        assert adapter._get_voice_id("drew") == "29vD33N1CtxCmqQRPOHJ"

    async def test_model_selection(self):
        """Test model selection logic"""
        adapter = ElevenLabsAdapter({
            "elevenlabs_api_key": "test-key",
            "elevenlabs_model": "eleven_multilingual_v2"
        })

        assert adapter.default_model == "eleven_multilingual_v2"

        # Test language-based model selection
        request_en = TTSRequest(text="Test", language="en")
        request_ja = TTSRequest(text="Test", language="ja")

        # English can use monolingual model
        model_en = adapter._select_model(request_en)
        assert model_en == "eleven_multilingual_v2"  # Uses configured default

        # Japanese requires multilingual model
        model_ja = adapter._select_model(request_ja)
        assert model_ja == "eleven_multilingual_v2"

    @patch.object(ElevenLabsAdapter, '_fetch_user_voices')
    async def test_user_voice_fetching(self, mock_fetch):
        """Test fetching user voices from API"""
        mock_fetch.return_value = None

        adapter = ElevenLabsAdapter({"elevenlabs_api_key": "test-key"})
        await adapter.initialize()

        mock_fetch.assert_called_once()

    async def test_voice_settings_configuration(self):
        """Test voice settings configuration"""
        adapter = ElevenLabsAdapter({
            "elevenlabs_api_key": "test-key",
            "elevenlabs_stability": 0.7,
            "elevenlabs_similarity_boost": 0.8,
            "elevenlabs_style": 0.5,
            "elevenlabs_speaker_boost": False
        })

        assert adapter.stability == 0.7
        assert adapter.similarity_boost == 0.8
        assert adapter.style == 0.5
        assert adapter.use_speaker_boost is False

    async def test_streaming_context_manager(self):
        """Test streaming uses context manager properly"""
        adapter = ElevenLabsAdapter({"elevenlabs_api_key": "test-key"})
        await adapter.initialize()

        call_state = {"count": 0}

        async def fake_stream(*args, **kwargs):
            call_state["count"] += 1
            yield b"chunk1"
            yield b"chunk2"

        with patch('tldw_Server_API.app.core.TTS.adapters.elevenlabs_adapter.astream_bytes', new=fake_stream):
            request = TTSRequest(text="Test", voice="rachel", stream=True)
            response = await adapter.generate(request)

            chunks = []
            async for chunk in response.audio_stream:
                chunks.append(chunk)

            assert len(chunks) > 0
            assert call_state["count"] == 1

    async def test_error_handling_during_streaming(self):
        """Test error handling during streaming"""
        adapter = ElevenLabsAdapter({"elevenlabs_api_key": "test-key"})
        await adapter.initialize()

        # Mock a failing stream
        async def mock_stream_error(*args, **kwargs):
            raise httpx.NetworkError("Connection lost", request=MagicMock())

        with patch('tldw_Server_API.app.core.TTS.adapters.elevenlabs_adapter.astream_bytes', new=mock_stream_error):
            request = TTSRequest(text="Test", voice="rachel", stream=True)

            response = await adapter.generate(request)
            with pytest.raises(Exception):  # Should raise network error
                async for _ in response.audio_stream:
                    pass

    async def test_format_header_mapping(self):
        """Test audio format to Accept header mapping"""
        adapter = ElevenLabsAdapter({})

        assert adapter._get_accept_header(AudioFormat.MP3) == "audio/mpeg"
        assert adapter._get_accept_header(AudioFormat.PCM) == "audio/pcm"
        assert adapter._get_accept_header(AudioFormat.WAV) == "audio/wav"
        assert adapter._get_accept_header(AudioFormat.ULAW) == "audio/mpeg"

        # Unknown format defaults to MP3
        assert adapter._get_accept_header(AudioFormat.OGG) == "audio/mpeg"

    async def test_text_preprocessing(self):
        """Test text preprocessing"""
        adapter = ElevenLabsAdapter({})

        # Test truncation for long text
        long_text = "a" * 6000
        processed = adapter.preprocess_text(long_text)
        assert len(processed) == 5000

        # Test normal text passes through
        normal_text = "Hello world"
        processed = adapter.preprocess_text(normal_text)
        assert processed == "Hello world"

    async def test_cleanup_on_close(self):
        """Test resource cleanup on close"""
        adapter = ElevenLabsAdapter({"elevenlabs_api_key": "test-key"})
        await adapter.initialize()

        assert adapter._initialized is True
        assert adapter._status == ProviderStatus.AVAILABLE
        assert adapter.client is not None

        await adapter.close()

        assert adapter._initialized is False
        assert adapter._status == ProviderStatus.DISABLED

#######################################################################################################################
#
# End of test_elevenlabs_adapter_mock.py
