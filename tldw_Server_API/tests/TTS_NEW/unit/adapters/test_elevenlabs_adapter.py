"""
Unit tests for ElevenLabs TTS adapter.

Tests the ElevenLabs TTS adapter with minimal mocking - only mocking
the actual ElevenLabs API calls.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import httpx
import json

from tldw_Server_API.app.core.TTS.adapters.elevenlabs_adapter import ElevenLabsTTSAdapter
from tldw_Server_API.app.core.TTS.adapters.base import (
    TTSRequest,
    TTSResponse,
    AudioFormat,
    VoiceSettings
)
from tldw_Server_API.app.core.TTS.tts_exceptions import (
    TTSProviderNotConfiguredError,
    TTSGenerationError,
    TTSRateLimitError,
    TTSValidationError,
    TTSQuotaExceededError
)

# ========================================================================
# Adapter Initialization Tests
# ========================================================================

class TestElevenLabsAdapterInitialization:
    """Test ElevenLabs adapter initialization and configuration."""

    @pytest.mark.unit
    def test_adapter_initialization_with_config(self):
        """Test adapter initialization with configuration."""
        config = {
            "api_key": "test-elevenlabs-key",
            "base_url": "https://api.elevenlabs.io/v1",
            "timeout": 60
        }

        adapter = ElevenLabsTTSAdapter(config)

        assert adapter.provider == "elevenlabs"
        assert adapter.is_available
        assert adapter.api_key == "test-elevenlabs-key"

    @pytest.mark.unit
    def test_adapter_initialization_without_api_key(self, monkeypatch):
        """Test adapter initialization without API key."""
        config = {
            "base_url": "https://api.elevenlabs.io/v1"
        }

        monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)

        with pytest.raises(TTSProviderNotConfiguredError):
            ElevenLabsTTSAdapter(config)

    @pytest.mark.unit
    def test_adapter_supported_models(self):
        """Test adapter reports supported models."""
        config = {"api_key": "test-key"}
        adapter = ElevenLabsTTSAdapter(config)

        assert "eleven_monolingual_v1" in adapter.supported_models
        assert "eleven_multilingual_v2" in adapter.supported_models
        assert "eleven_turbo_v2" in adapter.supported_models

    @pytest.mark.unit
    @patch('tldw_Server_API.app.core.TTS.adapters.elevenlabs_adapter.afetch')
    async def test_fetch_available_voices(self, mock_get):
        """Test fetching available voices from API."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "voices": [
                {"voice_id": "21m00Tcm4TlvDq8ikWAM", "name": "Rachel"},
                {"voice_id": "AZnzlk1XvdvUeBnXmlld", "name": "Domi"},
                {"voice_id": "EXAVITQu4vr4xnSDxMaL", "name": "Bella"}
            ]
        }
        mock_get.return_value = mock_response

        adapter = ElevenLabsTTSAdapter({"api_key": "test-key"})
        voices = await adapter.fetch_voices()

        assert len(voices) == 3
        assert "Rachel" in [v["name"] for v in voices]

# ========================================================================
# Request Validation Tests
# ========================================================================

class TestRequestValidation:
    """Test request validation in ElevenLabs adapter."""

    @pytest.mark.unit
    async def test_validate_valid_request(self):
        """Test validation of valid request."""
        adapter = ElevenLabsTTSAdapter({"api_key": "test-key"})

        request = TTSRequest(
            text="Hello world",
            voice="rachel",  # ElevenLabs uses voice names or IDs
            model="eleven_monolingual_v1"
        )

        # Should not raise
        await adapter.validate_request(request)

    @pytest.mark.unit
    async def test_validate_with_voice_id(self):
        """Test validation with voice ID instead of name."""
        adapter = ElevenLabsTTSAdapter({"api_key": "test-key"})

        request = TTSRequest(
            text="Hello",
            voice="21m00Tcm4TlvDq8ikWAM",  # Rachel's voice ID
            model="eleven_multilingual_v2"
        )

        await adapter.validate_request(request)

    @pytest.mark.unit
    async def test_validate_invalid_model(self):
        """Test validation rejects invalid model."""
        adapter = ElevenLabsTTSAdapter({"api_key": "test-key"})

        request = TTSRequest(
            text="Hello",
            voice="rachel",
            model="invalid-model"
        )

        with pytest.raises(TTSValidationError):
            await adapter.validate_request(request)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_legacy_model_alias_keeps_request_case_and_sends_canonical_payload(self):
        captured: dict[str, object] = {}

        async def _stream(**kwargs):
            captured.update(kwargs)
            yield b"audio"

        adapter = ElevenLabsTTSAdapter({"api_key": "test-key"})
        adapter.client = object()
        request = TTSRequest(
            text="Hello",
            voice="rachel",
            model="ELEVEN_MULTILINGUAL_V2",
        )

        await adapter.validate_request(request)
        with patch(
            "tldw_Server_API.app.core.TTS.adapters.elevenlabs_adapter.astream_bytes",
            _stream,
        ):
            chunks = [chunk async for chunk in adapter.generate_stream(request)]

        assert chunks == [b"audio"]
        assert request.model == "ELEVEN_MULTILINGUAL_V2"
        assert captured["json"]["model_id"] == "eleven_multilingual_v2"

    @pytest.mark.unit
    async def test_validate_text_too_long(self):
        """Test validation rejects text exceeding limit."""
        adapter = ElevenLabsTTSAdapter({"api_key": "test-key"})

        # ElevenLabs has 5000 char limit
        long_text = "a" * 6000
        request = TTSRequest(
            text=long_text,
            voice="rachel",
            model="eleven_monolingual_v1"
        )

        with pytest.raises(TTSValidationError) as exc_info:
            await adapter.validate_request(request)

        assert "exceeds maximum" in str(exc_info.value)

    @pytest.mark.unit
    async def test_validate_voice_settings(self):
        """Test validation with voice settings."""
        adapter = ElevenLabsTTSAdapter({"api_key": "test-key"})

        request = TTSRequest(
            text="Hello",
            voice="rachel",
            model="eleven_monolingual_v1",
            voice_settings=VoiceSettings(
                stability=0.5,
                similarity_boost=0.75,
                style=0.3,
                use_speaker_boost=True
            )
        )

        await adapter.validate_request(request)

# ========================================================================
# Audio Generation Tests
# ========================================================================

class TestAudioGeneration:
    """Test audio generation functionality."""

    @pytest.mark.unit
    @patch('tldw_Server_API.app.core.TTS.adapters.elevenlabs_adapter.afetch')
    async def test_generate_basic_audio(self, mock_post):
        """Test basic audio generation."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"fake_audio_data"
        mock_response.headers = {"content-type": "audio/mpeg"}
        mock_post.return_value = mock_response

        adapter = ElevenLabsTTSAdapter({"api_key": "test-key"})
        request = TTSRequest(
            text="Hello world",
            voice="rachel",
            model="eleven_monolingual_v1"
        )

        response = await adapter.generate(request)

        assert response.audio_content == b"fake_audio_data"
        assert response.provider == "elevenlabs"
        assert response.model == "eleven_monolingual_v1"

    @pytest.mark.unit
    @patch('tldw_Server_API.app.core.TTS.adapters.elevenlabs_adapter.afetch')
    async def test_generate_with_multilingual_model(self, mock_post):
        """Test generation with multilingual model."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"multilingual_audio"
        mock_post.return_value = mock_response

        adapter = ElevenLabsTTSAdapter({"api_key": "test-key"})
        request = TTSRequest(
            text="Bonjour le monde",
            voice="rachel",
            model="eleven_multilingual_v2",
            language="fr"
        )

        response = await adapter.generate(request)

        assert response.model == "eleven_multilingual_v2"
        assert response.audio_content == b"multilingual_audio"

    @pytest.mark.unit
    @patch('tldw_Server_API.app.core.TTS.adapters.elevenlabs_adapter.afetch')
    async def test_generate_with_voice_settings(self, mock_post):
        """Test generation with custom voice settings."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"custom_audio"
        mock_post.return_value = mock_response

        adapter = ElevenLabsTTSAdapter({"api_key": "test-key"})
        request = TTSRequest(
            text="Custom voice",
            voice="bella",
            model="eleven_monolingual_v1",
            voice_settings=VoiceSettings(
                stability=0.3,
                similarity_boost=0.9
            )
        )

        response = await adapter.generate(request)

        # Verify voice settings were included
        call_kwargs = mock_post.call_args[1]
        assert "voice_settings" in call_kwargs['json']
        assert call_kwargs['json']['voice_settings']['stability'] == 0.3

    @pytest.mark.unit
    @patch('tldw_Server_API.app.core.TTS.adapters.elevenlabs_adapter.afetch')
    async def test_generate_with_turbo_model(self, mock_post):
        """Test generation with turbo model for low latency."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"turbo_audio"
        mock_post.return_value = mock_response

        adapter = ElevenLabsTTSAdapter({"api_key": "test-key"})
        request = TTSRequest(
            text="Fast generation",
            voice="rachel",
            model="eleven_turbo_v2"
        )

        response = await adapter.generate(request)

        assert response.model == "eleven_turbo_v2"
        assert response.metadata.get("turbo") is True

# ========================================================================
# Streaming Generation Tests
# ========================================================================

class TestStreamingGeneration:
    """Test streaming audio generation."""

    @pytest.mark.unit
    async def test_streaming_generation(self):
        """Test streaming audio generation."""
        async def fake_stream(*args, **kwargs):
            for chunk in [b"chunk1", b"chunk2", b"chunk3"]:
                yield chunk

        with patch('tldw_Server_API.app.core.TTS.adapters.elevenlabs_adapter.astream_bytes', new=fake_stream):
            adapter = ElevenLabsTTSAdapter({"api_key": "test-key"})
            request = TTSRequest(
                text="Stream this",
                voice="domi",
                model="eleven_monolingual_v1"
            )

            chunks = []
            async for chunk in adapter.generate_stream(request):
                chunks.append(chunk)

            assert chunks == [b"chunk1", b"chunk2", b"chunk3"]

    @pytest.mark.unit
    async def test_streaming_with_websocket(self):
        """Test streaming via WebSocket for ultra-low latency."""
        # ElevenLabs supports WebSocket streaming for certain models
        adapter = ElevenLabsTTSAdapter({
            "api_key": "test-key",
            "use_websocket": True
        })

        # Mock WebSocket behavior would go here
        # This is a placeholder showing the test structure
        pass

# ========================================================================
# Voice Management Tests
# ========================================================================

class TestVoiceManagement:
    """Test voice management features."""

    @pytest.mark.unit
    @patch('tldw_Server_API.app.core.TTS.adapters.elevenlabs_adapter.afetch')
    async def test_get_voice_details(self, mock_get):
        """Test getting details for a specific voice."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "voice_id": "21m00Tcm4TlvDq8ikWAM",
            "name": "Rachel",
            "labels": {"accent": "american", "gender": "female"},
            "settings": {
                "stability": 0.75,
                "similarity_boost": 0.75
            }
        }
        mock_get.return_value = mock_response

        adapter = ElevenLabsTTSAdapter({"api_key": "test-key"})
        voice_info = await adapter.get_voice_info("21m00Tcm4TlvDq8ikWAM")

        assert voice_info["name"] == "Rachel"
        assert voice_info["labels"]["gender"] == "female"

    @pytest.mark.unit
    @patch('tldw_Server_API.app.core.TTS.adapters.elevenlabs_adapter.afetch')
    async def test_clone_voice(self, mock_post):
        """Test voice cloning functionality."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "voice_id": "new_voice_id_123"
        }
        mock_post.return_value = mock_response

        adapter = ElevenLabsTTSAdapter({"api_key": "test-key"})

        # This would require audio samples in real usage
        voice_id = await adapter.clone_voice(
            name="Custom Voice",
            samples=[b"audio_sample_1", b"audio_sample_2"]
        )

        assert voice_id == "new_voice_id_123"

# ========================================================================
# Error Handling Tests
# ========================================================================

class TestErrorHandling:
    """Test error handling in ElevenLabs adapter."""

    @pytest.mark.unit
    @patch('tldw_Server_API.app.core.TTS.adapters.elevenlabs_adapter.afetch')
    async def test_handle_rate_limit_error(self, mock_post):
        """Test handling of rate limit errors."""
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.headers = {"retry-after": "30"}
        mock_response.json.return_value = {
            "detail": {
                "status": "rate_limit_exceeded",
                "message": "Rate limit exceeded"
            }
        }
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "429", request=Mock(), response=mock_response
        )
        mock_post.return_value = mock_response

        adapter = ElevenLabsTTSAdapter({"api_key": "test-key"})
        request = TTSRequest(text="Test", voice="rachel", model="eleven_monolingual_v1")

        with pytest.raises(TTSRateLimitError) as exc_info:
            await adapter.generate(request)

        assert exc_info.value.retry_after == 30

    @pytest.mark.unit
    @patch('tldw_Server_API.app.core.TTS.adapters.elevenlabs_adapter.afetch')
    async def test_handle_quota_exceeded(self, mock_post):
        """Test handling of quota exceeded errors."""
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.json.return_value = {
            "detail": {
                "status": "quota_exceeded",
                "message": "Character quota exceeded"
            }
        }
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "429", request=Mock(), response=mock_response
        )
        mock_post.return_value = mock_response

        adapter = ElevenLabsTTSAdapter({"api_key": "test-key"})
        request = TTSRequest(text="Test", voice="rachel", model="eleven_monolingual_v1")

        with pytest.raises(TTSQuotaExceededError):
            await adapter.generate(request)

    @pytest.mark.unit
    @patch('tldw_Server_API.app.core.TTS.adapters.elevenlabs_adapter.afetch')
    async def test_handle_invalid_voice_error(self, mock_post):
        """Test handling of invalid voice errors."""
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.json.return_value = {
            "detail": {
                "status": "invalid_voice_id",
                "message": "Voice not found"
            }
        }
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "400", request=Mock(), response=mock_response
        )
        mock_post.return_value = mock_response

        adapter = ElevenLabsTTSAdapter({"api_key": "test-key"})
        request = TTSRequest(text="Test", voice="invalid_voice", model="eleven_monolingual_v1")

        with pytest.raises(TTSValidationError) as exc_info:
            await adapter.generate(request)

        assert "Voice not found" in str(exc_info.value)

# ========================================================================
# Usage and Quota Tests
# ========================================================================

class TestUsageAndQuota:
    """Test usage tracking and quota management."""

    @pytest.mark.unit
    @patch('tldw_Server_API.app.core.TTS.adapters.elevenlabs_adapter.afetch')
    async def test_get_usage_info(self, mock_get):
        """Test getting usage information."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "character_count": 10000,
            "character_limit": 100000,
            "can_extend_character_limit": True,
            "allowed_to_extend_character_limit": True
        }
        mock_get.return_value = mock_response

        adapter = ElevenLabsTTSAdapter({"api_key": "test-key"})
        usage = await adapter.get_usage()

        assert usage["character_count"] == 10000
        assert usage["character_limit"] == 100000
        assert usage["remaining"] == 90000
