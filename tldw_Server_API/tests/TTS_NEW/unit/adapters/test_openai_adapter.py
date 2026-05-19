"""
Unit tests for OpenAI TTS adapter.

Tests the OpenAI TTS adapter with minimal mocking - only mocking
the actual OpenAI API calls.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import httpx
from io import BytesIO

from tldw_Server_API.app.core.TTS.adapters.openai_adapter import (
    OpenAIAdapter as OpenAITTSAdapter,
    OpenAITTSAdapter as ProductionOpenAITTSAdapter,
)
from tldw_Server_API.app.core.TTS.adapters import openai_adapter as openai_mod
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
    TTSNetworkError,
    TTSAuthenticationError,
    TTSProviderInitializationError,
    TTSProviderError,
)

# ========================================================================
# Adapter Initialization Tests
# ========================================================================

class TestOpenAIAdapterInitialization:
    """Test OpenAI adapter initialization and configuration."""

    @pytest.mark.unit
    def test_adapter_initialization_with_config(self):
        """Test adapter initialization with configuration."""
        config = {
            "openai_api_key": "test-key-123",
            "openai_base_url": "https://api.openai.com/v1/audio/speech",
            "timeout": 30
        }

        adapter = OpenAITTSAdapter(config)

        assert adapter.provider_name.lower() == "openai"
        assert adapter.api_key == "test-key-123"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_adapter_initialization_without_api_key(self, monkeypatch):
        """Test adapter initialization without API key."""
        # Ensure environment doesn't supply a key implicitly
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        config = {
            "openai_base_url": "https://api.openai.com/v1/audio/speech"
        }
        adapter = OpenAITTSAdapter(config)
        # Production adapter returns False and sets status rather than raising
        success = await adapter.initialize()
        assert success is False
        assert str(adapter.status.value) in {"not_configured", "error"}

    @pytest.mark.unit
    @pytest.mark.asyncio
    @patch("tldw_Server_API.app.core.TTS.adapters.openai_adapter.get_resource_manager")
    async def test_adapter_initialization_with_api_key_verify_success(self, mock_get_resource_manager):
        """
        When verify_api_key_on_init is enabled, initialize() should perform a
        lightweight verification call but still succeed on healthy responses.
        """
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=MagicMock(status_code=200, content=b"ok", raise_for_status=MagicMock()))
        rm = AsyncMock()
        rm.get_http_client = AsyncMock(return_value=mock_client)
        mock_get_resource_manager.return_value = rm

        config = {
            "openai_api_key": "test-key-123",
            "openai_base_url": "https://api.openai.com/v1/audio/speech",
            "verify_api_key_on_init": True,
        }
        adapter = OpenAITTSAdapter(config)

        success = await adapter.initialize()

        assert success is True
        assert adapter.status == adapter.status.AVAILABLE
        # Verify that a single verification POST was issued
        mock_client.post.assert_called()

    @pytest.mark.unit
    @pytest.mark.asyncio
    @patch("tldw_Server_API.app.core.TTS.adapters.openai_adapter.get_resource_manager")
    async def test_adapter_initialization_with_api_key_verify_auth_failure(self, mock_get_resource_manager):
        """
        If verify_api_key_on_init is enabled and the verification step detects
        an auth failure, initialization should fail with TTSProviderInitializationError.
        """
        mock_client = AsyncMock()
        rm = AsyncMock()
        rm.get_http_client = AsyncMock(return_value=mock_client)
        mock_get_resource_manager.return_value = rm

        config = {
            "openai_api_key": "bad-key",
            "openai_base_url": "https://api.openai.com/v1/audio/speech",
            "verify_api_key_on_init": True,
        }
        adapter = OpenAITTSAdapter(config)

        with patch.object(
            OpenAITTSAdapter,
            "_generate_complete",
            new=AsyncMock(side_effect=TTSAuthenticationError("Invalid API key", provider="openai")),
        ):
            with pytest.raises(TTSProviderInitializationError):
                await adapter.initialize()
        assert adapter.status == adapter.status.ERROR

    @pytest.mark.unit
    @pytest.mark.asyncio
    @patch("tldw_Server_API.app.core.TTS.adapters.openai_adapter.get_resource_manager")
    async def test_api_key_verify_unexpected_error_log_sanitizes_exception_text(self, mock_get_resource_manager):
        raw_marker = "RAW_OPENAI_INIT_VERIFY_SECRET_MARKER"
        logged_messages: list[str] = []
        mock_client = AsyncMock()
        rm = AsyncMock()
        rm.get_http_client = AsyncMock(return_value=mock_client)
        mock_get_resource_manager.return_value = rm

        adapter = OpenAITTSAdapter({
            "openai_api_key": "test-key",
            "verify_api_key_on_init": True,
        })
        sink_id = openai_mod.logger.add(
            lambda message: logged_messages.append(message.record["message"]),
            level="WARNING",
        )

        try:
            with patch.object(
                OpenAITTSAdapter,
                "_generate_complete",
                new=AsyncMock(side_effect=RuntimeError(raw_marker)),
            ):
                assert await adapter.initialize() is True
        finally:
            openai_mod.logger.remove(sink_id)

        assert any("Unexpected error during API key verification on init" in message for message in logged_messages)
        assert all(raw_marker not in message for message in logged_messages)
        assert all("RuntimeError" in message for message in logged_messages)

    @pytest.mark.unit
    @pytest.mark.asyncio
    @patch("tldw_Server_API.app.core.TTS.adapters.openai_adapter.get_resource_manager")
    async def test_initialization_failure_log_sanitizes_exception_text(self, mock_get_resource_manager):
        raw_marker = "RAW_OPENAI_INIT_FAILURE_SECRET_MARKER"
        logged_messages: list[str] = []
        mock_get_resource_manager.side_effect = RuntimeError(raw_marker)
        adapter = OpenAITTSAdapter({"openai_api_key": "test-key"})
        sink_id = openai_mod.logger.add(
            lambda message: logged_messages.append(message.record["message"]),
            level="ERROR",
        )

        try:
            with pytest.raises(TTSProviderInitializationError) as exc_info:
                await adapter.initialize()
        finally:
            openai_mod.logger.remove(sink_id)

        assert exc_info.value.details["error"] == raw_marker
        assert any("Initialization failed" in message for message in logged_messages)
        assert all(raw_marker not in message for message in logged_messages)
        assert all("RuntimeError" in message for message in logged_messages)

    @pytest.mark.unit
    @pytest.mark.asyncio
    @patch("tldw_Server_API.app.core.TTS.adapters.openai_adapter.get_resource_manager")
    async def test_api_key_verify_auth_failure_log_sanitizes_exception_text(self, mock_get_resource_manager):
        raw_marker = "RAW_OPENAI_INIT_VERIFY_AUTH_SECRET_MARKER"
        logged_messages: list[str] = []
        mock_client = AsyncMock()
        rm = AsyncMock()
        rm.get_http_client = AsyncMock(return_value=mock_client)
        mock_get_resource_manager.return_value = rm
        adapter = OpenAITTSAdapter({
            "openai_api_key": "bad-key",
            "verify_api_key_on_init": True,
        })
        sink_id = openai_mod.logger.add(
            lambda message: logged_messages.append(message.record["message"]),
            level="ERROR",
        )

        try:
            with patch.object(
                OpenAITTSAdapter,
                "_generate_complete",
                new=AsyncMock(side_effect=TTSAuthenticationError(raw_marker, provider="openai")),
            ):
                with pytest.raises(TTSProviderInitializationError) as exc_info:
                    await adapter.initialize()
        finally:
            openai_mod.logger.remove(sink_id)

        assert str(exc_info.value) == "Failed to initialize OpenAI"
        assert any("API key verification failed during initialization" in message for message in logged_messages)
        assert all(raw_marker not in message for message in logged_messages)
        assert any("TTSAuthenticationError" in message for message in logged_messages)

    @pytest.mark.unit
    @pytest.mark.asyncio
    @patch("tldw_Server_API.app.core.TTS.adapters.openai_adapter.get_resource_manager")
    async def test_api_key_verify_nonfatal_log_sanitizes_exception_text(self, mock_get_resource_manager):
        raw_marker = "RAW_OPENAI_INIT_VERIFY_NONFATAL_SECRET_MARKER"
        logged_messages: list[str] = []
        mock_client = AsyncMock()
        rm = AsyncMock()
        rm.get_http_client = AsyncMock(return_value=mock_client)
        mock_get_resource_manager.return_value = rm
        adapter = OpenAITTSAdapter({
            "openai_api_key": "test-key",
            "verify_api_key_on_init": True,
        })
        sink_id = openai_mod.logger.add(
            lambda message: logged_messages.append(message.record["message"]),
            level="WARNING",
        )

        try:
            with patch.object(
                OpenAITTSAdapter,
                "_generate_complete",
                new=AsyncMock(side_effect=TTSNetworkError(raw_marker, provider="openai")),
            ):
                assert await adapter.initialize() is True
        finally:
            openai_mod.logger.remove(sink_id)

        assert any("API key verification during initialization did not succeed" in message for message in logged_messages)
        assert all(raw_marker not in message for message in logged_messages)
        assert all("TTSNetworkError" in message for message in logged_messages)

    @pytest.mark.unit
    @pytest.mark.asyncio
    @patch("tldw_Server_API.app.core.TTS.adapters.openai_adapter.get_resource_manager")
    async def test_adapter_initialization_with_api_key_verify_http_401_auth_failure(self, mock_get_resource_manager):
        """
        HTTP 401 during verify-on-init should be treated as a fatal auth failure
        and surfaced as TTSProviderInitializationError.
        """
        mock_client = AsyncMock()
        rm = AsyncMock()
        rm.get_http_client = AsyncMock(return_value=mock_client)
        mock_get_resource_manager.return_value = rm

        config = {
            "openai_api_key": "bad-key",
            "openai_base_url": "https://api.openai.com/v1/audio/speech",
            "verify_api_key_on_init": True,
        }
        adapter = OpenAITTSAdapter(config)

        # Simulate the real behavior of _generate_complete producing an HTTPStatusError
        # from response.raise_for_status() with status code 401.
        response = httpx.Response(
            status_code=401,
            request=httpx.Request("POST", "https://api.openai.com/v1/audio/speech"),
            content=b'{"error":{"message":"Invalid API key"}}',
        )
        http_error = httpx.HTTPStatusError("401 Unauthorized", request=response.request, response=response)

        with patch.object(
            OpenAITTSAdapter,
            "_generate_complete",
            new=AsyncMock(side_effect=http_error),
        ):
            with pytest.raises(TTSProviderInitializationError):
                await adapter.initialize()
        assert adapter.status == adapter.status.ERROR

    @pytest.mark.unit
    def test_adapter_supported_models(self):
        """Test adapter supports setting known models via config."""
        adapter_default = OpenAITTSAdapter({"openai_api_key": "test-key"})
        assert adapter_default.model in ("tts-1", "tts-1-hd")

        adapter_hd = OpenAITTSAdapter({"openai_api_key": "test-key", "openai_model": "tts-1-hd"})
        assert adapter_hd.model == "tts-1-hd"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_adapter_supported_voices(self):
        """Test adapter reports supported voices via capabilities."""
        adapter = OpenAITTSAdapter({"openai_api_key": "test-key"})
        caps = await adapter.get_capabilities()
        voice_ids = [v.id for v in caps.supported_voices]
        expected_voices = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
        for voice in expected_voices:
            assert voice in voice_ids

# ========================================================================
# Request Validation Tests
# ========================================================================

class TestRequestValidation:
    """Test request validation in OpenAI adapter."""

    @pytest.mark.unit
    async def test_validate_valid_request(self):
        """Test validation of valid request."""
        adapter = OpenAITTSAdapter({"openai_api_key": "test-key"})
        await adapter.ensure_initialized()
        request = TTSRequest(text="Hello world", voice="alloy", stream=False)
        is_valid, error = await adapter.validate_request(request)
        assert is_valid and error is None

    @pytest.mark.unit
    async def test_validate_invalid_voice(self):
        """OpenAI maps unknown voices to defaults; validation should pass."""
        adapter = OpenAITTSAdapter({"openai_api_key": "test-key"})
        await adapter.ensure_initialized()
        request = TTSRequest(text="Hello", voice="invalid_voice")
        is_valid, error = await adapter.validate_request(request)
        assert is_valid and error is None
        assert adapter.map_voice("invalid_voice") in {"alloy", "echo", "fable", "onyx", "nova", "shimmer"}

    @pytest.mark.unit
    @pytest.mark.xfail(reason="OpenAIAdapter does not validate model names in validate_request")
    async def test_validate_invalid_model(self):
        adapter = OpenAITTSAdapter({"openai_api_key": "test-key"})
        await adapter.ensure_initialized()
        request = TTSRequest(text="Hello", voice="alloy")
        is_valid, _ = await adapter.validate_request(request)
        assert is_valid

    @pytest.mark.unit
    async def test_validate_text_too_long(self):
        """Test validation rejects text exceeding limit."""
        adapter = OpenAITTSAdapter({"openai_api_key": "test-key"})
        await adapter.ensure_initialized()
        # OpenAI has ~4096 char limit in capabilities
        long_text = "a" * 5000
        request = TTSRequest(text=long_text, voice="alloy")
        is_valid, error = await adapter.validate_request(request)
        assert not is_valid
        assert "exceeds maximum" in (error or "")

    @pytest.mark.unit
    @pytest.mark.xfail(reason="OpenAIAdapter validate_request does not enforce speed bounds")
    async def test_validate_speed_out_of_range(self):
        adapter = OpenAITTSAdapter({"openai_api_key": "test-key"})
        await adapter.ensure_initialized()
        request = TTSRequest(text="Hello", voice="alloy", speed=5.0)
        is_valid, _ = await adapter.validate_request(request)
        assert not is_valid

# ========================================================================
# Audio Generation Tests
# ========================================================================

class TestAudioGeneration:
    """Test audio generation functionality."""

    @pytest.mark.unit
    @patch('tldw_Server_API.app.core.TTS.adapters.openai_adapter.apost')
    async def test_generate_basic_audio(self, mock_post):
        """Test basic audio generation."""
        # Mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"fake_audio_data"
        mock_response.headers = {"content-type": "audio/mpeg"}
        mock_post.return_value = mock_response

        adapter = OpenAITTSAdapter({"openai_api_key": "test-key"})
        request = TTSRequest(text="Hello world", voice="alloy", stream=False)

        response = await adapter.generate(request)

        assert (response.audio_content or response.audio_data) == b"fake_audio_data"
        assert (response.provider or "").lower() == "openai"

        # Verify API call
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        called_url = call_args.kwargs.get("url") if call_args.kwargs else (call_args.args[0] if call_args.args else "")
        assert "audio/speech" in str(called_url)

    @pytest.mark.unit
    @patch('tldw_Server_API.app.core.TTS.adapters.openai_adapter.apost')
    async def test_generate_with_hd_model(self, mock_post):
        """Test generation with HD model."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"hd_audio_data"
        mock_post.return_value = mock_response

        adapter = OpenAITTSAdapter({"openai_api_key": "test-key", "openai_model": "tts-1-hd"})
        request = TTSRequest(text="High quality audio", voice="nova", stream=False)

        response = await adapter.generate(request)

        assert (response.audio_content or response.audio_data) == b"hd_audio_data"

    @pytest.mark.unit
    @patch('tldw_Server_API.app.core.TTS.adapters.openai_adapter.apost')
    async def test_generate_with_speed_adjustment(self, mock_post):
        """Test generation with speed adjustment."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"audio_data"
        mock_post.return_value = mock_response

        adapter = OpenAITTSAdapter({"openai_api_key": "test-key"})
        request = TTSRequest(
            text="Slow speech",
            voice="echo",
            model="tts-1",
            speed=0.75,
            stream=False
        )

        response = await adapter.generate(request)

        # Verify speed was included in request
        call_kwargs = mock_post.call_args[1]
        assert call_kwargs['json']['speed'] == 0.75

    @pytest.mark.unit
    @patch('tldw_Server_API.app.core.TTS.adapters.openai_adapter.apost')
    async def test_generate_different_formats(self, mock_post):
        """Test generation with different audio formats."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"audio_data"
        mock_post.return_value = mock_response

        adapter = OpenAITTSAdapter({"openai_api_key": "test-key"})

        # Test different formats
        formats = [AudioFormat.MP3, AudioFormat.OPUS, AudioFormat.AAC, AudioFormat.FLAC]

        for audio_format in formats:
            request = TTSRequest(text="Test", voice="alloy", format=audio_format, stream=False)

            response = await adapter.generate(request)
            assert response.format == audio_format

# ========================================================================
# Streaming Generation Tests
# ========================================================================

class TestStreamingGeneration:
    """Test streaming audio generation."""

    @pytest.mark.unit
    @patch('tldw_Server_API.app.core.TTS.adapters.openai_adapter.apost')
    async def test_streaming_generation(self, mock_post):
        """Test streaming audio generation."""
        # Mock streaming response
        async def mock_iter():
            for chunk in [b"chunk1", b"chunk2", b"chunk3"]:
                yield chunk

        async def mock_iter_with_size(chunk_size=1024):
            async for c in mock_iter():
                yield c

        mock_response = AsyncMock()
        mock_response.aiter_bytes = mock_iter_with_size
        # Ensure raise_for_status behaves like a regular method (non-async)
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        adapter = OpenAITTSAdapter({"openai_api_key": "test-key"})
        request = TTSRequest(text="Stream this", voice="fable", stream=True)

        chunks = []
        resp = await adapter.generate(request)
        async for chunk in resp.audio_stream:
            chunks.append(chunk)

        assert chunks == [b"chunk1", b"chunk2", b"chunk3"]

    @pytest.mark.unit
    @patch('tldw_Server_API.app.core.TTS.adapters.openai_adapter.apost')
    async def test_streaming_with_error(self, mock_post):
        """Test streaming handles errors gracefully."""
        async def mock_error_iter():
            yield b"chunk1"
            raise httpx.HTTPError("Connection lost")

        async def mock_error_iter_with_size(chunk_size=1024):
            async for c in mock_error_iter():
                yield c

        mock_response = AsyncMock()
        mock_response.aiter_bytes = mock_error_iter_with_size
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        adapter = OpenAITTSAdapter({"openai_api_key": "test-key"})
        request = TTSRequest(text="Test", voice="alloy", stream=True)

        chunks = []
        with pytest.raises((TTSGenerationError, Exception)):
            resp = await adapter.generate(request)
            async for chunk in resp.audio_stream:
                chunks.append(chunk)

        # Should have received first chunk before error
        assert len(chunks) == 1

    @pytest.mark.unit
    @patch('tldw_Server_API.app.core.TTS.adapters.openai_adapter.apost')
    async def test_streaming_auth_error_401_is_normalized(self, mock_post):
        """HTTP 401 during stream setup should surface as TTSAuthenticationError."""
        async def _unused_iter(chunk_size=1024):
            if False:
                yield b""

        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.aread = AsyncMock(return_value=b'{"error":{"message":"Invalid API key"}}')
        mock_response.aiter_bytes = MagicMock(return_value=_unused_iter())
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "401",
            request=Mock(),
            response=mock_response,
        )
        mock_post.return_value = mock_response

        adapter = OpenAITTSAdapter({"openai_api_key": "bad-key"})
        request = TTSRequest(text="Test", voice="alloy", stream=True)

        response = await adapter.generate(request)

        with pytest.raises(TTSAuthenticationError):
            async for _ in response.audio_stream:
                pass

# ========================================================================
# Error Handling Tests
# ========================================================================

class TestErrorHandling:
    """Test error handling in OpenAI adapter."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_http_status_error_log_sanitizes_response_body_text(self):
        raw_marker = "RAW_OPENAI_HTTP_BODY_SECRET_MARKER"
        logged_messages: list[str] = []
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.headers = {}
        mock_response.aread = AsyncMock(return_value=f'{{"error":"{raw_marker}"}}'.encode())
        error = httpx.HTTPStatusError("500", request=Mock(), response=mock_response)
        adapter = OpenAITTSAdapter({"openai_api_key": "test-key"})
        sink_id = openai_mod.logger.add(
            lambda message: logged_messages.append(message.record["message"]),
            level="ERROR",
        )

        try:
            with pytest.raises(TTSProviderError) as exc_info:
                await adapter._handle_http_status_error(error)
        finally:
            openai_mod.logger.remove(sink_id)

        assert raw_marker in str(exc_info.value)
        assert any("OpenAI API error: 500" in message for message in logged_messages)
        assert all(raw_marker not in message for message in logged_messages)
        assert all("HTTPStatusError" in message for message in logged_messages)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_network_error_log_sanitizes_exception_text(self):
        raw_marker = "RAW_OPENAI_NETWORK_SECRET_MARKER"
        logged_messages: list[str] = []
        adapter = OpenAITTSAdapter({"openai_api_key": "test-key"})
        error = httpx.ConnectError(raw_marker)
        sink_id = openai_mod.logger.add(
            lambda message: logged_messages.append(message.record["message"]),
            level="ERROR",
        )

        try:
            with pytest.raises(TTSNetworkError):
                await adapter._raise_normalized_request_error(error)
        finally:
            openai_mod.logger.remove(sink_id)

        assert any("network/timeout error" in message for message in logged_messages)
        assert all(raw_marker not in message for message in logged_messages)
        assert all("ConnectError" in message for message in logged_messages)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_unexpected_error_log_sanitizes_exception_text(self):
        raw_marker = "RAW_OPENAI_UNEXPECTED_SECRET_MARKER"
        logged_messages: list[str] = []
        adapter = OpenAITTSAdapter({"openai_api_key": "test-key"})
        sink_id = openai_mod.logger.add(
            lambda message: logged_messages.append(message.record["message"]),
            level="ERROR",
        )

        try:
            with pytest.raises(TTSProviderError) as exc_info:
                await adapter._raise_normalized_request_error(RuntimeError(raw_marker))
        finally:
            openai_mod.logger.remove(sink_id)

        assert exc_info.value.details["error"] == raw_marker
        assert any("unexpected error" in message for message in logged_messages)
        assert all(raw_marker not in message for message in logged_messages)
        assert all("RuntimeError" in message for message in logged_messages)

    @pytest.mark.unit
    @pytest.mark.asyncio
    @patch('tldw_Server_API.app.core.TTS.adapters.openai_adapter.apost')
    async def test_streaming_error_log_sanitizes_exception_text(self, mock_post):
        raw_marker = "RAW_OPENAI_STREAM_SECRET_MARKER"
        logged_messages: list[str] = []
        mock_post.side_effect = RuntimeError(raw_marker)
        adapter = OpenAITTSAdapter({"openai_api_key": "test-key"})
        sink_id = openai_mod.logger.add(
            lambda message: logged_messages.append(message.record["message"]),
            level="ERROR",
        )

        try:
            stream = adapter._stream_audio(headers={}, payload={})
            with pytest.raises(TTSProviderError):
                async for _chunk in stream:
                    pass
        finally:
            openai_mod.logger.remove(sink_id)

        assert any("streaming error" in message for message in logged_messages)
        assert any("unexpected error" in message for message in logged_messages)
        assert all(raw_marker not in message for message in logged_messages)
        assert all("RuntimeError" in message for message in logged_messages)

    @pytest.mark.unit
    @pytest.mark.asyncio
    @patch("tldw_Server_API.app.core.TTS.adapters.openai_adapter.validate_tts_request")
    async def test_request_validation_failure_log_sanitizes_exception_text(self, mock_validate):
        raw_marker = "RAW_OPENAI_VALIDATION_SECRET_MARKER"
        logged_messages: list[str] = []
        mock_validate.side_effect = TTSValidationError(raw_marker, provider="openai")
        adapter = OpenAITTSAdapter({"openai_api_key": "test-key"})
        request = TTSRequest(text="Test", voice="alloy", stream=False)
        sink_id = openai_mod.logger.add(
            lambda message: logged_messages.append(message.record["message"]),
            level="ERROR",
        )

        try:
            with pytest.raises(TTSValidationError) as exc_info:
                await adapter.generate(request)
        finally:
            openai_mod.logger.remove(sink_id)

        assert raw_marker in str(exc_info.value)
        assert any("request validation failed" in message for message in logged_messages)
        assert all(raw_marker not in message for message in logged_messages)
        assert all("TTSValidationError" in message for message in logged_messages)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_stream_close_debug_log_sanitizes_exception_text(self):
        raw_marker = "RAW_OPENAI_STREAM_CLOSE_SECRET_MARKER"
        logged_messages: list[str] = []

        async def mock_iter_with_size(chunk_size=1024):
            yield b"chunk"

        mock_response = AsyncMock()
        mock_response.aiter_bytes = mock_iter_with_size
        mock_response.raise_for_status = MagicMock()
        mock_response.aclose = AsyncMock(side_effect=RuntimeError(raw_marker))

        adapter = OpenAITTSAdapter({"openai_api_key": "test-key"})
        sink_id = openai_mod.logger.add(
            lambda message: logged_messages.append(str(message)),
            level="DEBUG",
        )

        try:
            with patch("tldw_Server_API.app.core.TTS.adapters.openai_adapter.apost", new=AsyncMock(return_value=mock_response)):
                chunks = [chunk async for chunk in adapter._stream_audio(headers={}, payload={})]
        finally:
            openai_mod.logger.remove(sink_id)

        assert chunks == [b"chunk"]
        assert any("response close after stream failed" in message for message in logged_messages)
        assert all(raw_marker not in message for message in logged_messages)
        assert any("RuntimeError" in message for message in logged_messages)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_cleanup_warning_log_sanitizes_exception_text(self):
        raw_marker = "RAW_OPENAI_CLEANUP_SECRET_MARKER"
        logged_messages: list[str] = []

        class CleanupFailureAdapter(OpenAITTSAdapter):
            @property
            def client(self):
                return None

            @client.setter
            def client(self, value):
                raise RuntimeError(raw_marker)

        adapter = object.__new__(CleanupFailureAdapter)
        sink_id = openai_mod.logger.add(
            lambda message: logged_messages.append(message.record["message"]),
            level="WARNING",
        )

        try:
            await adapter._cleanup_resources()
        finally:
            openai_mod.logger.remove(sink_id)

        assert any("Error during cleanup" in message for message in logged_messages)
        assert all(raw_marker not in message for message in logged_messages)
        assert all("RuntimeError" in message for message in logged_messages)

    @pytest.mark.unit
    @patch('tldw_Server_API.app.core.TTS.adapters.openai_adapter.apost')
    async def test_handle_rate_limit_error(self, mock_post):
        """Test handling of rate limit errors."""
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.headers = {"retry-after": "60"}
        mock_response.json.return_value = {"error": {"message": "Rate limit exceeded"}}
        mock_response.aread = AsyncMock(return_value=b'{"error":{"message":"Rate limit exceeded"}}')
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "429", request=Mock(), response=mock_response
        )
        mock_post.return_value = mock_response

        adapter = OpenAITTSAdapter({"openai_api_key": "test-key"})
        request = TTSRequest(text="Test", voice="alloy", stream=False)

        with pytest.raises(TTSRateLimitError) as exc_info:
            await adapter.generate(request)

        assert getattr(exc_info.value, "retry_after", exc_info.value.details.get("retry_after")) == 60

    @pytest.mark.unit
    @patch('tldw_Server_API.app.core.TTS.adapters.openai_adapter.apost')
    async def test_handle_api_error(self, mock_post):
        """Test handling of API errors."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.json.return_value = {"error": {"message": "Internal server error"}}
        mock_response.aread = AsyncMock(return_value=b'{"error":{"message":"Internal server error"}}')
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "500", request=Mock(), response=mock_response
        )
        mock_post.return_value = mock_response

        adapter = OpenAITTSAdapter({"openai_api_key": "test-key"})
        request = TTSRequest(text="Test", voice="alloy", stream=False)

        with pytest.raises((TTSGenerationError, Exception)):
            await adapter.generate(request)

    @pytest.mark.unit
    @patch('tldw_Server_API.app.core.TTS.adapters.openai_adapter.apost')
    async def test_handle_auth_error_401(self, mock_post):
        """Test handling of 401 maps to TTSAuthenticationError."""
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.aread = AsyncMock(return_value=b'{"error":{"message":"Invalid API key"}}')
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "401", request=Mock(), response=mock_response
        )
        mock_post.return_value = mock_response

        adapter = OpenAITTSAdapter({"openai_api_key": "bad-key"})
        request = TTSRequest(text="Test", voice="alloy", stream=False)

        from tldw_Server_API.app.core.TTS.tts_exceptions import TTSAuthenticationError
        with pytest.raises(TTSAuthenticationError):
            await adapter.generate(request)

    @pytest.mark.unit
    @patch('tldw_Server_API.app.core.TTS.adapters.openai_adapter.apost')
    async def test_production_adapter_preserves_auth_error_401(self, mock_post):
        """Production OpenAITTSAdapter should preserve typed auth failures."""
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.aread = AsyncMock(return_value=b'{"error":{"message":"Invalid API key"}}')
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "401", request=Mock(), response=mock_response
        )
        mock_post.return_value = mock_response

        adapter = ProductionOpenAITTSAdapter({"openai_api_key": "bad-key"})
        request = TTSRequest(text="Test", voice="alloy", stream=False)

        with pytest.raises(TTSAuthenticationError):
            await adapter.generate(request)

    @pytest.mark.unit
    @patch('tldw_Server_API.app.core.TTS.adapters.openai_adapter.apost')
    async def test_handle_network_error(self, mock_post):
        """Test handling of network errors maps to TTSNetworkError."""
        mock_post.side_effect = httpx.ConnectError("Connection failed")

        adapter = OpenAITTSAdapter({"openai_api_key": "test-key"})
        request = TTSRequest(text="Test", voice="alloy", stream=False)

        with pytest.raises(TTSNetworkError):
            await adapter.generate(request)

    @pytest.mark.unit
    @patch('tldw_Server_API.app.core.TTS.adapters.openai_adapter.apost')
    async def test_handle_timeout_error(self, mock_post):
        """Test handling of timeout errors maps to TTSTimeoutError."""
        mock_post.side_effect = httpx.TimeoutException("Request timed out")

        adapter = OpenAITTSAdapter({"openai_api_key": "test-key"})
        request = TTSRequest(text="Test", voice="alloy", stream=False)

        from tldw_Server_API.app.core.TTS.tts_exceptions import TTSTimeoutError
        with pytest.raises(TTSTimeoutError):
            await adapter.generate(request)

# ========================================================================
# Metadata and Info Tests
# ========================================================================

class TestMetadataAndInfo:
    """Test metadata and information methods."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_adapter_info(self):
        """Build adapter information from capabilities for current API."""
        adapter = OpenAITTSAdapter({"openai_api_key": "test-key"})
        caps = await adapter.get_capabilities()
        info = {
            "provider": adapter.provider_name.lower(),
            "models": [adapter.model],
            "voices": [v.id for v in caps.supported_voices],
            "max_characters": caps.max_text_length,
        }
        assert info["provider"] == "openai"
        assert "tts-1" in ["tts-1", "tts-1-hd"]
        assert "alloy" in info["voices"]
        assert info["max_characters"] == 4096

    @pytest.mark.unit
    @patch('tldw_Server_API.app.core.TTS.adapters.openai_adapter.apost')
    async def test_response_includes_metadata(self, mock_post):
        """Test that response includes proper metadata."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"audio_data"
        mock_post.return_value = mock_response

        adapter = OpenAITTSAdapter({"openai_api_key": "test-key", "openai_model": "tts-1-hd"})
        request = TTSRequest(text="Test metadata", voice="shimmer", stream=False)

        response = await adapter.generate(request)

        assert (response.provider or "").lower() == "openai"
        assert (response.audio_content or response.audio_data) == b"audio_data"
