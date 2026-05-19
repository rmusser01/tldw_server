"""
Unit and integration tests for external transcription provider support.
"""

import pytest
import numpy as np
from unittest.mock import patch, AsyncMock
from pathlib import Path
import tempfile
from tldw_Server_API.app.core.exceptions import NetworkError

pytestmark = pytest.mark.unit


class _DummyResp:
    def __init__(self, status_code: int, text: str = "", json_data: dict | None = None, headers: dict | None = None):
        self.status_code = status_code
        self.text = text
        self._json_data = json_data or {}
        self.headers = headers or {}

    def json(self):

        return self._json_data

    async def aclose(self):
        return None


class TestExternalProvider:
    """Test suite for external transcription provider."""

    @pytest.fixture
    def sample_audio(self):
        """Generate sample audio data."""
        sample_rate = 16000
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        audio = np.sin(440 * 2 * np.pi * t).astype(np.float32)
        return audio, sample_rate

    @pytest.fixture
    def provider_config(self):
        """Create test provider configuration."""
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_External_Provider import (
            ExternalProviderConfig
        )

        return ExternalProviderConfig(
            base_url="https://api.example.com/v1/audio/transcriptions",
            api_key="test-api-key",
            model="whisper-1",
            timeout=30.0,
            max_retries=2,
            verify_ssl=True,
            response_format="json"
        )

    def test_import_module(self):

        """Test module imports."""
        try:
            from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_External_Provider import (
                transcribe_with_external_provider,
                ExternalProviderConfig,
                validate_external_provider_config,
                add_external_provider,
                list_external_providers
            )
            assert transcribe_with_external_provider is not None
            assert ExternalProviderConfig is not None
        except ImportError as e:
            pytest.skip(f"External provider module not available: {e}")

    def test_config_validation(self, provider_config):

        """Test configuration validation."""
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_External_Provider import (
            validate_external_provider_config,
            ExternalProviderConfig
        )

        # Valid config
        is_valid, error = validate_external_provider_config(provider_config)
        assert is_valid == True
        assert error is None

        # Invalid URL
        invalid_config = ExternalProviderConfig(
            base_url="not-a-url",
            api_key="key"
        )
        is_valid, error = validate_external_provider_config(invalid_config)
        assert is_valid == False
        assert "Invalid base URL" in error

        # Invalid timeout
        invalid_config = ExternalProviderConfig(
            base_url="https://api.example.com",
            timeout=-1
        )
        is_valid, error = validate_external_provider_config(invalid_config)
        assert is_valid == False
        assert "Timeout must be positive" in error

        # Invalid temperature
        invalid_config = ExternalProviderConfig(
            base_url="https://api.example.com",
            temperature=3.0
        )
        is_valid, error = validate_external_provider_config(invalid_config)
        assert is_valid == False
        assert "Temperature must be between" in error

        # Invalid response format
        invalid_config = ExternalProviderConfig(
            base_url="https://api.example.com",
            response_format="invalid"
        )
        is_valid, error = validate_external_provider_config(invalid_config)
        assert is_valid == False
        assert "Invalid response format" in error

    def test_add_and_list_providers(self, provider_config):

        """Test adding and listing providers."""
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_External_Provider import (
            add_external_provider,
            list_external_providers,
            remove_external_provider,
            _provider_configs
        )

        # Clear existing providers
        _provider_configs.clear()

        # Add provider
        success = add_external_provider("test_provider", provider_config)
        assert success == True

        # List providers
        providers = list_external_providers()
        assert "test_provider" in providers

        # Remove provider
        removed = remove_external_provider("test_provider")
        assert removed == True

        # Verify removed
        providers = list_external_providers()
        assert "test_provider" not in providers

    @pytest.mark.asyncio
    async def test_transcribe_with_mock_api(self, sample_audio, provider_config):
        """Test transcription with mocked API."""
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_External_Provider import (
            transcribe_with_external_provider_async
        )

        audio_data, sample_rate = sample_audio

        # Mock HTTP response
        mock_response = _DummyResp(
            200,
            json_data={
                'text': 'Mocked transcription result',
                'task': 'transcribe',
                'language': 'en'
            },
        )

        with patch(
            'tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_External_Provider.afetch',
            new=AsyncMock(return_value=mock_response),
        ) as mock_afetch:
            result = await transcribe_with_external_provider_async(
                audio_data,
                sample_rate,
                config=provider_config
            )

            assert result == 'Mocked transcription result'
            assert mock_afetch.call_count == 1

            # Verify request parameters
            call_kwargs = mock_afetch.call_args.kwargs
            assert provider_config.base_url in call_kwargs.get("url", "")
            assert 'headers' in call_kwargs
            assert 'Authorization' in call_kwargs['headers']

    @pytest.mark.asyncio
    async def test_retry_on_rate_limit(self, sample_audio, provider_config):
        """Test retry logic on rate limiting."""
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_External_Provider import (
            transcribe_with_external_provider_async
        )

        audio_data, sample_rate = sample_audio
        provider_config.max_retries = 3

        async def _fake_afetch(**kwargs):
            retry = kwargs.get("retry")
            assert retry is not None
            assert retry.attempts == provider_config.max_retries
            assert 429 in retry.retry_on_status
            assert retry.retry_on_unsafe is True
            return _DummyResp(200, json_data={'text': 'Success after retry'})

        with patch(
            'tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_External_Provider.afetch',
            new=AsyncMock(side_effect=_fake_afetch),
        ) as mock_afetch:
            result = await transcribe_with_external_provider_async(
                audio_data,
                sample_rate,
                config=provider_config
            )

        assert result == 'Success after retry'
        assert mock_afetch.call_count == 1

    @pytest.mark.asyncio
    async def test_timeout_handling(self, sample_audio, provider_config):
        """Test timeout handling."""
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_External_Provider import (
            transcribe_with_external_provider_async
        )

        audio_data, sample_rate = sample_audio
        provider_config.max_retries = 2
        provider_config.timeout = 0.1  # Very short timeout

        with patch(
            'tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_External_Provider.afetch',
            new=AsyncMock(
                side_effect=NetworkError("Request timeout at /private/provider/audio.wav")
            ),
        ) as mock_afetch:
            result = await transcribe_with_external_provider_async(
                audio_data,
                sample_rate,
                config=provider_config
            )

        assert result == "[Error: External transcription request failed]"
        assert "Request timeout" not in result
        assert "/private/provider/audio.wav" not in result
        assert mock_afetch.call_count == 1

    @pytest.mark.asyncio
    async def test_invalid_config_error_is_sanitized(self, sample_audio, provider_config):
        """Invalid provider config should not expose validation internals in transcripts."""
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_External_Provider import (
            transcribe_with_external_provider_async
        )

        audio_data, sample_rate = sample_audio
        provider_config.base_url = "/private/provider/config.txt"

        result = await transcribe_with_external_provider_async(
            audio_data,
            sample_rate,
            config=provider_config,
        )

        assert result == "[Error: Invalid external provider configuration]"
        assert "Invalid base URL" not in result
        assert "/private/provider/config.txt" not in result

    @pytest.mark.asyncio
    async def test_api_error_response_body_is_sanitized(self, sample_audio, provider_config):
        """Provider error bodies should not be copied into transcript output."""
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_External_Provider import (
            transcribe_with_external_provider_async
        )

        audio_data, sample_rate = sample_audio
        mock_response = _DummyResp(
            500,
            text="backend exploded at /private/provider/trace.log",
            json_data={
                "error": {
                    "message": "backend exploded at /private/provider/trace.log",
                }
            },
        )

        with patch(
            'tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_External_Provider.afetch',
            new=AsyncMock(return_value=mock_response),
        ):
            result = await transcribe_with_external_provider_async(
                audio_data,
                sample_rate,
                config=provider_config,
            )

        assert result == "[Error: API returned 500]"
        assert "backend exploded" not in result
        assert "/private/provider/trace.log" not in result

    @pytest.mark.asyncio
    async def test_async_runtime_error_sanitizes_internal_message(
        self,
        sample_audio,
        provider_config,
        monkeypatch,
    ):
        """Unexpected async processing failures should not leak backend details."""
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio import (
            Audio_Transcription_External_Provider as external_provider,
        )

        audio_data, sample_rate = sample_audio

        def _fail_write(*_args, **_kwargs):
            raise RuntimeError("audio encoder failed at /private/provider/encode.wav")

        monkeypatch.setattr(external_provider.sf, "write", _fail_write)

        result = await external_provider.transcribe_with_external_provider_async(
            audio_data,
            sample_rate,
            config=provider_config,
        )

        assert result == "[Error: External transcription failed]"
        assert "audio encoder failed" not in result
        assert "/private/provider/encode.wav" not in result

    @pytest.mark.asyncio
    async def test_different_response_formats(self, sample_audio, provider_config):
        """Test handling different response formats."""
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_External_Provider import (
            transcribe_with_external_provider_async
        )

        audio_data, sample_rate = sample_audio

        # Test JSON format
        provider_config.response_format = 'json'
        mock_response = _DummyResp(200, json_data={'text': 'JSON response'})

        with patch(
            'tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_External_Provider.afetch',
            new=AsyncMock(return_value=mock_response),
        ):
            result = await transcribe_with_external_provider_async(
                audio_data,
                sample_rate,
                config=provider_config
            )

        assert result == 'JSON response'

        # Test text format
        provider_config.response_format = 'text'
        mock_response = _DummyResp(200, text='Plain text response')

        with patch(
            'tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_External_Provider.afetch',
            new=AsyncMock(return_value=mock_response),
        ):
            result = await transcribe_with_external_provider_async(
                audio_data,
                sample_rate,
                config=provider_config
            )

        assert result == 'Plain text response'

        # Test SRT format
        provider_config.response_format = 'srt'
        mock_response = _DummyResp(200, text='1\n00:00:00,000 --> 00:00:01,000\nSRT subtitle')

        with patch(
            'tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_External_Provider.afetch',
            new=AsyncMock(return_value=mock_response),
        ):
            result = await transcribe_with_external_provider_async(
                audio_data,
                sample_rate,
                config=provider_config
            )

        assert 'SRT subtitle' in result

    def test_synchronous_wrapper(self, sample_audio, provider_config):

        """Test synchronous wrapper function."""
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_External_Provider import (
            transcribe_with_external_provider
        )

        audio_data, sample_rate = sample_audio

        mock_response = _DummyResp(200, json_data={'text': 'Sync wrapper result'})

        with patch(
            'tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_External_Provider.afetch',
            new=AsyncMock(return_value=mock_response),
        ):
            result = transcribe_with_external_provider(
                audio_data,
                sample_rate,
                config=provider_config
            )

        assert result == 'Sync wrapper result'

    def test_synchronous_wrapper_sanitizes_runtime_errors(
        self,
        sample_audio,
        provider_config,
        monkeypatch,
    ):
        """Synchronous wrapper failures should not leak internal exception text."""
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio import (
            Audio_Transcription_External_Provider as external_provider,
        )

        audio_data, sample_rate = sample_audio

        def _fail_async(*_args, **_kwargs):
            raise RuntimeError("event loop failed at /private/provider/loop")

        monkeypatch.setattr(
            external_provider,
            "transcribe_with_external_provider_async",
            _fail_async,
        )

        result = external_provider.transcribe_with_external_provider(
            audio_data,
            sample_rate,
            config=provider_config,
        )

        assert result == "[Error: External transcription failed]"
        assert "event loop failed" not in result
        assert "/private/provider/loop" not in result

    @pytest.mark.asyncio
    async def test_with_file_path(self, provider_config):
        """Test transcription with file path input."""
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_External_Provider import (
            transcribe_with_external_provider_async
        )

        # Create temporary audio file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            sample_rate = 16000
            audio = np.random.randn(sample_rate).astype(np.float32)
            import soundfile as sf
            sf.write(tmp_file.name, audio, sample_rate)
            tmp_path = tmp_file.name

        try:
            mock_response = _DummyResp(200, json_data={'text': 'File path result'})

            with patch(
                'tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_External_Provider.afetch',
                new=AsyncMock(return_value=mock_response),
            ):
                result = await transcribe_with_external_provider_async(
                    tmp_path,
                    config=provider_config
                )

            assert result == 'File path result'

        finally:
            # Clean up
            import os
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    @pytest.mark.asyncio
    async def test_provider_test_function(self, provider_config):
        """Test the provider test function."""
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_External_Provider import (
            test_external_provider
        )

        mock_response = _DummyResp(200, json_data={'text': 'Test successful'})

        with patch(
            'tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_External_Provider.afetch',
            new=AsyncMock(return_value=mock_response),
        ):
            result = await test_external_provider(config=provider_config)

        assert result['success'] == True
        assert result['result'] == 'Test successful'
        assert 'elapsed_time' in result
        assert result['elapsed_time'] >= 0

    @pytest.mark.asyncio
    async def test_provider_test_function_sanitizes_runtime_errors(
        self,
        provider_config,
        monkeypatch,
    ):
        """Provider self-test failures should not expose backend details."""
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio import (
            Audio_Transcription_External_Provider as external_provider,
        )

        async def fail_transcription(*_args, **_kwargs):
            raise RuntimeError("provider self-test failed at /private/provider/test.wav")

        monkeypatch.setattr(
            external_provider,
            "transcribe_with_external_provider_async",
            fail_transcription,
        )

        result = await external_provider.test_external_provider(config=provider_config)

        assert result["success"] is False
        assert result["error"] == "External provider test failed"
        assert "provider self-test failed" not in result["error"]
        assert "/private/provider/test.wav" not in result["error"]


@pytest.mark.integration
class TestExternalProviderIntegration:
    """Integration tests for external provider."""

    def test_integration_with_main_library(self):

        """Test integration with main transcription library."""
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_External_Provider import (
            register_external_provider_with_library
        )

        # This should not raise any errors
        register_external_provider_with_library()

    @pytest.mark.asyncio
    async def test_load_from_environment(self):
        """Test loading configuration from environment variables."""
        import os

        # Set environment variables
        os.environ['EXTERNAL_TRANSCRIPTION_TEST_BASE_URL'] = 'https://test.api.com'
        os.environ['EXTERNAL_TRANSCRIPTION_TEST_API_KEY'] = 'test-key'
        os.environ['EXTERNAL_TRANSCRIPTION_TEST_MODEL'] = 'test-model'

        try:
            from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_External_Provider import (
                load_external_provider_config
            )

            config = load_external_provider_config('test')

            if config:  # May be None if config loading not implemented
                assert config.base_url == 'https://test.api.com'
                assert config.api_key == 'test-key'
                assert config.model == 'test-model'

        finally:
            # Clean up environment
            for key in ['BASE_URL', 'API_KEY', 'MODEL']:
                env_key = f'EXTERNAL_TRANSCRIPTION_TEST_{key}'
                if env_key in os.environ:
                    del os.environ[env_key]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
