# test_tts_exceptions.py
# Description: Comprehensive tests for TTS exception hierarchy
#
# Imports

import pytest

#
# Local Imports
from tldw_Server_API.app.core.TTS.tts_exceptions import (
    TTSAuthenticationError,
    TTSError,
    TTSGenerationError,
    TTSGPUError,
    TTSInsufficientMemoryError,
    TTSModelLoadError,
    TTSModelNotFoundError,
    TTSNetworkError,
    TTSProviderError,
    TTSProviderInitializationError,
    TTSProviderNotConfiguredError,
    TTSProviderUnavailableError,
    TTSQuotaExceededError,
    TTSRateLimitError,
    TTSResourceError,
    TTSTextTooLongError,
    TTSTimeoutError,
    TTSUnsupportedFormatError,
    TTSValidationError,
    TTSVoiceNotFoundError,
    auth_error,
    categorize_error,
    get_http_status_for_error,
    is_retryable_error,
    network_error,
    rate_limit_error,
    resource_error,
    timeout_error,
    validation_error,
)

#
#######################################################################################################################
#
# Test Exception Hierarchy


class TestTTSExceptionHierarchy:
    """Test the TTS exception hierarchy structure"""

    def test_base_exception_initialization(self):
        """Test base TTSError initialization"""
        error = TTSError(
            message="Test error", provider="test_provider", error_code="TEST_001", details={"key": "value"}
        )

        assert str(error) == "Test error"
        assert error.provider == "test_provider"
        assert error.error_code == "TEST_001"
        assert error.details == {"key": "value"}
        assert error.timestamp is not None

    def test_exception_inheritance(self):
        """Test that all exceptions inherit from TTSError"""
        exceptions = [
            TTSProviderNotConfiguredError("test"),
            TTSProviderInitializationError("test"),
            TTSModelNotFoundError("test"),
            TTSGenerationError("test"),
            TTSValidationError("test"),
            TTSAuthenticationError("test"),
            TTSRateLimitError("test"),
            TTSNetworkError("test"),
            TTSResourceError("test"),
        ]

        for exc in exceptions:
            assert isinstance(exc, TTSError)
            assert isinstance(exc, Exception)

    def test_validation_exception_subtypes(self):
        """Test validation exception subtypes"""
        text_error = TTSTextTooLongError(
            "Text too long", provider="test", details={"max_length": 1000, "actual_length": 2000}
        )
        assert isinstance(text_error, TTSValidationError)
        assert text_error.details["max_length"] == 1000
        assert text_error.details["actual_length"] == 2000

        voice_error = TTSVoiceNotFoundError(
            "Voice not found", provider="test", details={"requested_voice": "test", "available_voices": ["v1", "v2"]}
        )
        assert isinstance(voice_error, TTSValidationError)
        assert voice_error.details["requested_voice"] == "test"
        assert voice_error.details["available_voices"] == ["v1", "v2"]

        format_error = TTSUnsupportedFormatError(
            "Unsupported format",
            provider="test",
            details={"requested_format": "test", "supported_formats": ["mp3", "wav"]},
        )
        assert isinstance(format_error, TTSValidationError)
        assert format_error.details["requested_format"] == "test"
        assert format_error.details["supported_formats"] == ["mp3", "wav"]

    def test_resource_exception_subtypes(self):
        """Test resource exception subtypes"""
        memory_error = TTSInsufficientMemoryError(
            "Out of memory", provider="test", details={"required_mb": 4096, "available_mb": 2048}
        )
        assert isinstance(memory_error, TTSResourceError)
        assert memory_error.details["required_mb"] == 4096
        assert memory_error.details["available_mb"] == 2048

        gpu_error = TTSGPUError("GPU error", provider="test", details={"cuda_available": False, "gpu_memory_mb": 0})
        assert isinstance(gpu_error, TTSResourceError)
        assert gpu_error.details is not None


class TestErrorCategorization:
    """Test error categorization functions"""

    def test_categorize_error(self):
        """Test error categorization"""
        # Configuration errors
        assert categorize_error(TTSProviderNotConfiguredError("test")) == "configuration"
        assert categorize_error(TTSProviderInitializationError("test")) == "configuration"

        # Model errors
        assert categorize_error(TTSModelNotFoundError("test")) == "model"
        assert categorize_error(TTSModelLoadError("test")) == "model"

        # Validation errors
        assert categorize_error(TTSValidationError("test")) == "validation"
        assert categorize_error(TTSTextTooLongError("test")) == "validation"

        # API errors
        assert categorize_error(TTSAuthenticationError("test")) == "authentication"
        assert categorize_error(TTSRateLimitError("test")) == "rate_limit"
        assert categorize_error(TTSQuotaExceededError("test")) == "quota"
        assert (
            categorize_error(TTSProviderUnavailableError("test"))
            == "provider_unavailable"
        )

        # Network errors
        assert categorize_error(TTSNetworkError("test")) == "network"
        assert categorize_error(TTSTimeoutError("test")) == "timeout"

        # Resource errors
        assert categorize_error(TTSResourceError("test")) == "resource"
        assert categorize_error(TTSInsufficientMemoryError("test")) == "resource"

        # Unknown errors
        assert categorize_error(Exception("test")) == "unknown"
        assert categorize_error(ValueError("test")) == "unknown"

    def test_is_retryable_error(self):
        """Test retryable error detection"""
        # Retryable errors
        assert is_retryable_error(TTSNetworkError("test")) is True
        assert is_retryable_error(TTSTimeoutError("test")) is True
        assert is_retryable_error(TTSRateLimitError("test")) is True
        assert is_retryable_error(TTSProviderError("test")) is True
        assert is_retryable_error(TTSResourceError("test")) is True

        # Non-retryable errors
        assert is_retryable_error(TTSValidationError("test")) is False
        assert is_retryable_error(TTSAuthenticationError("test")) is False
        assert is_retryable_error(TTSProviderNotConfiguredError("test")) is False
        assert is_retryable_error(TTSModelNotFoundError("test")) is False
        assert is_retryable_error(ValueError("test")) is False

    def test_get_http_status_for_error(self):
        """Test HTTP status code mapping"""
        # 400 Bad Request
        assert get_http_status_for_error(TTSValidationError("test")) == 400
        assert get_http_status_for_error(TTSTextTooLongError("test")) == 400
        assert get_http_status_for_error(TTSVoiceNotFoundError("test")) == 400

        # 401 Unauthorized
        assert get_http_status_for_error(TTSAuthenticationError("test")) == 401

        # 404 Not Found
        assert get_http_status_for_error(TTSModelNotFoundError("test")) == 404

        # 429 Too Many Requests
        assert get_http_status_for_error(TTSRateLimitError("test")) == 429

        # 500 Internal Server Error
        assert get_http_status_for_error(TTSProviderInitializationError("test")) == 500
        assert get_http_status_for_error(TTSModelLoadError("test")) == 500
        assert get_http_status_for_error(TTSGenerationError("test")) == 500

        # 502 Bad Gateway
        assert get_http_status_for_error(TTSProviderError("test")) == 502

        # 503 Service Unavailable
        assert get_http_status_for_error(TTSProviderNotConfiguredError("test")) == 503
        assert get_http_status_for_error(TTSResourceError("test")) == 503
        assert get_http_status_for_error(TTSInsufficientMemoryError("test")) == 503

        # 504 Gateway Timeout
        assert get_http_status_for_error(TTSTimeoutError("test")) == 504
        assert get_http_status_for_error(TTSNetworkError("test")) == 504

        # Default 500 for unknown
        assert get_http_status_for_error(Exception("test")) == 500


class TestConvenienceFunctions:
    """Test convenience error creation functions"""

    def test_auth_error(self):
        """Test auth_error convenience function"""
        error = auth_error("test_provider", "Invalid API key")

        assert isinstance(error, TTSAuthenticationError)
        assert error.provider == "test_provider"
        assert "Invalid API key" in str(error)
        assert error.details["suggestion"] == "Check your API key configuration"

    def test_rate_limit_error(self):
        """Test rate_limit_error convenience function"""
        error = rate_limit_error("test_provider", retry_after=60)

        assert isinstance(error, TTSRateLimitError)
        assert error.provider == "test_provider"
        assert error.details["retry_after"] == 60
        assert "Rate limit exceeded" in str(error)

    def test_network_error(self):
        """Test network_error convenience function"""
        sentinel_url = "https://user:secret@tts-helper.invalid/path?token=private"
        original_error = ConnectionError(sentinel_url)
        error = network_error(
            "test_provider",
            original_error,
            details={"original_error": sentinel_url},
        )

        assert isinstance(error, TTSNetworkError)
        assert error.provider == "test_provider"
        assert str(error) == "Network request failed for test_provider"
        assert error.details == {"error_type": "ConnectionError"}
        assert sentinel_url not in repr(error)
        assert sentinel_url not in repr(error.details)

    def test_timeout_error(self):
        """Test timeout_error convenience function"""
        error = timeout_error("test_provider", timeout_seconds=30)

        assert isinstance(error, TTSTimeoutError)
        assert error.provider == "test_provider"
        assert error.details["timeout_seconds"] == 30

    def test_validation_error(self):
        """Test validation_error convenience function"""
        error = validation_error(
            field="text", value="test", constraint="max_length", details={"max": 100, "actual": 200}
        )

        assert isinstance(error, TTSValidationError)
        assert error.details["field"] == "text"
        assert error.details["constraint"] == "max_length"
        assert error.details["max"] == 100

    def test_resource_error(self):
        """Test resource_error convenience function"""
        error = resource_error("test_provider", resource_type="memory", required=4096, available=2048)

        assert isinstance(error, TTSResourceError)
        assert error.provider == "test_provider"
        assert error.details["resource_type"] == "memory"
        assert error.details["required"] == 4096
        assert error.details["available"] == 2048


class TestErrorHandlingIntegration:
    """Test error handling in integrated scenarios"""

    def test_error_chain_preservation(self):
        """Test that error chains preserve information"""
        original = ValueError("Original error")

        try:
            raise TTSModelLoadError(
                "Failed to load model", provider="test", details={"original_error": str(original)}
            ) from original
        except TTSModelLoadError as e:
            assert e.__cause__ == original
            assert "Original error" in e.details["original_error"]

    def test_error_context_aggregation(self):
        """Test aggregating context across error handling"""
        error = TTSGenerationError("Generation failed", provider="test_provider", details={"step": "synthesis"})

        # Add more context
        error.details["model"] = "test_model"
        error.details["attempt"] = 1

        assert error.details["step"] == "synthesis"
        assert error.details["model"] == "test_model"
        assert error.details["attempt"] == 1

    def test_error_recovery_metadata(self):
        """Test that errors include recovery metadata"""
        error = TTSRateLimitError(
            "Rate limit hit", provider="test", details={"retry_after": 60, "limit": 100, "window": "1m"}
        )

        assert is_retryable_error(error)
        assert error.details["retry_after"] == 60
        assert get_http_status_for_error(error) == 429


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
