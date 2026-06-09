# tts_exceptions.py
# Description: Custom exception hierarchy for TTS module
#
# Imports
from datetime import datetime
from typing import Any, Optional

#
#######################################################################################################################
#
# TTS Exception Hierarchy

class TTSError(Exception):
    """Base exception for all TTS-related errors"""

    def __init__(
        self,
        message: str,
        provider: Optional[str] = None,
        error_code: Optional[str] = None,
        details: Optional[dict[str, Any]] = None
    ):
        """
        Initialize TTS error.

        Args:
            message: Human-readable error message
            provider: Name of the TTS provider that failed
            error_code: Machine-readable error code
            details: Additional error details
        """
        super().__init__(message)
        self.message = message
        self.provider = provider
        self.error_code = error_code
        self.details = details or {}
        self.timestamp = datetime.utcnow()

    def to_dict(self) -> dict[str, Any]:
        """Convert exception to dictionary for API responses"""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "provider": self.provider,
            "error_code": self.error_code,
            "details": self.details
        }


# Configuration and Initialization Errors
class TTSConfigurationError(TTSError):
    """Error in TTS configuration or setup"""
    pass


class TTSProviderNotConfiguredError(TTSConfigurationError):
    """TTS provider is not properly configured"""
    pass


class TTSProviderInitializationError(TTSConfigurationError):
    """Error initializing TTS provider"""
    pass


class TTSModelNotFoundError(TTSConfigurationError):
    """Requested TTS model not found"""
    pass


# Request and Validation Errors
class TTSValidationError(TTSError):
    """Error validating TTS request"""
    pass


class TTSInvalidInputError(TTSValidationError):
    """Invalid input text for TTS generation"""
    pass


class TTSUnsupportedFormatError(TTSValidationError):
    """Requested audio format not supported"""
    pass


class TTSUnsupportedLanguageError(TTSValidationError):
    """Requested language not supported by provider"""
    pass


class TTSVoiceNotFoundError(TTSValidationError):
    """Requested voice not found"""
    pass


class TTSTextTooLongError(TTSValidationError):
    """Input text exceeds maximum length limit"""
    pass


class TTSInvalidVoiceReferenceError(TTSValidationError):
    """Invalid voice reference audio for voice cloning"""
    pass


# Provider and API Errors
class TTSProviderError(TTSError):
    """General TTS provider error"""
    pass


class TTSProviderUnavailableError(TTSProviderError):
    """TTS provider is currently unavailable"""
    pass


class TTSProviderBusyError(TTSProviderError):
    """TTS provider cannot be modified because it has in-flight work."""
    pass


class TTSAuthenticationError(TTSProviderError):
    """Authentication failed with TTS provider"""
    pass


class TTSRateLimitError(TTSProviderError):
    """Rate limit exceeded for TTS provider"""
    @property
    def retry_after(self) -> Optional[int]:
        try:
            ra = self.details.get("retry_after") if hasattr(self, "details") else None
            return int(ra) if ra is not None else None
        except Exception:
            return None


class TTSQuotaExceededError(TTSProviderError):
    """Usage quota exceeded for TTS provider"""
    pass


class TTSNetworkError(TTSProviderError):
    """Network error communicating with TTS provider"""
    pass


class TTSTimeoutError(TTSProviderError):
    """Timeout waiting for TTS provider response"""
    pass


# Generation and Processing Errors
class TTSGenerationError(TTSError):
    """Error during TTS audio generation"""
    pass


class TTSStreamingError(TTSGenerationError):
    """Error during streaming TTS generation"""
    pass


class TTSAudioProcessingError(TTSGenerationError):
    """Error processing generated audio"""
    pass


class TTSFormatConversionError(TTSAudioProcessingError):
    """Error converting audio format"""
    pass


class TTSVoiceCloningError(TTSGenerationError):
    """Error during voice cloning process"""
    pass


class TTSAudioQualityError(TTSProviderError):
    """Generated audio failed quality checks (silent/truncated)"""
    pass


# Circuit Breaker and Fallback Errors
class TTSCircuitOpenError(TTSProviderError):
    """Circuit breaker is open for provider"""
    pass


class TTSAllProvidersFailedError(TTSError):
    """All available TTS providers failed"""
    pass


class TTSFallbackExhaustedError(TTSError):
    """All fallback providers exhausted"""
    pass


# Resource and System Errors
class TTSResourceError(TTSError):
    """Error with system resources"""
    pass


class TTSInsufficientMemoryError(TTSResourceError):
    """Insufficient memory for TTS generation"""
    pass


class TTSInsufficientStorageError(TTSResourceError):
    """Insufficient storage space"""
    pass


class TTSModelLoadError(TTSResourceError):
    """Error loading TTS model"""
    pass


class TTSGPUError(TTSResourceError):
    """Error with GPU processing"""
    pass


# Convenience functions for common error patterns
def validation_error(field: str, value: Any, constraint: str, **kwargs) -> TTSValidationError:
    """Create a validation error with standard format"""
    details = kwargs.get("details", {})
    details["field"] = field
    details["value"] = value
    details["constraint"] = constraint

    message = f"Validation failed for field '{field}': {constraint}"
    return TTSValidationError(message, details=details)


def provider_error(message: str, provider: str, error_code: Optional[str] = None, **kwargs) -> TTSProviderError:
    """Create a provider error with standard format"""
    return TTSProviderError(message, provider=provider, error_code=error_code, **kwargs)


def network_error(provider: str, original_error: Exception, **kwargs) -> TTSNetworkError:
    """Create a network error with standard format"""
    details = kwargs.get("details", {})
    details["original_error"] = str(original_error)
    return TTSNetworkError(str(original_error), provider=provider, error_code="NETWORK_ERROR", details=details)


def auth_error(provider: str, message: str, **kwargs) -> TTSAuthenticationError:
    """Create an authentication error with standard format"""
    details = kwargs.get("details", {})
    details["suggestion"] = "Check your API key configuration"
    return TTSAuthenticationError(message, provider=provider, error_code="AUTH_ERROR", details=details)


def rate_limit_error(provider: str, retry_after: Optional[int] = None, **kwargs) -> TTSRateLimitError:
    """Create a rate limit error with standard format"""
    details = kwargs.get("details", {})
    if retry_after:
        details["retry_after"] = retry_after

    return TTSRateLimitError(
        f"Rate limit exceeded for {provider}",
        provider=provider,
        error_code="RATE_LIMIT",
        details=details
    )


def timeout_error(provider: str, timeout_seconds: Optional[int] = None, **kwargs) -> TTSTimeoutError:
    """Create a timeout error with standard format"""
    details = kwargs.get("details", {})
    if timeout_seconds:
        details["timeout_seconds"] = timeout_seconds

    return TTSTimeoutError(
        f"Timeout waiting for {provider} response",
        provider=provider,
        error_code="TIMEOUT",
        details=details
    )


# Error categorization helper
def categorize_error(error: Exception) -> str:
    """
    Categorize error for better handling decisions.

    Args:
        error: Exception to categorize

    Returns:
        Error category string
    """
    if isinstance(error, TTSNetworkError):
        return "network"
    elif isinstance(error, TTSTimeoutError):
        return "timeout"
    elif isinstance(error, TTSRateLimitError):
        return "rate_limit"
    elif isinstance(error, TTSAuthenticationError):
        return "authentication"
    elif isinstance(error, TTSProviderBusyError):
        return "provider_busy"
    elif isinstance(error, (TTSProviderUnavailableError, TTSProviderError)):
        return "provider_error"
    elif isinstance(error, (TTSValidationError, TTSInvalidInputError, TTSTextTooLongError)):
        return "validation"
    elif isinstance(error, (TTSModelNotFoundError, TTSModelLoadError)):
        return "model"
    elif isinstance(error, (TTSProviderNotConfiguredError, TTSProviderInitializationError, TTSConfigurationError)):
        return "configuration"
    elif isinstance(error, (TTSResourceError, TTSInsufficientMemoryError)):
        return "resource"
    else:
        return "unknown"


# HTTP status code mapping for API responses
ERROR_STATUS_CODES = {
    TTSValidationError: 400,
    TTSInvalidInputError: 400,
    TTSUnsupportedFormatError: 400,
    TTSUnsupportedLanguageError: 400,
    TTSVoiceNotFoundError: 400,
    TTSTextTooLongError: 400,
    TTSInvalidVoiceReferenceError: 422,
    TTSAuthenticationError: 401,
    TTSModelNotFoundError: 404,
    TTSProviderBusyError: 409,
    TTSRateLimitError: 429,
    TTSQuotaExceededError: 429,
    TTSProviderError: 502,
    TTSAudioQualityError: 502,
    TTSProviderUnavailableError: 503,
    TTSProviderNotConfiguredError: 503,
    TTSResourceError: 503,
    TTSInsufficientMemoryError: 503,
    TTSInsufficientStorageError: 503,
    TTSNetworkError: 504,
    TTSTimeoutError: 504,
    TTSConfigurationError: 500,
    TTSProviderInitializationError: 500,
    TTSModelLoadError: 500,
    TTSGenerationError: 500,
    TTSStreamingError: 500,
    TTSAudioProcessingError: 500,
    TTSFormatConversionError: 500,
    TTSVoiceCloningError: 500,
    TTSCircuitOpenError: 503,
    TTSAllProvidersFailedError: 503,
    TTSFallbackExhaustedError: 503,
    TTSGPUError: 500,
}


def get_http_status_code(error: TTSError) -> int:
    """
    Get appropriate HTTP status code for TTS error.

    Args:
        error: TTS error instance

    Returns:
        HTTP status code
    """
    return ERROR_STATUS_CODES.get(type(error), 500)


def get_http_status_for_error(error: Exception) -> int:
    """
    Get appropriate HTTP status code for any error (alias for compatibility).

    Args:
        error: The error

    Returns:
        HTTP status code
    """
    if isinstance(error, TTSError):
        return get_http_status_code(error)
    return 500


def is_retryable_error(error: Exception) -> bool:
    """
    Check if an error is retryable.

    Args:
        error: The exception to check

    Returns:
        True if the error is retryable
    """
    # Non-retryable errors (check these first since some inherit from retryable types)
    if isinstance(error, (TTSValidationError, TTSAuthenticationError,
                         TTSProviderNotConfiguredError, TTSModelNotFoundError,
                         TTSConfigurationError)):
        return False

    # Retryable errors
    return isinstance(error, (TTSNetworkError, TTSTimeoutError, TTSRateLimitError,
                              TTSProviderError, TTSProviderUnavailableError,
                              TTSResourceError))


def resource_error(provider: str, resource_type: str, required: Any = None,
                  available: Any = None, **kwargs) -> TTSResourceError:
    """
    Create a resource error.

    Args:
        provider: Provider name
        resource_type: Type of resource (memory, gpu, etc)
        required: Required amount
        available: Available amount
        **kwargs: Additional details

    Returns:
        TTSResourceError instance
    """
    details = {
        'resource_type': resource_type,
        'required': required,
        'available': available,
        **kwargs
    }

    message = f"Insufficient {resource_type} for {provider}"
    if required and available:
        message += f" (required: {required}, available: {available})"

    return TTSResourceError(message, provider=provider, details=details)

#
# End of tts_exceptions.py
#######################################################################################################################
