# exceptions.py
# Description: Custom exceptions for Chatbook module
#
"""
Chatbook Module Exceptions
--------------------------

Provides comprehensive exception hierarchy for proper error handling
and context preservation in the Chatbook module.
"""

from typing import Any, Optional


class ChatbookException(Exception):
    """Base exception for all Chatbook-related errors."""

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        context: Optional[dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        """
        Initialize ChatbookException.

        Args:
            message: Error message
            error_code: Unique error code for identification
            context: Additional context information
            cause: Original exception that caused this error
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code or "CHATBOOK_ERROR"
        self.context = context or {}
        self.cause = cause

    def to_dict(self) -> dict[str, Any]:
        """Convert exception to dictionary for logging/API responses."""
        return {
            "error": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code,
            "context": self.context,
            "cause": str(self.cause) if self.cause else None
        }


class ValidationError(ChatbookException):
    """Raised when input validation fails."""

    def __init__(self, message: str, field: Optional[str] = None, **kwargs):
        """
        Initialize ValidationError.

        Args:
            message: Validation error message
            field: Field that failed validation
            **kwargs: Additional context
        """
        context = kwargs.get('context', {})
        if field:
            context['field'] = field
        kwargs['context'] = context
        kwargs['error_code'] = kwargs.get('error_code', 'VALIDATION_ERROR')
        super().__init__(message, **kwargs)


class MediaQualityError(ValidationError):
    """Raised when media quality value is invalid."""

    def __init__(self, message: str, quality_value: Optional[str] = None, **kwargs):
        """
        Initialize MediaQualityError.

        Args:
            message: Error message
            quality_value: The invalid quality value provided
            **kwargs: Additional context
        """
        context = kwargs.get('context', {})
        if quality_value:
            context['quality_value'] = quality_value
        context['valid_values'] = ['thumbnail', 'compressed', 'original']
        kwargs['context'] = context
        kwargs['error_code'] = kwargs.get('error_code', 'MEDIA_QUALITY_ERROR')
        super().__init__(message, field='media_quality', **kwargs)


class AuthorValidationError(ValidationError):
    """Raised when author field validation fails."""

    def __init__(self, message: str, author_value: Optional[str] = None, **kwargs):
        """
        Initialize AuthorValidationError.

        Args:
            message: Error message
            author_value: The invalid author value (truncated for safety)
            **kwargs: Additional context
        """
        context = kwargs.get('context', {})
        if author_value:
            # Truncate for safety - don't log potentially large/malicious values
            context['author_value'] = author_value[:50] if len(author_value) > 50 else author_value
        kwargs['context'] = context
        kwargs['error_code'] = kwargs.get('error_code', 'AUTHOR_VALIDATION_ERROR')
        super().__init__(message, field='author', **kwargs)


class FileOperationError(ChatbookException):
    """Raised when file operations fail."""

    def __init__(self, message: str, file_path: Optional[str] = None, operation: Optional[str] = None, **kwargs):
        """
        Initialize FileOperationError.

        Args:
            message: Error message
            file_path: Path to the file
            operation: Operation that failed (read, write, delete, etc.)
            **kwargs: Additional context
        """
        context = kwargs.get('context', {})
        if file_path:
            context['file_path'] = file_path
        if operation:
            context['operation'] = operation
        kwargs['context'] = context
        kwargs['error_code'] = kwargs.get('error_code', 'FILE_OPERATION_ERROR')
        super().__init__(message, **kwargs)


class DatabaseError(ChatbookException):
    """Raised when database operations fail."""

    def __init__(self, message: str, query: Optional[str] = None, **kwargs):
        """
        Initialize DatabaseError.

        Args:
            message: Error message
            query: SQL query that failed
            **kwargs: Additional context
        """
        context = kwargs.get('context', {})
        if query:
            # Truncate long queries for logging
            context['query'] = query[:500] if len(query) > 500 else query
        kwargs['context'] = context
        kwargs['error_code'] = kwargs.get('error_code', 'DATABASE_ERROR')
        super().__init__(message, **kwargs)


class QuotaExceededError(ChatbookException):
    """Raised when user exceeds quota limits."""

    def __init__(self, message: str, quota_type: str, limit: Any = None, current: Any = None, **kwargs):
        """
        Initialize QuotaExceededError.

        Args:
            message: Error message
            quota_type: Type of quota exceeded
            limit: Quota limit
            current: Current usage
            **kwargs: Additional context
        """
        context = kwargs.get('context', {})
        context.update({
            'quota_type': quota_type,
            'limit': limit,
            'current': current
        })
        kwargs['context'] = context
        kwargs['error_code'] = kwargs.get('error_code', 'QUOTA_EXCEEDED')
        super().__init__(message, **kwargs)


class SecurityError(ChatbookException):
    """Raised when security violations are detected."""

    def __init__(self, message: str, violation_type: str, **kwargs):
        """
        Initialize SecurityError.

        Args:
            message: Error message
            violation_type: Type of security violation
            **kwargs: Additional context
        """
        context = kwargs.get('context', {})
        context['violation_type'] = violation_type
        kwargs['context'] = context
        kwargs['error_code'] = kwargs.get('error_code', 'SECURITY_ERROR')
        super().__init__(message, **kwargs)


class JobError(ChatbookException):
    """Raised when job processing fails."""

    def __init__(self, message: str, job_id: Optional[str] = None, job_type: Optional[str] = None, **kwargs):
        """
        Initialize JobError.

        Args:
            message: Error message
            job_id: ID of the failed job
            job_type: Type of job that failed
            **kwargs: Additional context
        """
        context = kwargs.get('context', {})
        if job_id:
            context['job_id'] = job_id
        if job_type:
            context['job_type'] = job_type
        kwargs['context'] = context
        kwargs['error_code'] = kwargs.get('error_code', 'JOB_ERROR')
        super().__init__(message, **kwargs)


class ChatbookImportError(ChatbookException):
    """Raised when chatbook import fails."""

    def __init__(self, message: str, import_file: Optional[str] = None, item_type: Optional[str] = None, **kwargs):
        """
        Initialize ChatbookImportError.

        Args:
            message: Error message
            import_file: File being imported
            item_type: Type of item that failed to import
            **kwargs: Additional context
        """
        context = kwargs.get('context', {})
        if import_file:
            context['import_file'] = import_file
        if item_type:
            context['item_type'] = item_type
        kwargs['context'] = context
        kwargs['error_code'] = kwargs.get('error_code', 'IMPORT_ERROR')
        super().__init__(message, **kwargs)


class ExportError(ChatbookException):
    """Raised when chatbook export fails."""

    def __init__(self, message: str, export_name: Optional[str] = None, item_type: Optional[str] = None, **kwargs):
        """
        Initialize ExportError.

        Args:
            message: Error message
            export_name: Name of the export
            item_type: Type of item that failed to export
            **kwargs: Additional context
        """
        context = kwargs.get('context', {})
        if export_name:
            context['export_name'] = export_name
        if item_type:
            context['item_type'] = item_type
        kwargs['context'] = context
        kwargs['error_code'] = kwargs.get('error_code', 'EXPORT_ERROR')
        super().__init__(message, **kwargs)


class ArchiveError(ChatbookException):
    """Raised when archive operations fail."""

    def __init__(self, message: str, archive_path: Optional[str] = None, **kwargs):
        """
        Initialize ArchiveError.

        Args:
            message: Error message
            archive_path: Path to the archive
            **kwargs: Additional context
        """
        context = kwargs.get('context', {})
        if archive_path:
            context['archive_path'] = archive_path
        kwargs['context'] = context
        kwargs['error_code'] = kwargs.get('error_code', 'ARCHIVE_ERROR')
        super().__init__(message, **kwargs)


class ConflictError(ChatbookException):
    """Raised when import/export conflicts occur."""

    def __init__(self, message: str, conflict_type: str, existing_item: Optional[str] = None, new_item: Optional[str] = None, **kwargs):
        """
        Initialize ConflictError.

        Args:
            message: Error message
            conflict_type: Type of conflict
            existing_item: Existing item causing conflict
            new_item: New item causing conflict
            **kwargs: Additional context
        """
        context = kwargs.get('context', {})
        context.update({
            'conflict_type': conflict_type,
            'existing_item': existing_item,
            'new_item': new_item
        })
        kwargs['context'] = context
        kwargs['error_code'] = kwargs.get('error_code', 'CONFLICT_ERROR')
        super().__init__(message, **kwargs)


class RetryableError(ChatbookException):
    """Base class for errors that can be retried."""

    def __init__(self, message: str, retry_after: Optional[int] = None, max_retries: int = 3, **kwargs):
        """
        Initialize RetryableError.

        Args:
            message: Error message
            retry_after: Seconds to wait before retry
            max_retries: Maximum number of retries
            **kwargs: Additional context
        """
        context = kwargs.get('context', {})
        context.update({
            'retry_after': retry_after,
            'max_retries': max_retries,
            'retryable': True
        })
        kwargs['context'] = context
        super().__init__(message, **kwargs)


class TemporaryError(RetryableError):
    """Raised for temporary failures that should be retried."""

    def __init__(self, message: str, **kwargs):
        """Initialize TemporaryError."""
        kwargs['error_code'] = kwargs.get('error_code', 'TEMPORARY_ERROR')
        super().__init__(message, **kwargs)


class NetworkError(RetryableError):
    """Raised for network-related failures."""

    def __init__(self, message: str, url: Optional[str] = None, **kwargs):
        """
        Initialize NetworkError.

        Args:
            message: Error message
            url: URL that failed
            **kwargs: Additional context
        """
        context = kwargs.get('context', {})
        if url:
            context['url'] = url
        kwargs['context'] = context
        kwargs['error_code'] = kwargs.get('error_code', 'NETWORK_ERROR')
        super().__init__(message, **kwargs)


class ChatbookTimeoutError(ChatbookException):
    """Raised when operations timeout."""

    def __init__(self, message: str, timeout_seconds: Optional[int] = None, **kwargs):
        """
        Initialize ChatbookTimeoutError.

        Args:
            message: Error message
            timeout_seconds: Timeout duration
            **kwargs: Additional context
        """
        context = kwargs.get('context', {})
        if timeout_seconds:
            context['timeout_seconds'] = timeout_seconds
        kwargs['context'] = context
        kwargs['error_code'] = kwargs.get('error_code', 'TIMEOUT_ERROR')
        super().__init__(message, **kwargs)


class ConfigurationError(ChatbookException):
    """Raised when configuration is invalid or missing."""

    def __init__(self, message: str, config_key: Optional[str] = None, **kwargs):
        """
        Initialize ConfigurationError.

        Args:
            message: Error message
            config_key: Configuration key that is invalid or missing
            **kwargs: Additional context
        """
        context = kwargs.get('context', {})
        if config_key:
            context['config_key'] = config_key
        kwargs['context'] = context
        kwargs['error_code'] = kwargs.get('error_code', 'CONFIGURATION_ERROR')
        super().__init__(message, **kwargs)


# Error recovery utilities

def is_retryable(error: Exception) -> bool:
    """
    Check if an error is retryable.

    Args:
        error: Exception to check

    Returns:
        True if error can be retried
    """
    return isinstance(error, RetryableError)


def get_retry_delay(error: Exception, attempt: int = 1) -> int:
    """
    Get retry delay for an error.

    Args:
        error: Exception that occurred
        attempt: Current attempt number

    Returns:
        Seconds to wait before retry
    """
    if isinstance(error, RetryableError) and error.context.get('retry_after'):
        return error.context['retry_after']

    # Exponential backoff with jitter
    import secrets
    base_delay = 2 ** attempt
    jitter = secrets.randbelow(1000) / 1000
    return int(base_delay + jitter)


def should_circuit_break(error_count: int, threshold: int = 5) -> bool:
    """
    Determine if circuit breaker should open.

    Args:
        error_count: Number of consecutive errors
        threshold: Error threshold

    Returns:
        True if circuit should open
    """
    return error_count >= threshold


# Public API
__all__ = [
    # Base exceptions
    "ChatbookException",
    "ValidationError",
    "MediaQualityError",
    "AuthorValidationError",
    "FileOperationError",
    "DatabaseError",
    "QuotaExceededError",
    "SecurityError",
    "JobError",
    # Import/Export exceptions
    "ChatbookImportError",
    "ChatbookTimeoutError",
    "ExportError",
    "ArchiveError",
    "ConflictError",
    "TemporaryError",
    "RetryableError",
    "ConfigurationError",
    "NetworkError",
    # Utility functions
    "is_retryable",
    "get_retry_delay",
    "should_circuit_break",
]
