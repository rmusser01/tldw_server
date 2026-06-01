from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from loguru import logger

if hasattr(status, "HTTP_422_UNPROCESSABLE_CONTENT"):
    DEFAULT_VALIDATION_STATUS = status.HTTP_422_UNPROCESSABLE_CONTENT
else:
    DEFAULT_VALIDATION_STATUS = status.HTTP_422_UNPROCESSABLE_ENTITY


class VideoProcessingError(Exception):
    """Raised when video processing fails."""


class EgressPolicyError(Exception):
    """Raised when an outbound URL violates the egress/SSRF policy."""


class NetworkError(Exception):
    """Raised for network transport errors (connect/read timeouts, DNS, TLS, etc.)."""


class RetryExhaustedError(Exception):
    """Raised when a request exhausts all retry attempts without success."""


class JSONDecodeError(Exception):
    """Raised when a response expected to be JSON cannot be decoded or is invalid."""


class TokenizerUnavailable(Exception):
    """Raised when tokenizer support is unavailable."""


class BadRequestError(ValueError):
    """Raised when a caller provides invalid arguments for an operation."""


class RecipeEnqueueError(RuntimeError):
    """Raised when a recipe run cannot be enqueued into Jobs."""

    def __init__(
        self,
        message: str = "Failed to enqueue recipe run.",
        *,
        error_code: str = "recipe_run_enqueue_failed",
    ) -> None:
        super().__init__(message)
        self.error_code = error_code


class CodeGraphJobError(RuntimeError):
    """Raised when a CodeGraph Jobs worker rejects or fails a job."""

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


class PrototypeJobError(RuntimeError):
    """Worker-visible prototype job failure with explicit retry metadata."""

    def __init__(
        self,
        message: str,
        *,
        retryable: bool,
        failure_code: str,
        backoff_seconds: int | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.failure_code = failure_code
        self.backoff_seconds = backoff_seconds
        self.details = details or {}


class PrototypeTerminalRuntimeError(PrototypeJobError):
    """Terminal prototype runtime state failure that should not be retried."""

    def __init__(
        self,
        message: str,
        *,
        failure_code: str = "runtime_terminal",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message,
            retryable=False,
            failure_code=failure_code,
            details=details,
        )


class PrototypeJobPayloadError(ValueError):
    """Terminal prototype job payload error that should not be retried."""

    def __init__(
        self,
        message: str,
        *,
        failure_code: str = "invalid_job_payload",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.retryable = False
        self.failure_code = failure_code
        self.backoff_seconds = None
        self.details = details or {}


class AuditLogError(RuntimeError):
    """Raised when persisting an audit event fails."""


class ValidationError(BadRequestError):
    """Raised when validation of input parameters fails."""


class IngestionSourceValidationError(ValidationError):
    """Raised when an ingestion source payload fails validation."""


class ReferenceImportError(RuntimeError):
    """Raised when a reference-manager item cannot be persisted correctly."""


class StructuredOutputParseError(ValueError):
    """Base error for structured-output parsing/normalization failures.

    Raised as the common parent for JSON extraction and schema-shape errors in
    structured-output handling paths.
    """


class StructuredOutputNoPayloadError(StructuredOutputParseError):
    """Raised when no parseable JSON payload can be produced from model output.

    Typical triggers include empty/whitespace payloads or candidate parsing
    attempts that all fail JSON decoding.
    """


class StructuredOutputSchemaError(StructuredOutputParseError):
    """Raised when parsed JSON exists but fails expected structural constraints.

    Typical triggers include missing/invalid wrapper keys, non-list containers,
    or list items that are not valid object entries for the target schema.
    """


class APIValidationError(HTTPException):
    """Raised when API input validation fails and should return HTTP 422."""

    def __init__(self, detail: Any, *, status_code: int | None = None) -> None:
        resolved_status = status_code if status_code is not None else DEFAULT_VALIDATION_STATUS
        super().__init__(status_code=resolved_status, detail=detail)


class VNScriptAuthoringError(ValueError):
    """Raised when VN script authoring preview/apply input cannot be patched."""

    def __init__(
        self,
        code: str,
        message: str,
        status_code: int = 400,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code
        self.details = details or {}


class SyncCallInEventLoopError(BadRequestError):
    """Raised when a sync chat call is made inside a running event loop."""


class StreamingProtocolError(Exception):
    """Raised for streaming protocol violations (e.g., malformed SSE)."""


class DownloadError(Exception):
    """Raised when a download fails or post-download validation fails (checksum, size)."""


class TranscriptionCancelled(RuntimeError):
    """Raised when transcription/conversion is cancelled."""


class CancelCheckError(RuntimeError):
    """Raised when a cancellation check fails unexpectedly."""


class STTTranscriptionError(RuntimeError):
    """Raised when an STT backend fails to produce a valid transcription."""


class SecurityAlertWebhookError(Exception):
    """Raised when delivery of a security alert to a webhook fails.

    Carries a concise message including HTTP status and a truncated response body
    to aid debugging without leaking excessive data.
    """


class SecurityAlertEmailError(Exception):
    """Raised when delivery of a security alert via email fails.

    Message should concisely describe the failure (e.g., STARTTLS/login/send).
    """


class SecurityAlertFileError(Exception):
    """Raised when writing a security alert to a file sink fails."""


class StoragePathValidationError(Exception):
    """Base exception for storage path validation failures."""


class InvalidStoragePathError(StoragePathValidationError):
    """Raised when a storage path is invalid or outside its allowed base."""


class StorageUnavailableError(StoragePathValidationError):
    """Raised when storage base directories cannot be resolved."""


class WorkspaceArtifactExportStateError(ValueError):
    """Raised when a workspace artifact version is not eligible for export."""


class InvalidStorageUserIdError(StoragePathValidationError):
    """Raised when a storage path resolution is attempted with an invalid user id."""


class UnsafeUserPathError(StoragePathValidationError):
    """Raised when a user-derived path escapes an allowed base directory."""


class InvalidFirstRunTransition(ValueError):
    """Raised when a setup state transition would violate first-run rules."""


class AdminDataOpsError(ValueError):
    """Base exception for admin data ops validation errors."""


class ToolCatalogConflictError(AdminDataOpsError):
    """Raised when a tool catalog already exists."""


class UnknownBackupDatasetError(AdminDataOpsError):
    """Raised when a backup request references an unknown dataset."""


class InvalidBackupUserIdError(AdminDataOpsError):
    """Raised when a backup request references an invalid user id."""


class InvalidBackupPathError(AdminDataOpsError):
    """Raised when a backup path is invalid or unsafe."""


class InvalidBackupIdError(AdminDataOpsError):
    """Raised when a backup id is malformed or unsafe."""


class InvalidRetentionPolicyError(AdminDataOpsError):
    """Raised when a retention policy key is unknown."""


class InvalidRetentionRangeError(AdminDataOpsError):
    """Raised when a retention policy update is out of range."""


class ByokValidationError(AdminDataOpsError):
    """Base exception for admin BYOK validation run errors."""


class ByokValidationDisabledError(ByokValidationError):
    """Raised when BYOK validation is disabled for the current deployment."""


class ByokValidationActiveRunError(ByokValidationError):
    """Raised when another BYOK validation run is already active."""


class ByokValidationRunNotFoundError(ByokValidationError):
    """Raised when a BYOK validation run id cannot be resolved."""


class BundleError(AdminDataOpsError):
    """Base exception for backup bundle operations."""

    error_code: str = "bundle_error"

    def __init__(self, message: str = "", *, error_code: str | None = None):
        super().__init__(message)
        if error_code is not None:
            self.error_code = error_code


class BundleExportError(BundleError):
    """Raised when bundle export fails."""

    error_code: str = "export_error"


class BundleImportError(BundleError):
    """Raised when bundle import fails."""

    error_code: str = "import_error"


class BundleNotFoundError(BundleError):
    """Raised when a bundle ID cannot be resolved to a file."""

    error_code: str = "bundle_not_found"


class BundleSchemaIncompatibleError(BundleError):
    """Raised when a bundle's schema version is incompatible with the current app."""

    error_code: str = "schema_incompatible"


class BundleDiskSpaceError(BundleError):
    """Raised when insufficient disk space is available for a bundle operation."""

    error_code: str = "insufficient_disk_space"


class BundleRateLimitError(BundleError):
    """Raised when a bundle operation exceeds the rate limit."""

    error_code: str = "rate_limit_exceeded"


class BundleConcurrencyError(BundleError):
    """Raised when another bundle operation is already in progress."""

    error_code: str = "bundle_operation_in_progress"


class TemplateStoreError(Exception):
    """Base exception for watchlist template store errors."""


class TemplateValidationError(TemplateStoreError, ValueError):
    """Raised when a watchlist template validation check fails."""


class InvalidTemplateNameError(TemplateValidationError):
    """Raised when a template name fails validation."""


class InvalidTemplateFormatError(TemplateValidationError):
    """Raised when a template format is invalid."""


class InvalidTemplatePathError(TemplateValidationError):
    """Raised when a template path escapes the allowed base directory."""


class InvalidSecretRedactionParametersError(ValueError):
    """Raised when secret redaction parameters are invalid."""

    def __init__(self, message: str = "head and tail must be non-negative"):
        super().__init__(message)


class FileArtifactsError(Exception):
    """Base exception for file artifact operations."""

    def __init__(self, code: str, detail: Any | None = None) -> None:
        super().__init__(code)
        self.code = code
        self.detail = detail


class FileArtifactsValidationError(FileArtifactsError):
    """Raised when file artifacts payload validation fails."""


FILE_ARTIFACTS_ERROR_STATUS: dict[str, int] = {
    "unsupported_file_type": status.HTTP_400_BAD_REQUEST,
    "persist_required": status.HTTP_400_BAD_REQUEST,
    "image_backend_unavailable": status.HTTP_400_BAD_REQUEST,
    "reference_image_invalid": DEFAULT_VALIDATION_STATUS,
    "reference_image_not_found": DEFAULT_VALIDATION_STATUS,
    "reference_image_unsupported_by_backend": DEFAULT_VALIDATION_STATUS,
    "reference_image_unsupported_by_model": DEFAULT_VALIDATION_STATUS,
    "reference_image_storage_unavailable": status.HTTP_503_SERVICE_UNAVAILABLE,
    "storage_quota_exceeded": status.HTTP_507_INSUFFICIENT_STORAGE,
    "storage_persist_failed": status.HTTP_500_INTERNAL_SERVER_ERROR,
    "unsupported_export_format": DEFAULT_VALIDATION_STATUS,
    "invalid_export_mode": DEFAULT_VALIDATION_STATUS,
    "invalid_async_mode": DEFAULT_VALIDATION_STATUS,
    "export_size_exceeded": DEFAULT_VALIDATION_STATUS,
    "row_limit_exceeded": DEFAULT_VALIDATION_STATUS,
    "cell_limit_exceeded": DEFAULT_VALIDATION_STATUS,
    "export_failed": status.HTTP_500_INTERNAL_SERVER_ERROR,
    "export_job_enqueue_failed": status.HTTP_500_INTERNAL_SERVER_ERROR,
    "image_generation_failed": status.HTTP_500_INTERNAL_SERVER_ERROR,
}


def file_artifacts_http_status(exc: FileArtifactsError) -> int:
    """Resolve HTTP status code for file artifact errors."""
    status_code = FILE_ARTIFACTS_ERROR_STATUS.get(exc.code)
    if status_code is None:
        if isinstance(exc, FileArtifactsValidationError):
            return DEFAULT_VALIDATION_STATUS
        return status.HTTP_500_INTERNAL_SERVER_ERROR
    return status_code


class AdapterInitializationError(FileArtifactsError):
    """Raised when a file adapter fails to initialize."""

    def __init__(self, name: str, spec: Any, exc: Exception) -> None:
        message = f"Failed to initialize adapter '{name}' (spec={spec!r}): {exc}"
        super().__init__("adapter_initialization_failed", detail=message)
        self.adapter_name = name
        self.spec = spec
        self.original_exception = exc


class ResourceNotFoundError(Exception):
    """Generic resource-not-found error for domain-level lookups."""

    def __init__(self, resource: str, identifier: str | None = None, detail: str | None = None):
        message = f"{resource} not found"
        if identifier:
            message = f"{message}: {identifier}"
        if detail:
            message = f"{message} ({detail})"
        super().__init__(message)
        self.resource = resource
        self.identifier = identifier
        self.detail = detail


class InactiveUserError(Exception):
    """Raised when an authenticated user account is inactive."""


class ServiceInitializationError(Exception):
    """Raised when a service fails to initialize or coordination fails."""


class ServiceInitializationTimeoutError(ServiceInitializationError):
    """Raised when a service initialization exceeds its timeout."""


class DataTablesJobError(RuntimeError):
    """Raised for data table job processing failures."""

    def __init__(self, message: str, *, retryable: bool = False, backoff_seconds: int | None = None) -> None:
        super().__init__(message)
        self.retryable = retryable
        if backoff_seconds is not None:
            self.backoff_seconds = backoff_seconds


class FileArtifactsJobError(RuntimeError):
    """Raised for file artifact job processing failures."""

    def __init__(self, message: str, *, retryable: bool = False, backoff_seconds: int | None = None) -> None:
        super().__init__(message)
        self.retryable = retryable
        if backoff_seconds is not None:
            self.backoff_seconds = backoff_seconds


class ReadingDigestJobError(RuntimeError):
    """Raised for reading digest job processing failures."""

    def __init__(self, message: str, *, retryable: bool = False, backoff_seconds: int | None = None) -> None:
        super().__init__(message)
        self.retryable = retryable
        if backoff_seconds is not None:
            self.backoff_seconds = backoff_seconds


class WorkflowAdapterError(Exception):
    """Base exception for workflow adapter errors."""


class AdapterError(WorkflowAdapterError):
    """Workflow adapter-specific error."""


async def video_processing_exception_handler(
    _request: Request,
    exc: VideoProcessingError,
) -> JSONResponse:
    logger.error("Video processing failed: {}", exc)
    return JSONResponse(
        status_code=500,
        content={"message": f"An error occurred during video processing: {exc!s}"},
    )


def setup_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(VideoProcessingError, video_processing_exception_handler)
