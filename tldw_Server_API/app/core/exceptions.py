from __future__ import annotations

from collections.abc import Mapping
import re
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from loguru import logger

from .AuthNZ.exceptions import DatabaseError as AuthNZDatabaseError
from .exception_types import PromptCatalogError  # noqa: F401 - re-exported for compatibility.

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


class AudioQuotaStoreUnavailable(AuthNZDatabaseError):
    """Raised when canonical audio daily-minute quota storage is unavailable."""


class RetryExhaustedError(Exception):
    """Raised when a request exhausts all retry attempts without success."""


class JSONDecodeError(Exception):
    """Raised when a response expected to be JSON cannot be decoded or is invalid."""


class ThirdPartyHTTPStatusError(RuntimeError):
    """Raised when a Third_Party provider returns an HTTP error status."""

    def __init__(self, status_code: int, reason: str | None = None) -> None:
        self.status_code = int(status_code)
        self.reason = reason or ""
        message = f"HTTP Error: {self.status_code}"
        if self.reason:
            message = f"{message} - {self.reason}"
        super().__init__(message)


class TokenizerUnavailable(Exception):
    """Raised when tokenizer support is unavailable."""


class BadRequestError(ValueError):
    """Raised when a caller provides invalid arguments for an operation."""


class JobSubmissionLimitError(BadRequestError):
    """Raised when a Jobs submission limit rejects the whole request."""

    def __init__(self, message: str, *, code: str, retry_after: int | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.retry_after = retry_after


class InvalidMetadataOrderKeyError(ValueError):
    """Raised when a metadata order key cannot be safely used."""


SafeJsonScalar = str | int | float | bool | None
SafeDetail = dict[str, SafeJsonScalar]

EmbeddingErrorCode = Literal[
    "empty_input",
    "invalid_input_type",
    "too_many_inputs",
    "input_too_long",
    "invalid_token_array",
    "unknown_provider",
    "provider_model_mismatch",
    "invalid_dimensions",
    "provider_denied",
    "model_denied",
    "model_required",
    "provider_unsupported",
    "missing_provider_credentials",
    "provider_malformed_response",
    "provider_rate_limited",
    "provider_unavailable",
    "fallback_exhausted",
    "circuit_breaker_open",
    "internal_execution_failure",
]

_EMBEDDING_REDACTED = "[redacted]"
_EMBEDDING_SENSITIVE_KEY_PARTS = (
    "api_key",
    "apikey",
    "access_token",
    "authorization",
    "body",
    "credential",
    "header",
    "input",
    "password",
    "raw",
    "secret",
    "text",
)
_EMBEDDING_SAFE_NUMERIC_TOKEN_KEYS = {"tokens", "token_count", "prompt_tokens", "total_tokens"}
_EMBEDDING_SENSITIVE_VALUE_PARTS = (
    "api_key",
    "authorization",
    "bearer ",
    "password",
    "secret",
    "sk-",
    "token",
)


def _is_embedding_sensitive_key(key: str) -> bool:
    normalized = key.lower().replace("-", "_")
    return any(part in normalized for part in _EMBEDDING_SENSITIVE_KEY_PARTS)


def _is_embedding_sensitive_string(value: str) -> bool:
    normalized = value.lower()
    return any(part in normalized for part in _EMBEDDING_SENSITIVE_VALUE_PARTS)


def _is_embedding_safe_numeric_token_count(key: str, value: object) -> bool:
    normalized = key.lower().replace("-", "_")
    return (
        normalized in _EMBEDDING_SAFE_NUMERIC_TOKEN_KEYS
        and isinstance(value, (int, float))
        and not isinstance(value, bool)
    )


def sanitize_embedding_public_details(details: list[SafeDetail] | None) -> list[SafeDetail]:
    """Return public embedding error details with sensitive values redacted."""
    if not isinstance(details, list):
        return []

    sanitized: list[SafeDetail] = []
    for item in details:
        if not isinstance(item, Mapping):
            continue

        safe_item: SafeDetail = {}
        for raw_key, raw_value in item.items():
            if not isinstance(raw_key, str):
                continue

            if _is_embedding_safe_numeric_token_count(raw_key, raw_value):
                safe_item[raw_key] = raw_value
                continue

            if _is_embedding_sensitive_key(raw_key):
                safe_item[raw_key] = _EMBEDDING_REDACTED
                continue

            if isinstance(raw_value, str):
                safe_item[raw_key] = (
                    _EMBEDDING_REDACTED if _is_embedding_sensitive_string(raw_value) else raw_value
                )
            elif isinstance(raw_value, (int, float, bool)) or raw_value is None:
                safe_item[raw_key] = raw_value

        if safe_item:
            sanitized.append(safe_item)

    return sanitized


def sanitize_embedding_scalar_mapping(values: Mapping[str, object] | None) -> dict[str, SafeJsonScalar]:
    """Return safe scalar observability values for embedding request metadata."""
    if not isinstance(values, Mapping):
        return {}

    sanitized: dict[str, SafeJsonScalar] = {}
    for raw_key, raw_value in values.items():
        if not isinstance(raw_key, str):
            continue

        if _is_embedding_safe_numeric_token_count(raw_key, raw_value):
            sanitized[raw_key] = raw_value
            continue

        if _is_embedding_sensitive_key(raw_key):
            sanitized[raw_key] = _EMBEDDING_REDACTED
            continue

        if isinstance(raw_value, str):
            sanitized[raw_key] = (
                _EMBEDDING_REDACTED if _is_embedding_sensitive_string(raw_value) else raw_value
            )
        elif isinstance(raw_value, (int, float, bool)) or raw_value is None:
            sanitized[raw_key] = raw_value

    return sanitized


class EmbeddingDomainError(Exception):
    """Base domain error for embedding request planning and execution."""

    def __init__(
        self,
        code: EmbeddingErrorCode,
        message: str,
        *,
        retryable: bool = False,
        provider: str | None = None,
        model: str | None = None,
        retry_after: int | float | None = None,
        cause_class: str | None = None,
        details: list[SafeDetail] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable
        self.provider = provider
        self.model = model
        self.retry_after = retry_after
        self.cause_class = cause_class
        self.details = sanitize_embedding_public_details(details)

    def to_http_payload(self) -> dict[str, SafeJsonScalar | list[SafeDetail]]:
        """Return a stable, sanitized payload safe for HTTP responses."""
        return {
            "error_code": self.code,
            "message": self.message,
            "provider": self.provider,
            "model": self.model,
            "retryable": self.retryable,
            "retry_after": self.retry_after,
            "details": sanitize_embedding_public_details(self.details),
            "cause_class": self.cause_class,
        }


class EmbeddingInputError(EmbeddingDomainError):
    """Input validation failed before provider execution."""


class EmbeddingPolicyError(EmbeddingDomainError):
    """Provider or model policy rejected the request."""


class EmbeddingProviderError(EmbeddingDomainError):
    """Provider returned an error response or malformed result."""


class EmbeddingRateLimitError(EmbeddingProviderError):
    """Provider rate limit or quota throttled the request."""


class EmbeddingExecutionError(EmbeddingDomainError):
    """Embedding execution failed after request planning."""


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


class WritingAnnotationReviewEnqueueError(RuntimeError):
    """Raised when a manuscript scene annotation review cannot be queued."""

    def __init__(
        self,
        message: str = "Failed to enqueue manuscript scene annotation review.",
        *,
        error_code: str = "writing_annotation_review_enqueue_failed",
    ) -> None:
        super().__init__(message)
        self.error_code = error_code


class ExplainerValidationError(ValueError):
    """Raised when an Explainer API request violates workspace rules."""


class ExplainerNotFoundError(LookupError):
    """Raised when an Explainer resource is not visible to the requesting user."""


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


class RPGError(Exception):
    """Base exception for RPG runtime errors."""


class RPGNotFoundError(RPGError):
    """Raised when an RPG resource cannot be found."""


class RPGValidationError(RPGError):
    """Raised when RPG input fails domain validation."""


class RPGConflictError(RPGError):
    """Raised when an RPG write conflicts with current state."""


class SetupError(RuntimeError):
    """Base class for first-run setup and installer failures."""


class SetupSubprocessError(SetupError):
    """Raised when a setup-managed subprocess fails or times out."""


class SetupLockTimeoutError(SetupError):
    """Raised when setup state persistence cannot acquire its lock."""


class InvalidGovernanceCandidateError(ValueError):
    """Raised when a policy source returns a malformed governance candidate."""


class ResearchDiscoveryError(Exception):
    """Base class for public research discovery service failures."""

    def __init__(self, public_detail: str) -> None:
        super().__init__(public_detail)
        self.public_detail = public_detail


class ResearchDiscoveryValidationError(ResearchDiscoveryError):
    """Raised when a research discovery request fails validation."""


class ResearchDiscoveryBadRequestError(ResearchDiscoveryError):
    """Raised when a research discovery request is malformed or unsupported."""


class ResearchDiscoveryTimeoutError(ResearchDiscoveryError):
    """Raised when research discovery exceeds its configured time budget."""


class ResearchDiscoveryUpstreamError(ResearchDiscoveryError):
    """Raised when all selected research discovery providers fail."""


class IngestionSourceValidationError(ValidationError):
    """Raised when an ingestion source payload fails validation."""


class ReferenceImportError(RuntimeError):
    """Raised when a reference-manager item cannot be persisted correctly."""


class ConnectorServiceError(RuntimeError):
    """Raised when External_Sources connector service operations fail."""


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


class AudioProtocolError(StreamingProtocolError):
    """Raised when a websocket audio frame violates the strict audio contract."""

    def __init__(self, code: str, message: str, close_code: int = 4400) -> None:
        """Initialize a client-safe audio protocol error.

        Args:
            code: Machine-readable error code sent to the websocket client.
            message: Human-readable error message sent to the websocket client.
            close_code: WebSocket close code used after emitting the error payload.
        """
        super().__init__(message)
        self.code = code
        self.message = message
        self.close_code = close_code


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


class WorkspaceMembershipAdapterError(Exception):
    """Fail-closed adapter error for Workspace cross-resource memberships."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        status_code: int = 404,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = dict(details or {})


class WorkspaceMembershipServiceError(Exception):
    """API-facing service error for Workspace cross-resource memberships."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        status_code: int = 409,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = dict(details or {})


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


_RESEARCH_WORKSPACE_OUTPUT_ERROR_CODE_RE = re.compile(r"^[a-z][a-z0-9_]{0,95}$")


def _safe_research_workspace_output_error_code(value: str) -> str:
    """Return a stable public code for Research Workspace output failures."""
    raw = str(value or "").strip().lower()
    if _RESEARCH_WORKSPACE_OUTPUT_ERROR_CODE_RE.fullmatch(raw):
        return raw
    return "research_workspace_output_failed"


class ResearchWorkspaceOutputJobError(RuntimeError):
    """Worker-visible Research Workspace output failure with retry metadata."""

    def __init__(
        self,
        public_code: str,
        *,
        status_code: int = 400,
        retryable: bool = False,
        backoff_seconds: int | None = None,
    ) -> None:
        super().__init__(public_code)
        self.public_code = _safe_research_workspace_output_error_code(public_code)
        self.status_code = status_code
        self.retryable = retryable
        self.backoff_seconds = backoff_seconds
        self.failure_code = self.public_code


class ReadingDigestJobError(RuntimeError):
    """Raised for reading digest job processing failures."""

    def __init__(self, message: str, *, retryable: bool = False, backoff_seconds: int | None = None) -> None:
        super().__init__(message)
        self.retryable = retryable
        if backoff_seconds is not None:
            self.backoff_seconds = backoff_seconds


class WritingAnnotationReviewJobError(RuntimeError):
    """Raised for controlled Writing annotation review worker failures."""

    def __init__(
        self,
        message: str,
        *,
        retryable: bool = False,
        failure_code: str = "writing_annotation_review_job_failed",
    ) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.failure_code = failure_code


class SkillsMCPNotFoundError(ValueError):
    """Bounded public not-found response from the Skills MCP module."""


class SkillsMCPRenderedTooLargeError(ValueError):
    """Bounded public rendered-output rejection from the Skills MCP module."""


class SkillsMCPContextIntegrityError(PermissionError):
    """Bounded public render-time integrity rejection from the Skills MCP module."""


class SkillsMCPDatabaseCloseError(Exception):
    """Internal marker for a logged Skills MCP database close failure."""


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
