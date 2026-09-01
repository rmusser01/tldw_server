from __future__ import annotations

import email.utils
import re
import weakref
from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, NoReturn

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from loguru import logger

from .AuthNZ.exceptions import DatabaseError as AuthNZDatabaseError
from .exception_types import PromptCatalogError  # noqa: F401 - re-exported for compatibility.

if TYPE_CHECKING:
    from .Admin_Webhooks.domain import WebhookErrorCode

if hasattr(status, "HTTP_422_UNPROCESSABLE_CONTENT"):
    DEFAULT_VALIDATION_STATUS = status.HTTP_422_UNPROCESSABLE_CONTENT
else:
    DEFAULT_VALIDATION_STATUS = status.HTTP_422_UNPROCESSABLE_ENTITY


_PROMPT_IMPROVEMENT_DISPATCH_MESSAGES = {
    "missing_model": "Select an active chat model and try again.",
    "unsupported_model": "The selected chat model is not available.",
    "provider_not_configured": "The active provider is not configured for this request.",
    "provider_rate_limited": "The active provider is temporarily rate limited.",
    "provider_timeout": "The active provider timed out.",
    "provider_unavailable": "The active provider is temporarily unavailable.",
    "model_refusal": "The active model did not provide an improvement candidate.",
    "invalid_model_output": "The active model returned an unusable response.",
    "internal_error": "The prompt improvement request could not be completed.",
}
_MAX_PROMPT_IMPROVEMENT_RETRY_AFTER_SECONDS = 86_400


class TransactionPassthroughError(Exception):
    """Sanitized domain failure that may cross a rolled-back DB transaction."""


class WebhookError(TransactionPassthroughError):
    """Expected webhook domain failure with no caller-controlled message text."""

    def __init__(
        self,
        code: WebhookErrorCode,
        http_status: int | None = None,
    ) -> None:
        self.code = code
        self.http_status = http_status or code.http_status
        super().__init__(code.value)


class PromptImprovementError(RuntimeError):
    """Stable domain failure suitable for endpoint error mapping."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class RecurringQuestionRAGError(Exception):
    """Expected Recurring Question RAG execution failure."""

    def __init__(self, code: str, *, retryable: bool = False, details: dict[str, Any] | None = None) -> None:
        super().__init__(code)
        self.code = code
        self.retryable = retryable
        self.details = details or {}


class RecurringQuestionWorkerRetryableError(Exception):
    """Raised after durable run state is updated so WorkerSDK can retry the Jobs job."""


class ClaimsAnalyticsExportError(RuntimeError):
    """Safe domain failure surfaced by Claims analytics export operations."""

    def __init__(
        self,
        public_message: str,
        *,
        code: str,
        retryable: bool = False,
        http_status: int = 400,
    ) -> None:
        super().__init__(public_message)
        self.public_message = public_message
        self.code = code
        self.retryable = retryable
        self.http_status = http_status


class NotesOrganizationValidationError(ValueError):
    """Validation failure with a stable Notes organization Sync error code."""

    def __init__(self, error_code: str, message: str) -> None:
        super().__init__(message)
        self.error_code = error_code


class NotesLinkValidationError(ValueError):
    """Validation failure for a non-canonical Notes link payload."""

    error_code = "notes_link_payload_invalid"


class NoteAttachmentPolicyError(ValueError):
    """Raised when attachment metadata is outside the canonical Notes policy."""


class NotesTaskContractError(ValueError):
    """Stable fail-closed error for Notes task Sync contract violations."""


class LegacyAttachmentSourceError(RuntimeError):
    """Sanitized failure while reading a legacy Notes attachment source."""

    def __init__(self, error_code: str) -> None:
        super().__init__(error_code)
        self.error_code = error_code


class NotesAttachmentBootstrapInterrupted(RuntimeError):
    """Testable interruption that deliberately leaves durable progress resumable."""


class NotesTaskBootstrapInterrupted(RuntimeError):
    """Testable interruption that leaves durable task bootstrap progress resumable."""


class NotesTaskActivitySourceInvalid(RuntimeError):
    """Malformed legacy task activity encountered during trusted bootstrap."""


class NotesTaskActivitySourceChanged(RuntimeError):
    """Previously observed task activity changed during trusted bootstrap."""


class NotesAttachmentMutationError(RuntimeError):
    """Stable failure for a coordinated Notes attachment mutation."""


class NotesAttachmentSyncNotReadyError(NotesAttachmentMutationError):
    """Raised when canonical attachment mutation is not writable."""


class ProfileTransactionError(RuntimeError):
    """Base class for sanitized, transport-neutral profile transaction failures."""

    code = "profile_update_failed"
    retry_after_seconds: int | None = None


class PersonalContextError(RuntimeError):
    """Base error for canonical Personal Context operations."""


class ProfileStorageLockedError(PersonalContextError):
    """Report unavailable or unauthenticated server profile key material."""


class ProfileIntegrityError(PersonalContextError):
    """Report canonical or encrypted object authentication failure."""


class ProfileUnsupportedSchemaError(ProfileIntegrityError):
    """Report authenticated profile data from an unsupported newer schema."""


class ProfileAlreadyExistsError(PersonalContextError):
    """Report an attempt to create a second profile in one user database."""


class ProfileKeyAlreadyExistsError(PersonalContextError):
    """Report an attempt to replace existing wrapped profile keys."""


class ConcurrentProfileUpdateError(PersonalContextError):
    """Report an optimistic object-head mismatch."""


class ProfileSemanticKeyCollisionError(PersonalContextError):
    """Report an active same-scope canonical semantic-key collision."""


class ProfileQuotaExceededError(PersonalContextError):
    """Report a bounded Personal Context operational quota violation."""


class ProfileConflictError(PersonalContextError):
    """Report a stale canonical or runtime version supplied by a caller."""


class ProfileKeyCollisionError(PersonalContextError):
    """Report an active same-scope semantic key collision."""


class ProfileUnsupportedOperationError(PersonalContextError):
    """Report an operation supported by another owner but not this server."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class ProfileNotFoundError(PersonalContextError, KeyError):
    """Report that the authenticated user has no canonical profile."""


class ProfileDatabaseBusy(ProfileTransactionError):
    """Raised when a profile transaction cannot acquire the database in time."""

    code = "database_busy"

    def __init__(self, *, retry_after_seconds: int) -> None:
        super().__init__("Database is temporarily busy")
        self.retry_after_seconds = retry_after_seconds


class ProfileUpdateConcurrencyConflict(ProfileTransactionError):
    """Raised when a profile write loses a serialization or deadlock race."""

    code = "profile_update_concurrency_conflict"

    def __init__(self) -> None:
        super().__init__("Profile update conflicted")


class ProfileTransactionFailed(ProfileTransactionError):
    """Raised for sanitized non-retryable profile transaction failures."""

    def __init__(self) -> None:
        super().__init__("Profile update transaction failed")


class PromptImprovementDispatchError(RuntimeError):
    """Sanitized infrastructure failure for endpoint error mapping."""

    def __init__(
        self,
        code: str,
        *,
        internal_detail: object | None = None,
        retryable: bool = False,
        retry_after_seconds: int | None = None,
    ) -> None:
        del internal_detail
        public_message = _PROMPT_IMPROVEMENT_DISPATCH_MESSAGES.get(
            code,
            _PROMPT_IMPROVEMENT_DISPATCH_MESSAGES["internal_error"],
        )
        super().__init__(public_message)
        self.code = (
            code
            if code in _PROMPT_IMPROVEMENT_DISPATCH_MESSAGES
            else "internal_error"
        )
        self.retryable = bool(retryable)
        try:
            retry_after = int(retry_after_seconds)
        except (TypeError, ValueError):
            retry_after = None
        if retry_after is not None and retry_after < 0:
            retry_after = None
        self.retry_after_seconds = (
            min(retry_after, _MAX_PROMPT_IMPROVEMENT_RETRY_AFTER_SECONDS)
            if retry_after is not None
            else None
        )


class PromptsDatabaseError(Exception):
    """Base exception for Prompts database failures."""


class PromptsConflictError(PromptsDatabaseError):
    """Report a Prompts database concurrent-modification conflict."""

    def __init__(
        self,
        message: str = "Conflict detected: Record modified concurrently.",
        entity: Any = None,
        identifier: Any = None,
    ) -> None:
        super().__init__(message)
        self.entity = entity
        self.identifier = identifier

    def __str__(self) -> str:
        base = super().__str__()
        details = []
        if self.entity:
            details.append(f"Entity: {self.entity}")
        if self.identifier:
            details.append(f"ID: {self.identifier}")
        return f"{base} ({', '.join(details)})" if details else base


class UnknownServicePromptDefinition(ValueError):
    """Raised when a caller requests a definition outside the static registry."""

    def __init__(self, definition_id: str) -> None:
        self.definition_id = definition_id
        super().__init__(f"Unknown Service Prompt definition: {definition_id}")


class ServicePromptValidationError(ValueError):
    """Raised with safe, immutable validation errors keyed by registered part."""

    def __init__(self, field_errors: Mapping[str, str]) -> None:
        safe_errors = dict(field_errors)
        self.field_errors: Mapping[str, str] = MappingProxyType(safe_errors)
        super().__init__("Service Prompt validation failed for: " + ", ".join(safe_errors))


class ServicePromptCorruptOverride(RuntimeError):
    """Raised when a saved override cannot be parsed or validated."""

    def __init__(self, revision: str) -> None:
        self.revision = revision
        super().__init__(f"Stored Service Prompt override is corrupt at revision {revision}.")


class ServicePromptRevisionConflict(PromptsConflictError):
    """Report the revision observed during a failed conditional write."""

    def __init__(self, current_revision: str | None) -> None:
        super().__init__("Service Prompt override changed concurrently.")
        self.current_revision = current_revision


class VideoProcessingError(Exception):
    """Raised when video processing fails."""


class EgressPolicyError(Exception):
    """Raised when an outbound URL violates the egress/SSRF policy."""

    def __init__(self, message: str, *, reason_code: str | None = None) -> None:
        super().__init__(message)
        self.reason_code = reason_code


NetworkErrorClassification = Literal["timeout"]


class NetworkError(Exception):
    """Raised for sanitized transport failures, optionally with an HTTP status."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        classification: NetworkErrorClassification | None = None,
    ) -> None:
        if status_code is not None:
            if type(status_code) is not int:
                raise TypeError("status_code must be an integer")
            if not 100 <= status_code <= 599:
                raise ValueError("status_code must be a valid HTTP status")
        if classification not in (None, "timeout"):
            raise ValueError("Unsupported network error classification")
        self.status_code = status_code
        self.classification = classification
        super().__init__(message)


HTTPHopErrorCode = Literal[
    "invalid_request",
    "dns_resolution_failed",
    "dns_timeout",
    "dns_address_denied",
    "connect_timeout",
    "read_timeout",
    "write_timeout",
    "total_timeout",
    "peer_verification_failed",
    "tls_error",
    "protocol_error",
    "response_headers_too_large",
    "response_too_large",
    "decompressed_response_too_large",
    "parser_input_too_large",
    "unsupported_content_encoding",
    "invalid_content_encoding",
    "transport_error",
]

_HTTP_HOP_ERROR_MESSAGES: dict[HTTPHopErrorCode, str] = {
    "invalid_request": "The outbound request is invalid.",
    "dns_resolution_failed": "The destination could not be resolved.",
    "dns_timeout": "Destination resolution timed out.",
    "dns_address_denied": "The destination address is not allowed.",
    "connect_timeout": "The destination connection timed out.",
    "read_timeout": "The destination response timed out.",
    "write_timeout": "The outbound request timed out.",
    "total_timeout": "The outbound request exceeded its time limit.",
    "peer_verification_failed": "The connected destination could not be verified.",
    "tls_error": "The secure destination connection failed.",
    "protocol_error": "The destination returned an invalid response.",
    "response_headers_too_large": "The destination response headers are too large.",
    "response_too_large": "The destination response is too large.",
    "decompressed_response_too_large": "The decoded destination response is too large.",
    "parser_input_too_large": "The destination response exceeds the parser limit.",
    "unsupported_content_encoding": "The destination used an unsupported content encoding.",
    "invalid_content_encoding": "The destination returned invalid encoded content.",
    "transport_error": "The destination request failed.",
}


class HTTPHopError(Exception):
    """A stable, sanitized failure from the one-hop HTTP boundary."""

    def __init__(self, code: HTTPHopErrorCode, *, retryable: bool = False) -> None:
        message = _HTTP_HOP_ERROR_MESSAGES.get(code)
        if message is None:
            raise ValueError("Unsupported HTTP hop error code")
        if not isinstance(retryable, bool):
            raise TypeError("retryable must be a boolean")
        self.code = code
        self.retryable = retryable
        super().__init__(message)


DiscoveryGatewayErrorCode = Literal[
    "request_rejected",
    "policy_inactive",
    "hop_failed",
    "invalid_hop_response",
]

_DISCOVERY_GATEWAY_ERROR_MESSAGES: dict[DiscoveryGatewayErrorCode, str] = {
    "request_rejected": "Discovery gateway request rejected",
    "policy_inactive": "Discovery gateway policy inactive",
    "hop_failed": "Discovery gateway hop failed",
    "invalid_hop_response": "Discovery gateway hop response rejected",
}


class DiscoveryGatewayError(Exception):
    """Stable failure without request, response, or provider detail."""

    __slots__ = ("code", "retryable", "timed_out")

    def __init__(
        self,
        code: DiscoveryGatewayErrorCode,
        *,
        retryable: bool = False,
        timed_out: bool = False,
    ) -> None:
        if code not in _DISCOVERY_GATEWAY_ERROR_MESSAGES:
            raise ValueError("Unsupported discovery gateway error code")
        if type(retryable) is not bool:
            raise TypeError("retryable must be a boolean")
        if type(timed_out) is not bool:
            raise TypeError("timed_out must be a boolean")
        self.code = code
        self.retryable = retryable
        self.timed_out = timed_out
        super().__init__(_DISCOVERY_GATEWAY_ERROR_MESSAGES[code])


class DiscoveryExecutionError(ValueError):
    """Stable executor failure containing only a sanitized code."""

    __slots__ = ("code",)

    def __init__(self, code: str) -> None:
        if type(code) is not str or not code:
            raise TypeError("execution_error_code_must_be_nonempty_string")
        self.code = code
        super().__init__(code)


_DISCOVERY_ADAPTER_ERROR_CODES = frozenset(
    {
        "provider_rate_limited",
        "provider_response_rejected",
        "provider_payload_invalid",
        "provider_parse_limit_exceeded",
        "provider_parse_deadline_exceeded",
    }
)
_DISCOVERY_RETRY_AFTER_DELTA_SECONDS_RE = re.compile(r"[0-9]+\Z")


def _valid_discovery_retry_after(value: object) -> bool:
    """Return whether one discovery retry hint is delta-seconds or strict IMF-fixdate."""
    if type(value) is not str:
        return False
    if _DISCOVERY_RETRY_AFTER_DELTA_SECONDS_RE.fullmatch(value) is not None:
        return True
    try:
        parsed = email.utils.parsedate_to_datetime(value)
        return email.utils.format_datetime(parsed, usegmt=True) == value
    except (TypeError, ValueError):
        return False


class DiscoveryAdapterError(ValueError):
    """Stable adapter failure containing only allowlisted metadata."""

    __slots__ = (
        "code",
        "retry_after",
        "__weakref__",
    )

    def __init__(self, code: str, *, retry_after: str | None = None) -> None:
        if type(code) is not str:
            raise TypeError("adapter_error_code_must_be_string")
        if code not in _DISCOVERY_ADAPTER_ERROR_CODES:
            raise ValueError("adapter_error_code_invalid")
        if retry_after is not None:
            if code != "provider_rate_limited":
                raise ValueError("retry_after_requires_rate_limit")
            if not _valid_discovery_retry_after(retry_after):
                raise ValueError("retry_after_invalid")
        self.code = code
        self.retry_after = retry_after
        super().__init__(code)
        _DISCOVERY_ADAPTER_ERROR_SEALS[self] = (code, retry_after)


_DISCOVERY_ADAPTER_ERROR_SEALS: weakref.WeakKeyDictionary[
    DiscoveryAdapterError,
    tuple[str, str | None],
] = weakref.WeakKeyDictionary()


def _trusted_discovery_adapter_error(error: BaseException) -> tuple[str, str | None] | None:
    """Snapshot one exact, unmodified discovery adapter failure."""
    if type(error) is not DiscoveryAdapterError:
        return None
    try:
        code = error.code
        retry_after = error.retry_after
        if (
            type(code) is not str
            or (retry_after is not None and type(retry_after) is not str)
            or _DISCOVERY_ADAPTER_ERROR_SEALS.get(error) != (code, retry_after)
            or error.args != (code,)
            or code not in _DISCOVERY_ADAPTER_ERROR_CODES
            or (retry_after is not None and not _valid_discovery_retry_after(retry_after))
            or (code != "provider_rate_limited" and retry_after is not None)
        ):
            return None
    except Exception:  # noqa: BLE001 - malformed adapter failures fail closed.
        return None
    return code, retry_after


class PlanningError(ValueError):
    """Typed pure-planning failure."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class _PayloadInvalid(Exception):
    pass


class _ParseLimitExceeded(Exception):
    pass


class _ParseDeadlineExceeded(Exception):
    pass


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


class ChatAPIError(Exception):
    """Base exception for chat API call errors."""

    def __init__(
        self,
        message: str = "An error occurred during the chat API call.",
        status_code: int = 500,
        provider: str | None = None,
    ) -> None:
        self.message = message
        self.status_code = status_code
        self.provider = provider
        super().__init__(self.message)


class ChatAuthenticationError(ChatAPIError):
    """Raised when a chat provider rejects request credentials."""

    def __init__(
        self,
        message: str = "Authentication failed with the chat provider.",
        provider: str | None = None,
        status_code: int = 401,
    ) -> None:
        preserved_status = 403 if status_code == 403 else 401
        super().__init__(message, status_code=preserved_status, provider=provider)


class ChatConfigurationError(ChatAPIError):
    """Raised for missing or invalid chat-provider configuration."""

    _ERROR_CODES = frozenset(
        {
            "provider_configuration_invalid",
            "missing_provider_credentials",
        }
    )

    def __init__(
        self,
        message: str = "Chat provider configuration error.",
        provider: str | None = None,
        error_code: str = "provider_configuration_invalid",
    ) -> None:
        self.error_code = (
            error_code
            if error_code in self._ERROR_CODES
            else "provider_configuration_invalid"
        )
        super().__init__(message, status_code=500, provider=provider)


class ChatBadRequestError(ChatAPIError):
    """Raised when a chat provider rejects request parameters."""

    def __init__(
        self,
        message: str = "Invalid request sent to the chat provider.",
        provider: str | None = None,
    ) -> None:
        super().__init__(message, status_code=400, provider=provider)


class ChatRateLimitError(ChatAPIError):
    """Raised when a chat provider rate-limits a request."""

    def __init__(
        self,
        message: str = "Rate limit exceeded with the chat provider.",
        provider: str | None = None,
    ) -> None:
        super().__init__(message, status_code=429, provider=provider)


class ChatProviderError(ChatAPIError):
    """Raised for a general upstream chat-provider error."""

    def __init__(
        self,
        message: str = "Error received from the chat provider API.",
        status_code: int = 502,
        provider: str | None = None,
        details: Any = None,
    ) -> None:
        self.details = details
        super().__init__(message, status_code=status_code, provider=provider)


class ProviderCredentialTerminalError(RuntimeError):
    """Carry one bounded credential code through chat execution layers."""

    _ERROR_CODES = frozenset(
        {
            "provider_authentication_failed",
            "invalid_provider_credentials",
            "missing_provider_credentials",
            "credential_store_unavailable",
            "credential_scope_revoked",
            "provider_configuration_invalid",
            "provider_unavailable",
            "provider_disabled",
            "model_not_allowed",
        }
    )

    def __init__(self, code: str) -> None:
        self.code = code if code in self._ERROR_CODES else "provider_configuration_invalid"
        super().__init__(self.code)


class SanitizedProviderStreamError(ChatAPIError):
    """Safe provider-stream signal with explicit, fail-closed replay metadata."""

    def __init__(
        self,
        *,
        code: str,
        message: str,
        status_code: int,
        replay_certified: bool = False,
        credential_refresh_retry_certified: bool = False,
    ) -> None:
        self.code = code
        self.upstream_dispatched = replay_certified is not True
        self.output_emitted = False
        self.allow_non_stream_fallback = replay_certified is True
        self.credential_refresh_retry_safe = credential_refresh_retry_certified is True
        super().__init__(message=message, status_code=status_code)


class TTSPublicHTTPException(HTTPException):
    """Marker for TTS errors whose traceback must be dropped at serialization."""


def raise_detached_error(error: BaseException) -> NoReturn:
    """Raise a safe replacement without retaining an active private exception.

    ``raise ... from None`` suppresses display of an exception chain but Python
    still stores the handled exception in ``__context__``.  Public adapter
    boundaries use this helper so tracing and audit consumers cannot recover a
    provider response, credential, or endpoint detail from the safe error.
    """

    try:
        raise error from None
    except BaseException as detached:
        detached.__cause__ = None
        detached.__context__ = None
        detached.__suppress_context__ = True
        raise


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


class EmbeddingWorkflowTraceError(ValueError):
    """Workflow trace data violated its safety or boundedness contract."""


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


class IngestionSourceSchemaError(RuntimeError):
    """Raised when the ingestion-source schema definition is incomplete."""


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


class STTExecutionPlanError(BadRequestError):
    """Raised when a planned STT execution cannot be honored."""


class STTExecutionUnsupportedError(STTExecutionPlanError):
    """Raised when an adapter cannot safely expose the benchmark contract."""


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


class SharedWorkspaceAccessError(RuntimeError):
    """Base error for recipient shared-workspace access resolution."""


class SharedWorkspaceNotFound(SharedWorkspaceAccessError):
    """Raised for every missing, inactive, or unauthorized shared target."""

    def __init__(self) -> None:
        super().__init__("Shared workspace not found")


class SharedWorkspaceUnavailable(SharedWorkspaceAccessError):
    """Raised when an authorized shared target cannot be resolved operationally."""

    def __init__(self) -> None:
        super().__init__("Shared workspace is temporarily unavailable")


class SharedWorkspaceCloneNotAllowed(SharedWorkspaceAccessError):
    """Raised when the authoritative share policy disables recipient cloning."""

    def __init__(self) -> None:
        super().__init__("Shared workspace cloning is not allowed")


class SharedWorkspaceChatServiceError(RuntimeError):
    """Base shared-chat error with a stable code and disclosure-safe message."""

    code = "shared_workspace_unavailable"
    retryable = True


class SharedWorkspaceSourceScopeInvalid(SharedWorkspaceChatServiceError):
    """Raised when a requested shared source scope is invalid."""

    code = "invalid_shared_chat_request"
    retryable = False

    def __init__(self) -> None:
        super().__init__("The shared chat request is invalid.")


class SharedWorkspaceSourceSubsetRequired(SharedWorkspaceChatServiceError):
    """Raised when all queryable sources exceed the shared-chat cap."""

    code = "source_subset_required"
    retryable = False

    def __init__(self) -> None:
        super().__init__("Select a smaller set of shared sources.")


class SharedWorkspaceSourceChanged(SharedWorkspaceChatServiceError):
    """Raised when a frozen source authorization snapshot no longer matches."""

    code = "shared_source_changed"
    retryable = False

    def __init__(self) -> None:
        super().__init__("The selected shared sources changed.")


class SharedWorkspaceRetrievalUnavailable(SharedWorkspaceChatServiceError):
    """Raised when retrieval cannot produce a fully verified result."""

    code = "retrieval_unavailable"
    retryable = True

    def __init__(self) -> None:
        super().__init__("Shared workspace retrieval is temporarily unavailable.")


class SharedWorkspaceNoRelevantEvidence(SharedWorkspaceChatServiceError):
    """Raised when retrieval returns no usable verified evidence."""

    code = "no_relevant_evidence"
    retryable = False

    def __init__(self) -> None:
        super().__init__("No relevant shared evidence was found.")


class SharedWorkspaceChatContextTooLarge(SharedWorkspaceChatServiceError):
    """Raised before credentials when a grounded prompt cannot fit."""

    code = "shared_chat_context_too_large"
    retryable = False

    def __init__(self) -> None:
        super().__init__("The shared chat question is too large for this model.")


class SharedWorkspaceNoProviderConfigured(SharedWorkspaceChatServiceError):
    """Raised when no authorized recipient generation credential is usable."""

    code = "no_provider_configured"
    retryable = False

    def __init__(self) -> None:
        super().__init__("No usable generation provider is configured.")


class SharedWorkspaceGenerationFailed(SharedWorkspaceChatServiceError):
    """Raised for every shared-workspace provider or structured-output failure."""

    code = "generation_failed"
    retryable = True

    def __init__(self) -> None:
        super().__init__("Shared workspace generation is temporarily unavailable.")


class _SharedWorkspaceDataUnavailable(SharedWorkspaceChatServiceError):
    """Internal marker for unavailable canonical shared-workspace data."""

    def __init__(self) -> None:
        super().__init__("Shared workspace data is temporarily unavailable.")


class _NonQueryableSource(ValueError):
    """Internal marker for canonical sources that cannot be queried."""


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


class MacroValidationError(ValueError):
    """Raised when a chat macro definition or invocation fails validation."""


class MacroStorageError(RuntimeError):
    """Raised when chat macro definition or run storage fails."""


class MacroNotFoundError(MacroStorageError):
    """Raised when a requested chat macro or run record is missing."""


class MacroExecutionError(RuntimeError):
    """Raised when chat macro execution fails."""


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
