"""Workflow state and trace contracts for Embeddings execution."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal, Protocol, TypeAlias
from uuid import uuid4

from tldw_Server_API.app.core.exceptions import EmbeddingWorkflowTraceError

EmbeddingWorkflowPhase = Literal[
    "created",
    "normalizing",
    "resolving_policy",
    "planning",
    "serving_cache",
    "executing",
    "postprocessing",
    "persisting_outputs",
    "finalizing",
]
EmbeddingWorkflowStatus = Literal[
    "running",
    "completed",
    "failed",
    "paused",
    "cancelled",
    "retry_scheduled",
]
EmbeddingWorkflowItemState = Literal[
    "pending",
    "normalized",
    "cache_hit",
    "cache_miss",
    "provider_pending",
    "provider_succeeded",
    "postprocessed",
    "output_recorded",
    "failed",
]
EmbeddingWorkflowEventType = Literal[
    "workflow_started",
    "phase_changed",
    "prepare_completed",
    "execute_completed",
    "workflow_completed",
    "workflow_failed",
    "item_state_changed",
]
EmbeddingWorkflowRunnerMode = Literal["inline", "durable"]

SafeWorkflowScalar: TypeAlias = str | int | bool | None
SafeWorkflowMetadataValue: TypeAlias = SafeWorkflowScalar | tuple[SafeWorkflowScalar, ...]

FORBIDDEN_METADATA_FIELDS = frozenset(
    {
        "raw_input",
        "input",
        "texts",
        "token_arrays",
        "api_key",
        "authorization",
        "cookie",
        "nonce",
        "provider_response",
        "provider_body",
    }
)
FORBIDDEN_FIELD_SUBSTRINGS = ("secret", "password")
FORBIDDEN_VALUE_SUBSTRINGS = (
    "api_key",
    "apikey",
    "authorization",
    "bearer ",
    "password",
    "provider body",
    "raw input",
    "secret",
)
FORBIDDEN_METADATA_FIELD_FRAGMENTS = FORBIDDEN_METADATA_FIELDS
SAFE_TOKEN_COUNT_FIELDS = frozenset(
    {"token_count", "token_counts", "total_tokens", "prompt_tokens"}
)
SAFE_METADATA_ENUM_VALUES: Mapping[str, frozenset[str]] = MappingProxyType(
    {
        "endpoint_path": frozenset({"/api/v1/embeddings"}),
        "execution_path": frozenset({"adapter", "legacy"}),
        "failure_kind": frozenset({"domain", "unexpected"}),
        "phase": frozenset(
            {
                "created",
                "normalizing",
                "resolving_policy",
                "planning",
                "serving_cache",
                "executing",
                "postprocessing",
                "persisting_outputs",
                "finalizing",
            }
        ),
        "runner_mode": frozenset({"durable", "inline"}),
    }
)
SAFE_METADATA_NONNEGATIVE_INTEGER_FIELDS = frozenset(
    {
        "cache_hits",
        "cache_misses",
        "fallback_chain_length",
        "item_count",
        "prompt_tokens",
        "response_header_count",
        "token_count",
        "total_tokens",
        "vector_count",
    }
)
SAFE_METADATA_OPTIONAL_NONNEGATIVE_INTEGER_FIELDS = frozenset({"dimensions"})
SAFE_METADATA_BOOLEAN_FIELDS = frozenset(
    {"adapter_used", "fallback_allowed", "retryable"}
)
SAFE_METADATA_INTEGER_SEQUENCE_FIELDS = frozenset({"token_counts"})
SAFE_METADATA_FIELDS = frozenset(SAFE_METADATA_ENUM_VALUES).union(
    SAFE_METADATA_NONNEGATIVE_INTEGER_FIELDS,
    SAFE_METADATA_OPTIONAL_NONNEGATIVE_INTEGER_FIELDS,
    SAFE_METADATA_BOOLEAN_FIELDS,
    SAFE_METADATA_INTEGER_SEQUENCE_FIELDS,
)
MAX_METADATA_LIST_ITEMS = 128
MAX_METADATA_STRING_LENGTH = 4096
SAFE_METADATA_STRING_PATTERN = re.compile(r"^[A-Za-z0-9_./:+-]+$")
WORKFLOW_ID_PATTERN = re.compile(r"^emb-wf-[0-9a-f]{32}$")
SENSITIVE_VALUE_PATTERN = re.compile(
    r"(?:sk-(?:proj-)?[a-z0-9_-]{8,}|hf_[a-z0-9]{8,}|github_pat_[a-z0-9_]{8,}|"
    r"gh[pousr]_[a-z0-9]{8,}|xox[baprs]-[a-z0-9-]{8,}|"
    r"aiza[a-z0-9_-]{20,}|(?:akia|asia)[a-z0-9]{12,}|"
    r"eyj[a-z0-9_-]{8,}\.eyj[a-z0-9_-]{8,}\.[a-z0-9_-]{8,})",
    re.IGNORECASE,
)


def _validate_workflow_id(workflow_id: str) -> None:
    """Reject workflow identifiers outside the generated Stage 1 format."""
    if WORKFLOW_ID_PATTERN.fullmatch(workflow_id) is None:
        raise EmbeddingWorkflowTraceError("Workflow id does not match the approved format")


def _validate_metadata_name(name: str) -> None:
    """Reject trace metadata names outside the safe field contract."""
    normalized = name.strip().lower()
    if any(fragment in normalized for fragment in FORBIDDEN_METADATA_FIELD_FRAGMENTS):
        raise EmbeddingWorkflowTraceError(f"Unsafe workflow metadata field: {name}")
    if "token" in normalized and normalized not in SAFE_TOKEN_COUNT_FIELDS:
        raise EmbeddingWorkflowTraceError(f"Unsafe workflow metadata field: {name}")
    if any(part in normalized for part in FORBIDDEN_FIELD_SUBSTRINGS):
        raise EmbeddingWorkflowTraceError(f"Unsafe workflow metadata field: {name}")
    if normalized not in SAFE_METADATA_FIELDS:
        raise EmbeddingWorkflowTraceError(f"Unsupported workflow metadata field: {name}")


def _safe_metadata_string(value: str, *, field_name: str) -> str:
    """Validate a fixed-enum string for an approved metadata field."""
    if len(value) > MAX_METADATA_STRING_LENGTH:
        raise EmbeddingWorkflowTraceError("Workflow metadata string value exceeds maximum length")
    normalized = value.casefold()
    if SENSITIVE_VALUE_PATTERN.search(value) is not None:
        raise EmbeddingWorkflowTraceError("Workflow metadata string value resembles a credential")
    if any(part in normalized for part in FORBIDDEN_VALUE_SUBSTRINGS):
        raise EmbeddingWorkflowTraceError("Workflow metadata string value contains sensitive content")
    if not value or SAFE_METADATA_STRING_PATTERN.fullmatch(value) is None:
        raise EmbeddingWorkflowTraceError("Workflow metadata string values must be safe identifiers")
    if value not in SAFE_METADATA_ENUM_VALUES[field_name]:
        raise EmbeddingWorkflowTraceError(
            f"Unsupported workflow metadata value for field: {field_name}"
        )
    return value


def _safe_nonnegative_integer(value: object, *, field_name: str) -> int:
    """Return a strict non-negative integer or reject the trace value."""
    if type(value) is not int or value < 0:
        raise EmbeddingWorkflowTraceError(
            f"Workflow metadata field {field_name} must be a non-negative integer"
        )
    return value


def _safe_metadata_value(field_name: str, value: object) -> SafeWorkflowMetadataValue:
    """Validate and freeze one allowlisted metadata value."""
    if field_name in SAFE_METADATA_ENUM_VALUES:
        if not isinstance(value, str):
            raise EmbeddingWorkflowTraceError(
                f"Workflow metadata field {field_name} must be an approved identifier"
            )
        return _safe_metadata_string(value, field_name=field_name)
    if field_name in SAFE_METADATA_BOOLEAN_FIELDS:
        if type(value) is not bool:
            raise EmbeddingWorkflowTraceError(
                f"Workflow metadata field {field_name} must be a boolean"
            )
        return value
    if field_name in SAFE_METADATA_NONNEGATIVE_INTEGER_FIELDS:
        return _safe_nonnegative_integer(value, field_name=field_name)
    if field_name in SAFE_METADATA_OPTIONAL_NONNEGATIVE_INTEGER_FIELDS:
        if value is None:
            return None
        return _safe_nonnegative_integer(value, field_name=field_name)
    if field_name in SAFE_METADATA_INTEGER_SEQUENCE_FIELDS:
        if not isinstance(value, (list, tuple)):
            raise EmbeddingWorkflowTraceError(
                f"Workflow metadata field {field_name} must be a bounded integer sequence"
            )
        if len(value) > MAX_METADATA_LIST_ITEMS:
            raise EmbeddingWorkflowTraceError("Workflow metadata list value exceeds maximum item count")
        return tuple(_safe_nonnegative_integer(item, field_name=field_name) for item in value)
    raise EmbeddingWorkflowTraceError(f"Unsupported workflow metadata field: {field_name}")


def safe_workflow_metadata(
    metadata: Mapping[str, object] | None = None,
) -> Mapping[str, SafeWorkflowMetadataValue]:
    """Return an immutable, field-validated workflow metadata snapshot."""
    if not metadata:
        return MappingProxyType({})
    safe: dict[str, SafeWorkflowMetadataValue] = {}
    for key, value in metadata.items():
        field_name = str(key)
        _validate_metadata_name(field_name)
        safe[field_name] = _safe_metadata_value(field_name, value)
    return MappingProxyType(safe)


@dataclass(frozen=True, slots=True)
class EmbeddingWorkflowContext:
    """Safe request-level identity and runner context for one workflow."""

    workflow_id: str
    runner_mode: EmbeddingWorkflowRunnerMode
    endpoint_path: str = "/api/v1/embeddings"

    def __post_init__(self) -> None:
        """Validate direct construction against the workflow ID contract."""
        _validate_workflow_id(self.workflow_id)

    @classmethod
    def create(
        cls,
        *,
        endpoint_path: str,
        runner_mode: EmbeddingWorkflowRunnerMode,
    ) -> EmbeddingWorkflowContext:
        """Create a context with a generated workflow-local identifier."""
        return cls(
            workflow_id=f"emb-wf-{uuid4().hex}",
            endpoint_path=endpoint_path,
            runner_mode=runner_mode,
        )


@dataclass(frozen=True, slots=True)
class EmbeddingWorkflowEvent:
    """Immutable typed event emitted by an Embeddings workflow runner."""

    event_type: EmbeddingWorkflowEventType
    workflow_id: str
    phase: EmbeddingWorkflowPhase | None = None
    status: EmbeddingWorkflowStatus | None = None
    item_index: int | None = None
    item_state: EmbeddingWorkflowItemState | None = None
    metadata: Mapping[str, SafeWorkflowMetadataValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate identity and freeze metadata before collection."""
        _validate_workflow_id(self.workflow_id)
        object.__setattr__(self, "metadata", safe_workflow_metadata(dict(self.metadata)))


class EmbeddingWorkflowTraceCollector(Protocol):
    """Collector contract for bounded workflow event recording."""

    enabled: bool

    def record(self, event: EmbeddingWorkflowEvent) -> None:
        """Record one validated workflow event."""
        raise NotImplementedError


class EmbeddingNoopWorkflowTraceCollector:
    """Disabled production-default collector that retains no events."""

    enabled = False

    def record(self, event: EmbeddingWorkflowEvent) -> None:
        """Discard a workflow event without retaining state."""
        del event


@dataclass(slots=True)
class EmbeddingInMemoryWorkflowTraceCollector:
    """Bounded in-memory collector for tests and isolated callers."""

    max_events: int = 256
    enabled: bool = True
    events: list[EmbeddingWorkflowEvent] = field(default_factory=list)

    def record(self, event: EmbeddingWorkflowEvent) -> None:
        """Append an event or fail closed when the event cap is reached."""
        if len(self.events) >= self.max_events:
            raise EmbeddingWorkflowTraceError("Workflow trace event limit exceeded")
        self.events.append(event)


__all__ = [
    "EmbeddingInMemoryWorkflowTraceCollector",
    "EmbeddingNoopWorkflowTraceCollector",
    "EmbeddingWorkflowContext",
    "EmbeddingWorkflowEvent",
    "EmbeddingWorkflowEventType",
    "EmbeddingWorkflowItemState",
    "EmbeddingWorkflowPhase",
    "EmbeddingWorkflowRunnerMode",
    "EmbeddingWorkflowStatus",
    "EmbeddingWorkflowTraceCollector",
    "EmbeddingWorkflowTraceError",
    "FORBIDDEN_FIELD_SUBSTRINGS",
    "FORBIDDEN_METADATA_FIELDS",
    "FORBIDDEN_VALUE_SUBSTRINGS",
    "MAX_METADATA_LIST_ITEMS",
    "MAX_METADATA_STRING_LENGTH",
    "SAFE_METADATA_BOOLEAN_FIELDS",
    "SAFE_METADATA_ENUM_VALUES",
    "SAFE_METADATA_FIELDS",
    "SAFE_METADATA_INTEGER_SEQUENCE_FIELDS",
    "SAFE_METADATA_NONNEGATIVE_INTEGER_FIELDS",
    "SAFE_METADATA_OPTIONAL_NONNEGATIVE_INTEGER_FIELDS",
    "SAFE_METADATA_STRING_PATTERN",
    "SAFE_TOKEN_COUNT_FIELDS",
    "SENSITIVE_VALUE_PATTERN",
    "SafeWorkflowMetadataValue",
    "SafeWorkflowScalar",
    "WORKFLOW_ID_PATTERN",
    "safe_workflow_metadata",
]
