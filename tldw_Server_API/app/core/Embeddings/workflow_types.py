"""Workflow state and trace contracts for Embeddings execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol, TypeAlias
from uuid import uuid4

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

SafeWorkflowScalar: TypeAlias = str | int | float | bool | None
SafeWorkflowMetadataValue: TypeAlias = SafeWorkflowScalar | list[SafeWorkflowScalar]

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
SAFE_TOKEN_COUNT_FIELDS = frozenset({"token_count", "token_counts", "total_tokens", "prompt_tokens"})


class EmbeddingWorkflowTraceError(ValueError):
    """Raised when workflow trace metadata is unsafe or exceeds collector bounds."""


def _validate_metadata_name(name: str) -> None:
    normalized = name.strip().lower()
    if normalized in FORBIDDEN_METADATA_FIELDS:
        raise EmbeddingWorkflowTraceError(f"Unsafe workflow metadata field: {name}")
    if "token" in normalized and normalized not in SAFE_TOKEN_COUNT_FIELDS:
        raise EmbeddingWorkflowTraceError(f"Unsafe workflow metadata field: {name}")
    if any(part in normalized for part in FORBIDDEN_FIELD_SUBSTRINGS):
        raise EmbeddingWorkflowTraceError(f"Unsafe workflow metadata field: {name}")


def _safe_metadata_value(value: object) -> SafeWorkflowMetadataValue:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list) and all(item is None or isinstance(item, (str, int, float, bool)) for item in value):
        return value
    raise EmbeddingWorkflowTraceError("Workflow metadata values must be safe scalars or lists of safe scalars")


def safe_workflow_metadata(metadata: dict[str, object] | None = None) -> dict[str, SafeWorkflowMetadataValue]:
    if not metadata:
        return {}
    safe: dict[str, SafeWorkflowMetadataValue] = {}
    for key, value in metadata.items():
        _validate_metadata_name(str(key))
        safe[str(key)] = _safe_metadata_value(value)
    return safe


@dataclass(frozen=True, slots=True)
class EmbeddingWorkflowContext:
    workflow_id: str
    runner_mode: EmbeddingWorkflowRunnerMode
    request_id: str | None = None
    user_id: str | None = None
    endpoint_path: str = "/api/v1/embeddings"

    @classmethod
    def from_request(
        cls,
        *,
        request_id: str | None,
        user_id: str | int | None,
        endpoint_path: str,
        runner_mode: EmbeddingWorkflowRunnerMode,
    ) -> "EmbeddingWorkflowContext":
        workflow_id = request_id or f"emb-wf-{uuid4().hex}"
        return cls(
            workflow_id=workflow_id,
            request_id=request_id,
            user_id=str(user_id) if user_id is not None else None,
            endpoint_path=endpoint_path,
            runner_mode=runner_mode,
        )


@dataclass(frozen=True, slots=True)
class EmbeddingWorkflowEvent:
    event_type: EmbeddingWorkflowEventType
    workflow_id: str
    phase: EmbeddingWorkflowPhase | None = None
    status: EmbeddingWorkflowStatus | None = None
    item_index: int | None = None
    item_state: EmbeddingWorkflowItemState | None = None
    metadata: dict[str, SafeWorkflowMetadataValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", safe_workflow_metadata(dict(self.metadata)))


class EmbeddingWorkflowTraceCollector(Protocol):
    enabled: bool

    def record(self, event: EmbeddingWorkflowEvent) -> None:
        raise NotImplementedError


class EmbeddingNoopWorkflowTraceCollector:
    enabled = False

    def record(self, event: EmbeddingWorkflowEvent) -> None:
        del event


@dataclass(slots=True)
class EmbeddingInMemoryWorkflowTraceCollector:
    max_events: int = 256
    enabled: bool = True
    events: list[EmbeddingWorkflowEvent] = field(default_factory=list)

    def record(self, event: EmbeddingWorkflowEvent) -> None:
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
    "SafeWorkflowMetadataValue",
    "SafeWorkflowScalar",
    "safe_workflow_metadata",
]
