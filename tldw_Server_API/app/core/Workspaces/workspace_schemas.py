"""Core-owned Workspace models, re-exported by the API schemas package (TASK-13950.15)."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    ValidationInfo,
    field_validator,
    model_validator,
)

WorkspaceAssistantKind = Literal["persona"]
WorkspacePersonaMemoryMode = Literal["read_only", "read_write"]
WorkspaceEffectiveAssistantDefaultStatus = Literal["available", "unavailable", "none"]
WorkspaceEffectiveAssistantDefaultSource = Literal["workspace", "none"]
WorkspaceAssistantDefaultDegradedReason = Literal[
    "persona_deleted",
    "persona_unavailable",
    "persona_feature_disabled",
    "permission_denied",
    "invalid_default",
    "unsupported_assistant_kind",
]


class WorkspaceAssistantDefaults(BaseModel):
    """Persisted default assistant selection for new Workspace-scoped chats."""

    assistant_kind: WorkspaceAssistantKind
    assistant_id: str = Field(..., min_length=1, max_length=128)
    persona_memory_mode: WorkspacePersonaMemoryMode = "read_only"
    voice: None = None
    style: None = None
    tool_policy_profile_id: None = None

    @field_validator("voice", "style", "tool_policy_profile_id", mode="before")
    @classmethod
    def _deferred_fields_must_be_null(cls, value: Any, info: ValidationInfo) -> None:
        if value is not None:
            raise ValueError(f"{info.field_name} must be null")
        return None


class WorkspaceEffectiveAssistantDefault(BaseModel):
    """Resolved Workspace assistant default exposed to clients with degradation state."""

    status: WorkspaceEffectiveAssistantDefaultStatus
    source: WorkspaceEffectiveAssistantDefaultSource
    assistant_kind: WorkspaceAssistantKind | None = None
    assistant_id: str | None = None
    label: str | None = None
    persona_memory_mode: WorkspacePersonaMemoryMode | None = None
    degraded_reason: WorkspaceAssistantDefaultDegradedReason | None = None

    @model_validator(mode="after")
    def _validate_status_relations(self) -> WorkspaceEffectiveAssistantDefault:
        if self.status == "available":
            if self.assistant_kind is None or self.assistant_id is None:
                raise ValueError("available assistant defaults require assistant_kind and assistant_id")
            if self.degraded_reason is not None:
                raise ValueError("available assistant defaults must not include degraded_reason")
            return self
        if self.status == "unavailable":
            if self.degraded_reason is None:
                raise ValueError("unavailable assistant defaults require degraded_reason")
            return self
        populated_none_fields = [
            field_name
            for field_name in (
                "assistant_kind",
                "assistant_id",
                "label",
                "persona_memory_mode",
                "degraded_reason",
            )
            if getattr(self, field_name) is not None
        ]
        if populated_none_fields:
            raise ValueError("none assistant defaults must not include assistant or degradation fields")
        return self


WorkspaceOperationStatus = Literal[
    "queued",
    "running",
    "succeeded",
    "failed",
    "conflicted",
    "expired",
]


class WorkspaceOperationBase(BaseModel):
    """Serialized operation fields shared by Workspace-domain APIs."""

    operation_id: str
    workspace_id: str
    command: str
    status: WorkspaceOperationStatus
    started_at: str
    updated_at: str
    retryable: bool = False
    diagnostics: dict[str, Any] = Field(default_factory=dict)
    poll_href: str


WorkspaceArtifactReviewState = Literal[
    "draft",
    "reviewing",
    "accepted",
    "needs_revision",
    "rejected",
    "exported",
    "assigned",
    "archived",
]


class WorkspaceArtifactRedaction(BaseModel):
    model_config = ConfigDict(extra="allow")

    support_safe: StrictBool = True
    redacted: StrictBool = False
    retention_class: str | None = None
    redacted_fields: list[str] = Field(default_factory=list)


class WorkspaceArtifactResponse(BaseModel):
    id: str
    workspace_id: str
    artifact_type: str
    title: str
    status: str = "pending"
    content: str | None = None
    content_type: str = "text/markdown"
    preview_text: str | None = None
    summary: str | None = None
    review_state: WorkspaceArtifactReviewState = "draft"
    owner_scope: str = "user"
    owner_id: str | None = None
    project_id: str | None = None
    task_id: str | None = None
    source_collection_id: str | None = None
    root_artifact_id: str
    artifact_version_id: str
    previous_version_id: str | None = None
    producer_metadata: dict[str, Any] = Field(default_factory=dict)
    source_lineage: dict[str, Any] = Field(default_factory=dict)
    review_metadata: dict[str, Any] = Field(default_factory=dict)
    version_metadata: dict[str, Any] = Field(default_factory=dict)
    export_refs: list[dict[str, Any]] = Field(default_factory=list)
    redaction: WorkspaceArtifactRedaction = Field(default_factory=WorkspaceArtifactRedaction)
    schema_version: int = 1
    total_tokens: int | None = None
    total_cost_usd: float | None = None
    created_at: str
    completed_at: str | None = None
    version: int
