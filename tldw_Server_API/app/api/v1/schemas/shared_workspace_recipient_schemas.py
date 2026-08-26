"""Strict recipient-facing schemas for shared research workspaces."""
from __future__ import annotations

from typing import Annotated, Literal
from uuid import UUID

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    model_validator,
)

from tldw_Server_API.app.api.v1.schemas.workspace_schemas import (
    WorkspaceOperationBase,
)

Identifier = Annotated[str, StringConstraints(min_length=1, max_length=512)]
ReasonCode = Annotated[str, StringConstraints(min_length=1, max_length=128)]
ShortCode = Annotated[str, StringConstraints(min_length=1, max_length=128)]
CloneWarningCode = Annotated[
    str,
    StringConstraints(min_length=1, max_length=64, pattern=r"^[a-z][a-z0-9_]{0,63}$"),
]


class _RecipientModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SharedWorkspaceErrorDetail(_RecipientModel):
    code: ShortCode
    message: str = Field(min_length=1, max_length=320)
    retryable: bool
    recovery_action: Literal["retry", "refresh", "reselect_sources"] | None = None
    retry_after_ms: int | None = Field(default=None, ge=0, le=1_800_000)


class SharedWorkspaceErrorResponse(_RecipientModel):
    detail: SharedWorkspaceErrorDetail


class SharedWorkspaceCloneRequest(_RecipientModel):
    name: str | None = Field(default=None, min_length=1, max_length=255)

    @model_validator(mode="after")
    def _name_is_not_blank(self):
        if self.name is not None and not self.name.strip():
            raise ValueError("name must not be blank")
        return self


class SharedWorkspaceCloneProgress(_RecipientModel):
    phase: Literal[
        "queued",
        "authorizing",
        "preparing",
        "sources",
        "notes",
        "artifacts",
        "finalizing",
    ]
    percent: int = Field(ge=0, le=100)
    message_code: CloneWarningCode


class SharedWorkspaceCloneCounts(_RecipientModel):
    sources_attempted: int = Field(ge=0, le=1_000_000_000)
    sources_copied: int = Field(ge=0, le=1_000_000_000)
    sources_failed: int = Field(ge=0, le=1_000_000_000)
    notes_attempted: int = Field(ge=0, le=1_000_000_000)
    notes_copied: int = Field(ge=0, le=1_000_000_000)
    notes_failed: int = Field(ge=0, le=1_000_000_000)
    artifacts_attempted: int = Field(ge=0, le=1_000_000_000)
    artifacts_copied: int = Field(ge=0, le=1_000_000_000)
    artifacts_failed: int = Field(ge=0, le=1_000_000_000)
    media_attempted: int = Field(ge=0, le=1_000_000_000)
    media_copied: int = Field(ge=0, le=1_000_000_000)
    media_failed: int = Field(ge=0, le=1_000_000_000)
    operation_owned_media_count: int = Field(ge=0, le=1_000_000_000)

    @model_validator(mode="after")
    def _counts_are_consistent(self):
        for item_name in ("sources", "notes", "artifacts", "media"):
            attempted = getattr(self, f"{item_name}_attempted")
            copied = getattr(self, f"{item_name}_copied")
            failed = getattr(self, f"{item_name}_failed")
            if copied + failed > attempted:
                raise ValueError(
                    f"{item_name} copied plus failed cannot exceed attempted"
                )
        if self.operation_owned_media_count > self.media_copied:
            raise ValueError(
                "operation_owned_media_count cannot exceed media_copied"
            )
        return self


class SharedWorkspaceCloneReadiness(_RecipientModel):
    text_search: Literal["ready", "unavailable"]
    citations: Literal["ready", "unavailable"]
    vector_search: Literal["ready", "needs_indexing", "not_configured"]


class SharedWorkspaceCloneWarning(_RecipientModel):
    code: CloneWarningCode
    count: int = Field(ge=0, le=1_000_000_000)


class SharedWorkspaceCloneResult(_RecipientModel):
    schema_version: Literal[1] = 1
    outcome: Literal["complete", "partial"]
    workspace_id: str = Field(min_length=1, max_length=255)
    name: str = Field(min_length=1, max_length=255)
    publication_confirmed: bool
    counts: SharedWorkspaceCloneCounts
    readiness: SharedWorkspaceCloneReadiness
    warnings: list[SharedWorkspaceCloneWarning] = Field(
        default_factory=list,
        max_length=8,
    )


class SharedWorkspaceCloneError(_RecipientModel):
    code: ShortCode
    message_key: str = Field(min_length=1, max_length=160)
    message: str = Field(min_length=1, max_length=320)
    cleanup_state: Literal["complete", "pending", "unknown"]


class SharedWorkspaceCloneOperationResponse(WorkspaceOperationBase):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    command: Literal["shared_workspace_clone"] = "shared_workspace_clone"
    status: Literal["queued", "running", "succeeded", "failed"]
    share_id: int = Field(gt=0)
    progress: SharedWorkspaceCloneProgress | None = None
    result: SharedWorkspaceCloneResult | None = None
    error: SharedWorkspaceCloneError | None = None

    @model_validator(mode="after")
    def _terminal_shape_matches_status(self):
        if self.status in {"queued", "running"}:
            if self.progress is None or self.result is not None or self.error is not None:
                raise ValueError(
                    f"{self.status} clone operations require progress only"
                )
        elif self.status == "succeeded":
            if (
                self.progress is not None
                or self.result is None
                or self.error is not None
                or not self.result.publication_confirmed
            ):
                raise ValueError(
                    "succeeded clone operations require a confirmed result only"
                )
        elif self.progress is not None or self.result is not None or self.error is None:
            raise ValueError("failed clone operations require an error only")
        return self


class SharedWorkspacePartialError(_RecipientModel):
    area: str = Field(min_length=1, max_length=64)
    code: ShortCode
    message: str = Field(min_length=1, max_length=320)
    retryable: bool


class SharedWorkspaceAllowedAction(_RecipientModel):
    allowed: bool
    reason_code: ReasonCode | None = None

    @model_validator(mode="after")
    def _reason_matches_decision(self):
        if self.allowed and self.reason_code is not None:
            raise ValueError("allowed actions cannot include a denial reason")
        if not self.allowed and self.reason_code is None:
            raise ValueError("denied actions require a reason code")
        return self


class SharedWorkspaceAllowedActions(_RecipientModel):
    inspect_sources: SharedWorkspaceAllowedAction
    ask_grounded_questions: SharedWorkspaceAllowedAction
    add_sources: SharedWorkspaceAllowedAction
    edit_workspace: SharedWorkspaceAllowedAction
    clone_workspace: SharedWorkspaceAllowedAction


class SharedWorkspaceGenerationDefault(_RecipientModel):
    provider: str | None = Field(default=None, min_length=1, max_length=128)
    model: str | None = Field(default=None, min_length=1, max_length=512)
    ready: bool
    reason_code: ReasonCode | None = None

    @model_validator(mode="after")
    def _validate_readiness(self):
        if self.ready:
            if self.provider is None or self.model is None or self.reason_code is not None:
                raise ValueError("ready generation defaults require provider and model only")
        elif self.provider is not None or self.model is not None or self.reason_code is None:
            raise ValueError("unavailable generation defaults require only a reason code")
        return self


class SharedWorkspacePagination(_RecipientModel):
    offset: int = Field(ge=0)
    limit: int = Field(ge=1, le=200)
    total: int = Field(ge=0)
    has_more: bool


class SharedWorkspaceSource(_RecipientModel):
    source_id: Identifier
    title: str = Field(max_length=512)
    source_type: str = Field(min_length=1, max_length=64)
    origin_url: str | None = Field(default=None, min_length=1, max_length=2_048)
    origin_host: str | None = Field(default=None, min_length=1, max_length=255)
    state: str = Field(min_length=1, max_length=64)
    reason_code: ReasonCode | None = None
    citation_ready: bool
    retrieval_ready: bool
    position: int = Field(ge=0)
    added_at: AwareDatetime | None = None


class SharedWorkspaceSourceSummary(_RecipientModel):
    total: int = Field(ge=0)
    queryable: int = Field(ge=0)
    processing: int = Field(ge=0)
    failed: int = Field(ge=0)


class SharedWorkspaceSourcePage(_RecipientModel):
    items: list[SharedWorkspaceSource] = Field(max_length=200)
    pagination: SharedWorkspacePagination
    summary: SharedWorkspaceSourceSummary
    partial_errors: list[SharedWorkspacePartialError] = Field(default_factory=list, max_length=8)


class SharedWorkspacePreviewSnippet(_RecipientModel):
    kind: Literal["content_excerpt", "chunk"]
    text: str = Field(min_length=1, max_length=12_000)
    start_char: int | None = Field(default=None, ge=0)
    end_char: int | None = Field(default=None, ge=0)
    chunk_index: int | None = Field(default=None, ge=0)


class SharedWorkspaceSourcePreview(_RecipientModel):
    source_id: Identifier
    title: str = Field(max_length=512)
    source_type: str = Field(min_length=1, max_length=64)
    origin_url: str | None = Field(default=None, min_length=1, max_length=2_048)
    origin_host: str | None = Field(default=None, min_length=1, max_length=255)
    state: str = Field(min_length=1, max_length=64)
    reason_code: ReasonCode | None = None
    content_available: bool
    preview_mode: str = Field(min_length=1, max_length=64)
    unavailable_reason: ReasonCode | None = None
    text_preview: str | None = Field(default=None, max_length=12_000)
    text_total_chars: int | None = Field(default=None, ge=0)
    text_truncated: bool
    snippets: list[SharedWorkspacePreviewSnippet] = Field(max_length=10)
    generated_at: AwareDatetime


class SharedWorkspaceCitationLocator(_RecipientModel):
    chunk: int | None = Field(default=None, ge=0)
    start_char: int | None = Field(default=None, ge=0)
    end_char: int | None = Field(default=None, ge=0)


class SharedWorkspaceCitation(_RecipientModel):
    citation_id: Identifier
    source_id: Identifier
    source_title: str = Field(max_length=512)
    locator: SharedWorkspaceCitationLocator
    quote: str = Field(min_length=1, max_length=1_000)
    score: float = Field(allow_inf_nan=False)


class SharedWorkspaceMessage(_RecipientModel):
    message_id: Identifier
    role: Literal["user", "assistant"]
    content: str = Field(max_length=100_000)
    created_at: AwareDatetime
    citations: list[SharedWorkspaceCitation] = Field(default_factory=list, max_length=20)


class SharedWorkspaceMessagePage(_RecipientModel):
    conversation_id: Identifier | None = None
    messages: list[SharedWorkspaceMessage] = Field(max_length=100)
    next_before: str | None = Field(default=None, min_length=1, max_length=2_048)


class SharedWorkspaceShare(_RecipientModel):
    share_id: int = Field(gt=0)
    access_level: str = Field(min_length=1, max_length=64)
    allow_clone: bool
    owner_display_name: str = Field(min_length=1, max_length=128)
    shared_at: AwareDatetime | None = None


class SharedWorkspaceIdentity(_RecipientModel):
    workspace_id: Identifier
    name: str = Field(max_length=512)
    description: str = Field(max_length=2_000)


class SharedWorkspaceBootstrapSources(_RecipientModel):
    items: list[SharedWorkspaceSource] = Field(max_length=50)
    pagination: SharedWorkspacePagination


class SharedWorkspaceBootstrapResponse(_RecipientModel):
    schema_version: Literal[1] = 1
    generated_at: AwareDatetime
    share: SharedWorkspaceShare
    workspace: SharedWorkspaceIdentity
    allowed_actions: SharedWorkspaceAllowedActions
    generation_default: SharedWorkspaceGenerationDefault
    source_summary: SharedWorkspaceSourceSummary
    sources: SharedWorkspaceBootstrapSources
    conversation: SharedWorkspaceMessagePage
    partial_errors: list[SharedWorkspacePartialError] = Field(default_factory=list, max_length=8)


class SharedWorkspaceSourceScope(_RecipientModel):
    mode: Literal["all", "include"]
    source_ids: list[Identifier] = Field(default_factory=list, max_length=500)

    @model_validator(mode="after")
    def _validate_mode(self):
        if self.mode == "include" and not self.source_ids:
            raise ValueError("include mode requires source IDs")
        if self.mode == "all" and self.source_ids:
            raise ValueError("all mode cannot include source IDs")
        if len(self.source_ids) != len(set(self.source_ids)):
            raise ValueError("source IDs must be unique")
        return self


class SharedWorkspaceChatRequest(_RecipientModel):
    request_id: UUID
    query: str = Field(min_length=1, max_length=10_000)
    source_scope: SharedWorkspaceSourceScope
    provider: str | None = Field(default=None, min_length=1, max_length=128)
    model: str | None = Field(default=None, min_length=1, max_length=512)

    @model_validator(mode="after")
    def _query_is_not_blank(self):
        if not self.query.strip():
            raise ValueError("query must not be blank")
        return self


# Compatibility for tests and callers introduced with the read-plane scaffold.
SharedWorkspaceChatSourceScope = SharedWorkspaceSourceScope


class SharedWorkspaceTurnMessage(_RecipientModel):
    message_id: Identifier
    role: Literal["user", "assistant"]
    content: str = Field(max_length=100_000)
    created_at: AwareDatetime


class SharedWorkspaceChatTurn(_RecipientModel):
    user_message: SharedWorkspaceTurnMessage
    assistant_message: SharedWorkspaceTurnMessage


class SharedWorkspaceChatGeneration(_RecipientModel):
    provider: str = Field(min_length=1, max_length=128)
    model: str = Field(min_length=1, max_length=512)


class SharedWorkspaceEffectiveSourceScope(_RecipientModel):
    mode: Literal["all", "include"]
    effective_source_count: int = Field(ge=1, le=500)


class SharedWorkspaceChatReplay(_RecipientModel):
    replayed: bool


class SharedWorkspaceChatResponse(_RecipientModel):
    schema_version: Literal[1] = 1
    request_id: UUID
    conversation_id: Identifier
    turn: SharedWorkspaceChatTurn
    citations: list[SharedWorkspaceCitation] = Field(min_length=1, max_length=20)
    generation: SharedWorkspaceChatGeneration
    source_scope: SharedWorkspaceEffectiveSourceScope
    replay: SharedWorkspaceChatReplay


__all__ = [
    "SharedWorkspaceAllowedAction",
    "SharedWorkspaceBootstrapResponse",
    "SharedWorkspaceChatRequest",
    "SharedWorkspaceChatResponse",
    "SharedWorkspaceChatSourceScope",
    "SharedWorkspaceCitation",
    "SharedWorkspaceCloneError",
    "SharedWorkspaceCloneOperationResponse",
    "SharedWorkspaceCloneProgress",
    "SharedWorkspaceCloneRequest",
    "SharedWorkspaceCloneResult",
    "SharedWorkspaceErrorDetail",
    "SharedWorkspaceErrorResponse",
    "SharedWorkspaceGenerationDefault",
    "SharedWorkspaceMessage",
    "SharedWorkspaceMessagePage",
    "SharedWorkspacePagination",
    "SharedWorkspacePartialError",
    "SharedWorkspaceSource",
    "SharedWorkspaceSourceScope",
    "SharedWorkspaceSourcePage",
    "SharedWorkspaceSourcePreview",
]
