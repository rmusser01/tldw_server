"""Pydantic schemas for Chat Macros API endpoints."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

MacroRunMode = Literal["background"]


class ChatMacroSummary(BaseModel):
    """Catalog metadata for one chat macro."""

    name: str
    command: str
    description: str | None = None
    enabled: bool
    source: Literal["builtin", "user"]
    immutable: bool
    digest: str
    builtin_version: int | None = None
    schema_version: int


class ChatMacroListResponse(BaseModel):
    """Pageless response containing all available chat macros."""

    macros: list[ChatMacroSummary]
    count: int = Field(ge=0)


class ChatMacroDetail(BaseModel):
    """Full macro definition and editable source content."""

    summary: ChatMacroSummary
    definition: dict[str, Any]
    raw: str
    supporting_files: dict[str, str] = Field(default_factory=dict)


class ChatMacroCreateRequest(BaseModel):
    """Request to persist a user-defined macro."""

    name: str = Field(min_length=1, max_length=64)
    raw: str = Field(min_length=1, max_length=500_000)
    supporting_files: dict[str, str] | None = None


class ChatMacroUpdateRequest(BaseModel):
    """Request to replace a macro definition or change its enabled state."""

    raw: str | None = Field(default=None, min_length=1, max_length=500_000)
    supporting_files: dict[str, str] | None = None
    enabled: bool | None = None

    @model_validator(mode="after")
    def _requires_update_body(self) -> ChatMacroUpdateRequest:
        """Require a source or enabled-state change and reject orphaned files."""
        if self.raw is None and self.enabled is None:
            raise ValueError("raw or enabled is required")
        if self.raw is None and self.supporting_files is not None:
            raise ValueError("supporting_files requires raw")
        return self


class ChatMacroValidateRequest(BaseModel):
    """Macro source submitted for validation only."""

    raw: str = Field(min_length=1, max_length=500_000)


class ChatMacroValidateResponse(BaseModel):
    """Validation result with a parsed definition or bounded error."""

    valid: bool
    macro: dict[str, Any] | None = None
    error: str | None = None


class ChatMacroSettingsRequest(BaseModel):
    """User-level macro settings to validate and persist."""

    settings: dict[str, Any] = Field(default_factory=dict)


class ChatMacroSettingsResponse(BaseModel):
    """Normalized user-level macro settings."""

    settings: dict[str, Any]


class ChatMacroCloneRequest(BaseModel):
    """Requested identity for a mutable clone of a built-in macro."""

    name: str = Field(min_length=1, max_length=64)
    command: str | None = Field(default=None, min_length=1, max_length=64)


class ChatMacroRunRequest(BaseModel):
    """Request to dispatch a macro with structured arguments and context."""

    macro_name: str = Field(min_length=1, max_length=64)
    args: dict[str, Any] = Field(default_factory=dict)
    mode: MacroRunMode = "background"
    surface: str | None = Field(default=None, max_length=64)
    conversation_id: str | None = Field(default=None, max_length=128)
    workspace_id: str | None = Field(default=None, max_length=128)
    acp_session_id: str | None = Field(default=None, max_length=128)
    output_profile: str | None = Field(default=None, max_length=128)
    context_snapshot: dict[str, Any] | None = None
    model_selection: dict[str, Any] | None = None


class ChatMacroRunResponse(BaseModel):
    """Accepted macro run identity and initial status."""

    run_id: str
    status: str
    detail_url: str
    job_id: str | None = None


class ChatMacroRunRecordResponse(BaseModel):
    """Durable macro run state exposed through the API."""

    run_id: str
    macro_name: str
    macro_command: str
    macro_source: str | None = None
    macro_version: int | None = None
    macro_digest: str | None = None
    normalized_args: dict[str, Any]
    status: str
    surface: str | None = None
    conversation_id: str | None = None
    workspace_id: str | None = None
    acp_session_id: str | None = None
    job_id: str | None = None
    output_profile: str | None = None
    status_message_id: str | None = None
    final_message_id: str | None = None
    final_output: str | None = None
    final_output_format: str | None = None
    final_post_status: str | None = None
    cancel_requested_at: str | None = None
    error_code: str | None = None
    error: str | None = None
    created_at: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    updated_at: str | None = None

    model_config = ConfigDict(from_attributes=True)


class ChatMacroBranchSummary(BaseModel):
    """Public status and output for one macro branch."""

    branch_id: str
    step_id: str
    label: str | None = None
    output_name: str | None = None
    status: str
    attempt_count: int = Field(ge=0)
    output: str | None = None
    retained: bool = False
    error_code: str | None = None
    error: str | None = None
    created_at: str | None = None
    started_at: str | None = None
    finished_at: str | None = None


class ChatMacroRunDetailResponse(BaseModel):
    """Macro run record together with its branch summaries."""

    run: ChatMacroRunRecordResponse
    branches: list[ChatMacroBranchSummary] = Field(default_factory=list)


class ChatMacroCancelResponse(BaseModel):
    """State returned after requesting macro run cancellation."""

    run_id: str
    status: str
    cancel_requested_at: str | None = None
