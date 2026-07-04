"""Pydantic schemas for Chat Macros API endpoints."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


MacroRunMode = Literal["background", "chat_native", "foreground"]


class ChatMacroSummary(BaseModel):
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
    macros: list[ChatMacroSummary]
    count: int = Field(ge=0)


class ChatMacroDetail(BaseModel):
    summary: ChatMacroSummary
    definition: dict[str, Any]
    raw: str
    supporting_files: dict[str, str] = Field(default_factory=dict)


class ChatMacroCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=64)
    raw: str = Field(min_length=1, max_length=500_000)
    supporting_files: dict[str, str] | None = None


class ChatMacroUpdateRequest(BaseModel):
    raw: str = Field(min_length=1, max_length=500_000)
    supporting_files: dict[str, str] | None = None


class ChatMacroValidateRequest(BaseModel):
    raw: str = Field(min_length=1, max_length=500_000)


class ChatMacroValidateResponse(BaseModel):
    valid: bool
    macro: dict[str, Any] | None = None
    error: str | None = None


class ChatMacroSettingsRequest(BaseModel):
    settings: dict[str, Any] = Field(default_factory=dict)


class ChatMacroSettingsResponse(BaseModel):
    settings: dict[str, Any]


class ChatMacroCloneRequest(BaseModel):
    name: str = Field(min_length=1, max_length=64)
    command: str | None = Field(default=None, min_length=1, max_length=64)


class ChatMacroRunRequest(BaseModel):
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
    run_id: str
    status: str
    detail_url: str
    job_id: str | None = None


class ChatMacroRunRecordResponse(BaseModel):
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
    run: ChatMacroRunRecordResponse
    branches: list[ChatMacroBranchSummary] = Field(default_factory=list)


class ChatMacroCancelResponse(BaseModel):
    run_id: str
    status: str
    cancel_requested_at: str | None = None
