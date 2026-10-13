"""Shared frozen Resume contracts and character completion result values."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, StrictInt, model_validator

from tldw_Server_API.app.core.Chat.history_selection_schemas import HistoryWireModel


class RoleplayResumeTailV1(HistoryWireModel):
    """Exact prior tail; empty chats require both fields to be null."""

    message_id: str | None = Field(..., min_length=1, max_length=255)
    message_version: StrictInt | None = Field(..., ge=1)

    @model_validator(mode="after")
    def _require_complete_tail(self) -> RoleplayResumeTailV1:
        if (self.message_id is None) != (self.message_version is None):
            raise ValueError("Resume tail identity and version must be provided together")
        return self


class RoleplayResumeFenceV1(HistoryWireModel):
    """Complete saved behavior and history expected by an original Resume command."""

    version: StrictInt = Field(ge=1, le=1)
    snapshot_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    settings_version: StrictInt = Field(ge=1)
    history_version: StrictInt = Field(ge=1)
    tail: RoleplayResumeTailV1


class RoleplayResumeUserAppendRequest(HistoryWireModel):
    """Caller-ID text append; generation and behavior overrides are separate."""

    message_id: str = Field(min_length=1, max_length=255)
    content: str = Field(min_length=1, max_length=1_000_000)
    expected: RoleplayResumeFenceV1


class RoleplayResumeUserAppendResponse(HistoryWireModel):
    """Immutable original admission receipt, including the resulting full fence."""

    version: StrictInt = Field(ge=1, le=1)
    conversation_id: str = Field(min_length=1)
    message_id: str = Field(min_length=1)
    message_version: StrictInt = Field(ge=1)
    parent_message_id: str | None
    fence: RoleplayResumeFenceV1


class RoleplayResumeCompletionV1(HistoryWireModel):
    """Exact acknowledged input and complete original generation precondition."""

    version: StrictInt = Field(ge=1, le=1)
    input_message_id: str = Field(min_length=1, max_length=255)
    input_message_version: StrictInt = Field(ge=1)
    assistant_message_id: str = Field(min_length=1, max_length=255)
    expected: RoleplayResumeFenceV1


class RoleplayResumeCompletionResultV1(HistoryWireModel):
    """Original dispatch fence and exact authoritative save outcome."""

    version: StrictInt = Field(ge=1, le=1)
    input_message_id: str = Field(min_length=1, max_length=255)
    input_message_version: StrictInt = Field(ge=1)
    assistant_message_id: str = Field(min_length=1, max_length=255)
    assistant_message_version: StrictInt | None = Field(default=None, ge=1)
    expected: RoleplayResumeFenceV1
    fence: RoleplayResumeFenceV1 | None = None
    unsaved_reason: str | None = Field(default=None, min_length=1, max_length=100)


class CharacterChatCompletionV2Response(BaseModel):
    tldw_roleplay_resume_v1: RoleplayResumeCompletionResultV1 | None = None
    chat_id: str
    character_id: int
    provider: str
    model: str | None = None
    saved: bool = False
    user_message_id: str | None = None
    assistant_message_id: str | None = None
    assistant_content: str
    speaker_character_id: int | None = None
    speaker_character_name: str | None = None
    mood_label: str | None = None
    mood_confidence: float | None = None
    mood_topic: str | None = None
    lorebook_diagnostics: list[dict[str, Any]] | None = Field(
        None,
        description="Lorebook/world book trigger diagnostics for this turn",
    )
