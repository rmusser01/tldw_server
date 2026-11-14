"""Core-owned Workspace chat session models, re-exported by the API schemas package (TASK-13950.15)."""

from collections.abc import Mapping
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from tldw_Server_API.app.core.Chat.assistant_startup import reject_assistant_startup_input
from tldw_Server_API.app.core.Chat.conversation_create_idempotency import normalize_client_conversation_id
from tldw_Server_API.app.core.Workspaces.chat_startup_schemas import (
    _validate_conversation_state,
)

ALLOWED_ASSISTANT_KINDS = ("character", "persona")
ALLOWED_PERSONA_MEMORY_MODES = ("read_only", "read_write")


class ChatSessionCreate(BaseModel):
    """Schema for creating a new chat session."""

    id: str | None = Field(
        None,
        description=(
            "Optional client-generated chat id: a UUID in 8-4-4-4-12 form, stored lowercase. "
            "The chat is created with this id. Repeating the same request returns the existing chat "
            "with 200 and `Idempotency-Replayed: true`. Reusing the id for a different request, or an id "
            "that is not available to the caller, returns 409 `chat_id_conflict`; repeating the request "
            "after the chat was moved to trash returns 410 `chat_deleted`. Omit it for a server-generated id."
        ),
        json_schema_extra={"format": "uuid"},
    )
    character_id: int | None = Field(None, description="ID of the character for this chat", gt=0)
    assistant_kind: Literal["character", "persona"] | None = Field(
        None,
        description="Normalized assistant identity kind for this chat",
    )
    assistant_id: str | None = Field(
        None,
        description="Normalized assistant identity ID for this chat",
        min_length=1,
    )
    persona_memory_mode: Literal["read_only", "read_write"] | None = Field(
        None,
        description="Persona durable memory behavior for this chat",
    )
    title: Optional[str] = Field(None, description="Optional title for the chat session")
    parent_conversation_id: Optional[str] = Field(None, description="Parent conversation ID for forked chats")
    forked_from_message_id: Optional[str] = Field(None, description="Message ID where this fork begins")
    state: Optional[str] = Field(None, description="Lifecycle state for the conversation")
    topic_label: Optional[str] = Field(None, description="Primary topic label for the conversation")
    cluster_id: Optional[str] = Field(None, description="Cluster/group identifier for navigation")
    source: Optional[str] = Field(None, description="Source of the conversation (e.g., email, issue)")
    external_ref: Optional[str] = Field(None, description="External reference/link for the conversation")
    scope_type: Literal["global", "workspace"] | None = Field(
        None,
        description="Scope type: 'global' for /chat, 'workspace' for per-workspace chats. Defaults to 'global'.",
    )
    workspace_id: Optional[str] = Field(None, description="Workspace ID (required when scope_type='workspace')")
    participant_character_ids: list[int] = Field(
        default_factory=list,
        max_length=32,
        description="Additional character participants captured at creation time",
    )
    prompt_preset_id: str | None = Field(None, min_length=1, max_length=128)
    memory_by_character_id: dict[str, str] = Field(default_factory=dict)
    provider: str | None = Field(None, min_length=1, max_length=128)
    model: str | None = Field(None, min_length=1, max_length=512)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p: float = Field(1.0, ge=0.0, le=1.0)
    repetition_penalty: float = Field(1.0, ge=0.0, le=3.0)
    stop: list[str] = Field(default_factory=list, max_length=64)

    model_config = {"json_schema_extra": {"example": {"character_id": 1, "title": "Evening Chat with Assistant"}}}

    @field_validator("id")
    @classmethod
    def _validate_client_chat_id(cls, value: Optional[str]) -> Optional[str]:
        """Accept only a canonical UUID so the client and server hold the same id."""
        return None if value is None else normalize_client_conversation_id(value)

    @field_validator("state")
    @classmethod
    def _validate_state(cls, value: Optional[str]) -> Optional[str]:
        return _validate_conversation_state(value)

    @model_validator(mode="before")
    @classmethod
    def _reject_startup_input(cls, value: Any) -> Any:
        """Reject caller-authored origin and strict selectors without closing legacy extras."""
        if isinstance(value, Mapping) and any(
            field in value
            for field in (
                "workspace_assistant_selection",
                "workspace_assistant_default_version",
            )
        ):
            raise ValueError("Workspace assistant selection requires the strict startup route")
        return reject_assistant_startup_input(value)

    @model_validator(mode="after")
    def _normalize_assistant_identity(self) -> "ChatSessionCreate":
        has_any_tracked_identity = any(
            value is not None
            for value in (
                self.character_id,
                self.assistant_kind,
                self.assistant_id,
                self.persona_memory_mode,
            )
        )
        if not has_any_tracked_identity:
            return self

        if self.assistant_kind is None:
            self.assistant_kind = "character" if self.character_id is not None else None
        if self.assistant_kind is None:
            raise ValueError("Provide either character_id or assistant_kind + assistant_id.")

        if self.assistant_kind not in ALLOWED_ASSISTANT_KINDS:
            raise ValueError(
                f"Invalid assistant_kind '{self.assistant_kind}'. Allowed: {', '.join(ALLOWED_ASSISTANT_KINDS)}"
            )

        if self.assistant_kind == "character":
            if self.character_id is None:
                if not self.assistant_id:
                    raise ValueError("Character chats require character_id or a numeric assistant_id.")
                try:
                    self.character_id = int(self.assistant_id)
                except ValueError as exc:
                    raise ValueError("Character assistant_id must be numeric.") from exc
            self.assistant_id = str(self.character_id)
            if self.persona_memory_mode is not None:
                raise ValueError("persona_memory_mode is only valid for persona chats.")
            return self

        if not self.assistant_id:
            raise ValueError("Persona chats require assistant_id.")
        self.character_id = None

        if self.persona_memory_mode is not None and self.persona_memory_mode not in ALLOWED_PERSONA_MEMORY_MODES:
            raise ValueError(
                f"Invalid persona_memory_mode '{self.persona_memory_mode}'. "
                f"Allowed: {', '.join(ALLOWED_PERSONA_MEMORY_MODES)}"
            )
        return self
