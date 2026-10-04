"""Request models for ``POST /api/v1/chats/import`` (D7 P8).

The body mirrors what the chat and message read models return: a conversation
(``id``, ``title``, ``state``, ``created_at``, ``last_modified``, assistant
binding, fork lineage) and its messages (``id``, ``parent_message_id``,
``role``, ``content``, ``timestamp``, ``images``, ``metadata``).

Every model is closed (``extra="forbid"``). An import is lossless or it is
refused, so a field the server would not store is an error, not something to
ignore. That also means the body cannot name an owner, a scope, a version or a
deleted flag: the chat always goes to the authenticated account, in the global
chat scope, as live version-1 rows.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import AwareDatetime, BaseModel, BeforeValidator, ConfigDict, Field, field_validator, model_validator

from tldw_Server_API.app.core.Chat.conversation_import import normalize_client_conversation_id
from tldw_Server_API.app.core.Workspaces.chat_startup_schemas import (
    STARTUP_TEXT_BYTE_LIMITS,
    _validate_conversation_state,
)

# Parse ceilings. The configured per-chat message limit (MAX_MESSAGES_PER_CHAT,
# 1000 by default) is enforced by the endpoint and is normally far lower.
CHAT_IMPORT_MAX_MESSAGES = 10_000
CHAT_IMPORT_MAX_IMAGES_PER_MESSAGE = 10
CHAT_IMPORT_TITLE_MAX_BYTES = STARTUP_TEXT_BYTE_LIMITS["title"]
CHAT_IMPORT_ID_MAX_LENGTH = 255

# Message ids become path segments (``/messages/{id}``), so they are limited to
# characters that need no escaping and cannot be a dot segment.
CHAT_IMPORT_MESSAGE_ID_PATTERN = r"^[A-Za-z0-9_][A-Za-z0-9_.:~-]*$"


def _iso_string(value: Any) -> Any:
    """Accept ISO 8601 text only, so a bare number is never guessed to be seconds or milliseconds."""
    if not isinstance(value, str):
        raise ValueError("Expected an ISO 8601 timestamp string with a UTC offset")
    return value


def _require_utf8(value: str) -> None:
    """Refuse half of a surrogate pair, which JSON can carry but no database can store.

    The message never quotes the value: a validation message holding an
    unpaired surrogate cannot itself be encoded.
    """
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        raise ValueError("text must be valid Unicode") from None


def _plain_reference(value: Any) -> Any:
    """Refuse an id that could not be looked up: control characters or half a surrogate pair."""
    if isinstance(value, str):
        if any(ord(character) < 32 or ord(character) == 127 for character in value):
            raise ValueError("ids must not contain control characters")
        _require_utf8(value)
    return value


ImportTimestamp = Annotated[AwareDatetime, BeforeValidator(_iso_string)]
ImportMessageId = Annotated[
    str, Field(min_length=1, max_length=CHAT_IMPORT_ID_MAX_LENGTH, pattern=CHAT_IMPORT_MESSAGE_ID_PATTERN)
]
ImportReference = Annotated[
    str, BeforeValidator(_plain_reference), Field(min_length=1, max_length=CHAT_IMPORT_ID_MAX_LENGTH)
]


class ChatImportMessage(BaseModel):
    """One message of an imported chat."""

    model_config = ConfigDict(extra="forbid")

    id: ImportMessageId = Field(
        ..., description="Client message id. It becomes the server message id, so it must not be in use."
    )
    parent_message_id: ImportMessageId | None = Field(
        None,
        description="Id of the parent message, which must be part of this import. Null for a first message.",
    )
    role: Literal["user", "assistant", "system"] = Field(..., description="Message sender role")
    content: str = Field(
        "",
        description="Message text, stored unchanged. May be empty only when the message has an image.",
    )
    timestamp: ImportTimestamp = Field(
        ...,
        description=(
            "When the message was written: ISO 8601 with a UTC offset. Stored as UTC to the millisecond "
            "and never replaced with the time of the import."
        ),
    )
    images: list[str] = Field(
        default_factory=list,
        max_length=CHAT_IMPORT_MAX_IMAGES_PER_MESSAGE,
        description="Ordered images as base64 data URLs (PNG, JPEG, GIF, WebP, BMP or ICO)",
    )
    metadata: dict[str, Any] | None = Field(
        None,
        description=(
            "Generation metadata for an assistant message. Only generation_status, model_id, provider, "
            "finish_reason and usage (prompt_tokens, completion_tokens, total_tokens) are accepted; "
            "any other key is refused."
        ),
    )


class ChatImportRequest(BaseModel):
    """A local chat to save to the signed-in account, with its whole message graph."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "id": "6b0f8c1e-2d4a-4f3b-9a71-5c2e8d9f0a14",
                "title": "Trip planning",
                "created_at": "2026-09-01T09:59:30.000Z",
                "messages": [
                    {
                        "id": "pa_1a2b-3c4d-5e6-7f80",
                        "role": "user",
                        "content": "Where should we go in spring?",
                        "timestamp": "2026-09-01T10:00:00.250Z",
                    },
                    {
                        "id": "pa_9f8e-7d6c-5b4-a392",
                        "parent_message_id": "pa_1a2b-3c4d-5e6-7f80",
                        "role": "assistant",
                        "content": "Lisbon is mild in April.",
                        "timestamp": "2026-09-01T10:00:04.900Z",
                        "metadata": {"model_id": "gpt-4o-mini", "provider": "openai", "generation_status": "complete"},
                    },
                ],
            }
        },
    )

    id: str = Field(
        ...,
        description=(
            "Client-generated chat id: a UUID in 8-4-4-4-12 form, stored lowercase. Repeating the same import "
            "returns the existing chat with 200 and `Idempotency-Replayed: true`. Reusing the id for a different "
            "import, or an id that is not available to the caller, returns 409 `chat_id_conflict`; repeating the "
            "import after the chat was moved to trash returns 410 `chat_deleted`."
        ),
        json_schema_extra={"format": "uuid"},
    )
    title: str = Field(..., min_length=1, description="Chat title")
    state: str | None = Field(None, description="Lifecycle state for the conversation")
    created_at: ImportTimestamp = Field(..., description="When the chat was created: ISO 8601 with a UTC offset")
    last_modified: ImportTimestamp | None = Field(
        None,
        description="When the chat was last changed. Defaults to its newest message, never to the time of the import.",
    )
    character_id: int | None = Field(None, gt=0, description="ID of the character for this chat")
    assistant_kind: Literal["character", "persona"] | None = Field(
        None, description="Normalized assistant identity kind for this chat"
    )
    assistant_id: ImportReference | None = Field(
        None, description="Normalized assistant identity ID for this chat"
    )
    persona_memory_mode: Literal["read_only", "read_write"] | None = Field(
        None, description="Persona durable memory behavior for this chat"
    )
    parent_conversation_id: ImportReference | None = Field(
        None,
        description="Server id of the chat this one was forked from. It must already belong to the caller.",
    )
    forked_from_message_id: ImportReference | None = Field(
        None,
        description="Message of the parent chat where this fork begins",
    )
    messages: list[ChatImportMessage] = Field(
        ...,
        min_length=1,
        max_length=CHAT_IMPORT_MAX_MESSAGES,
        description="Every message of the chat, on all branches. Parents may be listed in any order.",
    )

    @field_validator("id")
    @classmethod
    def _validate_id(cls, value: str) -> str:
        """Accept only a canonical UUID so the client and server hold the same id."""
        return normalize_client_conversation_id(value)

    @field_validator("title")
    @classmethod
    def _validate_title(cls, value: str) -> str:
        """Bound the title like other chat creation routes; the text itself is kept unchanged."""
        if not value.strip():
            raise ValueError("title cannot be blank")
        if len(value.encode("utf-8", errors="replace")) > CHAT_IMPORT_TITLE_MAX_BYTES:
            raise ValueError(f"title exceeds {CHAT_IMPORT_TITLE_MAX_BYTES} bytes")
        return value

    @field_validator("state")
    @classmethod
    def _validate_state(cls, value: str | None) -> str | None:
        """Accept the lifecycle states chat creation accepts."""
        if value is not None:
            _require_utf8(value)
        return _validate_conversation_state(value)

    @model_validator(mode="after")
    def _normalize_binding_and_lineage(self) -> ChatImportRequest:
        """Normalize the assistant binding the way chat creation does, and require a fork's parent."""
        if self.forked_from_message_id is not None and self.parent_conversation_id is None:
            raise ValueError("forked_from_message_id requires parent_conversation_id")

        if all(
            value is None
            for value in (self.character_id, self.assistant_kind, self.assistant_id, self.persona_memory_mode)
        ):
            return self
        if self.assistant_kind is None:
            if self.character_id is None:
                raise ValueError("Provide either character_id or assistant_kind + assistant_id.")
            self.assistant_kind = "character"
        if self.assistant_kind == "character":
            if self.persona_memory_mode is not None:
                raise ValueError("persona_memory_mode is only valid for persona chats.")
            if self.character_id is None:
                if not self.assistant_id:
                    raise ValueError("Character chats require character_id or a numeric assistant_id.")
                try:
                    self.character_id = int(self.assistant_id)
                except ValueError as exc:
                    raise ValueError("Character assistant_id must be numeric.") from exc
                if self.character_id <= 0:
                    raise ValueError("Character assistant_id must be a positive integer.")
            self.assistant_id = str(self.character_id)
            return self
        if not self.assistant_id:
            raise ValueError("Persona chats require assistant_id.")
        if self.character_id is not None:
            raise ValueError("character_id is not valid for persona chats.")
        return self


__all__ = [
    "CHAT_IMPORT_ID_MAX_LENGTH",
    "CHAT_IMPORT_MAX_IMAGES_PER_MESSAGE",
    "CHAT_IMPORT_MAX_MESSAGES",
    "CHAT_IMPORT_MESSAGE_ID_PATTERN",
    "CHAT_IMPORT_TITLE_MAX_BYTES",
    "ChatImportMessage",
    "ChatImportRequest",
    "ImportTimestamp",
]
