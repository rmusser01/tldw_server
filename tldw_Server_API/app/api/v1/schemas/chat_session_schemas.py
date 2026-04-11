# chat_session_schemas.py
"""
Pydantic schemas for character chat sessions and messages.
"""

from datetime import datetime
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator
from tldw_Server_API.app.core.LLM_Calls.routing.models import RoutingOverride

ALLOWED_CONVERSATION_STATES = ("in-progress", "resolved", "backlog", "non-viable")
ALLOWED_ASSISTANT_KINDS = ("character", "persona")
ALLOWED_PERSONA_MEMORY_MODES = ("read_only", "read_write")


# ========================================================================
# Chat Session Schemas
# ========================================================================


def _validate_conversation_state(value: Optional[str]) -> Optional[str]:
    """Shared validator for conversation state field."""
    if value is None:
        return None
    normalized = value.strip().lower()
    if not normalized:
        raise ValueError("state cannot be empty")
    if normalized not in ALLOWED_CONVERSATION_STATES:
        raise ValueError(f"Invalid state '{value}'. Allowed: {', '.join(ALLOWED_CONVERSATION_STATES)}")
    return normalized

class ChatSessionCreate(BaseModel):
    """Schema for creating a new chat session."""
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

    model_config = {"json_schema_extra": {
        "example": {
            "character_id": 1,
            "title": "Evening Chat with Assistant"
        }
    }}

    @field_validator("state")
    @classmethod
    def _validate_state(cls, value: Optional[str]) -> Optional[str]:
        return _validate_conversation_state(value)

    @model_validator(mode="after")
    def _normalize_assistant_identity(self) -> "ChatSessionCreate":
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

        if (
            self.persona_memory_mode is not None
            and self.persona_memory_mode not in ALLOWED_PERSONA_MEMORY_MODES
        ):
            raise ValueError(
                f"Invalid persona_memory_mode '{self.persona_memory_mode}'. "
                f"Allowed: {', '.join(ALLOWED_PERSONA_MEMORY_MODES)}"
            )
        return self


class ChatSessionUpdate(BaseModel):
    """Schema for updating a chat session."""
    title: Optional[str] = Field(None, description="New title for the chat")
    rating: Optional[int] = Field(None, ge=1, le=5, description="Rating for the conversation")
    state: Optional[str] = Field(None, description="Lifecycle state for the conversation")
    topic_label: Optional[str] = Field(None, description="Primary topic label for the conversation")
    cluster_id: Optional[str] = Field(None, description="Cluster/group identifier for navigation")
    source: Optional[str] = Field(None, description="Source of the conversation (e.g., email, issue)")
    external_ref: Optional[str] = Field(None, description="External reference/link for the conversation")

    model_config = {"json_schema_extra": {
        "example": {
            "title": "Updated Chat Title",
            "rating": 5
        }
    }}

    @field_validator("state")
    @classmethod
    def _validate_state(cls, value: Optional[str]) -> Optional[str]:
        return _validate_conversation_state(value)


class ChatSessionResponse(BaseModel):
    """Schema for chat session responses."""
    id: str = Field(..., description="UUID of the chat session")
    scope_type: Literal["global", "workspace"] = Field(
        "global",
        description="Conversation scope type",
    )
    workspace_id: Optional[str] = Field(
        None,
        description="Workspace ID when scope_type='workspace'",
    )
    character_id: int | None = Field(None, description="ID of the associated character")
    assistant_kind: Literal["character", "persona"] | None = Field(
        None,
        description="Normalized assistant identity kind for the chat",
    )
    assistant_id: str | None = Field(None, description="Normalized assistant identity ID for the chat")
    persona_memory_mode: Literal["read_only", "read_write"] | None = Field(
        None,
        description="Persona durable memory behavior for this chat",
    )
    title: Optional[str] = Field(None, description="Chat session title")
    rating: Optional[int] = Field(None, description="User rating of the conversation")
    state: str = Field("in-progress", description="Lifecycle state of the conversation")
    topic_label: Optional[str] = Field(None, description="Primary topic label for the conversation")
    cluster_id: Optional[str] = Field(None, description="Cluster/group identifier for navigation")
    source: Optional[str] = Field(None, description="Source of the conversation (e.g., email, issue)")
    external_ref: Optional[str] = Field(None, description="External reference/link for the conversation")
    created_at: datetime = Field(..., description="Creation timestamp")
    last_modified: datetime = Field(..., description="Last modification timestamp")
    message_count: Optional[int] = Field(0, description="Number of messages in the chat")
    version: int = Field(1, description="Version number for optimistic locking")
    parent_conversation_id: Optional[str] = Field(None, description="Parent conversation ID when forked")
    root_id: Optional[str] = Field(None, description="Root conversation ID for fork trees")
    forked_from_message_id: Optional[str] = Field(None, description="Source message ID for forked chats")
    settings: Optional[dict[str, Any]] = Field(
        None,
        description="Optional per-chat settings payload when explicitly requested.",
    )

    model_config = {"from_attributes": True}


class ChatSessionListResponse(BaseModel):
    """Schema for listing chat sessions."""
    chats: list[ChatSessionResponse] = Field(..., description="List of chat sessions")
    total: int = Field(..., description="Total number of chats")
    limit: int = Field(..., description="Number of items per page")
    offset: int = Field(..., description="Offset for pagination")


class ChatLinkedResearchRunResponse(BaseModel):
    """Compact research run status surfaced alongside a chat thread."""

    run_id: str = Field(..., description="Research run identifier")
    query: str = Field(..., description="Original research query")
    status: str = Field(..., description="Current research run status")
    phase: str = Field(..., description="Current research phase")
    control_state: str = Field(..., description="Current research control state")
    latest_checkpoint_id: Optional[str] = Field(None, description="Latest checkpoint ID, when present")
    updated_at: str = Field(..., description="Last research session update timestamp")

    model_config = {"from_attributes": True}


class ChatLinkedResearchRunsListResponse(BaseModel):
    """Compact linked deep research runs for a single chat thread."""

    runs: list[ChatLinkedResearchRunResponse] = Field(
        default_factory=list,
        description="Linked deep research runs for this chat",
    )


class ChatSettingsUpdate(BaseModel):
    """Schema for updating chat settings."""
    settings: dict[str, Any] = Field(..., description="Chat settings payload")


class ChatSettingsResponse(BaseModel):
    """Schema for chat settings responses."""
    conversation_id: str = Field(..., description="Conversation ID")
    settings: dict[str, Any] = Field(..., description="Stored chat settings")
    last_modified: datetime = Field(..., description="Settings last modified timestamp")
    warnings: Optional[list[str]] = Field(None, description="Optional warnings (e.g. greeting staleness)")


# ========================================================================
# Message Schemas
# ========================================================================

class MessageCreate(BaseModel):
    """Schema for creating a new message."""
    role: Literal["user", "assistant", "system"] = Field(..., description="Message sender role")
    content: Optional[str] = Field(
        None,
        description="Message content (required unless image_base64 is provided)",
        min_length=1,
    )
    parent_message_id: Optional[str] = Field(None, description="Parent message ID for branching")
    image_base64: Optional[str] = Field(None, description="Optional base64 encoded image")

    model_config = {"json_schema_extra": {
        "example": {
            "role": "user",
            "content": "Hello, how are you today?"
        }
    }}

    @model_validator(mode="after")
    def _validate_content_or_image(self) -> "MessageCreate":
        """Ensure at least one of content or image_base64 is provided.

        Allows image-only messages and enforces content non-empty when provided.
        """
        content_ok = bool(self.content and str(self.content).strip())
        image_ok = bool(self.image_base64 and str(self.image_base64).strip())
        if not content_ok and not image_ok:
            raise ValueError("Provide either non-empty content or image_base64.")
        return self


class MessageUpdate(BaseModel):
    """Schema for updating a message."""
    content: Optional[str] = Field(
        None,
        description="New message content",
        min_length=1,
    )
    pinned: Optional[bool] = Field(
        None,
        description="Optional pin state for this message.",
    )

    model_config = {"json_schema_extra": {
        "example": {
            "content": "Updated message content"
        }
    }}

    @model_validator(mode="after")
    def _validate_content_or_pin_update(self) -> "MessageUpdate":
        content_set = bool(self.content and str(self.content).strip())
        pin_set = self.pinned is not None
        if not content_set and not pin_set:
            raise ValueError("Provide either non-empty content or pinned.")
        return self


class MessageResponse(BaseModel):
    """Schema for message responses."""
    id: str = Field(..., description="UUID of the message")
    conversation_id: str = Field(..., description="ID of the parent conversation")
    parent_message_id: Optional[str] = Field(None, description="ID of parent message")
    sender: str = Field(..., description="Message sender (user/assistant/system)")
    content: str = Field(..., description="Message content")
    timestamp: datetime = Field(..., description="Message timestamp")
    ranking: Optional[int] = Field(None, description="Message ranking/rating")
    has_image: bool = Field(False, description="Whether message has an attached image")
    version: int = Field(1, description="Version number for optimistic locking")
    tool_calls: Optional[list[dict[str, Any]]] = Field(None, description="Tool calls associated with this message (if any)")
    metadata_extra: Optional[dict[str, Any]] = Field(None, description="Additional stored metadata for this message (if requested)")

    model_config = {"from_attributes": True}


class MessageListResponse(BaseModel):
    """Schema for listing messages."""
    messages: list[MessageResponse] = Field(..., description="List of messages")
    total: int = Field(..., description="Total number of messages")
    limit: int = Field(..., description="Number of items per page")
    offset: int = Field(..., description="Offset for pagination")


# ========================================================================
# Chat Completion Schemas
# ========================================================================

class CharacterChatCompletionV1Request(BaseModel):
    """Schema for character-specific chat completion requests."""
    message: str = Field(..., description="User message to respond to", min_length=1)
    max_tokens: Optional[int] = Field(500, ge=1, le=4096, description="Maximum tokens in response")
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0, description="Response randomness")
    stream: bool = Field(False, description="Enable streaming response")
    include_history: bool = Field(True, description="Include conversation history")
    history_limit: Optional[int] = Field(20, ge=0, le=100, description="Number of history messages to include")

    model_config = {"json_schema_extra": {
        "example": {
            "message": "What's your favorite color?",
            "max_tokens": 150,
            "temperature": 0.8,
            "stream": False
        }
    }}


class CharacterChatCompletionV1Response(BaseModel):
    """Schema for chat completion responses."""
    response: str = Field(..., description="AI response")
    message_id: str = Field(..., description="ID of the created message")
    usage: Optional[dict[str, int]] = Field(None, description="Token usage statistics")

    model_config = {"json_schema_extra": {
        "example": {
            "response": "My favorite color is blue, like the vast ocean!",
            "message_id": "550e8400-e29b-41d4-a716-446655440000",
            "usage": {
                "prompt_tokens": 45,
                "completion_tokens": 12,
                "total_tokens": 57
            }
        }
    }}


# ========================================================================
# Character Chat v2 + Prep Schemas (centralized)
# ========================================================================

class MessageSteeringPromptOverrides(BaseModel):
    """Optional prompt text overrides for single-turn message steering."""

    continue_as_user: Optional[str] = Field(
        None,
        max_length=2_000,
        description="Override prompt text used when continue_as_user is enabled.",
    )
    impersonate_user: Optional[str] = Field(
        None,
        max_length=2_000,
        description="Override prompt text used when impersonate_user is enabled.",
    )
    force_narrate: Optional[str] = Field(
        None,
        max_length=2_000,
        description="Override prompt text used when force_narrate is enabled.",
    )


class CharacterChatCompletionPrepRequest(BaseModel):
    """Prepare chat messages for use with the main Chat API.

    Controls pagination and optional user message appending for preparation.
    """
    include_character_context: bool = Field(True, description="Include character system context")
    limit: int = Field(100, ge=1, le=1000, description="Max messages to include")
    offset: int = Field(0, ge=0, description="Messages offset")
    append_user_message: Optional[str] = Field(
        None,
        description="Optional user message to append to the end",
        max_length=1_000_000,
    )
    directed_character_id: Optional[int] = Field(
        None,
        gt=0,
        description="Optional participant character ID to direct the next response to.",
    )
    continue_as_user: bool = Field(
        False,
        description="Single response: continue in the user's voice.",
    )
    impersonate_user: bool = Field(
        False,
        description="Single response: write the reply as if authored by the user.",
    )
    force_narrate: bool = Field(
        False,
        description="Single response: force narration style for the assistant reply.",
    )
    message_steering_prompts: Optional[MessageSteeringPromptOverrides] = Field(
        None,
        description="Optional prompt text overrides for steering instructions.",
    )
    prompt_preset: Optional[str] = Field(
        None,
        description="Optional single-turn prompt preset override.",
    )


class CharacterChatCompletionPrepResponse(BaseModel):
    chat_id: str
    character_id: int
    character_name: Optional[str] = None
    messages: list[dict[str, Any]]
    total: int
    usage_instructions: str = "Use these messages with POST /api/v1/chat/completions"


class CharacterChatCompletionV2Request(BaseModel):
    """Character Chat completion (v2) - builds context and calls a provider.

    Includes provider/model controls, optional appended user message,
    persistence toggle, and streaming control.
    """
    # Character/context controls
    include_character_context: bool = Field(True, description="Include character system context")
    limit: int = Field(100, ge=1, le=1000, description="Max messages to include")
    offset: int = Field(0, ge=0, description="Messages offset")
    append_user_message: Optional[str] = Field(
        None,
        description="Optional user message to append and (optionally) persist",
        max_length=1_000_000,
    )
    directed_character_id: Optional[int] = Field(
        None,
        gt=0,
        description="Optional participant character ID to direct this response to.",
    )
    save_to_db: Optional[bool] = Field(None, description="Persist appended user and assistant messages to this chat")
    # Message steering (single-response controls)
    continue_as_user: bool = Field(
        False,
        description="Single response: continue in the user's voice.",
    )
    impersonate_user: bool = Field(
        False,
        description="Single response: write the reply as if authored by the user.",
    )
    force_narrate: bool = Field(
        False,
        description="Single response: force narration style for the assistant reply.",
    )
    message_steering_prompts: Optional[MessageSteeringPromptOverrides] = Field(
        None,
        description="Optional prompt text overrides for steering instructions.",
    )
    mood_label: Optional[str] = Field(
        None,
        max_length=80,
        description="Optional mood/expression label to persist with this assistant turn.",
    )
    mood_confidence: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Optional confidence score for mood_label (0.0-1.0).",
    )
    mood_topic: Optional[str] = Field(
        None,
        max_length=200,
        description="Optional short topic cue associated with the detected mood.",
    )
    prompt_preset: Optional[str] = Field(
        None,
        description="Optional single-turn prompt preset override.",
    )
    # LLM controls
    provider: Optional[str] = Field(
        None,
        description="LLM provider (e.g., openai, anthropic, local-llm). When omitted, server default provider settings are used.",
    )
    routing: Optional[RoutingOverride] = Field(
        None,
        description=(
            "Optional server-side model-router overrides when model='auto', including routing "
            "mode, objective, provider boundary, and failure handling."
        ),
    )
    model: Optional[str] = Field(None, description="Model identifier. Defaults to a local test model if omitted.")
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0, description="Nucleus sampling probability")
    repetition_penalty: Optional[float] = Field(None, ge=0.0, le=3.0, description="Repetition penalty")
    stop: Optional[Union[str, list[str]]] = Field(None, description="Stop sequence(s)")
    max_tokens: Optional[int] = Field(None, description="Max tokens in the completion")
    tools: Optional[list[dict[str, Any]]] = Field(None, description="Tool definitions")
    tool_choice: Optional[dict[str, Any]] = Field(None, description="Tool choice specification")
    stream: Optional[bool] = Field(False, description="If true, stream the assistant response (SSE)")

    @model_validator(mode="after")
    def _validate_routing_requires_auto_model(self) -> "CharacterChatCompletionV2Request":
        if self.routing is not None and str(self.model or "").strip().lower() != "auto":
            raise ValueError("routing overrides are only allowed when model='auto'")
        return self


class CharacterChatCompletionV2Response(BaseModel):
    chat_id: str
    character_id: int
    provider: str
    model: Optional[str] = None
    saved: bool = False
    user_message_id: Optional[str] = None
    assistant_message_id: Optional[str] = None
    assistant_content: str
    speaker_character_id: Optional[int] = None
    speaker_character_name: Optional[str] = None
    mood_label: Optional[str] = None
    mood_confidence: Optional[float] = None
    mood_topic: Optional[str] = None
    lorebook_diagnostics: Optional[list[dict[str, Any]]] = Field(
        None,
        description="Lorebook/world book trigger diagnostics for this turn",
    )


# New: Persist streamed assistant content after SSE
class CharacterChatStreamPersistRequest(BaseModel):
    """Persist assistant content produced via streaming.

    Use after a streamed completion where the assistant content was not persisted.
    """
    assistant_content: str = Field(..., min_length=1, max_length=1_000_000, description="Assistant text to persist (max 1MB)")
    assistant_message_id: Optional[str] = Field(
        None,
        min_length=1,
        max_length=255,
        description="Optional stable assistant message ID used for idempotent persist retries.",
    )
    user_message_id: Optional[str] = Field(None, description="Optional parent user message id to link threading")
    speaker_character_id: Optional[int] = Field(
        None,
        gt=0,
        description="Optional speaker character ID for multi-character chats.",
    )
    speaker_character_name: Optional[str] = Field(
        None,
        description="Optional speaker character display name for multi-character chats.",
    )
    mood_label: Optional[str] = Field(
        None,
        max_length=80,
        description="Optional mood/expression label to persist in message metadata.",
    )
    mood_confidence: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Optional confidence score for mood_label (0.0-1.0).",
    )
    mood_topic: Optional[str] = Field(
        None,
        max_length=200,
        description="Optional short topic cue associated with the detected mood.",
    )
    tool_calls: Optional[list[dict[str, Any]]] = Field(None, description="Optional tool_calls metadata to store")
    usage: Optional[dict[str, int]] = Field(None, description="Optional token usage stats: prompt_tokens, completion_tokens, total_tokens")
    chat_rating: Optional[int] = Field(None, ge=1, le=5, description="Optional conversation rating to set (1-5)")
    ranking: Optional[int] = Field(None, description="Optional ranking for the assistant message")

    @field_validator("assistant_message_id")
    @classmethod
    def validate_assistant_message_id(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("assistant_message_id cannot be blank")
        return normalized


class CharacterChatStreamPersistResponse(BaseModel):
    chat_id: str
    assistant_message_id: Optional[str] = None
    saved: bool = False


# ========================================================================
# Export Schemas
# ========================================================================

class ChatExportFormat(BaseModel):
    """Schema for chat export format options."""
    format: Literal["json", "markdown", "text"] = Field("json", description="Export format")
    include_metadata: bool = Field(True, description="Include chat metadata")
    include_character: bool = Field(True, description="Include character information")


class ChatHistoryExport(BaseModel):
    """Schema for exported chat history."""
    chat_id: str = Field(..., description="Chat session ID")
    character_name: str = Field(..., description="Character name")
    character_id: int = Field(..., description="Character ID")
    title: Optional[str] = Field(None, description="Chat title")
    created_at: datetime = Field(..., description="Creation timestamp")
    messages: list[dict[str, Any]] = Field(..., description="Message history")
    metadata: Optional[dict[str, Any]] = Field(None, description="Additional metadata")

    model_config = {"json_schema_extra": {
        "example": {
            "chat_id": "550e8400-e29b-41d4-a716-446655440000",
            "character_name": "Assistant",
            "character_id": 1,
            "title": "Evening Chat",
            "created_at": "2024-01-15T20:30:00Z",
            "messages": [
                {
                    "role": "user",
                    "content": "Hello!",
                    "timestamp": "2024-01-15T20:30:00Z"
                },
                {
                    "role": "assistant",
                    "content": "Hello! How can I help you?",
                    "timestamp": "2024-01-15T20:30:05Z"
                }
            ]
        }
    }}


# ========================================================================
# Error Response Schemas
# ========================================================================

class ChatErrorResponse(BaseModel):
    """Schema for chat-related error responses."""
    error: str = Field(..., description="Error type")
    detail: str = Field(..., description="Detailed error message")
    chat_id: Optional[str] = Field(None, description="Related chat ID if applicable")
    message_id: Optional[str] = Field(None, description="Related message ID if applicable")

    model_config = {"json_schema_extra": {
        "example": {
            "error": "ChatNotFound",
            "detail": "Chat session with ID '550e8400-e29b-41d4-a716-446655440000' not found",
            "chat_id": "550e8400-e29b-41d4-a716-446655440000"
        }
    }}


# ========================================================================
# Filter Schemas
# ========================================================================

class CharacterTagFilter(BaseModel):
    """Schema for filtering characters by tags."""
    tags: list[str] = Field(..., description="Tags to filter by", min_length=1)
    match_all: bool = Field(False, description="Require all tags to match (AND) vs any tag (OR)")

    model_config = {"json_schema_extra": {
        "example": {
            "tags": ["fantasy", "wizard"],
            "match_all": False
        }
    }}


# ========================================================================
# Greeting List Picker Schemas (PRD 1 Stage A2)
# ========================================================================

class GreetingItem(BaseModel):
    """A single greeting entry from the character card."""
    index: int = Field(..., description="Zero-based index into the greetings list")
    text: str = Field(..., description="Full greeting text")
    preview: str = Field(..., description="First 120 characters of the greeting")


class GreetingListResponse(BaseModel):
    """Response for GET /{chat_id}/greetings."""
    chat_id: str
    character_id: Optional[str] = None
    character_name: Optional[str] = None
    greetings: list[GreetingItem] = Field(default_factory=list)
    current_selection: Optional[int] = Field(None, description="Currently selected greeting index")
    staleness_warning: Optional[str] = Field(None, description="Warning if greetings changed since chat creation")


class GreetingSelectRequest(BaseModel):
    """Request body for PUT /{chat_id}/greetings/select."""
    index: int = Field(..., description="Zero-based index of the greeting to select")


class GreetingSelectResponse(BaseModel):
    """Response for PUT /{chat_id}/greetings/select."""
    chat_id: str
    selected_index: int
    greeting_preview: str
    checksum_updated: bool


# ========================================================================
# Author Note Info Schemas (PRD 4 Stage D2)
# ========================================================================

class AuthorNoteInfoResponse(BaseModel):
    """Response for GET /{chat_id}/author-note/info."""
    chat_id: str
    text: str = Field("", description="Author note text for UI display")
    text_for_prompt: str = Field("", description="Author note text as injected into LLM prompt")
    tokens_estimated: int = Field(0, description="Estimated token count for UI text")
    tokens_for_prompt: int = Field(0, description="Estimated token count for prompt text")
    budget: int = Field(0, description="Token budget for author note")
    truncated: bool = Field(False, description="Whether the note exceeds the token budget")
    enabled: bool = Field(True, description="Whether the author note is enabled")
    gm_only: bool = Field(False, description="Whether the note is GM-only (visible but not in prompt)")
    exclude_from_prompt: bool = Field(False, description="Whether the note is excluded from prompt")
    scope: str = Field("shared", description="Memory scope: shared, character, or both")
    source: str = Field("none", description="Source of the note text: settings, character_default, or none")
    warnings: list[str] = Field(default_factory=list)


# ========================================================================
# Lorebook Diagnostic Export Schemas (PRD 6 Stage F2)
# ========================================================================

class DiagnosticTurnEntry(BaseModel):
    """A single turn's lorebook diagnostics."""
    message_id: str
    timestamp: Optional[str] = None
    turn_number: int
    message_preview: str = Field("", description="First 120 characters of the assistant message")
    diagnostics: list[dict] = Field(default_factory=list)


class LorebookDiagnosticExportResponse(BaseModel):
    """Response for GET /{chat_id}/diagnostics/lorebook."""
    chat_id: str
    character_id: Optional[str] = None
    total_turns_with_diagnostics: int = 0
    turns: list[DiagnosticTurnEntry] = Field(default_factory=list)
    page: int = 1
    size: int = 50


# ========================================================================
# Preset Editor Schemas (PRD 2 Stage B2)
# ========================================================================

class PresetTokenInfo(BaseModel):
    """Template token with description."""
    token: str = Field(..., description="Template token, e.g. '{{char}}'")
    description: str = Field(..., description="What the token resolves to")


class PresetDetail(BaseModel):
    """Full details of a prompt preset."""
    preset_id: str
    name: str
    builtin: bool = Field(False, description="Whether this is a built-in preset")
    section_order: list[str] = Field(default_factory=list, description="Ordered list of section keys")
    section_templates: dict[str, str] = Field(default_factory=dict, description="Section key → template string")
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class PresetListResponse(BaseModel):
    """Response for GET /presets."""
    presets: list[PresetDetail] = Field(default_factory=list)


class PresetCreate(BaseModel):
    """Request body for POST /presets."""
    preset_id: str = Field(..., min_length=1, max_length=128, description="Unique identifier for the preset")
    name: str = Field(..., min_length=1, max_length=256, description="Display name")
    section_order: list[str] = Field(..., description="Ordered list of section keys")
    section_templates: dict[str, str] = Field(..., description="Section key → template string")

    @field_validator("preset_id")
    @classmethod
    def validate_preset_id(cls, v: str) -> str:
        v = v.strip()
        if not v or v in ("default", "st_default"):
            raise ValueError("Cannot use a built-in preset ID")
        return v


class PresetUpdate(BaseModel):
    """Request body for PUT /presets/{preset_id}."""
    name: Optional[str] = Field(None, min_length=1, max_length=256)
    section_order: Optional[list[str]] = None
    section_templates: Optional[dict[str, str]] = None


#
# End of chat_session_schemas.py
######################################################################################################################
