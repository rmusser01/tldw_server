# Server_API/app/api/schemas/chat_request_models.py
# Description: This code provides schema models for the /chat API endpoints
#
# Imports
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


# Core-owned since TASK-13950.15; re-exported for the API layer.
from tldw_Server_API.app.core.Chat.chat_request_schemas import (  # noqa: F401 - re-exported (TASK-13950.15)
    ALL_SUPPORTED_PROVIDER_NAMES_LIST,
    normalize_catalog_provider_for_chat,
    _SERVER_MANAGED_CHAT_REQUEST_FIELDS,
    _SERVER_MANAGED_CHAT_REQUEST_COMPACT_FIELDS,
    _is_server_managed_chat_request_field,
    SUPPORTED_API_ENDPOINTS,
    MAX_PARAMETER_DEPTH,
    MAX_PARAMETER_SIZE_BYTES,
    _calculate_json_depth,
    _validate_json_schema_structure,
    FunctionDefinition,
    ToolDefinition,
    GeminiNativeToolDefinition,
    ToolChoiceFunction,
    ToolChoiceOption,
    MAX_MESSAGE_CONTENT_LENGTH,
    MAX_MESSAGE_NAME_LENGTH,
    ChatCompletionRequestMessageContentPartText,
    ChatCompletionRequestMessageContentPartImageURL,
    ChatCompletionRequestMessageContentPartImage,
    ChatCompletionRequestMessageContentPart,
    BaseMessage,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
    MAX_FUNCTION_ARGUMENTS_SIZE,
    FunctionCall,
    ToolCallFunctionPayload,
    MAX_TOOL_CALL_ID_LENGTH,
    ChatCompletionMessageToolCallParam,
    ChatCompletionAssistantMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionMessageParam,
    ResponseFormatJsonSchemaSpec,
    ResponseFormat,
    TLDWContinuationSpec,
    ResearchChatContextOutlineSection,
    ResearchChatContextClaim,
    ResearchChatContextVerificationSummary,
    ResearchChatContextSourceTrustSummary,
    ResearchChatContext,
    ChatCompletionRequest,
)

model_config = ConfigDict(extra="allow", from_attributes=True)


class ChatCompletionResponseChoice(BaseModel):
    """One non-streaming chat completion choice in the OpenAI-compatible response."""

    index: Optional[int] = Field(default=None, description="Zero-based choice index.")
    message: Optional[dict[str, Any]] = Field(
        default=None,
        description="Assistant message payload returned by the selected provider.",
    )
    finish_reason: Optional[str] = Field(
        default=None,
        description="Provider-reported completion stop reason.",
    )
    logprobs: Optional[Any] = Field(
        default=None,
        description="Provider-specific log probability payload when requested.",
    )

    model_config = ConfigDict(extra="allow")


class ChatCompletionUsage(BaseModel):
    """Token usage summary for a non-streaming chat completion response."""

    prompt_tokens: Optional[int] = Field(default=None, ge=0)
    completion_tokens: Optional[int] = Field(default=None, ge=0)
    total_tokens: Optional[int] = Field(default=None, ge=0)

    model_config = ConfigDict(extra="allow")


class ChatCompletionResponse(BaseModel):
    """Document the non-streaming OpenAI-compatible chat completion response shape."""

    id: Optional[str] = Field(default=None, description="Provider completion identifier.")
    object: Optional[str] = Field(default=None, description="OpenAI-compatible object type.")
    created: Optional[int] = Field(default=None, description="Unix timestamp for response creation.")
    model: Optional[str] = Field(default=None, description="Model used for generation.")
    choices: list[ChatCompletionResponseChoice] = Field(
        default_factory=list,
        description="Completion choices returned by the selected provider.",
    )
    usage: Optional[ChatCompletionUsage] = Field(
        default=None,
        description="Token usage reported by the selected provider.",
    )
    tldw_conversation_id: Optional[str] = Field(
        default=None,
        description="Conversation identifier persisted by tldw_server when chat history is enabled.",
    )
    meta: Optional[dict[str, Any]] = Field(
        default=None,
        description="tldw_server metadata such as persona diagnostics.",
    )

    model_config = ConfigDict(extra="allow")

#
# --- RAG Context Schemas for Citation Persistence ---
# These schemas define the structure for storing RAG context with messages


class RagContextDocument(BaseModel):
    """A single retrieved document within a RAG context."""

    id: Optional[str] = Field(None, description="Document or media ID")
    source_type: Optional[str] = Field(
        None,
        description="Type of source: media_db, notes, characters, chats, kanban"
    )
    title: Optional[str] = Field(None, description="Document title")
    score: Optional[float] = Field(None, description="Relevance score (0-1)")
    chunk_id: Optional[str] = Field(None, description="Chunk identifier if chunk-level retrieval")
    excerpt: Optional[str] = Field(None, description="Relevant excerpt from the document")
    url: Optional[str] = Field(None, description="URL or path to original document")
    page_number: Optional[int] = Field(None, description="Page number if applicable")
    line_range: Optional[list[int]] = Field(
        None,
        description="Line range [start, end] of the excerpt in the source",
        min_length=2,
        max_length=2
    )
    metadata: Optional[dict[str, Any]] = Field(
        None,
        description="Additional metadata from the source"
    )

    model_config = ConfigDict(extra="allow")


class RagContext(BaseModel):
    """
    RAG context to be stored with a message for citation persistence.
    This is stored in message_metadata.extra_json under the 'rag_context' key.
    """

    search_query: str = Field(..., description="The original search query")
    search_mode: Optional[str] = Field(
        "hybrid",
        description="Search mode used: fts, vector, or hybrid"
    )
    settings_snapshot: Optional[dict[str, Any]] = Field(
        None,
        description="Snapshot of key RAG settings used for this search"
    )
    retrieved_documents: list[RagContextDocument] = Field(
        default_factory=list,
        description="List of retrieved documents with their metadata and scores"
    )
    generated_answer: Optional[str] = Field(
        None,
        description="The AI-generated answer (if generation was enabled)"
    )
    citations: Optional[list[dict[str, Any]]] = Field(
        None,
        description="Citation metadata (academic and chunk-level)"
    )
    claims_verified: Optional[list[dict[str, Any]]] = Field(
        None,
        description="Results from claims verification if enabled"
    )
    timestamp: Optional[str] = Field(
        None,
        description="ISO timestamp of when the search was performed"
    )
    feedback_id: Optional[str] = Field(
        None,
        description="Feedback tracking ID for analytics"
    )

    model_config = ConfigDict(
        extra="allow",
        json_schema_extra={
            "example": {
                "search_query": "What are the key findings from the Q3 report?",
                "search_mode": "hybrid",
                "settings_snapshot": {
                    "top_k": 10,
                    "enable_reranking": True,
                    "enable_citations": True
                },
                "retrieved_documents": [
                    {
                        "id": "media_123",
                        "source_type": "media_db",
                        "title": "Q3 Financial Report 2025",
                        "score": 0.92,
                        "excerpt": "Revenue increased by 15% compared to Q2...",
                        "page_number": 5
                    }
                ],
                "generated_answer": "The key findings from the Q3 report include...",
                "timestamp": "2025-01-25T12:00:00Z"
            }
        }
    )


#
# End of chat_request_schemas.py
#######################################################################################################################
