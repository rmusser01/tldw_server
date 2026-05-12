# chatbook_schemas.py
# Description: Pydantic schemas for chatbook import/export operations
#
"""
Chatbook Schemas
----------------

Pydantic models for chatbook creation, import, export, and preview operations.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from tldw_Server_API.app.api.v1.schemas.pagination import OffsetPaginationMeta

# Import shared enums from the canonical models module to avoid divergence.
# ConflictResolution is intentionally redefined here to constrain API input
# to only the currently-supported strategies (skip, rename).
from tldw_Server_API.app.core.Chatbooks.chatbook_models import (
    ChatbookVersion,
    ContentType,
    ExportStatus,
    ImportStatus,
)


def _default_offset_pagination_aliases(response):
    if response.has_more is None:
        response.has_more = response.pagination.has_more
    if response.next_offset is None:
        response.next_offset = response.pagination.next_offset
    return response


class ConflictResolution(str, Enum):
    """How to handle conflicts during import (API-constrained subset)."""
    SKIP = "skip"          # Skip conflicting items
    RENAME = "rename"      # Rename imported items


class ChatbookImportSourceFormat(str, Enum):
    """Supported source formats for Chatbooks import surfaces."""
    CHATBOOK = "chatbook"
    OPENWEBUI_JSON = "openwebui_json"
    OPENWEBUI_DB = "openwebui_db"


class MediaQuality(str, Enum):
    """Media quality levels for export."""
    THUMBNAIL = "thumbnail"
    COMPRESSED = "compressed"
    ORIGINAL = "original"


# Allowed values for job listing sort field (prevents SQL injection)
JobOrderByField = Literal["created_at", "status", "chatbook_name", "updated_at", "completed_at"]

# Combined status type for job queries (supports both export and import)
JobStatusFilter = Union[ExportStatus, ImportStatus, None]


# Request Schemas

class CreateChatbookRequest(BaseModel):
    """Request for creating a chatbook."""
    name: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Name of the chatbook"
    )
    description: str = Field(
        ...,
        max_length=5000,
        description="Description of the chatbook"
    )
    content_selections: dict[ContentType, list[str]] = Field(
        ...,
        description="Content to include by type and IDs"
    )
    author: Optional[str] = Field(
        None,
        max_length=255,
        description="Author name"
    )
    include_media: bool = Field(False, description="Include media files")
    media_quality: MediaQuality = Field(
        MediaQuality.COMPRESSED,
        description="Media quality level"
    )
    include_embeddings: bool = Field(False, description="Include embeddings")
    include_generated_content: bool = Field(True, description="Include generated documents")
    tags: list[str] = Field(default_factory=list, max_length=50, description="Chatbook tags")
    categories: list[str] = Field(default_factory=list, max_length=20, description="Chatbook categories")
    async_mode: bool = Field(False, description="Run as background job")

    @field_validator('tags', 'categories', mode='before')
    @classmethod
    def validate_string_lists(cls, v):
        """Validate that list items are reasonable length."""
        if v is None:
            return []
        if isinstance(v, list):
            for item in v:
                if isinstance(item, str) and len(item) > 50:
                    raise ValueError(f"Item '{item[:20]}...' exceeds maximum length of 50 characters")
        return v

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "name": "My Research Chatbook",
            "description": "Collection of research conversations and notes",
            "content_selections": {
                "conversation": ["conv123", "conv456"],
                "note": ["note789"],
                "character": ["char001"]
            },
            "author": "Jane Doe",
            "include_media": False,
            "media_quality": MediaQuality.COMPRESSED.value,
            "include_embeddings": False,
            "include_generated_content": True,
            "tags": ["research", "AI"],
            "categories": ["Work"],
            "async_mode": False
        }
    })


class ImportChatbookRequest(BaseModel):
    """Request for importing a chatbook."""
    source_format: ChatbookImportSourceFormat = Field(
        ChatbookImportSourceFormat.CHATBOOK,
        description="Uploaded source format"
    )
    content_selections: Optional[dict[ContentType, list[str]]] = Field(
        None,
        description="Specific content to import, or None for all"
    )
    conflict_resolution: ConflictResolution = Field(
        ConflictResolution.SKIP,
        description="How to handle conflicts"
    )
    prefix_imported: bool = Field(
        False,
        description="Add [Imported] prefix to items"
    )
    import_media: bool = Field(False, description="Import media files (not supported yet)")
    import_embeddings: bool = Field(False, description="Import embeddings (not supported yet)")
    async_mode: bool = Field(False, description="Run as background job")
    selected_openwebui_user_id: Optional[str] = Field(
        None,
        description="Selected OpenWebUI source user id for database imports"
    )

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "conflict_resolution": "skip",
            "prefix_imported": True,
            "import_media": False,
            "import_embeddings": False,
            "async_mode": False
        }
    })


# Response Schemas

class ContentItemResponse(BaseModel):
    """Individual content item in a chatbook."""
    id: str
    type: ContentType
    title: str
    description: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    tags: list[str] = Field(default_factory=list, max_length=50)
    metadata: dict[str, Any] = Field(default_factory=dict)
    file_path: Optional[str] = None
    checksum: Optional[str] = None


class ChatbookManifestResponse(BaseModel):
    """Chatbook manifest information."""
    version: ChatbookVersion
    name: str
    description: str
    author: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    export_id: Optional[str] = None

    # Content summary
    content_items: list[ContentItemResponse] = Field(default_factory=list)

    # Configuration
    include_media: bool = False
    include_embeddings: bool = False
    include_generated_content: bool = True
    media_quality: str = "compressed"
    max_file_size_mb: int = 100

    # Statistics
    total_conversations: int = 0
    total_notes: int = 0
    total_characters: int = 0
    total_media_items: int = 0
    total_prompts: int = 0
    total_evaluations: int = 0
    total_embeddings: int = 0
    total_world_books: int = 0
    total_dictionaries: int = 0
    total_documents: int = 0
    total_size_bytes: int = 0

    # Metadata
    tags: list[str] = Field(default_factory=list)
    categories: list[str] = Field(default_factory=list)
    language: str = "en"
    license: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    truncation: dict[str, Any] = Field(default_factory=dict)


class ExportJobResponse(BaseModel):
    """Export job status."""
    job_id: str
    status: ExportStatus
    chatbook_name: str
    output_path: Optional[str] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    progress_percentage: int = Field(default=0, ge=0, le=100)
    total_items: int = Field(default=0, ge=0)
    processed_items: int = Field(default=0, ge=0)
    file_size_bytes: Optional[int] = Field(default=None, ge=0)
    download_url: Optional[str] = None
    expires_at: Optional[datetime] = None


class ImportJobResponse(BaseModel):
    """Import job status."""
    job_id: str
    status: ImportStatus
    chatbook_path: str
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    progress_percentage: int = Field(default=0, ge=0, le=100)
    total_items: int = Field(default=0, ge=0)
    processed_items: int = Field(default=0, ge=0)
    successful_items: int = Field(default=0, ge=0)
    failed_items: int = Field(default=0, ge=0)
    skipped_items: int = Field(default=0, ge=0)
    conflicts: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class ImportConflictResponse(BaseModel):
    """Details about an import conflict."""
    item_id: str
    item_type: ContentType
    item_title: str
    existing_id: str
    existing_title: str
    suggested_resolution: ConflictResolution
    user_resolution: Optional[ConflictResolution] = None
    new_title: Optional[str] = None


class OpenWebUIPreviewChatItem(BaseModel):
    """Lightweight preview row for one OpenWebUI chat."""
    external_ref: str
    title: str
    message_count: int = Field(ge=0)
    branched: bool = False
    duplicate: bool = False
    warning_count: int = Field(default=0, ge=0)


class OpenWebUIImportPreview(BaseModel):
    """OpenWebUI import preview counts."""
    chat_count: int = Field(default=0, ge=0)
    message_count: int = Field(default=0, ge=0)
    branched_chat_count: int = Field(default=0, ge=0)
    duplicate_chat_count: int = Field(default=0, ge=0)
    attachment_reference_count: int = Field(default=0, ge=0)
    malformed_chat_count: int = Field(default=0, ge=0)
    warnings: list[str] = Field(default_factory=list)
    items: list[OpenWebUIPreviewChatItem] = Field(default_factory=list)


class OpenWebUIDatabaseUserPreview(BaseModel):
    """OpenWebUI database preview counts for one source user."""
    source_user_id: str
    display_label: str
    email: Optional[str] = None
    chat_count: int = Field(default=0, ge=0)
    folder_count: int = Field(default=0, ge=0)
    message_count: int = Field(default=0, ge=0)
    branched_chat_count: int = Field(default=0, ge=0)
    duplicate_chat_count: int = Field(default=0, ge=0)
    archived_chat_count: int = Field(default=0, ge=0)
    pinned_chat_count: int = Field(default=0, ge=0)
    attachment_reference_count: int = Field(default=0, ge=0)
    warning_count: int = Field(default=0, ge=0)
    warnings: list[str] = Field(default_factory=list)


class OpenWebUIDatabasePreview(BaseModel):
    """OpenWebUI database import preview grouped by source user."""
    user_count: int = Field(default=0, ge=0)
    users: list[OpenWebUIDatabaseUserPreview] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class OpenWebUIImportResult(BaseModel):
    """OpenWebUI import result counts."""
    imported_chats: int = Field(default=0, ge=0)
    skipped_chats: int = Field(default=0, ge=0)
    failed_chats: int = Field(default=0, ge=0)
    imported_messages: int = Field(default=0, ge=0)
    skipped_messages: int = Field(default=0, ge=0)
    duplicate_chats: int = Field(default=0, ge=0)
    warnings: list[str] = Field(default_factory=list)


class OpenWebUIDatabaseImportResult(OpenWebUIImportResult):
    """OpenWebUI database import result for a selected source user."""
    selected_user_id: str
    selected_user_label: str
    mirrored_folders: int = Field(default=0, ge=0)
    folder_links: int = Field(default=0, ge=0)


class OpenWebUIHydrationScopeRequest(BaseModel):
    """Scope for OpenWebUI attachment hydration over imported tldw conversations."""
    conversation_ids: list[str] = Field(
        default_factory=list,
        max_length=1000,
        description="Imported tldw conversation ids to scan. Empty means no conversations are selected.",
    )
    source_user_id: Optional[str] = Field(
        default=None,
        min_length=1,
        max_length=255,
        description="Optional OpenWebUI source user id used for chat_file fallback lookups.",
    )

    @field_validator("conversation_ids")
    @classmethod
    def validate_conversation_ids(cls, value: list[str]) -> list[str]:
        """Trim and reject empty conversation ids."""
        cleaned: list[str] = []
        for item in value:
            text = str(item).strip()
            if not text:
                raise ValueError("conversation_ids must not contain empty values")
            cleaned.append(text)
        return cleaned

    @field_validator("source_user_id")
    @classmethod
    def validate_source_user_id(cls, value: Optional[str]) -> Optional[str]:
        """Trim and reject blank OpenWebUI source user ids."""
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            raise ValueError("source_user_id must not be empty")
        return text


class OpenWebUIHydrationPreviewRequest(BaseModel):
    """Request to preview OpenWebUI attachment hydration."""
    openwebui_data_root: str = Field(..., min_length=1, max_length=4096)
    scope: OpenWebUIHydrationScopeRequest = Field(default_factory=OpenWebUIHydrationScopeRequest)
    process_supported_files: bool = Field(
        default=False,
        description="Preview whether supported non-image files would be processed after registration.",
    )

    @field_validator("openwebui_data_root")
    @classmethod
    def validate_openwebui_data_root(cls, value: str) -> str:
        """Trim and reject empty roots."""
        text = str(value).strip()
        if not text:
            raise ValueError("openwebui_data_root must not be empty")
        return text


class OpenWebUIHydrationJobRequest(OpenWebUIHydrationPreviewRequest):
    """Request to enqueue an OpenWebUI attachment hydration job."""


class OpenWebUIHydrationItemResponse(BaseModel):
    """One user-safe OpenWebUI attachment hydration preview/status item."""
    conversation_id: Optional[str] = None
    message_id: Optional[str] = None
    file_id: Optional[str] = None
    status: str
    warning_code: Optional[str] = None
    raw_ref_index: Optional[int] = None
    source: Optional[str] = None
    raw_ref_shape: Optional[str] = None
    job_id: Optional[str] = None
    source_key: Optional[str] = None
    message_image_position: Optional[int] = None
    file_kind: Optional[str] = None
    mime_type: Optional[str] = None
    media_id: Optional[int] = None
    media_file_id: Optional[str] = None
    checksum: Optional[str] = None
    processing_status: Optional[str] = None


class OpenWebUIHydrationSummaryResponse(BaseModel):
    """Counts for an OpenWebUI attachment hydration preview or job result."""
    referenced_files: int = Field(default=0, ge=0)
    returned_items: int = Field(default=0, ge=0)
    omitted_items: int = Field(default=0, ge=0)
    resolved_files: int = Field(default=0, ge=0)
    image_files: int = Field(default=0, ge=0)
    media_files: int = Field(default=0, ge=0)
    missing_files: int = Field(default=0, ge=0)
    unsupported_files: int = Field(default=0, ge=0)
    failed_files: int = Field(default=0, ge=0)
    hydrated_images: int = Field(default=0, ge=0)
    registered_media_files: int = Field(default=0, ge=0)
    already_hydrated: int = Field(default=0, ge=0)
    processed_files: int = Field(default=0, ge=0)
    warning_count: int = Field(default=0, ge=0)


class OpenWebUIHydrationPreviewResponse(BaseModel):
    """Response for OpenWebUI attachment hydration preview."""
    scope: OpenWebUIHydrationScopeRequest
    process_supported_files: bool = False
    summary: OpenWebUIHydrationSummaryResponse
    items: list[OpenWebUIHydrationItemResponse] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class OpenWebUIHydrationJobResponse(BaseModel):
    """Response for OpenWebUI attachment hydration job creation/status."""
    job_id: str
    job_uuid: Optional[str] = None
    status: str
    domain: str = "chatbooks"
    queue: str = "default"
    job_type: str = "openwebui_attachment_hydration"
    owner_user_id: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    result: Optional[dict[str, Any]] = None
    error: Optional[str] = None


class CreateChatbookResponse(BaseModel):
    """Response for chatbook creation."""
    success: bool
    message: str
    job_id: Optional[str] = Field(None, description="Job ID if async mode")
    download_url: Optional[str] = Field(None, description="Download URL if sync mode")


class ImportChatbookResponse(BaseModel):
    """Response for chatbook import."""
    success: bool
    message: str
    source_format: ChatbookImportSourceFormat = ChatbookImportSourceFormat.CHATBOOK
    job_id: Optional[str] = Field(None, description="Job ID if async mode")
    imported_items: Optional[dict[str, int]] = Field(
        None,
        description="Count of imported items by type"
    )
    openwebui_result: Optional[OpenWebUIImportResult] = Field(
        None,
        description="Structured import result for OpenWebUI JSON sources"
    )
    openwebui_db_result: Optional[OpenWebUIDatabaseImportResult] = Field(
        None,
        description="Structured import result for OpenWebUI database sources"
    )
    warnings: Optional[list[str]] = Field(
        None,
        description="Validator and import warnings (sync imports only)"
    )


class PreviewChatbookResponse(BaseModel):
    """Response for chatbook preview."""
    source_format: ChatbookImportSourceFormat = ChatbookImportSourceFormat.CHATBOOK
    manifest: Optional[ChatbookManifestResponse] = None
    openwebui_preview: Optional[OpenWebUIImportPreview] = None
    openwebui_db_preview: Optional[OpenWebUIDatabasePreview] = None
    error: Optional[str] = None


class ListExportJobsResponse(BaseModel):
    """Response for listing export jobs."""
    jobs: list[ExportJobResponse]
    total: int
    has_more: bool | None = Field(default=None, description="Alias for pagination.has_more")
    next_offset: int | None = Field(default=None, ge=0, description="Alias for pagination.next_offset")
    pagination: OffsetPaginationMeta

    @model_validator(mode="after")
    def _default_pagination_aliases(self):
        return _default_offset_pagination_aliases(self)


class ListImportJobsResponse(BaseModel):
    """Response for listing import jobs."""
    jobs: list[ImportJobResponse]
    total: int
    has_more: bool | None = Field(default=None, description="Alias for pagination.has_more")
    next_offset: int | None = Field(default=None, ge=0, description="Alias for pagination.next_offset")
    pagination: OffsetPaginationMeta

    @model_validator(mode="after")
    def _default_pagination_aliases(self):
        return _default_offset_pagination_aliases(self)


class CleanupExpiredExportsResponse(BaseModel):
    """Response for cleanup operation."""
    deleted_count: int
    message: Optional[str] = None


class DownloadChatbookResponse(BaseModel):
    """Response for download request."""
    file_path: str
    file_name: str
    content_type: str = "application/zip"
    file_size: int


# Query Parameters

class ListJobsQuery(BaseModel):
    """Query parameters for listing jobs."""
    status: Optional[str] = Field(
        None,
        description="Filter by status (pending, in_progress, completed, failed, cancelled, expired)"
    )
    limit: int = Field(100, ge=1, le=1000, description="Maximum results")
    offset: int = Field(0, ge=0, description="Offset for pagination")
    order_by: JobOrderByField = Field("created_at", description="Sort field")
    order_desc: bool = Field(True, description="Sort descending")

    @field_validator('status', mode='before')
    @classmethod
    def validate_status(cls, v):
        """Validate status is a known value to prevent injection."""
        if v is None:
            return None
        # Whitelist of allowed status values (from both ExportStatus and ImportStatus)
        allowed_statuses = {
            'pending', 'in_progress', 'completed', 'failed',
            'cancelled', 'expired', 'validating'
        }
        if v.lower() not in allowed_statuses:
            raise ValueError(f"Invalid status '{v}'. Allowed: {', '.join(sorted(allowed_statuses))}")
        return v.lower()


class CancelJobResponse(BaseModel):
    """Response for job cancellation."""
    success: bool
    message: str
    job_id: str


class RemoveJobResponse(BaseModel):
    """Response for job removal."""
    success: bool
    message: str
    job_id: str


class ContinueExportRequest(BaseModel):
    """Request to continue a truncated chatbook export (e.g. evaluation runs)."""
    export_id: str = Field(..., description="Export ID from the original chatbook manifest")
    continuations: list[dict[str, Any]] = Field(
        ...,
        description="Continuation tokens from the original manifest's truncation metadata"
    )
    name: Optional[str] = Field(None, max_length=255, description="Override name for continuation chatbook")
    async_mode: bool = Field(False, description="Run as background job")


# Error Responses

class ChatbookErrorResponse(BaseModel):
    """Error response for chatbook operations."""
    detail: str
    error_type: str
    job_id: Optional[str] = None
    suggestions: list[str] = Field(default_factory=list)
