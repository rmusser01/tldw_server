# app/api/v1/schemas/notes_schemas.py
#
# Imports
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

# 3rd-party Libraries
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from tldw_Server_API.app.api.v1.schemas.pagination import OffsetPaginationMeta

from .notes_studio import NoteStudioDocumentSummaryResponse

#
# Local Imports
#
#######################################################################################################################
#
# Schemas:

def _default_offset_pagination_aliases(response):
    if response.has_more is None:
        response.has_more = response.pagination.has_more
    if response.next_offset is None:
        response.next_offset = response.pagination.next_offset
    return response


def _split_keywords(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return [p.strip() for p in value.split(',')]
    if isinstance(value, list):
        return [p.strip() for p in value if isinstance(p, str)]
    raise ValueError("Keywords must be a list of strings or a comma-separated string.")


# --- Note Schemas ---
class NoteBase(BaseModel):
    title: str = Field(..., min_length=1, max_length=255, description="Title of the note")
    content: str = Field(..., min_length=1, max_length=5000000, description="Content of the note (max 5MB)")
    conversation_id: str | None = Field(None, description="Optional conversation ID backlink")
    message_id: str | None = Field(None, description="Optional message ID backlink")


class NoteCreate(NoteBase):
    # Override to allow optional title when auto_title is used
    title: str | None = Field(
        None,
        min_length=1,
        max_length=255,
        description="Title of the note. Optional when auto_title=true."
    )
    id: str | None = Field(None,
                              description="Optional client-provided UUID for the note. If None, will be auto-generated.")
    keywords: str | list[str] | None = Field(
        default=None,
        description="Optional keywords to attach to the note. Accepts a list of strings or a comma-separated string."
    )
    # Title auto-generation controls
    # MVP: heuristic-only; Phase 2: 'llm' available behind flag
    auto_title: bool = Field(False, description="If true and no title provided, auto-generate a title from content.")
    title_strategy: Literal["heuristic", "llm", "llm_fallback"] = Field(
        "heuristic",
        description="Strategy for title generation. MVP supports 'heuristic'."
    )
    title_max_len: int = Field(250, ge=1, le=500, description="Max title length when auto-generating.")
    language: str | None = Field(None, description="Optional language hint for title generation.")

    # Normalize keywords input to a clean list of strings (if provided)
    @field_validator("keywords", mode="before")
    @classmethod
    def validate_keywords(cls, value: Any):
        parts = _split_keywords(value)
        if parts is None:
            return value
        for part in parts:
            if not part:
                continue
            if len(part) > 100:
                raise ValueError("Keyword entries must be 100 characters or fewer.")
        return value

    @property
    def normalized_keywords(self) -> list[str] | None:
        parts = _split_keywords(getattr(self, 'keywords', None))
        if parts is None:
            return None
        # Remove empties and deduplicate while preserving order
        seen = set()
        result: list[str] = []
        for p in parts:
            if not p:
                continue
            # Dedup case-insensitive
            key = p.lower()
            if key in seen:
                continue
            seen.add(key)
            result.append(p)
        return result or None


class NoteUpdate(BaseModel):
    title: str | None = Field(None, min_length=1, max_length=255, description="New title for the note")
    content: str | None = Field(None, min_length=1, max_length=5000000, description="New content for the note (max 5MB)")
    conversation_id: str | None = Field(None, description="Optional conversation ID backlink")
    message_id: str | None = Field(None, description="Optional message ID backlink")
    keywords: str | list[str] | None = Field(
        default=None,
        description="Optional keywords to attach to the note. Accepts a list of strings or a comma-separated string."
    )
    # Ensure at least one field is provided for update, or handle in endpoint if empty update is no-op
    # Pydantic v2: model_validator

    @field_validator("keywords", mode="before")
    @classmethod
    def validate_keywords(cls, value: Any):
        parts = _split_keywords(value)
        if parts is None:
            return value
        for part in parts:
            if not part:
                continue
            if len(part) > 100:
                raise ValueError("Keyword entries must be 100 characters or fewer.")
        return value

    @property
    def normalized_keywords(self) -> list[str] | None:
        parts = _split_keywords(getattr(self, 'keywords', None))
        if parts is None:
            return None
        seen = set()
        result: list[str] = []
        for p in parts:
            if not p:
                continue
            key = p.lower()
            if key in seen:
                continue
            seen.add(key)
            result.append(p)
        return result or None


class NoteKeywordSyncStatus(BaseModel):
    failed_count: int = Field(
        default=0,
        ge=0,
        description="Number of keywords that failed to attach during the save operation."
    )
    failed_keywords: list[str] = Field(
        default_factory=list,
        description="Best-effort list of keyword texts that failed to attach."
    )


class NoteFolderResponse(BaseModel):
    id: int = Field(..., description="Integer ID of the folder")
    name: str = Field(..., description="Folder display name")
    path: str = Field(..., description="Normalized relative folder path")
    parent_id: int | None = Field(default=None, description="Parent folder ID, if any")

    model_config = ConfigDict(from_attributes=True)


class NoteResponse(NoteBase):
    id: str = Field(..., description="UUID of the note")
    created_at: datetime = Field(..., description="Timestamp of note creation")
    last_modified: datetime = Field(..., description="Timestamp of last modification")
    version: int = Field(..., description="Version number for optimistic locking")
    client_id: str = Field(..., description="Client ID that last modified the note")
    deleted: bool = Field(..., description="Whether the note is soft-deleted")
    studio: NoteStudioDocumentSummaryResponse | None = Field(
        default=None,
        description="Optional lightweight Studio summary attached to single-note fetches.",
    )
    keywords: list[KeywordResponse] | None = Field(default=None, description="Keywords linked to this note")
    folders: list[NoteFolderResponse] | None = Field(default=None, description="Folders linked to this note")
    keyword_sync: NoteKeywordSyncStatus | None = Field(
        default=None,
        description="Present when note save succeeded but one or more keyword attach operations failed."
    )

    model_config = ConfigDict(from_attributes=True)  # Pydantic V2 (formerly orm_mode)


# --- Keyword Schemas ---
class KeywordBase(BaseModel):
    keyword: str = Field(..., min_length=1, max_length=100, description="The keyword text")


class KeywordCreate(KeywordBase):
    pass


class KeywordUpdate(KeywordBase):
    pass


class KeywordMergeRequest(BaseModel):
    target_keyword_id: int = Field(..., ge=1, description="The destination keyword ID.")
    expected_target_version: int | None = Field(
        default=None,
        ge=1,
        description="Optional optimistic-lock version for the target keyword."
    )


class KeywordMergeResponse(BaseModel):
    source_keyword_id: int
    target_keyword_id: int
    source_deleted_version: int
    target_version: int
    merged_note_links: int = Field(default=0, ge=0)
    merged_conversation_links: int = Field(default=0, ge=0)
    merged_collection_links: int = Field(default=0, ge=0)
    merged_flashcard_links: int = Field(default=0, ge=0)


class KeywordResponse(KeywordBase):
    id: int = Field(..., description="Integer ID of the keyword")
    created_at: datetime = Field(..., description="Timestamp of keyword creation")
    last_modified: datetime = Field(..., description="Timestamp of last modification")
    version: int = Field(..., description="Version number for optimistic locking")
    client_id: str = Field(..., description="Client ID that last modified the keyword")
    deleted: bool = Field(..., description="Whether the keyword is soft-deleted")
    note_count: int | None = Field(
        default=None,
        ge=0,
        description="Optional count of active notes currently linked to this keyword."
    )

    model_config = ConfigDict(from_attributes=True)


# --- Keyword Collection Schemas ---
class KeywordCollectionBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=255, description="Collection name")
    parent_id: int | None = Field(
        default=None,
        ge=1,
        description="Optional parent collection ID for hierarchical organization."
    )


class KeywordCollectionCreate(KeywordCollectionBase):
    keywords: str | list[str] | None = Field(
        default=None,
        description="Optional keywords for initial collection membership."
    )

    @field_validator("keywords", mode="before")
    @classmethod
    def validate_keywords(cls, value: Any):
        parts = _split_keywords(value)
        if parts is None:
            return value
        for part in parts:
            if not part:
                continue
            if len(part) > 100:
                raise ValueError("Keyword entries must be 100 characters or fewer.")
        return value

    @property
    def normalized_keywords(self) -> list[str] | None:
        parts = _split_keywords(getattr(self, 'keywords', None))
        if parts is None:
            return None
        seen = set()
        result: list[str] = []
        for p in parts:
            if not p:
                continue
            key = p.lower()
            if key in seen:
                continue
            seen.add(key)
            result.append(p)
        return result or None


class KeywordCollectionUpdate(BaseModel):
    name: str | None = Field(None, min_length=1, max_length=255, description="Updated collection name")
    parent_id: int | None = Field(
        default=None,
        ge=1,
        description="Updated parent collection ID. Use null to clear parent."
    )
    keywords: str | list[str] | None = Field(
        default=None,
        description="Optional full keyword list to sync for this collection."
    )

    @field_validator("keywords", mode="before")
    @classmethod
    def validate_keywords(cls, value: Any):
        parts = _split_keywords(value)
        if parts is None:
            return value
        for part in parts:
            if not part:
                continue
            if len(part) > 100:
                raise ValueError("Keyword entries must be 100 characters or fewer.")
        return value

    @property
    def normalized_keywords(self) -> list[str] | None:
        parts = _split_keywords(getattr(self, 'keywords', None))
        if parts is None:
            return None
        seen = set()
        result: list[str] = []
        for p in parts:
            if not p:
                continue
            key = p.lower()
            if key in seen:
                continue
            seen.add(key)
            result.append(p)
        return result or None


class KeywordCollectionResponse(BaseModel):
    id: int = Field(..., ge=1)
    name: str = Field(..., min_length=1, max_length=255)
    parent_id: int | None = Field(default=None)
    created_at: datetime
    last_modified: datetime
    version: int = Field(..., ge=1)
    client_id: str
    deleted: bool
    keywords: list[KeywordResponse] | None = Field(
        default=None,
        description="Optional keywords linked to this collection."
    )

    model_config = ConfigDict(from_attributes=True)


class KeywordCollectionsListResponse(BaseModel):
    collections: list[KeywordCollectionResponse]
    count: int = Field(..., ge=0)
    limit: int = Field(..., ge=1)
    offset: int = Field(..., ge=0)
    total: int = Field(..., ge=0)
    has_more: bool | None = Field(default=None, description="Alias for pagination.has_more")
    next_offset: int | None = Field(default=None, ge=0, description="Alias for pagination.next_offset")
    pagination: OffsetPaginationMeta

    @model_validator(mode="after")
    def _default_pagination_aliases(self):
        return _default_offset_pagination_aliases(self)


class CollectionKeywordLinkResponse(BaseModel):
    success: bool
    message: str | None = None


class CollectionKeywordLinkItem(BaseModel):
    collection_id: int = Field(..., ge=1)
    keyword_id: int = Field(..., ge=1)


class CollectionKeywordLinksResponse(BaseModel):
    links: list[CollectionKeywordLinkItem]
    count: int = Field(..., ge=0)
    limit: int = Field(..., ge=1)
    offset: int = Field(..., ge=0)
    total: int = Field(..., ge=0)
    has_more: bool | None = Field(default=None, description="Alias for pagination.has_more")
    next_offset: int | None = Field(default=None, ge=0, description="Alias for pagination.next_offset")
    pagination: OffsetPaginationMeta

    @model_validator(mode="after")
    def _default_pagination_aliases(self):
        return _default_offset_pagination_aliases(self)


class ConversationKeywordLinkResponse(BaseModel):
    success: bool
    message: str | None = None


class ConversationKeywordLinkItem(BaseModel):
    conversation_id: str
    keyword_id: int = Field(..., ge=1)


class ConversationKeywordLinksResponse(BaseModel):
    links: list[ConversationKeywordLinkItem]
    count: int = Field(..., ge=0)
    limit: int = Field(..., ge=1)
    offset: int = Field(..., ge=0)
    total: int = Field(..., ge=0)
    has_more: bool | None = Field(default=None, description="Alias for pagination.has_more")
    next_offset: int | None = Field(default=None, ge=0, description="Alias for pagination.next_offset")
    pagination: OffsetPaginationMeta

    @model_validator(mode="after")
    def _default_pagination_aliases(self):
        return _default_offset_pagination_aliases(self)


# --- Linking Schemas ---
class NoteKeywordLinkResponse(BaseModel):
    success: bool
    message: str | None = None


class KeywordsForNoteResponse(BaseModel):
    note_id: str
    keywords: list[KeywordResponse]


class NotesForKeywordResponse(BaseModel):
    keyword_id: int
    notes: list[NoteResponse]
    count: int = Field(..., ge=0)
    limit: int = Field(..., ge=1)
    offset: int = Field(..., ge=0)
    total: int = Field(..., ge=0)
    has_more: bool | None = Field(default=None, description="Alias for pagination.has_more")
    next_offset: int | None = Field(default=None, ge=0, description="Alias for pagination.next_offset")
    pagination: OffsetPaginationMeta

    @model_validator(mode="after")
    def _default_pagination_aliases(self):
        return _default_offset_pagination_aliases(self)


# --- Attachment Schemas ---
class NoteAttachmentResponse(BaseModel):
    file_name: str = Field(..., min_length=1, description="Stored attachment filename")
    original_file_name: str = Field(..., min_length=1, description="Original filename supplied by the client")
    content_type: str | None = Field(default=None, description="Detected or provided media type")
    size_bytes: int = Field(..., ge=0, description="Attachment size in bytes")
    uploaded_at: datetime = Field(..., description="Upload timestamp")
    url: str = Field(..., min_length=1, description="Download URL for this attachment")


class NoteAttachmentsListResponse(BaseModel):
    note_id: str = Field(..., min_length=1)
    attachments: list[NoteAttachmentResponse]
    count: int = Field(..., ge=0)


# --- General API Response Schemas ---
class DetailResponse(BaseModel):
    detail: str


# --- Bulk Create Schemas ---
class NoteBulkCreateRequest(BaseModel):
    notes: list[NoteCreate] = Field(..., min_length=1, max_length=200, description="List of notes to create")


class NoteBulkCreateItemResult(BaseModel):
    success: bool
    note: NoteResponse | None = None
    error: str | None = None


class NoteBulkCreateResponse(BaseModel):
    results: list[NoteBulkCreateItemResult]
    created_count: int = 0
    failed_count: int = 0


# --- List/Export Response Schemas ---
class NotesListResponse(BaseModel):
    notes: list[NoteResponse]
    items: list[NoteResponse]
    results: list[NoteResponse]
    count: int
    limit: int
    offset: int
    total: int | None = None
    has_more: bool | None = Field(default=None, description="Alias for pagination.has_more")
    next_offset: int | None = Field(default=None, ge=0, description="Alias for pagination.next_offset")
    pagination: OffsetPaginationMeta

    @model_validator(mode="after")
    def _default_pagination_aliases(self):
        return _default_offset_pagination_aliases(self)


class NotesExportResponse(BaseModel):
    notes: list[NoteResponse]
    data: list[NoteResponse]
    items: list[NoteResponse]
    results: list[NoteResponse]
    count: int
    total: int | None = None
    limit: int | None = None
    offset: int | None = None
    pagination: OffsetPaginationMeta | None = None
    exported_at: str


class NotesExportRequest(BaseModel):
    """Export request for selected notes.

    Accepts explicit note IDs and optional flags for including keywords and
    selecting the output format.
    """
    model_config = ConfigDict(extra='forbid')

    note_ids: list[str] = Field(..., description="List of note IDs to export")
    include_keywords: bool = Field(default=False)
    format: Literal['json', 'csv'] = Field(default='json', description="Use /export.csv for CSV exports.")


class NotesImportItem(BaseModel):
    file_name: str | None = Field(default=None, description="Optional source file name.")
    format: Literal["json", "markdown"] = Field(..., description="Import format for this item.")
    content: str = Field(..., min_length=1, max_length=5_000_000, description="Raw file content.")


class NotesImportRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')

    items: list[NotesImportItem] = Field(..., min_length=1, max_length=50)
    duplicate_strategy: Literal["skip", "overwrite", "create_copy"] = Field(
        default="create_copy",
        description="How to handle imported notes whose IDs already exist."
    )


class NotesImportFileResult(BaseModel):
    file_name: str | None = None
    source_format: Literal["json", "markdown"]
    detected_notes: int = 0
    created_count: int = 0
    updated_count: int = 0
    skipped_count: int = 0
    failed_count: int = 0
    errors: list[str] = Field(default_factory=list)


class NotesImportResponse(BaseModel):
    files: list[NotesImportFileResult]
    detected_notes: int = 0
    created_count: int = 0
    updated_count: int = 0
    skipped_count: int = 0
    failed_count: int = 0


# --- Title Suggestion Schemas ---
class TitleSuggestRequest(BaseModel):
    content: str = Field(..., min_length=1, max_length=5000000, description="Source content of the note")
    title_strategy: Literal["heuristic", "llm", "llm_fallback"] = Field(
        "heuristic",
        description="Strategy for title generation. MVP supports 'heuristic'."
    )
    title_max_len: int = Field(250, ge=1, le=500, description="Max title length for suggestion.")
    language: str | None = Field(None, description="Optional language hint.")


class TitleSuggestResponse(BaseModel):
    title: str = Field(..., description="Suggested title")


# Resolve forward references for nested schemas.
NoteResponse.model_rebuild()
KeywordsForNoteResponse.model_rebuild()
NotesForKeywordResponse.model_rebuild()

#
# End of notes_schemas.py
#######################################################################################################################
