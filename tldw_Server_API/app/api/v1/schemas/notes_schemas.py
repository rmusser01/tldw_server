# app/api/v1/schemas/notes_schemas.py
#
# Imports
from __future__ import annotations

from datetime import datetime
from pathlib import PurePosixPath, PureWindowsPath
from typing import Annotated, Any, Literal

# 3rd-party Libraries
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from tldw_Server_API.app.api.v1.schemas.pagination import OffsetPaginationMeta
from tldw_Server_API.app.core.Notes.wikilinks import (
    MAX_WIKILINK_RENAME_NOTES,
    MAX_WIKILINK_REPLACEMENTS_PER_NOTE,
    MAX_WIKILINK_TITLE_LENGTH,
    MAX_WIKILINK_TOKEN_TEXT_LENGTH,
    is_single_wikilink,
)
from tldw_Server_API.app.core.Sync.v2.notes_provenance_contract import validate_notes_provenance_payload

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


def _normalize_folder_paths(value: Any) -> list[str] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise ValueError("Folder paths must be a list of relative paths.")
    normalized: list[str] = []
    for raw_path in value:
        if not isinstance(raw_path, str):
            raise ValueError("Folder paths must be strings.")
        raw_text = raw_path.strip()
        posix_text = raw_text.replace("\\", "/")
        if PurePosixPath(posix_text).is_absolute() or PureWindowsPath(raw_text).drive:
            raise ValueError("Folder paths must be relative.")
        text = posix_text.strip("/")
        parts = [part.strip() for part in text.split("/") if part.strip() and part.strip() != "."]
        if not parts or any(part == ".." for part in parts):
            raise ValueError("Folder paths must be non-empty relative paths without parent traversal.")
        path = "/".join(parts)
        if len(path) > 500:
            raise ValueError("Folder paths must be 500 characters or fewer.")
        normalized.append(path)
    return normalized


# --- Note Schemas ---
ProvenanceVersion = Annotated[int, Field(strict=True, ge=0, le=9_007_199_254_740_991)]


class NoteProvenanceWrite(BaseModel):
    """Optional independent evidence replacement; omission preserves history."""

    knowledge_provenance: dict[str, Any] | None = None
    expected_provenance_version: ProvenanceVersion | None = None

    @field_validator("knowledge_provenance", mode="before")
    @classmethod
    def validate_provenance(cls, value):
        if value is None:
            raise ValueError("Omit knowledge_provenance to preserve history; null is not a replacement")
        return validate_notes_provenance_payload(value)

    @model_validator(mode="after")
    def require_independent_base(self):
        if self.knowledge_provenance is not None and self.expected_provenance_version is None:
            raise ValueError("knowledge_provenance requires expected_provenance_version")
        if self.knowledge_provenance is None and self.expected_provenance_version is not None:
            raise ValueError("expected_provenance_version requires knowledge_provenance")
        return self


class NoteProvenanceRestore(BaseModel):
    """Restore only the retained child at its exact independent head."""

    model_config = ConfigDict(extra="forbid")
    expected_provenance_version: Annotated[int, Field(strict=True, ge=1, le=9_007_199_254_740_991)]
    expected_provenance_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")


class NoteBase(BaseModel):
    title: str = Field(..., min_length=1, max_length=255, description="Title of the note")
    content: str = Field(..., min_length=1, max_length=5000000, description="Content of the note (max 5MB)")
    conversation_id: str | None = Field(None, description="Optional conversation ID backlink")
    message_id: str | None = Field(None, description="Optional message ID backlink")


class NoteCreate(NoteBase, NoteProvenanceWrite):
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
    folder_paths: list[str] | None = Field(
        default=None,
        description="Optional ordered relative folder paths to attach to the note.",
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

    @field_validator("folder_paths", mode="before")
    @classmethod
    def validate_folder_paths(cls, value: Any):
        return _normalize_folder_paths(value)

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

    @property
    def normalized_folder_paths(self) -> list[str] | None:
        paths = _normalize_folder_paths(getattr(self, "folder_paths", None))
        if paths is None:
            return None
        seen: set[str] = set()
        result: list[str] = []
        for path in paths:
            key = path.casefold()
            if key in seen:
                continue
            seen.add(key)
            result.append(path)
        return result


class NoteUpdate(NoteProvenanceWrite):
    title: str | None = Field(None, min_length=1, max_length=255, description="New title for the note")
    content: str | None = Field(None, min_length=1, max_length=5000000, description="New content for the note (max 5MB)")
    conversation_id: str | None = Field(None, description="Optional conversation ID backlink")
    message_id: str | None = Field(None, description="Optional message ID backlink")
    keywords: str | list[str] | None = Field(
        default=None,
        description="Optional keywords to attach to the note. Accepts a list of strings or a comma-separated string."
    )
    folder_paths: list[str] | None = Field(
        default=None,
        description="Optional ordered relative folder paths to attach to the note.",
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

    @field_validator("folder_paths", mode="before")
    @classmethod
    def validate_folder_paths(cls, value: Any):
        return _normalize_folder_paths(value)

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

    @property
    def normalized_folder_paths(self) -> list[str] | None:
        paths = _normalize_folder_paths(getattr(self, "folder_paths", None))
        if paths is None:
            return None
        seen: set[str] = set()
        result: list[str] = []
        for path in paths:
            key = path.casefold()
            if key in seen:
                continue
            seen.add(key)
            result.append(path)
        return result


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


class NoteFolderCreate(BaseModel):
    path: str = Field(..., min_length=1, max_length=500, description="Relative folder path to create or reuse")

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("Folder path cannot be empty.")
        return normalized


class NoteFolderResponse(BaseModel):
    id: int = Field(..., description="Integer ID of the folder")
    sync_id: str = Field(..., description="Stable opaque Sync identity")
    name: str = Field(..., description="Folder display name")
    path: str = Field(..., description="Normalized relative folder path")
    parent_id: int | None = Field(default=None, description="Parent folder ID, if any")

    model_config = ConfigDict(from_attributes=True)


class NoteFoldersListResponse(BaseModel):
    items: list[NoteFolderResponse] = Field(default_factory=list, description="Folder rows")
    folders: list[NoteFolderResponse] = Field(default_factory=list, description="Alias for items")
    count: int = Field(default=0, ge=0, description="Number of returned folders")


class NoteResponse(NoteBase):
    content: str = Field(..., description="Note body; portable export may add or remove provenance markers")
    knowledge_provenance_state: Literal["unsupported", "absent", "active", "deleted"] = "unsupported"
    knowledge_provenance_version: ProvenanceVersion | None = None
    knowledge_provenance_hash: str | None = None
    knowledge_provenance: dict[str, Any] | None = None
    knowledge_provenance_reconciliation: Literal["canonical_wins"] | None = None

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
    sync_id: str = Field(..., description="Stable opaque Sync identity")
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
    sync_id: str = Field(..., description="Stable opaque Sync identity")
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
NoteListSortBy = Literal["last_modified", "created_at", "title"]
NoteListSortOrder = Literal["asc", "desc"]


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


MAX_WIKILINK_RESOLVE_ITEMS = 200


class WikilinkResolveRequest(BaseModel):
    """Link texts from one note's ``[[Title]]`` and ``[[id:UUID]]`` wikilinks."""

    model_config = ConfigDict(extra="forbid")

    titles: list[str] = Field(
        default_factory=list,
        max_length=MAX_WIKILINK_RESOLVE_ITEMS,
        description="Link texts from [[Title]] links, as written.",
    )
    ids: list[str] = Field(
        default_factory=list,
        max_length=MAX_WIKILINK_RESOLVE_ITEMS,
        description="Note ids from [[id:UUID]] links, as written.",
    )
    source_note_id: str | None = Field(
        None,
        max_length=200,
        description="The linking note. It never resolves its own [[Title]] links.",
    )

    @field_validator("titles", "ids")
    @classmethod
    def _bounded_items(cls, values: list[str]) -> list[str]:
        if any(len(value) > 1024 for value in values):
            raise ValueError("each wikilink text must be at most 1024 characters")
        return values


class WikilinkTitleResolution(BaseModel):
    title: str = Field(..., description="The link text, as sent.")
    note_id: str | None = Field(
        None,
        description="The resolved note, or null when no live note has this title.",
    )
    note_title: str | None = Field(None, description="The resolved note's title.")
    candidate_count: int = Field(
        ...,
        ge=0,
        description=(
            "Live notes whose title matches, ignoring case and extra whitespace. Above 1 the title is "
            "ambiguous: an exact title match wins, then the oldest note, then the lowest id."
        ),
    )


class WikilinkIdResolution(BaseModel):
    id: str = Field(..., description="The note id, as sent.")
    note_id: str | None = Field(
        None,
        description="The canonical note id, or null when it is malformed or no live note has it.",
    )
    note_title: str | None = Field(None, description="The linked note's title.")


class WikilinkResolveResponse(BaseModel):
    titles: list[WikilinkTitleResolution] = Field(default_factory=list)
    ids: list[WikilinkIdResolution] = Field(default_factory=list)


# --- Updating [[Old title]] links after a rename -----------------------------

_NOTE_ID_MAX_LENGTH = 200


class WikilinkReferrersRequest(BaseModel):
    """Ask which live notes hold a ``[[Title]]`` link to a title."""

    model_config = ConfigDict(extra="forbid")

    title: str = Field(
        ...,
        min_length=1,
        max_length=MAX_WIKILINK_TITLE_LENGTH,
        description="The linked title. Matching ignores case and extra whitespace.",
    )
    exclude_note_id: str | None = Field(
        None,
        max_length=_NOTE_ID_MAX_LENGTH,
        description="A note to leave out, such as the note that was just renamed.",
    )
    unresolved_only: bool = Field(
        False,
        description=(
            "If true, only count notes where the link names no live note: the links a rename broke. "
            "A link that another live note with this title still answers is left out."
        ),
    )
    after_note_id: str | None = Field(
        None,
        max_length=_NOTE_ID_MAX_LENGTH,
        description="Cursor: list notes after this id. Pass the previous page's next_after_note_id.",
    )
    limit: int = Field(
        MAX_WIKILINK_RENAME_NOTES,
        ge=1,
        le=MAX_WIKILINK_RENAME_NOTES,
        description="Notes to list per page.",
    )


class WikilinkReferrerNote(BaseModel):
    id: str = Field(..., description="The linking note.")
    title: str = Field(..., description="The linking note's title.")
    version: int = Field(..., description="Its version now. Send it back as expected_version to rewrite it.")


class WikilinkReferrersResponse(BaseModel):
    title: str = Field(..., description="The linked title, as sent.")
    count: int = Field(..., ge=0, description="Linking notes across all pages.")
    notes: list[WikilinkReferrerNote] = Field(default_factory=list, description="This page, ordered by note id.")
    next_after_note_id: str | None = Field(
        None,
        description="Cursor for the next page, or null when this page is the last.",
    )


class WikilinkRewriteNote(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., min_length=1, max_length=_NOTE_ID_MAX_LENGTH, description="A linking note.")
    expected_version: int = Field(
        ...,
        ge=1,
        description="The version the caller read. A note at another version is skipped, not overwritten.",
    )


class WikilinkRewriteRequest(BaseModel):
    """Rewrite ``[[Old title]]`` links so they point at a renamed note again."""

    model_config = ConfigDict(extra="forbid")

    note_id: str = Field(
        ...,
        min_length=1,
        max_length=_NOTE_ID_MAX_LENGTH,
        description="The renamed note. Links are rewritten to its current title.",
    )
    old_title: str = Field(
        ...,
        min_length=1,
        max_length=MAX_WIKILINK_TITLE_LENGTH,
        description="The title the note had before the rename.",
    )
    notes: list[WikilinkRewriteNote] = Field(
        ...,
        min_length=1,
        max_length=MAX_WIKILINK_RENAME_NOTES,
        description="The linking notes to rewrite, from the referrers endpoint.",
    )


class WikilinkTokenReplacement(BaseModel):
    """One rewritten link: undo data, to be sent back unchanged."""

    model_config = ConfigDict(extra="forbid")

    token_index: int = Field(..., ge=0, description="The link's position among the note's [[...]] tokens.")
    original: str = Field(
        ...,
        min_length=1,
        max_length=MAX_WIKILINK_TOKEN_TEXT_LENGTH,
        description="The link text that was replaced.",
    )


WikilinkRewriteStatus = Literal[
    "updated",
    "skipped_conflict",
    "skipped_no_match",
    "skipped_not_found",
    "skipped_resolved",
    "failed",
]


class WikilinkRewriteNoteResult(BaseModel):
    id: str
    title: str | None = Field(None, description="The note's title, when the note was found.")
    status: WikilinkRewriteStatus = Field(
        ...,
        description=(
            "updated: the links were rewritten. skipped_conflict: the note changed since expected_version. "
            "skipped_no_match: it holds no link to the old title. skipped_not_found: it is missing or in the "
            "trash. skipped_resolved: another live note still has the old title, so the link is not broken. "
            "failed: it was not rewritten: the save failed, a link could not be rewritten in place, or it held "
            "more links than undo can restore. Its text is unchanged; with Sync active, a save that failed "
            "after it was accepted may still be applied later."
        ),
    )
    version: int | None = Field(None, description="The note's version after this request.")
    replaced_count: int = Field(0, ge=0)
    replacements: list[WikilinkTokenReplacement] = Field(
        default_factory=list,
        description="Undo data for an updated note.",
    )


class WikilinkRewriteResponse(BaseModel):
    old_title: str
    new_title: str = Field(..., description="The renamed note's current title.")
    link_form: Literal["title", "id"] = Field(
        ...,
        description=(
            "How the links were written. 'id' ([[id:UUID]]) is used when another live note shares the new "
            "title, or when no title link can name it. A note whose id is not a UUID keeps 'title' for a "
            "shared title only when that link resolves to it."
        ),
    )
    replacement: str = Field(..., description="The link text written in place of each old link.")
    new_title_shared: bool = Field(
        ...,
        description="Another live note has the new title, ignoring case and extra whitespace.",
    )
    updated_count: int = Field(..., ge=0)
    skipped_count: int = Field(..., ge=0)
    results: list[WikilinkRewriteNoteResult] = Field(default_factory=list)


class WikilinkRewriteUndoNote(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., min_length=1, max_length=_NOTE_ID_MAX_LENGTH)
    expected_version: int = Field(
        ...,
        ge=1,
        description="The version the rewrite returned. A note edited since is skipped.",
    )
    replacements: list[WikilinkTokenReplacement] = Field(
        ...,
        min_length=1,
        max_length=MAX_WIKILINK_REPLACEMENTS_PER_NOTE,
        description="The rewrite's undo data for this note.",
    )


class WikilinkRewriteUndoRequest(BaseModel):
    """Restore the text a wikilink rewrite replaced."""

    model_config = ConfigDict(extra="forbid")

    old_title: str = Field(..., min_length=1, max_length=MAX_WIKILINK_TITLE_LENGTH)
    replacement: str = Field(
        ...,
        min_length=1,
        max_length=MAX_WIKILINK_TITLE_LENGTH + 4,
        description="The link text the rewrite wrote, as it returned it.",
    )
    notes: list[WikilinkRewriteUndoNote] = Field(
        ...,
        min_length=1,
        max_length=MAX_WIKILINK_RENAME_NOTES,
    )

    @field_validator("replacement")
    @classmethod
    def _replacement_is_one_link(cls, value: str) -> str:
        if not is_single_wikilink(value):
            raise ValueError("replacement must be exactly one [[Title]] or [[id:UUID]] link")
        return value


class WikilinkRewriteUndoNoteResult(BaseModel):
    id: str
    title: str | None = None
    status: Literal["restored", "skipped_conflict", "skipped_no_match", "skipped_not_found", "failed"] = Field(
        ...,
        description=(
            "restored: the previous text is back. skipped_conflict: the note changed since the rewrite. "
            "skipped_no_match: its text does not hold the rewritten links. skipped_not_found: it is missing "
            "or in the trash. failed: the save failed. Its text is unchanged; with Sync active, a save that "
            "failed after it was accepted may still be applied later."
        ),
    )
    version: int | None = Field(None, description="The note's version after this request.")


class WikilinkRewriteUndoResponse(BaseModel):
    restored_count: int = Field(..., ge=0)
    skipped_count: int = Field(..., ge=0)
    results: list[WikilinkRewriteUndoNoteResult] = Field(default_factory=list)


# Resolve forward references for nested schemas.
NoteResponse.model_rebuild()
KeywordsForNoteResponse.model_rebuild()
NotesForKeywordResponse.model_rebuild()

#
# End of notes_schemas.py
#######################################################################################################################
