"""Schemas for chat document upload processing decisions and handoff drafts."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


DocumentProcessingMode = Literal["add_to_chat", "ocr_pages", "ingest_to_library"]
DocumentProcessingStatus = Literal["available", "unavailable", "blocked"]


class DocumentUploadPreflightFile(BaseModel):
    """Client-supplied document metadata used before upload processing."""

    client_id: str = Field(..., min_length=1, max_length=128)
    filename: str = Field(..., min_length=1, max_length=512)
    mime_type: str | None = Field(default=None, max_length=255)
    size_bytes: int = Field(..., ge=0)
    page_count: int | None = Field(default=None, ge=0)
    estimated_tokens: int | None = Field(default=None, ge=0)


class DocumentUploadPreflightRequest(BaseModel):
    """Batch request for document processing capability checks."""

    files: list[DocumentUploadPreflightFile] = Field(default_factory=list, max_length=50)


class DocumentModeCapability(BaseModel):
    """Availability result for one document processing mode."""

    available: bool
    status: DocumentProcessingStatus
    reason: str | None = None


class DocumentUploadPreflightItem(BaseModel):
    """Per-file processing choices returned to the chat UI."""

    client_id: str
    filename: str
    media_type: Literal["pdf", "document", "ebook", "unsupported"]
    default_mode: DocumentProcessingMode | None
    modes: dict[DocumentProcessingMode, DocumentModeCapability]
    max_size_bytes: int
    max_pages: int | None
    max_chat_tokens: int
    estimated_pages: int | None = None
    estimated_tokens: int | None = None
    requires_send_time_estimate: bool = False


class DocumentUploadPreflightResponse(BaseModel):
    """Preflight response for all requested document files."""

    files: list[DocumentUploadPreflightItem]


class ChatDocumentDraftCreateRequest(BaseModel):
    """Transient sidepanel-to-WebUI document handoff payload."""

    payload: dict[str, Any] = Field(default_factory=dict)


class ChatDocumentDraftCreateResponse(BaseModel):
    """Identifier and expiry for a stored document handoff draft."""

    draft_id: str
    expires_at: str


class ChatDocumentDraftReadResponse(BaseModel):
    """Stored document handoff draft payload and timestamps."""

    draft_id: str
    created_at: str
    expires_at: str
    payload: dict[str, Any]
