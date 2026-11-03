"""Schemas for chat document upload processing decisions and handoff drafts."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

# Core-owned since TASK-13950.15; re-exported for the API layer.
from tldw_Server_API.app.core.Ingestion_Media_Processing.document_upload_processing import (  # noqa: F401 - re-exported (TASK-13950.15)
    DocumentModeCapability,
    DocumentProcessingMode,
    DocumentProcessingStatus,
    DocumentUploadPreflightFile,
    DocumentUploadPreflightItem,
)


class DocumentUploadPreflightRequest(BaseModel):
    """Batch request for document processing capability checks."""

    files: list[DocumentUploadPreflightFile] = Field(default_factory=list, max_length=50)


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
