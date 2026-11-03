"""Pydantic schemas for file artifact APIs."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

# Core-owned since TASK-13950.15; re-exported for the API layer.
from tldw_Server_API.app.core.File_Artifacts.file_artifacts_schemas import (  # noqa: F401 - re-exported (TASK-13950.15)
    AsyncMode,
    ExportFormat,
    ExportMode,
    FileArtifact,
    FileCreateOptions,
    FileCreateRequest,
    FileExportInfo,
    FileExportRequest,
    FileType,
    FileValidationIssue,
    FileValidationResult,
)


class FileCreateResponse(BaseModel):
    """Response wrapper for artifact creation."""
    artifact: FileArtifact


class FileArtifactResponse(BaseModel):
    """Response wrapper for fetching an artifact."""
    artifact: FileArtifact


class FileDeleteResponse(BaseModel):
    """Response for delete operations."""
    success: bool
    file_deleted: bool = False


class ReferenceImageListItem(BaseModel):
    """Picker-safe metadata for a reference image candidate."""

    file_id: int
    title: str
    mime_type: str
    width: int | None = None
    height: int | None = None
    created_at: datetime


class ReferenceImageListResponse(BaseModel):
    """Response wrapper for picker-safe reference image candidates."""

    items: list[ReferenceImageListItem] = Field(default_factory=list)


class FileArtifactsPurgeRequest(BaseModel):
    """Request to purge file artifacts."""
    delete_files: bool = False
    soft_deleted_grace_days: int = Field(default=30, ge=0)
    include_retention: bool = True


class FileArtifactsPurgeResponse(BaseModel):
    """Response for purge summary."""
    removed: int
    files_deleted: int
