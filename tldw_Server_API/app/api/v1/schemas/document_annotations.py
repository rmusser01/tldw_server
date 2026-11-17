# Document Annotations Schemas
# Schemas for document annotation CRUD operations
#
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Self

from pydantic import BaseModel, Field, model_validator


class AnnotationColor(str, Enum):
    """Available highlight colors for annotations."""

    yellow = "yellow"
    green = "green"
    blue = "blue"
    pink = "pink"


class AnnotationType(str, Enum):
    """Type of annotation."""

    highlight = "highlight"
    page_note = "page_note"


class AnnotationCreate(BaseModel):
    """Request body for creating a new annotation."""

    location: str = Field(
        ..., description="Page number (for PDF) or EPUB CFI string"
    )
    text: str = Field(..., description="The selected/highlighted text (or note content for page_note)")
    color: AnnotationColor = Field(default=AnnotationColor.yellow, description="Highlight color")
    note: str | None = Field(None, description="Additional note attached to the annotation")
    annotation_type: AnnotationType = Field(
        default=AnnotationType.highlight, description="Type of annotation"
    )
    chapter_title: str | None = Field(
        None, description="Chapter title for EPUB annotations"
    )
    percentage: float | None = Field(
        None, ge=0, le=100, description="Reading percentage (0-100) for EPUB annotations"
    )


class AnnotationUpdate(BaseModel):
    """Request body for updating an existing annotation."""

    text: str | None = Field(None, description="Updated highlighted text")
    color: AnnotationColor | None = Field(None, description="Updated highlight color")
    note: str | None = Field(None, description="Updated note content")


class AnnotationResponse(BaseModel):
    """A single annotation in response."""

    id: str = Field(..., description="Unique annotation ID")
    media_id: int = Field(..., description="ID of the media item")
    location: str = Field(..., description="Page number or EPUB CFI")
    text: str = Field(..., description="The highlighted/annotated text")
    color: AnnotationColor = Field(..., description="Highlight color")
    note: str | None = Field(None, description="Additional note")
    annotation_type: AnnotationType = Field(
        default=AnnotationType.highlight, description="Type of annotation"
    )
    chapter_title: str | None = Field(
        None, description="Chapter title for EPUB annotations"
    )
    percentage: float | None = Field(
        None, description="Reading percentage (0-100) for EPUB annotations"
    )
    created_at: datetime = Field(..., description="When the annotation was created")
    updated_at: datetime = Field(..., description="When the annotation was last updated")


class AnnotationListResponse(BaseModel):
    """Response containing all annotations for a document."""

    media_id: int = Field(..., description="ID of the media item")
    annotations: list[AnnotationResponse] = Field(
        default_factory=list, description="List of annotations"
    )
    total_count: int = Field(..., ge=0, description="Total number of annotations")


class AnnotationSyncRequest(BaseModel):
    """Request body for batch syncing annotations."""

    annotations: list[AnnotationCreate] = Field(
        ..., description="List of annotations to sync"
    )
    client_ids: list[str] | None = Field(
        None,
        description="Optional client-generated IDs to match with server IDs in response",
    )


    @model_validator(mode="after")
    def validate_client_identities(self) -> Self:
        """A supplied creation identity must cover exactly one annotation."""
        if self.client_ids is not None:
            if len(self.client_ids) != len(self.annotations) or len(set(self.client_ids)) != len(self.client_ids):
                raise ValueError("client_ids must be unique and match the annotation count")
            if any(not value.strip() or len(value) > 128 for value in self.client_ids):
                raise ValueError("client_ids must be nonblank and at most 128 characters")
            try:
                for value in self.client_ids:
                    value.encode("utf-8")
            except UnicodeEncodeError as exc:
                raise ValueError("client_ids must contain valid Unicode") from exc
        return self


class AnnotationSyncResponse(BaseModel):
    """Response from batch sync operation."""

    media_id: int = Field(..., description="ID of the media item")
    synced_count: int = Field(..., ge=0, description="Number of annotations synced")
    annotations: list[AnnotationResponse] = Field(
        default_factory=list, description="Synced annotations with server IDs"
    )
    id_mapping: dict | None = Field(
        None, description="Mapping from client_ids to server IDs if client_ids were provided"
    )


__all__ = [
    "AnnotationColor",
    "AnnotationType",
    "AnnotationCreate",
    "AnnotationUpdate",
    "AnnotationResponse",
    "AnnotationListResponse",
    "AnnotationSyncRequest",
    "AnnotationSyncResponse",
]
