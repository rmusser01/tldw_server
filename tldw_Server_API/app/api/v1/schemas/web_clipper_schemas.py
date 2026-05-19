"""Schemas for the browser extension web clipper."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

WebClipperDestination = Literal["note", "workspace", "both"]
WebClipperOutcomeState = Literal["saved", "saved_with_warnings", "partially_saved", "failed"]
WebClipperEnrichmentStatus = Literal["pending", "running", "complete", "failed"]
WebClipperEnrichmentType = Literal["ocr", "vlm"]


class WebClipperSaveRequest(BaseModel):
    """Typed save request for a browser clip."""

    class NotePayload(BaseModel):
        title: str | None = Field(default=None, min_length=1, description="Optional user-edited note title.")
        comment: str | None = Field(default=None, description="Optional user comment for the saved note.")
        folder_id: int | None = Field(default=None, ge=1, description="Optional note folder destination.")
        keywords: list[str] = Field(default_factory=list, description="Optional filing keywords for the note.")

        model_config = ConfigDict(extra="forbid")

    class WorkspacePayload(BaseModel):
        workspace_id: str = Field(..., min_length=1, description="Workspace destination identifier.")

        model_config = ConfigDict(extra="forbid")

    class ContentPayload(BaseModel):
        visible_body: str | None = Field(default=None, description="Primary visible clip body for the saved note.")
        full_extract: str | None = Field(default=None, description="Full extracted page/article text.")
        selected_text: str | None = Field(default=None, description="Explicit user-selected text, when available.")

        model_config = ConfigDict(extra="forbid")

    class AttachmentPayload(BaseModel):
        slot: str = Field(..., min_length=1, description="Deterministic attachment slot name.")
        file_name: str | None = Field(default=None, min_length=1, description="Original file name supplied by the client.")
        media_type: str = Field(..., min_length=1, description="Attachment media type.")
        text_content: str | None = Field(default=None, description="UTF-8 text content for small text attachments.")
        content_base64: str | None = Field(default=None, description="Base64-encoded attachment bytes.")
        source_url: str | None = Field(default=None, description="Optional source URL for the captured asset.")

        model_config = ConfigDict(extra="forbid")

    class EnhancementOptions(BaseModel):
        run_ocr: bool = Field(default=False, description="Whether OCR should run after save.")
        run_vlm: bool = Field(default=False, description="Whether VLM analysis should run after save.")

        model_config = ConfigDict(extra="forbid")

    clip_id: str = Field(..., min_length=1, description="Client idempotency key for the clip save.")
    clip_type: str = Field(..., min_length=1, description="Clip type chosen by the user.")
    source_url: str = Field(..., min_length=1, description="Source URL for the clip.")
    source_title: str = Field(..., min_length=1, description="Source page title.")
    destination_mode: WebClipperDestination = Field(default="note", description="Visible destination choice.")
    note: NotePayload = Field(default_factory=NotePayload, description="Note filing and visible-body options.")
    workspace: WorkspacePayload | None = Field(
        default=None,
        description="Workspace placement target when destination_mode includes a workspace.",
    )
    content: ContentPayload = Field(default_factory=ContentPayload, description="Structured extracted content payload.")
    attachments: list[AttachmentPayload] = Field(
        default_factory=list,
        description="Structured attachment inputs for deterministic slot persistence.",
    )
    enhancements: EnhancementOptions = Field(
        default_factory=EnhancementOptions,
        description="Requested OCR/VLM follow-up actions.",
    )
    capture_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Structured capture metadata and fallback information.",
    )
    source_note_version: int | None = Field(
        default=None,
        ge=1,
        description="Optional source note version for idempotent follow-up stages.",
    )

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    @model_validator(mode="after")
    def validate_workspace_destination(self) -> "WebClipperSaveRequest":
        if self.destination_mode in {"workspace", "both"} and self.workspace is None:
            raise ValueError("workspace is required when destination_mode targets a workspace.")
        return self


class WebClipperEnrichmentPayload(BaseModel):
    """Structured payload for OCR/VLM follow-up work."""

    clip_id: str = Field(..., min_length=1)
    enrichment_type: WebClipperEnrichmentType = Field(..., description="OCR or VLM enrichment kind.")
    status: WebClipperEnrichmentStatus = Field(default="pending")
    inline_summary: str | None = Field(default=None, description="Concise inline summary for the note body.")
    structured_payload: dict[str, Any] = Field(default_factory=dict, description="Full structured enrichment payload.")
    source_note_version: int = Field(
        ...,
        ge=1,
        description="Source note version used when the enrichment was produced.",
    )
    error: str | None = Field(default=None, description="Optional failure reason.")

    model_config = ConfigDict(extra="forbid")


class WebClipperSavedNote(BaseModel):
    """Canonical note information returned from clipper flows."""

    id: str = Field(..., min_length=1)
    title: str = Field(..., min_length=1)
    version: int = Field(..., ge=1)

    model_config = ConfigDict(extra="forbid")


class WebClipperWorkspacePlacement(BaseModel):
    """Read-only workspace placement summary for a canonical clip note."""

    workspace_id: str = Field(..., min_length=1)
    workspace_note_id: int = Field(..., ge=1)
    source_note_id: str = Field(..., min_length=1)
    source_note_version: int | None = Field(default=None, ge=1)

    model_config = ConfigDict(extra="forbid")


class WebClipperAttachmentRecord(BaseModel):
    """Persisted attachment metadata returned by the clipper service."""

    slot: str = Field(..., min_length=1)
    file_name: str = Field(..., min_length=1)
    original_file_name: str = Field(..., min_length=1)
    content_type: str | None = Field(default=None)
    size_bytes: int = Field(..., ge=0)
    uploaded_at: datetime
    url: str = Field(..., min_length=1)

    model_config = ConfigDict(extra="forbid")


class WebClipperSaveResponse(BaseModel):
    """Detailed outcome payload for a clip save."""

    clip_id: str = Field(..., min_length=1)
    status: WebClipperOutcomeState = Field(..., description="High-level save outcome state.")
    note: WebClipperSavedNote | None = None
    workspace_placement: WebClipperWorkspacePlacement | None = None
    attachments: list[WebClipperAttachmentRecord] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    note_id: str = Field(..., min_length=1, description="Backward-compatible canonical note identifier.")
    workspace_placement_saved: bool = Field(default=False)
    workspace_placement_count: int = Field(default=0, ge=0)

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class WebClipperSaveResult(BaseModel):
    """Backward-compatible minimal outcome payload for earlier Task 1 tests."""

    clip_id: str = Field(..., min_length=1)
    note_id: str = Field(..., min_length=1)
    status: WebClipperOutcomeState = Field(..., description="High-level save outcome state.")
    workspace_placement_saved: bool = Field(default=False)
    workspace_placement_count: int = Field(default=0, ge=0)
    warnings: list[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class WebClipperStatusResponse(BaseModel):
    """Current canonical state for a saved clip."""

    clip_id: str = Field(..., min_length=1)
    status: WebClipperOutcomeState = Field(...)
    note: WebClipperSavedNote
    workspace_placements: list[WebClipperWorkspacePlacement] = Field(default_factory=list)
    attachments: list[WebClipperAttachmentRecord] = Field(default_factory=list)
    analysis: dict[str, Any] = Field(default_factory=dict)
    content_budget: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")


class WebClipperEnrichmentResponse(BaseModel):
    """Result of storing an OCR/VLM enrichment payload."""

    clip_id: str = Field(..., min_length=1)
    enrichment_type: WebClipperEnrichmentType
    status: WebClipperEnrichmentStatus
    source_note_version: int = Field(..., ge=1)
    inline_applied: bool = Field(default=False)
    inline_summary: str | None = Field(default=None)
    conflict_reason: str | None = Field(default=None)
    warnings: list[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")
