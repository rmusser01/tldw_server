"""Core schemas for the browser extension web clipper."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

WebClipperDestination = Literal["note", "workspace", "both"]
WebClipperOutcomeState = Literal["saved", "saved_with_warnings", "partially_saved", "failed"]
WebClipperEnrichmentStatus = Literal["pending", "running", "complete", "failed"]
WebClipperEnrichmentType = Literal["ocr", "vlm"]

_MAX_CLIP_ID_CHARS = 256
_MAX_CLIP_TYPE_CHARS = 64
_MAX_SOURCE_URL_CHARS = 4096
_MAX_TITLE_CHARS = 512
_MAX_COMMENT_CHARS = 20_000
_MAX_KEYWORDS = 64
_MAX_KEYWORD_CHARS = 128
_MAX_BODY_CHARS = 1_000_000
_MAX_ATTACHMENTS = 16
_MAX_ATTACHMENT_SLOT_CHARS = 128
_MAX_ATTACHMENT_FILENAME_CHARS = 180
_MAX_ATTACHMENT_MEDIA_TYPE_CHARS = 127
_MAX_ATTACHMENT_TEXT_CHARS = 1_000_000
_MAX_ATTACHMENT_BASE64_CHARS = 35_000_000
MAX_CAPTURE_METADATA_JSON_CHARS = 65_536
_MAX_ENRICHMENT_INLINE_SUMMARY_CHARS = 20_000
_MAX_STRUCTURED_PAYLOAD_JSON_CHARS = 262_144


def _json_size_chars(value: Any) -> int:
    try:
        return len(json.dumps(value, ensure_ascii=False))
    except (TypeError, ValueError) as exc:
        raise ValueError("Value must be JSON serializable.") from exc


def validate_capture_metadata_json_size(value: dict[str, Any]) -> dict[str, Any]:
    """Return capture metadata after enforcing the canonical JSON size limit."""
    if _json_size_chars(value) > MAX_CAPTURE_METADATA_JSON_CHARS:
        raise ValueError(f"capture_metadata JSON must be {MAX_CAPTURE_METADATA_JSON_CHARS} characters or fewer.")
    return value


class WebClipperSaveRequest(BaseModel):
    """Typed save request for a browser clip."""

    class NotePayload(BaseModel):
        title: str | None = Field(
            default=None,
            min_length=1,
            max_length=_MAX_TITLE_CHARS,
            description="Optional user-edited note title.",
        )
        comment: str | None = Field(
            default=None,
            max_length=_MAX_COMMENT_CHARS,
            description="Optional user comment for the saved note.",
        )
        folder_id: int | None = Field(default=None, ge=1, description="Optional note folder destination.")
        keywords: list[str] = Field(
            default_factory=list,
            max_length=_MAX_KEYWORDS,
            description="Optional filing keywords for the note.",
        )

        model_config = ConfigDict(extra="forbid")

        @field_validator("keywords")
        @classmethod
        def validate_keyword_lengths(cls, value: list[str]) -> list[str]:
            for keyword in value:
                if len(str(keyword or "").strip()) > _MAX_KEYWORD_CHARS:
                    raise ValueError(f"Keywords must be {_MAX_KEYWORD_CHARS} characters or fewer.")
            return value

    class WorkspacePayload(BaseModel):
        workspace_id: str = Field(..., min_length=1, max_length=128, description="Workspace destination identifier.")

        model_config = ConfigDict(extra="forbid")

    class ContentPayload(BaseModel):
        visible_body: str | None = Field(
            default=None,
            max_length=_MAX_BODY_CHARS,
            description="Primary visible clip body for the saved note.",
        )
        full_extract: str | None = Field(
            default=None,
            max_length=_MAX_BODY_CHARS,
            description="Full extracted page/article text.",
        )
        selected_text: str | None = Field(
            default=None,
            max_length=_MAX_BODY_CHARS,
            description="Explicit user-selected text, when available.",
        )

        model_config = ConfigDict(extra="forbid")

    class AttachmentPayload(BaseModel):
        slot: str = Field(
            ..., min_length=1, max_length=_MAX_ATTACHMENT_SLOT_CHARS, description="Deterministic attachment slot name."
        )
        file_name: str | None = Field(
            default=None,
            min_length=1,
            max_length=_MAX_ATTACHMENT_FILENAME_CHARS,
            description="Original file name supplied by the client.",
        )
        media_type: str = Field(
            ...,
            min_length=1,
            max_length=_MAX_ATTACHMENT_MEDIA_TYPE_CHARS,
            description="Attachment media type.",
        )
        text_content: str | None = Field(
            default=None,
            max_length=_MAX_ATTACHMENT_TEXT_CHARS,
            description="UTF-8 text content for small text attachments.",
        )
        content_base64: str | None = Field(
            default=None,
            max_length=_MAX_ATTACHMENT_BASE64_CHARS,
            description="Base64-encoded attachment bytes.",
        )
        source_url: str | None = Field(
            default=None, max_length=_MAX_SOURCE_URL_CHARS, description="Optional source URL for the captured asset."
        )

        model_config = ConfigDict(extra="forbid")

    class EnhancementOptions(BaseModel):
        run_ocr: bool = Field(default=False, description="Whether OCR should run after save.")
        run_vlm: bool = Field(default=False, description="Whether VLM analysis should run after save.")

        model_config = ConfigDict(extra="forbid")

    clip_id: str = Field(
        ..., min_length=1, max_length=_MAX_CLIP_ID_CHARS, description="Client idempotency key for the clip save."
    )
    clip_type: str = Field(
        ..., min_length=1, max_length=_MAX_CLIP_TYPE_CHARS, description="Clip type chosen by the user."
    )
    source_url: str = Field(..., min_length=1, max_length=_MAX_SOURCE_URL_CHARS, description="Source URL for the clip.")
    source_title: str = Field(..., min_length=1, max_length=_MAX_TITLE_CHARS, description="Source page title.")
    destination_mode: WebClipperDestination = Field(default="note", description="Visible destination choice.")
    note: NotePayload = Field(default_factory=NotePayload, description="Note filing and visible-body options.")
    workspace: WorkspacePayload | None = Field(
        default=None,
        description="Workspace placement target when destination_mode includes a workspace.",
    )
    content: ContentPayload = Field(default_factory=ContentPayload, description="Structured extracted content payload.")
    attachments: list[AttachmentPayload] = Field(
        default_factory=list,
        max_length=_MAX_ATTACHMENTS,
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

    @field_validator("capture_metadata")
    @classmethod
    def validate_capture_metadata_size(cls, value: dict[str, Any]) -> dict[str, Any]:
        return validate_capture_metadata_json_size(value)

    @model_validator(mode="after")
    def validate_workspace_destination(self) -> "WebClipperSaveRequest":
        if self.destination_mode in {"workspace", "both"} and self.workspace is None:
            raise ValueError("workspace is required when destination_mode targets a workspace.")
        return self


class WebClipperEnrichmentPayload(BaseModel):
    """Structured payload for OCR/VLM follow-up work."""

    clip_id: str = Field(..., min_length=1, max_length=_MAX_CLIP_ID_CHARS)
    enrichment_type: WebClipperEnrichmentType = Field(..., description="OCR or VLM enrichment kind.")
    status: WebClipperEnrichmentStatus = Field(default="pending")
    inline_summary: str | None = Field(
        default=None,
        max_length=_MAX_ENRICHMENT_INLINE_SUMMARY_CHARS,
        description="Concise inline summary for the note body.",
    )
    structured_payload: dict[str, Any] = Field(default_factory=dict, description="Full structured enrichment payload.")
    source_note_version: int = Field(
        ...,
        ge=1,
        description="Source note version used when the enrichment was produced.",
    )
    error: str | None = Field(default=None, max_length=4096, description="Optional failure reason.")

    model_config = ConfigDict(extra="forbid")

    @field_validator("structured_payload")
    @classmethod
    def validate_structured_payload_size(cls, value: dict[str, Any]) -> dict[str, Any]:
        if _json_size_chars(value) > _MAX_STRUCTURED_PAYLOAD_JSON_CHARS:
            raise ValueError(
                f"structured_payload JSON must be {_MAX_STRUCTURED_PAYLOAD_JSON_CHARS} characters or fewer."
            )
        return value


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
