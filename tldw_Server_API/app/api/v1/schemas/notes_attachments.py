"""Strict public schemas for canonical Notes attachment lifecycle APIs."""

from __future__ import annotations

import re
from typing import Any, Literal
from uuid import RFC_4122, UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    field_validator,
    model_validator,
)

from tldw_Server_API.app.core.exceptions import NoteAttachmentPolicyError
from tldw_Server_API.app.core.Notes.attachment_policy import (
    canonicalize_note_attachment_file_name,
    validate_note_attachment_content_type,
    validate_note_attachment_original_file_name,
)
from tldw_Server_API.app.core.Sync.v2.models import normalize_sync_timestamp

NotesAttachmentAvailability = Literal[
    "available",
    "metadata_only",
    "missing",
    "verify_failed",
    "quarantined",
    "deleted",
]
NotesAttachmentState = Literal["live", "tombstoned"]
NotesAttachmentSourceKind = Literal["upload", "sync", "legacy_bootstrap"]

_UUID4_RE = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\Z"
)
_OBJECT_HASH_RE = re.compile(r"[0-9a-f]{64}\Z")
_BLOB_HASH_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_ETAG_RE = re.compile(
    r'"att-([0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12})'
    r"-v([1-9][0-9]*)-([0-9a-f]{64})\"\Z"
)


def _canonical_uuid4(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or _UUID4_RE.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a canonical lowercase UUIDv4")
    parsed = UUID(value)
    if parsed.version != 4 or parsed.variant != RFC_4122 or str(parsed) != value:
        raise ValueError(f"{field_name} must be a canonical lowercase UUIDv4")
    return value


def _visible_ascii(value: Any, *, field_name: str, maximum: int) -> str:
    if (
        not isinstance(value, str)
        or not 1 <= len(value) <= maximum
        or any(ord(character) < 0x21 or ord(character) > 0x7E for character in value)
    ):
        raise ValueError(f"{field_name} must be bounded visible ASCII")
    return value


def _canonical_timestamp(value: Any, field_name: str) -> str:
    normalized = normalize_sync_timestamp(value)
    if normalized is None:
        raise ValueError(f"{field_name} must be a canonical timestamp")
    return normalized


def format_notes_attachment_etag(
    attachment_id: str,
    version: int,
    object_hash: str,
) -> str:
    """Return the sole supported strong Notes attachment ETag."""

    canonical_id = _canonical_uuid4(attachment_id, "attachment_id")
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise ValueError("version must be a positive integer")
    if not isinstance(object_hash, str) or _OBJECT_HASH_RE.fullmatch(object_hash) is None:
        raise ValueError("object_hash must be a canonical lowercase SHA-256 digest")
    return f'"att-{canonical_id}-v{version}-{object_hash}"'


def parse_notes_attachment_if_match(value: Any) -> tuple[str, int, str]:
    """Parse exactly one strong Notes attachment validator."""

    if not isinstance(value, str):
        raise ValueError("If-Match requires one strong attachment ETag")
    match = _ETAG_RE.fullmatch(value)
    if match is None:
        raise ValueError("If-Match requires one strong attachment ETag")
    attachment_id, raw_version, object_hash = match.groups()
    _canonical_uuid4(attachment_id, "attachment_id")
    return attachment_id, int(raw_version), object_hash


def validate_notes_attachment_idempotency_key(value: Any) -> str:
    """Validate the public 128-byte visible-ASCII idempotency-key boundary."""

    return _visible_ascii(value, field_name="Idempotency-Key", maximum=128)


def validate_notes_attachment_keyset_cursor(value: Any) -> str:
    """Validate the public 512-byte opaque keyset-cursor boundary."""

    return _visible_ascii(value, field_name="cursor", maximum=512)


class _StrictAttachmentModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class NotesAttachmentRenameRequest(_StrictAttachmentModel):
    """Rename-only canonical attachment mutation."""

    file_name: str

    @field_validator("file_name")
    @classmethod
    def _canonicalize_file_name(cls, value: Any) -> str:
        try:
            return canonicalize_note_attachment_file_name(value)[0]
        except NoteAttachmentPolicyError as exc:
            raise ValueError(str(exc)) from exc


class NotesAttachmentReasonRequest(_StrictAttachmentModel):
    """Optional bounded reason for attachment delete or restore."""

    reason: str | None = Field(default=None, max_length=256)

    @field_validator("reason")
    @classmethod
    def _validate_reason(cls, value: str | None) -> str | None:
        if value is not None and any(ord(character) < 32 for character in value):
            raise ValueError("reason must not contain control characters")
        return value


class NotesAttachmentFromUploadRequest(_StrictAttachmentModel):
    """Bind one completed immutable upload session to its Notes intent."""

    upload_id: str

    @field_validator("upload_id")
    @classmethod
    def _validate_upload_id(cls, value: Any) -> str:
        return _visible_ascii(value, field_name="upload_id", maximum=128)


class NotesAttachmentItem(_StrictAttachmentModel):
    """Canonical owner-authorized attachment metadata response."""

    dataset_id: str = Field(..., min_length=1, max_length=255)
    note_id: str
    attachment_id: str
    file_name: str
    original_file_name: str
    content_type: str
    size_bytes: StrictInt = Field(..., ge=1)
    blob_hash: str
    version: StrictInt = Field(..., ge=1)
    object_hash: str
    state: NotesAttachmentState
    deleted_at: str | None = None
    delete_reason: str | None = Field(default=None, max_length=256)
    created_at: str
    last_modified: str
    created_by: str = Field(..., min_length=1, max_length=255)
    source_kind: NotesAttachmentSourceKind
    availability: NotesAttachmentAvailability
    etag: str

    @field_validator("note_id")
    @classmethod
    def _validate_note_id(cls, value: Any) -> str:
        return _canonical_uuid4(value, "note_id")

    @field_validator("dataset_id", "created_by")
    @classmethod
    def _validate_bounded_authority(cls, value: str, info: Any) -> str:
        if value != value.strip():
            raise ValueError(f"{info.field_name} must already be normalized")
        return value

    @field_validator("attachment_id")
    @classmethod
    def _validate_attachment_id(cls, value: Any) -> str:
        return _canonical_uuid4(value, "attachment_id")

    @field_validator("file_name")
    @classmethod
    def _validate_file_name(cls, value: Any) -> str:
        try:
            display_name, _ = canonicalize_note_attachment_file_name(value)
        except NoteAttachmentPolicyError as exc:
            raise ValueError(str(exc)) from exc
        if value != display_name:
            raise ValueError("file_name must already be canonical")
        return display_name

    @field_validator("original_file_name")
    @classmethod
    def _validate_original_file_name(cls, value: Any) -> str:
        try:
            return validate_note_attachment_original_file_name(value)
        except NoteAttachmentPolicyError as exc:
            raise ValueError(str(exc)) from exc

    @field_validator("content_type")
    @classmethod
    def _validate_content_type(cls, value: Any) -> str:
        try:
            return validate_note_attachment_content_type(value)
        except NoteAttachmentPolicyError as exc:
            raise ValueError(str(exc)) from exc

    @field_validator("blob_hash")
    @classmethod
    def _validate_blob_hash(cls, value: Any) -> str:
        if not isinstance(value, str) or _BLOB_HASH_RE.fullmatch(value) is None:
            raise ValueError("blob_hash must be a canonical lowercase SHA-256 digest")
        return value

    @field_validator("object_hash")
    @classmethod
    def _validate_object_hash(cls, value: Any) -> str:
        if not isinstance(value, str) or _OBJECT_HASH_RE.fullmatch(value) is None:
            raise ValueError("object_hash must be a canonical lowercase SHA-256 digest")
        return value

    @field_validator("created_at", "last_modified")
    @classmethod
    def _validate_timestamp(cls, value: Any, info: Any) -> str:
        return _canonical_timestamp(value, info.field_name)

    @field_validator("deleted_at")
    @classmethod
    def _validate_deleted_at(cls, value: Any) -> str | None:
        return None if value is None else _canonical_timestamp(value, "deleted_at")

    @model_validator(mode="after")
    def _validate_lifecycle_and_etag(self) -> NotesAttachmentItem:
        if self.state == "live" and (self.deleted_at is not None or self.delete_reason is not None):
            raise ValueError("live attachment lifecycle metadata is inconsistent")
        if self.state == "tombstoned" and self.deleted_at is None:
            raise ValueError("tombstoned attachments require deleted_at")
        expected = format_notes_attachment_etag(
            self.attachment_id,
            self.version,
            self.object_hash,
        )
        if self.etag != expected:
            raise ValueError("attachment ETag does not match its immutable identity")
        return self


class NotesAttachmentMutationResponse(NotesAttachmentItem):
    """Canonical attachment mutation response with exact-replay evidence."""

    idempotent_replay: bool = False


class NotesAttachmentPage(_StrictAttachmentModel):
    """Bounded attachment-ID keyset page."""

    items: list[NotesAttachmentItem] = Field(default_factory=list, max_length=200)
    next_cursor: str | None = None
    has_more: bool = False

    @field_validator("next_cursor")
    @classmethod
    def _validate_next_cursor(cls, value: Any) -> str | None:
        return None if value is None else validate_notes_attachment_keyset_cursor(value)

    @model_validator(mode="after")
    def _validate_pagination_state(self) -> NotesAttachmentPage:
        if self.has_more != (self.next_cursor is not None):
            raise ValueError("next_cursor must be present exactly when has_more is true")
        return self


__all__ = [
    "NotesAttachmentAvailability",
    "NotesAttachmentFromUploadRequest",
    "NotesAttachmentItem",
    "NotesAttachmentMutationResponse",
    "NotesAttachmentPage",
    "NotesAttachmentReasonRequest",
    "NotesAttachmentRenameRequest",
    "NotesAttachmentSourceKind",
    "NotesAttachmentState",
    "format_notes_attachment_etag",
    "parse_notes_attachment_if_match",
    "validate_notes_attachment_idempotency_key",
    "validate_notes_attachment_keyset_cursor",
]
