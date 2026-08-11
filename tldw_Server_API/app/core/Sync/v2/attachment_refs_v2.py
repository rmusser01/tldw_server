"""Strict wire contract for Notes ``attachment.ref`` adapter version 2."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from datetime import datetime
from typing import Literal, cast
from uuid import RFC_4122, UUID

from pydantic import (
    UUID4,
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    ValidationError,
    field_validator,
)

from .models import normalize_sync_timestamp

_LOWERCASE_SHA256_RE = re.compile(r"sha256:[0-9a-f]{64}")
_IMMUTABLE_FIELDS = (
    "attachment_id",
    "parent_domain",
    "parent_object_id",
    "original_file_name",
    "created_at",
    "created_by",
)


class AttachmentRefV2ValidationError(ValueError):
    """Stable validation failure for the attachment-ref v2 contract."""


class AttachmentRefV2Payload(BaseModel):
    """Canonical whole-object payload for one Notes attachment reference."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    attachment_id: UUID4
    parent_domain: Literal["notes.note"]
    parent_object_id: UUID4
    file_name: str = Field(min_length=1)
    original_file_name: str = Field(min_length=1)
    content_type: str = Field(min_length=1)
    size_bytes: StrictInt = Field(ge=1)
    blob_hash: str
    created_at: str
    last_modified: str
    created_by: str = Field(min_length=1)

    @field_validator("attachment_id", "parent_object_id", mode="before")
    @classmethod
    def _validate_uuid4(cls, value: object) -> object:
        if not isinstance(value, str):
            raise ValueError("attachment.ref v2 IDs must be canonical lowercase UUIDv4 strings")
        try:
            parsed = UUID(value)
        except ValueError as exc:
            raise ValueError(
                "attachment.ref v2 IDs must be canonical lowercase UUIDv4 strings"
            ) from exc
        if parsed.version != 4 or parsed.variant != RFC_4122 or str(parsed) != value:
            raise ValueError(
                "attachment.ref v2 IDs must be canonical lowercase UUIDv4 strings"
            )
        return value

    @field_validator("blob_hash")
    @classmethod
    def _validate_blob_hash(cls, value: str) -> str:
        if _LOWERCASE_SHA256_RE.fullmatch(value) is None:
            raise ValueError(
                "attachment.ref v2 blob_hash must be a canonical lowercase SHA-256 digest"
            )
        return value

    @field_validator("created_at", "last_modified")
    @classmethod
    def _validate_timestamp(cls, value: str) -> str:
        return _canonical_timestamp(value)


def parse_attachment_ref_v2_payload(
    payload: Mapping[str, object],
) -> AttachmentRefV2Payload:
    """Parse one strict v2 payload without mutating or enriching it."""

    if not isinstance(payload, Mapping):
        raise AttachmentRefV2ValidationError(
            "attachment.ref v2 payload must be an object"
        )
    try:
        return AttachmentRefV2Payload.model_validate(dict(payload))
    except ValidationError as exc:
        message = str(exc).replace(
            "Extra inputs are not permitted", "extra inputs are not permitted"
        )
        raise AttachmentRefV2ValidationError(message) from exc


def validate_attachment_ref_v2(
    payload: Mapping[str, object] | AttachmentRefV2Payload,
    *,
    envelope_created_at_client: str,
    authenticated_device_id: str,
    prior_payload: Mapping[str, object] | AttachmentRefV2Payload | None = None,
    trusted_server_origin: bool = False,
    verified_bootstrap: bool = False,
) -> AttachmentRefV2Payload:
    """Validate immutable identity and replay-stable creation provenance."""

    parsed = (
        payload
        if isinstance(payload, AttachmentRefV2Payload)
        else parse_attachment_ref_v2_payload(payload)
    )
    prior = None
    if prior_payload is not None:
        prior = (
            prior_payload
            if isinstance(prior_payload, AttachmentRefV2Payload)
            else parse_attachment_ref_v2_payload(prior_payload)
        )

    mutation_timestamp = _normalized_envelope_timestamp(envelope_created_at_client)
    if parsed.last_modified != mutation_timestamp:
        raise AttachmentRefV2ValidationError(
            "attachment.ref v2 provenance requires last_modified to match the canonical mutation timestamp"
        )

    if prior is not None:
        for field_name in _IMMUTABLE_FIELDS:
            if getattr(parsed, field_name) != getattr(prior, field_name):
                raise AttachmentRefV2ValidationError(
                    f"attachment.ref v2 immutable field changed: {field_name}"
                )
        return parsed

    if verified_bootstrap:
        if not trusted_server_origin:
            raise AttachmentRefV2ValidationError(
                "attachment.ref v2 legacy provenance requires a verified trusted bootstrap"
            )
        return parsed

    if parsed.created_at != mutation_timestamp:
        raise AttachmentRefV2ValidationError(
            "attachment.ref v2 creation provenance must match created_at_client"
        )
    if parsed.created_by != authenticated_device_id:
        raise AttachmentRefV2ValidationError(
            "attachment.ref v2 creation provenance must match the authenticated device"
        )
    return parsed


def validate_attachment_ref_v2_object_id(value: str) -> str:
    """Return one canonical lowercase UUIDv4 object ID or raise."""

    try:
        parsed = AttachmentRefV2Payload.model_validate(
            {
                "attachment_id": value,
                "parent_domain": "notes.note",
                "parent_object_id": value,
                "file_name": "_",
                "original_file_name": "_",
                "content_type": "application/octet-stream",
                "size_bytes": 1,
                "blob_hash": "sha256:" + "0" * 64,
                "created_at": "1970-01-01T00:00:00+00:00",
                "last_modified": "1970-01-01T00:00:00+00:00",
                "created_by": "_",
            }
        )
    except ValidationError as exc:
        raise AttachmentRefV2ValidationError(
            "attachment.ref v2 object_id must be a canonical lowercase UUIDv4 string"
        ) from exc
    return str(parsed.attachment_id)


def validate_attachment_ref_v2_routing_metadata(
    operation: str,
    routing_metadata: Mapping[str, object],
) -> dict[str, object]:
    """Validate restore intent without allowing it into the protected payload."""

    normalized = dict(routing_metadata)
    restore_intent = normalized.get("restore_intent")
    if (restore_intent is not None and restore_intent is not True) or (
        restore_intent is True and operation != "upsert"
    ):
        raise AttachmentRefV2ValidationError(
            "attachment.ref v2 restore_intent must be boolean true on an upsert"
        )
    return normalized


def attachment_ref_v2_object_hash(
    payload: Mapping[str, object] | AttachmentRefV2Payload,
) -> str:
    """Hash only the canonical semantic attachment payload fields."""

    parsed = (
        payload
        if isinstance(payload, AttachmentRefV2Payload)
        else parse_attachment_ref_v2_payload(payload)
    )
    canonical = json.dumps(
        parsed.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(canonical).hexdigest()


def _normalized_envelope_timestamp(value: str) -> str:
    normalized = normalize_sync_timestamp(value)
    if normalized is None:
        raise AttachmentRefV2ValidationError(
            "attachment.ref v2 mutation timestamp is required"
        )
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise AttachmentRefV2ValidationError(
            "attachment.ref v2 mutation timestamp is invalid"
        ) from exc
    if parsed.tzinfo is None:
        raise AttachmentRefV2ValidationError(
            "attachment.ref v2 mutation timestamp must include a timezone"
        )
    return normalized


def _canonical_timestamp(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("attachment.ref v2 timestamps must be canonical UTC strings")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(
            "attachment.ref v2 timestamps must be canonical UTC strings"
        ) from exc
    normalized = normalize_sync_timestamp(parsed)
    if parsed.tzinfo is None or normalized != value:
        raise ValueError("attachment.ref v2 timestamps must be canonical UTC strings")
    return cast(str, normalized)


__all__ = [
    "AttachmentRefV2Payload",
    "AttachmentRefV2ValidationError",
    "attachment_ref_v2_object_hash",
    "parse_attachment_ref_v2_payload",
    "validate_attachment_ref_v2",
    "validate_attachment_ref_v2_object_id",
    "validate_attachment_ref_v2_routing_metadata",
]
