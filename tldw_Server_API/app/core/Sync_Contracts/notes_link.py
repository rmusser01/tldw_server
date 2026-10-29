"""Strict Sync v2 payload contract for explicit Notes links."""

from __future__ import annotations

import json
import math
import uuid
from collections.abc import Mapping
from datetime import datetime
from typing import Any, Literal, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    ValidationError,
    field_validator,
    model_validator,
)

from tldw_Server_API.app.core.exceptions import NotesLinkValidationError

from .models import normalize_sync_timestamp
from .notes_link_contract import (
    NOTES_LINK_LABEL_MAX_CHARS,
    NOTES_LINK_PROPERTIES_MAX_BYTES,
    NOTES_LINK_PROPERTIES_MAX_DEPTH,
    NOTES_LINK_PROPERTIES_MAX_KEYS,
    NOTES_LINK_REASON_MAX_CHARS,
    NOTES_LINK_WEIGHT_MAX,
)


class NotesLinkUpsertPayload(BaseModel):
    """Canonical protected payload for one live explicit Notes link."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_note_id: str
    target_note_id: str
    type: Literal["manual"]
    directed: StrictBool
    weight: float = Field(ge=0.0, le=NOTES_LINK_WEIGHT_MAX)
    label: str | None = Field(default=None, max_length=NOTES_LINK_LABEL_MAX_CHARS)
    properties: dict[str, Any] = Field(default_factory=dict)
    created_at: str
    last_modified: str
    created_by: str = Field(min_length=1)

    @field_validator("source_note_id", "target_note_id")
    @classmethod
    def _validate_note_id(cls, value: str) -> str:
        return _canonical_uuid4(value, field_name="note endpoint")

    @field_validator("weight", mode="before")
    @classmethod
    def _validate_weight(cls, value: object) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("notes.link weight must be a finite number")
        try:
            normalized = float(value)
        except (OverflowError, TypeError, ValueError) as exc:
            raise ValueError("notes.link weight must be a finite number") from exc
        if not math.isfinite(normalized):
            raise ValueError("notes.link weight must be a finite number")
        return normalized

    @field_validator("properties")
    @classmethod
    def _validate_properties(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _canonical_properties(value)

    @field_validator("created_at", "last_modified")
    @classmethod
    def _validate_timestamp(cls, value: str) -> str:
        return _canonical_timestamp(value)

    @model_validator(mode="after")
    def _validate_identity(self) -> NotesLinkUpsertPayload:
        if self.source_note_id == self.target_note_id:
            raise ValueError("notes.link self-links are not allowed; endpoints must differ")
        if not self.directed and self.source_note_id > self.target_note_id:
            raise ValueError("undirected notes.link endpoints must use canonical order")
        return self


class NotesLinkTombstonePayload(NotesLinkUpsertPayload):
    """Canonical protected payload for one explicit-link tombstone."""

    deleted_at: str
    reason: str | None = Field(default=None, max_length=NOTES_LINK_REASON_MAX_CHARS)

    @field_validator("deleted_at")
    @classmethod
    def _validate_deleted_at(cls, value: str) -> str:
        return _canonical_timestamp(value)


def parse_notes_link_payload(
    operation: str,
    payload: Mapping[str, object],
) -> dict[str, object]:
    """Validate and return one canonical ``notes.link`` payload."""

    if operation not in {"upsert", "tombstone"}:
        raise NotesLinkValidationError(f"unsupported notes.link operation: {operation}")
    if not isinstance(payload, Mapping):
        raise NotesLinkValidationError("notes.link payload must be an object")
    model = NotesLinkUpsertPayload if operation == "upsert" else NotesLinkTombstonePayload
    try:
        parsed = model.model_validate(dict(payload))
    except ValidationError as exc:
        raise NotesLinkValidationError(str(exc)) from exc
    return cast(dict[str, object], parsed.model_dump())


def validate_notes_link_object_id(value: str) -> str:
    """Return a canonical lowercase UUIDv4 edge ID or raise."""

    return _canonical_uuid4(value, field_name="notes.link object_id")


def validate_notes_link_properties(value: object) -> dict[str, Any]:
    """Return one canonical bounded properties object or raise."""

    try:
        return _canonical_properties(value)
    except ValueError as exc:
        raise NotesLinkValidationError(str(exc)) from exc


def validate_notes_link_provenance(
    payload: Mapping[str, object],
    *,
    envelope_created_at_client: str,
    authenticated_device_id: str,
    prior_payload: Mapping[str, object] | None,
    trusted_bootstrap: bool = False,
) -> None:
    """Validate replay-stable creation and mutation provenance."""

    if trusted_bootstrap and prior_payload is None:
        return
    envelope_timestamp = _canonical_timestamp(envelope_created_at_client)
    if payload.get("last_modified") != envelope_timestamp:
        raise NotesLinkValidationError("notes.link last_modified must match created_at_client")
    if "deleted_at" in payload and payload.get("deleted_at") != envelope_timestamp:
        raise NotesLinkValidationError("notes.link deleted_at must match created_at_client")

    if prior_payload is not None:
        for field_name in ("created_at", "created_by"):
            if payload.get(field_name) != prior_payload.get(field_name):
                raise NotesLinkValidationError(f"notes.link {field_name} must match the current object")
        return

    if payload.get("created_at") != envelope_timestamp:
        raise NotesLinkValidationError("notes.link created_at must match created_at_client on create")
    if payload.get("created_by") != authenticated_device_id:
        raise NotesLinkValidationError("notes.link created_by must match the authenticated device")


def _canonical_uuid4(value: object, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise NotesLinkValidationError(f"{field_name} must be a canonical UUIDv4 string")
    try:
        parsed = uuid.UUID(value)
    except ValueError as exc:
        raise NotesLinkValidationError(f"{field_name} must be a canonical UUIDv4 string") from exc
    if parsed.version != 4 or parsed.variant != uuid.RFC_4122 or str(parsed) != value:
        raise NotesLinkValidationError(f"{field_name} must be a canonical UUIDv4 string")
    return value


def _canonical_timestamp(value: object) -> str:
    if not isinstance(value, str):
        raise NotesLinkValidationError("notes.link timestamp must be a canonical UTC string")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise NotesLinkValidationError("notes.link timestamp must be a canonical UTC string") from exc
    normalized = normalize_sync_timestamp(parsed)
    if parsed.tzinfo is None or normalized != value:
        raise NotesLinkValidationError("notes.link timestamp must be a canonical UTC string")
    return value


def _canonical_properties(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("notes.link properties must be an object")
    if len(value) > NOTES_LINK_PROPERTIES_MAX_KEYS:
        raise ValueError(f"notes.link properties may contain at most {NOTES_LINK_PROPERTIES_MAX_KEYS} keys")
    if not _json_object_keys_are_strings(value):
        raise ValueError("notes.link properties keys must be strings")
    if _json_depth(value) > NOTES_LINK_PROPERTIES_MAX_DEPTH:
        raise ValueError(f"notes.link properties depth must not exceed {NOTES_LINK_PROPERTIES_MAX_DEPTH}")
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("notes.link properties must contain canonical JSON values") from exc
    if len(encoded) > NOTES_LINK_PROPERTIES_MAX_BYTES:
        raise ValueError(f"notes.link properties must not exceed {NOTES_LINK_PROPERTIES_MAX_BYTES} bytes")
    return cast(dict[str, Any], json.loads(encoded))


def _json_depth(value: object) -> int:
    if isinstance(value, dict):
        return 1 + max((_json_depth(item) for item in value.values()), default=0)
    if isinstance(value, (list, tuple)):
        return 1 + max((_json_depth(item) for item in value), default=0)
    return 0


def _json_object_keys_are_strings(value: object) -> bool:
    if isinstance(value, dict):
        return all(isinstance(key, str) and _json_object_keys_are_strings(item) for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return all(_json_object_keys_are_strings(item) for item in value)
    return True


__all__ = [
    "NOTES_LINK_LABEL_MAX_CHARS",
    "NOTES_LINK_PROPERTIES_MAX_BYTES",
    "NOTES_LINK_PROPERTIES_MAX_DEPTH",
    "NOTES_LINK_PROPERTIES_MAX_KEYS",
    "NOTES_LINK_REASON_MAX_CHARS",
    "NOTES_LINK_WEIGHT_MAX",
    "NotesLinkTombstonePayload",
    "NotesLinkUpsertPayload",
    "NotesLinkValidationError",
    "parse_notes_link_payload",
    "validate_notes_link_object_id",
    "validate_notes_link_properties",
    "validate_notes_link_provenance",
]
