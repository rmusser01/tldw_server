from __future__ import annotations

"""Security helpers for Sync v2 private payload handling."""

from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from typing import Any

from .models import SyncEnvelope, SyncKeyRecord

_REDACTED = "<redacted>"

_PRIVATE_CLEAR_PAYLOAD_ALLOWED_KEYS = {
    "archive_status",
    "archived",
    "attachment_id",
    "attachment_ids",
    "availability",
    "content_type",
    "deleted",
    "entity_kind",
    "entity_type",
    "link_type",
    "media_id",
    "order_key",
    "parent_entity_id",
    "parent_entity_kind",
    "payload_hash",
    "payload_size_bytes",
    "position",
    "record_type",
    "relation_type",
    "relationship",
    "size_bytes",
    "soft_deleted",
    "sort_key",
    "source_id",
    "stable_key",
    "status",
    "sync_status",
    "tag_ids",
    "target_entity_id",
    "target_entity_kind",
    "tombstone",
    "workspace_id",
}

_SENSITIVE_KEYS = {
    "abstract",
    "body",
    "ciphertext",
    "content",
    "description",
    "display_name",
    "excerpt",
    "kdf_metadata",
    "label",
    "message",
    "name",
    "note",
    "payload_ciphertext",
    "payload_clear",
    "plain_text",
    "plaintext",
    "prompt",
    "summary",
    "text",
    "title",
    "transcript",
    "wrapped_key",
    "wrapped_key_blob",
}


class PrivatePayloadValidationError(ValueError):
    """Raised when a private Sync v2 payload exposes plaintext fields."""


def _normalized_key(key: object) -> str:
    return str(key).strip().lower().replace("-", "_")


def _is_sensitive_key(key: object) -> bool:
    normalized = _normalized_key(key)
    return normalized in _SENSITIVE_KEYS or normalized.endswith("_ciphertext")


def validate_private_payload(
    *,
    payload_ciphertext: str | None,
    payload_clear: Mapping[str, Any] | None,
) -> None:
    """Validate that private payload cleartext only contains routing metadata."""

    del payload_ciphertext
    for key in payload_clear or {}:
        if _normalized_key(key) not in _PRIVATE_CLEAR_PAYLOAD_ALLOWED_KEYS:
            raise PrivatePayloadValidationError(
                f"payload_clear.{key} is not allowed for private Sync v2 payloads"
            )


def redact_private_mapping_for_log(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return a log-safe copy of a mapping containing private Sync values."""

    redacted: dict[str, Any] = {}
    for key, item in value.items():
        if _is_sensitive_key(key):
            redacted[str(key)] = _REDACTED
        else:
            redacted[str(key)] = _redact_private_value(item)
    return redacted


def _redact_private_value(item: Any) -> Any:
    if isinstance(item, Mapping):
        return redact_private_mapping_for_log(item)
    if isinstance(item, list):
        return [_redact_private_value(element) for element in item]
    return item


def _dataclass_to_dict(value: object) -> dict[str, Any]:
    if not is_dataclass(value):
        raise TypeError("Sync redaction helpers expect dataclass inputs")
    return asdict(value)


def redact_envelope_for_log(envelope: SyncEnvelope) -> dict[str, Any]:
    """Return a log-safe Sync envelope representation."""

    data = _dataclass_to_dict(envelope)
    data["payload_ciphertext"] = _REDACTED if envelope.payload_ciphertext else None
    data["payload_clear"] = _REDACTED
    return data


def redact_key_record_for_log(record: SyncKeyRecord) -> dict[str, Any]:
    """Return a log-safe Sync key-record representation."""

    data = _dataclass_to_dict(record)
    data["wrapped_key_blob"] = _REDACTED
    data["kdf_metadata"] = _REDACTED
    return data


__all__ = [
    "PrivatePayloadValidationError",
    "redact_envelope_for_log",
    "redact_key_record_for_log",
    "redact_private_mapping_for_log",
    "validate_private_payload",
]
