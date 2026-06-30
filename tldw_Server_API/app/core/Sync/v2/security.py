from __future__ import annotations

"""Security helpers for Sync v2 private payload handling."""

import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any

from .models import SyncEnvelope, SyncKeyRecord

_REDACTED = "<redacted>"
_SERVER_TRUSTED_POLICY = "server_trusted_v1"
_AT_REST_ATTESTATION_SCOPE = "user_database_directory"
_AT_REST_COVERED_FILES = ["Sync_v2.db", "ChaChaNotes.db"]
_VALID_AT_REST_ENCRYPTION_MODES = frozenset(
    {"encrypted_volume", "managed_storage", "development_unencrypted"}
)
_AT_REST_COVERED_MODES = frozenset({"encrypted_volume", "managed_storage"})
_TRUE_ENV_VALUES = frozenset({"1", "true", "yes", "on"})

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


@dataclass(frozen=True, slots=True)
class SyncV2ServerTrustedEncryptionStatus:
    """Explicit Sync v2 M1 server-trusted encryption readiness posture."""

    mode: str | None
    enabled: bool
    auth_mode: str | None
    ready: bool
    configured: bool
    development: bool = False
    _warnings: tuple[dict[str, str], ...] = field(default_factory=tuple)

    @property
    def encryption(self) -> dict[str, Any]:
        """Return the public capability shape for server_trusted_v1."""

        return {
            "policy": _SERVER_TRUSTED_POLICY,
            "ready": self.ready,
            "attestation": {
                "scope": _AT_REST_ATTESTATION_SCOPE,
                "covers": list(_AT_REST_COVERED_FILES),
                "configured": self.configured,
                "mode": self.mode,
                "server_trusted_enabled": self.enabled,
                "auth_mode": self.auth_mode,
                "development": self.development,
            },
        }

    @property
    def warnings(self) -> list[dict[str, str]]:
        """Return copy-safe warnings explaining a not-ready posture."""

        return [dict(item) for item in self._warnings]


def server_trusted_encryption_status_from_env() -> SyncV2ServerTrustedEncryptionStatus:
    """Build Sync v2 server-trusted readiness from deterministic environment config."""

    return server_trusted_encryption_status_from_config(
        mode=os.getenv("SYNC_V2_AT_REST_ENCRYPTION_MODE"),
        server_trusted_enabled=os.getenv("SYNC_V2_SERVER_TRUSTED_ENABLED"),
        auth_mode=os.getenv("AUTH_MODE"),
    )


def server_trusted_encryption_status_from_config(
    *,
    mode: str | None,
    server_trusted_enabled: bool | str | None,
    auth_mode: str | None,
) -> SyncV2ServerTrustedEncryptionStatus:
    """Evaluate deterministic Sync v2 M1 server-trusted readiness settings."""

    normalized_mode = _normalize_mode(mode)
    enabled = _parse_bool(server_trusted_enabled)
    normalized_auth_mode = _normalize_optional_string(auth_mode)
    recognized_mode = (
        normalized_mode if normalized_mode in _VALID_AT_REST_ENCRYPTION_MODES else None
    )
    development = recognized_mode == "development_unencrypted"
    configured = bool(enabled and recognized_mode)
    ready = bool(enabled and recognized_mode in _AT_REST_COVERED_MODES)
    warnings = _server_trusted_warnings(
        mode=normalized_mode,
        recognized_mode=recognized_mode,
        ready=ready,
        configured=configured,
        development=development,
    )
    return SyncV2ServerTrustedEncryptionStatus(
        mode=recognized_mode,
        enabled=enabled,
        auth_mode=normalized_auth_mode,
        ready=ready,
        configured=configured,
        development=development,
        _warnings=tuple(warnings),
    )


def _normalize_optional_string(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _normalize_mode(value: str | None) -> str | None:
    normalized = _normalize_optional_string(value)
    if normalized is None:
        return None
    return normalized.lower()


def _parse_bool(value: bool | str | None) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in _TRUE_ENV_VALUES


def _server_trusted_warnings(
    *,
    mode: str | None,
    recognized_mode: str | None,
    ready: bool,
    configured: bool,
    development: bool,
) -> list[dict[str, str]]:
    if ready:
        return []
    if development:
        return [
            {
                "code": "sync_development_unencrypted",
                "message": (
                    "Sync v2 M1 server_trusted_v1 is configured for development "
                    "unencrypted storage and is not ready for M1 sync."
                ),
            }
        ]
    if mode is not None and recognized_mode is None:
        return [
            {
                "code": "sync_encryption_attestation_required",
                "message": (
                    "Sync v2 M1 requires a valid at-rest encryption mode for the "
                    "user database directory."
                ),
            }
        ]
    if not configured:
        return [
            {
                "code": "sync_encryption_attestation_required",
                "message": (
                    "Sync v2 M1 requires deployment-level at-rest encryption "
                    "coverage for the user database directory."
                ),
            }
        ]
    return [
        {
            "code": "sync_encryption_attestation_required",
            "message": (
                "Sync v2 M1 requires at-rest encryption coverage for both "
                "Sync_v2.db and ChaChaNotes.db."
            ),
        }
    ]


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
    "SyncV2ServerTrustedEncryptionStatus",
    "redact_envelope_for_log",
    "redact_key_record_for_log",
    "redact_private_mapping_for_log",
    "server_trusted_encryption_status_from_config",
    "server_trusted_encryption_status_from_env",
    "validate_private_payload",
]
