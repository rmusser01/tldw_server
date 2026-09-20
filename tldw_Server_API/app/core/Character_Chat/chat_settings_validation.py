"""Shared storage boundary for character-chat conversation settings."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

from tldw_Server_API.app.core.Character_Chat.character_behavior_snapshot import (
    DEFAULT_MAX_SNAPSHOT_BYTES,
    is_credential_key,
)
from tldw_Server_API.app.core.DB_Management.chacha.conversation_resume_store import (
    validate_materialized_behavior_settings,
    validate_roleplay_readiness_settings,
)
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import InputError

MAX_CHAT_SETTINGS_BYTES = 200_000
MAX_CHAT_SETTINGS_DEPTH = 32
PENDING_GREETING_SETTINGS_KEY = "roleplayPendingGreetingV1"
INTERNAL_CHAT_SETTINGS_KEYS = frozenset(
    {"roleplayResumeV1", "roleplayBehaviorV1", PENDING_GREETING_SETTINGS_KEY}
)
MAX_CHAT_SETTINGS_TOTAL_BYTES = (
    MAX_CHAT_SETTINGS_BYTES + 2 * DEFAULT_MAX_SNAPSHOT_BYTES + 16 * 1024
)


class ChatSettingsSizeError(InputError):
    """Raised when canonical settings exceed the storage byte budget."""


def _canonical_bytes(value: Any, *, label: str) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (RecursionError, TypeError, ValueError) as exc:
        raise InputError(f"Invalid {label}: {exc}") from exc


def _snapshot_binding(value: Any) -> dict[str, Any] | None:
    if isinstance(value, Mapping):
        if value.get("status") not in {None, "valid"}:
            return None
        schema_version = value.get("schema_version")
        digest = value.get("digest")
    else:
        schema_version = getattr(value, "schema_version", None)
        digest = getattr(value, "digest", None)
    if type(schema_version) is not int or not isinstance(digest, str):
        return None
    return {"schema_version": schema_version, "digest": digest}


def _pending_greeting_digest(values: Mapping[str, Any]) -> str:
    canonical = _canonical_bytes(
        {"schemaVersion": 1, "values": dict(values)},
        label="pending greeting authority",
    )
    if len(canonical) > DEFAULT_MAX_SNAPSHOT_BYTES:
        raise InputError("Pending greeting authority exceeds the size limit.")
    return f"sha256:{hashlib.sha256(canonical).hexdigest()}"


def build_pending_greeting_record(values: Mapping[str, Any]) -> dict[str, Any]:
    """Build one bounded, digest-protected pending greeting record."""
    normalized = json.loads(
        _canonical_bytes(dict(values), label="pending greeting authority")
    )
    return {
        "schemaVersion": 1,
        "digest": _pending_greeting_digest(normalized),
        "values": normalized,
    }


def validate_pending_greeting_record(
    value: Any,
    *,
    settings: Mapping[str, Any],
    behavior_snapshot: Any,
    conversation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate pending greeting shape, digest, and immutable bindings."""
    if not isinstance(value, Mapping) or set(value) != {
        "schemaVersion",
        "digest",
        "values",
    }:
        raise InputError("Stored pending greeting authority is invalid.")
    values = value.get("values")
    if not isinstance(values, Mapping) or set(values) != {
        "base_snapshot",
        "character_id",
        "greetings_checksum",
        "greeting",
    }:
        raise InputError("Stored pending greeting authority is invalid.")
    greeting = values.get("greeting")
    if not isinstance(greeting, Mapping) or set(greeting) != {
        "content",
        "selection_id",
        "source",
        "source_index",
        "character_version",
    }:
        raise InputError("Stored pending greeting authority is invalid.")
    binding = _snapshot_binding(behavior_snapshot)
    if (
        value.get("schemaVersion") != 1
        or not isinstance(value.get("digest"), str)
        or binding is None
        or values.get("base_snapshot") != binding
        or not isinstance(values.get("greetings_checksum"), str)
        or values.get("greetings_checksum") != settings.get("greetingsChecksum")
        or not isinstance(greeting.get("content"), str)
        or greeting.get("selection_id") != settings.get("greetingSelectionId")
        or greeting.get("source") not in {"first_message", "alternate_greeting"}
        or type(greeting.get("source_index")) is not int
        or greeting["source_index"] < 0
        or type(greeting.get("character_version")) is not int
        or greeting["character_version"] < 1
        or (
            conversation is not None
            and values.get("character_id") != int(conversation.get("character_id"))
        )
        or value.get("digest") != _pending_greeting_digest(values)
    ):
        raise InputError("Stored pending greeting authority is invalid.")
    return dict(value)


def _validate_json_value(
    value: Any,
    *,
    depth: int,
    active_containers: set[int],
    reject_credentials: bool,
) -> None:
    if depth > MAX_CHAT_SETTINGS_DEPTH:
        raise InputError(
            f"Settings payload exceeds maximum JSON depth {MAX_CHAT_SETTINGS_DEPTH}."
        )
    if value is None or isinstance(value, (bool, str, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise InputError("Settings payload numbers must be finite JSON values.")
        return
    if isinstance(value, Mapping):
        container_id = id(value)
        if container_id in active_containers:
            raise InputError("Settings payload must not contain cycles.")
        active_containers.add(container_id)
        try:
            for key, item in value.items():
                if not isinstance(key, str):
                    raise InputError("Settings payload object keys must be strings.")
                if reject_credentials and is_credential_key(key):
                    raise InputError(
                        f"Settings payload contains credential-bearing key {key!r}."
                    )
                _validate_json_value(
                    item,
                    depth=depth + 1,
                    active_containers=active_containers,
                    reject_credentials=reject_credentials,
                )
        finally:
            active_containers.remove(container_id)
        return
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray, memoryview),
    ):
        container_id = id(value)
        if container_id in active_containers:
            raise InputError("Settings payload must not contain cycles.")
        active_containers.add(container_id)
        try:
            for item in value:
                _validate_json_value(
                    item,
                    depth=depth + 1,
                    active_containers=active_containers,
                    reject_credentials=reject_credentials,
                )
        finally:
            active_containers.remove(container_id)
        return
    raise InputError(
        f"Settings payload contains non-JSON value {type(value).__name__}."
    )


def validate_chat_settings_storage(
    settings: Mapping[str, Any],
    *,
    reject_credentials: bool = False,
    allow_internal: bool = False,
    behavior_snapshot: Any = None,
    conversation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate public settings and independently bounded server authority."""
    if not isinstance(settings, Mapping):
        raise InputError("Settings payload must be an object.")
    normalized = dict(settings)
    internal_keys = INTERNAL_CHAT_SETTINGS_KEYS.intersection(normalized)
    if internal_keys and not allow_internal:
        raise InputError(f"{sorted(internal_keys)[0]} is reserved server-owned state.")
    public = {
        key: value
        for key, value in normalized.items()
        if key not in INTERNAL_CHAT_SETTINGS_KEYS
    }
    _validate_json_value(
        public,
        depth=0,
        active_containers=set(),
        reject_credentials=reject_credentials,
    )
    public_encoded = _canonical_bytes(public, label="settings payload")
    if len(public_encoded) > MAX_CHAT_SETTINGS_BYTES:
        raise ChatSettingsSizeError(
            f"Settings payload exceeds {MAX_CHAT_SETTINGS_BYTES} bytes."
        )

    if allow_internal and internal_keys:
        snapshot_binding = _snapshot_binding(behavior_snapshot)
        behavior = normalized.get("roleplayBehaviorV1")
        if behavior is not None:
            if snapshot_binding is None:
                raise InputError("Stored roleplay behavior requires a valid snapshot.")
            behavior = validate_materialized_behavior_settings(
                behavior,
                snapshot_binding=snapshot_binding,
            )
        readiness = normalized.get("roleplayResumeV1")
        if readiness is None:
            raise InputError("Stored roleplay authority requires a readiness marker.")
        readiness = validate_roleplay_readiness_settings(
            readiness,
            materialized_behavior=behavior,
        )
        pending = normalized.get(PENDING_GREETING_SETTINGS_KEY)
        if pending is not None:
            if behavior is not None or readiness.get("resumeEligible") is not False:
                raise InputError("Pending greeting authority requires an ineligible chat.")
            validate_pending_greeting_record(
                pending,
                settings=normalized,
                behavior_snapshot=behavior_snapshot,
                conversation=conversation,
            )

    total_encoded = _canonical_bytes(normalized, label="settings storage row")
    if len(total_encoded) > MAX_CHAT_SETTINGS_TOTAL_BYTES:
        raise ChatSettingsSizeError(
            f"Settings storage row exceeds {MAX_CHAT_SETTINGS_TOTAL_BYTES} bytes."
        )
    return normalized


__all__ = [
    "ChatSettingsSizeError",
    "INTERNAL_CHAT_SETTINGS_KEYS",
    "MAX_CHAT_SETTINGS_BYTES",
    "MAX_CHAT_SETTINGS_DEPTH",
    "MAX_CHAT_SETTINGS_TOTAL_BYTES",
    "PENDING_GREETING_SETTINGS_KEY",
    "build_pending_greeting_record",
    "validate_pending_greeting_record",
    "validate_chat_settings_storage",
]
