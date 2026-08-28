"""Shared storage boundary for character-chat conversation settings."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

from tldw_Server_API.app.core.Character_Chat.character_behavior_snapshot import (
    is_credential_key,
)
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import InputError

MAX_CHAT_SETTINGS_BYTES = 200_000
MAX_CHAT_SETTINGS_DEPTH = 32


class ChatSettingsSizeError(InputError):
    """Raised when canonical settings exceed the storage byte budget."""


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
) -> dict[str, Any]:
    """Validate the final finite JSON settings object before persistence."""
    if not isinstance(settings, Mapping):
        raise InputError("Settings payload must be an object.")
    _validate_json_value(
        settings,
        depth=0,
        active_containers=set(),
        reject_credentials=reject_credentials,
    )
    normalized = dict(settings)
    try:
        encoded = json.dumps(
            normalized,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise InputError(f"Invalid settings payload: {exc}") from exc
    if len(encoded) > MAX_CHAT_SETTINGS_BYTES:
        raise ChatSettingsSizeError(
            f"Settings payload exceeds {MAX_CHAT_SETTINGS_BYTES} bytes."
        )
    return normalized


__all__ = [
    "ChatSettingsSizeError",
    "MAX_CHAT_SETTINGS_BYTES",
    "MAX_CHAT_SETTINGS_DEPTH",
    "validate_chat_settings_storage",
]
