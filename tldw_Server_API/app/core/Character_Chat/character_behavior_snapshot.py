"""Canonical version-1 behavior snapshots for character conversations."""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from tldw_Server_API.app.core.exceptions import BehaviorSnapshotValidationError

SNAPSHOT_SCHEMA_VERSION = 1
DEFAULT_MAX_SNAPSHOT_BYTES = 1024 * 1024

_SNAPSHOT_KEYS = frozenset({"schema_version", "participants", "routing_defaults"})
_PARTICIPANT_KEYS = frozenset(
    {
        "source",
        "identity",
        "prompt",
        "greeting",
        "generation_defaults",
        "exemplars",
        "world_books",
        "default_memory",
    }
)
_SOURCE_KEYS = frozenset({"kind", "id", "version"})
_SOURCE_KINDS = frozenset({"character"})
_IDENTITY_KEYS = frozenset({"name", "aliases"})
_PROMPT_KEYS = frozenset(
    {
        "system_prompt",
        "description",
        "personality",
        "scenario",
        "message_example",
        "post_history_instructions",
        "prompt_relevant_extensions",
    }
)
_PROMPT_TEXT_KEYS = _PROMPT_KEYS - {"prompt_relevant_extensions"}
_GREETING_KEYS = frozenset({"content", "source", "source_index"})
_ROUTING_KEYS = frozenset({"turn_taking_mode"})
_CREDENTIAL_KEYS = frozenset(
    {
        "api_key",
        "apikey",
        "api_token",
        "access_token",
        "auth_token",
        "authorization",
        "bearer_token",
        "client_secret",
        "csrf_token",
        "credential",
        "credentials",
        "github_token",
        "hf_token",
        "huggingface_token",
        "id_token",
        "oauth_token",
        "password",
        "private_key",
        "refresh_token",
        "secret",
        "session_token",
        "x_api_key",
    }
)
_CREDENTIAL_SUFFIXES = tuple(
    f"_{term}" for term in sorted(_CREDENTIAL_KEYS) if "_" in term
)
_CREDENTIAL_SEPARATOR_RE = re.compile(r"[\W_]+")
_LOWER_TO_UPPER_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_ACRONYM_TO_WORD_RE = re.compile(r"(?<=[A-Z])(?=[A-Z][a-z])")


@dataclass(frozen=True)
class BehaviorSnapshotV1:
    """Frozen boundary around a canonical behavior-snapshot payload."""

    schema_version: int
    canonical_bytes: bytes
    digest: str
    size_bytes: int

    @property
    def payload(self) -> dict[str, Any]:
        """Return a defensive payload copy decoded from the canonical authority."""
        return json.loads(self.canonical_bytes)


def build_behavior_snapshot(
    payload: Mapping[str, Any],
    *,
    max_bytes: int = DEFAULT_MAX_SNAPSHOT_BYTES,
) -> BehaviorSnapshotV1:
    """Validate, copy, and canonically encode a version-1 behavior snapshot."""
    if not isinstance(payload, Mapping):
        raise BehaviorSnapshotValidationError("snapshot must be an object")
    if type(max_bytes) is not int or max_bytes <= 0:
        raise BehaviorSnapshotValidationError("max_bytes must be a positive integer")

    normalized = _normalize_json(dict(payload), path="snapshot")
    _validate_snapshot(normalized)
    canonical_bytes = json.dumps(
        normalized,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if len(canonical_bytes) > max_bytes:
        raise BehaviorSnapshotValidationError(
            f"behavior snapshot size {len(canonical_bytes)} exceeds maximum {max_bytes} bytes"
        )
    return BehaviorSnapshotV1(
        schema_version=SNAPSHOT_SCHEMA_VERSION,
        canonical_bytes=canonical_bytes,
        digest=f"sha256:{hashlib.sha256(canonical_bytes).hexdigest()}",
        size_bytes=len(canonical_bytes),
    )


def _normalize_json(value: Any, *, path: str) -> Any:
    """Normalize JSON-compatible values into a canonical defensive copy."""
    if isinstance(value, str):
        return value.replace("\r\n", "\n").replace("\r", "\n")
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise BehaviorSnapshotValidationError(f"{path} floats must be finite")
        return value
    if isinstance(value, (bytes, bytearray, memoryview)):
        raise BehaviorSnapshotValidationError(f"{path} must not contain binary values")
    if isinstance(value, list):
        return [
            _normalize_json(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, dict):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise BehaviorSnapshotValidationError(
                    f"{path} keys must be JSON-compatible strings"
                )
            normalized_key = key.replace("\r\n", "\n").replace("\r", "\n")
            if normalized_key in normalized:
                raise BehaviorSnapshotValidationError(
                    f"{path} has duplicate keys after line-ending normalization"
                )
            normalized[normalized_key] = _normalize_json(
                item,
                path=f"{path}.{normalized_key}",
            )
        return normalized
    raise BehaviorSnapshotValidationError(
        f"{path} must contain only JSON-compatible values"
    )


def _validate_snapshot(snapshot: dict[str, Any]) -> None:
    """Validate the closed top-level behavior snapshot contract."""
    _require_exact_keys(snapshot, _SNAPSHOT_KEYS, path="snapshot")
    if type(snapshot["schema_version"]) is not int or snapshot["schema_version"] != 1:
        raise BehaviorSnapshotValidationError("snapshot.schema_version must equal 1")

    participants = snapshot["participants"]
    if not isinstance(participants, list):
        raise BehaviorSnapshotValidationError("snapshot.participants must be a list")
    if not participants:
        raise BehaviorSnapshotValidationError(
            "snapshot must contain at least one participant"
        )

    seen_sources: set[tuple[str, str]] = set()
    for index, participant in enumerate(participants):
        source_identity = _validate_participant(participant, index=index)
        if source_identity in seen_sources:
            raise BehaviorSnapshotValidationError(
                "snapshot contains duplicate participant source "
                f"{source_identity[0]}:{source_identity[1]}"
            )
        seen_sources.add(source_identity)

    routing = _require_object(snapshot["routing_defaults"], path="snapshot.routing_defaults")
    _require_exact_keys(routing, _ROUTING_KEYS, path="snapshot.routing_defaults")
    if routing["turn_taking_mode"] != "single":
        raise BehaviorSnapshotValidationError(
            "snapshot.routing_defaults.turn_taking_mode must equal 'single'"
        )


def _validate_participant(participant: Any, *, index: int) -> tuple[str, str]:
    """Validate one participant and return its stable source identity."""
    path = f"snapshot.participants[{index}]"
    participant = _require_object(participant, path=path)
    _require_exact_keys(participant, _PARTICIPANT_KEYS, path=path)

    source = _require_object(participant["source"], path=f"{path}.source")
    _require_exact_keys(source, _SOURCE_KEYS, path=f"{path}.source")
    if not isinstance(source["kind"], str) or source["kind"] not in _SOURCE_KINDS:
        raise BehaviorSnapshotValidationError(
            f"{path}.source.kind must be one of {sorted(_SOURCE_KINDS)}"
        )
    if not isinstance(source["id"], str) or not source["id"]:
        raise BehaviorSnapshotValidationError(
            f"{path}.source.id must be a non-empty string"
        )
    if type(source["version"]) is not int or source["version"] < 1:
        raise BehaviorSnapshotValidationError(
            f"{path}.source.version must be a positive integer"
        )

    identity = _require_object(participant["identity"], path=f"{path}.identity")
    _require_exact_keys(identity, _IDENTITY_KEYS, path=f"{path}.identity")
    if not isinstance(identity["name"], str) or not identity["name"]:
        raise BehaviorSnapshotValidationError(
            f"{path}.identity.name must be a non-empty string"
        )
    if not isinstance(identity["aliases"], list) or not all(
        isinstance(alias, str) for alias in identity["aliases"]
    ):
        raise BehaviorSnapshotValidationError(
            f"{path}.identity.aliases must be a list of strings"
        )

    prompt = _require_object(participant["prompt"], path=f"{path}.prompt")
    _require_exact_keys(prompt, _PROMPT_KEYS, path=f"{path}.prompt")
    for key in _PROMPT_TEXT_KEYS:
        if not isinstance(prompt[key], str):
            raise BehaviorSnapshotValidationError(
                f"{path}.prompt.{key} must be a string"
            )
    extensions = _require_object(
        prompt["prompt_relevant_extensions"],
        path=f"{path}.prompt.prompt_relevant_extensions",
    )
    _reject_credential_keys(extensions, path=f"{path}.prompt.prompt_relevant_extensions")

    greeting = _require_object(participant["greeting"], path=f"{path}.greeting")
    _require_exact_keys(greeting, _GREETING_KEYS, path=f"{path}.greeting")
    if not isinstance(greeting["content"], str):
        raise BehaviorSnapshotValidationError(
            f"{path}.greeting.content must be a string"
        )
    if not isinstance(greeting["source"], str) or not greeting["source"]:
        raise BehaviorSnapshotValidationError(
            f"{path}.greeting.source must be a non-empty string"
        )
    if type(greeting["source_index"]) is not int or greeting["source_index"] < 0:
        raise BehaviorSnapshotValidationError(
            f"{path}.greeting.source_index must be a non-negative integer"
        )

    generation_defaults = _require_object(
        participant["generation_defaults"],
        path=f"{path}.generation_defaults",
    )
    _reject_credential_keys(generation_defaults, path=f"{path}.generation_defaults")

    for field_name in ("exemplars", "world_books"):
        entries = participant[field_name]
        if not isinstance(entries, list):
            raise BehaviorSnapshotValidationError(
                f"{path}.{field_name} must be a list"
            )
        for entry_index, entry in enumerate(entries):
            entry_path = f"{path}.{field_name}[{entry_index}]"
            entry = _require_object(entry, path=entry_path)
            _reject_credential_keys(entry, path=entry_path)

    memory = participant["default_memory"]
    if memory is not None:
        memory = _require_object(memory, path=f"{path}.default_memory")
        _reject_credential_keys(memory, path=f"{path}.default_memory")

    return source["kind"], source["id"]


def _require_object(value: Any, *, path: str) -> dict[str, Any]:
    """Require a plain JSON object at the supplied snapshot path."""
    if not isinstance(value, dict):
        raise BehaviorSnapshotValidationError(f"{path} must be an object")
    return value


def _require_exact_keys(value: dict[str, Any], allowed: frozenset[str], *, path: str) -> None:
    """Require exactly the allowed keys for one closed-schema object."""
    unexpected = sorted(value.keys() - allowed)
    if unexpected:
        raise BehaviorSnapshotValidationError(
            f"{path} has unexpected keys: {unexpected}"
        )
    missing = sorted(allowed - value.keys())
    if missing:
        raise BehaviorSnapshotValidationError(f"{path} has missing keys: {missing}")


def _reject_credential_keys(value: Any, *, path: str) -> None:
    """Reject credential-like keys recursively from extensible snapshot fields."""
    if isinstance(value, list):
        for index, item in enumerate(value):
            _reject_credential_keys(item, path=f"{path}[{index}]")
        return
    if not isinstance(value, dict):
        return
    for key, item in value.items():
        if is_credential_key(key):
            raise BehaviorSnapshotValidationError(
                f"{path} contains credential-like key {key!r}"
            )
        _reject_credential_keys(item, path=f"{path}.{key}")


def is_credential_key(key: str) -> bool:
    """Return whether a JSON key names an explicitly classified credential."""
    normalized = unicodedata.normalize("NFKC", key)
    normalized = _LOWER_TO_UPPER_RE.sub("_", normalized)
    normalized = _ACRONYM_TO_WORD_RE.sub("_", normalized)
    normalized = _CREDENTIAL_SEPARATOR_RE.sub("_", normalized.casefold()).strip("_")
    if normalized in _CREDENTIAL_KEYS:
        return True
    if normalized.endswith(_CREDENTIAL_SUFFIXES):
        return True
    parts = normalized.split("_")
    if len(parts) < 2:
        return False
    suffix = tuple(parts[-2:])
    return (
        parts[-1] == "secret"
        or suffix
        in {
            ("api", "key"),
            ("api", "token"),
            ("private", "key"),
        }
        or (parts[-1] == "key" and "secret" in parts[:-1])
    )


__all__ = [
    "BehaviorSnapshotV1",
    "BehaviorSnapshotValidationError",
    "DEFAULT_MAX_SNAPSHOT_BYTES",
    "SNAPSHOT_SCHEMA_VERSION",
    "build_behavior_snapshot",
    "is_credential_key",
]
