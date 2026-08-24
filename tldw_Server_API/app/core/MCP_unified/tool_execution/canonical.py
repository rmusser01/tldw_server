"""Strict bounded canonical JSON helpers for MCP execution state."""

from __future__ import annotations

import json
import math
from typing import TypeAlias, cast

JsonValue: TypeAlias = (
    None
    | bool
    | int
    | float
    | str
    | list["JsonValue"]
    | dict[str, "JsonValue"]
)

TOOL_DEFINITION_MAX_BYTES = 1_000_000
SCOPE_REPORTING_MAX_BYTES = 256_000
PREPARED_HMAC_PAYLOAD_MAX_BYTES = 64_000
IDEMPOTENCY_RESULT_DEFAULT_MAX_BYTES = 256_000
IDEMPOTENCY_RESULT_HARD_MAX_BYTES = 1_000_000
ARGUMENTS_MAX_BYTES = 1_000_000


class CanonicalJsonTooLarge(ValueError):
    """Raised when canonical JSON exceeds its explicit byte budget."""

    def __init__(self, *, max_bytes: int, actual_bytes: int) -> None:
        self.max_bytes = max_bytes
        self.actual_bytes = actual_bytes
        super().__init__(f"Canonical JSON exceeds the {max_bytes}-byte limit")


def _validate_max_bytes(max_bytes: int) -> None:
    if type(max_bytes) is not int or max_bytes < 1:
        raise ValueError("max_bytes must be a positive non-boolean integer")


def _validate_json_value(value: object, active_containers: set[int]) -> None:
    value_type = type(value)
    if value is None or value_type in {bool, int, str}:
        return
    if value_type is float:
        if not math.isfinite(cast(float, value)):
            raise ValueError("Canonical JSON numbers must be finite")
        return
    if value_type not in {list, dict}:
        raise TypeError("Canonical JSON contains an unsupported value type")

    container_id = id(value)
    if container_id in active_containers:
        raise TypeError("Cyclic JSON structure is not supported")
    active_containers.add(container_id)
    try:
        if value_type is list:
            for item in cast(list[object], value):
                _validate_json_value(item, active_containers)
            return

        for key, item in cast(dict[object, object], value).items():
            if type(key) is not str:
                raise TypeError("Canonical JSON object keys must be strings")
            _validate_json_value(item, active_containers)
    finally:
        active_containers.remove(container_id)


def canonical_json_bytes(value: JsonValue, *, max_bytes: int) -> bytes:
    """Return strict sorted compact UTF-8 JSON within ``max_bytes``."""

    _validate_max_bytes(max_bytes)
    _validate_json_value(value, set())
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if len(encoded) > max_bytes:
        raise CanonicalJsonTooLarge(max_bytes=max_bytes, actual_bytes=len(encoded))
    return encoded


def _reject_json_constant(value: str) -> None:
    del value
    raise ValueError("Canonical JSON numbers must be finite")


def decode_canonical_json(encoded: bytes, *, max_bytes: int) -> JsonValue:
    """Decode strict JSON into a fresh validated object graph."""

    _validate_max_bytes(max_bytes)
    if type(encoded) is not bytes:
        raise TypeError("Canonical JSON encoding must be bytes")
    if len(encoded) > max_bytes:
        raise CanonicalJsonTooLarge(max_bytes=max_bytes, actual_bytes=len(encoded))
    value = json.loads(
        encoded.decode("utf-8"),
        parse_constant=_reject_json_constant,
    )
    _validate_json_value(value, set())
    return cast(JsonValue, value)


def decode_canonical_json_object(encoded: bytes, *, max_bytes: int) -> dict[str, JsonValue]:
    """Decode a fresh canonical JSON object and reject other top-level shapes."""

    value = decode_canonical_json(encoded, max_bytes=max_bytes)
    if type(value) is not dict:
        raise TypeError("Expected a top-level JSON object")
    return cast(dict[str, JsonValue], value)


def decode_canonical_json_object_or_none(
    encoded: bytes,
    *,
    max_bytes: int,
) -> dict[str, JsonValue] | None:
    """Decode a fresh canonical JSON object or JSON null."""

    value = decode_canonical_json(encoded, max_bytes=max_bytes)
    if value is None:
        return None
    if type(value) is not dict:
        raise TypeError("Expected a top-level JSON object or null")
    return cast(dict[str, JsonValue], value)
