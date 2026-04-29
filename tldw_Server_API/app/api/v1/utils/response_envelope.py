"""Helpers for constructing and detecting API response envelopes."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from tldw_Server_API.app.api.v1.schemas.response_envelope import ResponseEnvelope


T = TypeVar("T")

_ENVELOPE_PAYLOAD_KEYS = frozenset(("data", "error", "error_code", "metadata"))


def envelope_success(
    data: T | None = None,
    metadata: dict[str, Any] | None = None,
) -> ResponseEnvelope[T]:
    """Build a successful response envelope without changing route semantics."""

    return ResponseEnvelope[T](
        success=True,
        data=data,
        metadata=metadata,
    )


def envelope_error(
    error: str,
    *,
    error_code: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> ResponseEnvelope[None]:
    """Build an error response envelope for routes that opt into the contract."""

    return ResponseEnvelope[None](
        success=False,
        error=error,
        error_code=error_code,
        metadata=metadata,
    )


def is_response_envelope(value: object) -> bool:
    """Return true only for canonical envelope-shaped mappings.

    Several legacy endpoints return domain payloads with a top-level
    ``success`` flag. Requiring at least one canonical envelope contract key
    avoids misclassifying those legacy shapes before migration.
    """

    if not isinstance(value, Mapping):
        return False
    if not isinstance(value.get("success"), bool):
        return False
    return any(key in value for key in _ENVELOPE_PAYLOAD_KEYS)


__all__ = [
    "ResponseEnvelope",
    "envelope_error",
    "envelope_success",
    "is_response_envelope",
]
