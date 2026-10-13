"""Allow-listed generation metadata for settled assistant replies.

When the server settles an assistant reply it records how the reply was
produced in ``message_metadata.extra_json`` next to the existing keys
(``sender_role``, ``image_details`` and so on). These keys are the only
generation metadata the server writes or accepts:

``generation_status``
    How the reply ended. One of ``GENERATION_STATUSES``:

    * ``complete``: the provider finished normally.
    * ``length``: the provider stopped at its output-token limit
      (``finish_reason == "length"``).
    * ``interrupted``: generation ended early because the client went away,
      the provider failed mid-stream or the stream stalled. The content is
      the partial reply.
    * ``stopped``: generation ended early because a stop signal asked for
      it. The content is the partial reply.
    * ``error``: reserved for imported replies whose generation failed.
      Server settlement never writes it, because a reply with no usable
      output is not saved.

    The WebUI maps ``interrupted`` to ``generationInfo.interrupted`` and
    ``stopped`` to ``generationInfo.interrupted`` plus ``generationInfo.stopped``.
``model_id``
    The model the request resolved to.
``provider``
    The provider that produced the reply, after any fallback.
``finish_reason``
    The provider's finish reason, when it reported one.
``usage``
    Provider-reported token counts: ``prompt_tokens``, ``completion_tokens``
    and ``total_tokens``. Estimates are never stored here.

Anything else is stripped by ``sanitize_generation_metadata`` (server
settlement) or rejected by ``validate_generation_metadata`` (callers that must
refuse bad input, such as importing local chats). Both return a new dict.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

GENERATION_STATUS_COMPLETE = "complete"
GENERATION_STATUS_STOPPED = "stopped"
GENERATION_STATUS_INTERRUPTED = "interrupted"
GENERATION_STATUS_LENGTH = "length"
GENERATION_STATUS_ERROR = "error"

GENERATION_STATUSES = frozenset(
    {
        GENERATION_STATUS_COMPLETE,
        GENERATION_STATUS_STOPPED,
        GENERATION_STATUS_INTERRUPTED,
        GENERATION_STATUS_LENGTH,
        GENERATION_STATUS_ERROR,
    }
)

USAGE_KEYS = ("prompt_tokens", "completion_tokens", "total_tokens")

GENERATION_METADATA_KEYS = frozenset(
    {"model_id", "provider", "finish_reason", "usage", "generation_status"}
)

_MAX_MODEL_ID_LENGTH = 256
_MAX_TOKEN_COUNT = 2**53
_IDENTIFIER_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,63}")


class GenerationMetadataError(ValueError):
    """Raised when generation metadata has unknown keys or invalid values."""


def _clean_model_id(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    if not cleaned or len(cleaned) > _MAX_MODEL_ID_LENGTH or not cleaned.isprintable():
        return None
    return cleaned


def _clean_identifier(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    if not _IDENTIFIER_PATTERN.fullmatch(cleaned):
        return None
    return cleaned


def _clean_status(value: Any) -> str | None:
    return value if isinstance(value, str) and value in GENERATION_STATUSES else None


def _clean_token_count(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    if value < 0 or value > _MAX_TOKEN_COUNT:
        return None
    return value


def _clean_usage(value: Any, *, strict: bool) -> dict[str, int] | None:
    if not isinstance(value, Mapping):
        return None
    if strict and set(value) - set(USAGE_KEYS):
        return None
    cleaned: dict[str, int] = {}
    for key in USAGE_KEYS:
        if key not in value:
            continue
        count = _clean_token_count(value[key])
        if count is None:
            if strict:
                return None
            continue
        cleaned[key] = count
    return cleaned or None


_CLEANERS = {
    "generation_status": _clean_status,
    "model_id": _clean_model_id,
    "provider": _clean_identifier,
    "finish_reason": _clean_identifier,
}


def _clean_value(key: str, value: Any, *, strict: bool) -> Any | None:
    if key == "usage":
        return _clean_usage(value, strict=strict)
    return _CLEANERS[key](value)


def sanitize_usage(value: Any) -> dict[str, int] | None:
    """Return the allow-listed token counts in ``value``, or None if there are none."""
    return _clean_usage(value, strict=False)


def sanitize_generation_metadata(value: Any) -> dict[str, Any]:
    """Keep only allow-listed keys whose values are valid; drop everything else."""
    if not isinstance(value, Mapping):
        return {}
    cleaned: dict[str, Any] = {}
    for key in GENERATION_METADATA_KEYS:
        if key not in value:
            continue
        item = _clean_value(key, value[key], strict=False)
        if item is not None:
            cleaned[key] = item
    return cleaned


def validate_generation_metadata(value: Any) -> dict[str, Any]:
    """Return a normalized copy, or raise when any key or value is not allowed."""
    if not isinstance(value, Mapping):
        raise GenerationMetadataError("Generation metadata must be an object")
    unknown = sorted(str(key) for key in value if key not in GENERATION_METADATA_KEYS)
    if unknown:
        raise GenerationMetadataError(f"Unsupported generation metadata keys: {', '.join(unknown)}")
    cleaned: dict[str, Any] = {}
    for key, item in value.items():
        normalized = _clean_value(key, item, strict=True)
        if normalized is None:
            raise GenerationMetadataError(f"Invalid generation metadata value for {key}")
        cleaned[key] = normalized
    return cleaned


def build_generation_metadata(
    *,
    generation_status: str,
    model_id: Any = None,
    provider: Any = None,
    finish_reason: Any = None,
    usage: Any = None,
) -> dict[str, Any]:
    """Build settlement metadata; optional values that are missing or invalid are omitted."""
    if _clean_status(generation_status) is None:
        raise GenerationMetadataError(f"Unsupported generation status: {generation_status!r}")
    return sanitize_generation_metadata(
        {
            "generation_status": generation_status,
            "model_id": model_id,
            "provider": provider,
            "finish_reason": finish_reason,
            "usage": usage,
        }
    )


__all__ = [
    "GENERATION_METADATA_KEYS",
    "GENERATION_STATUSES",
    "GENERATION_STATUS_COMPLETE",
    "GENERATION_STATUS_ERROR",
    "GENERATION_STATUS_INTERRUPTED",
    "GENERATION_STATUS_LENGTH",
    "GENERATION_STATUS_STOPPED",
    "USAGE_KEYS",
    "GenerationMetadataError",
    "build_generation_metadata",
    "sanitize_generation_metadata",
    "sanitize_usage",
    "validate_generation_metadata",
]
