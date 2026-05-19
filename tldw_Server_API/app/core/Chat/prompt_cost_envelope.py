"""
Prompt cost envelope helpers.

These helpers produce bounded diagnostics for provider-bound chat messages.
They intentionally do not mutate provider payloads or persist prompt text.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

FINGERPRINT_VERSION = "prompt-v1"

_DATA_URI_RE = re.compile(r'(data:image[^,]*,)[^"\s]+', re.IGNORECASE)
_KNOWN_PART_TYPES = frozenset(
    {
        "text",
        "input_text",
        "image_url",
        "input_image",
        "file",
        "input_file",
        "tool_result",
    }
)
_SEGMENT_KINDS = ("static", "world_book", "retrieval_tool", "history", "user_turn")


@dataclass(frozen=True)
class PromptSegment:
    """A bounded diagnostic record for one prompt segment."""

    name: str
    kind: str
    fingerprint: str
    estimated_tokens: int
    text_length: int
    role: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_diagnostics(self) -> dict[str, Any]:
        """Return a prompt-safe diagnostic representation."""
        return {
            "name": self.name,
            "kind": self.kind,
            "role": self.role,
            "fingerprint": self.fingerprint,
            "estimated_tokens": self.estimated_tokens,
            "text_length": self.text_length,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class PromptCostEnvelope:
    """Bounded prompt diagnostics for a final provider-bound request."""

    fingerprint_version: str
    aggregate_fingerprint: str
    total_estimated_tokens: int
    segments: tuple[PromptSegment, ...]
    segment_token_totals: Mapping[str, int]
    message_count: int

    def to_diagnostics(self) -> dict[str, Any]:
        """Return prompt-safe envelope diagnostics without raw prompt text."""
        return {
            "fingerprint_version": self.fingerprint_version,
            "aggregate_fingerprint": self.aggregate_fingerprint,
            "total_estimated_tokens": self.total_estimated_tokens,
            "message_count": self.message_count,
            "segment_token_totals": dict(self.segment_token_totals),
            "segments": [segment.to_diagnostics() for segment in self.segments],
        }


def canonicalize_messages(messages: Sequence[Mapping[str, Any]]) -> str:
    """Canonicalize provider-bound messages for stable fingerprinting."""
    normalized = [
        _normalize_message(message)
        for message in messages
        if isinstance(message, Mapping)
    ]
    return _canonical_json(normalized)


def fingerprint_text(text: str, *, version: str = FINGERPRINT_VERSION) -> str:
    """Return a versioned SHA-256 fingerprint for text."""
    digest = hashlib.sha256(text.encode("utf-8", errors="surrogatepass")).hexdigest()
    return f"{version}:sha256:{digest}"


def estimate_segment_tokens(text: str) -> int:
    """Conservative deterministic estimate using the existing 4 chars/token heuristic."""
    if not text:
        return 0
    sanitized = _sanitize_data_uris(text)
    return max(0, (len(sanitized) + 3) // 4)


def build_prompt_cost_envelope(
    messages: Sequence[Mapping[str, Any]],
    *,
    world_book_text: str | None = None,
    retrieval_text: str | None = None,
    version: str = FINGERPRINT_VERSION,
) -> PromptCostEnvelope:
    """Build bounded diagnostics for final provider-bound chat messages."""
    message_list = [message for message in messages if isinstance(message, Mapping)]
    last_user_index = _last_role_index(message_list, "user")
    segments: list[PromptSegment] = []

    for index, message in enumerate(message_list):
        role = _coerce_role(message.get("role"))
        kind = _message_segment_kind(role, index, last_user_index)
        text_for_estimate = _message_text_for_estimate(message)
        canonical_message = canonicalize_messages([message])
        estimate_basis = text_for_estimate or canonical_message
        segments.append(
            PromptSegment(
                name=f"message:{index}",
                kind=kind,
                role=role,
                fingerprint=fingerprint_text(canonical_message, version=version),
                estimated_tokens=estimate_segment_tokens(estimate_basis),
                text_length=len(text_for_estimate),
                metadata={"message_index": index},
            )
        )

    if world_book_text:
        segments.append(_text_segment("world_book", "world_book", world_book_text, version=version))
    if retrieval_text:
        segments.append(_text_segment("retrieval", "retrieval_tool", retrieval_text, version=version))

    totals = {kind: 0 for kind in _SEGMENT_KINDS}
    for segment in segments:
        totals[segment.kind] = totals.get(segment.kind, 0) + segment.estimated_tokens

    aggregate_source = _canonical_json(
        {
            "version": version,
            "segments": [
                {
                    "name": segment.name,
                    "kind": segment.kind,
                    "fingerprint": segment.fingerprint,
                }
                for segment in segments
            ],
        }
    )
    total_tokens = sum(segment.estimated_tokens for segment in segments)
    return PromptCostEnvelope(
        fingerprint_version=version,
        aggregate_fingerprint=fingerprint_text(aggregate_source, version=version),
        total_estimated_tokens=total_tokens,
        segments=tuple(segments),
        segment_token_totals=totals,
        message_count=len(message_list),
    )


def _text_segment(name: str, kind: str, text: str, *, version: str) -> PromptSegment:
    return PromptSegment(
        name=name,
        kind=kind,
        role=None,
        fingerprint=fingerprint_text(_sanitize_data_uris(text), version=version),
        estimated_tokens=estimate_segment_tokens(text),
        text_length=len(text),
    )


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def _sanitize_data_uris(text: str) -> str:
    return _DATA_URI_RE.sub(r"\1<omitted>", text)


def _normalize_message(message: Mapping[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key in sorted(str(key) for key in message.keys()):
        value = message.get(key)
        if key == "content":
            normalized[key] = _normalize_content(value)
        else:
            normalized[key] = _normalize_value(value)
    return normalized


def _normalize_content(content: Any) -> Any:
    if isinstance(content, str):
        return _sanitize_data_uris(content)
    if isinstance(content, list):
        return [_normalize_content_part(part) for part in content]
    return _normalize_value(content)


def _normalize_content_part(part: Any) -> Any:
    if isinstance(part, str):
        return _sanitize_data_uris(part)
    if not isinstance(part, Mapping):
        return _normalize_value(part)

    part_type = part.get("type")
    if part_type is not None and str(part_type) in _KNOWN_PART_TYPES:
        return {
            str(key): _normalize_value(value)
            for key, value in sorted(part.items(), key=lambda item: str(item[0]))
        }

    return {
        "type": str(part_type) if part_type is not None else "unknown",
        "unsupported_part": True,
        "keys": sorted(str(key) for key in part.keys())[:16],
    }


def _normalize_value(value: Any, *, depth: int = 0) -> Any:
    if depth > 6:
        return {"truncated": True, "type": type(value).__name__}
    if value is None or isinstance(value, bool | int | float):
        return value
    if isinstance(value, str):
        return _sanitize_data_uris(value)
    if isinstance(value, Mapping):
        return {
            str(key): _normalize_value(val, depth=depth + 1)
            for key, val in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, list | tuple):
        return [_normalize_value(item, depth=depth + 1) for item in value]
    return {"type": type(value).__name__}


def _last_role_index(messages: Sequence[Mapping[str, Any]], role: str) -> int | None:
    for index in range(len(messages) - 1, -1, -1):
        if _coerce_role(messages[index].get("role")) == role:
            return index
    return None


def _coerce_role(role: Any) -> str | None:
    if role is None:
        return None
    return str(role)


def _message_segment_kind(role: str | None, index: int, last_user_index: int | None) -> str:
    if role in {"system", "developer"}:
        return "static"
    if role in {"tool", "function"}:
        return "retrieval_tool"
    if role == "user" and index == last_user_index:
        return "user_turn"
    return "history"


def _message_text_for_estimate(message: Mapping[str, Any]) -> str:
    return _extract_text(message.get("content"))


def _extract_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return _sanitize_data_uris(content)
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(_sanitize_data_uris(part))
            elif isinstance(part, Mapping):
                text = part.get("text")
                if isinstance(text, str):
                    parts.append(_sanitize_data_uris(text))
        return "\n".join(part for part in parts if part)
    if isinstance(content, Mapping):
        return _canonical_json(_normalize_value(content))
    return str(type(content).__name__)


__all__ = [
    "FINGERPRINT_VERSION",
    "PromptCostEnvelope",
    "PromptSegment",
    "build_prompt_cost_envelope",
    "canonicalize_messages",
    "estimate_segment_tokens",
    "fingerprint_text",
]
