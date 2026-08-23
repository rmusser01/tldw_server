"""Bounded context snapshots for chat macro execution."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

_SECRET_KEY_FRAGMENTS = (
    "authorization",
    "credential",
    "password",
    "secret",
    "token",
    "api_key",
    "apikey",
    "access_key",
)
_BEARER_RE = re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]+", re.IGNORECASE)
_SECRET_ASSIGNMENT_RE = re.compile(
    r"\b(api[_-]?key|apikey|token|password|secret|authorization)\s*[:=]\s*([^\s,;]+)",
    re.IGNORECASE,
)
_OPENAI_KEY_RE = re.compile(r"\bsk-[A-Za-z0-9][A-Za-z0-9_-]{6,}\b")
_BASIC_AUTH_RE = re.compile(r"\bBasic\s+[A-Za-z0-9+/=]+", re.IGNORECASE)
_AWS_ACCESS_KEY_RE = re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")
_GITHUB_TOKEN_RE = re.compile(
    r"\b(?:gh[opsu]_[A-Za-z0-9_]{10,}|github_pat_[A-Za-z0-9_]{10,})\b"
)
_JWT_RE = re.compile(r"\b[A-Za-z0-9_-]{4,}\.[A-Za-z0-9_-]{4,}\.[A-Za-z0-9_-]{4,}\b")
DEFAULT_MAX_EXCERPT_CHARS = 800
MAX_SANITIZE_DEPTH = 8


class MacroContextSnapshot(BaseModel):
    """JSON-safe, bounded macro execution context captured at dispatch time."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    conversation_id: str | None = None
    workspace_id: str | None = None
    acp_session_id: str | None = None
    messages: list[dict[str, Any]] = Field(default_factory=list)
    selected_message_ids: list[str] = Field(default_factory=list)
    selected_source_ids: dict[str, list[str]] = Field(default_factory=dict)
    model_selection: dict[str, Any] = Field(default_factory=dict)
    output_profile: str = "default"
    token_estimate: int = 0
    acp: dict[str, Any] = Field(default_factory=dict)


def build_macro_context_snapshot(
    *,
    chat_db: Any,
    conversation_id: str | None,
    workspace_id: str | None,
    acp_session_id: str | None,
    request_messages: Sequence[Any] | None,
    model_selection: Mapping[str, Any] | None,
    output_profile: str | None,
    request_metadata: Mapping[str, Any] | None = None,
    max_excerpt_chars: int = DEFAULT_MAX_EXCERPT_CHARS,
) -> MacroContextSnapshot:
    """Build a bounded, secret-safe snapshot for later macro execution."""
    del chat_db  # Reserved for future DB-backed message/source lookups.
    metadata = _sanitize_mapping(request_metadata or {})
    messages = [
        _message_excerpt(message, max_excerpt_chars=max_excerpt_chars)
        for message in (request_messages or [])
    ]
    messages = [message for message in messages if message is not None]

    selected_source_ids: dict[str, list[str]] = {}
    rag_ids = _string_list(metadata.get("selected_rag_ids"))
    media_ids = _string_list(metadata.get("selected_media_ids"))
    if rag_ids:
        selected_source_ids["rag"] = rag_ids
    if media_ids:
        selected_source_ids["media"] = media_ids

    safe_model_selection = _sanitize_model_selection(model_selection or {})
    token_estimate = sum(max(1, len(str(message.get("excerpt", ""))) // 4) for message in messages)

    return MacroContextSnapshot(
        conversation_id=conversation_id,
        workspace_id=workspace_id,
        acp_session_id=acp_session_id,
        messages=messages,
        selected_message_ids=_string_list(metadata.get("selected_message_ids")),
        selected_source_ids=selected_source_ids,
        model_selection=safe_model_selection,
        output_profile=output_profile or "default",
        token_estimate=token_estimate,
        acp=_sanitize_mapping(metadata.get("acp") if isinstance(metadata.get("acp"), Mapping) else {}),
    )


def snapshot_from_mapping(raw: Mapping[str, Any] | None) -> MacroContextSnapshot:
    """Normalize a stored snapshot mapping into ``MacroContextSnapshot``."""
    raw = raw or {}
    return MacroContextSnapshot(
        conversation_id=raw.get("conversation_id"),
        workspace_id=raw.get("workspace_id"),
        acp_session_id=raw.get("acp_session_id"),
        messages=_stored_messages(raw.get("messages")),
        selected_message_ids=_string_list(raw.get("selected_message_ids")),
        selected_source_ids={
            str(key): _string_list(value)
            for key, value in dict(raw.get("selected_source_ids") or {}).items()
        },
        model_selection=_sanitize_model_selection(dict(raw.get("model_selection") or {})),
        output_profile=str(raw.get("output_profile") or "default"),
        token_estimate=max(0, int(raw.get("token_estimate") or 0)),
        acp=_sanitize_mapping(raw.get("acp") if isinstance(raw.get("acp"), Mapping) else {}),
    )


def _message_excerpt(message: Any, *, max_excerpt_chars: int) -> dict[str, Any] | None:
    if isinstance(message, Mapping):
        message_id = message.get("id") or message.get("message_id")
        role = message.get("role")
        content = message.get("content")
    else:
        message_id = getattr(message, "id", None) or getattr(message, "message_id", None)
        role = getattr(message, "role", None)
        content = getattr(message, "content", None)
    text = _content_text(content)
    if not text:
        return None
    return {
        "id": str(message_id) if message_id is not None else None,
        "role": str(role or "unknown"),
        "excerpt": redact_sensitive_text(text)[: max(1, max_excerpt_chars)],
    }


def _stored_messages(raw_messages: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_messages, Sequence) or isinstance(raw_messages, (bytes, bytearray, str)):
        return []
    messages: list[dict[str, Any]] = []
    for message in raw_messages:
        if not isinstance(message, Mapping):
            continue
        excerpt = str(message.get("excerpt") or _content_text(message.get("content")) or "")
        if not excerpt:
            continue
        messages.append(
            {
                "id": str(message.get("id")) if message.get("id") is not None else None,
                "role": str(message.get("role") or "unknown"),
                "excerpt": redact_sensitive_text(excerpt)[:DEFAULT_MAX_EXCERPT_CHARS],
            }
        )
    return messages


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, Sequence) and not isinstance(content, (bytes, bytearray, str)):
        parts: list[str] = []
        for item in content:
            if isinstance(item, Mapping):
                if item.get("type") == "text":
                    parts.append(str(item.get("text") or ""))
            else:
                text = getattr(item, "text", None)
                if text:
                    parts.append(str(text))
        return "\n".join(part for part in parts if part)
    if content is None:
        return ""
    return str(content)


def _sanitize_model_selection(raw: Mapping[str, Any]) -> dict[str, Any]:
    sanitized = _sanitize_mapping(raw)
    allowed = {}
    for key in ("api_provider", "provider", "model"):
        if key in sanitized:
            allowed[key] = sanitized[key]
    return allowed


def _sanitize_mapping(raw: Mapping[str, Any], *, depth: int = 0) -> dict[str, Any]:
    if depth >= MAX_SANITIZE_DEPTH:
        return {"_truncated": True}
    safe: dict[str, Any] = {}
    for key, value in raw.items():
        key_text = str(key)
        if _is_secret_key(key_text):
            continue
        if isinstance(value, Mapping):
            safe[key_text] = _sanitize_mapping(value, depth=depth + 1)
        elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
            safe[key_text] = _sanitize_sequence(value, depth=depth + 1)
        elif isinstance(value, str):
            safe[key_text] = redact_sensitive_text(value)
        else:
            safe[key_text] = value
    return safe


def _sanitize_sequence(raw: Sequence[Any], *, depth: int) -> list[Any]:
    if depth >= MAX_SANITIZE_DEPTH:
        return ["[truncated]"]
    safe: list[Any] = []
    for value in raw:
        if isinstance(value, Mapping):
            safe.append(_sanitize_mapping(value, depth=depth + 1))
        elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
            safe.append(_sanitize_sequence(value, depth=depth + 1))
        elif isinstance(value, str):
            safe.append(redact_sensitive_text(value))
        else:
            safe.append(value)
    return safe


def _is_secret_key(key: str) -> bool:
    normalized = key.lower().replace("-", "_")
    return any(fragment in normalized for fragment in _SECRET_KEY_FRAGMENTS)


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (bytes, bytearray, str)):
        return []
    return [redact_sensitive_text(item) for item in value if item is not None]


def redact_sensitive_text(value: Any) -> str:
    """Redact common secret-bearing value patterns in persisted macro context/errors."""
    text = str(value or "")
    text = _BEARER_RE.sub("Bearer [redacted]", text)
    text = _BASIC_AUTH_RE.sub("Basic [redacted]", text)
    text = _SECRET_ASSIGNMENT_RE.sub(lambda match: f"{match.group(1)}=[redacted]", text)
    text = _OPENAI_KEY_RE.sub("sk-[redacted]", text)
    text = _AWS_ACCESS_KEY_RE.sub("[redacted]", text)
    text = _GITHUB_TOKEN_RE.sub("[redacted]", text)
    text = _JWT_RE.sub("[redacted]", text)
    return text
