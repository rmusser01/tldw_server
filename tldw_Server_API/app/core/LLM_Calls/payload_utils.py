from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any
from urllib.parse import quote, urlsplit

_HTTP_HEADER_NAME_RE = re.compile(r"^[!#$%&'*+\-.^_`|~0-9A-Za-z]+$")
_SERVER_MANAGED_EXTRA_HEADERS = frozenset(
    {
        "api-key",
        "authorization",
        "cf-connecting-ip",
        "connection",
        "content-length",
        "content-md5",
        "content-type",
        "cookie",
        "digest",
        "expect",
        "forwarded",
        "host",
        "keep-alive",
        "ocp-apim-subscription-key",
        "openai-organization",
        "openai-project",
        "proxy-authenticate",
        "proxy-authorization",
        "proxy-connection",
        "set-cookie",
        "te",
        "trailer",
        "transfer-encoding",
        "true-client-ip",
        "upgrade",
        "via",
        "www-authenticate",
        "x-amz-security-token",
        "x-amz-target",
        "x-api-key",
        "x-goog-api-key",
        "x-goog-user-project",
        "x-http-method",
        "x-http-method-override",
        "x-method-override",
        "x-original-url",
        "x-real-ip",
        "x-rewrite-url",
        "x-envoy-original-path",
    }
)
_SERVER_MANAGED_EXTRA_HEADER_SUFFIXES = (
    "-api-key",
    "-auth",
    "-authorization",
    "-credential",
    "-credentials",
    "-host",
    "-token",
)
_SERVER_MANAGED_EXTRA_HEADER_COMPACT_MARKERS = (
    "apikey",
    "authorization",
    "credential",
    "accesskeyid",
    "secretaccesskey",
    "sessiontoken",
    "baseurl",
    "apiurl",
    "endpoint",
)


def encode_provider_model_path(model: Any) -> str:
    """Return a URL-safe provider model path or raise a bounded validation error."""
    if not isinstance(model, str) or not model:
        raise ValueError("Invalid provider model identifier.")
    if any(char.isspace() or ord(char) < 32 or ord(char) == 127 for char in model):
        raise ValueError("Invalid provider model identifier.")
    if any(delimiter in model for delimiter in ("\\", "%", "?", "#")):
        raise ValueError("Invalid provider model identifier.")

    segments = model.split("/")
    if any(not segment or segment in {".", ".."} for segment in segments):
        raise ValueError("Invalid provider model identifier.")
    return "/".join(quote(segment, safe="-._~:@") for segment in segments)


def encode_google_model_path(model: Any) -> str:
    """Normalize Google's optional ``models/`` resource prefix and encode its model path."""
    if isinstance(model, str) and model.startswith("models/"):
        model = model[len("models/") :]
    return encode_provider_model_path(model)


def encode_huggingface_model_path(model: Any) -> str:
    """Encode a Hugging Face model path, accepting its legacy single leading slash."""
    if isinstance(model, str) and model.startswith("/") and not model.startswith("//"):
        model = model[1:]
    return encode_provider_model_path(model)


def is_server_managed_extra_header(header_name: Any) -> bool:
    """Return whether a public extension header can alter request trust or routing."""
    if not isinstance(header_name, str):
        return True
    normalized = header_name.strip().casefold()
    if not normalized or normalized != header_name.casefold():
        return True
    if _HTTP_HEADER_NAME_RE.fullmatch(header_name) is None:
        return True
    if "_" in normalized:
        return True
    if normalized in _SERVER_MANAGED_EXTRA_HEADERS:
        return True
    if normalized.startswith("x-amz-"):
        return True
    if normalized.startswith("x-forwarded-"):
        return True
    compact = re.sub(r"[^a-z0-9]", "", normalized)
    if any(marker in compact for marker in _SERVER_MANAGED_EXTRA_HEADER_COMPACT_MARKERS):
        return True
    return normalized.endswith(_SERVER_MANAGED_EXTRA_HEADER_SUFFIXES)


def is_safe_extra_header_value(value: Any) -> bool:
    """Return whether a public extension header value contains no controls."""
    try:
        rendered = "" if value is None else str(value)
    except (TypeError, ValueError):
        return False
    return not any(ord(char) < 32 or ord(char) == 127 for char in rendered)


def resolve_runtime_embedding_base_url(
    request: Mapping[str, Any],
    *,
    provider: str,
) -> str | None:
    """Return the server-proven request endpoint or reject an incomplete contract."""
    from tldw_Server_API.app.core.AuthNZ.byok_config import (
        is_runtime_base_url_override,
    )
    from tldw_Server_API.app.core.Chat.Chat_Deps import ChatConfigurationError

    base_url = request.get("base_url")
    provenance = request.get("_runtime_base_url_override")
    credentials_resolved = request.get("credentials_resolved") is True

    if not credentials_resolved:
        if base_url is None and provenance is None:
            return None
        raise ChatConfigurationError(
            provider=provider,
            message="Invalid runtime embedding endpoint configuration.",
        )
    if not is_runtime_base_url_override(provenance) or not isinstance(base_url, str):
        raise ChatConfigurationError(
            provider=provider,
            message="Invalid runtime embedding endpoint configuration.",
        )

    cleaned = base_url.strip()
    try:
        parsed = urlsplit(cleaned)
        invalid = (
            not cleaned
            or cleaned != base_url
            or any(char.isspace() or ord(char) < 32 or ord(char) == 127 for char in cleaned)
            or "\\" in cleaned
            or parsed.scheme.casefold() not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or bool(parsed.query)
            or bool(parsed.fragment)
        )
        _ = parsed.port
    except (TypeError, ValueError):
        invalid = True
    if invalid:
        raise ChatConfigurationError(
            provider=provider,
            message="Invalid runtime embedding endpoint configuration.",
        )
    return cleaned.rstrip("/")


EMBEDDING_REDIRECT_STATUS_CODES = frozenset({301, 302, 303, 307, 308})


def _summarize_message_content(content: Any) -> tuple[int, bool]:
    """Return (text_char_count, has_attachments) for a message content payload."""
    text_chars = 0
    has_attachments = False

    if content is None:
        return text_chars, has_attachments

    if isinstance(content, str):
        return len(content), has_attachments

    if isinstance(content, dict):
        # Handle single-part dicts (e.g., Gemini parts or Cohere history entries)
        possible_text = content.get("text") or content.get("message")
        if isinstance(possible_text, str):
            text_chars += len(possible_text)
        if any(key in content for key in ("image_url", "inline_data", "data", "file_id")):
            has_attachments = True
        if "parts" in content:
            extra_chars, extra_attach = _summarize_message_content(content.get("parts"))
            text_chars += extra_chars
            has_attachments = has_attachments or extra_attach
        return text_chars, has_attachments

    if isinstance(content, (list, tuple)):
        for part in content:
            if isinstance(part, dict):
                part_type = (part.get("type") or "").lower()
                if part_type in {"text", "input_text"} and isinstance(part.get("text"), str):
                    text_chars += len(part.get("text") or "")
                elif part_type in {"image_url", "input_image", "image"}:
                    has_attachments = True
                elif part_type in {"tool_use"}:
                    continue
                if "inline_data" in part or "image_url" in part:
                    has_attachments = True
                if "functionCall" in part and isinstance(part.get("functionCall", {}).get("args"), str):
                    text_chars += len(part["functionCall"]["args"])
            elif isinstance(part, str):
                text_chars += len(part)
    return text_chars, has_attachments


def _summarize_messages(messages: Any, key: str) -> dict[str, Any]:
    """Summarize a messages-like payload without logging raw content."""
    if messages is None:
        return {f"{key}_count": 0, f"{key}_text_chars": 0}

    messages_iterable = [messages] if not isinstance(messages, list) else messages

    role_counts: dict[str, int] = {}
    total_text_chars = 0
    has_attachments = False

    for entry in messages_iterable:
        if isinstance(entry, dict):
            role = entry.get("role")
            if isinstance(role, str):
                role_counts[role] = role_counts.get(role, 0) + 1
            entry_content = None
            if "content" in entry:
                entry_content = entry.get("content")
            elif "parts" in entry:
                entry_content = entry.get("parts")
            elif "message" in entry:
                entry_content = entry.get("message")
            elif "text" in entry:
                entry_content = entry.get("text")
            text_chars, attachments = _summarize_message_content(entry_content)
            total_text_chars += text_chars
            has_attachments = has_attachments or attachments
        elif isinstance(entry, str):
            total_text_chars += len(entry)

    summary: dict[str, Any] = {
        f"{key}_count": len(messages_iterable),
        f"{key}_text_chars": total_text_chars,
    }
    if role_counts:
        summary[f"{key}_roles"] = role_counts
    if has_attachments:
        summary[f"{key}_has_attachments"] = True
    return summary


def _summarize_dict_field(key: str, value: dict[str, Any]) -> dict[str, Any]:
    """Summarize dict values without exposing raw content."""
    if key == "response_format":
        summary: dict[str, Any] = {f"{key}_keys_count": len(value)}
        response_type = value.get("type")
        if isinstance(response_type, str):
            summary["response_format_type"] = response_type
        return summary

    if key == "generationConfig":
        summary = {f"{key}_keys_count": len(value)}
        for numeric_key in ("temperature", "topP", "topK", "maxOutputTokens", "candidateCount"):
            numeric_val = value.get(numeric_key)
            if isinstance(numeric_val, (int, float)):
                summary[f"{key}_{numeric_key}"] = numeric_val
        if isinstance(value.get("responseMimeType"), str):
            summary["response_mime_type"] = value["responseMimeType"]
        if isinstance(value.get("stopSequences"), (list, tuple)):
            summary[f"{key}_stop_sequences_count"] = len(value["stopSequences"])
        return summary

    if key == "logit_bias":
        return {f"{key}_size": len(value)}

    if key == "system_instruction":
        parts = value.get("parts")
        text_chars, attachments = _summarize_message_content(parts)
        summary = {
            f"{key}_parts_count": len(parts or []),
            f"{key}_text_chars": text_chars,
        }
        if attachments:
            summary[f"{key}_has_attachments"] = True
        return summary

    return {f"{key}_keys_count": len(value)}


def _summarize_list_field(key: str, value: Iterable[Any]) -> dict[str, Any]:
    """Summarize list/tuple values."""
    items = list(value)
    summary: dict[str, Any] = {f"{key}_count": len(items)}
    if key in {"stop", "stop_sequences", "stopSequences"}:
        summary[f"{key}_total_chars"] = sum(len(item) for item in items if isinstance(item, str))
    return summary


def _sanitize_payload_for_logging(
    payload: dict[str, Any] | None,
    *,
    message_keys: tuple[str, ...] = ("messages",),
    text_keys: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Build a metadata dict safe for logging, omitting raw prompts or filenames."""
    if not isinstance(payload, dict):
        return {}

    metadata: dict[str, Any] = {}

    model = payload.get("model")
    if isinstance(model, str):
        metadata["model"] = model

    if "stream" in payload:
        metadata["stream"] = bool(payload.get("stream"))

    for key in message_keys:
        if key in payload:
            metadata.update(_summarize_messages(payload.get(key), key))

    for key, value in payload.items():
        if key in message_keys or key in {"model", "stream"}:
            continue
        if value is None:
            continue
        if isinstance(value, (int, float, bool)):
            metadata[key] = value
        elif isinstance(value, str):
            if key in text_keys or key in {"stop"}:
                metadata[f"{key}_chars"] = len(value)
            elif key in {"tool_choice"}:
                metadata[key] = value
            else:
                metadata[f"{key}_present"] = True
        elif isinstance(value, dict):
            metadata.update(_summarize_dict_field(key, value))
        elif isinstance(value, (list, tuple, set)):
            metadata.update(_summarize_list_field(key, value))
        else:
            metadata[f"{key}_present"] = True

    return metadata


def merge_extra_body(payload: dict[str, Any], request: Mapping[str, Any]) -> dict[str, Any]:
    """Merge extra_body into payload without overriding existing payload keys."""
    extra = request.get("extra_body")
    if not isinstance(extra, Mapping) or not extra:
        return payload
    merged = dict(extra)
    merged.update(payload)
    return merged


def merge_extra_headers(headers: dict[str, str], request: Mapping[str, Any]) -> dict[str, str]:
    """Merge safe extension headers without overriding server-managed headers."""
    extra = request.get("extra_headers")
    if not isinstance(extra, Mapping) or not extra:
        return headers
    merged = dict(headers or {})
    existing_lower = {str(k).lower() for k in merged}
    for key, value in extra.items():
        if not isinstance(key, str):
            continue
        normalized = key.casefold()
        if is_server_managed_extra_header(key) or normalized in existing_lower:
            continue
        if not is_safe_extra_header_value(value):
            raise ValueError("Invalid provider extension header value.")
        merged[key] = str(value) if value is not None else ""
        existing_lower.add(normalized)
    return merged


__all__ = [
    "EMBEDDING_REDIRECT_STATUS_CODES",
    "_sanitize_payload_for_logging",
    "encode_google_model_path",
    "encode_huggingface_model_path",
    "encode_provider_model_path",
    "is_safe_extra_header_value",
    "is_server_managed_extra_header",
    "merge_extra_body",
    "merge_extra_headers",
    "resolve_runtime_embedding_base_url",
]
