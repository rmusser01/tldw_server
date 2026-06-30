"""Context minimization and redaction helpers for persona dialogue-tree runs."""

from __future__ import annotations

from collections.abc import Mapping
import re
from dataclasses import dataclass, field
from typing import Any


_DEFAULT_SENSITIVE_KEY_FAMILIES: tuple[str, ...] = (
    "authorization",
    "auth_header",
    "api_key",
    "api_keys",
    "token",
    "access_token",
    "secret",
    "client_secret",
    "password",
    "raw",
    "credential",
)

_REDACTED_PLACEHOLDER = "[REDACTED]"

_INLINE_SENSITIVE_TEXT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?i)\bAuthorization\s*:\s*Bearer\s+[^\s,;]+"),
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{6,}"),
    re.compile(r"\bsk-[A-Za-z0-9][A-Za-z0-9_-]{2,}\b"),
    re.compile(
        r"(?i)\b(api\s+key|access\s+token|client\s+secret)\s*[:=]\s*[^\s,;]+"
    ),
    re.compile(
        r"(?i)\b(api[_-]?key|access[_-]?token|token|client[_-]?secret|"
        r"credential|password)\s*[:=]\s*[^\s,;]+"
    ),
    re.compile(r"(?i)\b(api\s+key|access\s+token|client\s+secret)\s+[^\s,;]+"),
    re.compile(
        r"(?i)\b(api[_-]?key|access[_-]?token|token|client[_-]?secret|"
        r"credential|password)\s+[^\s,;]+"
    ),
)

_TOOL_SAFE_FIELD_NAMES: frozenset[str] = frozenset(
    {"tool", "name", "id", "summary", "status", "latency_ms", "error"}
)
_TOOL_PRIVATE_PAYLOAD_FIELD_NAMES: frozenset[str] = frozenset(
    {
        "raw",
        "response",
        "raw_response",
        "body",
        "headers",
        "content",
        "data",
        "payload",
        "output",
        "result",
    }
)


@dataclass(frozen=True)
class PersonaTreeContext:
    persona_id: str
    session_id: str
    user_message: str
    policy_snapshot: dict[str, Any]
    memory_entries: list[dict[str, Any]]
    state_docs: list[dict[str, Any]]
    exemplar_sections: list[dict[str, Any]]
    tool_results: list[dict[str, Any]]
    metadata: dict[str, Any] = field(default_factory=dict)

    def for_generator(self) -> dict[str, Any]:
        return {
            "persona_id": self.persona_id,
            "session_id": self.session_id,
            "user_message": self.user_message,
            "policy_snapshot": self.policy_snapshot,
            "memory_entries": self.memory_entries,
            "state_docs": self.state_docs,
            "exemplar_sections": self.exemplar_sections,
            "tool_results": self.tool_results,
            "metadata": dict(self.metadata),
        }


def redact_sensitive_payload(payload: Any) -> Any:
    redacted_payload, _ = _redact_sensitive_payload_with_metadata(payload=payload)
    return redacted_payload


def truncate_text_fields(
    payload: Any,
    *,
    max_length: int = 400,
) -> tuple[Any, dict[str, Any]]:
    if max_length < 1:
        raise ValueError("max_length must be >= 1")

    truncated_paths: list[str] = []
    truncated = _truncate_value(
        payload=payload,
        path="",
        max_length=max_length,
        truncated_paths=truncated_paths,
    )
    metadata = {
        "truncated_field_count": len(truncated_paths),
        "truncated_paths": sorted(set(truncated_paths)),
        "max_length": max_length,
    }
    return truncated, metadata


def build_runtime_tree_context(
    *,
    persona_id: str,
    session_id: str,
    user_message: str,
    policy_snapshot: dict[str, Any] | None = None,
    memory_entries: list[dict[str, Any]] | None = None,
    state_docs: list[dict[str, Any]] | None = None,
    exemplar_sections: list[tuple[str, str, int | float]] | None = None,
    tool_results: list[dict[str, Any]] | None = None,
    max_text_length: int = 400,
) -> PersonaTreeContext:
    return _build_tree_context(
        mode="runtime",
        persona_id=persona_id,
        session_id=session_id,
        user_message=user_message,
        policy_snapshot=policy_snapshot or {},
        memory_entries=memory_entries or [],
        state_docs=state_docs or [],
        exemplar_sections=exemplar_sections or [],
        tool_results=tool_results or [],
        max_text_length=max_text_length,
    )


def build_offline_tree_context(
    *,
    persona_id: str,
    session_id: str,
    user_message: str,
    policy_snapshot: dict[str, Any] | None = None,
    memory_entries: list[dict[str, Any]] | None = None,
    state_docs: list[dict[str, Any]] | None = None,
    exemplar_sections: list[tuple[str, str, int | float]] | None = None,
    tool_results: list[dict[str, Any]] | None = None,
    max_text_length: int = 800,
) -> PersonaTreeContext:
    return _build_tree_context(
        mode="offline",
        persona_id=persona_id,
        session_id=session_id,
        user_message=user_message,
        policy_snapshot=policy_snapshot or {},
        memory_entries=memory_entries or [],
        state_docs=state_docs or [],
        exemplar_sections=exemplar_sections or [],
        tool_results=tool_results or [],
        max_text_length=max_text_length,
    )


def _build_tree_context(
    *,
    mode: str,
    persona_id: str,
    session_id: str,
    user_message: str,
    policy_snapshot: dict[str, Any],
    memory_entries: list[dict[str, Any]],
    state_docs: list[dict[str, Any]],
    exemplar_sections: list[tuple[str, str, int | float]],
    tool_results: list[dict[str, Any]],
    max_text_length: int,
) -> PersonaTreeContext:
    omitted_context_categories: set[str] = set()
    redacted_paths: list[str] = []
    truncation_paths: list[str] = []

    bounded_user_message = _sanitize_text_field(
        value=user_message,
        max_length=max_text_length,
        path="user_message",
        redacted_paths=redacted_paths,
        truncated_paths=truncation_paths,
    )

    bounded_policy_snapshot = _sanitize_general_payload(
        payload=policy_snapshot,
        path_prefix="policy_snapshot",
        max_text_length=max_text_length,
        redacted_paths=redacted_paths,
        truncation_paths=truncation_paths,
    )

    bounded_memory_entries = _sanitize_entry_list(
        entries=memory_entries,
        allowed_fields=(
            "id",
            "summary",
            "content",
            "title",
            "source",
            "role",
            "score",
            "tags",
            "created_at",
        ),
        category_name="memory_entries",
        max_text_length=max_text_length,
        omitted_context_categories=omitted_context_categories,
        redacted_paths=redacted_paths,
        truncation_paths=truncation_paths,
    )

    bounded_state_docs = _sanitize_entry_list(
        entries=state_docs,
        allowed_fields=("id", "summary", "content", "title", "source", "kind", "score", "state"),
        category_name="state_docs",
        max_text_length=max_text_length,
        omitted_context_categories=omitted_context_categories,
        redacted_paths=redacted_paths,
        truncation_paths=truncation_paths,
    )

    bounded_exemplar_sections = _sanitize_exemplar_sections(
        sections=exemplar_sections,
        max_text_length=max_text_length,
        redacted_paths=redacted_paths,
        truncation_paths=truncation_paths,
    )

    bounded_tool_results = _sanitize_tool_results(
        tool_results=tool_results,
        mode=mode,
        max_text_length=max_text_length,
        omitted_context_categories=omitted_context_categories,
        redacted_paths=redacted_paths,
        truncation_paths=truncation_paths,
    )

    metadata = {
        "context_mode": mode,
        "omitted_context_categories": sorted(omitted_context_categories),
        "redacted_field_count": len(set(redacted_paths)),
        "redacted_paths": sorted(set(redacted_paths)),
        "truncated_field_count": len(set(truncation_paths)),
        "truncated_paths": sorted(set(truncation_paths)),
        "max_text_length": max_text_length,
    }

    return PersonaTreeContext(
        persona_id=persona_id,
        session_id=session_id,
        user_message=bounded_user_message,
        policy_snapshot=bounded_policy_snapshot,
        memory_entries=bounded_memory_entries,
        state_docs=bounded_state_docs,
        exemplar_sections=bounded_exemplar_sections,
        tool_results=bounded_tool_results,
        metadata=metadata,
    )


def _sanitize_general_payload(
    *,
    payload: Any,
    path_prefix: str,
    max_text_length: int,
    redacted_paths: list[str],
    truncation_paths: list[str],
) -> Any:
    redacted, local_redacted = _redact_sensitive_payload_with_metadata(payload=payload)
    redacted_paths.extend([f"{path_prefix}.{path}" if path else path_prefix for path in local_redacted])
    truncated, truncation_meta = truncate_text_fields(redacted, max_length=max_text_length)
    truncation_paths.extend(
        [
            f"{path_prefix}.{path}" if path else path_prefix
            for path in truncation_meta["truncated_paths"]
        ]
    )
    return truncated


def _sanitize_entry_list(
    *,
    entries: list[dict[str, Any]],
    allowed_fields: tuple[str, ...],
    category_name: str,
    max_text_length: int,
    omitted_context_categories: set[str],
    redacted_paths: list[str],
    truncation_paths: list[str],
) -> list[dict[str, Any]]:
    sanitized_entries: list[dict[str, Any]] = []
    allowed_lookup = {field.casefold() for field in allowed_fields}

    for index, entry in enumerate(entries):
        safe_entry: dict[str, Any] = {}
        for key, value in entry.items():
            if key.casefold() in allowed_lookup:
                safe_entry[key] = value
            else:
                omitted_context_categories.add(f"{category_name}.non_allowlisted_fields")

        sanitized_entries.append(
            _sanitize_general_payload(
                payload=safe_entry,
                path_prefix=f"{category_name}[{index}]",
                max_text_length=max_text_length,
                redacted_paths=redacted_paths,
                truncation_paths=truncation_paths,
            )
        )

    return sanitized_entries


def _sanitize_exemplar_sections(
    *,
    sections: list[tuple[str, str, int | float]],
    max_text_length: int,
    redacted_paths: list[str],
    truncation_paths: list[str],
) -> list[dict[str, Any]]:
    sanitized_sections: list[dict[str, Any]] = []
    for index, (section_id, section_text, score) in enumerate(sections):
        bounded_text = _sanitize_text_field(
            value=section_text,
            max_length=max_text_length,
            path=f"exemplar_sections[{index}].text",
            redacted_paths=redacted_paths,
            truncated_paths=truncation_paths,
        )
        sanitized_sections.append(
            {
                "section_id": section_id,
                "text": bounded_text,
                "score": score,
            }
        )

    return sanitized_sections


def _sanitize_tool_results(
    *,
    tool_results: list[dict[str, Any]],
    mode: str,
    max_text_length: int,
    omitted_context_categories: set[str],
    redacted_paths: list[str],
    truncation_paths: list[str],
) -> list[dict[str, Any]]:
    sanitized_results: list[dict[str, Any]] = []
    for index, result in enumerate(tool_results):
        if not isinstance(result, Mapping):
            omitted_context_categories.add("tool_results.invalid_entries")
            sanitized_results.append(
                _sanitize_general_payload(
                    payload={"tool": f"tool_{index}", "raw_omitted": True},
                    path_prefix=f"tool_results[{index}]",
                    max_text_length=max_text_length,
                    redacted_paths=redacted_paths,
                    truncation_paths=truncation_paths,
                )
            )
            continue

        sanitized: dict[str, Any] = {}
        tool_name = str(result.get("tool") or result.get("name") or f"tool_{index}")
        sanitized["tool"] = tool_name

        private_payload_omitted = False
        for key in result:
            normalized_key = key.casefold() if isinstance(key, str) else str(key).casefold()
            if normalized_key in _TOOL_SAFE_FIELD_NAMES:
                continue
            omitted_context_categories.add("tool_results.non_allowlisted_fields")
            if _is_private_tool_payload_key(key):
                private_payload_omitted = True

        if private_payload_omitted:
            omitted_context_categories.add("tool_results.private_response_fields")
            sanitized["raw_omitted"] = True

        if "raw" in result:
            omitted_context_categories.add("tool_results.raw")
            sanitized["raw_omitted"] = True

        if "id" in result:
            sanitized["id"] = result["id"]

        if "summary" in result:
            bounded_summary = _sanitize_text_field(
                value=str(result["summary"]),
                max_length=max_text_length,
                path=f"tool_results[{index}].summary",
                redacted_paths=redacted_paths,
                truncated_paths=truncation_paths,
            )
            sanitized["summary"] = bounded_summary

        if mode == "offline":
            diagnostics: dict[str, Any] = {}
            for key in ("status", "latency_ms", "error", "id"):
                if key in result:
                    diagnostics[key] = result[key]
            if diagnostics:
                sanitized["diagnostics"] = _sanitize_general_payload(
                    payload=diagnostics,
                    path_prefix=f"tool_results[{index}].diagnostics",
                    max_text_length=max_text_length,
                    redacted_paths=redacted_paths,
                    truncation_paths=truncation_paths,
                )

        sanitized_results.append(
            _sanitize_general_payload(
                payload=sanitized,
                path_prefix=f"tool_results[{index}]",
                max_text_length=max_text_length,
                redacted_paths=redacted_paths,
                truncation_paths=truncation_paths,
            )
        )

    return sanitized_results


def _redact_sensitive_payload_with_metadata(payload: Any) -> tuple[Any, list[str]]:
    redacted_paths: list[str] = []
    redacted = _redact_value(payload=payload, path="", redacted_paths=redacted_paths)
    return redacted, redacted_paths


def _redact_value(*, payload: Any, path: str, redacted_paths: list[str]) -> Any:
    if isinstance(payload, dict):
        sanitized_dict: dict[Any, Any] = {}
        for key, value in payload.items():
            key_path = f"{path}.{key}" if path else str(key)
            if _is_sensitive_key(key):
                sanitized_dict[key] = _redacted_placeholder_for(value)
                redacted_paths.append(key_path)
            else:
                sanitized_dict[key] = _redact_value(
                    payload=value,
                    path=key_path,
                    redacted_paths=redacted_paths,
                )
        return sanitized_dict

    if isinstance(payload, list):
        return [
            _redact_value(payload=item, path=f"{path}[{index}]", redacted_paths=redacted_paths)
            for index, item in enumerate(payload)
        ]

    if isinstance(payload, tuple):
        return tuple(
            _redact_value(payload=item, path=f"{path}[{index}]", redacted_paths=redacted_paths)
            for index, item in enumerate(payload)
        )

    if isinstance(payload, str):
        redacted_text = _redact_sensitive_text(payload)
        if redacted_text != payload:
            redacted_paths.append(path)
        return redacted_text

    return payload


def _redacted_placeholder_for(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {"redacted": True}
    if isinstance(value, list):
        return [_REDACTED_PLACEHOLDER]
    if isinstance(value, tuple):
        return (_REDACTED_PLACEHOLDER,)
    if isinstance(value, set):
        return {_REDACTED_PLACEHOLDER}
    return _REDACTED_PLACEHOLDER


def _is_sensitive_key(key: Any) -> bool:
    if not isinstance(key, str):
        return False
    normalized = key.casefold()
    if normalized in {"raw_omitted"}:
        return False
    if normalized == "raw" or normalized.startswith("raw_") or normalized.endswith("_raw"):
        return True
    return any(
        fragment in normalized
        for fragment in _DEFAULT_SENSITIVE_KEY_FAMILIES
        if fragment != "raw"
    )


def _is_private_tool_payload_key(key: Any) -> bool:
    if not isinstance(key, str):
        return False
    normalized = key.casefold()
    if normalized == "raw" or normalized.startswith("raw_") or normalized.endswith("_raw"):
        return True
    return normalized in _TOOL_PRIVATE_PAYLOAD_FIELD_NAMES or any(
        fragment in normalized for fragment in ("response", "headers", "body")
    )


def _redact_sensitive_text(value: str) -> str:
    redacted = value
    for pattern in _INLINE_SENSITIVE_TEXT_PATTERNS:
        redacted = pattern.sub(_redact_inline_match, redacted)
    return redacted


def _redact_inline_match(match: re.Match[str]) -> str:
    matched_text = match.group(0)
    if ":" in matched_text:
        prefix = matched_text.split(":", maxsplit=1)[0]
        return f"{prefix}: {_REDACTED_PLACEHOLDER}"
    if "=" in matched_text:
        prefix = matched_text.split("=", maxsplit=1)[0]
        return f"{prefix}={_REDACTED_PLACEHOLDER}"
    if matched_text.casefold().startswith("bearer "):
        return f"Bearer {_REDACTED_PLACEHOLDER}"
    if matched_text.casefold().startswith("sk-"):
        return f"sk-{_REDACTED_PLACEHOLDER}"
    if " " in matched_text:
        prefix = matched_text.rsplit(maxsplit=1)[0]
        return f"{prefix} {_REDACTED_PLACEHOLDER}"
    return _REDACTED_PLACEHOLDER


def _truncate_value(
    *,
    payload: Any,
    path: str,
    max_length: int,
    truncated_paths: list[str],
) -> Any:
    if isinstance(payload, str):
        return _truncate_string(
            value=payload,
            max_length=max_length,
            path=path,
            truncated_paths=truncated_paths,
        )

    if isinstance(payload, dict):
        return {
            key: _truncate_value(
                payload=value,
                path=f"{path}.{key}" if path else str(key),
                max_length=max_length,
                truncated_paths=truncated_paths,
            )
            for key, value in payload.items()
        }

    if isinstance(payload, list):
        return [
            _truncate_value(
                payload=item,
                path=f"{path}[{index}]",
                max_length=max_length,
                truncated_paths=truncated_paths,
            )
            for index, item in enumerate(payload)
        ]

    if isinstance(payload, tuple):
        return tuple(
            _truncate_value(
                payload=item,
                path=f"{path}[{index}]",
                max_length=max_length,
                truncated_paths=truncated_paths,
            )
            for index, item in enumerate(payload)
        )

    return payload


def _sanitize_text_field(
    *,
    value: str,
    max_length: int,
    path: str,
    redacted_paths: list[str],
    truncated_paths: list[str],
) -> str:
    redacted_value = _redact_sensitive_text(value)
    if redacted_value != value:
        redacted_paths.append(path)
    return _truncate_string(
        value=redacted_value,
        max_length=max_length,
        path=path,
        truncated_paths=truncated_paths,
    )


def _truncate_string(
    *,
    value: str,
    max_length: int,
    path: str,
    truncated_paths: list[str],
) -> str:
    if len(value) <= max_length:
        return value

    truncated_paths.append(path)
    if max_length <= 3:
        return value[:max_length]
    return f"{value[: max_length - 3]}..."


__all__ = [
    "PersonaTreeContext",
    "build_offline_tree_context",
    "build_runtime_tree_context",
    "redact_sensitive_payload",
    "truncate_text_fields",
]
