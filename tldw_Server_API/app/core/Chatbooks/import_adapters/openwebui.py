"""OpenWebUI chat export import adapter."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Callable


_ATTACHMENT_KEYS = (
    "files",
    "attachments",
    "images",
    "artifacts",
    "file_ids",
    "fileIds",
)


@dataclass(frozen=True)
class OpenWebUIMessagePlan:
    """Normalized OpenWebUI message node ready for preview/import planning."""

    source_id: str
    role: str | None
    content: str
    timestamp: Any = None
    parent_source_id: str | None = None
    children_source_ids: list[str] = field(default_factory=list)
    model: str | None = None
    attachment_refs: list[Any] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OpenWebUIConversationPlan:
    """Normalized OpenWebUI conversation plan."""

    external_ref: str
    title: str
    messages: list[OpenWebUIMessagePlan]
    history_current_id: str | None = None
    is_branched: bool = False
    attachment_reference_count: int = 0
    source_metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class OpenWebUIParsedExport:
    """Parsed OpenWebUI export with valid chats and aggregate warnings."""

    chats: list[OpenWebUIConversationPlan]
    malformed_chat_count: int = 0
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class OpenWebUIPreviewChatItem:
    """Lightweight per-chat preview row."""

    external_ref: str
    title: str
    message_count: int
    branched: bool
    duplicate: bool
    warning_count: int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable preview item."""
        return {
            "external_ref": self.external_ref,
            "title": self.title,
            "message_count": self.message_count,
            "branched": self.branched,
            "duplicate": self.duplicate,
            "warning_count": self.warning_count,
        }


@dataclass(frozen=True)
class OpenWebUIImportPreview:
    """Aggregate OpenWebUI import preview."""

    chat_count: int
    message_count: int
    branched_chat_count: int
    duplicate_chat_count: int
    attachment_reference_count: int
    malformed_chat_count: int
    warnings: list[str]
    items: list[OpenWebUIPreviewChatItem]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable preview payload."""
        return {
            "chat_count": self.chat_count,
            "message_count": self.message_count,
            "branched_chat_count": self.branched_chat_count,
            "duplicate_chat_count": self.duplicate_chat_count,
            "attachment_reference_count": self.attachment_reference_count,
            "malformed_chat_count": self.malformed_chat_count,
            "warnings": list(self.warnings),
            "items": [item.to_dict() for item in self.items],
        }


def _canonical_sha(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _coerce_optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalize_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if item is not None and str(item).strip()]
    if isinstance(value, tuple):
        return [str(item) for item in value if item is not None and str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def _collect_attachment_refs(message: dict[str, Any]) -> list[Any]:
    refs: list[Any] = []
    for key in _ATTACHMENT_KEYS:
        value = message.get(key)
        if not value:
            continue
        if isinstance(value, list):
            refs.extend(value)
        elif isinstance(value, dict):
            refs.append(value)
        else:
            refs.append(value)
    return refs


def _build_title(chat_payload: dict[str, Any], messages: list[OpenWebUIMessagePlan]) -> str:
    title = _coerce_optional_text(chat_payload.get("title") or chat_payload.get("name"))
    if title:
        return title
    for message in messages:
        if message.role == "user" and message.content.strip():
            excerpt = " ".join(message.content.split())
            return excerpt[:80] or f"OpenWebUI Import {date.today().isoformat()}"
    return f"OpenWebUI Import {date.today().isoformat()}"


def _extract_chat_payload(item: Any) -> tuple[dict[str, Any], dict[str, Any]] | None:
    if not isinstance(item, dict):
        return None
    chat = item.get("chat")
    if isinstance(chat, dict):
        return chat, item
    return item, {}


def _derive_external_ref(
    index: int,
    item: Any,
    wrapper: dict[str, Any],
    chat_payload: dict[str, Any],
) -> str:
    for candidate in (
        wrapper.get("id"),
        wrapper.get("chat_id"),
        chat_payload.get("id"),
        chat_payload.get("chat_id"),
    ):
        text = _coerce_optional_text(candidate)
        if text:
            return text
    return f"openwebui:{index}:{_canonical_sha(item)[:16]}"


def _parse_message(source_key: str, value: Any) -> OpenWebUIMessagePlan | None:
    if not isinstance(value, dict):
        return None
    source_id = _coerce_optional_text(value.get("id")) or str(source_key)
    metadata = {
        "model": value.get("model"),
        "done": value.get("done"),
        "context": value.get("context"),
        "info": value.get("info"),
    }
    metadata = {key: item for key, item in metadata.items() if item is not None}
    return OpenWebUIMessagePlan(
        source_id=source_id,
        role=_coerce_optional_text(value.get("role")),
        content=_coerce_text(value.get("content")),
        timestamp=value.get("timestamp"),
        parent_source_id=_coerce_optional_text(value.get("parentId") or value.get("parent_id")),
        children_source_ids=_normalize_string_list(value.get("childrenIds") or value.get("children_ids")),
        model=_coerce_optional_text(value.get("model")),
        attachment_refs=_collect_attachment_refs(value),
        metadata=metadata,
    )


def _is_branched(messages: list[OpenWebUIMessagePlan]) -> bool:
    if any(len(message.children_source_ids) > 1 for message in messages):
        return True
    source_ids = {message.source_id for message in messages}
    root_count = sum(1 for message in messages if not message.parent_source_id or message.parent_source_id not in source_ids)
    return root_count > 1


def load_openwebui_export(file_path: str | Path) -> OpenWebUIParsedExport:
    """Load and normalize an OpenWebUI chat export JSON file."""
    path = Path(file_path)
    try:
        with path.open("r", encoding="utf-8") as handle:
            raw_data = json.load(handle)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError("Malformed OpenWebUI JSON export") from exc
    except OSError as exc:
        raise ValueError("Unable to read OpenWebUI JSON export") from exc

    if not isinstance(raw_data, list):
        raise ValueError("OpenWebUI export top-level JSON value must be an array")

    chats: list[OpenWebUIConversationPlan] = []
    warnings: list[str] = []
    malformed_chat_count = 0

    for index, item in enumerate(raw_data):
        extracted = _extract_chat_payload(item)
        if extracted is None:
            malformed_chat_count += 1
            warnings.append(f"Chat at index {index} is not an object and was skipped.")
            continue
        chat_payload, wrapper = extracted
        history = chat_payload.get("history")
        messages_map = history.get("messages") if isinstance(history, dict) else None
        if not isinstance(messages_map, dict):
            malformed_chat_count += 1
            warnings.append(f"Chat at index {index} does not contain history.messages and was skipped.")
            continue

        chat_warnings: list[str] = []
        messages: list[OpenWebUIMessagePlan] = []
        for source_key, value in messages_map.items():
            message = _parse_message(str(source_key), value)
            if message is None:
                chat_warnings.append(f"Message {source_key} is malformed and was skipped.")
                continue
            messages.append(message)

        current_id = _coerce_optional_text(history.get("currentId") or history.get("current_id"))
        attachment_count = sum(len(message.attachment_refs) for message in messages)
        chats.append(
            OpenWebUIConversationPlan(
                external_ref=_derive_external_ref(index, item, wrapper, chat_payload),
                title=_build_title(chat_payload, messages),
                messages=messages,
                history_current_id=current_id,
                is_branched=_is_branched(messages),
                attachment_reference_count=attachment_count,
                source_metadata={
                    "models": chat_payload.get("models"),
                    "options": chat_payload.get("options"),
                    "meta": chat_payload.get("meta"),
                    "pinned": chat_payload.get("pinned"),
                    "folder_id": chat_payload.get("folder_id") or wrapper.get("folder_id"),
                    "created_at": chat_payload.get("created_at") or wrapper.get("created_at"),
                    "updated_at": chat_payload.get("updated_at") or wrapper.get("updated_at"),
                },
                warnings=chat_warnings,
            )
        )
        warnings.extend(chat_warnings)

    return OpenWebUIParsedExport(
        chats=chats,
        malformed_chat_count=malformed_chat_count,
        warnings=warnings,
    )


def preview_openwebui_export(
    file_path: str | Path,
    duplicate_lookup: Callable[[str], bool] | None = None,
) -> OpenWebUIImportPreview:
    """Build a preview for an OpenWebUI chat export JSON file."""
    parsed = load_openwebui_export(file_path)
    duplicate_lookup = duplicate_lookup or (lambda _external_ref: False)

    items: list[OpenWebUIPreviewChatItem] = []
    duplicate_chat_count = 0
    for chat in parsed.chats:
        duplicate = bool(duplicate_lookup(chat.external_ref))
        if duplicate:
            duplicate_chat_count += 1
        items.append(
            OpenWebUIPreviewChatItem(
                external_ref=chat.external_ref,
                title=chat.title,
                message_count=len(chat.messages),
                branched=chat.is_branched,
                duplicate=duplicate,
                warning_count=len(chat.warnings),
            )
        )

    return OpenWebUIImportPreview(
        chat_count=len(parsed.chats),
        message_count=sum(len(chat.messages) for chat in parsed.chats),
        branched_chat_count=sum(1 for chat in parsed.chats if chat.is_branched),
        duplicate_chat_count=duplicate_chat_count,
        attachment_reference_count=sum(chat.attachment_reference_count for chat in parsed.chats),
        malformed_chat_count=parsed.malformed_chat_count,
        warnings=list(parsed.warnings),
        items=items,
    )
