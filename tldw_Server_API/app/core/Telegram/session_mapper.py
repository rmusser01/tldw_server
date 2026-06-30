from __future__ import annotations

import json
import uuid
from typing import Any

_ASSISTANT_CONVERSATION_NAMESPACE = uuid.UUID("1f5d8d1d-8d7d-5b0e-8c7f-7e6f6d6d5d5a")
_PERSONA_SESSION_NAMESPACE = uuid.UUID("d7cf4f79-c4de-5b54-8d0c-6f989d8b2f11")
_CHARACTER_CONVERSATION_NAMESPACE = uuid.UUID("8f6d6df3-0b13-5c67-a1f4-5c62fa2d4f22")


def _coerce_nonempty_string(value: Any) -> str:
    if not isinstance(value, (str, int)) or isinstance(value, bool):
        raise ValueError("value must be a non-empty scalar string or integer")
    text = str(value).strip()
    if not text:
        raise ValueError("value must be a non-empty string")
    return text


def _coerce_nonempty_id(value: Any, *, field_name: str) -> str:
    text = _coerce_nonempty_string(value)
    return text if text else field_name


def _canonical_uuid_name(*components: str) -> str:
    """Serialize UUID name components with unambiguous boundaries."""
    return json.dumps(list(components), ensure_ascii=False, separators=(",", ":"))


def build_telegram_session_key(
    *,
    tenant_id: Any,
    chat_type: Any,
    telegram_user_id: Any,
    telegram_chat_id: Any | None = None,
    topic_or_thread_id: Any | None = None,
) -> str:
    tenant = _coerce_nonempty_string(tenant_id)
    normalized_chat_type = _coerce_nonempty_string(chat_type).lower()
    user = _coerce_nonempty_string(telegram_user_id)
    if normalized_chat_type == "private":
        return f"{tenant}:dm:{user}"
    if normalized_chat_type not in {"group", "supergroup"}:
        raise ValueError("chat_type must be private, group, or supergroup")

    chat = _coerce_nonempty_string(telegram_chat_id)
    topic = _coerce_nonempty_string(topic_or_thread_id) if topic_or_thread_id is not None else "root"
    return f"{tenant}:group:{chat}:topic:{topic}:user:{user}"


def derive_telegram_assistant_conversation_id(session_key: str) -> str:
    key = _coerce_nonempty_string(session_key)
    return str(uuid.uuid5(_ASSISTANT_CONVERSATION_NAMESPACE, key))


def derive_telegram_persona_session_id(session_key: str, *, persona_id: Any) -> str:
    key = _coerce_nonempty_string(session_key)
    persona = _coerce_nonempty_id(persona_id, field_name="persona")
    return str(
        uuid.uuid5(
            _PERSONA_SESSION_NAMESPACE,
            _canonical_uuid_name(key, "persona", persona),
        )
    )


def derive_telegram_character_conversation_id(session_key: str, *, character_id: Any) -> str:
    key = _coerce_nonempty_string(session_key)
    character = _coerce_nonempty_id(character_id, field_name="character")
    return str(
        uuid.uuid5(
            _CHARACTER_CONVERSATION_NAMESPACE,
            _canonical_uuid_name(key, "character", character),
        )
    )
