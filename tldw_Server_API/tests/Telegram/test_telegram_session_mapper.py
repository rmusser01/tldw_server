from __future__ import annotations

import uuid

import pytest

from tldw_Server_API.app.core.Telegram.session_mapper import (
    build_telegram_session_key,
    derive_telegram_assistant_conversation_id,
    derive_telegram_character_conversation_id,
    derive_telegram_persona_session_id,
)


def test_build_telegram_session_key_for_dm():
    key = build_telegram_session_key(
        tenant_id="tenant-a",
        chat_type="private",
        telegram_chat_id=999,
        telegram_user_id=200,
    )

    assert key == "tenant-a:dm:200"


def test_build_telegram_session_key_for_group_topic():
    key = build_telegram_session_key(
        tenant_id="tenant-a",
        chat_type="supergroup",
        telegram_chat_id=100,
        topic_or_thread_id=300,
        telegram_user_id=200,
    )

    assert key == "tenant-a:group:100:topic:300:user:200"


def test_build_telegram_session_key_for_group_root_without_topic():
    key = build_telegram_session_key(
        tenant_id="tenant-a",
        chat_type="group",
        telegram_chat_id=100,
        telegram_user_id=200,
    )

    assert key == "tenant-a:group:100:topic:root:user:200"


def test_build_telegram_session_key_rejects_missing_group_chat_id():
    with pytest.raises(ValueError):
        build_telegram_session_key(
            tenant_id="tenant-a",
            chat_type="group",
            telegram_chat_id=None,
            telegram_user_id=200,
        )


@pytest.mark.unit
def test_build_telegram_session_key_rejects_non_scalar_components() -> None:
    with pytest.raises(ValueError):
        build_telegram_session_key(
            tenant_id={"scope": "tenant-a"},
            chat_type="private",
            telegram_user_id=200,
        )


def test_assistant_conversation_id_is_stable_for_same_session_key():
    key = build_telegram_session_key(
        tenant_id="tenant-a",
        chat_type="private",
        telegram_chat_id=999,
        telegram_user_id=200,
    )

    first = derive_telegram_assistant_conversation_id(key)
    second = derive_telegram_assistant_conversation_id(key)

    assert first == second
    assert uuid.UUID(first)


def test_persona_session_ids_differ_for_different_persona_ids():
    key = build_telegram_session_key(
        tenant_id="tenant-a",
        chat_type="private",
        telegram_chat_id=999,
        telegram_user_id=200,
    )

    first = derive_telegram_persona_session_id(key, persona_id="persona-a")
    second = derive_telegram_persona_session_id(key, persona_id="persona-b")

    assert first != second
    assert uuid.UUID(first)
    assert uuid.UUID(second)


@pytest.mark.unit
def test_persona_session_id_uses_canonical_component_boundaries() -> None:
    first = derive_telegram_persona_session_id("tenant:dm:user:persona:alpha", persona_id="beta")
    second = derive_telegram_persona_session_id("tenant:dm:user", persona_id="alpha:persona:beta")

    assert first != second


@pytest.mark.unit
def test_persona_and_character_ids_reject_non_scalar_identifiers() -> None:
    key = build_telegram_session_key(
        tenant_id="tenant-a",
        chat_type="private",
        telegram_chat_id=999,
        telegram_user_id=200,
    )

    with pytest.raises(ValueError):
        derive_telegram_persona_session_id(key, persona_id={"id": "persona-a"})

    with pytest.raises(ValueError):
        derive_telegram_character_conversation_id(key, character_id=["character-a"])


def test_character_conversation_ids_differ_for_different_character_ids():
    key = build_telegram_session_key(
        tenant_id="tenant-a",
        chat_type="supergroup",
        telegram_chat_id=100,
        topic_or_thread_id=300,
        telegram_user_id=200,
    )

    first = derive_telegram_character_conversation_id(key, character_id="character-a")
    second = derive_telegram_character_conversation_id(key, character_id="character-b")

    assert first != second
    assert uuid.UUID(first)
    assert uuid.UUID(second)


def test_conversation_backed_ids_are_uuid_safe():
    key = build_telegram_session_key(
        tenant_id="tenant-a",
        chat_type="private",
        telegram_chat_id=999,
        telegram_user_id=200,
    )

    assistant_conversation_id = derive_telegram_assistant_conversation_id(key)
    persona_session_id = derive_telegram_persona_session_id(key, persona_id="persona-a")
    character_conversation_id = derive_telegram_character_conversation_id(key, character_id="character-a")

    assert uuid.UUID(assistant_conversation_id)
    assert uuid.UUID(persona_session_id)
    assert uuid.UUID(character_conversation_id)
