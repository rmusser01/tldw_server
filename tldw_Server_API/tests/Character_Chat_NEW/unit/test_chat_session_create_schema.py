import pytest
from pydantic import ValidationError

from tldw_Server_API.app.api.v1.schemas.chat_session_schemas import ChatSessionCreate


@pytest.mark.unit
def test_chat_session_create_allows_plain_chat_without_tracked_identity():
    chat = ChatSessionCreate(title="Plain WebUI chat", source="webui-chat")

    assert chat.character_id is None
    assert chat.assistant_kind is None
    assert chat.assistant_id is None
    assert chat.persona_memory_mode is None


@pytest.mark.unit
def test_chat_session_create_normalizes_tracked_character_identity():
    chat = ChatSessionCreate(character_id=7, title="Tracked character chat")

    assert chat.character_id == 7
    assert chat.assistant_kind == "character"
    assert chat.assistant_id == "7"


@pytest.mark.unit
def test_chat_session_create_requires_assistant_id_for_tracked_persona_chat():
    with pytest.raises(ValidationError, match="Persona chats require assistant_id"):
        ChatSessionCreate(assistant_kind="persona", title="Tracked persona chat")


@pytest.mark.unit
@pytest.mark.parametrize("memory_mode", [None, "read_only", "read_write"])
def test_chat_session_create_preserves_explicit_persona_memory_mode(memory_mode):
    chat = ChatSessionCreate(
        assistant_kind="persona",
        assistant_id="garden-helper",
        persona_memory_mode=memory_mode,
        title="Tracked persona chat",
    )

    assert chat.character_id is None
    assert chat.assistant_kind == "persona"
    assert chat.assistant_id == "garden-helper"
    assert chat.persona_memory_mode == memory_mode


@pytest.mark.unit
@pytest.mark.parametrize("memory_mode", ["read_only", "read_write"])
def test_chat_session_create_rejects_persona_memory_mode_for_character_chat(memory_mode):
    with pytest.raises(ValidationError, match="persona_memory_mode is only valid for persona chats"):
        ChatSessionCreate(character_id=7, persona_memory_mode=memory_mode)


@pytest.mark.unit
@pytest.mark.parametrize("invalid_mode", ["session", "", 123])
def test_chat_session_create_rejects_invalid_persona_memory_mode(invalid_mode):
    with pytest.raises(ValidationError, match="persona_memory_mode"):
        ChatSessionCreate(
            assistant_kind="persona",
            assistant_id="garden-helper",
            persona_memory_mode=invalid_mode,
        )


# --- D7 P3: optional client chat id ---------------------------------------------------------

_CLIENT_ID = "6b0f8c1e-2d4a-4f3b-9a71-5c2e8d9f0a14"


@pytest.mark.unit
def test_chat_session_create_id_is_optional_and_canonicalized():
    assert ChatSessionCreate().id is None
    assert ChatSessionCreate(id=_CLIENT_ID.upper()).id == _CLIENT_ID


@pytest.mark.unit
@pytest.mark.parametrize(
    "bad_id",
    [
        "",
        "not-a-uuid",
        _CLIENT_ID.replace("-", ""),
        f"{{{_CLIENT_ID}}}",
        f"urn:uuid:{_CLIENT_ID}",
        "00000000-0000-0000-0000-000000000000",
        7,
    ],
)
def test_chat_session_create_rejects_non_canonical_ids(bad_id):
    with pytest.raises(ValidationError):
        ChatSessionCreate(id=bad_id)


@pytest.mark.unit
def test_create_fingerprint_ignores_key_order_and_id_but_not_explicit_nulls():
    from tldw_Server_API.app.core.Chat.conversation_create_idempotency import conversation_create_fingerprint

    def fingerprint(payload, **options):
        body = ChatSessionCreate.model_validate(payload).model_dump(mode="json", exclude_unset=True, exclude={"id"})
        return conversation_create_fingerprint(body, {"seed_first_message": False, **options})

    base = fingerprint({"id": _CLIENT_ID, "title": "A", "state": "resolved"})
    assert len(base) == 64
    assert fingerprint({"state": "RESOLVED", "title": "A"}) == base
    assert fingerprint({"title": "A", "state": "resolved", "topic_label": None}) != base
    assert fingerprint({"title": "A", "state": "resolved"}, seed_first_message=True) != base
