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
def test_chat_session_create_rejects_persona_memory_mode_for_character_chat():
    with pytest.raises(ValidationError, match="persona_memory_mode is only valid for persona chats"):
        ChatSessionCreate(character_id=7, persona_memory_mode="read_only")


@pytest.mark.unit
@pytest.mark.parametrize("invalid_mode", ["session", "", 123])
def test_chat_session_create_rejects_invalid_persona_memory_mode(invalid_mode):
    with pytest.raises(ValidationError, match="persona_memory_mode"):
        ChatSessionCreate(
            assistant_kind="persona",
            assistant_id="garden-helper",
            persona_memory_mode=invalid_mode,
        )
