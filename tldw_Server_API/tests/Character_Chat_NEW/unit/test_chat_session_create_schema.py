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
