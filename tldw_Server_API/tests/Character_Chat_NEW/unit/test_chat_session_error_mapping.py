"""Direct route tests for chat session DB error mapping."""

from __future__ import annotations

import pytest
from fastapi import HTTPException, status

from tldw_Server_API.app.api.v1.endpoints import character_chat_sessions
from tldw_Server_API.app.api.v1.schemas.chat_session_schemas import (
    CharacterChatCompletionV2Request,
    CharacterChatStreamPersistRequest,
    ChatSessionUpdate,
)
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDBError,
    ConflictError,
)
from tldw_Server_API.app.core.Chat.Chat_Deps import ChatAPIError


pytestmark = pytest.mark.unit


def _test_user() -> User:
    return User(id=1, username="tester", email="tester@example.com", is_active=True)


def _conversation(*, deleted: bool = False) -> dict:
    return {
        "id": "chat-1",
        "client_id": "1",
        "character_id": 7,
        "title": "Test Chat",
        "version": 1,
        "deleted": deleted,
        "scope_type": "global",
        "workspace_id": None,
        "created_at": "2026-01-01T00:00:00Z",
        "last_modified": "2026-01-01T00:00:00Z",
    }


class _BrokenChatSessionDb:
    def __init__(self, exc: Exception, *, deleted: bool = False, raise_on_get: bool = False) -> None:
        self.exc = exc
        self.deleted = deleted
        self.raise_on_get = raise_on_get

    def get_conversation_by_id(self, chat_id: str, include_deleted: bool = False) -> dict:
        if self.raise_on_get:
            raise self.exc
        return _conversation(deleted=self.deleted)

    def update_conversation(self, *args, **kwargs) -> None:
        raise self.exc

    def get_messages_for_conversation(self, *args, **kwargs) -> list[dict]:
        return []

    def soft_delete_conversation(self, *args, **kwargs) -> None:
        raise self.exc

    def restore_conversation(self, *args, **kwargs) -> None:
        raise self.exc

    def count_messages_for_conversation(self, *args, **kwargs) -> int:
        return 0


class _CompletionReadyChatSessionDb:
    def get_conversation_by_id(self, chat_id: str, include_deleted: bool = False) -> dict:
        return _conversation()

    def get_conversation_settings(self, *args, **kwargs) -> dict:
        return {}

    def get_messages_for_conversation(self, *args, **kwargs) -> list[dict]:
        return [{"sender": "user", "content": "hello", "deleted": False}]

    def get_character_card_by_id(self, character_id: int) -> dict:
        return {"id": character_id, "name": "Assistant", "content": ""}

    def count_messages_for_conversation(self, *args, **kwargs) -> int:
        return 1

    def list_persona_memory_entries(self, *args, **kwargs) -> list[dict]:
        return []


class _NoopCharacterRateLimiter:
    async def check_soft_message_limit(self, *args, **kwargs) -> None:
        return None

    async def check_message_limit(self, *args, **kwargs) -> None:
        return None

    async def check_chat_completion_rate(self, *args, **kwargs) -> None:
        return None


class _ByokResolution:
    api_key = None
    app_config: dict = {}

    async def touch_last_used(self) -> None:
        return None


@pytest.mark.asyncio
async def test_update_chat_session_maps_db_error_to_sanitized_500():
    db = _BrokenChatSessionDb(CharactersRAGDBError("sqlite update exploded"))

    with pytest.raises(HTTPException) as exc_info:
        await character_chat_sessions.update_chat_session(
            chat_id="chat-1",
            update_data=ChatSessionUpdate(title="Updated"),
            expected_version=1,
            scope_type=None,
            workspace_id=None,
            db=db,
            current_user=_test_user(),
        )

    assert exc_info.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    assert exc_info.value.detail == "Failed to update chat session"


@pytest.mark.asyncio
async def test_update_chat_session_maps_conflict_error_to_409():
    db = _BrokenChatSessionDb(ConflictError("chat update conflict"))

    with pytest.raises(HTTPException) as exc_info:
        await character_chat_sessions.update_chat_session(
            chat_id="chat-1",
            update_data=ChatSessionUpdate(title="Updated"),
            expected_version=1,
            scope_type=None,
            workspace_id=None,
            db=db,
            current_user=_test_user(),
        )

    assert exc_info.value.status_code == status.HTTP_409_CONFLICT
    assert exc_info.value.detail == "chat update conflict"


@pytest.mark.asyncio
async def test_delete_chat_session_maps_db_error_to_sanitized_500():
    db = _BrokenChatSessionDb(CharactersRAGDBError("sqlite delete exploded"))

    with pytest.raises(HTTPException) as exc_info:
        await character_chat_sessions.delete_chat_session(
            chat_id="chat-1",
            expected_version=1,
            hard_delete=False,
            scope_type=None,
            workspace_id=None,
            db=db,
            current_user=_test_user(),
        )

    assert exc_info.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    assert exc_info.value.detail == "Failed to delete chat session"


@pytest.mark.asyncio
async def test_delete_chat_session_maps_conflict_error_to_409():
    db = _BrokenChatSessionDb(ConflictError("chat delete conflict"))

    with pytest.raises(HTTPException) as exc_info:
        await character_chat_sessions.delete_chat_session(
            chat_id="chat-1",
            expected_version=1,
            hard_delete=False,
            scope_type=None,
            workspace_id=None,
            db=db,
            current_user=_test_user(),
        )

    assert exc_info.value.status_code == status.HTTP_409_CONFLICT
    assert exc_info.value.detail == "chat delete conflict"


@pytest.mark.asyncio
async def test_restore_chat_session_maps_db_error_to_sanitized_500():
    db = _BrokenChatSessionDb(CharactersRAGDBError("sqlite restore exploded"), deleted=True)

    with pytest.raises(HTTPException) as exc_info:
        await character_chat_sessions.restore_chat_session(
            chat_id="chat-1",
            expected_version=1,
            scope_type=None,
            workspace_id=None,
            db=db,
            current_user=_test_user(),
        )

    assert exc_info.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    assert exc_info.value.detail == "Failed to restore chat session"


@pytest.mark.asyncio
async def test_restore_chat_session_maps_conflict_error_to_409():
    db = _BrokenChatSessionDb(ConflictError("chat restore conflict"), deleted=True)

    with pytest.raises(HTTPException) as exc_info:
        await character_chat_sessions.restore_chat_session(
            chat_id="chat-1",
            expected_version=1,
            scope_type=None,
            workspace_id=None,
            db=db,
            current_user=_test_user(),
        )

    assert exc_info.value.status_code == status.HTTP_409_CONFLICT
    assert exc_info.value.detail == "chat restore conflict"


@pytest.mark.asyncio
async def test_complete_v2_maps_db_error_to_sanitized_500():
    db = _BrokenChatSessionDb(CharactersRAGDBError("sqlite completion exploded"), raise_on_get=True)

    with pytest.raises(HTTPException) as exc_info:
        await character_chat_sessions.character_chat_completion(
            chat_id="chat-1",
            body=CharacterChatCompletionV2Request(),
            db=db,
            current_user=_test_user(),
        )

    assert exc_info.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    assert exc_info.value.detail == "Failed to complete character chat"


@pytest.mark.asyncio
async def test_complete_v2_maps_conflict_error_to_409():
    db = _BrokenChatSessionDb(ConflictError("chat completion conflict"), raise_on_get=True)

    with pytest.raises(HTTPException) as exc_info:
        await character_chat_sessions.character_chat_completion(
            chat_id="chat-1",
            body=CharacterChatCompletionV2Request(),
            db=db,
            current_user=_test_user(),
        )

    assert exc_info.value.status_code == status.HTTP_409_CONFLICT
    assert exc_info.value.detail == "chat completion conflict"


@pytest.mark.asyncio
async def test_complete_v2_maps_chat_api_server_error_to_sanitized_502(monkeypatch):
    db = _CompletionReadyChatSessionDb()

    async def fake_resolve_byok_credentials(*args, **kwargs):
        return _ByokResolution()

    def fake_provider_call(**kwargs):
        raise ChatAPIError(
            "provider leaked token and /private/provider/cache/path",
            status_code=status.HTTP_502_BAD_GATEWAY,
            provider="local-llm",
        )

    monkeypatch.setenv("ENABLE_LOCAL_LLM_PROVIDER", "true")
    monkeypatch.setattr(character_chat_sessions, "resolve_byok_credentials", fake_resolve_byok_credentials)
    monkeypatch.setattr(character_chat_sessions, "provider_requires_api_key", lambda provider: False)
    monkeypatch.setattr(character_chat_sessions, "get_character_rate_limiter", lambda: _NoopCharacterRateLimiter())
    monkeypatch.setattr(character_chat_sessions, "_should_enforce_char_chat_strict_model_selection", lambda: False)
    monkeypatch.setattr(character_chat_sessions, "perform_chat_api_call", fake_provider_call)

    with pytest.raises(HTTPException) as exc_info:
        await character_chat_sessions.character_chat_completion(
            chat_id="chat-1",
            body=CharacterChatCompletionV2Request(
                provider="local-llm",
                model="local-test",
                save_to_db=False,
                include_character_context=False,
            ),
            db=db,
            current_user=_test_user(),
        )

    assert exc_info.value.status_code == status.HTTP_502_BAD_GATEWAY
    assert exc_info.value.detail == "Chat provider error"


@pytest.mark.asyncio
async def test_persist_streamed_assistant_message_maps_db_error_to_sanitized_500():
    db = _BrokenChatSessionDb(CharactersRAGDBError("sqlite persist exploded"), raise_on_get=True)

    with pytest.raises(HTTPException) as exc_info:
        await character_chat_sessions.persist_streamed_assistant_message(
            chat_id="chat-1",
            body=CharacterChatStreamPersistRequest(assistant_content="hello"),
            db=db,
            current_user=_test_user(),
        )

    assert exc_info.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    assert exc_info.value.detail == "Failed to persist assistant message"


@pytest.mark.asyncio
async def test_persist_streamed_assistant_message_maps_conflict_error_to_409():
    db = _BrokenChatSessionDb(ConflictError("chat persist conflict"), raise_on_get=True)

    with pytest.raises(HTTPException) as exc_info:
        await character_chat_sessions.persist_streamed_assistant_message(
            chat_id="chat-1",
            body=CharacterChatStreamPersistRequest(assistant_content="hello"),
            db=db,
            current_user=_test_user(),
        )

    assert exc_info.value.status_code == status.HTTP_409_CONFLICT
    assert exc_info.value.detail == "chat persist conflict"
