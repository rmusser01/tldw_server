from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.api.v1.endpoints import character_chat_sessions
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_request_user
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB


pytestmark = pytest.mark.unit


def _build_app(db: CharactersRAGDB) -> TestClient:
    app = FastAPI()
    app.include_router(character_chat_sessions.router, prefix="/api/v1/chats")
    app.dependency_overrides[get_chacha_db_for_user] = lambda: db
    app.dependency_overrides[get_request_user] = lambda: SimpleNamespace(id="user-1")
    return TestClient(app)


def _seed_chat(
    db: CharactersRAGDB,
    *,
    conversation_id: str,
    title: str,
    character_id: int | None,
) -> str:
    payload = {
        "id": conversation_id,
        "root_id": conversation_id,
        "title": title,
        "character_id": character_id,
        "client_id": "user-1",
    }
    if character_id is None:
        payload["assistant_kind"] = "persona"
        payload["assistant_id"] = f"persona-{conversation_id}"
        payload["persona_memory_mode"] = "read_only"
    return db.add_conversation(payload)


def test_list_chat_sessions_rejects_incompatible_character_scope_and_character_id(
    tmp_path,
):
    db = CharactersRAGDB(db_path=str(tmp_path / "chacha.db"), client_id="user-1")
    app = _build_app(db)

    response = app.get(
        "/api/v1/chats/",
        params={"character_scope": "non_character", "character_id": 5},
    )

    assert response.status_code == 400
    assert "character_scope" in response.json()["detail"]


def test_list_chat_sessions_uses_batched_message_counts_and_filters_character_scope(
    tmp_path,
    monkeypatch,
):
    db = CharactersRAGDB(db_path=str(tmp_path / "chacha.db"), client_id="user-1")
    app = _build_app(db)

    character_id = db.add_character_card(
        {
            "name": "Test Character",
            "description": "desc",
            "personality": "helpful",
            "system_prompt": "You are helpful.",
            "client_id": "user-1",
        }
    )
    character_chat_id = _seed_chat(
        db,
        conversation_id="character-chat",
        title="Character chat",
        character_id=character_id,
    )
    _seed_chat(
        db,
        conversation_id="plain-chat",
        title="Plain chat",
        character_id=None,
    )

    db.add_message(
        {
            "id": "message-1",
            "conversation_id": character_chat_id,
            "sender": "user",
            "content": "hello",
            "client_id": "user-1",
        }
    )

    single_calls: list[str] = []
    batched_calls: list[list[str]] = []
    original_batch_counter = db.count_messages_for_conversations

    def fail_single(conversation_id: str, include_deleted: bool = False) -> int:
        single_calls.append(conversation_id)
        raise AssertionError("single counter should not be used for chat listing")

    def track_batch(conversation_ids: list[str], include_deleted: bool = False) -> dict[str, int]:
        batched_calls.append(list(conversation_ids))
        return original_batch_counter(conversation_ids, include_deleted=include_deleted)

    monkeypatch.setattr(db, "count_messages_for_conversation", fail_single)
    monkeypatch.setattr(db, "count_messages_for_conversations", track_batch)

    response = app.get(
        "/api/v1/chats/",
        params={"character_scope": "character"},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert [item["id"] for item in body["chats"]] == ["character-chat"]
    assert body["total"] == 1
    assert body["pagination"] == {
        "mode": "offset",
        "limit": 50,
        "offset": 0,
        "total": 1,
        "has_more": False,
        "next_offset": None,
    }
    assert body["has_more"] is False
    assert body["next_offset"] is None
    assert single_calls == []
    assert batched_calls == [["character-chat"]]


def test_list_chat_sessions_can_skip_message_counts_for_lean_overview(
    tmp_path,
    monkeypatch,
):
    db = CharactersRAGDB(db_path=str(tmp_path / "chacha.db"), client_id="user-1")
    app = _build_app(db)

    _seed_chat(
        db,
        conversation_id="plain-chat",
        title="Plain chat",
        character_id=None,
    )

    single_calls: list[str] = []
    batch_calls: list[list[str]] = []

    def track_single(conversation_id: str, include_deleted: bool = False) -> int:
        single_calls.append(conversation_id)
        return 0

    def track_batch(conversation_ids: list[str], include_deleted: bool = False) -> dict[str, int]:
        batch_calls.append(list(conversation_ids))
        return {}

    monkeypatch.setattr(db, "count_messages_for_conversation", track_single)
    monkeypatch.setattr(db, "count_messages_for_conversations", track_batch)

    response = app.get(
        "/api/v1/chats/",
        params={"include_message_counts": False},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert [item["id"] for item in body["chats"]] == ["plain-chat"]
    assert body["chats"][0]["message_count"] is None
    assert single_calls == []
    assert batch_calls == []


def test_list_chat_sessions_defaults_omitted_scope_to_global(tmp_path):
    db = CharactersRAGDB(db_path=str(tmp_path / "chacha.db"), client_id="user-1")
    app = _build_app(db)

    character_id = db.add_character_card(
        {
            "name": "Scope Character",
            "description": "desc",
            "personality": "helpful",
            "system_prompt": "You are helpful.",
            "client_id": "user-1",
        }
    )
    db.upsert_workspace("ws-1", "Workspace One")
    _seed_chat(
        db,
        conversation_id="global-chat",
        title="Global chat",
        character_id=character_id,
    )
    db.add_conversation(
        {
            "id": "workspace-chat",
            "root_id": "workspace-chat",
            "title": "Workspace chat",
            "character_id": character_id,
            "client_id": "user-1",
            "scope_type": "workspace",
            "workspace_id": "ws-1",
        }
    )

    response = app.get("/api/v1/chats/")

    assert response.status_code == 200, response.text
    body = response.json()
    assert [item["id"] for item in body["chats"]] == ["global-chat"]
    assert body["total"] == 1


def test_get_chat_session_requires_exact_scope_match(tmp_path):
    db = CharactersRAGDB(db_path=str(tmp_path / "chacha.db"), client_id="user-1")
    app = _build_app(db)

    character_id = db.add_character_card(
        {
            "name": "Scope Detail Character",
            "description": "desc",
            "personality": "helpful",
            "system_prompt": "You are helpful.",
            "client_id": "user-1",
        }
    )
    db.upsert_workspace("ws-1", "Workspace One")
    db.add_conversation(
        {
            "id": "workspace-chat",
            "root_id": "workspace-chat",
            "title": "Workspace chat",
            "character_id": character_id,
            "client_id": "user-1",
            "scope_type": "workspace",
            "workspace_id": "ws-1",
        }
    )

    missing_scope = app.get("/api/v1/chats/workspace-chat")
    assert missing_scope.status_code == 404

    wrong_scope = app.get(
        "/api/v1/chats/workspace-chat",
        params={"scope_type": "workspace", "workspace_id": "ws-2"},
    )
    assert wrong_scope.status_code == 404

    correct_scope = app.get(
        "/api/v1/chats/workspace-chat",
        params={"scope_type": "workspace", "workspace_id": "ws-1"},
    )
    assert correct_scope.status_code == 200, correct_scope.text
    payload = correct_scope.json()
    assert payload["scope_type"] == "workspace"
    assert payload["workspace_id"] == "ws-1"
