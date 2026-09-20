from __future__ import annotations

import json
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.api.v1.endpoints import chat as chat_router
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_request_user
from tldw_Server_API.app.core.Character_Chat.character_conversation_factory import (
    create_character_conversation,
)
from tldw_Server_API.app.core.Character_Chat.chat_settings_validation import (
    INTERNAL_CHAT_SETTINGS_KEYS,
    MAX_CHAT_SETTINGS_BYTES,
)
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB, ConflictError

pytestmark = pytest.mark.unit


def _build_app(db: CharactersRAGDB, user_id: int = 1) -> TestClient:
    app = FastAPI()
    app.include_router(chat_router.router, prefix="/api/v1/chat")
    app.dependency_overrides[get_chacha_db_for_user] = lambda: db
    app.dependency_overrides[get_request_user] = lambda: SimpleNamespace(id=user_id)
    return TestClient(app)


def _seed_conversation_with_messages(db: CharactersRAGDB, client_id: str = "1") -> str:
    char_id = db.add_character_card(
        {
            "name": "Knowledge QA Share Test",
            "description": "desc",
            "personality": "helpful",
            "system_prompt": "You are helpful.",
            "client_id": client_id,
        }
    )
    conversation_id = db.add_conversation(
        {
            "character_id": char_id,
            "title": "Shareable conversation",
            "client_id": client_id,
        }
    )
    user_message_id = db.add_message(
        {
            "conversation_id": conversation_id,
            "sender": "user",
            "content": "What is the summary?",
            "client_id": client_id,
        }
    )
    assistant_message_id = db.add_message(
        {
            "conversation_id": conversation_id,
            "sender": "assistant",
            "content": "The summary is positive. [1]",
            "parent_message_id": user_message_id,
            "client_id": client_id,
        }
    )
    assert assistant_message_id is not None
    assert db.set_message_rag_context(
        assistant_message_id,
        {
            "search_query": "What is the summary?",
            "generated_answer": "The summary is positive. [1]",
            "retrieved_documents": [
                {
                    "id": "doc-1",
                    "title": "Report",
                    "excerpt": "Positive findings.",
                }
            ],
        },
    )
    return conversation_id


def _seed_roleplay_conversation(db: CharactersRAGDB, client_id: str = "1") -> str:
    character_id = db.add_character_card(
        {
            "name": "Roleplay Share Test",
            "first_message": "Hello from the frozen card.",
            "client_id": client_id,
        }
    )
    return create_character_conversation(
        db,
        conversation_data={
            "character_id": character_id,
            "title": "Roleplay share conversation",
            "client_id": client_id,
        },
        provider="local-llm",
        model="local-test",
    )


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def test_share_link_create_list_revoke_and_public_resolve(tmp_path, monkeypatch):
    db_path = tmp_path / "chacha.db"
    db = CharactersRAGDB(db_path=str(db_path), client_id="1")
    client = _build_app(db, user_id=1)
    conversation_id = _seed_conversation_with_messages(db, client_id="1")

    async def _fake_get_db_for_user_id(user_id: int, _auth_user_id: str):
        assert int(user_id) == 1
        return db

    monkeypatch.setattr(chat_router, "get_chacha_db_for_user_id", _fake_get_db_for_user_id)

    create_response = client.post(
        f"/api/v1/chat/conversations/{conversation_id}/share-links",
        json={"permission": "view"},
    )
    assert create_response.status_code == 200, create_response.text
    created = create_response.json()
    assert created["permission"] == "view"
    assert created["share_id"]
    assert created["token"]
    assert created["share_path"].startswith("/knowledge/shared/")

    list_response = client.get(f"/api/v1/chat/conversations/{conversation_id}/share-links")
    assert list_response.status_code == 200, list_response.text
    listed = list_response.json()
    assert listed["conversation_id"] == conversation_id
    assert len(listed["links"]) == 1
    assert listed["links"][0]["id"] == created["share_id"]

    resolve_response = client.get(f"/api/v1/chat/shared/conversations/{created['token']}")
    assert resolve_response.status_code == 200, resolve_response.text
    resolved = resolve_response.json()
    assert resolved["conversation_id"] == conversation_id
    assert resolved["permission"] == "view"
    assert resolved["shared_by_user_id"] == "1"
    assert len(resolved["messages"]) >= 2
    assert any(message.get("rag_context") for message in resolved["messages"])

    revoke_response = client.delete(
        f"/api/v1/chat/conversations/{conversation_id}/share-links/{created['share_id']}"
    )
    assert revoke_response.status_code == 200, revoke_response.text
    revoked = revoke_response.json()
    assert revoked["success"] is True
    assert revoked["share_id"] == created["share_id"]

    revoked_resolve_response = client.get(
        f"/api/v1/chat/shared/conversations/{created['token']}"
    )
    assert revoked_resolve_response.status_code == 403
    assert revoked_resolve_response.json()["detail"] == "Share link revoked"


def test_share_link_write_preserves_valid_roleplay_authority(tmp_path):
    db = CharactersRAGDB(db_path=str(tmp_path / "roleplay-share.db"), client_id="1")
    client = _build_app(db, user_id=1)
    conversation_id = _seed_roleplay_conversation(db)
    before = db.get_roleplay_resume_state(conversation_id)

    response = client.post(
        f"/api/v1/chat/conversations/{conversation_id}/share-links",
        json={"permission": "view", "label": "Roleplay link"},
    )

    assert response.status_code == 200, response.text
    after = db.get_roleplay_resume_state(conversation_id)
    assert after["settings_version"] == before["settings_version"] + 1
    assert after["history_version"] == before["history_version"]
    assert after["settings"]["roleplayResumeV1"] == before["settings"][
        "roleplayResumeV1"
    ]
    assert after["settings"]["roleplayBehaviorV1"] == before["settings"][
        "roleplayBehaviorV1"
    ]


def test_share_link_rejects_public_projection_overflow_atomically(tmp_path):
    db = CharactersRAGDB(db_path=str(tmp_path / "roleplay-share-limit.db"), client_id="1")
    client = _build_app(db, user_id=1)
    conversation_id = _seed_roleplay_conversation(db)
    state = db.get_roleplay_resume_state(conversation_id)
    settings = dict(state["settings"])
    internal = {
        key: settings[key]
        for key in INTERNAL_CHAT_SETTINGS_KEYS
        if key in settings
    }
    public = {
        key: value
        for key, value in settings.items()
        if key not in INTERNAL_CHAT_SETTINGS_KEYS
    }
    public["boundaryPadding"] = ""
    padding_size = MAX_CHAT_SETTINGS_BYTES - len(_canonical_json_bytes(public))
    assert padding_size >= 0
    public["boundaryPadding"] = "x" * padding_size
    assert len(_canonical_json_bytes(public)) == MAX_CHAT_SETTINGS_BYTES
    bounded_settings = {**public, **internal}
    assert db.upsert_conversation_settings(
        conversation_id,
        bounded_settings,
        expected_settings_version=state["settings_version"],
    )
    before = db.get_roleplay_resume_state(conversation_id)

    response = client.post(
        f"/api/v1/chat/conversations/{conversation_id}/share-links",
        json={"permission": "view"},
    )

    assert response.status_code == 413, response.text
    after = db.get_roleplay_resume_state(conversation_id)
    assert after["settings_version"] == before["settings_version"]
    assert after["history_version"] == before["history_version"]
    assert after["settings"] == before["settings"]


def test_share_link_resolve_rejects_malformed_token(tmp_path):
    db_path = tmp_path / "chacha.db"
    db = CharactersRAGDB(db_path=str(db_path), client_id="1")
    client = _build_app(db, user_id=1)

    response = client.get("/api/v1/chat/shared/conversations/not-a-valid-token")
    assert response.status_code == 400
    assert response.json()["detail"] == "Malformed share token"


def test_share_link_settings_write_uses_version_cas_and_surfaces_conflict():
    class _ConflictingDB:
        def __init__(self) -> None:
            self.expected_settings_version: int | None = None

        def get_conversation_settings(self, conversation_id: str):
            return {
                "settings": {"preserved": True},
                "settings_version": 9,
            }

        def transaction(self):
            return nullcontext(object())

        def get_roleplay_resume_state(
            self,
            conversation_id: str,
            *,
            conn,
            lock_for_update: bool,
            owner_client_id: str | None = None,
        ):
            assert lock_for_update is True
            return {
                "conversation": {
                    "id": conversation_id,
                    "character_id": 1,
                    "client_id": owner_client_id,
                },
                "settings": {"preserved": True},
                "settings_version": 9,
                "behavior_snapshot": {"status": "missing"},
            }

        def upsert_conversation_settings(
            self,
            conversation_id: str,
            settings: dict,
            *,
            conn=None,
            expected_settings_version: int | None = None,
        ) -> bool:
            self.expected_settings_version = expected_settings_version
            raise ConflictError("stale settings")

    db = _ConflictingDB()
    settings, links, settings_version = chat_router._load_knowledge_qa_share_links(
        db, "conversation-1"
    )

    with pytest.raises(chat_router.HTTPException) as exc_info:
        chat_router._persist_knowledge_qa_share_links(
            db,
            "conversation-1",
            [{"id": "share-1"}],
            conversation={"id": "conversation-1", "character_id": 1, "client_id": "1"},
            expected_settings_version=settings_version,
        )

    assert db.expected_settings_version == 9
    assert exc_info.value.status_code == 409


def test_share_link_settings_write_expects_absent_row_on_first_write():
    class _EmptyDB:
        def __init__(self) -> None:
            self.expected_settings_version: int | None = None

        def get_conversation_settings(self, conversation_id: str):
            return None

        def transaction(self):
            return nullcontext(object())

        def get_roleplay_resume_state(
            self,
            conversation_id: str,
            *,
            conn,
            lock_for_update: bool,
            owner_client_id: str | None = None,
        ):
            assert lock_for_update is True
            return {
                "conversation": {
                    "id": conversation_id,
                    "character_id": 1,
                    "client_id": owner_client_id,
                },
                "settings": None,
                "settings_version": None,
                "behavior_snapshot": {"status": "missing"},
            }

        def upsert_conversation_settings(
            self,
            conversation_id: str,
            settings: dict,
            *,
            conn=None,
            expected_settings_version: int | None = None,
        ) -> bool:
            self.expected_settings_version = expected_settings_version
            return True

    db = _EmptyDB()
    settings, links, settings_version = chat_router._load_knowledge_qa_share_links(
        db, "conversation-1"
    )
    chat_router._persist_knowledge_qa_share_links(
        db,
        "conversation-1",
        [{"id": "share-1"}],
        conversation={"id": "conversation-1", "character_id": 1, "client_id": "1"},
        expected_settings_version=settings_version,
    )

    assert db.expected_settings_version == 0


def test_share_link_writer_uses_transactional_conversation_identity(monkeypatch):
    stale_conversation = {
        "id": "conversation-1",
        "character_id": 1,
        "client_id": "1",
    }
    current_conversation = {
        **stale_conversation,
        "version": 2,
        "character_id": 2,
        "assistant_kind": "character",
        "assistant_id": "2",
        "persona_memory_mode": None,
        "scope_type": "global",
        "workspace_id": None,
    }

    class _RacingDB:
        client_id = "1"

        def __init__(self) -> None:
            self.owner_client_id: str | None = None

        def transaction(self):
            return nullcontext(object())

        def get_roleplay_resume_state(
            self,
            _conversation_id: str,
            *,
            conn,
            lock_for_update: bool,
            owner_client_id: str | None = None,
        ):
            assert conn is not None
            assert lock_for_update is True
            self.owner_client_id = owner_client_id
            return {
                "conversation": current_conversation,
                "settings": {},
                "settings_version": 3,
                "behavior_snapshot": {"status": "missing"},
            }

        def upsert_conversation_settings(
            self,
            _conversation_id: str,
            _settings: dict,
            *,
            conn,
            expected_settings_version: int,
        ) -> bool:
            assert conn is not None
            assert expected_settings_version == 3
            return True

    seen: dict[str, object] = {}

    def _capture_validation(settings, **kwargs):
        seen["conversation"] = kwargs.get("conversation")
        return settings

    monkeypatch.setattr(
        chat_router,
        "validate_chat_settings_storage",
        _capture_validation,
    )
    db = _RacingDB()

    chat_router._persist_knowledge_qa_share_links(
        db,
        "conversation-1",
        [{"id": "share-1"}],
        conversation=stale_conversation,
        expected_settings_version=3,
    )

    assert seen["conversation"] == current_conversation
    assert db.owner_client_id == "1"


def test_share_link_create_requires_exact_scope_match(tmp_path):
    db_path = tmp_path / "chacha.db"
    db = CharactersRAGDB(db_path=str(db_path), client_id="1")
    client = _build_app(db, user_id=1)
    db.upsert_workspace("ws-1", "Workspace One")

    char_id = db.add_character_card(
        {
            "name": "Scoped Share Test",
            "description": "desc",
            "personality": "helpful",
            "system_prompt": "You are helpful.",
            "client_id": "1",
        }
    )
    conversation_id = db.add_conversation(
        {
            "id": "workspace-conversation",
            "character_id": char_id,
            "title": "Workspace-only conversation",
            "client_id": "1",
            "scope_type": "workspace",
            "workspace_id": "ws-1",
        }
    )
    assert conversation_id == "workspace-conversation"

    missing_scope = client.post(
        f"/api/v1/chat/conversations/{conversation_id}/share-links",
        json={"permission": "view"},
    )
    assert missing_scope.status_code == 404

    wrong_scope = client.post(
        f"/api/v1/chat/conversations/{conversation_id}/share-links",
        params={"scope_type": "workspace", "workspace_id": "ws-2"},
        json={"permission": "view"},
    )
    assert wrong_scope.status_code == 404

    correct_scope = client.post(
        f"/api/v1/chat/conversations/{conversation_id}/share-links",
        params={"scope_type": "workspace", "workspace_id": "ws-1"},
        json={"permission": "view"},
    )
    assert correct_scope.status_code == 200, correct_scope.text
