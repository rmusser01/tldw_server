from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.api.v1.endpoints import chat as chat_router
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_request_user
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB

pytestmark = pytest.mark.unit


def _build_app(db: CharactersRAGDB) -> TestClient:
    app = FastAPI()
    app.include_router(chat_router.router, prefix="/api/v1/chat")

    app.dependency_overrides[get_chacha_db_for_user] = lambda: db
    app.dependency_overrides[get_request_user] = lambda: SimpleNamespace(id="user-1")
    return TestClient(app)


def test_knowledge_save_creates_note_with_backlinks(tmp_path):


    db_path = tmp_path / "chacha.db"
    db = CharactersRAGDB(db_path=str(db_path), client_id="user-1")
    app = _build_app(db)

    conv_id = db.add_conversation(
        {"id": "conv-1", "root_id": "conv-1", "character_id": 1, "title": "Test Conv", "client_id": "user-1"}
    )
    msg_id = db.add_message({"conversation_id": conv_id, "sender": "user", "content": "hello"})

    payload = {
        "conversation_id": conv_id,
        "message_id": msg_id,
        "snippet": "important snippet",
        "tags": ["alpha", "beta"],
        "make_flashcard": False,
        "export_to": "none",
    }

    resp = app.post("/api/v1/chat/knowledge/save", json=payload)
    assert resp.status_code == 201, resp.text
    data = resp.json()
    note_id = data["note_id"]
    note = db.get_note_by_id(note_id)
    assert note is not None
    assert note.get("conversation_id") == conv_id
    assert note.get("message_id") == msg_id
    keywords = db.get_keywords_for_note(note_id)
    keyword_texts = sorted([k.get("keyword") for k in keywords])
    assert keyword_texts == ["alpha", "beta"]


def test_knowledge_save_export_guard(tmp_path, monkeypatch):


    db_path = tmp_path / "chacha.db"
    db = CharactersRAGDB(db_path=str(db_path), client_id="user-1")
    app = _build_app(db)

    conv_id = db.add_conversation(
        {"id": "conv-2", "root_id": "conv-2", "character_id": 1, "title": "Test Conv", "client_id": "user-1"}
    )

    payload = {
        "conversation_id": conv_id,
        "snippet": "another snippet",
        "tags": None,
        "make_flashcard": False,
        "export_to": "notion",
    }

    # Ensure flag is off
    monkeypatch.delenv("CHAT_CONNECTORS_V2_ENABLED", raising=False)
    resp = app.post("/api/v1/chat/knowledge/save", json=payload)
    assert resp.status_code == 201
    assert resp.json().get("export_status") == "skipped_disabled"

    # Enable and ensure it passes
    monkeypatch.setenv("CHAT_CONNECTORS_V2_ENABLED", "true")
    resp_ok = app.post("/api/v1/chat/knowledge/save", json=payload)
    assert resp_ok.status_code == 201


def test_knowledge_save_requires_exact_scope_match_for_workspace_conversations(tmp_path):
    db_path = tmp_path / "chacha.db"
    db = CharactersRAGDB(db_path=str(db_path), client_id="user-1")
    app = _build_app(db)

    db.upsert_workspace("ws-1", "Workspace One")
    conv_id = db.add_conversation(
        {
            "id": "workspace-conv-1",
            "root_id": "workspace-conv-1",
            "character_id": 1,
            "title": "Workspace Conv",
            "client_id": "user-1",
            "scope_type": "workspace",
            "workspace_id": "ws-1",
        }
    )
    msg_id = db.add_message({"conversation_id": conv_id, "sender": "assistant", "content": "scoped"})

    base_payload = {
        "conversation_id": conv_id,
        "message_id": msg_id,
        "snippet": "workspace snippet",
        "make_flashcard": False,
        "export_to": "none",
    }

    missing_scope = app.post("/api/v1/chat/knowledge/save", json=base_payload)
    assert missing_scope.status_code == 404, missing_scope.text

    wrong_scope = app.post(
        "/api/v1/chat/knowledge/save",
        json={**base_payload, "scope_type": "workspace", "workspace_id": "ws-2"},
    )
    assert wrong_scope.status_code == 404, wrong_scope.text

    correct_scope = app.post(
        "/api/v1/chat/knowledge/save",
        json={**base_payload, "scope_type": "workspace", "workspace_id": "ws-1"},
    )
    assert correct_scope.status_code == 201, correct_scope.text


@pytest.mark.parametrize("make_flashcard", [False, True])
def test_knowledge_save_excludes_reasoning_and_saves_completed_card(tmp_path, make_flashcard):
    db = CharactersRAGDB(db_path=str(tmp_path / "chacha.db"), client_id="user-1")
    try:
        app = _build_app(db)
        conv_id = db.add_conversation({"character_id": 1, "title": "Garden", "client_id": "user-1"})
        msg_id = db.add_message({"conversation_id": conv_id, "sender": "assistant", "content": "Garden answer"})
        response = app.post(
            "/api/v1/chat/knowledge/save",
            json={
                "conversation_id": conv_id,
                "message_id": msg_id,
                "snippet": "<think>Private model reasoning.</think>The garden has seven raised beds.",
                "make_flashcard": make_flashcard,
                "flashcard_front": "How many raised beds does the garden have?",
                "flashcard_back": "<reasoning>Private reasoning.</reasoning>Seven raised beds.",
            },
        )
        assert response.status_code == 201, response.text
        saved = response.json()
        assert db.get_note_by_id(saved["note_id"])["content"] == "The garden has seven raised beds."
        if make_flashcard:
            card = db.get_flashcard(saved["flashcard_id"])
            assert card["front"] == "How many raised beds does the garden have?"
            assert card["back"] == "Seven raised beds."
            assert card["conversation_id"] == conv_id
            assert card["message_id"] == msg_id

    finally:
        db.close_connection()


@pytest.mark.parametrize(
    "card_fields",
    [
        {},
        {"flashcard_front": "Question"},
        {"flashcard_back": "Answer"},
        {"flashcard_front": "Question", "flashcard_back": " "},
        {"flashcard_front": " ", "flashcard_back": "Answer"},
        {"flashcard_front": "Question", "flashcard_back": "<think>Still reasoning"},
    ],
)
def test_knowledge_save_rejects_incomplete_card_before_creating_note(tmp_path, card_fields):
    db = CharactersRAGDB(db_path=str(tmp_path / "chacha.db"), client_id="user-1")
    try:
        app = _build_app(db)
        conv_id = db.add_conversation({"character_id": 1, "title": "Garden", "client_id": "user-1"})
        response = app.post(
            "/api/v1/chat/knowledge/save",
            json={
                "conversation_id": conv_id,
                "snippet": "Answer",
                "make_flashcard": True,
                **card_fields,
            },
        )
        assert response.status_code == 422, response.text
        assert db.count_notes() == 0

    finally:
        db.close_connection()


@pytest.mark.parametrize("scope_type", ["global", "workspace"])
def test_legacy_knowledge_save_uses_owned_parent_question_and_public_excerpt(tmp_path, scope_type):
    """Chatbook's legacy shape produces a complete card from its linked public turn."""
    db = CharactersRAGDB(db_path=str(tmp_path / "chacha.db"), client_id="user-1")
    try:
        app = _build_app(db)
        scope = {"scope_type": scope_type}
        if scope_type == "workspace":
            db.upsert_workspace("garden", "Garden workspace")
            scope["workspace_id"] = "garden"
        conv_id = db.add_conversation({"character_id": 1, "title": "Garden", "client_id": "user-1", **scope})
        question_id = db.add_message({
            "conversation_id": conv_id, "sender": "user",
            "content": "<think>Private draft.</think>How many raised beds does the garden have?",
        })
        answer_id = db.add_message({
            "conversation_id": conv_id, "sender": "assistant", "parent_message_id": question_id,
            "content": "<reasoning>Private model draft.</reasoning>The garden has seven raised beds. They contain herbs.",
        })
        db.add_message({"conversation_id": conv_id, "sender": "user", "content": "Which plants grow there?"})
        response = app.post("/api/v1/chat/knowledge/save", json={
            "conversation_id": conv_id, "message_id": answer_id,
            "snippet": "<think>Private model draft.</think>The garden has seven raised beds.",
            "make_flashcard": True, **scope,
        })
        assert response.status_code == 201, response.text
        saved = response.json()
        card = db.get_flashcard(saved["flashcard_id"])
        assert card["front"] == "How many raised beds does the garden have?"
        assert card["back"] == "The garden has seven raised beds."
        assert card["conversation_id"] == conv_id
        assert card["message_id"] == answer_id
        assert card["source_ref_id"] == saved["note_id"]
        assert db.get_note_by_id(saved["note_id"])["content"] == "The garden has seven raised beds."
    finally:
        db.close_connection()


@pytest.mark.parametrize(
    ("question", "answer", "snippet", "parent_sender"),
    [
        ("How many beds?", "Seven beds.", "Nineteen beds.", "user"),
        ("How many beds?", "<think>Seven beds.</think>", "Seven beds.", "user"),
        ("<think>How many beds?</think>", "Seven beds.", "Seven beds.", "user"),
        ("How many beds?", "Seven beds.", "Seven beds.", "assistant"),
        ('{"type":"text","text":"How many beds?"}', "Seven beds.", "Seven beds.", "user"),
        ("How many beds?", '[{"type":"reasoning","text":"Private draft"}]', '[{"type":"reasoning","text":"Private draft"}]', "user"),
    ],
)
def test_legacy_knowledge_save_rejects_ungrounded_turn_before_writes(tmp_path, question, answer, snippet, parent_sender):
    db = CharactersRAGDB(db_path=str(tmp_path / "chacha.db"), client_id="user-1")
    try:
        app = _build_app(db)
        conv_id = db.add_conversation({"character_id": 1, "client_id": "user-1"})
        parent_id = db.add_message({"conversation_id": conv_id, "sender": parent_sender, "content": question})
        answer_id = db.add_message({
            "conversation_id": conv_id, "sender": "assistant", "content": answer, "parent_message_id": parent_id,
        })
        response = app.post("/api/v1/chat/knowledge/save", json={
            "conversation_id": conv_id, "message_id": answer_id, "snippet": snippet, "make_flashcard": True,
        })
        assert response.status_code == 422, response.text
        assert db.count_notes() == 0
        assert db.list_flashcards() == []
    finally:
        db.close_connection()


def test_legacy_knowledge_save_rejects_foreign_parent_before_writes(tmp_path):
    db = CharactersRAGDB(db_path=str(tmp_path / "chacha.db"), client_id="user-1")
    try:
        app = _build_app(db)
        conv_id = db.add_conversation({"character_id": 1, "client_id": "user-1"})
        foreign_id = db.add_conversation({"character_id": 1, "client_id": "other-user"})
        parent_id = db.add_message({"conversation_id": foreign_id, "sender": "user", "content": "Private question?"})
        answer_id = db.add_message({
            "conversation_id": conv_id, "sender": "assistant", "content": "Seven beds.", "parent_message_id": parent_id,
        })
        response = app.post("/api/v1/chat/knowledge/save", json={
            "conversation_id": conv_id, "message_id": answer_id, "snippet": "Seven beds.", "make_flashcard": True,
        })
        assert response.status_code == 403, response.text
        assert db.count_notes() == 0
        assert db.list_flashcards() == []
    finally:
        db.close_connection()
