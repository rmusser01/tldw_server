"""POST /api/v1/chats/import through the real app: auth, middleware and routing (D7 P8).

The contract itself is covered in
``Character_Chat_NEW/integration/test_chat_import_local_chats.py`` against a
small app. These tests run the same route inside the full application, where
authentication, the expected-user guard and the request middleware are real.
"""

from __future__ import annotations

from typing import Any

import pytest

from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.core.Chat.history_selection import resolve_history_selection
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB

pytestmark = pytest.mark.integration

PATH = "/api/v1/chats/import"
CHAT_ID = "4c1d7e2a-9b3f-4a6c-8d5e-2f1a0b9c8d7e"


def _body() -> dict[str, Any]:
    """A question with two answers; the second answer was continued."""
    return {
        "id": CHAT_ID,
        "title": "Saved from this device",
        "created_at": "2025-03-01T09:59:30.000Z",
        "messages": [
            {"id": "pa_full_q1", "role": "user", "content": "first question", "timestamp": "2025-03-01T10:00:00.000Z"},
            {
                "id": "pa_full_a1",
                "parent_message_id": "pa_full_q1",
                "role": "assistant",
                "content": "first answer",
                "timestamp": "2025-03-01T10:00:05.000Z",
                "metadata": {"model_id": "gpt-4o-mini", "provider": "openai", "generation_status": "complete"},
            },
            {
                "id": "pa_full_a2",
                "parent_message_id": "pa_full_q1",
                "role": "assistant",
                "content": "regenerated answer",
                "timestamp": "2025-03-01T10:00:09.000Z",
                "metadata": {"model_id": "gpt-4o", "generation_status": "stopped"},
            },
            {
                "id": "pa_full_q2",
                "parent_message_id": "pa_full_a2",
                "role": "user",
                "content": "follow-up",
                "timestamp": "2025-03-01T10:01:00.000Z",
            },
        ],
    }


@pytest.fixture
def import_api(test_client, populated_chacha_db, auth_headers):
    """The authenticated owner's store, shared with the request dependency."""
    db = CharactersRAGDB(db_path=populated_chacha_db.db_path_str, client_id="1", owner_user_id="1")
    test_client.app.dependency_overrides[get_chacha_db_for_user] = lambda: db
    yield test_client, db, auth_headers
    test_client.app.dependency_overrides.pop(get_chacha_db_for_user, None)
    db.close_connection()


def _message_count(db: CharactersRAGDB) -> int:
    return int(
        db.execute_query(
            "SELECT COUNT(*) AS n FROM messages WHERE conversation_id = ?", (CHAT_ID,), read_only=True
        ).fetchone()["n"]
    )


def test_import_is_readable_and_continuable_in_the_full_app(import_api) -> None:
    client, db, headers = import_api
    imported = client.post(PATH, headers=headers, json=_body())
    assert imported.status_code == 201, imported.text
    assert imported.json()["id"] == CHAT_ID
    assert imported.json()["message_count"] == 4
    assert db.get_conversation_by_id(CHAT_ID)["client_id"] == "1"

    listed = client.get(
        f"/api/v1/chats/{CHAT_ID}/messages", headers=headers, params={"include_metadata": True, "limit": 50}
    )
    assert listed.status_code == 200, listed.text
    rows = listed.json()["messages"]
    assert [(row["id"], row["parent_message_id"], row["sender"]) for row in rows] == [
        ("pa_full_q1", None, "user"),
        ("pa_full_a1", "pa_full_q1", "assistant"),
        ("pa_full_a2", "pa_full_q1", "assistant"),
        ("pa_full_q2", "pa_full_a2", "user"),
    ]
    assert rows[2]["metadata_extra"] == {"sender_role": "assistant", "model_id": "gpt-4o", "generation_status": "stopped"}

    captured = client.post(
        f"/api/v1/chat/conversations/{CHAT_ID}/history/selection",
        headers=headers,
        json={
            "purpose": "send",
            "view": {
                "view_session_id": "view-one",
                "conversation_id": CHAT_ID,
                "interpretation": {"kind": "parent_graph_v1"},
                "cursor": {"kind": "after_message", "message_id": "pa_full_q2"},
                "selection_revision": 1,
            },
        },
    )
    assert captured.status_code == 200, captured.text
    body = captured.json()
    assert body["status"] == "captured"
    assert [row["id"] for row in body["rows"]] == ["pa_full_q1", "pa_full_a2", "pa_full_q2"]

    # The imported branch can be continued through the versioned send path.
    selection = resolve_history_selection(body["snapshot"], body["view"], "send", "client-only")["selection"]
    sent = client.post(
        f"/api/v1/chats/{CHAT_ID}/messages",
        headers=headers,
        json={"id": "pa_full_q3", "role": "user", "content": "one more", "tldw_history_selection_v1": selection},
    )
    assert sent.status_code == 201, sent.text
    assert db.get_message_by_id("pa_full_q3")["parent_message_id"] == "pa_full_q2"

    replay = client.post(PATH, headers=headers, json=_body())
    assert replay.status_code == 200, replay.text
    assert replay.headers["Idempotency-Replayed"] == "true"
    assert replay.json()["message_count"] == 5


def test_import_for_an_account_the_client_no_longer_holds_is_412(import_api) -> None:
    """The chat is saved to the account the user saw, or not at all."""
    client, db, headers = import_api
    stale = client.post(PATH, headers={**headers, "X-TLDW-Expected-User-ID": "999"}, json=_body())
    assert stale.status_code == 412, stale.text
    assert stale.json()["detail"]["code"] == "request_config_scope_changed"
    assert db.get_conversation_by_id(CHAT_ID, include_deleted=True) is None
    assert _message_count(db) == 0
    current = client.post(PATH, headers={**headers, "X-TLDW-Expected-User-ID": "1"}, json=_body())
    assert current.status_code == 201, current.text


def test_import_without_credentials_is_refused_and_writes_nothing(import_api) -> None:
    client, db, _headers = import_api
    response = client.post(PATH, json=_body())
    assert response.status_code in {401, 403}, response.text
    assert db.get_conversation_by_id(CHAT_ID, include_deleted=True) is None
    assert _message_count(db) == 0


def test_invalid_import_in_the_full_app_does_not_echo_the_chat(import_api) -> None:
    client, db, headers = import_api
    body = _body()
    del body["title"]
    response = client.post(PATH, headers=headers, json=body)
    assert response.status_code == 422, response.text
    assert response.json()["detail"] == [{"type": "missing", "loc": ["body", "title"], "msg": "Field required"}]
    assert "first question" not in response.text
    assert _message_count(db) == 0
