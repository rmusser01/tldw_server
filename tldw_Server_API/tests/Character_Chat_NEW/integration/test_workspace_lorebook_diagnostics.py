"""Lorebook diagnostics use the same explicit ownership/scope as the saved chat."""

from datetime import datetime
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.api.v1.endpoints import character_chat_sessions
from tldw_Server_API.app.core.DB_Management.chacha.operation_scope import ChaChaOperationMiddleware
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB

pytestmark = pytest.mark.integration


@pytest.fixture(params=["sqlite", "postgres"])
def diagnostics_api(tmp_path, request):
    options = {"backend": request.getfixturevalue("pg_restricted_backend")} if request.param == "postgres" else {}
    db = CharactersRAGDB(str(tmp_path / "diagnostics.db"), client_id="1", owner_user_id="1", **options)
    try:
        db.upsert_workspace("workspace-a", "Research")
        chat = db.add_conversation({"title": "Research", "scope_type": "workspace", "workspace_id": "workspace-a"})
        app = FastAPI()
        app.add_middleware(ChaChaOperationMiddleware)
        app.include_router(character_chat_sessions.router, prefix="/api/v1/chats")
        assert "/api/v1/chats/{chat_id}/diagnostics/lorebook" in app.openapi()["paths"]  # nosec B101 -- pytest public route-fixture guard.
        app.dependency_overrides[get_chacha_db_for_user] = lambda: db
        app.dependency_overrides[character_chat_sessions.get_request_user] = lambda: SimpleNamespace(id=1)
        with TestClient(app) as client:
            yield client, db, chat, {"scope_type": "workspace", "workspace_id": "workspace-a"}
    finally:
        db.close_all_connections()


def test_scoped_workspace_lorebook_diagnostics(diagnostics_api):
    client, db, chat, scope = diagnostics_api
    ids = []
    for number in (1, 2):
        mid = str(uuid4())
        db.add_message(
            {
                "id": mid,
                "conversation_id": chat,
                "sender": "assistant",
                "content": f"Turn {number}",
                "timestamp": f"2026-10-29T00:00:0{number}Z",
                "parent_message_id": None,
            }
        )
        assert db.add_message_metadata(mid, extra={"lorebook_diagnostics": [{"entry_id": number, "keyword": "Selene"}]})  # nosec B101 -- pytest setup assertion.
        ids.append(mid)
    response = client.get(f"/api/v1/chats/{chat}/diagnostics/lorebook", params={**scope, "order": "desc", "size": 1})
    assert response.status_code == 200  # nosec B101 -- pytest behavior assertion.
    body = response.json()
    assert body["chat_id"] == chat and body["character_id"] is None  # nosec B101 -- pytest behavior assertion.
    assert body["total_turns_with_diagnostics"] == 2  # nosec B101 -- pytest behavior assertion.
    assert [row["message_id"] for row in body["turns"]] == [ids[1]]  # nosec B101 -- pytest behavior assertion.
    assert body["turns"][0]["diagnostics"][0]["entry_id"] == 2  # nosec B101 -- pytest behavior assertion.
    stored_time = db.get_message_by_id(ids[1])["timestamp"]
    expected_time = stored_time.isoformat() if isinstance(stored_time, datetime) else stored_time
    assert body["turns"][0]["timestamp"] == expected_time  # nosec B101 -- pytest wire-value assertion.
    second = client.get(
        f"/api/v1/chats/{chat}/diagnostics/lorebook", params={**scope, "order": "desc", "size": 1, "page": 2}
    )
    assert second.status_code == 200 and [row["message_id"] for row in second.json()["turns"]] == [ids[0]]  # nosec B101 -- pytest behavior assertion.


def test_default_global_lorebook_diagnostics_unchanged(diagnostics_api):
    client, db, _, _ = diagnostics_api
    chat = db.add_conversation({"title": "Global"})
    response = client.get(f"/api/v1/chats/{chat}/diagnostics/lorebook")
    assert response.status_code == 200 and response.json()["turns"] == []  # nosec B101 -- pytest behavior assertion.


@pytest.mark.parametrize(
    "kind,expected",
    [
        ("omitted", 404),
        ("wrong_workspace", 404),
        ("global", 404),
        ("missing_workspace", 400),
        ("global_with_workspace", 404),
        ("invalid_scope", 422),
        ("foreign_user", 403),
        ("deleted", 404),
    ],
)
def test_lorebook_scope_refused_before_history(diagnostics_api, monkeypatch, kind, expected):
    client, db, chat, scope = diagnostics_api
    if kind == "omitted":
        scope = {}
    elif kind == "wrong_workspace":
        scope = {**scope, "workspace_id": "other"}
    elif kind == "global":
        scope = {"scope_type": "global"}
    elif kind == "missing_workspace":
        scope = {"scope_type": "workspace"}
    elif kind == "global_with_workspace":
        scope = {**scope, "scope_type": "global"}
    elif kind == "invalid_scope":
        scope = {**scope, "scope_type": "invalid"}
    elif kind == "foreign_user":
        client.app.dependency_overrides[character_chat_sessions.get_request_user] = lambda: SimpleNamespace(id=2)
    elif kind == "deleted":
        assert db.soft_delete_conversation(chat, expected_version=db.get_conversation_by_id(chat)["version"])  # nosec B101 -- pytest setup assertion.

    def reject_history(*args, **kwargs):
        raise AssertionError("Scope denial read protected history")

    monkeypatch.setattr(db, "get_conversation_settings", reject_history)
    monkeypatch.setattr(db, "get_messages_for_conversation", reject_history)
    monkeypatch.setattr(db, "get_message_metadata", reject_history)
    response = client.get(f"/api/v1/chats/{chat}/diagnostics/lorebook", params=scope)
    assert response.status_code == expected  # nosec B101 -- pytest behavior assertion.
    assert "Turn" not in response.text and "Selene" not in response.text  # nosec B101 -- pytest behavior assertion.
