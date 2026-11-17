"""Workspace message creation must retain bounded citations atomically."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.requests import Request

from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.api.v1.endpoints import character_messages
from tldw_Server_API.app.core.Chat.history_selection import resolve_history_selection, snapshot_to_wire
from tldw_Server_API.app.core.Chat.persistence_service import native_history_owner_key
from tldw_Server_API.app.core.DB_Management.chacha.operation_scope import ChaChaOperationMiddleware
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB, InputError

pytestmark = pytest.mark.integration


@pytest.fixture(params=["sqlite", "postgres"])
def message_api(tmp_path, monkeypatch, request):
    options = {"backend": request.getfixturevalue("pg_restricted_backend")} if request.param == "postgres" else {}
    db = CharactersRAGDB(str(tmp_path / "messages.db"), client_id="1", owner_user_id="1", **options)
    db.upsert_workspace("workspace-a", "Research")
    chat = db.add_conversation({"title": "Research", "scope_type": "workspace", "workspace_id": "workspace-a"})
    app = FastAPI()
    app.add_middleware(ChaChaOperationMiddleware)
    app.include_router(character_messages.router, prefix="/api/v1")
    app.dependency_overrides[get_chacha_db_for_user] = lambda: db
    app.dependency_overrides[character_messages.get_request_user] = lambda: SimpleNamespace(id=1)
    monkeypatch.setattr(character_messages, "_active_message_sync_service", lambda *_: None)
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Chat.conversation_enrichment.schedule_auto_tagging", lambda *a, **kw: None
    )
    with TestClient(app) as client:
        yield client, db, chat, {"scope_type": "workspace", "workspace_id": "workspace-a"}
    db.close_all_connections()


def citation_payload():
    return {
        "role": "assistant",
        "content": "Selene [1].",
        "metadata_extra": {
            "workspace_rag_sources_v1": {
                "origin": "knowledge_qa",
                "sources": [
                    {
                        "originalId": 1,
                        "mediaId": 1,
                        "title": "Rowan",
                        "type": "text",
                        "sourceType": "text",
                        "excerpt": "The telescope is named Selene.",
                    }
                ],
            }
        },
    }


def test_workspace_citations_survive_real_create_and_reload(message_api):
    client, db, chat, scope = message_api
    payload = citation_payload()
    response = client.post(f"/api/v1/chats/{chat}/messages", params=scope, json=payload)
    assert response.status_code == 201  # nosec B101 -- pytest behavior assertion.
    messages = client.get(
        f"/api/v1/chats/{chat}/messages", params={**scope, "include_metadata": True, "render_placeholders": False}
    )
    assert messages.status_code == 200  # nosec B101 -- pytest behavior assertion.
    assert messages.json()["messages"][0]["metadata_extra"] == payload["metadata_extra"]  # nosec B101 -- pytest behavior assertion.
    assert db.count_messages_for_conversation(chat) == 1  # nosec B101 -- pytest behavior assertion.


@pytest.mark.parametrize("failure", ["exception", "false"])
def test_metadata_failure_rolls_back_message(message_api, monkeypatch, failure):
    client, db, chat, scope = message_api
    before = history_snapshot(db, chat).fences.history

    def fail(*a, **kw):
        if failure == "false":
            return False
        raise InputError("injected metadata failure")

    monkeypatch.setattr(db, "set_message_metadata_extra", fail)
    response = client.post(f"/api/v1/chats/{chat}/messages", params=scope, json=citation_payload())
    assert response.status_code == 400  # nosec B101 -- pytest behavior assertion.
    assert db.count_messages_for_conversation(chat) == 0  # nosec B101 -- pytest behavior assertion.
    assert history_snapshot(db, chat).fences.history == before  # nosec B101 -- pytest behavior assertion.


@pytest.mark.parametrize(
    "mutation",
    [
        "empty",
        "many",
        "bad_id",
        "bool_id",
        "url",
        "text_limit",
        "encoded_limit",
        "arbitrary_source",
        "wrong_origin",
        "user",
        "system",
        "long_url",
        "null_id",
        "bad_port",
        "bad_host",
    ],
)
def test_rejects_invalid_citations_without_writing(message_api, mutation):
    client, db, chat, scope = message_api
    payload = deepcopy(citation_payload())
    extra = payload["metadata_extra"]
    provenance = extra["workspace_rag_sources_v1"]
    source = provenance["sources"][0]
    if mutation == "empty":
        provenance["sources"] = []
    elif mutation == "many":
        provenance["sources"] = [source] * 65
    elif mutation == "bad_id":
        source["mediaId"] = -1
    elif mutation == "bool_id":
        source["mediaId"] = True
    elif mutation == "url":
        source["url"] = "javascript:alert(1)"
    elif mutation == "text_limit":
        source["excerpt"] = "x" * 100001
    elif mutation == "encoded_limit":
        source["excerpt"] = "界" * 40000
    elif mutation == "arbitrary_source":
        source["arbitrary"] = "value"
    elif mutation == "wrong_origin":
        provenance["origin"] = "reviewed_sources"
    elif mutation in {"user", "system"}:
        payload["role"] = mutation
    elif mutation == "long_url":
        source["url"] = "https://example.com/" + "x" * 4096
    elif mutation == "null_id":
        source["mediaId"] = None
    elif mutation == "bad_port":
        source["url"] = "http://example.com:not-a-port"
    elif mutation == "bad_host":
        source["url"] = "http://bad host/"
    response = client.post(f"/api/v1/chats/{chat}/messages", params=scope, json=payload)
    assert response.status_code == 422  # nosec B101 -- pytest behavior assertion.
    assert db.count_messages_for_conversation(chat) == 0  # nosec B101 -- pytest behavior assertion.


@pytest.mark.parametrize("refusal", ["global", "foreign_workspace", "foreign_user", "sync"])
def test_citations_do_not_bypass_scope_or_sync(message_api, monkeypatch, refusal):
    client, db, chat, scope = message_api
    if refusal == "global":
        chat = db.add_conversation({"title": "Global"})
        scope = {}
    elif refusal == "foreign_workspace":
        scope = {**scope, "workspace_id": "other"}
    elif refusal == "foreign_user":
        client.app.dependency_overrides[character_messages.get_request_user] = lambda: SimpleNamespace(id=2)
    else:
        monkeypatch.setattr(character_messages, "_active_message_sync_service", lambda *_: object())
    response = client.post(f"/api/v1/chats/{chat}/messages", params=scope, json=citation_payload())
    assert response.status_code == {"global": 400, "foreign_workspace": 404, "foreign_user": 403, "sync": 400}[refusal]  # nosec B101 -- pytest behavior assertion.
    assert db.count_messages_for_conversation(chat) == 0  # nosec B101 -- pytest behavior assertion.


def test_plain_messages_remain_supported(message_api):
    client, db, chat, scope = message_api
    response = client.post(f"/api/v1/chats/{chat}/messages", params=scope, json={"role": "user", "content": "Hello"})
    assert response.status_code == 201  # nosec B101 -- pytest behavior assertion.
    assert db.get_message_metadata(response.json()["id"]) is None  # nosec B101 -- pytest behavior assertion.


def history_snapshot(db, chat):
    request = Request(
        {"type": "http", "scheme": "http", "server": ("testserver", 80), "headers": [], "path": "/", "root_path": ""}
    )
    owner = native_history_owner_key(request, 1)
    return db.get_conversation_history_snapshot(chat, owner_client_id="1", owner_key=owner)


def test_versioned_citations_replay_and_changed_intent_conflict(message_api):
    client, db, chat, scope = message_api
    snapshot = snapshot_to_wire(history_snapshot(db, chat))
    selection = resolve_history_selection(
        snapshot,
        {
            "owner_key": snapshot["owner_key"],
            "conversation_id": chat,
            "interpretation": {"kind": "parent_graph_v1"},
            "cursor": {"kind": "empty"},
            "selection_revision": 1,
        },
        "send",
        "workspace",
    )["selection"]
    accepted = client.post(
        f"/api/v1/chats/{chat}/messages",
        params=scope,
        json={
            "id": "workspace-input",
            "role": "user",
            "content": "Which telescope?",
            "tldw_history_selection_v1": selection,
        },
    )
    assert accepted.status_code == 201  # nosec B101 -- pytest behavior assertion.
    admission = accepted.json()["tldw_history_admission_v1"]
    payload = {
        **citation_payload(),
        "id": "workspace-answer",
        "tldw_history_admission_v1": {
            key: admission[key]
            for key in (
                "version",
                "owner_key",
                "conversation_id",
                "input_message_id",
                "input_message_revision",
                "selection_digest",
            )
        },
    }
    for _ in range(2):
        saved = client.post(f"/api/v1/chats/{chat}/messages", params=scope, json=payload)
        assert saved.status_code == 201  # nosec B101 -- pytest behavior assertion.
    assert db.count_messages_for_conversation(chat) == 2  # nosec B101 -- pytest behavior assertion.
    assert db.get_message_metadata("workspace-answer")["extra"] == payload["metadata_extra"]  # nosec B101 -- pytest behavior assertion.
    changed = deepcopy(payload)
    changed["metadata_extra"]["workspace_rag_sources_v1"]["sources"][0]["excerpt"] = "Changed citation."
    conflict = client.post(f"/api/v1/chats/{chat}/messages", params=scope, json=changed)
    assert conflict.status_code == 409  # nosec B101 -- pytest behavior assertion.
    assert db.get_message_metadata("workspace-answer")["extra"] == payload["metadata_extra"]  # nosec B101 -- pytest behavior assertion.
    assert db.count_messages_for_conversation(chat) == 2  # nosec B101 -- pytest behavior assertion.


@pytest.mark.parametrize("role", ["user", "assistant", "system"])
def test_legacy_metadata_does_not_prevent_text_save(message_api, role):
    client, db, _chat, _scope = message_api
    chat = db.add_conversation({"title": "Legacy SDK message"})
    response = client.post(
        f"/api/v1/chats/{chat}/messages",
        json={
            "role": role,
            "content": "Existing SDK text",
            "metadata_extra": {
                "dynamic_ui": {"renderer": "openui", "status": "ready"},
                "generation_status": "complete",
            },
        },
    )
    assert response.status_code == 201  # nosec B101 -- pytest backward compatibility assertion.
    saved = db.get_message_by_id(response.json()["id"])
    assert saved["content"] == "Existing SDK text"  # nosec B101 -- pytest content preservation assertion.
    assert db.get_message_metadata(saved["id"]) is None  # nosec B101 -- untrusted legacy metadata must remain unstored.


@pytest.mark.parametrize("sibling", ["arbitrary", "generation_status", "dynamic_ui"])
def test_unsupported_metadata_siblings_are_not_persisted(message_api, sibling):
    client, db, chat, scope = message_api
    payload = citation_payload()
    expected = deepcopy(payload["metadata_extra"])
    payload["metadata_extra"][sibling] = "untrusted client value"
    response = client.post(f"/api/v1/chats/{chat}/messages", params=scope, json=payload)
    assert response.status_code == 201  # nosec B101 -- pytest compatible save assertion.
    assert db.get_message_by_id(response.json()["id"])["content"] == payload["content"]  # nosec B101 -- pytest text preservation assertion.
    assert db.get_message_metadata(response.json()["id"])["extra"] == expected  # nosec B101 -- only validated references may be retained.
