"""Regression tests for gated native turns in linked document chats."""

from uuid import uuid4

from tldw_Server_API.app.api.v1.endpoints import character_chat_sessions, character_messages
from tldw_Server_API.app.core.Chat.history_selection import resolve_history_selection
from tldw_Server_API.tests.Chat_NEW.integration.test_document_chat_links import (
    _attach,
    _create,
)
from tldw_Server_API.tests.Chat_NEW.integration.test_document_chat_links import (
    document_api as document_api,
)


def _linked_chat(document_api):
    client, db, media, headers = document_api
    media_uuid = media[1]["uuid"]
    created = _create(client, headers, media_uuid).json()
    cid = created["id"]
    assert _attach(client, headers, cid, created["document_chat_record"], media_uuid).status_code == 204
    return client, db, media, headers, cid, {"document_media_id": 1, "document_media_uuid": media_uuid}


def _capture(client, headers, cid, params, mid=None):
    cursor = {"kind": "empty"} if mid is None else {"kind": "after_message", "message_id": mid}
    return client.post(f"/api/v1/chat/conversations/{cid}/history/selection", headers=headers, params=params, json={
        "purpose": "send", "view": {
            "view_session_id": "document-view", "conversation_id": cid, "cursor": cursor,
            "interpretation": {"kind": "parent_graph_v1"}, "selection_revision": 1,
        },
    })


def _send(client, headers, cid, params, selection):
    return client.post(f"/api/v1/chats/{cid}/messages", headers=headers, params=params, json={
        "id": str(uuid4()), "role": "user", "content": "What does the author recommend?",
        "tldw_history_selection_v1": selection, "tldw_history_branch": False,
    })


def test_document_admission_refuses_revoked_media(document_api):
    client, db, media, headers, cid, params = _linked_chat(document_api)
    captured = _capture(client, headers, cid, params).json()
    selected = resolve_history_selection(captured["snapshot"], captured["view"], "send", "")["selection"]
    original_count = db.count_messages_for_conversation(cid)
    media.pop(1)
    response = _send(client, headers, cid, params, selected)
    assert response.status_code == 404, response.text
    assert _capture(client, headers, cid, params).status_code == 404
    assert db.count_messages_for_conversation(cid) == original_count


def test_linked_chat_retains_authorized_read_access_with_sync_v2(document_api, monkeypatch):
    client, db, media, headers, cid, params = _linked_chat(document_api)
    monkeypatch.setattr(character_messages, "_active_message_sync_service", lambda *_: object())
    monkeypatch.setattr(character_chat_sessions, "_active_chat_sync_service", lambda *_: object())
    capture = _capture(client, headers, cid, params)
    assert capture.status_code == 200, capture.text
    captured = capture.json()
    selection = resolve_history_selection(captured["snapshot"], captured["view"], "send", "")["selection"]
    response = _send(client, headers, cid, params, selection)
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "sync_owner_unsupported"
    assert db.count_messages_for_conversation(cid) == 0
    new_id = str(uuid4())
    assert _create(client, headers, media[1]["uuid"], id=new_id).status_code == 409
    assert db.get_conversation_by_id(new_id) is None


def test_document_reply_round_trips_citation(document_api):
    client, db, media, headers, cid, params = _linked_chat(document_api)
    captured = _capture(client, headers, cid, params).json()
    selected = resolve_history_selection(captured["snapshot"], captured["view"], "send", "")["selection"]
    accepted = _send(client, headers, cid, params, selected)
    assert accepted.status_code == 201, accepted.text
    stale = _send(client, headers, cid, params, selected)
    assert stale.status_code == 409
    assert stale.json()["detail"]["code"] == "history_branch_changed"
    assert db.count_messages_for_conversation(cid) == 1

    admission = accepted.json()["tldw_history_admission_v1"]
    input_id = admission["input_message_id"]
    reference = {key: admission[key] for key in (
        "version", "owner_key", "conversation_id", "input_message_id", "input_message_revision", "selection_digest",
    )}
    citation = {
        "media_uuid": media[1]["uuid"], "title": "Original text", "excerpt": "An excerpt on the first page.", "page": 1,
    }
    body = {
        "id": str(uuid4()), "role": "assistant", "content": "The author recommends reviewing the excerpt.",
        "tldw_history_admission_v1": reference, "document_chat_sources_v1": [citation],
    }
    endpoint = f"/api/v1/chats/{cid}/messages"
    wrong_citation = {**citation, "media_uuid": media[2]["uuid"]}
    assert client.post(endpoint, headers=headers, params=params, json={
        **body, "document_chat_sources_v1": [wrong_citation],
    }).status_code == 422
    assert client.post(endpoint, headers=headers, json=body).status_code == 422
    response = client.post(endpoint, headers=headers, params=params, json=body)
    assert response.status_code == 201, response.text
    assert response.json()["parent_message_id"] == input_id
    reopened = _capture(client, headers, cid, params, response.json()["id"]).json()
    assert reopened["selected_content"][-1]["extra_metadata"]["document_chat_sources_v1"] == [citation]
