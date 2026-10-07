"""D7 P1 contract: ``tldw_history_branch`` opts an admission into the server leaf check.

``false`` means "must extend the latest message" and is refused with 409
``history_branch_changed`` when the selected tip already has a live child;
``true`` admits an explicit branch; an absent field keeps today's behaviour.
"""

import json

import pytest

from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.core.Chat.history_selection import resolve_history_selection
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB


@pytest.fixture
def history_api(credentialed_test_client, populated_chacha_db, auth_headers):
    """Use the authenticated owner for API writes while retaining seeded data."""
    client = credentialed_test_client
    db = CharactersRAGDB(db_path=populated_chacha_db.db_path_str, client_id="1", owner_user_id="1")
    client.app.dependency_overrides[get_chacha_db_for_user] = lambda: db
    cid = db.add_conversation({"character_id": None, "title": "P1 leaf check", "client_id": "1"})
    yield client, db, cid, auth_headers
    client.app.dependency_overrides.pop(get_chacha_db_for_user, None)
    db.close_connection()


def select(client, cid, headers, after=None, context="client"):
    cursor = {"kind": "after_message", "message_id": after} if after else {"kind": "empty"}
    response = client.post(f"/api/v1/chat/conversations/{cid}/history/selection", headers=headers, json={
        "purpose": "send", "view": {"view_session_id": "view", "conversation_id": cid,
        "interpretation": {"kind": "parent_graph_v1"}, "cursor": cursor, "selection_revision": 1}})
    assert response.status_code == 200, response.text
    body = response.json()
    return resolve_history_selection(body["snapshot"], body["view"], "send", context)["selection"]


def send(client, cid, headers, selection, mid, **extra):
    return client.post(f"/api/v1/chats/{cid}/messages", headers=headers, json={
        "id": mid, "role": "user", "content": f"input {mid}", "tldw_history_selection_v1": selection, **extra})


def reply(client, cid, headers, admission, mid):
    reference = {key: admission[key] for key in (
        "version", "owner_key", "conversation_id", "input_message_id", "input_message_revision", "selection_digest")}
    response = client.post(f"/api/v1/chats/{cid}/messages", headers=headers, json={
        "id": mid, "role": "assistant", "content": f"reply {mid}", "tldw_history_admission_v1": reference})
    assert response.status_code == 201, response.text


def stale_tip(client, cid, headers):
    """u1 -> a1; another tab then sends u2 -> a2 from a1. Returns the old a1 selection."""
    first = send(client, cid, headers, select(client, cid, headers), "u1")
    assert first.status_code == 201, first.text
    reply(client, cid, headers, first.json()["tldw_history_admission_v1"], "a1")
    tip = select(client, cid, headers, after="a1")
    other_tab = send(client, cid, headers, tip, "u2", tldw_history_branch=False)
    assert other_tab.status_code == 201, other_tab.text
    reply(client, cid, headers, other_tab.json()["tldw_history_admission_v1"], "a2")
    return tip


def assert_branch_changed(response, db, cid):
    assert response.status_code == 409, response.text
    assert response.json()["detail"] == {
        "status": "stale_selection",
        "code": "history_branch_changed",
        "conversation_id": cid,
        "parent_message_id": "a1",
        "leaf_ids": ["a2"],
        "history_version": db.get_conversation_by_id(cid)["history_version"],
    }


def test_message_admission_refuses_stale_tip_and_writes_nothing(history_api):
    client, db, cid, headers = history_api
    tip = stale_tip(client, cid, headers)
    before = db.count_messages_for_conversation(cid)
    response = send(client, cid, headers, tip, "u3", tldw_history_branch=False)
    assert_branch_changed(response, db, cid)
    assert db.count_messages_for_conversation(cid) == before
    assert db.get_message_by_id("u3") is None


def test_message_admission_current_explicit_absent_and_replay(history_api):
    client, db, cid, headers = history_api
    tip = stale_tip(client, cid, headers)
    # The other tab's own message replays idempotently although its tip moved on.
    replay = send(client, cid, headers, tip, "u2", tldw_history_branch=False)
    assert replay.status_code == 201, replay.text
    assert replay.json()["id"] == "u2"
    branched = send(client, cid, headers, tip, "branch", tldw_history_branch=True)
    assert branched.status_code == 201, branched.text
    assert branched.json()["parent_message_id"] == "a1"
    unchecked = send(client, cid, headers, tip, "legacy-client")
    assert unchecked.status_code == 201, unchecked.text
    assert unchecked.json()["parent_message_id"] == "a1"
    current = send(client, cid, headers, select(client, cid, headers, after="a2"), "u3", tldw_history_branch=False)
    assert current.status_code == 201, current.text
    assert current.json()["parent_message_id"] == "a2"


def test_message_admission_ignores_deleted_child(history_api):
    client, db, cid, headers = history_api
    first = send(client, cid, headers, select(client, cid, headers), "u1")
    reply(client, cid, headers, first.json()["tldw_history_admission_v1"], "a1")
    tip = select(client, cid, headers, after="a1")
    assert send(client, cid, headers, tip, "u2").status_code == 201
    db.soft_delete_message("u2", expected_version=1)
    response = send(client, cid, headers, tip, "u3", tldw_history_branch=False)
    assert response.status_code == 201, response.text
    assert response.json()["parent_message_id"] == "a1"


@pytest.mark.parametrize("with_selection", [False, True])
def test_message_branch_field_requires_selection_and_a_strict_boolean(history_api, with_selection):
    client, db, cid, headers = history_api
    if with_selection:
        response = send(client, cid, headers, select(client, cid, headers), "x", tldw_history_branch="false")
    else:
        response = client.post(f"/api/v1/chats/{cid}/messages", headers=headers, json={
            "role": "user", "content": "plain", "tldw_history_branch": False})
    assert response.status_code == 422, response.text
    assert db.count_messages_for_conversation(cid) == 0


def completion(client, cid, headers, selection, **extra):
    from tldw_Server_API.app.core.Chat.rate_limiter import initialize_rate_limiter
    initialize_rate_limiter()  # Several turns per test; the token budget is not under test.
    return client.post("/api/v1/chat/completions", headers=headers, json={
        "model": "gpt-4o-mini", "conversation_id": cid, "save_to_db": True,
        "messages": [{"role": "user", "content": "server turn"}], "tldw_history_selection_v1": selection, **extra})


@pytest.mark.parametrize("stream", [False, True])
def test_server_completion_refuses_stale_tip_before_provider_or_writes(history_api, monkeypatch, stream):
    from tldw_Server_API.app.api.v1.endpoints import chat as endpoint
    client, db, cid, headers = history_api
    tip = stale_tip(client, cid, headers)
    before = (db.count_messages_for_conversation(cid), db.get_conversation_settings(cid))

    def forbidden(*args, **kwargs):
        raise AssertionError("a refused admission must not reach the provider")

    monkeypatch.setattr(endpoint, "perform_chat_api_call", forbidden)
    response = completion(client, cid, headers, tip, tldw_history_branch=False, stream=stream)
    assert_branch_changed(response, db, cid)
    assert (db.count_messages_for_conversation(cid), db.get_conversation_settings(cid)) == before


def test_server_completion_current_explicit_and_absent(history_api, monkeypatch):
    from tldw_Server_API.app.core.Chat import streaming_utils
    monkeypatch.setattr(streaming_utils, "CHAT_STREAM_INCLUDE_METADATA", False)
    client, db, cid, headers = history_api
    stale_tip(client, cid, headers)
    for context, branch in (("explicit", {"tldw_history_branch": True}), ("absent", {})):
        response = completion(client, cid, headers, select(client, cid, headers, after="a1", context=context), **branch)
        assert response.status_code == 200, response.text
        admission = response.json()["tldw_history_admission_v1"]
        assert db.get_message_by_id(admission["input_message_id"])["parent_message_id"] == "a1"
    current = completion(client, cid, headers, select(client, cid, headers, after="a2", context="current"),
                         tldw_history_branch=False, stream=True)
    assert current.status_code == 200, current.text
    frames = [json.loads(line[6:]) for line in current.text.splitlines()
              if line.startswith("data: ") and line[6:] != "[DONE]"]
    admission = next(frame["tldw_history_admission_v1"] for frame in frames if "tldw_history_admission_v1" in frame)
    assert db.get_message_by_id(admission["input_message_id"])["parent_message_id"] == "a2"


def test_completion_branch_field_requires_selection(history_api):
    client, db, cid, headers = history_api
    response = client.post("/api/v1/chat/completions", headers=headers, json={
        "model": "gpt-4o-mini", "messages": [{"role": "user", "content": "plain"}], "tldw_history_branch": False})
    assert response.status_code == 422, response.text
    assert db.count_messages_for_conversation(cid) == 0
