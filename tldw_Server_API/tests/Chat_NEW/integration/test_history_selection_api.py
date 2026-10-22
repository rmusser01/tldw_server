"""Authenticated native history capture, admission and settlement routes."""
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.core.Chat.history_selection import resolve_history_selection
from tldw_Server_API.app.core.DB_Management.backends.base import BackendType
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.tests.DB_Management.test_history_selection_transactions import history_db as history_db


@pytest.fixture
def history_api(credentialed_test_client, populated_chacha_db, auth_headers):
    """Use the authenticated owner for API writes while retaining seeded data."""
    client = credentialed_test_client
    # Match request dependency ownership; existing seed attribution is unchanged.
    db = CharactersRAGDB(
        db_path=populated_chacha_db.db_path_str,
        client_id="1",
        owner_user_id="1",
    )
    client.app.dependency_overrides[get_chacha_db_for_user] = lambda: db
    cid = db.add_conversation({"character_id": None, "title": "H1 API", "client_id": "1"})
    yield client, db, cid, auth_headers
    client.app.dependency_overrides.pop(get_chacha_db_for_user, None)
    db.close_connection()


def capture(client, cid, headers, **overrides):
    return client.post(f"/api/v1/chat/conversations/{cid}/history/selection", headers=headers, json={
        "purpose": "send", "view": {"view_session_id": "view-one", "conversation_id": cid,
        "interpretation": {"kind": "parent_graph_v1"}, "cursor": {"kind": "empty"},
        "selection_revision": 1, **overrides}})


def test_regenerate_capture_capability_is_opt_in_and_ordinary_wire_is_unchanged(history_api):
    client, db, cid, headers = history_api
    question = db.add_message({"id": "capability-question", "conversation_id": cid,
                               "sender": "user", "content": "why?"})
    ordinary = capture(client, cid, headers, cursor={"kind": "after_message", "message_id": question})
    assert ordinary.status_code == 200, ordinary.text  # nosec B101 # pytest contract assertion
    assert "acknowledged_capability" not in ordinary.json()  # nosec B101 # pytest contract assertion

    view = ordinary.json()["view"]
    opted_in = client.post(f"/api/v1/chat/conversations/{cid}/history/selection", headers=headers, json={
        "purpose": "send", "view": view,
        "required_capability": "tldw_history_existing_input_v1"})
    assert opted_in.status_code == 200, opted_in.text  # nosec B101 # pytest contract assertion
    assert opted_in.json()["status"] == "captured"  # nosec B101 # pytest contract assertion
    assert opted_in.json()["acknowledged_capability"] == "tldw_history_existing_input_v1"  # nosec B101 # pytest contract assertion
    assert db.count_messages_for_conversation(cid) == 1  # nosec B101 # pytest contract assertion


def test_capture_bootstrap_and_generic_admission_before_assistant(history_api):
    client, db, cid, headers = history_api
    response = capture(client, cid, headers)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["snapshot"]["owner_key"] == body["view"]["owner_key"]
    selection = resolve_history_selection(body["snapshot"], body["view"], "send", "client-only")["selection"]
    accepted = client.post(f"/api/v1/chats/{cid}/messages", headers=headers, json={
        "id": "api-input", "role": "user", "content": "hello", "tldw_history_selection_v1": selection})
    assert accepted.status_code == 201, accepted.text
    admission = accepted.json()["tldw_history_admission_v1"]
    reference = {k: admission[k] for k in ("version", "owner_key", "conversation_id", "input_message_id",
                                         "input_message_revision", "selection_digest")}
    saved = client.post(f"/api/v1/chats/{cid}/messages", headers=headers, json={
        "id": "api-reply", "role": "assistant", "content": "hi", "tldw_history_admission_v1": reference})
    assert saved.status_code == 201, saved.text
    assert db.get_message_by_id("api-reply")["parent_message_id"] == "api-input"


@pytest.mark.parametrize("sync_enrolled", [False, True])
def test_opted_in_native_completion_capability_matches_original_owner(history_api, monkeypatch, sync_enrolled):
    from tldw_Server_API.app.api.v1.endpoints import character_messages

    client, db, cid, headers = history_api
    monkeypatch.setattr(
        character_messages, "_active_message_sync_service", lambda *_: object() if sync_enrolled else None
    )
    legacy = capture(client, cid, headers)
    response = client.post(
        f"/api/v1/chat/conversations/{cid}/history/selection?include_native_completion=true",
        headers=headers,
        json={"purpose": "send", "view": legacy.json()["view"]},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["native_completion_v1"] == {
        "policy": "selected_history_v1",
        "supported": not sync_enrolled,
        "storage_context_digest": body["storage_context_digest"],
    }
    assert {k: v for k, v in body.items() if k != "native_completion_v1"} == legacy.json()
    assert db.count_messages_for_conversation(cid) == 0


@pytest.mark.parametrize(
    "purpose,query",
    [("send", ""), ("send", "?include_native_completion=false"), ("fork", "?include_native_completion=true")],
)
def test_unrequested_or_fork_capture_retains_legacy_json(history_api, purpose, query):
    client, db, cid, headers = history_api
    db.add_message(
        {
            "id": "nullable-root",
            "conversation_id": cid,
            "sender": "user",
            "content": "Legacy nullable parent",
            "parent_message_id": None,
        }
    )
    view = capture(client, cid, headers).json()["view"]
    response = client.post(
        f"/api/v1/chat/conversations/{cid}/history/selection{query}",
        headers=headers,
        json={"purpose": purpose, "view": view},
    )
    assert response.status_code == 200, response.text
    assert "native_completion_v1" not in response.json()
    assert response.json()["snapshot"]["native_fork_context"] is not None
    assert response.json()["snapshot"]["nodes"][0]["parent_id"] is None
    assert db.count_messages_for_conversation(cid) == 1


@pytest.mark.parametrize("scope_type", ["global", "workspace"])
def test_native_completion_capability_uses_original_chat_scope(history_api, monkeypatch, scope_type):
    from tldw_Server_API.app.api.v1.endpoints import character_messages

    client, db, cid, headers = history_api
    query = "?include_native_completion=true"
    if scope_type == "workspace":
        db.upsert_workspace("completion-workspace", "Completion workspace")
        created = client.post(
            "/api/v1/chats/",
            headers=headers,
            json={"title": "Completion scope", "scope_type": "workspace", "workspace_id": "completion-workspace"},
        )
        assert created.status_code == 201, created.text
        cid = created.json()["id"]
        query += "&scope_type=workspace&workspace_id=completion-workspace"
    observed = []

    def active_sync(_user, scope):
        observed.append({"scope_type": scope.scope_type, "workspace_id": scope.workspace_id})
        return object() if scope.scope_type == "global" else None

    monkeypatch.setattr(character_messages, "_active_message_sync_service", active_sync)
    response = client.post(
        f"/api/v1/chat/conversations/{cid}/history/selection{query}",
        headers=headers,
        json={
            "purpose": "send",
            "view": {
                "view_session_id": "scope-view",
                "conversation_id": cid,
                "interpretation": {"kind": "parent_graph_v1"},
                "cursor": {"kind": "empty"},
                "selection_revision": 0,
            },
        },
    )
    assert response.status_code == 200, response.text
    assert response.json()["native_completion_v1"]["supported"] is (scope_type == "workspace")
    assert observed == [
        {"scope_type": scope_type, "workspace_id": "completion-workspace" if scope_type == "workspace" else None}
    ]
    assert db.count_messages_for_conversation(cid) == 0


def test_capability_does_not_authorize_completion_after_sync_enrollment(history_api, monkeypatch):
    from tldw_Server_API.app.api.v1.endpoints import character_messages

    client, db, cid, headers = history_api
    monkeypatch.setattr(character_messages, "_active_message_sync_service", lambda *_: None)
    view = capture(client, cid, headers).json()["view"]
    captured = client.post(
        f"/api/v1/chat/conversations/{cid}/history/selection?include_native_completion=true",
        headers=headers,
        json={"purpose": "send", "view": view},
    )
    assert captured.status_code == 200, captured.text
    body = captured.json()
    assert body["native_completion_v1"]["supported"] is True
    selection = resolve_history_selection(body["snapshot"], body["view"], "send", "enrollment-race")["selection"]
    monkeypatch.setattr(character_messages, "_active_message_sync_service", lambda *_: object())
    response = client.post(
        "/api/v1/chat/completions",
        headers=headers,
        json={
            "api_provider": "openai",
            "model": "gpt-4o-mini",
            "conversation_id": cid,
            "save_to_db": True,
            "messages": [{"role": "user", "content": "enrollment race"}],
            "tldw_history_selection_v1": selection,
        },
    )
    assert response.status_code == 409, response.text
    assert response.json()["detail"] == {"code": "sync_owner_unsupported", "status": "unsupported_history_capability"}
    assert db.count_messages_for_conversation(cid) == 0
    assert db.get_conversation_settings(cid) is None


@pytest.mark.parametrize("kind", ["foreign-owner", "foreign-workspace", "expected-user"])
def test_opted_in_capability_keeps_authenticated_capture_boundaries(history_api, kind):
    client, db, cid, headers = history_api
    view = capture(client, cid, headers).json()["view"]
    query = "?include_native_completion=true"
    request_headers = dict(headers)
    expected = 409
    if kind == "foreign-owner":
        view["owner_key"] = "server:foreign/account:1"
    elif kind == "foreign-workspace":
        query += "&scope_type=workspace&workspace_id=foreign"
        expected = 404
    else:
        request_headers["X-TLDW-Expected-User-ID"] = "other"
        expected = 412
    response = client.post(
        f"/api/v1/chat/conversations/{cid}/history/selection{query}",
        headers=request_headers,
        json={"purpose": "send", "view": view},
    )
    assert response.status_code == expected, response.text
    assert "native_completion_v1" not in response.json()
    assert db.count_messages_for_conversation(cid) == 0


@pytest.mark.parametrize("kind", ["policy", "boolean", "integer", "context", "extra", "fork"])
def test_native_completion_capability_schema_is_strict_and_capture_bound(history_api, kind):
    from pydantic import ValidationError

    from tldw_Server_API.app.api.v1.schemas.history_selection_schemas import CapturedHistoryV1

    client, _db, cid, headers = history_api
    body = capture(client, cid, headers).json()
    capability = {
        "policy": "selected_history_v1",
        "supported": True,
        "storage_context_digest": body["storage_context_digest"],
    }
    if kind == "policy":
        capability["policy"] = "future_v2"
    elif kind == "boolean":
        capability["supported"] = "true"
    elif kind == "integer":
        capability["supported"] = 1
    elif kind == "context":
        capability["storage_context_digest"] = "foreign"
    elif kind == "extra":
        capability["owner_override"] = "foreign"
    else:
        body["purpose"] = "fork"
    with pytest.raises(ValidationError):
        CapturedHistoryV1.model_validate({**body, "native_completion_v1": capability})


def test_existing_question_regenerate_keeps_question_and_replays_only_exact_answer(history_api):
    client, db, cid, headers = history_api
    ancestor = db.add_message({"id": "ancestor-question", "conversation_id": cid, "sender": "user", "content": "first?"})
    ancestor_answer = db.add_message({"id": "ancestor-answer", "conversation_id": cid, "sender": "assistant",
                                      "content": "first answer", "parent_message_id": ancestor})
    question = db.add_message({"id": "old-question", "conversation_id": cid, "sender": "user",
                               "content": "why?", "parent_message_id": ancestor_answer})
    original = db.add_message({"id": "old-answer", "conversation_id": cid, "sender": "assistant",
                               "content": "because", "parent_message_id": question})
    later = db.add_message({"id": "later-question", "conversation_id": cid, "sender": "user",
                            "content": "and then?", "parent_message_id": original})
    suffix = db.add_message({"id": "later-answer", "conversation_id": cid, "sender": "assistant",
                             "content": "later", "parent_message_id": later})
    before_question = db.get_message_by_id(question)
    before_answer = db.get_message_by_id(original)
    before_suffix = [db.get_message_by_id(later), db.get_message_by_id(suffix)]
    with db.transaction() as conn:
        original_authority = conn.execute(
            "SELECT history_admission_json FROM messages WHERE id = ?", (question,)
        ).fetchone()["history_admission_json"]
    body = capture(client, cid, headers, cursor={"kind": "after_message", "message_id": question}).json()
    assert body["status"] == "captured"  # nosec B101 # pytest contract assertion
    selection = resolve_history_selection(body["snapshot"], body["view"], "send", "prepared-request")["selection"]
    assert [row["id"] for row in selection["messages"]] == [ancestor, ancestor_answer, question]  # nosec B101 # pytest contract assertion
    payload = {"id": "new-answer", "role": "assistant", "content": "another reason",
               "tldw_history_existing_input_v1": selection}
    first = client.post(f"/api/v1/chats/{cid}/messages", headers=headers, json=payload)
    assert first.status_code == 201, first.text  # nosec B101 # pytest contract assertion
    assert first.json()["id"] == "new-answer"  # nosec B101 # pytest contract assertion
    assert db.get_message_by_id("new-answer")["parent_message_id"] == question  # nosec B101 # pytest contract assertion
    assert db.get_message_by_id(question) == before_question  # nosec B101 # pytest contract assertion
    assert db.get_message_by_id(original) == before_answer  # nosec B101 # pytest contract assertion
    assert [db.get_message_by_id(later), db.get_message_by_id(suffix)] == before_suffix  # nosec B101 # pytest contract assertion
    with db.transaction() as conn:
        assert conn.execute(  # nosec B101 # pytest contract assertion
            "SELECT history_admission_json FROM messages WHERE id = ?", (question,)
        ).fetchone()["history_admission_json"] == original_authority
    refreshed = capture(client, cid, headers, cursor={"kind": "after_message", "message_id": question}).json()
    assert refreshed["status"] == "captured"  # nosec B101 # pytest contract assertion
    assert refreshed["rows"][-1]["revision"] == body["rows"][-1]["revision"]  # nosec B101 # pytest contract assertion
    assert db.count_messages_for_conversation(cid) == 7  # nosec B101 # pytest contract assertion

    # A changed retained ancestor makes the original selection stale for a
    # first write, while exact replay of an already committed ID still works.
    db.update_message(ancestor, {"content": "changed first question"}, expected_version=1)
    changed_path = capture(client, cid, headers, cursor={"kind": "after_message", "message_id": question}).json()
    assert changed_path["rows"][-1]["revision"] == body["rows"][-1]["revision"]  # nosec B101 # pytest contract assertion
    stale_new = client.post(f"/api/v1/chats/{cid}/messages", headers=headers,
                            json={**payload, "id": "new-after-stale-ancestor"})
    assert stale_new.status_code == 409, stale_new.text  # nosec B101 # pytest contract assertion
    replay = client.post(f"/api/v1/chats/{cid}/messages", headers=headers, json=payload)
    assert replay.status_code == 201, replay.text  # nosec B101 # pytest contract assertion
    assert replay.json()["id"] == "new-answer"  # nosec B101 # pytest contract assertion
    assert db.count_messages_for_conversation(cid) == 7  # nosec B101 # pytest contract assertion
    conflict = client.post(f"/api/v1/chats/{cid}/messages", headers=headers,
                           json={**payload, "content": "changed answer"})
    assert conflict.status_code == 409, conflict.text  # nosec B101 # pytest contract assertion
    assert db.get_message_by_id("new-answer")["content"] == "another reason"  # nosec B101 # pytest contract assertion
    db.update_message("new-answer", {"content": "edited later"}, expected_version=1)
    edited_replay = client.post(f"/api/v1/chats/{cid}/messages", headers=headers, json=payload)
    assert edited_replay.status_code == 409, edited_replay.text  # nosec B101 # pytest contract assertion


def test_existing_question_regenerate_rejects_stale_or_foreign_selection_without_write(history_api):
    client, db, cid, headers = history_api
    question = db.add_message({"id": "stale-question", "conversation_id": cid, "sender": "user", "content": "why?"})
    body = capture(client, cid, headers, cursor={"kind": "after_message", "message_id": question}).json()
    selection = resolve_history_selection(body["snapshot"], body["view"], "send", "prepared-request")["selection"]
    payload = {"id": "rejected-answer", "role": "assistant", "content": "answer",
               "tldw_history_existing_input_v1": selection}
    foreign = client.post(f"/api/v1/chats/{cid}/messages", headers=headers,
                          json={**payload, "tldw_history_existing_input_v1": {**selection, "owner_key": "foreign"}})
    assert foreign.status_code == 409, foreign.text  # nosec B101 # pytest contract assertion
    db.update_message(question, {"content": "edited?"}, expected_version=1)
    stale = client.post(f"/api/v1/chats/{cid}/messages", headers=headers, json=payload)
    assert stale.status_code == 409, stale.text  # nosec B101 # pytest contract assertion
    assert db.count_messages_for_conversation(cid) == 1  # nosec B101 # pytest contract assertion


def test_existing_question_regenerate_rejects_wrong_role_cursor_and_parent(history_api):
    client, db, cid, headers = history_api
    db.add_message({"id": "another-question", "conversation_id": cid,
                    "sender": "user", "content": "another question"})
    question = db.add_message({"id": "guard-question", "conversation_id": cid,
                               "sender": "user", "content": "why?",
                               "parent_message_id": "another-question"})
    selected = capture(client, cid, headers,
                       cursor={"kind": "after_message", "message_id": question}).json()
    selection = resolve_history_selection(
        selected["snapshot"], selected["view"], "send", "prepared-guard"
    )["selection"]
    payload = {"id": "guard-answer", "role": "assistant", "content": "because",
               "tldw_history_existing_input_v1": selection}
    wrong_role = client.post(f"/api/v1/chats/{cid}/messages", headers=headers,
                             json={**payload, "role": "user"})
    assert wrong_role.status_code == 422, wrong_role.text  # nosec B101 # pytest contract assertion
    competing_contract = client.post(f"/api/v1/chats/{cid}/messages", headers=headers,
                                     json={**payload, "tldw_history_admission_v1": {
                                         "version": 1, "owner_key": selection["owner_key"],
                                         "conversation_id": cid, "input_message_id": question,
                                         "input_message_revision": selection["messages"][-1]["revision"],
                                         "selection_digest": selection["selection_digest"]}})
    assert competing_contract.status_code == 422, competing_contract.text  # nosec B101 # pytest contract assertion
    wrong_parent = client.post(f"/api/v1/chats/{cid}/messages", headers=headers,
                               json={**payload, "parent_message_id": "another-question"})
    assert wrong_parent.status_code == 409, wrong_parent.text  # nosec B101 # pytest contract assertion
    wrong_cursor = client.post(f"/api/v1/chats/{cid}/messages", headers=headers,
                               json={**payload, "tldw_history_existing_input_v1": {
                                   **selection, "cursor": {"kind": "empty"}}})
    assert wrong_cursor.status_code == 409, wrong_cursor.text  # nosec B101 # pytest contract assertion
    assert db.count_messages_for_conversation(cid) == 2  # nosec B101 # pytest contract assertion

    answer = db.add_message({"id": "guard-original", "conversation_id": cid,
                             "sender": "assistant", "content": "original", "parent_message_id": question})
    selected_answer = capture(client, cid, headers,
                              cursor={"kind": "after_message", "message_id": answer}).json()
    answer_selection = resolve_history_selection(
        selected_answer["snapshot"], selected_answer["view"], "send", "prepared-answer"
    )["selection"]
    wrong_terminal_role = client.post(f"/api/v1/chats/{cid}/messages", headers=headers,
                                      json={**payload, "tldw_history_existing_input_v1": answer_selection})
    assert wrong_terminal_role.status_code == 409, wrong_terminal_role.text  # nosec B101 # pytest contract assertion
    assert db.count_messages_for_conversation(cid) == 3  # nosec B101 # pytest contract assertion


def test_existing_question_regenerate_refuses_wrong_workspace_scope(history_api):
    client, db, cid, headers = history_api
    question = db.add_message({"id": "workspace-question", "conversation_id": cid,
                               "sender": "user", "content": "why?"})
    body = capture(client, cid, headers, cursor={"kind": "after_message", "message_id": question}).json()
    selection = resolve_history_selection(body["snapshot"], body["view"], "send", "prepared-workspace")["selection"]
    response = client.post(f"/api/v1/chats/{cid}/messages?scope_type=workspace&workspace_id=other",
                           headers=headers, json={"id": "wrong-workspace-answer", "role": "assistant",
                                                  "content": "must not save", "tldw_history_existing_input_v1": selection})
    assert response.status_code in (403, 404, 409), response.text  # nosec B101 # pytest contract assertion
    assert db.count_messages_for_conversation(cid) == 1  # nosec B101 # pytest contract assertion


def test_existing_question_regenerate_preserves_reviewed_legacy_question(history_api):
    client, db, cid, headers = history_api
    first = db.add_message({"id": "legacy-first", "conversation_id": cid, "sender": "user", "content": "one"})
    question = db.add_message({"id": "legacy-question", "conversation_id": cid, "sender": "user", "content": "two"})
    unreviewed = capture(client, cid, headers).json()
    assert unreviewed["status"] == "legacy_review_required"  # nosec B101 # pytest contract assertion
    source = unreviewed["snapshot"]
    confirmation = {"version": 1, "projection_id": "regenerate-review", "owner_key": source["owner_key"],
                    "conversation_id": cid, "source_digest": source["source_digest"], "fences": source["fences"],
                    "source_members": [{"id": row["id"], "revision": row["revision"]} for row in source["nodes"]],
                    "ordered_path_ids": [first, question],
                    "cursor": {"kind": "after_message", "message_id": question}, "selection_revision": 2}
    reviewed = client.post(f"/api/v1/chat/conversations/{cid}/history/legacy-projection",
                           headers=headers, json={"confirmation": confirmation})
    assert reviewed.status_code == 200, reviewed.text  # nosec B101 # pytest contract assertion
    body = capture(client, cid, headers,
                   interpretation={"kind": "legacy_linear_v1", "projection_id": "regenerate-review"},
                   cursor={"kind": "after_message", "message_id": question}).json()
    selection = resolve_history_selection(body["snapshot"], body["view"], "send", "prepared-legacy")["selection"]
    before = db.get_message_by_id(question)
    response = client.post(f"/api/v1/chats/{cid}/messages", headers=headers, json={
        "id": "legacy-alternative", "role": "assistant", "content": "alternative",
        "tldw_history_existing_input_v1": selection})
    assert response.status_code == 201, response.text  # nosec B101 # pytest contract assertion
    assert response.json()["id"] == "legacy-alternative"  # nosec B101 # pytest contract assertion
    assert db.get_message_by_id(question) == before  # nosec B101 # pytest contract assertion
    assert db.get_message_by_id("legacy-alternative")["parent_message_id"] == question  # nosec B101 # pytest contract assertion
    assert db.count_messages_for_conversation(cid) == 3  # nosec B101 # pytest contract assertion
    followed = capture(client, cid, headers,
                       interpretation={"kind": "legacy_linear_v1", "projection_id": "regenerate-review"},
                       cursor={"kind": "after_message", "message_id": "legacy-alternative"}).json()
    assert followed["status"] == "captured"  # nosec B101 # pytest contract assertion
    assert [row["id"] for row in followed["rows"]] == [first, question, "legacy-alternative"]  # nosec B101 # pytest contract assertion


def test_existing_question_regenerate_rolls_back_when_protected_proof_fails(history_api, monkeypatch):
    client, db, cid, headers = history_api
    question = db.add_message({"id": "rollback-question", "conversation_id": cid, "sender": "user", "content": "why?"})
    body = capture(client, cid, headers, cursor={"kind": "after_message", "message_id": question}).json()
    selection = resolve_history_selection(body["snapshot"], body["view"], "send", "prepared-request")["selection"]
    from tldw_Server_API.app.core.Chat.history_selection import HistorySelectionError

    def reject_proof(*_args, **_kwargs):
        raise HistorySelectionError("proof_write_failed")

    monkeypatch.setattr(db.message_store, "_write_history_authority", reject_proof)
    response = client.post(f"/api/v1/chats/{cid}/messages", headers=headers, json={
        "id": "rollback-answer", "role": "assistant", "content": "answer",
        "tldw_history_existing_input_v1": selection})
    assert response.status_code == 409, response.text  # nosec B101 # pytest contract assertion
    assert db.count_messages_for_conversation(cid) == 1  # nosec B101 # pytest contract assertion


def test_capture_foreign_namespace_and_workspace_have_no_mutations(history_api):
    client, db, cid, headers = history_api
    response = capture(client, cid, headers, owner_key="server:foreign/account:1")
    assert response.status_code == 409
    response = client.post(f"/api/v1/chat/conversations/{cid}/history/selection?scope_type=workspace&workspace_id=other",
        headers=headers, json={"purpose": "send", "view": {"view_session_id": "v", "conversation_id": cid,
        "interpretation": {"kind": "parent_graph_v1"}, "cursor": {"kind": "empty"}, "selection_revision": 0}})
    assert response.status_code == 404
    assert db.count_messages_for_conversation(cid) == 0


def test_legacy_review_returns_complete_bound_source(history_api):
    client, db, cid, headers = history_api
    for mid in ("one", "two"):
        db.add_message({"id": mid, "conversation_id": cid, "sender": "user", "content": mid})
    response = capture(client, cid, headers)
    assert response.status_code == 200, response.text
    assert response.json()["status"] == "legacy_review_required"
    assert len(response.json()["snapshot"]["nodes"]) == 2
    snapshot = response.json()["snapshot"]
    confirmation = {"version": 1, "projection_id": "reviewed-api", "owner_key": snapshot["owner_key"],
        "conversation_id": cid, "source_digest": snapshot["source_digest"], "fences": snapshot["fences"],
        "source_members": [{"id": row["id"], "revision": row["revision"]} for row in snapshot["nodes"]],
        "ordered_path_ids": ["two", "one"], "cursor": {"kind": "after_message", "message_id": "one"}, "selection_revision": 2}
    confirmed = client.post(f"/api/v1/chat/conversations/{cid}/history/legacy-projection", headers=headers,
        json={"confirmation": confirmation})
    assert confirmed.status_code == 200, confirmed.text
    reviewed = capture(client, cid, headers, interpretation={"kind": "legacy_linear_v1", "projection_id": "reviewed-api"},
        cursor={"kind": "after_message", "message_id": "one"})
    assert reviewed.status_code == 200, reviewed.text
    assert [row["id"] for row in reviewed.json()["selected_content"]] == ["two", "one"]
    selection = resolve_history_selection(
        reviewed.json()["snapshot"], reviewed.json()["view"], "send", "legacy-api-retry"
    )["selection"]
    accepted = client.post(
        f"/api/v1/chats/{cid}/messages",
        headers=headers,
        json={
            "id": "legacy-api-input",
            "role": "user",
            "content": "new question",
            "tldw_history_selection_v1": selection,
        },
    )
    assert accepted.status_code == 201, accepted.text
    admission = accepted.json()["tldw_history_admission_v1"]
    reference = {
        key: admission[key]
        for key in (
            "version",
            "owner_key",
            "conversation_id",
            "input_message_id",
            "input_message_revision",
            "selection_digest",
        )
    }
    db.update_message("two", {"content": "old source changed"}, expected_version=1)
    payload = {
        "id": "legacy-api-result",
        "role": "assistant",
        "content": "late reply",
        "tldw_history_admission_v1": reference,
    }
    for _ in range(2):
        reply = client.post(f"/api/v1/chats/{cid}/messages", headers=headers, json=payload)
        assert reply.status_code == 201, reply.text
    assert db.get_message_by_id("legacy-api-result")["parent_message_id"] == "legacy-api-input"
    assert db.count_messages_for_conversation(cid) == 4


def test_character_settlement_uses_owner_admission_and_rejects_forged_retry(history_api, monkeypatch):
    client, db, cid, headers = history_api
    body = capture(client, cid, headers).json()
    selection = resolve_history_selection(body["snapshot"], body["view"], "send", "client-only")["selection"]
    accepted = client.post(f"/api/v1/chats/{cid}/messages", headers=headers, json={
        "id": "character-input", "role": "user", "content": "hello", "tldw_history_selection_v1": selection}).json()
    admission = accepted["tldw_history_admission_v1"]
    reference = {k: admission[k] for k in ("version", "owner_key", "conversation_id", "input_message_id",
                                         "input_message_revision", "selection_digest")}
    payload = {"assistant_message_id": "character-result", "assistant_content": "hi", "tldw_history_admission_v1": reference}
    saved = client.post(f"/api/v1/chats/{cid}/completions/persist", headers=headers, json=payload)
    assert saved.status_code == 200, saved.text
    assert db.get_message_by_id("character-result")["parent_message_id"] == "character-input"
    from tldw_Server_API.app.api.v1.endpoints import character_chat_sessions
    def no_live_context(*args, **kwargs):
        raise AssertionError("versioned settlement must not use current branch/card/visual context")
    monkeypatch.setattr(character_chat_sessions, "_resolve_chat_turn_context", no_live_context)
    monkeypatch.setattr(character_chat_sessions, "_safe_resolve_character_visual_identity", no_live_context)
    db.add_message({"id": "later-branch", "conversation_id": cid, "sender": "user", "content": "later", "parent_message_id": None})
    replay = client.post(f"/api/v1/chats/{cid}/completions/persist", headers=headers, json=payload)
    assert replay.status_code == 200, replay.text
    conflict = client.post(f"/api/v1/chats/{cid}/completions/persist", headers=headers,
        json={**payload, "speaker_character_name": "Different speaker"})
    assert conflict.status_code == 409, conflict.text
    reference["selection_digest"] = "forged"
    rejected = client.post(f"/api/v1/chats/{cid}/completions/persist", headers=headers, json=payload)
    assert rejected.status_code == 409, rejected.text
    assert db.count_messages_for_conversation(cid) == 3


def test_character_history_admission_runs_database_work_off_event_loop(history_api, monkeypatch):
    import asyncio

    client, db, cid, headers = history_api
    body = capture(client, cid, headers).json()
    selection = resolve_history_selection(body["snapshot"], body["view"], "send", "client-only")["selection"]
    original_append = db.append_selected_history_input
    original_settle = db.settle_history_admission
    observed = []

    def off_loop(operation):
        def invoke(*args, **kwargs):
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                observed.append(operation)
            else:
                raise AssertionError(f"{operation} blocked the request event loop")
            return (original_append if operation == "append" else original_settle)(*args, **kwargs)
        return invoke

    monkeypatch.setattr(db, "append_selected_history_input", off_loop("append"))
    monkeypatch.setattr(db, "settle_history_admission", off_loop("settle"))
    accepted = client.post(f"/api/v1/chats/{cid}/messages", headers=headers, json={
        "id": "threaded-input", "role": "user", "content": "hello", "tldw_history_selection_v1": selection})
    assert accepted.status_code == 201, accepted.text
    admission = accepted.json()["tldw_history_admission_v1"]
    reference = {key: admission[key] for key in ("version", "owner_key", "conversation_id", "input_message_id", "input_message_revision", "selection_digest")}
    saved = client.post(f"/api/v1/chats/{cid}/messages", headers=headers, json={
        "id": "threaded-result", "role": "assistant", "content": "hi", "tldw_history_admission_v1": reference})
    assert saved.status_code == 201, saved.text
    assert observed == ["append", "settle"]


def test_versioned_server_completion_selected_input_and_consumed_replay(history_api):
    client, db, cid, headers = history_api
    body = capture(client, cid, headers).json()
    selection = resolve_history_selection(body["snapshot"], body["view"], "send", "inert-client-hash")["selection"]
    payload = {"model": "gpt-4o-mini", "conversation_id": cid, "save_to_db": True,
               "messages": [{"role": "user", "content": "new question"}], "tldw_history_selection_v1": selection}
    response = client.post("/api/v1/chat/completions", headers=headers, json=payload)
    assert response.status_code == 200, response.text
    admission = response.json()["tldw_history_admission_v1"]
    saved = db.get_message_by_id(response.json()["tldw_message_id"])
    assert saved["parent_message_id"] == admission["input_message_id"]
    assert db.get_message_by_id(admission["input_message_id"])["parent_message_id"] is None
    count = db.count_messages_for_conversation(cid)
    repeated = client.post("/api/v1/chat/completions", headers=headers, json=payload)
    assert repeated.status_code == 409, repeated.text
    assert db.count_messages_for_conversation(cid) == count


def test_versioned_stream_returns_accepted_input_binding(history_api, monkeypatch):
    from tldw_Server_API.app.core.Chat import streaming_utils
    monkeypatch.setattr(streaming_utils, "CHAT_STREAM_INCLUDE_METADATA", False)
    import json
    client, db, cid, headers = history_api
    body = capture(client, cid, headers).json()
    selection = resolve_history_selection(body["snapshot"], body["view"], "send", "stream")["selection"]
    response = client.post("/api/v1/chat/completions", headers=headers, json={
        "model": "gpt-4o-mini", "conversation_id": cid, "save_to_db": True, "stream": True,
        "messages": [{"role": "user", "content": "stream question"}], "tldw_history_selection_v1": selection})
    assert response.status_code == 200, response.text
    frames = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ") and line[6:] != "[DONE]"]
    admission = next(frame["tldw_history_admission_v1"] for frame in frames if "tldw_history_admission_v1" in frame)
    replies = [row for row in db.get_messages_for_conversation(cid) if row["sender"] == "assistant"]
    assert replies and all(row["parent_message_id"] == admission["input_message_id"] for row in replies)
    assert {frame["tldw_message_id"] for frame in frames if "tldw_message_id" in frame} == {row["id"] for row in replies}


def test_native_owner_namespace_uses_transport_account_and_base_path():
    from starlette.requests import Request

    from tldw_Server_API.app.core.Chat.persistence_service import native_history_owner_key
    def key(host="EXAMPLE.COM:443", root="/Api", user="1", headers=()):
        return native_history_owner_key(Request({"type": "http", "scheme": "https", "method": "POST",
            "path": root + "/chat", "root_path": root, "server": ("example.com", 443),
            "query_string": b"", "headers": [(b"host", host.encode()), *headers]}), user)
    assert key() == key(host="example.com", root="/Api/")
    assert key() != key(user="2")
    assert key() != key(root="/api")
    assert key() != key(host="other.example")
    assert key() == key(headers=[(b"x-forwarded-host", b"forged.example")])


@pytest.mark.parametrize("send_scoped", [False, True])
def test_versioned_workspace_scope_and_stale_selection_do_not_update_defaults(history_api, send_scoped):
    """Verified id-only and explicit Workspace turns preserve history admission guards."""
    client, db, _, headers = history_api
    db.upsert_workspace("scope-workspace", "Scope Workspace")
    cid = db.add_conversation({"character_id": None, "scope_type": "workspace", "workspace_id": "scope-workspace",
                               "client_id": "1", "title": "Neutral workspace"})
    query = "?scope_type=workspace&workspace_id=scope-workspace"
    body = client.post(f"/api/v1/chat/conversations/{cid}/history/selection{query}", headers=headers, json={"purpose": "send", "view": {
        "view_session_id": "workspace", "conversation_id": cid, "interpretation": {"kind": "parent_graph_v1"},
        "cursor": {"kind": "empty"}, "selection_revision": 1}}).json()
    selection = resolve_history_selection(body["snapshot"], body["view"], "send", "workspace-model")["selection"]
    payload = {"api_provider": "openai", "model": "gpt-4o-mini", "conversation_id": cid, "save_to_db": True,
        "messages": [{"role": "user", "content": "workspace"}], "tldw_history_selection_v1": selection}
    before = db.get_conversation_settings(cid)
    rejected = client.post("/api/v1/chat/completions?scope_type=global", headers=headers, json=payload)
    assert rejected.status_code == 404, rejected.text
    assert db.get_conversation_settings(cid) == before
    assert db.count_messages_for_conversation(cid) == 0
    # Owner context drift must reject before saving explicitly requested defaults.
    db.upsert_conversation_settings(cid, {"model": "changed"})
    changed = db.get_conversation_settings(cid)
    send_query = query if send_scoped else ""
    rejected = client.post("/api/v1/chat/completions" + send_query, headers=headers, json=payload)
    assert rejected.status_code == 409, rejected.text
    assert db.get_conversation_settings(cid) == changed
    assert db.count_messages_for_conversation(cid) == 0
    refreshed = client.post(f"/api/v1/chat/conversations/{cid}/history/selection{query}", headers=headers, json={"purpose": "send", "view": body["view"]}).json()
    payload["tldw_history_selection_v1"] = resolve_history_selection(refreshed["snapshot"], refreshed["view"], "send", "new-model")["selection"]
    from tldw_Server_API.app.core.Chat.rate_limiter import initialize_rate_limiter
    initialize_rate_limiter()
    success = client.post("/api/v1/chat/completions" + send_query, headers=headers, json=payload)
    assert success.status_code == 200, success.text
    admission = success.json()["tldw_history_admission_v1"]
    replies = [row for row in db.get_messages_for_conversation(cid) if row["sender"] == "assistant"]
    assert replies and replies[0]["parent_message_id"] == admission["input_message_id"]
    assert db.get_conversation_settings(cid)["settings"]["model"] == "gpt-4o-mini"


def test_versioned_character_override_rejects_before_input_or_defaults(history_api):
    client, db, cid, headers = history_api
    other = db.add_character_card({"name": "Different H1 character", "description": "Different", "client_id": "1"})
    body = capture(client, cid, headers).json()
    selection = resolve_history_selection(body["snapshot"], body["view"], "send", "override")["selection"]
    response = client.post("/api/v1/chat/completions", headers=headers, json={"model": "gpt-4o-mini",
        "conversation_id": cid, "character_id": str(other), "save_to_db": True,
        "messages": [{"role": "user", "content": "hello"}], "tldw_history_selection_v1": selection})
    assert response.status_code == 409, response.text
    assert db.count_messages_for_conversation(cid) == 0
    assert db.get_conversation_settings(cid) is None


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("kind", ["template", "world_books", "skills"])
def test_unsupported_saved_context_rejects_before_provider_or_admission(history_api, monkeypatch, kind, stream):
    from tldw_Server_API.app.api.v1.endpoints import chat as endpoint

    client, db, cid, headers = history_api
    body = capture(client, cid, headers).json()
    selection = resolve_history_selection(body["snapshot"], body["view"], "send", "context")["selection"]

    def forbidden(*args, **kwargs):
        raise AssertionError("unsupported context must not reach provider or append")

    monkeypatch.setattr(db, "append_selected_history_inputs", forbidden)
    monkeypatch.setattr(endpoint, "perform_chat_api_call", forbidden)
    request = {
        "model": "gpt-4o-mini",
        "conversation_id": cid,
        "save_to_db": True,
        "stream": stream,
        "messages": [{"role": "user", "content": "hello"}],
        "tldw_history_selection_v1": selection,
    }
    if kind == "template":
        request["prompt_template_name"] = "mutable-custom-template"
    elif kind == "skills":
        monkeypatch.setattr(db, "history_skills_may_be_visible", lambda: True)
    else:
        original = db.get_roleplay_resume_state

        def with_unsupported_context(*args, **kwargs):
            state = original(*args, **kwargs)
            state["settings"] = {"conversationContext": {"world_book_ids": [1]}}
            return state

        monkeypatch.setattr(db, "get_roleplay_resume_state", with_unsupported_context)
    response = client.post("/api/v1/chat/completions", headers=headers, json=request)
    assert response.status_code == 409, response.text
    assert response.json()["detail"] == {
        "status": "unsupported_history_capability",
        "code": "unsupported_history_context_"
        + {"template": "prompt_template", "world_books": "saved_settings", "skills": "skills_registry"}[kind],
    }
    assert db.count_messages_for_conversation(cid) == 0


def test_selected_history_rejects_unresolved_skill_directory_before_admission(history_api, monkeypatch):
    from tldw_Server_API.app.api.v1.endpoints import chat as endpoint

    client, db, cid, headers = history_api
    body = capture(client, cid, headers).json()
    selection = resolve_history_selection(body["snapshot"], body["view"], "send", "skills-path")["selection"]

    class UnavailableChatPaths(endpoint.DatabasePaths):
        @staticmethod
        def get_user_base_directory(_user_id):
            raise OSError("user directory unavailable")

    monkeypatch.setattr(endpoint, "DatabasePaths", UnavailableChatPaths)
    response = client.post("/api/v1/chat/completions", headers=headers, json={
        "model": "gpt-4o-mini", "conversation_id": cid, "save_to_db": True,
        "messages": [{"role": "user", "content": "hello"}],
        "tldw_history_selection_v1": selection,
    })
    assert response.status_code == 409, response.text
    assert response.json()["detail"] == {
        "status": "unsupported_history_capability",
        "code": "unsupported_history_context_skills_unresolved",
    }
    assert db.count_messages_for_conversation(cid) == 0


def test_sync_owner_captures_but_server_settled_completion_is_rejected_before_mutation(history_api, monkeypatch):
    """A Sync v2 owner can capture; only the server-settled turn is still refused.

    Client-managed admission and settlement for a Sync v2 owner are covered in
    tests/Sync/test_sync_v2_native_history_capture.py.
    """
    from tldw_Server_API.app.api.v1.endpoints import character_messages
    client, db, cid, headers = history_api
    monkeypatch.setattr(character_messages, "_active_message_sync_service", lambda *_: object())
    captured = capture(client, cid, headers)
    assert captured.status_code == 200, captured.text
    assert captured.json()["status"] == "captured"
    body = captured.json()
    selection = resolve_history_selection(body["snapshot"], body["view"], "send", "sync")["selection"]
    response = client.post("/api/v1/chat/completions", headers=headers, json={"api_provider": "openai", "model": "gpt-4o-mini",
        "conversation_id": cid, "save_to_db": True, "messages": [{"role": "user", "content": "hello"}],
        "tldw_history_selection_v1": selection})
    assert response.status_code == 409, response.text
    assert response.json()["detail"] == {"code": "sync_owner_unsupported", "status": "unsupported_history_capability"}
    assert db.count_messages_for_conversation(cid) == 0
    assert db.get_conversation_settings(cid) is None


def test_before_first_admission_has_explicit_null_parent(history_api):
    client, db, cid, headers = history_api
    db.add_message({"id": "earlier", "conversation_id": cid, "sender": "user", "content": "earlier", "parent_message_id": None})
    body = capture(client, cid, headers, cursor={"kind": "before_message", "message_id": "earlier"}).json()
    selection = resolve_history_selection(body["snapshot"], body["view"], "send", "before-first")["selection"]
    response = client.post(f"/api/v1/chats/{cid}/messages", headers=headers, json={"id": "new-root", "role": "user",
        "content": "new branch", "tldw_history_selection_v1": selection})
    assert response.status_code == 201, response.text
    assert db.get_message_by_id("new-root")["parent_message_id"] is None


@pytest.mark.parametrize("invalid", ["digest", "author_note"])
def test_saved_materialized_context_rejects_in_owner_transaction_without_writes(history_api, monkeypatch, invalid):
    from tldw_Server_API.app.core.Character_Chat.character_conversation_factory import create_character_conversation
    from tldw_Server_API.app.core.DB_Management.chacha.conversation_resume_store import (
        build_materialized_behavior_settings,
    )

    client, db, _, headers = history_api
    monkeypatch.setattr(db, "client_id", "1")
    char = db.add_character_card({"name": "Bound character", "description": "saved", "client_id": "1"})
    cid = create_character_conversation(
        db,
        conversation_data={"character_id": char},
        provider="openai",
        model="gpt-4o-mini",
        prompt_preset_id="st_default",
    )
    with db.transaction() as conn:
        state = db.get_roleplay_resume_state(cid, conn=conn, owner_client_id="1")
    settings = state["settings"]
    values = settings["roleplayBehaviorV1"]["values"]
    if invalid == "author_note":
        values["behavior_controls"]["author_note"] = {"enabled": True, "text": "REQUIRED SAVED NOTE"}
    else:
        values["base_snapshot"]["digest"] = "sha256:" + "0" * 64
    settings["roleplayBehaviorV1"] = build_materialized_behavior_settings(values)
    db.upsert_conversation_settings(cid, settings)
    before = db.get_conversation_settings(cid)
    body = capture(client, cid, headers).json()
    selection = resolve_history_selection(body["snapshot"], body["view"], "send", "invalid-materialized")["selection"]
    reads = []
    original = db.get_roleplay_resume_state

    def checked_read(*args, **kwargs):
        reads.append(kwargs.get("conn") is not None and kwargs.get("lock_for_update") is True)
        return original(*args, **kwargs)

    monkeypatch.setattr(db, "get_roleplay_resume_state", checked_read)
    response = client.post(
        "/api/v1/chat/completions",
        headers=headers,
        json={
            "model": "gpt-4o-mini",
            "conversation_id": cid,
            "save_to_db": True,
            "messages": [{"role": "user", "content": "hello"}],
            "tldw_history_selection_v1": selection,
        },
    )
    assert response.status_code == 409, response.text
    expected = (
        "unsupported_history_context_author_note"
        if invalid == "author_note"
        else "unsupported_history_context_materialized_binding"
    )
    assert response.json()["detail"]["code"] == expected
    assert any(reads)
    assert db.count_messages_for_conversation(cid) == 0
    assert db.get_conversation_settings(cid) == before


@pytest.mark.parametrize("save_fails", [False, True])
def test_native_endpoint_strips_provider_receipts_and_only_acks_owner_save(history_api, monkeypatch, save_fails):
    import json

    from tldw_Server_API.app.api.v1.endpoints import chat
    from tldw_Server_API.app.core.Chat import streaming_utils

    monkeypatch.setattr(streaming_utils, "CHAT_STREAM_INCLUDE_METADATA", False)
    client, db, cid, headers = history_api
    body = capture(client, cid, headers).json()
    selection = resolve_history_selection(body["snapshot"], body["view"], "send", "spoof-boundary")["selection"]

    def provider(**_kwargs):
        yield "data: " + json.dumps({"choices": [{"delta": {"content": "answer"}, "finish_reason": "stop"}],
            "tldw_message_id": "forged-result", "tldw_history_admission_v1": {"input_message_id": "forged-input"}}) + "\n\n"
        yield "data: [DONE]\n\n"

    monkeypatch.setattr(chat, "perform_chat_api_call", provider)
    if save_fails:
        def fail_save(*_args, **_kwargs):
            raise RuntimeError("owner save failed")
        monkeypatch.setattr(db, "settle_history_admission", fail_save)
    response = client.post("/api/v1/chat/completions", headers=headers, json={
        "model": "gpt-4o-mini", "conversation_id": cid, "save_to_db": True, "stream": True,
        "messages": [{"role": "user", "content": "question"}], "tldw_history_selection_v1": selection})
    assert response.status_code == 200, response.text
    frames = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ") and line[6:] != "[DONE]"]
    admissions = [frame["tldw_history_admission_v1"] for frame in frames if "tldw_history_admission_v1" in frame]
    assert len(admissions) == 1
    assert "forged" not in response.text
    replies = [row for row in db.get_messages_for_conversation(cid) if row["sender"] == "assistant"]
    saved_ids = {frame["tldw_message_id"] for frame in frames if "tldw_message_id" in frame}
    if save_fails:
        assert not replies and not saved_ids
    else:
        assert replies and saved_ids == {row["id"] for row in replies}
        assert all(row["parent_message_id"] == admissions[0]["input_message_id"] for row in replies)


def test_native_fork_context_and_scoped_settings_expected_owner(history_api):
    client, db, cid, headers = history_api
    body = capture(client, cid, headers).json()
    assert body["snapshot"]["native_fork_context"] == {
        "policy": "plain_v1", "supported": True,
        "storage_context_digest": body["storage_context_digest"],
    }
    # Omitted expected-user remains backward compatible; a stale assertion cannot read or write.
    assert client.get(f"/api/v1/chats/{cid}/settings", headers=headers).status_code == 200
    stale = {**headers, "X-TLDW-Expected-User-ID": "another-account"}
    assert client.get(f"/api/v1/chats/{cid}", headers=stale).status_code == 412
    assert client.get(f"/api/v1/chats/{cid}/settings", headers=stale).status_code == 412
    assert client.put(f"/api/v1/chats/{cid}/settings", headers=stale, json={"settings": {"authorNote": "wrong owner"}}).status_code == 412
    assert db.get_conversation_settings(cid) is None


def test_limited_native_copy_explicit_plain_identity_and_child_chain_reopen(history_api):
    client, db, _, headers = history_api
    workspace = db.upsert_workspace("fork-workspace", "Fork workspace")
    db.update_workspace("fork-workspace", {"assistant_defaults_json": {
        "assistant_kind": "persona", "assistant_id": "would-be-required", "persona_memory_mode": "read_only"
    }}, expected_version=workspace["version"])
    created = client.post("/api/v1/chats/", headers=headers, json={
        "title": "Plain fork", "scope_type": "workspace", "workspace_id": "fork-workspace",
        "character_id": None, "assistant_kind": None, "assistant_id": None, "persona_memory_mode": None,
    })
    assert created.status_code == 201, created.text
    child = created.json()["id"]
    assert db.get_conversation_by_id(child)["assistant_kind"] is None
    assert db.get_conversation_by_id(child)["assistant_id"] is None
    scope = "?scope_type=workspace&workspace_id=fork-workspace"
    parent = None
    ids = []
    for role, content in (("user", "  exact user  "), ("assistant", "  exact reply  "), ("system", "system context")):
        response = client.post(f"/api/v1/chats/{child}/messages{scope}", headers=headers,
            json={"role": role, "content": content, "parent_message_id": parent})
        assert response.status_code == 201, response.text
        body = response.json()
        assert body["conversation_id"] == child
        assert body["parent_message_id"] == parent
        parent = body["id"]
        ids.append(parent)
    reopened = client.post(f"/api/v1/chat/conversations/{child}/history/selection{scope}", headers=headers,
        json={"purpose": "fork", "view": {"view_session_id": "child-view", "conversation_id": child,
            "interpretation": {"kind": "parent_graph_v1"}, "cursor": {"kind": "after_message", "message_id": parent}, "selection_revision": 0}})
    assert reopened.status_code == 200, reopened.text
    body = reopened.json()
    assert body["status"] == "captured"
    assert [row["id"] for row in body["rows"]] == ids
    assert [row["message"] for row in body["selected_content"]] == ["  exact user  ", "  exact reply  ", "system context"]
    edited = client.put(f"/api/v1/chats/{child}/settings{scope}", headers=headers, json={"settings": {"authorNote": "legitimate edit"}})
    assert edited.status_code == 200, edited.text
    assert client.get(f"/api/v1/chats/{child}/settings{scope}", headers=headers).json()["settings"]["authorNote"] == "legitimate edit"


@pytest.mark.parametrize("field", ["tool_calls_json", "extra_json"])
def test_capture_reports_corrupt_selected_metadata_as_conflict(history_api, field):
    client, db, cid, headers = history_api
    empty = capture(client, cid, headers).json()
    selection = resolve_history_selection(empty["snapshot"], empty["view"], "send", "metadata-test")["selection"]
    accepted = client.post(f"/api/v1/chats/{cid}/messages", headers=headers, json={
        "id": "corrupt", "role": "user", "content": "input", "tldw_history_selection_v1": selection,
    })
    assert accepted.status_code == 201, accepted.text
    db.set_message_metadata_extra("corrupt", {"valid": True})
    with db.transaction() as conn:
        sql = {
            "tool_calls_json": "UPDATE message_metadata SET tool_calls_json = ? WHERE message_id = ?",
            "extra_json": "UPDATE message_metadata SET extra_json = ? WHERE message_id = ?",
        }[field]
        conn.execute(sql, ("{broken", "corrupt"))
    response = capture(client, cid, headers, cursor={"kind": "after_message", "message_id": "corrupt"})
    assert response.status_code == 409, response.text
    assert response.json()["detail"]["code"] == "invalid_metadata"  # nosec B101 - regression assertion


@pytest.mark.parametrize(
    "strategy,index,raw",
    [
        ("default", None, "  {{char}} welcomes {{user}}.\n\nChoose a path.  "),
        (
            "alternate_index",
            0,
            "Alternate zero — {{char}} opens a map for {{user}}.\n  ",
        ),
    ],
)
def test_seeded_character_opening_is_an_owned_assistant_root(
    history_api, strategy, index, raw
):
    """Create through the real API, then select its exact canonical H1 row."""
    client, db, _source, headers = history_api
    character_id = db.add_character_card(
        {
            "name": "Native Greeter",
            "first_message": "  {{char}} welcomes {{user}}.\n\nChoose a path.  ",
            "alternate_greetings": [
                "Alternate zero — {{char}} opens a map for {{user}}.\n  "
            ],
        }
    )
    params = {"seed_first_message": True, "greeting_strategy": strategy}
    if index is not None:
        params["alternate_index"] = index
    created = client.post(
        "/api/v1/chats/",
        headers=headers,
        params=params,
        json={
            "character_id": character_id,
            "provider": "local-llm",
            "model": "local-test",
            "temperature": 0.0,
            "top_p": 0.0,
        },
    )
    assert created.status_code == 201, created.text
    cid = created.json()["id"]
    listed = client.get(f"/api/v1/chats/{cid}/messages", headers=headers)
    assert listed.status_code == 200, listed.text
    messages = listed.json()["messages"]
    assert len(messages) == 1
    mid = messages[0]["id"]
    selected = capture(
        client, cid, headers, cursor={"kind": "after_message", "message_id": mid}
    )
    assert selected.status_code == 200, selected.text
    body = selected.json()
    assert body["status"] == "captured"
    assert [
        (row["id"], row["role"], row["settled"], row["parent_id"])
        for row in body["rows"]
    ] == [(mid, "assistant", True, None)]
    assert body["snapshot"]["interpretation_status"] == {"kind": "parent_graph_v1"}
    assert body["view"]["owner_key"] == body["snapshot"]["owner_key"]
    assert body["selected_content"][0]["message"] == raw
    assert db.get_message_by_id(mid)["content"] == raw
    assert messages[0]["sender"] == "assistant"
    state = db.get_roleplay_resume_state(cid)
    effective = state["effective_completion"]
    assert (
        effective["provider"],
        effective["model"],
        effective["sampling"]["temperature"],
        effective["sampling"]["top_p"],
    ) == ("local-llm", "local-test", 0.0, 0.0)
    assert (
        client.get(f"/api/v1/chats/{cid}", headers=headers).json()["character_name"]
        == "Native Greeter"
    )
    assert db.count_messages_for_conversation(cid) == 1


@pytest.mark.parametrize("seed,greeting", [(False, "An unused greeting"), (True, "")])
def test_factory_opening_change_preserves_supplied_and_existing_named_rows(
    history_api, seed, greeting
):
    """The seed fix must not reclassify caller data or stored legacy history."""
    from tldw_Server_API.app.core.Character_Chat.character_conversation_factory import (
        create_character_conversation,
    )

    _client, db, source, _headers = history_api
    old_id = db.add_message(
        {
            "conversation_id": source,
            "sender": "Legacy Greeter",
            "content": "Old raw {{char}}",
        }
    )
    old = db.get_message_by_id(old_id)
    character_id = db.add_character_card(
        {"name": "New Greeter", "first_message": greeting}
    )
    cid = create_character_conversation(
        db,
        conversation_data={
            "character_id": character_id,
            "client_id": "1",
            "title": "Compatibility",
        },
        provider="local-llm",
        model="local-test",
        seed_first_message=seed,
        initial_messages=[
            {
                "id": "supplied-root",
                "sender": "Supplied Greeter",
                "content": "Keep {{char}} raw",
            }
        ],
    )
    assert [
        (row["id"], row["sender"], row["content"], row["parent_message_id"])
        for row in db.get_messages_for_conversation(cid)
    ] == [("supplied-root", "Supplied Greeter", "Keep {{char}} raw", None)]
    assert db.get_message_by_id(old_id) == old


@pytest.mark.parametrize("role", ["user", "assistant"])
@pytest.mark.parametrize("character", [False, True])
@pytest.mark.parametrize("workspace", [False, True])
def test_plain_message_save_preserves_owned_graph_and_raw_acknowledgement(
    history_api, role, character, workspace
):
    """Exercise real GET/CAS/H1: displayed PUT text is not raw saved content."""
    client, db, cid, headers = history_api
    scope = {"scope_type": "global"}
    if character or workspace:
        payload = {"title": "Native Save", "character_id": None}
        if character:
            payload["character_id"] = db.add_character_card({"name": "Mira"})
        if workspace:
            db.upsert_workspace("save-workspace", "Save workspace")
            scope = {"scope_type": "workspace", "workspace_id": "save-workspace"}
            payload.update(scope)
        created = client.post(
            "/api/v1/chats/",
            headers=headers,
            params={"seed_first_message": False},
            json=payload,
        )
        assert created.status_code == 201, created.text
        cid = created.json()["id"]
    bootstrap = client.post(
        f"/api/v1/chat/conversations/{cid}/history/selection",
        headers=headers,
        params=scope,
        json={
            "purpose": "send",
            "view": {
                "view_session_id": "save-fixture",
                "conversation_id": cid,
                "interpretation": {"kind": "parent_graph_v1"},
                "cursor": {"kind": "empty"},
                "selection_revision": 0,
            },
        },
    )
    assert bootstrap.status_code == 200, bootstrap.text
    body = bootstrap.json()
    assert body["status"] == "captured"
    selection = resolve_history_selection(
        body["snapshot"], body["view"], "send", "save-fixture"
    )["selection"]
    accepted = client.post(
        f"/api/v1/chats/{cid}/messages",
        headers=headers,
        params=scope,
        json={
            "id": "save-user",
            "role": "user",
            "content": "Original {{char}}  \n",
            "tldw_history_selection_v1": selection,
        },
    )
    assert accepted.status_code == 201, accepted.text
    ids = {"user": accepted.json()["id"]}
    admission = accepted.json()["tldw_history_admission_v1"]
    reference = {
        key: admission[key]
        for key in (
            "version",
            "owner_key",
            "conversation_id",
            "input_message_id",
            "input_message_revision",
            "selection_digest",
        )
    }
    for label in ("reply", "sibling"):
        added = client.post(
            f"/api/v1/chats/{cid}/messages",
            headers=headers,
            params=scope,
            json={
                "id": f"save-{label}",
                "role": "assistant",
                "content": "Original {{char}}  \n",
                "tldw_history_admission_v1": reference,
            },
        )
        assert added.status_code == 201, added.text
        ids[label] = added.json()["id"]
    target = ids["user" if role == "user" else "reply"]
    db.add_message_metadata(target, tool_calls=[{"id": "tool", "name": "lookup"}])
    db.set_message_metadata_extra(
        target,
        {
            "sender_role": role,
            "pinned": True,
            "nested": {"value": 1},
        },
    )
    metadata = db.get_message_metadata(target)
    sibling = db.get_message_by_id(ids["sibling"])
    view = {
        "view_session_id": "save-view",
        "conversation_id": cid,
        "interpretation": {"kind": "parent_graph_v1"},
        "cursor": {"kind": "after_message", "message_id": ids["reply"]},
        "selection_revision": 1,
    }
    before_response = client.post(
        f"/api/v1/chat/conversations/{cid}/history/selection",
        headers=headers,
        params=scope,
        json={"purpose": "send", "view": view},
    )
    assert before_response.status_code == 200, before_response.text
    before = before_response.json()
    raw = client.get(f"/api/v1/messages/{target}", headers=headers, params=scope)
    assert raw.status_code == 200, raw.text
    version = raw.json()["version"]
    edited = "  {{char}} answers {{user}}.\n\nExact authored edit  "
    saved = client.put(
        f"/api/v1/messages/{target}",
        headers=headers,
        params={**scope, "expected_version": version},
        json={"content": edited},
    )
    assert saved.status_code == 200, saved.text
    assert saved.json()["version"] == version + 1
    assert saved.json()["content"] != edited
    after_response = client.post(
        f"/api/v1/chat/conversations/{cid}/history/selection",
        headers=headers,
        params=scope,
        json={"purpose": "send", "view": before["view"]},
    )
    assert after_response.status_code == 200, after_response.text
    after = after_response.json()
    after_node = next(node for node in after["rows"] if node["id"] == target)
    before_node = next(node for node in before["rows"] if node["id"] == target)
    assert after_node["revision"] != before_node["revision"]
    assert after_node["revision"] != str(saved.json()["version"])
    for field in ("parent_id", "role", "settled", "metadata", "assets", "comparison"):
        assert after_node.get(field) == before_node.get(field)
    assert after["view"] == before["view"]
    selected = next(row for row in after["selected_content"] if row["id"] == target)
    assert selected["message"] == edited
    assert selected["tool_calls"] == metadata["tool_calls"]
    assert selected["extra_metadata"] == metadata["extra"]
    readback = client.get(f"/api/v1/messages/{target}", headers=headers, params=scope)
    assert readback.status_code == 200
    assert readback.json()["content"] == edited
    assert readback.json()["version"] == version + 1
    assert db.get_message_metadata(target) == metadata
    assert db.get_message_by_id(ids["sibling"]) == sibling
    conflict = client.put(
        f"/api/v1/messages/{target}",
        headers=headers,
        params={**scope, "expected_version": version},
        json={"content": "Lost edit"},
    )
    assert conflict.status_code == 409, conflict.text
    final_response = client.post(
        f"/api/v1/chat/conversations/{cid}/history/selection",
        headers=headers,
        params=scope,
        json={"purpose": "send", "view": before["view"]},
    )
    assert final_response.status_code == 200
    assert final_response.json() == after


@pytest.mark.parametrize("boundary", ["account", "workspace", "trashed"])
def test_plain_message_save_refuses_retired_owner_without_mutation(
    history_api, boundary
):
    """The real message route must reject stale owner/scope and hidden chats."""
    client, db, cid, headers = history_api
    mid = db.add_message(
        {
            "id": "owner-save",
            "conversation_id": cid,
            "sender": "user",
            "content": "Kept",
            "parent_message_id": None,
        }
    )
    request_headers = headers
    params = {"expected_version": 1}
    expected_status = 404
    if boundary == "account":
        request_headers = {**headers, "X-TLDW-Expected-User-ID": "another-account"}
        expected_status = 412
    elif boundary == "workspace":
        params.update({"scope_type": "workspace", "workspace_id": "other"})
    else:
        conversation = db.get_conversation_by_id(cid)
        assert db.soft_delete_conversation(
            cid, expected_version=conversation["version"]
        )
    before = db.get_message_by_id(mid, include_deleted=True)
    read = client.get(f"/api/v1/messages/{mid}", headers=request_headers, params=params)
    assert read.status_code == expected_status, read.text
    written = client.put(
        f"/api/v1/messages/{mid}",
        headers=request_headers,
        params=params,
        json={"content": "Wrong owner edit"},
    )
    assert written.status_code == expected_status, written.text
    assert db.get_message_by_id(mid, include_deleted=True) == before


@pytest.mark.parametrize(
    "operation", ["read", "list", "search", "edit", "delete", "add"]
)
def test_message_routes_reject_stale_expected_account(history_api, operation):
    """Every message route shares the original-account assertion at the router."""
    client, db, cid, headers = history_api
    mid = db.add_message(
        {
            "id": "scoped-message",
            "conversation_id": cid,
            "sender": "user",
            "content": "Kept",
            "parent_message_id": None,
        }
    )
    stale = {**headers, "X-TLDW-Expected-User-ID": "another-account"}
    before = db.get_message_by_id(mid, include_deleted=True)
    if operation == "read":
        response = client.get(f"/api/v1/messages/{mid}", headers=stale)
    elif operation == "list":
        response = client.get(f"/api/v1/chats/{cid}/messages", headers=stale)
    elif operation == "search":
        response = client.get(
            f"/api/v1/chats/{cid}/messages/search",
            headers=stale,
            params={"query": "Kept"},
        )
    elif operation == "edit":
        response = client.put(
            f"/api/v1/messages/{mid}",
            headers=stale,
            params={"expected_version": 1},
            json={"content": "Wrong account"},
        )
    elif operation == "delete":
        response = client.delete(
            f"/api/v1/messages/{mid}", headers=stale, params={"expected_version": 1}
        )
    else:
        response = client.post(
            f"/api/v1/chats/{cid}/messages",
            headers=stale,
            json={"role": "user", "content": "Wrong account"},
        )
    assert response.status_code == 412, response.text
    assert response.json()["detail"]["code"] == "request_config_scope_changed"
    assert response.headers["cache-control"] == "no-store"
    assert db.get_message_by_id(mid, include_deleted=True) == before
    assert db.count_messages_for_conversation(cid) == 1


@pytest.mark.parametrize("expected_account", [None, "1"])
def test_message_routes_accept_current_expected_account(history_api, expected_account):
    """Absent-header clients and matching scoped clients retain GET/PUT behavior."""
    client, db, cid, headers = history_api
    mid = db.add_message(
        {
            "id": "current-save",
            "conversation_id": cid,
            "sender": "user",
            "content": "Original",
            "parent_message_id": None,
        }
    )
    current = (
        headers
        if expected_account is None
        else {
            **headers,
            "X-TLDW-Expected-User-ID": expected_account,
        }
    )
    read = client.get(f"/api/v1/messages/{mid}", headers=current)
    assert read.status_code == 200, read.text
    assert read.json()["content"] == "Original"
    saved = client.put(
        f"/api/v1/messages/{mid}",
        headers=current,
        params={"expected_version": read.json()["version"]},
        json={"content": "Exact saved edit  \n"},
    )
    assert saved.status_code == 200, saved.text
    assert saved.json()["version"] == read.json()["version"] + 1
    assert db.get_message_by_id(mid)["content"] == "Exact saved edit  \n"


def test_native_leaf_delete_refuses_descendants_through_real_storage(history_api):
    client, db, cid, headers = history_api
    parent = db.add_message({"conversation_id": cid, "sender": "user", "content": "Parent"})
    child = db.add_message(
        {"conversation_id": cid, "sender": "assistant", "content": "Child", "parent_message_id": parent}
    )
    before = db.get_messages_for_conversation(cid)
    response = client.delete(f"/api/v1/messages/{parent}?expected_version=1&require_leaf=true", headers=headers)
    assert response.status_code == 409, response.text
    assert "message_has_descendants" in response.text
    assert db.get_messages_for_conversation(cid) == before
    assert db.get_message_by_id(child)["parent_message_id"] == parent


def test_native_leaf_delete_acknowledges_exact204_and_keeps_parent_graph(history_api):
    client, db, cid, headers = history_api
    parent = db.add_message({"conversation_id": cid, "sender": "user", "content": "Parent"})
    leaf = db.add_message(
        {"conversation_id": cid, "sender": "assistant", "content": "Leaf", "parent_message_id": parent}
    )
    response = client.delete(f"/api/v1/messages/{leaf}?expected_version=1&require_leaf=true", headers=headers)
    assert response.status_code == 204, response.text
    assert response.content == b""
    captured = capture(client, cid, headers, cursor={"kind": "after_message", "message_id": parent})
    assert captured.status_code == 200, captured.text
    assert captured.json()["status"] == "captured"
    assert [row["id"] for row in captured.json()["snapshot"]["nodes"]] == [parent]


def test_native_leaf_delete_preserves_real_wrapper_cas_conflict(history_api, monkeypatch):
    from tldw_Server_API.app.api.v1.endpoints import character_messages

    client, db, cid, headers = history_api
    leaf = db.add_message({"conversation_id": cid, "sender": "user", "content": "Original"})
    verify = character_messages._verify_message_access

    def held_precheck(*args, **kwargs):
        original = verify(*args, **kwargs)
        db.update_message(leaf, {"content": "Changed after endpoint read"}, 1)
        return original

    monkeypatch.setattr(character_messages, "_verify_message_access", held_precheck)
    response = client.delete(f"/api/v1/messages/{leaf}?expected_version=1&require_leaf=true", headers=headers)
    assert response.status_code == 409, response.text
    assert db.get_message_by_id(leaf)["content"] == "Changed after endpoint read"


@pytest.mark.parametrize("native_options", [{"require_leaf": "true"}, {"expected_revision": "opaque"}])
def test_native_leaf_delete_refuses_enrolled_sync_before_product_or_sync_mutation(
    history_api, monkeypatch, native_options
):
    from tldw_Server_API.app.api.v1.endpoints import character_messages

    client, db, cid, headers = history_api
    leaf = db.add_message({"conversation_id": cid, "sender": "user", "content": "Keep current"})
    before = db.get_message_by_id(leaf)
    monkeypatch.setattr(character_messages, "_active_message_sync_service", lambda *_: object())

    def no_sync_write(*_args, **_kwargs):
        raise AssertionError("Unsupported native Sync deletion cannot write")

    monkeypatch.setattr(character_messages, "delete_unenrolled_server_origin_objects", no_sync_write)
    monkeypatch.setattr(character_messages, "capture_server_origin_mutation", no_sync_write)
    response = client.delete(
        f"/api/v1/messages/{leaf}", headers=headers, params={"expected_version": 1, **native_options}
    )
    assert response.status_code == 409, response.text
    assert "unsupported_native_sync_delete" in response.text
    assert db.get_message_by_id(leaf) == before


def test_legacy_non_leaf_delete_semantics_stay_explicitly_unchanged(history_api):
    client, db, cid, headers = history_api
    parent = db.add_message({"conversation_id": cid, "sender": "user", "content": "Parent"})
    child = db.add_message(
        {"conversation_id": cid, "sender": "assistant", "content": "Child", "parent_message_id": parent}
    )
    response = client.delete(f"/api/v1/messages/{parent}?expected_version=1", headers=headers)
    assert response.status_code == 204, response.text
    assert db.get_message_by_id(child)["parent_message_id"] == parent


@pytest.fixture
def native_mutation_api(history_db, credentialed_test_client, auth_headers):
    """Exercise real HTTP mutation on the standard SQLite/Postgres fixture."""
    db, _seed = history_db
    client = credentialed_test_client
    cid = db.add_conversation({"character_id": None, "title": "Native mutation HTTP", "client_id": "1"})
    client.app.dependency_overrides[get_chacha_db_for_user] = lambda: db
    yield client, db, cid, {**auth_headers, "X-TLDW-Expected-User-ID": "1"}
    client.app.dependency_overrides.pop(get_chacha_db_for_user, None)


@pytest.mark.parametrize("mutation", [None, "metadata", "image"])
def test_native_leaf_delete_enforces_opaque_cas_through_http(native_mutation_api, mutation):
    client, db, cid, headers = native_mutation_api
    leaf = db.add_message({"conversation_id": cid, "sender": "user", "content": "Exact {{char}}  \n"})
    selected = capture(client, cid, headers, cursor={"kind": "after_message", "message_id": leaf})
    assert selected.status_code == 200, selected.text
    revision = selected.json()["snapshot"]["nodes"][0]["revision"]
    if mutation == "metadata":
        db.set_message_metadata_extra(leaf, {"pinned": True})
    elif mutation == "image":
        db.append_message_image(leaf, b"image changed after capture", "image/png")
    before = db.get_message_by_id(leaf, include_deleted=True)
    assert before["version"] == 1
    response = client.delete(
        f"/api/v1/messages/{leaf}",
        headers=headers,
        params={"expected_version": 1, "require_leaf": "true", "expected_revision": revision},
    )
    if mutation is None:
        assert response.status_code == 204, response.text
        assert response.content == b""
        assert db.get_message_by_id(leaf) is None
    else:
        assert response.status_code == 409, response.text
        assert "message_changed" in response.text
        assert db.get_message_by_id(leaf, include_deleted=True) == before


@pytest.mark.parametrize("first", ["put", "delete"])
def test_actual_put_and_native_delete_use_conversation_first_order(native_mutation_api, monkeypatch, first):
    client, db, cid, headers = native_mutation_api
    leaf = db.add_message({"conversation_id": cid, "sender": "user", "content": "Original"})
    selected = capture(client, cid, headers, cursor={"kind": "after_message", "message_id": leaf})
    assert selected.status_code == 200, selected.text
    revision = selected.json()["snapshot"]["nodes"][0]["revision"]
    writing, release, started, lock_attempt = (threading.Event() for _ in range(4))
    advance = db.message_store._advance_history_version
    lock = db.message_store._lock_message_conversation
    resume = db.get_roleplay_resume_state

    def pause_after_first_write(conn, conversation_id):
        if not writing.is_set():
            writing.set()
            assert release.wait(10)
        advance(conn, conversation_id)

    def observe_lock(*args, **kwargs):
        if writing.is_set():
            lock_attempt.set()
        return lock(*args, **kwargs)

    def observe_resume(*args, **kwargs):
        if writing.is_set():
            lock_attempt.set()
        return resume(*args, **kwargs)

    monkeypatch.setattr(db.message_store, "_advance_history_version", pause_after_first_write)
    monkeypatch.setattr(db.message_store, "_lock_message_conversation", observe_lock)
    monkeypatch.setattr(db, "get_roleplay_resume_state", observe_resume)

    def request(kind, second=False):
        from fastapi.testclient import TestClient

        # Independent portals let the two synchronous storage transactions race.
        request_client = TestClient(client.app)
        if second:
            started.set()
        if kind == "put":
            return request_client.put(
                f"/api/v1/messages/{leaf}",
                headers=headers,
                params={"expected_version": 1},
                json={"content": "Saved edit", "pinned": True},
            )
        return request_client.delete(
            f"/api/v1/messages/{leaf}",
            headers=headers,
            params={"expected_version": 1, "require_leaf": "true", "expected_revision": revision},
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        a = pool.submit(request, first)
        try:
            assert writing.wait(10)
            b = pool.submit(request, "delete" if first == "put" else "put", True)
            assert started.wait(10)
            if db.backend_type == BackendType.POSTGRESQL:
                assert lock_attempt.wait(10)
            assert not b.done()
        finally:
            release.set()
        original, conflict = a.result(timeout=10), b.result(timeout=10)
    assert original.status_code == (200 if first == "put" else 204), original.text
    assert conflict.status_code in ({409} if first == "put" else {404, 409}), conflict.text
    stored = db.get_message_by_id(leaf, include_deleted=True)
    if first == "put":
        assert not stored["deleted"]
        assert stored["content"] == "Saved edit"
        assert db.get_message_metadata(leaf)["extra"]["pinned"] is True
    else:
        assert stored["deleted"]
        assert stored["content"] == "Original"
        assert not (db.get_message_metadata(leaf) or {}).get("extra", {}).get("pinned")


def test_native_admission_capture_is_opt_in_read_only_and_binds_the_real_hash(history_api):
    client, db, cid, headers = history_api
    empty = capture(client, cid, headers).json()
    selection = resolve_history_selection(empty["snapshot"], empty["view"], "send", "native retry")["selection"]
    accepted = client.post(f"/api/v1/chats/{cid}/messages", headers=headers, json={
        "id": "retry-input", "role": "user", "content": "Reply with ORBIT-742.", "tldw_history_selection_v1": selection})
    assert accepted.status_code == 201  # nosec B101: Actual native owner admission.
    reference = {key: accepted.json()["tldw_history_admission_v1"][key] for key in (
        "version", "owner_key", "conversation_id", "input_message_id", "input_message_revision", "selection_digest")}
    ordinary = capture(client, cid, headers, cursor={"kind": "after_message", "message_id": "retry-input"}).json()
    assert "validated_admission" not in ordinary  # nosec B101: Ordinary capture wire remains unchanged.
    assert ordinary["rows"][-1]["revision"] != reference["input_message_revision"]  # nosec B101: Exercise actual distinct revision domains.
    payload = {"purpose": "send", "view": ordinary["view"], "accepted_input_admission": reference}
    checked = client.post(f"/api/v1/chat/conversations/{cid}/history/selection", headers=headers, json=payload)
    assert checked.status_code == 200, checked.text  # nosec B101: Opt-in accepted-input proof is supported.
    assert checked.json()["validated_admission"] == {"admission": reference, "input_revision": ordinary["rows"][-1]["revision"]}  # nosec B101: Proof acknowledges the exact receipt and composite capture.
    assert db.count_messages_for_conversation(cid) == 1  # nosec B101: No new message or admission.
    db.add_message_metadata("retry-input", extra={"changed": True})
    refused = client.post(f"/api/v1/chat/conversations/{cid}/history/selection", headers=headers, json=payload)
    assert refused.status_code == 409  # nosec B101: Protected state drift refuses capture before any generation.
    assert db.count_messages_for_conversation(cid) == 1  # nosec B101: Refusal does not append.


@pytest.mark.parametrize("invalid", ["fork", "cursor", "regenerate"])
def test_native_admission_capture_rejects_non_retry_requests(history_api, invalid):
    client, db, cid, headers = history_api
    ordinary = capture(client, cid, headers).json()
    admission = {"version": 1, "owner_key": ordinary["view"]["owner_key"], "conversation_id": cid,
                 "input_message_id": "question", "input_message_revision": "1", "selection_digest": "receipt"}
    view = {**ordinary["view"], "cursor": {"kind": "after_message", "message_id": "question"}}
    payload = {"purpose": "send", "view": view, "accepted_input_admission": admission}
    if invalid == "fork":
        payload["purpose"] = "fork"
    elif invalid == "cursor":
        view["cursor"] = {"kind": "empty"}
    else:
        payload["required_capability"] = "tldw_history_existing_input_v1"
    response = client.post(f"/api/v1/chat/conversations/{cid}/history/selection", headers=headers, json=payload)
    assert response.status_code == 422  # nosec B101: Invalid retry/proof request is refused before capture handling.
    assert db.count_messages_for_conversation(cid) == 0  # nosec B101: Invalid request never admits input.


def test_b02_server_existing_input_capability_is_distinct_and_read_only(history_api):
    client, db, cid, headers = history_api
    question = db.add_message({"id": "b02-capability-question", "conversation_id": cid,
                               "sender": "user", "content": "saved B02 question"})
    ordinary = capture(client, cid, headers, cursor={"kind": "after_message", "message_id": question}).json()
    before = db.get_message_by_id(question)
    checked = client.post(f"/api/v1/chat/conversations/{cid}/history/selection", headers=headers, json={
        "purpose": "send", "view": ordinary["view"],
        "required_capability": "tldw_history_server_existing_input_v1",
    })
    assert checked.status_code == 200, checked.text  # nosec B101 - Explicit completion capability, distinct from message settlement.
    assert checked.json()["acknowledged_capability"] == "tldw_history_server_existing_input_v1"  # nosec B101 - Old capability cannot authorize new completion semantics.
    assert db.get_message_by_id(question) == before  # nosec B101 - Capability capture never rewrites the saved question.
    assert db.count_messages_for_conversation(cid) == 1  # nosec B101 - No admission or reply during negotiation.


def test_b02_server_existing_input_completion_never_readmits_question(history_api, monkeypatch):
    import json
    from unittest.mock import AsyncMock

    from tldw_Server_API.app.api.v1.endpoints import chat

    client, db, cid, headers = history_api
    question = db.add_message({"id": "b02-existing-question", "conversation_id": cid,
                               "sender": "user", "content": "B02-CANONICAL-QUESTION"})
    before = db.get_message_by_id(question)
    captured = capture(client, cid, headers, cursor={"kind": "after_message", "message_id": question}).json()
    selection = resolve_history_selection(captured["snapshot"], captured["view"], "send", "b02-existing-answer")["selection"]
    calls = []
    live_skills = AsyncMock(return_value="LIVE-SKILLS-MUST-NOT-ENTER-FROZEN-CONTEXT")
    monkeypatch.setattr(chat, "build_system_message_with_skills_async", live_skills)

    def provider(**kwargs):
        calls.append(kwargs["messages_payload"])
        yield "data: " + json.dumps({"choices": [{"delta": {"content": "B02-ANSWER"}, "finish_reason": "stop"}]}) + "\n\n"
        yield "data: [DONE]\n\n"

    monkeypatch.setattr(chat, "perform_chat_api_call", provider)
    response = client.post("/api/v1/chat/completions", headers=headers, json={
        "model": "gpt-4o-mini", "conversation_id": cid, "save_to_db": True, "stream": True,
        "messages": [], "tldw_history_server_existing_input_v1": selection,
    })
    assert response.status_code == 200, response.text  # nosec B101 - Explicit existing-input completion must be supported.
    live_skills.assert_not_awaited()
    assert len(calls) == 1  # nosec B101 - Exactly one provider execution.
    assert len([m for m in calls[0] if m["role"] == "user"]) == 1  # nosec B101 - The canonical question is model context once.
    assert json.dumps(calls[0]).count("B02-CANONICAL-QUESTION") == 1  # nosec B101 - No duplicate current input is appended.
    messages = db.get_messages_for_conversation(cid)
    assert [m["sender"] for m in messages] == ["user", "assistant"]  # nosec B101 - No second user admission.
    assert db.get_message_by_id(question) == before  # nosec B101 - Preserve the exact original question and its authority.
    assert messages[1]["content"] == "B02-ANSWER"  # nosec B101 - The actual provider answer is saved.
    assert messages[1]["parent_message_id"] == question  # nosec B101 - Reply belongs to the existing canonical user.
    frames = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ") and line[6:] != "[DONE]"]
    assert {frame["tldw_message_id"] for frame in frames if "tldw_message_id" in frame} == {messages[1]["id"]}  # nosec B101 - Only owner-persisted assistant settlement is acknowledged.
    assert not any("tldw_history_admission_v1" in frame for frame in frames)  # nosec B101 - Do not manufacture another input admission.


@pytest.mark.parametrize("policy", ["block", "redact"])
def test_b02_existing_question_retry_keeps_input_policy(history_api, monkeypatch, policy):
    from unittest.mock import Mock

    from fastapi import HTTPException

    from tldw_Server_API.app.api.v1.endpoints import chat

    client, db, cid, headers = history_api
    question = db.add_message({"id": "b02-policy-question", "conversation_id": cid,
                               "sender": "user", "content": "B02 policy subject"})
    before = db.get_message_by_id(question)
    captured = capture(client, cid, headers, cursor={"kind": "after_message", "message_id": question}).json()
    selection = resolve_history_selection(captured["snapshot"], captured["view"], "send", "b02-input-policy")["selection"]
    observed = []

    async def input_policy(**kwargs):
        messages = kwargs["request_data"].messages
        observed.append([m.content for m in messages if m.role == "user"])
        if policy == "block":
            raise HTTPException(403, "Blocked by current input policy")
        if messages:
            messages[0].content = "Redacted by current input policy"

    monkeypatch.setattr(chat, "moderate_input_messages", input_policy)
    provider = Mock(side_effect=AssertionError("denied canonical input must not reach provider"))
    monkeypatch.setattr(chat, "perform_chat_api_call", provider)
    response = client.post("/api/v1/chat/completions", headers=headers, json={
        "model": "gpt-4o-mini", "conversation_id": cid, "save_to_db": True, "stream": True,
        "messages": [], "tldw_history_server_existing_input_v1": selection,
    })
    assert observed == [["B02 policy subject"]]  # nosec B101 - Moderate the actual saved question, not the empty wire list.
    assert response.status_code == 403, response.text  # nosec B101 - A saved question cannot bypass current policy or be silently rewritten.
    provider.assert_not_called()
    assert db.get_message_by_id(question) == before  # nosec B101 - Refusal never edits the authoritative question.
    assert db.count_messages_for_conversation(cid) == 1  # nosec B101 - No input or reply added after policy refusal.


@pytest.mark.parametrize("text", ["saved question exceeds four characters", "x" * 100001])
def test_b02_existing_question_obeys_payload_text_limit(history_api, monkeypatch, text):
    from unittest.mock import Mock

    from tldw_Server_API.app.api.v1.endpoints import chat

    client, db, cid, headers = history_api
    question = db.add_message({"id": "b02-sized-question", "conversation_id": cid,
                               "sender": "user", "content": text})
    captured = capture(client, cid, headers, cursor={"kind": "after_message", "message_id": question}).json()
    selection = resolve_history_selection(captured["snapshot"], captured["view"], "send", "b02-input-size")["selection"]
    monkeypatch.setattr(chat, "MAX_TEXT_LENGTH", 4)
    provider = Mock(side_effect=AssertionError("oversized canonical input must not reach provider"))
    monkeypatch.setattr(chat, "perform_chat_api_call", provider)
    response = client.post("/api/v1/chat/completions", headers=headers, json={
        "model": "gpt-4o-mini", "conversation_id": cid, "save_to_db": True, "stream": True,
        "messages": [], "tldw_history_server_existing_input_v1": selection,
    })
    assert response.status_code == 413, response.text  # nosec B101 - Existing-input replay validates real canonical text length.
    provider.assert_not_called()
    assert db.count_messages_for_conversation(cid) == 1  # nosec B101 - Rejection has no persistence side effects.


@pytest.mark.parametrize("change", ["owner", "question", "answered"])
def test_b02_existing_question_rejects_changed_authority(history_api, monkeypatch, change):
    from unittest.mock import Mock

    from tldw_Server_API.app.api.v1.endpoints import chat

    client, db, cid, headers = history_api
    question = db.add_message({"id": "b02-changed-question", "conversation_id": cid,
                               "sender": "user", "content": "original saved question"})
    captured = capture(client, cid, headers, cursor={"kind": "after_message", "message_id": question}).json()
    if change == "question":
        db.update_message(question, {"content": "edited saved question"}, expected_version=1)
    elif change == "answered":
        db.add_message({"id": "b02-already-answered", "conversation_id": cid, "sender": "assistant",
                        "content": "already answered", "parent_message_id": question})
        captured = capture(client, cid, headers, cursor={"kind": "after_message", "message_id": question}).json()
    selection = resolve_history_selection(captured["snapshot"], captured["view"], "send", "b02-changed-boundary")["selection"]
    if change == "owner":
        selection["owner_key"] = "different-owner"
    count = db.count_messages_for_conversation(cid)
    provider = Mock(side_effect=AssertionError("changed authority must not reach provider"))
    monkeypatch.setattr(chat, "perform_chat_api_call", provider)
    response = client.post("/api/v1/chat/completions", headers=headers, json={
        "model": "gpt-4o-mini", "conversation_id": cid, "save_to_db": True, "stream": True,
        "messages": [], "tldw_history_server_existing_input_v1": selection,
    })
    assert response.status_code == 409, response.text  # nosec B101 - Reject stale input, foreign owner, and an already answered question.
    provider.assert_not_called()
    assert db.count_messages_for_conversation(cid) == count  # nosec B101 - Refusal never appends another question or answer.


def test_b02_existing_question_refuses_answer_saved_during_generation(history_api, monkeypatch):
    import json

    from tldw_Server_API.app.api.v1.endpoints import chat

    client, db, cid, headers = history_api
    question = db.add_message({"id": "b02-racing-question", "conversation_id": cid, "sender": "user", "content": "question"})
    captured = capture(client, cid, headers, cursor={"kind": "after_message", "message_id": question}).json()
    selection = resolve_history_selection(captured["snapshot"], captured["view"], "send", "b02-racing-answer")["selection"]
    def provider(**kwargs):
        db.add_message({"id": "b02-other-answer", "conversation_id": cid, "sender": "assistant",
                        "content": "already answered", "parent_message_id": question})
        yield "data: " + json.dumps({"choices": [{"delta": {"content": "late answer"}, "finish_reason": "stop"}]}) + "\n\n"
        yield "data: [DONE]\n\n"
    monkeypatch.setattr(chat, "perform_chat_api_call", provider)
    response = client.post("/api/v1/chat/completions", headers=headers, json={
        "model": "gpt-4o-mini", "conversation_id": cid, "save_to_db": True, "stream": True,
        "messages": [], "tldw_history_server_existing_input_v1": selection,
    })
    assert [m["id"] for m in db.get_messages_for_conversation(cid)] == [question, "b02-other-answer"]  # nosec B101 - A concurrent answer forbids a second settlement.
    assert "tldw_message_id" not in response.text  # nosec B101 - Failed settlement cannot acknowledge an owner-persisted reply.


@pytest.mark.parametrize("conflict", ["messages", "selection", "branch", "stream", "tools", "empty-legacy"])
def test_b02_existing_question_schema_refuses_ambiguous_or_legacy_intent(history_api, conflict):
    client, db, cid, headers = history_api
    question = db.add_message({"id": "b02-schema-question", "conversation_id": cid, "sender": "user", "content": "question"})
    captured = capture(client, cid, headers, cursor={"kind": "after_message", "message_id": question}).json()
    selection = resolve_history_selection(captured["snapshot"], captured["view"], "send", "b02-schema")["selection"]
    payload = {"model": "gpt-4o-mini", "conversation_id": cid, "save_to_db": True, "stream": True,
               "messages": [], "tldw_history_server_existing_input_v1": selection}
    if conflict == "messages":
        payload["messages"] = [{"role": "user", "content": "another question"}]
    elif conflict == "selection":
        payload["tldw_history_selection_v1"] = selection
    elif conflict == "branch":
        payload["tldw_history_branch"] = False
    elif conflict == "stream":
        payload["stream"] = False
    elif conflict == "tools":
        payload["tools"] = [{"type": "function", "function": {"name": "tool", "parameters": {}}}]
    else:
        payload.pop("tldw_history_server_existing_input_v1")
    response = client.post("/api/v1/chat/completions", headers=headers, json=payload)
    assert response.status_code == 422, response.text  # nosec B101 - Never fall through to a legacy or new-question completion.
    assert db.count_messages_for_conversation(cid) == 1  # nosec B101 - Refusal cannot admit or settle anything.


def test_b02_existing_question_provider_failure_keeps_exact_saved_input(history_api, monkeypatch):
    from tldw_Server_API.tests.Chat_NEW.integration.test_server_settled_generation_metadata import _install_provider

    client, db, cid, headers = history_api
    question = db.add_message({"id": "b02-failed-existing", "conversation_id": cid, "sender": "user", "content": "saved question"})
    original = db.get_message_by_id(question)
    captured = capture(client, cid, headers, cursor={"kind": "after_message", "message_id": question}).json()
    selection = resolve_history_selection(captured["snapshot"], captured["view"], "send", "b02-existing-failure")["selection"]
    calls = []
    async def unavailable():
        calls.append("called")
        raise RuntimeError("private-b02-existing-provider-error")
        yield
    _install_provider(monkeypatch, unavailable)
    response = client.post("/api/v1/chat/completions", headers=headers, json={
        "model": "gpt-4o-mini", "conversation_id": cid, "save_to_db": True, "stream": True,
        "messages": [], "tldw_history_server_existing_input_v1": selection})
    assert response.status_code == 502, response.text  # nosec B101 - Failed provider preparation retains the existing HTTP refusal contract.
    assert calls == ["called"]  # nosec B101 - No hidden model retry.
    assert db.get_messages_for_conversation(cid) == [original]  # nosec B101 - No input duplication or false assistant settlement.
    assert "private-b02-existing-provider-error" not in response.text  # nosec B101 - Keep provider details private.
    assert "tldw_history_admission_v1" not in response.text  # nosec B101 - Never invent a new admission for the saved input.


def test_b02_existing_question_capability_refuses_sync_owner(history_api, monkeypatch):
    from tldw_Server_API.app.api.v1.endpoints import character_messages

    client, db, cid, headers = history_api
    question = db.add_message({"id": "b02-sync-question", "conversation_id": cid, "sender": "user", "content": "saved question"})
    original = db.get_message_by_id(question)
    body = capture(client, cid, headers, cursor={"kind": "after_message", "message_id": question}).json()
    monkeypatch.setattr(character_messages, "_active_message_sync_service", lambda *_: object())
    response = client.post(f"/api/v1/chat/conversations/{cid}/history/selection", headers=headers, json={
        "purpose": "send", "view": body["view"], "required_capability": "tldw_history_server_existing_input_v1"})
    assert response.status_code == 409, response.text  # nosec B101 - The new capability cannot bypass Sync settlement ownership.
    assert response.json()["detail"]["code"] == "sync_owner_unsupported"  # nosec B101 - Explicit refusal, never a legacy fallback.
    assert db.get_messages_for_conversation(cid) == [original]  # nosec B101 - Capability probing is read-only.


def test_b02_existing_question_uses_frozen_character_after_live_card_edit(history_api, monkeypatch):
    import json

    from tldw_Server_API.app.api.v1.endpoints import chat
    from tldw_Server_API.app.core.Character_Chat.character_conversation_factory import create_character_conversation

    client, db, _, headers = history_api
    character = db.add_character_card({"name": "B02 frozen character", "description": "FROZEN-B02-CHARACTER", "client_id": "1"})
    cid = create_character_conversation(db, conversation_data={"character_id": character},
        provider="openai", model="gpt-4o-mini", prompt_preset_id="st_default")
    question = db.add_message({"id": "b02-frozen-question", "conversation_id": cid, "sender": "user", "content": "saved question"})
    captured = capture(client, cid, headers, cursor={"kind": "after_message", "message_id": question}).json()
    selection = resolve_history_selection(captured["snapshot"], captured["view"], "send", "b02-frozen-context")["selection"]
    card = db.get_character_card_by_id(character)
    db.update_character_card(character, {"description": "LIVE-B02-CHARACTER-CHANGED"}, expected_version=card["version"])
    calls = []
    def provider(**kwargs):
        calls.append(kwargs)
        yield "data: " + json.dumps({"choices": [{"delta": {"content": "answer"}, "finish_reason": "stop"}]}) + "\n\n"
        yield "data: [DONE]\n\n"
    monkeypatch.setattr(chat, "perform_chat_api_call", provider)
    response = client.post("/api/v1/chat/completions", headers=headers, json={
        "model": "gpt-4o-mini", "conversation_id": cid, "save_to_db": True, "stream": True,
        "messages": [], "tldw_history_server_existing_input_v1": selection})
    assert response.status_code == 200, response.text  # nosec B101 - Saved behavior remains a valid explicit completion context.
    assert len(calls) == 1  # nosec B101 - One provider execution under the frozen character.
    model_context = json.dumps({"messages": calls[0]["messages_payload"], "system": calls[0].get("system_message")})
    assert "FROZEN-B02-CHARACTER" in model_context  # nosec B101 - Required saved character context reaches the model.
    assert "LIVE-B02-CHARACTER-CHANGED" not in model_context  # nosec B101 - No late mutable card lookup changes this answer.
    assert [m["sender"] for m in db.get_messages_for_conversation(cid)] == ["user", "assistant"]  # nosec B101 - Existing question is never readmitted.
