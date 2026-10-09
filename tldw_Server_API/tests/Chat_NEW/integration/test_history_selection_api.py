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
