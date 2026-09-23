"""Authenticated native history capture, admission and settlement routes."""
import pytest

from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.core.Chat.history_selection import resolve_history_selection


@pytest.fixture
def history_api(credentialed_test_client, populated_chacha_db, auth_headers):
    client, db = credentialed_test_client, populated_chacha_db
    client.app.dependency_overrides[get_chacha_db_for_user] = lambda: db
    cid = db.add_conversation({"character_id": None, "title": "H1 API", "client_id": "1"})
    yield client, db, cid, auth_headers
    client.app.dependency_overrides.pop(get_chacha_db_for_user, None)


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


def test_legacy_review_returns_complete_bound_source_and_sync_rejects(history_api, monkeypatch):
    from tldw_Server_API.app.api.v1.endpoints import character_messages
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
    monkeypatch.setattr(character_messages, "_active_message_sync_service", lambda *_: object())
    response = capture(client, cid, headers)
    assert response.status_code == 409
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


def test_versioned_workspace_scope_and_stale_selection_do_not_update_defaults(history_api):
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
    rejected = client.post("/api/v1/chat/completions", headers=headers, json=payload)
    assert rejected.status_code == 404, rejected.text
    assert db.get_conversation_settings(cid) == before
    assert db.count_messages_for_conversation(cid) == 0
    # Owner context drift must reject before saving explicitly requested defaults.
    db.upsert_conversation_settings(cid, {"model": "changed"})
    changed = db.get_conversation_settings(cid)
    rejected = client.post("/api/v1/chat/completions" + query, headers=headers, json=payload)
    assert rejected.status_code == 409, rejected.text
    assert db.get_conversation_settings(cid) == changed
    assert db.count_messages_for_conversation(cid) == 0
    refreshed = client.post(f"/api/v1/chat/conversations/{cid}/history/selection{query}", headers=headers, json={"purpose": "send", "view": body["view"]}).json()
    payload["tldw_history_selection_v1"] = resolve_history_selection(refreshed["snapshot"], refreshed["view"], "send", "new-model")["selection"]
    from tldw_Server_API.app.core.Chat.rate_limiter import initialize_rate_limiter
    initialize_rate_limiter()
    success = client.post("/api/v1/chat/completions" + query, headers=headers, json=payload)
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


@pytest.mark.parametrize("kind", ["template", "world_books", "skills"])
def test_unsupported_saved_context_rejects_before_provider_or_admission(history_api, monkeypatch, kind):
    from tldw_Server_API.app.api.v1.endpoints import chat as endpoint
    client, db, cid, headers = history_api
    body = capture(client, cid, headers).json()
    selection = resolve_history_selection(body["snapshot"], body["view"], "send", "context")["selection"]
    def forbidden(*args, **kwargs):
        raise AssertionError("unsupported context must not reach provider or append")
    monkeypatch.setattr(db, "append_selected_history_inputs", forbidden)
    monkeypatch.setattr(endpoint, "perform_chat_api_call", forbidden)
    request = {"model": "gpt-4o-mini", "conversation_id": cid, "save_to_db": True,
        "messages": [{"role": "user", "content": "hello"}], "tldw_history_selection_v1": selection}
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
    assert response.json()["detail"]["status"] == "unsupported_history_capability"
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


def test_sync_owner_rejects_admission_and_completion_before_mutation(history_api, monkeypatch):
    from tldw_Server_API.app.api.v1.endpoints import character_messages
    client, db, cid, headers = history_api
    body = capture(client, cid, headers).json()
    selection = resolve_history_selection(body["snapshot"], body["view"], "send", "sync")["selection"]
    monkeypatch.setattr(character_messages, "_active_message_sync_service", lambda *_: object())
    response = client.post(f"/api/v1/chats/{cid}/messages", headers=headers, json={
        "id": "unsupported-sync", "role": "user", "content": "hello", "tldw_history_selection_v1": selection})
    assert response.status_code == 409, response.text
    response = client.post("/api/v1/chat/completions", headers=headers, json={"api_provider": "openai", "model": "gpt-4o-mini",
        "conversation_id": cid, "save_to_db": True, "messages": [{"role": "user", "content": "hello"}],
        "tldw_history_selection_v1": selection})
    assert response.status_code == 409, response.text
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
