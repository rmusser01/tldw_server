"""Real storage and HTTP integration with provider, command, macro, and tool doubles."""

import json
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest

from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.api.v1.endpoints import chat as chat_endpoint
from tldw_Server_API.app.core.Chat import chat_service, command_router
from tldw_Server_API.app.core.Chat.tool_auto_exec import ToolExecutionBatchResult, ToolExecutionRecord
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB


@pytest.fixture
def durable_workspace_api(credentialed_test_client, populated_chacha_db, auth_headers):
    client = credentialed_test_client
    db = CharactersRAGDB(
        db_path=populated_chacha_db.db_path_str,
        client_id="1",
        owner_user_id="1",
    )
    workspace_id = "durable-settings-workspace"
    db.upsert_workspace(workspace_id, "Durable settings workspace")
    cid = db.add_conversation({
        "title": "Durable settings",
        "scope_type": "workspace",
        "workspace_id": workspace_id,
    })
    db.upsert_conversation_settings(cid, {
        "provider": "llama.cpp", "model": "previous-local-model", "temperature": 0.25,
    })
    client.app.dependency_overrides[get_chacha_db_for_user] = lambda: db
    try:
        yield client, db, cid, auth_headers, workspace_id
    finally:
        client.app.dependency_overrides.pop(get_chacha_db_for_user, None)
        db.close_all_connections()


@pytest.mark.integration
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("conflict", ["changed-content", "stale-turn", "appended-image"])
def test_rejected_durable_retry_preserves_workspace_settings_and_rows(durable_workspace_api, stream, conflict):
    client, db, cid, headers, workspace_id = durable_workspace_api
    uid = str(uuid4())
    db.insert_or_validate_user_turn(
        cid, uid, "Original question", owner_client_id=db.client_id,
        conversation_context=db.get_conversation_by_id(cid),
    )
    if conflict == "stale-turn":
        db.add_message({"conversation_id": cid, "sender": "user", "content": "Later question"})
    elif conflict == "appended-image":
        db.append_message_image(uid, b"independently appended image", "image/png")
        assert db.get_message_by_id(uid)["image_data"] is None
    before_settings = db.get_conversation_settings(cid)
    before_rows = db.get_messages_for_conversation(cid)
    before_images = db.get_message_images(uid, strict=True)
    before_conversation = db.get_conversation_by_id(cid)

    response = client.post(
        f"/api/v1/chat/completions?scope_type=workspace&workspace_id={workspace_id}",
        headers=headers,
        json={
            "api_provider": "openai", "model": "gpt-4o-mini", "conversation_id": cid,
            "save_to_db": True, "stream": stream, "tldw_turn": {"user_message_id": uid},
            "messages": [{"role": "user", "content": "Changed question" if conflict == "changed-content" else "Original question"}],
        },
    )
    assert response.status_code == 409, response.text
    assert db.get_conversation_settings(cid) == before_settings
    assert db.get_messages_for_conversation(cid) == before_rows
    assert db.get_message_images(uid, strict=True) == before_images
    assert db.get_conversation_by_id(cid) == before_conversation


@pytest.mark.integration
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("turn_mode", ["ordinary", "durable"])
def test_admitted_workspace_turn_retains_explicit_selection(durable_workspace_api, stream, turn_mode):
    client, db, cid, headers, workspace_id = durable_workspace_api
    before = db.get_conversation_settings(cid)
    body = {
        "api_provider": "openai", "model": "gpt-4o-mini", "conversation_id": cid,
        "save_to_db": True, "stream": stream,
        "messages": [{"role": "user", "content": "Original question"}],
    }
    if turn_mode == "durable":
        body["tldw_turn"] = {"user_message_id": str(uuid4())}
    response = client.post(
        f"/api/v1/chat/completions?scope_type=workspace&workspace_id={workspace_id}",
        headers=headers, json=body,
    )
    assert response.status_code == 200, response.text
    after = db.get_conversation_settings(cid)
    assert after["settings"] == {**before["settings"], "provider": "openai", "model": "gpt-4o-mini"}
    assert after["settings_version"] == before["settings_version"] + 1


@pytest.mark.integration
@pytest.mark.parametrize("stream", [False, True])
def test_durable_receipts_and_assistant_parent_on_repeated_http_attempts(
    credentialed_test_client,
    populated_chacha_db,
    auth_headers,
    stream,
):
    client, db = credentialed_test_client, populated_chacha_db
    cid = db.add_conversation({"title": "Durable HTTP"})
    uid = str(uuid4())
    client.app.dependency_overrides[get_chacha_db_for_user] = lambda: db
    try:
        for _ in range(2):
            response = client.post(
                "/api/v1/chat/completions",
                headers=auth_headers,
                json={
                    "model": "gpt-4o-mini",
                    "conversation_id": cid,
                    "save_to_db": True,
                    "stream": stream,
                    "tldw_turn": {"user_message_id": uid},
                    "messages": [
                        {"role": "system", "content": "Private retrieval for this attempt"},
                        {"role": "user", "content": "Original question"},
                    ],
                },
            )
            assert response.status_code == 200, response.text
            payloads = (
                [
                    json.loads(line[6:])
                    for line in response.text.splitlines()
                    if line.startswith("data: ") and line[6:] != "[DONE]"
                ]
                if stream
                else [response.json()]
            )
            assert any(payload.get("tldw_user_message_id") == uid for payload in payloads)
            if stream:
                first_receipt = next(payload for payload in payloads if payload.get("tldw_user_message_id"))
                assert "tldw_message_id" not in first_receipt
            assistant_id = next(
                payload["tldw_message_id"] for payload in reversed(payloads) if payload.get("tldw_message_id")
            )
            assert db.get_message_by_id(assistant_id)["parent_message_id"] == uid
        rows = db.get_messages_for_conversation(cid)
        assert [row["id"] for row in rows if row["sender"] == "user"] == [uid]
        assert len(rows) == 3
    finally:
        client.app.dependency_overrides.pop(get_chacha_db_for_user, None)


@pytest.mark.integration
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("text_parts", [False, True])
@pytest.mark.parametrize(
    ("text", "injection_mode"),
    [
        ("/time", "system"),
        ("/time", "preface"),
        ("/time", "replace"),
        ("/wrapup", "system"),
        ("/unknown_macro", "system"),
    ],
)
def test_durable_turn_rejects_enabled_slash_candidates_before_execution(
    credentialed_test_client,
    populated_chacha_db,
    auth_headers,
    monkeypatch,
    stream,
    text_parts,
    text,
    injection_mode,
):
    client, db = credentialed_test_client, populated_chacha_db
    cid, uid = db.add_conversation({"title": "Rejected slash turn"}), str(uuid4())
    monkeypatch.setenv("CHAT_COMMANDS_ENABLED", "1")
    dispatch = AsyncMock(return_value=command_router.CommandResult(
        ok=True, command="time", content="Command output", metadata={},
    ))
    macro_service = Mock(return_value=object())
    macro_item = Mock(return_value=object())
    completion = {
        "id": "chatmacro-test",
        "object": "chat.completion",
        "created": 1,
        "model": "gpt-4o-mini",
        "choices": [{"message": {"role": "assistant", "content": "Answer", "metadata": {}}}],
    }
    macro_run = Mock(return_value=completion)
    provider = Mock(wraps=chat_endpoint.perform_chat_api_call)
    monkeypatch.setattr(command_router, "async_dispatch_command", dispatch)
    monkeypatch.setattr(chat_endpoint, "_build_chat_macro_service", macro_service)
    monkeypatch.setattr(chat_endpoint, "_find_enabled_chat_macro", macro_item)
    monkeypatch.setattr(chat_endpoint, "_create_chat_macro_run_payload", macro_run)
    monkeypatch.setattr(chat_endpoint, "perform_chat_api_call", provider)
    content = [{"type": "text", "text": text}, {"type": "text", "text": "More context"}] if text_parts else text
    client.app.dependency_overrides[get_chacha_db_for_user] = lambda: db
    try:
        response = client.post(
            "/api/v1/chat/completions",
            headers=auth_headers,
            json={
                "model": "gpt-4o-mini",
                "conversation_id": cid,
                "save_to_db": True,
                "stream": stream,
                "tldw_turn": {"user_message_id": uid},
                "slash_command_injection_mode": injection_mode,
                "messages": [{"role": "user", "content": content}],
            },
        )
        assert response.status_code == 422, response.text
        detail = response.json()["detail"]
        assert "tldw_turn" in detail["message"] and "slash" in detail["message"].lower()
        dispatch.assert_not_called()
        macro_service.assert_not_called()
        macro_item.assert_not_called()
        macro_run.assert_not_called()
        provider.assert_not_called()
        assert db.get_message_by_id(uid) is None
        assert db.get_messages_for_conversation(cid) == []
    finally:
        client.app.dependency_overrides.pop(get_chacha_db_for_user, None)


@pytest.mark.integration
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("text", ["/time", "/wrapup", "/unknown_macro"])
def test_durable_turn_accepts_disabled_slash_commands_as_literal_text(
    credentialed_test_client,
    populated_chacha_db,
    auth_headers,
    monkeypatch,
    stream,
    text,
):
    client, db = credentialed_test_client, populated_chacha_db
    cid, uid = db.add_conversation({"title": "Literal slash turn"}), str(uuid4())
    monkeypatch.setenv("CHAT_COMMANDS_ENABLED", "0")
    client.app.dependency_overrides[get_chacha_db_for_user] = lambda: db
    try:
        response = client.post(
            "/api/v1/chat/completions",
            headers=auth_headers,
            json={
                "model": "gpt-4o-mini",
                "conversation_id": cid,
                "save_to_db": True,
                "stream": stream,
                "tldw_turn": {"user_message_id": uid},
                "messages": [{"role": "user", "content": text}],
            },
        )
        assert response.status_code == 200, response.text
        assert db.get_message_by_id(uid)["content"] == text
        assert [row["id"] for row in db.get_messages_for_conversation(cid) if row["sender"] == "user"] == [uid]
    finally:
        client.app.dependency_overrides.pop(get_chacha_db_for_user, None)


@pytest.mark.integration
@pytest.mark.parametrize("turn_mode", ["ordinary", "durable", "branch"])
def test_nonstream_tool_auto_continuation_preserves_turn_parent(
    credentialed_test_client,
    populated_chacha_db,
    auth_headers,
    monkeypatch,
    turn_mode,
):
    client, db = credentialed_test_client, populated_chacha_db
    cid, uid = db.add_conversation({"title": "Tool continuation"}), str(uuid4())
    monkeypatch.setenv("CHAT_AUTO_EXECUTE_TOOLS", "1")
    monkeypatch.setenv("CHAT_TOOL_AUTO_CONTINUE_ONCE", "1")
    tool_call = {"id": "call-1", "type": "function", "function": {"name": "notes.search", "arguments": "{}"}}
    initial = {"choices": [{"message": {"role": "assistant", "content": "Searching", "tool_calls": [tool_call]}, "finish_reason": "tool_calls"}]}
    followup = {"choices": [{"message": {"role": "assistant", "content": "Final answer"}, "finish_reason": "stop"}]}
    monkeypatch.setattr(chat_endpoint, "perform_chat_api_call", Mock(return_value=initial))
    monkeypatch.setattr(chat_service, "perform_chat_api_call_async", AsyncMock(return_value=followup))
    monkeypatch.setattr(chat_service, "execute_assistant_tool_calls", AsyncMock(return_value=ToolExecutionBatchResult(
        requested_calls=1,
        processed_calls=1,
        execution_attempts=1,
        executed_calls=1,
        truncated=False,
        results=[ToolExecutionRecord(tool_call_id="call-1", tool_name="notes.search", ok=True, content="Tool result")],
    )))
    body = {
        "model": "gpt-4o-mini",
        "conversation_id": cid,
        "save_to_db": True,
        "messages": [{"role": "user", "content": "Search my notes"}],
    }
    parent_id = None if turn_mode == "ordinary" else uid
    if turn_mode == "durable":
        body["tldw_turn"] = {"user_message_id": uid}
    elif turn_mode == "branch":
        db.add_message({"id": uid, "conversation_id": cid, "sender": "user", "content": "Earlier question"})
        body["tldw_continuation"] = {"from_message_id": uid, "mode": "branch"}
    client.app.dependency_overrides[get_chacha_db_for_user] = lambda: db
    try:
        response = client.post("/api/v1/chat/completions", headers=auth_headers, json=body)
        assert response.status_code == 200, response.text
        payload = response.json()
        assert payload["tldw_tool_auto_continue"] == {"attempted": True, "succeeded": True}
        if turn_mode == "ordinary":
            parent_id = next(
                row["id"] for row in db.get_messages_for_conversation(cid) if row["sender"] == "user"
            )
        final = db.get_message_by_id(payload["tldw_message_id"])
        assert final["content"] == "Final answer"
        assert final["parent_message_id"] == parent_id
        assistants = [row for row in db.get_messages_for_conversation(cid) if row["sender"] == final["sender"]]
        assert len(assistants) == 2
        assert all(row["parent_message_id"] == parent_id for row in assistants)
    finally:
        client.app.dependency_overrides.pop(get_chacha_db_for_user, None)


@pytest.mark.integration
def test_empty_provider_failure_then_retry_keeps_committed_user(
    credentialed_test_client,
    populated_chacha_db,
    auth_headers,
    monkeypatch,
):
    client, db = credentialed_test_client, populated_chacha_db
    cid, uid = db.add_conversation({"title": "Failed HTTP"}), str(uuid4())
    calls = []

    def provider(**kwargs):
        calls.append(kwargs)
        text = "" if len(calls) == 1 else "Recovered answer"
        return {"choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}]}

    monkeypatch.setattr(chat_endpoint, "perform_chat_api_call", provider)
    client.app.dependency_overrides[get_chacha_db_for_user] = lambda: db
    body = {
        "model": "gpt-4o-mini",
        "conversation_id": cid,
        "save_to_db": True,
        "tldw_turn": {"user_message_id": uid},
        "messages": [{"role": "user", "content": "Original question"}],
    }
    try:
        failed = client.post("/api/v1/chat/completions", headers=auth_headers, json=body)
        assert failed.status_code >= 400
        assert db.get_message_by_id(uid)["content"] == "Original question"
        recovered = client.post("/api/v1/chat/completions", headers=auth_headers, json=body)
        assert recovered.status_code == 200, recovered.text
        assert recovered.json()["tldw_user_message_id"] == uid
        assert len(db.get_messages_for_conversation(cid)) == 2
        assert all("tldw_turn" not in call for call in calls)
    finally:
        client.app.dependency_overrides.pop(get_chacha_db_for_user, None)
