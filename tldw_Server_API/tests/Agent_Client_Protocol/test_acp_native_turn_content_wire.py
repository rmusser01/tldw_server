"""Native notifications through actual Go/stdin and owned cookie endpoints."""
from __future__ import annotations

import copy

import pytest

from tldw_Server_API.app.api.v1.endpoints import agent_client_protocol as endpoints
from tldw_Server_API.tests.Agent_Client_Protocol.test_acp_prompt_permission_wire import ORIGINAL, receive_until
from tldw_Server_API.tests.Agent_Client_Protocol.test_acp_prompt_permission_wire import (
    compiled_runner as _compiled_runner,
)
from tldw_Server_API.tests.Agent_Client_Protocol.test_acp_prompt_permission_wire import (
    owned_runtime as _owned_runtime,
)
from tldw_Server_API.tests.Agent_Client_Protocol.test_acp_prompt_permission_wire import (
    private_paths as _private_paths,
)
from tldw_Server_API.tests.Agent_Client_Protocol.test_acp_prompt_permission_wire import (
    single_user_cookie_client as _cookie_client,
)

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]
compiled_runner = _compiled_runner
owned_runtime = _owned_runtime
private_paths = _private_paths
single_user_cookie_client = _cookie_client

FRAGMENTS = ["  CONTROLLED native ", "summary.\nExact.  "]
TEXT = "".join(FRAGMENTS)
USAGE = {"input_tokens": 2, "output_tokens": 3}

# No model, README read or tool action: this peer exercises native notifications
# and a legitimate stop-only ACP result. The marker is functional test input.
NATIVE_PEER = r'''
import json
import os
import pathlib
import sys
import threading
import time

session = "native-turn-" + str(os.getpid())
marker = pathlib.Path(sys.argv[1]).parent / "emit-native-late"
lock = threading.Lock()

def send(payload):
    with lock:
        sys.stdout.write(json.dumps({"jsonrpc": "2.0", **payload}) + "\n")
        sys.stdout.flush()

def update(kind, content=None, target=None, **extra):
    payload = {"sessionUpdate": kind, **extra}
    if content is not None:
        payload["content"] = {"type": "text", "text": content}
    send({"method": "session/update", "params": {"sessionId": target or session, "update": payload}})

def late():
    for _ in range(1000):
        if marker.exists():
            update("agent_message_chunk", "MUST_NOT_LEAK_LATE")
            update("usage_update", used=0, size=200000, cost={"amount": 0}, late_probe_complete=True)
            return
        time.sleep(0.01)

for line in sys.stdin:
    request = json.loads(line)
    method = request.get("method")
    if method == "initialize":
        send({"id": request["id"], "result": {"protocolVersion": 1, "agentCapabilities": {}}})
    elif method == "session/new":
        send({"id": request["id"], "result": {"sessionId": session}})
    elif method == "session/prompt":
        prompt = request["params"]["prompt"]
        action = prompt[-1].get("text", "")
        if action.startswith("spoof:"):
            update("agent_message_chunk", "MUST_NOT_LEAK_FOREIGN", target=action[6:])
        elif action == "overflow":
            for _ in range(35):
                update("agent_message_chunk", "x" * 32768)
        elif action == "thought-only":
            update("user_message_chunk", "user echo")
            update("agent_thought_chunk", "private thinking, not an answer")
            update("tool_call", toolCallId="controlled", title="No execution", status="completed")
        elif action != "empty":
            for text in ["  CONTROLLED native ", "summary.\nExact.  "]:
                update("agent_message_chunk", text)
        update("usage_update", used=0, size=200000, cost={"amount": 0})
        if action == "fail":
            send({"id": request["id"], "error": {"code": -32603, "message": "controlled failure"}})
        else:
            result = {"stopReason": "end_turn", "usage": {"input_tokens": 2, "output_tokens": 3},
                "_meta": {"tldw-test/received-prompt": prompt}}
            if action == "legacy":
                result["content"] = [{"type": "text", "text": "Legacy final only"}]
            if action == "cancelled":
                result["stopReason"] = "cancelled"
            send({"id": request["id"], "result": result})
            if action == "late":
                threading.Thread(target=late, daemon=True).start()
    elif method in {"session/close", "_tldw/session/close"}:
        send({"id": request["id"], "result": {}})
    elif method == "session/cancel":
        pass
    elif request.get("id") is not None:
        send({"id": request["id"], "error": {"code": -32601, "message": "method not found"}})
'''


@pytest.fixture(autouse=True)
def private_jobs_path(tmp_path, monkeypatch):
    monkeypatch.setenv("JOBS_DB_PATH", str(tmp_path / "jobs.db"))


def post(client, session_id, text):
    return client.post("/api/v1/acp/sessions/prompt",
        json={"session_id": session_id, "prompt": [{"role": "user", "content": text}]},
        headers={"X-CSRF-Token": client.cookies["csrf_token"]})


def record(client, session_id):
    async def read():
        return await (await endpoints.get_acp_session_store()).get_session(session_id)
    return client.portal.call(read)


@pytest.mark.parametrize("owned_runtime", [NATIVE_PEER], indirect=True)
def test_native_stop_only_rest_saves_exact_text_and_unmodified_raw_result(owned_runtime):
    client, _runner, session_id, output = owned_runtime
    response = post(client, session_id, "native")
    assert response.status_code == 200, response.json()  # nosec B101 # pytest regression oracle, not production authorization
    raw = copy.deepcopy(response.json()["raw_result"])
    assert set(raw) == {"stopReason", "usage", "_meta"}  # nosec B101 # pytest regression oracle, not production authorization
    saved = record(client, session_id)
    assert saved.messages[0]["raw_prompt"] == {"role": "user", "content": "native"}  # nosec B101 # pytest regression oracle, not production authorization
    assert saved.messages[1]["content"] == TEXT  # nosec B101 # pytest regression oracle, not production authorization
    assert saved.messages[1]["raw_result"] == raw  # nosec B101 # pytest regression oracle, not production authorization
    assert saved.bootstrap_ready is True  # nosec B101 # pytest regression oracle, not production authorization
    assert saved.usage.total_tokens == 5  # nosec B101 # pytest regression oracle, not production authorization
    detail = client.get(f"/api/v1/acp/sessions/{session_id}/detail")
    assert detail.status_code == 200  # nosec B101 # pytest regression oracle, not production authorization
    assert detail.json()["messages"][1]["content"] == TEXT  # nosec B101 # pytest regression oracle, not production authorization
    assert output.read_text() == ORIGINAL  # nosec B101 # pytest regression oracle, not production authorization


@pytest.mark.parametrize("owned_runtime", [NATIVE_PEER], indirect=True)
def test_native_websocket_kinds_and_original_nested_payload_arrive_before_stop(owned_runtime):
    client, _runner, session_id, _output = owned_runtime
    with client.websocket_connect(f"/api/v1/acp/sessions/{session_id}/stream", headers={"Origin": "http://localhost:3000"}) as ws:
        assert ws.receive_json()["type"] == "connected"  # nosec B101 # pytest regression oracle, not production authorization
        ws.send_json({"type": "prompt", "prompt": [{"role": "user", "content": "native"}]})
        frames = receive_until(ws, lambda items: any(item.get("type") == "prompt_complete" for item in items))
        updates = [frame for frame in frames if frame.get("type") == "update"]
        assert [frame["update_type"] for frame in updates] == ["agent_message_chunk", "agent_message_chunk", "usage_update"]  # nosec B101 # pytest regression oracle, not production authorization
        assert [frame["data"]["update"]["content"]["text"] for frame in updates[:2]] == FRAGMENTS  # nosec B101 # pytest regression oracle, not production authorization
        assert all(frame["data"]["sessionId"] == session_id for frame in updates)  # nosec B101 # pytest regression oracle, not production authorization
        assert set(frames[-1]["raw_result"]) == {"stopReason", "usage", "_meta"}  # nosec B101 # pytest regression oracle, not production authorization


@pytest.mark.parametrize("owned_runtime", [NATIVE_PEER], indirect=True)
def test_empty_successor_cannot_reuse_previous_native_text(owned_runtime):
    client, _runner, session_id, _output = owned_runtime
    assert post(client, session_id, "native").status_code == 200  # nosec B101 # pytest regression oracle, not production authorization
    assert post(client, session_id, "empty").status_code == 200  # nosec B101 # pytest regression oracle, not production authorization
    saved = record(client, session_id)
    assert saved.messages[1]["content"] == TEXT  # nosec B101 # pytest regression oracle, not production authorization
    assert saved.messages[3]["content"] is None  # nosec B101 # pytest regression oracle, not production authorization


@pytest.mark.parametrize("owned_runtime", [NATIVE_PEER], indirect=True)
def test_final_content_takes_precedence_without_duplicate_native_text(owned_runtime):
    client, _runner, session_id, _output = owned_runtime
    response = post(client, session_id, "legacy")
    assert response.status_code == 200  # nosec B101 # pytest regression oracle, not production authorization
    saved = record(client, session_id)
    assert saved.messages[1]["content"] == "Legacy final only"  # nosec B101 # pytest regression oracle, not production authorization
    assert saved.messages[1]["raw_result"] == response.json()["raw_result"]  # nosec B101 # pytest regression oracle, not production authorization


@pytest.mark.parametrize("owned_runtime", [NATIVE_PEER], indirect=True)
def test_nonassistant_native_updates_do_not_fabricate_an_answer(owned_runtime):
    client, _runner, session_id, _output = owned_runtime
    response = post(client, session_id, "thought-only")
    assert response.status_code == 200  # nosec B101 # pytest regression oracle, not production authorization
    assert record(client, session_id).messages[1]["content"] is None  # nosec B101 # pytest regression oracle, not production authorization


@pytest.mark.parametrize("owned_runtime", [NATIVE_PEER], indirect=True)
def test_error_does_not_persist_partial_text_or_leak_into_empty_next_turn(owned_runtime):
    client, _runner, session_id, _output = owned_runtime
    assert post(client, session_id, "fail").status_code == 502  # nosec B101 # pytest regression oracle, not production authorization
    assert record(client, session_id).messages == []  # nosec B101 # pytest regression oracle, not production authorization
    assert post(client, session_id, "empty").status_code == 200  # nosec B101 # pytest regression oracle, not production authorization
    assert record(client, session_id).messages[1]["content"] is None  # nosec B101 # pytest regression oracle, not production authorization


@pytest.mark.parametrize("owned_runtime", [NATIVE_PEER], indirect=True)
def test_native_saved_text_is_real_fork_bootstrap_history(owned_runtime):
    client, runner, session_id, _output = owned_runtime
    assert post(client, session_id, "native").status_code == 200  # nosec B101 # pytest regression oracle, not production authorization
    fork = client.post(f"/api/v1/acp/sessions/{session_id}/fork", json={"message_index": 1},
        headers={"X-CSRF-Token": client.cookies["csrf_token"]})
    assert fork.status_code == 200, fork.json()  # nosec B101 # pytest regression oracle, not production authorization
    fork_id = fork.json()["session_id"]
    try:
        response = post(client, fork_id, "followup")
        assert response.status_code == 200, response.json()  # nosec B101 # pytest regression oracle, not production authorization
        received = response.json()["raw_result"]["_meta"]["tldw-test/received-prompt"]
        assert [block["text"] for block in received] == ["native", TEXT, "followup"]  # nosec B101 # pytest regression oracle, not production authorization
        assert [block["_meta"]["tldw-server/legacy-message"]["role"] for block in received] == ["user", "assistant", "user"]  # nosec B101 # pytest regression oracle, not production authorization
        assert record(client, fork_id).needs_bootstrap is False  # nosec B101 # pytest regression oracle, not production authorization
    finally:
        client.portal.call(runner.close_session, fork_id)


@pytest.mark.parametrize("owned_runtime", [NATIVE_PEER], indirect=True)
def test_actual_downstream_cannot_spoof_another_known_session_notification(owned_runtime):
    client, runner, session_id, output = owned_runtime
    other_id = client.portal.call(runner.create_session, str(output.parent), None, "controlled", 1)
    try:
        assert post(client, session_id, "spoof:" + other_id).status_code == 200  # nosec B101 # pytest regression oracle, not production authorization
        assert runner.pop_updates(other_id) == []  # nosec B101 # pytest regression oracle, not production authorization
    finally:
        client.portal.call(runner.close_session, other_id)


@pytest.mark.parametrize("owned_runtime", [NATIVE_PEER], indirect=True)
def test_late_native_text_without_active_prompt_is_not_a_new_assistant_update(owned_runtime):
    client, _runner, session_id, output = owned_runtime
    assert post(client, session_id, "late").status_code == 200  # nosec B101 # pytest regression oracle, not production authorization
    with client.websocket_connect(f"/api/v1/acp/sessions/{session_id}/stream", headers={"Origin": "http://localhost:3000"}) as ws:
        assert ws.receive_json()["type"] == "connected"  # nosec B101 # pytest regression oracle, not production authorization
        output.parent.joinpath("emit-native-late").write_text("emit")
        frames = receive_until(ws, lambda items: any((item.get("data") or {}).get("update", {}).get("late_probe_complete") for item in items))
        assert [frame["update_type"] for frame in frames] == ["usage_update"]  # nosec B101 # pytest regression oracle, not production authorization
    assert post(client, session_id, "empty").status_code == 200  # nosec B101 # pytest regression oracle, not production authorization
    assert record(client, session_id).messages[-1]["content"] is None  # nosec B101 # pytest regression oracle, not production authorization


@pytest.mark.parametrize("owned_runtime", [NATIVE_PEER], indirect=True)
def test_native_aggregate_overflow_refuses_after_actual_reply_without_partial_row(owned_runtime):
    client, runner, session_id, _output = owned_runtime
    response = post(client, session_id, "overflow")
    assert response.status_code == 502  # nosec B101 # pytest regression oracle, not production authorization
    assert response.json()["detail"] == "ACP prompt failed"  # nosec B101 # pytest regression oracle, not production authorization
    assert record(client, session_id).messages == []  # nosec B101 # pytest regression oracle, not production authorization
    assert session_id not in runner._active_prompts  # nosec B101 # pytest regression oracle, not production authorization
    assert post(client, session_id, "empty").status_code == 200  # nosec B101 # pytest regression oracle, not production authorization


@pytest.mark.parametrize("owned_runtime", [NATIVE_PEER], indirect=True)
def test_native_cancelled_raw_result_does_not_publish_partial_answer(owned_runtime):
    client, _runner, session_id, _output = owned_runtime
    response = post(client, session_id, "cancelled")
    assert response.status_code == 200  # nosec B101 # pytest regression oracle, not production authorization
    assert response.json()["raw_result"]["stopReason"] == "cancelled"  # nosec B101 # pytest regression oracle, not production authorization
    saved = record(client, session_id)
    assert saved.messages[1]["content"] is None  # nosec B101 # pytest regression oracle, not production authorization
    assert saved.messages[1]["raw_result"] == response.json()["raw_result"]  # nosec B101 # pytest regression oracle, not production authorization
    assert saved.bootstrap_ready is False  # nosec B101 # pytest regression oracle, not production authorization


@pytest.mark.parametrize("owned_runtime", [NATIVE_PEER], indirect=True)
def test_raw_result_only_actual_runner_call_has_no_transcript_capture_limit(owned_runtime):
    client, runner, session_id, _output = owned_runtime
    result = client.portal.call(runner.prompt, session_id, [{"role": "user", "content": "overflow"}])
    assert result["stopReason"] == "end_turn"  # nosec B101 # pytest regression oracle, not production authorization
    assert set(result) == {"stopReason", "usage", "_meta"}  # nosec B101 # pytest regression oracle, not production authorization
    assert record(client, session_id).messages == []  # nosec B101 # pytest regression oracle, not production authorization
    assert session_id not in runner._active_prompts  # nosec B101 # pytest regression oracle, not production authorization


@pytest.mark.parametrize("owned_runtime", [NATIVE_PEER], indirect=True)
@pytest.mark.parametrize("action,expected", [("native", TEXT), ("legacy", "Legacy final only")])
def test_actual_workflow_adapter_persists_native_text_and_legacy_precedence(owned_runtime, monkeypatch, action, expected):
    from tldw_Server_API.app.core.Workflows.adapters.integration import acp as adapter
    client, runner, session_id, _output = owned_runtime
    async def get_runner():
        return runner
    monkeypatch.setattr(adapter, "get_runner_client", get_runner)
    monkeypatch.setattr(adapter, "get_acp_session_store", endpoints.get_acp_session_store)
    config = {"stage": "readonly", "session_id": session_id, "create_session": False, "prompt": action}
    result = client.portal.call(adapter.run_acp_stage_adapter, config, {"user_id": 1})
    assert result["status"] == "ok"  # nosec B101 # pytest regression oracle, not production authorization
    assert result["text"] == expected  # nosec B101 # pytest regression oracle, not production authorization
    saved = record(client, session_id)
    assert saved.messages[1]["content"] == expected  # nosec B101 # pytest regression oracle, not production authorization
    assert saved.messages[1]["raw_result"] == result["response"]  # nosec B101 # pytest regression oracle, not production authorization
    assert result["response"]["stopReason"] == "end_turn"  # nosec B101 # pytest regression oracle, not production authorization
    assert "text" not in result["response"]  # nosec B101 # pytest regression oracle, not production authorization
    before = copy.deepcopy(saved.messages)
    denied = client.portal.call(adapter.run_acp_stage_adapter, config, {"user_id": 2})
    assert denied["error"] == "session_access_denied"  # nosec B101 # pytest regression oracle, not production authorization
    assert denied["status"] == "error"  # nosec B101 # pytest regression oracle, not production authorization
    assert record(client, session_id).messages == before  # nosec B101 # pytest regression oracle, not production authorization
