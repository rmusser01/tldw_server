"""Prompt/permission control through the compiled runner and authenticated socket."""

from __future__ import annotations

import asyncio
import json
import subprocess  # nosec B404 - fixed normal project Go build in this controlled fixture
import sys
from pathlib import Path

import anyio
import pytest

from tldw_Server_API.app.api.v1.endpoints import agent_client_protocol as endpoints
from tldw_Server_API.app.core.Agent_Client_Protocol.config import ACPRunnerConfig
from tldw_Server_API.app.core.Agent_Client_Protocol.runner_client import ACPRunnerClient
from tldw_Server_API.app.core.Agent_Client_Protocol.stdio_client import ACPResponseError
from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
from tldw_Server_API.app.core.AuthNZ.repos.mcp_hub_repo import McpHubRepo
from tldw_Server_API.app.core.DB_Management.ACP_Sessions_DB import ACPSessionsDB
from tldw_Server_API.app.core.Utils.fastapi_routes import iter_served_routes
from tldw_Server_API.app.services import admin_acp_sessions_service
from tldw_Server_API.app.services.admin_acp_sessions_service import ACPSessionStore
from tldw_Server_API.tests.AuthNZ.integration.test_single_user_cookie_session import (
    _mint,
)
from tldw_Server_API.tests.AuthNZ.integration.test_single_user_cookie_session import (
    single_user_cookie_client as _single_user_cookie_client_fixture,
)

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]
single_user_cookie_client = _single_user_cookie_client_fixture
REPO_ROOT = Path(__file__).resolve().parents[3]
ORIGINAL = "ORIGINAL_WIRE_INPUT\n"
REPLACEMENT = "APPROVED_WIRE_OUTPUT\n"

# Only this controlled stdlib peer runs downstream. It makes actual ACP tool
# requests to the Go runner and reports the real tool response, never app results.
DOWNSTREAM = r'''
import json
import sys

session = "controlled-owned-session"
pending = {}
next_tool = 1000

def send(payload):
    sys.stdout.write(json.dumps({"jsonrpc": "2.0", **payload}) + "\n")
    sys.stdout.flush()

for line in sys.stdin:
    message = json.loads(line)
    method = message.get("method")
    params = message.get("params") or {}
    if method == "initialize":
        send({"id": message["id"], "result": {"protocolVersion": 1, "agentCapabilities": {}}})
    elif method == "session/new":
        send({"id": message["id"], "result": {"sessionId": session}})
    elif method == "session/prompt":
        next_tool += 1
        native_permission = any(item.get("text") == "Foreign permission" for item in params.get("prompt", []))
        pending[next_tool] = (message["id"], native_permission)
        if native_permission:
            send({"id": next_tool, "method": "session/request_permission", "params": {
                "sessionId": "foreign-permission-session", "tool": {
                    "name": "fs/write_text_file", "input": {
                        "path": sys.argv[1], "content": "APPROVED_WIRE_OUTPUT\n"}}}})
        else:
            send({"id": next_tool, "method": "fs/write_text_file", "params": {
                "sessionId": session, "path": sys.argv[1], "content": "APPROVED_WIRE_OUTPUT\n"}})
    elif method == "session/cancel":
        send({"method": "session/update", "params": {"sessionId": session,
            "update": {"sessionUpdate": "agent_message_chunk", "content": {
                "type": "text", "text": "CONTROLLED_STOP_RECEIVED"}}}})
    elif method in {"session/close", "_tldw/session/close"}:
        send({"id": message["id"], "result": {}})
    elif method is None and message.get("id") in pending:
        prompt_id, native_permission = pending.pop(message["id"])
        refused = bool(message.get("error"))
        if native_permission:
            refused = refused or (message.get("result") or {}).get("outcome", {}).get("outcome") != "approved"
        send({"id": prompt_id, "result": {"stopReason": "end_turn",
            "toolRefused": refused}})
    elif message.get("id") is not None:
        send({"id": message["id"], "error": {"code": -32601, "message": "method not found"}})
'''


@pytest.fixture(scope="module")
def compiled_runner(tmp_path_factory):
    """Build the actual installed Go project normally, without cache overrides."""
    binary = tmp_path_factory.mktemp("acp-compiled-runner") / "tldw-agent-acp"
    subprocess.run(  # nosec B603 B607 - fixed project Go build, no untrusted command
        ["go", "build", "-o", str(binary), "./cmd/tldw-agent-acp"],
        cwd=REPO_ROOT / "tools/tldw-agent",
        check=True,
        capture_output=True,
        text=True,
        timeout=300,
    )
    return binary


@pytest.fixture(autouse=True)
def private_paths(tmp_path, monkeypatch):
    """Keep all functional test storage inside this test's temporary directory."""
    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path / "users"))
    monkeypatch.setenv("SYSTEM_LOG_FILE_PATH", str(tmp_path / "system.log"))
    monkeypatch.setenv("MONITORING_ALERTS_DB", str(tmp_path / "alerts.db"))
    monkeypatch.setenv("ACP_SESSIONS_DB_PATH", str(tmp_path / "sessions.db"))


@pytest.fixture
def owned_runtime(single_user_cookie_client, compiled_runner, tmp_path, monkeypatch, request):
    """Wire real private storage/runner singletons into the actual cookie app."""
    client, api_key = single_user_cookie_client
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    output = workspace / "answer.txt"
    output.write_text(ORIGINAL)
    peer = tmp_path / "downstream.py"
    peer.write_text(getattr(request, "param", DOWNSTREAM))
    home = tmp_path / "home"
    config_dir = home / ".tldw-agent"
    config_dir.mkdir(parents=True)
    config_dir.joinpath("config.yaml").write_text(
        "agents:\n  default: controlled\n  agents:\n    - type: controlled\n"
        "      entrypoint_strategy: native_acp\n"
        f"      command: {json.dumps(sys.executable)}\n"
        f"      args: {json.dumps([str(peer), str(output)])}\n"
        "security:\n  require_approval_for_writes: true\n  require_approval_for_exec: true\n"
    )
    runner = ACPRunnerClient(ACPRunnerConfig(
        command=str(compiled_runner), env={"HOME": str(home)}, cwd=str(workspace),
        allowed_session_cwd_roots=[str(workspace)],
    ))
    store = ACPSessionStore(ACPSessionsDB(str(tmp_path / "owned-sessions.db")))

    async def get_runner():
        return runner

    async def get_store():
        return store

    monkeypatch.setattr(endpoints, "get_runner_client", get_runner)
    monkeypatch.setattr(endpoints, "get_acp_session_store", get_store)
    monkeypatch.setattr(admin_acp_sessions_service, "get_acp_session_store", get_store)
    if not any(route.path == "/api/v1/acp/sessions/{session_id}/stream"
               for route in iter_served_routes(client.app.routes)):
        client.app.include_router(endpoints.router, prefix="/api/v1")

    async def prepare():
        await runner.start()
        session_id = await runner.create_session(str(workspace), agent_type="controlled", user_id=1)
        await store.register_session(session_id, user_id=1, agent_type="controlled", cwd=str(workspace))
        repo = McpHubRepo(await get_db_pool())
        await repo.create_policy_assignment(
            target_type="default", target_id=None, owner_scope_type="user", owner_scope_id=1,
            profile_id=None, inline_policy_document={"allowed_tools": ["fs/write_text_file"],
                "approval_mode": "require_approval"}, approval_policy_id=None, actor_id=1,
        )
        snapshot = await runner._get_runtime_policy_snapshot(session_id, force_refresh=True)
        assert snapshot is not None  # nosec B101 - actual policy setup precondition
        assert snapshot.resolved_policy_document["allowed_tools"] == ["fs/write_text_file"]  # nosec B101
        assert await runner.verify_session_access(session_id, 1)  # nosec B101 - actual owner precondition
        assert not await runner.verify_session_access(session_id, 2)  # nosec B101
        return session_id

    session_id = client.portal.call(prepare)
    assert _mint(client, api_key).status_code == 200  # nosec B101 - normal cookie authentication
    try:
        yield client, runner, session_id, output
    finally:
        try:
            client.portal.call(runner.close_session, session_id)
        finally:
            try:
                client.portal.call(runner.shutdown)
            finally:
                client.portal.call(store.get_db().close)


def receive_json(ws):
    """Apply the unchanged deadline to the actual TestClient receive stream."""
    async def receive():
        with anyio.fail_after(3):
            message = await ws._send_rx.receive()
        ws._raise_on_close(message)
        return json.loads(message["text"])

    return ws.portal.call(receive)


def receive_type(ws, expected, runner, client):
    for _ in range(20):
        message = receive_json(ws)
        if message.get("type") == expected:
            return message
    raise AssertionError(f"Missing {expected} in bounded socket messages")


def receive_until(ws, complete):
    """Retain actual notifications while bounding the entire required reply set."""
    async def receive():
        messages = []
        with anyio.fail_after(3):
            for _ in range(20):
                message = await ws._send_rx.receive()
                ws._raise_on_close(message)
                messages.append(json.loads(message["text"]))
                if complete(messages):
                    return messages
        raise AssertionError("Required socket replies absent in bounded messages")

    return ws.portal.call(receive)


def test_compiled_runner_permission_keeps_stdio_response_reader_responsive(owned_runtime):
    client, runner, session_id, output = owned_runtime

    async def exercise():
        permission_requests = asyncio.Queue()
        await runner.register_websocket(session_id, permission_requests.put)
        prompt = asyncio.create_task(runner.prompt(session_id, [{"type": "text", "text": "Write once"}]))
        try:
            permission = await asyncio.wait_for(permission_requests.get(), timeout=3)
            assert permission["type"] == "permission_request"  # nosec B101
            catalog = await asyncio.wait_for(runner.list_agents(), timeout=3)
            assert catalog["agents"]  # nosec B101 - actual response during held permission
            assert output.read_text() == ORIGINAL  # nosec B101
            assert await runner.respond_to_permission(session_id, permission["request_id"], True)  # nosec B101
            result = await asyncio.wait_for(prompt, timeout=3)
            assert result["toolRefused"] is False  # nosec B101
            assert output.read_text() == REPLACEMENT  # nosec B101
        finally:
            await runner.cancel(session_id)
            await runner.unregister_websocket(session_id, permission_requests.put)
            prompt.cancel()
            await asyncio.gather(prompt, return_exceptions=True)

    client.portal.call(exercise)

# A separate controlled peer holds only the two new lifetime test prompts.
# The release file belongs to the temporary workspace and controls actual
# downstream settlement; it is never read by application code.
LIFETIME_DOWNSTREAM = r'''
import json
import pathlib
import sys
import threading
import time

session = "controlled-owned-session"
output = pathlib.Path(sys.argv[1])
release = output.parent / "release-prompt"
write_lock = threading.Lock()
state_lock = threading.Lock()
pending = {}
held = {}
next_tool = 1000

def send(payload):
    with write_lock:
        sys.stdout.write(json.dumps({"jsonrpc": "2.0", **payload}) + "\n")
        sys.stdout.flush()

def update(text):
    send({"method": "session/update", "params": {"sessionId": session,
        "update": {"sessionUpdate": "agent_thought_chunk", "content": {
            "type": "text", "text": text}}}})

def release_actual_reply():
    while True:
        if release.exists():
            with state_lock:
                replies = list(held.items())
                held.clear()
            for prompt_id, refused in replies:
                send({"id": prompt_id, "result": {"stopReason": "end_turn", "toolRefused": refused}})
                update("LIFETIME_SETTLED")
        time.sleep(0.01)

threading.Thread(target=release_actual_reply, daemon=True).start()
for line in sys.stdin:
    message = json.loads(line)
    method = message.get("method")
    params = message.get("params") or {}
    if method == "initialize":
        send({"id": message["id"], "result": {"protocolVersion": 1, "agentCapabilities": {}}})
    elif method == "session/new":
        send({"id": message["id"], "result": {"sessionId": session}})
    elif method == "session/prompt":
        hold = any(item.get("text") == "Hold original settlement" for item in params.get("prompt", []))
        update("LIFETIME_PROMPT_DISPATCHED")
        next_tool += 1
        pending[next_tool] = (message["id"], "initial" if hold else "fresh")
        send({"id": next_tool, "method": "fs/write_text_file", "params": {
            "sessionId": session, "path": str(output), "content": "APPROVED_WIRE_OUTPUT\n"}})
    elif method == "session/cancel":
        update("LIFETIME_STOP_RECEIVED")
    elif method in {"session/close", "_tldw/session/close"}:
        send({"id": message["id"], "result": {}})
    elif method is None and message.get("id") in pending:
        prompt_id, phase = pending.pop(message["id"])
        refused = bool(message.get("error"))
        if phase == "initial":
            next_tool += 1
            pending[next_tool] = (prompt_id, "late")
            send({"id": next_tool, "method": "fs/write_text_file", "params": {
                "sessionId": session, "path": str(output), "content": "LATE_OLD_WRITE\n"}})
        elif phase == "late":
            update("LIFETIME_LATE_REFUSED" if refused else "LIFETIME_LATE_APPROVED")
            with state_lock:
                held[prompt_id] = refused
        else:
            send({"id": prompt_id, "result": {"stopReason": "end_turn", "toolRefused": refused}})
    elif message.get("id") is not None:
        send({"id": message["id"], "error": {"code": -32601, "message": "method not found"}})
'''


@pytest.mark.parametrize("owned_runtime", [LIFETIME_DOWNSTREAM], indirect=True, ids=["held_downstream"])
@pytest.mark.parametrize("interruption", ["timeout", "caller_cancel"])
def test_interrupted_caller_keeps_original_prompt_owned_until_actual_reply(owned_runtime, interruption, monkeypatch):
    client, runner, session_id, output = owned_runtime

    async def exercise():
        # Configure only this real private stdio client's supported RPC deadline.
        monkeypatch.setattr(runner._client, "rpc_timeout_sec", 0.5 if interruption == "timeout" else 3)
        events = asyncio.Queue()
        await runner.register_websocket(session_id, events.put)
        original = asyncio.create_task(runner.prompt(
            session_id, [{"type": "text", "text": "Hold original settlement"}]))
        observed = []

        async def until_text(text):
            for message in observed:
                if message.get("data", {}).get("update", {}).get("content", {}).get("text") == text:
                    return message
            async with asyncio.timeout(3):
                while True:
                    message = await events.get()
                    observed.append(message)
                    if message.get("data", {}).get("update", {}).get("content", {}).get("text") == text:
                        return message

        try:
            async with asyncio.timeout(3):
                while True:
                    permission = await events.get()
                    observed.append(permission)
                    if permission.get("type") == "permission_request":
                        break
            assert output.read_text() == ORIGINAL  # nosec B101
            if interruption == "timeout":
                with pytest.raises(ACPResponseError, match="timed out waiting for session/prompt"):
                    await asyncio.wait_for(original, timeout=3)
            else:
                original.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await original

            # This must be the local admission fence, rather than Go's later
            # busy response after a new Python operation borrowed the session.
            with pytest.raises(ACPResponseError, match="^A prompt is already active for this session$") as busy:
                await runner.prompt(session_id, [{"type": "text", "text": "Must not dispatch"}])
            assert busy.value.code is None  # nosec B101 - local refusal, no RPC dispatch
            await until_text("LIFETIME_STOP_RECEIVED")
            await until_text("LIFETIME_LATE_REFUSED")
            assert output.read_text() == ORIGINAL  # nosec B101
            assert sum(item.get("type") == "permission_request" for item in observed) == 1  # nosec B101
            assert sum(item.get("data", {}).get("update", {}).get("content", {}).get("text") ==
                       "LIFETIME_PROMPT_DISPATCHED" for item in observed) == 1  # nosec B101
            assert not await runner.respond_to_permission(session_id, permission["request_id"], True)  # nosec B101
            assert runner.get_pending_permission_metadata(session_id, permission["request_id"]) == {}  # nosec B101

            output.parent.joinpath("release-prompt").write_text("RELEASE")
            await until_text("LIFETIME_SETTLED")
            await asyncio.wait_for(runner.list_agents(), timeout=3)
            monkeypatch.setattr(runner._client, "rpc_timeout_sec", 3)
            fresh = asyncio.create_task(runner.prompt(session_id, [{"type": "text", "text": "Fresh input"}]))
            async with asyncio.timeout(3):
                while True:
                    fresh_permission = await events.get()
                    if fresh_permission.get("type") == "permission_request":
                        break
            assert fresh_permission["request_id"] != permission["request_id"]  # nosec B101
            assert await runner.respond_to_permission(session_id, fresh_permission["request_id"], True)  # nosec B101
            assert (await asyncio.wait_for(fresh, timeout=3))["toolRefused"] is False  # nosec B101
            assert output.read_text() == REPLACEMENT  # nosec B101
        finally:
            output.parent.joinpath("release-prompt").write_text("RELEASE")
            await runner.cancel(session_id)
            await runner.unregister_websocket(session_id, events.put)
            if not original.done():
                original.cancel()
            await asyncio.gather(original, return_exceptions=True)

    client.portal.call(exercise)



def test_owned_same_websocket_approval_settles_original_prompt(owned_runtime):
    client, runner, session_id, output = owned_runtime
    path = f"/api/v1/acp/sessions/{session_id}/stream"
    with client.websocket_connect(path, headers={"Origin": "http://localhost:3000"}) as ws:
        assert receive_type(ws, "connected", runner, client)["session_id"] == session_id  # nosec B101
        ws.send_json({"type": "prompt", "prompt": [{"type": "text", "text": "Write once"}]})
        permission = receive_type(ws, "permission_request", runner, client)
        assert permission["tool_name"] == "fs/write_text_file"  # nosec B101
        assert permission["tool_arguments"] == {"path": str(output), "content": REPLACEMENT}  # nosec B101
        assert permission["policy_snapshot_fingerprint"]  # nosec B101
        assert output.read_text() == ORIGINAL  # nosec B101
        ws.send_json({"type": "permission_response", "request_id": permission["request_id"], "approved": True})
        complete = receive_type(ws, "prompt_complete", runner, client)
        assert complete["raw_result"]["toolRefused"] is False  # nosec B101
        assert output.read_text() == REPLACEMENT  # nosec B101


def observe_prompt_replies(monkeypatch, runner):
    # Retain the actual wire refusal oracle when Stop now rejects the public result.
    original_call = runner._client.call
    replies = []

    async def observed(method, *args, **kwargs):
        response = await original_call(method, *args, **kwargs)
        if method == "session/prompt":
            replies.append(response.result)
        return response

    monkeypatch.setattr(runner._client, "call", observed)
    return replies


def test_owned_same_websocket_stop_retires_permission_and_refuses_late_approval(owned_runtime, monkeypatch):
    client, runner, session_id, output = owned_runtime
    replies = observe_prompt_replies(monkeypatch, runner)
    path = f"/api/v1/acp/sessions/{session_id}/stream"
    with client.websocket_connect(path, headers={"Origin": "http://localhost:3000"}) as ws:
        receive_type(ws, "connected", runner, client)
        ws.send_json({"type": "prompt", "prompt": [{"type": "text", "text": "Write once"}]})
        permission = receive_type(ws, "permission_request", runner, client)
        ws.send_json({"type": "cancel"})
        outcomes = receive_until(ws, lambda messages: any(
            item.get("update_type") == "cancelled" for item in messages))
        assert any(item.get("update_type") == "cancelled" for item in outcomes)  # nosec B101
        ws.send_json({"type": "permission_response", "request_id": permission["request_id"], "approved": True})
        # A real downstream Stop update may interleave with both required replies.
        outcomes.extend(receive_until(ws, lambda messages: (
            any(item.get("code") == "permission_not_found" for item in outcomes + messages)
            and any(item.get("code") == "prompt_failed" and "cancelled" in item.get("message", "")
                    for item in outcomes + messages))))
        assert any(item.get("code") == "permission_not_found" for item in outcomes)  # nosec B101
        assert any(item.get("code") == "prompt_failed" and "cancelled" in item.get("message", "")
                   for item in outcomes)  # nosec B101 - Stop is never a public completion.
        assert replies[-1]["toolRefused"] is True  # nosec B101 - Original real wire refusal retained.
        assert not any(item.get("type") == "prompt_complete" for item in outcomes)  # nosec B101 - No false success.
        assert output.read_text() == ORIGINAL  # nosec B101
        assert runner.get_pending_permission_metadata(session_id, permission["request_id"]) == {}  # nosec B101


def test_foreign_cookie_session_cannot_open_owned_runner_socket(owned_runtime):
    client, runner, session_id, output = owned_runtime
    # Session owner remains1; this unknown ID cannot inherit the authenticated owner.
    from starlette.websockets import WebSocketDisconnect

    with pytest.raises(WebSocketDisconnect) as refusal:
        with client.websocket_connect(
            "/api/v1/acp/sessions/foreign-session/stream",
            headers={"Origin": "http://localhost:3000"},
        ):
            pytest.fail("Unknown session accepted")
    assert refusal.value.code == 4404  # nosec B101
    assert output.read_text() == ORIGINAL  # nosec B101


def test_owned_socket_busy_denial_and_fresh_prompt_admission(owned_runtime):
    client, runner, session_id, output = owned_runtime
    path = f"/api/v1/acp/sessions/{session_id}/stream"
    with client.websocket_connect(path, headers={"Origin": "http://localhost:3000"}) as ws:
        receive_type(ws, "connected", runner, client)
        prompt = {"type": "prompt", "prompt": [{"type": "text", "text": "Write once"}]}
        ws.send_json(prompt)
        permission = receive_type(ws, "permission_request", runner, client)
        ws.send_json(prompt)
        busy = receive_type(ws, "error", runner, client)
        assert busy["code"] == "prompt_busy"  # nosec B101 - held operation must not admit another
        ws.send_json({"type": "permission_response", "request_id": permission["request_id"], "approved": False})
        refused = receive_type(ws, "prompt_complete", runner, client)
        assert refused["raw_result"]["toolRefused"] is True  # nosec B101
        assert output.read_text() == ORIGINAL  # nosec B101
        ws.send_json(prompt)
        fresh = receive_type(ws, "permission_request", runner, client)
        assert fresh["request_id"] != permission["request_id"]  # nosec B101
        ws.send_json({"type": "permission_response", "request_id": fresh["request_id"], "approved": True})
        complete = receive_type(ws, "prompt_complete", runner, client)
        assert complete["raw_result"]["toolRefused"] is False  # nosec B101
        assert output.read_text() == REPLACEMENT  # nosec B101


def test_unknown_permission_session_is_refused_before_owned_callback(owned_runtime):
    client, runner, session_id, output = owned_runtime

    async def exercise():
        requests = asyncio.Queue()
        await runner.register_websocket(session_id, requests.put)
        try:
            result = await asyncio.wait_for(runner.prompt(
                session_id, [{"type": "text", "text": "Foreign permission"}]), timeout=3)
            assert result["toolRefused"] is True  # nosec B101
            assert requests.qsize() == 1  # nosec B101 - Only the owned RPC settlement may reach its owner.
            assert requests.get_nowait() == {"type": "prompt_settled", "session_id": session_id}  # nosec B101 - No foreign permission callback is admitted.
            assert requests.empty()  # nosec B101 - unknown session must not reach owner callback
            assert output.read_text() == ORIGINAL  # nosec B101
        finally:
            await runner.unregister_websocket(session_id, requests.put)

    client.portal.call(exercise)


def test_stop_revokes_permission_before_its_scheduled_sender_runs(owned_runtime, monkeypatch):
    client, runner, session_id, output = owned_runtime
    replies = observe_prompt_replies(monkeypatch, runner)

    async def exercise():
        events = asyncio.Queue()
        await runner.register_websocket(session_id, events.put)
        original = asyncio.create_task(runner.prompt(session_id, [{"type": "text", "text": "Write once"}]))
        stop = None
        try:
            async with asyncio.timeout(3):
                while True:
                    permission = await events.get()
                    if permission.get("type") == "permission_request":
                        break
            assert output.read_text() == ORIGINAL  # nosec B101
            stop = asyncio.create_task(runner.cancel(session_id))
            # FIFO task scheduling lets cancel register its owned Stop and reach
            # shield before the newly scheduled sender can run. This is normal
            # event-loop ordering, with no private operation-map mutation.
            await asyncio.sleep(0)
            assert not await runner.respond_to_permission(session_id, permission["request_id"], True)  # nosec B101
            await asyncio.wait_for(stop, timeout=3)
            with pytest.raises(ACPResponseError, match="cancelled"):
                await asyncio.wait_for(original, timeout=3)
            assert replies[-1]["toolRefused"] is True  # nosec B101 - Original real wire refusal retained.
            assert output.read_text() == ORIGINAL  # nosec B101
            assert runner.get_pending_permission_metadata(session_id, permission["request_id"]) == {}  # nosec B101
        finally:
            await runner.cancel(session_id)
            if stop is not None:
                await asyncio.gather(stop, return_exceptions=True)
            await runner.unregister_websocket(session_id, events.put)
            if not original.done():
                original.cancel()
            await asyncio.gather(original, return_exceptions=True)

    client.portal.call(exercise)
