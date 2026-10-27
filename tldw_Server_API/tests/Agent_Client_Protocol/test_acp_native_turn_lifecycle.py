"""Native transcript contracts over the actual stream client and owned ledger.

The injected byte transport does not provision a sandbox or execute a model.
"""
from __future__ import annotations

import asyncio
import copy
import json
import sys

import pytest

from tldw_Server_API.app.core.Agent_Client_Protocol.config import ACPRunnerConfig, ACPSandboxConfig
from tldw_Server_API.app.core.Agent_Client_Protocol.runner_client import ACPRunnerClient
from tldw_Server_API.app.core.Agent_Client_Protocol.sandbox_runner_client import (
    ACPSandboxRunnerManager,
    SandboxSessionHandle,
)
from tldw_Server_API.app.core.Agent_Client_Protocol.stdio_client import ACPMessage, ACPResponseError, ACPStdioClient
from tldw_Server_API.app.core.Agent_Client_Protocol.stream_client import ACPStreamClient
from tldw_Server_API.app.core.DB_Management.ACP_Sessions_DB import ACPSessionsDB
from tldw_Server_API.app.services.acp_runtime_policy_service import ACPRuntimePolicySnapshot
from tldw_Server_API.app.services.admin_acp_sessions_service import ACPSessionStore


@pytest.fixture
async def sandbox_stream():
    sent = asyncio.Queue()
    incoming = asyncio.Queue()

    async def send(data):
        await sent.put(json.loads(data))

    client = ACPStreamClient(send_bytes=send, rpc_timeout_sec=0)
    manager = ACPSandboxRunnerManager(ACPSandboxConfig(enabled=True, agent_command="controlled-peer"))
    client.set_notification_handler(lambda msg: manager._handle_notification(msg, source_client=client))
    client.set_request_handler(lambda msg: manager._handle_request(msg, source_client=client))
    reader = asyncio.create_task(manager._reader_loop("owned-run", incoming, client))
    manager._sessions["owned"] = SandboxSessionHandle("owned", 7, "sandbox", "owned-run", client, reader)
    try:
        yield manager, client, incoming, sent
    finally:
        await client.close()
        reader.cancel()
        await asyncio.gather(reader, return_exceptions=True)
        tasks = [op.task for op in manager._active_prompts.values() if op.task is not None]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)


async def emit(incoming, payload):
    await incoming.put({"type": "stdout", "data": json.dumps({"jsonrpc": "2.0", **payload}) + "\n"})


async def text(incoming, value, session="owned"):
    await emit(incoming, {"method": "session/update", "params": {"sessionId": session,
        "update": {"sessionUpdate": "agent_message_chunk", "content": {"type": "text", "text": value}}}})


@pytest.mark.parametrize("stop_reason", ["end_turn", "max_tokens", "cancelled"])
async def test_sandbox_native_callback_is_owned_exact_once_and_raw_unchanged(sandbox_stream, stop_reason):
    manager, _client, incoming, sent = sandbox_stream
    seen = []
    task = asyncio.create_task(manager.prompt("owned", [{"role": "user", "content": "authored"}], on_native_text=seen.append))
    request = await asyncio.wait_for(sent.get(), 1)
    await text(incoming, "  native ")
    await text(incoming, "exact\n ")
    raw = {"stopReason": stop_reason, "usage": {"input_tokens": 0}, "_meta": {"original": None}}
    original = copy.deepcopy(raw)
    await emit(incoming, {"id": request["id"], "result": raw})
    assert await asyncio.wait_for(task, 1) == original  # nosec B101 # pytest regression oracle, not production authorization
    assert seen == ([] if stop_reason == "cancelled" else ["  native exact\n "])  # nosec B101 # pytest regression oracle, not production authorization
    assert raw == original  # nosec B101 # pytest regression oracle, not production authorization
    assert manager._active_prompts == {}  # nosec B101 # pytest regression oracle, not production authorization


async def test_sandbox_stop_keeps_admission_until_reply_and_suppresses_partial(sandbox_stream):
    manager, _client, incoming, sent = sandbox_stream
    events = []
    async def receive(event):
        events.append(event)
    await manager.register_websocket("owned", receive)
    seen = []
    task = asyncio.create_task(manager.prompt("owned", [{"role": "user", "content": "held"}], on_native_text=seen.append))
    request = await asyncio.wait_for(sent.get(), 1)
    await text(incoming, "partial")
    await manager.cancel("owned")
    assert not any(event["type"] == "prompt_settled" for event in events)  # nosec B101 # Stop/deadline does not establish RPC settlement.
    assert (await sent.get())["method"] == "session/cancel"  # nosec B101 # pytest regression oracle, not production authorization
    with pytest.raises(ACPResponseError, match="already active"):
        await manager.prompt("owned", [{"role": "user", "content": "successor"}])
    await text(incoming, "late cancelled")
    await emit(incoming, {"id": request["id"], "result": {"stopReason": "end_turn"}})
    with pytest.raises(ACPResponseError, match="cancelled"):
        await asyncio.wait_for(task, 1)
    assert seen == []  # nosec B101 # pytest regression oracle, not production authorization
    assert [event for event in events if event["type"] == "prompt_settled"] == [{"type": "prompt_settled", "session_id": "owned"}]  # nosec B101 # Exactly one actual downstream settlement.
    assert manager._active_prompts == {}  # nosec B101 # pytest regression oracle, not production authorization


async def test_sandbox_caller_deadline_does_not_release_actual_rpc(sandbox_stream):
    manager, client, incoming, sent = sandbox_stream
    client.rpc_timeout_sec = 0.02
    events = []
    async def receive(event):
        events.append(event)
    await manager.register_websocket("owned", receive)
    seen = []
    task = asyncio.create_task(manager.prompt("owned", [{"role": "user", "content": "held"}], on_native_text=seen.append))
    request = await asyncio.wait_for(sent.get(), 1)
    with pytest.raises(ACPResponseError, match="timed out"):
        await task
    assert "owned" in manager._active_prompts  # nosec B101 # pytest regression oracle, not production authorization
    assert client._pending  # nosec B101 # pytest regression oracle, not production authorization
    assert (await sent.get())["method"] == "session/cancel"  # nosec B101 # pytest regression oracle, not production authorization
    operation = manager._active_prompts["owned"]
    assert not any(event["type"] == "prompt_settled" for event in events)  # nosec B101 # Stop/deadline does not establish RPC settlement.
    await emit(incoming, {"id": request["id"], "result": {"stopReason": "cancelled"}})
    await asyncio.wait_for(operation.task, 1)
    assert [event for event in events if event["type"] == "prompt_settled"] == [{"type": "prompt_settled", "session_id": "owned"}]  # nosec B101 # Exactly one actual downstream settlement.
    assert manager._active_prompts == {}  # nosec B101 # pytest regression oracle, not production authorization
    assert seen == []  # nosec B101 # pytest regression oracle, not production authorization


@pytest.mark.parametrize("frame", [{"type": "event", "event": "end"}, {"type": "stdout", "data": "[]\n"}])
async def test_originating_reader_eof_or_malformed_frame_settles_owned_prompt(sandbox_stream, frame):
    manager, client, incoming, sent = sandbox_stream
    seen = []
    task = asyncio.create_task(manager.prompt("owned", [{"role": "user", "content": "held"}], on_native_text=seen.append))
    await asyncio.wait_for(sent.get(), 1)
    await text(incoming, "partial")
    await incoming.put(frame)
    with pytest.raises(ACPResponseError, match="connection closed"):
        await asyncio.wait_for(task, 1)
    assert not client.is_running  # nosec B101 # pytest regression oracle, not production authorization
    assert manager._active_prompts == {}  # nosec B101 # pytest regression oracle, not production authorization
    assert seen == []  # nosec B101 # pytest regression oracle, not production authorization


async def test_old_source_cannot_deliver_text_or_borrow_successor_permissions(sandbox_stream):
    manager, client, incoming, sent = sandbox_stream
    seen = []
    task = asyncio.create_task(manager.prompt("owned", [{"role": "user", "content": "held"}], on_native_text=seen.append))
    request = await asyncio.wait_for(sent.get(), 1)
    async def discard(_data):
        pass
    old = ACPStreamClient(send_bytes=discard)
    msg = ACPMessage(jsonrpc="2.0", method="session/update", params={"sessionId": "owned", "update": {
        "sessionUpdate": "agent_message_chunk", "content": {"type": "text", "text": "foreign"}}})
    await manager._handle_notification(msg, source_client=old)
    permission = await manager._handle_request(ACPMessage(jsonrpc="2.0", id=20, method="session/request_permission",
        params={"sessionId": "owned", "tool": {"name": "fs.write", "input": {}}}), source_client=old)
    assert permission.result == {"outcome": {"outcome": "cancelled"}}  # nosec B101 # pytest regression oracle, not production authorization
    assert manager.pop_updates("owned") == []  # nosec B101 # pytest regression oracle, not production authorization
    await text(incoming, "legitimate")
    await emit(incoming, {"id": request["id"], "result": {"stopReason": "end_turn"}})
    await asyncio.wait_for(task, 1)
    assert seen == ["legitimate"]  # nosec B101 # pytest regression oracle, not production authorization
    after = await manager._handle_request(ACPMessage(jsonrpc="2.0", id=21, method="session/request_permission",
        params={"sessionId": "owned", "tool": {"name": "fs.write", "input": {}}}), source_client=client)
    assert after.result == {"outcome": {"outcome": "cancelled"}}  # nosec B101 # pytest regression oracle, not production authorization


@pytest.mark.parametrize("native,final,expected,ready", [
    ("  exact native\n ", {}, "  exact native\n ", True),
    ("native", {"content": [{"type": "text", "text": "Legacy"}]}, "Legacy", True),
    ("  \n ", {}, "  \n ", False),
    (None, {}, None, False),
])
async def test_store_db_share_exact_text_choice_and_bootstrap_truth(tmp_path, native, final, expected, ready):
    db = ACPSessionsDB(db_path=str(tmp_path / "sessions.db"))
    store = ACPSessionStore(db=db)
    await store.register_session(session_id="stored", user_id=7, agent_type="controlled", cwd="/tmp/project")  # nosec B108 # inert metadata in isolated pytest DB; no filesystem write
    raw = {"stopReason": "end_turn", "usage": {"input_tokens": 0}, **final}
    original = copy.deepcopy(raw)
    await store.record_prompt("stored", [{"role": "user", "content": "authored"}], raw, assistant_text=native)
    saved = await store.get_session("stored")
    assert db.get_messages("stored")[1]["content"] == (expected or "")  # nosec B101 # pytest regression oracle, not production authorization
    assert saved.messages[1]["content"] == (expected if ready else None)  # nosec B101 # pytest regression oracle, not production authorization
    assert saved.messages[1]["raw_result"] == original  # nosec B101 # pytest regression oracle, not production authorization
    assert saved.bootstrap_ready is ready  # nosec B101 # pytest regression oracle, not production authorization
    assert raw == original  # nosec B101 # pytest regression oracle, not production authorization
    db.close()


async def test_stream_zero_timeout_override_retains_default_deadline_contract():
    sent = asyncio.Queue()
    async def send(data):
        await sent.put(json.loads(data))
    client = ACPStreamClient(send_bytes=send, rpc_timeout_sec=0.01)
    with pytest.raises(ACPResponseError, match="timed out"):
        await client.call("default", {})
    await sent.get()
    task = asyncio.create_task(client.call("owned", {}, timeout_sec=0))
    request = await sent.get()
    await asyncio.sleep(0.02)
    assert not task.done()  # nosec B101 # pytest regression oracle, not production authorization
    await client.feed_bytes((json.dumps({"id": request["id"], "result": {"ok": True}}) + "\n").encode())
    assert (await task).result == {"ok": True}  # nosec B101 # pytest regression oracle, not production authorization
    task = asyncio.create_task(client.call("eof", {}, timeout_sec=0))
    await sent.get()
    await client.close()
    with pytest.raises(ACPResponseError, match="connection closed"):
        await task
    assert client._pending == {}  # nosec B101 # pytest regression oracle, not production authorization


async def test_sandbox_stop_before_retirement_refuses_late_batch_approval(sandbox_stream, monkeypatch):
    manager, _client, incoming, sent = sandbox_stream
    events = asyncio.Queue()
    await manager.register_websocket("owned", events.put)
    snapshot = ACPRuntimePolicySnapshot(
        session_id="owned", user_id=7, policy_snapshot_version="resolved-v1",
        policy_snapshot_fingerprint="controlled", policy_snapshot_refreshed_at="2026-10-09T00:00:00Z",
        policy_summary={}, policy_provenance_summary={},
        resolved_policy_document={"allowed_tools": ["fs/write_text_file"], "approval_mode": "require_approval"},
        approval_summary={"mode": "require_approval"}, context_summary={}, execution_config={},
    )
    async def current_snapshot(*_args, **_kwargs):
        return snapshot
    monkeypatch.setattr(manager, "_get_runtime_policy_snapshot", current_snapshot)
    seen = []
    prompt = asyncio.create_task(manager.prompt("owned", [{"role": "user", "content": "held"}], on_native_text=seen.append))
    request = await asyncio.wait_for(sent.get(), 1)
    await emit(incoming, {"id": 400, "method": "session/request_permission", "params": {
        "sessionId": "owned", "tool": {"name": "fs/write_text_file", "input": {"path": "unused"}}}})
    permission = await asyncio.wait_for(events.get(), 1)
    assert permission["type"] == "permission_request"  # nosec B101 # pytest regression oracle, not production authorization
    stop = asyncio.create_task(manager.cancel("owned"))
    # Stop marks the operation before its separately scheduled sender retires futures.
    await asyncio.sleep(0)
    assert not await manager.respond_to_permission("owned", permission["request_id"], True, "individual")  # nosec B101 # pytest regression oracle, not production authorization
    assert manager._ws_registry["owned"].batch_approved_tiers == set()  # nosec B101 # pytest regression oracle, not production authorization
    await asyncio.wait_for(stop, 1)
    outgoing = [await asyncio.wait_for(sent.get(), 1), await asyncio.wait_for(sent.get(), 1)]
    reply = next(frame for frame in outgoing if frame.get("id") == 400)
    assert reply["result"] == {"outcome": {"outcome": "cancelled"}}  # nosec B101 # pytest regression oracle, not production authorization
    assert any(frame.get("method") == "session/cancel" for frame in outgoing)  # nosec B101 # pytest regression oracle, not production authorization
    await emit(incoming, {"id": request["id"], "result": {"stopReason": "cancelled"}})
    assert await asyncio.wait_for(prompt, 1) == {"stopReason": "cancelled"}  # nosec B101 # pytest regression oracle, not production authorization
    assert seen == []  # nosec B101 # pytest regression oracle, not production authorization
    assert manager._active_prompts == {}  # nosec B101 # pytest regression oracle, not production authorization
    assert manager._ws_registry["owned"].pending_permissions == {}  # nosec B101 # pytest regression oracle, not production authorization


async def _assert_pre_reply_persistence(tmp_path, seen, raw):
    db = ACPSessionsDB(db_path=str(tmp_path / "ordered.db"))
    store = ACPSessionStore(db=db)
    try:
        await store.register_session(session_id="ordered", user_id=7, agent_type="controlled", cwd=str(tmp_path))
        await store.record_prompt("ordered", [{"role": "user", "content": "authored"}], raw, assistant_text=seen[0])
        saved = await store.get_session("ordered")
        assert saved.messages[1]["content"] == "  before reply\n"  # nosec B101 # pytest regression oracle, not production authorization
        assert saved.messages[1]["raw_result"] == raw  # nosec B101 # pytest regression oracle, not production authorization
        assert saved.bootstrap_ready is True  # nosec B101 # pytest regression oracle, not production authorization
    finally:
        db.close()


async def test_sandbox_same_frame_reply_closes_native_capture_before_late_chunk(sandbox_stream, tmp_path):
    manager, _client, incoming, sent = sandbox_stream
    seen = []
    task = asyncio.create_task(manager.prompt("owned", [{"role": "user", "content": "authored"}], on_native_text=seen.append))
    request = await asyncio.wait_for(sent.get(), 1)
    raw = {"stopReason": "end_turn", "usage": {"input_tokens": 0}, "_meta": {"original": None}}
    def chunk(value):
        return {"method": "session/update", "params": {"sessionId": "owned", "update": {
            "sessionUpdate": "agent_message_chunk", "content": {"type": "text", "text": value}}}}
    frames = [chunk("  before reply\n"), {"id": request["id"], "result": raw}, chunk("AFTER REPLY")]
    # One actual stdout frame is consumed in order without yielding at its reply.
    await incoming.put({"type": "stdout", "data": "".join(json.dumps({"jsonrpc": "2.0", **frame}) + "\n" for frame in frames)})
    assert await asyncio.wait_for(task, 1) == raw  # nosec B101 # pytest regression oracle, not production authorization
    assert seen == ["  before reply\n"]  # nosec B101 # pytest regression oracle, not production authorization
    updates = manager.pop_updates("owned")
    assert [update["update"]["content"]["text"] for update in updates] == seen  # nosec B101 # pytest regression oracle, not production authorization
    assert manager._active_prompts == {}  # nosec B101 # pytest regression oracle, not production authorization
    assert _client._response_observers == {}  # nosec B101 # pytest regression oracle, not production authorization
    await _assert_pre_reply_persistence(tmp_path, seen, raw)


ORDERED_STDIO_PEER = r'''
import json, sys
for line in sys.stdin:
    request = json.loads(line)
    method = request.get("method")
    if method == "initialize":
        frames = [{"id": request["id"], "result": {"protocolVersion": 1, "agentCapabilities": {}}}]
    elif method == "session/new":
        frames = [{"id": request["id"], "result": {"sessionId": "owned"}}]
    elif method == "session/prompt":
        def chunk(value):
            return {"method": "session/update", "params": {"sessionId": "owned", "update": {
                "sessionUpdate": "agent_message_chunk", "content": {"type": "text", "text": value}}}}
        frames = [chunk("  before reply\n"), {"id": request["id"], "result": {
            "stopReason": "end_turn", "usage": {"input_tokens": 0}, "_meta": {"original": None}}}, chunk("AFTER REPLY")]
    elif method == "observed":
        frames = [{"id": 9999, "result": {"foreign": True}}]
        outcome = request["params"]["outcome"]
        if outcome == "error":
            frames.append({"id": request["id"], "error": {"code": -32603, "message": "controlled error"}})
        elif outcome == "eof":
            sys.stdout.write("".join(json.dumps({"jsonrpc": "2.0", **frame}) + "\n" for frame in frames))
            sys.stdout.flush()
            break
    else:
        continue
    sys.stdout.write("".join(json.dumps({"jsonrpc": "2.0", **frame}) + "\n" for frame in frames))
    sys.stdout.flush()
'''


async def test_stdio_same_buffer_reply_closes_native_capture_before_late_chunk(tmp_path):
    manager = ACPRunnerClient(ACPRunnerConfig(command=sys.executable, args=["-u", "-c", ORDERED_STDIO_PEER],
        cwd=str(tmp_path), allowed_session_cwd_roots=[str(tmp_path)]))
    seen = []
    try:
        await manager.start()
        session = await manager.create_session(str(tmp_path), user_id=7)
        raw = await manager.prompt(session, [{"role": "user", "content": "authored"}], on_native_text=seen.append)
        assert raw == {"stopReason": "end_turn", "usage": {"input_tokens": 0}, "_meta": {"original": None}}  # nosec B101 # pytest regression oracle, not production authorization
        assert seen == ["  before reply\n"]  # nosec B101 # pytest regression oracle, not production authorization
        updates = manager.pop_updates(session)
        assert [update["update"]["content"]["text"] for update in updates] == seen  # nosec B101 # pytest regression oracle, not production authorization
        assert manager._active_prompts == {}  # nosec B101 # pytest regression oracle, not production authorization
        assert manager._client._response_observers == {}  # nosec B101 # pytest regression oracle, not production authorization
        await _assert_pre_reply_persistence(tmp_path, seen, raw)
    finally:
        await manager.shutdown()


@pytest.mark.parametrize("transport", ["stdio", "stream"])
@pytest.mark.parametrize("outcome", ["error", "timeout", "eof"])
async def test_response_observer_matches_only_owned_reply_and_cleans_all_exits(transport, outcome):
    sent = asyncio.Queue()
    async def send(data):
        await sent.put(json.loads(data))
    client = (ACPStdioClient(sys.executable, ["-u", "-c", ORDERED_STDIO_PEER], rpc_timeout_sec=0.02)
              if transport == "stdio" else ACPStreamClient(send_bytes=send, rpc_timeout_sec=0.02))
    seen = []
    try:
        await client.start()
        if transport == "stdio":
            await client.call("initialize", {}, timeout_sec=1)
        task = asyncio.create_task(client.call("observed", {"outcome": outcome}, on_response_received=lambda: seen.append("reply")))
        if transport == "stream":
            request = await asyncio.wait_for(sent.get(), 1)
            frames = [{"id": 9999, "result": {"foreign": True}}]
            if outcome == "error":
                frames.append({"id": request["id"], "error": {"code": -32603, "message": "controlled error"}})
            await client.feed_bytes("".join(json.dumps({"jsonrpc": "2.0", **frame}) + "\n" for frame in frames).encode())
            if outcome == "eof":
                await client.close()
        expected = {"error": "controlled error", "timeout": "timed out", "eof": "connection closed"}[outcome]
        with pytest.raises(ACPResponseError, match=expected):
            await asyncio.wait_for(task, 1)
        assert seen == (["reply"] if outcome == "error" else [])  # nosec B101 # pytest regression oracle, not production authorization
        assert client._pending == {}  # nosec B101 # pytest regression oracle, not production authorization
        assert client._response_observers == {}  # nosec B101 # pytest regression oracle, not production authorization
    finally:
        await client.close()


async def test_local_stop_rejects_sandbox_direct_success_after_actual_reply(sandbox_stream):
    manager, _client, incoming, sent = sandbox_stream
    seen = []
    task = asyncio.create_task(
        manager.prompt("owned", [{"role": "user", "content": "held"}], on_native_text=seen.append)
    )
    request = await asyncio.wait_for(sent.get(), 1)
    await manager.cancel("owned")
    assert (await sent.get())["method"] == "session/cancel"  # nosec B101 # Real Stop send.
    with pytest.raises(ACPResponseError, match="already active"):
        await manager.prompt("owned", [{"role": "user", "content": "successor"}])
    raw = {"stopReason": "end_turn", "taskCompletion": {"status": "completed", "summary": "partial"}}
    await emit(incoming, {"id": request["id"], "result": raw})
    with pytest.raises(ACPResponseError, match="cancelled"):
        await asyncio.wait_for(task, 1)
    assert seen == []  # nosec B101 # No cancelled transcript publication.
    assert manager._active_prompts == {}  # nosec B101 # Admission retires after actual reply.


STOPPED_SUCCESS_STDIO_PEER = r"""
import json, sys
pending = None
def send(value):
    sys.stdout.write(json.dumps({"jsonrpc": "2.0", **value}) + "\n")
    sys.stdout.flush()
for line in sys.stdin:
    req = json.loads(line)
    method = req.get("method")
    if method == "initialize":
        send({"id": req["id"], "result": {"protocolVersion": 1, "agentCapabilities": {}}})
    elif method == "session/new":
        send({"id": req["id"], "result": {"sessionId": "owned"}})
    elif method == "session/prompt":
        pending = req["id"]
        send({"method": "session/update", "params": {"sessionId": "owned", "update": {"sessionUpdate": "agent_message_chunk", "content": {"type": "text", "text": "partial"}}}})
    elif method == "release":
        send({"id": pending, "result": {"stopReason": "end_turn", "taskCompletion": {"status": "completed", "summary": "partial"}}})
        send({"id": req["id"], "result": {}})
"""


async def test_local_stop_rejects_stdio_direct_success_after_actual_reply(tmp_path):
    manager = ACPRunnerClient(
        ACPRunnerConfig(
            command=sys.executable,
            args=["-u", "-c", STOPPED_SUCCESS_STDIO_PEER],
            cwd=str(tmp_path),
            allowed_session_cwd_roots=[str(tmp_path)],
        )
    )
    task = None
    seen = []
    try:
        await manager.start()
        session = await manager.create_session(str(tmp_path), user_id=7)
        events = []
        async def receive(event):
            events.append(event)
        await manager.register_websocket(session, receive)
        task = asyncio.create_task(
            manager.prompt(session, [{"role": "user", "content": "held"}], on_native_text=seen.append)
        )
        async with asyncio.timeout(2):
            while not manager.pop_updates(session):
                await asyncio.sleep(0)
        await manager.cancel(session)
        assert not any(event["type"] == "prompt_settled" for event in events)  # nosec B101 # Stop alone must not retire actual RPC.
        with pytest.raises(ACPResponseError, match="already active"):
            await manager.prompt(session, [{"role": "user", "content": "successor"}])
        await manager._client.call("release", {}, timeout_sec=2)
        with pytest.raises(ACPResponseError, match="cancelled"):
            await asyncio.wait_for(task, 2)
        assert seen == []  # nosec B101 # No cancelled transcript publication.
        assert [event for event in events if event["type"] == "prompt_settled"] == [{"type": "prompt_settled", "session_id": session}]  # nosec B101 # Exactly one authoritative settlement.
        assert manager._active_prompts == {}  # nosec B101 # Admission retires after actual reply.
    finally:
        if task is not None and not task.done():
            task.cancel()
        if task is not None:
            await asyncio.gather(task, return_exceptions=True)
        await manager.shutdown()


@pytest.mark.parametrize("transport", ["sandbox", "stdio"])
async def test_stalled_settlement_notification_releases_admission_and_preserves_raw_result(
    sandbox_stream, tmp_path, transport,
):
    manager, _client, incoming, sent = sandbox_stream
    if transport == "stdio":
        manager = ACPRunnerClient(ACPRunnerConfig(
            command=sys.executable, args=["-u", "-c", LIVENESS_STDIO_PEER],
            cwd=str(tmp_path), allowed_session_cwd_roots=[str(tmp_path)],
        ))
        await manager.start()
        await manager.create_session(str(tmp_path), user_id=7)
    entered = asyncio.Event()
    release = asyncio.Event()
    events = []
    metadata = []
    metadata_received = asyncio.Event()
    async def receive(event):
        if event["type"] == "prompt_settled":
            events.append(event)
            entered.set()
            await release.wait()
        if event.get("update_type") == "session_info_update":
            metadata.append(event.get("prompt_active"))
            metadata_received.set()
    await manager.register_websocket("owned", receive)
    task = asyncio.create_task(manager.prompt("owned", [{"role": "user", "content": "held"}]))
    raw = {"stopReason": "end_turn", "taskCompletion": {"status": "completed", "summary": "partial"}}
    try:
        if transport == "sandbox":
            request = await asyncio.wait_for(sent.get(), 1)
            await emit(incoming, {"id": request["id"], "result": raw})
        else:
            async with asyncio.timeout(2):
                while not manager.pop_updates("owned"):
                    await asyncio.sleep(0)
            await manager._client.call("release", {}, timeout_sec=2)
        await asyncio.wait_for(entered.wait(), 1)
        operation = manager._active_prompts["owned"]
        assert operation.capture_closed and not operation.task.done()  # nosec B101 # Reply read while bounded final notification still holds admission.
        if transport == "sandbox":
            await emit(incoming, {"method": "session/update", "params": {"sessionId": "owned", "prompt_active": True, "update": {"sessionUpdate": "session_info_update"}}})
        else:
            await manager._client.call("metadata", {}, timeout_sec=2)
        await asyncio.wait_for(metadata_received.wait(), 1)
        assert metadata == [False]  # nosec B101 # Closed capture forbids adapter metadata from claiming a pending prompt.
        assert await asyncio.wait_for(asyncio.shield(task), 2) == raw  # nosec B101 # Notification backpressure must not replace the exact RPC result.
        assert manager._active_prompts == {}  # nosec B101 # A completed RPC must release admission despite a stalled browser.
        assert events == [{"type": "prompt_settled", "session_id": "owned"}]  # nosec B101 # Exactly one owned notification attempt.
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        if transport == "stdio":
            await manager.shutdown()


LIVENESS_STDIO_PEER = STOPPED_SUCCESS_STDIO_PEER.replace(
    '    elif method == "release":',
    '\n'.join([
        '    elif method == "metadata":',
        '        send({"method": "session/update", "params": {"sessionId": "owned", "prompt_active": True, "update": {"sessionUpdate": "session_info_update", "title": "public metadata"}}})',
        '        send({"id": req["id"], "result": {}})',
        '    elif method == "release":',
    ]),
)


@pytest.mark.parametrize("transport", ["sandbox", "stdio"])
async def test_update_liveness_tracks_actual_rpc_before_during_after_reply(sandbox_stream, tmp_path, transport):
    manager, _client, incoming, sent = sandbox_stream
    if transport == "stdio":
        manager = ACPRunnerClient(ACPRunnerConfig(
            command=sys.executable, args=["-u", "-c", LIVENESS_STDIO_PEER],
            cwd=str(tmp_path), allowed_session_cwd_roots=[str(tmp_path)],
        ))
        await manager.start()
        await manager.create_session(str(tmp_path), user_id=7)
    events = []
    metadata_received = asyncio.Event()
    native_received = asyncio.Event()
    async def receive(event):
        events.append(event)
        if event.get("update_type") == "session_info_update":
            metadata_received.set()
        if event.get("update_type") == "agent_message_chunk":
            native_received.set()
    await manager.register_websocket("owned", receive)
    task = None
    async def metadata():
        metadata_received.clear()
        if transport == "sandbox":
            await emit(incoming, {"method": "session/update", "params": {"sessionId": "owned", "prompt_active": True, "update": {"sessionUpdate": "session_info_update", "title": "public metadata"}}})
        else:
            await manager._client.call("metadata", {}, timeout_sec=2)
        await asyncio.wait_for(metadata_received.wait(), 2)
    try:
        await metadata()
        task = asyncio.create_task(manager.prompt("owned", [{"role": "user", "content": "held"}]))
        if transport == "sandbox":
            request = await asyncio.wait_for(sent.get(), 1)
            await text(incoming, "partial")
        await asyncio.wait_for(native_received.wait(), 2)
        await metadata()
        raw = {"stopReason": "end_turn", "taskCompletion": {"status": "completed", "summary": "partial"}}
        if transport == "sandbox":
            await emit(incoming, {"id": request["id"], "result": raw})
        else:
            await manager._client.call("release", {}, timeout_sec=2)
        assert await asyncio.wait_for(task, 2) == raw  # nosec B101 # Liveness metadata must not change the actual RPC result.
        await metadata()
        assert [e.get("prompt_active") for e in events if e.get("update_type") == "session_info_update"] == [False, True, False]  # nosec B101 # Actual admission/capture establishes activity, not metadata presence.
        assert [e.get("prompt_active") for e in events if e.get("update_type") == "agent_message_chunk"] == [True]  # nosec B101 # Real in-flight native activity remains visible.
    finally:
        if task is not None and not task.done():
            if transport == "stdio":
                await manager.shutdown()
            else:
                await _client.close()
        if task is not None:
            await asyncio.gather(task, return_exceptions=True)
        if transport == "stdio":
            await manager.shutdown()
