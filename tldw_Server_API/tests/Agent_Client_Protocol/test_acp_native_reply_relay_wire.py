"""Final downstream reply retires native relay through the actual compiled runner."""
from __future__ import annotations

import asyncio
import copy

import pytest

from tldw_Server_API.tests.Agent_Client_Protocol.test_acp_native_turn_content_wire import (
    ORIGINAL,
    TEXT,
    post,
    record,
)
from tldw_Server_API.tests.Agent_Client_Protocol.test_acp_native_turn_content_wire import (
    compiled_runner as _compiled_runner,
)
from tldw_Server_API.tests.Agent_Client_Protocol.test_acp_native_turn_content_wire import (
    owned_runtime as _owned_runtime,
)
from tldw_Server_API.tests.Agent_Client_Protocol.test_acp_native_turn_content_wire import (
    private_jobs_path as _private_jobs_path,
)
from tldw_Server_API.tests.Agent_Client_Protocol.test_acp_native_turn_content_wire import (
    private_paths as _private_paths,
)
from tldw_Server_API.tests.Agent_Client_Protocol.test_acp_native_turn_content_wire import (
    single_user_cookie_client as _cookie_client,
)

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]
compiled_runner = _compiled_runner
owned_runtime = _owned_runtime
private_jobs_path = _private_jobs_path
private_paths = _private_paths
single_user_cookie_client = _cookie_client


@pytest.fixture(autouse=True)
def serialized_reader_schedule(monkeypatch):
    """Use normal single-CPU Go scheduling in this isolated functional fixture."""
    monkeypatch.setenv("GOMAXPROCS", "1")


REPLY_THEN_CHUNK_PEER = r'''
import json, os, sys
session = "ordered-relay-" + str(os.getpid())
def update(kind, text=None, **extra):
    payload = {"sessionUpdate": kind, **extra}
    if text is not None:
        payload["content"] = {"type": "text", "text": text}
    return {"method": "session/update", "params": {"sessionId": session, "update": payload}}
for line in sys.stdin:
    request = json.loads(line)
    method = request.get("method")
    if method == "initialize":
        frames = [{"id": request["id"], "result": {"protocolVersion": 1, "agentCapabilities": {}}}]
    elif method == "session/new":
        frames = [{"id": request["id"], "result": {"sessionId": session}}]
    elif method == "session/prompt":
        prompt = request["params"]["prompt"]
        frames = [update("agent_message_chunk", "  CONTROLLED native "),
                  update("agent_message_chunk", "summary.\nExact.  "),
                  {"id": request["id"], "result": {"stopReason": "end_turn", "usage": {
                      "input_tokens": 2, "output_tokens": 3}, "_meta": {"original": None, "received": prompt}}},
                  update("agent_message_chunk", "AFTER_DOWNSTREAM_REPLY"),
                  update("usage_update", used=0, size=200000, relay_probe_complete=True)]
    elif method in {"session/close", "_tldw/session/close"}:
        frames = [{"id": request["id"], "result": {}}]
    else:
        continue
    sys.stdout.write("".join(json.dumps({"jsonrpc": "2.0", **frame}) + "\n" for frame in frames))
    sys.stdout.flush()
'''


CANONICAL_KIND_ALIAS_PEER = REPLY_THEN_CHUNK_PEER.replace(
    '    return {"method": "session/update",',
    '    if text == "AFTER_DOWNSTREAM_REPLY":\n'
    '        payload["SessionUpdate"] = "usage_update"\n'
    '    return {"method": "session/update",',
)


LEGACY_KIND_FALLBACK_PEER = REPLY_THEN_CHUNK_PEER.replace(
    '    return {"method": "session/update",',
    '    if text == "AFTER_DOWNSTREAM_REPLY":\n'
    '        payload.pop("sessionUpdate")\n'
    '        return {"method": "session/update", "params": {"sessionId": session,\n'
    '            "type": "agent_message_chunk", "update": payload}}\n'
    '    return {"method": "session/update",',
)


@pytest.mark.parametrize("owned_runtime", [REPLY_THEN_CHUNK_PEER,
    pytest.param(CANONICAL_KIND_ALIAS_PEER, id="canonical_kind_alias"),
    pytest.param(LEGACY_KIND_FALLBACK_PEER, id="legacy_kind_fallback")], indirect=True)
def test_compiled_relay_closes_native_text_at_downstream_reply_before_upstream_result(owned_runtime):
    client, runner, session_id, output = owned_runtime
    response = post(client, session_id, "authored ordered request")
    assert response.status_code == 200, response.json()  # nosec B101 # pytest regression oracle, not production authorization
    raw = copy.deepcopy(response.json()["raw_result"])
    saved = record(client, session_id)
    assert saved.messages[1]["content"] == TEXT  # nosec B101 # pytest regression oracle, not production authorization
    assert saved.messages[1]["raw_result"] == raw  # nosec B101 # pytest regression oracle, not production authorization
    assert raw == {"stopReason": "end_turn", "usage": {"input_tokens": 2, "output_tokens": 3},
                   "_meta": {"original": None, "received": [{"type": "text", "text": "authored ordered request",
                       "_meta": {"tldw-server/legacy-message": {"role": "user"}}}]}}  # nosec B101 # pytest regression oracle, not production authorization
    assert saved.messages[0]["raw_prompt"] == {"role": "user", "content": "authored ordered request"}  # nosec B101 # pytest regression oracle, not production authorization
    assert saved.bootstrap_ready is True  # nosec B101 # pytest regression oracle, not production authorization
    async def completed_updates():
        updates = []
        async with asyncio.timeout(1):
            while not any(item["update"].get("relay_probe_complete") for item in updates):
                updates.extend(runner.pop_updates(session_id))
                await asyncio.sleep(0)
        return updates
    updates = client.portal.call(completed_updates)
    chunks = [item["update"]["content"]["text"] for item in updates if item["update"]["sessionUpdate"] == "agent_message_chunk"]
    assert "".join(chunks) == TEXT  # nosec B101 # pytest regression oracle, not production authorization
    assert updates[-1]["update"] == {"sessionUpdate": "usage_update", "used": 0, "size": 200000,
                                     "relay_probe_complete": True}  # nosec B101 # pytest regression oracle, not production authorization
    assert output.read_text() == ORIGINAL  # nosec B101 # pytest regression oracle, not production authorization
