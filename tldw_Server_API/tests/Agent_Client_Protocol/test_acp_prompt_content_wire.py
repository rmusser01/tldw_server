"""Authored prompts through real initialized ACP stdio, without model execution."""

from __future__ import annotations

import asyncio
import copy

import pytest

from tldw_Server_API.app.api.v1.endpoints import agent_client_protocol as endpoints
from tldw_Server_API.app.core.Agent_Client_Protocol.config import ACPSandboxConfig
from tldw_Server_API.app.core.Agent_Client_Protocol.runner_client import ACPGovernanceDeniedError
from tldw_Server_API.app.core.Agent_Client_Protocol.sandbox_runner_client import (
    ACPSandboxRunnerManager,
    SandboxSessionHandle,
)
from tldw_Server_API.app.core.Agent_Client_Protocol.stdio_client import ACPResponseError
from tldw_Server_API.tests.Agent_Client_Protocol.test_acp_prompt_permission_wire import (
    ORIGINAL,
    receive_type,
    receive_until,
)
from tldw_Server_API.tests.Agent_Client_Protocol.test_acp_prompt_permission_wire import (
    compiled_runner as _compiled_runner_fixture,
)
from tldw_Server_API.tests.Agent_Client_Protocol.test_acp_prompt_permission_wire import (
    owned_runtime as _owned_runtime_fixture,
)
from tldw_Server_API.tests.Agent_Client_Protocol.test_acp_prompt_permission_wire import (
    private_paths as _private_paths_fixture,
)
from tldw_Server_API.tests.Agent_Client_Protocol.test_acp_prompt_permission_wire import (
    single_user_cookie_client as _cookie_fixture,
)

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]
compiled_runner = _compiled_runner_fixture
owned_runtime = _owned_runtime_fixture
private_paths = _private_paths_fixture
single_user_cookie_client = _cookie_fixture

# An external protocol peer accepts only native ACP v1 shapes. It echoes the
# received payload/count; it neither reads the README nor executes a model/tool.
STRICT_DOWNSTREAM = r'''
import json
import os
import sys

session = "content-wire-session-" + str(os.getpid())
count = 0

def send(payload):
    sys.stdout.write(json.dumps({"jsonrpc": "2.0", **payload}) + "\n")
    sys.stdout.flush()

def valid(block):
    if not isinstance(block, dict):
        return False
    kind = block.get("type")
    if kind == "text":
        return isinstance(block.get("text"), str)
    if kind in {"image", "audio"}:
        return isinstance(block.get("data"), str) and isinstance(block.get("mimeType"), str)
    if kind == "resource_link":
        return isinstance(block.get("uri"), str) and isinstance(block.get("name"), str)
    if kind == "resource":
        resource = block.get("resource")
        return isinstance(resource, dict) and isinstance(resource.get("uri"), str) and (
            isinstance(resource.get("text"), str) or isinstance(resource.get("blob"), str))
    return False

for line in sys.stdin:
    message = json.loads(line)
    method = message.get("method")
    if method == "initialize":
        send({"id": message["id"], "result": {"protocolVersion": 1,
            "agentCapabilities": {"promptCapabilities": {
                "image": True, "audio": True, "embeddedContext": True}}}})
    elif method == "session/new":
        send({"id": message["id"], "result": {"sessionId": session}})
    elif method == "session/prompt":
        count += 1
        params = message.get("params") or {}
        prompt = params.get("prompt")
        if params.get("sessionId") != session or not isinstance(prompt, list) or not prompt or not all(valid(b) for b in prompt):
            send({"id": message["id"], "error": {"code": -32602, "message": "Invalid params: ContentBlock[] required"}})
        else:
            send({"id": message["id"], "result": {"stopReason": "end_turn",
                "content": [{"type": "text", "text": "CONTROLLED_PROTOCOL_ACK"}],
                "receivedPrompt": prompt, "wireCount": count}})
    elif method in {"session/close", "_tldw/session/close"}:
        send({"id": message["id"], "result": {}})
    elif method == "session/cancel":
        pass
    elif message.get("id") is not None:
        send({"id": message["id"], "error": {"code": -32601, "message": "method not found"}})
'''


@pytest.fixture(autouse=True)
def owned_job_shutdown_path(tmp_path, monkeypatch):
    """Retain the official app's supported private Jobs path through teardown."""
    monkeypatch.setenv("JOBS_DB_PATH", str(tmp_path / "jobs.db"))


def _run(client, runner, session_id, prompt):
    return client.portal.call(runner.prompt, session_id, prompt)


@pytest.mark.parametrize("owned_runtime", [STRICT_DOWNSTREAM], indirect=True)
@pytest.mark.parametrize("role", ["user", "system", "assistant"])
def test_authored_role_content_reaches_initialized_native_stdio(owned_runtime, role):
    client, runner, session_id, output = owned_runtime
    authored = [{"role": role, "content": "  Read README.md only.\nExact text.  "}]
    result = _run(client, runner, session_id, authored)
    assert result["receivedPrompt"] == [{  # nosec B101 # pytest regression oracle, not production authorization
        "type": "text", "text": "  Read README.md only.\nExact text.  ",
        "_meta": {"tldw-server/legacy-message": {"role": role}},
    }]
    assert authored == [{"role": role, "content": "  Read README.md only.\nExact text.  "}]  # nosec B101 # pytest regression oracle, not production authorization
    assert output.read_text() == ORIGINAL  # nosec B101 # pytest regression oracle, not production authorization


@pytest.mark.parametrize("owned_runtime", [STRICT_DOWNSTREAM], indirect=True)
def test_cookie_rest_preserves_authored_storage_and_converts_only_wire_copy(owned_runtime):
    client, runner, session_id, output = owned_runtime
    authored = [{"role": "user", "content": "README question", "metadata": {"trace": [1, "literal"]}}]
    response = client.post("/api/v1/acp/sessions/prompt", json={"session_id": session_id, "prompt": authored},
        headers={"X-CSRF-Token": client.cookies["csrf_token"]})
    assert response.status_code == 200, response.json()  # nosec B101 # pytest regression oracle, not production authorization
    assert response.json()["raw_result"]["receivedPrompt"] == [{  # nosec B101 # pytest regression oracle, not production authorization
        "type": "text", "text": "README question", "_meta": {
            "tldw-server/legacy-message": {"role": "user", "metadata": {"trace": [1, "literal"]}}},
    }]

    async def read_record():
        return await (await endpoints.get_acp_session_store()).get_session(session_id)

    record = client.portal.call(read_record)
    assert record.messages[0]["role"] == "user"  # nosec B101 # pytest regression oracle, not production authorization
    assert record.messages[0]["content"] == "README question"  # nosec B101 # pytest regression oracle, not production authorization
    assert record.messages[0]["raw_prompt"] == authored[0]  # nosec B101 # pytest regression oracle, not production authorization
    assert record.usage.total_tokens == 0  # nosec B101 # pytest regression oracle, not production authorization
    assert output.read_text() == ORIGINAL  # nosec B101 # pytest regression oracle, not production authorization


@pytest.mark.parametrize("owned_runtime", [STRICT_DOWNSTREAM], indirect=True)
def test_cookie_websocket_authored_prompt_is_native_on_actual_wire(owned_runtime):
    client, runner, session_id, output = owned_runtime
    with client.websocket_connect(f"/api/v1/acp/sessions/{session_id}/stream", headers={"Origin": "http://localhost:3000"}) as ws:
        assert receive_type(ws, "connected", runner, client)["session_id"] == session_id  # nosec B101 # pytest regression oracle, not production authorization
        ws.send_json({"type": "prompt", "prompt": [{"role": "user", "content": "Read README only"}]})
        messages = receive_until(ws, lambda frames: any(frame.get("type") in {"prompt_complete", "error"} for frame in frames))
        complete = messages[-1]
        assert complete["type"] == "prompt_complete", complete  # nosec B101 # pytest regression oracle, not production authorization
        assert complete["raw_result"]["receivedPrompt"] == [{  # nosec B101 # pytest regression oracle, not production authorization
            "type": "text", "text": "Read README only",
            "_meta": {"tldw-server/legacy-message": {"role": "user"}},
        }]
    assert output.read_text() == ORIGINAL  # nosec B101 # pytest regression oracle, not production authorization


NATIVE_BLOCKS = [
    {"type": "text", "text": "\nExact native text  ", "_meta": None, "annotations": None},
    {"type": "image", "data": "aGVsbG8=", "mimeType": "image/png", "uri": None, "_meta": {"vendor/context": [1, None]}},
    {"type": "audio", "data": "", "mimeType": "audio/wav"},
    {"type": "resource_link", "uri": "file:///repo/README.md", "name": "README", "title": None, "size": 0},
    {"type": "resource", "resource": {"uri": "file:///repo/README.md", "text": "  Embedded exactly\n", "mimeType": "text/markdown"}},
    {"type": "resource", "resource": {"uri": "file:///repo/a.bin", "blob": "AA==", "_meta": None}},
]


@pytest.mark.parametrize("owned_runtime", [STRICT_DOWNSTREAM], indirect=True)
@pytest.mark.parametrize("block", NATIVE_BLOCKS, ids=["text-null-metadata", "image", "audio", "link", "text-resource", "blob-resource"])
def test_native_blocks_are_structure_equivalent_on_initialized_wire(owned_runtime, block):
    client, runner, session_id, _output = owned_runtime
    authored = copy.deepcopy([block])
    result = _run(client, runner, session_id, authored)
    assert result["receivedPrompt"] == [block]  # nosec B101 # pytest regression oracle, not production authorization
    assert authored == [block]  # nosec B101 # pytest regression oracle, not production authorization


@pytest.mark.parametrize("owned_runtime", [STRICT_DOWNSTREAM], indirect=True)
def test_mixed_native_and_authored_blocks_keep_exact_order_and_governance_input(owned_runtime, monkeypatch):
    client, runner, session_id, _output = owned_runtime
    original_check = runner.check_prompt_governance
    checked = []

    async def observe(session_id, prompt, **kwargs):
        checked.append(copy.deepcopy(prompt))
        return await original_check(session_id, prompt, **kwargs)

    monkeypatch.setattr(runner, "check_prompt_governance", observe)
    authored = [NATIVE_BLOCKS[0], {"role": "system", "content": "instructions", "extra": {"literal": False}}, NATIVE_BLOCKS[3]]
    result = _run(client, runner, session_id, copy.deepcopy(authored))
    assert result["receivedPrompt"] == [NATIVE_BLOCKS[0], {  # nosec B101 # pytest regression oracle, not production authorization
        "type": "text", "text": "instructions", "_meta": {"tldw-server/legacy-message": {
            "role": "system", "extra": {"literal": False}}}}, NATIVE_BLOCKS[3]]
    assert checked == [authored]  # nosec B101 # pytest regression oracle, not production authorization


MALFORMED = [
    [], [{"role": "user", "content": 7}], [{"content": "missing role"}],
    [{"role": "tool", "content": "unsupported role"}], [{"type": "text", "text": 4}],
    [{"type": "text", "text": "ambiguous", "role": "user", "content": "second"}],
    [{"type": "image", "data": "AA=="}], [{"type": "audio", "data": 1, "mimeType": "audio/wav"}],
    [{"type": "resource_link", "uri": "file:///repo/a"}],
    [{"type": "resource", "resource": {"uri": "file:///repo/a", "text": 3}}],
    [{"type": "text", "text": "literal", "_meta": "malformed"}],
    [{"type": "text", "text": "literal", "annotations": []}],
]


@pytest.mark.parametrize("owned_runtime", [STRICT_DOWNSTREAM], indirect=True)
@pytest.mark.parametrize("prompt", MALFORMED, ids=["empty", "nonstring-content", "missing-role", "unsupported-role", "nonstring-text", "ambiguous", "missing-image-mime", "nonstring-audio", "missing-link-name", "bad-embedded-text", "bad-meta", "bad-annotations"])
def test_malformed_prompt_is_rejected_before_governance_admission_or_wire(owned_runtime, prompt, monkeypatch):
    client, runner, session_id, output = owned_runtime
    original_check = runner.check_prompt_governance
    checked = []

    async def observe(session_id, prompt, **kwargs):
        checked.append(copy.deepcopy(prompt))
        return await original_check(session_id, prompt, **kwargs)

    monkeypatch.setattr(runner, "check_prompt_governance", observe)
    with pytest.raises(ACPResponseError):
        _run(client, runner, session_id, prompt)
    assert checked == []  # nosec B101 # pytest regression oracle, not production authorization
    assert session_id not in runner._active_prompts  # nosec B101 # pytest regression oracle, not production authorization
    result = _run(client, runner, session_id, [{"type": "text", "text": "valid next"}])
    assert result["wireCount"] == 1  # nosec B101 # pytest regression oracle, not production authorization
    assert output.read_text() == ORIGINAL  # nosec B101 # pytest regression oracle, not production authorization


@pytest.mark.parametrize("owned_runtime", [STRICT_DOWNSTREAM], indirect=True)
def test_malformed_owned_rest_refuses_before_bootstrap_and_governance(owned_runtime, monkeypatch):
    client, runner, session_id, output = owned_runtime
    prepared = []
    original_prepare = endpoints._prepare_acp_runtime_prompt

    async def observe(**kwargs):
        prepared.append(copy.deepcopy(kwargs["prompt"]))
        return await original_prepare(**kwargs)

    monkeypatch.setattr(endpoints, "_prepare_acp_runtime_prompt", observe)
    response = client.post("/api/v1/acp/sessions/prompt", json={"session_id": session_id, "prompt": [{"role": "user", "content": 8}]},
        headers={"X-CSRF-Token": client.cookies["csrf_token"]})
    assert response.status_code == 502  # nosec B101 # pytest regression oracle, not production authorization
    assert response.json()["detail"] == "ACP prompt failed"  # nosec B101 # pytest regression oracle, not production authorization
    assert prepared == []  # nosec B101 # pytest regression oracle, not production authorization
    assert _run(client, runner, session_id, [{"type": "text", "text": "valid next"}])["wireCount"] == 1  # nosec B101 # pytest regression oracle, not production authorization

    async def read_record():
        return await (await endpoints.get_acp_session_store()).get_session(session_id)

    record = client.portal.call(read_record)
    assert record.messages == []  # nosec B101 # pytest regression oracle, not production authorization
    assert record.usage.total_tokens == 0  # nosec B101 # pytest regression oracle, not production authorization
    assert output.read_text() == ORIGINAL  # nosec B101 # pytest regression oracle, not production authorization


@pytest.mark.parametrize("owned_runtime", [STRICT_DOWNSTREAM], indirect=True)
def test_sandbox_prompt_boundary_converts_authored_payload_on_real_stdio(owned_runtime):
    client, runner, session_id, _output = owned_runtime

    async def exercise():
        manager = ACPSandboxRunnerManager(ACPSandboxConfig())
        reader = asyncio.create_task(asyncio.Event().wait())
        manager._sessions[session_id] = SandboxSessionHandle(session_id=session_id, user_id=1,
            sandbox_session_id="controlled-sandbox", run_id="controlled-run", client=runner._client,
            reader_task=reader, agent_capabilities=runner.agent_capabilities)
        try:
            result = await manager.prompt(session_id, [{"role": "user", "content": "Sandbox boundary"}])
            assert result["receivedPrompt"] == [{"type": "text", "text": "Sandbox boundary",  # nosec B101 # pytest regression oracle, not production authorization
                "_meta": {"tldw-server/legacy-message": {"role": "user"}}}]
            native = await manager.prompt(session_id, NATIVE_BLOCKS)
            assert native["receivedPrompt"] == NATIVE_BLOCKS  # nosec B101 # pytest regression oracle, not production authorization
            with pytest.raises(ACPResponseError):
                await manager.prompt(session_id, [{"type": "text", "text": 8}])
            assert (await manager.prompt(session_id, [{"type": "text", "text": "after refusal"}]))["wireCount"] == 3  # nosec B101 # pytest regression oracle, not production authorization
        finally:
            reader.cancel()
            await asyncio.gather(reader, return_exceptions=True)
            manager._sessions.clear()

    client.portal.call(exercise)


@pytest.mark.parametrize("owned_runtime", [STRICT_DOWNSTREAM], indirect=True)
def test_malformed_cookie_websocket_refuses_before_bootstrap_preparation(owned_runtime, monkeypatch):
    client, runner, session_id, _output = owned_runtime
    prepared = []
    original_prepare = endpoints._prepare_acp_runtime_prompt

    async def observe(**kwargs):
        prepared.append(copy.deepcopy(kwargs["prompt"]))
        return await original_prepare(**kwargs)

    monkeypatch.setattr(endpoints, "_prepare_acp_runtime_prompt", observe)
    with client.websocket_connect(f"/api/v1/acp/sessions/{session_id}/stream", headers={"Origin": "http://localhost:3000"}) as ws:
        receive_type(ws, "connected", runner, client)
        ws.send_json({"type": "prompt", "prompt": [{"type": "text", "text": 9}]})
        refused = receive_type(ws, "error", runner, client)
        assert refused["code"] == "prompt_failed"  # nosec B101 # pytest regression oracle, not production authorization
        assert prepared == []  # nosec B101 # pytest regression oracle, not production authorization
    assert _run(client, runner, session_id, [{"type": "text", "text": "valid next"}])["wireCount"] == 1  # nosec B101 # pytest regression oracle, not production authorization


@pytest.mark.parametrize("owned_runtime", [STRICT_DOWNSTREAM], indirect=True)
def test_fork_bootstrap_keeps_authored_order_and_clears_only_after_success(owned_runtime):
    client, runner, session_id, _output = owned_runtime
    csrf_headers = {"X-CSRF-Token": client.cookies["csrf_token"]}
    first = client.post("/api/v1/acp/sessions/prompt", json={"session_id": session_id,
        "prompt": [{"role": "user", "content": "Original question"}]}, headers=csrf_headers)
    assert first.status_code == 200, first.json()  # nosec B101 # pytest regression oracle, not production authorization
    fork = client.post(f"/api/v1/acp/sessions/{session_id}/fork", json={"message_index": 1}, headers=csrf_headers)
    assert fork.status_code == 200, fork.json()  # nosec B101 # pytest regression oracle, not production authorization
    fork_id = fork.json()["session_id"]

    async def read_fork():
        return await (await endpoints.get_acp_session_store()).get_session(fork_id)

    try:
        assert client.portal.call(read_fork).needs_bootstrap is True  # nosec B101 # pytest regression oracle, not production authorization
        refused = client.post("/api/v1/acp/sessions/prompt", json={"session_id": fork_id,
            "prompt": [{"role": "user", "content": 7}]}, headers=csrf_headers)
        assert refused.status_code == 502  # nosec B101 # pytest regression oracle, not production authorization
        assert client.portal.call(read_fork).needs_bootstrap is True  # nosec B101 # pytest regression oracle, not production authorization
        complete = client.post("/api/v1/acp/sessions/prompt", json={"session_id": fork_id,
            "prompt": [{"role": "user", "content": "Follow-up"}]}, headers=csrf_headers)
        assert complete.status_code == 200, complete.json()  # nosec B101 # pytest regression oracle, not production authorization
        assert complete.json()["raw_result"]["receivedPrompt"] == [  # nosec B101 # pytest regression oracle, not production authorization
            {"type": "text", "text": "Original question", "_meta": {"tldw-server/legacy-message": {"role": "user"}}},
            {"type": "text", "text": "CONTROLLED_PROTOCOL_ACK", "_meta": {"tldw-server/legacy-message": {"role": "assistant"}}},
            {"type": "text", "text": "Follow-up", "_meta": {"tldw-server/legacy-message": {"role": "user"}}},
        ]
        assert complete.json()["raw_result"]["wireCount"] == 1  # nosec B101 # pytest regression oracle, not production authorization
        assert client.portal.call(read_fork).needs_bootstrap is False  # nosec B101 # pytest regression oracle, not production authorization
    finally:
        client.portal.call(runner.close_session, fork_id)


@pytest.mark.parametrize("owned_runtime", [STRICT_DOWNSTREAM], indirect=True)
def test_governance_denial_retains_authored_input_and_never_dispatches(owned_runtime, monkeypatch):
    client, runner, session_id, _output = owned_runtime
    authored = [{"role": "user", "content": "Denied question"}]
    checked = []

    async def deny(session_id, prompt, **_kwargs):
        checked.append(copy.deepcopy(prompt))
        return {"action": "deny", "status": "deny", "rollout_mode": "enforce"}

    original = runner.check_prompt_governance
    monkeypatch.setattr(runner, "check_prompt_governance", deny)
    with pytest.raises(ACPGovernanceDeniedError):
        _run(client, runner, session_id, authored)
    assert checked == [authored]  # nosec B101 # pytest regression oracle, not production authorization
    assert session_id not in runner._active_prompts  # nosec B101 # pytest regression oracle, not production authorization
    monkeypatch.setattr(runner, "check_prompt_governance", original)
    assert _run(client, runner, session_id, [{"type": "text", "text": "valid next"}])["wireCount"] == 1  # nosec B101 # pytest regression oracle, not production authorization
