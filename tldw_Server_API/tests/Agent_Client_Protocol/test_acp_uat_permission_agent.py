"""Closed controlled-worker recipes; live Go/browser composition is separate."""

import asyncio
import json
import sys
from pathlib import Path

import pytest
from Helper_Scripts.acp_uat_permission_agent import OUTPUT, Worker

pytestmark = pytest.mark.asyncio


@pytest.mark.parametrize("command,count", [("write", 1), ("repeat", 2), ("read", 1), ("boundary", 6)])
async def test_controlled_recipes_only_request_tools(tmp_path, command, count):
    worker = Worker(tmp_path)
    messages, calls = [], []
    worker.send = messages.append

    async def tool(method, **params):
        calls.append((method, params))
        return {"error": {"code": -1, "message": "denied"}}

    worker.tool = tool
    result = await worker.prompt({"prompt": [{"type": "text", "text": "C05 " + command}]})
    assert len(calls) == count  # nosec B101 # Closed fixture call-count oracle.
    assert result == {"stopReason": "end_turn"}  # nosec B101 # RPC settlement is distinct from a successful tool.
    assert '"error"' in messages[0]["params"]["update"]["content"]["text"]  # nosec B101 # Denial is retained, not synthesized success.
    assert not list(tmp_path.iterdir())  # nosec B101 # Fixture itself never writes files.
    if command == "boundary":
        targets = (tmp_path.parent / "sentinel.txt", tmp_path / "outside-link", tmp_path / ".env")
        assert calls == [
            (method, {"path": str(path), **({"content": OUTPUT} if method == "fs/write_text_file" else {})})
            for path in targets
            for method in ("fs/read_text_file", "fs/write_text_file")
        ]  # nosec B101 # Exact harmless target/method pairs.
    if command in {"write", "repeat"}:
        assert all(p == {"path": str(tmp_path / "output.txt"), "content": OUTPUT} for _, p in calls)  # nosec B101 # Exact disposable target only.


@pytest.mark.parametrize("prompt", ["write arbitrary.txt", "C05 rm", "C05 write ../other", "C05 read\nwrite"])
async def test_unknown_recipe_cannot_request_tool(tmp_path, prompt):
    worker = Worker(tmp_path)
    with pytest.raises(ValueError, match="declared"):
        await worker.prompt({"prompt": [{"text": prompt}]})
    assert worker.sequence == 0  # nosec B101 # Reject before any tool dispatch.


async def test_cancel_pending_tool_ignores_late_approval(tmp_path):
    worker = Worker(tmp_path)
    sent = []
    worker.send = sent.append
    task = asyncio.create_task(worker.tool("fs/write_text_file", path=str(tmp_path / "output.txt"), content=OUTPUT))
    await asyncio.sleep(0)
    response = worker.pending[sent[0]["id"]]
    worker.cancelled.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert response.cancelled() and not worker.pending  # nosec B101 # Late approval has no pending owner.
    assert not (tmp_path / "output.txt").exists()  # nosec B101 # No direct fixture write.


async def test_delayed_prompt_cancellation_has_no_tool_effect(tmp_path):
    worker = Worker(tmp_path)
    task = asyncio.create_task(worker.prompt({"prompt": [{"text": "C05 delayed-write"}]}))
    await asyncio.sleep(0)
    worker.cancelled.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert worker.sequence == 0  # nosec B101 # Stop precedes tool admission.


async def test_wrong_session_refused(tmp_path):
    worker = Worker(tmp_path)
    sent = []
    worker.send = sent.append
    await worker.handle(
        {"id": 1, "method": "session/prompt", "params": {"sessionId": "foreign", "prompt": [{"text": "C05 write"}]}}
    )
    assert sent[0]["error"]["message"] == "Controlled session mismatch"  # nosec B101 # Foreign session cannot use recipes.
    assert worker.sequence == 0  # nosec B101 # No dispatch on ownership failure.


async def start_pipe(root):
    script = Path(__file__).parents[3] / "Helper_Scripts/acp_uat_permission_agent.py"
    return await asyncio.create_subprocess_exec(
        sys.executable,
        str(script),
        "--root",
        str(root),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )


async def test_pipe_eof_retires_pending_tool(tmp_path):
    child = await start_pipe(tmp_path)
    try:
        child.stdin.write(
            (
                json.dumps({"jsonrpc": "2.0", "id": 1, "method": "session/new", "params": {"cwd": str(tmp_path)}})
                + "\n"
            ).encode()
        )
        await child.stdin.drain()
        session = json.loads(await asyncio.wait_for(child.stdout.readline(), 2))["result"]["sessionId"]
        child.stdin.write(
            (
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": 2,
                        "method": "session/prompt",
                        "params": {"sessionId": session, "prompt": [{"text": "C05 write"}]},
                    }
                )
                + "\n"
            ).encode()
        )
        await child.stdin.drain()
        tool = json.loads(await asyncio.wait_for(child.stdout.readline(), 2))
        assert tool["method"] == "fs/write_text_file"  # nosec B101 # Real pipe reaches actual tool request.
        child.stdin.close()
        assert await asyncio.wait_for(child.wait(), 2) == 0  # nosec B101 # EOF retires pending operation naturally.
        assert not (tmp_path / "output.txt").exists()  # nosec B101 # No write before approval.
    finally:
        if child.returncode is None:
            child.terminate()
            await child.wait()


async def test_unexpected_handler_failure_ends_pipe(tmp_path):
    child = await start_pipe(tmp_path)
    try:
        child.stdin.write(b'{"jsonrpc":"2.0","id":1,"method":"session/prompt","params":1}\n')
        await child.stdin.drain()
        assert await asyncio.wait_for(child.wait(), 2) != 0  # nosec B101 # Unexpected failure cannot leave a silent worker.
    finally:
        if child.returncode is None:
            child.terminate()
            await child.wait()
