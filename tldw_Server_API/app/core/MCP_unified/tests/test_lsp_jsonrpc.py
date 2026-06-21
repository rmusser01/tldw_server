import json
import sys
from pathlib import Path

import pytest
from mcp_unified.lsp import LspRuntimeConfig, LspToolError
from mcp_unified.lsp.jsonrpc import LspJsonRpcClient

FAKE_SERVER = Path(__file__).parent / "fixtures" / "fake_lsp_stdio_server.py"


async def _client(tmp_path: Path, *, config: LspRuntimeConfig | None = None) -> tuple[LspJsonRpcClient, Path]:
    trace_path = tmp_path / "trace.json"
    client = LspJsonRpcClient(
        argv=[sys.executable, str(FAKE_SERVER), str(trace_path)],
        workspace_root=tmp_path,
        config=config or LspRuntimeConfig(),
    )
    await client.start()
    return client, trace_path


async def test_initialize_uses_lsp_content_length_framing(tmp_path: Path):
    client, _ = await _client(tmp_path)
    try:
        result = await client.request("initialize", {"rootUri": tmp_path.as_uri()})
    finally:
        await client.close()

    assert "Content-Length:" in result["received_header"]
    assert result["capabilities"] == {}


async def test_requests_are_correlated_by_jsonrpc_id(tmp_path: Path):
    client, _ = await _client(tmp_path)
    try:
        first = await client.request("test/echo", {"value": "first"})
        second = await client.request("test/echo", {"value": "second"})
    finally:
        await client.close()

    assert first == {"value": "first", "request_id": 1}
    assert second == {"value": "second", "request_id": 2}


async def test_stderr_capture_is_bounded_and_redacted(tmp_path: Path):
    config = LspRuntimeConfig(max_stderr_bytes=48)
    client, _ = await _client(tmp_path, config=config)
    try:
        await client.request("test/stderr", {"message": f"failure in {tmp_path} " + ("x" * 200)})
        stderr_text = client.stderr_text(workspace_root=tmp_path)
    finally:
        await client.close()

    assert str(tmp_path) not in stderr_text
    assert "<workspace>" in stderr_text
    assert len(stderr_text) <= config.max_stderr_bytes


async def test_request_timeout_raises_backend_timeout(tmp_path: Path):
    config = LspRuntimeConfig(request_timeout_seconds=0.05, startup_timeout_seconds=1)
    client, _ = await _client(tmp_path, config=config)
    try:
        with pytest.raises(LspToolError) as exc:
            await client.request("test/sleep", {"seconds": 0.25})
    finally:
        await client.close()

    assert exc.value.reason_code == "backend_timeout"


async def test_close_sends_shutdown_and_exit_notifications(tmp_path: Path):
    client, trace_path = await _client(tmp_path)

    await client.request("initialize", {"rootUri": tmp_path.as_uri()})
    await client.close()

    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    assert "shutdown" in trace["methods"]
    assert "exit" in trace["methods"]
    assert client.closed is True
