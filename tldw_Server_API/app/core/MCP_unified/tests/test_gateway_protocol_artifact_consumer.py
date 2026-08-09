"""Installed-artifact contract for downstream strict stdio embedders."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import venv
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from Helper_Scripts import mcp_unified_rc

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[5]
SUCCESS_MARKER = "MCP_UNIFIED_ARTIFACT_CONSUMER_OK"
PROTOCOL_SUITES = (
    "test_gateway_protocol_contracts.py",
    "test_gateway_protocol_validation.py",
    "test_gateway_protocol_projection.py",
    "test_gateway_protocol_connection.py",
    "test_gateway_protocol_stdio.py",
)
FIXTURE_COMMIT = "5f5440bb26a62e2cf3440b92da5a667efa03b267"
FIXTURE_SHA256 = {
    "2026-07-28": "ef70b61f99b6d2e5e3b46863822eab08dff6a45bedc7a08914e0e5b133f40203",
    "2025-11-25": "268a5f82ba70fd7e4b6dc4aa1e64f116f74b4d0edcb69dc046829c79dd4e97e7",
    "2025-06-18": "af845e7e5b9d27107d1690f0936022546177a1403e63ffb11470135b296a2e01",
    "2025-03-26": "e720669548c8100a4282c49e580efd6ddf7f28899ea786fc8db251dbdb356131",
    "2024-11-05": "61cea2392d4f284092d09bc84b9ac488c0d5618ac2b38a56942fc5b99fd960ce",
}


def _load_boundary_tests() -> ModuleType:
    """Load the neighboring artifact build helper without package imports."""

    path = Path(__file__).with_name("test_runtime_package_boundary.py")
    spec = importlib.util.spec_from_file_location(
        "_mcp_unified_artifact_consumer_boundary",
        path,
    )
    assert spec is not None and spec.loader is not None  # nosec B101
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_build_standalone_distributions = _load_boundary_tests()._build_standalone_distributions

_DOWNSTREAM_CONSUMER = r"""
from __future__ import annotations

import asyncio
import inspect
import json
import os
import sys
import sysconfig
from pathlib import Path
from typing import Any

from mcp_unified.gateway import (
    GatewayApplicationError,
    GatewayInvalidApplicationResult,
    GatewayLimits,
    GatewayProtocolConnection,
    GatewayResourceNotFound,
    GatewayResultTooLarge,
    GatewayToolExecutionError,
    serve_stdio,
)

MARKER = "MCP_UNIFIED_ARTIFACT_CONSUMER_OK"
MODERN_META = {
    "io.modelcontextprotocol/protocolVersion": "2026-07-28",
    "io.modelcontextprotocol/clientCapabilities": {},
}


def modern(request_id: str | int, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": method,
        "params": {**(params or {}), "_meta": MODERN_META},
    }


class ValueWriter:
    def __init__(self) -> None:
        self.values: list[dict[str, Any]] = []

    async def __call__(self, value: dict[str, Any]) -> None:
        self.values.append(value)


class Runtime:
    name = "artifact-consumer"
    version = "1.0"

    def __init__(self, *, empty: bool = False) -> None:
        self.context_ids: list[str | int] = []
        self.tokens: list[Any] = []
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.release.set()
        self.tools = [] if empty else [
            {"name": "array", "inputSchema": {"type": "object"}, "outputSchema": {"type": "array"}},
            {"name": "application_error", "inputSchema": {"type": "object"}},
            {"name": "invalid_result", "inputSchema": {"type": "object"}},
            {"name": "null", "inputSchema": {"type": "object"}, "outputSchema": {"type": "null"}},
            {"name": "object", "inputSchema": {"type": "object"}, "outputSchema": {"type": "object"}},
            {"name": "result_too_large", "inputSchema": {"type": "object"}},
            {"name": "scalar", "inputSchema": {"type": "object"}, "outputSchema": {"type": "integer"}},
            {"name": "slow", "inputSchema": {"type": "object"}},
            {"name": "tool_error", "inputSchema": {"type": "object"}},
        ]
        self.resources = [] if empty else [{"name": "guide", "uri": "file:///guide.txt"}]
        self.templates = [] if empty else [{"name": "files", "uriTemplate": "file:///{name}"}]
        self.prompts = [] if empty else [{"name": "hello"}]

    def record(self, context: Any) -> None:
        self.context_ids.append(context.request_id)
        self.tokens.append(context.cancellation)

    async def list_tools(self, context: Any) -> list[dict[str, Any]]:
        self.record(context)
        return self.tools

    async def call_tool(self, name: str, arguments: dict[str, Any], context: Any) -> Any:
        del arguments
        self.record(context)
        if name == "slow":
            self.started.set()
            await self.release.wait()
            context.cancellation.raise_if_cancelled()
        if name == "object":
            return {"kind": "object"}
        if name == "array":
            return [1, "two"]
        if name == "scalar":
            return 7
        if name == "null":
            return None
        if name == "tool_error":
            raise GatewayToolExecutionError("Tool unavailable", reason_code="unavailable")
        if name == "result_too_large":
            raise GatewayResultTooLarge(limit_bytes=128)
        if name == "invalid_result":
            raise GatewayInvalidApplicationResult()
        raise GatewayApplicationError("Application unavailable", reason_code="unavailable")

    async def list_resources(self, context: Any) -> list[dict[str, Any]]:
        self.record(context)
        return self.resources

    async def list_resource_templates(self, context: Any) -> list[dict[str, Any]]:
        self.record(context)
        return self.templates

    async def read_resource(self, uri: str, context: Any) -> dict[str, Any]:
        self.record(context)
        if uri != "file:///guide.txt":
            raise GatewayResourceNotFound()
        return {"contents": [{"uri": uri, "text": "guide"}]}

    async def list_prompts(self, context: Any) -> list[dict[str, Any]]:
        self.record(context)
        return self.prompts

    async def get_prompt(self, name: str, arguments: dict[str, Any], context: Any) -> dict[str, Any]:
        del arguments
        self.record(context)
        if name != "hello":
            raise GatewayApplicationError("Prompt unavailable", reason_code="unavailable", kind="prompt")
        return {"messages": [{"role": "user", "content": {"type": "text", "text": "hello"}}]}


class ByteReader:
    def __init__(self, *lines: bytes) -> None:
        self.lines = list(lines)

    async def readline(self) -> bytes:
        return self.lines.pop(0)


class ByteWriter:
    def __init__(self) -> None:
        self.chunks: list[bytes] = []

    def write(self, data: bytes) -> None:
        self.chunks.append(data)

    async def drain(self) -> None:
        return None


async def receive(connection: GatewayProtocolConnection, writer: ValueWriter, payload: dict[str, Any]) -> dict[str, Any]:
    before = len(writer.values)
    await connection.receive(payload)
    await connection.wait_for_idle()
    assert len(writer.values) == before + 1
    return writer.values[-1]


async def main() -> None:
    purelib = Path(sysconfig.get_paths()["purelib"]).resolve()
    module_path = Path(inspect.getfile(GatewayLimits)).resolve()
    checkout = Path(os.environ["MCP_UNIFIED_FORBIDDEN_CHECKOUT"]).resolve()
    assert module_path.is_relative_to(purelib)
    assert not module_path.is_relative_to(checkout)
    assert sys.path[0] != str(checkout)

    limits = GatewayLimits(max_in_flight=1, default_catalog_page_size=2)
    runtime = Runtime()
    writer = ValueWriter()
    connection = GatewayProtocolConnection(runtime, writer, limits=limits)

    first_page = await receive(connection, writer, modern(1, "tools/list"))
    assert len(first_page["result"]["tools"]) == 2
    cursor = first_page["result"]["nextCursor"]
    second_page = await receive(connection, writer, modern("1", "tools/list", {"cursor": cursor}))
    assert len(second_page["result"]["tools"]) == 2
    assert runtime.context_ids[:2] == [1, "1"]
    assert type(runtime.context_ids[0]) is int and type(runtime.context_ids[1]) is str

    for request_id, method, field in (
        (10, "resources/list", "resources"),
        (11, "resources/templates/list", "resourceTemplates"),
        (12, "prompts/list", "prompts"),
    ):
        response = await receive(connection, writer, modern(request_id, method))
        assert response["result"][field]

    resource = await receive(connection, writer, modern(13, "resources/read", {"uri": "file:///guide.txt"}))
    assert resource["result"]["contents"][0]["text"] == "guide"
    prompt = await receive(connection, writer, modern(14, "prompts/get", {"name": "hello", "arguments": {}}))
    assert prompt["result"]["messages"][0]["content"]["text"] == "hello"

    expected = {
        "object": {"kind": "object"},
        "array": [1, "two"],
        "scalar": 7,
        "null": None,
    }
    for offset, (name, value) in enumerate(expected.items(), start=20):
        response = await receive(connection, writer, modern(offset, "tools/call", {"name": name, "arguments": {}}))
        assert response["result"]["structuredContent"] == value

    tool_error = await receive(connection, writer, modern(30, "tools/call", {"name": "tool_error", "arguments": {}}))
    assert tool_error["result"]["isError"] is True
    assert tool_error["result"]["_meta"]["io.github.rmusser01.mcp-unified/error"]["reasonCode"] == "unavailable"
    too_large = await receive(connection, writer, modern(31, "tools/call", {"name": "result_too_large", "arguments": {}}))
    assert too_large["error"]["code"] == -33001
    application_error = await receive(connection, writer, modern(35, "tools/call", {"name": "application_error", "arguments": {}}))
    assert application_error["error"]["code"] == -33002
    invalid = await receive(connection, writer, modern(32, "tools/call", {"name": "invalid_result", "arguments": {}}))
    assert invalid["error"]["code"] == -32603
    missing = await receive(connection, writer, modern(33, "resources/read", {"uri": "file:///missing"}))
    assert missing["error"]["code"] == -32602
    prompt_error = await receive(connection, writer, modern(34, "prompts/get", {"name": "missing", "arguments": {}}))
    assert prompt_error["error"]["code"] == -32602

    runtime.release.clear()
    await connection.receive(modern(40, "tools/call", {"name": "slow", "arguments": {}}))
    await runtime.started.wait()
    before_rejection = len(writer.values)
    await connection.receive(modern(41, "ping"))
    assert len(writer.values) == before_rejection + 1
    rejected = writer.values[-1]
    assert rejected["error"]["code"] == -32000
    await connection.receive({"jsonrpc": "2.0", "method": "notifications/cancelled", "params": {"requestId": 40}})
    runtime.release.set()
    await connection.wait_for_idle()
    assert not any(value.get("id") == 40 for value in writer.values)
    token = runtime.tokens[-1]
    assert token.cancelled is True
    assert await token.wait() is None
    await connection.shutdown()

    empty_writer = ValueWriter()
    empty_connection = GatewayProtocolConnection(Runtime(empty=True), empty_writer, limits=limits)
    for request_id, method, field in (
        (50, "tools/list", "tools"),
        (51, "resources/list", "resources"),
        (52, "resources/templates/list", "resourceTemplates"),
        (53, "prompts/list", "prompts"),
    ):
        response = await receive(empty_connection, empty_writer, modern(request_id, method))
        assert response["result"][field] == []
    await empty_connection.shutdown()

    legacy_runtime = Runtime()
    legacy_writer = ValueWriter()
    legacy = GatewayProtocolConnection(legacy_runtime, legacy_writer, limits=limits)
    initialized = await receive(
        legacy,
        legacy_writer,
        {"jsonrpc": "2.0", "id": "init", "method": "initialize", "params": {
            "protocolVersion": "2025-11-25", "capabilities": {},
            "clientInfo": {"name": "artifact-consumer", "version": "1"},
        }},
    )
    assert initialized["result"]["protocolVersion"] == "2025-11-25"
    legacy_array = await receive(
        legacy,
        legacy_writer,
        {"jsonrpc": "2.0", "id": 60, "method": "tools/call", "params": {"name": "array", "arguments": {}}},
    )
    assert "structuredContent" not in legacy_array["result"]
    assert legacy_array["result"]["content"] == [{"type": "text", "text": '[1,"two"]'}]
    await legacy.shutdown()

    stdio_writer = ByteWriter()
    ping = json.dumps(modern("stdio", "ping"), separators=(",", ":")).encode() + b"\n"
    exit_code = await serve_stdio(
        Runtime(empty=True),
        input_stream=ByteReader(ping, b""),
        output_stream=stdio_writer,
        limits=GatewayLimits(max_in_flight=1),
    )
    assert exit_code == 0
    assert len(stdio_writer.chunks) == 1
    assert json.loads(stdio_writer.chunks[0])["id"] == "stdio"


if __name__ == "__main__":
    asyncio.run(main())
    print(MARKER)
"""


@pytest.fixture(scope="module")
def standalone_protocol_distributions(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, Path]:
    """Build one wheel and one sdist for installed downstream checks."""

    return _build_standalone_distributions(tmp_path_factory.mktemp("mcp_unified_protocol_artifacts"))


def _venv_python(venv_dir: Path) -> Path:
    """Return the virtualenv Python path on the active platform."""

    return venv_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _sanitized_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    """Return an environment that cannot import package code from the checkout."""

    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    env.update(
        {
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "PIP_NO_CACHE_DIR": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "MCP_UNIFIED_FORBIDDEN_CHECKOUT": str(REPO_ROOT),
            **(extra or {}),
        }
    )
    return env


def _run_checked(command: list[str], *, cwd: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    """Run one artifact command and preserve its captured diagnostics."""

    result = subprocess.run(  # nosec B603
        command,
        cwd=cwd,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, (  # nosec B101
        f"command failed: {command}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    return result


@pytest.mark.parametrize("artifact_index", [0, 1], ids=["wheel", "sdist"])
def test_installed_artifact_supports_strict_downstream_consumer(
    standalone_protocol_distributions: tuple[Path, Path],
    artifact_index: int,
    tmp_path: Path,
) -> None:
    """Removing any public strict seam must break both distribution consumers."""

    artifact = standalone_protocol_distributions[artifact_index]
    venv_dir = tmp_path / "venv"
    work_dir = tmp_path / "consumer"
    work_dir.mkdir()
    script = work_dir / "downstream_consumer.py"
    script.write_text(_DOWNSTREAM_CONSUMER, encoding="utf-8")
    env = _sanitized_env()
    _run_checked(
        [sys.executable, "-m", "venv", str(venv_dir)],
        cwd=work_dir,
        env=env,
    )
    python = _venv_python(venv_dir)

    _run_checked([str(python), "-m", "pip", "install", str(artifact)], cwd=work_dir, env=env)
    result = _run_checked([str(python), str(script)], cwd=work_dir, env=env)

    assert result.stdout == f"{SUCCESS_MARKER}\n"  # nosec B101


def test_rc_artifact_gate_requires_installed_protocol_suites_and_provenance(
    standalone_protocol_distributions: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dropping installed suites, dependency checks, or fixture provenance must fail."""

    dist_dir = tmp_path / "dist"
    dist_dir.mkdir()
    for artifact in standalone_protocol_distributions:
        (dist_dir / artifact.name).write_bytes(artifact.read_bytes())
    paths = mcp_unified_rc.RcPaths(
        repo_root=REPO_ROOT,
        package_project=REPO_ROOT / "apps" / "mcp-unified",
        package_src=REPO_ROOT / "apps" / "mcp-unified" / "src" / "mcp_unified",
        evidence_dir=tmp_path / "evidence",
        dist_dir=dist_dir,
    )
    calls: list[dict[str, Any]] = []

    def fake_run_command(
        command: list[str],
        *,
        cwd: Path,
        timeout: int = 180,
        env: dict[str, str] | None = None,
    ) -> mcp_unified_rc.RcCommandResult:
        calls.append({"command": command, "cwd": cwd, "timeout": timeout, "env": env})
        return mcp_unified_rc.RcCommandResult(
            command=command,
            cwd=str(cwd),
            returncode=0,
            stdout="",
            stderr="",
            duration_ms=0,
        )

    monkeypatch.setattr(mcp_unified_rc, "run_command", fake_run_command)
    monkeypatch.setattr(venv.EnvBuilder, "create", lambda self, target: None)
    recorder = mcp_unified_rc._new_recorder(paths)

    redacted = mcp_unified_rc._redact_evidence_text(
        "../../../../var/folders/private-run/protocol-test.py",
        recorder,
    )
    assert "var/folders" not in redacted

    mcp_unified_rc._run_artifact_gate(paths, recorder)

    results = {entry["name"]: entry for entry in recorder.results}
    assert results["wheel_jsonschema_base_dependency"]["status"] == "passed"
    assert results["sdist_jsonschema_base_dependency"]["status"] == "passed"
    provenance = results["normative_fixture_provenance"]
    assert provenance["status"] == "passed"
    assert provenance["details"] == {
        "commit": FIXTURE_COMMIT,
        "sha256": FIXTURE_SHA256,
    }
    assert results["wheel_installed_protocol_suites"]["status"] == "passed"
    assert results["sdist_installed_protocol_suites"]["status"] == "passed"
    assert results["wheel_installed_protocol_import"]["status"] == "passed"
    assert results["sdist_installed_protocol_import"]["status"] == "passed"
    assert results["installed_artifact_consumer"]["status"] == "passed"

    protocol_calls = [
        call
        for call in calls
        if all(any(str(argument).endswith(suite) for argument in call["command"]) for suite in PROTOCOL_SUITES)
    ]
    assert len(protocol_calls) == 2
    assert all(not (call["env"] or {}).get("PYTHONPATH") for call in protocol_calls)
    assert all((call["env"] or {}).get("PYTHONNOUSERSITE") == "1" for call in protocol_calls)
    assert all(call["cwd"] != REPO_ROOT for call in protocol_calls)
    import_calls = [
        call for call in calls if any("mcp_unified.__file__" in str(argument) for argument in call["command"])
    ]
    assert len(import_calls) == 2
    install_calls = [call for call in calls if call["command"][2:4] == ["pip", "install"]]
    assert len(install_calls) == 2
    assert all((call["env"] or {}).get("PIP_NO_CACHE_DIR") == "1" for call in install_calls)
    consumer_calls = [
        call
        for call in calls
        if any(str(argument).endswith("test_gateway_protocol_artifact_consumer.py") for argument in call["command"])
    ]
    assert len(consumer_calls) == 1
