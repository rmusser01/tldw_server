from __future__ import annotations

import asyncio
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tldw_Server_API.app.core.MCP_unified.command_runtime.adapters import (
    derive_step_idempotency_key,
)
from tldw_Server_API.app.core.MCP_unified.command_runtime.executor import CommandRuntimeExecutor
from tldw_Server_API.app.core.MCP_unified.modules.base import ModuleConfig
from tldw_Server_API.app.core.MCP_unified.modules.implementations import (
    run_command_module as run_command_module_module,
)
from tldw_Server_API.app.core.MCP_unified.modules.implementations.run_command_module import (
    RunCommandModule,
)
from tldw_Server_API.app.core.MCP_unified.protocol import (
    ApprovalRequiredError,
    MCPProtocol,
    RequestContext,
)


@dataclass
class _PreparedCall:
    params: dict[str, Any]
    idempotency_key: str | None = None


class _ProtocolStub:
    def __init__(self) -> None:
        self.prepare_calls: list[_PreparedCall] = []
        self.execute_calls: list[_PreparedCall] = []
        self.tools_list_calls = 0
        self.prepare_errors: dict[str, BaseException] = {}
        self.execute_errors: dict[str, BaseException] = {}
        self.read_text_content = "hello"

    async def _handle_tools_list(self, params: dict[str, Any], context: RequestContext) -> dict[str, Any]:
        del params, context
        self.tools_list_calls += 1
        return {
            "tools": [
                {"name": "fs.list", "module": "filesystem", "canExecute": True},
                {"name": "fs.read", "module": "filesystem", "canExecute": True},
                {"name": "fs.read_text", "module": "filesystem", "canExecute": True},
                {"name": "fs.write", "module": "filesystem", "canExecute": True},
                {"name": "fs.write_text", "module": "filesystem", "canExecute": True},
                {"name": "fs.stat", "module": "filesystem", "canExecute": True},
                {"name": "fs.glob", "module": "filesystem", "canExecute": True},
                {"name": "fs.grep", "module": "filesystem", "canExecute": True},
            ]
        }

    async def prepare_tool_call(
        self,
        *,
        params: dict[str, Any],
        context: RequestContext,
        idempotency_key: str | None = None,
    ) -> _PreparedCall:
        del context
        prepared = _PreparedCall(params=dict(params), idempotency_key=idempotency_key)
        self.prepare_calls.append(prepared)
        tool_name = str(params.get("name") or "")
        error = self.prepare_errors.get(tool_name)
        if error is not None:
            raise error
        return prepared

    async def execute_prepared_tool_call(self, prepared: _PreparedCall) -> dict[str, Any]:
        self.execute_calls.append(prepared)
        tool_name = str(prepared.params.get("name") or "")
        error = self.execute_errors.get(tool_name)
        if error is not None:
            raise error
        if tool_name == "fs.list":
            return {
                "content": [
                    {
                        "type": "json",
                        "json": {
                            "path": ".",
                            "entries": [
                                {"name": "alpha.txt", "type": "file"},
                                {"name": "docs", "type": "directory"},
                            ],
                        },
                    }
                ],
                "tool": tool_name,
            }
        if tool_name == "fs.write_text":
            return {
                "content": [{"type": "json", "json": {"path": "notes.txt", "bytes_written": 5}}],
                "tool": tool_name,
            }
        if tool_name == "fs.write":
            return {
                "content": [{"type": "json", "json": {"path": "notes.txt", "bytes_written": 5}}],
                "tool": tool_name,
            }
        if tool_name == "fs.read":
            return {
                "content": [{"type": "json", "json": {"path": "notes.txt", "content": self.read_text_content}}],
                "tool": tool_name,
            }
        if tool_name == "fs.read_text":
            return {
                "content": [{"type": "json", "json": {"path": "notes.txt", "text": self.read_text_content}}],
                "tool": tool_name,
            }
        if tool_name == "fs.stat":
            return {
                "content": [
                    {
                        "type": "json",
                        "json": {
                            "path": "docs/readme.md",
                            "type": "file",
                            "size_bytes": 42,
                        },
                    }
                ],
                "tool": tool_name,
            }
        if tool_name == "fs.glob":
            return {
                "content": [
                    {
                        "type": "json",
                        "json": {
                            "matches": [
                                {"path": "src/app.py", "type": "file"},
                                {"path": "src/pkg", "type": "directory"},
                            ]
                        },
                    }
                ],
                "tool": tool_name,
            }
        if tool_name == "fs.grep":
            return {
                "content": [
                    {
                        "type": "json",
                        "json": {
                            "matches": [
                                {
                                    "path": "src/app.py",
                                    "line_number": 7,
                                    "line": "TODO: wire facade",
                                }
                            ]
                        },
                    }
                ],
                "tool": tool_name,
            }
        raise AssertionError(f"Unexpected tool execution: {tool_name}")


class _WorkspaceRootResolverStub:
    def __init__(self, workspace_root: Path) -> None:
        self.workspace_root = workspace_root
        self.calls: list[dict[str, Any]] = []

    async def resolve_for_context(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(dict(kwargs))
        return {
            "workspace_root": str(self.workspace_root),
            "workspace_id": kwargs.get("workspace_id") or "workspace-1",
            "source": "test",
            "reason": None,
        }


def _build_module(protocol: _ProtocolStub) -> RunCommandModule:
    return RunCommandModule(
        ModuleConfig(
            name="run",
            settings={"protocol": protocol},
        )
    )


@pytest.mark.unit
async def test_run_module_exposes_governed_bash_and_shell_alias_tools() -> None:
    protocol = _ProtocolStub()
    module = _build_module(protocol)

    tools = await module.get_tools()

    by_name = {tool["name"]: tool for tool in tools}
    assert list(by_name) == ["run", "bash", "shell", "powershell", "pwsh"]
    assert by_name["run"]["metadata"].get("canonical_tool") is None
    for alias_name in ("bash", "shell", "powershell", "pwsh"):
        alias = by_name[alias_name]
        assert alias["metadata"]["canonical_tool"] == "run"
        assert "not a raw host shell" in alias["description"]
        assert alias["inputSchema"]["properties"] == by_name["run"]["inputSchema"]["properties"]
        assert "timeout_seconds" in alias["inputSchema"]["properties"]
        assert "timeoutSeconds" in alias["inputSchema"]["properties"]
        assert "cwd" in alias["inputSchema"]["properties"]
        assert "workingDirectory" in alias["inputSchema"]["properties"]
        assert "retainOutputArtifacts" in alias["inputSchema"]["properties"]
        assert "retain_output_artifacts" in alias["inputSchema"]["properties"]
        assert "sandboxSessionId" in alias["inputSchema"]["properties"]
        assert "sandbox_session_id" in alias["inputSchema"]["properties"]
        assert "envFile" in alias["inputSchema"]["properties"]
        assert "env_file" in alias["inputSchema"]["properties"]
        assert "shellName" in alias["inputSchema"]["properties"]
        assert "shell_name" in alias["inputSchema"]["properties"]


@pytest.mark.asyncio
async def test_run_ls_uses_fs_list_and_returns_footer() -> None:
    protocol = _ProtocolStub()
    module = _build_module(protocol)
    context = RequestContext(request_id="run-ls", user_id="1", client_id="unit")

    rendered = await module.execute_tool("run", {"command": "ls"}, context=context)

    assert "alpha.txt" in rendered
    assert "docs/" in rendered
    assert "[exit:0 |" in rendered
    assert len(protocol.prepare_calls) == 1
    assert protocol.prepare_calls[0].params["name"] == "fs.list"
    assert protocol.prepare_calls[0].params["arguments"] == {"path": "."}


@pytest.mark.asyncio
async def test_shell_alias_uses_governed_run_implementation_without_raw_shell_delegation() -> None:
    protocol = _ProtocolStub()
    module = _build_module(protocol)
    context = RequestContext(request_id="run-shell-alias", user_id="1", client_id="unit")

    rendered = await module.execute_tool("shell", {"command": "echo unsafe"}, context=context)

    assert "Unknown command: echo" in rendered
    assert "[exit:127 |" in rendered
    assert protocol.prepare_calls == []
    assert protocol.execute_calls == []


@pytest.mark.asyncio
async def test_powershell_alias_uses_governed_run_implementation_without_raw_shell_delegation() -> None:
    """PowerShell aliases must stay governed virtual CLI facades."""

    protocol = _ProtocolStub()
    module = _build_module(protocol)
    context = RequestContext(request_id="run-powershell-alias", user_id="1", client_id="unit")

    rendered = await module.execute_tool("powershell", {"command": "ls"}, context=context)

    assert "alpha.txt" in rendered
    assert "docs/" in rendered
    assert "[exit:0 |" in rendered
    assert [call.params["name"] for call in protocol.prepare_calls] == ["fs.list"]
    assert [call.params["arguments"] for call in protocol.prepare_calls] == [{"path": "."}]


@pytest.mark.asyncio
async def test_run_timeout_seconds_cancels_slow_governed_execution(monkeypatch: pytest.MonkeyPatch) -> None:
    """Timeout handling should cancel an in-flight governed backend call deterministically."""

    class _BlockedProtocolStub(_ProtocolStub):
        def __init__(self) -> None:
            super().__init__()
            self.started = asyncio.Event()

        async def execute_prepared_tool_call(self, prepared: _PreparedCall) -> dict[str, Any]:
            self.execute_calls.append(prepared)
            self.started.set()
            await asyncio.Future()
            raise AssertionError("blocked backend call should be cancelled")

    protocol = _BlockedProtocolStub()

    async def _deterministic_wait_for(awaitable: Any, *, timeout: float | None = None) -> Any:
        assert timeout == 0.01
        task = asyncio.create_task(awaitable)
        await protocol.started.wait()
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task
        raise TimeoutError

    monkeypatch.setattr(run_command_module_module.asyncio, "wait_for", _deterministic_wait_for)
    module = _build_module(protocol)
    context = RequestContext(request_id="run-timeout", user_id="1", client_id="unit")

    rendered = await module.execute_tool("bash", {"command": "ls", "timeout_seconds": 0.01}, context=context)

    assert "Command timed out after 0.01s" in rendered
    assert "[exit:124 |" in rendered
    assert [call.params["name"] for call in protocol.prepare_calls] == ["fs.list"]
    assert len(protocol.execute_calls) == 1


@pytest.mark.parametrize(
    "arguments",
    [
        {"command": "ls", "timeout_seconds": 0},
        {"command": "ls", "timeout_seconds": "soon"},
        {"command": "ls", "timeout_seconds": float("inf")},
        {"command": "ls", "timeout_seconds": float("nan")},
        {"command": "ls", "timeout_seconds": 5, "timeoutSeconds": 6},
    ],
)
@pytest.mark.asyncio
async def test_run_rejects_invalid_timeout_arguments(arguments: dict[str, Any]) -> None:
    """Timeout arguments must be finite positive numbers with matching aliases."""

    protocol = _ProtocolStub()
    module = _build_module(protocol)

    with pytest.raises(ValueError, match="timeout"):
        await module.execute_tool("run", arguments, context=RequestContext(request_id="run-timeout-invalid"))


@pytest.mark.asyncio
async def test_run_applies_cwd_to_relative_workspace_file_arguments() -> None:
    """Relative filesystem arguments should be scoped under the requested cwd."""

    protocol = _ProtocolStub()
    module = _build_module(protocol)
    context = RequestContext(request_id="run-cwd", user_id="1", client_id="unit")

    await module.execute_tool(
        "run",
        {"command": "cat notes.txt ; write out.txt hello ; ls . ; rg TODO .", "cwd": "docs"},
        context=context,
    )

    assert [call.params["name"] for call in protocol.prepare_calls] == [
        "fs.read",
        "fs.write_text",
        "fs.list",
        "fs.grep",
    ]
    assert [call.params["arguments"] for call in protocol.prepare_calls] == [
        {"path": "docs/notes.txt"},
        {"path": "docs/out.txt", "content": "hello"},
        {"path": "docs"},
        {"pattern": "TODO", "base_path": "docs"},
    ]


@pytest.mark.asyncio
async def test_run_cwd_preserves_whitespace_in_relative_file_arguments() -> None:
    """Cwd path rewriting must not trim valid whitespace in path tokens."""

    protocol = _ProtocolStub()
    module = _build_module(protocol)
    context = RequestContext(request_id="run-cwd-whitespace", user_id="1", client_id="unit")

    await module.execute_tool(
        "run",
        {"command": 'cat " notes.txt " ; write " out.txt " hello ; ls " sub dir "', "cwd": "docs"},
        context=context,
    )

    assert [call.params["arguments"] for call in protocol.prepare_calls] == [
        {"path": "docs/ notes.txt "},
        {"path": "docs/ out.txt ", "content": "hello"},
        {"path": "docs/ sub dir "},
    ]


@pytest.mark.asyncio
async def test_run_cwd_participates_in_nested_idempotency_keys() -> None:
    """Cwd scopes should salt nested idempotency keys."""

    protocol = _ProtocolStub()
    module = _build_module(protocol)
    context = RequestContext(request_id="run-cwd-idempotency", user_id="1", client_id="unit")

    await module.execute_tool(
        "run",
        {"command": "cat notes.txt", "cwd": "docs", "idempotencyKey": "parent-idem-cwd"},
        context=context,
    )
    await module.execute_tool(
        "run",
        {"command": "cat notes.txt", "cwd": "src", "idempotencyKey": "parent-idem-cwd"},
        context=context,
    )

    assert protocol.prepare_calls[0].idempotency_key != protocol.prepare_calls[1].idempotency_key
    assert protocol.prepare_calls[0].idempotency_key.startswith("parent-idem-cwd:")
    assert protocol.prepare_calls[1].idempotency_key.startswith("parent-idem-cwd:")


def test_scoped_parent_idempotency_key_uses_unambiguous_scope_serialization() -> None:
    """Crafted scope values must not collide with separate scope components."""

    newline_encoded_key = RunCommandModule._scoped_parent_idempotency_key(
        "parent-idem",
        "docs\nsandbox_session_id=sandbox-1",
        None,
    )
    component_key = RunCommandModule._scoped_parent_idempotency_key(
        "parent-idem",
        "docs",
        "sandbox-1",
    )

    assert newline_encoded_key != component_key


@pytest.mark.unit
async def test_run_shell_name_participates_in_nested_idempotency_scope() -> None:
    """Explicit shell selection should scope nested idempotency without changing backend routing."""

    protocol = _ProtocolStub()
    module = _build_module(protocol)
    context = RequestContext(request_id="run-shell-name-idem", user_id="1", client_id="unit")

    await module.execute_tool(
        "run",
        {"command": "cat notes.txt", "shellName": "bash", "idempotencyKey": "parent-idem-shell"},
        context=context,
    )
    await module.execute_tool(
        "run",
        {"command": "cat notes.txt", "shellName": "powershell", "idempotencyKey": "parent-idem-shell"},
        context=context,
    )

    first_key = protocol.prepare_calls[0].idempotency_key
    second_key = protocol.prepare_calls[1].idempotency_key
    assert first_key != second_key
    assert first_key is not None and first_key.startswith("parent-idem-shell:")
    assert second_key is not None and second_key.startswith("parent-idem-shell:")
    assert [call.params["name"] for call in protocol.prepare_calls] == ["fs.read", "fs.read"]


@pytest.mark.parametrize(
    "tool_name, arguments",
    [
        ("run", {"command": "ls", "shellName": ""}),
        ("run", {"command": "ls", "shellName": "zsh"}),
        ("run", {"command": "ls", "shellName": 123}),
        ("run", {"command": "ls", "shellName": "bash", "shell_name": "powershell"}),
        ("bash", {"command": "ls", "shellName": "powershell"}),
        ("powershell", {"command": "ls", "shell_name": "bash"}),
        ("pwsh", {"command": "ls", "shellName": "powershell"}),
    ],
)
@pytest.mark.unit
async def test_run_rejects_invalid_or_conflicting_shell_name_arguments(
    tool_name: str,
    arguments: dict[str, Any],
) -> None:
    """Shell selection must be explicit, known, and compatible with pinned aliases."""

    protocol = _ProtocolStub()
    module = _build_module(protocol)

    with pytest.raises(ValueError, match="shellName|shell_name"):
        await module.execute_tool(tool_name, arguments, context=RequestContext(request_id="run-shell-name-invalid"))


@pytest.mark.asyncio
async def test_run_sandbox_session_id_uses_session_backed_sandbox_run() -> None:
    """Sandbox commands should pass sandboxSessionId as sandbox.run session_id."""

    class _SandboxProtocolStub(_ProtocolStub):
        async def _handle_tools_list(self, params: dict[str, Any], context: RequestContext) -> dict[str, Any]:
            del params, context
            return {"tools": [{"name": "sandbox.run", "module": "sandbox", "canExecute": True}]}

        async def execute_prepared_tool_call(self, prepared: _PreparedCall) -> dict[str, Any]:
            self.execute_calls.append(prepared)
            return {
                "content": [
                    {
                        "type": "json",
                        "json": {
                            "status": "completed",
                            "session_id": prepared.params["arguments"].get("session_id"),
                        },
                    }
                ],
                "tool": "sandbox.run",
            }

    protocol = _SandboxProtocolStub()
    module = _build_module(protocol)
    context = RequestContext(request_id="run-sandbox-session", user_id="1", client_id="unit")

    rendered = await module.execute_tool(
        "run",
        {"command": "sandbox python -V", "sandboxSessionId": "sandbox-session-1"},
        context=context,
    )

    assert "sandbox-session-1" in rendered
    assert protocol.prepare_calls[0].params["name"] == "sandbox.run"
    assert protocol.prepare_calls[0].params["arguments"] == {
        "session_id": "sandbox-session-1",
        "command": ["python", "-V"],
    }


@pytest.mark.unit
async def test_run_env_file_is_forwarded_only_to_sandbox_run_without_rendering_values(tmp_path: Path) -> None:
    """envFile should load workspace env values into sandbox.run without rendering secrets."""

    class _SandboxProtocolStub(_ProtocolStub):
        async def _handle_tools_list(self, params: dict[str, Any], context: RequestContext) -> dict[str, Any]:
            del params, context
            return {"tools": [{"name": "sandbox.run", "module": "sandbox", "canExecute": True}]}

        async def execute_prepared_tool_call(self, prepared: _PreparedCall) -> dict[str, Any]:
            self.execute_calls.append(prepared)
            return {"content": [{"type": "json", "json": {"status": "completed"}}], "tool": "sandbox.run"}

    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    env_file = workspace_root / ".env.sandbox"
    env_file.write_text(
        "\n".join(
            [
                "# comment",
                "API_TOKEN=super-secret",
                "export FEATURE_FLAG=enabled",
                "QUOTED=\"two words\"",
                "SINGLE='single quoted'",
            ]
        ),
        encoding="utf-8",
    )
    protocol = _SandboxProtocolStub()
    module = RunCommandModule(
        ModuleConfig(
            name="run",
            settings={
                "protocol": protocol,
                "workspace_root_resolver": _WorkspaceRootResolverStub(workspace_root),
            },
        )
    )
    context = RequestContext(request_id="run-env-file", user_id="1", client_id="unit")

    rendered = await module.execute_tool(
        "run",
        {"command": "sandbox python app.py", "envFile": ".env.sandbox"},
        context=context,
    )

    assert "[exit:0 |" in rendered
    assert "super-secret" not in rendered
    assert protocol.prepare_calls[0].params["name"] == "sandbox.run"
    assert protocol.prepare_calls[0].params["arguments"] == {
        "base_image": "python:3.11",
        "command": ["python", "app.py"],
        "env": {
            "API_TOKEN": "super-secret",
            "FEATURE_FLAG": "enabled",
            "QUOTED": "two words",
            "SINGLE": "single quoted",
        },
    }


@pytest.mark.unit
async def test_run_env_file_uses_cwd_to_resolve_workspace_relative_file(tmp_path: Path) -> None:
    """envFile should resolve beneath cwd without escaping the workspace root."""

    class _SandboxProtocolStub(_ProtocolStub):
        async def _handle_tools_list(self, params: dict[str, Any], context: RequestContext) -> dict[str, Any]:
            del params, context
            return {"tools": [{"name": "sandbox.run", "module": "sandbox", "canExecute": True}]}

        async def execute_prepared_tool_call(self, prepared: _PreparedCall) -> dict[str, Any]:
            self.execute_calls.append(prepared)
            return {"content": [{"type": "json", "json": {"ok": True}}], "tool": "sandbox.run"}

    workspace_root = tmp_path / "workspace"
    (workspace_root / "apps" / "api").mkdir(parents=True)
    (workspace_root / "apps" / "api" / ".env").write_text("APP_ENV=test\n", encoding="utf-8")
    protocol = _SandboxProtocolStub()
    module = RunCommandModule(
        ModuleConfig(
            name="run",
            settings={
                "protocol": protocol,
                "workspace_root_resolver": _WorkspaceRootResolverStub(workspace_root),
            },
        )
    )
    context = RequestContext(request_id="run-env-file-cwd", user_id="1", client_id="unit")

    await module.execute_tool(
        "run",
        {"command": "sandbox python app.py", "cwd": "apps/api", "envFile": ".env"},
        context=context,
    )

    assert protocol.prepare_calls[0].params["arguments"]["env"] == {"APP_ENV": "test"}


@pytest.mark.unit
async def test_run_env_file_participates_in_nested_idempotency_keys_without_secret_values(tmp_path: Path) -> None:
    """Env-file content changes should alter nested idempotency without exposing values in keys."""

    class _SandboxProtocolStub(_ProtocolStub):
        async def _handle_tools_list(self, params: dict[str, Any], context: RequestContext) -> dict[str, Any]:
            del params, context
            return {"tools": [{"name": "sandbox.run", "module": "sandbox", "canExecute": True}]}

        async def execute_prepared_tool_call(self, prepared: _PreparedCall) -> dict[str, Any]:
            self.execute_calls.append(prepared)
            return {"content": [{"type": "json", "json": {"ok": True}}], "tool": "sandbox.run"}

    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    env_file = workspace_root / ".env"
    env_file.write_text("TOKEN=first-secret\n", encoding="utf-8")
    protocol = _SandboxProtocolStub()
    module = RunCommandModule(
        ModuleConfig(
            name="run",
            settings={
                "protocol": protocol,
                "workspace_root_resolver": _WorkspaceRootResolverStub(workspace_root),
            },
        )
    )
    context = RequestContext(request_id="run-env-file-idem", user_id="1", client_id="unit")

    await module.execute_tool(
        "run",
        {"command": "sandbox python -V", "envFile": ".env", "idempotencyKey": "parent-idem-env"},
        context=context,
    )
    env_file.write_text("TOKEN=second-secret\n", encoding="utf-8")
    await module.execute_tool(
        "run",
        {"command": "sandbox python -V", "envFile": ".env", "idempotencyKey": "parent-idem-env"},
        context=context,
    )

    first_key = protocol.prepare_calls[0].idempotency_key
    second_key = protocol.prepare_calls[1].idempotency_key
    assert first_key != second_key
    assert first_key is not None and first_key.startswith("parent-idem-env:")
    assert second_key is not None and second_key.startswith("parent-idem-env:")
    assert "first-secret" not in first_key
    assert "second-secret" not in second_key


@pytest.mark.parametrize(
    "arguments",
    [
        {"command": "sandbox python -V", "envFile": ""},
        {"command": "sandbox python -V", "envFile": 123},
        {"command": "sandbox python -V", "envFile": "/tmp/.env"},
        {"command": "sandbox python -V", "envFile": "../.env"},
        {"command": "sandbox python -V", "envFile": "~/.env"},
        {"command": "sandbox python -V", "envFile": ".env", "env_file": "other.env"},
    ],
)
@pytest.mark.unit
async def test_run_rejects_invalid_env_file_arguments(arguments: dict[str, Any]) -> None:
    """envFile aliases must be safe, relative, and consistent."""

    protocol = _ProtocolStub()
    module = _build_module(protocol)

    with pytest.raises(ValueError, match="envFile|env_file"):
        await module.execute_tool("run", arguments, context=RequestContext(request_id="run-env-file-invalid"))


@pytest.mark.unit
async def test_run_env_file_rejects_non_sandbox_command_chains(tmp_path: Path) -> None:
    """envFile should fail closed instead of being silently ignored outside sandbox steps."""

    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    (workspace_root / ".env").write_text("TOKEN=secret\n", encoding="utf-8")
    protocol = _ProtocolStub()
    module = RunCommandModule(
        ModuleConfig(
            name="run",
            settings={
                "protocol": protocol,
                "workspace_root_resolver": _WorkspaceRootResolverStub(workspace_root),
            },
        )
    )

    rendered = await module.execute_tool(
        "run",
        {"command": "ls", "envFile": ".env"},
        context=RequestContext(request_id="run-env-file-non-sandbox"),
    )

    assert "envFile is only supported for command chains that include sandbox" in rendered
    assert protocol.prepare_calls == []
    assert protocol.execute_calls == []


@pytest.mark.unit
async def test_run_env_file_rejects_malformed_files(tmp_path: Path) -> None:
    """Malformed env files should fail before any sandbox tool call is prepared."""

    class _SandboxProtocolStub(_ProtocolStub):
        async def _handle_tools_list(self, params: dict[str, Any], context: RequestContext) -> dict[str, Any]:
            del params, context
            return {"tools": [{"name": "sandbox.run", "module": "sandbox", "canExecute": True}]}

    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    (workspace_root / ".env").write_text("GOOD=value\nNO_EQUALS\n", encoding="utf-8")
    protocol = _SandboxProtocolStub()
    module = RunCommandModule(
        ModuleConfig(
            name="run",
            settings={
                "protocol": protocol,
                "workspace_root_resolver": _WorkspaceRootResolverStub(workspace_root),
            },
        )
    )

    rendered = await module.execute_tool(
        "run",
        {"command": "sandbox python -V", "envFile": ".env"},
        context=RequestContext(request_id="run-env-file-malformed"),
    )

    assert "envFile line 2 must be KEY=value" in rendered
    assert protocol.prepare_calls == []
    assert protocol.execute_calls == []


@pytest.mark.unit
def test_run_env_file_parser_accepts_bom_and_rejects_unicode_keys() -> None:
    """Env parser should handle UTF-8 BOMs but enforce ASCII variable names."""

    parsed = RunCommandModule._parse_env_file_bytes(b"\xef\xbb\xbfAPI_TOKEN=ok\n")

    assert parsed == {"API_TOKEN": "ok"}
    with pytest.raises(run_command_module_module.RunEnvFileValidationError, match="invalid variable name"):
        RunCommandModule._parse_env_file_bytes("ÄPI_TOKEN=bad\n".encode())
    with pytest.raises(run_command_module_module.RunEnvFileValidationError, match="invalid variable name"):
        RunCommandModule._parse_env_file_bytes("ＡPI_TOKEN=bad\n".encode())


@pytest.mark.unit
def test_run_env_file_reader_uses_descriptor_read_without_path_read_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Env reader should not re-open the path after descriptor validation."""

    env_file = tmp_path / ".env"
    env_file.write_text("API_TOKEN=ok\n", encoding="utf-8")

    def _fail_read_bytes(_path: Path) -> bytes:
        raise AssertionError("Path.read_bytes must not be used for envFile reads")

    monkeypatch.setattr(Path, "read_bytes", _fail_read_bytes)

    assert RunCommandModule._read_env_file_bytes(env_file) == b"API_TOKEN=ok\n"


@pytest.mark.unit
def test_run_env_file_reader_maps_open_errors(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Env reader should surface OSError as a structured env-file validation error."""

    env_file = tmp_path / ".env"
    env_file.write_text("API_TOKEN=ok\n", encoding="utf-8")

    def _raise_open(*_args: Any, **_kwargs: Any) -> int:
        raise PermissionError("blocked")

    monkeypatch.setattr(run_command_module_module.os, "open", _raise_open)

    with pytest.raises(run_command_module_module.RunEnvFileValidationError, match="could not be read"):
        RunCommandModule._read_env_file_bytes(env_file)


@pytest.mark.asyncio
async def test_run_sandbox_session_id_participates_in_nested_idempotency_keys() -> None:
    """Sandbox session scopes should salt nested idempotency keys."""

    class _SandboxProtocolStub(_ProtocolStub):
        async def _handle_tools_list(self, params: dict[str, Any], context: RequestContext) -> dict[str, Any]:
            del params, context
            return {"tools": [{"name": "sandbox.run", "module": "sandbox", "canExecute": True}]}

        async def execute_prepared_tool_call(self, prepared: _PreparedCall) -> dict[str, Any]:
            self.execute_calls.append(prepared)
            return {"content": [{"type": "json", "json": {"ok": True}}], "tool": "sandbox.run"}

    protocol = _SandboxProtocolStub()
    module = _build_module(protocol)
    context = RequestContext(request_id="run-sandbox-session-idem", user_id="1", client_id="unit")

    await module.execute_tool(
        "run",
        {
            "command": "sandbox python -V",
            "idempotencyKey": "parent-idem-sandbox",
            "sandboxSessionId": "sandbox-session-1",
        },
        context=context,
    )
    await module.execute_tool(
        "run",
        {
            "command": "sandbox python -V",
            "idempotencyKey": "parent-idem-sandbox",
            "sandboxSessionId": "sandbox-session-2",
        },
        context=context,
    )

    assert protocol.prepare_calls[0].idempotency_key != protocol.prepare_calls[1].idempotency_key
    assert protocol.prepare_calls[0].idempotency_key.startswith("parent-idem-sandbox:")
    assert protocol.prepare_calls[1].idempotency_key.startswith("parent-idem-sandbox:")


@pytest.mark.parametrize(
    "arguments",
    [
        {"command": "sandbox python -V", "sandboxSessionId": ""},
        {"command": "sandbox python -V", "sandboxSessionId": 123},
        {
            "command": "sandbox python -V",
            "sandboxSessionId": "sandbox-session-1",
            "sandbox_session_id": "sandbox-session-2",
        },
    ],
)
@pytest.mark.asyncio
async def test_run_rejects_invalid_sandbox_session_arguments(arguments: dict[str, Any]) -> None:
    """Sandbox session aliases must be non-empty strings when provided."""

    protocol = _ProtocolStub()
    module = _build_module(protocol)

    with pytest.raises(ValueError, match="sandboxSessionId|sandbox_session_id"):
        await module.execute_tool("run", arguments, context=RequestContext(request_id="run-sandbox-session-invalid"))


@pytest.mark.parametrize(
    "arguments",
    [
        {"command": "ls", "cwd": "/tmp"},
        {"command": "ls", "cwd": "../private"},
        {"command": "ls", "cwd": "C:\\Users\\example"},
        {"command": "ls", "cwd": "docs", "workingDirectory": "src"},
    ],
)
@pytest.mark.asyncio
async def test_run_rejects_unsafe_cwd_arguments(arguments: dict[str, Any]) -> None:
    """Cwd must remain workspace-relative and alias-consistent."""

    protocol = _ProtocolStub()
    module = _build_module(protocol)

    with pytest.raises(ValueError, match="cwd|workingDirectory"):
        await module.execute_tool("run", arguments, context=RequestContext(request_id="run-cwd-invalid"))


@pytest.mark.parametrize(
    ("command", "expected_message"),
    [
        ("cat notes.txt > out.txt", "Unsupported shell feature: redirection"),
        ("cat $(cat secret.txt)", "Unsupported shell feature: command substitution"),
        ("cat $HOME/.ssh/id_rsa", "Unsupported shell feature: environment expansion"),
        ("TOKEN=secret cat notes.txt", "Unsupported shell feature: environment assignment"),
        ("ls &", "Unsupported shell feature: background execution"),
    ],
)
@pytest.mark.asyncio
async def test_shell_alias_rejects_unsupported_raw_shell_syntax_before_backend_calls(
    command: str,
    expected_message: str,
) -> None:
    """Unsupported raw shell syntax must fail before any governed backend call."""

    protocol = _ProtocolStub()
    module = _build_module(protocol)
    context = RequestContext(request_id="run-shell-unsupported", user_id="1", client_id="unit")

    rendered = await module.execute_tool("bash", {"command": command}, context=context)

    assert expected_message in rendered
    assert "[exit:2 |" in rendered
    assert protocol.prepare_calls == []
    assert protocol.execute_calls == []


@pytest.mark.parametrize(
    ("tool_name", "arguments", "expected_message"),
    [
        (
            "powershell",
            {"command": "& ./script.ps1"},
            "Unsupported PowerShell feature: invocation operator",
        ),
        (
            "pwsh",
            {"command": "& ./script.ps1"},
            "Unsupported PowerShell feature: invocation operator",
        ),
        (
            "run",
            {"command": "ForEach-Object { Get-ChildItem }", "shellName": "powershell"},
            "Unsupported PowerShell feature: script blocks",
        ),
    ],
)
@pytest.mark.unit
async def test_powershell_shell_selection_rejects_unsupported_platform_syntax_before_backend_calls(
    tool_name: str,
    arguments: dict[str, Any],
    expected_message: str,
) -> None:
    """PowerShell-only raw shell syntax must fail before any governed backend call."""

    protocol = _ProtocolStub()
    module = _build_module(protocol)
    context = RequestContext(request_id="run-powershell-unsupported", user_id="1", client_id="unit")

    rendered = await module.execute_tool(tool_name, arguments, context=context)

    assert expected_message in rendered
    assert "[exit:2 |" in rendered
    assert protocol.prepare_calls == []
    assert protocol.execute_calls == []


@pytest.mark.asyncio
async def test_run_filesystem_aliases_route_to_backing_tools() -> None:
    protocol = _ProtocolStub()
    module = _build_module(protocol)
    context = RequestContext(request_id="run-filesystem-aliases", user_id="1", client_id="unit")

    rendered = await module.execute_tool(
        "run",
        {"command": 'stat docs/readme.md ; glob "**/*.py" src ; find "*.md" docs ; rg TODO src ; grep-files FIXME docs'},
        context=context,
    )

    assert "[exit:0 |" in rendered
    assert [call.params["name"] for call in protocol.prepare_calls] == [
        "fs.stat",
        "fs.glob",
        "fs.glob",
        "fs.grep",
        "fs.grep",
    ]
    assert [call.params["arguments"] for call in protocol.prepare_calls] == [
        {"path": "docs/readme.md"},
        {"pattern": "**/*.py", "base_path": "src"},
        {"pattern": "*.md", "base_path": "docs"},
        {"pattern": "TODO", "base_path": "src"},
        {"pattern": "FIXME", "base_path": "docs"},
    ]


@pytest.mark.asyncio
async def test_run_plain_grep_still_filters_stdin_instead_of_calling_fs_grep() -> None:
    protocol = _ProtocolStub()
    protocol.read_text_content = "ERROR one\nINFO two\nerror three\n"
    module = _build_module(protocol)
    context = RequestContext(request_id="run-pure-grep", user_id="1", client_id="unit")

    rendered = await module.execute_tool("run", {"command": "cat app.log | grep ERROR"}, context=context)

    assert "ERROR one" in rendered
    assert "INFO two" not in rendered
    assert [call.params["name"] for call in protocol.prepare_calls] == ["fs.read"]
    assert [call.params["name"] for call in protocol.execute_calls] == ["fs.read"]


@pytest.mark.asyncio
async def test_run_cat_falls_back_to_legacy_read_text_when_structured_read_is_hidden() -> None:
    class _LegacyReadOnlyProtocolStub(_ProtocolStub):
        async def _handle_tools_list(self, params: dict[str, Any], context: RequestContext) -> dict[str, Any]:
            del params, context
            self.tools_list_calls += 1
            return {
                "tools": [
                    {"name": "fs.read_text", "module": "filesystem", "canExecute": True},
                ]
            }

    protocol = _LegacyReadOnlyProtocolStub()
    module = _build_module(protocol)
    context = RequestContext(request_id="run-cat-legacy-read", user_id="1", client_id="unit")

    rendered = await module.execute_tool("run", {"command": "cat notes.txt"}, context=context)

    assert "hello" in rendered
    assert [call.params["name"] for call in protocol.prepare_calls] == ["fs.read_text"]
    assert [call.params["arguments"] for call in protocol.prepare_calls] == [{"path": "notes.txt"}]


@pytest.mark.asyncio
async def test_run_cat_surfaces_structured_read_truncation_metadata() -> None:
    class _TruncatedReadProtocolStub(_ProtocolStub):
        async def execute_prepared_tool_call(self, prepared: _PreparedCall) -> dict[str, Any]:
            self.execute_calls.append(prepared)
            tool_name = str(prepared.params.get("name") or "")
            if tool_name == "fs.read":
                return {
                    "content": [
                        {
                            "type": "json",
                            "json": {
                                "path": "/private/workspace/notes.txt",
                                "content": "first chunk\n",
                                "truncated": True,
                                "bytes_returned": 12,
                                "bytes_total": 128,
                                "lines_returned": 1,
                                "truncation_reason": "byte_limit",
                            },
                        }
                    ],
                    "tool": tool_name,
                }
            return await super().execute_prepared_tool_call(prepared)

    protocol = _TruncatedReadProtocolStub()
    module = _build_module(protocol)
    context = RequestContext(request_id="run-cat-truncated", user_id="1", client_id="unit")

    rendered = await module.execute_tool("run", {"command": "cat notes.txt"}, context=context)

    assert "first chunk" in rendered
    assert "truncated" in rendered.lower()
    assert "bytes_returned=12" in rendered
    assert "bytes_total=128" in rendered
    assert "lines_returned=1" in rendered
    assert "truncation_reason=byte_limit" in rendered
    assert "/private/workspace" not in rendered


@pytest.mark.asyncio
async def test_run_help_policy_filters_filesystem_aliases() -> None:
    class _RestrictedProtocolStub(_ProtocolStub):
        async def _handle_tools_list(self, params: dict[str, Any], context: RequestContext) -> dict[str, Any]:
            del params, context
            return {
                "tools": [
                    {"name": "fs.stat", "module": "filesystem", "canExecute": True},
                    {"name": "fs.grep", "module": "filesystem", "canExecute": True},
                ]
            }

    module = _build_module(_RestrictedProtocolStub())
    context = RequestContext(request_id="run-help-filesystem-aliases", user_id="1", client_id="unit")

    rendered = await module.execute_tool("run", {"command": "--help"}, context=context)

    assert "stat" in rendered
    assert "rg" in rendered
    assert "grep-files" in rendered
    assert "grep" in rendered
    assert "glob" not in rendered
    assert "find" not in rendered


@pytest.mark.asyncio
async def test_run_help_shows_write_create_only_when_structured_write_is_visible() -> None:
    class _StructuredWriteProtocolStub(_ProtocolStub):
        async def _handle_tools_list(self, params: dict[str, Any], context: RequestContext) -> dict[str, Any]:
            del params, context
            return {
                "tools": [
                    {"name": "fs.write", "module": "filesystem", "canExecute": True},
                ]
            }

    module = _build_module(_StructuredWriteProtocolStub())
    context = RequestContext(request_id="run-help-write-create", user_id="1", client_id="unit")

    rendered = await module.execute_tool("run", {"command": "--help"}, context=context)

    commands = {line.split()[0] for line in rendered.splitlines() if line.startswith("  ")}
    assert "write-create" in commands
    assert "write" not in commands


@pytest.mark.asyncio
async def test_run_help_shows_legacy_write_only_when_legacy_write_text_is_visible() -> None:
    class _LegacyWriteProtocolStub(_ProtocolStub):
        async def _handle_tools_list(self, params: dict[str, Any], context: RequestContext) -> dict[str, Any]:
            del params, context
            return {
                "tools": [
                    {"name": "fs.write_text", "module": "filesystem", "canExecute": True},
                ]
            }

    module = _build_module(_LegacyWriteProtocolStub())
    context = RequestContext(request_id="run-help-legacy-write", user_id="1", client_id="unit")

    rendered = await module.execute_tool("run", {"command": "--help"}, context=context)

    commands = {line.split()[0] for line in rendered.splitlines() if line.startswith("  ")}
    assert "write" in commands
    assert "write-create" not in commands


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("command", "command_name"),
    [
        ("knowledge search vector", "knowledge"),
        ("media search lecture", "media"),
        ("mcp tools", "mcp"),
    ],
)
async def test_run_returns_unknown_for_hidden_multi_backend_commands(
    command: str,
    command_name: str,
) -> None:
    protocol = _ProtocolStub()
    module = _build_module(protocol)
    context = RequestContext(request_id=f"run-hidden-{command_name}", user_id="1", client_id="unit")

    rendered = await module.execute_tool("run", {"command": command}, context=context)

    assert f"Unknown command: {command_name}" in rendered
    assert "unavailable in this context" not in rendered
    assert "[exit:127 |" in rendered
    assert protocol.prepare_calls == []
    assert protocol.execute_calls == []


@pytest.mark.asyncio
async def test_run_mcp_catalogs_does_not_prepare_hidden_catalogs_subtool() -> None:
    class _McpToolsOnlyProtocolStub(_ProtocolStub):
        async def _handle_tools_list(self, params: dict[str, Any], context: RequestContext) -> dict[str, Any]:
            del params, context
            return {
                "tools": [
                    {"name": "mcp.tools.list", "module": "mcp", "canExecute": True},
                ]
            }

    protocol = _McpToolsOnlyProtocolStub()
    module = _build_module(protocol)
    context = RequestContext(request_id="run-mcp-hidden-catalogs", user_id="1", client_id="unit")

    rendered = await module.execute_tool("run", {"command": "mcp catalogs"}, context=context)

    assert "unavailable in this context" in rendered
    assert "mcp.catalogs.list" in rendered
    assert "[exit:2 |" in rendered
    assert protocol.prepare_calls == []
    assert protocol.execute_calls == []


@pytest.mark.asyncio
async def test_bash_alias_returns_governed_alias_usage_errors() -> None:
    protocol = _ProtocolStub()
    module = _build_module(protocol)
    context = RequestContext(request_id="run-bash-alias-usage", user_id="1", client_id="unit")

    rendered = await module.execute_tool("bash", {"command": "rg"}, context=context)

    assert "usage: rg <pattern> [base_path]" in rendered
    assert "[exit:2 |" in rendered
    assert protocol.prepare_calls == []
    assert protocol.execute_calls == []


@pytest.mark.asyncio
async def test_run_cat_without_path_returns_usage() -> None:
    protocol = _ProtocolStub()
    module = _build_module(protocol)
    context = RequestContext(request_id="run-cat-usage", user_id="1", client_id="unit")

    rendered = await module.execute_tool("run", {"command": "cat"}, context=context)

    assert "usage: cat <path>" in rendered.lower()
    assert "[exit:2 |" in rendered
    assert protocol.prepare_calls == []
    assert protocol.execute_calls == []


@pytest.mark.asyncio
async def test_run_write_create_uses_structured_write_create_mode() -> None:
    protocol = _ProtocolStub()
    module = _build_module(protocol)
    context = RequestContext(request_id="run-write-create", user_id="1", client_id="unit")

    rendered = await module.execute_tool("run", {"command": "write-create notes.txt hello"}, context=context)

    assert "wrote 5 bytes to notes.txt" in rendered
    assert protocol.prepare_calls[0].params["name"] == "fs.write"
    assert protocol.prepare_calls[0].params["arguments"] == {
        "path": "notes.txt",
        "content": "hello",
        "mode": "create",
    }


@pytest.mark.asyncio
async def test_run_preflights_write_chain_before_executing_first_step() -> None:
    protocol = _ProtocolStub()
    protocol.prepare_errors["fs.write_text"] = PermissionError("blocked by policy")
    module = _build_module(protocol)
    context = RequestContext(request_id="run-preflight", user_id="1", client_id="unit")

    rendered = await module.execute_tool("run", {"command": "ls ; write notes.txt hello"}, context=context)

    assert "blocked by policy" in rendered
    assert "[exit:1 |" in rendered
    assert [call.params["name"] for call in protocol.prepare_calls] == ["fs.list", "fs.write_text"]
    assert protocol.execute_calls == []


@pytest.mark.asyncio
async def test_run_derives_step_idempotency_from_parent_key() -> None:
    protocol = _ProtocolStub()
    module = _build_module(protocol)
    context = RequestContext(request_id="run-idempotency", user_id="1", client_id="unit")

    first = await module.execute_tool(
        "run",
        {"command": "write notes.txt hello", "idempotencyKey": "parent-idem-1"},
        context=context,
    )
    second = await module.execute_tool(
        "run",
        {"command": "write notes.txt hello", "idempotencyKey": "parent-idem-1"},
        context=context,
    )

    assert first.split("\n[exit:0 |", 1)[0] == second.split("\n[exit:0 |", 1)[0]
    assert "[exit:0 |" in first
    assert "[exit:0 |" in second
    assert len(protocol.prepare_calls) == 2
    assert protocol.prepare_calls[0].params["name"] == "fs.write_text"
    assert protocol.prepare_calls[0].idempotency_key == derive_step_idempotency_key(
        "parent-idem-1",
        ["write", "notes.txt", "hello"],
        0,
    )
    assert protocol.prepare_calls[0].idempotency_key == protocol.prepare_calls[1].idempotency_key


@pytest.mark.asyncio
async def test_run_uses_lexical_preflighted_step_for_identical_command_after_skipped_branch() -> None:
    protocol = _ProtocolStub()
    protocol.execute_errors["fs.read"] = FileNotFoundError("Path not found: missing.txt")
    module = _build_module(protocol)
    context = RequestContext(request_id="run-skipped-branch", user_id="1", client_id="unit")

    rendered = await module.execute_tool(
        "run",
        {
            "command": "cat missing.txt && write notes.txt hi ; write notes.txt hi",
            "idempotencyKey": "parent-idem-skip-1",
        },
        context=context,
    )

    assert "[exit:0 |" in rendered
    assert [call.params["name"] for call in protocol.prepare_calls] == [
        "fs.read",
        "fs.write_text",
        "fs.write_text",
    ]
    assert [call.params["name"] for call in protocol.execute_calls] == [
        "fs.read",
        "fs.write_text",
    ]
    assert protocol.execute_calls[1].idempotency_key == derive_step_idempotency_key(
        "parent-idem-skip-1",
        ["write", "notes.txt", "hi"],
        2,
    )


@pytest.mark.asyncio
async def test_run_converts_governed_file_errors_into_cli_result() -> None:
    protocol = _ProtocolStub()
    protocol.execute_errors["fs.read"] = FileNotFoundError("Path not found: notes.txt")
    module = _build_module(protocol)
    context = RequestContext(request_id="run-cat-missing", user_id="1", client_id="unit")

    rendered = await module.execute_tool("run", {"command": "cat notes.txt"}, context=context)

    assert "Path not found: notes.txt" in rendered
    assert "[exit:1 |" in rendered


@pytest.mark.asyncio
async def test_run_preserves_approval_required_errors() -> None:
    protocol = _ProtocolStub()
    protocol.prepare_errors["fs.write_text"] = ApprovalRequiredError(
        "approval required",
        approval={"reason": "path_outside_current_folder_scope"},
    )
    module = _build_module(protocol)
    context = RequestContext(request_id="run-approval", user_id="1", client_id="unit")

    with pytest.raises(ApprovalRequiredError, match="approval required"):
        await module.execute_tool("run", {"command": "write notes.txt hello"}, context=context)


@pytest.mark.asyncio
async def test_run_help_keeps_argument_sensitive_allowed_commands_visible() -> None:
    class _PatternProtocolStub:
        async def _handle_tools_list(self, params: dict[str, Any], context: RequestContext) -> dict[str, Any]:
            del params, context
            return {
                "tools": [
                    {"name": "sandbox.run", "module": "sandbox", "canExecute": True},
                ]
            }

        def _extract_allowed_tools(self, context: RequestContext) -> list[str]:
            del context
            return ["sandbox.run(ls *)"]

        async def _resolve_effective_tool_policy(self, context: RequestContext) -> dict[str, Any]:
            del context
            return {
                "enabled": True,
                "allowed_tools": ["sandbox.run(ls *)"],
                "denied_tools": [],
            }

        def _is_tool_allowed_by_context(self, tool_name: str, tool_args: dict[str, Any], context: RequestContext) -> bool:
            del tool_name, tool_args, context
            return False

        def _is_tool_allowed_by_effective_policy(
            self,
            tool_name: str,
            tool_args: dict[str, Any],
            policy: dict[str, Any],
        ) -> bool:
            del tool_name, tool_args, policy
            return False

    module = RunCommandModule(
        ModuleConfig(name="run", settings={"protocol": _PatternProtocolStub()}),
    )
    context = RequestContext(
        request_id="run-help-arg-sensitive",
        user_id="1",
        client_id="unit",
        metadata={"allowed_tools": ["sandbox.run(ls *)"]},
    )

    rendered = await module.execute_tool("run", {"command": "help"}, context=context)

    assert "sandbox" in rendered


@pytest.mark.asyncio
async def test_run_uses_configured_spill_settings_and_workspace_relative_spill_dir(tmp_path: Path) -> None:
    protocol = _ProtocolStub()
    protocol.read_text_content = "line one\nline two\nline three\n"
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir(parents=True, exist_ok=True)
    resolver = _WorkspaceRootResolverStub(workspace_root)
    module = RunCommandModule(
        ModuleConfig(
            name="run",
            settings={
                "protocol": protocol,
                "spill_dir": ".mcp/spills",
                "spill_threshold_bytes": 8,
                "preview_line_limit": 1,
                "preview_byte_limit": 8,
                "workspace_root_resolver": resolver,
            },
        )
    )
    context = RequestContext(
        request_id="run-spill-settings",
        user_id="1",
        client_id="unit",
        metadata={"workspace_id": "workspace-1"},
    )

    rendered = await module.execute_tool("run", {"command": "cat notes.txt"}, context=context)

    spill_root = workspace_root / ".mcp" / "spills"
    assert spill_root.exists()
    assert list(spill_root.iterdir()) == []
    assert "line one" in rendered
    assert "line two" not in rendered
    assert "stored internally" in rendered
    assert str(spill_root) not in rendered
    assert resolver.calls


@pytest.mark.asyncio
async def test_run_cleans_up_internal_spill_files_after_rendering(tmp_path: Path) -> None:
    """Default rendering should delete oversized internal spill files."""

    protocol = _ProtocolStub()
    protocol.read_text_content = "line\n" * 500
    spill_root = tmp_path / "spills"
    module = RunCommandModule(
        ModuleConfig(
            name="run",
            settings={
                "protocol": protocol,
                "spill_dir": spill_root,
                "spill_threshold_bytes": 32,
            },
        )
    )
    context = RequestContext(request_id="run-spill-cleanup", user_id="1", client_id="unit")

    rendered = await module.execute_tool("run", {"command": "cat notes.txt"}, context=context)

    assert "--- stdout truncated" in rendered
    assert "stored internally" in rendered
    assert spill_root.exists()
    assert list(spill_root.iterdir()) == []


@pytest.mark.asyncio
async def test_run_timeout_removes_spill_files_created_before_cancellation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Timeout cleanup should remove spills even when no execution result is returned."""

    async def _timeout_after_spill(self: CommandRuntimeExecutor, _chain: Any) -> Any:
        await self._spill_payload_async("line\n" * 500, kind="stdout")
        raise TimeoutError

    monkeypatch.setattr(CommandRuntimeExecutor, "execute", _timeout_after_spill)
    protocol = _ProtocolStub()
    spill_root = tmp_path / "spills"
    module = RunCommandModule(
        ModuleConfig(
            name="run",
            settings={
                "protocol": protocol,
                "spill_dir": spill_root,
                "spill_threshold_bytes": 32,
            },
        )
    )
    context = RequestContext(request_id="run-spill-timeout-cleanup", user_id="1", client_id="unit")

    rendered = await module.execute_tool("run", {"command": "cat notes.txt", "timeout_seconds": 1}, context=context)

    assert "Command timed out after 1s" in rendered
    assert spill_root.exists()
    assert list(spill_root.iterdir()) == []


@pytest.mark.asyncio
async def test_run_can_retain_spill_files_as_redacted_output_artifacts(tmp_path: Path) -> None:
    """Retained spill artifacts should use redacted handles and keep files private."""

    protocol = _ProtocolStub()
    protocol.read_text_content = "line\n" * 500
    spill_root = tmp_path / "spills"
    module = RunCommandModule(
        ModuleConfig(
            name="run",
            settings={
                "protocol": protocol,
                "spill_dir": spill_root,
                "spill_threshold_bytes": 32,
            },
        )
    )
    context = RequestContext(request_id="run-spill-retain", user_id="1", client_id="unit")

    rendered = await module.execute_tool(
        "run",
        {"command": "cat notes.txt", "retainOutputArtifacts": True},
        context=context,
    )

    retained_dirs = list(spill_root.iterdir())
    assert len(retained_dirs) == 1
    assert retained_dirs[0].is_dir()
    retained = list(retained_dirs[0].iterdir())
    assert len(retained) == 1
    assert retained[0].read_text(encoding="utf-8") == protocol.read_text_content
    assert "--- stdout truncated" in rendered
    assert "artifact: mcp-run-output://stdout/" in rendered
    assert retained[0].name in rendered
    assert str(spill_root) not in rendered


@pytest.mark.parametrize(
    "arguments",
    [
        {"command": "ls", "retainOutputArtifacts": "yes"},
        {"command": "ls", "retain_output_artifacts": 1},
        {"command": "ls", "retainOutputArtifacts": True, "retain_output_artifacts": False},
    ],
)
@pytest.mark.asyncio
async def test_run_rejects_invalid_retain_output_artifact_arguments(arguments: dict[str, Any]) -> None:
    """Artifact retention aliases must be booleans with matching values."""

    protocol = _ProtocolStub()
    module = _build_module(protocol)

    with pytest.raises(ValueError, match="retainOutputArtifacts|retain_output_artifacts"):
        await module.execute_tool("run", arguments, context=RequestContext(request_id="run-retain-invalid"))


@pytest.mark.asyncio
async def test_run_supports_json_paths_with_escaped_dots() -> None:
    class _JsonProtocolStub(_ProtocolStub):
        async def execute_prepared_tool_call(self, prepared: _PreparedCall) -> dict[str, Any]:
            self.execute_calls.append(prepared)
            return {
                "content": [
                    {
                        "type": "json",
                        "json": {
                            "path": "payload.json",
                            "text": '{"a.b": {"nested.value": 7}}',
                        },
                    }
                ],
                "tool": prepared.params.get("name"),
            }

    protocol = _JsonProtocolStub()
    module = _build_module(protocol)
    context = RequestContext(request_id="run-json-dot-path", user_id="1", client_id="unit")

    rendered = await module.execute_tool(
        "run",
        {"command": r"cat payload.json | json a\.b.nested\.value"},
        context=context,
    )

    assert "\n7\n" in f"\n{rendered}\n"
    assert "[exit:0 |" in rendered


@pytest.mark.asyncio
async def test_run_resolve_protocol_uses_standalone_protocol_when_server_protocol_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core.MCP_unified import server as server_module

    monkeypatch.setattr(server_module, "get_mcp_server", lambda: SimpleNamespace(protocol=None))

    module = RunCommandModule(ModuleConfig(name="run"))

    protocol = await module._resolve_protocol()

    assert isinstance(protocol, MCPProtocol)


def test_run_write_classification_detects_structured_write_create_command() -> None:
    module = _build_module(_ProtocolStub())

    assert module.is_write_tool_call("run", {"command": "write-create notes.txt hi"}) is True
