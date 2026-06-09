from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tldw_Server_API.app.core.MCP_unified.command_runtime.adapters import (
    derive_step_idempotency_key,
)
from tldw_Server_API.app.core.MCP_unified.modules.base import ModuleConfig
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


@pytest.mark.asyncio
async def test_run_module_exposes_governed_bash_and_shell_alias_tools() -> None:
    protocol = _ProtocolStub()
    module = _build_module(protocol)

    tools = await module.get_tools()

    by_name = {tool["name"]: tool for tool in tools}
    assert list(by_name) == ["run", "bash", "shell"]
    assert by_name["run"]["metadata"].get("canonical_tool") is None
    for alias_name in ("bash", "shell"):
        alias = by_name[alias_name]
        assert alias["metadata"]["canonical_tool"] == "run"
        assert "not a raw host shell" in alias["description"]
        assert alias["inputSchema"]["properties"] == by_name["run"]["inputSchema"]["properties"]


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
