from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tldw_Server_API.app.core.MCP_unified.modules.base import ModuleConfig
from tldw_Server_API.app.core.MCP_unified.modules.implementations.filesystem_module import (
    FilesystemModule,
)
from tldw_Server_API.app.core.MCP_unified.protocol import InvalidParamsException
from tldw_Server_API.app.core.MCP_unified.protocol import MCPProtocol
from tldw_Server_API.app.core.MCP_unified.protocol import RequestContext
from tldw_Server_API.app.services.mcp_hub_workspace_root_resolver import (
    McpHubWorkspaceRootResolver,
)


class _FakeWorkspaceRootResolver:
    def __init__(self, result: dict[str, Any]) -> None:
        self.result = dict(result)
        self.calls: list[dict[str, Any]] = []

    async def resolve_for_context(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(dict(kwargs))
        return dict(self.result)


class _FilesystemRegistry:
    def __init__(self, module: FilesystemModule) -> None:
        self.module = module
        self._tool_names = {"fs.list", "fs.read_text", "fs.write_text"}

    async def find_module_for_tool(self, tool_name: str):  # noqa: ANN001
        if tool_name in self._tool_names:
            return self.module
        return None

    def get_module_id_for_tool(self, tool_name: str) -> str | None:
        if tool_name in self._tool_names:
            return self.module.name
        return None


class _FakeSandboxService:
    def get_session_workspace_path_for_user(self, session_id: str, user_id: str) -> str | None:
        return None

    def list_workspace_paths_for_user_workspace(self, user_id: str, workspace_id: str) -> list[str]:
        return []


class _FakeSharedRegistryRepo:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = list(rows)
        self.calls: list[dict[str, Any]] = []

    async def list_shared_workspace_entries(self, **kwargs: Any) -> list[dict[str, Any]]:
        self.calls.append(dict(kwargs))
        scope_type = kwargs.get("owner_scope_type")
        scope_id = kwargs.get("owner_scope_id")
        workspace_id = kwargs.get("workspace_id")
        rows = list(self.rows)
        if scope_type is not None:
            rows = [row for row in rows if row.get("owner_scope_type") == scope_type]
        if scope_id is not None or scope_type == "global":
            rows = [row for row in rows if row.get("owner_scope_id") == scope_id]
        if workspace_id is not None:
            rows = [row for row in rows if row.get("workspace_id") == workspace_id]
        return rows


@pytest.mark.asyncio
async def test_filesystem_rejects_session_only_context_without_user_binding() -> None:
    class _Resolver:
        def __init__(self) -> None:
            self.calls = 0

        async def resolve_for_context(self, **kwargs):
            self.calls += 1
            raise AssertionError("resolver should not be called for session-only non-shared contexts")

    resolver = _Resolver()
    mod = FilesystemModule(ModuleConfig(name="filesystem"), workspace_root_resolver=resolver)
    ctx = RequestContext(
        request_id="req-fs-session-only",
        session_id="sess-1",
        user_id=None,
        metadata={"session_id": "sess-1", "workspace_id": "ws-1"},
    )

    with pytest.raises(PermissionError, match="workspace_root_unavailable"):
        await mod.execute_tool("fs.list", {"path": "."}, context=ctx)
    assert resolver.calls == 0  # nosec B101


@pytest.mark.asyncio
async def test_filesystem_allows_session_only_context_with_shared_registry_trust_source(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    docs_dir = workspace_root / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / "hello.txt").write_text("hello world", encoding="utf-8")

    repo = _FakeSharedRegistryRepo(
        [
            {
                "workspace_id": "ws-1",
                "absolute_root": str(workspace_root),
                "owner_scope_type": "team",
                "owner_scope_id": 21,
                "is_active": True,
            }
        ]
    )
    resolver = McpHubWorkspaceRootResolver(sandbox_service=_FakeSandboxService(), repo=repo)
    mod = FilesystemModule(ModuleConfig(name="filesystem"), workspace_root_resolver=resolver)
    ctx = RequestContext(
        request_id="req-fs-shared-registry-session-only",
        session_id="sess-1",
        user_id=None,
        metadata={
            "session_id": "sess-1",
            "workspace_id": "ws-1",
            "selected_workspace_trust_source": "shared_registry",
            "selected_workspace_scope_type": "team",
            "selected_workspace_scope_id": 21,
        },
    )

    listed = await mod.execute_tool("fs.list", {"path": "docs"}, context=ctx)

    assert listed["path"] == "docs"  # nosec B101
    assert any(entry["name"] == "hello.txt" for entry in listed["entries"])  # nosec B101
    assert repo.calls[0]["owner_scope_type"] == "team"  # nosec B101
    assert repo.calls[0]["owner_scope_id"] == 21  # nosec B101
    assert repo.calls[0]["workspace_id"] == "ws-1"  # nosec B101


@pytest.mark.asyncio
async def test_server_registers_filesystem_module_by_default(monkeypatch, tmp_path: Path) -> None:
    from tldw_Server_API.app.core.MCP_unified.server import MCPServer

    server = MCPServer()
    registered_module_ids: list[str] = []

    async def _capture_registration(module_id, module_type, config):  # noqa: ANN001, ARG001
        registered_module_ids.append(str(module_id))

    monkeypatch.setattr(server.module_registry, "register_module", _capture_registration)
    monkeypatch.setenv("MCP_MODULES_CONFIG", str(tmp_path / "missing-modules.yaml"))
    monkeypatch.setenv("MCP_MODULES", "")
    monkeypatch.setenv("MCP_ENABLE_MEDIA_MODULE", "0")
    monkeypatch.setenv("MCP_ENABLE_FILESYSTEM_MODULE", "1")

    await server._register_default_modules()

    assert "filesystem" in registered_module_ids  # nosec B101


@pytest.mark.asyncio
async def test_filesystem_tools_include_path_scope_metadata() -> None:
    resolver = _FakeWorkspaceRootResolver(
        {
            "workspace_root": "/workspace/mcp-filesystem-workspace",
            "workspace_id": "workspace-1",
            "source": "sandbox_workspace_lookup",
            "reason": None,
        }
    )
    mod = FilesystemModule(ModuleConfig(name="filesystem"), workspace_root_resolver=resolver)

    tools = await mod.get_tools()
    by_name = {tool["name"]: tool for tool in tools}

    assert {"fs.list", "fs.read_text", "fs.write_text"} <= set(by_name)  # nosec B101

    for tool_name in ("fs.list", "fs.read_text", "fs.write_text"):
        metadata = by_name[tool_name]["metadata"]
        assert metadata["uses_filesystem"] is True  # nosec B101
        assert metadata["path_boundable"] is True  # nosec B101
        assert metadata["path_argument_hints"] == ["path"]  # nosec B101

    assert by_name["fs.write_text"]["metadata"]["category"] == "management"  # nosec B101


@pytest.mark.asyncio
async def test_filesystem_list_read_and_write_text_within_workspace(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    docs_dir = workspace_root / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    source_file = docs_dir / "hello.txt"
    source_file.write_text("hello world", encoding="utf-8")

    resolver = _FakeWorkspaceRootResolver(
        {
            "workspace_root": str(workspace_root),
            "workspace_id": "workspace-1",
            "source": "sandbox_workspace_lookup",
            "reason": None,
        }
    )
    mod = FilesystemModule(ModuleConfig(name="filesystem"), workspace_root_resolver=resolver)
    context = RequestContext(
        request_id="req-filesystem-roundtrip",
        user_id="7",
        session_id="sess-1",
        metadata={"workspace_id": "workspace-1"},
    )

    listed = await mod.execute_tool("fs.list", {"path": "docs"}, context=context)
    assert listed["path"] == "docs"  # nosec B101
    assert any(entry["name"] == "hello.txt" and entry["type"] == "file" for entry in listed["entries"])  # nosec B101

    read_result = await mod.execute_tool("fs.read_text", {"path": "docs/hello.txt"}, context=context)
    assert read_result["path"] == "docs/hello.txt"  # nosec B101
    assert read_result["text"] == "hello world"  # nosec B101

    write_result = await mod.execute_tool(
        "fs.write_text",
        {"path": "docs/new.txt", "content": "created by fs.write_text"},
        context=context,
    )
    assert write_result["path"] == "docs/new.txt"  # nosec B101
    assert write_result["bytes_written"] == len("created by fs.write_text".encode("utf-8"))  # nosec B101
    assert (docs_dir / "new.txt").read_text(encoding="utf-8") == "created by fs.write_text"  # nosec B101
    assert len(resolver.calls) == 3  # nosec B101


@pytest.mark.asyncio
async def test_protocol_rejects_unknown_fs_read_text_arguments(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    docs_dir = workspace_root / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / "hello.txt").write_text("hello world", encoding="utf-8")

    resolver = _FakeWorkspaceRootResolver(
        {
            "workspace_root": str(workspace_root),
            "workspace_id": "workspace-1",
            "source": "sandbox_workspace_lookup",
            "reason": None,
        }
    )
    mod = FilesystemModule(ModuleConfig(name="filesystem"), workspace_root_resolver=resolver)

    protocol = MCPProtocol()
    protocol.module_registry = _FilesystemRegistry(mod)

    async def _resolve_effective_policy(_context):
        return {"enabled": True, "allowed_tools": ["fs.read_text"], "policy_document": {"path_scope_mode": "none"}}

    async def _allow(*_args, **_kwargs) -> bool:
        return True

    protocol._resolve_effective_tool_policy = _resolve_effective_policy  # type: ignore[method-assign]
    protocol._has_module_permission = _allow  # type: ignore[method-assign]
    protocol._has_tool_permission = _allow  # type: ignore[method-assign]
    protocol._is_tool_allowed_by_context = lambda *_args, **_kwargs: True  # type: ignore[method-assign]

    context = RequestContext(
        request_id="req-fs-read-unknown-arg",
        user_id="7",
        session_id="sess-1",
        metadata={"workspace_id": "workspace-1"},
    )

    with pytest.raises(InvalidParamsException, match="Unknown parameters"):
        await protocol._handle_tools_call(
            {"name": "fs.read_text", "arguments": {"path": "docs/hello.txt", "unknown": "boom"}},
            context,
        )


@pytest.mark.asyncio
async def test_filesystem_list_does_not_leak_symlink_targets_outside_workspace(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    docs_dir = workspace_root / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir(parents=True, exist_ok=True)
    outside_file = outside_dir / "secret.txt"
    outside_file.write_text("outside", encoding="utf-8")
    (docs_dir / "secret-link").symlink_to(outside_file)

    resolver = _FakeWorkspaceRootResolver(
        {
            "workspace_root": str(workspace_root),
            "workspace_id": "workspace-1",
            "source": "sandbox_workspace_lookup",
            "reason": None,
        }
    )
    mod = FilesystemModule(ModuleConfig(name="filesystem"), workspace_root_resolver=resolver)
    context = RequestContext(
        request_id="req-filesystem-symlink-leak",
        user_id="7",
        session_id="sess-1",
        metadata={"workspace_id": "workspace-1"},
    )

    listed = await mod.execute_tool("fs.list", {"path": "docs"}, context=context)
    symlink_entry = next(entry for entry in listed["entries"] if entry["name"] == "secret-link")

    assert listed["path"] == "docs"  # nosec B101
    assert symlink_entry["path"] == "docs/secret-link"  # nosec B101
    assert symlink_entry["type"] == "symlink"  # nosec B101
    assert str(outside_file.resolve()) not in str(listed)  # nosec B101


@pytest.mark.asyncio
async def test_filesystem_read_text_rejects_binary_payload(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir(parents=True, exist_ok=True)
    binary_path = workspace_root / "blob.bin"
    binary_path.write_bytes(b"\x00\x01\x02\x03")

    resolver = _FakeWorkspaceRootResolver(
        {
            "workspace_root": str(workspace_root),
            "workspace_id": "workspace-1",
            "source": "sandbox_workspace_lookup",
            "reason": None,
        }
    )
    mod = FilesystemModule(ModuleConfig(name="filesystem"), workspace_root_resolver=resolver)
    context = RequestContext(
        request_id="req-filesystem-binary",
        user_id="7",
        metadata={"workspace_id": "workspace-1"},
    )

    with pytest.raises(ValueError, match="binary"):
        await mod.execute_tool("fs.read_text", {"path": "blob.bin"}, context=context)


@pytest.mark.asyncio
async def test_filesystem_read_text_rejects_files_over_size_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir(parents=True, exist_ok=True)
    large_path = workspace_root / "large.txt"
    large_path.write_text("x" * 32, encoding="utf-8")

    resolver = _FakeWorkspaceRootResolver(
        {
            "workspace_root": str(workspace_root),
            "workspace_id": "workspace-1",
            "source": "sandbox_workspace_lookup",
            "reason": None,
        }
    )
    mod = FilesystemModule(
        ModuleConfig(name="filesystem", settings={"max_read_bytes": 8}),
        workspace_root_resolver=resolver,
    )
    context = RequestContext(
        request_id="req-filesystem-large-read",
        user_id="7",
        metadata={"workspace_id": "workspace-1"},
    )

    def _fail_read_bytes(self):  # noqa: ANN001
        raise AssertionError("unexpected full file read")

    monkeypatch.setattr(Path, "read_bytes", _fail_read_bytes)

    with pytest.raises(ValueError, match="exceeds fs.read_text limit"):
        await mod.execute_tool("fs.read_text", {"path": "large.txt"}, context=context)


@pytest.mark.asyncio
async def test_filesystem_write_text_rejects_path_escape(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir(parents=True, exist_ok=True)

    resolver = _FakeWorkspaceRootResolver(
        {
            "workspace_root": str(workspace_root),
            "workspace_id": "workspace-1",
            "source": "sandbox_workspace_lookup",
            "reason": None,
        }
    )
    mod = FilesystemModule(ModuleConfig(name="filesystem"), workspace_root_resolver=resolver)
    context = RequestContext(
        request_id="req-filesystem-escape",
        user_id="7",
        metadata={"workspace_id": "workspace-1"},
    )

    with pytest.raises(PermissionError, match="outside"):
        await mod.execute_tool(
            "fs.write_text",
            {"path": "../escape.txt", "content": "forbidden"},
            context=context,
        )


@pytest.mark.asyncio
async def test_filesystem_list_caps_large_directories(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    docs_dir = workspace_root / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    for index in range(12):
        (docs_dir / f"file-{index:02d}.txt").write_text("x", encoding="utf-8")

    resolver = _FakeWorkspaceRootResolver(
        {
            "workspace_root": str(workspace_root),
            "workspace_id": "workspace-1",
            "source": "sandbox_workspace_lookup",
            "reason": None,
        }
    )
    mod = FilesystemModule(
        ModuleConfig(name="filesystem", settings={"list_entry_limit": 5}),
        workspace_root_resolver=resolver,
    )
    context = RequestContext(
        request_id="req-filesystem-list-cap",
        user_id="7",
        session_id="sess-1",
        metadata={"workspace_id": "workspace-1"},
    )

    listed = await mod.execute_tool("fs.list", {"path": "docs"}, context=context)

    assert listed["truncated"] is True  # nosec B101
    assert listed["remaining_count"] == 7  # nosec B101
    assert len(listed["entries"]) == 5  # nosec B101


@pytest.mark.asyncio
async def test_server_resolves_env_placeholders_in_module_settings(monkeypatch, tmp_path: Path) -> None:
    from tldw_Server_API.app.core.MCP_unified.server import MCPServer

    config_path = tmp_path / "mcp_modules.yaml"
    config_path.write_text(
        """
modules:
  - id: run_command
    class: tldw_Server_API.app.core.MCP_unified.modules.implementations.run_command_module:RunCommandModule
    enabled: true
    name: Run Command
    version: "0.1.0"
    department: system
    settings:
      spill_dir: ${MCP_RUN_COMMAND_SPILL_DIR:-.mcp/spills}
""".strip(),
        encoding="utf-8",
    )

    server = MCPServer()
    captured_settings: dict[str, Any] = {}

    async def _capture_registration(module_id, module_type, config):  # noqa: ANN001, ARG001
        if str(module_id) == "run_command":
            captured_settings.update(dict(config.settings or {}))

    monkeypatch.setattr(server.module_registry, "register_module", _capture_registration)
    monkeypatch.setenv("MCP_MODULES_CONFIG", str(config_path))
    monkeypatch.setenv("MCP_MODULES", "")
    monkeypatch.setenv("MCP_ENABLE_MEDIA_MODULE", "0")
    monkeypatch.setenv("MCP_ENABLE_FILESYSTEM_MODULE", "0")
    monkeypatch.setenv("MCP_RUN_COMMAND_SPILL_DIR", ".workspace-spills")

    await server._register_default_modules()

    assert captured_settings["spill_dir"] == ".workspace-spills"  # nosec B101
