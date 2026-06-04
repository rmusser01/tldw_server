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
async def test_filesystem_tools_include_stat_glob_and_grep_metadata() -> None:
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

    assert {"fs.stat", "fs.glob", "fs.grep"} <= set(by_name)  # nosec B101
    for tool_name in ("fs.stat", "fs.glob", "fs.grep"):
        schema = by_name[tool_name]["inputSchema"]
        metadata = by_name[tool_name]["metadata"]
        assert schema["additionalProperties"] is False  # nosec B101
        assert metadata["uses_filesystem"] is True  # nosec B101
        assert metadata["path_boundable"] is True  # nosec B101
        assert "filesystem.read" in metadata["capabilities"]  # nosec B101
        assert metadata["readOnlyHint"] is True  # nosec B101


def test_filesystem_validates_new_filesystem_helper_arguments() -> None:
    mod = FilesystemModule(
        ModuleConfig(
            name="filesystem",
            settings={
                "grep_max_pattern_length": 4,
            },
        ),
        workspace_root_resolver=_FakeWorkspaceRootResolver({"workspace_root": "/workspace/root"}),
    )

    mod.validate_tool_arguments("fs.stat", {"path": "docs/readme.md"})
    mod.validate_tool_arguments("fs.glob", {"pattern": "**/*.py", "limit": 10})
    mod.validate_tool_arguments(
        "fs.grep",
        {
            "pattern": "TODO",
            "include": ["*.py", "**/*.md"],
            "exclude": ["**/.venv/**"],
            "limit": 10,
            "max_file_bytes": 1024,
        },
    )

    invalid_cases = [
        ("fs.stat", {"path": "docs/readme.md", "extra": True}, "unknown arguments"),
        ("fs.stat", {}, "path is required"),
        ("fs.stat", {"path": ""}, "path is required"),
        ("fs.stat", {"path": "docs/readme.md", "follow_symlinks": "yes"}, "follow_symlinks must be a boolean"),
        ("fs.glob", {"pattern": "**/*.py", "unknown": True}, "unknown arguments"),
        ("fs.glob", {}, "pattern is required"),
        ("fs.glob", {"pattern": ""}, "pattern is required"),
        ("fs.glob", {"pattern": "**/*.py", "base_path": 7}, "base_path must be a string"),
        ("fs.glob", {"pattern": "**/*.py", "include_hidden": "yes"}, "include_hidden must be a boolean"),
        ("fs.glob", {"pattern": "**/*.py", "include_files": "yes"}, "include_files must be a boolean"),
        ("fs.glob", {"pattern": "**/*.py", "include_directories": "yes"}, "include_directories must be a boolean"),
        ("fs.glob", {"pattern": "**/*.py", "follow_symlinks": "yes"}, "follow_symlinks must be a boolean"),
        ("fs.glob", {"pattern": "**/*.py", "case_sensitive": "yes"}, "case_sensitive must be a boolean"),
        ("fs.glob", {"pattern": "**/*.py", "limit": 0}, "limit must be a positive integer"),
        ("fs.grep", {"pattern": "TODO", "unknown": True}, "unknown arguments"),
        ("fs.grep", {}, "pattern is required"),
        ("fs.grep", {"pattern": ""}, "pattern is required"),
        ("fs.grep", {"pattern": "TODO", "base_path": 7}, "base_path must be a string"),
        ("fs.grep", {"pattern": "TODO", "include": "*.py"}, "include must be a list of strings"),
        ("fs.grep", {"pattern": "TODO", "include": [1]}, "include must be a list of strings"),
        ("fs.grep", {"pattern": "TODO", "exclude": "*.py"}, "exclude must be a list of strings"),
        ("fs.grep", {"pattern": "TODO", "regex": "yes"}, "regex must be a boolean"),
        ("fs.grep", {"pattern": "TODO", "case_sensitive": "yes"}, "case_sensitive must be a boolean"),
        ("fs.grep", {"pattern": "TODO", "include_hidden": "yes"}, "include_hidden must be a boolean"),
        ("fs.grep", {"pattern": "TODO", "follow_symlinks": "yes"}, "follow_symlinks must be a boolean"),
        ("fs.grep", {"pattern": "TODO", "limit": 0}, "limit must be a positive integer"),
        ("fs.grep", {"pattern": "TODO", "max_file_bytes": 0}, "max_file_bytes must be a positive integer"),
        (
            "fs.grep",
            {"pattern": "abcde", "regex": True},
            "pattern exceeds grep regex length limit",
        ),
    ]

    for tool_name, arguments, expected_message in invalid_cases:
        with pytest.raises(ValueError, match=expected_message):
            mod.validate_tool_arguments(tool_name, arguments)


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
async def test_filesystem_stat_file_and_directory_metadata(tmp_path: Path) -> None:
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
        request_id="req-filesystem-stat",
        user_id="7",
        session_id="sess-1",
        metadata={"workspace_id": "workspace-1"},
    )

    file_stat = await mod.execute_tool("fs.stat", {"path": "docs/hello.txt"}, context=context)
    dir_stat = await mod.execute_tool("fs.stat", {"path": "docs"}, context=context)

    assert file_stat["path"] == "docs/hello.txt"  # nosec B101
    assert file_stat["name"] == "hello.txt"  # nosec B101
    assert file_stat["type"] == "file"  # nosec B101
    assert file_stat["size"] == len("hello world".encode("utf-8"))  # nosec B101
    assert file_stat["is_symlink"] is False  # nosec B101
    assert isinstance(file_stat["modified_at"], str) and file_stat["modified_at"]  # nosec B101
    assert isinstance(file_stat.get("mode"), int)  # nosec B101
    assert dir_stat["path"] == "docs"  # nosec B101
    assert dir_stat["name"] == "docs"  # nosec B101
    assert dir_stat["type"] == "directory"  # nosec B101
    assert dir_stat["is_symlink"] is False  # nosec B101


@pytest.mark.asyncio
async def test_filesystem_stat_rejects_missing_and_escaped_paths(tmp_path: Path) -> None:
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
        request_id="req-filesystem-stat-missing",
        user_id="7",
        metadata={"workspace_id": "workspace-1"},
    )

    with pytest.raises(FileNotFoundError, match="path not found"):
        await mod.execute_tool("fs.stat", {"path": "missing.txt"}, context=context)
    with pytest.raises(PermissionError, match="outside"):
        await mod.execute_tool("fs.stat", {"path": "../escape.txt"}, context=context)


@pytest.mark.asyncio
async def test_filesystem_stat_symlink_policy_does_not_leak_targets(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    docs_dir = workspace_root / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir(parents=True, exist_ok=True)
    outside_file = outside_dir / "secret.txt"
    outside_file.write_text("outside", encoding="utf-8")
    link_path = docs_dir / "secret-link"
    try:
        link_path.symlink_to(outside_file)
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable on this platform: {exc}")

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
        request_id="req-filesystem-stat-symlink",
        user_id="7",
        metadata={"workspace_id": "workspace-1"},
    )

    link_stat = await mod.execute_tool("fs.stat", {"path": "docs/secret-link"}, context=context)

    assert link_stat["path"] == "docs/secret-link"  # nosec B101
    assert link_stat["name"] == "secret-link"  # nosec B101
    assert link_stat["type"] == "symlink"  # nosec B101
    assert link_stat["is_symlink"] is True  # nosec B101
    assert str(outside_file.resolve()) not in str(link_stat)  # nosec B101
    with pytest.raises(PermissionError, match="outside"):
        await mod.execute_tool("fs.stat", {"path": "docs/secret-link", "follow_symlinks": True}, context=context)


@pytest.mark.asyncio
async def test_filesystem_glob_matches_sorted_paths_and_normalizes_patterns(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    package_dir = workspace_root / "pkg"
    package_dir.mkdir(parents=True, exist_ok=True)
    (workspace_root / "app.py").write_text("print('root')", encoding="utf-8")
    (package_dir / "app.py").write_text("print('pkg')", encoding="utf-8")
    (package_dir / "README.md").write_text("# docs", encoding="utf-8")

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
        request_id="req-filesystem-glob",
        user_id="7",
        metadata={"workspace_id": "workspace-1"},
    )

    result = await mod.execute_tool("fs.glob", {"pattern": "**/*.py"}, context=context)
    backslash_result = await mod.execute_tool("fs.glob", {"pattern": r"**\*.py"}, context=context)

    assert [match["path"] for match in result["matches"]] == ["app.py", "pkg/app.py"]  # nosec B101
    assert [match["path"] for match in backslash_result["matches"]] == ["app.py", "pkg/app.py"]  # nosec B101
    assert result["base_path"] == "."  # nosec B101
    assert result["pattern"] == "**/*.py"  # nosec B101
    assert result["truncated"] is False  # nosec B101


@pytest.mark.asyncio
async def test_filesystem_glob_case_hidden_and_limit_behavior(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    hidden_dir = workspace_root / ".secret"
    hidden_dir.mkdir(parents=True, exist_ok=True)
    (workspace_root / "App.PY").write_text("print('mixed')", encoding="utf-8")
    (workspace_root / "visible.py").write_text("print('visible')", encoding="utf-8")
    (workspace_root / ".hidden.py").write_text("print('hidden')", encoding="utf-8")
    (hidden_dir / "nested.py").write_text("print('nested')", encoding="utf-8")

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
        request_id="req-filesystem-glob-hidden",
        user_id="7",
        metadata={"workspace_id": "workspace-1"},
    )

    sensitive = await mod.execute_tool("fs.glob", {"pattern": "**/*.py"}, context=context)
    insensitive = await mod.execute_tool(
        "fs.glob",
        {"pattern": "**/*.py", "case_sensitive": False},
        context=context,
    )
    hidden = await mod.execute_tool(
        "fs.glob",
        {"pattern": "**/*.py", "case_sensitive": False, "include_hidden": True},
        context=context,
    )
    limited = await mod.execute_tool(
        "fs.glob",
        {"pattern": "**/*.py", "case_sensitive": False, "include_hidden": True, "limit": 2},
        context=context,
    )

    assert [match["path"] for match in sensitive["matches"]] == ["visible.py"]  # nosec B101
    assert [match["path"] for match in insensitive["matches"]] == ["App.PY", "visible.py"]  # nosec B101
    assert [match["path"] for match in hidden["matches"]] == [  # nosec B101
        ".hidden.py",
        ".secret/nested.py",
        "App.PY",
        "visible.py",
    ]
    assert limited["truncated"] is True  # nosec B101
    assert limited["remaining_count"] == 2  # nosec B101
    assert len(limited["matches"]) == 2  # nosec B101


@pytest.mark.asyncio
async def test_filesystem_glob_rejects_unsafe_patterns(tmp_path: Path) -> None:
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
        request_id="req-filesystem-glob-unsafe",
        user_id="7",
        metadata={"workspace_id": "workspace-1"},
    )

    for pattern in ("/tmp/*", "C:/tmp/*", "//server/share/*", "../*"):
        with pytest.raises(ValueError, match="unsafe pattern"):
            await mod.execute_tool("fs.glob", {"pattern": pattern}, context=context)


@pytest.mark.asyncio
async def test_filesystem_glob_caps_walk_and_rejects_outside_symlink_dirs(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    docs_dir = workspace_root / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    for index in range(5):
        (docs_dir / f"file-{index}.py").write_text("x", encoding="utf-8")
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir(parents=True, exist_ok=True)
    (outside_dir / "secret.py").write_text("secret", encoding="utf-8")
    link_path = workspace_root / "outside-link"
    try:
        link_path.symlink_to(outside_dir, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable on this platform: {exc}")

    resolver = _FakeWorkspaceRootResolver(
        {
            "workspace_root": str(workspace_root),
            "workspace_id": "workspace-1",
            "source": "sandbox_workspace_lookup",
            "reason": None,
        }
    )
    mod = FilesystemModule(
        ModuleConfig(name="filesystem", settings={"glob_walk_entry_limit": 2}),
        workspace_root_resolver=resolver,
    )
    context = RequestContext(
        request_id="req-filesystem-glob-cap",
        user_id="7",
        metadata={"workspace_id": "workspace-1"},
    )

    capped = await mod.execute_tool("fs.glob", {"pattern": "**/*.py", "base_path": "docs"}, context=context)
    no_follow = await mod.execute_tool("fs.glob", {"pattern": "**/*.py"}, context=context)

    assert capped["truncated"] is True  # nosec B101
    assert len(capped["matches"]) <= 2  # nosec B101
    assert "outside-link/secret.py" not in [match["path"] for match in no_follow["matches"]]  # nosec B101
    with pytest.raises(PermissionError, match="outside"):
        await mod.execute_tool("fs.glob", {"pattern": "**/*.py", "follow_symlinks": True}, context=context)


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
