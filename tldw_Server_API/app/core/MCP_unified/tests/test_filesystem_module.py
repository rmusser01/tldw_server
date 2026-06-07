from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest

from mcp_unified.interfaces.path_scope import PathScopeCandidate
from tldw_Server_API.app.core.MCP_unified.modules.base import ModuleConfig
from tldw_Server_API.app.core.MCP_unified.modules.implementations.filesystem_module import (
    FilesystemModule,
)
from tldw_Server_API.app.core.MCP_unified.protocol import InvalidParamsException, MCPProtocol, RequestContext
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
        self._tool_names = {
            "fs.list",
            "fs.read",
            "fs.read_text",
            "fs.patch",
            "fs.write",
            "fs.write_text",
            "fs.stat",
            "fs.glob",
            "fs.grep",
        }

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


_PATCH_MODIFY_STORY = """--- a/docs/story.txt
+++ b/docs/story.txt
@@ -1,3 +1,3 @@
 alpha
-beta
+BETTA
 gamma
"""

_PATCH_CREATE_NOTES = """--- /dev/null
+++ b/docs/notes.txt
@@ -0,0 +1,2 @@
+first
+second
"""


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

    assert {"fs.list", "fs.read", "fs.read_text", "fs.patch", "fs.write", "fs.write_text"} <= set(by_name)  # nosec B101

    for tool_name in ("fs.list", "fs.read", "fs.read_text", "fs.patch", "fs.write", "fs.write_text"):
        metadata = by_name[tool_name]["metadata"]
        assert metadata["uses_filesystem"] is True  # nosec B101
        assert metadata["path_boundable"] is True  # nosec B101

    read_metadata = by_name["fs.read"]["metadata"]
    assert read_metadata["path_argument_hints"] == ["path"]  # nosec B101
    assert read_metadata["readOnlyHint"] is True  # nosec B101
    assert read_metadata["path_scope_action"] == "read"  # nosec B101
    assert "filesystem.read" in read_metadata["capabilities"]  # nosec B101
    assert read_metadata["eval"]["task_families"] == ["filesystem_read"]  # nosec B101
    assert read_metadata["eval"]["expected_result_kind"] == "structured_filesystem_read"  # nosec B101
    assert by_name["fs.read"]["inputSchema"]["additionalProperties"] is False  # nosec B101
    patch_metadata = by_name["fs.patch"]["metadata"]
    assert patch_metadata["path_scope_candidate_source"] == "module"  # nosec B101
    assert patch_metadata["write_capable"] is True  # nosec B101
    assert patch_metadata["eval"]["task_families"] == ["filesystem_edit"]  # nosec B101
    assert patch_metadata["eval"]["expected_result_kind"] == "structured_filesystem_edit"  # nosec B101
    assert by_name["fs.patch"]["inputSchema"]["additionalProperties"] is False  # nosec B101
    write_metadata = by_name["fs.write"]["metadata"]
    assert write_metadata["path_argument_hints"] == ["path"]  # nosec B101
    assert write_metadata["path_scope_action"] == "write"  # nosec B101
    assert write_metadata["write_capable"] is True  # nosec B101
    assert write_metadata["eval"]["task_families"] == ["filesystem_write"]  # nosec B101
    assert write_metadata["eval"]["expected_result_kind"] == "structured_filesystem_write"  # nosec B101
    assert by_name["fs.write"]["inputSchema"]["additionalProperties"] is False  # nosec B101
    read_text_metadata = by_name["fs.read_text"]["metadata"]
    assert read_text_metadata["legacy_tool"] is True  # nosec B101
    assert read_text_metadata["replacement_tool"] == "fs.read"  # nosec B101
    assert read_text_metadata["path_scope_action"] == "read"  # nosec B101
    write_text_metadata = by_name["fs.write_text"]["metadata"]
    assert write_text_metadata["legacy_tool"] is True  # nosec B101
    assert write_text_metadata["replacement_tools"] == ["fs.patch", "fs.write"]  # nosec B101
    assert write_text_metadata["path_scope_action"] == "write"  # nosec B101
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
                "grep_allow_regex": True,
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
    assert write_result["bytes_written"] == len(b"created by fs.write_text")  # nosec B101
    assert (docs_dir / "new.txt").read_text(encoding="utf-8") == "created by fs.write_text"  # nosec B101


@pytest.mark.asyncio
async def test_filesystem_read_returns_content_hash_and_receipt(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    docs_dir = workspace_root / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    content = "alpha\nbeta\ngamma\n"
    (docs_dir / "story.txt").write_text(content, encoding="utf-8")
    resolver = _FakeWorkspaceRootResolver(
        {
            "workspace_root": str(workspace_root),
            "workspace_id": "workspace-1",
            "source": "sandbox_workspace_lookup",
            "reason": None,
        }
    )
    mod = FilesystemModule(
        ModuleConfig(name="filesystem", settings={"read_receipt_secret": "unit-test-secret"}),
        workspace_root_resolver=resolver,
    )
    context = RequestContext(request_id="req-fs-read", user_id="1", metadata={})

    result = await mod.execute_tool("fs.read", {"path": "docs/story.txt"}, context=context)

    assert result["path"] == "docs/story.txt"  # nosec B101
    assert result["content"] == content  # nosec B101
    assert result["start_line"] == 1  # nosec B101
    assert result["end_line"] == 3  # nosec B101
    assert result["line_count_total"] == 3  # nosec B101
    assert result["bytes_read"] == len(content.encode("utf-8"))  # nosec B101
    assert result["bytes_total"] == len(content.encode("utf-8"))  # nosec B101
    assert result["sha256"] == hashlib.sha256(content.encode("utf-8")).hexdigest()  # nosec B101
    assert result["read_receipt"]  # nosec B101
    assert result["newline_style"] == "lf"  # nosec B101
    assert result["truncated"] is False  # nosec B101
    assert result["eval"] == {  # nosec B101
        "tool_name": "fs.read",
        "tool_prompt_id": "mcp.fs.read.v1",
        "tool_prompt_version": "2026.06.04",
        "action_family": "filesystem_read",
        "result_kind": "structured_filesystem_read",
        "path_filter_used": True,
        "truncated": False,
    }


@pytest.mark.asyncio
async def test_filesystem_read_truncates_and_omits_receipt(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir(parents=True, exist_ok=True)
    (workspace_root / "long.txt").write_text("abcdef\n", encoding="utf-8")
    resolver = _FakeWorkspaceRootResolver(
        {
            "workspace_root": str(workspace_root),
            "workspace_id": "workspace-1",
            "source": "sandbox_workspace_lookup",
            "reason": None,
        }
    )
    mod = FilesystemModule(
        ModuleConfig(name="filesystem", settings={"read_receipt_secret": "unit-test-secret"}),
        workspace_root_resolver=resolver,
    )
    context = RequestContext(request_id="req-fs-read-truncated", user_id="1", metadata={})

    result = await mod.execute_tool("fs.read", {"path": "long.txt", "max_bytes": 3}, context=context)

    assert result["content"] == "abc"  # nosec B101
    assert result["bytes_read"] == 3  # nosec B101
    assert result["truncated"] is True  # nosec B101
    assert result["truncation_reason"] == "max_bytes"  # nosec B101
    assert "read_receipt" not in result  # nosec B101
    assert result["eval"]["truncated"] is True  # nosec B101
    assert result["eval"]["path_filter_used"] is True  # nosec B101


@pytest.mark.asyncio
async def test_filesystem_read_can_include_line_numbers(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir(parents=True, exist_ok=True)
    content = "alpha\nbeta\ngamma\n"
    (workspace_root / "story.txt").write_text(content, encoding="utf-8")
    resolver = _FakeWorkspaceRootResolver(
        {
            "workspace_root": str(workspace_root),
            "workspace_id": "workspace-1",
            "source": "sandbox_workspace_lookup",
            "reason": None,
        }
    )
    mod = FilesystemModule(
        ModuleConfig(name="filesystem", settings={"read_receipt_secret": "unit-test-secret"}),
        workspace_root_resolver=resolver,
    )
    context = RequestContext(request_id="req-fs-read-numbered", user_id="1", metadata={})

    result = await mod.execute_tool(
        "fs.read",
        {"path": "story.txt", "start_line": 2, "max_lines": 2, "include_line_numbers": True},
        context=context,
    )

    assert result["content"] == "2\tbeta\n3\tgamma\n"  # nosec B101
    assert result["start_line"] == 2  # nosec B101
    assert result["end_line"] == 3  # nosec B101
    assert result["line_count_total"] == 3  # nosec B101
    assert result["bytes_read"] == len("beta\ngamma\n".encode("utf-8"))  # nosec B101
    assert result["truncated"] is False  # nosec B101


@pytest.mark.asyncio
async def test_filesystem_read_applies_configured_caps(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir(parents=True, exist_ok=True)
    (workspace_root / "story.txt").write_text("one\ntwo\nthree\n", encoding="utf-8")
    resolver = _FakeWorkspaceRootResolver(
        {
            "workspace_root": str(workspace_root),
            "workspace_id": "workspace-1",
            "source": "sandbox_workspace_lookup",
            "reason": None,
        }
    )
    byte_capped_mod = FilesystemModule(
        ModuleConfig(
            name="filesystem",
            settings={
                "read_max_bytes": 4,
                "read_receipt_secret": "unit-test-secret",
            },
        ),
        workspace_root_resolver=resolver,
    )
    line_capped_mod = FilesystemModule(
        ModuleConfig(
            name="filesystem",
            settings={
                "read_max_lines": 2,
                "read_receipt_secret": "unit-test-secret",
            },
        ),
        workspace_root_resolver=resolver,
    )
    context = RequestContext(request_id="req-fs-read-capped", user_id="1", metadata={})

    byte_capped = await byte_capped_mod.execute_tool("fs.read", {"path": "story.txt", "max_bytes": 100}, context=context)
    line_capped = await line_capped_mod.execute_tool("fs.read", {"path": "story.txt", "max_lines": 100}, context=context)

    assert byte_capped["content"] == "one\n"  # nosec B101
    assert byte_capped["bytes_read"] == 4  # nosec B101
    assert byte_capped["truncation_reason"] == "max_bytes"  # nosec B101
    assert "read_receipt" not in byte_capped  # nosec B101
    assert line_capped["content"] == "one\ntwo\n"  # nosec B101
    assert line_capped["truncation_reason"] == "max_lines"  # nosec B101


@pytest.mark.asyncio
async def test_filesystem_read_rejects_binary_payload(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir(parents=True, exist_ok=True)
    (workspace_root / "blob.bin").write_bytes(b"abc\x00def")
    resolver = _FakeWorkspaceRootResolver(
        {
            "workspace_root": str(workspace_root),
            "workspace_id": "workspace-1",
            "source": "sandbox_workspace_lookup",
            "reason": None,
        }
    )
    mod = FilesystemModule(ModuleConfig(name="filesystem"), workspace_root_resolver=resolver)
    context = RequestContext(request_id="req-fs-read-binary", user_id="1", metadata={})

    with pytest.raises(ValueError, match="binary content is not supported"):
        await mod.execute_tool("fs.read", {"path": "blob.bin"}, context=context)
    assert len(resolver.calls) == 1  # nosec B101


@pytest.mark.asyncio
async def test_filesystem_patch_extracts_path_scope_candidates() -> None:
    resolver = _FakeWorkspaceRootResolver({"workspace_root": "/workspace/root"})
    mod = FilesystemModule(ModuleConfig(name="filesystem"), workspace_root_resolver=resolver)

    modify_candidates = await mod.extract_path_scope_candidates(
        "fs.patch",
        {"diff": _PATCH_MODIFY_STORY},
    )
    create_candidates = await mod.extract_path_scope_candidates(
        "fs.patch",
        {"diff": _PATCH_CREATE_NOTES},
    )

    assert modify_candidates == [  # nosec B101
        PathScopeCandidate(
            path="docs/story.txt",
            action="edit",
            source="filesystem_diff",
            requires_existing_file=True,
        )
    ]
    assert create_candidates == [  # nosec B101
        PathScopeCandidate(
            path="docs/notes.txt",
            action="write",
            source="filesystem_diff",
            creates_file=True,
        )
    ]


@pytest.mark.asyncio
async def test_filesystem_patch_requires_preimage_for_existing_file(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    docs_dir = workspace_root / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / "story.txt").write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    resolver = _FakeWorkspaceRootResolver({"workspace_root": str(workspace_root)})
    mod = FilesystemModule(ModuleConfig(name="filesystem"), workspace_root_resolver=resolver)
    context = RequestContext(request_id="req-fs-patch-preimage", user_id="1", metadata={})

    with pytest.raises(ValueError, match="patch_preimage_required"):
        await mod.execute_tool("fs.patch", {"diff": _PATCH_MODIFY_STORY}, context=context)


@pytest.mark.asyncio
async def test_filesystem_patch_applies_existing_file_with_expected_hash(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    docs_dir = workspace_root / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    original = "alpha\nbeta\ngamma\n"
    target = docs_dir / "story.txt"
    target.write_text(original, encoding="utf-8")
    resolver = _FakeWorkspaceRootResolver({"workspace_root": str(workspace_root), "workspace_id": "ws-1"})
    mod = FilesystemModule(ModuleConfig(name="filesystem"), workspace_root_resolver=resolver)
    context = RequestContext(request_id="req-fs-patch-expected", user_id="1", metadata={"workspace_id": "ws-1"})

    result = await mod.execute_tool(
        "fs.patch",
        {
            "diff": _PATCH_MODIFY_STORY,
            "expected_sha256_by_path": {"docs/story.txt": hashlib.sha256(original.encode("utf-8")).hexdigest()},
        },
        context=context,
    )

    assert target.read_text(encoding="utf-8") == "alpha\nBETTA\ngamma\n"  # nosec B101
    assert result["applied"] is True  # nosec B101
    assert result["dry_run"] is False  # nosec B101
    assert result["files"][0]["path"] == "docs/story.txt"  # nosec B101
    assert result["files"][0]["action"] == "edit"  # nosec B101
    assert result["files"][0]["created"] is False  # nosec B101
    assert "content" not in str(result)  # nosec B101
    assert result["eval"]["action_family"] == "filesystem_edit"  # nosec B101
    assert result["eval"]["path_filter_used"] is True  # nosec B101


@pytest.mark.asyncio
async def test_filesystem_patch_dry_run_does_not_write(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    docs_dir = workspace_root / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    original = "alpha\nbeta\ngamma\n"
    target = docs_dir / "story.txt"
    target.write_text(original, encoding="utf-8")
    resolver = _FakeWorkspaceRootResolver({"workspace_root": str(workspace_root)})
    mod = FilesystemModule(ModuleConfig(name="filesystem"), workspace_root_resolver=resolver)
    context = RequestContext(request_id="req-fs-patch-dry-run", user_id="1", metadata={})

    result = await mod.execute_tool(
        "fs.patch",
        {
            "diff": _PATCH_MODIFY_STORY,
            "expected_sha256_by_path": {"docs/story.txt": hashlib.sha256(original.encode("utf-8")).hexdigest()},
            "dry_run": True,
        },
        context=context,
    )

    assert target.read_text(encoding="utf-8") == original  # nosec B101
    assert result["applied"] is False  # nosec B101
    assert result["dry_run"] is True  # nosec B101
    assert result["files"][0]["sha256_after"] == hashlib.sha256("alpha\nBETTA\ngamma\n".encode("utf-8")).hexdigest()  # nosec B101


@pytest.mark.asyncio
async def test_filesystem_patch_applies_existing_file_with_read_receipt(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    docs_dir = workspace_root / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    target = docs_dir / "story.txt"
    target.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    resolver = _FakeWorkspaceRootResolver({"workspace_root": str(workspace_root), "workspace_id": "ws-1"})
    mod = FilesystemModule(
        ModuleConfig(name="filesystem", settings={"read_receipt_secret": "unit-test-secret"}),
        workspace_root_resolver=resolver,
    )
    context = RequestContext(request_id="req-fs-patch-receipt", user_id="1", metadata={"workspace_id": "ws-1"})
    read_result = await mod.execute_tool("fs.read", {"path": "docs/story.txt"}, context=context)

    result = await mod.execute_tool(
        "fs.patch",
        {"diff": _PATCH_MODIFY_STORY, "read_receipt_by_path": {"docs/story.txt": read_result["read_receipt"]}},
        context=context,
    )

    assert target.read_text(encoding="utf-8") == "alpha\nBETTA\ngamma\n"  # nosec B101
    assert result["files"][0]["sha256_before"] == read_result["sha256"]  # nosec B101


@pytest.mark.asyncio
async def test_filesystem_patch_rejects_stale_hash_and_context_mismatch(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    docs_dir = workspace_root / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    target = docs_dir / "story.txt"
    target.write_text("alpha\nchanged\ngamma\n", encoding="utf-8")
    resolver = _FakeWorkspaceRootResolver({"workspace_root": str(workspace_root)})
    mod = FilesystemModule(ModuleConfig(name="filesystem"), workspace_root_resolver=resolver)
    context = RequestContext(request_id="req-fs-patch-stale", user_id="1", metadata={})

    with pytest.raises(ValueError, match="patch_preimage_mismatch"):
        await mod.execute_tool(
            "fs.patch",
            {
                "diff": _PATCH_MODIFY_STORY,
                "expected_sha256_by_path": {"docs/story.txt": "0" * 64},
            },
            context=context,
        )

    current_sha = hashlib.sha256(target.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="patch_context_mismatch"):
        await mod.execute_tool(
            "fs.patch",
            {
                "diff": _PATCH_MODIFY_STORY,
                "expected_sha256_by_path": {"docs/story.txt": current_sha},
            },
            context=context,
        )


@pytest.mark.asyncio
async def test_filesystem_patch_can_create_file_when_allowed(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir(parents=True, exist_ok=True)
    resolver = _FakeWorkspaceRootResolver({"workspace_root": str(workspace_root)})
    mod = FilesystemModule(ModuleConfig(name="filesystem"), workspace_root_resolver=resolver)
    context = RequestContext(request_id="req-fs-patch-create", user_id="1", metadata={})

    result = await mod.execute_tool(
        "fs.patch",
        {"diff": _PATCH_CREATE_NOTES, "allow_create": True, "create_parent_directories": True},
        context=context,
    )

    assert (workspace_root / "docs" / "notes.txt").read_text(encoding="utf-8") == "first\nsecond\n"  # nosec B101
    assert result["files"][0]["path"] == "docs/notes.txt"  # nosec B101
    assert result["files"][0]["action"] == "write"  # nosec B101
    assert result["files"][0]["created"] is True  # nosec B101


@pytest.mark.asyncio
async def test_filesystem_write_create_creates_new_text_file(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir(parents=True, exist_ok=True)
    resolver = _FakeWorkspaceRootResolver({"workspace_root": str(workspace_root)})
    mod = FilesystemModule(ModuleConfig(name="filesystem"), workspace_root_resolver=resolver)
    context = RequestContext(request_id="req-fs-write-create", user_id="1", metadata={})

    result = await mod.execute_tool(
        "fs.write",
        {"path": "docs/new.txt", "content": "created\n", "mode": "create"},
        context=context,
    )

    assert (workspace_root / "docs" / "new.txt").read_text(encoding="utf-8") == "created\n"  # nosec B101
    assert result["path"] == "docs/new.txt"  # nosec B101
    assert result["written"] is True  # nosec B101
    assert result["created"] is True  # nosec B101
    assert result["bytes_written"] == len("created\n".encode("utf-8"))  # nosec B101
    assert result["eval"]["action_family"] == "filesystem_write"  # nosec B101

    with pytest.raises(ValueError, match="write_target_exists"):
        await mod.execute_tool(
            "fs.write",
            {"path": "docs/new.txt", "content": "again\n", "mode": "create"},
            context=context,
        )


@pytest.mark.asyncio
async def test_filesystem_write_replace_requires_preimage(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir(parents=True, exist_ok=True)
    (workspace_root / "story.txt").write_text("old\n", encoding="utf-8")
    resolver = _FakeWorkspaceRootResolver({"workspace_root": str(workspace_root)})
    mod = FilesystemModule(ModuleConfig(name="filesystem"), workspace_root_resolver=resolver)
    context = RequestContext(request_id="req-fs-write-preimage", user_id="1", metadata={})

    with pytest.raises(ValueError, match="write_preimage_required"):
        await mod.execute_tool(
            "fs.write",
            {"path": "story.txt", "content": "new\n", "mode": "replace"},
            context=context,
        )


@pytest.mark.asyncio
async def test_filesystem_write_replace_with_expected_hash(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir(parents=True, exist_ok=True)
    target = workspace_root / "story.txt"
    target.write_text("old\n", encoding="utf-8")
    resolver = _FakeWorkspaceRootResolver({"workspace_root": str(workspace_root)})
    mod = FilesystemModule(ModuleConfig(name="filesystem"), workspace_root_resolver=resolver)
    context = RequestContext(request_id="req-fs-write-expected", user_id="1", metadata={})
    expected_sha = hashlib.sha256("old\n".encode("utf-8")).hexdigest()

    result = await mod.execute_tool(
        "fs.write",
        {"path": "story.txt", "content": "new\n", "mode": "replace", "expected_sha256": expected_sha},
        context=context,
    )

    assert target.read_text(encoding="utf-8") == "new\n"  # nosec B101
    assert result["written"] is True  # nosec B101
    assert result["created"] is False  # nosec B101
    assert result["sha256_before"] == expected_sha  # nosec B101
    assert result["sha256_after"] == hashlib.sha256("new\n".encode("utf-8")).hexdigest()  # nosec B101


@pytest.mark.asyncio
async def test_filesystem_write_replace_with_read_receipt(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir(parents=True, exist_ok=True)
    target = workspace_root / "story.txt"
    target.write_text("old\n", encoding="utf-8")
    resolver = _FakeWorkspaceRootResolver({"workspace_root": str(workspace_root), "workspace_id": "ws-1"})
    mod = FilesystemModule(
        ModuleConfig(name="filesystem", settings={"read_receipt_secret": "unit-test-secret"}),
        workspace_root_resolver=resolver,
    )
    context = RequestContext(request_id="req-fs-write-receipt", user_id="1", metadata={"workspace_id": "ws-1"})
    read_result = await mod.execute_tool("fs.read", {"path": "story.txt"}, context=context)

    result = await mod.execute_tool(
        "fs.write",
        {
            "path": "story.txt",
            "content": "new\n",
            "mode": "replace",
            "read_receipt": read_result["read_receipt"],
        },
        context=context,
    )

    assert target.read_text(encoding="utf-8") == "new\n"  # nosec B101
    assert result["sha256_before"] == read_result["sha256"]  # nosec B101


@pytest.mark.asyncio
async def test_filesystem_write_rejects_stale_hash_and_dry_run_does_not_write(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir(parents=True, exist_ok=True)
    target = workspace_root / "story.txt"
    target.write_text("old\n", encoding="utf-8")
    resolver = _FakeWorkspaceRootResolver({"workspace_root": str(workspace_root)})
    mod = FilesystemModule(ModuleConfig(name="filesystem"), workspace_root_resolver=resolver)
    context = RequestContext(request_id="req-fs-write-stale", user_id="1", metadata={})
    expected_sha = hashlib.sha256("old\n".encode("utf-8")).hexdigest()

    with pytest.raises(ValueError, match="write_preimage_mismatch"):
        await mod.execute_tool(
            "fs.write",
            {"path": "story.txt", "content": "new\n", "mode": "replace", "expected_sha256": "0" * 64},
            context=context,
        )

    result = await mod.execute_tool(
        "fs.write",
        {
            "path": "story.txt",
            "content": "dry-run\n",
            "mode": "replace",
            "expected_sha256": expected_sha,
            "dry_run": True,
        },
        context=context,
    )

    assert target.read_text(encoding="utf-8") == "old\n"  # nosec B101
    assert result["written"] is False  # nosec B101
    assert result["dry_run"] is True  # nosec B101
    assert result["sha256_after"] == hashlib.sha256("dry-run\n".encode("utf-8")).hexdigest()  # nosec B101


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
    assert file_stat["size"] == len(b"hello world")  # nosec B101
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
async def test_filesystem_stat_rejects_parent_symlink_escape_without_following(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir(parents=True, exist_ok=True)
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir(parents=True, exist_ok=True)
    outside_file = outside_dir / "secret.txt"
    outside_file.write_text("outside", encoding="utf-8")
    link_dir = workspace_root / "outside-dir"
    try:
        link_dir.symlink_to(outside_dir, target_is_directory=True)
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
        request_id="req-filesystem-stat-parent-symlink",
        user_id="7",
        metadata={"workspace_id": "workspace-1"},
    )

    with pytest.raises(PermissionError, match="outside"):
        await mod.execute_tool("fs.stat", {"path": "outside-dir/secret.txt"}, context=context)


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
async def test_filesystem_glob_marks_file_size_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir(parents=True, exist_ok=True)
    source_file = workspace_root / "unreadable-size.txt"
    source_file.write_text("content", encoding="utf-8")

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
        request_id="req-filesystem-glob-size-unavailable",
        user_id="7",
        metadata={"workspace_id": "workspace-1"},
    )
    original_stat = Path.stat
    stat_calls_by_path: dict[Path, int] = {}

    def _raise_for_source_file(self: Path, *args: Any, **kwargs: Any):
        stat_calls_by_path[self] = stat_calls_by_path.get(self, 0) + 1
        if self == source_file and stat_calls_by_path[self] > 1:
            raise OSError("metadata unavailable")
        return original_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", _raise_for_source_file)

    result = await mod.execute_tool("fs.glob", {"pattern": "*.txt"}, context=context)

    assert result["matches"] == [  # nosec B101
        {
            "path": "unreadable-size.txt",
            "type": "file",
            "size": None,
            "size_unavailable": True,
        }
    ]


@pytest.mark.asyncio
async def test_filesystem_glob_rejects_excessive_double_star_patterns(tmp_path: Path) -> None:
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
        request_id="req-filesystem-glob-double-star-cap",
        user_id="7",
        metadata={"workspace_id": "workspace-1"},
    )

    with pytest.raises(ValueError, match="too many double-star"):
        await mod.execute_tool("fs.glob", {"pattern": "**/**/**/**/**/**/*.py"}, context=context)


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
async def test_filesystem_glob_returns_symlink_directories_without_traversing(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir(parents=True, exist_ok=True)
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
    mod = FilesystemModule(ModuleConfig(name="filesystem"), workspace_root_resolver=resolver)
    context = RequestContext(
        request_id="req-filesystem-glob-symlink-dir-entry",
        user_id="7",
        metadata={"workspace_id": "workspace-1"},
    )

    result = await mod.execute_tool("fs.glob", {"pattern": "*"}, context=context)

    assert result["matches"] == [  # nosec B101
        {"path": "outside-link", "type": "symlink"}
    ]


@pytest.mark.asyncio
async def test_filesystem_grep_literal_regex_case_and_newlines(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    docs_dir = workspace_root / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / "app.py").write_text("alpha\nTODO: fix root\nError mixed\n", encoding="utf-8")
    (docs_dir / "notes.txt").write_text("first\r\nTODO: docs\rthird\r", encoding="utf-8", newline="")

    resolver = _FakeWorkspaceRootResolver(
        {
            "workspace_root": str(workspace_root),
            "workspace_id": "workspace-1",
            "source": "sandbox_workspace_lookup",
            "reason": None,
        }
    )
    mod = FilesystemModule(
        ModuleConfig(name="filesystem", settings={"grep_allow_regex": True}),
        workspace_root_resolver=resolver,
    )
    context = RequestContext(
        request_id="req-filesystem-grep",
        user_id="7",
        metadata={"workspace_id": "workspace-1"},
    )

    literal = await mod.execute_tool(
        "fs.grep",
        {"pattern": "TODO", "base_path": "docs", "include": ["**/*.py"]},
        context=context,
    )
    regex = await mod.execute_tool(
        "fs.grep",
        {"pattern": r"TODO:\s+\w+", "base_path": "docs", "regex": True},
        context=context,
    )
    insensitive = await mod.execute_tool(
        "fs.grep",
        {"pattern": "error", "base_path": "docs", "case_sensitive": False},
        context=context,
    )
    newline = await mod.execute_tool(
        "fs.grep",
        {"pattern": "third", "base_path": "docs"},
        context=context,
    )

    assert literal["matches"] == [  # nosec B101
        {
            "path": "docs/app.py",
            "line_number": 2,
            "line": "TODO: fix root",
            "match_text": "TODO",
        }
    ]
    assert regex["matches"][0]["match_text"] == "TODO: fix"  # nosec B101
    assert insensitive["matches"][0]["match_text"] == "Error"  # nosec B101
    assert newline["matches"][0]["line_number"] == 3  # nosec B101
    assert newline["matches"][0]["line"] == "third"  # nosec B101


@pytest.mark.asyncio
async def test_filesystem_grep_rejects_regex_when_disabled_by_default(tmp_path: Path) -> None:
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
        request_id="req-filesystem-grep-regex-disabled",
        user_id="7",
        metadata={"workspace_id": "workspace-1"},
    )

    with pytest.raises(ValueError, match="regex grep is disabled"):
        await mod.execute_tool("fs.grep", {"pattern": "TODO", "regex": True}, context=context)


@pytest.mark.asyncio
async def test_filesystem_grep_skips_binary_decode_and_large_files(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    src_dir = workspace_root / "src"
    src_dir.mkdir(parents=True, exist_ok=True)
    (src_dir / "a.py").write_text("TODO small\n", encoding="utf-8")
    (src_dir / "b.md").write_text("TODO excluded\n", encoding="utf-8")
    (src_dir / "large.txt").write_text("TODO " + ("x" * 64), encoding="utf-8")
    (src_dir / "blob.bin").write_bytes(b"\x00TODO")
    (src_dir / "bad.txt").write_bytes(b"\xffTODO")

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
        request_id="req-filesystem-grep-skips",
        user_id="7",
        metadata={"workspace_id": "workspace-1"},
    )

    result = await mod.execute_tool(
        "fs.grep",
        {
            "pattern": "TODO",
            "base_path": "src",
            "include": ["**/*"],
            "exclude": ["**/*.md"],
            "max_file_bytes": 32,
        },
        context=context,
    )

    assert [match["path"] for match in result["matches"]] == ["src/a.py"]  # nosec B101
    assert result["skipped"]["binary"] == 1  # nosec B101
    assert result["skipped"]["decode_error"] == 1  # nosec B101
    assert result["skipped"]["too_large"] == 1  # nosec B101


@pytest.mark.asyncio
async def test_filesystem_grep_limits_and_regex_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workspace_root = tmp_path / "workspace"
    nested_dir = workspace_root / "a"
    workspace_root.mkdir(parents=True, exist_ok=True)
    nested_dir.mkdir(parents=True, exist_ok=True)
    (workspace_root / "z-root.txt").write_text("TODO root\n", encoding="utf-8")
    (nested_dir / "a-nested.txt").write_text("TODO nested\n", encoding="utf-8")
    for index in range(5):
        (workspace_root / f"file-{index}.txt").write_text("TODO\n", encoding="utf-8")

    resolver = _FakeWorkspaceRootResolver(
        {
            "workspace_root": str(workspace_root),
            "workspace_id": "workspace-1",
            "source": "sandbox_workspace_lookup",
            "reason": None,
        }
    )
    limit_mod = FilesystemModule(
        ModuleConfig(name="filesystem", settings={"grep_allow_regex": True}),
        workspace_root_resolver=resolver,
    )
    cap_mod = FilesystemModule(
        ModuleConfig(name="filesystem", settings={"grep_walk_entry_limit": 2}),
        workspace_root_resolver=resolver,
    )
    context = RequestContext(
        request_id="req-filesystem-grep-limits",
        user_id="7",
        metadata={"workspace_id": "workspace-1"},
    )

    limited = await limit_mod.execute_tool("fs.grep", {"pattern": "TODO", "limit": 2}, context=context)
    sorted_limited = await limit_mod.execute_tool("fs.grep", {"pattern": "TODO", "limit": 1}, context=context)
    capped = await cap_mod.execute_tool("fs.grep", {"pattern": "TODO"}, context=context)

    assert limited["truncated"] is True  # nosec B101
    assert len(limited["matches"]) == 2  # nosec B101
    assert sorted_limited["matches"][0]["path"] == "a/a-nested.txt"  # nosec B101
    assert capped["truncated"] is True  # nosec B101
    assert len(capped["matches"]) <= 2  # nosec B101

    def _fail_read_bytes(self):  # noqa: ANN001
        raise AssertionError("invalid regex should be rejected before file reads")

    monkeypatch.setattr(Path, "read_bytes", _fail_read_bytes)
    with pytest.raises(ValueError, match="invalid regex pattern"):
        await limit_mod.execute_tool("fs.grep", {"pattern": "[", "regex": True}, context=context)


@pytest.mark.asyncio
async def test_filesystem_grep_caps_global_io_budgets(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir(parents=True, exist_ok=True)
    for name in ("a.txt", "b.txt", "c.txt"):
        (workspace_root / name).write_text("TODO " + ("x" * 16), encoding="utf-8")

    resolver = _FakeWorkspaceRootResolver(
        {
            "workspace_root": str(workspace_root),
            "workspace_id": "workspace-1",
            "source": "sandbox_workspace_lookup",
            "reason": None,
        }
    )
    byte_budget_mod = FilesystemModule(
        ModuleConfig(name="filesystem", settings={"grep_max_total_bytes": 24}),
        workspace_root_resolver=resolver,
    )
    file_budget_mod = FilesystemModule(
        ModuleConfig(name="filesystem", settings={"grep_max_files": 1}),
        workspace_root_resolver=resolver,
    )
    context = RequestContext(
        request_id="req-filesystem-grep-io-budget",
        user_id="7",
        metadata={"workspace_id": "workspace-1"},
    )

    byte_limited = await byte_budget_mod.execute_tool("fs.grep", {"pattern": "TODO"}, context=context)
    file_limited = await file_budget_mod.execute_tool("fs.grep", {"pattern": "TODO"}, context=context)

    assert [match["path"] for match in byte_limited["matches"]] == ["a.txt"]  # nosec B101
    assert byte_limited["truncated"] is True  # nosec B101
    assert byte_limited["remaining_count_known"] is False  # nosec B101
    assert "io_budget" in byte_limited["truncation_reasons"]  # nosec B101
    assert [match["path"] for match in file_limited["matches"]] == ["a.txt"]  # nosec B101
    assert file_limited["truncated"] is True  # nosec B101
    assert file_limited["remaining_count_known"] is False  # nosec B101
    assert "file_budget" in file_limited["truncation_reasons"]  # nosec B101


@pytest.mark.asyncio
async def test_filesystem_grep_caps_directory_only_walks(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir(parents=True, exist_ok=True)
    for index in range(5):
        (workspace_root / f"empty-{index}").mkdir()

    resolver = _FakeWorkspaceRootResolver(
        {
            "workspace_root": str(workspace_root),
            "workspace_id": "workspace-1",
            "source": "sandbox_workspace_lookup",
            "reason": None,
        }
    )
    mod = FilesystemModule(
        ModuleConfig(name="filesystem", settings={"grep_walk_entry_limit": 2}),
        workspace_root_resolver=resolver,
    )
    context = RequestContext(
        request_id="req-filesystem-grep-dir-cap",
        user_id="7",
        metadata={"workspace_id": "workspace-1"},
    )

    result = await mod.execute_tool("fs.grep", {"pattern": "missing"}, context=context)

    assert result["matches"] == []  # nosec B101
    assert result["truncated"] is True  # nosec B101


@pytest.mark.asyncio
async def test_filesystem_grep_symlink_policy_and_loop_avoidance(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir(parents=True, exist_ok=True)
    (workspace_root / "root.txt").write_text("TODO root\n", encoding="utf-8")
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir(parents=True, exist_ok=True)
    (outside_dir / "secret.txt").write_text("TODO secret\n", encoding="utf-8")
    outside_link = workspace_root / "outside-link"
    loop_link = workspace_root / "loop"
    try:
        outside_link.symlink_to(outside_dir, target_is_directory=True)
        loop_link.symlink_to(workspace_root, target_is_directory=True)
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
        request_id="req-filesystem-grep-symlink",
        user_id="7",
        metadata={"workspace_id": "workspace-1"},
    )

    no_follow = await mod.execute_tool("fs.grep", {"pattern": "TODO"}, context=context)
    outside_link.unlink()
    loop_safe = await mod.execute_tool("fs.grep", {"pattern": "TODO", "follow_symlinks": True}, context=context)

    assert [match["path"] for match in no_follow["matches"]] == ["root.txt"]  # nosec B101
    assert [match["path"] for match in loop_safe["matches"]] == ["root.txt"]  # nosec B101
    outside_link.symlink_to(outside_dir, target_is_directory=True)
    with pytest.raises(PermissionError, match="outside"):
        await mod.execute_tool("fs.grep", {"pattern": "TODO", "follow_symlinks": True}, context=context)


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
async def test_protocol_rejects_unknown_new_filesystem_helper_arguments(tmp_path: Path) -> None:
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
        return {
            "enabled": True,
            "allowed_tools": ["fs.stat", "fs.glob", "fs.grep"],
            "policy_document": {"path_scope_mode": "none"},
        }

    async def _allow(*_args, **_kwargs) -> bool:
        return True

    protocol._resolve_effective_tool_policy = _resolve_effective_policy  # type: ignore[method-assign]
    protocol._has_module_permission = _allow  # type: ignore[method-assign]
    protocol._has_tool_permission = _allow  # type: ignore[method-assign]
    protocol._is_tool_allowed_by_context = lambda *_args, **_kwargs: True  # type: ignore[method-assign]

    context = RequestContext(
        request_id="req-fs-new-helper-unknown-arg",
        user_id="7",
        session_id="sess-1",
        metadata={"workspace_id": "workspace-1"},
    )

    cases = [
        ("fs.stat", {"path": "docs/hello.txt", "unknown": "boom"}),
        ("fs.glob", {"pattern": "**/*.txt", "unknown": "boom"}),
        ("fs.grep", {"pattern": "hello", "unknown": "boom"}),
    ]
    for tool_name, arguments in cases:
        with pytest.raises(InvalidParamsException, match="Unknown parameters"):
            await protocol._handle_tools_call({"name": tool_name, "arguments": arguments}, context)


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
