from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from tldw_Server_API.app.core.MCP_unified.modules.base import ModuleConfig
from tldw_Server_API.app.core.MCP_unified.modules.implementations.codegraph_module import (
    CodeGraphModule,
)
from tldw_Server_API.app.core.MCP_unified.protocol import InvalidParamsException
from tldw_Server_API.app.core.MCP_unified.protocol import MCPProtocol
from tldw_Server_API.app.core.MCP_unified.protocol import RequestContext


class _FakeWorkspaceRootResolver:
    def __init__(self, result: dict[str, Any]) -> None:
        self.result = dict(result)
        self.calls: list[dict[str, Any]] = []

    async def resolve_for_context(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(dict(kwargs))
        return dict(self.result)


class _CodeGraphRegistry:
    def __init__(self, module: CodeGraphModule) -> None:
        self.module = module
        self._tool_names = {"codegraph.status", "codegraph.index", "codegraph.sync", "codegraph.files"}

    async def find_module_for_tool(self, tool_name: str):  # noqa: ANN001
        if tool_name in self._tool_names:
            return self.module
        return None

    def get_module_id_for_tool(self, tool_name: str) -> str | None:
        if tool_name in self._tool_names:
            return self.module.name
        return None


def _context() -> RequestContext:
    return RequestContext(
        request_id="req-codegraph",
        user_id="7",
        session_id="sess-1",
        metadata={"workspace_id": "workspace-1"},
    )


def _module(tmp_path: Path, workspace_root: Path) -> CodeGraphModule:
    resolver = _FakeWorkspaceRootResolver(
        {
            "workspace_root": str(workspace_root),
            "workspace_id": "workspace-1",
            "source": "sandbox_workspace_lookup",
            "reason": None,
        }
    )
    return CodeGraphModule(
        ModuleConfig(name="CodeGraph", settings={"index_base_dir": str(tmp_path / "indexes")}),
        workspace_root_resolver=resolver,
    )


@pytest.mark.asyncio
async def test_codegraph_exposes_stage1_tools_only(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    module = _module(tmp_path, workspace_root)

    tools = await module.get_tools()
    by_name = {tool["name"]: tool for tool in tools}

    assert set(by_name) == {"codegraph.status", "codegraph.index", "codegraph.sync", "codegraph.files"}  # nosec B101
    assert by_name["codegraph.status"]["metadata"]["readOnlyHint"] is True  # nosec B101
    assert by_name["codegraph.files"]["metadata"]["readOnlyHint"] is True  # nosec B101
    assert by_name["codegraph.index"]["metadata"]["category"] == "management"  # nosec B101
    assert by_name["codegraph.sync"]["metadata"]["category"] == "management"  # nosec B101

    for tool in by_name.values():
        assert tool["metadata"]["uses_filesystem"] is True  # nosec B101
        assert tool["metadata"]["path_boundable"] is True  # nosec B101
        assert tool["inputSchema"]["additionalProperties"] is False  # nosec B101

    assert by_name["codegraph.status"]["metadata"]["path_argument_hints"] == []  # nosec B101
    assert by_name["codegraph.index"]["metadata"]["path_argument_hints"] == []  # nosec B101
    assert by_name["codegraph.sync"]["metadata"]["path_argument_hints"] == []  # nosec B101
    assert by_name["codegraph.files"]["metadata"]["path_argument_hints"] == ["path"]  # nosec B101


@pytest.mark.asyncio
async def test_codegraph_status_is_read_only_when_index_is_absent(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    index_base = tmp_path / "indexes"
    module = _module(tmp_path, workspace_root)

    status = await module.execute_tool("codegraph.status", {}, context=_context())

    assert status["workspace_key"].startswith("ws_")  # nosec B101
    assert status["index_present"] is False  # nosec B101
    assert status["counts"] == {"files": 0, "nodes": 0, "edges": 0, "unresolved_refs": 0}  # nosec B101
    assert status["last_index_run"] is None  # nosec B101
    assert not index_base.exists()  # nosec B101


@pytest.mark.asyncio
async def test_codegraph_index_and_files_roundtrip(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    (workspace_root / "app.py").write_text("x = 1\n", encoding="utf-8")
    module = _module(tmp_path, workspace_root)

    index_result = await module.execute_tool(
        "codegraph.index",
        {"mode": "foreground", "force": True, "max_files": 10},
        context=_context(),
    )
    files_result = await module.execute_tool("codegraph.files", {"limit": 10}, context=_context())

    assert index_result["status"] == "complete"  # nosec B101
    assert index_result["counters"]["files_indexed"] == 1  # nosec B101
    assert [item["path"] for item in files_result["files"]] == ["app.py"]  # nosec B101


@pytest.mark.asyncio
async def test_codegraph_offloads_blocking_repository_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    (workspace_root / "app.py").write_text("x = 1\n", encoding="utf-8")
    module = _module(tmp_path, workspace_root)
    offloaded: list[str] = []

    async def _fake_to_thread(func, /, *args, **kwargs):  # noqa: ANN001
        offloaded.append(func.__name__)
        return func(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", _fake_to_thread)

    await module.execute_tool("codegraph.index", {"mode": "foreground"}, context=_context())
    await module.execute_tool("codegraph.files", {}, context=_context())
    await module.execute_tool("codegraph.sync", {"mode": "foreground"}, context=_context())

    assert len(offloaded) >= 3  # nosec B101


@pytest.mark.asyncio
async def test_protocol_rejects_unknown_codegraph_index_arguments(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    module = _module(tmp_path, workspace_root)

    protocol = MCPProtocol()
    protocol.module_registry = _CodeGraphRegistry(module)

    async def _resolve_effective_policy(_context):
        return {"enabled": True, "allowed_tools": ["codegraph.index"], "policy_document": {"path_scope_mode": "none"}}

    async def _allow(*_args, **_kwargs) -> bool:
        return True

    protocol._resolve_effective_tool_policy = _resolve_effective_policy  # type: ignore[method-assign]
    protocol._has_module_permission = _allow  # type: ignore[method-assign]
    protocol._has_tool_permission = _allow  # type: ignore[method-assign]
    protocol._is_tool_allowed_by_context = lambda *_args, **_kwargs: True  # type: ignore[method-assign]

    with pytest.raises(InvalidParamsException, match="Unknown parameters"):
        await protocol._handle_tools_call(
            {"name": "codegraph.index", "arguments": {"mode": "foreground", "unknown": "boom"}},
            _context(),
        )
