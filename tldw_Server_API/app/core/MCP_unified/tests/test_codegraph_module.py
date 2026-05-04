from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from tldw_Server_API.app.core.MCP_unified.modules.base import ModuleConfig
from tldw_Server_API.app.core.MCP_unified.modules.implementations.codegraph_module import (
    CodeGraphModule,
)
from tldw_Server_API.app.core.MCP_unified.protocol import InvalidParamsException, MCPProtocol, RequestContext


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
        self._tool_names = {
            "codegraph.status",
            "codegraph.index",
            "codegraph.sync",
            "codegraph.files",
            "codegraph.search",
            "codegraph.node",
            "codegraph.callers",
            "codegraph.callees",
        }

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
    return _module_with_settings(tmp_path, workspace_root, {})


def _module_with_settings(
    tmp_path: Path,
    workspace_root: Path,
    settings: dict[str, Any],
) -> CodeGraphModule:
    resolver = _FakeWorkspaceRootResolver(
        {
            "workspace_root": str(workspace_root),
            "workspace_id": "workspace-1",
            "source": "sandbox_workspace_lookup",
            "reason": None,
        }
    )
    module_settings = {"index_base_dir": str(tmp_path / "indexes"), **settings}
    return CodeGraphModule(
        ModuleConfig(name="CodeGraph", settings=module_settings),
        workspace_root_resolver=resolver,
    )


@pytest.mark.asyncio
async def test_codegraph_exposes_stage2_tools(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    module = _module(tmp_path, workspace_root)

    tools = await module.get_tools()
    by_name = {tool["name"]: tool for tool in tools}

    assert set(by_name) == {  # nosec B101
        "codegraph.status",
        "codegraph.index",
        "codegraph.sync",
        "codegraph.files",
        "codegraph.search",
        "codegraph.node",
        "codegraph.callers",
        "codegraph.callees",
    }
    assert by_name["codegraph.status"]["metadata"]["readOnlyHint"] is True  # nosec B101
    assert by_name["codegraph.files"]["metadata"]["readOnlyHint"] is True  # nosec B101
    assert by_name["codegraph.search"]["metadata"]["readOnlyHint"] is True  # nosec B101
    assert by_name["codegraph.node"]["metadata"]["readOnlyHint"] is True  # nosec B101
    assert by_name["codegraph.callers"]["metadata"]["readOnlyHint"] is True  # nosec B101
    assert by_name["codegraph.callees"]["metadata"]["readOnlyHint"] is True  # nosec B101
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
    assert by_name["codegraph.search"]["metadata"]["path_argument_hints"] == []  # nosec B101


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
    (workspace_root / "ui.ts").write_text("export const x = 1;\n", encoding="utf-8")
    module = _module(tmp_path, workspace_root)

    index_result = await module.execute_tool(
        "codegraph.index",
        {"mode": "foreground", "force": True, "max_files": 10},
        context=_context(),
    )
    files_result = await module.execute_tool("codegraph.files", {"limit": 10}, context=_context())
    filtered_result = await module.execute_tool(
        "codegraph.files",
        {"limit": 10, "pattern": "*.py"},
        context=_context(),
    )

    assert index_result["status"] == "complete"  # nosec B101
    assert index_result["counters"]["files_indexed"] == 2  # nosec B101
    assert [item["path"] for item in files_result["files"]] == ["app.py", "ui.ts"]  # nosec B101
    assert [item["path"] for item in filtered_result["files"]] == ["app.py"]  # nosec B101


@pytest.mark.asyncio
async def test_codegraph_files_clamps_large_limits_to_configured_max(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    for index in range(3):
        (workspace_root / f"file_{index}.py").write_text("x = 1\n", encoding="utf-8")
    module = _module_with_settings(tmp_path, workspace_root, {"max_search_results": 2})

    await module.execute_tool(
        "codegraph.index",
        {"mode": "foreground", "force": True, "max_files": 10},
        context=_context(),
    )
    files_result = await module.execute_tool("codegraph.files", {"limit": 999}, context=_context())

    assert len(files_result["files"]) == 2  # nosec B101
    assert files_result["truncated"] is True  # nosec B101


@pytest.mark.asyncio
async def test_codegraph_search_node_callers_and_callees_roundtrip(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    (workspace_root / "app.py").write_text(
        """
class Greeter:
    def greet(self, name):
        return helper(name)


def helper(value):
    return value.upper()
""",
        encoding="utf-8",
    )
    module = _module(tmp_path, workspace_root)

    await module.execute_tool(
        "codegraph.index",
        {"mode": "foreground", "force": True, "max_files": 10},
        context=_context(),
    )
    search = await module.execute_tool(
        "codegraph.search",
        {"query": " helper ", "kind": " function ", "limit": 10},
        context=_context(),
    )
    node = await module.execute_tool("codegraph.node", {"symbol": " helper "}, context=_context())
    callers = await module.execute_tool("codegraph.callers", {"symbol": " helper "}, context=_context())
    callees = await module.execute_tool("codegraph.callees", {"symbol": " Greeter.greet "}, context=_context())

    assert [item["qualified_name"] for item in search["results"]] == ["helper"]  # nosec B101
    assert node["node"]["qualified_name"] == "helper"  # nosec B101
    assert [item["source"]["qualified_name"] for item in callers["relationships"]] == ["Greeter.greet"]  # nosec B101
    assert [item["target"]["qualified_name"] for item in callees["relationships"]] == ["helper"]  # nosec B101


def test_codegraph_rejects_ambiguous_node_selectors(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    module = _module(tmp_path, workspace_root)

    with pytest.raises(ValueError, match="node_id and symbol are mutually exclusive"):
        module.validate_tool_arguments(
            "codegraph.node",
            {"node_id": "node_helper", "symbol": "helper"},
        )


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
    await module.execute_tool("codegraph.search", {"query": "app"}, context=_context())
    await module.execute_tool("codegraph.sync", {"mode": "foreground"}, context=_context())

    assert len(offloaded) >= 4  # nosec B101


@pytest.mark.asyncio
async def test_protocol_rejects_unknown_codegraph_index_arguments(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    module = _module(tmp_path, workspace_root)

    protocol = MCPProtocol()
    protocol.module_registry = _CodeGraphRegistry(module)

    async def _resolve_effective_policy(_context: RequestContext) -> dict[str, Any]:
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


@pytest.mark.asyncio
async def test_protocol_rejects_unknown_codegraph_search_arguments(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    module = _module(tmp_path, workspace_root)

    protocol = MCPProtocol()
    protocol.module_registry = _CodeGraphRegistry(module)

    async def _resolve_effective_policy(_context: RequestContext) -> dict[str, Any]:
        return {"enabled": True, "allowed_tools": ["codegraph.search"], "policy_document": {"path_scope_mode": "none"}}

    async def _allow(*_args, **_kwargs) -> bool:
        return True

    protocol._resolve_effective_tool_policy = _resolve_effective_policy  # type: ignore[method-assign]
    protocol._has_module_permission = _allow  # type: ignore[method-assign]
    protocol._has_tool_permission = _allow  # type: ignore[method-assign]
    protocol._is_tool_allowed_by_context = lambda *_args, **_kwargs: True  # type: ignore[method-assign]

    with pytest.raises(InvalidParamsException, match="Unknown parameters"):
        await protocol._handle_tools_call(
            {"name": "codegraph.search", "arguments": {"query": "helper", "unknown": "boom"}},
            _context(),
        )
