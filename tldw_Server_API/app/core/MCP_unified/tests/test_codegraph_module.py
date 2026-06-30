"""Tests for the Unified MCP CodeGraph module."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from tldw_Server_API.app.core.CodeGraph.extractors.tree_sitter_loader import load_parser
from tldw_Server_API.app.core.CodeGraph.models import CodeGraphNode
from tldw_Server_API.app.core.MCP_unified.modules.base import ModuleConfig
from tldw_Server_API.app.core.MCP_unified.modules.implementations.codegraph_module import (
    CodeGraphModule,
    _relationship_neighborhood,
)
from tldw_Server_API.app.core.MCP_unified.protocol import InvalidParamsException, MCPProtocol, RequestContext


class _FakeWorkspaceRootResolver:
    """Capture workspace resolution requests and return a fixed workspace."""

    def __init__(self, result: dict[str, Any]) -> None:
        self.result = dict(result)
        self.calls: list[dict[str, Any]] = []

    async def resolve_for_context(self, **kwargs: Any) -> dict[str, Any]:
        """Return the configured fake workspace resolution payload."""
        self.calls.append(dict(kwargs))
        return dict(self.result)


class _CodeGraphRegistry:
    """Minimal registry wrapper exposing one CodeGraph module to protocol tests."""

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
            "codegraph.impact",
            "codegraph.context",
        }

    async def find_module_for_tool(self, tool_name: str):  # noqa: ANN001
        """Return the CodeGraph module when the tool name belongs to it."""
        if tool_name in self._tool_names:
            return self.module
        return None

    def get_module_id_for_tool(self, tool_name: str) -> str | None:
        """Return the CodeGraph module id for known CodeGraph tools."""
        if tool_name in self._tool_names:
            return self.module.name
        return None


class _FakeJobManager:
    """Capture CodeGraph Jobs rows created by MCP job-mode tests."""

    def __init__(self) -> None:
        self.created: list[dict[str, Any]] = []

    def create_job(self, **kwargs: Any) -> dict[str, Any]:
        """Record a fake Jobs create call."""
        self.created.append(dict(kwargs))
        return {"id": len(self.created), "uuid": f"job-{len(self.created)}", "status": "queued", **kwargs}

    def get_job(self, job_id: int) -> dict[str, Any] | None:
        """Return the recorded fake Jobs row by id."""
        if job_id < 1 or job_id > len(self.created):
            return None
        created = self.created[job_id - 1]
        return {"id": job_id, "uuid": f"job-{job_id}", "status": "queued", **created}


def _context() -> RequestContext:
    """Build a request context with a workspace id for CodeGraph tests."""
    return RequestContext(
        request_id="req-codegraph",
        user_id="7",
        session_id="sess-1",
        metadata={"workspace_id": "workspace-1"},
    )


def _require_c_family_parsers() -> None:
    """Skip C/C++ MCP coverage unless both C-family parsers load."""
    if not (load_parser("c").available and load_parser("cpp").available):
        pytest.skip("tree-sitter-c/cpp parsers are not available")


def _require_jvm_parsers() -> None:
    """Skip JVM MCP coverage unless both Java and Kotlin parsers load."""
    if not (load_parser("java").available and load_parser("kotlin").available):
        pytest.skip("tree-sitter-java/kotlin parsers are not available")


def _require_typescript_parsers() -> None:
    """Skip TypeScript MCP coverage unless TS and TSX parsers load."""
    if not (load_parser("typescript").available and load_parser("tsx").available):
        pytest.skip("tree-sitter-typescript parser is not available")


def _require_csharp_parser() -> None:
    """Skip C# MCP coverage unless the C# parser loads."""
    if not load_parser("csharp").available:
        pytest.skip("tree-sitter-c-sharp parser is not available")


def _module(tmp_path: Path, workspace_root: Path) -> CodeGraphModule:
    """Create a CodeGraph module with default test settings."""
    return _module_with_settings(tmp_path, workspace_root, {})


def _module_with_settings(
    tmp_path: Path,
    workspace_root: Path,
    settings: dict[str, Any],
    *,
    job_manager_factory: Callable[[], Any] | None = None,
) -> CodeGraphModule:
    """Create a CodeGraph module with overridden test settings."""
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
        job_manager_factory=job_manager_factory,
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
        "codegraph.impact",
        "codegraph.context",
    }
    assert by_name["codegraph.status"]["metadata"]["readOnlyHint"] is True  # nosec B101
    assert by_name["codegraph.files"]["metadata"]["readOnlyHint"] is True  # nosec B101
    assert by_name["codegraph.search"]["metadata"]["readOnlyHint"] is True  # nosec B101
    assert by_name["codegraph.node"]["metadata"]["readOnlyHint"] is True  # nosec B101
    assert by_name["codegraph.callers"]["metadata"]["readOnlyHint"] is True  # nosec B101
    assert by_name["codegraph.callees"]["metadata"]["readOnlyHint"] is True  # nosec B101
    assert by_name["codegraph.impact"]["metadata"]["readOnlyHint"] is True  # nosec B101
    assert by_name["codegraph.context"]["metadata"]["readOnlyHint"] is True  # nosec B101
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
    assert by_name["codegraph.impact"]["metadata"]["path_argument_hints"] == []  # nosec B101
    assert by_name["codegraph.context"]["metadata"]["path_argument_hints"] == []  # nosec B101


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
async def test_codegraph_index_job_mode_enqueues_without_creating_index(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    (workspace_root / "app.py").write_text("x = 1\n", encoding="utf-8")
    jobs = _FakeJobManager()
    module = _module_with_settings(
        tmp_path,
        workspace_root,
        {},
        job_manager_factory=lambda: jobs,
    )

    result = await module.execute_tool(
        "codegraph.index",
        {"mode": "job", "force": True, "languages": ["python"], "max_files": 10},
        context=_context(),
    )

    assert result["status"] == "queued"  # nosec B101
    assert result["mode"] == "job"  # nosec B101
    assert result["operation"] == "index"  # nosec B101
    assert result["job_id"] == 1  # nosec B101
    assert result["job_uuid"] == "job-1"  # nosec B101
    assert result["job_status"] == "queued"  # nosec B101
    assert result["workspace_id"] == "workspace-1"  # nosec B101
    assert result["workspace_key"].startswith("ws_")  # nosec B101
    assert not result["index_db_path"].endswith("app.py")  # nosec B101
    assert not Path(result["index_db_path"]).exists()  # nosec B101

    created = jobs.created[0]
    assert created["domain"] == "codegraph"  # nosec B101
    assert created["queue"] == "default"  # nosec B101
    assert created["job_type"] == "codegraph_index"  # nosec B101
    assert created["owner_user_id"] == "7"  # nosec B101
    assert created["max_retries"] == 0  # nosec B101
    assert created["payload"]["operation"] == "index"  # nosec B101
    assert created["payload"]["force"] is True  # nosec B101
    assert created["payload"]["languages"] == ["python"]  # nosec B101
    assert created["payload"]["max_files"] == 10  # nosec B101


@pytest.mark.asyncio
async def test_codegraph_sync_background_mode_enqueues_sync(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    jobs = _FakeJobManager()
    module = _module_with_settings(
        tmp_path,
        workspace_root,
        {},
        job_manager_factory=lambda: jobs,
    )

    result = await module.execute_tool(
        "codegraph.sync",
        {"mode": "background", "languages": ["python"], "max_files": 5},
        context=_context(),
    )

    assert result["status"] == "queued"  # nosec B101
    assert result["mode"] == "background"  # nosec B101
    assert result["operation"] == "sync"  # nosec B101
    assert result["job_id"] == 1  # nosec B101
    assert result["queue"] == "default"  # nosec B101
    assert jobs.created[0]["payload"]["operation"] == "sync"  # nosec B101
    assert jobs.created[0]["payload"]["force"] is False  # nosec B101
    assert jobs.created[0]["payload"]["languages"] == ["python"]  # nosec B101
    assert jobs.created[0]["payload"]["max_files"] == 5  # nosec B101


def test_codegraph_rejects_unknown_execution_mode(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    module = _module(tmp_path, workspace_root)

    with pytest.raises(ValueError, match="mode must be foreground, job, or background"):
        module.validate_tool_arguments("codegraph.index", {"mode": "later"})


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


@pytest.mark.asyncio
async def test_codegraph_read_tools_return_cross_file_relationships(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    package = workspace_root / "pkg"
    package.mkdir(parents=True)
    (package / "util.py").write_text(
        "def helper(value):\n    return value.upper()\n",
        encoding="utf-8",
    )
    (package / "app.py").write_text(
        "from pkg.util import helper\n\n\ndef entry(value):\n    return helper(value)\n",
        encoding="utf-8",
    )
    module = _module(tmp_path, workspace_root)

    index_result = await module.execute_tool(
        "codegraph.index",
        {"mode": "foreground", "force": True, "max_files": 10},
        context=_context(),
    )
    helper_search = await module.execute_tool(
        "codegraph.search",
        {"query": "helper", "kind": "function", "limit": 10},
        context=_context(),
    )
    entry_search = await module.execute_tool(
        "codegraph.search",
        {"query": "entry", "kind": "function", "limit": 10},
        context=_context(),
    )
    helper_id = helper_search["results"][0]["id"]
    entry_id = entry_search["results"][0]["id"]

    callers = await module.execute_tool(
        "codegraph.callers",
        {"node_id": helper_id, "limit": 10},
        context=_context(),
    )
    callees = await module.execute_tool(
        "codegraph.callees",
        {"node_id": entry_id, "limit": 10},
        context=_context(),
    )
    impact = await module.execute_tool(
        "codegraph.impact",
        {"node_id": helper_id, "direction": "incoming", "depth": 1, "limit": 10},
        context=_context(),
    )
    context_result = await module.execute_tool(
        "codegraph.context",
        {"task": "helper", "max_nodes": 5, "max_files": 2},
        context=_context(),
    )

    assert index_result["status"] == "complete"  # nosec B101
    assert [item["source"]["file_path"] for item in callers["relationships"]] == ["pkg/app.py"]  # nosec B101
    assert [item["target"]["file_path"] for item in callees["relationships"]] == ["pkg/util.py"]  # nosec B101
    assert "pkg/app.py" in {item["source"]["file_path"] for item in impact["relationships"]}  # nosec B101
    assert any(
        item["target"]["file_path"] == "pkg/util.py" for item in context_result["relationships"]
    )  # nosec B101


@pytest.mark.asyncio
async def test_codegraph_search_finds_typescript_component_after_index(tmp_path: Path) -> None:
    _require_typescript_parsers()
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    (workspace_root / "Card.tsx").write_text(
        "export function Card() { return <div />; }\n",
        encoding="utf-8",
    )
    module = _module(tmp_path, workspace_root)

    index_result = await module.execute_tool(
        "codegraph.index",
        {"mode": "foreground", "force": True, "max_files": 10},
        context=_context(),
    )
    search = await module.execute_tool(
        "codegraph.search",
        {"query": "Card", "kind": "component", "language": "typescript", "limit": 10},
        context=_context(),
    )

    assert index_result["status"] == "complete"  # nosec B101
    assert [item["qualified_name"] for item in search["results"]] == ["Card"]  # nosec B101
    assert [item["file_path"] for item in search["results"]] == ["Card.tsx"]  # nosec B101


@pytest.mark.asyncio
async def test_codegraph_search_finds_java_kotlin_symbols_after_index(tmp_path: Path) -> None:
    _require_jvm_parsers()
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    (workspace_root / "Service.java").write_text(
        """
package com.example.app;

public class Service {
    public String greet(String name) {
        return helper(name);
    }

    private String helper(String value) {
        return value.trim();
    }
}
""",
        encoding="utf-8",
    )
    (workspace_root / "Greeter.kt").write_text(
        """
package com.example.app

class Greeter {
    fun greet(name: String): String {
        return helper(name)
    }

    private fun helper(value: String): String {
        return value.trim()
    }
}
""",
        encoding="utf-8",
    )
    module = _module(tmp_path, workspace_root)

    index_result = await module.execute_tool(
        "codegraph.index",
        {"mode": "foreground", "force": True, "max_files": 10},
        context=_context(),
    )
    java_search = await module.execute_tool(
        "codegraph.search",
        {"query": "helper", "kind": "method", "language": "java", "limit": 10},
        context=_context(),
    )
    kotlin_search = await module.execute_tool(
        "codegraph.search",
        {"query": "helper", "kind": "function", "language": "kotlin", "limit": 10},
        context=_context(),
    )

    assert index_result["status"] == "complete"  # nosec B101
    assert index_result["counters"]["files_indexed"] == 2  # nosec B101
    assert [item["qualified_name"] for item in java_search["results"]] == [  # nosec B101
        "com.example.app.Service.helper"
    ]
    assert [item["qualified_name"] for item in kotlin_search["results"]] == [  # nosec B101
        "com.example.app.Greeter.helper"
    ]


@pytest.mark.asyncio
async def test_codegraph_search_finds_csharp_symbols_after_index(tmp_path: Path) -> None:
    _require_csharp_parser()
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    (workspace_root / "Greeter.cs").write_text(
        """
using System;

namespace Example.App;

public class Greeter {
    public string Greet(string name) {
        return Helper(name);
    }

    private string Helper(string value) {
        return value.Trim();
    }
}
""",
        encoding="utf-8",
    )
    module = _module(tmp_path, workspace_root)

    index_result = await module.execute_tool(
        "codegraph.index",
        {"mode": "foreground", "force": True, "max_files": 10},
        context=_context(),
    )
    search = await module.execute_tool(
        "codegraph.search",
        {"query": "Helper", "kind": "method", "language": "csharp", "limit": 10},
        context=_context(),
    )

    assert index_result["status"] == "complete"  # nosec B101
    assert index_result["counters"]["files_indexed"] == 1  # nosec B101
    assert [item["qualified_name"] for item in search["results"]] == [  # nosec B101
        "Example.App.Greeter.Helper"
    ]


@pytest.mark.asyncio
async def test_codegraph_search_finds_c_cpp_symbols_after_index(tmp_path: Path) -> None:
    _require_c_family_parsers()
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    (workspace_root / "greeter.c").write_text(
        """
#include <stdio.h>

int helper(int value) {
    return value + 1;
}

int greet(int name) {
    return helper(name);
}
""",
        encoding="utf-8",
    )
    (workspace_root / "Greeter.cpp").write_text(
        """
#include <string>

namespace demo {
class Greeter {
public:
    std::string greet(std::string name) {
        return helper(name);
    }

private:
    std::string helper(std::string value) {
        return value;
    }
};
}
""",
        encoding="utf-8",
    )
    module = _module(tmp_path, workspace_root)

    index_result = await module.execute_tool(
        "codegraph.index",
        {"mode": "foreground", "force": True, "max_files": 10},
        context=_context(),
    )
    c_search = await module.execute_tool(
        "codegraph.search",
        {"query": "helper", "kind": "function", "language": "c", "limit": 10},
        context=_context(),
    )
    cpp_search = await module.execute_tool(
        "codegraph.search",
        {"query": "helper", "kind": "method", "language": "cpp", "limit": 10},
        context=_context(),
    )

    assert index_result["status"] == "complete"  # nosec B101
    assert index_result["counters"]["files_indexed"] == 2  # nosec B101
    assert [item["qualified_name"] for item in c_search["results"]] == ["helper"]  # nosec B101
    assert [item["qualified_name"] for item in cpp_search["results"]] == ["demo.Greeter.helper"]  # nosec B101


def test_codegraph_rejects_ambiguous_node_selectors(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    module = _module(tmp_path, workspace_root)

    with pytest.raises(ValueError, match="node_id and symbol are mutually exclusive"):
        module.validate_tool_arguments(
            "codegraph.node",
            {"node_id": "node_helper", "symbol": "helper"},
        )


def test_codegraph_impact_rejects_invalid_arguments(tmp_path: Path) -> None:
    """Reject invalid selectors, directions, depth, and limit arguments."""
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    module = _module(tmp_path, workspace_root)

    with pytest.raises(ValueError, match="unknown arguments"):
        module.validate_tool_arguments("codegraph.impact", {"symbol": "helper", "unknown": True})
    with pytest.raises(ValueError, match="direction"):
        module.validate_tool_arguments("codegraph.impact", {"symbol": "helper", "direction": "sideways"})
    with pytest.raises(ValueError, match="depth"):
        module.validate_tool_arguments("codegraph.impact", {"symbol": "helper", "depth": 5})
    with pytest.raises(ValueError, match="limit"):
        module.validate_tool_arguments("codegraph.impact", {"symbol": "helper", "limit": 0})


@pytest.mark.asyncio
async def test_codegraph_impact_missing_index_is_read_only(tmp_path: Path) -> None:
    """Return an empty read-only impact response when no index database exists."""
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    index_base = tmp_path / "indexes"
    module = _module(tmp_path, workspace_root)

    result = await module.execute_tool("codegraph.impact", {"symbol": "helper"}, context=_context())

    assert result["index_present"] is False  # nosec B101
    assert result["root"] is None  # nosec B101
    assert result["nodes"] == []  # nosec B101
    assert result["relationships"] == []  # nosec B101
    assert not index_base.exists()  # nosec B101


@pytest.mark.asyncio
async def test_codegraph_impact_returns_bounded_relationship_neighborhood(tmp_path: Path) -> None:
    """Return a bounded incoming impact neighborhood for an indexed symbol."""
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
    result = await module.execute_tool(
        "codegraph.impact",
        {"symbol": " helper ", "direction": "incoming", "depth": 1, "limit": 10},
        context=_context(),
    )

    assert result["index_present"] is True  # nosec B101
    assert result["root"]["qualified_name"] == "helper"  # nosec B101
    assert result["depth"] == 1  # nosec B101
    assert result["direction"] == "incoming"  # nosec B101
    assert [node["qualified_name"] for node in result["nodes"]] == ["Greeter.greet", "helper"]  # nosec B101
    assert [relationship["source"]["qualified_name"] for relationship in result["relationships"]] == [
        "Greeter.greet"
    ]  # nosec B101
    assert result["truncated"] is False  # nosec B101


def test_codegraph_context_rejects_invalid_arguments(tmp_path: Path) -> None:
    """Reject invalid task, bounds, include_code, and unknown context arguments."""
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    module = _module(tmp_path, workspace_root)

    with pytest.raises(ValueError, match="unknown arguments"):
        module.validate_tool_arguments("codegraph.context", {"task": "helper", "unknown": True})
    with pytest.raises(ValueError, match="task"):
        module.validate_tool_arguments("codegraph.context", {"task": " "})
    with pytest.raises(ValueError, match="max_nodes"):
        module.validate_tool_arguments("codegraph.context", {"task": "helper", "max_nodes": 0})
    with pytest.raises(ValueError, match="max_files"):
        module.validate_tool_arguments("codegraph.context", {"task": "helper", "max_files": 0})
    with pytest.raises(ValueError, match="include_code"):
        module.validate_tool_arguments("codegraph.context", {"task": "helper", "include_code": "yes"})


def test_relationship_neighborhood_uses_repository_batch_traversal() -> None:
    """Collect context relationships with one repository batch traversal call."""
    node = CodeGraphNode(
        id="node_helper",
        identity_key="helper",
        kind="function",
        name="helper",
        qualified_name="helper",
        file_path="app.py",
        language="python",
    )
    relationship = {
        "id": "edge_call",
        "source": {"id": "node_caller", "qualified_name": "caller"},
        "target": {"id": "node_helper", "qualified_name": "helper"},
    }

    class _FakeRepository:
        """Record impact traversal calls made by relationship-neighborhood helper."""

        def __init__(self) -> None:
            self.calls: list[tuple[tuple[str, ...], int, str, int]] = []

        def traverse_impact_many(
            self,
            node_ids: tuple[str, ...],
            *,
            depth: int,
            direction: str,
            limit: int,
        ):  # noqa: ANN202
            """Return one fake impact relationship and record traversal arguments."""
            self.calls.append((node_ids, depth, direction, limit))
            return type("Impact", (), {"relationships": (relationship,)})()

        def traverse_impact(self, *_args, **_kwargs):  # noqa: ANN202
            """Fail if legacy single-node traversal is used by context assembly."""
            raise AssertionError("relationship neighborhood should use batch traversal")

    repository = _FakeRepository()

    result = _relationship_neighborhood(repository, (node,), limit=10)  # type: ignore[arg-type]

    assert result == [relationship]  # nosec B101
    assert repository.calls == [(("node_helper",), 1, "both", 10)]  # nosec B101


@pytest.mark.asyncio
async def test_codegraph_context_missing_index_is_read_only(tmp_path: Path) -> None:
    """Return an empty read-only context response when no index database exists."""
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    index_base = tmp_path / "indexes"
    module = _module(tmp_path, workspace_root)

    result = await module.execute_tool("codegraph.context", {"task": "helper"}, context=_context())

    assert result["index_present"] is False  # nosec B101
    assert result["nodes"] == []  # nosec B101
    assert result["files"] == []  # nosec B101
    assert result["relationships"] == []  # nosec B101
    assert not index_base.exists()  # nosec B101


@pytest.mark.asyncio
async def test_codegraph_context_returns_bounded_source_context(tmp_path: Path) -> None:
    """Build bounded context with related nodes, relationships, and source snippets."""
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
    result = await module.execute_tool(
        "codegraph.context",
        {"task": " helper ", "max_nodes": 5, "max_files": 2, "include_code": True},
        context=_context(),
    )

    assert result["index_present"] is True  # nosec B101
    assert result["task"] == "helper"  # nosec B101
    assert result["query"] == "helper"  # nosec B101
    assert [node["qualified_name"] for node in result["nodes"]] == ["helper"]  # nosec B101
    assert [relationship["source"]["qualified_name"] for relationship in result["relationships"]] == [
        "Greeter.greet"
    ]  # nosec B101
    assert [file_context["path"] for file_context in result["files"]] == ["app.py"]  # nosec B101
    assert "def helper" in result["files"][0]["snippets"][0]["text"]  # nosec B101
    assert result["truncation"]["truncated"] is False  # nosec B101


@pytest.mark.asyncio
async def test_codegraph_context_ranks_task_and_relationship_relevance(tmp_path: Path) -> None:
    """Select related task-token matches for multi-word context requests."""
    workspace_root = tmp_path / "workspace"
    package = workspace_root / "pkg"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "util.py").write_text(
        "def helper(value):\n    return value + 1\n",
        encoding="utf-8",
    )
    (package / "app.py").write_text(
        "from pkg.util import helper\n\n\ndef entry(value):\n    return helper(value)\n",
        encoding="utf-8",
    )
    (package / "noise.py").write_text(
        "def helper_noise(value):\n    return value - 1\n",
        encoding="utf-8",
    )
    for index in range(12):
        (package / f"noise_{index:02}.py").write_text(
            f"def update_noise_{index}(value):\n    return value - {index}\n",
            encoding="utf-8",
        )
    module = _module_with_settings(tmp_path, workspace_root, {"max_search_results": 3})

    await module.execute_tool(
        "codegraph.index",
        {"mode": "foreground", "force": True, "max_files": 25},
        context=_context(),
    )
    result = await module.execute_tool(
        "codegraph.context",
        {"task": "update entry helper flow", "max_nodes": 2, "max_files": 2, "include_code": False},
        context=_context(),
    )

    assert {node["name"] for node in result["nodes"]} == {"entry", "helper"}  # nosec B101
    assert "helper_noise" not in {node["name"] for node in result["nodes"]}  # nosec B101
    assert {file_context["path"] for file_context in result["files"]} == {"pkg/app.py", "pkg/util.py"}  # nosec B101
    assert any(  # nosec B101
        relationship["source"]["name"] == "entry" and relationship["target"]["name"] == "helper"
        for relationship in result["relationships"]
    )


@pytest.mark.asyncio
async def test_codegraph_context_treats_null_include_code_as_default_true(tmp_path: Path) -> None:
    """Treat JSON null include_code as the default source-including context mode."""
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    (workspace_root / "app.py").write_text("def helper():\n    return 1\n", encoding="utf-8")
    module = _module(tmp_path, workspace_root)

    await module.execute_tool(
        "codegraph.index",
        {"mode": "foreground", "force": True, "max_files": 10},
        context=_context(),
    )
    result = await module.execute_tool(
        "codegraph.context",
        {"task": "helper", "include_code": None},
        context=_context(),
    )

    assert "def helper" in result["files"][0]["snippets"][0]["text"]  # nosec B101


@pytest.mark.asyncio
async def test_codegraph_context_can_return_metadata_without_source_text(tmp_path: Path) -> None:
    """Return file metadata without source snippets when include_code is false."""
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    (workspace_root / "app.py").write_text("def helper():\n    return 1\n", encoding="utf-8")
    module = _module(tmp_path, workspace_root)

    await module.execute_tool(
        "codegraph.index",
        {"mode": "foreground", "force": True, "max_files": 10},
        context=_context(),
    )
    result = await module.execute_tool(
        "codegraph.context",
        {"task": "helper", "include_code": False},
        context=_context(),
    )

    assert result["files"][0]["path"] == "app.py"  # nosec B101
    assert result["files"][0]["snippets"] == []  # nosec B101
    assert result["truncation"]["used_chars"] == 0  # nosec B101


@pytest.mark.asyncio
async def test_codegraph_offloads_blocking_repository_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    (workspace_root / "app.py").write_text("x = 1\n", encoding="utf-8")
    jobs = _FakeJobManager()
    module = _module_with_settings(tmp_path, workspace_root, {}, job_manager_factory=lambda: jobs)
    offloaded: list[str] = []

    async def _fake_to_thread(func, /, *args, **kwargs):  # noqa: ANN001
        offloaded.append(func.__name__)
        return func(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", _fake_to_thread)

    await module.execute_tool("codegraph.index", {"mode": "foreground"}, context=_context())
    await module.execute_tool("codegraph.files", {}, context=_context())
    await module.execute_tool("codegraph.search", {"query": "app"}, context=_context())
    await module.execute_tool("codegraph.sync", {"mode": "foreground"}, context=_context())
    await module.execute_tool("codegraph.index", {"mode": "job"}, context=_context())
    await module.execute_tool("codegraph.impact", {"symbol": "app"}, context=_context())
    await module.execute_tool("codegraph.context", {"task": "app"}, context=_context())

    assert "_run_index" in offloaded  # nosec B101
    assert "_run_sync" in offloaded  # nosec B101
    assert "_enqueue_index_job" in offloaded  # nosec B101
    assert "_impact" in offloaded  # nosec B101
    assert "_build_context" in offloaded  # nosec B101


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
