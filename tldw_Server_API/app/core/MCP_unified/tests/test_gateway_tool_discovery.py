"""Tests for profile-scoped MCP gateway tool discovery helpers."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

from mcp_unified.gateway import tool_discovery
from mcp_unified.gateway.tool_discovery import (
    describe_profile_tool,
    list_profile_tools,
    resolve_profile_tool_call,
    search_profile_tools,
)
from mcp_unified.profiles.models import MCPProfile, ProfilePolicy


REPO_ROOT = Path(__file__).resolve().parents[5]
TOOL_DISCOVERY_PATH = REPO_ROOT / "mcp_unified" / "gateway" / "tool_discovery.py"


def _profile(
    *,
    capabilities: list[str] | None = None,
    allowed_tools: list[str] | None = None,
    denied_tools: list[str] | None = None,
    recommended_tools: list[dict[str, Any]] | None = None,
) -> MCPProfile:
    return MCPProfile(
        id="profile",
        name="Profile",
        policy_document=ProfilePolicy(
            allowed_tools=allowed_tools or [],
            denied_tools=denied_tools or [],
            capabilities=capabilities or [],
        ),
        metadata={
            "tooling": {
                "recommended_tools": recommended_tools or [],
                "progressive_disclosure": {
                    "direct_categories": [],
                    "deferred_categories": ["browser", "code"],
                    "max_direct_tools": 24,
                },
            }
        },
    )


def test_tool_search_filters_by_profile_before_bm25() -> None:
    profile = MCPProfile(
        id="reviewer",
        name="Reviewer",
        policy_document=ProfilePolicy(capabilities=["code_search"]),
    )
    tools = [
        {
            "name": "code.search",
            "description": "Search code",
            "metadata": {"capability": "code_search", "category": "code"},
        },
        {
            "name": "shell.run",
            "description": "Run shell commands",
            "metadata": {"capability": "process.execute", "category": "shell"},
        },
    ]

    results = search_profile_tools(profile, tools, query="run search")

    assert [item["tool_id"] for item in results] == ["code.search"]


def test_tool_search_orders_installed_before_unavailable_then_bm25() -> None:
    profile = MCPProfile(
        id="frontend",
        name="Frontend",
        policy_document=ProfilePolicy(capabilities=["browser.inspect"]),
        metadata={
            "tooling": {
                "recommended_tools": [
                    {
                        "id": "browser.trace",
                        "category": "browser",
                        "description": "Browser trace capture",
                        "capability": "browser.inspect",
                        "activation": "requires_browser_runtime",
                    }
                ],
                "progressive_disclosure": {
                    "direct_categories": [],
                    "deferred_categories": ["browser"],
                    "max_direct_tools": 24,
                },
            }
        },
    )
    installed = [
        {
            "name": "browser.snapshot",
            "description": "Browser DOM snapshot",
            "metadata": {
                "capability": "browser.inspect",
                "category": "browser",
            },
        }
    ]

    results = search_profile_tools(
        profile,
        installed,
        query="browser",
        category="browser",
    )

    assert [item["tool_id"] for item in results] == [
        "browser.snapshot",
        "browser.trace",
    ]
    assert results[0]["installation_status"] == "installed"
    assert results[1]["installation_status"] == "recommended_unavailable"


def test_describe_profile_tool_returns_visible_installed_and_recommended_only() -> None:
    profile = _profile(
        capabilities=["code_search"],
        recommended_tools=[
            {
                "id": "code.index",
                "category": "code",
                "description": "Build a code index after setup",
                "capability": "code_search",
                "activation": "requires_index_runtime",
            }
        ],
    )
    tools = [
        {
            "name": "code.search",
            "description": "Search code",
            "metadata": {
                "capability": "code_search",
                "category": "code",
                "display_name": "Code Search",
            },
        },
        {
            "name": "shell.run",
            "description": "Run shell commands",
            "metadata": {"capability": "process.execute", "category": "shell"},
        },
    ]

    installed = describe_profile_tool(profile, tools, "code.search")
    recommended = describe_profile_tool(profile, tools, "code.index")
    denied = describe_profile_tool(profile, tools, "shell.run")

    assert installed is not None
    assert installed["tool_id"] == "code.search"
    assert installed["installation_status"] == "installed"
    assert installed["display_name"] == "Code Search"
    assert recommended is not None
    assert recommended["tool_id"] == "code.index"
    assert recommended["installation_status"] == "recommended_unavailable"
    assert recommended["activation"] == "requires_index_runtime"
    assert denied is None


def test_resolve_profile_tool_call_resolves_installed_and_rejects_recommended() -> None:
    profile = _profile(
        capabilities=["browser.inspect"],
        recommended_tools=[
            {
                "id": "browser.trace",
                "category": "browser",
                "description": "Browser trace capture",
                "capability": "browser.inspect",
                "activation": "requires_browser_runtime",
            }
        ],
    )
    backend_tool = {
        "name": "browser.snapshot",
        "description": "Browser DOM snapshot",
        "metadata": {"capability": "browser.inspect", "category": "browser"},
    }
    tools = [
        backend_tool,
        {
            "name": "shell.run",
            "description": "Run shell commands",
            "metadata": {"capability": "process.execute", "category": "shell"},
        },
    ]

    resolved = resolve_profile_tool_call(profile, tools, "browser.snapshot")
    unavailable = resolve_profile_tool_call(profile, tools, "browser.trace")
    denied = resolve_profile_tool_call(profile, tools, "shell.run")

    assert resolved == {
        "status": "resolved",
        "tool_id": "browser.snapshot",
        "tool_name": "browser.snapshot",
        "tool": backend_tool,
    }
    assert unavailable["status"] == "unavailable"
    assert unavailable["reason_code"] == "tool_not_enabled"
    assert unavailable["tool_id"] == "browser.trace"
    assert unavailable["installation_status"] == "recommended_unavailable"
    assert denied == {
        "status": "not_found",
        "reason_code": "tool_not_found",
        "tool_id": "shell.run",
    }


def test_ungranted_recommended_tools_are_not_discoverable() -> None:
    profile = _profile(
        capabilities=["code_search"],
        recommended_tools=[
            {
                "id": "code.index",
                "category": "code",
                "description": "No capability recommendation",
                "activation": "requires_index_runtime",
            },
            {
                "id": "shell.run",
                "category": "shell",
                "description": "Run shell commands",
                "capability": "process.execute",
                "activation": "requires_operator_enablement",
            },
        ],
    )

    results = search_profile_tools(profile, [], query="index shell")
    no_capability_description = describe_profile_tool(profile, [], "code.index")
    denied_capability_description = describe_profile_tool(profile, [], "shell.run")
    no_capability_resolution = resolve_profile_tool_call(profile, [], "code.index")
    denied_capability_resolution = resolve_profile_tool_call(profile, [], "shell.run")

    assert results == []
    assert no_capability_description is None
    assert denied_capability_description is None
    assert no_capability_resolution == {
        "status": "not_found",
        "reason_code": "tool_not_found",
        "tool_id": "code.index",
    }
    assert denied_capability_resolution == {
        "status": "not_found",
        "reason_code": "tool_not_found",
        "tool_id": "shell.run",
    }


def test_recommended_tools_preserve_explicit_allowed_tools_semantics() -> None:
    profile = _profile(
        allowed_tools=["browser.trace"],
        recommended_tools=[
            {
                "id": "browser.trace",
                "category": "browser",
                "description": "Browser trace capture",
                "activation": "requires_browser_runtime",
            },
            {
                "id": "code.index",
                "category": "code",
                "description": "Code index setup",
                "capability": "code_search",
                "activation": "requires_index_runtime",
            },
        ],
    )

    results = search_profile_tools(profile, [], query="browser code")
    allowed = describe_profile_tool(profile, [], "browser.trace")
    denied = describe_profile_tool(profile, [], "code.index")
    resolved = resolve_profile_tool_call(profile, [], "browser.trace")

    assert [item["tool_id"] for item in results] == ["browser.trace"]
    assert allowed is not None
    assert allowed["installation_status"] == "recommended_unavailable"
    assert denied is None
    assert resolved["status"] == "unavailable"
    assert resolved["reason_code"] == "tool_not_enabled"
    assert resolved["tool_id"] == "browser.trace"


def test_unknown_backend_descriptors_do_not_crash_or_leak() -> None:
    profile = _profile(capabilities=["code_search"])
    tools: list[Any] = [
        None,
        "code.search",
        {"description": "missing name", "metadata": {"capability": "code_search"}},
        {"name": "   ", "metadata": {"capability": "code_search"}},
        {
            "name": "future.tool",
            "description": "Future descriptor shape",
            "metadata": {"category": "code"},
            "unexpected": {"capability": "code_search"},
        },
        {
            "name": "code.search",
            "description": "Search code",
            "metadata": {"capability": "code_search", "category": "code"},
        },
    ]

    results = search_profile_tools(profile, tools, query="future search")
    listed = list_profile_tools(profile, tools)

    assert [item["tool_id"] for item in results] == ["code.search"]
    assert [item["tool_id"] for item in listed["tools"]] == ["code.search"]


def test_tool_search_uses_standard_library_bm25_metadata() -> None:
    profile = _profile(capabilities=["code_search"])
    tools = [
        {
            "name": "code.search",
            "description": "Search code",
            "metadata": {"capability": "code_search", "category": "code"},
        }
    ]

    payload = list_profile_tools(profile, tools)
    results = search_profile_tools(profile, tools, query="code")

    assert payload["ranking"]["semantic_search"] is False
    assert payload["ranking"]["scoring"] == "bm25_standard_library"
    assert results[0]["ranking"]["semantic_search"] is False
    assert results[0]["ranking"]["scoring"] == "bm25_standard_library"


def test_tool_discovery_module_keeps_package_boundary_clean() -> None:
    tree = ast.parse(TOOL_DISCOVERY_PATH.read_text(encoding="utf-8"))
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)

    assert "tldw_Server_API" not in imports
    assert all(not item.startswith("tldw_Server_API.") for item in imports)
    assert tool_discovery.__name__ == "mcp_unified.gateway.tool_discovery"
