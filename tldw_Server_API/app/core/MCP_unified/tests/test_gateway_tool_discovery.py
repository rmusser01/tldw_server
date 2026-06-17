"""Tests for profile-scoped MCP gateway tool discovery helpers."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

from mcp_unified.gateway import tool_discovery
from mcp_unified.gateway.tool_discovery import (
    describe_profile_tool,
    find_direct_profile_backend_tool,
    list_profile_tools,
    profile_has_deferred_installed_tools,
    profile_tool_availability,
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


def test_list_profile_tools_consolidates_category_counts_by_normalized_name() -> None:
    profile = _profile(capabilities=["code_search"])
    tools = [
        {
            "name": "code.search",
            "description": "Search code",
            "metadata": {"capability": "code_search", "category": "Code"},
        },
        {
            "name": "code.symbols",
            "description": "Find symbols",
            "metadata": {"capability": "code_search", "category": "code"},
        },
    ]

    payload = list_profile_tools(profile, tools)
    results = search_profile_tools(profile, tools, query="", category="CODE")

    assert payload["categories"] == [
        {
            "category": "code",
            "count": 2,
            "direct_count": 0,
            "deferred_installed_count": 2,
            "installed_count": 2,
            "recommended_unavailable_count": 0,
        }
    ]
    assert {item["tool_id"] for item in results} == {"code.search", "code.symbols"}


def test_profile_tools_classify_direct_deferred_and_recommended_availability() -> None:
    profile = MCPProfile(
        id="engineer",
        name="Engineer",
        policy_document=ProfilePolicy(capabilities=["code_search", "browser.inspect"]),
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
                    "direct_categories": ["code"],
                    "deferred_categories": ["browser"],
                    "max_direct_tools": 24,
                },
            }
        },
    )
    tools = [
        {
            "name": "code.search",
            "description": "Search code",
            "metadata": {"capability": "code_search", "category": "code"},
        },
        {
            "name": "browser.snapshot",
            "description": "Browser DOM snapshot",
            "metadata": {"capability": "browser.inspect", "category": "browser"},
        },
    ]

    payload = list_profile_tools(profile, tools)
    entries = {item["tool_id"]: item for item in payload["tools"]}
    categories = {item["category"]: item for item in payload["categories"]}

    assert entries["code.search"]["exposure"] == "direct"
    assert entries["code.search"]["availability_reason_code"] == "installed_direct"
    assert entries["browser.snapshot"]["exposure"] == "deferred"
    assert entries["browser.snapshot"]["availability_reason_code"] == "installed_deferred"
    assert entries["browser.trace"]["exposure"] == "recommended_unavailable"
    assert entries["browser.trace"]["availability_reason_code"] == "recommended_unavailable"
    assert categories["code"]["direct_count"] == 1
    assert categories["code"]["deferred_installed_count"] == 0
    assert categories["browser"]["direct_count"] == 0
    assert categories["browser"]["deferred_installed_count"] == 1
    assert categories["browser"]["recommended_unavailable_count"] == 1
    assert payload["availability"] == {
        "count": 3,
        "installed_count": 2,
        "direct_count": 1,
        "deferred_installed_count": 1,
        "recommended_unavailable_count": 1,
    }


def test_profile_tool_availability_reports_callable_deferred_tools_only() -> None:
    profile = MCPProfile(
        id="engineer",
        name="Engineer",
        policy_document=ProfilePolicy(capabilities=["code_search", "browser.inspect"]),
        metadata={
            "tooling": {
                "recommended_tools": [
                    {
                        "id": "browser.trace",
                        "category": "browser",
                        "description": "Browser trace capture",
                        "capability": "browser.inspect",
                    }
                ],
                "progressive_disclosure": {
                    "direct_categories": ["code"],
                    "deferred_categories": ["browser"],
                    "max_direct_tools": 24,
                },
            }
        },
    )
    tools = [
        {
            "name": "code.search",
            "description": "Search code",
            "metadata": {"capability": "code_search", "category": "code"},
        },
        {
            "name": "browser.snapshot",
            "description": "Browser DOM snapshot",
            "metadata": {"capability": "browser.inspect", "category": "browser"},
        },
    ]

    availability = profile_tool_availability(profile, tools)
    direct_tool = find_direct_profile_backend_tool(profile, tools, "code.search")

    assert [tool["name"] for tool in availability.direct_tools] == ["code.search"]
    assert availability.direct_tools[0] is not tools[0]
    assert availability.has_deferred_installed_tools is True
    assert availability.has_recommended_unavailable_tools is True
    assert profile_has_deferred_installed_tools(profile, tools) is True
    assert direct_tool == tools[0]
    assert direct_tool is not tools[0]
    assert find_direct_profile_backend_tool(profile, tools, "browser.snapshot") is None


def test_profile_tools_ignore_malformed_direct_category_metadata() -> None:
    profile = MCPProfile(
        id="engineer",
        name="Engineer",
        policy_document=ProfilePolicy(capabilities=["code_search"]),
        metadata={
            "tooling": {
                "progressive_disclosure": {
                    "direct_categories": {"code": True},
                    "deferred_categories": ["code"],
                    "max_direct_tools": 24,
                },
            }
        },
    )
    tools = [
        {
            "name": "code.search",
            "description": "Search code",
            "metadata": {"capability": "code_search", "category": "code"},
        }
    ]

    payload = list_profile_tools(profile, tools)

    assert payload["tools"][0]["tool_id"] == "code.search"
    assert payload["tools"][0]["exposure"] == "deferred"
    assert payload["availability"]["direct_count"] == 0
    assert payload["availability"]["deferred_installed_count"] == 1


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


def test_recommendations_without_executable_grants_are_discoverable_not_callable() -> None:
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
    ungranted_capability_description = describe_profile_tool(profile, [], "shell.run")
    no_capability_resolution = resolve_profile_tool_call(profile, [], "code.index")
    ungranted_capability_resolution = resolve_profile_tool_call(profile, [], "shell.run")

    assert {item["tool_id"] for item in results} == {"code.index", "shell.run"}
    assert no_capability_description is not None
    assert no_capability_description["installation_status"] == "recommended_unavailable"
    assert no_capability_description["capabilities"] == []
    assert ungranted_capability_description is not None
    assert ungranted_capability_description["installation_status"] == "recommended_unavailable"
    assert ungranted_capability_description["capabilities"] == ["process.execute"]
    assert no_capability_resolution["status"] == "unavailable"
    assert no_capability_resolution["reason_code"] == "tool_not_enabled"
    assert no_capability_resolution["tool_id"] == "code.index"
    assert ungranted_capability_resolution["status"] == "unavailable"
    assert ungranted_capability_resolution["reason_code"] == "tool_not_enabled"
    assert ungranted_capability_resolution["tool_id"] == "shell.run"


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
    recommended = describe_profile_tool(profile, [], "code.index")
    resolved = resolve_profile_tool_call(profile, [], "browser.trace")
    inactive = resolve_profile_tool_call(profile, [], "code.index")

    assert [item["tool_id"] for item in results] == ["browser.trace", "code.index"]
    assert allowed is not None
    assert allowed["installation_status"] == "recommended_unavailable"
    assert recommended is not None
    assert recommended["installation_status"] == "recommended_unavailable"
    assert resolved["status"] == "unavailable"
    assert resolved["reason_code"] == "tool_not_enabled"
    assert resolved["tool_id"] == "browser.trace"
    assert inactive["status"] == "unavailable"
    assert inactive["reason_code"] == "tool_not_enabled"
    assert inactive["tool_id"] == "code.index"


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
