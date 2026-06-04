"""Default profile tooling metadata and recommendation catalog helpers."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

DEFAULT_MAX_DIRECT_TOOLS = 24

CHROME_DEVTOOLS_MCP_OPTION: dict[str, Any] = {
    "id": "chrome-devtools-mcp",
    "category": "browser",
    "kind": "external_mcp",
    "install_target": "ChromeDevTools/chrome-devtools-mcp",
    "credential_slots": [],
    "required_scopes": [],
    "risk_classes": ["external_network"],
    "maturity": "exact_target",
    "setup_url": "https://github.com/ChromeDevTools/chrome-devtools-mcp",
}


def tooling_metadata(
    *,
    enabled_tools: list[str],
    enabled_capabilities: list[str],
    direct_categories: list[str],
    deferred_categories: list[str],
    recommended_tools: list[dict[str, Any]] | None = None,
    recommended_servers: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return caller-owned tooling metadata for a role preset."""
    return {
        "enabled_tools": list(enabled_tools),
        "enabled_capabilities": list(enabled_capabilities),
        "recommended_tools": deepcopy(recommended_tools or []),
        "recommended_servers": deepcopy(recommended_servers or []),
        "recommendation_catalog_patchable": True,
        "progressive_disclosure": {
            "direct_categories": list(direct_categories),
            "deferred_categories": list(deferred_categories),
            "max_direct_tools": DEFAULT_MAX_DIRECT_TOOLS,
        },
        "tool_search": {
            "ranking": ["profile_grants", "installation_status", "category", "bm25"],
            "semantic_search": False,
        },
    }


def recommended_tool(
    tool_id: str,
    *,
    category: str,
    description: str,
    activation: str = "requires_operator_enablement",
) -> dict[str, Any]:
    """Return recommendation-only metadata for an optional tool."""
    return {
        "id": tool_id,
        "category": category,
        "description": description,
        "activation": activation,
        "required": False,
        "authority": "recommendation_only",
    }


def browser_server_recommendation() -> dict[str, Any]:
    """Return the browser/CDP recommendation category."""
    return {
        "category": "browser",
        "required": False,
        "binding_options": [deepcopy(CHROME_DEVTOOLS_MCP_OPTION)],
    }


def web_search_server_recommendation() -> dict[str, Any]:
    """Return vendor-neutral web-search recommendation metadata."""
    return {
        "category": "web_search",
        "required": False,
        "binding_options": [
            {
                "id": "configured-web-search",
                "category": "web_search",
                "kind": "external_mcp",
                "install_target": None,
                "credential_slots": [],
                "required_scopes": ["search:read"],
                "risk_classes": ["external_network"],
                "maturity": "category_placeholder",
                "activation": "requires_configured_provider",
            }
        ],
    }


def issue_tracker_server_recommendation() -> dict[str, Any]:
    """Return vendor-neutral issue-tracker recommendation metadata."""
    return {
        "category": "issue_tracker",
        "required": False,
        "binding_options": [
            {
                "id": "jira",
                "category": "issue_tracker",
                "kind": "external_mcp",
                "install_target": None,
                "credential_slots": ["jira_api_token"],
                "required_scopes": ["issues:read", "issues:write"],
                "risk_classes": ["external_network", "mutating"],
                "maturity": "category_placeholder",
            },
            {
                "id": "linear",
                "category": "issue_tracker",
                "kind": "external_mcp",
                "install_target": None,
                "credential_slots": ["linear_api_key"],
                "required_scopes": ["issues:read", "issues:write"],
                "risk_classes": ["external_network", "mutating"],
                "maturity": "category_placeholder",
            },
        ],
    }


def merge_tooling_recommendations(
    tooling: dict[str, Any],
    patch: dict[str, Any],
) -> dict[str, Any]:
    """Return patched recommendation metadata without changing policy."""
    merged = deepcopy(tooling)
    for key in ("recommended_tools", "recommended_servers"):
        additions = patch.get(key)
        if isinstance(additions, list):
            merged.setdefault(key, [])
            merged[key].extend(deepcopy(additions))
    return merged
