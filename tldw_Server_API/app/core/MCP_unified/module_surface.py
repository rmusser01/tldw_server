"""User-facing summaries of the effective Unified MCP module surface."""

from __future__ import annotations

from typing import Any


MODULE_RISK_TIERS: dict[str, tuple[str, str]] = {
    "media": ("read_only", "Search and retrieve existing media records."),
    "knowledge": ("read_only", "Search and retrieve knowledge records."),
    "chats": ("read_only", "Read chat/session context."),
    "prompts": ("read_only", "Read prompt library entries."),
    "prompts_catalog": ("read_only", "Expose configured prompt catalogs."),
    "mcp_discovery": ("read_only", "Inspect MCP capabilities."),
    "governance": ("write", "Manage or inspect policy/governance state."),
    "notes": ("write", "Create or modify note data."),
    "template": ("write", "Create or modify generated content templates."),
    "quizzes": ("write", "Create or modify quiz data."),
    "flashcards": ("write", "Create or modify flashcard data."),
    "kanban": ("write", "Create or modify board/task data."),
    "slides": ("write", "Create or export slide artifacts."),
    "characters": ("write", "Manage character-related data."),
    "persona_visuals": ("write", "Manage persona visual assets."),
    "filesystem": ("local_files", "Read or write configured local file scopes."),
    "codegraph": ("local_files", "Index and inspect configured source workspaces."),
    "git": ("local_process", "Inspect configured Git workspaces through allowlisted commands."),
    "external_federation": ("external_network", "Connect to external MCP servers."),
    "web_fetch": ("external_network", "Fetch bounded external web content."),
    "web_search": ("external_network", "Search configured external web providers."),
    "web_research": ("external_network", "Compose bounded external search and fetch workflows."),
    "run_command": ("local_process", "Run configured local command families."),
    "sandbox": ("local_process", "Run code or workloads in configured sandboxes."),
    "browser_cdp": ("local_process", "Control a configured browser over the Chrome DevTools Protocol."),
}

TIER_LABELS: dict[str, str] = {
    "read_only": "Read-only data access",
    "write": "Writes to TLDW data",
    "local_files": "Local filesystem or workspace access",
    "external_network": "External server or network access",
    "local_process": "Local process or sandbox execution",
    "unknown": "Unclassified module",
}

EXPLICIT_OPT_IN_TIERS = {"local_files", "local_process", "external_network"}

_DISABLED_STATUSES = {"disabled", "not_loaded", "inactive"}


def _is_enabled(payload: Any) -> bool:
    """Return whether a status/config payload describes an enabled module."""
    if isinstance(payload, dict):
        if payload.get("enabled") is False:
            return False
        status = str(payload.get("status", "")).strip().lower()
        return status not in _DISABLED_STATUSES
    return True


def _disabled_next_action(module_name: str, tier: str) -> str:
    """Return user-facing guidance for enabling a disabled high-risk module."""
    if tier == "local_files":
        risk = "local file or workspace access"
    elif tier == "local_process":
        risk = "local process or sandbox execution"
    elif tier == "external_network":
        risk = "external network or federated MCP access"
    else:
        risk = "this capability"
    return (
        f"Enable `{module_name}` in Config_Files/mcp_modules.yaml only if this "
        f"deployment should expose {risk}; restart TLDW Server and recheck /api/v1/mcp/status."
    )


def describe_module_surface(modules: dict[str, Any]) -> dict[str, Any]:
    """Group enabled MCP modules into user-facing capability risk tiers."""
    tiers: dict[str, dict[str, Any]] = {
        key: {"label": label, "modules": []}
        for key, label in TIER_LABELS.items()
    }
    enabled_count = 0
    disabled_available: list[dict[str, Any]] = []

    for module_name, payload in sorted(modules.items()):
        tier, description = MODULE_RISK_TIERS.get(
            module_name,
            ("unknown", "No risk tier is registered yet."),
        )
        if not _is_enabled(payload):
            explicitly_disabled = isinstance(payload, dict) and payload.get("enabled") is False
            if explicitly_disabled and tier in EXPLICIT_OPT_IN_TIERS:
                disabled_available.append(
                    {
                        "id": module_name,
                        "tier": tier,
                        "label": TIER_LABELS.get(tier, TIER_LABELS["unknown"]),
                        "description": description,
                        "requires_explicit_opt_in": True,
                        "next_action": _disabled_next_action(module_name, tier),
                    }
                )
            continue

        enabled_count += 1
        tiers[tier]["modules"].append({"id": module_name, "description": description})

    return {
        "enabled_count": enabled_count,
        "disabled_available_count": len(disabled_available),
        "disabled_available": disabled_available,
        "tiers": {
            key: value
            for key, value in tiers.items()
            if value["modules"]
        },
    }
