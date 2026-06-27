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

_DISABLED_STATUSES = {"disabled", "not_loaded", "inactive"}


def _is_enabled(payload: Any) -> bool:
    """Return whether a status/config payload describes an enabled module."""
    if isinstance(payload, dict):
        if payload.get("enabled") is False:
            return False
        status = str(payload.get("status", "")).strip().lower()
        return status not in _DISABLED_STATUSES
    return True


def describe_module_surface(modules: dict[str, Any]) -> dict[str, Any]:
    """Group enabled MCP modules into user-facing capability risk tiers."""
    tiers: dict[str, dict[str, Any]] = {
        key: {"label": label, "modules": []}
        for key, label in TIER_LABELS.items()
    }
    enabled_count = 0

    for module_name, payload in sorted(modules.items()):
        if not _is_enabled(payload):
            continue

        enabled_count += 1
        tier, description = MODULE_RISK_TIERS.get(
            module_name,
            ("unknown", "No risk tier is registered yet."),
        )
        tiers[tier]["modules"].append({"id": module_name, "description": description})

    return {
        "enabled_count": enabled_count,
        "tiers": {
            key: value
            for key, value in tiers.items()
            if value["modules"]
        },
    }
