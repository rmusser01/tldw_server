from __future__ import annotations

from dataclasses import dataclass


Classification = str


@dataclass(frozen=True)
class CompatibilityItem:
    name: str
    classification: Classification
    upstream_reference: str


@dataclass(frozen=True)
class CompatibilityInventory:
    items: tuple[CompatibilityItem, ...]


def load_builtin_inventory() -> CompatibilityInventory:
    items = (
        CompatibilityItem("cli:task-list-plain", "golden-required", "CLI-INSTRUCTIONS.md"),
        CompatibilityItem("cli:task-view-plain", "golden-required", "CLI-INSTRUCTIONS.md"),
        CompatibilityItem("cli:search-plain", "golden-required", "CLI-INSTRUCTIONS.md"),
        CompatibilityItem("cli:config-list", "golden-required", "ADVANCED-CONFIG.md"),
        CompatibilityItem("mcp:workflow-overview", "golden-required", "agent-nudge.md"),
        CompatibilityItem("mcp:task-search", "golden-required", "MCP tools"),
        CompatibilityItem("browser:kanban-drag-drop", "browser-deferred", "README.md"),
        CompatibilityItem("cli:interactive-board", "interactive-deferred", "CLI-INSTRUCTIONS.md"),
    )
    return CompatibilityInventory(items=items)
