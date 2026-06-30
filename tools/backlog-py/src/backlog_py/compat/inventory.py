from __future__ import annotations

from dataclasses import dataclass


Classification = str


@dataclass(frozen=True)
class CompatibilityItem:
    name: str
    classification: Classification
    upstream_reference: str
    expected: str
    status: str
    fixture: str | None = None
    deferred_reason: str | None = None


@dataclass(frozen=True)
class CompatibilityInventory:
    items: tuple[CompatibilityItem, ...]


def load_builtin_inventory() -> CompatibilityInventory:
    items = (
        _golden("cli:help", "CLI-INSTRUCTIONS.md", "backlog --help"),
        _golden("cli:task-list-plain", "CLI-INSTRUCTIONS.md", "backlog task list --plain"),
        _golden("cli:task-view-plain", "CLI-INSTRUCTIONS.md", "backlog task <id> --plain"),
        _golden("cli:search-plain", "CLI-INSTRUCTIONS.md", "backlog search <query> --plain"),
        _golden("cli:board", "CLI-INSTRUCTIONS.md", "backlog board"),
        _golden("cli:config-list", "ADVANCED-CONFIG.md", "backlog config list"),
        _golden("cli:task-create", "CLI-INSTRUCTIONS.md", "backlog task create <title> --status <status> --plain"),
        _golden("cli:task-edit", "CLI-INSTRUCTIONS.md", "backlog task edit <id> --append-notes <text> --plain"),
        _golden("cli:doc-list", "CLI-INSTRUCTIONS.md", "backlog doc list"),
        _golden("cli:doc-view", "CLI-INSTRUCTIONS.md", "backlog doc view <path-or-id>"),
        _golden("cli:doc-create", "CLI-INSTRUCTIONS.md", "backlog doc create <path> --title <title> --content <body>"),
        _golden("cli:doc-update", "CLI-INSTRUCTIONS.md", "backlog doc update <path-or-id> --title <title>"),
        _golden("cli:milestone-list", "CLI-INSTRUCTIONS.md", "backlog milestone list"),
        _golden("cli:milestone-add", "CLI-INSTRUCTIONS.md", "backlog milestone add <name>"),
        _golden("cli:milestone-rename", "CLI-INSTRUCTIONS.md", "backlog milestone rename <old> <new>"),
        _golden("cli:milestone-remove", "CLI-INSTRUCTIONS.md", "backlog milestone remove <name>"),
        _golden("cli:milestone-archive", "CLI-INSTRUCTIONS.md", "backlog milestone archive <name>"),
        _golden("cli:config-dod-defaults-get", "ADVANCED-CONFIG.md", "backlog config dod-defaults-get"),
        _golden(
            "cli:config-dod-defaults-upsert",
            "ADVANCED-CONFIG.md",
            "backlog config dod-defaults-upsert [item...]",
        ),
        _golden("mcp:workflow-overview", "agent-nudge.md", "backlog://workflow/overview"),
        _golden("mcp:task-workflow-alias", "agent-nudge.md", "backlog://docs/task-workflow"),
        _golden("mcp:task-search", "MCP tools", "task_search(project, query, limit=10)"),
        _golden("mcp:task-view", "MCP tools", "task_view(project, task_id)"),
        _golden("mcp:task-create", "MCP tools", "task_create(project, **kwargs)"),
        _golden("mcp:task-edit", "MCP tools", "task_edit(project, task_id, **kwargs)"),
        _golden("mcp:document-list", "MCP tools", "document_list(project, query=None, limit=100)"),
        _golden("mcp:document-search", "MCP tools", "document_list(project, query=<query>, limit=100)"),
        _golden("mcp:document-view", "MCP tools", "document_view(project, path_or_id)"),
        _golden("mcp:document-create", "MCP tools", "document_create(project, **kwargs)"),
        _golden("mcp:document-update", "MCP tools", "document_update(project, path_or_id, **kwargs)"),
        _golden("mcp:milestone-list", "MCP tools", "milestone_list(project)"),
        _golden("mcp:milestone-add", "MCP tools", "milestone_add(project, name, description='')"),
        _golden(
            "mcp:milestone-rename",
            "MCP tools",
            "milestone_rename(project, old_name, new_name, update_tasks=False)",
        ),
        _golden("mcp:milestone-remove", "MCP tools", "milestone_remove(project, name, clear_tasks=False)"),
        _golden("mcp:milestone-archive", "MCP tools", "milestone_archive(project, name)"),
        _golden(
            "mcp:definition-of-done-defaults-get",
            "MCP tools",
            "definition_of_done_defaults_get(project)",
        ),
        _golden(
            "mcp:definition-of-done-defaults-upsert",
            "MCP tools",
            "definition_of_done_defaults_upsert(project, items)",
        ),
        _deferred(
            "browser:kanban-drag-drop",
            "browser-deferred",
            "README.md",
            "backlog browser",
            "Browser UI parity is tracked in the browser deferral milestone.",
        ),
        _deferred(
            "cli:interactive-board",
            "interactive-deferred",
            "CLI-INSTRUCTIONS.md",
            "backlog board interactive controls",
            "Interactive terminal controls are deferred behind non-interactive agent workflows.",
        ),
        _deferred(
            "cli:rich-colored-output",
            "interactive-deferred",
            "CLI-INSTRUCTIONS.md",
            "ANSI-rich terminal rendering",
            "Plain output is the cutover blocker; rich color is later polish.",
        ),
        _deferred(
            "cli:shell-completion-install",
            "completion-deferred",
            "CLI-INSTRUCTIONS.md",
            "backlog completion install",
            "Shell completion installation is not needed for agent runtime cutover.",
        ),
        _deferred(
            "core:on-status-change",
            "automation-deferred",
            "ADVANCED-CONFIG.md",
            "onStatusChange hooks",
            "Hook execution remains disabled until a separate safety review.",
        ),
        _deferred(
            "git:remote-operations",
            "git-deferred",
            "ADVANCED-CONFIG.md",
            "remote git operations",
            "Remote git behavior is outside the first local-file compatibility gate.",
        ),
        _deferred(
            "git:auto-commit",
            "git-deferred",
            "ADVANCED-CONFIG.md",
            "autoCommit",
            "Automatic commits are deferred to keep mutation review explicit.",
        ),
        _deferred(
            "git:hook-bypass",
            "git-deferred",
            "ADVANCED-CONFIG.md",
            "bypassGitHooks",
            "Hook bypass remains unsupported for safety.",
        ),
    )
    return CompatibilityInventory(items=items)


def _golden(name: str, upstream_reference: str, expected: str) -> CompatibilityItem:
    return CompatibilityItem(
        name=name,
        classification="golden-required",
        upstream_reference=upstream_reference,
        expected=expected,
        status="implemented",
        fixture=name,
    )


def _deferred(
    name: str,
    classification: str,
    upstream_reference: str,
    expected: str,
    deferred_reason: str,
) -> CompatibilityItem:
    return CompatibilityItem(
        name=name,
        classification=classification,
        upstream_reference=upstream_reference,
        expected=expected,
        status="deferred",
        deferred_reason=deferred_reason,
    )
