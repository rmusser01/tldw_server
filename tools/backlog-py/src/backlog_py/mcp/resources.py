from __future__ import annotations


WORKFLOW_OVERVIEW_RESOURCE = """# Backlog.md Python MCP Workflow

This compatibility layer exposes Backlog.md read helpers and safe mutation
helpers implemented in Python. It does not shell out to the Node.js Backlog.md
CLI.

Supported resources:
- backlog://workflow/overview
- backlog://docs/task-workflow

Supported tools:
- task_search(project, query, limit=10)
- task_view(project, task_id)
- task_create(project, **kwargs)
- task_edit(project, task_id, **kwargs)
- document_list(project, query=None, limit=100)
- document_view(project, path_or_id)
- document_create(project, **kwargs)
- document_update(project, path_or_id, **kwargs)
- milestone_list(project)
- milestone_add(project, name, description="")
- milestone_rename(project, old_name, new_name, update_tasks=False)
- milestone_remove(project, name, clear_tasks=False)
- milestone_archive(project, name)
- definition_of_done_defaults_get(project)
- definition_of_done_defaults_upsert(project, items)

All write-capable helpers must use the safe core services, path-containment
checks, and atomic file writes. Do not use this registry for shell execution.
"""

_RESOURCE_ALIASES = {
    "backlog://workflow/overview": "backlog://workflow/overview",
    "backlog://docs/task-workflow": "backlog://workflow/overview",
}

_RESOURCES = {
    "backlog://workflow/overview": WORKFLOW_OVERVIEW_RESOURCE,
}


def read_resource(uri: str) -> str:
    """Return static read-only MCP resource content for a supported URI."""
    canonical_uri = _RESOURCE_ALIASES.get(uri)
    if canonical_uri is None:
        raise KeyError(f"Unsupported Backlog MCP resource: {uri}")
    return _RESOURCES[canonical_uri]
