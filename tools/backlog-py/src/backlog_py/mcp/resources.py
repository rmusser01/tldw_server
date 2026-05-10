from __future__ import annotations


WORKFLOW_OVERVIEW_RESOURCE = """# Backlog.md Python MCP Workflow

This compatibility layer currently exposes read-only Backlog.md MCP behavior.

Supported resources:
- backlog://workflow/overview
- backlog://docs/task-workflow

Supported tools:
- task_search(project, query, limit=10)
- task_view(project, task_id)

Mutation tools are intentionally unavailable until Task 7 adds the safe
mutation core. Do not use this registry for shell execution or live writes.
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
