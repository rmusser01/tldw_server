"""Pure MCP registry helpers for Backlog.py compatibility."""

from backlog_py.mcp.resources import read_resource
from backlog_py.mcp.tools import (
    definition_of_done_defaults_get,
    definition_of_done_defaults_upsert,
    document_create,
    document_list,
    document_update,
    document_view,
    milestone_add,
    milestone_archive,
    milestone_list,
    milestone_remove,
    milestone_rename,
    task_create,
    task_edit,
    task_search,
    task_view,
)

__all__ = [
    "definition_of_done_defaults_get",
    "definition_of_done_defaults_upsert",
    "document_create",
    "document_list",
    "document_update",
    "document_view",
    "milestone_add",
    "milestone_archive",
    "milestone_list",
    "milestone_remove",
    "milestone_rename",
    "read_resource",
    "task_create",
    "task_edit",
    "task_search",
    "task_view",
]
