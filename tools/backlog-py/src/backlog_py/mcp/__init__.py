"""Pure MCP registry helpers for Backlog.py compatibility."""

from backlog_py.mcp.resources import read_resource
from backlog_py.mcp.tools import task_create, task_edit, task_search, task_view

__all__ = [
    "read_resource",
    "task_create",
    "task_edit",
    "task_search",
    "task_view",
]
