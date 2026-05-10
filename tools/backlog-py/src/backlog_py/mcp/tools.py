from __future__ import annotations

from typing import Any

from backlog_py.core.models import BacklogProject
from backlog_py.core.repository import ReadOnlyRepository, TaskRecord


_MUTATION_NOT_IMPLEMENTED = (
    "Task mutation MCP tools are not implemented until Task 7; "
    "this Task 6 registry is read-only."
)


def task_search(project: BacklogProject, query: str, limit: int = 10) -> list[dict[str, Any]]:
    """Search tasks through the read-only repository and return JSON-safe rows."""
    if limit <= 0:
        return []
    repository = ReadOnlyRepository(project)
    return [_task_summary(project, task) for task in repository.search_tasks(query)[:limit]]


def task_view(project: BacklogProject, task_id: str) -> dict[str, Any]:
    """Return one task through the read-only repository as a JSON-safe mapping."""
    repository = ReadOnlyRepository(project)
    return _task_detail(project, repository.get_task(task_id))


def task_create(project: BacklogProject, **kwargs: Any) -> dict[str, Any]:
    """Placeholder for future MCP task creation support."""
    raise NotImplementedError(_MUTATION_NOT_IMPLEMENTED)


def task_edit(project: BacklogProject, task_id: str, **kwargs: Any) -> dict[str, Any]:
    """Placeholder for future MCP task editing support."""
    raise NotImplementedError(_MUTATION_NOT_IMPLEMENTED)


def _task_summary(project: BacklogProject, task: TaskRecord) -> dict[str, Any]:
    return {
        "id": task.id,
        "title": task.title,
        "status": task.status,
        "description": task.description,
        "path": _relative_task_path(project, task),
    }


def _task_detail(project: BacklogProject, task: TaskRecord) -> dict[str, Any]:
    detail = _task_summary(project, task)
    detail["raw_source"] = task.raw_source
    return detail


def _relative_task_path(project: BacklogProject, task: TaskRecord) -> str:
    return task.path.relative_to(project.root).as_posix()
