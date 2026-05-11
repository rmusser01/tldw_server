from __future__ import annotations

from typing import Any

from backlog_py.core.models import BacklogProject
from backlog_py.core.repository import MutableRepository, ReadOnlyRepository, TaskRecord


_MUTATION_NOT_IMPLEMENTED = "Task mutation MCP tools are not implemented until Task 7 for this argument shape."


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
    """Create a task through the safe mutation repository."""
    task_id = kwargs.get("task_id") or kwargs.get("id")
    repository = MutableRepository(project)
    task = repository.create_task(
        title=str(kwargs.get("title") or ""),
        task_id=None if task_id is None else str(task_id),
        status=_optional_string(kwargs.get("status")),
        description=str(kwargs.get("description") or ""),
        acceptance_criteria=_string_list(kwargs.get("acceptanceCriteria") or kwargs.get("acceptance_criteria")),
        definition_of_done=_string_list(kwargs.get("definitionOfDone") or kwargs.get("definition_of_done")),
        dependencies=_string_list(kwargs.get("dependencies")),
        on_status_change=_optional_bool(kwargs.get("onStatusChange") or kwargs.get("on_status_change")),
    )
    return _task_detail(project, task)


def task_edit(project: BacklogProject, task_id: str, **kwargs: Any) -> dict[str, Any]:
    """Edit supported task sections through the safe mutation repository."""
    if "title" in kwargs:
        raise NotImplementedError(_MUTATION_NOT_IMPLEMENTED)
    repository = MutableRepository(project)
    task = repository.edit_task(
        task_id,
        description=_optional_string(kwargs.get("description")),
        append_notes=_optional_string(kwargs.get("appendNotes") or kwargs.get("append_notes")),
        final_summary=_optional_string(kwargs.get("finalSummary") or kwargs.get("final_summary")),
        check_ac=_int_list(kwargs.get("checkAc") or kwargs.get("check_ac")),
        check_dod=_int_list(kwargs.get("checkDod") or kwargs.get("check_dod")),
        uncheck_ac=_int_list(kwargs.get("uncheckAc") or kwargs.get("uncheck_ac")),
        uncheck_dod=_int_list(kwargs.get("uncheckDod") or kwargs.get("uncheck_dod")),
        dependencies=_string_list(kwargs.get("dependencies")) if "dependencies" in kwargs else None,
        status=_optional_string(kwargs.get("status")),
        on_status_change=_optional_bool(kwargs.get("onStatusChange") or kwargs.get("on_status_change")),
    )
    return _task_detail(project, task)


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


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    return bool(value)


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def _int_list(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, int):
        return [value]
    return [int(item) for item in value]
