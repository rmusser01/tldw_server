from __future__ import annotations

from typing import Any

from backlog_py.core.models import BacklogProject
from backlog_py.core.documents import DocumentRecord, DocumentService
from backlog_py.core.milestones import MilestoneRecord, MilestoneService
from backlog_py.core.repository import MutableRepository, ReadOnlyRepository, TaskRecord
from backlog_py.storage.config import get_definition_of_done_defaults, load_config, replace_definition_of_done_defaults


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
    task_id = _get_alias(kwargs, "task_id", "id")
    repository = MutableRepository(_fresh_project(project))
    task = repository.create_task(
        title=str(kwargs.get("title") or ""),
        task_id=None if task_id is None else str(task_id),
        status=_optional_string(_get_alias(kwargs, "status")),
        description=str(kwargs.get("description") or ""),
        acceptance_criteria=_optional_string_list(_get_alias(kwargs, "acceptanceCriteria", "acceptance_criteria")),
        definition_of_done=_optional_string_list(_get_alias(kwargs, "definitionOfDone", "definition_of_done")),
        definition_of_done_add=_optional_string_list(
            _get_alias(kwargs, "definitionOfDoneAdd", "definition_of_done_add")
        ),
        disable_definition_of_done_defaults=bool(
            _get_alias(kwargs, "disableDefinitionOfDoneDefaults", "disable_definition_of_done_defaults") or False
        ),
        dependencies=_optional_string_list(_get_alias(kwargs, "dependencies")),
        on_status_change=_optional_bool(_get_alias(kwargs, "onStatusChange", "on_status_change")),
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
        append_notes=_optional_string(_get_alias(kwargs, "appendNotes", "append_notes")),
        final_summary=_optional_string(_get_alias(kwargs, "finalSummary", "final_summary")),
        check_ac=_int_list(_get_alias(kwargs, "checkAc", "check_ac")),
        check_dod=_int_list(_get_alias(kwargs, "checkDod", "check_dod")),
        uncheck_ac=_int_list(_get_alias(kwargs, "uncheckAc", "uncheck_ac")),
        uncheck_dod=_int_list(_get_alias(kwargs, "uncheckDod", "uncheck_dod")),
        dependencies=_string_list(kwargs.get("dependencies")) if "dependencies" in kwargs else None,
        status=_optional_string(kwargs.get("status")),
        on_status_change=_optional_bool(_get_alias(kwargs, "onStatusChange", "on_status_change")),
    )
    return _task_detail(project, task)


def document_list(project: BacklogProject, query: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
    """List or search documents through the safe document service."""
    if limit <= 0:
        return []
    service = DocumentService(project)
    documents = service.list_documents() if query is None else service.search_documents(query)
    return [_document_detail(project, document) for document in documents[:limit]]


def document_view(project: BacklogProject, path_or_id: str) -> dict[str, Any]:
    """Return one document by docs-relative path or frontmatter id."""
    return _document_detail(project, DocumentService(project).view_document(path_or_id))


def document_create(project: BacklogProject, **kwargs: Any) -> dict[str, Any]:
    """Create a document under backlog/docs."""
    document = DocumentService(project).create_document(
        str(kwargs.get("path") or ""),
        title=str(kwargs.get("title") or ""),
        content=str(kwargs.get("content") or ""),
        metadata=_dict_value(kwargs.get("metadata")),
    )
    return _document_detail(project, document)


def document_update(project: BacklogProject, path_or_id: str, **kwargs: Any) -> dict[str, Any]:
    """Update a document while preserving omitted metadata."""
    document = DocumentService(project).update_document(
        path_or_id,
        title=_optional_string(kwargs.get("title")),
        content=_optional_string(kwargs.get("content")),
    )
    return _document_detail(project, document)


def milestone_list(project: BacklogProject) -> list[dict[str, Any]]:
    """List active milestone files."""
    return [_milestone_detail(project, milestone) for milestone in MilestoneService(project).list_milestones()]


def milestone_add(project: BacklogProject, name: str, description: str = "") -> dict[str, Any]:
    """Create a milestone file."""
    return _milestone_detail(project, MilestoneService(project).add_milestone(name, description=description))


def milestone_rename(
    project: BacklogProject,
    old_name: str,
    new_name: str,
    update_tasks: bool = False,
) -> dict[str, Any]:
    """Rename a milestone file and optionally update task references."""
    milestone = MilestoneService(project).rename_milestone(old_name, new_name, update_tasks=update_tasks)
    return _milestone_detail(project, milestone)


def milestone_remove(project: BacklogProject, name: str, clear_tasks: bool = False) -> dict[str, Any]:
    """Remove a milestone file and optionally clear task references."""
    return _milestone_detail(project, MilestoneService(project).remove_milestone(name, clear_tasks=clear_tasks))


def milestone_archive(project: BacklogProject, name: str) -> dict[str, Any]:
    """Archive a milestone file."""
    return _milestone_detail(project, MilestoneService(project).archive_milestone(name))


def definition_of_done_defaults_get(project: BacklogProject) -> dict[str, list[str]]:
    """Return project-level Definition of Done default checklist items."""
    return {"items": get_definition_of_done_defaults(project)}


def definition_of_done_defaults_upsert(project: BacklogProject, items: list[str]) -> dict[str, list[str]]:
    """Replace project-level Definition of Done default checklist items."""
    return {"items": replace_definition_of_done_defaults(project, items)}


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


def _document_detail(project: BacklogProject, document: DocumentRecord) -> dict[str, Any]:
    return {
        "id": document.id,
        "title": document.title,
        "path": document.path_relative,
        "content": document.content,
        "frontmatter": dict(document.frontmatter),
        "raw_source": document.raw_source,
        "project_path": document.path.relative_to(project.root).as_posix(),
    }


def _milestone_detail(project: BacklogProject, milestone: MilestoneRecord) -> dict[str, Any]:
    return {
        "name": milestone.name,
        "path": milestone.path_relative,
        "content": milestone.content,
        "frontmatter": dict(milestone.frontmatter),
        "archived": milestone.archived,
        "project_path": milestone.path.relative_to(project.root).as_posix(),
    }


def _relative_task_path(project: BacklogProject, task: TaskRecord) -> str:
    return task.path.relative_to(project.root).as_posix()


def _fresh_project(project: BacklogProject) -> BacklogProject:
    return BacklogProject(
        root=project.root,
        backlog_dir=project.backlog_dir,
        config_path=project.config_path,
        config=load_config(project.config_path),
    )


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


def _optional_string_list(value: Any) -> list[str] | None:
    if value is None:
        return None
    return _string_list(value)


def _int_list(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, int):
        return [value]
    return [int(item) for item in value]


def _dict_value(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise TypeError("Expected mapping")
    return dict(value)


def _get_alias(mapping: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in mapping:
            return mapping[name]
    return None
