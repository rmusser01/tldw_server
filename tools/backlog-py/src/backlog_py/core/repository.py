from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from backlog_py.core.models import BacklogProject, ParsedTaskMarkdown
from backlog_py.markdown.task_parser import parse_task_markdown
from backlog_py.search.simple import contains_query
from backlog_py.storage.project import discover_project


@dataclass(frozen=True)
class TaskRecord:
    id: str
    title: str
    status: str
    path: Path
    parsed: ParsedTaskMarkdown

    @property
    def description(self) -> str:
        section = self.parsed.sections.get("DESCRIPTION")
        return "" if section is None else section.content.strip()

    @property
    def body(self) -> str:
        return self.parsed.body

    @property
    def raw_source(self) -> str:
        return self.parsed.raw_source


class ReadOnlyRepository:
    def __init__(self, project: BacklogProject) -> None:
        self.project = project

    @classmethod
    def from_path(cls, cwd: Path) -> "ReadOnlyRepository":
        return cls(discover_project(Path.cwd(), explicit_cwd=cwd))

    def list_tasks(self) -> list[TaskRecord]:
        return sorted(self._load_tasks(), key=lambda task: (_task_sort_key(task.id), task.title))

    def get_task(self, task_id: str) -> TaskRecord:
        normalized_id = task_id.casefold()
        for task in self.list_tasks():
            if task.id.casefold() == normalized_id:
                return task
        raise KeyError(f"Task not found: {task_id}")

    def search_tasks(self, query: str) -> list[TaskRecord]:
        return [
            task
            for task in self.list_tasks()
            if contains_query(_search_text(task), query)
        ]

    def board(self) -> "OrderedDict[str, list[TaskRecord]]":
        statuses = self.project.config.statuses or _statuses_from_tasks(self.list_tasks())
        board: OrderedDict[str, list[TaskRecord]] = OrderedDict((status, []) for status in statuses)
        for task in self.list_tasks():
            board.setdefault(task.status, []).append(task)
        return board

    def _load_tasks(self) -> list[TaskRecord]:
        task_dir = self.project.backlog_dir / "tasks"
        if not task_dir.is_dir():
            return []
        return [_load_task(path) for path in sorted(task_dir.glob("*.md"))]


def _load_task(path: Path) -> TaskRecord:
    parsed = parse_task_markdown(path.read_text(encoding="utf-8"))
    frontmatter = parsed.frontmatter
    task_id = str(frontmatter.get("id") or _id_from_filename(path))
    return TaskRecord(
        id=task_id,
        title=str(frontmatter.get("title") or ""),
        status=str(frontmatter.get("status") or "To Do"),
        path=path,
        parsed=parsed,
    )


def _id_from_filename(path: Path) -> str:
    stem = path.stem
    if " - " in stem:
        return stem.split(" - ", 1)[0].upper()
    return stem.upper()


def _search_text(task: TaskRecord) -> str:
    return "\n".join([task.id, task.title, task.status, task.raw_source])


def _statuses_from_tasks(tasks: Iterable[TaskRecord]) -> list[str]:
    statuses: list[str] = []
    for task in tasks:
        if task.status not in statuses:
            statuses.append(task.status)
    return statuses


def _task_sort_key(task_id: str) -> tuple[str, tuple[tuple[int, int | str], ...]]:
    prefix, _, suffix = task_id.partition("-")
    return prefix, tuple(_sort_segment(segment) for segment in suffix.replace(".", "-").split("-"))


def _sort_segment(segment: str) -> tuple[int, int | str]:
    if segment.isdigit():
        return 0, int(segment)
    return 1, segment
