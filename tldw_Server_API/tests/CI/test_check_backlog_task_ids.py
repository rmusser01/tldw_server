"""Every backlog task file has its own id (TASK-13960)."""

from pathlib import Path

import pytest
from Helper_Scripts.ci.check_backlog_task_ids import duplicate_task_ids, main

pytestmark = pytest.mark.unit


def _task(path: Path, task_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"---\nid: {task_id}\ntitle: t\nstatus: Done\n---\n", encoding="utf-8")


def test_reports_ids_shared_across_tasks_completed_and_archive(tmp_path: Path) -> None:
    _task(tmp_path / "backlog/tasks/task-1 - A.md", "TASK-1")
    _task(tmp_path / "backlog/completed/task-1 - B.md", "TASK-1")
    _task(tmp_path / "backlog/archive/tasks/task-2 - C.md", "TASK-2")
    _task(tmp_path / "backlog/tasks/task-2.1 - D.md", "TASK-2.1")

    assert duplicate_task_ids(tmp_path) == {
        "TASK-1": ["backlog/completed/task-1 - B.md", "backlog/tasks/task-1 - A.md"]
    }


def test_main_annotates_each_file_and_points_to_the_fix(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _task(tmp_path / "backlog/tasks/task-3 - A.md", "TASK-3")
    _task(tmp_path / "backlog/tasks/task-3 - B.md", "TASK-3")

    assert main([str(tmp_path)]) == 1
    out = capsys.readouterr().out
    assert "::error file=backlog/tasks/task-3 - A.md::TASK-3 is also the id of backlog/tasks/task-3 - B.md" in out
    assert "AGENTS.md" in out and "task renumber" in out


def test_main_passes_when_every_id_is_unique(tmp_path: Path) -> None:
    _task(tmp_path / "backlog/tasks/task-4 - A.md", "TASK-4")
    assert main([str(tmp_path)]) == 0


def test_main_fails_when_no_task_directory_exists(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    # A moved backlog/ or a sparse checkout must not pass as "no duplicates".
    assert main([str(tmp_path)]) == 2
    assert capsys.readouterr().out.startswith("::error::")
