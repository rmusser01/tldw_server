"""Every backlog task file has its own id (TASK-13960)."""

from pathlib import Path

import pytest
import yaml
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


def test_a_file_holds_the_id_in_its_name_too(tmp_path: Path) -> None:
    # backlog-py reads ids from file names, so a file whose id: line says otherwise holds both ids.
    _task(tmp_path / "backlog/tasks/task-5 - A.md", "TASK-6")
    _task(tmp_path / "backlog/tasks/task-5 - B.md", "TASK-5")

    assert duplicate_task_ids(tmp_path) == {"TASK-5": ["backlog/tasks/task-5 - A.md", "backlog/tasks/task-5 - B.md"]}


def test_an_id_needs_no_id_line_and_ignores_case(tmp_path: Path) -> None:
    (tmp_path / "backlog/tasks").mkdir(parents=True)
    (tmp_path / "backlog/tasks/task-9 - A.md").write_text("---\ntitle: t\n---\n", encoding="utf-8")
    _task(tmp_path / "backlog/completed/task-90 - B.md", "task-9")

    assert duplicate_task_ids(tmp_path) == {
        "TASK-9": ["backlog/completed/task-90 - B.md", "backlog/tasks/task-9 - A.md"]
    }


def test_a_trailing_dot_is_not_part_of_the_id(tmp_path: Path) -> None:
    _task(tmp_path / "backlog/tasks/task-8 - A.md", "TASK-7.")
    _task(tmp_path / "backlog/tasks/task-7 - B.md", "TASK-7")

    assert list(duplicate_task_ids(tmp_path)) == ["TASK-7"]


def test_main_escapes_the_annotation(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    # GitHub's workflow commands: %, CR and LF in the message; also : and , in a property such as file=.
    _task(tmp_path / "backlog/tasks/task-3 - a,b:1%.md", "TASK-3")
    _task(tmp_path / "backlog/tasks/task-3 - B.md", "TASK-3")

    assert main([str(tmp_path)]) == 1
    assert capsys.readouterr().out.splitlines()[:2] == [
        "::error file=backlog/tasks/task-3 - B.md::TASK-3 is also the id of backlog/tasks/task-3 - a,b:1%25.md",
        "::error file=backlog/tasks/task-3 - a%2Cb%3A1%25.md::TASK-3 is also the id of backlog/tasks/task-3 - B.md",
    ]


def test_main_annotates_a_file_it_cannot_read(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "backlog/tasks/task-4 - A.md"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"---\nid: TASK-4\ntitle: \xff\n---\n")

    with pytest.raises(UnicodeDecodeError):
        main([str(tmp_path)])
    assert capsys.readouterr().out.startswith(f"::error file={path.as_posix()}::")


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


def test_the_repository_has_no_duplicate_ids() -> None:
    assert duplicate_task_ids(Path(__file__).resolve().parents[3]) == {}


def test_backend_required_runs_the_check_on_every_pr() -> None:
    path = Path(__file__).resolve().parents[3] / ".github/workflows/backend-required.yml"
    workflow = yaml.safe_load(path.read_text(encoding="utf-8"))
    # Every PR triggers it: no paths filter on the trigger. (YAML 1.1 reads the `on:` key as True.)
    assert {"paths", "paths-ignore"}.isdisjoint(workflow[True]["pull_request"])
    jobs = workflow["jobs"]
    checks = [step for step in jobs["changes"]["steps"]
              if "Helper_Scripts/ci/check_backlog_task_ids.py" in step.get("run", "")]
    # One step in the changes job, which every PR runs, with no condition of its own, and it blocks:
    # no continue-on-error on the step or the job, and nothing appended to the command (no `|| true`).
    assert [(step.get("if"), step.get("continue-on-error"), step["run"].strip()) for step in checks] == [
        (None, None, "python3 Helper_Scripts/ci/check_backlog_task_ids.py .")
    ]
    assert jobs["changes"].get("continue-on-error") is None
    # These tests run in a required step too, so the lines above are enforced, not decoration.
    contracts = next(step for step in jobs["backend-required"]["steps"]
                     if step.get("name") == "Enforce CI contracts and code ratchets")
    assert "tldw_Server_API/tests/CI/test_check_backlog_task_ids.py" in contracts["run"]
    assert "tldw_Server_API/tests/Helper_Scripts/test_renumber_duplicate_task_ids.py" in contracts["run"]
