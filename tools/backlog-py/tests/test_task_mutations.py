from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from click.testing import CliRunner

from backlog_py.cli.main import main
from backlog_py.core.repository import MutableRepository, TaskMutationError
from backlog_py.mcp.tools import task_create, task_edit
from backlog_py.storage.project import discover_project


FIXTURE_REPO = Path(__file__).parent / "fixtures" / "repos" / "basic"


def _copy_fixture(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    shutil.copytree(FIXTURE_REPO, repo)
    return repo


def _repository(repo: Path) -> MutableRepository:
    return MutableRepository.from_path(repo)


def _project(repo: Path):
    return discover_project(Path.cwd(), explicit_cwd=repo)


def _task_file(repo: Path, task_id: str = "task-1") -> Path:
    matches = sorted((repo / "backlog" / "tasks").glob(f"{task_id} -*.md"))
    assert len(matches) == 1
    return matches[0]


def _snapshot_tasks(repo: Path) -> dict[Path, str]:
    task_dir = repo / "backlog" / "tasks"
    return {
        path.relative_to(task_dir): path.read_text(encoding="utf-8")
        for path in sorted(task_dir.glob("*.md"))
    }


def test_create_task_writes_valid_task_in_fixture_repo(tmp_path):
    repo = _copy_fixture(tmp_path)

    task = _repository(repo).create_task(
        title="New safe mutation task",
        task_id="TASK-2",
        description="Created through the safe mutation core.",
        acceptance_criteria=["Task can be viewed"],
        definition_of_done=["Tests pass"],
    )

    assert task.id == "TASK-2"
    assert task.title == "New safe mutation task"
    written = _task_file(repo, "task-2").read_text(encoding="utf-8")
    assert "id: TASK-2" in written
    assert "Created through the safe mutation core." in written
    assert "- [ ] #1 Task can be viewed" in written
    assert "- [ ] #1 Tests pass" in written
    assert _repository(repo).get_task("TASK-2").description == "Created through the safe mutation core."


def test_edit_task_updates_owned_sections_and_checklists_without_rewriting_unowned_body(tmp_path):
    repo = _copy_fixture(tmp_path)
    task_path = _task_file(repo)
    before = task_path.read_text(encoding="utf-8")

    edited = _repository(repo).edit_task(
        "TASK-1",
        description="Edited description.",
        append_notes="- Added implementation note.",
        final_summary="Finalized through safe edit.",
        check_ac=[2],
        check_dod=[2],
    )

    after = task_path.read_text(encoding="utf-8")
    assert edited.description == "Edited description."
    assert "Unowned body content before acceptance criteria must be preserved." in after
    assert "Trailing unowned body content must also round trip." in after
    assert "custom_field: preserve-me" in after
    assert "- Keep unknown body text stable." in after
    assert "- Added implementation note." in after
    assert "Finalized through safe edit." in after
    assert "- [x] #2 Preserve incomplete acceptance criteria raw line" in after
    assert "- [x] #2 Verification recorded" in after
    assert before != after


def test_edit_task_can_uncheck_acceptance_criteria_and_definition_of_done(tmp_path):
    repo = _copy_fixture(tmp_path)

    _repository(repo).edit_task("TASK-1", uncheck_ac=[1], uncheck_dod=[1])

    after = _task_file(repo).read_text(encoding="utf-8")
    assert "- [ ] #1 Preserve completed acceptance criteria raw line" in after
    assert "- [ ] #1 Tests written" in after


def test_invalid_checklist_index_is_rejected_before_write(tmp_path):
    repo = _copy_fixture(tmp_path)
    before = _snapshot_tasks(repo)

    with pytest.raises(TaskMutationError, match="AC checklist index 99"):
        _repository(repo).edit_task("TASK-1", check_ac=[99])

    assert _snapshot_tasks(repo) == before


def test_invalid_dod_checklist_index_is_rejected_before_write(tmp_path):
    repo = _copy_fixture(tmp_path)
    before = _snapshot_tasks(repo)

    with pytest.raises(TaskMutationError, match="DOD checklist index 99"):
        _repository(repo).edit_task("TASK-1", check_dod=[99])

    assert _snapshot_tasks(repo) == before


def test_duplicate_task_id_is_rejected_before_write(tmp_path):
    repo = _copy_fixture(tmp_path)
    before = _snapshot_tasks(repo)

    with pytest.raises(TaskMutationError, match="already exists"):
        _repository(repo).create_task(title="Duplicate", task_id="TASK-1")

    assert _snapshot_tasks(repo) == before


def test_circular_dependencies_are_rejected_before_write(tmp_path):
    repo = _copy_fixture(tmp_path)
    repository = _repository(repo)
    repository.create_task(title="Child", task_id="TASK-2", dependencies=["TASK-1"])
    before = _snapshot_tasks(repo)

    with pytest.raises(TaskMutationError, match="Circular dependency"):
        repository.edit_task("TASK-1", dependencies=["TASK-2"])

    assert _snapshot_tasks(repo) == before


def test_nonexistent_dependencies_are_rejected_before_write(tmp_path):
    repo = _copy_fixture(tmp_path)
    before = _snapshot_tasks(repo)

    with pytest.raises(TaskMutationError, match="Dependency not found: TASK-99"):
        _repository(repo).create_task(title="Missing dependency", task_id="TASK-2", dependencies=["TASK-99"])

    assert _snapshot_tasks(repo) == before


def test_edit_nonexistent_dependencies_are_rejected_before_write(tmp_path):
    repo = _copy_fixture(tmp_path)
    before = _snapshot_tasks(repo)

    with pytest.raises(TaskMutationError, match="Dependency not found: TASK-99"):
        _repository(repo).edit_task("TASK-1", dependencies=["TASK-99"])

    assert _snapshot_tasks(repo) == before


def test_unknown_status_is_rejected_before_write(tmp_path):
    repo = _copy_fixture(tmp_path)
    before = _snapshot_tasks(repo)

    with pytest.raises(TaskMutationError, match="Unknown status: Mystery"):
        _repository(repo).create_task(title="Bad status", task_id="TASK-2", status="Mystery")

    assert _snapshot_tasks(repo) == before


def test_edit_unknown_status_is_rejected_before_write(tmp_path):
    repo = _copy_fixture(tmp_path)
    before = _snapshot_tasks(repo)

    with pytest.raises(TaskMutationError, match="Unknown status: Mystery"):
        _repository(repo).edit_task("TASK-1", status="Mystery")

    assert _snapshot_tasks(repo) == before


def test_path_traversal_task_id_is_rejected_without_partial_file(tmp_path):
    repo = _copy_fixture(tmp_path)
    before = _snapshot_tasks(repo)

    with pytest.raises(TaskMutationError, match="Invalid task id"):
        _repository(repo).create_task(title="Escape", task_id="../TASK-9")

    assert _snapshot_tasks(repo) == before
    assert not (repo / "backlog" / "TASK-9.md").exists()


def test_symlinked_task_directory_escape_is_rejected_without_outside_write(tmp_path):
    repo = _copy_fixture(tmp_path)
    task_dir = repo / "backlog" / "tasks"
    outside = tmp_path / "outside"
    outside.mkdir()
    shutil.rmtree(task_dir)
    try:
        task_dir.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")

    with pytest.raises(TaskMutationError, match="outside allowed base"):
        _repository(repo).create_task(title="Escaped", task_id="TASK-2")

    assert list(outside.iterdir()) == []


def test_symlinked_task_file_escape_is_rejected_without_outside_write(tmp_path):
    repo = _copy_fixture(tmp_path)
    task_path = _task_file(repo)
    outside = tmp_path / "outside"
    outside.mkdir()
    outside_task = outside / task_path.name
    original = task_path.read_text(encoding="utf-8")
    outside_task.write_text(original, encoding="utf-8")
    task_path.unlink()
    try:
        task_path.symlink_to(outside_task)
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")

    with pytest.raises(TaskMutationError, match="outside allowed base"):
        _repository(repo).edit_task("TASK-1", description="Escaped edit.")

    assert outside_task.read_text(encoding="utf-8") == original


def test_on_status_change_is_disabled_by_default(tmp_path):
    repo = _copy_fixture(tmp_path)

    with pytest.raises(TaskMutationError, match="onStatusChange is disabled"):
        _repository(repo).edit_task("TASK-1", status="Done", on_status_change=True)


def test_cli_task_create_and_edit_use_safe_core(tmp_path):
    repo = _copy_fixture(tmp_path)
    runner = CliRunner()

    create = runner.invoke(
        main,
        [
            "--cwd",
            str(repo),
            "task",
            "create",
            "CLI mutation task",
            "--id",
            "TASK-2",
            "--description",
            "Created from CLI.",
            "--plain",
        ],
    )
    assert create.exit_code == 0
    assert "TASK-2 [To Do] CLI mutation task" in create.output

    edit = runner.invoke(
        main,
        [
            "--cwd",
            str(repo),
            "task",
            "edit",
            "TASK-2",
            "--append-notes",
            "- CLI note.",
            "--final-summary",
            "CLI final summary.",
            "--plain",
        ],
    )
    assert edit.exit_code == 0
    assert "TASK-2 [To Do] CLI mutation task" in edit.output
    written = _task_file(repo, "task-2").read_text(encoding="utf-8")
    assert "- CLI note." in written
    assert "CLI final summary." in written

    uncheck = runner.invoke(
        main,
        [
            "--cwd",
            str(repo),
            "task",
            "edit",
            "TASK-1",
            "--uncheck-ac",
            "1",
            "--uncheck-dod",
            "1",
            "--plain",
        ],
    )
    assert uncheck.exit_code == 0
    task_one = _task_file(repo).read_text(encoding="utf-8")
    assert "- [ ] #1 Preserve completed acceptance criteria raw line" in task_one
    assert "- [ ] #1 Tests written" in task_one


def test_mcp_task_create_and_edit_use_safe_core(tmp_path):
    repo = _copy_fixture(tmp_path)
    project = _project(repo)

    created = task_create(
        project,
        title="MCP mutation task",
        description="Created from MCP.",
        acceptanceCriteria=["MCP create works"],
    )
    assert created["id"] == "TASK-2"
    assert created["description"] == "Created from MCP."

    edited = task_edit(
        project,
        task_id="TASK-2",
        appendNotes="- MCP note.",
        finalSummary="MCP final summary.",
        checkAc=[1],
    )
    assert edited["id"] == "TASK-2"

    unchecked = task_edit(project, task_id="TASK-2", uncheckAc=[1])
    assert unchecked["id"] == "TASK-2"
    written = _task_file(repo, "task-2").read_text(encoding="utf-8")
    assert "- MCP note." in written
    assert "MCP final summary." in written
    assert "- [ ] #1 MCP create works" in written
