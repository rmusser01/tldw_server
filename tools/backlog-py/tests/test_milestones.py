from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from click.testing import CliRunner

import backlog_py.core.milestones as milestones_module
from backlog_py.cli.main import main
from backlog_py.core.milestones import MilestoneMutationError, MilestoneService
from backlog_py.mcp.tools import (
    milestone_add,
    milestone_archive,
    milestone_list,
    milestone_remove,
    milestone_rename,
)
from backlog_py.storage.project import discover_project


FIXTURE_REPO = Path(__file__).parent / "fixtures" / "repos" / "basic"


def _copy_fixture(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    shutil.copytree(FIXTURE_REPO, repo)
    return repo


def _project(repo: Path):
    return discover_project(Path.cwd(), explicit_cwd=repo)


def _service(repo: Path) -> MilestoneService:
    return MilestoneService(_project(repo))


def _task_path(repo: Path, task_id: str = "task-1") -> Path:
    matches = sorted((repo / "backlog" / "tasks").glob(f"{task_id} -*.md"))
    assert len(matches) == 1
    return matches[0]


def _set_task_milestone(repo: Path, milestone: str, task_id: str = "task-1") -> None:
    path = _task_path(repo, task_id=task_id)
    source = path.read_text(encoding="utf-8")
    path.write_text(source.replace("status: In Progress\n", f"status: In Progress\nmilestone: {milestone}\n"), encoding="utf-8")


def _create_task_with_milestone(repo: Path, task_id: str, title: str, milestone: str) -> Path:
    source = _task_path(repo).read_text(encoding="utf-8")
    source = source.replace("id: TASK-1\n", f"id: {task_id.upper()}\n")
    source = source.replace("title: Example task\n", f"title: {title}\n")
    source = source.replace("status: In Progress\n", f"status: In Progress\nmilestone: {milestone}\n")
    path = repo / "backlog" / "tasks" / f"{task_id.lower()} - {title.replace(' ', '-')}.md"
    path.write_text(source, encoding="utf-8")
    return path


def test_add_list_rename_remove_and_archive_milestones(tmp_path):
    repo = _copy_fixture(tmp_path)
    service = _service(repo)

    added = service.add_milestone("Alpha", description="First release")
    assert added.name == "Alpha"
    assert "First release" in added.content
    assert [milestone.name for milestone in service.list_milestones()] == ["Alpha"]

    renamed = service.rename_milestone("Alpha", "Beta")
    assert renamed.name == "Beta"
    assert not (repo / "backlog" / "milestones" / "Alpha.md").exists()
    assert (repo / "backlog" / "milestones" / "Beta.md").exists()

    service.remove_milestone("Beta")
    assert service.list_milestones() == []

    service.add_milestone("Release 1", description="Ship it")
    archived = service.archive_milestone("Release 1")
    assert archived.archived is True
    assert not (repo / "backlog" / "milestones" / "Release-1.md").exists()
    assert (repo / "backlog" / "archive" / "milestones" / "Release-1.md").exists()


def test_rename_and_remove_can_update_task_milestone_references(tmp_path):
    repo = _copy_fixture(tmp_path)
    service = _service(repo)
    service.add_milestone("Alpha")
    _set_task_milestone(repo, "Alpha")

    service.rename_milestone("Alpha", "Beta", update_tasks=True)
    assert "milestone: Beta" in _task_path(repo).read_text(encoding="utf-8")

    service.remove_milestone("Beta", clear_tasks=True)
    source = _task_path(repo).read_text(encoding="utf-8")
    assert "milestone:" not in source


def test_rename_and_remove_update_task_refs_when_lookup_uses_different_case(tmp_path):
    repo = _copy_fixture(tmp_path)
    service = _service(repo)
    service.add_milestone("Alpha")
    _set_task_milestone(repo, "Alpha")

    service.rename_milestone("alpha", "Beta", update_tasks=True)
    assert "milestone: Beta" in _task_path(repo).read_text(encoding="utf-8")

    service.remove_milestone("beta", clear_tasks=True)
    assert "milestone:" not in _task_path(repo).read_text(encoding="utf-8")


def test_rename_same_slug_milestone_updates_display_name_and_task_refs(tmp_path):
    repo = _copy_fixture(tmp_path)
    service = _service(repo)
    service.add_milestone("Release 1")
    _set_task_milestone(repo, "Release 1")

    renamed = service.rename_milestone("Release 1", "Release-1", update_tasks=True)

    assert renamed.name == "Release-1"
    assert renamed.path == repo / "backlog" / "milestones" / "Release-1.md"
    assert [path.name for path in sorted((repo / "backlog" / "milestones").glob("*.md"))] == ["Release-1.md"]
    assert "milestone: Release-1" in _task_path(repo).read_text(encoding="utf-8")


def test_rename_with_task_reference_symlink_escape_is_rejected_before_milestone_changes(tmp_path):
    repo = _copy_fixture(tmp_path)
    service = _service(repo)
    service.add_milestone("Alpha")
    _set_task_milestone(repo, "Alpha")
    task_path = _task_path(repo)
    original_task_source = task_path.read_text(encoding="utf-8")
    outside = tmp_path / "outside"
    outside.mkdir()
    outside_task = outside / task_path.name
    outside_task.write_text(original_task_source, encoding="utf-8")
    task_path.unlink()
    try:
        task_path.symlink_to(outside_task)
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")

    with pytest.raises(MilestoneMutationError, match="outside allowed base"):
        service.rename_milestone("Alpha", "Beta", update_tasks=True)

    assert (repo / "backlog" / "milestones" / "Alpha.md").exists()
    assert not (repo / "backlog" / "milestones" / "Beta.md").exists()
    assert outside_task.read_text(encoding="utf-8") == original_task_source


def test_same_slug_rename_rolls_back_original_milestone_when_task_write_fails(tmp_path, monkeypatch):
    repo = _copy_fixture(tmp_path)
    service = _service(repo)
    service.add_milestone("Release 1")
    _set_task_milestone(repo, "Release 1")
    milestone_path = repo / "backlog" / "milestones" / "Release-1.md"
    original_milestone_source = milestone_path.read_text(encoding="utf-8")
    original_task_source = _task_path(repo).read_text(encoding="utf-8")
    original_writer = milestones_module._atomic_write_text

    def fail_on_task(path: Path, source: str) -> None:
        if path.name.startswith("task-1"):
            raise OSError("simulated task write failure")
        original_writer(path, source)

    monkeypatch.setattr(milestones_module, "_atomic_write_text", fail_on_task)

    with pytest.raises(OSError, match="simulated task write failure"):
        service.rename_milestone("Release 1", "Release-1", update_tasks=True)

    assert milestone_path.read_text(encoding="utf-8") == original_milestone_source
    assert _task_path(repo).read_text(encoding="utf-8") == original_task_source


def test_list_milestones_rejects_symlinked_file_escape_before_read(tmp_path):
    repo = _copy_fixture(tmp_path)
    service = _service(repo)
    service.add_milestone("Alpha")
    milestone_path = repo / "backlog" / "milestones" / "Alpha.md"
    outside = tmp_path / "outside.md"
    outside.write_text("---\nname: Outside\n---\n\nSecret\n", encoding="utf-8")
    milestone_path.unlink()
    try:
        milestone_path.symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")

    with pytest.raises(MilestoneMutationError, match="outside allowed base"):
        service.list_milestones()


def test_rename_rolls_back_milestone_and_task_refs_when_task_write_fails(tmp_path, monkeypatch):
    repo = _copy_fixture(tmp_path)
    service = _service(repo)
    service.add_milestone("Alpha")
    _set_task_milestone(repo, "Alpha")
    second_task = _create_task_with_milestone(repo, "TASK-2", "Second task", "Alpha")
    first_task = _task_path(repo)
    original_first_source = first_task.read_text(encoding="utf-8")
    original_second_source = second_task.read_text(encoding="utf-8")
    original_writer = milestones_module._atomic_write_text

    def fail_on_second_task(path: Path, source: str) -> None:
        if path.name.startswith("task-2"):
            raise OSError("simulated task write failure")
        original_writer(path, source)

    monkeypatch.setattr(milestones_module, "_atomic_write_text", fail_on_second_task)

    with pytest.raises(OSError, match="simulated task write failure"):
        service.rename_milestone("Alpha", "Beta", update_tasks=True)

    assert (repo / "backlog" / "milestones" / "Alpha.md").exists()
    assert not (repo / "backlog" / "milestones" / "Beta.md").exists()
    assert first_task.read_text(encoding="utf-8") == original_first_source
    assert second_task.read_text(encoding="utf-8") == original_second_source


def test_remove_rolls_back_task_refs_when_task_write_fails(tmp_path, monkeypatch):
    repo = _copy_fixture(tmp_path)
    service = _service(repo)
    service.add_milestone("Alpha")
    _set_task_milestone(repo, "Alpha")
    second_task = _create_task_with_milestone(repo, "TASK-2", "Second task", "Alpha")
    first_task = _task_path(repo)
    original_first_source = first_task.read_text(encoding="utf-8")
    original_second_source = second_task.read_text(encoding="utf-8")
    original_writer = milestones_module._atomic_write_text

    def fail_on_second_task(path: Path, source: str) -> None:
        if path.name.startswith("task-2"):
            raise OSError("simulated task write failure")
        original_writer(path, source)

    monkeypatch.setattr(milestones_module, "_atomic_write_text", fail_on_second_task)

    with pytest.raises(OSError, match="simulated task write failure"):
        service.remove_milestone("Alpha", clear_tasks=True)

    assert (repo / "backlog" / "milestones" / "Alpha.md").exists()
    assert first_task.read_text(encoding="utf-8") == original_first_source
    assert second_task.read_text(encoding="utf-8") == original_second_source


def test_rename_leaves_task_references_when_update_not_requested(tmp_path):
    repo = _copy_fixture(tmp_path)
    service = _service(repo)
    service.add_milestone("Alpha")
    _set_task_milestone(repo, "Alpha")

    service.rename_milestone("Alpha", "Beta", update_tasks=False)

    assert "milestone: Alpha" in _task_path(repo).read_text(encoding="utf-8")


def test_cli_milestone_commands_use_safe_service(tmp_path):
    repo = _copy_fixture(tmp_path)
    runner = CliRunner()

    add = runner.invoke(main, ["--cwd", str(repo), "milestone", "add", "Alpha"])
    assert add.exit_code == 0
    assert "Alpha" in add.output

    rename = runner.invoke(main, ["--cwd", str(repo), "milestone", "rename", "Alpha", "Beta"])
    assert rename.exit_code == 0
    assert "Beta" in rename.output

    listed = runner.invoke(main, ["--cwd", str(repo), "milestone", "list"])
    assert listed.exit_code == 0
    assert "Beta" in listed.output

    archive = runner.invoke(main, ["--cwd", str(repo), "milestone", "archive", "Beta"])
    assert archive.exit_code == 0
    assert "archived" in archive.output


def test_mcp_milestone_tools_use_safe_service(tmp_path):
    repo = _copy_fixture(tmp_path)
    project = _project(repo)

    added = milestone_add(project, "Alpha")
    assert added["name"] == "Alpha"
    assert [milestone["name"] for milestone in milestone_list(project)] == ["Alpha"]

    renamed = milestone_rename(project, "Alpha", "Beta")
    assert renamed["name"] == "Beta"

    milestone_remove(project, "Beta")
    assert milestone_list(project) == []

    milestone_add(project, "Release 1")
    archived = milestone_archive(project, "Release 1")
    assert archived["archived"] is True
