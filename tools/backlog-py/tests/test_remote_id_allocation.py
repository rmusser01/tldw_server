"""Collision-safe task-id allocation across origin/dev and open PRs."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from backlog_py.core import remote_ids
from backlog_py.core import repository as repository_module
from backlog_py.core.remote_ids import RemoteIdState, collect_remote_ids
from backlog_py.core.repository import MutableRepository, TaskMutationError

FIXTURE_REPO = Path(__file__).parent / "fixtures" / "repos" / "basic"

pytestmark = pytest.mark.unit


def _copy_fixture(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    shutil.copytree(FIXTURE_REPO, repo)
    return repo


def _state(ids_by_source: dict[str, set[int]]) -> RemoteIdState:
    return RemoteIdState(ids_by_source=ids_by_source)


def test_allocation_uses_remote_max(tmp_path, monkeypatch):
    repo = _copy_fixture(tmp_path)
    monkeypatch.setattr(
        repository_module, "collect_remote_ids", lambda _root: _state({"origin/dev": {99}})
    )
    created = MutableRepository.from_path(repo).create_task(title="Remote-aware")
    assert created.id == "TASK-100"


def test_explicit_id_colliding_with_remote_branch_is_rejected(tmp_path, monkeypatch):
    repo = _copy_fixture(tmp_path)
    monkeypatch.setattr(
        repository_module, "collect_remote_ids", lambda _root: _state({"origin/dev": {99}})
    )
    with pytest.raises(TaskMutationError, match="origin/dev.*TASK-99"):
        MutableRepository.from_path(repo).create_task(title="Collides", task_id="TASK-99")


def test_explicit_id_colliding_with_open_pr_is_rejected(tmp_path, monkeypatch):
    repo = _copy_fixture(tmp_path)
    monkeypatch.setattr(
        repository_module,
        "collect_remote_ids",
        lambda _root: _state({"open pull requests": {50}}),
    )
    with pytest.raises(TaskMutationError, match="open pull requests.*TASK-50"):
        MutableRepository.from_path(repo).create_task(title="PR collision", task_id="TASK-50")


def test_unreachable_sources_fall_back_to_local(tmp_path, monkeypatch):
    repo = _copy_fixture(tmp_path)
    monkeypatch.setattr(
        repository_module, "collect_remote_ids", lambda _root: RemoteIdState()
    )
    created = MutableRepository.from_path(repo).create_task(title="Local only")
    assert created.id  # local allocation still works; exact value is fixture-defined


def test_collect_remote_ids_parses_git_and_gh_outputs(tmp_path, monkeypatch):
    def fake_run(cmd, _cwd):
        if cmd[0] == "git":
            return "backlog/tasks/task-41 - Old.md\nbacklog/tasks/task-42 - New.md\n"
        if cmd[0] == "gh":
            return "backlog/tasks/task-77 - In-flight.md\n"
        return None

    monkeypatch.setattr(remote_ids, "_run", fake_run)
    state = collect_remote_ids(tmp_path)
    assert state.ids_by_source["origin/dev"] == {41, 42}
    assert state.ids_by_source["open pull requests"] == {77}
    assert state.max_id == 77
    assert state.source_of(77) == "open pull requests"


def test_collect_remote_ids_warns_when_nothing_reachable(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(remote_ids, "_run", lambda *_a, **_k: None)
    state = collect_remote_ids(tmp_path)
    assert not state.any_source_reachable
    assert state.max_id == 0
    assert "local-only" in capsys.readouterr().err


def test_collect_remote_ids_ignores_non_task_paths(tmp_path, monkeypatch):
    def fake_run(cmd, _cwd):
        return "backlog/config.yml\nbacklog/docs/mentions-task-999-not-a-file.md\n" if cmd[0] == "git" else None

    monkeypatch.setattr(remote_ids, "_run", fake_run)
    state = collect_remote_ids(tmp_path)
    # task-999 appears mid-filename without the task-<digits> boundary -> ignored;
    # config.yml carries no id. Only genuinely reachable sources are recorded.
    assert state.ids_by_source.get("origin/dev", set()) == set()
