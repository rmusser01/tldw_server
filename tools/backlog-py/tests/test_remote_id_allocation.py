"""Collision-safe task-id allocation across origin/dev and open PRs."""

from __future__ import annotations

import json
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


def _prs(*files_per_pr, changed=None):
    return json.dumps(
        [
            {"number": n + 1, "changedFiles": changed or len(f), "files": [{"path": p} for p in f]}
            for n, f in enumerate(files_per_pr)
        ]
    )


def _all_fail(cmd, _cwd, timeout):
    raise remote_ids.SourceError("down")


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
    assert created.id == "TASK-2"


def test_collect_remote_ids_parses_git_and_gh_outputs(tmp_path, monkeypatch):
    def fake_run(cmd, _cwd, timeout):
        if cmd[0] == "git":
            return "backlog/tasks/task-41 - Old.md\nbacklog/tasks/task-42 - New.md\n"
        return _prs(("backlog/tasks/task-77 - In-flight.md",))

    monkeypatch.setattr(remote_ids, "_run", fake_run)
    state = collect_remote_ids(tmp_path)
    assert state.ids_by_source["origin/dev"] == {41, 42}
    assert state.ids_by_source["open pull requests"] == {77}
    assert state.max_id == 77
    assert state.source_of(77) == "open pull requests"


def test_collect_remote_ids_warns_when_nothing_reachable(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(remote_ids, "_run", _all_fail)
    state = collect_remote_ids(tmp_path)
    assert not state.any_source_reachable
    assert state.max_id == 0
    assert "local-only" in capsys.readouterr().err


def test_collect_remote_ids_ignores_non_task_paths(tmp_path, monkeypatch):
    def fake_run(cmd, _cwd, timeout):
        if cmd[0] == "git":
            return "backlog/config.yml\nbacklog/docs/mentions-task-999-not-a-file.md\n"
        raise remote_ids.SourceError("down")

    monkeypatch.setattr(remote_ids, "_run", fake_run)
    state = collect_remote_ids(tmp_path)
    # task-999 appears mid-filename without the task-<digits> boundary -> ignored;
    # config.yml carries no id. Only genuinely reachable sources are recorded.
    assert state.ids_by_source.get("origin/dev", set()) == set()


def _recording_run(calls, gh_list, api=None):
    def fake_run(cmd, cwd, timeout):
        calls.append((cmd, timeout))
        if cmd[:2] == ["git", "fetch"]:
            return ""
        if cmd[0] == "git":
            return "backlog/tasks/task-41 - Old.md\n"
        if cmd[:3] == ["gh", "pr", "list"]:
            return gh_list
        if cmd[:2] == ["gh", "api"]:
            if api is None:
                raise remote_ids.SourceError("exit status 1")
            return api
        raise AssertionError(cmd)

    return fake_run


def test_each_failing_source_warns_by_name_and_allocation_continues(tmp_path, monkeypatch, capsys):
    def fake_run(cmd, _cwd, timeout):
        if cmd[0] == "gh":
            raise remote_ids.SourceError("timed out")
        return "backlog/tasks/task-41 - Old.md\n"

    monkeypatch.setattr(remote_ids, "_run", fake_run)
    state = collect_remote_ids(tmp_path)
    err = capsys.readouterr().err
    assert state.ids_by_source == {"origin/dev": {41}}
    assert "open PRs (gh): timed out" in err
    assert "continuing with the remaining sources" in err


def test_timeouts_are_5s_for_local_git_and_30s_for_gh_and_fetch(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(remote_ids, "_run", _recording_run(calls, _prs(("backlog/tasks/task-9 - X.md",))))
    collect_remote_ids(tmp_path)
    timeouts = {tuple(c[:3]): t for c, t in calls}
    assert timeouts[("git", "fetch", "--quiet")] == 30.0
    assert timeouts[("git", "ls-tree", "-r")] == 5.0
    assert timeouts[("gh", "pr", "list")] == 30.0


def test_origin_dev_is_fetched_before_it_is_read(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(remote_ids, "_run", _recording_run(calls, _prs()))
    collect_remote_ids(tmp_path)
    cmds = [c for c, _ in calls]
    fetch = ["git", "fetch", "--quiet", "--no-tags", "origin", "dev"]
    assert fetch in cmds
    assert cmds.index(fetch) < next(i for i, c in enumerate(cmds) if c[:2] == ["git", "ls-tree"])


def test_failed_fetch_warns_and_still_uses_the_local_ref(tmp_path, monkeypatch, capsys):
    def fake_run(cmd, _cwd, timeout):
        if cmd[:2] == ["git", "fetch"]:
            raise remote_ids.SourceError("exit status 128")
        if cmd[0] == "git":
            return "backlog/tasks/task-41 - Old.md\n"
        return _prs()

    monkeypatch.setattr(remote_ids, "_run", fake_run)
    state = collect_remote_ids(tmp_path)
    assert state.ids_by_source["origin/dev"] == {41}
    assert "as of the last fetch" in capsys.readouterr().err


def test_pr_over_100_files_is_paged_so_its_backlog_ids_count(tmp_path, monkeypatch):
    calls = []
    listing = _prs(("Docs/a.md",) * 100, changed=146)
    api = "Docs/a.md\nbacklog/tasks/task-13408 - Hidden.md\n"
    monkeypatch.setattr(remote_ids, "_run", _recording_run(calls, listing, api))
    state = collect_remote_ids(tmp_path)
    assert state.ids_by_source["open pull requests"] == {13408}
    api_calls = [(c, t) for c, t in calls if c[:2] == ["gh", "api"]]
    assert api_calls[0][0][2] == "repos/{owner}/{repo}/pulls/1/files"
    assert "--paginate" in api_calls[0][0] and api_calls[0][1] == 30.0


def test_failed_paged_lookup_warns_about_the_pr(tmp_path, monkeypatch, capsys):
    listing = _prs(("Docs/a.md",) * 100, changed=146)
    monkeypatch.setattr(remote_ids, "_run", _recording_run([], listing, None))
    collect_remote_ids(tmp_path)
    err = capsys.readouterr().err
    assert "PR #1 exceeds 100 files" in err and "may be missing" in err


def test_gh_paths_outside_backlog_are_ignored(tmp_path, monkeypatch):
    listing = _prs((".superpowers/sdd/task-12968.2-task3-report.md",))
    monkeypatch.setattr(remote_ids, "_run", _recording_run([], listing))
    state = collect_remote_ids(tmp_path)
    assert "open pull requests" not in state.ids_by_source


def test_child_id_skips_the_remote_lookup(tmp_path, monkeypatch):
    repo = _copy_fixture(tmp_path)
    calls = []
    monkeypatch.setattr(
        repository_module, "collect_remote_ids", lambda _root: calls.append(1) or RemoteIdState()
    )
    created = MutableRepository.from_path(repo).create_task(title="Child", task_id="TASK-1.1")
    assert created.id == "TASK-1.1"
    assert calls == []
