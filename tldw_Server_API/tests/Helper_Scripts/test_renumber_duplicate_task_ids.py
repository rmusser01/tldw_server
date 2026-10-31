"""The keeper rule, the open-PR cache and the open-PR duplicate list of the duplicate-id renumbering (TASK-13960)."""

import json
from pathlib import Path

import pytest
from Helper_Scripts import renumber_duplicate_task_ids as renumber
from Helper_Scripts.renumber_duplicate_task_ids import Task, choose_keeper

pytestmark = pytest.mark.unit


def _copy(directory: str, name: str, origin_index: int, **evidence: object) -> Task:
    task = Task(path=f"backlog/{directory}/{name}", id="TASK-7", title=name, parent="", deps=[], size=100)
    task.origin_index = origin_index
    for key, value in evidence.items():
        setattr(task, key, value)
    return task


def test_the_first_added_copy_keeps_the_id_and_completed_wins_a_tie() -> None:
    old, new = _copy("tasks", "task-7 - Old.md", 1), _copy("tasks", "task-7 - New.md", 2)
    assert choose_keeper("TASK-7", [new, old], []) == (old, "d", "")
    active, done = _copy("tasks", "task-7 - A.md", 1), _copy("completed", "task-7 - B.md", 1)
    assert choose_keeper("TASK-7", [active, done], [])[0] is done


def test_the_copy_that_reached_dev_first_keeps_the_id() -> None:
    # Added first (origin_index 1) on a long-lived branch whose merge is dev's 9th first-parent commit;
    # the other copy was added later but reached dev at the 5th, so dev has held it longer.
    branch = _copy("tasks", "task-7 - Branch.md", 1, arrival=9)
    landed = _copy("tasks", "task-7 - Landed.md", 2, arrival=5)
    assert choose_keeper("TASK-7", [branch, landed], []) == (landed, "d", "")


def test_a_copy_tests_name_or_that_owns_subtasks_beats_age() -> None:
    old, cited = _copy("tasks", "task-7 - Old.md", 1), _copy("tasks", "task-7 - Cited.md", 2, code_refs=["t.py"])
    assert choose_keeper("TASK-7", [old, cited], []) == (cited, "a", "")
    parent = _copy("tasks", "task-7 - Parent.md", 3, owns=["TASK-7.1"])
    assert choose_keeper("TASK-7", [old, parent], []) == (parent, "b", "")


def test_conflicting_rules_and_unowned_subtasks_wait_for_the_owner(monkeypatch: pytest.MonkeyPatch) -> None:
    parent = _copy("tasks", "task-7 - Parent.md", 1, owns=["TASK-7.1"])
    edited = _copy("tasks", "task-7 - Edited.md", 2, open_prs=[3000])
    monkeypatch.setattr(renumber, "OWNER_DECISIONS", {})
    keeper, rule, why = choose_keeper("TASK-7", [parent, edited], [])
    assert (keeper, rule) == (None, "owner") and "(b)" in why and "(c)" in why
    assert choose_keeper("TASK-7", [_copy("tasks", "task-7 - Old.md", 1)], ["TASK-7.2"])[0] is None

    monkeypatch.setattr(renumber, "OWNER_DECISIONS", {"TASK-7": ("task-7 - Edited.md", "owner said so")})
    assert choose_keeper("TASK-7", [parent, edited], [])[:2] == (edited, "owner")


def test_the_pr_cache_is_fetched_again_once_origin_dev_moves(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fetches: list[tuple[str, ...]] = []

    def fake_gh(*argv: str, check: bool = True) -> str:
        fetches.append(argv[:3])
        if argv[:3] == ("gh", "pr", "list"):
            return json.dumps([{"number": 1, "title": "t", "author": {"login": "a"}, "headRefName": "theirs"}])
        return "added\tbacklog/tasks/task-1 - A.md\t\n"

    dev = {"tip": "aaa"}
    monkeypatch.setattr(renumber, "run", fake_gh)
    monkeypatch.setattr(renumber, "git", lambda *args: dev["tip"] if args[-1] == "origin/dev" else "mine\n")
    cache = tmp_path / "prs.json"

    assert renumber.open_pr_files(cache)[0]["files"] == [
        {"status": "added", "path": "backlog/tasks/task-1 - A.md", "previous": ""}
    ]
    renumber.open_pr_files(cache)
    assert len(fetches) == 2  # the list and one PR's files; the second call read the cache
    dev["tip"] = "bbb"
    renumber.open_pr_files(cache)
    assert len(fetches) == 4  # origin/dev moved, so the lists were fetched again
    renumber.open_pr_files(cache, refresh=True)
    assert len(fetches) == 6


def test_a_pr_file_dev_already_holds_is_removed_and_a_renamed_one_renumbered() -> None:
    def task(path: str) -> Task:
        return Task(path=path, id=renumber.id_from_name(path), title="t", parent="", deps=[], size=1)

    dev = [task("backlog/tasks/task-7 - Kept.md"), task("backlog/tasks/task-15000 - Moved.md"),
           task("backlog/tasks/task-8 - Renamed.md")]
    pr = {"number": 1, "author": "a", "files": [
        {"status": "added", "path": "backlog/tasks/task-7 - Moved.md", "previous": ""},
        {"status": "added", "path": "backlog/tasks/task-7 - New.md", "previous": ""},
        {"status": "renamed", "path": "backlog/tasks/task-7 - Renamed.md", "previous": "backlog/tasks/task-8 - Renamed.md"},
        {"status": "added", "path": "backlog/tasks/task-9 - Free.md", "previous": ""},
    ]}

    found = {path: (holders, same) for _, path, holders, same in renumber.pr_duplicates([pr], dev)}

    kept = ["backlog/tasks/task-7 - Kept.md"]
    assert found == {
        "backlog/tasks/task-7 - Moved.md": (kept, ["backlog/tasks/task-15000 - Moved.md"]),  # remove
        "backlog/tasks/task-7 - New.md": (kept, []),  # renumber
        "backlog/tasks/task-7 - Renamed.md": (kept, []),  # its own rename source does not count
    }
