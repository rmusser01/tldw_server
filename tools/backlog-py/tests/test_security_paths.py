from pathlib import Path

import pytest

from backlog_py.security.paths import PathContainmentError, assert_path_within_base


def test_assert_path_within_base_allows_backlog_child(tmp_path):
    base = tmp_path / "repo" / "backlog"
    task_path = base / "tasks" / "task-2 - New.md"
    base.mkdir(parents=True)

    assert assert_path_within_base(base, task_path) == task_path.resolve()


def test_assert_path_within_base_rejects_parent_traversal(tmp_path):
    base = tmp_path / "repo" / "backlog"
    base.mkdir(parents=True)
    escaped = base / ".." / "outside.md"

    with pytest.raises(PathContainmentError, match="outside allowed base"):
        assert_path_within_base(base, escaped)


def test_assert_path_within_base_rejects_sibling_prefix(tmp_path):
    base = tmp_path / "repo" / "backlog"
    sibling = tmp_path / "repo" / "backlog-other" / "task.md"
    base.mkdir(parents=True)
    sibling.parent.mkdir(parents=True)

    with pytest.raises(PathContainmentError):
        assert_path_within_base(base, sibling)


def test_assert_path_within_base_rejects_symlinked_base_escape(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    base = tmp_path / "repo" / "backlog"
    base.parent.mkdir(parents=True)
    try:
        base.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")

    with pytest.raises(PathContainmentError):
        assert_path_within_base(base, base / "task.md")
