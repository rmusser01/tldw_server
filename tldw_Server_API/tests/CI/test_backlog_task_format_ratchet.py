"""Ratchet: task files a change adds or edits must be in backlog-py's canonical form.

ADR-059 (TASK-13440) makes tools/backlog-py the repository's task editor. The Node
Backlog.md CLI writes ``SECTION:NOTES`` and, when it edits a backlog-py task, nests
the existing ``SECTION:IMPLEMENTATION_NOTES`` inside a new NOTES block and duplicates
the FINAL_SUMMARY markers. ``backlog-py task normalize`` rewrites all of that into one
canonical form, and every backlog-py edit leaves the file it touches canonical.

The check reads only the task files changed between the merge base with
``BACKLOG_TASK_FORMAT_BASE`` and HEAD. backend-required passes the PR's base commit;
a local run defaults to ``origin/dev``. Scoping it to the change means a PR is never
failed for a file it did not touch: a non-canonical file that reached dev through a
PR this gate did not run on (a backlog-only PR) cannot block unrelated work, and the
legacy files TASK-13441 normalizes need no baseline.

A failure here usually means a task file was written by the Node CLI. Fix it with:

    PYTHONPATH=tools/backlog-py/src python -m backlog_py --cwd . task normalize <file>
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Iterable
from pathlib import Path, PurePosixPath

import pytest
import yaml
from backlog_py.markdown.task_parser import normalize_task_markdown

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "backend-required.yml"
BASE_ENV = "BACKLOG_TASK_FORMAT_BASE"
_TASK_DIR = PurePosixPath("backlog/tasks")
_FIX_HINT = "PYTHONPATH=tools/backlog-py/src python -m backlog_py --cwd . task normalize <file>"


def _changed_task_files(repo_root: Path, base: str) -> list[Path]:
    """Task files added or modified on HEAD since its merge base with ``base`` (deletions excluded)."""
    result = subprocess.run(  # fixed git argv, no shell
        ["git", "diff", "--name-only", "-z", "--diff-filter=d", f"{base}...HEAD", "--", str(_TASK_DIR)],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    names = [name for name in result.stdout.split("\0") if name]
    return [
        repo_root / name
        for name in names
        if PurePosixPath(name).parent == _TASK_DIR and name.endswith(".md")
    ]


def _task_format_failures(paths: Iterable[Path]) -> list[str]:
    """One line per file backlog-py cannot read or would rewrite, naming the file and the reason."""
    failures = []
    for path in paths:
        try:
            source = path.read_bytes().decode("utf-8")  # bytes, so CRLF is seen as backlog-py reads it
            if normalize_task_markdown(source) != source:
                failures.append(f"{path.name}: not in backlog-py's canonical form")
        except ValueError as exc:
            failures.append(f"{path.name}: {exc}")
    return failures


def test_task_files_changed_by_this_change_are_canonical() -> None:
    base = os.environ.get(BASE_ENV) or "origin/dev"
    try:
        changed = _changed_task_files(REPO_ROOT, base)
    except (OSError, subprocess.CalledProcessError) as exc:
        if os.environ.get(BASE_ENV):
            raise
        pytest.skip(f"cannot diff against {base}; fetch it or set {BASE_ENV} to a base commit ({exc})")

    failures = _task_format_failures(changed)

    assert not failures, (
        "These changed task files are not in backlog-py's canonical form. Use backlog-py, not the Node "
        f"`backlog` CLI/MCP, and normalize them with: {_FIX_HINT}\n" + "\n".join(failures)
    )


def test_backend_required_gives_the_ratchet_the_pr_base() -> None:
    """Without the base the required gate would fall back to origin/dev, or skip."""
    steps = [
        step
        for job in yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))["jobs"].values()
        for step in job.get("steps", [])
        if "test_backlog_task_format_ratchet.py" in str(step.get("run", ""))
    ]

    assert len(steps) == 1
    assert "base_sha" in str(steps[0].get("env", {}).get(BASE_ENV, ""))


_CROSSED = (
    "---\nid: TASK-9\n---\n\n"
    "<!-- SECTION:NOTES:BEGIN -->\n<!-- SECTION:PLAN:BEGIN -->\ntext\n"
    "<!-- SECTION:NOTES:END -->\n<!-- SECTION:PLAN:END -->\n"
)
_NODE_NOTES = "---\nid: TASK-8\n---\n\n<!-- SECTION:NOTES:BEGIN -->\nnote\n<!-- SECTION:NOTES:END -->\n"
_CANONICAL = "---\nid: TASK-7\n---\n\n<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->\nnote\n<!-- SECTION:IMPLEMENTATION_NOTES:END -->\n"


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", "-c", "commit.gpgsign=false", *args],
        cwd=repo,
        check=True,
        capture_output=True,
    )


def _commit_file(repo: Path, relative: str, content: str) -> None:
    path = repo / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    _git(repo, "add", relative)
    _git(repo, "commit", "-q", "-m", f"add {relative}")


def test_changed_task_files_are_the_branch_own_changes_only(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "dev")
    _commit_file(repo, "backlog/tasks/task-1 - Old.md", _NODE_NOTES)
    _git(repo, "checkout", "-q", "-b", "feature")
    _commit_file(repo, "backlog/tasks/task-2 - Mine.md", _CANONICAL)
    _commit_file(repo, "tools/other.md", _NODE_NOTES)
    _git(repo, "checkout", "-q", "dev")
    _commit_file(repo, "backlog/tasks/task-3 - Landed on dev unchecked.md", _NODE_NOTES)
    _git(repo, "checkout", "-q", "feature")

    changed = _changed_task_files(repo, "dev")

    assert [path.name for path in changed] == ["task-2 - Mine.md"]


def test_failures_name_every_file_backlog_py_would_rewrite_or_reject(tmp_path: Path) -> None:
    files = {"task-9 - Crossed.md": _CROSSED, "task-8 - Node.md": _NODE_NOTES, "task-7 - Fine.md": _CANONICAL}
    for name, content in files.items():
        (tmp_path / name).write_text(content, encoding="utf-8")

    failures = _task_format_failures(sorted(tmp_path.glob("*.md")))

    assert len(failures) == 2
    assert failures[0].startswith("task-8 - Node.md: ")
    assert failures[1].startswith("task-9 - Crossed.md: ") and "crossed" in failures[1]
