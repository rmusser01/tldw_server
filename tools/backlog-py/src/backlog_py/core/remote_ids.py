"""Remote task-id awareness for collision-safe allocation.

Task ids collide when concurrent sessions each allocate the next id from
their local ``backlog/`` alone (the AGENTS.md recipe: take the max across
``origin/dev`` and open PR branches). This module implements that recipe,
best-effort: every source is optional, and failures degrade to local-only
behaviour with a one-time stderr warning so offline use never breaks.
"""

from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

_TASK_PATH_ID_RE = re.compile(r"(?:^|/)task-(\d+)(?:[ .]|$)")

_TIMEOUT_S = 5.0


@dataclass(frozen=True)
class RemoteIdState:
    """Task ids known to exist outside this checkout, by source label."""

    ids_by_source: dict[str, set[int]] = field(default_factory=dict)

    @property
    def any_source_reachable(self) -> bool:
        return bool(self.ids_by_source)

    @property
    def max_id(self) -> int:
        return max((i for ids in self.ids_by_source.values() for i in ids), default=0)

    def source_of(self, numeric_id: int) -> str | None:
        for source, ids in self.ids_by_source.items():
            if numeric_id in ids:
                return source
        return None


def _run(cmd: list[str], cwd: Path) -> str | None:
    """Run a read-only command; return stdout on success, None otherwise."""
    try:
        proc = subprocess.run(
            cmd, cwd=cwd, capture_output=True, text=True, timeout=_TIMEOUT_S
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return proc.stdout if proc.returncode == 0 else None


def _ids_from_output(output: str) -> set[int]:
    return {int(m.group(1)) for m in _TASK_PATH_ID_RE.finditer(output)}


def collect_remote_ids(root: Path) -> RemoteIdState:
    """Collect task ids visible on origin's dev branch and in open PR files."""
    state = RemoteIdState()

    git_out = _run(
        ["git", "ls-tree", "-r", "--name-only", "origin/dev", "--", "backlog/"], root
    )
    if git_out is None:
        git_out = _run(
            ["git", "ls-tree", "-r", "--name-only", "origin/HEAD", "--", "backlog/"], root
        )
        source = "origin/HEAD"
    else:
        source = "origin/dev"
    if git_out is not None:
        ids = _ids_from_output(git_out)
        if ids:
            state.ids_by_source[source] = ids

    gh_out = _run(
        [
            "gh",
            "pr",
            "list",
            "--state",
            "open",
            "--limit",
            "500",
            "--json",
            "files",
            "--jq",
            ".[].files[].path",
        ],
        root,
    )
    if gh_out is not None:
        ids = _ids_from_output(gh_out)
        if ids:
            state.ids_by_source["open pull requests"] = ids

    if not state.any_source_reachable:
        print(
            "warning: no remote task-id source reachable (origin/dev, open PRs); "
            "falling back to local-only id allocation",
            file=sys.stderr,
        )
    return state
