"""Remote task-id awareness for collision-safe allocation.

Task ids collide when concurrent sessions each allocate the next id from
their local ``backlog/`` alone (the AGENTS.md recipe: take the max across
``origin/dev`` and open PR branches). This module implements that recipe,
best-effort: every source is optional, and failures degrade to local-only
behaviour with a one-time stderr warning so offline use never breaks.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

_TASK_PATH_ID_RE = re.compile(r"(?:^|/)task-(\d+)(?:[ .]|$)")

_LOCAL_TIMEOUT_S = 5.0  # git ls-tree reads local objects
_NETWORK_TIMEOUT_S = 30.0  # git fetch and every gh call (gh pr list took 2.5-5.5 s with 47 PRs)
_GH_FILE_CAP = 100  # gh pr list --json files returns at most this many files per PR


class SourceError(Exception):
    """A remote id source could not be read; the message says why."""


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


def _run(cmd: list[str], cwd: Path, timeout: float) -> str:
    """Run a read-only command; return stdout or raise SourceError saying why."""
    try:
        proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError:
        raise SourceError("not installed") from None
    except subprocess.TimeoutExpired:
        raise SourceError(f"timed out after {timeout:g}s") from None
    except (OSError, subprocess.SubprocessError) as exc:
        raise SourceError(str(exc)) from None
    if proc.returncode != 0:
        raise SourceError(f"exit status {proc.returncode}")
    return proc.stdout


def _warn(message: str) -> None:
    print(f"warning: {message}", file=sys.stderr)


def _backlog_ids(paths: Iterable[str]) -> set[int]:
    ids: set[int] = set()
    for path in paths:
        if path.startswith("backlog/"):
            match = _TASK_PATH_ID_RE.search(path)
            if match:
                ids.add(int(match.group(1)))
    return ids


def _origin_dev_ids(root: Path) -> tuple[str, set[int]] | None:
    try:
        _run(["git", "fetch", "--quiet", "--no-tags", "origin", "dev"], root, _NETWORK_TIMEOUT_S)
    except SourceError as exc:
        _warn(f"git fetch origin dev failed ({exc}); the origin/dev source is as of the last fetch")
    reasons = []
    for ref in ("origin/dev", "origin/HEAD"):
        try:
            out = _run(["git", "ls-tree", "-r", "--name-only", ref, "--", "backlog/"], root, _LOCAL_TIMEOUT_S)
        except SourceError as exc:
            reasons.append(f"{ref}: {exc}")
            continue
        return ref, _backlog_ids(out.splitlines())
    _warn(f"origin/dev unreachable ({'; '.join(reasons)}); continuing with the remaining sources")
    return None


def _open_pr_ids(root: Path) -> set[int] | None:
    try:
        listing = json.loads(
            _run(
                ["gh", "pr", "list", "--state", "open", "--limit", "500", "--json", "number,changedFiles,files"],
                root,
                _NETWORK_TIMEOUT_S,
            )
        )
    except (SourceError, ValueError) as exc:
        _warn(f"open PRs (gh): {exc}; continuing with the remaining sources")
        return None
    paths: list[str] = []
    for pr in listing:
        files = [f["path"] for f in pr.get("files", [])]
        if len(files) < pr.get("changedFiles", 0):
            try:
                files = _run(
                    ["gh", "api", f"repos/{{owner}}/{{repo}}/pulls/{pr['number']}/files",
                     "--paginate", "--jq", ".[].filename"],
                    root,
                    _NETWORK_TIMEOUT_S,
                ).splitlines()
            except SourceError as exc:
                _warn(
                    f"PR #{pr['number']} exceeds {_GH_FILE_CAP} files and its file lookup failed ({exc}); "
                    "its task ids may be missing"
                )
        paths.extend(files)
    return _backlog_ids(paths)


def collect_remote_ids(root: Path) -> RemoteIdState:
    """Collect task ids on origin/dev (fetched first) and in open PR files."""
    ids_by_source: dict[str, set[int]] = {}
    origin = _origin_dev_ids(root)
    if origin and origin[1]:
        ids_by_source[origin[0]] = origin[1]
    pr_ids = _open_pr_ids(root)
    if pr_ids:
        ids_by_source["open pull requests"] = pr_ids
    if not ids_by_source:
        _warn("no remote task-id source reachable (origin/dev, open PRs); falling back to local-only id allocation")
    return RemoteIdState(ids_by_source=ids_by_source)
