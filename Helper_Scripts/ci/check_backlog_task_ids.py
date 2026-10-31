"""Fail when two backlog task files share an id (TASK-N1).

Ids come from each file's ``id:`` frontmatter, or its file name when that is missing.
backend-required's ``changes`` job runs this on every PR, backlog-only ones included.
Standard library only: that job installs nothing.
"""

from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

_TASK_DIRS = ("backlog/tasks", "backlog/completed", "backlog/archive/tasks")
_ID_RE = re.compile(r"^id:\s*['\"]?([A-Za-z]+-[\d.]+)", re.MULTILINE)
_FIX = (
    "Give the newer file a free id: find one with the AGENTS.md recipe (Backlog.md Task Tracking), then run\n"
    '  PYTHONPATH=tools/backlog-py/src python -m backlog_py --cwd . task renumber "<path>" --id TASK-<free id>'
)


def _task_id(path: Path) -> str:
    head = path.read_text(encoding="utf-8").split("\n---", 1)[0]
    match = _ID_RE.search(head)
    return (match.group(1) if match else path.name.split(" - ", 1)[0].removesuffix(".md")).upper()


def duplicate_task_ids(root: Path) -> dict[str, list[str]]:
    """Each id held by more than one task file, with those files (relative to ``root``)."""
    by_id: dict[str, list[str]] = defaultdict(list)
    for directory in _TASK_DIRS:
        for path in sorted((root / directory).glob("task-*.md")):
            by_id[_task_id(path)].append(path.relative_to(root).as_posix())
    return {task_id: sorted(paths) for task_id, paths in sorted(by_id.items()) if len(paths) > 1}


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    root = Path(args[0] if args else ".")
    if not any((root / directory).is_dir() for directory in _TASK_DIRS):
        print(f"::error::No backlog task directory under {root} ({', '.join(_TASK_DIRS)}); nothing was checked.")
        return 2
    duplicates = duplicate_task_ids(root)
    for task_id, paths in duplicates.items():
        for path in paths:
            others = ", ".join(other for other in paths if other != path)
            print(f"::error file={path}::{task_id} is also the id of {others}")
    if duplicates:
        print(f"{len(duplicates)} task id(s) are used by more than one file. {_FIX}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
