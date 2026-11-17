"""Fail when two backlog task files share an id (TASK-13960).

A file holds the id in its name, which is what backlog-py reads, and the one in its ``id:`` frontmatter.
backend-required's ``changes`` job runs this on every PR, backlog-only ones included.
Standard library only: that job installs nothing.
"""

from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

_TASK_DIRS = ("backlog/tasks", "backlog/completed", "backlog/archive/tasks")
_ID_RE = re.compile(r"^id:\s*['\"]?([A-Za-z]+-\d+(?:\.\d+)*)", re.MULTILINE)
_FIX = (
    "Give the newer file a free id: find one with the AGENTS.md recipe (Backlog.md Task Tracking), then run\n"
    '  PYTHONPATH=tools/backlog-py/src python -m backlog_py --cwd . task renumber "<path>" --id TASK-<free id>'
)


def _escape(value: str, *, prop: bool = False) -> str:
    """Escape a workflow command's message, or with ``prop`` a property such as ``file=``, as GitHub's toolkit does."""
    value = value.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
    return value.replace(":", "%3A").replace(",", "%2C") if prop else value


def _task_ids(path: Path) -> set[str]:
    """The id in the file's name and the one in its ``id:`` line; the same id unless the two disagree."""
    try:
        head = path.read_text(encoding="utf-8").split("\n---", 1)[0]
    except (OSError, UnicodeDecodeError) as exc:
        print(f"::error file={_escape(path.as_posix(), prop=True)}::{_escape(f'Cannot read this task file: {exc}')}")
        raise
    match = _ID_RE.search(head)
    name_id = path.name.split(" - ", 1)[0].removesuffix(".md")
    return {name_id.upper(), (match.group(1) if match else name_id).upper()}


def duplicate_task_ids(root: Path) -> dict[str, list[str]]:
    """Each id held by more than one task file, with those files (relative to ``root``)."""
    by_id: dict[str, list[str]] = defaultdict(list)
    for directory in _TASK_DIRS:
        for path in sorted((root / directory).glob("task-*.md")):
            for task_id in _task_ids(path):
                by_id[task_id].append(path.relative_to(root).as_posix())
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
            print(f"::error file={_escape(path, prop=True)}::{_escape(f'{task_id} is also the id of {others}')}")
    if duplicates:
        print(f"{len(duplicates)} task id(s) are used by more than one file. {_FIX}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
