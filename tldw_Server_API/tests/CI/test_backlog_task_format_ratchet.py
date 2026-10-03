"""Ratchet: backlog task files stay readable by backlog-py, and Node-format debt only shrinks.

ADR-059 (TASK-13440) makes tools/backlog-py the repository's task editor. The Node
Backlog.md CLI writes ``SECTION:NOTES`` and, when it edits a backlog-py task, nests
the existing ``SECTION:IMPLEMENTATION_NOTES`` inside a new NOTES block and duplicates
the FINAL_SUMMARY markers. ``backlog-py task normalize`` rewrites all of that into one
canonical form. The number of task files it would still rewrite is frozen here and
may only go down; TASK-13441 normalizes the backlog and lowers the baseline to zero.

A failure here usually means a task file was written by the Node CLI. Fix it with:

    PYTHONPATH=tools/backlog-py/src python -m backlog_py --cwd . task normalize <file>
"""

from __future__ import annotations

from pathlib import Path

import pytest
from backlog_py.markdown.task_parser import normalize_task_markdown, parse_task_markdown

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
TASK_DIR = REPO_ROOT / "backlog" / "tasks"

# Files `task normalize --check` lists on this branch's base. May only be lowered.
NEEDS_NORMALIZATION_BASELINE = 2308

_FIX_HINT = "PYTHONPATH=tools/backlog-py/src python -m backlog_py --cwd . task normalize <file>"


def _task_sources() -> list[tuple[str, str]]:
    # Bytes, so CRLF files are seen exactly as backlog-py reads them.
    return [(path.name, path.read_bytes().decode("utf-8")) for path in sorted(TASK_DIR.glob("*.md"))]


def test_every_task_file_parses_with_backlog_py() -> None:
    failures = []
    for name, source in _task_sources():
        try:
            parse_task_markdown(source)
        except ValueError as exc:
            failures.append(f"{name}: {exc}")

    assert not failures, "backlog-py cannot parse these task files:\n" + "\n".join(failures)


def test_task_files_needing_normalization_do_not_exceed_baseline() -> None:
    needing = [name for name, source in _task_sources() if normalize_task_markdown(source) != source]

    assert len(needing) <= NEEDS_NORMALIZATION_BASELINE, (
        f"{len(needing)} task files are not in backlog-py's canonical form (baseline "
        f"{NEEDS_NORMALIZATION_BASELINE}); a change added Node Backlog.md sections. Use backlog-py, not the "
        f"Node `backlog` CLI/MCP, and normalize the new files with: {_FIX_HINT}"
    )
