"""Classify a NUL-delimited git diff for the opt-in server smoke tier."""

from __future__ import annotations

import sys

_MAPPED = frozenset(b"tldw_Server_API/app/core/Chunking/strategies/" + file for file in (b"words.py", b"paragraphs.py"))
_DOC_ROOTS = (b"Docs/", b"backlog/tasks/", b"backlog/completed/")


def select_tier(diff: bytes) -> str:
    """Choose light only for reviewed edits, and keep unknown changes broad."""
    fields = diff.split(b"\0")
    if len(fields) < 3 or fields[-1] or len(fields) % 2 == 0:
        return "full"
    mapped = False
    for status, path in zip(fields[:-1:2], fields[1:-1:2], strict=True):
        if status == b"M" and path in _MAPPED:
            mapped = True
        elif not (status in {b"A", b"M"} and path.startswith(_DOC_ROOTS) and path.endswith(b".md")):
            return "full"
    return "light" if mapped else "unchanged"


if __name__ == "__main__":
    print(select_tier(sys.stdin.buffer.read()))
