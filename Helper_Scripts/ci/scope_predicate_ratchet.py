#!/usr/bin/env python3
"""Fail when a new query makes its tenant predicate conditional.

    if user_id:
        conditions.append("user_id = ?")
        params.append(user_id)

A falsy scope -- ``None``, ``""``, ``0`` -- silently drops the predicate and
returns the whole table instead of raising.  On SQLite that is usually
harmless, because each user has their own file.  On shared PostgreSQL tables it
returns every tenant's rows.

Not every match is a defect.  ``list_all_schedules(user_id=None)`` is an
explicitly named unscoped read for a system worker, which is the shape this
codebase should use when an unscoped read is genuinely needed -- the name makes
it visible in review and in grep.  So this is a ratchet, not a ban: what exists
is recorded, and the list may shrink but never grow.

Detection is deliberately narrow.  It fires only when the guarded block adds a
SQL fragment comparing *that same* scope name, so ``if status:`` filters and
``if user_id:`` blocks that do something other than build a predicate are not
reported.
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path
from typing import Iterator

REPO_ROOT = Path(__file__).resolve().parents[2]
APP_ROOT = REPO_ROOT / "tldw_Server_API" / "app"
BASELINE_PATH = Path(__file__).resolve().parent / "scope_predicate_baseline.txt"

# Column names that carry the tenant boundary.
SCOPE_NAMES = frozenset(
    {
        "account_id",
        "client_id",
        "created_by",
        "owner_id",
        "owner_user_id",
        "tenant_id",
        "user_id",
    }
)


class RatchetError(RuntimeError):
    """Raised when the source tree cannot be read or parsed."""


def _guarded_scope(test: ast.expr) -> str | None:
    """Return the scope name for a bare truthiness test, else None.

    Only ``if user_id:`` and ``if self.user_id:`` count.  An explicit
    ``if user_id is not None:`` is a deliberate check, not the footgun.
    """
    if isinstance(test, ast.Name) and test.id in SCOPE_NAMES:
        return test.id
    if isinstance(test, ast.Attribute) and test.attr in SCOPE_NAMES:
        return test.attr
    return None


def _adds_predicate_on(block: list[ast.stmt], name: str) -> bool:
    """True when the block contributes a SQL comparison on *name*."""
    pattern = re.compile(rf"\b{re.escape(name)}\b\s*(=|==|!=|\bIN\b|\bLIKE\b)", re.IGNORECASE)
    for statement in block:
        for node in ast.walk(statement):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if pattern.search(node.value):
                    return True
            elif isinstance(node, ast.JoinedStr):
                for part in node.values:
                    if (
                        isinstance(part, ast.Constant)
                        and isinstance(part.value, str)
                        and pattern.search(part.value)
                    ):
                        return True
    return False


def iter_findings(root: Path = APP_ROOT) -> Iterator[str]:
    """Yield ``"relative/path.py:LINE if <scope>"`` for each conditional predicate."""
    for path in sorted(root.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise RatchetError(f"could not read {path}: {exc}") from exc
        except SyntaxError as exc:
            raise RatchetError(f"could not parse {path}: {exc}") from exc
        for node in ast.walk(tree):
            if not isinstance(node, ast.If):
                continue
            name = _guarded_scope(node.test)
            if name is None or not _adds_predicate_on(node.body, name):
                continue
            yield f"{path.relative_to(REPO_ROOT)}:{node.lineno} if {name}"


def read_baseline(path: Path = BASELINE_PATH) -> set[str]:
    """Return baseline entries, ignoring comments and blank lines."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RatchetError(f"could not read the baseline at {path}: {exc}") from exc
    return {
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.startswith("#")
    }


def diff_against_baseline(
    current: list[str] | set[str], baseline: set[str]
) -> tuple[list[str], list[str]]:
    """Return ``(added, stale)``, each sorted.

    Stale entries fail too: a line number that no longer holds a conditional
    predicate would otherwise keep covering whatever moves onto that line.
    """
    live = set(current)
    return sorted(live - baseline), sorted(baseline - live)


def main(argv: list[str] | None = None) -> int:
    """Return 0 when the baseline matches the tree exactly, else 1."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write-baseline",
        action="store_true",
        help="record the current conditional scope predicates as the baseline",
    )
    args = parser.parse_args(argv)

    current = sorted(iter_findings())

    if args.write_baseline:
        BASELINE_PATH.write_text(
            "# Queries whose tenant predicate is applied only when the scope is truthy.\n"
            "#\n"
            "# A falsy scope drops the predicate instead of raising. Harmless on\n"
            "# SQLite (one file per user), a full-table read on shared PostgreSQL.\n"
            "#\n"
            "# Some entries are correct: an explicitly named unscoped read for a\n"
            "# system worker, such as list_all_schedules(user_id=None), is the shape\n"
            "# to prefer when an unscoped read is genuinely needed.\n"
            "#\n"
            "# Line numbers move, so regenerate when you touch a listed file:\n"
            "#   python Helper_Scripts/ci/scope_predicate_ratchet.py --write-baseline\n"
            + "\n".join(current)
            + "\n",
            encoding="utf-8",
        )
        print(f"Recorded {len(current)} conditional scope predicates.")
        return 0

    added, stale = diff_against_baseline(current, read_baseline())
    if added:
        print("New conditional tenant predicates:", file=sys.stderr)
        for entry in added:
            print(f"  {entry}", file=sys.stderr)
        print(
            "\nMake the scope required and raise on a falsy value, or -- if the read\n"
            "is meant to be unscoped -- give it an explicitly named method and\n"
            "regenerate the baseline.",
            file=sys.stderr,
        )
    if stale:
        print("Baseline entries that no longer match the tree:", file=sys.stderr)
        for entry in stale:
            print(f"  {entry}", file=sys.stderr)
        print(
            "\nUsually a line number moved. Regenerate with --write-baseline.",
            file=sys.stderr,
        )
    if added or stale:
        return 1

    print(f"{len(current)} conditional scope predicates; baseline exact.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
