"""Ratchet: endpoints must not grow new raw SQL.

Docs/Architecture.md states "no raw SQL in endpoints", and CLAUDE.md repeats it as
"Use /app/core/DB_Management/ abstractions (no raw SQL outside)". The 2026-09-21
core-module review measured the violation at 20 files; this test freezes the current
count per file so it can only go down.

What counts, precisely. A SQL string literal (or an f-string whose literal head is
SQL, or a module-level name bound to one) passed as the FIRST argument to a call
named execute/executemany/executescript/fetchone/fetchall/fetchval/fetch/_execute/
execute_query/query/exec_driver_sql/run.

That definition is deliberately narrower than "a file mentions SELECT". A grep-shaped
count reports 98 files and 470 statements, but most of those are docstrings, error
messages, OpenAPI examples, and text2sql.py -- which is an endpoint *about* SQL and
does not execute any of its own. Executed SQL is the thing the rule is about, and
that is 22 files.

Two related problems the review found in the same code, NOT ratcheted here because a
count is the wrong instrument for them:

* Ten of these files hand-roll the SQLite/PostgreSQL dialect branch. That is why
  core/AuthNZ/database.py:1966 _normalize_sqlite_sql exists at all; its own docstring
  calls itself a safety net for when a dollar-style query slips through.
* Five sites reach past a core module's private API -- jm._connect, jm._pg_cursor,
  vector_store_batches_db._connect, PromptStudioDatabase._execute and
  Collections_DB._coerce_bool_flag -- so those owners cannot refactor their internals
  without breaking an endpoint. A broad AST sweep for private-attribute access from
  endpoints finds 147 reaches across 54 files, but most are endpoint-to-endpoint
  aliases rather than core reaches, so the number needs the refactor's judgement
  rather than a frozen total.

The review checked every f-string site individually and found no injectable
interpolation and no DDL in endpoints: the scope here is layering, not security.

Companion to test_core_to_api_import_boundary.py and
test_endpoint_auth_deps_import_boundary.py, which guard the import direction.

TASK-13317 tracks draining this baseline.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
ENDPOINTS_ROOT = REPO_ROOT / "tldw_Server_API" / "app" / "api" / "v1" / "endpoints"

_SQL_HEAD = re.compile(
    r"^\s*(SELECT|INSERT\s+INTO|UPDATE|DELETE\s+FROM|CREATE\s+(TABLE|INDEX|UNIQUE)"
    r"|ALTER\s+TABLE|DROP\s+(TABLE|INDEX)|WITH)\b",
    re.IGNORECASE,
)

# Call names that execute their first argument as SQL.
_EXECUTORS = frozenset({
    "execute", "executemany", "executescript",
    "fetchone", "fetchall", "fetchval", "fetch",
    "_execute", "execute_query", "query", "exec_driver_sql", "run",
})

# Frozen at the 2026-09-21 review. Entries may shrink or disappear, never grow, and a
# file that reaches zero must be deleted from this map -- that is what keeps it honest.
RAW_SQL_BASELINE: dict[str, int] = {
}


def _looks_like_sql(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return bool(_SQL_HEAD.match(node.value))
    if isinstance(node, ast.JoinedStr):
        head = "".join(
            part.value
            for part in node.values
            if isinstance(part, ast.Constant) and isinstance(part.value, str)
        )
        return bool(_SQL_HEAD.match(head))
    return False


def _executed_sql_count(tree: ast.AST) -> int:
    """Count SQL literals passed as the first argument to an executor call."""
    sql_names = {
        target.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign) and _looks_like_sql(node.value)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    count = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        func = node.func
        name = (
            func.attr
            if isinstance(func, ast.Attribute)
            else func.id
            if isinstance(func, ast.Name)
            else None
        )
        if name not in _EXECUTORS:
            continue
        first = node.args[0]
        if _looks_like_sql(first) or (
            isinstance(first, ast.Name) and first.id in sql_names
        ):
            count += 1
    return count


def _measure() -> dict[str, int]:
    found: dict[str, int] = {}
    for path in sorted(ENDPOINTS_ROOT.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:  # pragma: no cover - a broken file fails elsewhere
            continue
        count = _executed_sql_count(tree)
        if count:
            found[str(path.relative_to(ENDPOINTS_ROOT))] = count
    return found


def test_no_endpoint_gains_raw_sql() -> None:
    """A file not in the baseline must not start executing SQL."""
    found = _measure()
    new_files = sorted(set(found) - set(RAW_SQL_BASELINE))
    assert not new_files, (  # nosec B101
        "These endpoint files newly execute raw SQL. Docs/Architecture.md says "
        "endpoints must not: route the query through its core DB_Management owner "
        f"instead of adding it here. {new_files}"
    )


def test_no_endpoint_executes_more_sql_than_its_baseline() -> None:
    """An existing offender must not get worse."""
    found = _measure()
    worse = {
        name: (RAW_SQL_BASELINE[name], count)
        for name, count in found.items()
        if name in RAW_SQL_BASELINE and count > RAW_SQL_BASELINE[name]
    }
    assert not worse, (  # nosec B101
        "These endpoint files execute MORE raw SQL than the frozen baseline "
        f"(baseline, now): {worse}"
    )


def test_baseline_has_no_stale_entries() -> None:
    """A file that no longer executes SQL must be removed from the baseline.

    Without this the map would silently record debt that has already been paid, and
    the next person could not tell how much is left.
    """
    found = _measure()
    stale = sorted(set(RAW_SQL_BASELINE) - set(found))
    assert not stale, (  # nosec B101
        "These files no longer execute raw SQL. Delete them from RAW_SQL_BASELINE: "
        f"{stale}"
    )


def test_baseline_counts_are_exact() -> None:
    """Report progress: a file that improved must have its baseline lowered."""
    found = _measure()
    improved = {
        name: (RAW_SQL_BASELINE[name], found[name])
        for name in sorted(set(RAW_SQL_BASELINE) & set(found))
        if found[name] < RAW_SQL_BASELINE[name]
    }
    assert not improved, (  # nosec B101
        "These files execute LESS raw SQL than the baseline, which is good. Lower "
        f"their entries in RAW_SQL_BASELINE to lock the gain in (baseline, now): {improved}"
    )


# --- Database-internals reaches (TASK-13317 AC3) ------------------------------------
# Endpoints must not open connections or cursors on a core owner's behalf; the owner
# exposes a method instead. These private names are the ones that did exactly that.
_DB_INTERNALS = frozenset({"_connect", "_pg_cursor", "_execute", "_coerce_bool_flag", "_cursor_exec"})


def test_no_endpoint_reaches_database_internals() -> None:
    offenders = []
    for path in sorted(ENDPOINTS_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            name = None
            if isinstance(node, ast.Attribute) and node.attr in _DB_INTERNALS:
                # self._x inside an endpoint's own helper class is not a core reach.
                if not (isinstance(node.value, ast.Name) and node.value.id == "self"):
                    name = node.attr
            elif isinstance(node, ast.ImportFrom) and (node.module or "").startswith("tldw_Server_API.app.core"):
                name = next((a.name for a in node.names if a.name in _DB_INTERNALS), None)
            if name:
                offenders.append(f"{path.relative_to(ENDPOINTS_ROOT)}:{node.lineno} {name}")
    assert not offenders, offenders
