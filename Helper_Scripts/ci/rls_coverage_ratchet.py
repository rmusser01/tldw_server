"""Ratchet for row-level-security coverage over tenant-owned tables.

Isolation in the PostgreSQL deployment rests on two things: the WHERE clause a
developer remembered to type, and an RLS policy as a backstop when they did
not. Coverage of the second is currently a hand-maintained allowlist, so a new
table carrying a user column ships with no database-level protection and
nothing fails.

This module turns that into a ratchet. It scans the source for CREATE TABLE
statements whose column list names an ownership column, cross-references every
CREATE POLICY in the tree, and reports the tables left uncovered. A recorded
baseline pins today's set: adding a new uncovered table fails, and covering an
existing one is expected to shrink the baseline.

It is deliberately a static scan rather than a live `information_schema` query
so that it runs in ordinary CI with no database, which is where new tables
actually arrive.

Python sources are also read through the AST, because policies are often built
from f-string placeholders: ``add_tenant_policy("notes", ...)`` or
``for table in ("a", "b"): add(f"CREATE POLICY ... ON {table}")``. A regex over
the text never sees the table name next to CREATE POLICY there, so those tables
would read as unprotected forever.
"""

from __future__ import annotations

import ast
import itertools
import re
from dataclasses import dataclass
from pathlib import Path

# Columns that mark a row as belonging to one account.
OWNERSHIP_COLUMNS = (
    "user_id",
    "owner_user_id",
    "owner_id",
    "client_id",
    "created_by",
    "tenant_id",
    "account_id",
)

_OWNER_RE = re.compile(r"\b(" + "|".join(OWNERSHIP_COLUMNS) + r")\b", re.IGNORECASE)
_CREATE_TABLE_RE = re.compile(
    r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?[\"'`]?([A-Za-z_0-9.]+)[\"'`]?\s*\(",
    re.IGNORECASE,
)
_CREATE_POLICY_RE = re.compile(
    r"CREATE\s+POLICY\s+[A-Za-z_0-9]+\s+ON\s+[\"'`]?([A-Za-z_0-9.]+)[\"'`]?",
    re.IGNORECASE,
)
# A policy only isolates anything once the table has RLS switched on, and FORCE
# is what makes it bind the table owner. A CREATE POLICY with neither is text,
# not protection, so coverage requires both to appear for the same table.
_ENABLE_RLS_RE = re.compile(
    r"ALTER\s+TABLE\s+(?:IF\s+EXISTS\s+)?[\"'`]?([A-Za-z_0-9.]+)[\"'`]?\s+"
    r"ENABLE\s+ROW\s+LEVEL\s+SECURITY",
    re.IGNORECASE,
)
_FORCE_RLS_RE = re.compile(
    r"ALTER\s+TABLE\s+(?:IF\s+EXISTS\s+)?[\"'`]?([A-Za-z_0-9.]+)[\"'`]?\s+"
    r"FORCE\s+ROW\s+LEVEL\s+SECURITY",
    re.IGNORECASE,
)
# CREATE TABLE forms whose name is built at runtime, e.g. psycopg's
# sql.Identifier composition. The name cannot be resolved statically, so these
# are reported rather than ignored: silently skipping them is how a new
# tenant-owned table would slip past this guard entirely.
_DYNAMIC_CREATE_TABLE_RE = re.compile(
    r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?[{%$]",
    re.IGNORECASE,
)

SCANNED_SUFFIXES = (".py", ".sql")

# Only modules mentioning RLS can build a policy; skipping the rest keeps the
# AST pass off the thousands of files that cannot matter.
_RLS_HINT_RE = re.compile(r"CREATE\s+POLICY|ROW\s+LEVEL\s+SECURITY", re.IGNORECASE)
# Rendered in place of a placeholder the scan cannot resolve. No identifier
# pattern above matches it, so an unknown name never counts as coverage.
_UNRESOLVED = "\x00"
_PARAM_SENTINEL = "__rls_param_{}__"


class RatchetError(AssertionError):
    """Raised when coverage regresses against the recorded baseline."""


@dataclass(frozen=True)
class CoverageReport:
    """What one scan of the source tree found.

    Attributes:
        owned_tables: Tables whose CREATE TABLE names an ownership column, and
            which therefore hold rows belonging to a particular account.
        policy_tables: Tables that both have a CREATE POLICY and have row-level
            security enabled and forced for the owner. A policy alone is not protection; all three are
            required before a table counts as covered.
        dynamic_table_sites: ``path:line`` for CREATE TABLE statements whose
            name is built at runtime. A static scan cannot resolve those, so
            they are reported rather than silently omitted -- if one of them is
            tenant-owned, this guard cannot see it.
    """

    owned_tables: frozenset[str]
    policy_tables: frozenset[str]
    dynamic_table_sites: tuple[str, ...] = ()

    @property
    def uncovered(self) -> frozenset[str]:
        """Tenant-owned tables with no enforced policy, i.e. no DB-level backstop."""
        return frozenset(self.owned_tables - self.policy_tables)


def _normalise(name: str) -> str:
    return name.split(".")[-1].strip().lower()


def _column_list(text: str, open_paren_index: int) -> str:
    """Return the balanced parenthesised body starting at ``open_paren_index``."""
    depth = 0
    for index in range(open_paren_index, len(text)):
        char = text[index]
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return text[open_paren_index + 1 : index]
    return text[open_paren_index + 1 :]


def _literal_strings(node: ast.AST | None, scopes: list[ast.AST]) -> list[str] | None:
    """String constants of a tuple/list/set literal or dict keys; None if not all literal."""
    if isinstance(node, ast.Name):
        node = _assigned_value(node.id, scopes)
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        items: list[ast.expr | None] = list(node.elts)
    elif isinstance(node, ast.Dict):
        items = list(node.keys)
    else:
        return None
    values = [
        item.value
        for item in items
        if isinstance(item, ast.Constant) and isinstance(item.value, str)
    ]
    return values if len(values) == len(items) else None


def _assigned_value(name: str, scopes: list[ast.AST]) -> ast.AST | None:
    """The value of the single assignment to ``name``, innermost scope first."""
    for scope in scopes:
        nodes = ast.walk(scope) if not isinstance(scope, ast.Module) else iter(scope.body)
        values = [
            node.value
            for node in nodes
            if isinstance(node, (ast.Assign, ast.AnnAssign))
            and node.value is not None
            and any(
                isinstance(target, ast.Name) and target.id == name
                for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
            )
        ]
        if values:
            return values[0] if len(values) == 1 else None
    return None


def _loop_bindings(loop: ast.For, scopes: list[ast.AST]) -> list[dict[str, str]] | None:
    """One {name: value} per iteration of a loop over literal strings, else None.

    Handles ``for t in (...)`` and ``for t, p in d.items()``, where the iterable
    is a literal or a name assigned one.
    """
    target, source = loop.target, loop.iter
    if (
        isinstance(source, ast.Call)
        and isinstance(source.func, ast.Attribute)
        and source.func.attr == "items"
        and isinstance(target, ast.Tuple)
    ):
        source, target = source.func.value, target.elts[0]
    values = _literal_strings(source, scopes)
    if values is None or not isinstance(target, ast.Name):
        return None
    return [{target.id: value} for value in values]


def _render(node: ast.JoinedStr, binding: dict[str, str]) -> str:
    parts = []
    for value in node.values:
        if isinstance(value, ast.Constant):
            parts.append(str(value.value))
        elif isinstance(value, ast.FormattedValue) and isinstance(value.value, ast.Name):
            parts.append(binding.get(value.value.id, _UNRESOLVED))
        else:
            parts.append(_UNRESOLVED)
    return "".join(parts)


_POLICY_KINDS = (
    ("policy", _CREATE_POLICY_RE),
    ("enabled", _ENABLE_RLS_RE),
    ("forced", _FORCE_RLS_RE),
)


def _scan_python_policies(text: str, path: Path, found: dict[str, set[str]]) -> None:
    """Add tables whose policy DDL is built from f-string placeholders.

    Two shapes are resolved: an f-string inside loops over literal names, and
    a helper whose f-strings put a parameter in the table position, credited
    with each literal (or loop-bound) argument it is called with in the module.
    Anything else stays unresolved and earns no coverage.
    """
    try:
        tree = ast.parse(text, filename=str(path))
    except SyntaxError as exc:
        raise RatchetError(
            f"Could not parse {path} while scanning for RLS policies. Coverage "
            "cannot be established from partial input."
        ) from exc

    parents: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent

    def context(
        node: ast.AST,
    ) -> tuple[list[dict[str, str]], list[ast.AST], ast.FunctionDef | ast.AsyncFunctionDef | None]:
        """Loop bindings around ``node``, its enclosing scopes, and its function."""
        loops: list[ast.For] = []
        scopes: list[ast.AST] = []
        function: ast.FunctionDef | ast.AsyncFunctionDef | None = None
        current = parents.get(node)
        while current is not None:
            if isinstance(current, ast.For) and not scopes:
                loops.append(current)
            elif isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Module)):
                scopes.append(current)
                if function is None and not isinstance(current, ast.Module):
                    function = current
            current = parents.get(current)
        per_loop = [b for b in (_loop_bindings(loop, scopes) for loop in loops) if b is not None]
        bindings = [
            {key: value for binding in combo for key, value in binding.items()}
            for combo in itertools.product(*per_loop)
        ]
        return bindings, scopes, function

    # helper name -> [(parameter index, parameter name, kind)]
    helpers: dict[str, list[tuple[int, str, str]]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.JoinedStr):
            continue
        bindings, _scopes, function = context(node)
        # sentinel -> (index, name) for the enclosing function's positional parameters
        params: dict[str, tuple[int, str]] = {}
        if function is not None:
            positional = function.args.posonlyargs + function.args.args
            params = {
                _PARAM_SENTINEL.format(i): (i, arg.arg) for i, arg in enumerate(positional)
            }
        as_placeholders = {name: sentinel for sentinel, (_i, name) in params.items()}
        for binding in bindings:
            rendered = _render(node, {**as_placeholders, **binding})
            for kind, pattern in _POLICY_KINDS:
                for match in pattern.finditer(rendered):
                    table = _normalise(match.group(1))
                    if table in params and function is not None:
                        index, name = params[table]
                        helpers.setdefault(function.name, []).append((index, name, kind))
                    elif not table.startswith("__rls_param_"):
                        found[kind].add(table)

    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
            continue
        facts = helpers.get(node.func.id)
        if not facts:
            continue
        bindings, _scopes, _function = context(node)
        for index, name, kind in facts:
            argument = node.args[index] if index < len(node.args) else next(
                (kw.value for kw in node.keywords if kw.arg == name), None
            )
            if isinstance(argument, ast.Starred):
                continue
            if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                found[kind].add(_normalise(argument.value))
            elif isinstance(argument, ast.Name):
                found[kind].update(
                    _normalise(b[argument.id]) for b in bindings if argument.id in b
                )


def scan_source(roots: list[Path]) -> CoverageReport:
    """Collect tenant-owned table names and tables carrying an RLS policy."""
    owned: set[str] = set()
    with_policy: set[str] = set()
    rls_enabled: set[str] = set()
    rls_forced: set[str] = set()
    dynamic: list[str] = []

    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if path.suffix not in SCANNED_SUFFIXES or not path.is_file():
                continue
            try:
                text = path.read_text(errors="ignore")
            except OSError as exc:
                # Skipping an unreadable file would drop its tables and its
                # policies from the report, so the guard could pass on partial
                # input. Refuse instead.
                raise RatchetError(
                    f"Could not read {path} while scanning for tenant-owned "
                    "tables. Coverage cannot be established from partial input."
                ) from exc

            for match in _CREATE_POLICY_RE.finditer(text):
                with_policy.add(_normalise(match.group(1)))

            for match in _ENABLE_RLS_RE.finditer(text):
                rls_enabled.add(_normalise(match.group(1)))

            for match in _FORCE_RLS_RE.finditer(text):
                rls_forced.add(_normalise(match.group(1)))

            if path.suffix == ".py" and _RLS_HINT_RE.search(text):
                _scan_python_policies(
                    text,
                    path,
                    {"policy": with_policy, "enabled": rls_enabled, "forced": rls_forced},
                )

            for match in _CREATE_TABLE_RE.finditer(text):
                body = _column_list(text, match.end() - 1)
                if _OWNER_RE.search(body):
                    owned.add(_normalise(match.group(1)))

            for match in _DYNAMIC_CREATE_TABLE_RE.finditer(text):
                line = text[: match.start()].count("\n") + 1
                dynamic.append(f"{path}:{line}")

    # Coverage requires a policy, ENABLE, and FORCE to bind the table owner.
    policied = with_policy & rls_enabled & rls_forced
    return CoverageReport(
        frozenset(owned), frozenset(policied), tuple(sorted(dynamic))
    )


BASELINE_HEADER = """\
# rls_coverage_baseline.txt - tenant-owned tables with no RLS policy at guard
# introduction. The guard stays green on these but fails on a NEWLY uncovered
# table. Shrink this list by adding policies in
# tldw_Server_API/app/core/DB_Management/backends/pg_rls_policies.py.
# Regenerate with: python Helper_Scripts/ci/rls_coverage_ratchet.py --write-baseline
"""


DEFAULT_EXEMPTIONS = Path(__file__).resolve().parent / "rls_coverage_exemptions.txt"


def load_exemptions(path: Path) -> dict[str, str]:
    """Return {table: reason} for tables that must not get an owner-scoped policy.

    Distinct from the baseline: a baseline entry is work not yet done, an
    exemption is a decision that the obvious policy would break the feature.
    Every entry must carry a reason, so the decision survives past the pull
    request that made it.

    Raises ``RatchetError`` if the file is unreadable or an entry has no reason.
    """
    try:
        text = path.read_text()
    except OSError as exc:
        raise RatchetError(f"could not read the exemptions at {path}: {exc}") from exc

    exemptions: dict[str, str] = {}
    for lineno, line in enumerate(text.splitlines(), start=1):
        entry = line.strip()
        if not entry or entry.startswith("#"):
            continue
        table, separator, reason = entry.partition(":")
        if not separator or not reason.strip():
            raise RatchetError(
                f"{path}:{lineno}: exemption '{entry}' has no reason. "
                "Write 'table_name: why the obvious policy breaks the feature'."
            )
        exemptions[_normalise(table)] = reason.strip()
    return exemptions


def load_baseline(path: Path) -> frozenset[str]:
    names = []
    for line in path.read_text().splitlines():
        entry = line.strip()
        if entry and not entry.startswith("#"):
            names.append(_normalise(entry))
    return frozenset(names)


def write_baseline(path: Path, uncovered: frozenset[str]) -> None:
    path.write_text(BASELINE_HEADER + "\n".join(sorted(uncovered)) + "\n")


def compare(
    report: CoverageReport,
    baseline: frozenset[str],
    exemptions: dict[str, str] | None = None,
) -> tuple[frozenset[str], frozenset[str]]:
    """Return (newly uncovered, newly covered) against the baseline.

    Exempted tables are accounted for separately and take part in neither: they
    are a standing decision, not work in progress.
    """
    uncovered = report.uncovered - frozenset(exemptions or {})
    covered_baseline = baseline - frozenset(exemptions or {})
    return frozenset(uncovered - covered_baseline), frozenset(covered_baseline - uncovered)


def enforce(
    report: CoverageReport,
    baseline: frozenset[str],
    exemptions: dict[str, str] | None = None,
) -> frozenset[str]:
    """Raise when new uncovered tables appeared; return the ones now covered."""
    regressions, improvements = compare(report, baseline, exemptions)
    if regressions:
        listed = "\n  ".join(sorted(regressions))
        raise RatchetError(
            "These tables carry an ownership column but have no RLS policy:\n  "
            f"{listed}\n\n"
            "On PostgreSQL every user's rows share these tables, so the only "
            "thing keeping one account out of another's data is the WHERE "
            "clause each query happens to carry. Add a policy (see "
            "app/core/DB_Management/backends/pg_rls_policies.py). If the table "
            "is genuinely global -- a catalog, migration bookkeeping, or an "
            "authentication table that must be read before a tenant is known -- "
            "add it to the baseline. If the obvious policy would actively break "
            "the feature, as it does for anonymous share-link redemption, add it "
            "to rls_coverage_exemptions.txt with the reason, so the decision "
            "outlives the pull request that made it."
        )
    return improvements


DEFAULT_APP_ROOT = Path(__file__).resolve().parents[2] / "tldw_Server_API" / "app"
DEFAULT_BASELINE = Path(__file__).resolve().parent / "rls_coverage_baseline.txt"


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write-baseline",
        action="store_true",
        help="Record the current uncovered set instead of checking against it.",
    )
    args = parser.parse_args(argv)

    report = scan_source([DEFAULT_APP_ROOT])
    try:
        exemptions = load_exemptions(DEFAULT_EXEMPTIONS)
    except RatchetError as exc:
        print(exc)
        return 1

    if args.write_baseline:
        recorded = report.uncovered - frozenset(exemptions)
        write_baseline(DEFAULT_BASELINE, recorded)
        print(
            f"Wrote {len(recorded)} entries to {DEFAULT_BASELINE} "
            f"({len(exemptions)} exempted)"
        )
        return 0

    try:
        improvements = enforce(report, load_baseline(DEFAULT_BASELINE), exemptions)
    except RatchetError as exc:
        print(exc)
        return 1
    if improvements:
        print(
            f"{len(improvements)} table(s) gained a policy. Re-run with "
            "--write-baseline to hold the better position."
        )
    if report.dynamic_table_sites:
        print(
            f"note: {len(report.dynamic_table_sites)} CREATE TABLE statements build "
            "their name at runtime and cannot be checked by a static scan. If any "
            "of them is tenant-owned, this guard will not see it:"
        )
        for site in report.dynamic_table_sites[:10]:
            print(f"  {site}")
    outstanding = len(report.uncovered) - len(exemptions)
    print(
        f"{outstanding} tenant-owned tables still lack an RLS policy; "
        f"{len(exemptions)} exempted with a recorded reason."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
