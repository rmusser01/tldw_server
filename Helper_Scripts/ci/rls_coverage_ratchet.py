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
"""

from __future__ import annotations

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
# CREATE TABLE forms whose name is built at runtime, e.g. psycopg's
# sql.Identifier composition. The name cannot be resolved statically, so these
# are reported rather than ignored: silently skipping them is how a new
# tenant-owned table would slip past this guard entirely.
_DYNAMIC_CREATE_TABLE_RE = re.compile(
    r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?[{%$]",
    re.IGNORECASE,
)

SCANNED_SUFFIXES = (".py", ".sql")


class RatchetError(AssertionError):
    """Raised when coverage regresses against the recorded baseline."""


@dataclass(frozen=True)
class CoverageReport:
    """What one scan of the source tree found.

    Attributes:
        owned_tables: Tables whose CREATE TABLE names an ownership column, and
            which therefore hold rows belonging to a particular account.
        policy_tables: Tables that both have a CREATE POLICY and have row-level
            security switched on. A policy alone is not protection, so both are
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


def scan_source(roots: list[Path]) -> CoverageReport:
    """Collect tenant-owned table names and tables carrying an RLS policy."""
    owned: set[str] = set()
    with_policy: set[str] = set()
    rls_enabled: set[str] = set()
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

            for match in _CREATE_TABLE_RE.finditer(text):
                body = _column_list(text, match.end() - 1)
                if _OWNER_RE.search(body):
                    owned.add(_normalise(match.group(1)))

            for match in _DYNAMIC_CREATE_TABLE_RE.finditer(text):
                line = text[: match.start()].count("\n") + 1
                dynamic.append(f"{path}:{line}")

    # Coverage means a policy AND row-level security switched on for the table.
    policied = with_policy & rls_enabled
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
