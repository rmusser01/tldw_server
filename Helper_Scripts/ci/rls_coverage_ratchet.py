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

SCANNED_SUFFIXES = (".py", ".sql")


class RatchetError(AssertionError):
    """Raised when coverage regresses against the recorded baseline."""


@dataclass(frozen=True)
class CoverageReport:
    owned_tables: frozenset[str]
    policy_tables: frozenset[str]

    @property
    def uncovered(self) -> frozenset[str]:
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
    policied: set[str] = set()

    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if path.suffix not in SCANNED_SUFFIXES or not path.is_file():
                continue
            try:
                text = path.read_text(errors="ignore")
            except OSError:
                continue

            for match in _CREATE_POLICY_RE.finditer(text):
                policied.add(_normalise(match.group(1)))

            for match in _CREATE_TABLE_RE.finditer(text):
                body = _column_list(text, match.end() - 1)
                if _OWNER_RE.search(body):
                    owned.add(_normalise(match.group(1)))

    return CoverageReport(frozenset(owned), frozenset(policied))


BASELINE_HEADER = """\
# rls_coverage_baseline.txt - tenant-owned tables with no RLS policy at guard
# introduction. The guard stays green on these but fails on a NEWLY uncovered
# table. Shrink this list by adding policies in
# tldw_Server_API/app/core/DB_Management/backends/pg_rls_policies.py.
# Regenerate with: python Helper_Scripts/ci/rls_coverage_ratchet.py --write-baseline
"""


def load_baseline(path: Path) -> frozenset[str]:
    names = []
    for line in path.read_text().splitlines():
        entry = line.strip()
        if entry and not entry.startswith("#"):
            names.append(_normalise(entry))
    return frozenset(names)


def write_baseline(path: Path, uncovered: frozenset[str]) -> None:
    path.write_text(BASELINE_HEADER + "\n".join(sorted(uncovered)) + "\n")


def compare(report: CoverageReport, baseline: frozenset[str]) -> tuple[frozenset[str], frozenset[str]]:
    """Return (newly uncovered, newly covered) against the baseline."""
    uncovered = report.uncovered
    return frozenset(uncovered - baseline), frozenset(baseline - uncovered)


def enforce(report: CoverageReport, baseline: frozenset[str]) -> frozenset[str]:
    """Raise when new uncovered tables appeared; return the ones now covered."""
    regressions, improvements = compare(report, baseline)
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
            "add it to the baseline and say why in the pull request."
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
    if args.write_baseline:
        write_baseline(DEFAULT_BASELINE, report.uncovered)
        print(f"Wrote {len(report.uncovered)} entries to {DEFAULT_BASELINE}")
        return 0

    try:
        improvements = enforce(report, load_baseline(DEFAULT_BASELINE))
    except RatchetError as exc:
        print(exc)
        return 1
    if improvements:
        print(
            f"{len(improvements)} table(s) gained a policy. Re-run with "
            "--write-baseline to hold the better position."
        )
    print(f"{len(report.uncovered)} tenant-owned tables still lack an RLS policy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
