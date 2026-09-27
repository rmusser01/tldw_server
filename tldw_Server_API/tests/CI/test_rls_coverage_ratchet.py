"""New tenant-owned tables must not ship without an RLS policy.

On SQLite each user gets their own database file, so a forgotten
``WHERE user_id = ?`` is harmless. On PostgreSQL every user's rows share one
table and the same query leaks. Today RLS covers a small fraction of the tables
that carry an ownership column, and nothing fails when that fraction shrinks.

This ratchet pins the current set. It may shrink; it may not grow. Covering a
table is expected to make this test tell you to update the baseline.

Runs unconditionally and needs no database -- it is a static scan, because new
tables arrive in pull requests, not in a running cluster.
"""

from pathlib import Path

import pytest
from Helper_Scripts.ci.rls_coverage_ratchet import (
    DEFAULT_BASELINE,
    DEFAULT_EXEMPTIONS,
    OWNERSHIP_COLUMNS,
    CoverageReport,
    RatchetError,
    compare,
    enforce,
    load_baseline,
    load_exemptions,
    scan_source,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]


def _write(sql: str):
    """Write one throwaway .sql file and return the root to scan."""
    import tempfile

    root = Path(tempfile.mkdtemp())
    (root / "schema.sql").write_text(sql)
    return root

APP_ROOT = REPO_ROOT / "tldw_Server_API" / "app"
BASELINE = REPO_ROOT / "Helper_Scripts" / "ci" / "rls_coverage_baseline.txt"


@pytest.fixture(scope="module")
def report() -> CoverageReport:
    return scan_source([APP_ROOT])


def test_no_new_tenant_owned_table_lacks_an_rls_policy(report):
    """The gate. A new uncovered table fails here instead of leaking in prod."""
    baseline = load_baseline(BASELINE)
    exemptions = load_exemptions(DEFAULT_EXEMPTIONS)

    newly_covered = enforce(report, baseline, exemptions)

    if newly_covered:
        listed = ", ".join(sorted(newly_covered))
        pytest.fail(
            f"These tables now have an RLS policy: {listed}. "
            "Remove them from rls_coverage_baseline.txt so the ratchet holds "
            "the new, better position."
        )


def test_scan_finds_the_tables_and_policies_that_exist(report):
    """Guards the scanner itself; a silently empty scan would pass everything."""
    assert len(report.owned_tables) > 100
    assert len(report.policy_tables) > 5
    # Prompt Studio is the reference policy set in the tree.
    assert "prompt_studio_projects" in report.policy_tables


def test_baseline_only_lists_tables_the_scan_still_finds():
    """A stale baseline entry hides a table that was renamed or dropped."""
    current = scan_source([APP_ROOT])
    stale = load_baseline(BASELINE) - current.owned_tables
    if stale:
        listed = ", ".join(sorted(stale))
        pytest.fail(
            f"Baseline lists tables that no longer carry an ownership column: {listed}. "
            "Remove them from rls_coverage_baseline.txt."
        )


def test_ratchet_rejects_a_newly_uncovered_table():
    """Proves the gate is load-bearing rather than vacuously green."""
    report = CoverageReport(
        owned_tables=frozenset({"already_known", "brand_new_leak"}),
        policy_tables=frozenset(),
    )

    with pytest.raises(RatchetError, match="brand_new_leak"):
        enforce(report, frozenset({"already_known"}))


def test_ratchet_accepts_a_table_that_gained_a_policy():
    report = CoverageReport(
        owned_tables=frozenset({"now_protected"}),
        policy_tables=frozenset({"now_protected"}),
    )

    regressions, improvements = compare(report, frozenset({"now_protected"}))

    assert not regressions
    assert improvements == frozenset({"now_protected"})


def test_ownership_columns_cover_the_conventions_used_in_this_repo():
    for column in ("user_id", "owner_user_id", "client_id", "created_by"):
        assert column in OWNERSHIP_COLUMNS


def test_a_policy_without_enabled_rls_is_not_coverage():
    """A CREATE POLICY on a table with RLS switched off protects nothing.

    The scanner previously counted any policy-shaped text as coverage, so an
    assertion string or a policy created without ENABLE ROW LEVEL SECURITY
    removed a table from the uncovered set while giving no isolation.
    """
    report = scan_source([_write(
        "CREATE TABLE thing (id INT, user_id TEXT);\n"
        "CREATE POLICY thing_iso ON thing USING (true);\n"
    )])

    assert "thing" in report.owned_tables
    assert "thing" not in report.policy_tables
    assert "thing" in report.uncovered


def test_a_policy_with_enabled_rls_is_coverage():
    report = scan_source([_write(
        "CREATE TABLE thing (id INT, user_id TEXT);\n"
        "ALTER TABLE thing ENABLE ROW LEVEL SECURITY;\n"
        "CREATE POLICY thing_iso ON thing USING (true);\n"
    )])

    assert "thing" in report.policy_tables
    assert "thing" not in report.uncovered


def test_runtime_built_table_names_are_reported_not_ignored():
    """A name composed at runtime cannot be checked, so it must be surfaced."""
    report = scan_source([_write(
        'cur.execute(f"CREATE TABLE {ident(name)} (id INT, user_id TEXT)")\n'
    )])

    assert report.dynamic_table_sites, "a dynamic CREATE TABLE must be reported"


def test_an_unreadable_source_file_fails_instead_of_being_skipped(tmp_path):
    """Partial input must not be able to produce a passing coverage report."""
    root = tmp_path / "src"
    root.mkdir()
    bad = root / "unreadable.sql"
    bad.write_text("CREATE TABLE t (id INT, user_id TEXT);")
    bad.chmod(0o000)
    try:
        with pytest.raises(RatchetError, match="Could not read"):
            scan_source([root])
    finally:
        bad.chmod(0o644)


class TestExemptions:
    """Exemptions are decisions, and a decision without a reason is a guess."""

    def test_every_exemption_carries_a_reason(self) -> None:
        exemptions = load_exemptions(DEFAULT_EXEMPTIONS)
        assert exemptions, "the exemptions file should not be empty"
        for table, reason in exemptions.items():
            assert len(reason) > 40, f"{table} needs a real reason, got {reason!r}"

    def test_a_reasonless_exemption_is_refused(self, tmp_path) -> None:
        path = tmp_path / "exemptions.txt"
        path.write_text("# comment\nsome_table\n", encoding="utf-8")
        with pytest.raises(RatchetError) as exc_info:
            load_exemptions(path)
        assert "no reason" in str(exc_info.value)

    def test_an_unreadable_exemptions_file_fails_loudly(self, tmp_path) -> None:
        """Otherwise a missing file would silently exempt nothing, or everything."""
        with pytest.raises(RatchetError):
            load_exemptions(tmp_path / "does-not-exist.txt")

    def test_exemptions_and_baseline_do_not_overlap(self) -> None:
        """A table listed in both would look like work in progress and a decision."""
        exemptions = set(load_exemptions(DEFAULT_EXEMPTIONS))
        baseline = set(load_baseline(DEFAULT_BASELINE))
        assert not (exemptions & baseline), sorted(exemptions & baseline)

    def test_exempted_table_is_not_reported_as_a_regression(self) -> None:
        """The whole point: an exempted table never fails the guard."""
        report = CoverageReport(
            owned_tables=frozenset({"share_tokens"}), policy_tables=frozenset()
        )
        regressions, _ = compare(
            report, frozenset(), {"share_tokens": "anonymous redemption"}
        )
        assert regressions == frozenset()

    def test_a_table_with_no_exemption_still_fails(self) -> None:
        report = CoverageReport(
            owned_tables=frozenset({"some_new_table"}), policy_tables=frozenset()
        )
        regressions, _ = compare(report, frozenset(), {"share_tokens": "x"})
        assert regressions == frozenset({"some_new_table"})
