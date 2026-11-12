"""The merge helper must reject unfinished and stale release checks."""

import os
import shutil
import subprocess  # nosec B404 - executes a controlled shell fixture
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]


# Scenarios that must land; every other scenario must stop before merging.
LANDS = {"success", "success_no_merge", "advisory_failed", "advisory_failed_e2e_skipped"}


@pytest.mark.parametrize(
    "scenario",
    [
        "audit_pending", "audit_failed", "gate_pending", "stale_gate", "changed_head", "success", "success_no_merge",
        # frontend-e2e-tiers.yml's run also holds the advisory ux-regression job: only its
        # required `E2E Critical` check decides, and skipped passes as on GitHub.
        "advisory_failed", "advisory_failed_e2e_skipped", "e2e_failed", "e2e_check_missing",
        # The six other gates are still judged by their workflow run, not by a check run.
        "six_gate_run_failed",
    ],
)
def test_release_helper_requires_success_for_the_current_head(
    tmp_path: Path, scenario: str
) -> None:
    """Only finished checks for an unchanged PR head may authorize merging."""
    gh = tmp_path / "gh"
    gh.write_text("""#!/bin/sh
case "$1 $2" in
  'pr view')
    if [ "$SCENARIO" = changed_head ] && [ -f "$MARKER.view" ]; then echo new-head; else echo release-head; fi
    touch "$MARKER.view";;
  'run list')
    case "$*" in
      *'Frontend License Gate Audit'*)
        case "$SCENARIO" in
          audit_pending) echo '1 in_progress pending release-head';;
          audit_failed) echo '1 completed failure release-head';;
          *) echo '1 completed success release-head';;
        esac;;
      *)
        case "$SCENARIO:$*" in
          gate_pending:*) echo '2 in_progress pending release-head';;
          stale_gate:*) echo '2 completed success old-head';;
          six_gate_run_failed:*'--workflow backend-required '*) echo '2 completed failure release-head';;
          advisory_failed*:*'--workflow frontend-e2e-tiers.yml '*) echo '3 completed failure release-head';;
          e2e_*:*'--workflow frontend-e2e-tiers.yml '*) echo '3 completed failure release-head';;
          *) echo '2 completed success release-head';;
        esac;;
    esac;;
  'api -X')
    printf '%s\n' "$*" >> "$MARKER.api"
    case "$SCENARIO" in
      advisory_failed_e2e_skipped) echo skipped;;
      e2e_failed) echo failure;;
      e2e_check_missing) echo '';;
      *) echo success;;
    esac;;
  'run rerun') printf '%s\n' "$*" >> "$MARKER.rerun";;
  'pr merge') printf '%s' "$*" > "$MARKER";;
  *) exit 9;;
esac
""")
    gh.chmod(0o755)
    sleep = tmp_path / "sleep"
    sleep.write_text("#!/bin/sh\nexit 0\n")
    sleep.chmod(0o755)
    marker = tmp_path / "merged"
    shell = shutil.which("sh")
    if shell is None:
        pytest.skip("POSIX shell unavailable for this shell-helper regression")
    result = subprocess.run(  # nosec B603 - trusted local helper and fixed fixture arguments
        [
            shell,
            str(ROOT / "Helper_Scripts/ci/land_required_gates.sh"),
            "release",
            "3027",
            *(["--no-merge"] if scenario == "success_no_merge" else []),
        ],
        env={
            **os.environ,
            "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"],
            "SCENARIO": scenario,
            "MARKER": str(marker),
        },
        capture_output=True,
        text=True,
        timeout=30,
    )
    if scenario in LANDS:
        assert result.returncode == 0, result.stdout + result.stderr  # nosec B101 - regression assertion
        if scenario != "success_no_merge":
            assert "--match-head-commit release-head" in marker.read_text()  # nosec B101 - regression assertion
        else:
            assert not marker.exists()  # nosec B101 - regression assertion
    else:
        assert result.returncode != 0, result.stdout + result.stderr  # nosec B101 - regression assertion
        assert not marker.exists()  # nosec B101 - regression assertion
    if scenario.startswith("advisory_failed"):
        # The passing check is enough: the run that only the advisory job failed is not re-run.
        assert not Path(f"{marker}.rerun").exists()  # nosec B101 - regression assertion
        assert "check_name=E2E Critical (Tier 1 + Journeys)" in Path(f"{marker}.api").read_text()  # nosec B101 - regression assertion
    if scenario == "e2e_failed":
        assert "frontend-e2e-tiers.yml:failure" in result.stdout  # nosec B101 - regression assertion
    if scenario == "e2e_check_missing":
        assert "frontend-e2e-tiers.yml:missing" in result.stdout  # nosec B101 - regression assertion
    if scenario == "six_gate_run_failed":
        assert "backend-required:failure" in result.stdout  # nosec B101 - regression assertion
