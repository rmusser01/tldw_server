"""The merge helper must reject unfinished and stale release checks."""

import os
import shutil
import subprocess  # nosec B404 - executes a controlled shell fixture
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize(
    "scenario",
    ["audit_pending", "audit_failed", "gate_pending", "stale_gate", "changed_head", "success", "success_no_merge"],
)
def test_release_helper_requires_success_for_the_current_head(tmp_path, scenario):
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
        case "$SCENARIO" in
          gate_pending) echo '2 in_progress pending release-head';;
          stale_gate) echo '2 completed success old-head';;
          *) echo '2 completed success release-head';;
        esac;;
    esac;;
  'run rerun') exit 0;;
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
    if scenario.startswith("success"):
        assert result.returncode == 0, result.stdout + result.stderr  # nosec B101 - regression assertion
        if scenario == "success":
            assert "--match-head-commit release-head" in marker.read_text()  # nosec B101 - regression assertion
        else:
            assert not marker.exists()  # nosec B101 - regression assertion
    else:
        assert result.returncode != 0, result.stdout + result.stderr  # nosec B101 - regression assertion
        assert not marker.exists()  # nosec B101 - regression assertion
