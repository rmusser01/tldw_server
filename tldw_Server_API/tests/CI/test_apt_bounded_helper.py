"""Behavior of .github/actions/apt-bounded.sh: a stalled apt-get is cut off and retried (TASK-13415)."""

import shutil
import subprocess
import time
from pathlib import Path

HELPER = Path(".github/actions/apt-bounded.sh").resolve()

_TIMEOUT_SHIM = """#!/usr/bin/env python3
import subprocess, sys
args = [a for a in sys.argv[1:] if not a.startswith("--kill-after")]
proc = subprocess.Popen(args[1:])
try:
    sys.exit(proc.wait(timeout=float(args[0])))
except subprocess.TimeoutExpired:
    proc.kill()
    proc.wait()
    sys.exit(124)
"""

# Hangs on the first call, then succeeds; always fails; or always hangs. Stubs exec
# their sleep so the timeout kills the process holding the output pipe, like apt-get.
_APT_GET_STUB = """#!/bin/bash
case "$APT_STUB_MODE" in
  hang-once)
    if [ ! -f "$APT_STUB_STATE" ]; then touch "$APT_STUB_STATE"; exec sleep 30; fi
    echo "apt-get ok: $*" ;;
  fail) echo "apt-get called"; exit 100 ;;
  hang) exec sleep 30 ;;
esac
"""


def _run_helper(
    tmp_path: Path, mode: str, *, total_seconds: str = "600", dpkg_body: str = "exit 0"
) -> subprocess.CompletedProcess[str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    stubs = {
        "apt-get": _APT_GET_STUB,
        "sudo": '#!/bin/bash\nexec "$@"\n',
        "dpkg": f"#!/bin/bash\n{dpkg_body}\n",
    }
    if shutil.which("timeout") is None:  # macOS has no coreutils timeout
        stubs["timeout"] = _TIMEOUT_SHIM
    for name, body in stubs.items():
        stub = bin_dir / name
        stub.write_text(body, encoding="utf-8")
        stub.chmod(0o755)
    script = f'source "{HELPER}"; apt_bounded update "${{apt_opts[@]}}"'
    env = {
        "PATH": f"{bin_dir}:/usr/bin:/bin",
        "APT_ATTEMPT_SECONDS": "2",
        "APT_TOTAL_SECONDS": total_seconds,
        "APT_RETRY_BASE_SECONDS": "0",
        "APT_STUB_MODE": mode,
        "APT_STUB_STATE": str(tmp_path / "seen"),
    }
    return subprocess.run(
        ["bash", "-c", script], env=env, capture_output=True, text=True, timeout=60
    )


def test_stalled_attempt_is_cut_off_and_retried(tmp_path: Path) -> None:
    result = _run_helper(tmp_path, "hang-once")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "::warning::apt-get update attempt 1 failed or timed out" in result.stdout
    assert "apt-get ok: update -o Acquire::Retries=3" in result.stdout


def test_persistent_failure_stops_after_three_attempts(tmp_path: Path) -> None:
    result = _run_helper(tmp_path, "fail")

    assert result.returncode == 1
    assert result.stdout.count("apt-get called") == 3
    # Only the first two failures announce a retry; the third ends with the error.
    assert result.stdout.count("::warning::apt-get update attempt") == 2
    assert "::error::apt-get update failed within the apt time budget" in result.stdout


def test_shared_deadline_stops_retries_before_three_attempts(tmp_path: Path) -> None:
    start = time.monotonic()
    result = _run_helper(tmp_path, "hang", total_seconds="3")

    assert result.returncode == 1
    assert time.monotonic() - start < 20
    assert result.stdout.count("::warning::apt-get update attempt") < 3
    assert "::error::apt-get update failed within the apt time budget" in result.stdout


def test_hanging_dpkg_recovery_is_bounded(tmp_path: Path) -> None:
    start = time.monotonic()
    result = _run_helper(tmp_path, "fail", total_seconds="4", dpkg_body="exec sleep 30")

    assert result.returncode == 1
    assert time.monotonic() - start < 25
