"""Behavior of .github/actions/apt-bounded.sh: a stalled apt-get is cut off and retried (TASK-13415)."""

import shutil
import subprocess
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

# Hangs on the first call, then succeeds; or always fails.
_APT_GET_STUB = """#!/bin/bash
case "$APT_STUB_MODE" in
  hang-once)
    if [ ! -f "$APT_STUB_STATE" ]; then touch "$APT_STUB_STATE"; sleep 30; fi
    echo "apt-get ok: $*" ;;
  fail) exit 100 ;;
esac
"""


def _run_helper(tmp_path: Path, mode: str) -> subprocess.CompletedProcess[str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    stubs = {
        "apt-get": _APT_GET_STUB,
        "sudo": '#!/bin/bash\nexec "$@"\n',
        "dpkg": "#!/bin/bash\nexit 0\n",
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
    assert result.stdout.count("::warning::apt-get update attempt") == 3
    assert "::error::apt-get update failed after 3 bounded attempts" in result.stdout
