"""Behavior of .github/actions/apt-bounded.sh: a stalled apt-get is cut off and retried (TASK-13415)."""

import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

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


@pytest.mark.parametrize("action", ["setup-ffmpeg", "wait-for-postgres"])
@pytest.mark.parametrize("layout", ["classic-and-mirror", "mirror-only", "absent"])
def test_action_preflight_normalizes_active_apt_mirrors(tmp_path: Path, action: str, layout: str) -> None:
    """Run the owning Linux normalization without touching host apt configuration."""
    action_path = HELPER.parent / action / "action.yml"
    steps = yaml.safe_load(action_path.read_text(encoding="utf-8"))["runs"]["steps"]
    linux_script = next(step["run"] for step in steps if "runner.os == 'Linux'" in step.get("if", ""))
    preflight = linux_script[linux_script.index("if [ -d /etc/apt/") :].split('source "', 1)[0]
    apt_dir = tmp_path / "apt"
    apt_dir.mkdir()
    original = "http://azure.archive.ubuntu.com/ubuntu\nhttps://security.ubuntu.com/ubuntu\n"
    normalized = "https://archive.ubuntu.com/ubuntu\nhttps://security.ubuntu.com/ubuntu\n"
    mirror_list = apt_dir / "apt-mirrors.txt"
    if layout != "absent":
        mirror_list.write_text(original, encoding="utf-8")
    if layout == "classic-and-mirror":
        sources_dir = apt_dir / "sources.list.d"
        sources_dir.mkdir()
        for path in [apt_dir / "sources.list", sources_dir / "ubuntu.sources"]:
            path.write_text(original, encoding="utf-8")
        (sources_dir / "microsoft.list").write_text("https://packages.microsoft.com/ubuntu\n", encoding="utf-8")
        (sources_dir / "unrelated.list").write_text("https://example.com/ubuntu\n", encoding="utf-8")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    sudo = bin_dir / "sudo"
    # The actions use GNU sed; macOS needs its native in-place spelling for the same expression.
    sed_compat = (
        'if [ "$1" = sed ] && [ "$2" = -i ]; then shift 2; exec /usr/bin/sed -i "" "$@"; fi\n'
        if sys.platform == "darwin"
        else ""
    )
    sudo.write_text("#!/bin/bash\n" + sed_compat + 'exec "$@"\n', encoding="utf-8")
    sudo.chmod(0o755)
    preflight = preflight.replace("/etc/apt", str(apt_dir))
    assert "/etc/apt" not in preflight
    # Execute fixed repository code with test-owned paths and literal arguments.
    result = subprocess.run(
        ["/bin/bash", "-e", "-o", "pipefail", "-c", preflight],
        env={"PATH": f"{bin_dir}:/usr/bin:/bin"},
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    if layout == "absent":
        assert list(apt_dir.iterdir()) == []
    else:
        assert mirror_list.read_text(encoding="utf-8") == normalized
    if layout == "classic-and-mirror":
        for path in [apt_dir / "sources.list", sources_dir / "ubuntu.sources"]:
            assert path.read_text(encoding="utf-8") == normalized
        assert not (sources_dir / "microsoft.list").exists()
        assert (sources_dir / "unrelated.list").read_text(encoding="utf-8") == "https://example.com/ubuntu\n"
