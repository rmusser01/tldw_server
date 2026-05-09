from __future__ import annotations

import os
import re
import signal
import socket
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


IMAGE_DIR = Path(__file__).resolve().parents[1]
SMOKE_SCRIPT = IMAGE_DIR / "scripts" / "run-host-e2e-smoke.sh"


def _run_smoke_script(*args: str, env_overrides: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.update(env_overrides or {})
    return subprocess.run(
        [str(SMOKE_SCRIPT), *args],
        cwd=IMAGE_DIR,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )


def test_host_e2e_smoke_script_help_mentions_required_bundle() -> None:
    result = _run_smoke_script("--help")

    assert result.returncode == 0
    assert "Usage:" in result.stdout
    assert "--bundle PATH" in result.stdout
    assert "--include-failure-drills" in result.stdout


def test_host_e2e_smoke_script_requires_bundle() -> None:
    result = _run_smoke_script("--dry-run")

    assert result.returncode != 0
    assert "--bundle is required" in result.stderr


def test_host_e2e_smoke_script_dry_run_prints_helper_and_pytest_commands(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "kernel").write_bytes(b"kernel")
    (bundle / "rootfs.img").write_bytes(b"rootfs")
    helper = tmp_path / "macos-vz-helper"
    entitlements = tmp_path / "helper.entitlements"
    entitlements.write_text("<plist/>", encoding="utf-8")
    socket_path = tmp_path / "helper.sock"
    serial_log_dir = tmp_path / "serial"

    result = _run_smoke_script(
        "--dry-run",
        "--bundle",
        str(bundle),
        "--helper",
        str(helper),
        "--entitlements",
        str(entitlements),
        "--socket",
        str(socket_path),
        "--serial-log-dir",
        str(serial_log_dir),
        "--python",
        sys.executable,
    )

    assert result.returncode == 0, result.stderr
    assert "dry-run" in result.stdout
    assert "swift build" in result.stdout
    assert "codesign --force --sign - --entitlements" in result.stdout
    assert f"TLDW_SANDBOX_MACOS_HELPER_SOCKET={socket_path}" in result.stdout
    assert f"TLDW_SANDBOX_VZ_LINUX_BUNDLE_PATH={bundle}" in result.stdout
    assert f"TLDW_SANDBOX_VZ_LINUX_E2E_BASE_IMAGE={bundle}" in result.stdout
    assert f"TLDW_SANDBOX_VZ_LINUX_SERIAL_LOG_DIR={serial_log_dir}" in result.stdout  # nosec B101
    assert "test_macos_virtualization_helper_daemon_host_gated.py" in result.stdout
    assert "test_vz_linux_real_host_e2e.py" in result.stdout
    assert "-m vz_linux_host_smoke" in result.stdout
    assert "vz_linux_host_failure_drill" not in result.stdout
    assert not serial_log_dir.exists()


def test_host_e2e_smoke_script_dry_run_includes_failure_drills_when_requested(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "kernel").write_bytes(b"kernel")
    (bundle / "rootfs.img").write_bytes(b"rootfs")
    helper = tmp_path / "macos-vz-helper"

    result = _run_smoke_script(
        "--dry-run",
        "--include-failure-drills",
        "--bundle",
        str(bundle),
        "--helper",
        str(helper),
        "--python",
        sys.executable,
    )

    assert result.returncode == 0, result.stderr
    assert "-m vz_linux_host_smoke" in result.stdout
    assert "-m vz_linux_host_failure_drill" in result.stdout


def test_host_e2e_smoke_script_default_dry_run_omits_restart_lease(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "kernel").write_bytes(b"kernel")
    (bundle / "rootfs.img").write_bytes(b"rootfs")
    helper = tmp_path / "macos-vz-helper"

    result = _run_smoke_script(
        "--dry-run",
        "--bundle",
        str(bundle),
        "--helper",
        str(helper),
        "--python",
        sys.executable,
    )

    assert result.returncode == 0, result.stderr
    assert "TLDW_SANDBOX_VZ_LINUX_E2E_HELPER_RESTART_ALLOWED" not in result.stdout
    assert "TLDW_SANDBOX_VZ_LINUX_E2E_HELPER_PID_FILE" not in result.stdout


def test_host_e2e_smoke_script_failure_drill_dry_run_includes_restart_lease(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "kernel").write_bytes(b"kernel")
    (bundle / "rootfs.img").write_bytes(b"rootfs")
    helper = tmp_path / "macos-vz-helper"

    result = _run_smoke_script(
        "--dry-run",
        "--include-failure-drills",
        "--bundle",
        str(bundle),
        "--helper",
        str(helper),
        "--python",
        sys.executable,
    )

    assert result.returncode == 0, result.stderr
    assert "TLDW_SANDBOX_VZ_LINUX_E2E_HELPER_RESTART_ALLOWED=1" in result.stdout
    assert "TLDW_SANDBOX_VZ_LINUX_E2E_HELPER_PID_FILE=" in result.stdout
    assert "/helper.pid" in result.stdout
    assert f"TLDW_SANDBOX_MACOS_HELPER_BINARY={helper}" in result.stdout


def test_host_e2e_smoke_script_default_socket_uses_private_runtime_dir(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "kernel").write_bytes(b"kernel")
    (bundle / "rootfs.img").write_bytes(b"rootfs")
    helper = tmp_path / "macos-vz-helper"

    result = _run_smoke_script(
        "--dry-run",
        "--bundle",
        str(bundle),
        "--helper",
        str(helper),
        "--python",
        sys.executable,
    )

    assert result.returncode == 0, result.stderr
    assert "TLDW_SANDBOX_MACOS_HELPER_SOCKET=" in result.stdout
    assert "/tldw-vz-helper-e2e-" in result.stdout
    assert "/helper.sock" in result.stdout
    assert "TLDW_SANDBOX_VZ_LINUX_SERIAL_LOG_DIR=" in result.stdout
    assert "/serial" in result.stdout


def test_host_e2e_smoke_script_default_runtime_dir_is_private_for_real_run(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "kernel").write_bytes(b"kernel")
    (bundle / "rootfs.img").write_bytes(b"rootfs")
    tmp_dir = tmp_path / "tmp"
    tmp_dir.mkdir()
    fake_python = tmp_path / "fake-python"
    fake_helper = tmp_path / "fake-helper"
    fake_python.write_text(
        "#!/usr/bin/env python3\n"
        "import sys\n"
        "sys.exit(0 if sys.argv[1:3] == ['-m', 'pytest'] else 2)\n",
        encoding="utf-8",
    )
    fake_helper.write_text(
        "#!/usr/bin/env python3\n"
        "import signal\n"
        "import sys\n"
        "import time\n"
        "signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))\n"
        "while True:\n"
        "    time.sleep(0.1)\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    fake_helper.chmod(0o755)

    result = _run_smoke_script(
        "--bundle",
        str(bundle),
        "--helper",
        str(fake_helper),
        "--python",
        str(fake_python),
        "--skip-build",
        "--skip-sign",
        env_overrides={
            "TMPDIR": str(tmp_dir),
            "TLDW_HOST_E2E_SMOKE_SKIP_SOCKET_WAIT": "1",
        },
    )

    assert result.returncode == 0, result.stderr
    socket_match = re.search(
        r"TLDW_SANDBOX_MACOS_HELPER_SOCKET=([^ ]+/helper\.sock)",
        result.stdout,
    )
    assert socket_match is not None
    runtime_dir = Path(socket_match.group(1)).parent
    assert runtime_dir.exists()
    assert runtime_dir.stat().st_mode & 0o077 == 0
    serial_dir = runtime_dir / "serial"
    assert serial_dir.exists()
    assert serial_dir.stat().st_mode & 0o077 == 0


def test_host_e2e_smoke_script_removes_stale_socket_before_helper_start(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "kernel").write_bytes(b"kernel")
    (bundle / "rootfs.img").write_bytes(b"rootfs")
    serial_log_dir = tmp_path / "serial"
    fake_python = tmp_path / "fake-python"
    fake_helper = tmp_path / "fake-helper"
    fake_python.write_text(
        "#!/usr/bin/env python3\n"
        "import sys\n"
        "sys.exit(0 if sys.argv[1:3] == ['-m', 'pytest'] else 2)\n",
        encoding="utf-8",
    )
    fake_helper.write_text(
        "#!/usr/bin/env python3\n"
        "import signal\n"
        "import sys\n"
        "import time\n"
        "signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))\n"
        "while True:\n"
        "    time.sleep(0.1)\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    fake_helper.chmod(0o755)

    with tempfile.TemporaryDirectory(prefix="vz-smoke-", dir="/tmp") as socket_dir:
        socket_path = Path(socket_dir) / "helper.sock"
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as stale_socket:
            try:
                stale_socket.bind(str(socket_path))
            except PermissionError:
                pytest.skip("AF_UNIX socket binding is not permitted in this sandbox")

            result = _run_smoke_script(
                "--bundle",
                str(bundle),
                "--helper",
                str(fake_helper),
                "--socket",
                str(socket_path),
                "--serial-log-dir",
                str(serial_log_dir),
                "--python",
                str(fake_python),
                "--skip-build",
                "--skip-sign",
                env_overrides={"TLDW_HOST_E2E_SMOKE_SKIP_SOCKET_WAIT": "1"},
            )

        assert result.returncode == 0, result.stderr
    assert not socket_path.exists()


def test_host_e2e_smoke_script_refuses_non_private_socket_parent(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "kernel").write_bytes(b"kernel")
    (bundle / "rootfs.img").write_bytes(b"rootfs")
    serial_log_dir = tmp_path / "serial"
    fake_python = tmp_path / "fake-python"
    fake_helper = tmp_path / "fake-helper"
    fake_python.write_text(
        "#!/usr/bin/env python3\n"
        "import sys\n"
        "sys.exit(0 if sys.argv[1:3] == ['-m', 'pytest'] else 2)\n",
        encoding="utf-8",
    )
    fake_helper.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
    fake_python.chmod(0o755)
    fake_helper.chmod(0o755)
    socket_dir = tmp_path / "public-runtime"
    socket_dir.mkdir(mode=0o755)
    socket_dir.chmod(0o755)

    result = _run_smoke_script(
        "--bundle",
        str(bundle),
        "--helper",
        str(fake_helper),
        "--socket",
        str(socket_dir / "helper.sock"),
        "--serial-log-dir",
        str(serial_log_dir),
        "--python",
        str(fake_python),
        "--skip-build",
        "--skip-sign",
        env_overrides={"TLDW_HOST_E2E_SMOKE_SKIP_SOCKET_WAIT": "1"},
    )

    assert result.returncode != 0
    assert "helper socket directory must be owner-only" in result.stderr


def test_host_e2e_smoke_script_refuses_non_private_serial_log_directory(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "kernel").write_bytes(b"kernel")
    (bundle / "rootfs.img").write_bytes(b"rootfs")
    socket_dir = tmp_path / "private-runtime"
    socket_dir.mkdir(mode=0o700)
    socket_dir.chmod(0o700)
    serial_log_dir = tmp_path / "public-serial"
    serial_log_dir.mkdir(mode=0o755)
    serial_log_dir.chmod(0o755)
    fake_python = tmp_path / "fake-python"
    fake_helper = tmp_path / "fake-helper"
    fake_python.write_text(
        "#!/usr/bin/env python3\n"
        "import sys\n"
        "sys.exit(0 if sys.argv[1:3] == ['-m', 'pytest'] else 2)\n",
        encoding="utf-8",
    )
    fake_helper.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
    fake_python.chmod(0o755)
    fake_helper.chmod(0o755)

    result = _run_smoke_script(
        "--bundle",
        str(bundle),
        "--helper",
        str(fake_helper),
        "--socket",
        str(socket_dir / "helper.sock"),
        "--serial-log-dir",
        str(serial_log_dir),
        "--python",
        str(fake_python),
        "--skip-build",
        "--skip-sign",
        env_overrides={"TLDW_HOST_E2E_SMOKE_SKIP_SOCKET_WAIT": "1"},
    )

    assert result.returncode != 0  # nosec B101
    assert "serial log directory must be owner-only" in result.stderr  # nosec B101


def test_host_e2e_smoke_script_refuses_regular_file_socket_path(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "kernel").write_bytes(b"kernel")
    (bundle / "rootfs.img").write_bytes(b"rootfs")
    serial_log_dir = tmp_path / "serial"
    fake_python = tmp_path / "fake-python"
    fake_helper = tmp_path / "fake-helper"
    fake_python.write_text(
        "#!/usr/bin/env python3\n"
        "import sys\n"
        "sys.exit(0 if sys.argv[1:3] == ['-m', 'pytest'] else 2)\n",
        encoding="utf-8",
    )
    fake_helper.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
    fake_python.chmod(0o755)
    fake_helper.chmod(0o755)
    socket_path = tmp_path / "not-a-socket"
    socket_path.write_text("do not delete", encoding="utf-8")

    result = _run_smoke_script(
        "--bundle",
        str(bundle),
        "--helper",
        str(fake_helper),
        "--socket",
        str(socket_path),
        "--serial-log-dir",
        str(serial_log_dir),
        "--python",
        str(fake_python),
        "--skip-build",
        "--skip-sign",
        env_overrides={"TLDW_HOST_E2E_SMOKE_SKIP_SOCKET_WAIT": "1"},
    )

    assert result.returncode != 0
    assert "helper socket path already exists and is not a UNIX socket" in result.stderr
    assert socket_path.read_text(encoding="utf-8") == "do not delete"


def test_host_e2e_smoke_script_cleanup_uses_replacement_helper_pid(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "kernel").write_bytes(b"kernel")
    (bundle / "rootfs.img").write_bytes(b"rootfs")
    tmp_dir = tmp_path / "tmp"
    tmp_dir.mkdir()
    marker = tmp_path / "replacement_pid.txt"
    fake_python = tmp_path / "fake-python"
    fake_helper = tmp_path / "fake-helper"
    fake_python.write_text(
        "#!/usr/bin/env python3\n"
        "import os\n"
        "import subprocess\n"
        "import sys\n"
        "if sys.argv[1:3] != ['-m', 'pytest']:\n"
        "    sys.exit(2)\n"
        "if 'vz_linux_host_failure_drill' in sys.argv:\n"
        "    pid_file = os.environ['TLDW_SANDBOX_VZ_LINUX_E2E_HELPER_PID_FILE']\n"
        "    marker = os.environ['TLDW_TEST_REPLACEMENT_PID_MARKER']\n"
        "    proc = subprocess.Popen(\n"
        "        [os.environ['TLDW_SANDBOX_MACOS_HELPER_BINARY']],\n"
        "        stdin=subprocess.DEVNULL,\n"
        "        stdout=subprocess.DEVNULL,\n"
        "        stderr=subprocess.DEVNULL,\n"
        "    )\n"
        "    with open(pid_file, 'w', encoding='utf-8') as handle:\n"
        "        handle.write(str(proc.pid) + '\\n')\n"
        "    os.chmod(pid_file, 0o600)\n"
        "    with open(marker, 'w', encoding='utf-8') as handle:\n"
        "        handle.write(str(proc.pid) + '\\n')\n"
        "sys.exit(0)\n",
        encoding="utf-8",
    )
    fake_helper.write_text(
        "#!/usr/bin/env python3\n"
        "import signal\n"
        "import sys\n"
        "import time\n"
        "signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))\n"
        "while True:\n"
        "    time.sleep(0.1)\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    fake_helper.chmod(0o755)

    result = _run_smoke_script(
        "--bundle",
        str(bundle),
        "--helper",
        str(fake_helper),
        "--python",
        str(fake_python),
        "--skip-build",
        "--skip-sign",
        "--include-failure-drills",
        env_overrides={
            "TMPDIR": str(tmp_dir),
            "TLDW_HOST_E2E_SMOKE_SKIP_SOCKET_WAIT": "1",
            "TLDW_TEST_REPLACEMENT_PID_MARKER": str(marker),
        },
    )

    assert result.returncode == 0, result.stderr
    replacement_pid = int(marker.read_text(encoding="utf-8").strip())
    try:
        with pytest.raises(ProcessLookupError):
            os.kill(replacement_pid, 0)
    finally:
        try:
            os.kill(replacement_pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
