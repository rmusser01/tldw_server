from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


IMAGE_DIR = Path(__file__).resolve().parents[1]
SMOKE_SCRIPT = IMAGE_DIR / "scripts" / "run-host-e2e-smoke.sh"


def _run_smoke_script(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(SMOKE_SCRIPT), *args],
        cwd=IMAGE_DIR,
        check=False,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )


def test_host_e2e_smoke_script_help_mentions_required_bundle() -> None:
    result = _run_smoke_script("--help")

    assert result.returncode == 0
    assert "Usage:" in result.stdout
    assert "--bundle PATH" in result.stdout


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
    assert "test_macos_virtualization_helper_daemon_host_gated.py" in result.stdout
    assert "test_vz_linux_real_host_e2e.py" in result.stdout
    assert not serial_log_dir.exists()
