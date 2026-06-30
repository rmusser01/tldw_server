from __future__ import annotations

import os
import shutil
import subprocess  # nosec B404 - subprocess is required to launch the repo-local helper daemon in this opt-in smoke test.
import sys
import tempfile
import time
import warnings
from pathlib import Path

import pytest

from tldw_Server_API.app.core.Sandbox.macos_virtualization.helper_client import (
    MacOSVirtualizationHelperClient,
    MacOSVirtualizationHelperFailure,
    MacOSVirtualizationHelperUnavailable,
)
from tldw_Server_API.app.core.testing import is_truthy


def _require_helper_daemon_smoke() -> Path:
    if sys.platform != "darwin":
        pytest.skip("macOS host only")
    if not is_truthy(os.getenv("TLDW_SANDBOX_MACOS_HELPER_DAEMON_SMOKE")):
        pytest.skip("Set TLDW_SANDBOX_MACOS_HELPER_DAEMON_SMOKE=1 to enable this test")

    repo_root = Path(__file__).resolve().parents[3]
    binary_override = str(os.getenv("TLDW_SANDBOX_MACOS_HELPER_BINARY") or "").strip()
    candidates = [Path(binary_override)] if binary_override else []
    candidates.append(repo_root / "tools" / "macos-vz-helper" / ".build" / "debug" / "macos-vz-helper")

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    pytest.skip("macOS helper daemon binary is not available; build tools/macos-vz-helper first")


def _require_canonical_bundle_smoke() -> Path:
    if not is_truthy(os.getenv("TLDW_SANDBOX_VZ_LINUX_BUNDLE_SMOKE")):
        pytest.skip("Set TLDW_SANDBOX_VZ_LINUX_BUNDLE_SMOKE=1 to enable canonical bundle smoke")
    bundle_text = str(os.getenv("TLDW_SANDBOX_VZ_LINUX_BUNDLE_PATH") or "").strip()
    if not bundle_text:
        pytest.skip(
            "TLDW_SANDBOX_VZ_LINUX_BUNDLE_PATH is required; point it at the "
            "bundle/ output from tools/vz-linux-image/scripts/build-debian-bundle.sh"
        )
    bundle_path = Path(bundle_text)
    if not bundle_path.exists():
        pytest.skip(f"canonical bundle path does not exist: {bundle_path}")
    return bundle_path


def test_macos_helper_daemon_smoke_over_real_unix_socket(monkeypatch, tmp_path: Path) -> None:
    binary_path = _require_helper_daemon_smoke()
    monkeypatch.delenv("TEST_MODE", raising=False)
    socket_dir = Path(
        tempfile.mkdtemp(  # nosec B108 - AF_UNIX paths must stay short on macOS, so this smoke test intentionally allocates a private directory under /tmp.
            prefix="macos-vz-helper-smoke-",
            dir="/tmp",  # nosec B108
        )
    )
    socket_dir.chmod(0o700)
    socket_path = socket_dir / "helper.sock"
    template_path = tmp_path / "template.img"
    template_path.write_bytes(b"")

    env = os.environ.copy()
    env["TLDW_SANDBOX_MACOS_HELPER_SOCKET"] = str(socket_path)
    process = subprocess.Popen(  # nosec B603 - binary_path is repo-controlled or explicitly user-provided for this smoke test.
        [str(binary_path)],
        cwd=str(binary_path.parent),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )

    try:
        client = MacOSVirtualizationHelperClient(socket_path=str(socket_path), timeout_sec=0.5)
        deadline = time.time() + 5.0
        ping = None
        while time.time() < deadline:
            if process.poll() is not None:
                stderr_text = process.stderr.read() if process.stderr else ""
                pytest.fail(f"helper daemon exited early: {stderr_text}")
            try:
                ping = client.ping()
                break
            except MacOSVirtualizationHelperUnavailable:
                time.sleep(0.1)

        if ping is None:
            pytest.fail("helper daemon did not accept ping within timeout")

        if ping.protocol_version != "1":
            pytest.fail(f"expected helper protocol_version '1', got {ping.protocol_version!r}")
        if ping.status != "ok":
            pytest.fail(f"expected helper status 'ok', got {ping.status!r}")
        if ping.details.get("transport") != "unix":
            pytest.fail(f"expected helper transport 'unix', got {ping.details.get('transport')!r}")

        validation = client.validate_template(
            {"runtime": "vz_linux", "template": str(template_path)}
        )
        if validation["ready"] is not True:
            pytest.fail(f"expected template validation ready=True, got {validation!r}")
        if validation["template_id"] != "vz_linux:template.img":
            pytest.fail(f"expected template_id 'vz_linux:template.img', got {validation['template_id']!r}")
        if validation.get("boot_mode") != "raw_disk":
            pytest.fail(f"expected boot_mode 'raw_disk', got {validation.get('boot_mode')!r}")
        if validation.get("validation_strength") != "compatibility":
            pytest.fail(
                "expected validation_strength 'compatibility', "
                f"got {validation.get('validation_strength')!r}"
            )
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        shutil.rmtree(socket_dir, ignore_errors=True)


def test_macos_helper_daemon_canonical_bundle_boot_smoke(monkeypatch, tmp_path: Path) -> None:
    binary_path = _require_helper_daemon_smoke()
    bundle_path = _require_canonical_bundle_smoke()
    monkeypatch.delenv("TEST_MODE", raising=False)
    socket_dir = Path(
        tempfile.mkdtemp(  # nosec B108 - AF_UNIX paths must stay short on macOS, so this smoke test intentionally allocates a private directory under /tmp.
            prefix="macos-vz-helper-bundle-",
            dir="/tmp",  # nosec B108
        )
    )
    socket_dir.chmod(0o700)
    socket_path = socket_dir / "helper.sock"

    env = os.environ.copy()
    env["TLDW_SANDBOX_MACOS_HELPER_SOCKET"] = str(socket_path)
    process = subprocess.Popen(  # nosec B603 - binary_path is repo-controlled or explicitly user-provided for this smoke test.
        [str(binary_path)],
        cwd=str(binary_path.parent),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )

    created_vm_id: str | None = None
    cleanup_error: Exception | None = None
    try:
        boot_timeout_sec = float(os.getenv("TLDW_SANDBOX_VZ_LINUX_BUNDLE_BOOT_TIMEOUT_SEC") or "60")
        client = MacOSVirtualizationHelperClient(socket_path=str(socket_path), timeout_sec=max(5.0, boot_timeout_sec + 5.0))
        deadline = time.time() + 5.0
        while time.time() < deadline:
            if process.poll() is not None:
                stderr_text = process.stderr.read() if process.stderr else ""
                pytest.fail(f"helper daemon exited early: {stderr_text}")
            try:
                client.ping()
                break
            except MacOSVirtualizationHelperUnavailable:
                time.sleep(0.1)
        else:
            pytest.fail("helper daemon did not accept ping within timeout")

        validation = client.validate_template(
            {"runtime": "vz_linux", "template": str(bundle_path)}
        )
        if validation["ready"] is not True:
            reasons = ", ".join(str(reason) for reason in validation.get("reasons", []))
            pytest.skip(f"canonical bundle validation unavailable: {reasons or 'template_invalid'}")
        if validation.get("boot_mode") != "bundle":
            pytest.fail(f"expected boot_mode 'bundle', got {validation.get('boot_mode')!r}")
        if validation.get("validation_strength") != "strong":
            pytest.fail(
                "expected validation_strength 'strong', "
                f"got {validation.get('validation_strength')!r}"
            )

        try:
            response = client.create_vm(
                {
                    "runtime": "vz_linux",
                    "vm_name": "bundle-smoke-vm",
                    "template": str(bundle_path),
                    "workspace_path": str(tmp_path),
                    "timeout_sec": boot_timeout_sec,
                }
            )
        except MacOSVirtualizationHelperFailure as exc:
            pytest.fail(f"unexpected helper create_vm failure: {exc.error_code}")
        else:
            if response.vm_id != "bundle-smoke-vm":
                pytest.fail(f"expected vm_id 'bundle-smoke-vm', got {response.vm_id!r}")
            if response.state != "running":
                pytest.fail(f"expected state 'running', got {response.state!r}")
            created_vm_id = response.vm_id
            status = client.get_vm_status(response.vm_id)
            if status.state != "running":
                pytest.fail(f"expected running vm status, got {status.state!r}")
            if status.healthy is not True:
                pytest.fail(f"expected healthy=True, got {status.healthy!r}")
    finally:
        if created_vm_id:
            try:
                client = MacOSVirtualizationHelperClient(socket_path=str(socket_path), timeout_sec=5.0)
                client.terminate_vm(created_vm_id)
            except Exception as exc:  # pragma: no cover - cleanup warning path only
                cleanup_error = exc
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        shutil.rmtree(socket_dir, ignore_errors=True)
        if cleanup_error is not None:
            warnings.warn(
                f"failed to terminate helper smoke vm {created_vm_id}: {cleanup_error}",
                stacklevel=1,
            )
