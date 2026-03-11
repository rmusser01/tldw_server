from __future__ import annotations

import os
import subprocess  # nosec B404 - subprocess is required to launch the repo-local helper daemon in this opt-in smoke test.
import sys
import tempfile
import time
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


def test_macos_helper_daemon_smoke_over_real_unix_socket(monkeypatch, tmp_path: Path) -> None:
    binary_path = _require_helper_daemon_smoke()
    monkeypatch.delenv("TEST_MODE", raising=False)
    socket_fd, socket_name = tempfile.mkstemp(  # nosec B108 - AF_UNIX paths must stay short on macOS, so this smoke test intentionally allocates under /tmp.
        prefix="macos-vz-helper-smoke-",
        suffix=".sock",
        dir="/tmp",  # nosec B108
    )
    os.close(socket_fd)
    socket_path = Path(socket_name)
    socket_path.unlink(missing_ok=True)
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

        with pytest.raises(MacOSVirtualizationHelperFailure) as excinfo:
            client.create_vm(
                {
                    "runtime": "vz_linux",
                    "vm_name": "smoke-vm",
                    "template": str(template_path),
                    "workspace_path": str(tmp_path),
                    "timeout_sec": 1,
                }
            )

        if excinfo.value.error_code != "boot_not_implemented":
            pytest.fail(
                f"expected boot_not_implemented helper error, got {excinfo.value.error_code!r}"
            )
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        socket_path.unlink(missing_ok=True)
