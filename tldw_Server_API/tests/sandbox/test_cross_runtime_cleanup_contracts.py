from __future__ import annotations

from pathlib import Path

import pytest

import tldw_Server_API.app.core.Sandbox.runners.firecracker_runner as firecracker_module
import tldw_Server_API.app.core.Sandbox.runners.lima_runner as lima_module
from tldw_Server_API.app.core.Sandbox.models import RunSpec, RuntimeType
from tldw_Server_API.app.core.Sandbox.runners.firecracker_runner import FirecrackerRunner
from tldw_Server_API.app.core.Sandbox.runners.lima_runner import LimaRunner


def _spec(runtime: RuntimeType, *, network_policy: str | None = "deny_all") -> RunSpec:
    return RunSpec(
        session_id=None,
        runtime=runtime,
        base_image=None,
        command=["/bin/echo", "unused"],
        network_policy=network_policy,
        timeout_sec=5,
    )


def test_lima_real_run_cleans_run_dir_when_setup_fails_after_workspace_create(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "lima-run"

    def _mkdtemp(prefix: str) -> str:
        if prefix != "tldw_lima_":
            pytest.fail(f"unexpected Lima temp prefix: {prefix}")
        run_dir.mkdir()
        return str(run_dir)

    monkeypatch.setattr(lima_module.tempfile, "mkdtemp", _mkdtemp)

    with pytest.raises(RuntimeError, match="strict_allowlist_not_supported"):
        LimaRunner()._run_real(
            "run-lima-cleanup",
            _spec(RuntimeType.lima, network_policy="allowlist"),
        )

    if run_dir.exists():
        pytest.fail("LimaRunner should remove the run directory on setup failure")


def test_firecracker_real_run_cleans_run_dir_when_setup_fails_after_workspace_create(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "firecracker-run"
    kernel = tmp_path / "vmlinux"
    rootfs = tmp_path / "rootfs.ext4"
    kernel.write_bytes(b"kernel")
    rootfs.write_bytes(b"rootfs")

    def _mkdtemp(prefix: str) -> str:
        if prefix != "tldw_fc_":
            pytest.fail(f"unexpected Firecracker temp prefix: {prefix}")
        run_dir.mkdir()
        return str(run_dir)

    def _fail_write_env_file(workspace: str, env: dict[str, str]) -> None:
        del workspace, env
        raise RuntimeError("env write failed")

    monkeypatch.setenv("SANDBOX_FC_KERNEL_PATH", str(kernel))
    monkeypatch.setenv("SANDBOX_FC_ROOTFS_PATH", str(rootfs))
    monkeypatch.setattr(firecracker_module.tempfile, "mkdtemp", _mkdtemp)
    monkeypatch.setattr(firecracker_module, "_write_env_file", _fail_write_env_file)

    with pytest.raises(RuntimeError, match="env write failed"):
        FirecrackerRunner()._run_real(
            "run-firecracker-cleanup",
            _spec(RuntimeType.firecracker),
        )

    if run_dir.exists():
        pytest.fail("FirecrackerRunner should remove the run directory on setup failure")
