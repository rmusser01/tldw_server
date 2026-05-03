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


def test_lima_real_run_cleans_run_dir_when_setup_is_interrupted(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "lima-interrupted-run"

    def _mkdtemp(prefix: str) -> str:
        if prefix != "tldw_lima_":
            pytest.fail(f"unexpected Lima temp prefix: {prefix}")
        run_dir.mkdir()
        return str(run_dir)

    def _interrupt_entry_script(workspace: str, command: list[str]) -> None:
        del workspace, command
        raise KeyboardInterrupt

    monkeypatch.setattr(lima_module.tempfile, "mkdtemp", _mkdtemp)
    monkeypatch.setattr(LimaRunner, "_write_entry_script", staticmethod(_interrupt_entry_script))

    with pytest.raises(KeyboardInterrupt):
        LimaRunner()._run_real("run-lima-interrupted", _spec(RuntimeType.lima))

    if run_dir.exists():
        pytest.fail("LimaRunner should remove the run directory on setup interruption")


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


def test_firecracker_real_run_waits_for_spawned_processes_on_setup_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    kernel = tmp_path / "vmlinux"
    rootfs = tmp_path / "rootfs.ext4"
    kernel.write_bytes(b"kernel")
    rootfs.write_bytes(b"rootfs")
    processes: list[_FakeProcess] = []

    def _fake_popen(args: list[str], **kwargs: object) -> _FakeProcess:
        proc = _FakeProcess()
        processes.append(proc)
        if "--api-sock" in args:
            cwd = kwargs.get("cwd")
            if not isinstance(cwd, str):
                pytest.fail("Firecracker Popen should receive a string cwd")
            (Path(cwd) / "fc.sock").touch()
        return proc

    def _fail_api_request(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise RuntimeError("api configure failed")

    monkeypatch.setenv("SANDBOX_FC_KERNEL_PATH", str(kernel))
    monkeypatch.setenv("SANDBOX_FC_ROOTFS_PATH", str(rootfs))
    monkeypatch.setattr(firecracker_module.subprocess, "Popen", _fake_popen)
    monkeypatch.setattr(firecracker_module, "_fc_api_request", _fail_api_request)

    with pytest.raises(RuntimeError, match="api configure failed"):
        FirecrackerRunner()._run_real("run-firecracker-proc-cleanup", _spec(RuntimeType.firecracker))

    if len(processes) != 1:
        pytest.fail(f"expected one Firecracker process, got {len(processes)}")
    if processes[0].terminate_calls != 1:
        pytest.fail("Firecracker process should be terminated on setup failure")
    if processes[0].wait_calls == 0:
        pytest.fail("Firecracker process should be waited after terminate")


def test_firecracker_real_run_logs_status_parse_failures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    kernel = tmp_path / "vmlinux"
    rootfs = tmp_path / "rootfs.ext4"
    kernel.write_bytes(b"kernel")
    rootfs.write_bytes(b"rootfs")
    warnings: list[str] = []

    class _Logger:
        def warning(self, message: str, *args: object, **kwargs: object) -> None:
            del args, kwargs
            warnings.append(message)

    def _fake_popen(args: list[str], **kwargs: object) -> _FakeProcess:
        proc = _FakeProcess()
        if "--api-sock" in args:
            cwd = kwargs.get("cwd")
            if not isinstance(cwd, str):
                pytest.fail("Firecracker Popen should receive a string cwd")
            (Path(cwd) / "fc.sock").touch()
        if "--socket-path" in args:
            source_arg = next((arg for arg in args if arg.startswith("source=")), None)
            if source_arg is None:
                pytest.fail("virtiofsd Popen should receive a source mount option")
            workspace = Path(source_arg.removeprefix("source="))
            (workspace / ".sandbox_status.json").write_text("{not-json", encoding="utf-8")
        return proc

    monkeypatch.setenv("SANDBOX_FC_KERNEL_PATH", str(kernel))
    monkeypatch.setenv("SANDBOX_FC_ROOTFS_PATH", str(rootfs))
    monkeypatch.setenv("TLDW_SANDBOX_FIRECRACKER_VERSION", "test-version")
    monkeypatch.setattr(firecracker_module.subprocess, "Popen", _fake_popen)
    monkeypatch.setattr(firecracker_module, "_fc_api_request", lambda *args, **kwargs: None)
    monkeypatch.setattr(firecracker_module, "logger", _Logger(), raising=False)

    FirecrackerRunner()._run_real("run-firecracker-status-parse", _spec(RuntimeType.firecracker))

    if not warnings:
        pytest.fail("Firecracker status parse failures should be logged")
    if "status" not in warnings[0]:
        pytest.fail(f"expected status parse context in warning, got: {warnings[0]}")


class _FakeProcess:
    def __init__(self) -> None:
        self.terminate_calls = 0
        self.kill_calls = 0
        self.wait_calls = 0

    def terminate(self) -> None:
        self.terminate_calls += 1

    def kill(self) -> None:
        self.kill_calls += 1

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        self.wait_calls += 1
        return 0
