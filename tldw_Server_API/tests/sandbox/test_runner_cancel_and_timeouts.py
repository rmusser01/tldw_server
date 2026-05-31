from __future__ import annotations

import signal
import types
from typing import Any, List

import pytest

from tldw_Server_API.app.core.Sandbox.runners.docker_runner import DockerRunner
from tldw_Server_API.app.core.Sandbox.models import RunSpec, RuntimeType
from tldw_Server_API.app.core.Sandbox.runners.seatbelt_runner import SeatbeltRunner
from tldw_Server_API.app.core.Sandbox.streams import get_hub


def _spec(cmd: List[str], *, network_policy: str = "deny_all") -> RunSpec:
    return RunSpec(
        session_id=None,
        runtime=RuntimeType.docker,
        base_image="python:3.11-slim",
        command=list(cmd),
        env={},
        timeout_sec=5,
        startup_timeout_sec=1,
        network_policy=network_policy,
    )


class _EmptyPipe:
    def readline(self) -> bytes:
        return b""

    def peek(self) -> bytes:
        return b""


class _DockerLogsPopen:
    stdout = _EmptyPipe()
    stderr = _EmptyPipe()

    def poll(self) -> int:
        return 0


def _clear_docker_tracking(run_id: str) -> None:
    with DockerRunner._active_lock:  # type: ignore[attr-defined]
        DockerRunner._active_cid.pop(run_id, None)  # type: ignore[attr-defined]
    with DockerRunner._egress_lock:  # type: ignore[attr-defined]
        DockerRunner._egress_net.pop(run_id, None)  # type: ignore[attr-defined]
        DockerRunner._egress_label.pop(run_id, None)  # type: ignore[attr-defined]


def _assert_docker_tracking_cleared(run_id: str) -> None:
    with DockerRunner._active_lock:  # type: ignore[attr-defined]
        assert run_id not in DockerRunner._active_cid  # type: ignore[attr-defined]
    with DockerRunner._egress_lock:  # type: ignore[attr-defined]
        assert run_id not in DockerRunner._egress_net  # type: ignore[attr-defined]
        assert run_id not in DockerRunner._egress_label  # type: ignore[attr-defined]


def _docker_tracking_present(run_id: str) -> bool:
    with DockerRunner._active_lock:  # type: ignore[attr-defined]
        active_present = run_id in DockerRunner._active_cid  # type: ignore[attr-defined]
    with DockerRunner._egress_lock:  # type: ignore[attr-defined]
        egress_present = (
            run_id in DockerRunner._egress_net  # type: ignore[attr-defined]
            or run_id in DockerRunner._egress_label  # type: ignore[attr-defined]
        )
    return active_present or egress_present


def _stub_docker_resource_probes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(DockerRunner, "_docker_version", staticmethod(lambda: "test-docker"))
    monkeypatch.setattr(DockerRunner, "_resolve_cgroup_cpu_file_by_cid", staticmethod(lambda cid: None))
    monkeypatch.setattr(DockerRunner, "_read_cgroup_cpu_time_sec_by_cid", staticmethod(lambda cid: None))
    monkeypatch.setattr(DockerRunner, "_read_cgroup_mem_peak_mb_by_cid", staticmethod(lambda cid: None))
    monkeypatch.setattr(DockerRunner, "_get_mem_usage_mb", staticmethod(lambda cid: 0))
    monkeypatch.setattr(DockerRunner, "_get_cpu_time_sec", staticmethod(lambda cid, started, finished: 0))
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Sandbox.runners.docker_runner.delete_rules_by_label",
        lambda label: None,
    )


@pytest.mark.unit
def test_runner_cancel_term_grace_no_duplicate_end(monkeypatch: pytest.MonkeyPatch) -> None:
    # Ensure runner uses small grace
    from tldw_Server_API.app.core.config import settings as app_settings
    monkeypatch.setattr(app_settings, "SANDBOX_CANCEL_GRACE_SECONDS", 0, raising=False)

    # Seed an active container mapping
    rid = "run_cancel_1"
    cid = "cid123"
    with DockerRunner._active_lock:  # type: ignore[attr-defined]
        DockerRunner._active_cid[rid] = cid  # type: ignore[attr-defined]

    # Simulate TERM stops container quickly
    states = {"running": True}

    def _is_running(_cid: str) -> bool:
        return states["running"]

    def _subproc_run(args: List[str], check: bool = False, **kwargs: Any):
        # When TERM is sent, mark as not running
        if args[:3] == ["docker", "kill", "--signal"] and args[3] == "TERM":
            states["running"] = False
        # Return a simple object like CompletedProcess
        cp = types.SimpleNamespace(returncode=0)
        return cp

    monkeypatch.setattr(DockerRunner, "_is_container_running", staticmethod(_is_running))
    monkeypatch.setattr("subprocess.run", _subproc_run)

    # Clear any prior frames for this run_id
    hub = get_hub()
    hub._buffers.pop(rid, None)  # type: ignore[attr-defined]

    ok = DockerRunner.cancel_run(rid)
    assert ok is True

    # Runner should NOT publish an end event; service layer does it
    frames = list(hub._buffers.get(rid, []))  # type: ignore[attr-defined]
    assert not any(f.get("type") == "event" and f.get("event") == "end" for f in frames)


@pytest.mark.unit
def test_runner_completed_run_clears_active_container_and_egress_maps(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TLDW_SANDBOX_DOCKER_FAKE_EXEC", "0")
    monkeypatch.setattr("tldw_Server_API.app.core.Sandbox.runners.docker_runner.docker_available", lambda: True)
    _stub_docker_resource_probes(monkeypatch)

    rid = "run_docker_complete_cleanup"
    cid = "cid-complete"
    _clear_docker_tracking(rid)

    def _check_output(args: List[str], text: bool = False, timeout: int | None = None, **kwargs: Any):
        if args[:2] == ["docker", "create"]:
            return cid
        if args[:3] == ["docker", "image", "inspect"]:
            return "sha256:test-image"
        return ""

    check_calls: list[list[str]] = []

    def _check_call(args: List[str], timeout: int | None = None, **kwargs: Any):
        check_calls.append(list(args))
        return 0

    def _run(args: List[str], capture_output: bool = False, text: bool = False, timeout: int | None = None, **kwargs: Any):
        if args[:2] == ["docker", "wait"]:
            return types.SimpleNamespace(returncode=0, stdout="0")
        return types.SimpleNamespace(returncode=0, stdout="")

    monkeypatch.setattr("subprocess.check_output", _check_output)
    monkeypatch.setattr("subprocess.check_call", _check_call)
    monkeypatch.setattr("subprocess.run", _run)
    monkeypatch.setattr("subprocess.Popen", lambda *args, **kwargs: _DockerLogsPopen())

    try:
        rs = DockerRunner().start_run(rid, _spec(["python", "-c", "print('x')"]))
    finally:
        # Keep this test isolated even while RED exposes a cleanup leak.
        leaked_before_cleanup = _docker_tracking_present(rid)
        _clear_docker_tracking(rid)

    assert rs.phase.value == "completed"
    assert rs.exit_code == 0
    assert ["docker", "rm", "-f", cid] in check_calls
    assert not leaked_before_cleanup
    _assert_docker_tracking_cleared(rid)


@pytest.mark.unit
def test_runner_startup_timeout_on_create(monkeypatch: pytest.MonkeyPatch) -> None:
    # Make docker available and non-fake
    monkeypatch.setenv("TLDW_SANDBOX_DOCKER_FAKE_EXEC", "0")
    monkeypatch.setattr("tldw_Server_API.app.core.Sandbox.runners.docker_runner.docker_available", lambda: True)

    import subprocess

    def _raise_timeout(*args: Any, **kwargs: Any):  # type: ignore[no-untyped-def]
        raise subprocess.TimeoutExpired(cmd=args[0] if args else "docker create", timeout=1)

    monkeypatch.setattr("subprocess.check_output", _raise_timeout)

    dr = DockerRunner()
    rid = "run_to_create_to"
    hub = get_hub()
    hub._buffers.pop(rid, None)  # type: ignore[attr-defined]
    rs = dr.start_run(rid, _spec(["python", "-c", "print('x')"]))
    assert rs.phase.value == "timed_out"
    assert (rs.message or "").startswith("startup_timeout")
    # Ensure WS end has reason=startup_timeout
    frames = list(hub._buffers.get(rid, []))  # type: ignore[attr-defined]
    assert any(f.get("type") == "event" and f.get("event") == "end" and f.get("data", {}).get("reason") == "startup_timeout" for f in frames)


@pytest.mark.unit
def test_runner_startup_timeout_on_create_removes_precreated_egress_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TLDW_SANDBOX_DOCKER_FAKE_EXEC", "0")
    monkeypatch.setenv("SANDBOX_EGRESS_ENFORCEMENT", "true")
    monkeypatch.setenv("SANDBOX_EGRESS_GRANULAR_ENFORCEMENT", "true")
    monkeypatch.setattr("tldw_Server_API.app.core.Sandbox.runners.docker_runner.docker_available", lambda: True)
    _stub_docker_resource_probes(monkeypatch)

    import subprocess

    rid = "run_create_timeout_cleanup"
    run_calls: list[list[str]] = []

    def _run(args: List[str], check: bool = False, **kwargs: Any):
        run_calls.append(list(args))
        return types.SimpleNamespace(returncode=0, stdout="")

    def _raise_timeout(*args: Any, **kwargs: Any):  # type: ignore[no-untyped-def]
        raise subprocess.TimeoutExpired(cmd=args[0] if args else "docker create", timeout=1)

    monkeypatch.setattr("subprocess.run", _run)
    monkeypatch.setattr("subprocess.check_output", _raise_timeout)

    rs = DockerRunner().start_run(
        rid,
        _spec(["python", "-c", "print('x')"], network_policy="allowlist"),
    )

    assert rs.phase.value == "timed_out"
    assert rs.message == "startup_timeout"
    assert ["docker", "network", "rm", f"tldw_sbx_{rid[:12]}"] in run_calls
    _assert_docker_tracking_cleared(rid)


@pytest.mark.unit
def test_runner_execution_timeout_on_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    # Make docker available and non-fake
    monkeypatch.setenv("TLDW_SANDBOX_DOCKER_FAKE_EXEC", "0")
    monkeypatch.setattr("tldw_Server_API.app.core.Sandbox.runners.docker_runner.docker_available", lambda: True)

    # Simulate docker create returns a CID, cp/start succeed, wait times out
    _stub_docker_resource_probes(monkeypatch)

    def _check_output(args: List[str], text: bool = False, timeout: int | None = None):
        if args and args[0] == "docker" and args[1] not in ("image",):
            # docker create returns a CID once
            return "cid999"
        # image inspect can return some digest; return empty to skip
        return ""

    def _check_call(args: List[str], timeout: int | None = None):
        return 0

    import subprocess

    def _run(args: List[str], capture_output: bool = False, text: bool = False, timeout: int | None = None):
        # docker wait should time out
        if args and args[0] == "docker" and args[1] == "wait":
            raise subprocess.TimeoutExpired(cmd=args, timeout=timeout or 1)
        return types.SimpleNamespace(returncode=0, stdout="0")

    monkeypatch.setattr("subprocess.check_output", _check_output)
    monkeypatch.setattr("subprocess.check_call", _check_call)
    monkeypatch.setattr("subprocess.run", _run)

    dr = DockerRunner()
    rid = "run_to_wait_to"
    hub = get_hub()
    hub._buffers.pop(rid, None)  # type: ignore[attr-defined]
    rs = dr.start_run(rid, _spec(["python", "-c", "print('x')"]))
    assert rs.phase.value == "timed_out"
    assert (rs.message or "") == "execution_timeout"
    frames = list(hub._buffers.get(rid, []))  # type: ignore[attr-defined]
    assert any(f.get("type") == "event" and f.get("event") == "end" and f.get("data", {}).get("reason") == "execution_timeout" for f in frames)
    leaked_before_cleanup = _docker_tracking_present(rid)
    _clear_docker_tracking(rid)
    assert not leaked_before_cleanup
    _assert_docker_tracking_cleared(rid)


@pytest.mark.unit
def test_runner_startup_timeout_after_container_create_clears_tracking(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TLDW_SANDBOX_DOCKER_FAKE_EXEC", "0")
    monkeypatch.setenv("SANDBOX_EGRESS_ENFORCEMENT", "true")
    monkeypatch.setenv("SANDBOX_EGRESS_GRANULAR_ENFORCEMENT", "true")
    monkeypatch.setattr("tldw_Server_API.app.core.Sandbox.runners.docker_runner.docker_available", lambda: True)
    _stub_docker_resource_probes(monkeypatch)

    import subprocess

    rid = "run_docker_start_timeout_cleanup"
    cid = "cid-start-timeout"
    _clear_docker_tracking(rid)

    def _check_output(args: List[str], text: bool = False, timeout: int | None = None):
        if args[:2] == ["docker", "create"]:
            return cid
        return ""

    check_calls: list[list[str]] = []

    def _check_call(args: List[str], timeout: int | None = None):
        check_calls.append(list(args))
        if args[:2] == ["docker", "start"]:
            raise subprocess.TimeoutExpired(cmd=args, timeout=timeout or 1)
        return 0

    run_calls: list[list[str]] = []

    def _run(args: List[str], check: bool = False, **kwargs: Any):
        run_calls.append(list(args))
        return types.SimpleNamespace(returncode=0, stdout="")

    monkeypatch.setattr("subprocess.check_output", _check_output)
    monkeypatch.setattr("subprocess.check_call", _check_call)
    monkeypatch.setattr("subprocess.run", _run)

    rs = DockerRunner().start_run(
        rid,
        _spec(["python", "-c", "print('x')"], network_policy="allowlist"),
    )
    assert rs.phase.value == "timed_out"
    assert rs.message == "startup_timeout"
    assert ["docker", "rm", "-f", cid] in check_calls
    assert ["docker", "network", "rm", f"tldw_sbx_{rid[:12]}"] in run_calls
    leaked_before_cleanup = _docker_tracking_present(rid)
    _clear_docker_tracking(rid)
    assert not leaked_before_cleanup
    _assert_docker_tracking_cleared(rid)


@pytest.mark.unit
def test_seatbelt_cancel_run_kills_active_process_group_and_removes_run_dir(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    rid = "run_seatbelt_cancel_1"
    run_dir = tmp_path / "seatbelt-run-dir"
    run_dir.mkdir()

    class _FakeProc:
        pid = 9876

        def wait(self, timeout: int | None = None) -> int:
            del timeout
            return 0

    with SeatbeltRunner._active_lock:  # type: ignore[attr-defined]
        SeatbeltRunner._active_proc[rid] = _FakeProc()  # type: ignore[attr-defined]
        SeatbeltRunner._active_run_dir[rid] = str(run_dir)  # type: ignore[attr-defined]

    killpg_calls: list[tuple[int, int]] = []
    monkeypatch.setattr("os.killpg", lambda pid, sig: killpg_calls.append((pid, sig)))
    monkeypatch.setattr(SeatbeltRunner, "_cancel_grace_seconds", classmethod(lambda cls: 0))

    try:
        ok = SeatbeltRunner.cancel_run(rid)
    finally:
        with SeatbeltRunner._active_lock:  # type: ignore[attr-defined]
            SeatbeltRunner._cancelled_runs.discard(rid)  # type: ignore[attr-defined]

    assert ok is True
    assert killpg_calls == [(9876, signal.SIGTERM)]
    assert not run_dir.exists()
    with SeatbeltRunner._active_lock:  # type: ignore[attr-defined]
        assert rid not in SeatbeltRunner._active_proc  # type: ignore[attr-defined]
        assert rid not in SeatbeltRunner._active_run_dir  # type: ignore[attr-defined]
