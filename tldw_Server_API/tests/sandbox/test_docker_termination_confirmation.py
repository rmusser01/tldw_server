"""Docker terminal states require confirmed removal of the owned container."""

import io
import json
import subprocess
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tldw_Server_API.app.core.Sandbox.models import RunPhase, RunSpec, RuntimeType
from tldw_Server_API.app.core.Sandbox.runners import docker_runner as module


@pytest.fixture
def docker_backend(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("TLDW_SANDBOX_DOCKER_FAKE_EXEC", "0")
    monkeypatch.setenv("TLDW_SANDBOX_DOCKER_AVAILABLE", "1")
    monkeypatch.setattr(module.DockerRunner, "_active_cid", {})
    monkeypatch.setattr(module.DockerRunner, "_egress_net", {})
    monkeypatch.setattr(module.DockerRunner, "_egress_label", {})
    for method in (
        "_resolve_cgroup_cpu_file_by_cid",
        "_read_cgroup_cpu_time_sec_by_cid",
        "_read_cgroup_mem_peak_mb_by_cid",
    ):
        monkeypatch.setattr(module.DockerRunner, method, staticmethod(lambda *_: None))
    monkeypatch.setattr(module.DockerRunner, "_get_mem_usage_mb", staticmethod(lambda *_: 0))
    monkeypatch.setattr(module.DockerRunner, "_get_cpu_time_sec", staticmethod(lambda *_: 0))
    monkeypatch.setattr(module.DockerRunner, "_docker_version", staticmethod(lambda: "test"))
    monkeypatch.setattr(module, "delete_rules_by_label", lambda *_: None)
    hub = Mock()
    hub.get_log_bytes.return_value = 0
    monkeypatch.setattr(module, "get_hub", lambda: hub)
    monkeypatch.setattr(
        module.subprocess,
        "Popen",
        lambda *_, **__: SimpleNamespace(
            stdout=io.BytesIO(),
            stderr=io.BytesIO(),
            wait=lambda **_: 0,
        ),
    )
    monkeypatch.setattr(
        module.subprocess,
        "check_output",
        lambda args, **_: "container-owned" if args[:2] == ["docker", "create"] else "sha256:image",
    )

    def configure(outcome: str, *, removal_fails: bool = True) -> None:
        def check_call(args, **kwargs):
            if args[:2] == ["docker", "start"] and outcome == "startup_timeout":
                raise subprocess.TimeoutExpired(args, 1)
            if args[:2] == ["docker", "start"] and outcome == "startup_error":
                raise subprocess.CalledProcessError(1, args)
            if args[:3] == ["docker", "rm", "-f"] and removal_fails:
                raise subprocess.CalledProcessError(1, args)
            if args[:2] == ["docker", "kill"]:
                raise subprocess.CalledProcessError(1, args)
            return 0

        def run(args, **kwargs):
            if args[:2] == ["docker", "wait"]:
                if outcome == "execution_timeout":
                    raise subprocess.TimeoutExpired(args, 1)
                if outcome == "wait_error":
                    raise OSError("daemon unavailable")
                if outcome == "wait_failed":
                    return SimpleNamespace(returncode=1, stdout="0", stderr="daemon unavailable")
                return SimpleNamespace(returncode=0, stdout="0", stderr="")
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        monkeypatch.setattr(module.subprocess, "check_call", check_call)
        monkeypatch.setattr(module.subprocess, "run", run)

    return configure, hub


@pytest.mark.parametrize(
    "outcome",
    [
        "startup_timeout",
        "startup_error",
        "execution_timeout",
        "wait_error",
        "wait_failed",
        "completed",
    ],
)
def test_unconfirmed_docker_removal_preserves_active_ownership(docker_backend, outcome: str) -> None:
    configure, hub = docker_backend
    configure(outcome)

    status = module.DockerRunner().start_run(
        "uncertain",
        RunSpec(None, RuntimeType.docker, "image", ["true"]),
    )

    assert status.phase == RunPhase.running
    assert status.message == "backend_termination_unconfirmed"
    assert status.finished_at is None
    assert status.exit_code is None
    assert module.DockerRunner._active_cid["uncertain"] == "container-owned"
    assert not any(call.args[1] == "end" for call in hub.publish_event.call_args_list)


@pytest.mark.parametrize(
    "outcome, phase",
    [
        ("completed", RunPhase.completed),
        ("execution_timeout", RunPhase.timed_out),
        ("startup_timeout", RunPhase.timed_out),
        ("wait_failed", RunPhase.failed),
    ],
)
def test_confirmed_docker_removal_allows_terminal_state(docker_backend, outcome: str, phase: RunPhase) -> None:
    configure, hub = docker_backend
    configure(outcome, removal_fails=False)

    status = module.DockerRunner().start_run(
        "confirmed",
        RunSpec(None, RuntimeType.docker, "image", ["true"]),
    )

    assert status.phase == phase
    assert "confirmed" not in module.DockerRunner._active_cid
    assert any(call.args[1] == "end" for call in hub.publish_event.call_args_list)


def test_docker_allowlist_rejects_ipv6_network_before_user_execution(docker_backend, monkeypatch) -> None:
    configure, _hub = docker_backend
    configure("completed", removal_fails=False)
    monkeypatch.setenv("SANDBOX_EGRESS_ENFORCEMENT", "true")
    monkeypatch.setenv("SANDBOX_EGRESS_GRANULAR_ENFORCEMENT", "true")
    monkeypatch.setattr(
        module.subprocess,
        "check_output",
        lambda args, **_: (
            "container-owned"
            if args[:2] == ["docker", "create"]
            else json.dumps(
                {
                    "tldw_sbx_ipv6-review": {"IPAddress": "172.18.0.2", "GlobalIPv6Address": "fd00::2"},
                }
            )
        ),
    )

    with pytest.raises(RuntimeError, match="IPv6 allowlist enforcement is unsupported"):
        module.DockerRunner().start_run(
            "ipv6-review",
            RunSpec(None, RuntimeType.docker, "image", ["true"], network_policy="allowlist"),
        )


def test_docker_empty_egress_rules_cleanup_preserves_original_failure(docker_backend, monkeypatch) -> None:
    configure, _hub = docker_backend
    configure("completed", removal_fails=False)
    monkeypatch.setenv("SANDBOX_EGRESS_ENFORCEMENT", "true")
    monkeypatch.setenv("SANDBOX_EGRESS_GRANULAR_ENFORCEMENT", "true")
    monkeypatch.setattr(
        module.subprocess,
        "check_output",
        lambda args, **_: (
            "container-owned"
            if args[:2] == ["docker", "create"]
            else json.dumps(
                {
                    "tldw_sbx_empty-rules": {"IPAddress": "172.18.0.2"},
                }
            )
        ),
    )
    monkeypatch.setattr(module, "apply_egress_rules_atomic", lambda *_: [])
    removed = False

    def check_call(args, **kwargs):
        nonlocal removed
        if args[:3] == ["docker", "rm", "-f"]:
            if removed:
                raise subprocess.CalledProcessError(1, args)
            removed = True
        return 0

    monkeypatch.setattr(module.subprocess, "check_call", check_call)

    with pytest.raises(RuntimeError, match="no rules were installed"):
        module.DockerRunner().start_run(
            "empty-rules",
            RunSpec(None, RuntimeType.docker, "image", ["true"], network_policy="allowlist"),
        )
