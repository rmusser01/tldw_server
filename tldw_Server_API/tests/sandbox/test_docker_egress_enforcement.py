from __future__ import annotations

import json
import os
import shutil
import subprocess
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

import tldw_Server_API.app.core.Sandbox.network_policy as network_policy_module
import tldw_Server_API.app.core.Sandbox.runners.docker_runner as docker_module
from tldw_Server_API.app.core.Sandbox.runners.docker_runner import DockerRunner
from tldw_Server_API.app.core.Sandbox.models import RunSpec, RuntimeType


@pytest.mark.unit
@pytest.mark.sandbox_real_docker
def test_docker_runner_uses_network_none_when_allowlist_enforced_non_granular(monkeypatch):
     # Make docker appear available and ensure execution path is taken
    monkeypatch.setenv("TLDW_SANDBOX_DOCKER_AVAILABLE", "1")
    # Build a spec with allowlist policy
    spec = RunSpec(
        session_id=None,
        runtime=RuntimeType.docker,
        base_image="python:3.11-slim",
        command=["python", "-c", "print('ok')"],
        timeout_sec=5,
        network_policy="allowlist",
    )
    # Enforce allowlist but disable granular; expect --network none
    monkeypatch.setenv("SANDBOX_EGRESS_ENFORCEMENT", "true")
    monkeypatch.setenv("SANDBOX_EGRESS_GRANULAR_ENFORCEMENT", "false")

    recorded_cmds: List[List[str]] = []

    class _Called(Exception):
        pass

    def fake_check_output(cmd, text=False, timeout=None):  # type: ignore[no-redef]
        nonlocal recorded_cmds
        recorded_cmds.append(list(cmd))
        # Simulate failure after capture so we don't need the rest of the flow
        raise _Called()

    monkeypatch.setattr("subprocess.check_output", fake_check_output)
    runner = DockerRunner()
    with pytest.raises(_Called):
        runner.start_run(run_id="rid1234567890", spec=spec)
    # Assert the docker create command contains '--network', 'none'
    create = next((c for c in recorded_cmds if c[:2] == ["docker", "create"]), [])
    assert create, f"docker create not issued; got: {recorded_cmds}"
    # Find '--network' flag
    if "--network" in create:
        idx = create.index("--network")
        assert create[idx + 1] == "none"
    else:
        pytest.fail(f"--network not present in docker create: {create}")


@pytest.mark.unit
@pytest.mark.sandbox_real_docker
def test_docker_runner_creates_dedicated_network_when_granular_enabled(monkeypatch):
    monkeypatch.setenv("TLDW_SANDBOX_DOCKER_AVAILABLE", "1")
    spec = RunSpec(
        session_id=None,
        runtime=RuntimeType.docker,
        base_image="python:3.11-slim",
        command=["python", "-c", "print('ok')"],
        timeout_sec=5,
        network_policy="allowlist",
    )
    monkeypatch.setenv("SANDBOX_EGRESS_ENFORCEMENT", "true")
    monkeypatch.setenv("SANDBOX_EGRESS_GRANULAR_ENFORCEMENT", "true")

    recorded_cmds: List[List[str]] = []

    class _Called(Exception):
        pass

    def fake_run(args, check=False, timeout=None):  # docker network create/remove
        recorded_cmds.append(list(args))
        return 0

    def fake_check_output(cmd, text=False, timeout=None):

        recorded_cmds.append(list(cmd))
        raise _Called()

    monkeypatch.setattr("subprocess.run", fake_run)
    monkeypatch.setattr("subprocess.check_output", fake_check_output)
    runner = DockerRunner()
    with pytest.raises(_Called):
        runner.start_run(run_id="abcd1234efgh", spec=spec)
    create = next((c for c in recorded_cmds if c[:2] == ["docker", "create"]), [])
    assert create, f"docker create not issued; got: {recorded_cmds}"
    assert "--network" in create
    idx = create.index("--network")
    net_name = create[idx + 1]
    assert net_name.startswith("tldw_sbx_")


@pytest.mark.unit
def test_apply_egress_rules_atomic_raises_when_fallback_rules_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    def _failing_run(args: list[str], **kwargs: object) -> SimpleNamespace:
        del kwargs
        calls.append(list(args))
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(network_policy_module.subprocess, "run", _failing_run)

    with pytest.raises(RuntimeError, match="iptables"):
        network_policy_module.apply_egress_rules_atomic(
            "172.18.0.2",
            ["1.1.1.1/32"],
            "tldw-run-egress-fail",
        )

    assert any(call[:1] == ["iptables-restore"] for call in calls)
    assert any(call[:1] == ["iptables"] and "-j" in call and "DROP" in call for call in calls)


@pytest.mark.unit
def test_docker_runner_fails_closed_when_granular_egress_rules_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TLDW_SANDBOX_DOCKER_AVAILABLE", "1")
    monkeypatch.delenv("TLDW_SANDBOX_DOCKER_FAKE_EXEC", raising=False)
    monkeypatch.setenv("SANDBOX_EGRESS_ENFORCEMENT", "true")
    monkeypatch.setenv("SANDBOX_EGRESS_GRANULAR_ENFORCEMENT", "true")
    monkeypatch.setenv("SANDBOX_EGRESS_ALLOWLIST", "1.1.1.1")
    check_calls: list[list[str]] = []
    net_name = "tldw_sbx_runegressfai"

    def _fake_run(args: list[str], **kwargs: object) -> SimpleNamespace:
        del kwargs
        check_calls.append(list(args))
        return SimpleNamespace(returncode=0)

    def _fake_check_output(cmd, text: bool = False, timeout: int | None = None) -> str:
        del text, timeout
        cmd_list = list(cmd)
        if cmd_list[:3] == ["docker", "version", "--format"]:
            return "24.0.0"
        if cmd_list[:2] == ["docker", "create"]:
            return "cid-egress-fail\n"
        if cmd_list[:2] == ["docker", "inspect"]:
            return json.dumps({net_name: {"IPAddress": "172.18.0.2"}})
        raise AssertionError(f"Unexpected check_output call: {cmd_list!r}")

    def _fake_check_call(cmd, timeout: int | None = None) -> int:
        del timeout
        cmd_list = list(cmd)
        check_calls.append(cmd_list)
        if cmd_list[:2] == ["docker", "start"]:
            return 0
        if cmd_list[:3] == ["docker", "rm", "-f"]:
            return 0
        raise AssertionError(f"Unexpected check_call call: {cmd_list!r}")

    def _fail_apply(container_ip: str, allow_targets: list[str], label: str) -> list[str]:
        del container_ip, allow_targets, label
        raise RuntimeError("iptables unavailable")

    class _LogsReached(Exception):
        pass

    monkeypatch.setattr(docker_module.subprocess, "run", _fake_run)
    monkeypatch.setattr(docker_module.subprocess, "check_output", _fake_check_output)
    monkeypatch.setattr(docker_module.subprocess, "check_call", _fake_check_call)
    monkeypatch.setattr(docker_module.subprocess, "Popen", lambda *args, **kwargs: (_ for _ in ()).throw(_LogsReached()))
    monkeypatch.setattr(docker_module, "apply_egress_rules_atomic", _fail_apply)
    monkeypatch.setattr(DockerRunner, "_docker_version", staticmethod(lambda: "24.0.0"))

    spec = RunSpec(
        session_id=None,
        runtime=RuntimeType.docker,
        base_image="python:3.11-slim",
        command=["python", "-c", "print('ok')"],
        timeout_sec=5,
        network_policy="allowlist",
    )

    with pytest.raises(RuntimeError, match="egress allowlist"):
        DockerRunner().start_run(run_id="runegressfail", spec=spec)

    assert ["docker", "rm", "-f", "cid-egress-fail"] in check_calls


@pytest.mark.integration
def test_apply_iptables_rules_on_supported_hosts(monkeypatch):
     # Only run when explicitly allowed
    if os.getenv("SANDBOX_TEST_ALLOW_IPTABLES_MUTATION") not in {"1", "true", "on", "yes"}:
        pytest.skip("iptables mutation not enabled")
    if shutil.which("iptables") is None:
        pytest.skip("iptables not available on host")
    # Ensure DOCKER-USER chain exists; if not, skip to avoid altering host firewall unexpectedly
    import subprocess
    try:
        subprocess.check_output(["iptables", "-S", "DOCKER-USER"])  # may fail if chain missing
    except Exception:
        pytest.skip("DOCKER-USER chain not present; skipping")

    from tldw_Server_API.app.core.Sandbox.network_policy import apply_egress_rules_atomic, delete_rules_by_label
    label = "tldw-test-egress-allowlist"
    rules = apply_egress_rules_atomic("172.18.0.2", ["1.1.1.1/32"], label=label)
    try:
        # Minimal assertion: rules list is non-empty
        assert isinstance(rules, list) and rules
    finally:
        # Cleanup
        delete_rules_by_label(label)
