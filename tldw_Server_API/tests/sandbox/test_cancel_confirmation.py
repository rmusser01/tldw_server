"""Cancellation must not turn backend uncertainty into terminal success."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tldw_Server_API.app.core.Sandbox.models import RunPhase, RunStatus, RuntimeType
from tldw_Server_API.app.core.Sandbox.runners.firecracker_runner import FirecrackerRunner
from tldw_Server_API.app.core.Sandbox.service import SandboxService
from tldw_Server_API.app.core.Sandbox.store import InMemoryStore, SQLiteStore


def _service(runtime: RuntimeType = RuntimeType.docker):
    status = RunStatus(id="cancel-confirmation", phase=RunPhase.running, runtime=runtime)
    service = SandboxService.__new__(SandboxService)
    service._orch = SimpleNamespace(
        get_run=Mock(return_value=status), update_run=Mock(),
        cancel_queued_run=Mock(return_value=False),
    )
    return service, status


@pytest.mark.parametrize("outcome", [False, RuntimeError("backend unavailable")])
def test_unconfirmed_cancel_preserves_active_status(monkeypatch, outcome):
    service, status = _service()
    cancel = Mock(side_effect=outcome) if isinstance(outcome, Exception) else Mock(return_value=outcome)
    monkeypatch.setattr("tldw_Server_API.app.core.Sandbox.service.DockerRunner.cancel_run", cancel)
    hub = Mock()
    monkeypatch.setattr("tldw_Server_API.app.core.Sandbox.service.get_hub", lambda: hub)

    assert service.cancel_run(status.id) is False
    assert status.phase == RunPhase.running
    assert status.finished_at is None
    assert not any(call.args[1] == "end" for call in hub.publish_event.call_args_list)


def test_firecracker_cancel_is_dispatched_before_marking_killed(monkeypatch):
    service, status = _service(RuntimeType.firecracker)
    cancel = Mock(return_value=True)
    monkeypatch.setattr(FirecrackerRunner, "cancel_run", cancel, raising=False)

    assert service.cancel_run(status.id) is True
    cancel.assert_called_once_with(status.id)
    assert status.phase == RunPhase.killed


def test_firecracker_cleanup_failure_is_not_confirmed(monkeypatch):
    from tldw_Server_API.app.core.Sandbox.runners.firecracker_runner import _terminate_process

    proc = Mock()
    proc.terminate.side_effect = OSError("permission denied")
    assert _terminate_process(proc, "firecracker") is False


def test_real_firecracker_requires_implemented_host_containment(monkeypatch):
    from tldw_Server_API.app.core.Sandbox.runners import firecracker_runner as runner

    monkeypatch.setattr(runner.sys, "platform", "linux")
    monkeypatch.setattr(runner.os.path, "exists", lambda _: True)
    monkeypatch.setattr(runner.shutil, "which", lambda _: "/usr/bin/firecracker")
    assert "firecracker_host_containment_not_implemented" in runner._preflight_errors()


@pytest.fixture(params=["memory", "sqlite"])
def store(request, tmp_path):
    return InMemoryStore() if request.param == "memory" else SQLiteStore(str(tmp_path / "runs.db"))


def test_queued_cancellation_fences_future_dispatch(store):
    status = RunStatus(id="queued", phase=RunPhase.queued, runtime=RuntimeType.docker)
    store.put_run("user", status)
    assert store.cancel_queued_run(status.id) is True
    assert store.try_claim_run(status.id, worker_id="late-worker") is None
    assert store.get_run(status.id).phase == RunPhase.killed


def test_queued_cancellation_cannot_kill_started_workload(store):
    status = RunStatus(id="running", phase=RunPhase.running, runtime=RuntimeType.docker)
    store.put_run("user", status)
    assert store.cancel_queued_run(status.id) is False
    assert store.get_run(status.id).phase == RunPhase.running


def test_backend_identity_survives_store_reopen_and_stale_updates(tmp_path):
    path = str(tmp_path / "identity.db")
    store = SQLiteStore(path)
    status = RunStatus(id="owned", phase=RunPhase.running, runtime=RuntimeType.docker,
                       backend_id="tldw-sbx-owned", backend_owner="daemon-host")
    store.put_run("user", status)
    reopened = SQLiteStore(path)
    assert reopened.get_run(status.id).backend_id == "tldw-sbx-owned"
    reopened.update_run(RunStatus(id=status.id, phase=RunPhase.completed))
    assert reopened.get_run(status.id).backend_owner == "daemon-host"


def test_confirmed_cancellation_cannot_be_overwritten_by_late_runner(store):
    status = RunStatus(id="late", phase=RunPhase.killed, runtime=RuntimeType.docker,
                       message="canceled_by_user")
    store.put_run("user", status)
    store.update_run(RunStatus(id=status.id, phase=RunPhase.completed, exit_code=0))
    assert store.get_run(status.id).phase == RunPhase.killed


@pytest.mark.parametrize("module_name,class_name", [
    ("seatbelt_runner", "SeatbeltRunner"), ("worktree_runner", "WorktreeRunner"),
])
def test_host_runner_cancel_failure_retains_ownership(monkeypatch, module_name, class_name):
    import importlib
    import subprocess

    module = importlib.import_module(f"tldw_Server_API.app.core.Sandbox.runners.{module_name}")
    runner = getattr(module, class_name)
    proc = Mock(pid=12345)
    proc.wait.side_effect = subprocess.TimeoutExpired("wait", 1)
    monkeypatch.setattr(runner, "_active_proc", {"retained": proc})
    monkeypatch.setattr(runner, "_terminate_process_group", Mock())

    assert runner.cancel_run("retained") is False
    assert runner._active_proc["retained"] is proc


def test_vz_cancel_failure_retains_ownership(monkeypatch):
    from tldw_Server_API.app.core.Sandbox.runners.vz_linux_runner import VZLinuxRunner

    monkeypatch.setattr(VZLinuxRunner, "_active_vm", {"retained": "vm-owned"})
    helper = Mock()
    helper.terminate_vm.return_value = False
    monkeypatch.setattr(VZLinuxRunner, "helper_client_cls", staticmethod(lambda: helper))

    assert VZLinuxRunner.cancel_run("retained") is False
    assert VZLinuxRunner._active_vm["retained"] == "vm-owned"


def test_expired_dispatch_lease_does_not_release_backend_capacity(store):
    from datetime import datetime, timedelta, timezone

    active = RunStatus(id="orphan", phase=RunPhase.running, runtime=RuntimeType.docker,
                       backend_id="tldw-sbx-orphan", claim_owner="dead-worker",
                       claim_expires_at=datetime.now(timezone.utc) - timedelta(seconds=60))
    queued = RunStatus(id="next", phase=RunPhase.queued, runtime=RuntimeType.docker)
    store.put_run("user", active)
    store.put_run("user", queued)
    store.try_claim_run(queued.id, worker_id="new-worker")
    assert store.try_admit_run_start(queued.id, worker_id="new-worker", max_active_runs=1) is None


def test_docker_cancel_failure_retains_tracking(monkeypatch):
    from tldw_Server_API.app.core.Sandbox.runners import docker_runner as module

    monkeypatch.setattr(module.DockerRunner, "_active_cid", {"retained": "container"})
    monkeypatch.setattr(module.DockerRunner, "_is_container_running", lambda _: True)
    monkeypatch.setattr(module.subprocess, "run", Mock(return_value=SimpleNamespace(returncode=1)))
    monkeypatch.setattr(module.app_settings, "SANDBOX_CANCEL_GRACE_SECONDS", 0, raising=False)
    assert module.DockerRunner.cancel_run("retained") is False
    assert module.DockerRunner._active_cid["retained"] == "container"


def test_docker_inspection_failure_is_not_evidence_of_termination(monkeypatch):
    from tldw_Server_API.app.core.Sandbox.runners import docker_runner as module

    monkeypatch.setattr(module.subprocess, "check_output", Mock(side_effect=OSError("daemon unavailable")))
    assert module.DockerRunner._is_container_running("owned") is True


def test_docker_cancel_recovers_owned_name_without_process_local_map(monkeypatch):
    import json

    from tldw_Server_API.app.core.Sandbox.runners import docker_runner as module

    monkeypatch.setattr(module.DockerRunner, "_active_cid", {})
    inspect = Mock(return_value=json.dumps([{
        "Id": "container-owned", "Config": {"Labels": {"tldw.run_id": "recovered"}},
        "NetworkSettings": {"Networks": {}},
    }]))
    monkeypatch.setattr(module.subprocess, "check_output", inspect)
    monkeypatch.setattr(module.DockerRunner, "_is_container_running", lambda _: False)
    monkeypatch.setattr(module.subprocess, "run", Mock(return_value=SimpleNamespace(returncode=0)))
    monkeypatch.setattr(module.DockerRunner, "_cleanup_egress_resources", Mock())
    assert module.DockerRunner.cancel_run("recovered") is True
    assert "tldw-sbx-recovered" in inspect.call_args.args[0]


def test_docker_cancel_cannot_use_an_unrelated_named_container(monkeypatch):
    import json

    from tldw_Server_API.app.core.Sandbox.runners import docker_runner as module

    monkeypatch.setattr(module.DockerRunner, "_active_cid", {})
    monkeypatch.setattr(module.subprocess, "check_output", Mock(return_value=json.dumps([{
        "Id": "unrelated", "Config": {"Labels": {"tldw.run_id": "another-run"}},
    }])))
    kill = Mock()
    monkeypatch.setattr(module.subprocess, "run", kill)
    assert module.DockerRunner.cancel_run("recovered") is False
    kill.assert_not_called()


def test_vz_launch_persists_actual_identity_before_execution(monkeypatch, tmp_path):
    from tldw_Server_API.app.core.Sandbox.models import RunSpec
    from tldw_Server_API.app.core.Sandbox.runners.vz_linux_runner import VZLinuxRunner

    monkeypatch.delenv("TLDW_SANDBOX_VZ_LINUX_FAKE_EXEC", raising=False)
    identities = []
    helper = Mock()
    helper.validate_template.return_value = {"ready": True, "source": "ubuntu"}
    helper.create_vm.return_value = SimpleNamespace(vm_id="actual-vm-id", details={})
    helper.exec_guest.side_effect = lambda **_: (
        identities == ["actual-vm-id"] and SimpleNamespace(exit_code=0, stdout=b"", stderr=b"", details={})
    )
    helper.terminate_vm.return_value = True
    monkeypatch.setattr(VZLinuxRunner, "helper_client_cls", staticmethod(lambda: helper))
    spec = RunSpec(session_id=None, runtime=RuntimeType.vz_linux, base_image="ubuntu", command=["true"],
                   on_backend_started=identities.append)
    result = VZLinuxRunner().start_run("durable-vm", spec, str(tmp_path))
    assert identities == ["actual-vm-id"]
    assert result.phase == RunPhase.completed


def test_vz_failed_cleanup_keeps_run_active_and_ownership(monkeypatch, tmp_path):
    from tldw_Server_API.app.core.Sandbox.models import RunSpec
    from tldw_Server_API.app.core.Sandbox.runners.vz_linux_runner import VZLinuxRunner

    monkeypatch.delenv("TLDW_SANDBOX_VZ_LINUX_FAKE_EXEC", raising=False)
    monkeypatch.setattr(VZLinuxRunner, "_active_vm", {})
    helper = Mock()
    helper.validate_template.return_value = {"ready": True, "source": "ubuntu"}
    helper.create_vm.return_value = SimpleNamespace(vm_id="still-live", details={})
    helper.exec_guest.return_value = SimpleNamespace(exit_code=0, stdout=b"", stderr=b"", details={})
    helper.terminate_vm.return_value = False
    monkeypatch.setattr(VZLinuxRunner, "helper_client_cls", staticmethod(lambda: helper))
    spec = RunSpec(session_id=None, runtime=RuntimeType.vz_linux, base_image="ubuntu", command=["true"])
    result = VZLinuxRunner().start_run("unconfirmed-vm", spec, str(tmp_path))
    assert result.phase == RunPhase.running
    assert result.finished_at is None
    assert VZLinuxRunner._active_vm["unconfirmed-vm"] == "still-live"


def test_vz_cancel_can_use_durable_backend_identity(monkeypatch):
    from tldw_Server_API.app.core.Sandbox.runners.vz_linux_runner import VZLinuxRunner

    monkeypatch.setattr(VZLinuxRunner, "_active_vm", {})
    helper = Mock()
    helper.terminate_vm.return_value = True
    monkeypatch.setattr(VZLinuxRunner, "helper_client_cls", staticmethod(lambda: helper))
    assert VZLinuxRunner.cancel_run("rehydrated", backend_id="actual-vm-id") is True
    helper.terminate_vm.assert_called_once_with("actual-vm-id")


def test_lima_failed_stop_retains_ownership(monkeypatch):
    from tldw_Server_API.app.core.Sandbox.runners import lima_runner as module

    monkeypatch.setattr(module.LimaRunner, "_active_vm", {"retained": "lima-owned"})
    monkeypatch.setattr(module.subprocess, "run", Mock(return_value=SimpleNamespace(returncode=1)))
    assert module.LimaRunner.cancel_run("retained") is False
    assert module.LimaRunner._active_vm["retained"] == "lima-owned"


def test_service_persists_backend_identity_before_launch_and_updates_actual_id():
    from tldw_Server_API.app.core.Sandbox.models import RunSpec

    store = InMemoryStore()
    store.put_run("user", RunStatus(id="launch", phase=RunPhase.starting, runtime=RuntimeType.vz_linux))
    service = SandboxService.__new__(SandboxService)
    service._claim_worker_id = "worker"
    service._orch = SimpleNamespace(
        get_run=store.get_run, update_run=lambda _id, status: store.update_run(status),
        renew_run_claim=Mock(return_value=True),
    )
    spec = RunSpec(session_id=None, runtime=RuntimeType.vz_linux, base_image="ubuntu", command=["true"])

    def launch():
        saved = store.get_run("launch")
        assert saved.backend_id == "launch"
        assert saved.backend_owner == service._backend_host_identity()
        spec.on_backend_started("actual-backend")
        assert store.get_run("launch").backend_id == "actual-backend"
        return "executed"

    assert service._run_with_claim_lease("launch", launch, spec=spec) == "executed"
    assert spec.on_backend_started is None


def test_background_unconfirmed_workload_does_not_emit_end_or_completion_audit(monkeypatch):
    from tldw_Server_API.app.core.Sandbox import service as module
    from tldw_Server_API.app.core.Sandbox.models import RunSpec
    from tldw_Server_API.app.core.Sandbox.runtime_capabilities import RuntimePreflightResult

    monkeypatch.setenv("SANDBOX_ENABLE_EXECUTION", "1")
    monkeypatch.setenv("SANDBOX_BACKGROUND_EXECUTION", "1")
    service = SandboxService()
    service._orch._store = InMemoryStore()
    monkeypatch.setattr(service, "_collect_runtime_preflights", lambda **_: {
        RuntimeType.docker: RuntimePreflightResult(runtime=RuntimeType.docker, available=True,
                                                 enforcement_ready={"deny_all": True}),
    })
    monkeypatch.setattr(service, "_submit_background_worker", lambda worker: worker())
    monkeypatch.setattr(module.DockerRunner, "start_run", lambda *args: RunStatus(
        id="", phase=RunPhase.running, message="backend_termination_unconfirmed",
    ))
    hub = Mock()
    monkeypatch.setattr(module, "get_hub", lambda: hub)
    audit = Mock()
    monkeypatch.setattr(module, "UnifiedAuditService", audit)
    spec = RunSpec(session_id=None, runtime=RuntimeType.docker, base_image="python:3.11", command=["true"])

    status = service.start_run_scaffold(user_id="user", spec=spec, spec_version="1.0", idem_key=None,
                                        raw_body={"command": ["true"]})
    assert status.phase == RunPhase.running
    assert not any(call.args[1] == "end" for call in hub.publish_event.call_args_list)
    audit.assert_not_called()


def test_vz_ambiguous_session_execution_does_not_release_capacity(monkeypatch, tmp_path):
    from tldw_Server_API.app.core.Sandbox.models import RunSpec
    from tldw_Server_API.app.core.Sandbox.runners.vz_linux_runner import VZLinuxRunner

    monkeypatch.delenv("TLDW_SANDBOX_VZ_LINUX_FAKE_EXEC", raising=False)
    monkeypatch.setattr(VZLinuxRunner, "_active_vm", {})
    helper = Mock()
    helper.validate_template.return_value = {"ready": True, "source": "ubuntu"}
    helper.create_vm.return_value = SimpleNamespace(vm_id="session-uncertain", details={})
    helper.exec_guest.side_effect = TimeoutError("command submitted but reply lost")
    helper.terminate_vm.return_value = False
    monkeypatch.setattr(VZLinuxRunner, "helper_client_cls", staticmethod(lambda: helper))
    spec = RunSpec(session_id="session", runtime=RuntimeType.vz_linux, base_image="ubuntu", command=["true"])

    status = VZLinuxRunner().start_run("uncertain", spec, str(tmp_path))
    assert status.phase == RunPhase.running
    assert status.finished_at is None
    assert VZLinuxRunner._active_vm["uncertain"] == "session-uncertain"
    helper.terminate_vm.assert_called_once_with("session-uncertain")


def test_vz_startup_failure_preserves_actual_identity_before_validation(monkeypatch, tmp_path):
    from tldw_Server_API.app.core.Sandbox.models import RunSpec
    from tldw_Server_API.app.core.Sandbox.runners import vz_linux_runner as module

    monkeypatch.delenv("TLDW_SANDBOX_VZ_LINUX_FAKE_EXEC", raising=False)
    monkeypatch.setattr(module.VZLinuxRunner, "_active_vm", {})
    helper = Mock()
    helper.validate_template.return_value = {"ready": True, "source": "ubuntu"}
    helper.create_vm.return_value = SimpleNamespace(vm_id="early-vm", details={"invalid": True})
    helper.terminate_vm.return_value = False
    monkeypatch.setattr(module.VZLinuxRunner, "helper_client_cls", staticmethod(lambda: helper))
    monkeypatch.setattr(module, "classify_vz_linux_guest_agent", lambda _: {
        "compatibility": "mismatch", "reasons": ["wrong_guest"],
    })
    identities = []
    spec = RunSpec(session_id=None, runtime=RuntimeType.vz_linux, base_image="ubuntu", command=["true"],
                   on_backend_started=identities.append)

    status = module.VZLinuxRunner().start_run("early", spec, str(tmp_path))
    assert status.phase == RunPhase.running
    assert identities == ["early-vm"]
    assert module.VZLinuxRunner._active_vm["early"] == "early-vm"
