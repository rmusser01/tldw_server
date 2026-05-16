from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.VLLM_Management.executors.base import ProbeResult
from tldw_Server_API.app.core.VLLM_Management.reconciler import VLLMReconciler
from tldw_Server_API.app.core.VLLM_Management.sqlite_repo import SqliteVLLMInstanceRepository
from tldw_Server_API.app.api.v1.schemas.vllm_management import VLLMInstanceCreateRequest


class _ProbeExecutor:
    def probe(self, instance):  # noqa: ANN001
        return ProbeResult(
            status="healthy",
            reachable=True,
            base_url="http://127.0.0.1:8015/v1",
            capabilities={"chat": True, "vision": True},
        )


@pytest.mark.unit
def test_reconciler_updates_observed_state_and_base_url(tmp_path):
    repo = SqliteVLLMInstanceRepository(db_path=tmp_path / "vllm_instances.db")
    instance = repo.create_instance(
        VLLMInstanceCreateRequest(
            name="vision-box",
            execution_mode="local",
            launch_spec={"model": "Qwen/Qwen2.5-VL-7B-Instruct", "port": 8015},
            declared_capabilities={"chat": True, "vision": True},
        ).to_domain()
    )
    repo.update_instance_runtime(
        instance.instance_id,
        {
            "desired_state": "running",
            "observed_state": "starting",
        },
    )

    reconciler = VLLMReconciler(repository=repo, executors={"local": _ProbeExecutor()})

    summary = reconciler.reconcile_once()
    updated = repo.get_instance(instance.instance_id)

    assert summary["reconciled"] == 1
    assert updated is not None
    assert updated.observed_state == "healthy"
    assert updated.last_known_base_url == "http://127.0.0.1:8015/v1"
    assert updated.effective_capabilities["vision"] is True


@pytest.mark.asyncio
@pytest.mark.unit
async def test_run_loop_reconciles_starting_running_instance_and_stops_cleanly():
    stop_event = asyncio.Event()
    probed_instance_ids: list[str] = []
    managed_instance = SimpleNamespace(
        instance_id="instance-123",
        desired_state="running",
        observed_state="starting",
    )

    class _Repository:
        def list_instances(self):  # noqa: ANN001
            return [managed_instance]

    class _Service:
        repository = _Repository()

        def probe_instance(self, instance_id: str) -> None:
            probed_instance_ids.append(instance_id)
            stop_event.set()

    reconciler = VLLMReconciler(service=_Service(), interval_seconds=60)

    await asyncio.wait_for(reconciler.run_loop(stop_event), timeout=1.0)

    assert probed_instance_ids == ["instance-123"]
    assert stop_event.is_set() is True


@pytest.mark.asyncio
@pytest.mark.unit
async def test_run_loop_survives_probe_exception_and_retries(monkeypatch: pytest.MonkeyPatch):
    stop_event = asyncio.Event()
    probe_attempts: list[str] = []
    managed_instance = SimpleNamespace(
        instance_id="instance-456",
        desired_state="running",
        observed_state="starting",
    )

    class _Repository:
        def list_instances(self):  # noqa: ANN001
            return [managed_instance]

    class _Service:
        repository = _Repository()

        def probe_instance(self, instance_id: str) -> None:
            probe_attempts.append(instance_id)
            if len(probe_attempts) == 1:
                raise RuntimeError("transient probe failure")
            stop_event.set()

    reconciler = VLLMReconciler(service=_Service(), interval_seconds=1)
    monkeypatch.setattr(reconciler, "interval_seconds", 0.01)

    await asyncio.wait_for(reconciler.run_loop(stop_event), timeout=1.0)

    assert probe_attempts == ["instance-456", "instance-456"]
    assert stop_event.is_set() is True


@pytest.mark.asyncio
@pytest.mark.unit
async def test_run_loop_stops_before_probing_remaining_instances_once_stop_requested():
    stop_event = asyncio.Event()
    probe_attempts: list[str] = []
    instances = [
        SimpleNamespace(instance_id="instance-1", desired_state="running", observed_state="starting"),
        SimpleNamespace(instance_id="instance-2", desired_state="running", observed_state="starting"),
    ]

    class _Repository:
        def list_instances(self):  # noqa: ANN001
            return instances

    class _Service:
        repository = _Repository()

        def probe_instance(self, instance_id: str) -> None:
            probe_attempts.append(instance_id)
            if instance_id == "instance-1":
                stop_event.set()

    reconciler = VLLMReconciler(service=_Service(), interval_seconds=60)

    await asyncio.wait_for(reconciler.run_loop(stop_event), timeout=1.0)

    assert probe_attempts == ["instance-1"]
    assert stop_event.is_set() is True
