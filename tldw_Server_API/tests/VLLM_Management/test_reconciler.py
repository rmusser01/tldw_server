from __future__ import annotations

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
