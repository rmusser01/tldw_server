from __future__ import annotations

import asyncio

import pytest

from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.VLLM_Management.executors.base import LifecycleResult, ProbeResult, StopResult
from tldw_Server_API.app.core.VLLM_Management.service import VLLMManagementService
from tldw_Server_API.app.core.VLLM_Management.sqlite_repo import SqliteVLLMInstanceRepository
from tldw_Server_API.app.api.v1.schemas.vllm_management import VLLMInstanceCreateRequest
from tldw_Server_API.app.services import vllm_management_worker


class _StubExecutor:
    def __init__(self, *, stop_event: asyncio.Event | None = None) -> None:
        self.stop_event = stop_event

    def start(self, instance):  # noqa: ANN001
        if self.stop_event is not None:
            self.stop_event.set()
        return LifecycleResult(
            status="started",
            base_url="http://127.0.0.1:8014/v1",
            handle={"pid": 321},
            log_handle={"stdout_path": "/tmp/vllm.out"},
        )

    def stop(self, instance, handle):  # noqa: ANN001
        return StopResult(status="stopped", forced=False)

    def probe(self, instance):  # noqa: ANN001
        return ProbeResult(
            status="healthy",
            reachable=True,
            base_url="http://127.0.0.1:8014/v1",
            capabilities={"chat": True, "embeddings": True},
        )


@pytest.mark.asyncio
async def test_worker_processes_start_job_and_updates_instance_state(tmp_path):
    repo = SqliteVLLMInstanceRepository(db_path=tmp_path / "vllm_instances.db")
    instance = repo.create_instance(
        VLLMInstanceCreateRequest(
            name="worker-box",
            execution_mode="local",
            launch_spec={"model": "BAAI/bge-m3", "port": 8014},
            declared_capabilities={"chat": True, "embeddings": True},
        ).to_domain()
    )
    jobs_db = tmp_path / "jobs.db"
    jm = JobManager(jobs_db)
    stop_event = asyncio.Event()
    executor = _StubExecutor(stop_event=stop_event)
    service = VLLMManagementService(
        repository=repo,
        job_manager=jm,
        executors={"local": executor},
    )
    job = service.enqueue_start(instance.instance_id, owner_user_id="1")

    await asyncio.wait_for(
        vllm_management_worker.run_vllm_management_worker(
            stop_event=stop_event,
            job_manager=jm,
            service=service,
        ),
        timeout=2,
    )

    job_row = jm.get_job(int(job["id"])) or {}
    updated = repo.get_instance(instance.instance_id)

    assert job_row["status"] == "completed"
    assert updated is not None
    assert updated.observed_state == "healthy"
    assert updated.last_known_base_url == "http://127.0.0.1:8014/v1"
    assert updated.executor_handle["pid"] == 321
    assert updated.effective_capabilities["embeddings"] is True
