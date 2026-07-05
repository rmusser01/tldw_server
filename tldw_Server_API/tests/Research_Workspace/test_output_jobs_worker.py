from __future__ import annotations

from contextlib import contextmanager
import importlib
from typing import Any

import pytest


@pytest.fixture
def fake_job_manager() -> object:
    return object()


def _worker_module() -> Any:
    return importlib.import_module("tldw_Server_API.app.services.research_workspace_output_jobs_worker")


def _output_jobs_module() -> Any:
    return importlib.import_module("tldw_Server_API.app.core.Research_Workspace.output_jobs")


@pytest.mark.asyncio
async def test_worker_rejects_unrelated_job_type(fake_job_manager: Any) -> None:
    worker = _worker_module()
    output_jobs = _output_jobs_module()

    with pytest.raises(output_jobs.ResearchWorkspaceOutputJobError) as excinfo:
        await worker.process_research_workspace_output_job(
            {"id": 1, "job_type": "other", "payload": {}},
            job_manager=fake_job_manager,
        )

    assert excinfo.value.retryable is False


@pytest.mark.asyncio
async def test_worker_processes_valid_job_with_open_databases(
    monkeypatch: pytest.MonkeyPatch,
    fake_job_manager: Any,
) -> None:
    worker = _worker_module()
    output_jobs = _output_jobs_module()
    notes_db = object()
    media_db = object()
    closed_dbs: list[object] = []
    media_calls: list[dict[str, object]] = []
    delegated: dict[str, object] = {}

    async def _open_notes_db(user_id: int) -> object:
        assert user_id == 7
        return notes_db

    @contextmanager
    def _managed_media_database(client_id: str, **kwargs: object):
        media_calls.append({"client_id": client_id, **kwargs})
        yield media_db

    async def _process_payload(**kwargs: object) -> dict[str, object]:
        delegated.update(kwargs)
        return {"artifact_id": "artifact-1"}

    monkeypatch.setattr(worker, "open_research_workspace_output_notes_db", _open_notes_db)
    monkeypatch.setattr(worker, "close_research_workspace_output_notes_db", closed_dbs.append)
    monkeypatch.setattr(worker, "managed_media_database", _managed_media_database)
    monkeypatch.setattr(worker, "process_research_workspace_output_payload", _process_payload)

    result = await worker.process_research_workspace_output_job(
        {
            "id": 10,
            "job_type": output_jobs.RESEARCH_WORKSPACE_OUTPUT_JOB_TYPE,
            "payload": {"user_id": "7", "artifact_id": "artifact-1"},
        },
        job_manager=fake_job_manager,
    )

    assert result == {"artifact_id": "artifact-1"}
    assert closed_dbs == [notes_db]
    assert media_calls == [
        {
            "client_id": "research_workspace_output_worker",
            "db_path": str(worker.DatabasePaths.get_media_db_path(7)),
            "initialize": False,
        }
    ]
    assert delegated["workspace_db"] is notes_db
    assert delegated["media_db"] is media_db
    assert delegated["job_manager"] is fake_job_manager
    assert delegated["user_id"] == 7
    assert delegated["payload"] == {"user_id": "7", "artifact_id": "artifact-1"}


@pytest.mark.asyncio
async def test_worker_runner_filters_worker_sdk_to_research_output_jobs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker = _worker_module()
    output_jobs = _output_jobs_module()
    captured: dict[str, object] = {}

    class _FakeWorkerSDK:
        def __init__(self, job_manager: object, config: object) -> None:
            captured["job_manager"] = job_manager
            captured["config"] = config

        async def run(self, **kwargs: object) -> None:
            captured["run_kwargs"] = kwargs

        def stop(self) -> None:
            captured["stopped"] = True

    monkeypatch.setenv("RESEARCH_WORKSPACE_OUTPUT_JOBS_WORKER_ID", "rw-output-test")
    monkeypatch.setenv("RESEARCH_WORKSPACE_OUTPUT_JOBS_QUEUE", "high")
    monkeypatch.setattr(worker, "WorkerSDK", _FakeWorkerSDK)

    await worker.run_research_workspace_output_jobs_worker()

    config = captured["config"]
    assert config.worker_id == "rw-output-test"
    assert config.queue == "high"
    assert captured["run_kwargs"]["job_type"] == output_jobs.RESEARCH_WORKSPACE_OUTPUT_JOB_TYPE
    assert callable(captured["run_kwargs"]["handler"])
    assert callable(captured["run_kwargs"]["progress_cb"])
