from __future__ import annotations

import importlib
from typing import Any

import pytest


@pytest.fixture
def fake_job_manager() -> object:
    return object()


@pytest.mark.asyncio
async def test_worker_rejects_unrelated_job_type(fake_job_manager: Any) -> None:
    worker = importlib.import_module("tldw_Server_API.app.services.research_workspace_output_jobs_worker")
    output_jobs = importlib.import_module("tldw_Server_API.app.core.Research_Workspace.output_jobs")

    with pytest.raises(output_jobs.ResearchWorkspaceOutputJobError) as excinfo:
        await worker.process_research_workspace_output_job(
            {"id": 1, "job_type": "other", "payload": {}},
            job_manager=fake_job_manager,
        )

    assert excinfo.value.retryable is False
