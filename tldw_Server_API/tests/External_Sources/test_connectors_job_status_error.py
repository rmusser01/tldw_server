import pytest
from fastapi import HTTPException

from tldw_Server_API.app.api.v1.endpoints import connectors


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_job_status_sanitizes_job_manager_error(monkeypatch):
    class _BrokenJobManager:
        def get_job(self, job_id):
            assert job_id == 123
            raise RuntimeError("job backend exploded")

    import tldw_Server_API.app.core.Jobs.manager as jobs_manager

    monkeypatch.setattr(jobs_manager, "JobManager", _BrokenJobManager)

    with pytest.raises(HTTPException) as exc_info:
        await connectors.get_job_status(123)

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to get connector job status"
