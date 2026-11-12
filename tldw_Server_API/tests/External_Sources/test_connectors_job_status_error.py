import pytest
from fastapi import HTTPException

from tldw_Server_API.app.api.v1.endpoints import connectors
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_job_status_sanitizes_job_manager_error(monkeypatch):
    class _BrokenJobManager:
        def get_job(self, job_id, owner_user_id=None):
            assert job_id == 123
            assert owner_user_id == "7"
            raise RuntimeError("job backend exploded")

    import tldw_Server_API.app.core.Jobs.manager as jobs_manager

    monkeypatch.setattr(jobs_manager, "JobManager", _BrokenJobManager)

    with pytest.raises(HTTPException) as exc_info:
        await connectors.get_job_status(123, principal=AuthPrincipal(kind="user", user_id=7))

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to get connector job status"
