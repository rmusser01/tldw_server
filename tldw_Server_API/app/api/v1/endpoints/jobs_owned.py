"""Authenticated owner status reads, deliberately separate from Jobs administration."""

import os
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_auth_principal
from tldw_Server_API.app.api.v1.schemas.jobs_owned import OwnedJobStatus
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Jobs.owned_status import get_owned_job_status

router = APIRouter()


@router.get("/jobs/owned/{job_uuid}", response_model=OwnedJobStatus)
def get_owned_job_status_endpoint(
    job_uuid: UUID, principal: AuthPrincipal = Depends(get_auth_principal)
) -> OwnedJobStatus:
    """Resolve a logical Job without revealing foreign or unsupported Jobs."""
    db_url = os.getenv("JOBS_DB_URL")
    backend = "postgres" if db_url and db_url.startswith("postgres") else None
    manager = JobManager(backend=backend, db_url=db_url)
    job = get_owned_job_status(manager, job_uuid=str(job_uuid), owner_user_id=str(principal.user_id))
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job
