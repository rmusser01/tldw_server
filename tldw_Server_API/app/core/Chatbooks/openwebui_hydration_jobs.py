"""Core Jobs helpers for OpenWebUI attachment hydration."""

from __future__ import annotations

from typing import Any


CHATBOOKS_DOMAIN = "chatbooks"
OPENWEBUI_ATTACHMENT_HYDRATION_JOB_TYPE = "openwebui_attachment_hydration"


def create_openwebui_hydration_job(
    jobs_manager: Any,
    payload: dict[str, Any],
    *,
    owner_user_id: str,
    request_id: str | None = None,
) -> dict[str, Any]:
    """Enqueue an OpenWebUI attachment hydration job in core Jobs."""
    return jobs_manager.create_job(
        domain=CHATBOOKS_DOMAIN,
        queue="default",
        job_type=OPENWEBUI_ATTACHMENT_HYDRATION_JOB_TYPE,
        payload=dict(payload),
        owner_user_id=str(owner_user_id),
        priority=5,
        max_retries=3,
        request_id=request_id,
    )


def get_openwebui_hydration_job(jobs_manager: Any, job_id: str) -> dict[str, Any] | None:
    """Fetch an OpenWebUI hydration job by numeric id or uuid and verify its type."""
    raw_job_id = str(job_id).strip()
    if not raw_job_id:
        return None
    job: dict[str, Any] | None
    if raw_job_id.isdigit():
        job = jobs_manager.get_job(int(raw_job_id))
    else:
        get_by_uuid = getattr(jobs_manager, "get_job_by_uuid", None)
        if get_by_uuid is None:
            return None
        job = get_by_uuid(raw_job_id)
    if not isinstance(job, dict):
        return None
    if str(job.get("domain") or "") != CHATBOOKS_DOMAIN:
        return None
    if str(job.get("job_type") or "") != OPENWEBUI_ATTACHMENT_HYDRATION_JOB_TYPE:
        return None
    return job
