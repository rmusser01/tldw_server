"""Jobs-backed contract helpers for llama.cpp asset acquisition."""

from __future__ import annotations

from typing import Any

from tldw_Server_API.app.api.v1.schemas.llamacpp_admin_schemas import (
    LlamaCppAcquisitionJobListResponse,
    LlamaCppAcquisitionJobResponse,
    LlamaCppAssetDownloadRequest,
)
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Local_LLM import llamacpp_acquisition_service


LLAMACPP_ACQUISITION_DOMAIN = "llamacpp"
LLAMACPP_ACQUISITION_QUEUE = "acquisition"
LLAMACPP_DOWNLOAD_JOB_TYPE = "llamacpp_asset_download"


def create_download_job(
    job_manager: JobManager,
    payload: LlamaCppAssetDownloadRequest,
    *,
    owner_user_id: str | None,
) -> LlamaCppAcquisitionJobResponse:
    """Validate a download request and create a sanitized llama.cpp acquisition job."""
    validated = llamacpp_acquisition_service.validate_download_request(payload)
    job_payload = {
        "operation": "download",
        "source_url": validated.source_url,
        "source_label": validated.source_label,
        "destination_path": str(validated.destination_path),
        "expected_sha256": validated.expected_sha256,
        "expected_size_bytes": validated.expected_size_bytes,
        "overwrite": validated.overwrite,
        "register_asset": validated.register_asset,
        "warnings": list(validated.warnings),
    }
    job = job_manager.create_job(
        domain=LLAMACPP_ACQUISITION_DOMAIN,
        queue=LLAMACPP_ACQUISITION_QUEUE,
        job_type=LLAMACPP_DOWNLOAD_JOB_TYPE,
        payload=job_payload,
        owner_user_id=owner_user_id,
        priority=5,
        max_retries=3,
    )
    job_id = job.get("id")
    if job_id is not None:
        refreshed = job_manager.get_job(int(job_id))
        if refreshed is not None:
            job = refreshed
    return job_to_response(job)


def get_download_job(job_manager: JobManager, job_id: int) -> LlamaCppAcquisitionJobResponse | None:
    """Return one llama.cpp acquisition job if it belongs to the acquisition domain."""
    job = job_manager.get_job(int(job_id))
    if not _is_llamacpp_download_job(job):
        return None
    return job_to_response(job)


def list_download_jobs(job_manager: JobManager, *, limit: int = 100) -> LlamaCppAcquisitionJobListResponse:
    """Return recent llama.cpp acquisition download jobs."""
    jobs = job_manager.list_jobs(
        domain=LLAMACPP_ACQUISITION_DOMAIN,
        queue=LLAMACPP_ACQUISITION_QUEUE,
        job_type=LLAMACPP_DOWNLOAD_JOB_TYPE,
        limit=limit,
    )
    return LlamaCppAcquisitionJobListResponse(jobs=[job_to_response(job) for job in jobs])


def cancel_download_job(job_manager: JobManager, job_id: int) -> LlamaCppAcquisitionJobResponse | None:
    """Request cancellation for a llama.cpp acquisition job and return its new state."""
    job = job_manager.get_job(int(job_id))
    if not _is_llamacpp_download_job(job):
        return None
    job_manager.cancel_job(int(job_id), reason="cancelled_by_admin")
    return get_download_job(job_manager, int(job_id))


def job_to_response(job: dict[str, Any]) -> LlamaCppAcquisitionJobResponse:
    """Map a raw Jobs row into the llama.cpp acquisition API response."""
    payload = _dict_value(job.get("payload"))
    result = _dict_value(job.get("result"))
    progress = _dict_value(payload.get("progress"))
    if result:
        progress.update(_dict_value(result.get("progress")))
    warnings = _string_list(payload.get("warnings")) + _string_list(result.get("warnings") if result else None)
    return LlamaCppAcquisitionJobResponse(
        job_id=str(job.get("id") or ""),
        status=str(job.get("status") or "unknown"),
        operation="download",
        queue=str(job.get("queue") or LLAMACPP_ACQUISITION_QUEUE),
        source_label=_optional_str(payload.get("source_label")),
        destination_path=_optional_str(payload.get("destination_path")),
        asset_id=_optional_str(result.get("asset_id") if result else None),
        progress=progress,
        warnings=warnings,
        error_message=_optional_str(job.get("last_error") or job.get("error_message")),
    )


def _is_llamacpp_download_job(job: dict[str, Any] | None) -> bool:
    return bool(
        job
        and job.get("domain") == LLAMACPP_ACQUISITION_DOMAIN
        and job.get("queue") == LLAMACPP_ACQUISITION_QUEUE
        and job.get("job_type") == LLAMACPP_DOWNLOAD_JOB_TYPE
    )


def _dict_value(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]
