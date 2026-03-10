"""Worker handlers for managed vLLM lifecycle jobs."""

from __future__ import annotations

from typing import Any

from .service import VLLMManagementService, VLLM_JOB_TYPE_BY_ACTION

ACTION_BY_JOB_TYPE = {value: key for key, value in VLLM_JOB_TYPE_BY_ACTION.items()}


async def handle_vllm_management_job(
    job: dict[str, Any],
    *,
    service: VLLMManagementService,
) -> dict[str, Any]:
    payload = job.get("payload") or {}
    if not isinstance(payload, dict):
        payload = {}
    job_type = str(job.get("job_type") or "").strip().lower()
    action = str(payload.get("action") or ACTION_BY_JOB_TYPE.get(job_type) or "").strip().lower()
    if action not in VLLM_JOB_TYPE_BY_ACTION:
        raise ValueError(f"Unsupported managed vLLM job action: {action or job_type}")
    instance_id = str(payload.get("instance_id") or "").strip()
    if not instance_id:
        raise ValueError("Managed vLLM job payload missing instance_id")
    return service.execute_action(action, instance_id)
