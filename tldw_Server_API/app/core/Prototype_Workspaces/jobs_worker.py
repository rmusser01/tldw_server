"""Jobs worker for prototype workspace runtime orchestration."""
from __future__ import annotations

import asyncio
import contextlib
import os
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.Jobs.worker_sdk import WorkerConfig, WorkerSDK
from tldw_Server_API.app.core.Jobs.worker_utils import coerce_int as _coerce_int
from tldw_Server_API.app.core.Jobs.worker_utils import jobs_manager_from_env as _jobs_manager

from .jobs import PROTOTYPE_DOMAIN, PROTOTYPE_QUEUE
from .models import PrototypeJobType
from .service import PrototypeWorkspaceService


async def handle_prototype_job(
    job: dict[str, Any],
    *,
    service: PrototypeWorkspaceService,
) -> dict[str, Any]:
    """Dispatch a single prototype runtime job through the service layer."""
    payload = job.get("payload") or {}
    job_type = str(job.get("job_type") or payload.get("job_type") or "").strip().lower()
    if not job_type:
        raise ValueError("missing prototype job_type")

    if job_type == PrototypeJobType.BRANCH_SESSION_BOOTSTRAP.value:
        result = await service.create_or_reuse_branch_session(
            prototype_workspace_id=str(payload.get("prototype_workspace_id") or ""),
            base_snapshot_id=payload.get("base_snapshot_id"),
            actor_type=str(payload.get("actor_type") or ""),
            actor_user_id=payload.get("actor_user_id"),
            actor_shared_actor_id=payload.get("actor_shared_actor_id"),
            request_nonce=payload.get("request_nonce"),
            share_link_id=payload.get("share_link_id"),
            expires_at=payload.get("expires_at"),
        )
        session = result.get("session") or {}
        return {
            "status": "ok",
            "session_id": session.get("id"),
            "created": bool(result.get("created")),
        }

    if job_type == PrototypeJobType.PREVIEW_BOOT.value:
        grant = await service.boot_preview(
            prototype_workspace_id=str(payload.get("prototype_workspace_id") or ""),
            prototype_session_id=payload.get("prototype_session_id"),
            snapshot_id=str(payload.get("snapshot_id") or ""),
            runtime_target_url=str(payload.get("runtime_target_url") or ""),
            metadata=payload.get("metadata") or {},
            runtime_policy_profile=payload.get("runtime_policy_profile"),
        )
        return {
            "status": "ok",
            "preview_handle": grant.get("preview_handle"),
            "preview_url": grant.get("preview_url"),
        }

    if job_type == PrototypeJobType.SNAPSHOT_SAVE.value:
        snapshot = await service.save_session_snapshot(
            prototype_session_id=str(payload.get("prototype_session_id") or ""),
            snapshot_id=payload.get("snapshot_id"),
            storage_ref=payload.get("storage_ref"),
            diff_summary=payload.get("diff_summary") or {},
            prompt_summary=payload.get("prompt_summary"),
            preview_health=payload.get("preview_health") or {},
        )
        return {
            "status": "ok",
            "snapshot_id": snapshot.get("snapshot_id"),
        }

    if job_type == PrototypeJobType.PUBLISH_VALIDATE_AND_PROMOTE.value:
        result = await service.promote_candidate(
            prototype_workspace_id=str(payload.get("prototype_workspace_id") or ""),
            candidate_snapshot_id=str(payload.get("candidate_snapshot_id") or ""),
            reviewer_user_id=int(payload.get("reviewer_user_id")),
            review_baseline_snapshot_id=payload.get("review_baseline_snapshot_id"),
            promotion_request_id=payload.get("promotion_request_id"),
            review_notes=payload.get("review_notes"),
        )
        return result

    raise ValueError(f"unsupported prototype job_type: {job_type}")


async def run_prototype_jobs_worker(
    *,
    service: PrototypeWorkspaceService,
    stop_event: asyncio.Event | None = None,
) -> None:
    """Run the prototype workspace jobs worker until stopped."""
    worker_id = (os.getenv("PROTOTYPE_WORKSPACE_WORKER_ID") or f"prototype-jobs-{os.getpid()}").strip()
    queue = (os.getenv("PROTOTYPE_WORKSPACE_JOBS_QUEUE") or PROTOTYPE_QUEUE).strip() or PROTOTYPE_QUEUE
    lease_seconds = _coerce_int(
        os.getenv("PROTOTYPE_WORKSPACE_JOBS_LEASE_SECONDS") or os.getenv("JOBS_LEASE_SECONDS"),
        60,
    )
    renew_jitter = _coerce_int(
        os.getenv("PROTOTYPE_WORKSPACE_JOBS_RENEW_JITTER_SECONDS")
        or os.getenv("JOBS_LEASE_RENEW_JITTER_SECONDS"),
        5,
    )
    renew_threshold = _coerce_int(
        os.getenv("PROTOTYPE_WORKSPACE_JOBS_RENEW_THRESHOLD_SECONDS")
        or os.getenv("JOBS_LEASE_RENEW_THRESHOLD_SECONDS"),
        10,
    )
    cfg = WorkerConfig(
        domain=PROTOTYPE_DOMAIN,
        queue=queue,
        worker_id=worker_id,
        lease_seconds=lease_seconds,
        renew_jitter_seconds=renew_jitter,
        renew_threshold_seconds=renew_threshold,
    )
    sdk = WorkerSDK(_jobs_manager(), cfg)
    _stop_watcher_task: asyncio.Task[None] | None = None

    if stop_event is not None:
        async def _watch_stop() -> None:
            await stop_event.wait()
            sdk.stop()

        _stop_watcher_task = asyncio.create_task(_watch_stop())

    logger.info("Prototype workspace jobs worker starting (queue={}, worker_id={})", queue, worker_id)
    try:
        await sdk.run(handler=lambda job: handle_prototype_job(job, service=service))
    finally:
        if _stop_watcher_task is not None and not _stop_watcher_task.done():
            _stop_watcher_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await _stop_watcher_task


if __name__ == "__main__":
    raise SystemExit(
        "Prototype workspace jobs worker requires an injected PrototypeWorkspaceService. "
        "Import run_prototype_jobs_worker() from the application bootstrap."
    )
