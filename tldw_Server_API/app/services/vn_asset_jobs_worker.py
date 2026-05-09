"""Jobs worker entrypoint for VN asset generation."""

from __future__ import annotations

import asyncio
import contextlib
import os
from pathlib import Path
from typing import Any

from loguru import logger

from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user_id
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.DB_Management.VNAssetPacks_DB import VNAssetPacksRepository
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Jobs.worker_sdk import WorkerConfig, WorkerSDK
from tldw_Server_API.app.core.testing import env_flag_enabled
from tldw_Server_API.app.core.VN_Assets.jobs import (
    VN_ASSET_ENQUEUE_BATCH_JOB_TYPE,
    VN_ASSET_GENERATE_VARIANT_JOB_TYPE,
    VN_ASSETS_DOMAIN,
    VN_PACK_EXPORT_JOB_TYPE,
    VN_PACK_IMPORT_COMMIT_JOB_TYPE,
    VN_PACK_IMPORT_PREVIEW_JOB_TYPE,
    vn_asset_generation_jobs_queue,
    vn_asset_jobs_queue,
)
from tldw_Server_API.app.core.VN_Assets.storage import resolve_vn_asset_storage_path
from tldw_Server_API.app.core.VN_Assets.worker import VNAssetGenerationWorker
from tldw_Server_API.app.services.storage_quota_service import get_storage_service


def _close_worker_database(db: Any) -> None:
    if db is None:
        return
    if hasattr(db, "release_context_connection"):
        db.release_context_connection()
        return
    if hasattr(db, "close_connection"):
        db.close_connection()


async def handle_vn_asset_job(job: dict[str, Any], *, job_manager: JobManager | None = None) -> dict[str, Any]:
    if str(job.get("job_type") or "").strip() not in {
        VN_ASSET_ENQUEUE_BATCH_JOB_TYPE,
        VN_ASSET_GENERATE_VARIANT_JOB_TYPE,
        VN_PACK_EXPORT_JOB_TYPE,
        VN_PACK_IMPORT_COMMIT_JOB_TYPE,
        VN_PACK_IMPORT_PREVIEW_JOB_TYPE,
    }:
        raise ValueError("unsupported_vn_asset_worker_job_type")

    job_type = str(job.get("job_type") or "").strip()
    payload = job.get("payload") or {}
    owner_user_id = _job_owner_user_id(job)
    payload_user_id = _payload_user_id(payload)
    if payload_user_id != owner_user_id:
        raise ValueError("vn_asset_job_owner_mismatch")

    note_db = await get_chacha_db_for_user_id(
        owner_user_id,
        client_id=f"vn-asset-worker-{owner_user_id}",
    )
    try:
        repo = VNAssetPacksRepository.initialized(note_db)
        worker_dependencies: dict[str, Any] = {}
        if job_type == VN_PACK_EXPORT_JOB_TYPE:
            worker_dependencies.update(await _export_dependencies(owner_user_id))
        if job_type == VN_PACK_IMPORT_COMMIT_JOB_TYPE:
            worker_dependencies.update(await _import_commit_dependencies())
        worker = VNAssetGenerationWorker(
            repo=repo,
            jobs_manager=job_manager or JobManager(),
            **worker_dependencies,
        )
        return await worker.handle_job_async(job)
    finally:
        _close_worker_database(note_db)


def _job_owner_user_id(job: dict[str, Any]) -> int:
    try:
        owner_user_id = int(str(job.get("owner_user_id") or "").strip())
    except (TypeError, ValueError) as exc:
        raise ValueError("missing_owner_user_id") from exc
    if owner_user_id <= 0:
        raise ValueError("missing_owner_user_id")
    return owner_user_id


def _payload_user_id(payload: Any) -> int:
    try:
        user_id = int(payload["user_id"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("missing_user_id") from exc
    if user_id <= 0:
        raise ValueError("missing_user_id")
    return user_id


async def _export_dependencies(owner_user_id: int) -> dict[str, Any]:
    storage_service = await get_storage_service()
    files_repo = await storage_service.get_generated_files_repo()
    return {
        "generated_files_repo": files_repo,
        "read_generated_file_bytes": _read_generated_file_bytes_for_user(owner_user_id),
        "export_staging_root": _vn_pack_export_staging_root(owner_user_id),
    }


async def _import_commit_dependencies() -> dict[str, Any]:
    storage_service = await get_storage_service()
    return {
        "unregister_generated_file": storage_service.unregister_generated_file,
        "preflight_storage_quota": storage_service.check_combined_quota,
    }


def _read_generated_file_bytes_for_user(owner_user_id: int) -> Any:
    def _reader(file_record: dict[str, Any]) -> bytes:
        storage_path = str(file_record.get("storage_path") or "")
        resolved = resolve_vn_asset_storage_path(
            user_id=owner_user_id,
            storage_path=storage_path,
        )
        return resolved.read_bytes()

    return _reader


def _vn_pack_export_staging_root(owner_user_id: int) -> Path:
    configured = (os.getenv("VN_PACK_EXPORT_STAGING_ROOT") or "").strip()
    if configured:
        return Path(configured) / str(owner_user_id)
    return DatabasePaths.get_user_temp_outputs_dir(owner_user_id) / "vn_pack_exports"


async def _should_cancel(job: dict[str, Any], *, job_manager: JobManager | None = None) -> bool:
    jm = job_manager or JobManager()
    current = jm.get_job(int(job["id"]))
    if not current:
        return False
    if current.get("cancel_requested_at"):
        jm.finalize_cancelled(int(job["id"]), reason=str(current.get("cancellation_reason") or "requested"))
        return True
    return str(current.get("status") or "").strip().lower() == "cancelled"


async def run_vn_asset_jobs_worker(stop_event: asyncio.Event | None = None) -> None:
    await _run_vn_asset_jobs_worker(
        stop_event,
        queue=vn_asset_jobs_queue(),
        worker_id_env="VN_ASSET_JOBS_WORKER_ID",
        worker_id_default=f"vn-asset-worker-{os.getpid()}",
    )


async def run_vn_asset_generation_jobs_worker(stop_event: asyncio.Event | None = None) -> None:
    await _run_vn_asset_jobs_worker(
        stop_event,
        queue=vn_asset_generation_jobs_queue(),
        worker_id_env="VN_ASSET_GENERATION_JOBS_WORKER_ID",
        worker_id_default=f"vn-asset-generation-worker-{os.getpid()}",
    )


async def _run_vn_asset_jobs_worker(
    stop_event: asyncio.Event | None,
    *,
    queue: str,
    worker_id_env: str,
    worker_id_default: str,
) -> None:
    worker_id = (os.getenv(worker_id_env) or worker_id_default).strip()
    cfg = WorkerConfig(
        domain=VN_ASSETS_DOMAIN,
        queue=queue,
        worker_id=worker_id,
        lease_seconds=int(os.getenv("VN_ASSET_JOBS_LEASE_SECONDS", os.getenv("JOBS_LEASE_SECONDS", "120")) or "120"),
        renew_threshold_seconds=int(os.getenv("VN_ASSET_JOBS_RENEW_THRESHOLD_SECONDS", "10") or "10"),
        renew_jitter_seconds=int(os.getenv("VN_ASSET_JOBS_RENEW_JITTER_SECONDS", "0") or "0"),
    )
    jm = JobManager()
    sdk = WorkerSDK(jm, cfg)
    stop_task: asyncio.Task[None] | None = None
    if stop_event is not None:

        async def _watch_stop() -> None:
            await stop_event.wait()
            sdk.stop()

        stop_task = asyncio.create_task(_watch_stop(), name="vn_asset_jobs_worker_stop_watch")

    logger.info("VN asset Jobs worker starting: queue={} worker_id={}", cfg.queue, worker_id)
    try:
        await sdk.run(
            handler=lambda job_row: handle_vn_asset_job(job_row, job_manager=jm),
            cancel_check=lambda job_row: _should_cancel(job_row, job_manager=jm),
        )
    finally:
        if stop_task is not None:
            stop_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await stop_task


async def start_vn_asset_jobs_worker(stop_event: asyncio.Event | None = None) -> asyncio.Task | None:
    if not env_flag_enabled("VN_ASSET_JOBS_WORKER_ENABLED"):
        return None
    return asyncio.create_task(
        run_vn_asset_jobs_worker(stop_event),
        name="vn_asset_jobs_worker",
    )


async def start_vn_asset_generation_jobs_worker(stop_event: asyncio.Event | None = None) -> asyncio.Task | None:
    if not env_flag_enabled("VN_ASSET_GENERATION_JOBS_WORKER_ENABLED"):
        return None
    return asyncio.create_task(
        run_vn_asset_generation_jobs_worker(stop_event),
        name="vn_asset_generation_jobs_worker",
    )
