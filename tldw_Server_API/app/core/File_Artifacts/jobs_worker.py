"""
File Artifacts Jobs worker.

- Consumes core Jobs entries for file artifact exports.
- domain = "files"
- queue = os.getenv("FILES_JOBS_QUEUE", "default")
- job_type = "file_artifact_export"

Payload fields:
- file_id (required)
- user_id (required)
- export_format (required)
- max_bytes (optional)
- export_ttl_seconds (optional)

Usage:
  python -m tldw_Server_API.app.core.File_Artifacts.jobs_worker
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
from typing import Any

from loguru import logger

from tldw_Server_API.app.api.v1.schemas.file_artifacts_schemas import FileCreateOptions
from tldw_Server_API.app.core.DB_Management.Collections_DB import CollectionsDatabase
from tldw_Server_API.app.core.exceptions import FileArtifactsJobError
from tldw_Server_API.app.core.File_Artifacts.file_artifacts_service import FileArtifactsService
from tldw_Server_API.app.core.Jobs.worker_sdk import WorkerConfig, WorkerSDK
from tldw_Server_API.app.core.Jobs.worker_utils import coerce_int as _coerce_int
from tldw_Server_API.app.core.Jobs.worker_utils import jobs_manager_from_env as _jobs_manager

FILES_DOMAIN = "files"
FILES_JOB_TYPE = "file_artifact_export"


def _resolve_user_id(job: dict[str, Any], payload: dict[str, Any]) -> str:
    candidate = payload.get("user_id") or job.get("owner_user_id")
    if candidate is None or str(candidate).strip() == "":
        raise FileArtifactsJobError("missing user_id", retryable=False)
    return str(candidate)


async def _handle_export_job(job: dict[str, Any]) -> dict[str, Any]:
    payload = job.get("payload") or {}
    job_type = str(job.get("job_type") or payload.get("job_type") or "").strip().lower()
    if job_type and job_type != FILES_JOB_TYPE:
        raise FileArtifactsJobError(f"unsupported job_type: {job_type}", retryable=False)

    user_id = _resolve_user_id(job, payload)
    file_id_raw = payload.get("file_id")
    try:
        file_id = int(file_id_raw)
    except (TypeError, ValueError):
        raise FileArtifactsJobError("invalid file_id", retryable=False) from None

    export_format = str(payload.get("export_format") or "").strip().lower()
    if not export_format:
        raise FileArtifactsJobError("missing export_format", retryable=False)

    with CollectionsDatabase.for_user(user_id=user_id) as cdb:
        service = FileArtifactsService(cdb, user_id=user_id)
        try:
            row = cdb.get_file_artifact(file_id)
        except KeyError:
            raise FileArtifactsJobError("file_artifact_not_found", retryable=False) from None

        if row.export_status == "ready" and row.export_storage_path:
            return {"skipped": True, "status": "ready"}

        file_type = row.file_type
        if file_type in {"csv_table", "json_table"}:
            file_type = "data_table"
        adapter = service.get_adapter(file_type)
        if adapter is None:
            raise FileArtifactsJobError(f"adapter_missing:{file_type}", retryable=False)

        try:
            structured = json.loads(row.structured_json)
        except (json.JSONDecodeError, TypeError) as exc:
            raise FileArtifactsJobError(f"structured_json_invalid:{exc}", retryable=False) from exc

        options = FileCreateOptions(
            persist=True,
            max_bytes=payload.get("max_bytes"),
            export_ttl_seconds=payload.get("export_ttl_seconds"),
        )
        try:
            export_info = await service.export_artifact_for_job(
                adapter=adapter,
                structured=structured,
                file_id=file_id,
                export_format=export_format,
                options=options,
            )
            return {
                "status": export_info.status,
                "bytes": export_info.bytes,
                "expires_at": export_info.expires_at.isoformat() if export_info.expires_at else None,
            }
        except Exception as exc:
            logger.error("file_artifacts worker: export failed error_type={}", type(exc).__name__)
            try:
                cdb.update_file_artifact_export(
                    file_id,
                    export_status="none",
                    export_format=row.export_format or export_format,
                    export_storage_path=None,
                    export_bytes=row.export_bytes,
                    export_content_type=row.export_content_type,
                    export_job_id=row.export_job_id,
                    export_expires_at=None,
                    export_consumed_at=None,
                )
            except Exception as reset_exc:
                logger.warning(
                    "file_artifacts worker: failed to reset export status error_type={}",
                    type(reset_exc).__name__,
                )
            raise FileArtifactsJobError(str(exc), retryable=False) from exc


async def run_file_artifacts_jobs_worker(stop_event: asyncio.Event | None = None) -> None:
    """Run the file artifacts jobs worker until stopped."""
    worker_id = (os.getenv("FILES_JOBS_WORKER_ID") or f"files-jobs-{os.getpid()}").strip()
    queue = (os.getenv("FILES_JOBS_QUEUE") or "default").strip() or "default"
    lease_seconds = _coerce_int(os.getenv("FILES_JOBS_LEASE_SECONDS") or os.getenv("JOBS_LEASE_SECONDS"), 60)
    renew_jitter = _coerce_int(os.getenv("FILES_JOBS_RENEW_JITTER_SECONDS") or os.getenv("JOBS_LEASE_RENEW_JITTER_SECONDS"), 5)
    renew_threshold = _coerce_int(os.getenv("FILES_JOBS_RENEW_THRESHOLD_SECONDS") or os.getenv("JOBS_LEASE_RENEW_THRESHOLD_SECONDS"), 10)
    cfg = WorkerConfig(
        domain=FILES_DOMAIN,
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

    logger.info("File Artifacts Jobs worker starting (queue={}, worker_id={})", queue, worker_id)
    try:
        await sdk.run(handler=_handle_export_job)
    finally:
        if _stop_watcher_task is not None and not _stop_watcher_task.done():
            _stop_watcher_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await _stop_watcher_task


if __name__ == "__main__":
    asyncio.run(run_file_artifacts_jobs_worker())
