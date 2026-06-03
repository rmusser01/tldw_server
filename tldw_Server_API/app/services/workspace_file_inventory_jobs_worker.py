from __future__ import annotations

import asyncio
import contextlib
import json
import os
from pathlib import Path
from typing import Any, Sequence

from loguru import logger

from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user_id
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Jobs.worker_sdk import WorkerConfig, WorkerSDK
from tldw_Server_API.app.core.Jobs.worker_utils import coerce_int as _coerce_int
from tldw_Server_API.app.core.Jobs.worker_utils import jobs_manager_from_env as _jobs_manager
from tldw_Server_API.app.core.Workspaces.file_inventory_ignore import build_inventory_ignore_policy
from tldw_Server_API.app.core.Workspaces.file_inventory_jobs import (
    WORKSPACE_FILE_INVENTORY_JOB_TYPE,
    WORKSPACE_JOBS_DOMAIN,
    workspace_file_inventory_jobs_queue,
)
from tldw_Server_API.app.core.Workspaces.file_inventory_models import normalize_inventory_counts
from tldw_Server_API.app.core.Workspaces.file_inventory_scanner import (
    InventoryScanBounds,
    scan_workspace_file_inventory,
)
from tldw_Server_API.app.core.Workspaces.root_binding_service import (
    SandboxInventoryMountResolver,
    resolve_workspace_root_for_inventory_scan,
)


class WorkspaceFileInventoryJobError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        retryable: bool = False,
        failure_code: str = "workspace_file_inventory_job_failed",
    ) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.failure_code = failure_code


async def handle_workspace_file_inventory_job(
    job: dict[str, Any],
    *,
    db: Any | None = None,
    allowed_roots: Sequence[Path | str] | None = None,
    sandbox_mount_resolver: SandboxInventoryMountResolver | None = None,
) -> dict[str, Any]:
    payload = _validate_job_payload(job)
    worker_db = db
    loaded_db = False
    if worker_db is None:
        worker_db = await _get_database_for_job(job)
        loaded_db = True
    try:
        return await asyncio.to_thread(
            _handle_workspace_file_inventory_job_sync,
            payload,
            worker_db,
            allowed_roots,
            sandbox_mount_resolver,
        )
    finally:
        if loaded_db:
            _close_worker_database(worker_db)


def _handle_workspace_file_inventory_job_sync(
    payload: dict[str, Any],
    db: Any,
    allowed_roots: Sequence[Path | str] | None,
    sandbox_mount_resolver: SandboxInventoryMountResolver | None,
) -> dict[str, Any]:
    workspace_id = str(payload["workspace_id"])
    root_id = str(payload["root_id"])
    scan_id = str(payload["scan_id"])
    root_version = int(payload["root_version"])
    policy_fingerprint = str(payload["ignore_policy_fingerprint"])

    root = db.get_workspace_primary_root(workspace_id)
    if root is None or str(root.get("root_id") or "") != root_id:
        return _complete_failed_scan(
            db,
            scan_id=scan_id,
            code="workspace_project_root_missing",
            message="Workspace project root was not found.",
        )
    if int(root.get("version") or 0) != root_version:
        return _complete_failed_scan(
            db,
            scan_id=scan_id,
            code="root_version_mismatch",
            message="Workspace project root changed before scan started.",
        )

    resolution = resolve_workspace_root_for_inventory_scan(
        root=root,
        allowed_roots=allowed_roots,
        sandbox_mount_resolver=sandbox_mount_resolver,
    )
    if not resolution.ok or resolution.local_path is None:
        return _complete_failed_scan(
            db,
            scan_id=scan_id,
            code=resolution.failure_code or "workspace_project_root_unavailable",
            message=resolution.message or "Workspace project root is unavailable.",
        )

    db.mark_workspace_file_inventory_scanning(scan_id)
    policy = build_inventory_ignore_policy()
    scan_result = scan_workspace_file_inventory(
        resolution.local_path,
        policy=policy,
        bounds=InventoryScanBounds(),
    )
    state = "current" if scan_result.coverage_complete else "partial"
    items_written = db.replace_workspace_file_inventory_items(
        workspace_id,
        root_id,
        scan_id,
        list(scan_result.items),
        scan_coverage_complete=scan_result.coverage_complete,
    )
    completed = db.complete_workspace_file_inventory_scan(
        scan_id,
        state,
        scan_result.counts,
        list(scan_result.diagnostics),
        root_snapshot_token=resolution.root_snapshot_token,
    )
    return {
        "scan_id": scan_id,
        "state": completed["state"],
        "counts": scan_result.counts,
        "diagnostics": list(scan_result.diagnostics),
        "items_written": items_written,
        "ignore_policy_fingerprint": policy_fingerprint,
    }


def _validate_job_payload(job: dict[str, Any]) -> dict[str, Any]:
    job_type = str(job.get("job_type") or "").strip()
    if job_type != WORKSPACE_FILE_INVENTORY_JOB_TYPE:
        raise WorkspaceFileInventoryJobError(
            f"unsupported job_type: {job_type or '<missing>'}",
            retryable=False,
            failure_code="unsupported_job_type",
        )
    payload = _coerce_payload(job.get("payload"))
    required = ("workspace_id", "root_id", "scan_id", "root_version", "ignore_policy_fingerprint")
    for field_name in required:
        if payload.get(field_name) in (None, ""):
            raise WorkspaceFileInventoryJobError(
                f"missing {field_name}",
                retryable=False,
                failure_code="invalid_job_payload",
            )
    try:
        payload["root_version"] = int(payload["root_version"])
    except (TypeError, ValueError) as exc:
        raise WorkspaceFileInventoryJobError(
            "invalid root_version",
            retryable=False,
            failure_code="invalid_job_payload",
        ) from exc
    return payload


def _coerce_payload(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        try:
            loaded = json.loads(value)
        except json.JSONDecodeError as exc:
            raise WorkspaceFileInventoryJobError(
                "payload must be an object",
                retryable=False,
                failure_code="invalid_job_payload",
            ) from exc
        if isinstance(loaded, dict):
            return dict(loaded)
    raise WorkspaceFileInventoryJobError(
        "payload must be an object",
        retryable=False,
        failure_code="invalid_job_payload",
    )


def _complete_failed_scan(db: Any, *, scan_id: str, code: str, message: str) -> dict[str, Any]:
    diagnostics = [{"code": code, "message": message}]
    completed = db.complete_workspace_file_inventory_scan(
        scan_id,
        "failed",
        normalize_inventory_counts({}),
        diagnostics,
        root_snapshot_token=None,
    )
    return {
        "scan_id": scan_id,
        "state": completed["state"],
        "counts": normalize_inventory_counts({}),
        "diagnostics": diagnostics,
        "items_written": 0,
    }


async def _get_database_for_job(job: dict[str, Any]) -> Any:
    owner_user_id = str(job.get("owner_user_id") or "").strip()
    if not owner_user_id:
        raise WorkspaceFileInventoryJobError(
            "missing owner_user_id",
            retryable=False,
            failure_code="invalid_job_payload",
        )
    try:
        normalized_user_id = int(owner_user_id)
    except ValueError as exc:
        raise WorkspaceFileInventoryJobError(
            "invalid owner_user_id",
            retryable=False,
            failure_code="invalid_job_payload",
        ) from exc
    return await get_chacha_db_for_user_id(
        normalized_user_id,
        client_id=f"workspace-file-inventory-worker-{normalized_user_id}",
    )


def _close_worker_database(db: Any) -> None:
    if db is None:
        return
    if hasattr(db, "release_context_connection"):
        db.release_context_connection()
        return
    if hasattr(db, "close_connection"):
        db.close_connection()


async def _should_cancel(job: dict[str, Any], *, job_manager: JobManager) -> bool:
    current = job_manager.get_job(int(job["id"]))
    if not current:
        return False
    if str(current.get("status") or "").strip().lower() == "cancelled":
        return True
    if current.get("cancel_requested_at"):
        job_manager.finalize_cancelled(
            int(job["id"]),
            reason=str(current.get("cancellation_reason") or "requested"),
        )
        return True
    return False


async def run_workspace_file_inventory_jobs_worker(stop_event: asyncio.Event | None = None) -> None:
    worker_id = (
        os.getenv("WORKSPACE_FILE_INVENTORY_JOBS_WORKER_ID")
        or f"workspace-file-inventory-worker-{os.getpid()}"
    ).strip()
    cfg = WorkerConfig(
        domain=WORKSPACE_JOBS_DOMAIN,
        queue=workspace_file_inventory_jobs_queue(),
        worker_id=worker_id,
        lease_seconds=_coerce_int(
            os.getenv("WORKSPACE_FILE_INVENTORY_JOBS_LEASE_SECONDS") or os.getenv("JOBS_LEASE_SECONDS"),
            60,
        ),
        renew_jitter_seconds=_coerce_int(
            os.getenv("WORKSPACE_FILE_INVENTORY_JOBS_RENEW_JITTER_SECONDS")
            or os.getenv("JOBS_LEASE_RENEW_JITTER_SECONDS"),
            5,
        ),
        renew_threshold_seconds=_coerce_int(
            os.getenv("WORKSPACE_FILE_INVENTORY_JOBS_RENEW_THRESHOLD_SECONDS")
            or os.getenv("JOBS_LEASE_RENEW_THRESHOLD_SECONDS"),
            10,
        ),
    )
    jm = _jobs_manager()
    sdk = WorkerSDK(jm, cfg)
    stop_watcher_task: asyncio.Task[None] | None = None

    if stop_event is not None:

        async def _watch_stop() -> None:
            await stop_event.wait()
            sdk.stop()

        stop_watcher_task = asyncio.create_task(_watch_stop())

    logger.info("Workspace file inventory Jobs worker starting: queue={} worker_id={}", cfg.queue, worker_id)
    try:
        await sdk.run(
            handler=handle_workspace_file_inventory_job,
            cancel_check=lambda job_row: _should_cancel(job_row, job_manager=jm),
        )
    finally:
        if stop_watcher_task is not None and not stop_watcher_task.done():
            stop_watcher_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await stop_watcher_task


__all__ = [
    "WorkspaceFileInventoryJobError",
    "handle_workspace_file_inventory_job",
    "run_workspace_file_inventory_jobs_worker",
]
