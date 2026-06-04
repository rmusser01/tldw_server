"""Jobs-backed enqueue helpers for Workspace file inventory scans.

This module owns the small boundary between Workspace root state stored in
ChaChaNotes and the shared Jobs manager. It creates durable scan records before
enqueueing Jobs work, attaches the resulting Jobs row back to the scan record,
and reports enqueue failures back into the Workspace inventory projection.
"""

from __future__ import annotations

import os
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.Jobs.manager import JobManager

WORKSPACE_JOBS_DOMAIN = "workspaces"
WORKSPACE_FILE_INVENTORY_JOB_TYPE = "workspace_file_inventory_scan"


class WorkspaceFileInventoryEnqueueError(RuntimeError):
    """Raised when a Workspace inventory scan record cannot be queued as a Jobs task."""

    def __init__(
        self,
        message: str = "Failed to enqueue Workspace file inventory scan.",
        *,
        error_code: str = "workspace_file_inventory_enqueue_failed",
    ) -> None:
        super().__init__(message)
        self.error_code = error_code


def workspace_file_inventory_jobs_queue() -> str:
    """Return the Jobs queue name for Workspace file inventory scan work.

    The queue defaults to ``"default"`` and can be overridden with the
    ``WORKSPACE_FILE_INVENTORY_JOBS_QUEUE`` environment variable. Empty or
    whitespace-only overrides are ignored so workers and enqueuers always share
    a non-empty queue name.
    """
    queue = (os.getenv("WORKSPACE_FILE_INVENTORY_JOBS_QUEUE") or "default").strip()
    return queue or "default"


def build_workspace_file_inventory_job_payload(
    *,
    workspace_id: str,
    root_id: str,
    root_version: int,
    scan_id: str,
    policy_fingerprint: str,
    requested_by: str | None,
) -> dict[str, Any]:
    """Build the durable Jobs payload for a Workspace inventory scan.

    Args:
        workspace_id: Workspace identifier whose primary root is being scanned.
        root_id: Workspace project root identifier captured for the scan.
        root_version: Project root version that the scan should operate on.
        scan_id: Durable scan record identifier created before enqueue.
        policy_fingerprint: Ignore-policy fingerprint used for idempotency and
            stale-scan detection.
        requested_by: Optional user or system actor that requested the scan.

    Returns:
        A JSON-serializable payload for ``JobManager.create_job``.
    """
    payload: dict[str, Any] = {
        "workspace_id": str(workspace_id).strip(),
        "root_id": str(root_id).strip(),
        "root_version": int(root_version),
        "scan_id": str(scan_id).strip(),
        "ignore_policy_fingerprint": str(policy_fingerprint).strip(),
    }
    if requested_by is not None:
        payload["requested_by"] = str(requested_by).strip()
    return payload


def enqueue_workspace_file_inventory_scan_job(
    *,
    db: Any,
    workspace_id: str,
    root_id: str,
    root_version: int,
    policy_fingerprint: str,
    requested_by: str | None,
    owner_user_id: str | None,
    job_manager: JobManager | None = None,
) -> dict[str, Any]:
    """Create a scan record, enqueue Jobs work, and return the attached status.

    Args:
        db: CharactersRAGDB-like object that supports Workspace file inventory
            scan persistence methods.
        workspace_id: Workspace identifier whose root should be scanned.
        root_id: Project root identifier to scan.
        root_version: Version of the root snapshot requested by the caller.
        policy_fingerprint: Ignore-policy fingerprint for this scan request.
        requested_by: Optional user or system actor requesting the scan.
        owner_user_id: Optional Jobs owner used for ownership/RLS attribution.
        job_manager: Optional Jobs manager injection for tests; when omitted a
            default ``JobManager`` is constructed.

    Returns:
        A mapping with ``scan``, ``job``, and ``status`` entries after the Jobs
        row has been attached back to the scan record.

    Raises:
        WorkspaceFileInventoryEnqueueError: If ``JobManager.create_job`` fails.

    Notes:
        The scan record is created before enqueue so failure state can be
        projected back to the Workspace. The Jobs idempotency key is based on the
        durable scan id, so retries for the same scan do not create duplicate
        Jobs rows.
    """
    manager = job_manager or JobManager()
    scan = db.begin_workspace_file_inventory_scan(
        workspace_id,
        root_id,
        root_version,
        policy_fingerprint,
        requested_by=requested_by,
    )
    payload = build_workspace_file_inventory_job_payload(
        workspace_id=workspace_id,
        root_id=root_id,
        root_version=int(scan["root_version"]),
        scan_id=str(scan["scan_id"]),
        policy_fingerprint=policy_fingerprint,
        requested_by=requested_by,
    )
    try:
        created_job = manager.create_job(
            domain=WORKSPACE_JOBS_DOMAIN,
            queue=workspace_file_inventory_jobs_queue(),
            job_type=WORKSPACE_FILE_INVENTORY_JOB_TYPE,
            payload=payload,
            owner_user_id=owner_user_id,
            max_retries=0,
            idempotency_key=f"workspace-file-inventory-scan:{scan['scan_id']}",
        )
    except Exception as exc:
        db.mark_workspace_file_inventory_enqueue_failed(
            str(scan["scan_id"]),
            [
                {
                    "code": "job_enqueue_failed",
                    "message": "Workspace file inventory scan could not be enqueued.",
                }
            ],
        )
        raise WorkspaceFileInventoryEnqueueError() from exc

    job = _reload_job(manager, created_job)
    attached_scan = db.attach_workspace_file_inventory_job(str(scan["scan_id"]), job)
    status = db.get_workspace_file_inventory_status(workspace_id, policy_fingerprint=policy_fingerprint)
    return {"scan": attached_scan, "job": job, "status": status}


def _reload_job(manager: Any, created_job: dict[str, Any]) -> dict[str, Any]:
    try:
        job_id = int(created_job["id"])
    except (KeyError, TypeError, ValueError):
        return created_job
    if not hasattr(manager, "get_job"):
        return created_job
    try:
        return manager.get_job(job_id) or created_job
    except Exception:
        logger.warning(f"Workspace file inventory job reload failed for job_id={job_id}")
        return created_job


__all__ = [
    "WORKSPACE_FILE_INVENTORY_JOB_TYPE",
    "WORKSPACE_JOBS_DOMAIN",
    "WorkspaceFileInventoryEnqueueError",
    "build_workspace_file_inventory_job_payload",
    "enqueue_workspace_file_inventory_scan_job",
    "workspace_file_inventory_jobs_queue",
]
