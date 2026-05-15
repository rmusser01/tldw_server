"""Jobs helpers for prototype workspace runtime orchestration."""
from __future__ import annotations

import hashlib
import json
import os
from typing import Any

from tldw_Server_API.app.core.AuthNZ.repos.prototype_workspaces_repo import (
    PrototypeWorkspacesRepo,
)
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Jobs.worker_utils import jobs_manager_from_env

from .models import PrototypeJobType, actor_key_for_session

PROTOTYPE_DOMAIN = "prototype_workspaces"
PROTOTYPE_QUEUE = "default"
PROTOTYPE_JOB_TYPES = {
    PrototypeJobType.BRANCH_SESSION_BOOTSTRAP.value,
    PrototypeJobType.PREVIEW_BOOT.value,
    PrototypeJobType.SNAPSHOT_SAVE.value,
    PrototypeJobType.PUBLISH_VALIDATE_AND_PROMOTE.value,
}
_DEFAULT_JOBS_MANAGER: JobManager | None = None
_DEFAULT_JOBS_MANAGER_KEY: tuple[str, str] | None = None


def _jobs_manager_environment_key() -> tuple[str, str]:
    return (
        str(os.getenv("JOBS_DB_URL") or ""),
        str(os.getenv("JOBS_DB_PATH") or ""),
    )


def _get_default_jobs_manager() -> JobManager:
    global _DEFAULT_JOBS_MANAGER
    global _DEFAULT_JOBS_MANAGER_KEY

    key = _jobs_manager_environment_key()
    if _DEFAULT_JOBS_MANAGER is None or key != _DEFAULT_JOBS_MANAGER_KEY:
        _DEFAULT_JOBS_MANAGER = jobs_manager_from_env()
        _DEFAULT_JOBS_MANAGER_KEY = key
    return _DEFAULT_JOBS_MANAGER


def build_branch_session_bootstrap_idempotency_key(
    *,
    prototype_workspace_id: str,
    base_snapshot_id: str,
    actor_type: str,
    actor_user_id: int | None = None,
    actor_shared_actor_id: str | None = None,
    request_nonce: str,
) -> str:
    actor_key = actor_key_for_session(
        actor_type=actor_type,
        actor_user_id=actor_user_id,
        actor_shared_actor_id=actor_shared_actor_id,
    )
    nonce = str(request_nonce or "").strip()
    if not nonce:
        raise ValueError("request_nonce is required")
    return (
        "prototype:branch:"
        f"{prototype_workspace_id}:{actor_key}:{base_snapshot_id}:{nonce}"
    )


def build_preview_boot_idempotency_key(
    *,
    prototype_session_id: str | None,
    prototype_workspace_id: str,
    snapshot_id: str,
    runtime_target_url: str,
    runtime_profile_version: str | None = None,
) -> str:
    scope_token = prototype_session_id or f"canonical:{prototype_workspace_id}"
    profile_version = str(runtime_profile_version or "v1")
    target_fingerprint = hashlib.sha256(
        str(runtime_target_url or "").encode("utf-8")
    ).hexdigest()[:16]
    return f"prototype:preview:{scope_token}:{snapshot_id}:{profile_version}:{target_fingerprint}"


def build_snapshot_save_idempotency_key(
    *,
    prototype_session_id: str,
    save_request_id: str,
) -> str:
    request_id = str(save_request_id or "").strip()
    if not request_id:
        raise ValueError("save_request_id is required")
    return f"prototype:snapshot-save:{prototype_session_id}:{request_id}"


def build_promote_idempotency_key(
    *,
    prototype_workspace_id: str,
    candidate_snapshot_id: str,
    canonical_snapshot_id: str,
) -> str:
    return (
        "prototype:promote:"
        f"{prototype_workspace_id}:{candidate_snapshot_id}:{canonical_snapshot_id}"
    )


class PrototypeWorkspaceJobs:
    """Create stable Jobs entries for prototype runtime operations."""

    def __init__(
        self,
        *,
        repo: PrototypeWorkspacesRepo,
        jobs_manager: JobManager | None = None,
        queue: str = PROTOTYPE_QUEUE,
    ) -> None:
        self._repo = repo
        self._jobs_manager = jobs_manager if jobs_manager is not None else _get_default_jobs_manager()
        self._queue = str(queue or PROTOTYPE_QUEUE)

    @staticmethod
    def _normalize_job_row(job: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(job)
        payload = normalized.get("payload")
        if isinstance(payload, str):
            try:
                decoded = json.loads(payload)
            except (TypeError, ValueError, json.JSONDecodeError):
                decoded = payload
            normalized["payload"] = decoded
        return normalized

    async def enqueue_branch_session_bootstrap(
        self,
        *,
        prototype_workspace_id: str,
        actor_type: str,
        actor_user_id: int | None = None,
        actor_shared_actor_id: str | None = None,
        request_nonce: str,
        share_link_id: int | None = None,
        expires_at: str | None = None,
    ) -> dict[str, Any]:
        workspace = await self._repo.get_workspace(prototype_workspace_id)
        if not workspace:
            raise ValueError("prototype workspace not found")

        base_snapshot_id = str(
            workspace.get("canonical_snapshot_id")
            or workspace.get("last_known_good_snapshot_id")
            or ""
        ).strip()
        if not base_snapshot_id:
            raise ValueError("prototype workspace does not have a canonical snapshot")

        payload = {
            "prototype_workspace_id": prototype_workspace_id,
            "base_snapshot_id": base_snapshot_id,
            "actor_type": actor_type,
            "actor_user_id": actor_user_id,
            "actor_shared_actor_id": actor_shared_actor_id,
            "share_link_id": share_link_id,
            "expires_at": expires_at,
            "request_nonce": request_nonce,
        }
        return self._normalize_job_row(self._jobs_manager.create_job(
            domain=PROTOTYPE_DOMAIN,
            queue=self._queue,
            job_type=PrototypeJobType.BRANCH_SESSION_BOOTSTRAP.value,
            payload=payload,
            owner_user_id=str(workspace["owner_user_id"]),
            idempotency_key=build_branch_session_bootstrap_idempotency_key(
                prototype_workspace_id=prototype_workspace_id,
                base_snapshot_id=base_snapshot_id,
                actor_type=actor_type,
                actor_user_id=actor_user_id,
                actor_shared_actor_id=actor_shared_actor_id,
                request_nonce=request_nonce,
            ),
        ))

    async def enqueue_preview_boot(
        self,
        *,
        prototype_workspace_id: str,
        snapshot_id: str,
        runtime_target_url: str,
        prototype_session_id: str | None = None,
        runtime_profile_version: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        workspace = await self._repo.get_workspace(prototype_workspace_id)
        if not workspace:
            raise ValueError("prototype workspace not found")

        resolved_runtime_profile_version = runtime_profile_version or "v1"
        runtime_metadata = {
            **(metadata or {}),
            "runtime_profile_version": resolved_runtime_profile_version,
        }
        return self._normalize_job_row(self._jobs_manager.create_job(
            domain=PROTOTYPE_DOMAIN,
            queue=self._queue,
            job_type=PrototypeJobType.PREVIEW_BOOT.value,
            payload={
                "prototype_workspace_id": prototype_workspace_id,
                "prototype_session_id": prototype_session_id,
                "snapshot_id": snapshot_id,
                "runtime_target_url": runtime_target_url,
                "metadata": runtime_metadata,
                "runtime_profile_version": resolved_runtime_profile_version,
            },
            owner_user_id=str(workspace["owner_user_id"]),
            idempotency_key=build_preview_boot_idempotency_key(
                prototype_workspace_id=prototype_workspace_id,
                prototype_session_id=prototype_session_id,
                snapshot_id=snapshot_id,
                runtime_target_url=runtime_target_url,
                runtime_profile_version=resolved_runtime_profile_version,
            ),
        ))

    async def enqueue_snapshot_save(
        self,
        *,
        prototype_session_id: str,
        save_request_id: str,
        snapshot_id: str | None = None,
        storage_ref: str | None = None,
        diff_summary: dict[str, Any] | None = None,
        prompt_summary: str | None = None,
        preview_health: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        session = await self._repo.get_session(prototype_session_id)
        if not session:
            raise ValueError("prototype session not found")
        workspace = await self._repo.get_workspace(str(session["prototype_workspace_id"]))
        if not workspace:
            raise ValueError("prototype workspace not found")

        return self._normalize_job_row(self._jobs_manager.create_job(
            domain=PROTOTYPE_DOMAIN,
            queue=self._queue,
            job_type=PrototypeJobType.SNAPSHOT_SAVE.value,
            payload={
                "prototype_session_id": prototype_session_id,
                "snapshot_id": snapshot_id,
                "save_request_id": save_request_id,
                "storage_ref": storage_ref,
                "diff_summary": diff_summary or {},
                "prompt_summary": prompt_summary,
                "preview_health": preview_health or {},
            },
            owner_user_id=str(workspace["owner_user_id"]),
            idempotency_key=build_snapshot_save_idempotency_key(
                prototype_session_id=prototype_session_id,
                save_request_id=save_request_id,
            ),
        ))

    async def enqueue_publish_validate_and_promote(
        self,
        *,
        prototype_workspace_id: str,
        candidate_snapshot_id: str,
        reviewer_user_id: int,
        review_baseline_snapshot_id: str | None = None,
        promotion_request_id: str | None = None,
        review_notes: str | None = None,
    ) -> dict[str, Any]:
        workspace = await self._repo.get_workspace(prototype_workspace_id)
        if not workspace:
            raise ValueError("prototype workspace not found")

        review_baseline = str(
            review_baseline_snapshot_id
            or workspace.get("last_known_good_snapshot_id")
            or workspace.get("canonical_snapshot_id")
            or ""
        ).strip()
        if not review_baseline:
            raise ValueError("prototype workspace does not have a canonical snapshot")

        return self._normalize_job_row(self._jobs_manager.create_job(
            domain=PROTOTYPE_DOMAIN,
            queue=self._queue,
            job_type=PrototypeJobType.PUBLISH_VALIDATE_AND_PROMOTE.value,
            payload={
                "prototype_workspace_id": prototype_workspace_id,
                "candidate_snapshot_id": candidate_snapshot_id,
                "reviewer_user_id": int(reviewer_user_id),
                "review_baseline_snapshot_id": review_baseline,
                "promotion_request_id": promotion_request_id,
                "review_notes": review_notes,
                "canonical_snapshot_id": review_baseline,
            },
            owner_user_id=str(workspace["owner_user_id"]),
            idempotency_key=build_promote_idempotency_key(
                prototype_workspace_id=prototype_workspace_id,
                candidate_snapshot_id=candidate_snapshot_id,
                canonical_snapshot_id=review_baseline,
            ),
        ))


class PrototypeRuntimeJobs(PrototypeWorkspaceJobs):
    """Compatibility alias for tests and future callers."""
