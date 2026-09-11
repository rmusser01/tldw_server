from __future__ import annotations

import contextlib
import json
import os
from datetime import datetime
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Prompt_Management.prompt_studio.quota_config import (
    apply_prompt_studio_quota_defaults,
)

_PROMPT_STUDIO_DOMAIN = "prompt_studio"
_ENTITY_LOOKUP_PAGE_SIZE = 100


def _jobs_backend() -> str:
    backend = (os.getenv("PROMPT_STUDIO_JOBS_BACKEND") or os.getenv("TLDW_JOBS_BACKEND") or "").strip().lower()
    if backend and backend != "core":
        logger.warning("Prompt Studio jobs backend override ignored; only core Jobs is supported now.")
    return "core"


def _jobs_queue() -> str:
    queue = (os.getenv("PROMPT_STUDIO_JOBS_QUEUE") or "default").strip()
    return queue or "default"


def _jobs_manager() -> JobManager:
    apply_prompt_studio_quota_defaults()
    db_url = (os.getenv("JOBS_DB_URL") or "").strip()
    if not db_url:
        return JobManager()
    backend = "postgres" if db_url.startswith("postgres") else None
    return JobManager(backend=backend, db_url=db_url)


def _map_status(raw_status: str | None) -> str:
    status = str(raw_status or "").lower()
    if status == "quarantined":
        return "failed"
    if status in {"queued", "processing", "completed", "failed", "cancelled"}:
        return status
    return "queued"


def _normalize_payload(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _require_user_id(user_id: str | None) -> str:
    """Return one explicit tenant owner or fail before querying core Jobs."""

    owner_id = str(user_id).strip() if user_id is not None else ""
    if not owner_id:
        raise ValueError("Prompt Studio job owner is required")
    return owner_id


def _format_datetime(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _cursor_datetime(value: Any) -> datetime:
    """Return a Jobs cursor timestamp or fail instead of truncating the lookup."""
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise RuntimeError("Prompt Studio job lookup returned an invalid cursor") from exc
    raise RuntimeError("Prompt Studio job lookup returned an invalid cursor")


def _entity_id_from_payload(payload: dict[str, Any]) -> int | None:
    for key in ("optimization_id", "evaluation_id", "generation_id", "entity_id"):
        value = payload.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
    return None


class PromptStudioJobsAdapter:
    """Bridge Prompt Studio job views to the core Jobs table when enabled."""

    def __init__(
        self,
        *,
        backend: str | None = None,
    ) -> None:
        if backend and backend != "core":
            logger.warning("Prompt Studio jobs adapter forced to core backend; legacy backend removed.")
        self._backend = "core"
        self._jm = _jobs_manager()

    @property
    def backend(self) -> str:
        return self._backend

    def get_job(
        self,
        job_id: str,
        *,
        db,
        user_id: str | None,
        job_type: str | None = None,
    ) -> dict[str, Any] | None:
        owner_id = _require_user_id(user_id)
        if self._backend == "core":
            job = self._lookup_core_job(
                job_id,
                user_id=owner_id,
                job_type=job_type,
            )
            if job is not None:
                return self._format_job(job)
        return None

    def list_jobs(
        self,
        *,
        db,
        user_id: str | None,
        job_type: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        owner_id = _require_user_id(user_id)
        if self._backend == "core":
            jobs = self._jm.list_jobs(
                domain=_PROMPT_STUDIO_DOMAIN,
                queue=None,
                status=None,
                owner_user_id=owner_id,
                job_type=job_type,
                limit=max(1, int(limit)),
            )
            return [
                self._format_job(job)
                for job in jobs
                if self._matches(
                    job,
                    user_id=owner_id,
                    job_type=job_type,
                )
            ]
        return []

    def get_latest_job_for_entity(
        self,
        *,
        db,
        user_id: str | None,
        job_type: str,
        entity_id: int,
    ) -> dict[str, Any] | None:
        jobs = self.list_jobs_for_entity(
            db=db,
            user_id=user_id,
            job_type=job_type,
            entity_id=entity_id,
            limit=1,
            ascending=False,
        )
        return jobs[0] if jobs else None

    def list_jobs_for_entity(
        self,
        *,
        db,
        user_id: str | None,
        job_type: str,
        entity_id: int,
        limit: int = 50,
        ascending: bool = True,
    ) -> list[dict[str, Any]]:
        owner_id = _require_user_id(user_id)
        if self._backend == "core":
            requested_limit = max(0, int(limit))
            if requested_limit == 0:
                return []

            matched: list[dict[str, Any]] = []
            created_before: datetime | None = None
            before_id: int | None = None
            while len(matched) < requested_limit:
                raw_jobs = self._jm.list_jobs(
                    domain=_PROMPT_STUDIO_DOMAIN,
                    queue=None,
                    status=None,
                    owner_user_id=owner_id,
                    job_type=job_type,
                    created_before=created_before,
                    before_id=before_id,
                    limit=_ENTITY_LOOKUP_PAGE_SIZE,
                    sort_by="created_at",
                    sort_order="desc",
                )
                for job in raw_jobs:
                    if not self._matches(
                        job,
                        user_id=owner_id,
                        job_type=job_type,
                    ):
                        continue
                    payload = _normalize_payload(job.get("payload"))
                    if _entity_id_from_payload(payload) == int(entity_id):
                        matched.append(job)
                        if len(matched) == requested_limit:
                            break

                if len(matched) == requested_limit or len(raw_jobs) < _ENTITY_LOOKUP_PAGE_SIZE:
                    break

                cursor_job = raw_jobs[-1]
                try:
                    next_before_id = int(cursor_job["id"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise RuntimeError(
                        "Prompt Studio job lookup returned an invalid cursor"
                    ) from exc
                next_created_before = _cursor_datetime(cursor_job.get("created_at"))
                if (next_created_before, next_before_id) == (created_before, before_id):
                    raise RuntimeError("Prompt Studio job lookup did not advance")
                created_before, before_id = next_created_before, next_before_id

            matched.sort(key=lambda row: _format_datetime(row.get("created_at")) or "", reverse=not ascending)
            return [self._format_job(job) for job in matched]
        return []

    def create_job(
        self,
        *,
        user_id: str | None,
        job_type: str,
        entity_id: int | None,
        payload: dict[str, Any] | None,
        project_id: int | None = None,
        priority: int = 5,
        max_retries: int = 3,
        request_id: str | None = None,
        trace_id: str | None = None,
    ) -> dict[str, Any]:
        owner_id = _require_user_id(user_id)
        payload_dict: dict[str, Any] = dict(payload or {})
        normalized_job_type = str(job_type).strip().lower()
        if normalized_job_type == "optimization":
            optimization_uuid = payload_dict.get("optimization_uuid")
            if not isinstance(optimization_uuid, str) or not optimization_uuid.strip():
                raise ValueError(
                    "Prompt Studio optimization_uuid is required for durable jobs"
                )
            payload_dict["optimization_uuid"] = optimization_uuid.strip()
        if entity_id is not None:
            try:
                payload_dict.setdefault("entity_id", int(entity_id))
            except (TypeError, ValueError):
                payload_dict.setdefault("entity_id", entity_id)
        return self._jm.create_job(
            domain=_PROMPT_STUDIO_DOMAIN,
            queue=_jobs_queue(),
            job_type=str(job_type),
            payload=payload_dict,
            owner_user_id=owner_id,
            project_id=project_id,
            priority=priority,
            max_retries=max_retries,
            request_id=request_id,
            trace_id=trace_id,
        )

    def cancel_job(
        self,
        job_id: str,
        *,
        user_id: str | None,
        reason: str | None = None,
        job_type: str | None = None,
    ) -> bool:
        owner_id = _require_user_id(user_id)
        job = self._lookup_core_job(
            job_id,
            user_id=owner_id,
            job_type=job_type,
        )
        if not job:
            return False
        job_uuid = str(job.get("uuid") or "").strip()
        resolved_job_type = str(job.get("job_type") or "").strip()
        if not job_uuid or not resolved_job_type:
            return False
        try:
            return bool(
                self._jm.cancel_job(
                    int(job["id"]),
                    reason=reason,
                    expected_uuid=job_uuid,
                    expected_domain=_PROMPT_STUDIO_DOMAIN,
                    expected_job_type=resolved_job_type,
                )
            )
        except Exception:
            return False

    def _lookup_core_job(
        self,
        job_id: str,
        *,
        user_id: str | None,
        job_type: str | None,
    ) -> dict[str, Any] | None:
        owner_id = _require_user_id(user_id)
        job = None
        if job_id:
            try:
                job = self._jm.get_job_by_uuid(str(job_id))
            except Exception:
                job = None
        if job is None and str(job_id).isdigit():
            try:
                job = self._jm.get_job(int(job_id))
            except Exception:
                job = None
        if job and self._matches(job, user_id=owner_id, job_type=job_type):
            return job

        try:
            jobs = self._jm.list_jobs(
                domain=_PROMPT_STUDIO_DOMAIN,
                queue=None,
                status=None,
                owner_user_id=owner_id,
                job_type=job_type,
                limit=200,
            )
        except Exception as exc:
            logger.debug(f"Prompt studio jobs adapter list failed: {exc}")
            return None

        for candidate in jobs:
            if self._matches(candidate, user_id=owner_id, job_type=job_type):
                cid = candidate.get("uuid") or candidate.get("id")
                if cid and str(cid) == str(job_id):
                    return candidate
        return None

    def _matches(self, job: dict[str, Any], *, user_id: str | None, job_type: str | None) -> bool:
        owner_id = _require_user_id(user_id)
        if not job:
            return False
        if str(job.get("domain")) != _PROMPT_STUDIO_DOMAIN:
            return False
        if job_type and str(job.get("job_type")) != str(job_type):
            return False
        owner = job.get("owner_user_id")
        return owner is not None and str(owner) == owner_id

    def _format_job(self, job: dict[str, Any]) -> dict[str, Any]:
        payload = _normalize_payload(job.get("payload"))
        result = _normalize_payload(job.get("result"))
        progress = job.get("progress_percent")
        formatted: dict[str, Any] = {
            "id": str(job.get("uuid") or job.get("id")),
            "uuid": job.get("uuid"),
            "job_type": job.get("job_type"),
            "status": _map_status(job.get("status")),
            "entity_id": _entity_id_from_payload(payload),
            "project_id": payload.get("project_id") or job.get("project_id"),
            "priority": job.get("priority"),
            "payload": json.dumps(payload),
            "result": json.dumps(result),
            "error_message": job.get("error_message") or job.get("last_error"),
            "created_at": _format_datetime(job.get("created_at")),
            "updated_at": _format_datetime(job.get("updated_at")),
            "started_at": _format_datetime(job.get("started_at") or job.get("acquired_at")),
            "completed_at": _format_datetime(job.get("completed_at")),
        }
        if progress is not None:
            with contextlib.suppress(TypeError, ValueError):
                formatted["progress"] = float(progress) / 100.0
        return formatted

__all__ = ["PromptStudioJobsAdapter"]
