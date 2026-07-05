from __future__ import annotations

import json
import os
import re
import uuid
from typing import Any, cast

from tldw_Server_API.app.api.v1.schemas.research_workspace_outputs import (
    ResearchWorkspaceOutputArtifactType,
    ResearchWorkspaceOutputSettings,
    ResearchWorkspaceOutputStatus,
    ResearchWorkspaceOutputStatusResponse,
    ResearchWorkspaceOutputSubmitRequest,
    ResearchWorkspaceOutputSubmitResponse,
)
from tldw_Server_API.app.api.v1.schemas.workspace_schemas import WorkspaceArtifactResponse
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.Jobs.manager import JobManager

RESEARCH_WORKSPACE_OUTPUT_JOB_DOMAIN = "research_workspace"
RESEARCH_WORKSPACE_OUTPUT_JOB_QUEUE = "default"
RESEARCH_WORKSPACE_OUTPUT_JOBS_QUEUE_ENV = "RESEARCH_WORKSPACE_OUTPUT_JOBS_QUEUE"
RESEARCH_WORKSPACE_OUTPUT_JOB_TYPE = "research_workspace_output"

_OUTPUT_ARTIFACT_TYPES = {"video_overview", "infographic"}
_PUBLIC_ERROR_CODE_RE = re.compile(r"^[a-z][a-z0-9_]{0,95}$")


def _safe_public_error_code(value: str) -> str:
    raw = str(value or "").strip().lower()
    if _PUBLIC_ERROR_CODE_RE.fullmatch(raw):
        return raw
    return "research_workspace_output_failed"


class ResearchWorkspaceOutputJobError(RuntimeError):
    def __init__(
        self,
        public_code: str,
        *,
        status_code: int = 400,
        retryable: bool = False,
        backoff_seconds: int | None = None,
    ) -> None:
        super().__init__(public_code)
        self.public_code = _safe_public_error_code(public_code)
        self.status_code = status_code
        self.retryable = retryable
        self.backoff_seconds = backoff_seconds


def research_workspace_output_jobs_queue() -> str:
    raw = (os.getenv(RESEARCH_WORKSPACE_OUTPUT_JOBS_QUEUE_ENV) or "").strip().lower()
    if raw in {"default", "high", "low"}:
        return raw
    return RESEARCH_WORKSPACE_OUTPUT_JOB_QUEUE


def submit_research_workspace_output_job(
    *,
    workspace_id: str,
    request: ResearchWorkspaceOutputSubmitRequest,
    workspace_db: CharactersRAGDB,
    job_manager: JobManager,
    user_id: int | str,
) -> ResearchWorkspaceOutputSubmitResponse:
    _validate_workspace_sources(workspace_db, workspace_id, request.source_ids)
    artifact_id = f"{request.artifact_type}-{uuid.uuid4().hex}"
    workspace_db.add_workspace_artifact(
        workspace_id,
        _pending_artifact_payload(
            artifact_id=artifact_id,
            artifact_type=request.artifact_type,
            source_ids=request.source_ids,
            user_id=str(user_id),
            settings=request.settings,
        ),
    )
    try:
        job = job_manager.create_job(
            domain=RESEARCH_WORKSPACE_OUTPUT_JOB_DOMAIN,
            queue=research_workspace_output_jobs_queue(),
            job_type=RESEARCH_WORKSPACE_OUTPUT_JOB_TYPE,
            owner_user_id=str(user_id),
            payload={
                "workspace_id": workspace_id,
                "artifact_id": artifact_id,
                "artifact_type": request.artifact_type,
                "source_ids": request.source_ids,
                "settings": request.settings.model_dump(exclude_none=True),
                "user_id": str(user_id),
            },
            max_retries=1,
        )
    except Exception as exc:
        try:
            workspace_db.delete_workspace_artifact(workspace_id, artifact_id)
        except Exception as cleanup_exc:
            raise ResearchWorkspaceOutputJobError(
                "output_job_enqueue_failed",
                status_code=503,
            ) from cleanup_exc
        raise ResearchWorkspaceOutputJobError(
            "output_job_enqueue_failed",
            status_code=503,
        ) from exc
    return ResearchWorkspaceOutputSubmitResponse(
        job_id=int(job["id"]),
        status=_public_job_status(job.get("status")),
        workspace_id=workspace_id,
        artifact_id=artifact_id,
        artifact_type=request.artifact_type,
    )


def get_research_workspace_output_job_status(
    *,
    workspace_id: str,
    job_id: int,
    workspace_db: CharactersRAGDB,
    job_manager: JobManager,
    user_id: int | str,
) -> ResearchWorkspaceOutputStatusResponse:
    try:
        job = job_manager.get_job(int(job_id))
    except Exception as exc:
        raise ResearchWorkspaceOutputJobError(
            "output_job_status_unavailable",
            status_code=503,
        ) from exc
    if job is None:
        raise ResearchWorkspaceOutputJobError("job_not_found", status_code=404)
    _validate_job_scope(job, workspace_id=workspace_id, user_id=str(user_id))

    payload = _normalize_job_mapping(job.get("payload"))
    result = _normalize_job_mapping(job.get("result"))
    artifact_id = str(payload.get("artifact_id") or result.get("artifact_id") or "").strip()
    artifact_type = str(payload.get("artifact_type") or result.get("artifact_type") or "").strip()
    if not artifact_id or artifact_type not in _OUTPUT_ARTIFACT_TYPES:
        raise ResearchWorkspaceOutputJobError("job_payload_invalid", status_code=500)

    artifact_row = workspace_db.get_workspace_artifact(workspace_id, artifact_id)
    artifact = WorkspaceArtifactResponse.model_validate(artifact_row) if artifact_row else None
    return ResearchWorkspaceOutputStatusResponse(
        job_id=int(job["id"]),
        status=_public_job_status(job.get("status")),
        progress_percent=job.get("progress_percent"),
        progress_message=job.get("progress_message"),
        workspace_id=workspace_id,
        artifact_id=artifact_id,
        artifact_type=cast(ResearchWorkspaceOutputArtifactType, artifact_type),
        artifact=artifact,
        error=_job_error(job),
        result=result,
    )


def _validate_workspace_sources(
    workspace_db: CharactersRAGDB,
    workspace_id: str,
    source_ids: list[str],
) -> None:
    existing_ids = {str(source.get("id") or "").strip() for source in workspace_db.list_workspace_sources(workspace_id)}
    if not existing_ids or any(source_id not in existing_ids for source_id in source_ids):
        raise ResearchWorkspaceOutputJobError("workspace_sources_not_found", status_code=404)


def _pending_artifact_payload(
    *,
    artifact_id: str,
    artifact_type: ResearchWorkspaceOutputArtifactType,
    source_ids: list[str],
    user_id: str,
    settings: ResearchWorkspaceOutputSettings,
) -> dict[str, Any]:
    title = settings.title_hint or artifact_type.replace("_", " ").title()
    content_type = "video/mp4" if artifact_type == "video_overview" else "image/png"
    return {
        "id": artifact_id,
        "artifact_type": artifact_type,
        "title": title,
        "status": "pending",
        "content": None,
        "content_type": content_type,
        "owner_scope": "user",
        "owner_id": user_id,
        "producer_metadata": {
            "origin": "research_workspace_output_job",
            "status": "queued",
            "settings": settings.model_dump(exclude_none=True),
        },
        "source_lineage": {"source_ids": source_ids},
        "version_metadata": {"schema_version": 1},
        "export_refs": [],
        "schema_version": 1,
    }


def _validate_job_scope(job: dict[str, Any], *, workspace_id: str, user_id: str) -> None:
    if job.get("job_type") != RESEARCH_WORKSPACE_OUTPUT_JOB_TYPE:
        raise ResearchWorkspaceOutputJobError("job_not_found", status_code=404)
    if job.get("domain") != RESEARCH_WORKSPACE_OUTPUT_JOB_DOMAIN:
        raise ResearchWorkspaceOutputJobError("job_not_found", status_code=404)

    payload = _normalize_job_mapping(job.get("payload"))
    owner_user_id = str(job.get("owner_user_id") or payload.get("owner_user_id") or payload.get("user_id") or "")
    if str(payload.get("workspace_id") or "").strip() != workspace_id or owner_user_id != user_id:
        raise ResearchWorkspaceOutputJobError("job_not_found", status_code=404)


def _public_job_status(value: Any) -> ResearchWorkspaceOutputStatus:
    raw = str(value or "").strip().lower()
    if raw in {"queued", "processing", "completed", "failed", "cancelled"}:
        return cast(ResearchWorkspaceOutputStatus, raw)
    if raw in {"running", "retrying"}:
        return "processing"
    return "queued"


def _job_error(job: dict[str, Any]) -> str | None:
    if _public_job_status(job.get("status")) != "failed":
        return None
    return _safe_public_error_code(str(job.get("error_code") or job.get("last_error") or job.get("error_message") or ""))


def _normalize_job_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}
