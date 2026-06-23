"""Jobs helpers for Audio Studio generation, rendering, export, and migration."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from tldw_Server_API.app.core.Audio_Studio.security import sanitize_audio_studio_payload

AUDIO_STUDIO_DOMAIN = "audio_studio"
AUDIO_STUDIO_QUEUE = "default"
JOB_TYPE_GENERATE = "audio_studio_generate"
JOB_TYPE_RENDER = "audio_studio_render"
JOB_TYPE_EXPORT = "audio_studio_export"
JOB_TYPE_MIGRATE = "audio_studio_migrate"

_REPLAYABLE_JOB_STATUSES = {
    "queued",
    "pending",
    "scheduled",
    "processing",
    "completed",
    "failed",
    "cancelled",
    "quarantined",
}


@dataclass(frozen=True)
class AudioStudioJobAccepted:
    """Accepted Audio Studio Jobs row projected for API responses."""

    job_id: str
    job_type: str
    project_id: str
    provider: str | None
    kind: str | None
    status: str


class AudioStudioTerminalJobError(ValueError):
    """Raised when Jobs idempotency returns a terminal row."""


def build_audio_studio_idempotency_key(
    *,
    user_id: str,
    project_id: str,
    job_type: str,
    target_resource_kind: str,
    target_resource_id: str,
    target_revision_id: str,
    caller_idempotency_key: str,
) -> str:
    """Build a stable Jobs idempotency key for Audio Studio work."""

    raw = "\x1f".join(
        [
            str(user_id),
            str(project_id),
            str(job_type),
            str(target_resource_kind),
            str(target_resource_id),
            str(target_revision_id),
            str(caller_idempotency_key),
        ]
    )
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return f"audio-studio:{digest}"


def enqueue_audio_studio_generation_job(
    *,
    jm: Any,
    collections_db: Any,
    user_id: str,
    project_id: str,
    workflow: str,
    kind: str,
    provider: str,
    target_resource_kind: str,
    target_resource_id: str,
    target_revision_id: str,
    idempotency_key: str,
    options: dict[str, Any],
    text: str | None = None,
    prompt: str | None = None,
    priority: int = 5,
) -> AudioStudioJobAccepted:
    """Create an idempotent Audio Studio generation Jobs row and domain index."""

    return _enqueue_audio_studio_job(
        jm=jm,
        collections_db=collections_db,
        user_id=user_id,
        project_id=project_id,
        workflow=workflow,
        job_type=JOB_TYPE_GENERATE,
        target_resource_kind=target_resource_kind,
        target_resource_id=target_resource_id,
        target_revision_id=target_revision_id,
        idempotency_key=idempotency_key,
        options=options,
        kind=kind,
        provider=provider,
        text=text,
        prompt=prompt,
        priority=priority,
    )


def enqueue_audio_studio_render_job(
    *,
    jm: Any,
    collections_db: Any,
    user_id: str,
    project_id: str,
    target_resource_kind: str,
    target_resource_id: str,
    target_revision_id: str,
    idempotency_key: str,
    options: dict[str, Any],
    priority: int = 5,
) -> AudioStudioJobAccepted:
    """Create an idempotent deferred Audio Studio render job."""

    return _enqueue_audio_studio_job(
        jm=jm,
        collections_db=collections_db,
        user_id=user_id,
        project_id=project_id,
        workflow="",
        job_type=JOB_TYPE_RENDER,
        target_resource_kind=target_resource_kind,
        target_resource_id=target_resource_id,
        target_revision_id=target_revision_id,
        idempotency_key=idempotency_key,
        options=options,
        priority=priority,
    )


def enqueue_audio_studio_export_job(
    *,
    jm: Any,
    collections_db: Any,
    user_id: str,
    project_id: str,
    target_resource_kind: str,
    target_resource_id: str,
    target_revision_id: str,
    idempotency_key: str,
    options: dict[str, Any],
    priority: int = 5,
) -> AudioStudioJobAccepted:
    """Create an idempotent deferred Audio Studio export job."""

    return _enqueue_audio_studio_job(
        jm=jm,
        collections_db=collections_db,
        user_id=user_id,
        project_id=project_id,
        workflow="",
        job_type=JOB_TYPE_EXPORT,
        target_resource_kind=target_resource_kind,
        target_resource_id=target_resource_id,
        target_revision_id=target_revision_id,
        idempotency_key=idempotency_key,
        options=options,
        priority=priority,
    )


def enqueue_audio_studio_migration_job(
    *,
    jm: Any,
    collections_db: Any,
    user_id: str,
    project_id: str,
    target_resource_kind: str,
    target_resource_id: str,
    target_revision_id: str,
    idempotency_key: str,
    options: dict[str, Any],
    priority: int = 5,
) -> AudioStudioJobAccepted:
    """Create an idempotent deferred Audio Studio migration job."""

    return _enqueue_audio_studio_job(
        jm=jm,
        collections_db=collections_db,
        user_id=user_id,
        project_id=project_id,
        workflow="",
        job_type=JOB_TYPE_MIGRATE,
        target_resource_kind=target_resource_kind,
        target_resource_id=target_resource_id,
        target_revision_id=target_revision_id,
        idempotency_key=idempotency_key,
        options=options,
        priority=priority,
    )


def _enqueue_audio_studio_job(
    *,
    jm: Any,
    collections_db: Any,
    user_id: str,
    project_id: str,
    workflow: str,
    job_type: str,
    target_resource_kind: str,
    target_resource_id: str,
    target_revision_id: str,
    idempotency_key: str,
    options: dict[str, Any],
    kind: str | None = None,
    provider: str | None = None,
    text: str | None = None,
    prompt: str | None = None,
    priority: int = 5,
) -> AudioStudioJobAccepted:
    project = collections_db.get_audio_studio_project_by_project_id(project_id)
    _validate_target_revision(collections_db, project, target_revision_id)
    sanitized_options = sanitize_audio_studio_payload(options or {})
    payload = {
        "project_id": project_id,
        "workflow": workflow or getattr(project, "workflow", ""),
        "kind": kind,
        "provider": provider,
        "text": text,
        "prompt": prompt,
        "target_resource_kind": target_resource_kind,
        "target_resource_id": target_resource_id,
        "target_revision_id": target_revision_id,
        "provider_options": sanitized_options,
    }
    job_idempotency_key = build_audio_studio_idempotency_key(
        user_id=user_id,
        project_id=project_id,
        job_type=job_type,
        target_resource_kind=target_resource_kind,
        target_resource_id=target_resource_id,
        target_revision_id=target_revision_id,
        caller_idempotency_key=idempotency_key,
    )
    job = jm.create_job(
        domain=AUDIO_STUDIO_DOMAIN,
        queue=AUDIO_STUDIO_QUEUE,
        job_type=job_type,
        payload=payload,
        owner_user_id=str(user_id),
        project_id=getattr(project, "id", None),
        priority=priority,
        max_retries=2,
        idempotency_key=job_idempotency_key,
    )
    status = str(job.get("status") or "queued").strip().lower()
    if status not in _REPLAYABLE_JOB_STATUSES:
        raise AudioStudioTerminalJobError(
            f"Audio Studio job returned unknown status {status or 'unknown'}"
        )
    job_id = str(job.get("uuid") or job.get("id"))
    if job_type == JOB_TYPE_GENERATE:
        _record_generation_index(
            collections_db=collections_db,
            project_row_id=getattr(project, "id"),
            job_id=job_id,
            provider=str(provider or ""),
            kind=str(kind or ""),
            target_resource_kind=target_resource_kind,
            target_resource_id=target_resource_id,
            target_revision_id=target_revision_id,
            idempotency_key=job_idempotency_key,
            status=status,
            request_json=json.dumps(payload, sort_keys=True),
        )
    return AudioStudioJobAccepted(
        job_id=job_id,
        job_type=job_type,
        project_id=project_id,
        provider=provider,
        kind=kind,
        status=status,
    )


def _record_generation_index(
    *,
    collections_db: Any,
    project_row_id: int,
    job_id: str,
    provider: str,
    kind: str,
    target_resource_kind: str,
    target_resource_id: str,
    target_revision_id: str,
    idempotency_key: str,
    status: str,
    request_json: str,
) -> None:
    getter = getattr(collections_db, "get_audio_studio_generation_job", None)
    if callable(getter):
        try:
            getter(project_row_id=project_row_id, job_id=job_id)
            return
        except KeyError:
            pass
    collections_db.record_audio_studio_generation_job(
        project_row_id=project_row_id,
        job_id=job_id,
        provider=provider,
        operation=f"{kind}.generate.v1",
        target_resource_kind=target_resource_kind,
        target_resource_id=target_resource_id,
        target_revision_id=target_revision_id,
        idempotency_key=idempotency_key,
        status=status,
        request_json=request_json,
        result_json=None,
    )


def _validate_target_revision(collections_db: Any, project: Any, target_revision_id: str) -> None:
    if str(getattr(project, "current_revision_id", "")) != str(target_revision_id):
        raise ValueError("stale_target_revision")
    try:
        collections_db.get_audio_studio_revision(target_revision_id)
    except KeyError as exc:
        raise ValueError("stale_target_revision") from exc
