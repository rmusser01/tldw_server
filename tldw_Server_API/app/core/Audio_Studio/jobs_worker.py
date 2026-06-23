"""Worker entrypoint and handler for Audio Studio Jobs."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import os
from collections.abc import Callable
from typing import Any
from uuid import uuid4

from loguru import logger

from tldw_Server_API.app.core.Audio_Studio.jobs import (
    AUDIO_STUDIO_DOMAIN,
    AUDIO_STUDIO_QUEUE,
    JOB_TYPE_EXPORT,
    JOB_TYPE_GENERATE,
    JOB_TYPE_MIGRATE,
    JOB_TYPE_RENDER,
)
from tldw_Server_API.app.core.Audio_Studio.models import AudioGenerationRequest
from tldw_Server_API.app.core.Audio_Studio.providers.registry import (
    AudioStudioProviderRegistry,
    build_audio_studio_provider_registry,
)
from tldw_Server_API.app.core.DB_Management.Collections_DB import CollectionsDatabase
from tldw_Server_API.app.core.Jobs.worker_sdk import WorkerConfig, WorkerSDK
from tldw_Server_API.app.core.Jobs.worker_utils import coerce_int, jobs_manager_from_env


class AudioStudioJobError(Exception):
    """Audio Studio worker error with Jobs retry metadata."""

    def __init__(self, message: str, *, retryable: bool = True):
        super().__init__(message)
        self.retryable = retryable


async def run_audio_studio_jobs_worker(stop_event: asyncio.Event | None = None) -> None:
    """Run the Audio Studio Jobs worker loop until stopped."""

    worker_id = os.getenv("AUDIO_STUDIO_JOBS_WORKER_ID") or f"audio-studio-jobs-{os.getpid()}"
    cfg = WorkerConfig(
        domain=AUDIO_STUDIO_DOMAIN,
        queue=AUDIO_STUDIO_QUEUE,
        worker_id=worker_id,
        lease_seconds=coerce_int(os.getenv("AUDIO_STUDIO_JOBS_LEASE_SECONDS") or os.getenv("JOBS_LEASE_SECONDS"), 60),
    )
    sdk = WorkerSDK(jobs_manager_from_env(), cfg)
    stop_task: asyncio.Task[None] | None = None
    if stop_event is not None:

        async def _watch_stop() -> None:
            await stop_event.wait()
            sdk.stop()

        stop_task = asyncio.create_task(_watch_stop())

    logger.info("Audio Studio Jobs worker starting: queue={} worker_id={}", AUDIO_STUDIO_QUEUE, worker_id)
    try:
        await sdk.run(handler=build_audio_studio_job_handler())
    finally:
        if stop_task is not None and not stop_task.done():
            stop_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await stop_task


def build_audio_studio_job_handler(
    *,
    collections_db_factory: Callable[[str], CollectionsDatabase] | None = None,
    provider_registry_factory: Callable[[], AudioStudioProviderRegistry] = build_audio_studio_provider_registry,
) -> Callable[[dict[str, Any]], Any]:
    """Build the concrete Audio Studio job handler used by the worker loop."""

    db_factory = collections_db_factory or (lambda owner_user_id: CollectionsDatabase.for_user(user_id=owner_user_id))

    async def _handler(job: dict[str, Any]) -> dict[str, Any]:
        owner_user_id = str(job.get("owner_user_id") or "").strip()
        if not owner_user_id:
            raise AudioStudioJobError("missing_owner_user_id", retryable=False)
        return await handle_audio_studio_job(
            job,
            collections_db=db_factory(owner_user_id),
            provider_registry=provider_registry_factory(),
        )

    return _handler


async def handle_audio_studio_job(
    job: dict[str, Any],
    *,
    collections_db: Any,
    provider_registry: AudioStudioProviderRegistry,
) -> dict[str, Any]:
    """Handle one Audio Studio job."""

    payload = job.get("payload") or {}
    if not isinstance(payload, dict):
        raise AudioStudioJobError("audio_studio_job_payload_invalid", retryable=False)
    job_type = str(job.get("job_type") or payload.get("job_type") or "").strip()
    if job_type == JOB_TYPE_GENERATE:
        return await _handle_generation_job(job, payload, collections_db=collections_db, provider_registry=provider_registry)
    if job_type == JOB_TYPE_RENDER:
        return {"status": "deferred", "reason": "audio_studio_render_not_implemented"}
    if job_type == JOB_TYPE_EXPORT:
        return {"status": "deferred", "reason": "audio_studio_export_not_implemented"}
    if job_type == JOB_TYPE_MIGRATE:
        return {"status": "deferred", "reason": "audio_studio_migrate_not_implemented"}
    raise AudioStudioJobError(f"unsupported_audio_studio_job_type:{job_type}", retryable=False)


async def _handle_generation_job(
    job: dict[str, Any],
    payload: dict[str, Any],
    *,
    collections_db: Any,
    provider_registry: AudioStudioProviderRegistry,
) -> dict[str, Any]:
    project_id = _required_text(payload.get("project_id"), "project_id")
    owner_user_id = _required_text(job.get("owner_user_id"), "owner_user_id")
    target_revision_id = _required_text(payload.get("target_revision_id"), "target_revision_id")
    project = collections_db.get_audio_studio_project_by_project_id(project_id)
    if str(getattr(project, "current_revision_id", "")) != target_revision_id:
        return {"status": "skipped", "reason": "stale_target_revision"}
    try:
        collections_db.get_audio_studio_revision(target_revision_id)
    except KeyError:
        return {"status": "skipped", "reason": "stale_target_revision"}

    kind = _required_text(payload.get("kind"), "kind")
    provider = _required_text(payload.get("provider"), "provider")
    adapter = provider_registry.get_adapter(provider, kind)
    request = AudioGenerationRequest(
        workflow=_required_text(payload.get("workflow") or getattr(project, "workflow", ""), "workflow"),
        kind=kind,
        prompt=payload.get("prompt") if payload.get("prompt") is None else str(payload.get("prompt")),
        text=payload.get("text") if payload.get("text") is None else str(payload.get("text")),
        provider_options=payload.get("provider_options") if isinstance(payload.get("provider_options"), dict) else {},
        target_resource_kind=_required_text(payload.get("target_resource_kind"), "target_resource_kind"),
        target_resource_id=_required_text(payload.get("target_resource_id"), "target_resource_id"),
        target_revision_id=target_revision_id,
    )
    result = await adapter.generate(request, user_id=int(owner_user_id) if owner_user_id.isdigit() else None)
    artifact_id = f"art_{uuid4().hex[:16]}"
    content_hash = hashlib.sha256(result.content_bytes).hexdigest()
    job_id = str(job.get("uuid") or job.get("id") or "")
    metadata = dict(result.metadata or {})
    metadata["job_id"] = job_id
    metadata["source"] = "audio_studio_generation"
    artifact = collections_db.create_audio_studio_artifact(
        project_row_id=getattr(project, "id"),
        artifact_id=artifact_id,
        artifact_type="generated_audio",
        provider=result.provider,
        output_id=None,
        storage_path=None,
        mime_type=result.mime_type,
        size_bytes=len(result.content_bytes),
        source_resource_kind=request.target_resource_kind,
        source_resource_id=request.target_resource_id,
        source_revision_id=request.target_revision_id,
        content_hash=content_hash,
        metadata_json=json.dumps(metadata, sort_keys=True),
    )
    response = {
        "status": "completed",
        "artifact_id": artifact_id,
        "mime_type": result.mime_type,
        "size_bytes": len(result.content_bytes),
    }
    updater = getattr(collections_db, "update_audio_studio_generation_job", None)
    if callable(updater):
        updater(
            project_row_id=getattr(project, "id"),
            job_id=job_id,
            status="completed",
            result_json=json.dumps(response, sort_keys=True),
        )
    return {
        **response,
        "artifact_row_id": getattr(artifact, "id", None),
    }


def _required_text(value: Any, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise AudioStudioJobError(f"missing_{field_name}", retryable=False)
    return text


if __name__ == "__main__":
    asyncio.run(run_audio_studio_jobs_worker())
