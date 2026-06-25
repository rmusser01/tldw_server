"""Worker entrypoint and handler for Audio Studio Jobs."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import os
from collections.abc import Callable
from pathlib import Path
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
from tldw_Server_API.app.core.Audio_Studio.export import (
    create_audio_studio_export_manifest,
    package_audio_studio_export,
    record_audio_studio_export_artifact,
    resolve_audio_studio_export_artifact_rows,
)
from tldw_Server_API.app.core.Audio_Studio.models import AudioGenerationRequest
from tldw_Server_API.app.core.Audio_Studio.migration import commit_audio_studio_audiobook_migration
from tldw_Server_API.app.core.Audio_Studio.providers.registry import (
    AudioStudioProviderRegistry,
    build_audio_studio_provider_registry,
)
from tldw_Server_API.app.core.Audio_Studio.render import (
    build_render_plan,
    record_audio_studio_render_artifact,
    render_audio_studio_mix,
)
from tldw_Server_API.app.core.Audio_Studio.security import redact_audio_studio_secret
from tldw_Server_API.app.core.DB_Management.Collections_DB import CollectionsDatabase
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
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
        collections_db = db_factory(owner_user_id)
        if hasattr(collections_db, "__enter__") and hasattr(collections_db, "__exit__"):
            with collections_db as scoped_db:
                return await handle_audio_studio_job(
                    job,
                    collections_db=scoped_db,
                    provider_registry=provider_registry_factory(),
                )
        return await handle_audio_studio_job(
            job,
            collections_db=collections_db,
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
        return await _handle_render_job(job, payload, collections_db=collections_db)
    if job_type == JOB_TYPE_EXPORT:
        return await _handle_export_job(job, payload, collections_db=collections_db)
    if job_type == JOB_TYPE_MIGRATE:
        return await _handle_migration_job(job, payload, collections_db=collections_db)
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
        response = {"status": "skipped", "reason": "stale_target_revision"}
        _update_generation_job_result(collections_db, project=project, job=job, status="skipped", result=response)
        return response
    try:
        collections_db.get_audio_studio_revision(target_revision_id)
    except KeyError:
        response = {"status": "skipped", "reason": "stale_target_revision"}
        _update_generation_job_result(collections_db, project=project, job=job, status="skipped", result=response)
        return response

    try:
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
    except Exception as exc:
        reason = _redact_audio_studio_error(str(exc) or type(exc).__name__)
        response = {
            "status": "failed",
            "reason": reason,
            "retryable": bool(getattr(exc, "retryable", True)),
        }
        _update_generation_job_result(collections_db, project=project, job=job, status="failed", result=response)
        raise
    artifact_id = f"art_{uuid4().hex[:16]}"
    content_hash = hashlib.sha256(result.content_bytes).hexdigest()
    job_id = str(job.get("uuid") or job.get("id") or "")
    storage_path = _write_generation_artifact_file(
        collections_db,
        project_id=project_id,
        job=job,
        artifact_id=artifact_id,
        mime_type=result.mime_type,
        content_bytes=result.content_bytes,
    )
    metadata = dict(result.metadata or {})
    metadata["job_id"] = job_id
    metadata["source"] = "audio_studio_generation"
    artifact = collections_db.create_audio_studio_artifact(
        project_row_id=getattr(project, "id"),
        artifact_id=artifact_id,
        artifact_type="generated_audio",
        provider=result.provider,
        output_id=None,
        storage_path=str(storage_path),
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
    _update_generation_job_result(collections_db, project=project, job=job, status="completed", result=response)
    return {
        **response,
        "artifact_row_id": getattr(artifact, "id", None),
    }


def _update_generation_job_result(
    collections_db: Any,
    *,
    project: Any,
    job: dict[str, Any],
    status: str,
    result: dict[str, Any],
) -> None:
    updater = getattr(collections_db, "update_audio_studio_generation_job", None)
    if callable(updater):
        updater(
            project_row_id=getattr(project, "id"),
            job_id=str(job.get("uuid") or job.get("id") or ""),
            status=status,
            result_json=json.dumps(result, sort_keys=True),
        )


async def _handle_render_job(
    job: dict[str, Any],
    payload: dict[str, Any],
    *,
    collections_db: Any,
) -> dict[str, Any]:
    project_id = _required_text(payload.get("project_id"), "project_id")
    target_revision_id = _required_text(payload.get("target_revision_id"), "target_revision_id")
    target_resource_id = _required_text(payload.get("target_resource_id"), "target_resource_id")
    project = collections_db.get_audio_studio_project_by_project_id(project_id)
    options = _provider_options(payload)
    artifact_refs = _artifact_refs(options)
    try:
        plan = build_render_plan(
            collections_db=collections_db,
            project=project,
            render_id=target_resource_id,
            target_revision_id=target_revision_id,
            artifact_refs=artifact_refs,
            output_format=str(options.get("output_format") or "wav"),
            loudness_normalize=bool(options.get("loudness_normalize")),
            render_type=str(options.get("render_type") or "preview_mix"),
        )
    except ValueError as exc:
        if str(exc) in {"stale_target_revision", "stale_artifact_revision"}:
            return {"status": "skipped", "reason": str(exc)}
        raise
    output_dir = _audio_studio_job_output_dir(collections_db, project_id=project_id, job=job, folder="renders")
    render_result = await render_audio_studio_mix(plan, output_dir=output_dir)
    recorded = record_audio_studio_render_artifact(
        collections_db=collections_db,
        project=project,
        plan=plan,
        render_result=render_result,
        artifact_id_prefix=str(job.get("uuid") or job.get("id") or target_resource_id),
    )
    return {
        "status": "completed",
        "render_id": plan.render_id,
        "mix_artifact_id": recorded.mix_artifact_id,
        "manifest_artifact_id": recorded.manifest_artifact_id,
        "mime_type": render_result.mime_type,
        "size_bytes": render_result.size_bytes,
        "content_hash": render_result.content_hash,
    }


async def _handle_export_job(
    job: dict[str, Any],
    payload: dict[str, Any],
    *,
    collections_db: Any,
) -> dict[str, Any]:
    project_id = _required_text(payload.get("project_id"), "project_id")
    target_revision_id = _required_text(payload.get("target_revision_id"), "target_revision_id")
    target_resource_id = _required_text(payload.get("target_resource_id"), "target_resource_id")
    project = collections_db.get_audio_studio_project_by_project_id(project_id)
    options = _provider_options(payload)
    artifact_refs = _artifact_refs(options)
    export_type = str(options.get("export_type") or "zip_package")
    source_render_id = options.get("source_render_id") if isinstance(options.get("source_render_id"), str) else None
    try:
        manifest = create_audio_studio_export_manifest(
            collections_db=collections_db,
            project=project,
            export_id=target_resource_id,
            export_type=export_type,
            target_revision_id=target_revision_id,
            artifact_refs=artifact_refs,
            source_render_id=source_render_id,
            settings=options.get("settings") if isinstance(options.get("settings"), dict) else {},
        )
        source_rows = resolve_audio_studio_export_artifact_rows(
            collections_db=collections_db,
            project=project,
            artifact_refs=artifact_refs,
            source_render_id=source_render_id,
        )
    except ValueError as exc:
        if str(exc) in {"stale_target_revision", "stale_artifact_revision"}:
            return {"status": "skipped", "reason": str(exc)}
        raise
    output_dir = _audio_studio_job_output_dir(collections_db, project_id=project_id, job=job, folder="exports")
    package_result = package_audio_studio_export(
        manifest=manifest,
        source_artifacts=source_rows,
        export_type=export_type,
        output_dir=output_dir,
        collections_db=collections_db,
    )
    recorded = record_audio_studio_export_artifact(
        collections_db=collections_db,
        project=project,
        manifest=manifest,
        package_result=package_result,
        artifact_id_prefix=str(job.get("uuid") or job.get("id") or target_resource_id),
    )
    return {
        "status": "completed",
        "export_id": target_resource_id,
        "package_artifact_id": recorded.package_artifact_id,
        "manifest_artifact_id": recorded.manifest_artifact_id,
        "mime_type": package_result.mime_type,
        "size_bytes": package_result.size_bytes,
        "content_hash": package_result.content_hash,
    }


async def _handle_migration_job(
    job: dict[str, Any],
    payload: dict[str, Any],
    *,
    collections_db: Any,
) -> dict[str, Any]:
    options = _provider_options(payload)
    project_payload = options.get("project_payload")
    if not isinstance(project_payload, dict):
        raise AudioStudioJobError("missing_project_payload", retryable=False)
    committed = commit_audio_studio_audiobook_migration(
        collections_db=collections_db,
        project_payload=project_payload,
        legacy_project_id=options.get("legacy_project_id") if isinstance(options.get("legacy_project_id"), str) else None,
        idempotency_key=options.get("idempotency_key") if isinstance(options.get("idempotency_key"), str) else None,
        user_id=job.get("owner_user_id"),
    )
    return {
        "status": "completed",
        "project_id": committed.project.project_id,
        "imported_section_count": committed.imported_section_count,
        "audio_reference_count": committed.audio_reference_count,
        "needs_regeneration_count": committed.needs_regeneration_count,
        "fingerprint": committed.fingerprint,
        "replayed": committed.replayed,
    }


def _provider_options(payload: dict[str, Any]) -> dict[str, Any]:
    options = payload.get("provider_options")
    return options if isinstance(options, dict) else {}


def _artifact_refs(options: dict[str, Any]) -> list[dict[str, Any] | str]:
    refs = options.get("artifact_refs")
    return refs if isinstance(refs, list) else []


def _write_generation_artifact_file(
    collections_db: Any,
    *,
    project_id: str,
    job: dict[str, Any],
    artifact_id: str,
    mime_type: str,
    content_bytes: bytes,
) -> Path:
    output_dir = _audio_studio_job_output_dir(collections_db, project_id=project_id, job=job, folder="generations")
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{_safe_path_component(artifact_id)}{_extension_for_mime(mime_type)}"
    path.write_bytes(content_bytes)
    return path


def _extension_for_mime(mime_type: str) -> str:
    normalized = str(mime_type or "").split(";", 1)[0].strip().lower()
    return {
        "audio/mpeg": ".mp3",
        "audio/mp3": ".mp3",
        "audio/wav": ".wav",
        "audio/x-wav": ".wav",
        "audio/flac": ".flac",
        "audio/ogg": ".ogg",
        "audio/opus": ".opus",
        "audio/mp4": ".m4a",
    }.get(normalized, ".audio")


def _audio_studio_job_output_dir(
    collections_db: Any,
    *,
    project_id: str,
    job: dict[str, Any],
    folder: str,
) -> Path:
    owner = str(getattr(collections_db, "user_id", "") or job.get("owner_user_id") or "0")
    user_id = int(owner) if owner.isdigit() else 0
    job_id = _safe_path_component(str(job.get("uuid") or job.get("id") or uuid4().hex))
    return DatabasePaths.get_user_outputs_dir(user_id) / "audio_studio" / _safe_path_component(project_id) / folder / job_id


def _safe_path_component(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(value or "").strip())
    return safe[:120].strip("_") or "audio_studio"


def _redact_audio_studio_error(message: str) -> str:
    secrets = [
        value
        for key, value in os.environ.items()
        if key.startswith("AUDIO_STUDIO_")
        and any(part in key for part in ("KEY", "SECRET", "TOKEN", "PASSWORD"))
    ]
    return redact_audio_studio_secret(message, secrets=secrets)


def _required_text(value: Any, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise AudioStudioJobError(f"missing_{field_name}", retryable=False)
    return text


if __name__ == "__main__":
    asyncio.run(run_audio_studio_jobs_worker())
