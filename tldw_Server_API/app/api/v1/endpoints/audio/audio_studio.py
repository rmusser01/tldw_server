"""Audio Studio API endpoint skeleton.

This module exposes server-backed project and timeline resource contracts for
Audio Studio without provider execution, rendering, exporting, or migration.
"""

from __future__ import annotations

import hashlib
import json
from typing import Annotated, Any
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Path, Query
from starlette import status

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import User, get_request_user
from tldw_Server_API.app.api.v1.API_Deps.Collections_DB_Deps import get_collections_db_for_user
from tldw_Server_API.app.api.v1.schemas.audio_studio_schemas import (
    AudioStudioArtifactListResponse,
    AudioStudioArtifactResponse,
    AudioStudioClipResponse,
    AudioStudioClipUpsert,
    AudioStudioProjectArchiveRequest,
    AudioStudioProjectCreate,
    AudioStudioProjectListResponse,
    AudioStudioProjectResponse,
    AudioStudioProjectUpdate,
    AudioStudioGenerationCreate,
    AudioStudioGenerationJobResponse,
    AudioStudioProviderListResponse,
    AudioStudioProviderResponse,
    AudioStudioSectionResponse,
    AudioStudioSectionUpsert,
    AudioStudioTrackResponse,
    AudioStudioTrackUpsert,
    AudioStudioWorkflow,
)
from tldw_Server_API.app.core.DB_Management.Collections_DB import (
    AudioStudioClipRow,
    AudioStudioArtifactRow,
    AudioStudioGenerationJobRow,
    AudioStudioProjectRow,
    AudioStudioSectionRow,
    AudioStudioTrackRow,
    CollectionsDatabase,
)
from tldw_Server_API.app.core.DB_Management.backends.base import BackendType, DatabaseError
from tldw_Server_API.app.core.Audio_Studio.jobs import enqueue_audio_studio_generation_job
from tldw_Server_API.app.core.Audio_Studio.providers.registry import build_audio_studio_provider_registry
from tldw_Server_API.app.core.Jobs.manager import JobManager


router = APIRouter(prefix="/audio-studio", tags=["audio-studio"])
_AUDIO_STUDIO_ID_PATTERN = r"^[A-Za-z0-9_-]+$"
AudioStudioIdPath = Annotated[str, Path(min_length=1, max_length=120, pattern=_AUDIO_STUDIO_ID_PATTERN)]


def _new_project_id() -> str:
    return f"ast_{uuid4().hex[:16]}"


def _new_revision_id() -> str:
    return f"rev_{uuid4().hex[:16]}"


def _json_dumps(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _content_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(_json_dumps(payload).encode("utf-8")).hexdigest()


def _settings_json(
    *,
    settings: dict[str, Any] | None,
    metadata: dict[str, Any] | None,
    description: str | None = None,
) -> str:
    return _json_dumps(
        {
            "settings": settings or {},
            "metadata": metadata or {},
            "description": description,
        }
    )


def _settings_payload(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {"settings": {}, "metadata": {}, "description": None}
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        return {"settings": {}, "metadata": {}, "description": None}
    if not isinstance(parsed, dict):
        return {"settings": {}, "metadata": {}, "description": None}
    if "settings" in parsed or "metadata" in parsed or "description" in parsed:
        return {
            "settings": parsed.get("settings") if isinstance(parsed.get("settings"), dict) else {},
            "metadata": parsed.get("metadata") if isinstance(parsed.get("metadata"), dict) else {},
            "description": parsed.get("description") if isinstance(parsed.get("description"), str) else None,
        }
    return {"settings": parsed, "metadata": {}, "description": None}


def _project_response(row: AudioStudioProjectRow) -> AudioStudioProjectResponse:
    payload = _settings_payload(row.settings_json)
    return AudioStudioProjectResponse(
        project_id=row.project_id,
        title=row.title,
        description=payload["description"],
        workflow=row.workflow,
        status=row.status,
        settings=payload["settings"],
        metadata=payload["metadata"],
        current_revision_id=row.current_revision_id,
        created_at=row.created_at,
        updated_at=row.updated_at,
        archived_at=row.archived_at,
    )


def _section_response(row: AudioStudioSectionRow) -> AudioStudioSectionResponse:
    return AudioStudioSectionResponse(
        section_id=row.section_id,
        workflow=row.workflow,
        title=row.title,
        body_text=row.body_text,
        speaker_id=row.speaker_id,
        order_index=row.order_index,
        settings=_settings_payload(row.settings_json)["settings"],
        current_revision_id=row.current_revision_id,
        archived_at=row.archived_at,
    )


def _track_response(row: AudioStudioTrackRow) -> AudioStudioTrackResponse:
    return AudioStudioTrackResponse(
        track_id=row.track_id,
        name=row.name,
        kind=row.kind,
        order_index=row.order_index,
        muted=bool(row.muted),
        solo=bool(row.solo),
        volume=float(row.volume),
        settings=_settings_payload(row.settings_json)["settings"],
        current_revision_id=row.current_revision_id,
        archived_at=row.archived_at,
    )


def _clip_response(row: AudioStudioClipRow) -> AudioStudioClipResponse:
    return AudioStudioClipResponse(
        clip_id=row.clip_id,
        section_id=row.section_id,
        track_id=row.track_id,
        title=row.title,
        clip_type=row.clip_type,
        start_ms=row.start_ms,
        duration_ms=row.duration_ms,
        volume=float(row.volume),
        fade_in_ms=row.fade_in_ms,
        fade_out_ms=row.fade_out_ms,
        muted=bool(row.muted),
        artifact_id=row.artifact_id,
        settings=_settings_payload(row.settings_json)["settings"],
        current_revision_id=row.current_revision_id,
        archived_at=row.archived_at,
    )


def _generation_job_response(
    row: AudioStudioGenerationJobRow,
    *,
    project_id: str,
) -> AudioStudioGenerationJobResponse:
    request_payload = _parse_json_object(row.request_json)
    result_payload = _parse_json_object(row.result_json)
    return AudioStudioGenerationJobResponse(
        job_id=row.job_id,
        project_id=project_id,
        provider=row.provider,
        kind=str(request_payload.get("kind") or row.operation.split(".", 1)[0]),
        status=row.status,
        target_resource_kind=row.target_resource_kind,
        target_resource_id=row.target_resource_id,
        target_revision_id=row.target_revision_id,
        result=result_payload or None,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def _artifact_response(row: AudioStudioArtifactRow) -> AudioStudioArtifactResponse:
    return AudioStudioArtifactResponse(
        artifact_id=row.artifact_id,
        artifact_type=row.artifact_type,
        provider=row.provider,
        mime_type=row.mime_type,
        size_bytes=row.size_bytes,
        source_resource_kind=row.source_resource_kind,
        source_resource_id=row.source_resource_id,
        source_revision_id=row.source_revision_id,
        metadata=_parse_json_object(row.metadata_json),
        created_at=row.created_at,
    )


def _parse_json_object(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _load_project_or_404(collections_db: CollectionsDatabase, project_id: str) -> AudioStudioProjectRow:
    try:
        return collections_db.get_audio_studio_project_by_project_id(project_id)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="audio_studio_project_not_found") from exc


def _raise_conflict_for_stale_base(exc: ValueError) -> None:
    if str(exc) == "stale_base_revision":
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="stale_base_revision") from exc
    if str(exc) in {
        "audio_studio_track_not_found",
        "audio_studio_section_not_found",
        "audio_studio_artifact_not_found",
    }:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="invalid_audio_studio_request") from exc


def _section_text_for_generation(
    collections_db: CollectionsDatabase,
    *,
    project_row_id: int,
    target_resource_kind: str,
    target_resource_id: str,
) -> str | None:
    if target_resource_kind != "section":
        return None
    row = collections_db.backend.execute(
        "SELECT body_text FROM audio_studio_sections "
        "WHERE project_row_id = ? AND section_id = ? AND deleted = ?",
        (
            project_row_id,
            target_resource_id,
            collections_db._coerce_bool_flag(  # noqa: SLF001
                False,
                postgres=collections_db.backend.backend_type == BackendType.POSTGRESQL,
            ),
        ),
    ).first
    if not row:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="audio_studio_section_not_found")
    return row.get("body_text")


@router.get("/workflows")
async def list_audio_studio_workflows(current_user: User = Depends(get_request_user)) -> dict[str, list[dict[str, str]]]:
    """Return first-class Audio Studio workflows."""
    _ = current_user
    return {
        "workflows": [
            {"id": AudioStudioWorkflow.NARRATION.value, "label": "Narration"},
            {"id": AudioStudioWorkflow.PODCAST.value, "label": "Podcast"},
            {"id": AudioStudioWorkflow.BRIEFING.value, "label": "Briefing"},
            {"id": AudioStudioWorkflow.MUSIC.value, "label": "Music"},
        ]
    }


@router.get("/providers", response_model=AudioStudioProviderListResponse)
async def list_audio_studio_providers(
    current_user: User = Depends(get_request_user),
) -> AudioStudioProviderListResponse:
    """Return configured, secret-free Audio Studio generation providers."""
    _ = current_user
    registry = build_audio_studio_provider_registry()
    return AudioStudioProviderListResponse(
        providers=[AudioStudioProviderResponse(**row) for row in registry.list_providers()]
    )


@router.post("/projects", response_model=AudioStudioProjectResponse)
async def create_audio_studio_project(
    request: AudioStudioProjectCreate,
    current_user: User = Depends(get_request_user),
    collections_db: CollectionsDatabase = Depends(get_collections_db_for_user),
) -> AudioStudioProjectResponse:
    """Create an Audio Studio project and initial revision."""
    _ = current_user
    project_id = _new_project_id()
    revision_id = _new_revision_id()
    settings_json = _settings_json(
        settings=request.settings,
        metadata=request.metadata,
        description=request.description,
    )
    payload = request.model_dump(mode="json")
    try:
        project = collections_db.create_audio_studio_project(
            project_id=project_id,
            title=request.title,
            workflow=request.workflow.value,
            revision_id=revision_id,
            mutation_kind="project.create",
            resource_kind="project",
            resource_id=project_id,
            content_hash=_content_hash(payload),
            payload_json=_json_dumps(payload),
            settings_json=settings_json,
        )
        return _project_response(project)
    except DatabaseError as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="audio_studio_project_create_failed") from exc


@router.get("/projects", response_model=AudioStudioProjectListResponse)
async def list_audio_studio_projects(
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    current_user: User = Depends(get_request_user),
    collections_db: CollectionsDatabase = Depends(get_collections_db_for_user),
) -> AudioStudioProjectListResponse:
    """List current user's non-archived Audio Studio projects."""
    _ = current_user
    rows = collections_db.list_audio_studio_projects(limit=limit, offset=offset)
    return AudioStudioProjectListResponse(
        projects=[_project_response(row) for row in rows],
        limit=limit,
        offset=offset,
    )


@router.get("/projects/{project_id}", response_model=AudioStudioProjectResponse)
async def get_audio_studio_project(
    project_id: AudioStudioIdPath,
    current_user: User = Depends(get_request_user),
    collections_db: CollectionsDatabase = Depends(get_collections_db_for_user),
) -> AudioStudioProjectResponse:
    """Fetch a project owned by the current user."""
    _ = current_user
    return _project_response(_load_project_or_404(collections_db, project_id))


@router.patch("/projects/{project_id}", response_model=AudioStudioProjectResponse)
async def update_audio_studio_project(
    request: AudioStudioProjectUpdate,
    project_id: AudioStudioIdPath,
    current_user: User = Depends(get_request_user),
    collections_db: CollectionsDatabase = Depends(get_collections_db_for_user),
) -> AudioStudioProjectResponse:
    """Update project metadata with optimistic concurrency."""
    _ = current_user
    project = _load_project_or_404(collections_db, project_id)
    payload = _settings_payload(project.settings_json)
    settings = request.settings if request.settings is not None else payload["settings"]
    metadata = request.metadata if request.metadata is not None else payload["metadata"]
    description = request.description if "description" in request.model_fields_set else payload["description"]
    revision_id = _new_revision_id()
    revision_payload = request.model_dump(mode="json", exclude_unset=True)
    try:
        updated = collections_db.mutate_audio_studio_project(
            project_row_id=project.id,
            base_revision_id=request.base_revision_id,
            revision_id=revision_id,
            mutation_kind="project.update",
            resource_kind="project",
            resource_id=project.project_id,
            content_hash=_content_hash(revision_payload),
            payload_json=_json_dumps(revision_payload),
            title=request.title,
            status=request.status.value if request.status is not None else None,
            settings_json=_settings_json(settings=settings, metadata=metadata, description=description),
        )
        return _project_response(updated)
    except ValueError as exc:
        _raise_conflict_for_stale_base(exc)
    except DatabaseError as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="audio_studio_project_update_failed") from exc


@router.delete("/projects/{project_id}")
async def archive_audio_studio_project(
    request: AudioStudioProjectArchiveRequest,
    project_id: AudioStudioIdPath,
    current_user: User = Depends(get_request_user),
    collections_db: CollectionsDatabase = Depends(get_collections_db_for_user),
) -> dict[str, str | bool]:
    """Archive a project owned by the current user."""
    _ = current_user
    project = _load_project_or_404(collections_db, project_id)
    revision_id = _new_revision_id()
    payload = request.model_dump(mode="json")
    try:
        archived = collections_db.archive_audio_studio_project(
            project_row_id=project.id,
            base_revision_id=request.base_revision_id,
            revision_id=revision_id,
            content_hash=_content_hash(payload),
            payload_json=_json_dumps(payload),
        )
        return {"project_id": archived.project_id, "archived": True, "current_revision_id": archived.current_revision_id}
    except ValueError as exc:
        _raise_conflict_for_stale_base(exc)
    except DatabaseError as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="audio_studio_project_archive_failed") from exc


@router.put("/projects/{project_id}/sections/{section_id}", response_model=AudioStudioSectionResponse)
async def upsert_audio_studio_section(
    request: AudioStudioSectionUpsert,
    project_id: AudioStudioIdPath,
    section_id: AudioStudioIdPath,
    current_user: User = Depends(get_request_user),
    collections_db: CollectionsDatabase = Depends(get_collections_db_for_user),
) -> AudioStudioSectionResponse:
    """Create or update a section within the current user's project."""
    _ = current_user
    project = _load_project_or_404(collections_db, project_id)
    revision_id = _new_revision_id()
    payload = request.model_dump(mode="json")
    try:
        row = collections_db.upsert_audio_studio_section(
            project_row_id=project.id,
            section_id=section_id,
            base_revision_id=request.base_revision_id,
            revision_id=revision_id,
            workflow=project.workflow,
            title=request.title,
            body_text=request.body_text,
            speaker_id=request.speaker_id,
            order_index=request.order_index,
            settings_json=_settings_json(settings=request.settings, metadata=request.metadata),
            content_hash=_content_hash(payload),
            payload_json=_json_dumps(payload),
        )
        return _section_response(row)
    except ValueError as exc:
        _raise_conflict_for_stale_base(exc)
    except DatabaseError as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="audio_studio_section_upsert_failed") from exc


@router.put("/projects/{project_id}/tracks/{track_id}", response_model=AudioStudioTrackResponse)
async def upsert_audio_studio_track(
    request: AudioStudioTrackUpsert,
    project_id: AudioStudioIdPath,
    track_id: AudioStudioIdPath,
    current_user: User = Depends(get_request_user),
    collections_db: CollectionsDatabase = Depends(get_collections_db_for_user),
) -> AudioStudioTrackResponse:
    """Create or update a timeline track within the current user's project."""
    _ = current_user
    project = _load_project_or_404(collections_db, project_id)
    revision_id = _new_revision_id()
    payload = request.model_dump(mode="json")
    try:
        row = collections_db.upsert_audio_studio_track(
            project_row_id=project.id,
            track_id=track_id,
            base_revision_id=request.base_revision_id,
            revision_id=revision_id,
            name=request.name,
            kind=request.kind.value,
            order_index=request.order_index,
            muted=request.muted,
            solo=request.solo,
            volume=request.volume,
            settings_json=_settings_json(settings=request.settings, metadata=request.metadata),
            content_hash=_content_hash(payload),
            payload_json=_json_dumps(payload),
        )
        return _track_response(row)
    except ValueError as exc:
        _raise_conflict_for_stale_base(exc)
    except DatabaseError as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="audio_studio_track_upsert_failed") from exc


@router.put("/projects/{project_id}/clips/{clip_id}", response_model=AudioStudioClipResponse)
async def upsert_audio_studio_clip(
    request: AudioStudioClipUpsert,
    project_id: AudioStudioIdPath,
    clip_id: AudioStudioIdPath,
    current_user: User = Depends(get_request_user),
    collections_db: CollectionsDatabase = Depends(get_collections_db_for_user),
) -> AudioStudioClipResponse:
    """Create or update a timeline clip within the current user's project."""
    _ = current_user
    project = _load_project_or_404(collections_db, project_id)
    revision_id = _new_revision_id()
    payload = request.model_dump(mode="json")
    try:
        row = collections_db.upsert_audio_studio_clip(
            project_row_id=project.id,
            clip_id=clip_id,
            base_revision_id=request.base_revision_id,
            revision_id=revision_id,
            section_id=request.section_id,
            track_id=request.track_id,
            title=request.title,
            clip_type=request.clip_type.value,
            start_ms=request.start_ms,
            duration_ms=request.duration_ms,
            volume=request.volume,
            fade_in_ms=request.fade_in_ms,
            fade_out_ms=request.fade_out_ms,
            muted=request.muted,
            artifact_id=request.artifact_id,
            settings_json=_settings_json(settings=request.settings, metadata=request.metadata),
            content_hash=_content_hash(payload),
            payload_json=_json_dumps(payload),
        )
        return _clip_response(row)
    except ValueError as exc:
        _raise_conflict_for_stale_base(exc)
    except DatabaseError as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="audio_studio_clip_upsert_failed") from exc


@router.post(
    "/projects/{project_id}/generations",
    response_model=AudioStudioGenerationJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def create_audio_studio_generation(
    request: AudioStudioGenerationCreate,
    project_id: AudioStudioIdPath,
    current_user: User = Depends(get_request_user),
    collections_db: CollectionsDatabase = Depends(get_collections_db_for_user),
) -> AudioStudioGenerationJobResponse:
    """Create an idempotent Audio Studio generation job."""

    project = _load_project_or_404(collections_db, project_id)
    provider = request.provider
    if not isinstance(provider, str):
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="provider must be a string")
    text = _section_text_for_generation(
        collections_db,
        project_row_id=project.id,
        target_resource_kind=request.target_resource_kind.value,
        target_resource_id=request.target_resource_id,
    )
    prompt = request.options.get("prompt") if isinstance(request.options.get("prompt"), str) else None
    try:
        accepted = enqueue_audio_studio_generation_job(
            jm=JobManager(),
            collections_db=collections_db,
            user_id=str(current_user.id),
            project_id=project.project_id,
            workflow=project.workflow,
            kind=request.kind,
            provider=provider,
            target_resource_kind=request.target_resource_kind.value,
            target_resource_id=request.target_resource_id,
            target_revision_id=request.target_revision_id,
            idempotency_key=request.idempotency_key,
            options=request.options,
            text=text,
            prompt=prompt,
        )
        row = collections_db.get_audio_studio_generation_job(
            project_row_id=project.id,
            job_id=accepted.job_id,
        )
        return _generation_job_response(row, project_id=project.project_id)
    except ValueError as exc:
        if str(exc) == "stale_target_revision":
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="stale_target_revision") from exc
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except DatabaseError as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="audio_studio_generation_create_failed") from exc


@router.get(
    "/projects/{project_id}/generations/{job_id}",
    response_model=AudioStudioGenerationJobResponse,
)
async def get_audio_studio_generation(
    project_id: AudioStudioIdPath,
    job_id: AudioStudioIdPath,
    current_user: User = Depends(get_request_user),
    collections_db: CollectionsDatabase = Depends(get_collections_db_for_user),
) -> AudioStudioGenerationJobResponse:
    """Fetch an Audio Studio generation job owned by the current user."""

    _ = current_user
    project = _load_project_or_404(collections_db, project_id)
    try:
        row = collections_db.get_audio_studio_generation_job(project_row_id=project.id, job_id=job_id)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="audio_studio_generation_job_not_found") from exc
    return _generation_job_response(row, project_id=project.project_id)


@router.get("/projects/{project_id}/artifacts", response_model=AudioStudioArtifactListResponse)
async def list_audio_studio_artifacts(
    project_id: AudioStudioIdPath,
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    current_user: User = Depends(get_request_user),
    collections_db: CollectionsDatabase = Depends(get_collections_db_for_user),
) -> AudioStudioArtifactListResponse:
    """List Audio Studio artifacts for a project owned by the current user."""

    _ = current_user
    project = _load_project_or_404(collections_db, project_id)
    rows = collections_db.list_audio_studio_artifacts(project_row_id=project.id, limit=limit, offset=offset)
    return AudioStudioArtifactListResponse(
        artifacts=[_artifact_response(row) for row in rows],
        limit=limit,
        offset=offset,
    )
