"""Audio Studio API endpoints.

This module exposes server-backed project and timeline resource contracts for
Audio Studio provider execution, rendering, exporting, and migration.
"""

from __future__ import annotations

import hashlib
import json
import mimetypes
import re
from collections.abc import Iterator
from datetime import datetime, timedelta
from pathlib import Path as FileSystemPath
from typing import Annotated, Any
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Path, Query, Request, Response
from fastapi.responses import FileResponse, StreamingResponse
from starlette import status

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import User, get_request_user
from tldw_Server_API.app.api.v1.API_Deps.Collections_DB_Deps import get_collections_db_for_user
from tldw_Server_API.app.api.v1.schemas.audio_studio_schemas import (
    AudioStudioArtifactListResponse,
    AudioStudioArtifactResponse,
    AudioStudioClipResponse,
    AudioStudioClipUpsert,
    AudioStudioExportCreate,
    AudioStudioExportJobResponse,
    AudioStudioGenerationCreate,
    AudioStudioGenerationJobResponse,
    AudioStudioMediaTicketCreate,
    AudioStudioMediaTicketPurpose,
    AudioStudioMediaTicketResponse,
    AudioStudioMigrationCommit,
    AudioStudioMigrationCommitResponse,
    AudioStudioMigrationPreview,
    AudioStudioMigrationPreviewResponse,
    AudioStudioProjectArchiveRequest,
    AudioStudioProjectCreate,
    AudioStudioProjectListResponse,
    AudioStudioProjectResponse,
    AudioStudioProjectUpdate,
    AudioStudioProviderListResponse,
    AudioStudioProviderResponse,
    AudioStudioRenderCreate,
    AudioStudioRenderJobResponse,
    AudioStudioSectionResponse,
    AudioStudioSectionUpsert,
    AudioStudioTrackResponse,
    AudioStudioTrackUpsert,
    AudioStudioWorkflow,
)
from tldw_Server_API.app.core.Audio_Studio.export import create_audio_studio_export_manifest
from tldw_Server_API.app.core.Audio_Studio.jobs import (
    AUDIO_STUDIO_DOMAIN,
    JOB_TYPE_EXPORT,
    JOB_TYPE_RENDER,
    enqueue_audio_studio_export_job,
    enqueue_audio_studio_generation_job,
    enqueue_audio_studio_render_job,
)
from tldw_Server_API.app.core.Audio_Studio.media_tickets import (
    AudioStudioMediaTicketRow,
    AudioStudioMediaTicketStore,
    hash_media_ticket_token,
    utc_now,
)
from tldw_Server_API.app.core.Audio_Studio.migration import (
    commit_audio_studio_audiobook_migration,
    preview_audio_studio_audiobook_migration,
)
from tldw_Server_API.app.core.Audio_Studio.providers.registry import build_audio_studio_provider_registry
from tldw_Server_API.app.core.Audio_Studio.render import build_render_plan
from tldw_Server_API.app.core.AuthNZ.db_config import AuthDatabaseConfig
from tldw_Server_API.app.core.DB_Management.backends.base import BackendType, DatabaseError
from tldw_Server_API.app.core.DB_Management.Collections_DB import (
    AudioStudioArtifactRow,
    AudioStudioClipRow,
    AudioStudioGenerationJobRow,
    AudioStudioProjectRow,
    AudioStudioSectionRow,
    AudioStudioTrackRow,
    CollectionsDatabase,
)
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.exceptions import (
    InvalidStoragePathError,
    InvalidStorageUserIdError,
    StorageUnavailableError,
)
from tldw_Server_API.app.core.Jobs.manager import JobManager

router = APIRouter(prefix="/audio-studio", tags=["audio-studio"])
_AUDIO_STUDIO_ID_PATTERN = r"^[A-Za-z0-9_-]+$"
AudioStudioIdPath = Annotated[str, Path(min_length=1, max_length=120, pattern=_AUDIO_STUDIO_ID_PATTERN)]
_PLAYABLE_AUDIO_MIME_TYPES = {
    "audio/aac",
    "audio/flac",
    "audio/m4a",
    "audio/mp4",
    "audio/mpeg",
    "audio/ogg",
    "audio/wav",
    "audio/wave",
    "audio/webm",
    "audio/x-m4a",
    "audio/x-wav",
}
_AUDIO_MIME_SUFFIXES = {
    "audio/aac": {".aac"},
    "audio/flac": {".flac"},
    "audio/m4a": {".m4a"},
    "audio/mp4": {".m4a", ".mp4"},
    "audio/mpeg": {".mp3"},
    "audio/ogg": {".ogg"},
    "audio/wav": {".wav"},
    "audio/wave": {".wav"},
    "audio/webm": {".webm"},
    "audio/x-m4a": {".m4a"},
    "audio/x-wav": {".wav"},
}
_AUDIO_ALLOWED_SUFFIXES = {suffix for suffixes in _AUDIO_MIME_SUFFIXES.values() for suffix in suffixes}
_HASH_HEX_CHARS = set("0123456789abcdefABCDEF")
_MEDIA_TICKET_TOKEN_PATTERN = re.compile(r"^[A-Za-z0-9_-]{32,256}$")
_PLAYBACK_TICKET_TTL = timedelta(minutes=30)
_DOWNLOAD_TICKET_TTL = timedelta(minutes=10)
_MEDIA_TICKET_HASH_CACHE_MAX_ENTRIES = 4096
_MediaTicketHashCacheKey = tuple[str, str]
_MediaTicketHashCacheIdentity = tuple[str, int | None, int | None, int, int, int]
_MEDIA_TICKET_HASH_VERIFICATION_CACHE: dict[_MediaTicketHashCacheKey, _MediaTicketHashCacheIdentity] = {}


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


def _render_job_response(
    *,
    accepted_or_job: Any,
    project_id: str,
    render_id: str,
    render_type: str,
    target_resource_kind: str,
    target_resource_id: str,
    target_revision_id: str,
    manifest: dict[str, Any],
) -> AudioStudioRenderJobResponse:
    if isinstance(accepted_or_job, dict):
        status_value = str(accepted_or_job.get("status") or "queued")
        result = accepted_or_job.get("result") if isinstance(accepted_or_job.get("result"), dict) else None
        created_at = _string_or_none(accepted_or_job.get("created_at"))
        updated_at = _string_or_none(accepted_or_job.get("updated_at"))
        job_id = str(accepted_or_job.get("uuid") or accepted_or_job.get("id") or "")
    else:
        status_value = str(getattr(accepted_or_job, "status", "queued"))
        result = None
        created_at = None
        updated_at = None
        job_id = str(getattr(accepted_or_job, "job_id", ""))
    return AudioStudioRenderJobResponse(
        job_id=job_id,
        project_id=project_id,
        job_type=JOB_TYPE_RENDER,
        render_id=render_id,
        render_type=render_type,
        status=status_value,
        target_resource_kind=target_resource_kind,
        target_resource_id=target_resource_id,
        target_revision_id=target_revision_id,
        manifest=manifest,
        result=result,
        created_at=created_at,
        updated_at=updated_at,
    )


def _export_job_response(
    *,
    accepted_or_job: Any,
    project_id: str,
    export_id: str,
    export_type: str,
    target_resource_kind: str,
    target_resource_id: str,
    target_revision_id: str,
    source_render_id: str | None,
    manifest: dict[str, Any],
) -> AudioStudioExportJobResponse:
    if isinstance(accepted_or_job, dict):
        status_value = str(accepted_or_job.get("status") or "queued")
        result = accepted_or_job.get("result") if isinstance(accepted_or_job.get("result"), dict) else None
        created_at = _string_or_none(accepted_or_job.get("created_at"))
        updated_at = _string_or_none(accepted_or_job.get("updated_at"))
        job_id = str(accepted_or_job.get("uuid") or accepted_or_job.get("id") or "")
    else:
        status_value = str(getattr(accepted_or_job, "status", "queued"))
        result = None
        created_at = None
        updated_at = None
        job_id = str(getattr(accepted_or_job, "job_id", ""))
    return AudioStudioExportJobResponse(
        job_id=job_id,
        project_id=project_id,
        job_type=JOB_TYPE_EXPORT,
        export_id=export_id,
        export_type=export_type,
        status=status_value,
        target_resource_kind=target_resource_kind,
        target_resource_id=target_resource_id,
        target_revision_id=target_revision_id,
        source_render_id=source_render_id,
        manifest=manifest,
        result=result,
        created_at=created_at,
        updated_at=updated_at,
    )


def _job_payload(job: dict[str, Any]) -> dict[str, Any]:
    payload = job.get("payload")
    return payload if isinstance(payload, dict) else {}


def _provider_options(payload: dict[str, Any]) -> dict[str, Any]:
    options = payload.get("provider_options")
    return options if isinstance(options, dict) else {}


def _load_audio_studio_job_or_404(
    *,
    job_id: str,
    project: AudioStudioProjectRow,
    current_user: User,
    job_type: str,
) -> dict[str, Any]:
    job = JobManager().get_job_by_uuid(job_id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="audio_studio_job_not_found")
    payload = _job_payload(job)
    if (
        job.get("domain") != AUDIO_STUDIO_DOMAIN
        or job.get("job_type") != job_type
        or str(job.get("owner_user_id") or "") != str(current_user.id)
        or str(payload.get("project_id") or "") != project.project_id
    ):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="audio_studio_job_not_found")
    return job


def _artifact_refs_from_request(request: AudioStudioRenderCreate | AudioStudioExportCreate) -> list[dict[str, Any] | str]:
    if isinstance(request.options.get("artifact_refs"), list):
        return request.options["artifact_refs"]
    if isinstance(request.settings.get("artifact_refs"), list):
        return request.settings["artifact_refs"]
    return []


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


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


def _load_audio_studio_artifact_or_404(
    collections_db: CollectionsDatabase,
    project: AudioStudioProjectRow,
    artifact_id: str,
) -> AudioStudioArtifactRow:
    rows = collections_db.list_audio_studio_artifacts(
        project_row_id=project.id,
        artifact_id=artifact_id,
        limit=1,
    )
    if not rows:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="audio_studio_artifact_not_found")
    return rows[0]


_DANGEROUS_DOWNLOAD_SUFFIXES = {
    ".app",
    ".bat",
    ".cmd",
    ".com",
    ".cpl",
    ".dll",
    ".dmg",
    ".exe",
    ".hta",
    ".htm",
    ".html",
    ".jar",
    ".js",
    ".jse",
    ".mjs",
    ".msi",
    ".ps1",
    ".scr",
    ".sh",
    ".svg",
    ".vb",
    ".vbe",
    ".vbs",
    ".ws",
    ".wsf",
}

_DANGEROUS_DOWNLOAD_MIME_TYPES = {
    "application/javascript",
    "application/x-msdownload",
    "application/x-sh",
    "application/x-shellscript",
    "image/svg+xml",
    "text/html",
    "text/javascript",
    "text/x-shellscript",
}


def _audio_studio_artifact_roots(user_id: int | str) -> list[FileSystemPath]:
    return [
        DatabasePaths.get_user_base_directory(user_id) / "outputs",
        DatabasePaths.get_user_temp_outputs_dir(user_id),
    ]


def _resolve_contained_artifact_file(
    candidate: FileSystemPath,
    roots: list[FileSystemPath],
) -> FileSystemPath:
    try:
        resolved_file = candidate.resolve(strict=True)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="audio_studio_artifact_file_not_found",
        ) from exc
    except OSError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="invalid_audio_studio_artifact_path",
        ) from exc
    if not resolved_file.is_file():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="invalid_audio_studio_artifact_path",
        )

    for root in roots:
        try:
            resolved_root = root.resolve(strict=True)
        except OSError:
            continue
        try:
            resolved_file.relative_to(resolved_root)
        except ValueError:
            continue
        return resolved_file
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="invalid_audio_studio_artifact_path",
    )


def _is_sha256_hex(value: str | None) -> bool:
    return bool(value and len(value) == 64 and all(char in _HASH_HEX_CHARS for char in value))


def _sha256_file(path: FileSystemPath) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(64 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_hash_cache_identity(path: FileSystemPath) -> _MediaTicketHashCacheIdentity:
    stat_result = path.stat()
    return (
        str(path),
        getattr(stat_result, "st_dev", None),
        getattr(stat_result, "st_ino", None),
        stat_result.st_size,
        stat_result.st_mtime_ns,
        stat_result.st_ctime_ns,
    )


def _remember_ticket_hash_verification(
    cache_key: _MediaTicketHashCacheKey,
    identity: _MediaTicketHashCacheIdentity,
) -> None:
    if len(_MEDIA_TICKET_HASH_VERIFICATION_CACHE) >= _MEDIA_TICKET_HASH_CACHE_MAX_ENTRIES:
        _MEDIA_TICKET_HASH_VERIFICATION_CACHE.clear()
    _MEDIA_TICKET_HASH_VERIFICATION_CACHE[cache_key] = identity


def _validate_artifact_content_hash(
    artifact: AudioStudioArtifactRow,
    media_path: FileSystemPath,
    *,
    ticket_token_hash: str | None = None,
) -> None:
    if not _is_sha256_hex(artifact.content_hash):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="audio_studio_artifact_hash_unavailable",
        )
    expected_hash = artifact.content_hash.lower()
    cache_key = (ticket_token_hash, expected_hash) if ticket_token_hash is not None else None
    identity = _file_hash_cache_identity(media_path) if cache_key is not None else None
    if cache_key is not None and _MEDIA_TICKET_HASH_VERIFICATION_CACHE.get(cache_key) == identity:
        return

    if _sha256_file(media_path) != expected_hash:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="audio_studio_artifact_hash_mismatch",
        )
    if cache_key is not None and identity is not None:
        refreshed_identity = _file_hash_cache_identity(media_path)
        _remember_ticket_hash_verification(cache_key, refreshed_identity)


def _resolve_audio_studio_artifact_path(
    *,
    collections_db: CollectionsDatabase,
    user_id: int | str,
    artifact: AudioStudioArtifactRow,
) -> FileSystemPath:
    raw_storage_path = str(artifact.storage_path or "").strip()
    if not raw_storage_path or "://" in raw_storage_path:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="invalid_audio_studio_artifact_path",
        )

    roots = _audio_studio_artifact_roots(user_id)
    raw_path = FileSystemPath(raw_storage_path)
    if raw_path.is_absolute():
        return _resolve_contained_artifact_file(raw_path, roots)

    try:
        relative_filename = collections_db.resolve_output_storage_path(raw_storage_path)
    except (InvalidStoragePathError, InvalidStorageUserIdError, StorageUnavailableError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="invalid_audio_studio_artifact_path",
        ) from exc

    candidates = [
        _resolve_contained_artifact_file(root / relative_filename, roots)
        for root in roots
        if (root / relative_filename).exists()
    ]
    unique_candidates: list[FileSystemPath] = []
    for candidate in candidates:
        if candidate not in unique_candidates:
            unique_candidates.append(candidate)

    if not unique_candidates:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="audio_studio_artifact_file_not_found",
        )
    if len(unique_candidates) == 1:
        return unique_candidates[0]

    if not _is_sha256_hex(artifact.content_hash):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="audio_studio_artifact_path_ambiguous",
        )
    matches = [candidate for candidate in unique_candidates if _sha256_file(candidate) == artifact.content_hash.lower()]
    if len(matches) != 1:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="audio_studio_artifact_path_ambiguous",
        )
    return matches[0]


def _normalize_audio_mime(mime_type: str | None, path: FileSystemPath) -> str:
    normalized_mime = str(mime_type or "").split(";", 1)[0].strip().lower()
    inferred_mime = str(mimetypes.guess_type(path.name)[0] or "").split(";", 1)[0].strip().lower()
    if not normalized_mime:
        normalized_mime = inferred_mime
    suffix = path.suffix.lower()
    if (
        not normalized_mime
        or normalized_mime not in _PLAYABLE_AUDIO_MIME_TYPES
        or suffix not in _AUDIO_ALLOWED_SUFFIXES
        or suffix not in _AUDIO_MIME_SUFFIXES.get(normalized_mime, set())
    ):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="unsupported_audio_studio_artifact_media_type",
        )
    return normalized_mime


def _normalize_download_mime(mime_type: str | None, path: FileSystemPath) -> str:
    normalized = str(mime_type or "").split(";", 1)[0].strip().lower()
    inferred = str(mimetypes.guess_type(path.name)[0] or "").split(";", 1)[0].strip().lower()
    if not normalized:
        normalized = inferred or "application/octet-stream"
    suffix = path.suffix.lower()
    if suffix in _DANGEROUS_DOWNLOAD_SUFFIXES or normalized in _DANGEROUS_DOWNLOAD_MIME_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="unsupported_audio_studio_artifact_download_type",
        )
    return normalized


def _safe_artifact_filename_base(artifact: AudioStudioArtifactRow, *, fallback: str) -> str:
    safe_base = "".join(
        char
        for char in artifact.artifact_id
        if char.isascii() and (char.isalnum() or char in "._-") and char not in {'"', "'", "\r", "\n", "/", "\\"}
    ).strip("._-")
    if not safe_base:
        return fallback
    return safe_base


def _safe_audio_artifact_filename(
    *,
    artifact: AudioStudioArtifactRow,
    media_path: FileSystemPath,
    mime_type: str,
) -> str:
    safe_base = _safe_artifact_filename_base(artifact, fallback="audio-artifact")
    suffix = media_path.suffix.lower()
    allowed_suffixes = _AUDIO_MIME_SUFFIXES.get(mime_type, set())
    if suffix not in allowed_suffixes:
        suffix = sorted(allowed_suffixes)[0] if allowed_suffixes else ".audio"
    return f"{safe_base}{suffix}"


def _safe_download_artifact_filename(
    *,
    artifact: AudioStudioArtifactRow,
    media_path: FileSystemPath,
    mime_type: str,
) -> str:
    safe_base = _safe_artifact_filename_base(artifact, fallback="artifact-download")
    suffix = media_path.suffix.lower()
    if (
        not suffix
        or suffix in _DANGEROUS_DOWNLOAD_SUFFIXES
        or not suffix.startswith(".")
        or not all(char.isascii() and (char.isalnum() or char in "._-") for char in suffix)
    ):
        suffix = str(mimetypes.guess_extension(mime_type) or ".bin").lower()
    if suffix in _DANGEROUS_DOWNLOAD_SUFFIXES:
        suffix = ".bin"
    return f"{safe_base}{suffix}"


def _content_disposition(
    *,
    artifact: AudioStudioArtifactRow,
    media_path: FileSystemPath,
    mime_type: str,
    download: bool,
) -> str:
    disposition = "attachment" if download else "inline"
    filename = (
        _safe_download_artifact_filename(artifact=artifact, media_path=media_path, mime_type=mime_type)
        if download
        else _safe_audio_artifact_filename(artifact=artifact, media_path=media_path, mime_type=mime_type)
    )
    return f'{disposition}; filename="{filename}"'


def _raise_invalid_byte_range(total_size: int) -> None:
    raise HTTPException(
        status_code=status.HTTP_416_REQUESTED_RANGE_NOT_SATISFIABLE,
        detail="invalid_range",
        headers={
            "Content-Range": f"bytes */{total_size}",
            "Accept-Ranges": "bytes",
            "X-Content-Type-Options": "nosniff",
        },
    )


def _parse_single_byte_range(range_header: str, total_size: int) -> tuple[int, int]:
    if not range_header.lower().startswith("bytes="):
        _raise_invalid_byte_range(total_size)
    range_spec = range_header.split("=", 1)[1].strip()
    if not range_spec or "," in range_spec or "-" not in range_spec:
        _raise_invalid_byte_range(total_size)
    start_text, end_text = range_spec.split("-", 1)
    try:
        if not start_text:
            suffix_length = int(end_text)
            if suffix_length <= 0 or total_size <= 0:
                raise ValueError("invalid_suffix_range")
            start = max(total_size - suffix_length, 0)
            end = total_size - 1
        else:
            start = int(start_text)
            end = int(end_text) if end_text else total_size - 1
            if start < 0 or end < start or start >= total_size:
                raise ValueError("invalid_range")
            end = min(end, total_size - 1)
    except (TypeError, ValueError):
        _raise_invalid_byte_range(total_size)
    return start, end


def _iter_file_range(path: FileSystemPath, *, start: int, length: int) -> Iterator[bytes]:
    with path.open("rb") as file_obj:
        file_obj.seek(start)
        remaining = length
        while remaining > 0:
            chunk = file_obj.read(min(64 * 1024, remaining))
            if not chunk:
                break
            remaining -= len(chunk)
            yield chunk


def get_audio_studio_media_ticket_store() -> AudioStudioMediaTicketStore:
    """Return the AuthNZ-backed Audio Studio media ticket store."""

    users_db = AuthDatabaseConfig().get_user_database(client_id="audio_studio_media_tickets")
    return AudioStudioMediaTicketStore(users_db.backend)


def _auth_mode_label(request: Request) -> str | None:
    if request.headers.get("authorization"):
        return "jwt"
    if request.headers.get("x-api-key"):
        return "single_user"
    return None


def _ticket_path(raw_token: str) -> str:
    return f"/api/v1/audio-studio/media-tickets/{raw_token}"


def _raise_invalid_media_ticket() -> None:
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="audio_studio_media_ticket_not_found")


def _ticket_is_expired(ticket: AudioStudioMediaTicketRow) -> bool:
    expiry = datetime.fromisoformat(ticket.expires_at.replace("Z", "+00:00"))
    return expiry <= utc_now()


def _validate_ticket_for_redemption(
    raw_token: str,
    store: AudioStudioMediaTicketStore,
) -> AudioStudioMediaTicketRow:
    if not _MEDIA_TICKET_TOKEN_PATTERN.fullmatch(raw_token):
        _raise_invalid_media_ticket()
    ticket = store.get_by_hash(hash_media_ticket_token(raw_token))
    if ticket is None:
        _raise_invalid_media_ticket()
    if ticket.revoked_at:
        raise HTTPException(status_code=status.HTTP_410_GONE, detail="audio_studio_media_ticket_revoked")
    if ticket.consumed_at:
        raise HTTPException(status_code=status.HTTP_410_GONE, detail="audio_studio_media_ticket_consumed")
    if _ticket_is_expired(ticket):
        raise HTTPException(status_code=status.HTTP_410_GONE, detail="audio_studio_media_ticket_expired")
    return ticket


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


@router.post("/migrations/audiobook/preview", response_model=AudioStudioMigrationPreviewResponse)
async def preview_audiobook_studio_migration(
    request: AudioStudioMigrationPreview,
    current_user: User = Depends(get_request_user),
) -> AudioStudioMigrationPreviewResponse:
    """Preview a legacy local Audiobook Studio project without writing server records."""

    try:
        preview = preview_audio_studio_audiobook_migration(
            project_payload=request.project_payload,
            legacy_project_id=request.legacy_project_id,
            user_id=str(current_user.id),
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    return AudioStudioMigrationPreviewResponse(
        preview_id=preview.preview_id,
        fingerprint=preview.fingerprint,
        workflow=AudioStudioWorkflow.NARRATION,
        project_count=preview.project_count,
        section_count=preview.section_count,
        audio_reference_count=preview.audio_reference_count,
        needs_regeneration_count=preview.needs_regeneration_count,
        warnings=preview.warnings,
    )


@router.post(
    "/migrations/audiobook/commit",
    response_model=AudioStudioMigrationCommitResponse,
    status_code=status.HTTP_201_CREATED,
)
async def commit_audiobook_studio_migration(
    request: AudioStudioMigrationCommit,
    response: Response,
    current_user: User = Depends(get_request_user),
    collections_db: CollectionsDatabase = Depends(get_collections_db_for_user),
) -> AudioStudioMigrationCommitResponse:
    """Commit a sanitized legacy Audiobook Studio project into Narration."""

    try:
        committed = commit_audio_studio_audiobook_migration(
            collections_db=collections_db,
            project_payload=request.project_payload,
            legacy_project_id=None,
            idempotency_key=request.idempotency_key,
            user_id=str(current_user.id),
        )
    except ValueError as exc:
        if str(exc) == "audio_studio_idempotency_conflict":
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    if committed.replayed:
        response.status_code = status.HTTP_200_OK
    return AudioStudioMigrationCommitResponse(
        project=_project_response(committed.project),
        imported_section_count=committed.imported_section_count,
        audio_reference_count=committed.audio_reference_count,
        needs_regeneration_count=committed.needs_regeneration_count,
        fingerprint=committed.fingerprint,
        replayed=committed.replayed,
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
    workflow: AudioStudioWorkflow | None = Query(None),
    include_archived: bool = Query(False),
    current_user: User = Depends(get_request_user),
    collections_db: CollectionsDatabase = Depends(get_collections_db_for_user),
) -> AudioStudioProjectListResponse:
    """List current user's non-archived Audio Studio projects."""
    _ = current_user
    rows = collections_db.list_audio_studio_projects(
        limit=limit,
        offset=offset,
        include_archived=include_archived,
        workflow=workflow.value if workflow else None,
    )
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


@router.post(
    "/projects/{project_id}/renders",
    response_model=AudioStudioRenderJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def create_audio_studio_render(
    request: AudioStudioRenderCreate,
    project_id: AudioStudioIdPath,
    current_user: User = Depends(get_request_user),
    collections_db: CollectionsDatabase = Depends(get_collections_db_for_user),
) -> AudioStudioRenderJobResponse:
    """Create an idempotent Audio Studio render job with validated source pins."""

    project = _load_project_or_404(collections_db, project_id)
    output_format = str(request.options.get("output_format") or request.settings.get("output_format") or "wav")
    loudness_normalize = bool(request.options.get("loudness_normalize") or request.settings.get("loudness_normalize"))
    try:
        plan = build_render_plan(
            collections_db=collections_db,
            project=project,
            render_id=request.target_resource_id,
            target_revision_id=request.target_revision_id,
            artifact_refs=_artifact_refs_from_request(request),
            output_format=output_format,
            loudness_normalize=loudness_normalize,
            render_type=request.render_type,
        )
        job_options = {
            **request.options,
            "render_type": request.render_type,
            "output_format": plan.output_format,
            "loudness_normalize": plan.loudness_normalize,
            "artifact_refs": request.options.get("artifact_refs") or request.settings.get("artifact_refs") or [],
            "manifest": plan.manifest,
        }
        accepted = enqueue_audio_studio_render_job(
            jm=JobManager(),
            collections_db=collections_db,
            user_id=str(current_user.id),
            project_id=project.project_id,
            target_resource_kind=request.target_resource_kind.value,
            target_resource_id=request.target_resource_id,
            target_revision_id=request.target_revision_id,
            idempotency_key=request.idempotency_key,
            options=job_options,
        )
        return _render_job_response(
            accepted_or_job=accepted,
            project_id=project.project_id,
            render_id=request.target_resource_id,
            render_type=request.render_type,
            target_resource_kind=request.target_resource_kind.value,
            target_resource_id=request.target_resource_id,
            target_revision_id=request.target_revision_id,
            manifest=plan.manifest,
        )
    except ValueError as exc:
        if str(exc) in {"stale_target_revision", "stale_artifact_revision"}:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except DatabaseError as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="audio_studio_render_create_failed") from exc


@router.get(
    "/projects/{project_id}/renders/{job_id}",
    response_model=AudioStudioRenderJobResponse,
)
async def get_audio_studio_render(
    project_id: AudioStudioIdPath,
    job_id: AudioStudioIdPath,
    current_user: User = Depends(get_request_user),
    collections_db: CollectionsDatabase = Depends(get_collections_db_for_user),
) -> AudioStudioRenderJobResponse:
    """Fetch an Audio Studio render job owned by the current user."""

    project = _load_project_or_404(collections_db, project_id)
    job = _load_audio_studio_job_or_404(
        job_id=job_id,
        project=project,
        current_user=current_user,
        job_type=JOB_TYPE_RENDER,
    )
    payload = _job_payload(job)
    options = _provider_options(payload)
    return _render_job_response(
        accepted_or_job=job,
        project_id=project.project_id,
        render_id=str(payload.get("target_resource_id") or ""),
        render_type=str(options.get("render_type") or "preview_mix"),
        target_resource_kind=str(payload.get("target_resource_kind") or "render"),
        target_resource_id=str(payload.get("target_resource_id") or ""),
        target_revision_id=str(payload.get("target_revision_id") or ""),
        manifest=options.get("manifest") if isinstance(options.get("manifest"), dict) else {},
    )


@router.post(
    "/projects/{project_id}/exports",
    response_model=AudioStudioExportJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def create_audio_studio_export(
    request: AudioStudioExportCreate,
    project_id: AudioStudioIdPath,
    current_user: User = Depends(get_request_user),
    collections_db: CollectionsDatabase = Depends(get_collections_db_for_user),
) -> AudioStudioExportJobResponse:
    """Create an idempotent Audio Studio export job with source provenance."""

    project = _load_project_or_404(collections_db, project_id)
    settings = {**request.settings, **({"source_render_id": request.source_render_id} if request.source_render_id else {})}
    try:
        manifest = create_audio_studio_export_manifest(
            collections_db=collections_db,
            project=project,
            export_id=request.target_resource_id,
            export_type=request.export_type,
            target_revision_id=request.target_revision_id,
            artifact_refs=_artifact_refs_from_request(request),
            source_render_id=request.source_render_id,
            settings=settings,
        )
        job_options = {
            **request.options,
            "export_type": request.export_type,
            "source_render_id": request.source_render_id,
            "artifact_refs": request.options.get("artifact_refs") or request.settings.get("artifact_refs") or [],
            "manifest": manifest,
        }
        accepted = enqueue_audio_studio_export_job(
            jm=JobManager(),
            collections_db=collections_db,
            user_id=str(current_user.id),
            project_id=project.project_id,
            target_resource_kind=request.target_resource_kind.value,
            target_resource_id=request.target_resource_id,
            target_revision_id=request.target_revision_id,
            idempotency_key=request.idempotency_key,
            options=job_options,
        )
        return _export_job_response(
            accepted_or_job=accepted,
            project_id=project.project_id,
            export_id=request.target_resource_id,
            export_type=request.export_type,
            target_resource_kind=request.target_resource_kind.value,
            target_resource_id=request.target_resource_id,
            target_revision_id=request.target_revision_id,
            source_render_id=request.source_render_id,
            manifest=manifest,
        )
    except ValueError as exc:
        if str(exc) in {"stale_target_revision", "stale_artifact_revision"}:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except DatabaseError as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="audio_studio_export_create_failed") from exc


@router.get(
    "/projects/{project_id}/exports/{job_id}",
    response_model=AudioStudioExportJobResponse,
)
async def get_audio_studio_export(
    project_id: AudioStudioIdPath,
    job_id: AudioStudioIdPath,
    current_user: User = Depends(get_request_user),
    collections_db: CollectionsDatabase = Depends(get_collections_db_for_user),
) -> AudioStudioExportJobResponse:
    """Fetch an Audio Studio export job owned by the current user."""

    project = _load_project_or_404(collections_db, project_id)
    job = _load_audio_studio_job_or_404(
        job_id=job_id,
        project=project,
        current_user=current_user,
        job_type=JOB_TYPE_EXPORT,
    )
    payload = _job_payload(job)
    options = _provider_options(payload)
    return _export_job_response(
        accepted_or_job=job,
        project_id=project.project_id,
        export_id=str(payload.get("target_resource_id") or ""),
        export_type=str(options.get("export_type") or "zip_package"),
        target_resource_kind=str(payload.get("target_resource_kind") or "export"),
        target_resource_id=str(payload.get("target_resource_id") or ""),
        target_revision_id=str(payload.get("target_revision_id") or ""),
        source_render_id=options.get("source_render_id") if isinstance(options.get("source_render_id"), str) else None,
        manifest=options.get("manifest") if isinstance(options.get("manifest"), dict) else {},
    )


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


@router.post(
    "/projects/{project_id}/artifacts/{artifact_id}/tickets",
    response_model=AudioStudioMediaTicketResponse,
)
async def create_audio_studio_artifact_media_ticket(
    request: Request,
    project_id: AudioStudioIdPath,
    artifact_id: AudioStudioIdPath,
    ticket_request: AudioStudioMediaTicketCreate,
    current_user: User = Depends(get_request_user),
    collections_db: CollectionsDatabase = Depends(get_collections_db_for_user),
    ticket_store: AudioStudioMediaTicketStore = Depends(get_audio_studio_media_ticket_store),
) -> AudioStudioMediaTicketResponse:
    """Create a scoped short-lived ticket for artifact playback or download."""

    project = _load_project_or_404(collections_db, project_id)
    artifact = _load_audio_studio_artifact_or_404(collections_db, project, artifact_id)
    media_path = _resolve_audio_studio_artifact_path(
        collections_db=collections_db,
        user_id=current_user.id,
        artifact=artifact,
    )
    actual_size = media_path.stat().st_size
    if artifact.size_bytes is not None and artifact.size_bytes != actual_size:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="audio_studio_artifact_size_mismatch")

    purpose = ticket_request.purpose.value
    if ticket_request.purpose == AudioStudioMediaTicketPurpose.PLAYBACK:
        _normalize_audio_mime(artifact.mime_type, media_path)
        ttl = _PLAYBACK_TICKET_TTL
    else:
        _normalize_download_mime(artifact.mime_type, media_path)
        ttl = _DOWNLOAD_TICKET_TTL

    ticket_store.cleanup(retention=timedelta(hours=1))
    raw_token, ticket = ticket_store.create_ticket(
        user_id=int(current_user.id),
        project_id=project.project_id,
        artifact_id=artifact.artifact_id,
        purpose=purpose,
        expires_at=utc_now() + ttl,
        created_by_auth_mode=_auth_mode_label(request),
    )
    return AudioStudioMediaTicketResponse(
        ticket_path=_ticket_path(raw_token),
        expires_at=ticket.expires_at,
        purpose=ticket_request.purpose,
        artifact_id=artifact.artifact_id,
    )


@router.get("/media-tickets/{token}", response_model=None)
async def redeem_audio_studio_media_ticket(
    request: Request,
    token: str,
    ticket_store: AudioStudioMediaTicketStore = Depends(get_audio_studio_media_ticket_store),
) -> Response:
    """Redeem a scoped Audio Studio artifact media ticket."""

    ticket = _validate_ticket_for_redemption(token, ticket_store)
    if ticket.purpose == AudioStudioMediaTicketPurpose.DOWNLOAD.value:
        consumed = ticket_store.consume_download_ticket(ticket.token_hash)
        if consumed is None:
            refreshed = ticket_store.get_by_hash(ticket.token_hash)
            if refreshed and refreshed.consumed_at:
                raise HTTPException(status_code=status.HTTP_410_GONE, detail="audio_studio_media_ticket_consumed")
            if refreshed and refreshed.revoked_at:
                raise HTTPException(status_code=status.HTTP_410_GONE, detail="audio_studio_media_ticket_revoked")
            if refreshed and _ticket_is_expired(refreshed):
                raise HTTPException(status_code=status.HTTP_410_GONE, detail="audio_studio_media_ticket_expired")
            _raise_invalid_media_ticket()
        ticket = consumed
    else:
        ticket_store.touch_redeemed(ticket.token_hash)

    collections_db = CollectionsDatabase.for_user(user_id=ticket.user_id)
    project = _load_project_or_404(collections_db, ticket.project_id)
    artifact = _load_audio_studio_artifact_or_404(collections_db, project, ticket.artifact_id)
    media_path = _resolve_audio_studio_artifact_path(
        collections_db=collections_db,
        user_id=ticket.user_id,
        artifact=artifact,
    )
    actual_size = media_path.stat().st_size
    if artifact.size_bytes is not None and artifact.size_bytes != actual_size:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="audio_studio_artifact_size_mismatch")
    _validate_artifact_content_hash(
        artifact,
        media_path,
        ticket_token_hash=ticket.token_hash if ticket.purpose == AudioStudioMediaTicketPurpose.PLAYBACK.value else None,
    )

    if ticket.purpose == AudioStudioMediaTicketPurpose.PLAYBACK.value:
        mime_type = _normalize_audio_mime(artifact.mime_type, media_path)
        headers = {
            "Accept-Ranges": "bytes",
            "Cache-Control": "private, no-store",
            "Content-Disposition": _content_disposition(
                artifact=artifact,
                media_path=media_path,
                mime_type=mime_type,
                download=False,
            ),
            "Referrer-Policy": "no-referrer",
            "X-Content-Type-Options": "nosniff",
        }
        range_header = request.headers.get("range")
        if range_header is not None:
            start, end = _parse_single_byte_range(range_header, actual_size)
            content_length = end - start + 1
            return StreamingResponse(
                _iter_file_range(media_path, start=start, length=content_length),
                status_code=status.HTTP_206_PARTIAL_CONTENT,
                headers={
                    **headers,
                    "Content-Range": f"bytes {start}-{end}/{actual_size}",
                    "Content-Length": str(content_length),
                    "Content-Type": mime_type,
                },
            )
        return FileResponse(str(media_path), media_type=mime_type, headers=headers)

    mime_type = _normalize_download_mime(artifact.mime_type, media_path)
    return StreamingResponse(
        _iter_file_range(media_path, start=0, length=actual_size),
        media_type=mime_type,
        headers={
            "Cache-Control": "private, no-store",
            "Content-Disposition": _content_disposition(
                artifact=artifact,
                media_path=media_path,
                mime_type=mime_type,
                download=True,
            ),
            "Content-Length": str(actual_size),
            "Referrer-Policy": "no-referrer",
            "X-Content-Type-Options": "nosniff",
        },
    )


@router.get("/projects/{project_id}/artifacts/{artifact_id}/media", response_model=None)
async def get_audio_studio_artifact_media(
    request: Request,
    project_id: AudioStudioIdPath,
    artifact_id: AudioStudioIdPath,
    download: bool = Query(False),
    current_user: User = Depends(get_request_user),
    collections_db: CollectionsDatabase = Depends(get_collections_db_for_user),
) -> Response:
    """Serve an Audio Studio audio artifact for playback or download."""

    project = _load_project_or_404(collections_db, project_id)
    artifact = _load_audio_studio_artifact_or_404(collections_db, project, artifact_id)
    media_path = _resolve_audio_studio_artifact_path(
        collections_db=collections_db,
        user_id=current_user.id,
        artifact=artifact,
    )
    mime_type = _normalize_audio_mime(artifact.mime_type, media_path)
    actual_size = media_path.stat().st_size
    if artifact.size_bytes is not None and artifact.size_bytes != actual_size:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="audio_studio_artifact_size_mismatch",
        )

    headers = {
        "Accept-Ranges": "bytes",
        "Cache-Control": "private, no-store",
        "Content-Disposition": _content_disposition(
            artifact=artifact,
            media_path=media_path,
            mime_type=mime_type,
            download=download,
        ),
        "X-Content-Type-Options": "nosniff",
    }
    range_header = request.headers.get("range")
    if range_header is not None:
        start, end = _parse_single_byte_range(range_header, actual_size)
        content_length = end - start + 1
        range_headers = {
            **headers,
            "Content-Range": f"bytes {start}-{end}/{actual_size}",
            "Content-Length": str(content_length),
            "Content-Type": mime_type,
        }
        return StreamingResponse(
            _iter_file_range(media_path, start=start, length=content_length),
            status_code=status.HTTP_206_PARTIAL_CONTENT,
            headers=range_headers,
        )

    return FileResponse(
        str(media_path),
        media_type=mime_type,
        headers=headers,
    )
