"""Audiobook API endpoints.

Provides parsing, job management, project and voice profile CRUD, and subtitle export.
All endpoints require authentication and operate on per-user collections data.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path as PathlibPath
from typing import Any
from uuid import uuid4

from cachetools import LRUCache
from fastapi import APIRouter, Depends, HTTPException, Path, Query
from fastapi.responses import PlainTextResponse
from loguru import logger
from starlette import status
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import check_rate_limit, get_request_user, User

from tldw_Server_API.app.api.v1.API_Deps.Collections_DB_Deps import get_collections_db_for_user
from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user
from tldw_Server_API.app.api.v1.endpoints._pagination_utils import build_offset_pagination_meta
from tldw_Server_API.app.api.v1.schemas.audiobook_schemas import (
    AlignmentPayload,
    ArtifactInfo,
    AudiobookArtifactsResponse,
    AudiobookChapterInfo,
    AudiobookChapterListResponse,
    AudiobookJobCreateResponse,
    AudiobookJobRequest,
    AudiobookJobStatusResponse,
    AudiobookParseRequest,
    AudiobookParseResponse,
    AudiobookProjectInfo,
    AudiobookProjectListResponse,
    AudiobookProjectResponse,
    ChapterPreview,
    SubtitleExportRequest,
    VoiceProfileCreateRequest,
    VoiceProfileDeleteResponse,
    VoiceProfileListResponse,
    VoiceProfileResponse,
)
from tldw_Server_API.app.core.Audiobooks.subtitle_generator import generate_subtitles
from tldw_Server_API.app.core.Audiobooks.subtitle_parser import normalize_subtitle_source
from tldw_Server_API.app.core.Audiobooks.tag_parser import (
    build_chapters_from_markers,
    parse_tagged_text,
)
from tldw_Server_API.app.core.Chunking.strategies.ebook_chapters import EbookChapterChunkingStrategy
from tldw_Server_API.app.core.config import get_config_value
from tldw_Server_API.app.core.DB_Management.Collections_DB import CollectionsDatabase
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.Ingestion_Media_Processing.Books.Book_Processing_Lib import (
    extract_epub_metadata_from_text,
    process_epub,
)
from tldw_Server_API.app.core.Ingestion_Media_Processing.path_utils import resolve_safe_local_path
from tldw_Server_API.app.core.Ingestion_Media_Processing.PDF.PDF_Processing_Lib import (
    extract_text_and_format_from_pdf,
)
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.testing import is_truthy

router = APIRouter(prefix="/audiobooks", tags=["audiobooks"])

MAX_CACHED_JOB_MANAGER_INSTANCES = 4
_job_manager_cache: LRUCache = LRUCache(maxsize=MAX_CACHED_JOB_MANAGER_INSTANCES)
_job_manager_lock = threading.Lock()
MAX_CACHED_PROJECT_LOOKUPS = 512
_project_lookup_cache: LRUCache = LRUCache(maxsize=MAX_CACHED_PROJECT_LOOKUPS)
_project_lookup_lock = threading.Lock()
_PROJECT_CACHE_MISS = object()

_AUDIOBOOKS_COERCE_EXCEPTIONS = (
    TypeError,
    ValueError,
    AttributeError,
)

_AUDIOBOOKS_JSON_EXCEPTIONS = (
    json.JSONDecodeError,
    TypeError,
    ValueError,
)

_AUDIOBOOKS_DB_OPERATION_EXCEPTIONS = (
    OSError,
    ValueError,
    TypeError,
    KeyError,
    RuntimeError,
    AttributeError,
)


def _not_implemented(detail: str) -> None:
    raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail=detail)


def _normalize_subtitles(text: str, input_type: str) -> str:
    return normalize_subtitle_source(text, input_type)


def _parse_bool(value: str | bool | None) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    return is_truthy(str(value).strip().lower())


def _resolve_subtitle_persist(request: SubtitleExportRequest) -> bool:
    if request.persist is not None:
        return bool(request.persist)
    env_val = os.getenv("AUDIOBOOK_SUBTITLES_PERSIST")
    cfg_val = get_config_value("Audiobooks", "subtitles_persist")
    raw = env_val if env_val not in (None, "") else cfg_val
    resolved = _parse_bool(raw)
    return bool(resolved) if resolved is not None else False


def _record_has_tts_hint(record: dict[str, Any] | None) -> bool:
    if not isinstance(record, dict):
        return False
    if record.get("tts_provider") or record.get("tts_model"):
        return True
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        return False
    return bool(metadata.get("tts_provider") or metadata.get("tts_model"))


def _apply_default_audiobook_tts_hints(payload: dict[str, Any]) -> None:
    # Keep the persisted job payload aligned with the API schema:
    # when subtitles are requested without an explicit TTS hint, the request
    # follows the Kokoro/alignment-capable path by default.
    if payload.get("subtitles") is not None and not _record_has_tts_hint(payload):
        payload["tts_model"] = "kokoro"

    items = payload.get("items")
    if not isinstance(items, list):
        return

    inherited_has_tts_hint = _record_has_tts_hint(payload)
    for item in items:
        if not isinstance(item, dict):
            continue
        if item.get("subtitles") is None:
            continue
        if inherited_has_tts_hint or _record_has_tts_hint(item):
            continue
        item["tts_model"] = "kokoro"


def _resolve_subtitle_ttl_hours(request: SubtitleExportRequest) -> int | None:
    if request.cache_ttl_hours is not None:
        return int(request.cache_ttl_hours)
    env_val = os.getenv("AUDIOBOOK_SUBTITLES_CACHE_TTL_HOURS")
    cfg_val = get_config_value("Audiobooks", "subtitles_cache_ttl_hours")
    raw = env_val if env_val not in (None, "") else cfg_val
    if raw is None:
        return None
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _subtitle_cache_key(
    *,
    alignment: AlignmentPayload,
    alignment_output_id: int | None,
    request: SubtitleExportRequest,
) -> str:
    base: dict[str, object] = {
        "alignment_output_id": alignment_output_id,
        "format": request.format,
        "mode": request.mode,
        "variant": request.variant,
        "words_per_cue": request.words_per_cue,
        "max_chars": request.max_chars,
        "max_lines": request.max_lines,
    }
    if alignment_output_id is None:
        try:
            alignment_dump = alignment.model_dump(mode="json")
        except _AUDIOBOOKS_COERCE_EXCEPTIONS:
            alignment_dump = alignment.dict() if hasattr(alignment, "dict") else alignment
        alignment_hash = hashlib.sha256(
            json.dumps(alignment_dump, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        base["alignment_hash"] = alignment_hash
    digest = hashlib.sha256(
        json.dumps(base, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return digest[:16]


def _subtitle_cache_title(cache_key: str) -> str:
    return f"audiobook_subtitle_{cache_key}"


def _subtitle_storage_filename(cache_key: str, fmt: str) -> str:
    return f"subtitle_{cache_key}.{fmt}"


def _load_alignment_from_output(
    collections_db: CollectionsDatabase,
    *,
    output_id: int,
    user_id: int,
) -> tuple[AlignmentPayload, dict[str, object], str]:
    try:
        row = collections_db.get_output_artifact(output_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="alignment_output_not_found") from exc
    if row.type != "audiobook_alignment":
        raise HTTPException(status_code=400, detail="invalid_alignment_output")
    outputs_dir = DatabasePaths.get_user_outputs_dir(user_id)
    # Defense-in-depth: validate storage_path doesn't escape outputs_dir
    storage_path = str(row.storage_path or "")
    if ".." in PathlibPath(storage_path).parts or PathlibPath(storage_path).is_absolute():
        raise HTTPException(status_code=400, detail="invalid_storage_path")
    alignment_path = outputs_dir / storage_path
    if not alignment_path.exists():
        raise HTTPException(status_code=404, detail="alignment_file_not_found")
    try:
        raw = alignment_path.read_text(encoding="utf-8")
        payload = json.loads(raw)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="alignment_parse_failed") from exc
    try:
        alignment = AlignmentPayload(**payload)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="alignment_invalid") from exc
    meta: dict[str, object] = {}
    if row.metadata_json:
        try:
            parsed = json.loads(row.metadata_json)
            if isinstance(parsed, dict):
                meta = parsed
        except _AUDIOBOOKS_JSON_EXCEPTIONS:
            meta = {}
    return alignment, meta, row.storage_path


def _resolve_upload_path(upload_id: str, user_id: int) -> PathlibPath | None:
    if not upload_id:
        return None
    base_dir = DatabasePaths.get_user_temp_outputs_dir(user_id)
    safe_path = resolve_safe_local_path(PathlibPath(upload_id), base_dir)
    if safe_path is None:
        return None
    if not safe_path.exists():
        return None
    return safe_path


def _detect_chapters(
    text: str,
    *,
    language: str | None = None,
    custom_pattern: str | None = None,
) -> list[ChapterPreview]:
    if not text or not text.strip():
        return []
    lang = language or "en"
    strategy = EbookChapterChunkingStrategy(language=lang)
    word_count = max(1, len(text.split()))
    results = strategy.chunk_with_metadata(
        text,
        max_size=word_count,
        overlap=0,
        custom_chapter_pattern=custom_pattern,
    )
    chapters: list[ChapterPreview] = []
    for result in results:
        options = getattr(result.metadata, "options", {}) or {}
        title = options.get("chapter_title")
        chapter_id = f"ch_{result.metadata.index + 1:03d}"
        chapters.append(
            ChapterPreview(
                chapter_id=chapter_id,
                title=title,
                start_offset=result.metadata.start_char,
                end_offset=result.metadata.end_char,
                word_count=result.metadata.word_count,
            )
        )
    return chapters


def _get_job_manager() -> JobManager:
    db_url = (os.getenv("JOBS_DB_URL") or "").strip()
    db_path = (os.getenv("JOBS_DB_PATH") or "").strip()
    if db_url:
        cache_key = f"url:{db_url}"
    elif db_path:
        cache_key = f"path:{db_path}"
    else:
        cache_key = "default"
    with _job_manager_lock:
        cached = _job_manager_cache.get(cache_key)
        if cached is not None:
            return cached

        if not db_url:
            job_manager = JobManager(db_path=PathlibPath(db_path)) if db_path else JobManager()
        else:
            backend = "postgres" if db_url.startswith("postgres") else None
            job_manager = JobManager(backend=backend, db_url=db_url)

        _job_manager_cache[cache_key] = job_manager
        return job_manager


def _normalize_job_status(job_status: str | None) -> str:
    if not job_status:
        return "queued"
    norm = job_status.lower()
    if norm == "cancelled":
        return "canceled"
    if norm == "quarantined":
        return "failed"
    return norm


def _job_project_id(job_row: dict) -> str:
    payload = job_row.get("payload") if isinstance(job_row, dict) else None
    if isinstance(payload, dict):
        project_id = payload.get("project_id")
        if isinstance(project_id, str) and project_id:
            return project_id
    return f"abk_{job_row.get('id', 0)}"


def _safe_json_loads(raw: str | None) -> dict:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except _AUDIOBOOKS_JSON_EXCEPTIONS:
        return {}
    if isinstance(parsed, dict):
        return parsed
    return {}


def _project_row_to_info(row) -> AudiobookProjectInfo:
    settings = _safe_json_loads(row.settings_json)
    source_ref = _safe_json_loads(row.source_ref)
    project_id_val = None
    if getattr(row, "project_id", None):
        project_id_val = str(row.project_id)
    else:
        project_id = settings.get("project_id")
        if isinstance(project_id, str):
            project_id_val = project_id
    return AudiobookProjectInfo(
        project_db_id=int(row.id),
        project_id=project_id_val,
        title=row.title,
        status=row.status,
        source_ref=source_ref or None,
        settings=settings or None,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def _resolve_project_row(collections_db: CollectionsDatabase, project_ref: str):
    if project_ref.isdigit():
        try:
            return collections_db.get_audiobook_project(int(project_ref))
        except KeyError:
            pass
    try:
        return collections_db.get_audiobook_project_by_project_id(project_ref)
    except KeyError:
        pass
    cache_key = (str(getattr(collections_db, "user_id", "")), project_ref)
    with _project_lookup_lock:
        cached = _project_lookup_cache.get(cache_key, _PROJECT_CACHE_MISS)
    if cached is not _PROJECT_CACHE_MISS:
        if cached is None:
            raise KeyError("audiobook_project_not_found")
        return cached
    # Legacy fallback: scan settings_json for pre-backfill entries; cache results to avoid repeated O(n) scans.
    offset = 0
    limit = 200
    while True:
        rows = collections_db.list_audiobook_projects(limit=limit, offset=offset)
        if not rows:
            break
        for row in rows:
            settings = _safe_json_loads(row.settings_json)
            project_id = settings.get("project_id")
            if isinstance(project_id, str) and project_id == project_ref:
                with _project_lookup_lock:
                    _project_lookup_cache[cache_key] = row
                return row
        offset += limit
    with _project_lookup_lock:
        _project_lookup_cache[cache_key] = None
    raise KeyError("audiobook_project_not_found")


def _project_row_project_id(row, fallback: str) -> str:
    if getattr(row, "project_id", None):
        return str(row.project_id)
    settings = _safe_json_loads(row.settings_json)
    project_id = settings.get("project_id")
    if isinstance(project_id, str) and project_id:
        return project_id
    return fallback


def _parse_progress_message(
    message: str | None,
) -> tuple[str, int | None, int | None, int | None, int | None]:
    if not message:
        return "audiobook_job", None, None, None, None
    raw = str(message).strip()
    if raw.startswith("{") and raw.endswith("}"):
        parsed = _safe_json_loads(raw)
        if parsed:
            stage = parsed.get("stage") or "audiobook_job"
            try:
                chapter_index = int(parsed.get("chapter_index")) if parsed.get("chapter_index") is not None else None
            except _AUDIOBOOKS_COERCE_EXCEPTIONS:
                chapter_index = None
            try:
                chapters_total = int(parsed.get("chapters_total")) if parsed.get("chapters_total") is not None else None
            except _AUDIOBOOKS_COERCE_EXCEPTIONS:
                chapters_total = None
            try:
                item_index = int(parsed.get("item_index")) if parsed.get("item_index") is not None else None
            except _AUDIOBOOKS_COERCE_EXCEPTIONS:
                item_index = None
            try:
                items_total = int(parsed.get("items_total")) if parsed.get("items_total") is not None else None
            except _AUDIOBOOKS_COERCE_EXCEPTIONS:
                items_total = None
            return str(stage), chapter_index, chapters_total, item_index, items_total
    return raw, None, None, None, None


@router.post(
    "/parse",
    response_model=AudiobookParseResponse,
    summary="Parse audiobook source and detect chapters",
    dependencies=[Depends(check_rate_limit)],
)
async def parse_audiobook_source(
    request: AudiobookParseRequest,
    _current_user: User = Depends(get_request_user),
    media_db: Any = Depends(get_media_db_for_user),
) -> AudiobookParseResponse:
    source = request.source
    input_type = source.input_type
    text: str | None = None
    metadata: dict = {"source_type": input_type}

    if source.raw_text:
        text = source.raw_text
        title, author = extract_epub_metadata_from_text(text)
        if title:
            metadata["title"] = title
        if author:
            metadata["author"] = author
    elif source.media_id is not None:
        try:
            media_id = int(source.media_id)
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail="invalid_media_id") from exc
        record = media_db.get_media_by_id(media_id)
        if not record:
            raise HTTPException(status_code=404, detail="media_not_found")
        text = record.get("content") or ""
        metadata["title"] = record.get("title")
        if record.get("author"):
            metadata["author"] = record.get("author")
    elif source.upload_id:
        upload_path = _resolve_upload_path(source.upload_id, int(_current_user.id))
        if upload_path is None:
            raise HTTPException(status_code=404, detail="upload_not_found")
        try:
            if input_type == "epub":
                result = await asyncio.to_thread(
                    process_epub, str(upload_path), perform_chunking=False
                )
                text = result.get("content") or ""
                if result.get("metadata"):
                    meta = result["metadata"]
                    if meta.get("title"):
                        metadata["title"] = meta.get("title")
                    if meta.get("author"):
                        metadata["author"] = meta.get("author")
            elif input_type == "pdf":
                text = await asyncio.to_thread(
                    extract_text_and_format_from_pdf, str(upload_path)
                )
            else:
                text = await asyncio.to_thread(
                    upload_path.read_text, encoding="utf-8", errors="ignore"
                )
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Failed to parse upload for audiobook")
            raise HTTPException(status_code=500, detail="parse_failed") from exc

    if text is None:
        raise HTTPException(status_code=400, detail="no_source_text")

    normalized_text = _normalize_subtitles(text, input_type)
    tag_result = parse_tagged_text(normalized_text)
    normalized_text = tag_result.clean_text
    if request.max_chars and len(normalized_text) > request.max_chars:
        normalized_text = normalized_text[: request.max_chars]

    chapters: list[ChapterPreview] = []
    if tag_result.chapter_markers:
        chapters = build_chapters_from_markers(normalized_text, tag_result.chapter_markers)
    elif request.detect_chapters:
        try:
            chapters = _detect_chapters(
                normalized_text,
                language=request.language,
                custom_pattern=request.custom_chapter_pattern,
            )
        except Exception as exc:
            logger.warning("Chapter detection failed")
            raise HTTPException(status_code=400, detail="chapter_detection_failed") from exc

    if tag_result.chapter_markers or tag_result.voice_markers or tag_result.speed_markers or tag_result.ts_markers:
        metadata["tag_markers"] = tag_result.as_metadata()

    project_id = f"abk_{uuid4().hex[:12]}"
    return AudiobookParseResponse(
        project_id=project_id,
        normalized_text=normalized_text,
        chapters=chapters,
        metadata=metadata,
    )


@router.post(
    "/jobs",
    response_model=AudiobookJobCreateResponse,
    summary="Create an audiobook generation job",
    dependencies=[Depends(check_rate_limit)],
)
async def create_audiobook_job(
    request: AudiobookJobRequest,
    _current_user: User = Depends(get_request_user),
) -> AudiobookJobCreateResponse:
    project_id = f"abk_{uuid4().hex[:12]}"
    priority = 5
    if request.queue is not None:
        try:
            priority = int(request.queue.priority)
        except (TypeError, ValueError, AttributeError):
            priority = 5
    if priority < 1:
        priority = 1
    if priority > 10:
        priority = 10

    payload = request.model_dump()
    _apply_default_audiobook_tts_hints(payload)
    payload["project_id"] = project_id
    if request.items:
        # Preserve per-item subtitle overrides only when explicitly set.
        for idx, item in enumerate(request.items):
            if item.subtitles is None and "subtitles" not in item.model_fields_set:
                try:
                    if isinstance(payload.get("items"), list):
                        payload["items"][idx].pop("subtitles", None)
                except _AUDIOBOOKS_COERCE_EXCEPTIONS:
                    logger.debug("Failed to remove subtitle override")

    job_manager = _get_job_manager()
    batch_group = request.queue.batch_group if request.queue is not None else None
    job = job_manager.create_job(
        domain="audiobooks",
        queue="default",
        job_type="audiobook_generate",
        payload=payload,
        owner_user_id=str(getattr(_current_user, "id", "")),
        batch_group=batch_group,
        priority=priority,
    )

    return AudiobookJobCreateResponse(
        job_id=int(job["id"]),
        project_id=project_id,
        status=_normalize_job_status(job_status=job.get("status")),
    )


@router.get(
    "/jobs/{job_id}",
    response_model=AudiobookJobStatusResponse,
    summary="Get audiobook job status",
)
async def get_audiobook_job_status(
    job_id: int = Path(..., ge=1, description="Audiobook job id"),
    _current_user: User = Depends(get_request_user),
) -> AudiobookJobStatusResponse:
    job_manager = _get_job_manager()
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job_not_found")
    owner_user_id = job.get("owner_user_id") if isinstance(job, dict) else getattr(job, "owner_user_id", None)
    user_id = getattr(_current_user, "id", None)
    if user_id is None:
        raise HTTPException(status_code=401, detail="unauthorized")
    if not owner_user_id or str(owner_user_id) != str(user_id):
        raise HTTPException(status_code=404, detail="job_not_found")

    job_status = _normalize_job_status(job_status=job.get("status"))
    project_id = _job_project_id(job)

    errors: list[str] = []
    for key in ("error_message", "last_error", "error_code"):
        value = job.get(key)
        if value:
            errors.append(str(value))

    progress = None
    percent = job.get("progress_percent")
    stage = job.get("progress_message")
    if percent is not None or stage:
        try:
            percent_val = int(percent) if percent is not None else None
        except _AUDIOBOOKS_COERCE_EXCEPTIONS:
            percent_val = None
        stage_val, chapter_index, chapters_total, item_index, items_total = _parse_progress_message(stage)
        progress = {
            "stage": stage_val,
            "percent": percent_val,
            "chapter_index": chapter_index,
            "chapters_total": chapters_total,
            "item_index": item_index,
            "items_total": items_total,
        }

    return AudiobookJobStatusResponse(
        job_id=int(job["id"]),
        project_id=project_id,
        status=job_status,
        progress=progress,
        errors=errors,
    )


@router.get(
    "/jobs/{job_id}/artifacts",
    response_model=AudiobookArtifactsResponse,
    summary="List audiobook job artifacts",
    dependencies=[Depends(check_rate_limit)],
)
async def get_audiobook_job_artifacts(
    job_id: int = Path(..., ge=1, description="Audiobook job id"),
    _current_user: User = Depends(get_request_user),
    collections_db: CollectionsDatabase = Depends(get_collections_db_for_user),
) -> AudiobookArtifactsResponse:
    job_manager = _get_job_manager()
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job_not_found")
    project_id = _job_project_id(job)
    owner_user_id = job.get("owner_user_id") if isinstance(job, dict) else getattr(job, "owner_user_id", None)
    current_user_id = getattr(_current_user, "id", None)
    if current_user_id is None:
        raise HTTPException(status_code=401, detail="unauthorized")
    if not owner_user_id or str(owner_user_id) != str(current_user_id):
        raise HTTPException(status_code=404, detail="job_not_found")

    limit = 200
    offset = 0
    rows, total = collections_db.list_output_artifacts(job_id=int(job_id), limit=limit, offset=offset)
    artifacts: list[ArtifactInfo] = []
    type_map = {
        "audiobook_audio": "audio",
        "audiobook_subtitle": "subtitle",
        "audiobook_alignment": "alignment",
        "audiobook_package": "package",
    }
    for row in rows:
        metadata = {}
        if row.metadata_json:
            try:
                metadata = json.loads(row.metadata_json)
            except _AUDIOBOOKS_JSON_EXCEPTIONS:
                metadata = {}
        artifact_type = metadata.get("artifact_type")
        if not artifact_type:
            artifact_type = type_map.get(str(row.type), "audio")
        artifact_type = str(artifact_type)
        scope = metadata.get("scope")
        chapter_id = metadata.get("chapter_id")
        download_url = f"/api/v1/outputs/{row.id}/download"
        artifacts.append(
            ArtifactInfo(
                artifact_type=artifact_type,
                format=row.format,
                scope=scope,
                chapter_id=chapter_id,
                output_id=row.id,
                download_url=download_url,
            )
        )
    return AudiobookArtifactsResponse(
        project_id=project_id,
        artifacts=artifacts,
        limit=limit,
        offset=offset,
        pagination=build_offset_pagination_meta(
            total=total,
            limit=limit,
            offset=offset,
            count=len(artifacts),
        ),
    )


@router.get(
    "/projects",
    response_model=AudiobookProjectListResponse,
    summary="List audiobook projects",
    dependencies=[Depends(check_rate_limit)],
)
async def list_audiobook_projects(
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    _current_user: User = Depends(get_request_user),
    collections_db: CollectionsDatabase = Depends(get_collections_db_for_user),
) -> AudiobookProjectListResponse:
    try:
        rows = collections_db.list_audiobook_projects(limit=limit, offset=offset)
        total = collections_db.count_audiobook_projects()
    except Exception as exc:
        logger.exception("Failed to list audiobook projects")
        raise HTTPException(status_code=500, detail="audiobook_project_list_failed") from exc
    projects = [_project_row_to_info(row) for row in rows]
    return AudiobookProjectListResponse(
        projects=projects,
        total=total,
        limit=limit,
        offset=offset,
        pagination=build_offset_pagination_meta(
            total=total,
            limit=limit,
            offset=offset,
            count=len(projects),
        ),
    )


@router.get(
    "/projects/{project_ref}",
    response_model=AudiobookProjectResponse,
    summary="Get audiobook project",
)
async def get_audiobook_project(
    project_ref: str = Path(..., min_length=1, description="Project id or database id"),
    _current_user: User = Depends(get_request_user),
    collections_db: CollectionsDatabase = Depends(get_collections_db_for_user),
) -> AudiobookProjectResponse:
    try:
        row = _resolve_project_row(collections_db, project_ref)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="audiobook_project_not_found") from exc
    except Exception as exc:
        logger.exception("Failed to fetch audiobook project")
        raise HTTPException(status_code=500, detail="audiobook_project_fetch_failed") from exc
    return AudiobookProjectResponse(project=_project_row_to_info(row))


@router.get(
    "/projects/{project_ref}/chapters",
    response_model=AudiobookChapterListResponse,
    summary="List audiobook project chapters",
    dependencies=[Depends(check_rate_limit)],
)
async def list_audiobook_project_chapters(
    project_ref: str = Path(..., min_length=1, description="Project id or database id"),
    limit: int = Query(200, ge=1, le=2000),
    offset: int = Query(0, ge=0),
    _current_user: User = Depends(get_request_user),
    collections_db: CollectionsDatabase = Depends(get_collections_db_for_user),
) -> AudiobookChapterListResponse:
    try:
        project_row = _resolve_project_row(collections_db, project_ref)
        chapter_rows = collections_db.list_audiobook_chapters(
            project_id=int(project_row.id),
            limit=limit + 1,
            offset=offset,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="audiobook_project_not_found") from exc
    except Exception as exc:
        logger.exception("Failed to list audiobook chapters")
        raise HTTPException(status_code=500, detail="audiobook_chapters_list_failed") from exc
    chapters: list[AudiobookChapterInfo] = []
    has_more = len(chapter_rows) > limit
    for row in chapter_rows[:limit]:
        metadata = _safe_json_loads(row.metadata_json)
        chapters.append(
            AudiobookChapterInfo(
                id=int(row.id),
                chapter_index=int(row.chapter_index),
                title=row.title,
                start_offset=row.start_offset,
                end_offset=row.end_offset,
                voice_profile_id=row.voice_profile_id,
                speed=float(row.speed) if row.speed is not None else None,
                metadata=metadata,
            )
        )
    project_id = _project_row_project_id(project_row, project_ref)
    return AudiobookChapterListResponse(
        project_id=project_id,
        chapters=chapters,
        limit=limit,
        offset=offset,
        pagination=build_offset_pagination_meta(
            total=None,
            limit=limit,
            offset=offset,
            count=len(chapters),
            has_more=has_more,
        ),
    )


@router.get(
    "/projects/{project_ref}/artifacts",
    response_model=AudiobookArtifactsResponse,
    summary="List audiobook project artifacts",
    dependencies=[Depends(check_rate_limit)],
)
async def list_audiobook_project_artifacts(
    project_ref: str = Path(..., min_length=1, description="Project id or database id"),
    limit: int = Query(200, ge=1, le=2000),
    offset: int = Query(0, ge=0),
    _current_user: User = Depends(get_request_user),
    collections_db: CollectionsDatabase = Depends(get_collections_db_for_user),
) -> AudiobookArtifactsResponse:
    try:
        project_row = _resolve_project_row(collections_db, project_ref)
        artifact_rows = collections_db.list_audiobook_artifacts(
            project_id=int(project_row.id),
            limit=limit + 1,
            offset=offset,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="audiobook_project_not_found") from exc
    except Exception as exc:
        logger.exception("Failed to list audiobook artifacts")
        raise HTTPException(status_code=500, detail="audiobook_artifacts_list_failed") from exc
    artifacts: list[ArtifactInfo] = []
    has_more = len(artifact_rows) > limit
    for row in artifact_rows[:limit]:
        metadata = _safe_json_loads(row.metadata_json)
        artifacts.append(
            ArtifactInfo(
                artifact_type=str(row.artifact_type),
                format=row.format,
                scope=metadata.get("scope"),
                chapter_id=metadata.get("chapter_id"),
                output_id=int(row.output_id),
                download_url=f"/api/v1/outputs/{int(row.output_id)}/download",
            )
        )
    project_id = _project_row_project_id(project_row, project_ref)
    return AudiobookArtifactsResponse(
        project_id=project_id,
        artifacts=artifacts,
        limit=limit,
        offset=offset,
        pagination=build_offset_pagination_meta(
            total=None,
            limit=limit,
            offset=offset,
            count=len(artifacts),
            has_more=has_more,
        ),
    )


@router.post(
    "/voices/profiles",
    response_model=VoiceProfileResponse,
    summary="Create a voice profile",
    dependencies=[Depends(check_rate_limit)],
)
async def create_voice_profile(
    request: VoiceProfileCreateRequest,
    _current_user: User = Depends(get_request_user),
    collections_db: CollectionsDatabase = Depends(get_collections_db_for_user),
) -> VoiceProfileResponse:
    profile_id = f"vp_{uuid4().hex[:12]}"
    overrides = [override.model_dump() for override in (request.chapter_overrides or [])]
    overrides_json = json.dumps(overrides) if overrides else None
    try:
        row = collections_db.create_voice_profile(
            profile_id=profile_id,
            name=request.name,
            default_voice=request.default_voice,
            default_speed=request.default_speed,
            chapter_overrides_json=overrides_json,
        )
    except Exception as exc:
        logger.exception("Failed to create audiobook voice profile")
        raise HTTPException(status_code=500, detail="voice_profile_create_failed") from exc
    return VoiceProfileResponse(
        profile_id=row.profile_id,
        name=row.name,
        default_voice=row.default_voice,
        default_speed=float(row.default_speed),
        chapter_overrides=overrides,
    )


@router.get(
    "/voices/profiles",
    response_model=VoiceProfileListResponse,
    summary="List voice profiles",
    dependencies=[Depends(check_rate_limit)],
)
async def list_voice_profiles(
    limit: int = Query(100, ge=1, le=200, description="Maximum number of voice profiles to return"),
    offset: int = Query(0, ge=0, description="Number of voice profiles to skip"),
    _current_user: User = Depends(get_request_user),
    collections_db: CollectionsDatabase = Depends(get_collections_db_for_user),
) -> VoiceProfileListResponse:
    try:
        rows = collections_db.list_voice_profiles(limit=limit, offset=offset)
        total = collections_db.count_voice_profiles()
    except Exception as exc:
        logger.exception("Failed to list audiobook voice profiles")
        raise HTTPException(status_code=500, detail="voice_profile_list_failed") from exc
    profiles: list[VoiceProfileResponse] = []
    for row in rows:
        overrides = []
        if row.chapter_overrides_json:
            try:
                overrides = json.loads(row.chapter_overrides_json)
            except _AUDIOBOOKS_JSON_EXCEPTIONS:
                overrides = []
        profiles.append(
            VoiceProfileResponse(
                profile_id=row.profile_id,
                name=row.name,
                default_voice=row.default_voice,
                default_speed=float(row.default_speed),
                chapter_overrides=overrides,
            )
        )
    return VoiceProfileListResponse(
        profiles=profiles,
        total=total,
        limit=limit,
        offset=offset,
        pagination=build_offset_pagination_meta(
            total=total,
            limit=limit,
            offset=offset,
            count=len(profiles),
        ),
    )


@router.delete(
    "/voices/profiles/{profile_id}",
    response_model=VoiceProfileDeleteResponse,
    summary="Delete a voice profile",
)
async def delete_voice_profile(
    profile_id: str = Path(..., min_length=1, description="Voice profile id"),
    _current_user: User = Depends(get_request_user),
    collections_db: CollectionsDatabase = Depends(get_collections_db_for_user),
) -> VoiceProfileDeleteResponse:
    try:
        collections_db.delete_voice_profile(profile_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="voice_profile_not_found") from exc
    except Exception as exc:
        logger.exception("Failed to delete audiobook voice profile")
        raise HTTPException(status_code=500, detail="voice_profile_delete_failed") from exc
    return VoiceProfileDeleteResponse(profile_id=profile_id, deleted=True)


@router.post(
    "/subtitles",
    response_class=PlainTextResponse,
    summary="Render subtitles from alignment data",
    dependencies=[Depends(check_rate_limit)],
    responses={
        200: {
            "content": {
                "text/plain": {
                    "example": "1\n00:00:00,000 --> 00:00:00,900\nHello world",
                }
            }
        }
    },
)
async def export_subtitles(
    request: SubtitleExportRequest,
    _current_user: User = Depends(get_request_user),
    collections_db: CollectionsDatabase = Depends(get_collections_db_for_user),
) -> PlainTextResponse:
    user_id = getattr(_current_user, "id", None)
    if user_id is None:
        raise HTTPException(status_code=401, detail="unauthorized")

    alignment_meta: dict[str, object] = {}
    alignment_output_id = request.alignment_output_id
    if alignment_output_id is not None:
        alignment, alignment_meta, _storage_path = _load_alignment_from_output(
            collections_db,
            output_id=int(alignment_output_id),
            user_id=int(user_id),
        )
    else:
        if request.alignment is None:
            raise HTTPException(status_code=400, detail="alignment_required")
        alignment = request.alignment

    persist = _resolve_subtitle_persist(request)
    ttl_hours = _resolve_subtitle_ttl_hours(request)
    cache_key = _subtitle_cache_key(
        alignment=alignment,
        alignment_output_id=alignment_output_id,
        request=request,
    )
    cache_title = _subtitle_cache_title(cache_key)

    outputs_dir = DatabasePaths.get_user_outputs_dir(int(user_id))
    outputs_dir.mkdir(parents=True, exist_ok=True)

    cached_row = None
    if persist:
        try:
            cached_row = collections_db.get_output_artifact_by_title(
                cache_title,
                format_=request.format,
                include_deleted=False,
            )
        except KeyError:
            cached_row = None
        if cached_row is not None:
            cached_path = outputs_dir / cached_row.storage_path
            if cached_path.exists():
                content = cached_path.read_text(encoding="utf-8")
                response = PlainTextResponse(content)
                response.headers["X-Subtitle-Output-Id"] = str(cached_row.id)
                response.headers["X-Subtitle-Download-Url"] = f"/api/v1/outputs/{cached_row.id}/download"
                response.headers["X-Subtitle-Cache-Key"] = cache_key
                response.headers["X-Subtitle-Cache-Hit"] = "1"
                return response
            try:
                collections_db.delete_output_artifact(cached_row.id, hard=True)
            except _AUDIOBOOKS_DB_OPERATION_EXCEPTIONS:
                logger.warning("audiobook subtitles: failed to prune missing cache output")

    try:
        content = generate_subtitles(
            alignment,
            format=request.format,
            mode=request.mode,
            variant=request.variant,
            words_per_cue=request.words_per_cue,
            max_chars=request.max_chars,
            max_lines=request.max_lines,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not persist:
        return PlainTextResponse(content)

    storage_filename = _subtitle_storage_filename(cache_key, request.format)
    subtitle_path = outputs_dir / storage_filename
    subtitle_path.write_text(content, encoding="utf-8")
    size_bytes = subtitle_path.stat().st_size
    retention_until = None
    if ttl_hours is not None:
        retention_until = (datetime.now(timezone.utc) + timedelta(hours=ttl_hours)).isoformat()

    metadata: dict[str, object] = {}
    if alignment_meta:
        metadata.update(alignment_meta)
    if request.metadata:
        metadata.update(request.metadata)
    if request.project_id and "project_id" not in metadata:
        metadata["project_id"] = request.project_id
    if request.chapter_id and "chapter_id" not in metadata:
        metadata["chapter_id"] = request.chapter_id
    if request.chapter_index is not None and "chapter_index" not in metadata:
        metadata["chapter_index"] = request.chapter_index
    if request.item_index is not None and "item_index" not in metadata:
        metadata["item_index"] = request.item_index
    metadata.update(
        {
            "artifact_type": "subtitle",
            "format": request.format,
            "subtitle_mode": request.mode,
            "subtitle_variant": request.variant,
            "alignment_output_id": alignment_output_id,
            "cache_key": cache_key,
            "byte_size": size_bytes,
        }
    )

    row = collections_db.create_output_artifact(
        type_="audiobook_subtitle",
        title=cache_title,
        format_=request.format,
        storage_path=storage_filename,
        metadata_json=json.dumps(metadata),
        retention_until=retention_until,
    )
    try:
        collections_db.update_audiobook_output_usage(size_bytes)
    except _AUDIOBOOKS_DB_OPERATION_EXCEPTIONS:
        logger.warning("audiobook_quota: failed to increment subtitle usage")

    project_id = metadata.get("project_id")
    if project_id:
        try:
            project_row = collections_db.get_audiobook_project_by_project_id(str(project_id))
            collections_db.create_audiobook_artifact(
                project_id=int(project_row.id),
                artifact_type="subtitle",
                format_=request.format,
                output_id=int(row.id),
                metadata_json=json.dumps(metadata),
            )
        except KeyError:
            pass
        except _AUDIOBOOKS_DB_OPERATION_EXCEPTIONS:
            logger.warning("audiobook subtitles: failed to link artifact")

    response = PlainTextResponse(content)
    response.headers["X-Subtitle-Output-Id"] = str(row.id)
    response.headers["X-Subtitle-Download-Url"] = f"/api/v1/outputs/{row.id}/download"
    response.headers["X-Subtitle-Cache-Key"] = cache_key
    response.headers["X-Subtitle-Cache-Hit"] = "0"
    return response
