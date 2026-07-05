from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
import json
import os
import re
import uuid
from typing import Any, cast

from loguru import logger
from pydantic import ValidationError

from tldw_Server_API.app.api.v1.schemas.audio_schemas import OpenAISpeechRequest
from tldw_Server_API.app.api.v1.schemas.chat_request_schemas import DEFAULT_LLM_PROVIDER
from tldw_Server_API.app.api.v1.schemas.research_workspace_outputs import (
    ResearchWorkspaceOutputArtifactType,
    ResearchWorkspaceOutputSettings,
    ResearchWorkspaceOutputStatus,
    ResearchWorkspaceOutputStatusResponse,
    ResearchWorkspaceOutputSubmitRequest,
    ResearchWorkspaceOutputSubmitResponse,
)
from tldw_Server_API.app.api.v1.schemas.workspace_schemas import WorkspaceArtifactResponse
from tldw_Server_API.app.core.exceptions import (
    FileArtifactsError,
    FileArtifactsValidationError,
    file_artifacts_http_status,
)
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.Collections_DB import CollectionsDatabase
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.DB_Management.media_db import api as media_db_api
from tldw_Server_API.app.core.DB_Management.media_db.errors import DatabaseError as MediaDatabaseError
from tldw_Server_API.app.core.File_Artifacts.adapters.image_adapter import ImageAdapter
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Slides.presentation_rendering import PresentationRenderError, render_presentation_video
from tldw_Server_API.app.core.Slides.slides_db import SlidesDatabase
from tldw_Server_API.app.core.Slides.slides_generator import SlidesGenerator
from tldw_Server_API.app.core.TTS.tts_service_v2 import get_tts_service_v2

RESEARCH_WORKSPACE_OUTPUT_JOB_DOMAIN = "research_workspace"
RESEARCH_WORKSPACE_OUTPUT_JOB_QUEUE = "default"
RESEARCH_WORKSPACE_OUTPUT_JOBS_QUEUE_ENV = "RESEARCH_WORKSPACE_OUTPUT_JOBS_QUEUE"
RESEARCH_WORKSPACE_OUTPUT_JOB_TYPE = "research_workspace_output"

_OUTPUT_ARTIFACT_TYPES = {"video_overview", "infographic"}
_PUBLIC_ERROR_CODE_RE = re.compile(r"^[a-z][a-z0-9_]{0,95}$")
_SOURCE_CONTEXT_TOTAL_CHAR_LIMIT = 18_000
_SOURCE_CONTEXT_PER_SOURCE_CHAR_LIMIT = 6_000
_SOURCE_CONTEXT_PREVIEW_CHAR_LIMIT = 1_000
_INFOGRAPHIC_PROMPT_SOURCE_CHAR_LIMIT = 8_000
_VIDEO_OVERVIEW_MAX_SLIDES = 8
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_SOURCE_CONTEXT_MEDIA_ERRORS = (
    AttributeError,
    MediaDatabaseError,
    RuntimeError,
    TypeError,
    ValueError,
    json.JSONDecodeError,
)
_REQUIRED_OUTPUT_METADATA_KEYS = frozenset(
    {"origin", "workspace_id", "workspace_artifact_id", "content_type", "byte_size"}
)
_PATHISH_OUTPUT_METADATA_KEY_TOKENS = frozenset(
    {"path", "file", "filename", "filepath", "folder", "directory", "dir", "storage"}
)
_NON_RETRYABLE_FILE_ARTIFACT_CODES = frozenset(
    {
        "unsupported_file_type",
        "persist_required",
        "unsupported_export_format",
        "invalid_export_mode",
        "invalid_async_mode",
        "export_size_exceeded",
        "row_limit_exceeded",
        "cell_limit_exceeded",
        "storage_quota_exceeded",
        "reference_image_invalid",
        "reference_image_not_found",
        "reference_image_unsupported_by_backend",
        "reference_image_unsupported_by_model",
    }
)
_UNSAFE_OUTPUT_METADATA_VALUE = object()


@dataclass(frozen=True)
class ResearchWorkspaceOutputSourceContext:
    text: str
    source_lineage: dict[str, Any]
    preview_text: str


@dataclass(frozen=True)
class ResearchWorkspacePersistedOutput:
    output_id: int
    download_url: str
    format: str
    content_type: str
    byte_size: int


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
        self.failure_code = self.public_code


def research_workspace_output_jobs_queue() -> str:
    raw = (os.getenv(RESEARCH_WORKSPACE_OUTPUT_JOBS_QUEUE_ENV) or "").strip().lower()
    if raw in {"default", "high", "low"}:
        return raw
    return RESEARCH_WORKSPACE_OUTPUT_JOB_QUEUE


def normalize_research_workspace_output_payload(value: Any) -> dict[str, Any]:
    return dict(_normalize_job_mapping(value))


def build_research_workspace_output_source_context(
    *,
    workspace_db: Any,
    media_db: Any,
    workspace_id: str,
    source_ids: list[str],
    max_chars: int = _SOURCE_CONTEXT_TOTAL_CHAR_LIMIT,
) -> ResearchWorkspaceOutputSourceContext:
    selected_source_ids = [
        str(source_id).strip()
        for source_id in source_ids
        if str(source_id).strip()
    ]
    source_by_id = {
        str(source.get("id") or "").strip(): source
        for source in workspace_db.list_workspace_sources(workspace_id)
    }
    context_char_limit = min(max(int(max_chars), 1), _SOURCE_CONTEXT_TOTAL_CHAR_LIMIT)
    remaining_chars = context_char_limit
    parts: list[str] = []
    usable_source_ids: list[str] = []
    skipped_source_ids: list[str] = []
    media_ids: list[int] = []
    truncated = False

    for source_id in selected_source_ids:
        source = source_by_id.get(source_id)
        media_id = _source_media_id(source)
        if source is None or media_id is None or remaining_chars <= 0:
            skipped_source_ids.append(source_id)
            truncated = truncated or remaining_chars <= 0
            continue

        content = _media_source_text(media_db, media_id)
        if not content:
            skipped_source_ids.append(source_id)
            continue

        content_excerpt = content[:_SOURCE_CONTEXT_PER_SOURCE_CHAR_LIMIT].strip()
        if not content_excerpt:
            skipped_source_ids.append(source_id)
            continue

        title = str(source.get("title") or source_id).strip() or source_id
        separator_chars = 2 if parts else 0
        available_chars = remaining_chars - separator_chars
        if available_chars <= 0:
            skipped_source_ids.append(source_id)
            truncated = True
            continue

        if available_chars < len("# a\n\nb"):
            skipped_source_ids.append(source_id)
            truncated = True
            continue

        body_budget = min(len(content_excerpt), max(1, available_chars // 2))
        title_budget = available_chars - len("# \n\n") - body_budget
        if title_budget < 1:
            title_budget = 1
            body_budget = available_chars - len("# \n\n") - title_budget
        if body_budget < 1:
            skipped_source_ids.append(source_id)
            truncated = True
            continue

        title_excerpt = title[:title_budget].rstrip() or title[:1]
        body_excerpt = content_excerpt[:body_budget].rstrip() or content_excerpt[:1]
        part = f"# {title_excerpt}\n\n{body_excerpt}"
        parts.append(part)
        usable_source_ids.append(source_id)
        media_ids.append(media_id)
        remaining_chars -= separator_chars + len(part)
        truncated = (
            truncated
            or len(title) > len(title_excerpt)
            or len(content_excerpt) > len(body_excerpt)
            or len(content) > len(content_excerpt)
        )

    text = "\n\n".join(parts).strip()
    if not text:
        raise ResearchWorkspaceOutputJobError("source_context_empty", retryable=False)

    return ResearchWorkspaceOutputSourceContext(
        text=text,
        preview_text=text[:_SOURCE_CONTEXT_PREVIEW_CHAR_LIMIT].strip(),
        source_lineage={
            "selected_source_ids": selected_source_ids,
            "usable_source_ids": usable_source_ids,
            "skipped_source_ids": skipped_source_ids,
            "media_ids": media_ids,
            "context_char_limit": context_char_limit,
            "context_truncated": truncated,
        },
    )


def persist_research_workspace_output_bytes(
    *,
    collections_db: CollectionsDatabase,
    user_id: int,
    job_id: int,
    artifact_type: str,
    title: str,
    content: bytes,
    format_: str,
    content_type: str,
    workspace_id: str,
    workspace_artifact_id: str,
    metadata: Mapping[str, Any] | None = None,
) -> ResearchWorkspacePersistedOutput:
    if not content:
        raise ResearchWorkspaceOutputJobError("empty_output", retryable=False)

    try:
        if int(getattr(collections_db, "user_id")) != int(user_id):
            raise ResearchWorkspaceOutputJobError("output_user_mismatch", retryable=False)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ResearchWorkspaceOutputJobError("output_user_mismatch", retryable=False) from exc

    outputs_dir = DatabasePaths.get_user_outputs_dir(user_id)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    filename = f"research-workspace-{workspace_artifact_id}-{uuid.uuid4().hex}.{format_}"
    storage_path = collections_db.resolve_output_storage_path(filename)
    path = outputs_dir / storage_path
    path.write_bytes(content)

    try:
        row = collections_db.create_output_artifact(
            job_id=job_id,
            type_=f"research_workspace_{artifact_type}",
            title=title,
            format_=format_,
            storage_path=storage_path,
            workspace_tag=f"workspace:{workspace_id}",
            metadata_json=json.dumps(
                {
                    **_safe_caller_output_metadata(metadata),
                    "origin": "research_workspace",
                    "workspace_id": workspace_id,
                    "workspace_artifact_id": workspace_artifact_id,
                    "content_type": content_type,
                    "byte_size": len(content),
                },
                ensure_ascii=False,
            ),
        )
    except Exception as exc:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
        raise ResearchWorkspaceOutputJobError(
            "output_artifact_create_failed",
            retryable=False,
        ) from exc
    return ResearchWorkspacePersistedOutput(
        output_id=int(row.id),
        download_url=f"/api/v1/outputs/{row.id}/download",
        format=format_,
        content_type=content_type,
        byte_size=len(content),
    )


async def process_research_workspace_output_payload(
    *,
    job: dict[str, Any],
    payload: dict[str, Any],
    workspace_db: Any,
    media_db: Any,
    user_id: int,
    job_manager: JobManager,
    progress: Any | None = None,
) -> dict[str, Any]:
    normalized_payload = normalize_research_workspace_output_payload(payload)
    workspace_id = str(normalized_payload.get("workspace_id") or "").strip()
    artifact_id = str(normalized_payload.get("artifact_id") or "").strip()
    artifact_type = str(normalized_payload.get("artifact_type") or "").strip()
    if not workspace_id or not artifact_id or artifact_type not in _OUTPUT_ARTIFACT_TYPES:
        raise ResearchWorkspaceOutputJobError("job_payload_invalid", status_code=400)
    if str(normalized_payload.get("user_id") or user_id) != str(user_id):
        raise ResearchWorkspaceOutputJobError("owner_user_id_mismatch", status_code=403)
    try:
        settings = ResearchWorkspaceOutputSettings.model_validate(
            _normalize_job_mapping(normalized_payload.get("settings"))
        )
    except ValidationError as exc:
        raise ResearchWorkspaceOutputJobError("output_settings_invalid", status_code=400) from exc

    job_id = _validated_research_workspace_output_job_id(job)
    if artifact_type == "video_overview":
        return await _process_video_overview_output_payload(
            job=job,
            payload=normalized_payload,
            workspace_db=workspace_db,
            media_db=media_db,
            user_id=user_id,
            settings=settings,
            job_id=job_id,
            job_manager=job_manager,
            progress=progress,
        )
    return await _process_infographic_output_payload(
        job=job,
        payload=normalized_payload,
        workspace_db=workspace_db,
        media_db=media_db,
        user_id=user_id,
        settings=settings,
        job_id=job_id,
    )


def generate_infographic_prompt(
    *,
    source_context: ResearchWorkspaceOutputSourceContext,
    settings: ResearchWorkspaceOutputSettings,
) -> str:
    title = f"Title: {settings.title_hint.strip()}\n" if settings.title_hint else ""
    source_text = source_context.text[:_INFOGRAPHIC_PROMPT_SOURCE_CHAR_LIMIT].strip()
    return (
        "Create one clean, information-dense infographic image. "
        "Use only facts from the source context. "
        "Prefer labeled sections, concrete numbers, timelines, and comparisons where relevant. "
        "Do not invent facts.\n\n"
        f"{title}Source context:\n{source_text}"
    ).strip()


async def _process_infographic_output_payload(
    *,
    job: dict[str, Any],
    payload: dict[str, Any],
    workspace_db: Any,
    media_db: Any,
    user_id: int,
    settings: ResearchWorkspaceOutputSettings,
    job_id: int,
) -> dict[str, Any]:
    workspace_id = str(payload["workspace_id"])
    artifact_id = str(payload["artifact_id"])
    source_ids = [
        str(source_id).strip()
        for source_id in payload.get("source_ids") or []
        if str(source_id).strip()
    ]
    title = settings.title_hint or "Infographic"

    try:
        source_context = build_research_workspace_output_source_context(
            workspace_db=workspace_db,
            media_db=media_db,
            workspace_id=workspace_id,
            source_ids=source_ids,
        )
        prompt = generate_infographic_prompt(source_context=source_context, settings=settings)
        adapter = ImageAdapter()
        structured = adapter.normalize(
            {
                "backend": settings.image_backend,
                "prompt": prompt,
                "width": settings.image_width or 1024,
                "height": settings.image_height or 1024,
                "model": settings.model,
            }
        )
        issues = adapter.validate(structured)
        if issues:
            raise ResearchWorkspaceOutputJobError(str(issues[0].code), retryable=False)
        export = await asyncio.to_thread(adapter.export, structured, format="png")
        content = getattr(export, "content", None)
        if not isinstance(content, bytes) or not content:
            raise ResearchWorkspaceOutputJobError("empty_output", retryable=False)
        if not _is_png_export_content(content, getattr(export, "content_type", None)):
            raise ResearchWorkspaceOutputJobError("image_content_type_invalid", retryable=False)
        with CollectionsDatabase.for_user(user_id=str(user_id)) as collections_db:
            persisted = persist_research_workspace_output_bytes(
                collections_db=collections_db,
                user_id=user_id,
                job_id=job_id,
                artifact_type="infographic",
                title=title,
                content=content,
                format_="png",
                content_type="image/png",
                workspace_id=workspace_id,
                workspace_artifact_id=artifact_id,
                metadata={"image_backend": structured.get("backend")},
            )
    except ResearchWorkspaceOutputJobError as exc:
        _try_mark_workspace_output_artifact_failed(
            workspace_db=workspace_db,
            workspace_id=workspace_id,
            artifact_id=artifact_id,
            artifact_type="infographic",
            job_id=job_id,
            error_code=exc.public_code,
        )
        raise
    except FileArtifactsError as exc:
        job_error = _research_workspace_output_error_from_file_artifacts_error(exc)
        _try_mark_workspace_output_artifact_failed(
            workspace_db=workspace_db,
            workspace_id=workspace_id,
            artifact_id=artifact_id,
            artifact_type="infographic",
            job_id=job_id,
            error_code=job_error.public_code,
        )
        raise job_error from exc
    except Exception as exc:
        public_code = "infographic_generation_failed"
        _try_mark_workspace_output_artifact_failed(
            workspace_db=workspace_db,
            workspace_id=workspace_id,
            artifact_id=artifact_id,
            artifact_type="infographic",
            job_id=job_id,
            error_code=public_code,
        )
        raise ResearchWorkspaceOutputJobError(public_code, retryable=False) from exc

    export_ref = {
        "id": persisted.output_id,
        "fileId": persisted.output_id,
        "format": persisted.format,
        "url": persisted.download_url,
        "status": "ready",
        "content_type": persisted.content_type,
        "bytes": persisted.byte_size,
    }
    _update_workspace_output_artifact(
        workspace_db=workspace_db,
        workspace_id=workspace_id,
        artifact_id=artifact_id,
        updates={
            "status": "complete",
            "content_type": "image/png",
            "preview_text": source_context.preview_text,
            "summary": "Generated infographic",
            "source_lineage": source_context.source_lineage,
            "producer_metadata": {
                "origin": "research_workspace_output_job",
                "job_id": job_id,
                "artifact_type": "infographic",
                "backend": structured.get("backend"),
            },
            "export_refs": [export_ref],
            "completed_at": _utc_now_iso(),
        },
    )
    return {
        "workspace_id": workspace_id,
        "artifact_id": artifact_id,
        "artifact_type": "infographic",
        "output_id": persisted.output_id,
        "download_url": persisted.download_url,
        "format": persisted.format,
        "content_type": persisted.content_type,
        "bytes": persisted.byte_size,
    }


async def _process_video_overview_output_payload(
    *,
    job: dict[str, Any],
    payload: dict[str, Any],
    workspace_db: Any,
    media_db: Any,
    user_id: int,
    settings: ResearchWorkspaceOutputSettings,
    job_id: int,
    job_manager: JobManager,
    progress: Any | None = None,
) -> dict[str, Any]:
    workspace_id = str(payload["workspace_id"])
    artifact_id = str(payload["artifact_id"])
    source_ids = [
        str(source_id).strip()
        for source_id in payload.get("source_ids") or []
        if str(source_id).strip()
    ]
    tts_model = settings.tts_model or "kokoro"
    tts_voice = settings.tts_voice or "af_heart"
    title = settings.title_hint or "Video Overview"

    try:
        _update_output_job_progress(job_manager, job_id, progress, 15.0, "build_source_context")
        source_context = build_research_workspace_output_source_context(
            workspace_db=workspace_db,
            media_db=media_db,
            workspace_id=workspace_id,
            source_ids=source_ids,
        )
        _update_output_job_progress(job_manager, job_id, progress, 30.0, "generate_slides")
        generated = SlidesGenerator().generate_from_text(
            source_text=source_context.text,
            title_hint=settings.title_hint,
            provider=settings.provider or DEFAULT_LLM_PROVIDER,
            model=settings.model,
            api_key=None,
            temperature=None,
            max_tokens=None,
            max_source_tokens=None,
            max_source_chars=None,
            enable_chunking=True,
            chunk_size_tokens=None,
            summary_tokens=None,
            visual_style_snapshot=None,
        )
        title = str(generated.get("title") or title).strip() or title
        slides = _normalize_video_overview_slides(generated.get("slides"))

        with CollectionsDatabase.for_user(user_id=str(user_id)) as collections_db:
            _update_output_job_progress(job_manager, job_id, progress, 50.0, "generate_narration")
            for slide in slides:
                audio_bytes = await generate_tts_audio_bytes(
                    text=_slide_narration_text(slide),
                    provider=settings.tts_provider,
                    model=tts_model,
                    voice=tts_voice,
                    user_id=user_id,
                )
                audio_output = persist_research_workspace_output_bytes(
                    collections_db=collections_db,
                    user_id=user_id,
                    job_id=job_id,
                    artifact_type="video_overview_narration",
                    title=f"{title} narration {int(slide['order']) + 1}",
                    content=audio_bytes,
                    format_="mp3",
                    content_type="audio/mpeg",
                    workspace_id=workspace_id,
                    workspace_artifact_id=artifact_id,
                    metadata={"output_format": "mp3", "content_type": "audio/mpeg"},
                )
                slide.setdefault("metadata", {}).setdefault("studio", {}).setdefault("audio", {})[
                    "asset_ref"
                ] = f"output:{audio_output.output_id}"

            slides_db = SlidesDatabase(
                db_path=str(DatabasePaths.get_slides_db_path(user_id)),
                client_id=str(user_id),
            )
            try:
                presentation = slides_db.create_presentation(
                    presentation_id=None,
                    title=title,
                    description=None,
                    theme="black",
                    marp_theme=None,
                    template_id=None,
                    visual_style_id=None,
                    visual_style_scope=None,
                    visual_style_name=None,
                    visual_style_version=None,
                    visual_style_snapshot=None,
                    settings=json.dumps({"origin": "research_workspace"}, ensure_ascii=False),
                    studio_data=None,
                    slides=json.dumps(slides, ensure_ascii=False),
                    slides_text=_video_overview_slides_text(slides),
                    source_type="research_workspace",
                    source_ref=json.dumps(
                        {"workspace_id": workspace_id, "artifact_id": artifact_id},
                        ensure_ascii=False,
                    ),
                    source_query=None,
                    custom_css=None,
                )
            finally:
                close_slides_db = getattr(slides_db, "close_connection", None)
                if callable(close_slides_db):
                    close_slides_db()

            _update_output_job_progress(job_manager, job_id, progress, 75.0, "render_video")
            render_result = await asyncio.to_thread(
                render_presentation_video,
                presentation_id=str(presentation.id),
                presentation_version=int(presentation.version),
                title=str(presentation.title),
                slides=slides,
                output_format="mp4",
                output_dir=DatabasePaths.get_user_outputs_dir(user_id),
                collections_db=collections_db,
                user_id=user_id,
            )
            final_artifact = collections_db.create_output_artifact(
                job_id=job_id,
                type_="research_workspace_video_overview",
                title=title,
                format_="mp4",
                storage_path=render_result.storage_path,
                workspace_tag=f"workspace:{workspace_id}",
                metadata_json=json.dumps(
                    {
                        "origin": "research_workspace",
                        "workspace_id": workspace_id,
                        "workspace_artifact_id": artifact_id,
                        "presentation_id": str(presentation.id),
                        "presentation_version": int(presentation.version),
                        "slide_count": len(slides),
                        "content_type": "video/mp4",
                        "byte_size": int(render_result.byte_size),
                        "tts_provider": settings.tts_provider,
                        "tts_model": tts_model,
                        "tts_voice": tts_voice,
                        "output_format": render_result.output_format,
                    },
                    ensure_ascii=False,
                ),
            )
    except ResearchWorkspaceOutputJobError as exc:
        _try_mark_workspace_output_artifact_failed(
            workspace_db=workspace_db,
            workspace_id=workspace_id,
            artifact_id=artifact_id,
            artifact_type="video_overview",
            job_id=job_id,
            error_code=exc.public_code,
        )
        raise
    except PresentationRenderError as exc:
        job_error = ResearchWorkspaceOutputJobError(
            str(exc.code),
            retryable=getattr(exc, "retryable", False),
        )
        _try_mark_workspace_output_artifact_failed(
            workspace_db=workspace_db,
            workspace_id=workspace_id,
            artifact_id=artifact_id,
            artifact_type="video_overview",
            job_id=job_id,
            error_code=job_error.public_code,
        )
        raise job_error from exc
    except Exception as exc:
        public_code = "video_overview_generation_failed"
        _try_mark_workspace_output_artifact_failed(
            workspace_db=workspace_db,
            workspace_id=workspace_id,
            artifact_id=artifact_id,
            artifact_type="video_overview",
            job_id=job_id,
            error_code=public_code,
        )
        raise ResearchWorkspaceOutputJobError(public_code, retryable=False) from exc

    persisted = ResearchWorkspacePersistedOutput(
        output_id=int(final_artifact.id),
        download_url=f"/api/v1/outputs/{final_artifact.id}/download",
        format=str(render_result.output_format),
        content_type="video/mp4",
        byte_size=int(render_result.byte_size),
    )
    export_ref = {
        "id": persisted.output_id,
        "fileId": persisted.output_id,
        "format": persisted.format,
        "url": persisted.download_url,
        "status": "ready",
        "content_type": persisted.content_type,
        "bytes": persisted.byte_size,
    }
    _update_workspace_output_artifact(
        workspace_db=workspace_db,
        workspace_id=workspace_id,
        artifact_id=artifact_id,
        updates={
            "status": "complete",
            "content_type": "video/mp4",
            "preview_text": source_context.preview_text,
            "summary": "Generated video overview",
            "source_lineage": source_context.source_lineage,
            "producer_metadata": {
                "origin": "research_workspace_output_job",
                "job_id": job_id,
                "artifact_type": "video_overview",
                "presentation_id": str(presentation.id),
                "presentation_version": int(presentation.version),
                "slide_count": len(slides),
                "tts_provider": settings.tts_provider,
                "tts_model": tts_model,
                "tts_voice": tts_voice,
            },
            "export_refs": [export_ref],
            "completed_at": _utc_now_iso(),
        },
    )
    return {
        "workspace_id": workspace_id,
        "artifact_id": artifact_id,
        "artifact_type": "video_overview",
        "output_id": persisted.output_id,
        "download_url": persisted.download_url,
        "format": persisted.format,
        "content_type": persisted.content_type,
        "bytes": persisted.byte_size,
    }


async def generate_tts_audio_bytes(
    *,
    text: str,
    provider: str | None,
    model: str,
    voice: str,
    user_id: int,
) -> bytes:
    request = OpenAISpeechRequest(
        model=model,
        input=text,
        voice=voice,
        response_format="mp3",
        speed=1.0,
        stream=False,
    )
    tts_service = await get_tts_service_v2()
    chunks = b""
    async for chunk in tts_service.generate_speech(
        request,
        provider=provider,
        fallback=True,
        user_id=user_id,
    ):
        chunks += chunk
    if not chunks:
        raise ResearchWorkspaceOutputJobError("empty_output", retryable=False)
    return chunks


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


def _source_media_id(source: Mapping[str, Any] | None) -> int | None:
    if source is None:
        return None
    try:
        return int(source.get("media_id")) if source.get("media_id") is not None else None
    except (TypeError, ValueError):
        return None


def _media_source_text(media_db: Any, media_id: int) -> str:
    try:
        media = media_db_api.get_media_by_id(media_db, media_id)
    except _SOURCE_CONTEXT_MEDIA_ERRORS:
        media = None
    if not media:
        return ""

    text = _content_text((media or {}).get("content"))
    if text:
        return text
    return ""


def _content_text(value: Any) -> str:
    if isinstance(value, Mapping):
        return str(value.get("content") or value.get("text") or "").strip()
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("{") or text.startswith("["):
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                return text
            if isinstance(parsed, Mapping):
                return str(parsed.get("content") or parsed.get("text") or text).strip()
        return text
    return ""


def _safe_caller_output_metadata(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key, value in dict(metadata or {}).items():
        key_text = str(key)
        if _is_unsafe_output_metadata_key(key_text):
            continue
        sanitized = _sanitize_output_metadata_value(value)
        if sanitized is _UNSAFE_OUTPUT_METADATA_VALUE:
            continue
        safe[key_text] = sanitized
    return safe


def _sanitize_output_metadata_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        safe: dict[str, Any] = {}
        for key, child in value.items():
            key_text = str(key)
            if _is_unsafe_output_metadata_key(key_text):
                continue
            sanitized = _sanitize_output_metadata_value(child)
            if sanitized is not _UNSAFE_OUTPUT_METADATA_VALUE:
                safe[key_text] = sanitized
        return safe
    if isinstance(value, list | tuple):
        safe_items = [
            sanitized
            for item in value
            if (sanitized := _sanitize_output_metadata_value(item)) is not _UNSAFE_OUTPUT_METADATA_VALUE
            and sanitized not in ({}, [])
        ]
        return safe_items
    if isinstance(value, str):
        text = value.strip()
        if text.startswith(("{", "[")):
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                pass
            else:
                if isinstance(parsed, Mapping | list):
                    sanitized = _sanitize_output_metadata_value(parsed)
                    if sanitized is _UNSAFE_OUTPUT_METADATA_VALUE or sanitized in ({}, []):
                        return _UNSAFE_OUTPUT_METADATA_VALUE
                    return sanitized
        if _looks_fileish_or_pathish_text(value):
            return _UNSAFE_OUTPUT_METADATA_VALUE
    return value


def _is_unsafe_output_metadata_key(key_text: str) -> bool:
    key_lower = key_text.lower()
    if (
        key_lower in _REQUIRED_OUTPUT_METADATA_KEYS
        or "path" in key_lower
        or _looks_absolute_path_like(key_text)
    ):
        return True
    key_tokens = re.split(r"[^a-z0-9]+", key_lower)
    return any(token in _PATHISH_OUTPUT_METADATA_KEY_TOKENS for token in key_tokens)


def _looks_fileish_or_pathish_text(value: str) -> bool:
    text = value.strip()
    return (
        _looks_absolute_path_like(text)
        or re.search(
            r"(^|[^\w])(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+(?:\.[A-Za-z0-9]{1,12})?(?=$|[^\w])",
            text,
        )
        is not None
        or re.search(r"(?i)(^|[^\w])[A-Za-z0-9_.-]+\.[a-z][a-z0-9]{0,11}(?=$|[^\w])", text)
        is not None
    )


def _looks_absolute_path_like(value: str) -> bool:
    text = value.strip()
    return (
        re.search(r"(^|[^\w])(?:[/\\](?:\S|$)|~[/\\](?:\S|$)|[A-Za-z]:[/\\])", text)
        is not None
    )


def _validated_research_workspace_output_job_id(job: Mapping[str, Any]) -> int:
    try:
        job_id = int(job.get("id"))
    except (TypeError, ValueError) as exc:
        raise ResearchWorkspaceOutputJobError("invalid_job_id", status_code=400, retryable=False) from exc
    if job_id <= 0:
        raise ResearchWorkspaceOutputJobError("invalid_job_id", status_code=400, retryable=False)
    return job_id


def _research_workspace_output_error_from_file_artifacts_error(exc: FileArtifactsError) -> ResearchWorkspaceOutputJobError:
    retryable = not isinstance(exc, FileArtifactsValidationError) and exc.code not in _NON_RETRYABLE_FILE_ARTIFACT_CODES
    return ResearchWorkspaceOutputJobError(
        str(exc.code),
        status_code=file_artifacts_http_status(exc),
        retryable=retryable,
    )


def _is_png_export_content(content: bytes, content_type: Any) -> bool:
    normalized_content_type = str(content_type or "").split(";", 1)[0].strip().lower()
    if normalized_content_type and normalized_content_type != "image/png":
        return False
    return content.startswith(_PNG_SIGNATURE)


def _normalize_video_overview_slides(value: Any) -> list[dict[str, Any]]:
    raw_slides = value if isinstance(value, list) else []
    slides: list[dict[str, Any]] = []
    for index, item in enumerate(raw_slides[:_VIDEO_OVERVIEW_MAX_SLIDES]):
        slide = dict(item) if isinstance(item, Mapping) else {}
        metadata = slide.get("metadata") if isinstance(slide.get("metadata"), dict) else {}
        slides.append(
            {
                "order": index,
                "layout": str(slide.get("layout") or "content"),
                "title": str(slide.get("title") or f"Slide {index + 1}").strip(),
                "content": str(slide.get("content") or "").strip(),
                "speaker_notes": str(slide.get("speaker_notes") or "").strip(),
                "metadata": dict(metadata),
            }
        )
    if not slides:
        raise ResearchWorkspaceOutputJobError("slides_generation_empty", retryable=False)
    return slides


def _slide_narration_text(slide: Mapping[str, Any]) -> str:
    text = str(slide.get("speaker_notes") or slide.get("content") or slide.get("title") or "").strip()
    if not text:
        raise ResearchWorkspaceOutputJobError("slide_narration_empty", retryable=False)
    return text


def _video_overview_slides_text(slides: list[Mapping[str, Any]]) -> str:
    parts: list[str] = []
    for slide in slides:
        parts.extend(
            str(slide.get(key) or "").strip()
            for key in ("title", "content", "speaker_notes")
            if str(slide.get(key) or "").strip()
        )
    return "\n".join(parts)


def _update_output_job_progress(
    job_manager: Any,
    job_id: int,
    progress: Any | None,
    percent: float,
    message: str,
) -> None:
    if progress is not None:
        progress.percent = percent
        progress.message = message
    update = getattr(job_manager, "update_job_progress", None)
    if callable(update):
        update(job_id, progress_percent=percent, progress_message=message)


def _update_workspace_output_artifact(
    *,
    workspace_db: Any,
    workspace_id: str,
    artifact_id: str,
    updates: dict[str, Any],
) -> dict[str, Any]:
    existing = workspace_db.get_workspace_artifact(workspace_id, artifact_id)
    if existing is None:
        raise ResearchWorkspaceOutputJobError("workspace_artifact_not_found", status_code=404)
    try:
        expected_version = int(existing.get("version"))
    except (AttributeError, TypeError, ValueError) as exc:
        raise ResearchWorkspaceOutputJobError("workspace_artifact_version_invalid", status_code=409) from exc
    if expected_version < 1:
        raise ResearchWorkspaceOutputJobError("workspace_artifact_version_invalid", status_code=409)
    try:
        return workspace_db.update_workspace_artifact(
            workspace_id,
            artifact_id,
            updates,
            expected_version=expected_version,
        )
    except ResearchWorkspaceOutputJobError:
        raise
    except Exception as exc:
        raise ResearchWorkspaceOutputJobError("workspace_artifact_update_failed", status_code=409) from exc


def _try_mark_workspace_output_artifact_failed(
    *,
    workspace_db: Any,
    workspace_id: str,
    artifact_id: str,
    artifact_type: str,
    job_id: int,
    error_code: str,
) -> None:
    try:
        _mark_workspace_output_artifact_failed(
            workspace_db=workspace_db,
            workspace_id=workspace_id,
            artifact_id=artifact_id,
            artifact_type=artifact_type,
            job_id=job_id,
            error_code=error_code,
        )
    except Exception:
        logger.warning(
            "Failed to mark research workspace output artifact failed: workspace_id={} artifact_id={} job_id={} error_code={}",
            workspace_id,
            artifact_id,
            job_id,
            _safe_public_error_code(error_code),
        )


def _mark_workspace_output_artifact_failed(
    *,
    workspace_db: Any,
    workspace_id: str,
    artifact_id: str,
    artifact_type: str,
    job_id: int,
    error_code: str,
) -> None:
    public_code = _safe_public_error_code(error_code)
    _update_workspace_output_artifact(
        workspace_db=workspace_db,
        workspace_id=workspace_id,
        artifact_id=artifact_id,
        updates={
            "status": "failed",
            "content_type": "image/png" if artifact_type == "infographic" else "video/mp4",
            "producer_metadata": {
                "origin": "research_workspace_output_job",
                "job_id": job_id,
                "artifact_type": artifact_type,
                "error": public_code,
            },
        },
    )


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


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
