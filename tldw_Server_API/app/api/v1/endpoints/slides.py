"""Slides/Presentations API endpoints."""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
import os
import re
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request, Response, status
from fastapi.encoders import jsonable_encoder
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.routing import APIRoute
from loguru import logger
from pydantic import ValidationError
from starlette.responses import JSONResponse

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import RequirePermission, User, get_request_user, rbac_rate_limit
from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user, get_chacha_db_for_user_id
from tldw_Server_API.app.api.v1.API_Deps.Collections_DB_Deps import get_collections_db_for_user
from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user
from tldw_Server_API.app.api.v1.API_Deps.Slides_DB_Deps import get_slides_db_for_user
from tldw_Server_API.app.api.v1.endpoints._pagination_utils import build_offset_pagination_meta
from tldw_Server_API.app.api.v1.schemas.chat_request_schemas import DEFAULT_LLM_PROVIDER
from tldw_Server_API.app.api.v1.schemas.slides_schemas import (
    ExportFormat,
    GenerateFromChatRequest,
    GenerateFromMediaRequest,
    GenerateFromNotesRequest,
    GenerateFromPromptRequest,
    GenerateFromRagRequest,
    PresentationCreateRequest,
    PresentationDeleteResponse,
    PresentationListResponse,
    PresentationPatchRequest,
    PresentationRenderArtifactInfo,
    PresentationRenderArtifactListResponse,
    PresentationRenderFormat,
    PresentationRenderJobResponse,
    PresentationRenderJobStatusResponse,
    PresentationRenderRequest,
    PresentationReorderRequest,
    PresentationResponse,
    PresentationSearchResponse,
    PresentationUpdateRequest,
    PresentationVersionListResponse,
    Slide,
    SlidesHealthResponse,
    SlidesTemplateListResponse,
    SlidesTemplateResponse,
    StructuredPresentationResponse,
    VisualStyleCreateRequest,
    VisualStyleListResponse,
    VisualStylePatchRequest,
    VisualStyleResponse,
)
from tldw_Server_API.app.api.v1.utils.http_errors import map_db_error_to_http
from tldw_Server_API.app.core.AuthNZ.permissions import MEDIA_CREATE, MEDIA_DELETE, MEDIA_READ, MEDIA_UPDATE
from tldw_Server_API.app.core.Claims_Extraction.artifact_verification import (
    ArtifactVerificationResult,
    ArtifactVerificationUnit,
    verify_generated_artifact_against_sources,
)
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.Collections_DB import CollectionsDatabase
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.DB_Management.media_db.api import (
    MediaDbSession,
    get_document_version,
    get_latest_transcription,
)
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Metrics.metrics_manager import get_metrics_registry
from tldw_Server_API.app.core.RAG.rag_service.types import Document
from tldw_Server_API.app.core.RAG.rag_service.unified_pipeline import unified_rag_pipeline
from tldw_Server_API.app.core.Slides.presentation_service import (
    CONTENT_KIND_HEADER,
    STANDALONE_HTML,
    PresentationService,
    PresentationServiceError,
    merge_vary_header,
    parse_accepted_content_kinds,
    presentation_detail,
    presentation_summary,
    snapshot_detail,
)
from tldw_Server_API.app.core.Slides.slides_assets import (
    MAX_RESOLVED_SLIDE_ASSET_BYTES,
    resolve_slide_asset,
)
from tldw_Server_API.app.core.Slides.slides_db import (
    ConflictError,
    InputError,
    SlidesDatabase,
    SlidesDatabaseError,
    VisualStyleRow,
    decode_presentation_version_payload,
)
from tldw_Server_API.app.core.Slides.slides_export import (
    SlidesAssetsMissingError,
    SlidesExportError,
    SlidesExportInputError,
    export_presentation_bundle,
    export_presentation_json,
    export_presentation_markdown,
    export_presentation_pdf,
)
from tldw_Server_API.app.core.Slides.slides_generator import (
    SlidesGenerationError,
    SlidesGenerationInputError,
    SlidesGenerationOutputError,
    SlidesGenerator,
    SlidesSourceTooLargeError,
)
from tldw_Server_API.app.core.Slides.slides_images import (
    SlidesImageError,
    collect_image_alt_text,
    validate_images_payload,
)
from tldw_Server_API.app.core.Slides.slides_templates import (
    SlidesTemplate,
    SlidesTemplateInvalidError,
    SlidesTemplateNotFoundError,
    get_slide_template,
    list_slide_templates,
)
from tldw_Server_API.app.core.Slides.standalone_html_contracts import (
    StandaloneHtmlValidationError,
)
from tldw_Server_API.app.core.Slides.standalone_html_validation_pool import (
    StandaloneHtmlValidationPool,
)
from tldw_Server_API.app.core.Slides.visual_style_resolver import (
    ResolvedBuiltinVisualStyle,
    resolve_builtin_visual_style,
)
from tldw_Server_API.app.core.Slides.visual_styles import (
    get_builtin_visual_style,
    list_builtin_visual_styles,
)
from tldw_Server_API.app.core.testing import is_test_mode, is_truthy

_VALIDATION_POOL_ATTR = "standalone_html_validation_pool"
_VALIDATION_POOL_LOCK_ATTR = "standalone_html_validation_pool_lock"


class _SlidesRoute(APIRoute):
    """Preserve content-kind cache variation through downstream HTTP errors."""

    def get_route_handler(self) -> Callable[[Request], Awaitable[Response]]:
        original = super().get_route_handler()
        negotiated = any(
            str(field.alias).lower() == CONTENT_KIND_HEADER.lower() for field in self.dependant.header_params
        )

        async def route_handler(request: Request) -> Response:
            try:
                response = await original(request)
            except RequestValidationError as exc:
                response = await request_validation_exception_handler(request, exc)
                if negotiated:
                    merge_vary_header(response.headers)
                return response
            except HTTPException as exc:
                headers = dict(exc.headers or {})
                if negotiated:
                    merge_vary_header(headers)
                detail = exc.detail
                if (
                    isinstance(detail, dict)
                    and detail.get("code") == "operation_not_supported_for_content_kind"
                    and set(detail) == {"code", "operation", "content_kind"}
                ):
                    return JSONResponse(
                        status_code=exc.status_code,
                        content={
                            "detail": detail["code"],
                            "operation": detail["operation"],
                            "content_kind": detail["content_kind"],
                        },
                        headers=headers,
                    )
                exc.headers = headers or None
                raise
            if negotiated:
                merge_vary_header(response.headers)
            return response

        return route_handler


@contextlib.asynccontextmanager
async def _slides_lifespan(app: Any) -> AsyncIterator[None]:
    """Own the single validator pool shared by Slides requests and workers."""
    if getattr(app.state, _VALIDATION_POOL_LOCK_ATTR, None) is None:
        setattr(app.state, _VALIDATION_POOL_LOCK_ATTR, asyncio.Lock())
    try:
        yield
    finally:
        pool = getattr(app.state, _VALIDATION_POOL_ATTR, None)
        try:
            if pool is not None:
                await pool.close()
        finally:
            with contextlib.suppress(AttributeError, KeyError):
                delattr(app.state, _VALIDATION_POOL_ATTR)
            with contextlib.suppress(AttributeError, KeyError):
                delattr(app.state, _VALIDATION_POOL_LOCK_ATTR)


router = APIRouter(
    prefix="/slides",
    tags=["slides"],
    lifespan=_slides_lifespan,
    route_class=_SlidesRoute,
)

_ALLOWED_THEMES = {
    "black",
    "white",
    "league",
    "beige",
    "sky",
    "night",
    "serif",
    "simple",
    "solarized",
    "blood",
    "moon",
    "dracula",
}

_SETTINGS_ALLOWLIST: dict[str, tuple[type, ...]] = {
    "transition": (str,),
    "backgroundTransition": (str,),
    "slideNumber": (bool,),
    "controls": (bool,),
    "progress": (bool,),
    "hash": (bool,),
    "center": (bool,),
    "width": (int, float),
    "height": (int, float),
    "margin": (int, float),
    "minScale": (int, float),
    "maxScale": (int, float),
    "viewDistance": (int, float),
    "keyboard": (bool,),
    "touch": (bool,),
    "loop": (bool,),
    "rtl": (bool,),
    "navigationMode": (str,),
}

_SETTINGS_STRING_ENUMS: dict[str, set[str]] = {
    "transition": {"none", "fade", "slide", "convex", "concave", "zoom"},
    "backgroundTransition": {"none", "fade", "slide", "convex", "concave", "zoom"},
    "navigationMode": {"default", "linear", "grid"},
}

_ETAG_RE = re.compile(r'^(W/)?"v(?P<version>\d+)"$')
_MARP_THEME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

_SLIDES_NONCRITICAL_EXCEPTIONS = (
    AssertionError,
    AttributeError,
    ConnectionError,
    ImportError,
    KeyError,
    LookupError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
    json.JSONDecodeError,
)
_SLIDES_HEALTH_EXCEPTIONS = (*_SLIDES_NONCRITICAL_EXCEPTIONS, SlidesDatabaseError)

_PRESENTATION_STUDIO_TRANSITIONS = {"fade", "cut", "wipe", "zoom"}
_PRESENTATION_STUDIO_TIMING_MODES = {"auto", "manual"}


@dataclass(frozen=True)
class PresentationVisualStyleApplication:
    """Resolved visual-style application data for presentation writes."""

    style_id: str
    scope: str
    name: str
    version: int | None
    snapshot: dict[str, Any]
    appearance_defaults: dict[str, Any]


def _parse_etag(raw: str | None) -> int:
    if not raw:
        raise HTTPException(status_code=428, detail="if_match_required")
    match = _ETAG_RE.match(raw.strip())
    if not match:
        raise HTTPException(status_code=400, detail="invalid_if_match")
    return int(match.group("version"))


def _format_etag(version: int, content_kind: str = "structured_slides") -> str:
    prefix = "" if content_kind == STANDALONE_HTML else "W/"
    return f'{prefix}"v{version}"'


def _accepted_content_kinds(
    raw: str | None,
    response: Response | None = None,
) -> frozenset[str]:
    try:
        accepted = parse_accepted_content_kinds(raw)
    except PresentationServiceError as exc:
        raise _map_presentation_service_error(exc) from exc
    if response is not None:
        merge_vary_header(response.headers)
    return accepted


def _map_standalone_validation_error(exc: StandaloneHtmlValidationError) -> HTTPException:
    headers: dict[str, str] = {}
    if exc.retry_after is not None:
        headers["Retry-After"] = str(exc.retry_after)
    return HTTPException(
        status_code=exc.status_code,
        detail=exc.code,
        headers=headers or None,
    )


def _map_presentation_service_error(exc: PresentationServiceError) -> HTTPException:
    detail: str | dict[str, str] = exc.code
    if (
        exc.code == "operation_not_supported_for_content_kind"
        and exc.operation is not None
        and exc.content_kind is not None
    ):
        detail = {
            "code": exc.code,
            "operation": exc.operation,
            "content_kind": exc.content_kind,
        }
    return HTTPException(
        status_code=exc.status_code,
        detail=detail,
    )


def _map_precondition_conflict(exc: ConflictError) -> HTTPException:
    return map_db_error_to_http(
        exc,
        conflict_status_code=status.HTTP_412_PRECONDITION_FAILED,
        conflict_detail="precondition_failed",
    )


async def _get_standalone_html_validation_pool(request: Request) -> StandaloneHtmlValidationPool:
    """Return the app-owned pool shared with the Task 8 generation worker."""
    state = request.app.state
    pool = getattr(state, _VALIDATION_POOL_ATTR, None)
    if pool is not None:
        return pool
    lock = getattr(state, _VALIDATION_POOL_LOCK_ATTR, None)
    if lock is None:
        lock = asyncio.Lock()
        setattr(state, _VALIDATION_POOL_LOCK_ATTR, lock)
    async with lock:
        pool = getattr(state, _VALIDATION_POOL_ATTR, None)
        if pool is None:
            pool = StandaloneHtmlValidationPool()
            setattr(state, _VALIDATION_POOL_ATTR, pool)
        return pool


def _slides_jobs_manager() -> JobManager:
    db_url = (os.getenv("JOBS_DB_URL") or "").strip()
    if not db_url:
        return JobManager()
    backend = "postgres" if db_url.startswith("postgres") else None
    return JobManager(backend=backend, db_url=db_url)


def _render_enabled() -> bool:
    return is_truthy(os.getenv("PRESENTATION_RENDER_ENABLED", "true"))


def _presentation_render_queue_name() -> str:
    configured_queue = (os.getenv("PRESENTATION_RENDER_JOBS_QUEUE") or "").strip().lower()
    if configured_queue in {"default", "high", "low"}:
        return configured_queue
    if configured_queue.endswith("-high"):
        return "high"
    if configured_queue.endswith("-low"):
        return "low"
    if configured_queue.endswith("-default"):
        return "default"
    return "default"


def _normalize_dt(value: str) -> datetime:
    try:
        dt = datetime.fromisoformat(value)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="invalid_timestamp") from exc
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _slide_from_obj(obj: Any) -> Slide:
    validator = getattr(Slide, "model_validate", None)
    if callable(validator):
        return validator(obj)
    return Slide.parse_obj(obj)


def _normalize_presentation_studio_transition(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    return normalized if normalized in _PRESENTATION_STUDIO_TRANSITIONS else "fade"


def _normalize_presentation_studio_manual_duration_ms(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return None
    if numeric_value <= 0:
        return None
    return int(round(numeric_value))


def _normalize_presentation_studio_timing_mode(value: Any, *, has_manual_duration: bool) -> str:
    normalized = str(value or "").strip().lower()
    if normalized not in _PRESENTATION_STUDIO_TIMING_MODES:
        return "manual" if has_manual_duration else "auto"
    if normalized == "manual" and not has_manual_duration:
        return "auto"
    return normalized


def _normalize_slide_studio_metadata(metadata: dict[str, Any]) -> None:
    studio = metadata.get("studio")
    if studio is None:
        return
    if not isinstance(studio, dict):
        raise HTTPException(status_code=422, detail="slide_studio_metadata_invalid")

    manual_duration_ms = _normalize_presentation_studio_manual_duration_ms(studio.get("manual_duration_ms"))
    studio["transition"] = _normalize_presentation_studio_transition(studio.get("transition"))
    studio["manual_duration_ms"] = manual_duration_ms
    studio["timing_mode"] = _normalize_presentation_studio_timing_mode(
        studio.get("timing_mode"),
        has_manual_duration=manual_duration_ms is not None,
    )


def _normalize_slides(slides: list[Slide]) -> list[Slide]:
    orders = [slide.order for slide in slides]
    if any(order < 0 for order in orders):
        raise HTTPException(status_code=422, detail="slide_order_negative")
    if len(set(orders)) != len(orders):
        raise HTTPException(status_code=422, detail="slide_order_not_unique")
    ordered = sorted(slides, key=lambda s: s.order)
    for idx, slide in enumerate(ordered):
        slide.order = idx
        if slide.metadata is None:
            slide.metadata = {}
        if not isinstance(slide.metadata, dict):
            raise HTTPException(status_code=422, detail="slide_metadata_invalid")
        _normalize_slide_studio_metadata(slide.metadata)
        _validate_slide_images(slide.metadata)
    return ordered


def _validate_slide_images(metadata: dict[str, Any]) -> None:
    images = metadata.get("images")
    if images is None:
        return
    try:
        normalized = validate_images_payload(images)
    except SlidesImageError as exc:
        raise HTTPException(status_code=422, detail=exc.code) from exc
    metadata["images"] = normalized


def _flatten_slides_text(slides: list[Slide]) -> str:
    parts: list[str] = []
    for slide in slides:
        if slide.title:
            parts.append(slide.title)
        if slide.content:
            parts.append(slide.content)
        if slide.speaker_notes:
            parts.append(slide.speaker_notes)
        metadata = slide.metadata if isinstance(slide.metadata, dict) else None
        if metadata:
            images = metadata.get("images")
            parts.extend(collect_image_alt_text(images if isinstance(images, list) else None))
    return "\n".join(parts)


def _slide_claim_lines(value: str | None) -> list[str]:
    claims: list[str] = []
    for raw_line in str(value or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        line = re.sub(r"^\s*(?:[-*+]\s+|\d+[.)]\s+|#{1,6}\s+|>\s*)", "", line).strip()
        if line:
            claims.append(line)
    return claims


def _build_slide_verification_units(slides: list[Slide]) -> list[ArtifactVerificationUnit]:
    units: list[ArtifactVerificationUnit] = []
    for slide in slides:
        parts: list[str] = []
        claims: list[str] = []
        layout_value = str(slide.layout.value if hasattr(slide.layout, "value") else slide.layout)
        is_title_slide = layout_value == "title"
        if slide.title:
            parts.append(f"Title: {slide.title}")
        if slide.content:
            parts.append(slide.content)
            if not is_title_slide:
                claims.extend(_slide_claim_lines(slide.content))
        if slide.speaker_notes:
            parts.append(f"Speaker notes: {slide.speaker_notes}")
            if not is_title_slide:
                claims.extend(_slide_claim_lines(slide.speaker_notes))
        text = "\n".join(parts).strip()
        if not text:
            continue
        units.append(
            ArtifactVerificationUnit(
                unit_id=f"slide:{slide.order}",
                text=text,
                claims=claims,
                metadata={
                    "slide_order": slide.order,
                    "layout": layout_value,
                },
            )
        )
    return units


async def _verify_slides_against_source(
    *,
    slides: list[Slide],
    source_text: str,
    source_type: str,
    source_ref: Any | None,
    generation_provider: str | None,
    generation_model: str | None,
    verification_provider: str | None = None,
    verification_model: str | None = None,
) -> ArtifactVerificationResult:
    effective_provider = verification_provider or generation_provider
    effective_model = verification_model or generation_model
    differs_from_generation = (
        effective_provider != generation_provider
        or effective_model != generation_model
    )
    if is_test_mode():
        return ArtifactVerificationResult(
            verdict="grounded",
            report={"total_claims": len(slides), "claims": []},
            unit_results=[],
            metadata={
                "artifact_type": "slides",
                "generation_provider": generation_provider,
                "generation_model": generation_model,
                "verification_provider": effective_provider,
                "verification_model": effective_model,
                "verification_provider_configured": verification_provider is not None,
                "verification_model_configured": verification_model is not None,
                "verification_llm_is_default": not differs_from_generation,
                "verification_llm_differs_from_generation": differs_from_generation,
                "test_mode": True,
            },
        )

    return await verify_generated_artifact_against_sources(
        artifact_type="slides",
        units=_build_slide_verification_units(slides),
        source_documents=[
            Document(
                id=f"{source_type}:{source_ref if source_ref is not None else 'source'}",
                content=source_text,
                metadata={
                    "source_type": source_type,
                    "source_ref": source_ref,
                },
            )
        ],
        generation_provider=generation_provider,
        generation_model=generation_model,
        verification_provider=verification_provider,
        verification_model=verification_model,
        generation_context={"query": "generated presentation slides"},
    )


def _normalize_job_status(job_status: Any) -> str:
    status_value = str(job_status or "").strip().lower()
    return status_value or "queued"


def _safe_json_dict(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str) or not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _validate_theme(theme: str) -> None:
    if theme not in _ALLOWED_THEMES:
        raise HTTPException(status_code=422, detail="invalid_theme")


def _validate_marp_theme(marp_theme: str | None) -> str | None:
    if marp_theme is None:
        return None
    if not isinstance(marp_theme, str) or not marp_theme.strip():
        raise HTTPException(status_code=422, detail="invalid_marp_theme")
    if not _MARP_THEME_RE.match(marp_theme):
        raise HTTPException(status_code=422, detail="invalid_marp_theme")
    return marp_theme


def _validate_settings(settings: dict[str, Any] | None) -> dict[str, Any] | None:
    if settings is None:
        return None
    if not isinstance(settings, dict):
        raise HTTPException(status_code=422, detail="invalid_settings")
    unknown = [key for key in settings if key not in _SETTINGS_ALLOWLIST]
    if unknown:
        raise HTTPException(status_code=422, detail=f"invalid_settings: unknown keys {unknown}")
    for key, value in settings.items():
        expected = _SETTINGS_ALLOWLIST[key]
        if value is None:
            continue
        if not isinstance(value, expected):
            raise HTTPException(status_code=422, detail=f"invalid_settings: {key}")
        allowed_values = _SETTINGS_STRING_ENUMS.get(key)
        if allowed_values is not None and str(value).strip() not in allowed_values:
            raise HTTPException(status_code=422, detail=f"invalid_settings: {key}")
    return settings


def _validate_custom_css(
    custom_css: Any,
    *,
    detail: str = "invalid_custom_css",
) -> str | None:
    if custom_css is None:
        return None
    if not isinstance(custom_css, str):
        raise HTTPException(status_code=422, detail=detail)
    return custom_css


def _serialize_settings(settings: dict[str, Any] | None) -> str | None:
    if settings is None:
        return None
    return json.dumps(settings)


def _serialize_studio_data(studio_data: dict[str, Any] | None) -> str | None:
    if studio_data is None:
        return None
    if not isinstance(studio_data, dict):
        raise HTTPException(status_code=422, detail="invalid_studio_data")
    return json.dumps(studio_data)


def _deserialize_settings(value: str | None) -> dict[str, Any] | None:
    if not value:
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=422, detail="invalid_settings_json") from exc
    return parsed if isinstance(parsed, dict) else None


def _deserialize_studio_data(value: str | None) -> dict[str, Any] | None:
    if not value:
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=422, detail="invalid_studio_data_json") from exc
    return parsed if isinstance(parsed, dict) else None


def _serialize_visual_style_snapshot(snapshot: dict[str, Any] | None) -> str | None:
    """Serialize a validated visual-style snapshot for presentation persistence."""
    if snapshot is None:
        return None
    if not isinstance(snapshot, dict):
        raise HTTPException(status_code=422, detail="invalid_visual_style_snapshot")
    return json.dumps(snapshot, ensure_ascii=True)


def _deserialize_visual_style_snapshot(value: str | None) -> dict[str, Any] | None:
    """Deserialize a persisted visual-style snapshot into a dictionary payload."""
    if not value:
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=422, detail="invalid_visual_style_snapshot_json") from exc
    return parsed if isinstance(parsed, dict) else None


def _deserialize_source_ref(value: str | None) -> Any | None:
    if value is None:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _serialize_source_ref(value: Any | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, (list, dict)):
        return json.dumps(value)
    return str(value)


def _field_was_set(model: Any, field_name: str) -> bool:
    """Return whether a Pydantic model received an explicit value for a field."""

    fields_set = getattr(model, "model_fields_set", None)
    if isinstance(fields_set, set):
        return field_name in fields_set
    return field_name in getattr(model, "__fields_set__", set())


def _resolve_template(template_id: str | None) -> SlidesTemplate | None:
    """Resolve a template id into a template object or raise the appropriate API error."""

    if not template_id:
        return None
    try:
        return get_slide_template(template_id)
    except SlidesTemplateNotFoundError as exc:
        raise HTTPException(status_code=404, detail="template_not_found") from exc
    except SlidesTemplateInvalidError as exc:
        raise HTTPException(status_code=500, detail="Failed to resolve slide template") from exc


def _compact_visual_style_appearance_defaults(appearance_defaults: dict[str, Any]) -> dict[str, Any]:
    """Return a compact, response-safe copy of visual-style appearance defaults."""

    compact = deepcopy(appearance_defaults)
    if isinstance(compact, dict):
        compact.pop("custom_css", None)
    return compact


def _visual_style_application_from_builtin(
    resolved: ResolvedBuiltinVisualStyle,
) -> PresentationVisualStyleApplication:
    """Convert a resolved builtin style into the presentation write-model shape."""

    appearance_defaults = _validate_visual_style_appearance_defaults(resolved.appearance)
    return PresentationVisualStyleApplication(
        style_id=resolved.definition.style_id,
        scope="builtin",
        name=resolved.definition.name,
        version=resolved.definition.version,
        snapshot=deepcopy(resolved.snapshot),
        appearance_defaults=appearance_defaults,
    )


def _visual_style_application_from_row(row: VisualStyleRow) -> PresentationVisualStyleApplication:
    """Convert a stored user visual-style row into the presentation write-model shape."""

    payload = _deserialize_visual_style_payload(row.style_payload)
    appearance_defaults_raw = (
        payload.get("appearance_defaults") if isinstance(payload.get("appearance_defaults"), dict) else {}
    )
    appearance_defaults = _validate_visual_style_appearance_defaults(appearance_defaults_raw)
    generation_rules = payload.get("generation_rules") if isinstance(payload.get("generation_rules"), dict) else {}
    fallback_policy = payload.get("fallback_policy") if isinstance(payload.get("fallback_policy"), dict) else {}
    artifact_preferences_raw = payload.get("artifact_preferences")
    artifact_preferences = artifact_preferences_raw if isinstance(artifact_preferences_raw, list) else []
    version = payload.get("version")
    response = VisualStyleResponse(
        id=row.id,
        name=row.name,
        scope=row.scope,
        description=payload.get("description") if isinstance(payload.get("description"), str) else None,
        version=version if isinstance(version, int) else None,
        generation_rules=generation_rules,
        artifact_preferences=[str(item) for item in artifact_preferences],
        appearance_defaults=appearance_defaults,
        fallback_policy=fallback_policy,
        created_at=_normalize_dt(row.created_at),
        updated_at=_normalize_dt(row.updated_at),
    )
    return PresentationVisualStyleApplication(
        style_id=response.id,
        scope=response.scope,
        name=response.name,
        version=response.version,
        snapshot=_visual_style_snapshot_from_response(response),
        appearance_defaults=response.appearance_defaults,
    )


def _apply_template_defaults(
    *,
    request: Any,
    template: SlidesTemplate | None,
    visual_style_application: PresentationVisualStyleApplication | None = None,
) -> tuple[str, str | None, dict[str, Any] | None, str | None]:
    """Merge request values with visual-style and template defaults."""

    appearance_defaults = visual_style_application.appearance_defaults if visual_style_application is not None else {}
    theme = appearance_defaults.get("theme")
    marp_theme = appearance_defaults.get("marp_theme")
    settings = appearance_defaults.get("settings")
    custom_css = appearance_defaults.get("custom_css")
    if _field_was_set(request, "theme"):
        theme = request.theme
    if _field_was_set(request, "marp_theme"):
        marp_theme = request.marp_theme
    if _field_was_set(request, "settings"):
        settings = request.settings
    if _field_was_set(request, "custom_css"):
        custom_css = _validate_custom_css(request.custom_css)

    if template:
        if theme is None:
            theme = template.theme
        if marp_theme is None:
            marp_theme = template.marp_theme
        if settings is None:
            settings = template.settings
        if custom_css is None:
            custom_css = _validate_custom_css(template.custom_css)

    if theme is None:
        theme = "black"
    return theme, marp_theme, settings, custom_css


def _template_to_response(template: SlidesTemplate) -> SlidesTemplateResponse:
    slides_payload = template.default_slides
    slides: list[Slide] | None = None
    if slides_payload:
        try:
            slides = _normalize_slides([_slide_from_obj(item) for item in slides_payload])
        except HTTPException as exc:
            raise HTTPException(status_code=500, detail="template_slides_invalid") from exc
    return SlidesTemplateResponse(
        id=template.template_id,
        name=template.name,
        theme=template.theme,
        marp_theme=template.marp_theme,
        settings=template.settings,
        default_slides=slides,
        custom_css=template.custom_css,
    )


def _deserialize_visual_style_payload(value: str) -> dict[str, Any]:
    """Deserialize a stored visual-style payload and assert its top-level shape."""
    try:
        payload = json.loads(value)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail="visual_style_payload_invalid") from exc
    if not isinstance(payload, dict):
        raise HTTPException(status_code=500, detail="visual_style_payload_invalid")
    return payload


def _validate_visual_style_appearance_defaults(appearance_defaults: dict[str, Any]) -> dict[str, Any]:
    """Validate the appearance defaults section for a visual-style payload."""
    if not isinstance(appearance_defaults, dict):
        raise HTTPException(status_code=422, detail="invalid_visual_style_appearance_defaults")
    validated = dict(appearance_defaults)
    if "theme" in validated:
        if validated.get("theme") is not None:
            _validate_theme(validated.get("theme"))
    if "marp_theme" in validated:
        validated["marp_theme"] = _validate_marp_theme(validated.get("marp_theme"))
    if "settings" in validated:
        validated["settings"] = _validate_settings(validated.get("settings"))
    if "custom_css" in validated:
        validated["custom_css"] = _validate_custom_css(
            validated.get("custom_css"),
            detail="invalid_visual_style_custom_css",
        )
    return validated


def _serialize_visual_style_payload(
    *,
    description: str | None,
    generation_rules: dict[str, Any],
    artifact_preferences: list[str],
    appearance_defaults: dict[str, Any],
    fallback_policy: dict[str, Any],
) -> str:
    """Serialize a validated visual-style payload for database storage."""
    payload = {
        "description": description,
        "generation_rules": generation_rules,
        "artifact_preferences": artifact_preferences,
        "appearance_defaults": _validate_visual_style_appearance_defaults(appearance_defaults),
        "fallback_policy": fallback_policy,
    }
    return json.dumps(payload, ensure_ascii=True)


def _visual_style_response_from_builtin(
    resolved: ResolvedBuiltinVisualStyle,
) -> VisualStyleResponse:
    """Convert a resolved builtin style into the public API response shape."""

    return VisualStyleResponse(
        id=resolved.definition.style_id,
        name=resolved.definition.name,
        scope="builtin",
        category=resolved.definition.category,
        guide_number=resolved.definition.guide_number,
        tags=list(resolved.definition.tags),
        best_for=list(resolved.definition.best_for),
        description=resolved.definition.description,
        version=resolved.definition.version,
        generation_rules=deepcopy(resolved.definition.generation_rules),
        artifact_preferences=list(resolved.definition.artifact_preferences),
        appearance_defaults=_compact_visual_style_appearance_defaults(resolved.appearance),
        fallback_policy=deepcopy(resolved.definition.fallback_policy),
        created_at=None,
        updated_at=None,
    )


def _visual_style_response_from_row(row: VisualStyleRow) -> VisualStyleResponse:
    """Convert a stored visual-style row into the public API response shape."""
    payload = _deserialize_visual_style_payload(row.style_payload)
    generation_rules = payload.get("generation_rules") if isinstance(payload.get("generation_rules"), dict) else {}
    appearance_defaults = (
        payload.get("appearance_defaults") if isinstance(payload.get("appearance_defaults"), dict) else {}
    )
    fallback_policy = payload.get("fallback_policy") if isinstance(payload.get("fallback_policy"), dict) else {}
    artifact_preferences_raw = payload.get("artifact_preferences")
    artifact_preferences = artifact_preferences_raw if isinstance(artifact_preferences_raw, list) else []
    version = payload.get("version")
    return VisualStyleResponse(
        id=row.id,
        name=row.name,
        scope=row.scope,
        description=payload.get("description") if isinstance(payload.get("description"), str) else None,
        version=version if isinstance(version, int) else None,
        generation_rules=generation_rules,
        artifact_preferences=[str(item) for item in artifact_preferences],
        appearance_defaults=appearance_defaults,
        fallback_policy=fallback_policy,
        created_at=_normalize_dt(row.created_at),
        updated_at=_normalize_dt(row.updated_at),
    )


def _resolve_visual_style_response(style_id: str, db: SlidesDatabase) -> VisualStyleResponse:
    """Resolve either a builtin or stored style into the public API response shape."""

    resolved_builtin = resolve_builtin_visual_style(style_id, include_custom_css=False)
    if resolved_builtin is not None:
        return _visual_style_response_from_builtin(resolved_builtin)
    try:
        row = db.get_visual_style_by_id(style_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="visual_style_not_found") from None
    return _visual_style_response_from_row(row)


def _visual_style_snapshot_from_response(
    style: VisualStyleResponse,
    *,
    compact_resolution: dict[str, Any] | None = None,
) -> dict[str, Any]:
    snapshot = {
        "id": style.id,
        "scope": style.scope,
        "name": style.name,
        "version": style.version,
        "description": style.description,
        "generation_rules": deepcopy(style.generation_rules),
        "artifact_preferences": list(style.artifact_preferences),
        "appearance_defaults": deepcopy(style.appearance_defaults),
        "fallback_policy": deepcopy(style.fallback_policy),
    }
    if compact_resolution is not None:
        snapshot["resolution"] = deepcopy(compact_resolution)
    return snapshot


def _resolve_presentation_visual_style_application(
    *,
    visual_style_id: str | None,
    visual_style_scope: str | None,
    db: SlidesDatabase,
) -> PresentationVisualStyleApplication | None:
    """Resolve a presentation-level visual-style selection into an application payload."""

    if visual_style_id is None and visual_style_scope is None:
        return None
    if visual_style_id is None:
        raise HTTPException(status_code=422, detail="visual_style_id_required")
    if visual_style_scope is None:
        raise HTTPException(status_code=422, detail="visual_style_scope_required")

    resolved_id = visual_style_id.strip()
    if not resolved_id:
        raise HTTPException(status_code=422, detail="visual_style_id_required")

    resolved_scope = visual_style_scope.strip().lower()
    if resolved_scope == "builtin":
        resolved_builtin = resolve_builtin_visual_style(resolved_id)
        if resolved_builtin is None:
            raise HTTPException(status_code=404, detail="visual_style_not_found")
        return _visual_style_application_from_builtin(resolved_builtin)
    if resolved_scope == "user":
        try:
            row = db.get_visual_style_by_id(resolved_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="visual_style_not_found") from None
        return _visual_style_application_from_row(row)
    else:
        raise HTTPException(status_code=422, detail="invalid_visual_style_scope")


def _normalize_template_slides(slides_payload: list[Any]) -> list[Slide]:
    try:
        return _normalize_slides([_slide_from_obj(item) for item in slides_payload])
    except HTTPException as exc:
        raise HTTPException(status_code=500, detail="template_slides_invalid") from exc


def _load_version_payload(payload_json: str) -> dict[str, Any]:
    payload: dict[str, Any] | None = None
    try:
        payload = decode_presentation_version_payload(payload_json)
    except InputError:
        pass
    if payload is None:
        raise HTTPException(status_code=500, detail="version_payload_invalid")
    return payload


def _payload_to_presentation(payload: dict[str, Any], *, additive: bool = False) -> dict[str, Any]:
    try:
        result = snapshot_detail(payload)
    except PresentationServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.code) from exc
    if not additive and result.get("content_kind") == "structured_slides":
        result.pop("content_kind", None)
    return result


def _restore_structured_snapshot(
    *,
    db: SlidesDatabase,
    presentation_id: str,
    payload: dict[str, Any],
    expected_version: int,
):
    """Restore a structured snapshot through the legacy normalization pipeline."""

    try:
        restored = StructuredPresentationResponse.model_validate(_payload_to_presentation(payload, additive=True))
    except ValidationError as exc:
        raise InputError("version_payload_invalid") from exc
    theme = restored.theme
    _validate_theme(theme)
    marp_theme = _validate_marp_theme(restored.marp_theme)
    settings = _validate_settings(restored.settings)
    studio_data = (
        restored.studio_data if isinstance(restored.studio_data, dict) or restored.studio_data is None else None
    )
    slides = _normalize_slides(restored.slides)
    title = restored.title.strip()
    if not title:
        raise HTTPException(status_code=422, detail="title_required")
    return db.update_presentation(
        presentation_id=presentation_id,
        update_fields={
            "title": title,
            "description": restored.description,
            "theme": theme,
            "marp_theme": marp_theme,
            "template_id": restored.template_id,
            "visual_style_id": restored.visual_style_id,
            "visual_style_scope": restored.visual_style_scope,
            "visual_style_name": restored.visual_style_name,
            "visual_style_version": restored.visual_style_version,
            "visual_style_snapshot": _serialize_visual_style_snapshot(restored.visual_style_snapshot),
            "settings": _serialize_settings(settings),
            "studio_data": _serialize_studio_data(studio_data),
            "slides": json.dumps(
                [slide.model_dump() if hasattr(slide, "model_dump") else slide.dict() for slide in slides]
            ),
            "slides_text": _flatten_slides_text(slides),
            "custom_css": restored.custom_css,
            "source_type": restored.source_type,
            "source_ref": _serialize_source_ref(restored.source_ref),
            "source_query": restored.source_query,
            "deleted": 0,
        },
        expected_version=expected_version,
    )


def _build_presentation_response(row, *, additive: bool = False) -> dict[str, Any]:
    try:
        result = presentation_detail(row)
    except PresentationServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.code) from exc
    if not additive and result.get("content_kind") == "structured_slides":
        result.pop("content_kind", None)
    return result


def _build_summary(row, *, additive: bool = False) -> dict[str, Any]:
    result = presentation_summary(row)
    if not additive and result.get("content_kind") == "structured_slides":
        for field in ("content_kind", "provenance", "slide_count"):
            result.pop(field, None)
    return result


def _parse_sort(sort: str | None) -> tuple[str, str]:
    if not sort:
        return "created_at", "DESC"
    parts = sort.strip().split()
    col = parts[0]
    direction = parts[1] if len(parts) > 1 else "DESC"
    if col not in {"created_at", "last_modified", "title"}:
        raise HTTPException(status_code=400, detail="invalid_sort")
    if direction.upper() not in {"ASC", "DESC"}:
        raise HTTPException(status_code=400, detail="invalid_sort")
    return col, direction.upper()


def _resolve_provider(request_provider: str | None) -> str:
    provider = (request_provider or DEFAULT_LLM_PROVIDER or "openai").strip()
    return provider.lower() if provider else "openai"


def _resolve_media_source_text(
    *,
    media_db: MediaDbSession,
    media_row: dict[str, Any],
    media_id: int,
) -> str | None:
    transcript = get_latest_transcription(media_db, media_id)
    if isinstance(transcript, str) and transcript.strip():
        return transcript.strip()

    try:
        latest_document = get_document_version(
            db_instance=media_db,
            media_id=media_id,
            version_number=None,
            include_content=True,
        )
    except Exception:
        logger.debug("Failed to resolve latest document content for slides source media")
        latest_document = None

    if isinstance(latest_document, dict):
        document_content = latest_document.get("content")
        if isinstance(document_content, str) and document_content.strip():
            return document_content.strip()

    media_content = media_row.get("content")
    if isinstance(media_content, str) and media_content.strip():
        return media_content.strip()

    return None


def _format_chat_messages(messages: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for msg in messages:
        sender = msg.get("sender") or msg.get("role") or "unknown"
        content = msg.get("content") or ""
        if content:
            lines.append(f"{sender}: {content}")
    return "\n".join(lines).strip()


def _format_notes(notes: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for note in notes:
        title = note.get("title") or ""
        content = note.get("content") or ""
        if title:
            parts.append(f"# {title}")
        if content:
            parts.append(str(content))
    return "\n\n".join(parts).strip()


async def _resolve_notes_db_for_request(http_request: Request, current_user: User) -> CharactersRAGDB:
    """Resolve the per-user notes DB lazily while still honoring test/app dependency overrides."""

    override_fn = http_request.app.dependency_overrides.get(get_chacha_db_for_user)
    if override_fn is not None:
        result = override_fn()
        if inspect.isawaitable(result):
            result = await result
        return result
    return await get_chacha_db_for_user_id(current_user.id, str(current_user.id))


def _format_rag_documents(documents: list[Any]) -> str:
    parts: list[str] = []
    for doc in documents:
        metadata = getattr(doc, "metadata", {}) or {}
        title = metadata.get("title") or metadata.get("source_title") or getattr(doc, "id", "source")
        content = getattr(doc, "content", "")
        if title:
            parts.append(f"# {title}")
        if content:
            parts.append(str(content))
    return "\n\n".join(parts).strip()


async def _generate_presentation(
    *,
    response: Response,
    db: SlidesDatabase,
    request: Any,
    source_text: str,
    source_type: str,
    source_ref: Any | None,
    source_query: str | None,
) -> PresentationResponse:
    visual_style_application = _resolve_presentation_visual_style_application(
        visual_style_id=getattr(request, "visual_style_id", None),
        visual_style_scope=getattr(request, "visual_style_scope", None),
        db=db,
    )
    visual_style_snapshot_dict = visual_style_application.snapshot if visual_style_application else None
    template = _resolve_template(getattr(request, "template_id", None))
    theme, marp_theme, settings, custom_css = _apply_template_defaults(
        request=request,
        template=template,
        visual_style_application=visual_style_application,
    )
    _validate_theme(theme)
    marp_theme = _validate_marp_theme(marp_theme)
    settings = _validate_settings(settings)
    provider = _resolve_provider(request.provider)
    generator = SlidesGenerator()
    try:
        metrics = get_metrics_registry()
    except _SLIDES_NONCRITICAL_EXCEPTIONS:
        logger.debug("Failed to get metrics registry, metrics disabled")
        metrics = None
    started_at = time.perf_counter()

    def _record_generation_error(error_type: str) -> None:
        if metrics is None:
            return
        try:
            metrics.increment(
                "slides_generation_errors_total",
                labels={"source_type": source_type, "error": error_type},
            )
        except _SLIDES_NONCRITICAL_EXCEPTIONS:
            logger.debug("Failed to record slides generation error metric")

    try:
        generated = generator.generate_from_text(
            source_text=source_text,
            title_hint=request.title_hint,
            provider=provider,
            model=request.model,
            api_key=None,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            max_source_tokens=request.max_source_tokens,
            max_source_chars=request.max_source_chars,
            enable_chunking=request.enable_chunking,
            chunk_size_tokens=request.chunk_size_tokens,
            summary_tokens=request.summary_tokens,
            visual_style_snapshot=visual_style_snapshot_dict,
        )
    except SlidesSourceTooLargeError as exc:
        _record_generation_error("input_too_large")
        detail = {
            "detail": "Input exceeds size limits and chunking is disabled.",
            "code": "input_too_large",
        }
        if exc.max_source_tokens is not None:
            detail["max_source_tokens"] = exc.max_source_tokens
        if exc.max_source_chars is not None:
            detail["max_source_chars"] = exc.max_source_chars
        raise HTTPException(status_code=413, detail=detail) from exc
    except SlidesGenerationInputError as exc:
        _record_generation_error("input_error")
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except SlidesGenerationOutputError as exc:
        _record_generation_error("output_error")
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except SlidesGenerationError as exc:
        _record_generation_error("generation_error")
        raise HTTPException(status_code=500, detail="Failed to generate presentation") from exc

    try:
        slides = _normalize_slides([_slide_from_obj(s) for s in generated["slides"]])
    except HTTPException:
        raise
    except (ValidationError, KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail="invalid_generated_slides") from exc
    slides_text = _flatten_slides_text(slides)
    claim_verification = await _verify_slides_against_source(
        slides=slides,
        source_text=source_text,
        source_type=source_type,
        source_ref=source_ref,
        generation_provider=provider,
        generation_model=request.model,
        verification_provider=getattr(request, "claims_verification_provider", None),
        verification_model=getattr(request, "claims_verification_model", None),
    )
    claim_verification_payload = claim_verification.to_dict()
    if claim_verification.verdict != "grounded":
        raise HTTPException(
            status_code=422,
            detail={
                "code": "claim_verification_failed",
                "claimVerification": claim_verification_payload,
            },
        )
    studio_data = {"claimVerification": claim_verification_payload}
    row = db.create_presentation(
        presentation_id=None,
        title=generated["title"],
        description=None,
        theme=theme,
        marp_theme=marp_theme,
        template_id=template.template_id if template else None,
        visual_style_id=visual_style_application.style_id if visual_style_application else None,
        visual_style_scope=visual_style_application.scope if visual_style_application else None,
        visual_style_name=visual_style_application.name if visual_style_application else None,
        visual_style_version=visual_style_application.version if visual_style_application else None,
        visual_style_snapshot=_serialize_visual_style_snapshot(visual_style_snapshot_dict),
        settings=_serialize_settings(settings),
        studio_data=_serialize_studio_data(studio_data),
        slides=json.dumps([slide.model_dump() if hasattr(slide, "model_dump") else slide.dict() for slide in slides]),
        slides_text=slides_text,
        source_type=source_type,
        source_ref=_serialize_source_ref(source_ref),
        source_query=source_query,
        custom_css=custom_css,
    )
    if metrics is not None:
        try:
            metrics.observe(
                "slides_generation_latency_seconds",
                time.perf_counter() - started_at,
                labels={"source_type": source_type},
            )
        except _SLIDES_NONCRITICAL_EXCEPTIONS:
            logger.debug("Failed to record slides generation latency metric")
    response.headers["ETag"] = _format_etag(row.version)
    response.headers["Last-Modified"] = row.last_modified
    return _build_presentation_response(row)


@router.post(
    "/presentations",
    response_model=PresentationResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a presentation",
    dependencies=[Depends(RequirePermission(MEDIA_CREATE)), Depends(rbac_rate_limit("slides.create"))],
)
async def create_presentation(
    request: PresentationCreateRequest,
    response: Response,
    db: SlidesDatabase = Depends(get_slides_db_for_user),
) -> PresentationResponse:
    service = PresentationService(db)
    try:
        service.require_generic_create(request.content_kind)
        if any(
            _field_was_set(request, field)
            for field in (
                "html_document",
                "html_sha256",
                "html_bytes",
                "html_slide_count",
                "generation_job_uuid",
                "generation_provenance",
            )
        ):
            raise service.operation_not_supported(request.content_kind, "create")
    except PresentationServiceError as exc:
        raise _map_presentation_service_error(exc) from exc
    title = request.title.strip()
    if not title:
        raise HTTPException(status_code=422, detail="title_required")
    visual_style_application = _resolve_presentation_visual_style_application(
        visual_style_id=request.visual_style_id,
        visual_style_scope=request.visual_style_scope,
        db=db,
    )
    visual_style_snapshot_dict = visual_style_application.snapshot if visual_style_application else None
    template = _resolve_template(request.template_id)
    theme, marp_theme, settings, custom_css = _apply_template_defaults(
        request=request,
        template=template,
        visual_style_application=visual_style_application,
    )
    _validate_theme(theme)
    marp_theme = _validate_marp_theme(marp_theme)
    settings = _validate_settings(settings)
    if template and not _field_was_set(request, "slides") and template.default_slides:
        slides = _normalize_template_slides(template.default_slides)
    else:
        slides = _normalize_slides([_slide_from_obj(s) for s in request.slides])
    slides_text = _flatten_slides_text(slides)
    row = db.create_presentation(
        presentation_id=None,
        title=title,
        description=request.description,
        theme=theme,
        marp_theme=marp_theme,
        template_id=template.template_id if template else None,
        visual_style_id=visual_style_application.style_id if visual_style_application else None,
        visual_style_scope=visual_style_application.scope if visual_style_application else None,
        visual_style_name=visual_style_application.name if visual_style_application else None,
        visual_style_version=visual_style_application.version if visual_style_application else None,
        visual_style_snapshot=_serialize_visual_style_snapshot(visual_style_snapshot_dict),
        settings=_serialize_settings(settings),
        studio_data=_serialize_studio_data(request.studio_data),
        slides=json.dumps([slide.model_dump() if hasattr(slide, "model_dump") else slide.dict() for slide in slides]),
        slides_text=slides_text,
        source_type="manual",
        source_ref=None,
        source_query=None,
        custom_css=custom_css,
    )
    response.headers["ETag"] = _format_etag(row.version, row.content_kind)
    response.headers["Last-Modified"] = row.last_modified
    return _build_presentation_response(row)


@router.get(
    "/presentations",
    response_model=PresentationListResponse,
    summary="List presentations",
    dependencies=[Depends(RequirePermission(MEDIA_READ)), Depends(rbac_rate_limit("slides.list"))],
)
async def list_presentations(
    response: Response,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    sort: str | None = Query(None, description="Sort by created_at/last_modified/title, e.g. 'created_at desc'"),
    include_deleted: bool = Query(False),
    accept_content_kinds: str | None = Header(None, alias=CONTENT_KIND_HEADER),
    db: SlidesDatabase = Depends(get_slides_db_for_user),
) -> PresentationListResponse:
    accepted = _accepted_content_kinds(accept_content_kinds, response)
    sort_col, sort_dir = _parse_sort(sort)
    rows, total = PresentationService(db).list_summaries(
        limit=limit,
        offset=offset,
        include_deleted=include_deleted,
        sort_column=sort_col,
        sort_direction=sort_dir,
        accepted_content_kinds=accepted,
    )
    return PresentationListResponse(
        presentations=[_build_summary(row, additive=STANDALONE_HTML in accepted) for row in rows],
        total=total,
        limit=limit,
        offset=offset,
        pagination=build_offset_pagination_meta(
            limit=limit,
            offset=offset,
            total=total,
            count=len(rows),
        ),
    )


@router.get(
    "/presentations/search",
    response_model=PresentationSearchResponse,
    summary="Search presentations",
    dependencies=[Depends(RequirePermission(MEDIA_READ)), Depends(rbac_rate_limit("slides.search"))],
)
async def search_presentations(
    response: Response,
    q: str = Query(..., min_length=1),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    include_deleted: bool = Query(False),
    accept_content_kinds: str | None = Header(None, alias=CONTENT_KIND_HEADER),
    db: SlidesDatabase = Depends(get_slides_db_for_user),
) -> PresentationSearchResponse:
    accepted = _accepted_content_kinds(accept_content_kinds, response)
    try:
        rows, total = PresentationService(db).search_summaries(
            query=q,
            limit=limit,
            offset=offset,
            include_deleted=include_deleted,
            accepted_content_kinds=accepted,
        )
    except InputError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return PresentationSearchResponse(
        presentations=[_build_summary(row, additive=STANDALONE_HTML in accepted) for row in rows],
        total=total,
        limit=limit,
        offset=offset,
        pagination=build_offset_pagination_meta(
            limit=limit,
            offset=offset,
            total=total,
            count=len(rows),
        ),
    )


@router.get(
    "/presentations/{presentation_id}",
    response_model=PresentationResponse,
    summary="Get presentation",
    dependencies=[Depends(RequirePermission(MEDIA_READ)), Depends(rbac_rate_limit("slides.get"))],
)
async def get_presentation(
    presentation_id: str,
    response: Response,
    include_deleted: bool = Query(False),
    accept_content_kinds: str | None = Header(None, alias=CONTENT_KIND_HEADER),
    db: SlidesDatabase = Depends(get_slides_db_for_user),
) -> PresentationResponse:
    accepted = _accepted_content_kinds(accept_content_kinds, response)
    try:
        row = PresentationService(db).get_detail(
            presentation_id,
            accepted,
            include_deleted=include_deleted,
        )
    except KeyError:
        raise HTTPException(status_code=404, detail="presentation_not_found") from None
    except PresentationServiceError as exc:
        raise _map_presentation_service_error(exc) from exc
    response.headers["ETag"] = _format_etag(row.version, row.content_kind)
    response.headers["Last-Modified"] = row.last_modified
    if row.content_kind == STANDALONE_HTML:
        response.headers["Cache-Control"] = "private, no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
    return _build_presentation_response(row, additive=STANDALONE_HTML in accepted)


@router.put(
    "/presentations/{presentation_id}/html-source",
    response_model=PresentationResponse,
    summary="Save standalone HTML source",
    dependencies=[
        Depends(RequirePermission(MEDIA_UPDATE)),
        Depends(rbac_rate_limit("slides.update")),
    ],
)
async def save_standalone_html_source(
    presentation_id: str,
    request: Request,
    response: Response,
    if_match: str | None = Header(None, alias="If-Match"),
    content_type: str | None = Header(None, alias="Content-Type"),
    accept_content_kinds: str | None = Header(None, alias=CONTENT_KIND_HEADER),
    db: SlidesDatabase = Depends(get_slides_db_for_user),
    validation_pool: StandaloneHtmlValidationPool = Depends(_get_standalone_html_validation_pool),
) -> PresentationResponse:
    accepted = _accepted_content_kinds(accept_content_kinds, response)
    service = PresentationService(db, validation_pool=validation_pool)
    try:
        kind = service.guard_target(presentation_id, accepted)
        service.require_operation(kind.content_kind, "html_source")
    except KeyError:
        raise HTTPException(status_code=404, detail="presentation_not_found") from None
    except PresentationServiceError as exc:
        raise _map_presentation_service_error(exc) from exc
    if (content_type or "").split(";", 1)[0].strip().lower() != "application/octet-stream":
        raise HTTPException(status_code=415, detail="unsupported_media_type")
    expected_version = _parse_etag(if_match)
    source = await request.body()
    try:
        row = await service.save_html_source(
            presentation_id=presentation_id,
            html_document=source,
            expected_version=expected_version,
        )
    except StandaloneHtmlValidationError as exc:
        raise _map_standalone_validation_error(exc) from exc
    except KeyError:
        raise HTTPException(status_code=404, detail="presentation_not_found") from None
    except InputError as exc:
        raise map_db_error_to_http(exc, default_detail="Failed to save standalone HTML") from exc
    except ConflictError as exc:
        raise _map_precondition_conflict(exc) from exc
    except PresentationServiceError as exc:
        raise _map_presentation_service_error(exc) from exc
    response.headers["ETag"] = _format_etag(row.version, row.content_kind)
    response.headers["Last-Modified"] = row.last_modified
    response.headers["Cache-Control"] = "private, no-store"
    response.headers["X-Content-Type-Options"] = "nosniff"
    return _build_presentation_response(row, additive=True)


@router.put(
    "/presentations/{presentation_id}",
    response_model=PresentationResponse,
    summary="Update presentation",
    dependencies=[Depends(RequirePermission(MEDIA_UPDATE)), Depends(rbac_rate_limit("slides.update"))],
)
async def update_presentation(
    presentation_id: str,
    request: PresentationUpdateRequest,
    response: Response,
    if_match: str | None = Header(None, alias="If-Match"),
    accept_content_kinds: str | None = Header(None, alias=CONTENT_KIND_HEADER),
    db: SlidesDatabase = Depends(get_slides_db_for_user),
) -> PresentationResponse:
    accepted = _accepted_content_kinds(accept_content_kinds, response)
    service = PresentationService(db)
    try:
        kind = service.guard_target(presentation_id, accepted)
        if _field_was_set(request, "content_kind") and request.content_kind != kind.content_kind:
            raise PresentationServiceError("content_kind_immutable", status_code=409)
        service.require_operation(kind.content_kind, "update")
        if any(
            _field_was_set(request, field)
            for field in (
                "html_document",
                "html_sha256",
                "html_bytes",
                "html_slide_count",
                "generation_job_uuid",
                "generation_provenance",
            )
        ):
            raise service.operation_not_supported(kind.content_kind, "update")
    except KeyError:
        raise HTTPException(status_code=404, detail="presentation_not_found") from None
    except PresentationServiceError as exc:
        raise _map_presentation_service_error(exc) from exc
    expected_version = _parse_etag(if_match)
    title = request.title.strip()
    if not title:
        raise HTTPException(status_code=422, detail="title_required")
    visual_style_application = _resolve_presentation_visual_style_application(
        visual_style_id=request.visual_style_id,
        visual_style_scope=request.visual_style_scope,
        db=db,
    )
    visual_style_snapshot_dict = visual_style_application.snapshot if visual_style_application else None
    template = _resolve_template(request.template_id)
    theme, marp_theme, settings, custom_css = _apply_template_defaults(
        request=request,
        template=template,
        visual_style_application=visual_style_application,
    )
    _validate_theme(theme)
    marp_theme = _validate_marp_theme(marp_theme)
    settings = _validate_settings(settings)
    if template and not _field_was_set(request, "slides") and template.default_slides:
        slides = _normalize_template_slides(template.default_slides)
    else:
        slides = _normalize_slides([_slide_from_obj(s) for s in request.slides])
    slides_text = _flatten_slides_text(slides)
    try:
        row = db.update_presentation(
            presentation_id=presentation_id,
            update_fields={
                "title": title,
                "description": request.description,
                "theme": theme,
                "marp_theme": marp_theme,
                "template_id": template.template_id if template else None,
                "visual_style_id": visual_style_application.style_id if visual_style_application else None,
                "visual_style_scope": visual_style_application.scope if visual_style_application else None,
                "visual_style_name": visual_style_application.name if visual_style_application else None,
                "visual_style_version": visual_style_application.version if visual_style_application else None,
                "visual_style_snapshot": _serialize_visual_style_snapshot(visual_style_snapshot_dict),
                "settings": _serialize_settings(settings),
                "studio_data": _serialize_studio_data(request.studio_data),
                "slides": json.dumps(
                    [slide.model_dump() if hasattr(slide, "model_dump") else slide.dict() for slide in slides]
                ),
                "slides_text": slides_text,
                "custom_css": custom_css,
            },
            expected_version=expected_version,
        )
    except KeyError:
        raise HTTPException(status_code=404, detail="presentation_not_found") from None
    except InputError as exc:
        raise map_db_error_to_http(exc, default_detail="Failed to update presentation") from exc
    except ConflictError as exc:
        raise _map_precondition_conflict(exc) from exc
    response.headers["ETag"] = _format_etag(row.version, row.content_kind)
    response.headers["Last-Modified"] = row.last_modified
    return _build_presentation_response(row, additive=STANDALONE_HTML in accepted)


@router.patch(
    "/presentations/{presentation_id}",
    response_model=PresentationResponse,
    summary="Patch presentation",
    dependencies=[Depends(RequirePermission(MEDIA_UPDATE)), Depends(rbac_rate_limit("slides.update"))],
)
async def patch_presentation(
    presentation_id: str,
    request: PresentationPatchRequest,
    response: Response,
    if_match: str | None = Header(None, alias="If-Match"),
    accept_content_kinds: str | None = Header(None, alias=CONTENT_KIND_HEADER),
    db: SlidesDatabase = Depends(get_slides_db_for_user),
) -> PresentationResponse:
    accepted = _accepted_content_kinds(accept_content_kinds, response)
    service = PresentationService(db)
    try:
        kind = service.guard_target(presentation_id, accepted)
        if _field_was_set(request, "content_kind") and request.content_kind != kind.content_kind:
            raise PresentationServiceError("content_kind_immutable", status_code=409)
        service.require_operation(kind.content_kind, "update")
        if any(
            _field_was_set(request, field)
            for field in (
                "html_document",
                "html_sha256",
                "html_bytes",
                "html_slide_count",
                "generation_job_uuid",
                "generation_provenance",
            )
        ):
            raise service.operation_not_supported(kind.content_kind, "update")
    except KeyError:
        raise HTTPException(status_code=404, detail="presentation_not_found") from None
    except PresentationServiceError as exc:
        raise _map_presentation_service_error(exc) from exc
    expected_version = _parse_etag(if_match)
    update_fields: dict[str, Any] = {}
    builtin_appearance_defaults: dict[str, Any] | None = None
    theme_was_set = _field_was_set(request, "theme")
    marp_theme_was_set = _field_was_set(request, "marp_theme")
    settings_was_set = _field_was_set(request, "settings")
    custom_css_was_set = _field_was_set(request, "custom_css")
    if request.title is not None:
        title = request.title.strip()
        if not title:
            raise HTTPException(status_code=422, detail="title_required")
        update_fields["title"] = title
    if request.description is not None:
        update_fields["description"] = request.description
    if theme_was_set and request.theme is None:
        raise HTTPException(status_code=422, detail="invalid_theme")
    if request.theme is not None:
        _validate_theme(request.theme)
        update_fields["theme"] = request.theme
    if marp_theme_was_set:
        update_fields["marp_theme"] = _validate_marp_theme(request.marp_theme)
    if _field_was_set(request, "template_id"):
        template = _resolve_template(request.template_id)
        update_fields["template_id"] = template.template_id if template else None
    if _field_was_set(request, "visual_style_id") or _field_was_set(request, "visual_style_scope"):
        if request.visual_style_id is None and request.visual_style_scope is None:
            update_fields["visual_style_id"] = None
            update_fields["visual_style_scope"] = None
            update_fields["visual_style_name"] = None
            update_fields["visual_style_version"] = None
            update_fields["visual_style_snapshot"] = None
        else:
            visual_style_application = _resolve_presentation_visual_style_application(
                visual_style_id=request.visual_style_id,
                visual_style_scope=request.visual_style_scope,
                db=db,
            )
            if visual_style_application is None:
                update_fields["visual_style_id"] = None
                update_fields["visual_style_scope"] = None
                update_fields["visual_style_name"] = None
                update_fields["visual_style_version"] = None
                update_fields["visual_style_snapshot"] = None
            else:
                update_fields["visual_style_id"] = visual_style_application.style_id
                update_fields["visual_style_scope"] = visual_style_application.scope
                update_fields["visual_style_name"] = visual_style_application.name
                update_fields["visual_style_version"] = visual_style_application.version
                update_fields["visual_style_snapshot"] = _serialize_visual_style_snapshot(
                    visual_style_application.snapshot
                )
                if visual_style_application.scope == "builtin":
                    # Built-in presets provide deck-wide appearance defaults unless the caller
                    # overrides a specific field in this patch request.
                    builtin_appearance_defaults = visual_style_application.appearance_defaults
                    if not theme_was_set and "theme" not in update_fields:
                        update_fields["theme"] = builtin_appearance_defaults.get("theme") or "black"
                    if not marp_theme_was_set and "marp_theme" not in update_fields:
                        update_fields["marp_theme"] = builtin_appearance_defaults.get("marp_theme")
                    if not settings_was_set and "settings" not in update_fields:
                        update_fields["settings"] = _serialize_settings(builtin_appearance_defaults.get("settings"))
                    if not custom_css_was_set and "custom_css" not in update_fields:
                        update_fields["custom_css"] = builtin_appearance_defaults.get("custom_css")
    if settings_was_set:
        settings = _validate_settings(request.settings)
        update_fields["settings"] = _serialize_settings(settings)
    if _field_was_set(request, "studio_data"):
        update_fields["studio_data"] = _serialize_studio_data(request.studio_data)
    if request.slides is not None:
        slides = _normalize_slides([_slide_from_obj(s) for s in request.slides])
        update_fields["slides"] = json.dumps(
            [slide.model_dump() if hasattr(slide, "model_dump") else slide.dict() for slide in slides]
        )
        update_fields["slides_text"] = _flatten_slides_text(slides)
    if custom_css_was_set:
        update_fields["custom_css"] = _validate_custom_css(request.custom_css)
    if not update_fields:
        raise HTTPException(status_code=400, detail="no_fields_to_update")
    try:
        row = db.update_presentation(
            presentation_id=presentation_id,
            update_fields=update_fields,
            expected_version=expected_version,
        )
    except KeyError:
        raise HTTPException(status_code=404, detail="presentation_not_found") from None
    except InputError as exc:
        raise map_db_error_to_http(exc, default_detail="Failed to patch presentation") from exc
    except ConflictError as exc:
        raise _map_precondition_conflict(exc) from exc
    response.headers["ETag"] = _format_etag(row.version, row.content_kind)
    response.headers["Last-Modified"] = row.last_modified
    return _build_presentation_response(row, additive=STANDALONE_HTML in accepted)


@router.post(
    "/presentations/{presentation_id}/reorder",
    response_model=PresentationResponse,
    summary="Reorder slides in a presentation",
    dependencies=[Depends(RequirePermission(MEDIA_UPDATE)), Depends(rbac_rate_limit("slides.update"))],
)
async def reorder_presentation(
    presentation_id: str,
    request: PresentationReorderRequest,
    response: Response,
    if_match: str | None = Header(None, alias="If-Match"),
    accept_content_kinds: str | None = Header(None, alias=CONTENT_KIND_HEADER),
    db: SlidesDatabase = Depends(get_slides_db_for_user),
) -> PresentationResponse:
    accepted = _accepted_content_kinds(accept_content_kinds, response)
    service = PresentationService(db)
    try:
        kind = service.guard_target(presentation_id, accepted)
        service.require_operation(kind.content_kind, "reorder")
        row = db.get_presentation_by_id(presentation_id, include_deleted=False)
    except KeyError:
        raise HTTPException(status_code=404, detail="presentation_not_found") from None
    except PresentationServiceError as exc:
        raise _map_presentation_service_error(exc) from exc
    expected_version = _parse_etag(if_match)

    slides_raw = json.loads(row.slides)
    slides = _normalize_slides([_slide_from_obj(item) for item in slides_raw])
    order = request.order
    if len(order) != len(slides):
        raise HTTPException(status_code=422, detail="invalid_reorder_length")
    if set(order) != set(range(len(slides))):
        raise HTTPException(status_code=422, detail="invalid_reorder_indices")

    reordered = [slides[idx] for idx in order]
    for idx, slide in enumerate(reordered):
        slide.order = idx
    slides_text = _flatten_slides_text(reordered)
    try:
        row = db.update_presentation(
            presentation_id=presentation_id,
            update_fields={
                "slides": json.dumps(
                    [slide.model_dump() if hasattr(slide, "model_dump") else slide.dict() for slide in reordered]
                ),
                "slides_text": slides_text,
            },
            expected_version=expected_version,
        )
    except KeyError:
        raise HTTPException(status_code=404, detail="presentation_not_found") from None
    except InputError as exc:
        raise map_db_error_to_http(exc, default_detail="Failed to reorder presentation") from exc
    except ConflictError as exc:
        raise _map_precondition_conflict(exc) from exc

    response.headers["ETag"] = _format_etag(row.version, row.content_kind)
    response.headers["Last-Modified"] = row.last_modified
    return _build_presentation_response(row, additive=STANDALONE_HTML in accepted)


@router.delete(
    "/presentations/{presentation_id}",
    response_model=PresentationDeleteResponse,
    summary="Soft delete presentation",
    dependencies=[Depends(RequirePermission(MEDIA_DELETE)), Depends(rbac_rate_limit("slides.delete"))],
)
async def delete_presentation(
    presentation_id: str,
    response: Response,
    if_match: str | None = Header(None, alias="If-Match"),
    accept_content_kinds: str | None = Header(None, alias=CONTENT_KIND_HEADER),
    db: SlidesDatabase = Depends(get_slides_db_for_user),
) -> PresentationDeleteResponse:
    accepted = _accepted_content_kinds(accept_content_kinds, response)
    service = PresentationService(db)
    try:
        kind = service.guard_target(presentation_id, accepted)
        service.require_operation(kind.content_kind, "delete")
        expected_version = _parse_etag(if_match)
        result = service.delete_presentation(
            presentation_id=presentation_id,
            expected_version=expected_version,
        )
    except KeyError:
        raise HTTPException(status_code=404, detail="presentation_not_found") from None
    except InputError as exc:
        raise map_db_error_to_http(exc, default_detail="Failed to delete presentation") from exc
    except ConflictError as exc:
        raise _map_precondition_conflict(exc) from exc
    except PresentationServiceError as exc:
        raise _map_presentation_service_error(exc) from exc
    if isinstance(result, dict):
        return result
    response.headers["ETag"] = _format_etag(result.version, result.content_kind)
    response.headers["Last-Modified"] = result.last_modified
    return _build_presentation_response(result, additive=STANDALONE_HTML in accepted)


@router.post(
    "/presentations/{presentation_id}/restore",
    response_model=PresentationResponse,
    summary="Restore soft-deleted presentation",
    dependencies=[Depends(RequirePermission(MEDIA_UPDATE)), Depends(rbac_rate_limit("slides.restore"))],
)
async def restore_presentation(
    presentation_id: str,
    response: Response,
    if_match: str | None = Header(None, alias="If-Match"),
    accept_content_kinds: str | None = Header(None, alias=CONTENT_KIND_HEADER),
    db: SlidesDatabase = Depends(get_slides_db_for_user),
    validation_pool: StandaloneHtmlValidationPool = Depends(_get_standalone_html_validation_pool),
) -> PresentationResponse:
    accepted = _accepted_content_kinds(accept_content_kinds, response)
    service = PresentationService(db, validation_pool=validation_pool)
    try:
        kind = service.guard_target(presentation_id, accepted, include_deleted=True)
        service.require_operation(kind.content_kind, "restore")
        expected_version = _parse_etag(if_match)
        row = await service.restore_presentation(
            presentation_id=presentation_id,
            expected_version=expected_version,
        )
    except KeyError:
        raise HTTPException(status_code=404, detail="presentation_not_found") from None
    except InputError as exc:
        raise map_db_error_to_http(exc, default_detail="Failed to restore presentation") from exc
    except ConflictError as exc:
        raise _map_precondition_conflict(exc) from exc
    except StandaloneHtmlValidationError as exc:
        raise _map_standalone_validation_error(exc) from exc
    except PresentationServiceError as exc:
        raise _map_presentation_service_error(exc) from exc
    response.headers["ETag"] = _format_etag(row.version, row.content_kind)
    response.headers["Last-Modified"] = row.last_modified
    if row.content_kind == STANDALONE_HTML:
        response.headers["Cache-Control"] = "private, no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
    return _build_presentation_response(row, additive=STANDALONE_HTML in accepted)


@router.get(
    "/templates",
    response_model=SlidesTemplateListResponse,
    summary="List slide templates",
    dependencies=[Depends(RequirePermission(MEDIA_READ)), Depends(rbac_rate_limit("slides.templates.list"))],
)
async def list_templates() -> SlidesTemplateListResponse:
    try:
        templates = list_slide_templates()
    except SlidesTemplateInvalidError as exc:
        raise HTTPException(status_code=500, detail="Failed to list slide templates") from exc
    return SlidesTemplateListResponse(templates=[_template_to_response(t) for t in templates])


@router.get(
    "/templates/{template_id}",
    response_model=SlidesTemplateResponse,
    summary="Get slide template",
    dependencies=[Depends(RequirePermission(MEDIA_READ)), Depends(rbac_rate_limit("slides.templates.get"))],
)
async def get_template(template_id: str) -> SlidesTemplateResponse:
    try:
        template = get_slide_template(template_id)
    except SlidesTemplateNotFoundError as exc:
        raise HTTPException(status_code=404, detail="template_not_found") from exc
    except SlidesTemplateInvalidError as exc:
        raise HTTPException(status_code=500, detail="Failed to get slide template") from exc
    return _template_to_response(template)


@router.get(
    "/styles",
    response_model=VisualStyleListResponse,
    summary="List visual styles",
    dependencies=[Depends(RequirePermission(MEDIA_READ)), Depends(rbac_rate_limit("slides.styles.list"))],
)
async def list_visual_styles(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: SlidesDatabase = Depends(get_slides_db_for_user),
) -> VisualStyleListResponse:
    builtin_presets = list_builtin_visual_styles()
    builtin_total = len(builtin_presets)
    builtin_slice = builtin_presets[offset : offset + limit]
    remaining = limit - len(builtin_slice)
    user_offset = max(offset - builtin_total, 0)
    user_rows: list[VisualStyleRow] = []
    if remaining > 0:
        user_rows, _ = db.list_visual_styles(limit=remaining, offset=user_offset)
    total_count = builtin_total + db.count_visual_styles()
    styles = [
        *(
            _visual_style_response_from_builtin(resolved)
            for resolved in (
                resolve_builtin_visual_style(style.style_id, include_custom_css=False) for style in builtin_slice
            )
            if resolved is not None
        ),
        *(_visual_style_response_from_row(row) for row in user_rows),
    ]
    return VisualStyleListResponse(
        styles=styles,
        total_count=total_count,
        limit=limit,
        offset=offset,
        pagination=build_offset_pagination_meta(
            limit=limit,
            offset=offset,
            total=total_count,
            count=len(styles),
        ),
    )


@router.get(
    "/styles/{style_id}",
    response_model=VisualStyleResponse,
    summary="Get visual style",
    dependencies=[Depends(RequirePermission(MEDIA_READ)), Depends(rbac_rate_limit("slides.styles.get"))],
)
async def get_visual_style(
    style_id: str,
    db: SlidesDatabase = Depends(get_slides_db_for_user),
) -> VisualStyleResponse:
    return _resolve_visual_style_response(style_id, db)


@router.post(
    "/styles",
    response_model=VisualStyleResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create visual style",
    dependencies=[Depends(RequirePermission(MEDIA_CREATE)), Depends(rbac_rate_limit("slides.styles.create"))],
)
async def create_visual_style(
    request: VisualStyleCreateRequest,
    db: SlidesDatabase = Depends(get_slides_db_for_user),
) -> VisualStyleResponse:
    name = request.name.strip()
    if not name:
        raise HTTPException(status_code=422, detail="visual_style_name_required")
    row = db.create_visual_style(
        name=name,
        scope="user",
        style_payload=_serialize_visual_style_payload(
            description=request.description,
            generation_rules=request.generation_rules,
            artifact_preferences=request.artifact_preferences,
            appearance_defaults=request.appearance_defaults,
            fallback_policy=request.fallback_policy,
        ),
    )
    return _visual_style_response_from_row(row)


@router.patch(
    "/styles/{style_id}",
    response_model=VisualStyleResponse,
    summary="Patch visual style",
    dependencies=[Depends(RequirePermission(MEDIA_UPDATE)), Depends(rbac_rate_limit("slides.styles.update"))],
)
async def patch_visual_style(
    style_id: str,
    request: VisualStylePatchRequest,
    db: SlidesDatabase = Depends(get_slides_db_for_user),
) -> VisualStyleResponse:
    if get_builtin_visual_style(style_id) is not None:
        raise HTTPException(status_code=403, detail="builtin_visual_style_read_only")
    try:
        existing = db.get_visual_style_by_id(style_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="visual_style_not_found") from None
    payload = _deserialize_visual_style_payload(existing.style_payload)
    merged_description = request.description if _field_was_set(request, "description") else payload.get("description")
    merged_generation_rules = (
        request.generation_rules
        if _field_was_set(request, "generation_rules")
        else payload.get("generation_rules") or {}
    )
    if merged_generation_rules is None:
        merged_generation_rules = {}
    merged_artifact_preferences = (
        request.artifact_preferences
        if _field_was_set(request, "artifact_preferences")
        else payload.get("artifact_preferences") or []
    )
    if merged_artifact_preferences is None:
        merged_artifact_preferences = []
    merged_appearance_defaults = (
        request.appearance_defaults
        if _field_was_set(request, "appearance_defaults")
        else payload.get("appearance_defaults") or {}
    )
    if merged_appearance_defaults is None:
        merged_appearance_defaults = {}
    merged_fallback_policy = (
        request.fallback_policy if _field_was_set(request, "fallback_policy") else payload.get("fallback_policy") or {}
    )
    if merged_fallback_policy is None:
        merged_fallback_policy = {}
    name = request.name.strip() if _field_was_set(request, "name") and isinstance(request.name, str) else existing.name
    if not name:
        raise HTTPException(status_code=422, detail="visual_style_name_required")
    if not any(
        _field_was_set(request, field_name)
        for field_name in {
            "name",
            "description",
            "generation_rules",
            "artifact_preferences",
            "appearance_defaults",
            "fallback_policy",
        }
    ):
        raise HTTPException(status_code=400, detail="no_fields_to_update")
    try:
        row = db.update_visual_style(
            style_id=style_id,
            name=name,
            style_payload=_serialize_visual_style_payload(
                description=(
                    merged_description if isinstance(merged_description, str) or merged_description is None else None
                ),
                generation_rules=merged_generation_rules if isinstance(merged_generation_rules, dict) else {},
                artifact_preferences=(
                    [str(item) for item in merged_artifact_preferences]
                    if isinstance(merged_artifact_preferences, list)
                    else []
                ),
                appearance_defaults=merged_appearance_defaults if isinstance(merged_appearance_defaults, dict) else {},
                fallback_policy=merged_fallback_policy if isinstance(merged_fallback_policy, dict) else {},
            ),
            expected_updated_at=existing.updated_at,
        )
    except KeyError:
        raise HTTPException(status_code=404, detail="visual_style_not_found") from None
    except ConflictError as exc:
        raise map_db_error_to_http(
            exc,
            conflict_detail="visual_style_version_conflict",
        ) from exc
    return _visual_style_response_from_row(row)


@router.delete(
    "/styles/{style_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete visual style",
    dependencies=[Depends(RequirePermission(MEDIA_DELETE)), Depends(rbac_rate_limit("slides.styles.delete"))],
)
async def delete_visual_style(
    style_id: str,
    db: SlidesDatabase = Depends(get_slides_db_for_user),
) -> Response:
    if get_builtin_visual_style(style_id) is not None:
        raise HTTPException(status_code=403, detail="builtin_visual_style_read_only")
    try:
        deleted = db.delete_visual_style(style_id)
    except ConflictError as exc:
        raise map_db_error_to_http(
            exc,
            conflict_detail="visual_style_in_use",
        ) from exc
    if not deleted:
        raise HTTPException(status_code=404, detail="visual_style_not_found")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get(
    "/presentations/{presentation_id}/versions",
    response_model=PresentationVersionListResponse,
    summary="List presentation versions",
    dependencies=[Depends(RequirePermission(MEDIA_READ)), Depends(rbac_rate_limit("slides.versions.list"))],
)
async def list_presentation_versions(
    presentation_id: str,
    response: Response,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    accept_content_kinds: str | None = Header(None, alias=CONTENT_KIND_HEADER),
    db: SlidesDatabase = Depends(get_slides_db_for_user),
) -> PresentationVersionListResponse:
    accepted = _accepted_content_kinds(accept_content_kinds, response)
    try:
        kind = PresentationService(db).guard_target(presentation_id, accepted, include_deleted=True)
    except KeyError:
        raise HTTPException(status_code=404, detail="presentation_not_found") from None
    except PresentationServiceError as exc:
        raise _map_presentation_service_error(exc) from exc
    rows, total = db.list_presentation_version_metadata(presentation_id=presentation_id, limit=limit, offset=offset)
    additive = STANDALONE_HTML in accepted
    versions: list[dict[str, Any]] = []
    for row in rows:
        summary: dict[str, Any] = {
            "presentation_id": row.presentation_id,
            "version": row.version,
            "created_at": row.created_at,
            "title": row.title,
            "deleted": None if row.deleted is None else bool(row.deleted),
        }
        if additive:
            summary["content_kind"] = kind.content_kind
        versions.append(summary)
    return PresentationVersionListResponse(
        versions=versions,
        total=total,
        limit=limit,
        offset=offset,
        pagination=build_offset_pagination_meta(
            limit=limit,
            offset=offset,
            total=total,
            count=len(versions),
        ),
    )


@router.get(
    "/presentations/{presentation_id}/versions/{version}",
    response_model=PresentationResponse,
    summary="Get presentation version",
    dependencies=[Depends(RequirePermission(MEDIA_READ)), Depends(rbac_rate_limit("slides.versions.get"))],
)
async def get_presentation_version(
    presentation_id: str,
    version: int,
    response: Response,
    accept_content_kinds: str | None = Header(None, alias=CONTENT_KIND_HEADER),
    db: SlidesDatabase = Depends(get_slides_db_for_user),
) -> PresentationResponse:
    accepted = _accepted_content_kinds(accept_content_kinds, response)
    try:
        kind = PresentationService(db).guard_target(presentation_id, accepted, include_deleted=True)
        row = db.get_presentation_version(presentation_id=presentation_id, version=version)
    except KeyError:
        raise HTTPException(status_code=404, detail="presentation_version_not_found") from None
    except PresentationServiceError as exc:
        raise _map_presentation_service_error(exc) from exc
    payload = _load_version_payload(row.payload_json)
    payload_kind = payload.get("content_kind", "structured_slides")
    if payload_kind != kind.content_kind:
        raise HTTPException(status_code=409, detail="version_content_kind_mismatch")
    response.headers["ETag"] = _format_etag(version, kind.content_kind)
    if kind.content_kind == STANDALONE_HTML:
        response.headers["Cache-Control"] = "private, no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
    return _payload_to_presentation(payload, additive=STANDALONE_HTML in accepted)


@router.post(
    "/presentations/{presentation_id}/versions/{version}/restore",
    response_model=PresentationResponse,
    summary="Restore presentation to a previous version",
    dependencies=[Depends(RequirePermission(MEDIA_UPDATE)), Depends(rbac_rate_limit("slides.versions.restore"))],
)
async def restore_presentation_version(
    presentation_id: str,
    version: int,
    response: Response,
    if_match: str | None = Header(None, alias="If-Match"),
    accept_content_kinds: str | None = Header(None, alias=CONTENT_KIND_HEADER),
    db: SlidesDatabase = Depends(get_slides_db_for_user),
    validation_pool: StandaloneHtmlValidationPool = Depends(_get_standalone_html_validation_pool),
) -> PresentationResponse:
    accepted = _accepted_content_kinds(accept_content_kinds, response)
    service = PresentationService(db, validation_pool=validation_pool)
    try:
        kind = service.guard_target(presentation_id, accepted, include_deleted=True)
        service.require_operation(kind.content_kind, "restore")
        expected_version = _parse_etag(if_match)
        row = await service.restore_version(
            presentation_id=presentation_id,
            version=version,
            expected_version=expected_version,
            structured_restore=lambda payload: _restore_structured_snapshot(
                db=db,
                presentation_id=presentation_id,
                payload=dict(payload),
                expected_version=expected_version,
            ),
        )
    except KeyError:
        raise HTTPException(status_code=404, detail="presentation_version_not_found") from None
    except InputError as exc:
        raise map_db_error_to_http(exc, default_detail="Failed to restore presentation version") from exc
    except ConflictError as exc:
        raise _map_precondition_conflict(exc) from exc
    except StandaloneHtmlValidationError as exc:
        raise _map_standalone_validation_error(exc) from exc
    except PresentationServiceError as exc:
        raise _map_presentation_service_error(exc) from exc
    response.headers["ETag"] = _format_etag(row.version, row.content_kind)
    response.headers["Last-Modified"] = row.last_modified
    if row.content_kind == STANDALONE_HTML:
        response.headers["Cache-Control"] = "private, no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
    return _build_presentation_response(row, additive=STANDALONE_HTML in accepted)


@router.post(
    "/presentations/{presentation_id}/render-jobs",
    response_model=PresentationRenderJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit a presentation render job",
    dependencies=[Depends(RequirePermission(MEDIA_UPDATE)), Depends(rbac_rate_limit("slides.render.submit"))],
)
async def submit_presentation_render_job(
    presentation_id: str,
    request: PresentationRenderRequest,
    response: Response,
    if_match: str | None = Header(None, alias="If-Match"),
    accept_content_kinds: str | None = Header(None, alias=CONTENT_KIND_HEADER),
    db: SlidesDatabase = Depends(get_slides_db_for_user),
    current_user: User = Depends(get_request_user),
    job_manager: JobManager = Depends(_slides_jobs_manager),
) -> PresentationRenderJobResponse:
    accepted = _accepted_content_kinds(accept_content_kinds, response)
    service = PresentationService(db)
    try:
        kind = service.guard_target(presentation_id, accepted)
        service.require_operation(kind.content_kind, "render")
    except KeyError:
        raise HTTPException(status_code=404, detail="presentation_not_found") from None
    except PresentationServiceError as exc:
        raise _map_presentation_service_error(exc) from exc
    if not _render_enabled():
        raise HTTPException(status_code=503, detail="presentation_render_unavailable")
    expected_version = _parse_etag(if_match)
    try:
        row = db.get_presentation_by_id(presentation_id, include_deleted=False)
    except KeyError:
        raise HTTPException(status_code=404, detail="presentation_not_found") from None
    if int(row.version) != expected_version:
        raise HTTPException(status_code=412, detail="precondition_failed")

    render_format = str(request.format.value if hasattr(request.format, "value") else request.format)
    payload = {
        "user_id": int(current_user.id),
        "presentation_id": presentation_id,
        "presentation_version": int(row.version),
        "format": render_format,
        "theme": row.theme,
        "title": row.title,
    }
    job = await asyncio.to_thread(
        job_manager.create_job,
        domain="presentation_render",
        queue=_presentation_render_queue_name(),
        job_type="presentation_render",
        payload=payload,
        owner_user_id=str(current_user.id),
        priority=5,
        max_retries=2,
    )
    return PresentationRenderJobResponse(
        job_id=int(job["id"]),
        status=_normalize_job_status(job.get("status")),
        job_type="presentation_render",
        presentation_id=presentation_id,
        presentation_version=int(row.version),
        format=PresentationRenderFormat(render_format),
    )


@router.get(
    "/render-jobs/{job_id}",
    response_model=PresentationRenderJobStatusResponse,
    summary="Get presentation render job status",
    dependencies=[Depends(RequirePermission(MEDIA_READ)), Depends(rbac_rate_limit("slides.render.status"))],
)
async def get_presentation_render_job_status(
    job_id: int,
    current_user: User = Depends(get_request_user),
    job_manager: JobManager = Depends(_slides_jobs_manager),
) -> PresentationRenderJobStatusResponse:
    job = await asyncio.to_thread(job_manager.get_job, int(job_id))
    if not job:
        raise HTTPException(status_code=404, detail="job_not_found")
    if str(job.get("owner_user_id") or "") != str(current_user.id):
        raise HTTPException(status_code=404, detail="job_not_found")

    payload = job.get("payload") if isinstance(job.get("payload"), dict) else {}
    result = job.get("result") if isinstance(job.get("result"), dict) else {}
    render_format = payload.get("format")
    format_value = PresentationRenderFormat(render_format) if render_format in {"mp4", "webm"} else None
    error_text = None
    for key in ("last_error", "error_message", "error_code"):
        if job.get(key):
            error_text = str(job.get(key))
            break

    return PresentationRenderJobStatusResponse(
        job_id=int(job["id"]),
        status=_normalize_job_status(job.get("status")),
        job_type=str(job.get("job_type") or "presentation_render"),
        presentation_id=payload.get("presentation_id"),
        presentation_version=payload.get("presentation_version"),
        format=format_value,
        output_id=result.get("output_id"),
        download_url=result.get("download_url"),
        error=error_text,
    )


@router.get(
    "/presentations/{presentation_id}/render-artifacts",
    response_model=PresentationRenderArtifactListResponse,
    summary="List presentation render artifacts",
    dependencies=[Depends(RequirePermission(MEDIA_READ)), Depends(rbac_rate_limit("slides.render.artifacts"))],
)
async def list_presentation_render_artifacts(
    presentation_id: str,
    response: Response,
    accept_content_kinds: str | None = Header(None, alias=CONTENT_KIND_HEADER),
    db: SlidesDatabase = Depends(get_slides_db_for_user),
    collections_db: CollectionsDatabase = Depends(get_collections_db_for_user),
) -> PresentationRenderArtifactListResponse:
    accepted = _accepted_content_kinds(accept_content_kinds, response)
    service = PresentationService(db)
    try:
        kind = service.guard_target(presentation_id, accepted)
        service.require_operation(kind.content_kind, "render")
    except KeyError:
        raise HTTPException(status_code=404, detail="presentation_not_found") from None
    except PresentationServiceError as exc:
        raise _map_presentation_service_error(exc) from exc
    artifacts: list[PresentationRenderArtifactInfo] = []
    page_size = 200
    offset = 0
    total = 1
    while offset < total:
        rows, total = await asyncio.to_thread(
            collections_db.list_output_artifacts,
            limit=page_size,
            offset=offset,
            type_="presentation_render",
            metadata_origin="presentation_studio",
            metadata_presentation_id=presentation_id,
        )
        for row in rows:
            metadata = _safe_json_dict(getattr(row, "metadata_json", None))
            fmt = str(getattr(row, "format", "") or "").lower()
            if fmt not in {"mp4", "webm"}:
                continue
            created_at = getattr(row, "created_at", None)
            artifacts.append(
                PresentationRenderArtifactInfo(
                    output_id=int(row.id),
                    format=PresentationRenderFormat(fmt),
                    title=getattr(row, "title", None),
                    download_url=f"/api/v1/outputs/{int(row.id)}/download",
                    presentation_version=metadata.get("presentation_version"),
                    created_at=_normalize_dt(created_at) if isinstance(created_at, str) else None,
                )
            )
        offset += page_size
    return PresentationRenderArtifactListResponse(
        presentation_id=presentation_id,
        artifacts=artifacts,
    )


@router.post(
    "/generate",
    response_model=PresentationResponse,
    summary="Generate slides from prompt",
    dependencies=[Depends(RequirePermission(MEDIA_CREATE)), Depends(rbac_rate_limit("slides.generate"))],
)
async def generate_from_prompt(
    request: GenerateFromPromptRequest,
    response: Response,
    db: SlidesDatabase = Depends(get_slides_db_for_user),
) -> PresentationResponse:
    prompt = request.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=422, detail="prompt_required")
    return await _generate_presentation(
        response=response,
        db=db,
        request=request,
        source_text=prompt,
        source_type="prompt",
        source_ref=None,
        source_query=request.prompt,
    )


@router.post(
    "/generate/from-chat",
    response_model=PresentationResponse,
    summary="Generate slides from chat conversation",
    dependencies=[Depends(RequirePermission(MEDIA_CREATE)), Depends(rbac_rate_limit("slides.generate"))],
)
async def generate_from_chat(
    request: GenerateFromChatRequest,
    response: Response,
    http_request: Request,
    db: SlidesDatabase = Depends(get_slides_db_for_user),
    current_user: User = Depends(get_request_user),
) -> PresentationResponse:
    conversation_id = request.conversation_id.strip()
    if not conversation_id:
        raise HTTPException(status_code=422, detail="conversation_id_required")
    notes_db = await _resolve_notes_db_for_request(http_request, current_user)
    conversation = notes_db.get_conversation_by_id(conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="conversation_not_found")
    messages = notes_db.get_messages_for_conversation(conversation_id, limit=500, offset=0, order_by_timestamp="ASC")
    source_text = _format_chat_messages(messages)
    if not source_text:
        raise HTTPException(status_code=404, detail="conversation_empty")
    return await _generate_presentation(
        response=response,
        db=db,
        request=request,
        source_text=source_text,
        source_type="chat",
        source_ref=conversation_id,
        source_query=None,
    )


@router.post(
    "/generate/from-media",
    response_model=PresentationResponse,
    summary="Generate slides from media transcript",
    dependencies=[Depends(RequirePermission(MEDIA_CREATE)), Depends(rbac_rate_limit("slides.generate"))],
)
async def generate_from_media(
    request: GenerateFromMediaRequest,
    response: Response,
    db: SlidesDatabase = Depends(get_slides_db_for_user),
    media_db: MediaDbSession = Depends(get_media_db_for_user),
) -> PresentationResponse:
    try:
        media_id = int(request.media_id)
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail="media_id_invalid") from exc
    media_row = media_db.get_media_by_id(media_id, include_deleted=False, include_trash=False)
    if not media_row:
        raise HTTPException(status_code=404, detail="media_not_found")
    source_text = _resolve_media_source_text(
        media_db=media_db,
        media_row=media_row,
        media_id=media_id,
    )
    if not source_text:
        media_type = str(media_row.get("type") or "").strip().lower()
        detail = "media_transcript_not_found" if media_type in {"", "audio", "video"} else "media_content_not_found"
        raise HTTPException(status_code=404, detail=detail)
    return await _generate_presentation(
        response=response,
        db=db,
        request=request,
        source_text=source_text,
        source_type="media",
        source_ref=str(media_id),
        source_query=None,
    )


@router.post(
    "/generate/from-notes",
    response_model=PresentationResponse,
    summary="Generate slides from notes",
    dependencies=[Depends(RequirePermission(MEDIA_CREATE)), Depends(rbac_rate_limit("slides.generate"))],
)
async def generate_from_notes(
    request: GenerateFromNotesRequest,
    response: Response,
    http_request: Request,
    db: SlidesDatabase = Depends(get_slides_db_for_user),
    current_user: User = Depends(get_request_user),
) -> PresentationResponse:
    if not request.note_ids:
        raise HTTPException(status_code=422, detail="note_ids_required")
    notes_db = await _resolve_notes_db_for_request(http_request, current_user)
    notes: list[dict[str, Any]] = []
    missing: list[str] = []
    for note_id in request.note_ids:
        note = notes_db.get_note_by_id(note_id)
        if not note:
            missing.append(note_id)
            continue
        notes.append(note)
    if missing:
        raise HTTPException(status_code=404, detail={"missing_note_ids": missing})
    source_text = _format_notes(notes)
    if not source_text:
        raise HTTPException(status_code=404, detail="notes_empty")
    return await _generate_presentation(
        response=response,
        db=db,
        request=request,
        source_text=source_text,
        source_type="notes",
        source_ref=request.note_ids,
        source_query=None,
    )


@router.post(
    "/generate/from-rag",
    response_model=PresentationResponse,
    summary="Generate slides from RAG results",
    dependencies=[Depends(RequirePermission(MEDIA_CREATE)), Depends(rbac_rate_limit("slides.generate"))],
)
async def generate_from_rag(
    request: GenerateFromRagRequest,
    response: Response,
    db: SlidesDatabase = Depends(get_slides_db_for_user),
) -> PresentationResponse:
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=422, detail="query_required")
    try:
        user_id = int(db.client_id)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=500, detail="user_unavailable") from exc
    rag_result = await unified_rag_pipeline(
        query=query,
        top_k=request.top_k or 8,
        sources=["media_db", "notes", "chats"],
        media_db_path=str(DatabasePaths.get_media_db_path(user_id)),
        notes_db_path=str(DatabasePaths.get_chacha_db_path(user_id)),
        character_db_path=str(DatabasePaths.get_chacha_db_path(user_id)),
    )
    documents = rag_result.documents if hasattr(rag_result, "documents") else []
    source_text = _format_rag_documents(documents)
    if not source_text and hasattr(rag_result, "generated_answer") and rag_result.generated_answer:
        source_text = str(rag_result.generated_answer)
    if not source_text:
        raise HTTPException(status_code=404, detail="rag_no_results")
    return await _generate_presentation(
        response=response,
        db=db,
        request=request,
        source_text=source_text,
        source_type="rag",
        source_ref=None,
        source_query=query,
    )


@router.get(
    "/presentations/{presentation_id}/export",
    response_class=Response,
    responses={
        200: {
            "description": "Presentation export download.",
            "content": {
                "application/json": {},
                "application/pdf": {},
                "application/zip": {},
                "text/markdown": {},
            },
        },
    },
    summary="Export presentation",
    dependencies=[Depends(RequirePermission(MEDIA_READ)), Depends(rbac_rate_limit("slides.export"))],
)
async def export_presentation(
    presentation_id: str,
    format: ExportFormat = Query(ExportFormat.REVEAL),
    pdf_format: str | None = Query(None),
    pdf_width: str | None = Query(None),
    pdf_height: str | None = Query(None),
    pdf_landscape: bool | None = Query(None),
    pdf_margin_top: str | None = Query(None),
    pdf_margin_bottom: str | None = Query(None),
    pdf_margin_left: str | None = Query(None),
    pdf_margin_right: str | None = Query(None),
    accept_content_kinds: str | None = Header(None, alias=CONTENT_KIND_HEADER),
    db: SlidesDatabase = Depends(get_slides_db_for_user),
    collections_db: CollectionsDatabase = Depends(get_collections_db_for_user),
    validation_pool: StandaloneHtmlValidationPool = Depends(_get_standalone_html_validation_pool),
) -> Response:
    accepted = _accepted_content_kinds(accept_content_kinds)
    service = PresentationService(db, validation_pool=validation_pool)
    try:
        kind = service.guard_target(presentation_id, accepted)
        service.require_operation(
            kind.content_kind,
            "export",
            export_format=str(format.value),
        )
        if format == ExportFormat.HTML:
            raise service.operation_not_supported(kind.content_kind, "export")
        row = db.get_presentation_by_id(presentation_id, include_deleted=False)
        if row.content_kind == STANDALONE_HTML and format == ExportFormat.JSON:
            await service.validate_saved_standalone(row)
    except KeyError:
        raise HTTPException(status_code=404, detail="presentation_not_found") from None
    except StandaloneHtmlValidationError as exc:
        raise _map_standalone_validation_error(exc) from exc
    except PresentationServiceError as exc:
        raise _map_presentation_service_error(exc) from exc

    if row.content_kind == STANDALONE_HTML:
        payload = jsonable_encoder(_build_presentation_response(row, additive=True))
        body = export_presentation_json(payload).encode("utf-8")
        headers = {
            "Content-Disposition": f'attachment; filename="presentation_{presentation_id}.json"',
            "Cache-Control": "private, no-store",
            "X-Content-Type-Options": "nosniff",
        }
        merge_vary_header(headers)
        return Response(content=body, media_type="application/json", headers=headers)

    slides_raw = json.loads(row.slides)
    slides = [_slide_from_obj(item) for item in slides_raw]
    slides = _normalize_slides(slides)
    settings = _deserialize_settings(row.settings)
    visual_style_snapshot = _deserialize_visual_style_snapshot(getattr(row, "visual_style_snapshot", None))
    try:
        user_id = int(db.client_id)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=500, detail="user_unavailable") from exc

    def _asset_resolver(asset_ref: str) -> dict[str, Any]:
        return resolve_slide_asset(
            asset_ref,
            collections_db=collections_db,
            user_id=user_id,
            max_bytes=MAX_RESOLVED_SLIDE_ASSET_BYTES,
        )

    try:
        metrics = get_metrics_registry()
    except _SLIDES_NONCRITICAL_EXCEPTIONS:
        metrics = None
    started_at = time.perf_counter()

    if format == ExportFormat.JSON:
        payload = jsonable_encoder(_build_presentation_response(row, additive=STANDALONE_HTML in accepted))
        body = export_presentation_json(payload).encode("utf-8")
        filename = f"presentation_{presentation_id}.json"
        media_type = "application/json"
    elif format == ExportFormat.MARKDOWN:
        try:
            markdown_text = await asyncio.to_thread(
                export_presentation_markdown,
                title=row.title,
                slides=slides,
                theme=row.theme,
                marp_theme=getattr(row, "marp_theme", None),
                asset_resolver=_asset_resolver,
            )
            body = markdown_text.encode("utf-8")
        except SlidesExportInputError as exc:
            if metrics is not None:
                with contextlib.suppress(_SLIDES_NONCRITICAL_EXCEPTIONS):
                    metrics.increment(
                        "slides_export_errors_total",
                        labels={"format": format.value, "error": "input_error"},
                    )
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except SlidesExportError as exc:
            if metrics is not None:
                with contextlib.suppress(_SLIDES_NONCRITICAL_EXCEPTIONS):
                    metrics.increment(
                        "slides_export_errors_total",
                        labels={"format": format.value, "error": "export_error"},
                    )
            raise HTTPException(
                status_code=500,
                detail="Failed to export presentation as markdown",
            ) from exc
        filename = f"presentation_{presentation_id}.md"
        media_type = "text/markdown"
    elif format == ExportFormat.PDF:
        pdf_options = {
            "format": pdf_format,
            "width": pdf_width,
            "height": pdf_height,
            "landscape": pdf_landscape,
            "margin": {
                "top": pdf_margin_top,
                "bottom": pdf_margin_bottom,
                "left": pdf_margin_left,
                "right": pdf_margin_right,
            },
        }
        try:
            body = await asyncio.to_thread(
                export_presentation_pdf,
                title=row.title,
                slides=slides,
                theme=row.theme,
                settings=settings,
                custom_css=row.custom_css,
                visual_style_snapshot=visual_style_snapshot,
                pdf_options=pdf_options,
                asset_resolver=_asset_resolver,
            )
        except SlidesExportInputError as exc:
            if metrics is not None:
                with contextlib.suppress(_SLIDES_NONCRITICAL_EXCEPTIONS):
                    metrics.increment(
                        "slides_export_errors_total",
                        labels={"format": format.value, "error": "input_error"},
                    )
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except SlidesExportError as exc:
            if metrics is not None:
                with contextlib.suppress(_SLIDES_NONCRITICAL_EXCEPTIONS):
                    metrics.increment(
                        "slides_export_errors_total",
                        labels={"format": format.value, "error": "export_error"},
                    )
            raise HTTPException(
                status_code=500,
                detail="Failed to export presentation as pdf",
            ) from exc
        filename = f"presentation_{presentation_id}.pdf"
        media_type = "application/pdf"
    elif format == ExportFormat.REVEAL:
        try:
            body = await asyncio.to_thread(
                export_presentation_bundle,
                title=row.title,
                slides=slides,
                theme=row.theme,
                settings=settings,
                custom_css=row.custom_css,
                visual_style_snapshot=visual_style_snapshot,
                asset_resolver=_asset_resolver,
            )
        except SlidesAssetsMissingError as exc:
            if metrics is not None:
                with contextlib.suppress(_SLIDES_NONCRITICAL_EXCEPTIONS):
                    metrics.increment(
                        "slides_export_errors_total",
                        labels={"format": format.value, "error": "assets_missing"},
                    )
            raise HTTPException(status_code=500, detail="slides_assets_missing") from exc
        except SlidesExportInputError as exc:
            if metrics is not None:
                with contextlib.suppress(_SLIDES_NONCRITICAL_EXCEPTIONS):
                    metrics.increment(
                        "slides_export_errors_total",
                        labels={"format": format.value, "error": "input_error"},
                    )
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except SlidesExportError as exc:
            if metrics is not None:
                with contextlib.suppress(_SLIDES_NONCRITICAL_EXCEPTIONS):
                    metrics.increment(
                        "slides_export_errors_total",
                        labels={"format": format.value, "error": "export_error"},
                    )
            raise HTTPException(
                status_code=500,
                detail="Failed to export presentation as revealjs",
            ) from exc
        filename = f"presentation_{presentation_id}.zip"
        media_type = "application/zip"
    else:
        if metrics is not None:
            with contextlib.suppress(_SLIDES_NONCRITICAL_EXCEPTIONS):
                metrics.increment(
                    "slides_export_errors_total",
                    labels={"format": str(format), "error": "invalid_format"},
                )
        raise HTTPException(status_code=400, detail="invalid_export_format")

    if metrics is not None:
        with contextlib.suppress(_SLIDES_NONCRITICAL_EXCEPTIONS):
            metrics.observe(
                "slides_export_latency_seconds",
                time.perf_counter() - started_at,
                labels={"format": format.value},
            )

    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
    merge_vary_header(headers)
    if row.content_kind == STANDALONE_HTML:
        headers["Cache-Control"] = "private, no-store"
        headers["X-Content-Type-Options"] = "nosniff"
    return Response(content=body, media_type=media_type, headers=headers)


@router.get(
    "/health",
    summary="Slides health check",
    response_model=SlidesHealthResponse,
    dependencies=[Depends(rbac_rate_limit("slides.health"))],
)
async def slides_health(db: SlidesDatabase = Depends(get_slides_db_for_user)) -> SlidesHealthResponse:
    try:
        db.probe_health()
    except _SLIDES_HEALTH_EXCEPTIONS as exc:
        logger.warning("slides health check failed")
        raise HTTPException(status_code=500, detail="slides_db_unavailable") from exc
    return SlidesHealthResponse(service="slides", status="ok")
