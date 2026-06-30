"""API routes for canonical browser web clipper saves and enrichments."""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from loguru import logger
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_rate_limiter_dep, get_request_user, RateLimiter, User

from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import try_get_media_db_for_user
from tldw_Server_API.app.api.v1.API_Deps.jobs_deps import try_get_job_manager
from tldw_Server_API.app.api.v1.schemas.web_clipper_schemas import (
    WebClipperEnrichmentPayload,
    WebClipperEnrichmentResponse,
    WebClipperSaveRequest,
    WebClipperSaveResponse,
    WebClipperStatusResponse,
)
from tldw_Server_API.app.api.v1.utils.http_errors import map_db_error_to_http
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDB,
    CharactersRAGDBError,
    ConflictError,
    InputError,
)
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.WebClipper.service import WebClipperService
from tldw_Server_API.app.core.Workspaces.source_jobs import enqueue_workspace_source_ingest_job

router = APIRouter(tags=["web-clipper"])

_WEB_CLIPPER_ENDPOINT_EXCEPTIONS = (
    CharactersRAGDBError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)


async def _check_rate_limit(
    *,
    rate_limiter: RateLimiter,
    current_user: User,
    scope: str,
) -> None:
    try:
        allowed, meta = await rate_limiter.check_user_rate_limit(int(current_user.id), scope)
    except Exception as exc:
        logger.error("Web clipper rate limiter unavailable")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Rate limiter unavailable",
        ) from exc
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded for {scope}",
            headers={"Retry-After": str(meta.get("retry_after", 60))},
        )


@router.post("/save", response_model=WebClipperSaveResponse, summary="Save a browser clip")
async def save_web_clip(
    request: Request,
    payload: WebClipperSaveRequest,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    media_db: Any | None = Depends(try_get_media_db_for_user),
    jm: JobManager | None = Depends(try_get_job_manager),
    rate_limiter: RateLimiter = Depends(get_rate_limiter_dep),
    current_user: User = Depends(get_request_user),
) -> WebClipperSaveResponse:
    _ = request
    await _check_rate_limit(rate_limiter=rate_limiter, current_user=current_user, scope="web_clipper.save")
    service = WebClipperService(
        db=db,
        media_db=media_db,
        user_id=current_user.id,
        promote_workspace_sources=True,
        workspace_source_job_enqueuer=lambda workspace_id, src: enqueue_workspace_source_ingest_job(
            jm=jm,
            owner_user_id=current_user.id,
            workspace_id=workspace_id,
            src=src,
        ),
    )
    try:
        result = await asyncio.to_thread(service.save_clip, payload)
        if result.status == "failed":
            logger.error("Web clipper canonical save failed")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Canonical note save failed.",
            )
        return result
    except (InputError, ConflictError, CharactersRAGDBError) as exc:
        if isinstance(exc, CharactersRAGDBError) and not isinstance(exc, ConflictError):
            logger.error("Web clipper save failed")
        raise map_db_error_to_http(exc, default_detail="Internal server error") from exc
    except _WEB_CLIPPER_ENDPOINT_EXCEPTIONS as exc:
        logger.error("Web clipper save failed")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error") from exc


@router.get("/{clip_id}", response_model=WebClipperStatusResponse, summary="Get saved clip state")
async def get_web_clip_status(
    clip_id: str,
    request: Request,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    rate_limiter: RateLimiter = Depends(get_rate_limiter_dep),
    current_user: User = Depends(get_request_user),
) -> WebClipperStatusResponse:
    _ = request
    await _check_rate_limit(rate_limiter=rate_limiter, current_user=current_user, scope="web_clipper.status")
    service = WebClipperService(db=db, user_id=current_user.id)
    try:
        return await asyncio.to_thread(service.get_clip_status, clip_id)
    except ConflictError as exc:
        raise map_db_error_to_http(
            exc,
            conflict_status_code=status.HTTP_404_NOT_FOUND,
        ) from exc
    except (InputError, CharactersRAGDBError) as exc:
        if isinstance(exc, CharactersRAGDBError):
            logger.error("Web clipper status failed")
        raise map_db_error_to_http(exc, default_detail="Internal server error") from exc
    except _WEB_CLIPPER_ENDPOINT_EXCEPTIONS as exc:
        logger.error("Web clipper status failed")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error") from exc


@router.post(
    "/{clip_id}/enrichments",
    response_model=WebClipperEnrichmentResponse,
    summary="Persist OCR/VLM enrichment for a saved clip",
)
async def persist_web_clip_enrichment(
    clip_id: str,
    request: Request,
    payload: WebClipperEnrichmentPayload,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    rate_limiter: RateLimiter = Depends(get_rate_limiter_dep),
    current_user: User = Depends(get_request_user),
) -> WebClipperEnrichmentResponse:
    _ = request
    await _check_rate_limit(rate_limiter=rate_limiter, current_user=current_user, scope="web_clipper.enrichment")
    service = WebClipperService(db=db, user_id=current_user.id)
    try:
        return await asyncio.to_thread(service.persist_enrichment, clip_id, payload)
    except ConflictError as exc:
        raise map_db_error_to_http(
            exc,
            conflict_status_code=status.HTTP_404_NOT_FOUND,
        ) from exc
    except (InputError, CharactersRAGDBError) as exc:
        if isinstance(exc, CharactersRAGDBError):
            logger.error("Web clipper enrichment failed")
        raise map_db_error_to_http(exc, default_detail="Internal server error") from exc
    except _WEB_CLIPPER_ENDPOINT_EXCEPTIONS as exc:
        logger.error("Web clipper enrichment failed")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error") from exc
