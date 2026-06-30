from __future__ import annotations

import asyncio
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from threading import BoundedSemaphore
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from loguru import logger
from pydantic import ValidationError

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import (
    RequirePermission,
    User,
    get_request_user,
    rbac_rate_limit,
)
from tldw_Server_API.app.api.v1.schemas.media_playlist_preflight import (
    PlaylistPreflightRequest,
    PlaylistPreflightResponse,
)
from tldw_Server_API.app.core.AuthNZ.permissions import MEDIA_CREATE
from tldw_Server_API.app.core.Ingestion_Media_Processing.Video.playlist_preflight import (
    PlaylistPreflightData,
    preflight_playlist_url,
)


router = APIRouter()

_PLAYLIST_PREFLIGHT_MAX_WORKERS = 2
_PLAYLIST_PREFLIGHT_EXECUTOR = ThreadPoolExecutor(
    max_workers=_PLAYLIST_PREFLIGHT_MAX_WORKERS,
    thread_name_prefix="playlist-preflight",
)
_PLAYLIST_PREFLIGHT_CAPACITY = BoundedSemaphore(value=_PLAYLIST_PREFLIGHT_MAX_WORKERS)
PlaylistPreflightRawResult = PlaylistPreflightResponse | PlaylistPreflightData | Mapping[str, Any]


class PlaylistPreflightResultError(ValueError):
    """Raised when the preflight extractor returns an invalid response contract."""


def _coerce_preflight_response(result: PlaylistPreflightRawResult) -> PlaylistPreflightResponse:
    """Normalize supported extractor outputs into the public response model."""
    if isinstance(result, PlaylistPreflightResponse):
        return result
    try:
        if isinstance(result, PlaylistPreflightData):
            return PlaylistPreflightResponse.model_validate(result.to_dict())
        if isinstance(result, Mapping):
            return PlaylistPreflightResponse.model_validate(dict(result))
    except ValidationError as exc:
        raise PlaylistPreflightResultError("playlist_preflight_invalid_result") from exc
    raise PlaylistPreflightResultError("playlist_preflight_invalid_result")


def _preflight_playlist_url_with_capacity(url: str, *, max_items: int) -> PlaylistPreflightRawResult:
    """Run the blocking extractor and release API capacity after actual thread completion."""
    try:
        return preflight_playlist_url(url, max_items=max_items)
    finally:
        _PLAYLIST_PREFLIGHT_CAPACITY.release()


async def _run_preflight_with_timeout(payload: PlaylistPreflightRequest) -> PlaylistPreflightRawResult:
    """Run playlist extraction in a bounded executor with request-level timeout handling."""
    if not _PLAYLIST_PREFLIGHT_CAPACITY.acquire(blocking=False):
        raise HTTPException(status_code=429, detail="playlist_preflight_busy")

    loop = asyncio.get_running_loop()
    try:
        extraction = loop.run_in_executor(
            _PLAYLIST_PREFLIGHT_EXECUTOR,
            partial(
                _preflight_playlist_url_with_capacity,
                payload.url,
                max_items=payload.max_items,
            ),
        )
    except Exception:
        _PLAYLIST_PREFLIGHT_CAPACITY.release()
        raise

    return await asyncio.wait_for(extraction, timeout=payload.timeout_seconds)


@router.post(
    "/playlists/preflight",
    response_model=PlaylistPreflightResponse,
    summary="Preflight a playlist URL without downloading media",
    tags=["Media Playlist Preflight"],
    dependencies=[
        Depends(RequirePermission(MEDIA_CREATE)),
        Depends(rbac_rate_limit("media.create")),
    ],
)
async def preflight_playlist(
    payload: PlaylistPreflightRequest,
    _current_user: User = Depends(get_request_user),
) -> PlaylistPreflightResponse:
    try:
        result = await _run_preflight_with_timeout(payload)
        return _coerce_preflight_response(result)
    except asyncio.TimeoutError as exc:
        raise HTTPException(status_code=504, detail="playlist_preflight_timeout") from exc
    except PlaylistPreflightResultError as exc:
        raise HTTPException(status_code=502, detail="playlist_preflight_invalid_result") from exc
    except ValueError as exc:
        logger.warning("Playlist preflight validation failed for {}: {}", payload.url, exc)
        raise HTTPException(status_code=422, detail="playlist_preflight_invalid_request") from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning("Playlist preflight failed for {}: {}", payload.url, exc)
        raise HTTPException(status_code=502, detail="playlist_preflight_failed") from exc
