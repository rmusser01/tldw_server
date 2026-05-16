from __future__ import annotations

import asyncio
from functools import partial

from fastapi import APIRouter, Depends, HTTPException
from loguru import logger

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


def _coerce_preflight_response(result) -> PlaylistPreflightResponse:
    if isinstance(result, PlaylistPreflightResponse):
        return result
    if isinstance(result, PlaylistPreflightData):
        return PlaylistPreflightResponse.model_validate(result.to_dict())
    if isinstance(result, dict):
        return PlaylistPreflightResponse.model_validate(result)
    raise ValueError("playlist_preflight_invalid_result")


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
    loop = asyncio.get_running_loop()
    try:
        extraction = loop.run_in_executor(
            None,
            partial(
                preflight_playlist_url,
                payload.url,
                max_items=payload.max_items,
            ),
        )
        result = await asyncio.wait_for(extraction, timeout=payload.timeout_seconds)
        return _coerce_preflight_response(result)
    except asyncio.TimeoutError as exc:
        raise HTTPException(status_code=504, detail="playlist_preflight_timeout") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning("Playlist preflight failed for {}: {}", payload.url, exc)
        raise HTTPException(status_code=502, detail="playlist_preflight_failed") from exc
