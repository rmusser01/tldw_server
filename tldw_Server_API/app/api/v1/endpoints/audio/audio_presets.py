"""Audio preset endpoints for reusable TTS/STT configurations."""

from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from starlette import status

from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import (
    TokenScopeGuard,
    User,
    check_rate_limit,
    get_request_user,
)
from tldw_Server_API.app.api.v1.schemas.audio_presets import (
    AudioPresetCreateRequest,
    AudioPresetKind,
    AudioPresetListResponse,
    AudioPresetResponse,
    AudioPresetUpdateRequest,
    AudioPresetValidationResponse,
    AudioPresetValidationWarning,
)
from tldw_Server_API.app.api.v1.utils.http_errors import map_db_error_to_http
from tldw_Server_API.app.core.DB_Management.media_db.errors import (
    ConflictError,
    DatabaseError,
    InputError,
)
from tldw_Server_API.app.core.Utils.pydantic_compat import model_dump_compat

router = APIRouter(
    tags=["Audio"],
    responses={
        401: {"description": "Unauthorized"},
        404: {"description": "Not found"},
        409: {"description": "Conflict"},
        429: {"description": "Rate limit exceeded"},
    },
)

_BROWSER_TTS_WARNING = AudioPresetValidationWarning(
    code="browser_tts_revalidation_required",
    message="Browser TTS presets depend on the current browser and must be revalidated before use.",
    field="config.provider",
)


def _user_id(request_user: User) -> str:
    return str(request_user.id)


def _normalize_audio_preset_config(kind: str, config: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(config or {})
    provider = str(normalized.get("provider") or "").strip().lower()
    if kind == "tts" and provider == "browser":
        normalized["provider"] = "browser"
        normalized["browser_local"] = True
        normalized["requires_browser_revalidation"] = True
    return normalized


def _validation_warnings_for_preset(preset: dict[str, Any]) -> list[AudioPresetValidationWarning]:
    config = preset.get("config")
    if not isinstance(config, dict):
        return []
    provider = str(config.get("provider") or "").strip().lower()
    if preset.get("kind") == "tts" and provider == "browser":
        return [_BROWSER_TTS_WARNING]
    return []


def _preset_or_404(preset: dict[str, Any] | None) -> dict[str, Any]:
    if preset is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Audio preset not found")
    return preset


@router.get(
    "/presets",
    response_model=AudioPresetListResponse,
    summary="List reusable audio presets.",
    dependencies=[
        Depends(check_rate_limit),
        Depends(TokenScopeGuard("any", require_if_present=True, endpoint_id="audio.presets.list", count_as="call")),
    ],
)
async def list_audio_presets_endpoint(
    kind: Optional[AudioPresetKind] = Query(default=None),
    favorite: Optional[bool] = Query(default=None),
    is_default: Optional[bool] = Query(default=None),
    limit: int = Query(default=100, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    request_user: User = Depends(get_request_user),
    media_db: Any = Depends(get_media_db_for_user),
) -> AudioPresetListResponse:
    user_id = _user_id(request_user)
    try:
        items = media_db.list_audio_presets(
            user_id=user_id,
            kind=kind,
            favorite=favorite,
            is_default=is_default,
            limit=limit,
            offset=offset,
        )
        total = media_db.count_audio_presets(
            user_id=user_id,
            kind=kind,
            favorite=favorite,
            is_default=is_default,
        )
    except DatabaseError as exc:
        raise map_db_error_to_http(exc, default_detail="Failed to list audio presets") from exc
    return AudioPresetListResponse(items=items, total=total, limit=limit, offset=offset)


@router.post(
    "/presets",
    response_model=AudioPresetResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a reusable audio preset.",
    dependencies=[
        Depends(check_rate_limit),
        Depends(TokenScopeGuard("any", require_if_present=True, endpoint_id="audio.presets.create", count_as="call")),
    ],
)
async def create_audio_preset_endpoint(
    payload: AudioPresetCreateRequest,
    request_user: User = Depends(get_request_user),
    media_db: Any = Depends(get_media_db_for_user),
) -> dict[str, Any]:
    data = model_dump_compat(payload)
    data["config"] = _normalize_audio_preset_config(str(data["kind"]), data.get("config") or {})
    try:
        return media_db.create_audio_preset(
            user_id=_user_id(request_user),
            kind=str(data["kind"]),
            name=str(data["name"]),
            description=data.get("description"),
            favorite=bool(data.get("favorite", False)),
            is_default=bool(data.get("is_default", False)),
            config=data.get("config") or {},
            capability_assumptions=data.get("capability_assumptions") or {},
        )
    except (ConflictError, InputError, DatabaseError) as exc:
        raise map_db_error_to_http(exc, default_detail="Failed to create audio preset") from exc


@router.patch(
    "/presets/{preset_id}",
    response_model=AudioPresetResponse,
    summary="Update a reusable audio preset.",
    dependencies=[
        Depends(check_rate_limit),
        Depends(TokenScopeGuard("any", require_if_present=True, endpoint_id="audio.presets.update", count_as="call")),
    ],
)
async def update_audio_preset_endpoint(
    preset_id: str,
    payload: AudioPresetUpdateRequest,
    request_user: User = Depends(get_request_user),
    media_db: Any = Depends(get_media_db_for_user),
) -> dict[str, Any]:
    updates = model_dump_compat(payload, exclude_unset=True)
    try:
        if "config" in updates:
            current = media_db.get_audio_preset(user_id=_user_id(request_user), preset_id=preset_id)
            current = _preset_or_404(current)
            updates["config"] = _normalize_audio_preset_config(str(current["kind"]), updates.get("config") or {})
        return _preset_or_404(
            media_db.update_audio_preset(
                user_id=_user_id(request_user),
                preset_id=preset_id,
                updates=updates,
            )
        )
    except (ConflictError, InputError, DatabaseError) as exc:
        raise map_db_error_to_http(
            exc,
            not_found_substrings=("not found",),
            default_detail="Failed to update audio preset",
        ) from exc


@router.delete(
    "/presets/{preset_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a reusable audio preset.",
    dependencies=[
        Depends(check_rate_limit),
        Depends(TokenScopeGuard("any", require_if_present=True, endpoint_id="audio.presets.delete", count_as="call")),
    ],
)
async def delete_audio_preset_endpoint(
    preset_id: str,
    request_user: User = Depends(get_request_user),
    media_db: Any = Depends(get_media_db_for_user),
) -> Response:
    try:
        deleted = media_db.soft_delete_audio_preset(user_id=_user_id(request_user), preset_id=preset_id)
    except DatabaseError as exc:
        raise map_db_error_to_http(exc, default_detail="Failed to delete audio preset") from exc
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Audio preset not found")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post(
    "/presets/{preset_id}/validate",
    response_model=AudioPresetValidationResponse,
    summary="Validate a reusable audio preset against current runtime assumptions.",
    dependencies=[
        Depends(check_rate_limit),
        Depends(TokenScopeGuard("any", require_if_present=True, endpoint_id="audio.presets.validate", count_as="call")),
    ],
)
async def validate_audio_preset_endpoint(
    preset_id: str,
    request_user: User = Depends(get_request_user),
    media_db: Any = Depends(get_media_db_for_user),
) -> AudioPresetValidationResponse:
    try:
        preset = _preset_or_404(
            media_db.get_audio_preset(user_id=_user_id(request_user), preset_id=preset_id)
        )
    except DatabaseError as exc:
        raise map_db_error_to_http(exc, default_detail="Failed to validate audio preset") from exc
    warnings = _validation_warnings_for_preset(preset)
    return AudioPresetValidationResponse(preset=preset, valid=True, warnings=warnings)
