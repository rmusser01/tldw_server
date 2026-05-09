"""VN Play runtime endpoints."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, cast

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status

from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.api.v1.schemas.vn_play_schemas import (
    VNPlayBranchResponse,
    VNPlayCheckpointCreate,
    VNPlayCheckpointResponse,
    VNPlayEventResponse,
    VNPlayRestoreRequest,
    VNPlayRetryTurnRequest,
    VNPlaySceneStateResponse,
    VNPlayMode,
    VNPlaySessionCreate,
    VNPlaySessionResponse,
    VNPlaySessionUpdate,
    VNPlaySetupOptionsResponse,
    VNPlayTurnRequest,
    VNPlayTurnResponse,
)
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.VNPlay_DB import VNPlayRepository
from tldw_Server_API.app.core.VN_Play.constants import (
    ERROR_IDEMPOTENCY_KEY_CONFLICT,
    ERROR_STALE_SCENE_VERSION,
    ERROR_TURN_IN_PROGRESS,
    TURN_STATUS_MODEL_FAILED,
    TURN_STATUS_PARSE_FAILED,
)
from tldw_Server_API.app.core.VN_Play.setup_options import (
    DEFAULT_SETUP_LIMIT,
    MAX_SETUP_LIMIT,
    build_vn_play_setup_options,
)
from tldw_Server_API.app.core.VN_Play.service import (
    VNPlayConflictError,
    VNPlayNotFoundError,
    VNPlayService,
    VNPlaySession,
    VNPlayTurnError,
    VNPlayTurnResponse as ServiceTurnResponse,
)


router = APIRouter(prefix="/vn-play", tags=["vn-play"])
CONFLICT_ERROR_CODES = {
    ERROR_IDEMPOTENCY_KEY_CONFLICT,
    ERROR_STALE_SCENE_VERSION,
    ERROR_TURN_IN_PROGRESS,
}


def _service(
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    current_user: User = Depends(get_request_user),
) -> VNPlayService:
    owner_user_id = _owner_user_id(current_user)
    return VNPlayService(
        repo=VNPlayRepository.initialized(db),
        owner_user_id=owner_user_id,
    )


def _owner_user_id(current_user: User) -> int:
    owner_user_id = current_user.id_int
    if owner_user_id is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="invalid_user_id",
        )
    return owner_user_id


@router.get("/setup-options", response_model=VNPlaySetupOptionsResponse)
def setup_options(
    mode: str | None = Query(default=None, pattern="^(freeform|story)$"),
    character_query: str | None = Query(default=None, max_length=200),
    pack_query: str | None = Query(default=None, max_length=200),
    character_limit: int = Query(default=DEFAULT_SETUP_LIMIT, ge=1, le=MAX_SETUP_LIMIT),
    character_offset: int = Query(default=0, ge=0),
    pack_limit: int = Query(default=DEFAULT_SETUP_LIMIT, ge=1, le=MAX_SETUP_LIMIT),
    pack_offset: int = Query(default=0, ge=0),
    selected_character_id: int | None = Query(default=None, ge=1),
    content_rating: str | None = Query(default="general", min_length=1, max_length=100),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    current_user: User = Depends(get_request_user),
) -> VNPlaySetupOptionsResponse:
    return build_vn_play_setup_options(
        db=db,
        owner_user_id=_owner_user_id(current_user),
        mode=cast(VNPlayMode | None, mode),
        character_query=character_query,
        pack_query=pack_query,
        character_limit=character_limit,
        character_offset=character_offset,
        pack_limit=pack_limit,
        pack_offset=pack_offset,
        selected_character_id=selected_character_id,
        content_rating=content_rating,
    )


@router.post(
    "/sessions",
    response_model=VNPlaySessionResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_session(
    request: VNPlaySessionCreate,
    service: VNPlayService = Depends(_service),
) -> VNPlaySessionResponse:
    try:
        session = service.create_session(**request.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    return _session_response(service, session)


@router.get("/sessions", response_model=list[VNPlaySessionResponse])
def list_sessions(service: VNPlayService = Depends(_service)) -> list[VNPlaySessionResponse]:
    return [_session_response(service, session) for session in service.list_sessions()]


@router.get("/sessions/{session_id}", response_model=VNPlaySessionResponse)
def get_session(
    session_id: int,
    service: VNPlayService = Depends(_service),
) -> VNPlaySessionResponse:
    try:
        return _session_response(service, service.get_session(session_id))
    except VNPlayNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="not_found") from exc


@router.patch("/sessions/{session_id}", response_model=VNPlaySessionResponse)
def update_session(
    session_id: int,
    request: VNPlaySessionUpdate,
    service: VNPlayService = Depends(_service),
) -> VNPlaySessionResponse:
    fields = request.model_dump(exclude_none=True)
    row = service.repo.update_session(
        session_id,
        fields,
        owner_user_id=service.owner_user_id,
    )
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="not_found")
    return _session_response(service, VNPlaySession.from_row(row))


@router.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_session(
    session_id: int,
    service: VNPlayService = Depends(_service),
) -> Response:
    try:
        service.get_session(session_id)
    except VNPlayNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="not_found") from exc

    service.repo.update_session(
        session_id,
        {"deleted": True},
        owner_user_id=service.owner_user_id,
    )
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/sessions/{session_id}/turn", response_model=VNPlayTurnResponse)
async def submit_turn(
    session_id: int,
    request: VNPlayTurnRequest,
    service: VNPlayService = Depends(_service),
) -> VNPlayTurnResponse:
    try:
        response = await service.submit_turn(
            session_id,
            input_text=request.input_text,
            choice_id=request.choice_id,
            custom_action=request.custom_action,
            client_scene_version=request.client_scene_version,
            idempotency_key=request.idempotency_key,
        )
        return _turn_response(service, session_id, response)
    except (VNPlayConflictError, VNPlayNotFoundError, VNPlayTurnError) as exc:
        raise _http_error_for_service_exception(exc) from exc


@router.post("/sessions/{session_id}/retry-last-turn", response_model=VNPlayTurnResponse)
async def retry_last_turn(
    session_id: int,
    request: VNPlayRetryTurnRequest,
    service: VNPlayService = Depends(_service),
) -> VNPlayTurnResponse:
    try:
        response = await service.retry_last_turn(
            session_id,
            client_scene_version=request.client_scene_version,
            idempotency_key=request.idempotency_key,
        )
        return _turn_response(service, session_id, response)
    except (VNPlayConflictError, VNPlayNotFoundError, VNPlayTurnError) as exc:
        raise _http_error_for_service_exception(exc) from exc


@router.get("/sessions/{session_id}/events", response_model=list[VNPlayEventResponse])
def list_events(
    session_id: int,
    service: VNPlayService = Depends(_service),
) -> list[VNPlayEventResponse]:
    try:
        return [VNPlayEventResponse.model_validate(event) for event in service.list_events(session_id)]
    except VNPlayNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="not_found") from exc


@router.post("/sessions/{session_id}/checkpoint", response_model=VNPlayCheckpointResponse)
def create_checkpoint(
    session_id: int,
    request: VNPlayCheckpointCreate,
    service: VNPlayService = Depends(_service),
) -> VNPlayCheckpointResponse:
    try:
        return VNPlayCheckpointResponse.model_validate(
            service.create_checkpoint(session_id, label=request.label)
        )
    except VNPlayNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="not_found") from exc


@router.get("/sessions/{session_id}/checkpoints", response_model=list[VNPlayCheckpointResponse])
def list_checkpoints(
    session_id: int,
    service: VNPlayService = Depends(_service),
) -> list[VNPlayCheckpointResponse]:
    try:
        service.get_session(session_id)
    except VNPlayNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="not_found") from exc
    return [
        VNPlayCheckpointResponse.model_validate(checkpoint)
        for checkpoint in service.repo.list_checkpoints(
            session_id,
            owner_user_id=service.owner_user_id,
        )
    ]


@router.post("/sessions/{session_id}/restore", response_model=VNPlaySessionResponse)
def restore_checkpoint(
    session_id: int,
    request: VNPlayRestoreRequest,
    service: VNPlayService = Depends(_service),
) -> VNPlaySessionResponse:
    try:
        service.restore_checkpoint(session_id, request.checkpoint_id)
        return _session_response(service, service.get_session(session_id))
    except VNPlayNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="not_found") from exc


@router.get("/sessions/{session_id}/branches", response_model=list[VNPlayBranchResponse])
def list_branches(
    session_id: int,
    service: VNPlayService = Depends(_service),
) -> list[VNPlayBranchResponse]:
    try:
        return [VNPlayBranchResponse.model_validate(branch) for branch in service.list_branches(session_id)]
    except VNPlayNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="not_found") from exc


def _session_response(
    service: VNPlayService,
    session: VNPlaySession,
) -> VNPlaySessionResponse:
    payload = asdict(session)
    payload["deleted"] = False
    scene_state = _scene_state(service, session.id)
    payload["current_scene"] = scene_state
    payload["scene_state"] = scene_state
    return VNPlaySessionResponse.model_validate(payload)


def _turn_response(
    service: VNPlayService,
    session_id: int,
    response: ServiceTurnResponse,
) -> VNPlayTurnResponse:
    payload = response.to_payload()
    session = service.get_session(session_id)
    payload["session"] = _session_response(service, session).model_dump()
    payload["current_scene"] = _scene_state(service, session_id)
    return VNPlayTurnResponse.model_validate(payload)


def _scene_state(service: VNPlayService, session_id: int) -> dict[str, Any] | None:
    return service.get_enriched_scene_state(session_id)


def _http_error_for_service_exception(exc: Exception) -> HTTPException:
    detail = str(exc) or exc.__class__.__name__
    if isinstance(exc, VNPlayNotFoundError):
        return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="not_found")
    if isinstance(exc, VNPlayConflictError) or detail in CONFLICT_ERROR_CODES:
        return HTTPException(status_code=status.HTTP_409_CONFLICT, detail=detail)
    if detail in {TURN_STATUS_MODEL_FAILED, TURN_STATUS_PARSE_FAILED}:
        return HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=detail)
    return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=detail)
