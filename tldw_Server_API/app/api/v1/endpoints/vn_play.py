"""VN Play runtime endpoints."""

from __future__ import annotations

import json
from typing import Any, cast

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status

from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.api.v1.schemas.vn_play_schemas import (
    VNPlayBranchNavigationResponse,
    VNPlayBranchResponse,
    VNPlayBranchRestoreRequest,
    VNPlayBranchRestoreResponse,
    VNPlayCheckpointCreate,
    VNPlayCheckpointResponse,
    VNPlayEventResponse,
    VNPlayRestoreRequest,
    VNPlayRetryTurnRequest,
    VNPlaySaveSlotCreate,
    VNPlaySaveSlotResponse,
    VNPlaySaveSlotRestoreRequest,
    VNPlaySaveSlotUpdate,
    VNPlaySceneStateResponse,
    VNPlayScriptActionRequest,
    VNPlayScriptActionResponse,
    VNPlayScriptDebugStateResponse,
    VNPlayScriptStateResponse,
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
    ERROR_BRANCH_NOT_FOUND,
    ERROR_BRANCH_RESTORE_AMBIGUOUS,
    ERROR_BRANCH_RESTORE_NOT_ALLOWED,
    ERROR_BRANCH_RESTORE_TARGET_UNAVAILABLE,
    ERROR_IDEMPOTENCY_KEY_CONFLICT,
    ERROR_RESTORE_ACTION_IN_PROGRESS,
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
from tldw_Server_API.app.core.VN_Platform.errors import vn_error_detail


router = APIRouter(prefix="/vn-play", tags=["vn-play"])
CONFLICT_ERROR_CODES = {
    ERROR_IDEMPOTENCY_KEY_CONFLICT,
    ERROR_RESTORE_ACTION_IN_PROGRESS,
    ERROR_STALE_SCENE_VERSION,
    ERROR_TURN_IN_PROGRESS,
}
BAD_REQUEST_ERROR_CODES = {
    ERROR_BRANCH_RESTORE_AMBIGUOUS,
    ERROR_BRANCH_RESTORE_NOT_ALLOWED,
    ERROR_BRANCH_RESTORE_TARGET_UNAVAILABLE,
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
    mode: str | None = Query(default=None, pattern="^(freeform|story|scripted_story)$"),
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
        detail = str(exc) or "invalid_request"
        raise _vn_play_http_error(status.HTTP_400_BAD_REQUEST, detail) from exc
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


@router.get(
    "/sessions/{session_id}/branch-navigation",
    response_model=VNPlayBranchNavigationResponse,
)
def get_branch_navigation(
    session_id: int,
    service: VNPlayService = Depends(_service),
) -> VNPlayBranchNavigationResponse:
    try:
        return VNPlayBranchNavigationResponse.model_validate(
            service.get_branch_navigation(session_id)
        )
    except (VNPlayConflictError, VNPlayNotFoundError, VNPlayTurnError) as exc:
        raise _http_error_for_service_exception(exc) from exc


@router.get("/sessions/{session_id}/events", response_model=list[VNPlayEventResponse])
def list_events(
    session_id: int,
    response: Response,
    branch_id: int | None = Query(default=None, ge=1),
    after_sequence: int | None = Query(default=None, ge=0),
    limit: int | None = Query(default=None, ge=1, le=250),
    include_descendants: bool = Query(default=False),
    service: VNPlayService = Depends(_service),
) -> list[VNPlayEventResponse]:
    try:
        effective_limit = 100 if branch_id is not None and limit is None else limit
        result = service.list_events_with_metadata(
            session_id,
            branch_id=branch_id,
            after_sequence=after_sequence,
            limit=effective_limit,
            include_descendants=include_descendants,
        )
        warnings = result.get("warnings") or []
        if warnings:
            response.headers["X-VN-Play-Warnings"] = json.dumps(
                warnings,
                separators=(",", ":"),
            )
        return [
            VNPlayEventResponse.model_validate(event)
            for event in result.get("events", [])
        ]
    except (VNPlayConflictError, VNPlayNotFoundError, VNPlayTurnError) as exc:
        raise _http_error_for_service_exception(exc) from exc


@router.post("/sessions/{session_id}/checkpoint", response_model=VNPlayCheckpointResponse)
def create_checkpoint(
    session_id: int,
    request: VNPlayCheckpointCreate,
    service: VNPlayService = Depends(_service),
) -> VNPlayCheckpointResponse:
    try:
        return VNPlayCheckpointResponse.model_validate(
            service.public_checkpoint_payload(
                service.create_checkpoint(session_id, label=request.label)
            )
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
        VNPlayCheckpointResponse.model_validate(service.public_checkpoint_payload(checkpoint))
        for checkpoint in service.repo.list_checkpoints(
            session_id,
            owner_user_id=service.owner_user_id,
        )
    ]


@router.post(
    "/sessions/{session_id}/save-slots",
    response_model=VNPlaySaveSlotResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_save_slot(
    session_id: int,
    request: VNPlaySaveSlotCreate,
    service: VNPlayService = Depends(_service),
) -> VNPlaySaveSlotResponse:
    try:
        return VNPlaySaveSlotResponse.model_validate(
            service.create_save_slot(
                session_id,
                slot_key=request.slot_key,
                title=request.title,
                metadata=request.metadata,
                idempotency_key=request.idempotency_key,
            )
        )
    except (VNPlayConflictError, VNPlayNotFoundError, VNPlayTurnError) as exc:
        raise _http_error_for_service_exception(exc) from exc


@router.get("/sessions/{session_id}/save-slots", response_model=list[VNPlaySaveSlotResponse])
def list_save_slots(
    session_id: int,
    include_deleted: bool = Query(default=False),
    service: VNPlayService = Depends(_service),
) -> list[VNPlaySaveSlotResponse]:
    try:
        return [
            VNPlaySaveSlotResponse.model_validate(slot)
            for slot in service.list_save_slots(
                session_id,
                include_deleted=include_deleted,
            )
        ]
    except (VNPlayConflictError, VNPlayNotFoundError, VNPlayTurnError) as exc:
        raise _http_error_for_service_exception(exc) from exc


@router.get(
    "/sessions/{session_id}/save-slots/{save_slot_id}",
    response_model=VNPlaySaveSlotResponse,
)
def get_save_slot(
    session_id: int,
    save_slot_id: int,
    include_deleted: bool = Query(default=False),
    service: VNPlayService = Depends(_service),
) -> VNPlaySaveSlotResponse:
    try:
        return VNPlaySaveSlotResponse.model_validate(
            service.get_save_slot(
                session_id,
                save_slot_id,
                include_deleted=include_deleted,
            )
        )
    except (VNPlayConflictError, VNPlayNotFoundError, VNPlayTurnError) as exc:
        raise _http_error_for_service_exception(exc) from exc


@router.patch(
    "/sessions/{session_id}/save-slots/{save_slot_id}",
    response_model=VNPlaySaveSlotResponse,
)
def patch_save_slot(
    session_id: int,
    save_slot_id: int,
    request: VNPlaySaveSlotUpdate,
    service: VNPlayService = Depends(_service),
) -> VNPlaySaveSlotResponse:
    try:
        return VNPlaySaveSlotResponse.model_validate(
            service.update_save_slot(
                session_id,
                save_slot_id,
                title=request.title,
                metadata=request.metadata,
            )
        )
    except (VNPlayConflictError, VNPlayNotFoundError, VNPlayTurnError) as exc:
        raise _http_error_for_service_exception(exc) from exc


@router.delete(
    "/sessions/{session_id}/save-slots/{save_slot_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def delete_save_slot(
    session_id: int,
    save_slot_id: int,
    service: VNPlayService = Depends(_service),
) -> Response:
    try:
        service.delete_save_slot(session_id, save_slot_id)
    except (VNPlayConflictError, VNPlayNotFoundError, VNPlayTurnError) as exc:
        raise _http_error_for_service_exception(exc) from exc
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post(
    "/sessions/{session_id}/save-slots/{save_slot_id}/restore",
    response_model=VNPlayBranchRestoreResponse,
)
def restore_save_slot(
    session_id: int,
    save_slot_id: int,
    request: VNPlaySaveSlotRestoreRequest,
    service: VNPlayService = Depends(_service),
) -> VNPlayBranchRestoreResponse:
    try:
        return VNPlayBranchRestoreResponse.model_validate(
            service.restore_save_slot(
                session_id,
                save_slot_id,
                client_scene_version=request.client_scene_version,
                idempotency_key=request.idempotency_key,
            )
        )
    except (VNPlayConflictError, VNPlayNotFoundError, VNPlayTurnError) as exc:
        raise _http_error_for_service_exception(exc) from exc


@router.post("/sessions/{session_id}/restore", response_model=VNPlaySessionResponse)
def restore_checkpoint(
    session_id: int,
    request: VNPlayRestoreRequest,
    service: VNPlayService = Depends(_service),
) -> VNPlaySessionResponse:
    try:
        restore_response = service.restore_checkpoint(
            session_id,
            request.checkpoint_id,
            client_scene_version=request.client_scene_version,
            idempotency_key=request.idempotency_key,
        )
        return _checkpoint_restore_session_response(restore_response)
    except (VNPlayConflictError, VNPlayNotFoundError, VNPlayTurnError) as exc:
        raise _http_error_for_service_exception(exc) from exc


@router.get("/sessions/{session_id}/branches", response_model=list[VNPlayBranchResponse])
def list_branches(
    session_id: int,
    service: VNPlayService = Depends(_service),
) -> list[VNPlayBranchResponse]:
    try:
        return [VNPlayBranchResponse.model_validate(branch) for branch in service.list_branches(session_id)]
    except VNPlayNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="not_found") from exc


@router.post(
    "/sessions/{session_id}/branches/{branch_id}/restore",
    response_model=VNPlayBranchRestoreResponse,
)
def restore_branch(
    session_id: int,
    branch_id: int,
    request: VNPlayBranchRestoreRequest,
    service: VNPlayService = Depends(_service),
) -> VNPlayBranchRestoreResponse:
    try:
        return VNPlayBranchRestoreResponse.model_validate(
            service.restore_branch(
                session_id,
                branch_id=branch_id,
                client_scene_version=request.client_scene_version,
                idempotency_key=request.idempotency_key,
                target=request.target,
            )
        )
    except (VNPlayConflictError, VNPlayNotFoundError, VNPlayTurnError) as exc:
        raise _http_error_for_service_exception(exc) from exc


@router.post(
    "/sessions/{session_id}/story/start",
    response_model=VNPlayTurnResponse | VNPlayScriptActionResponse,
)
async def start_story(
    session_id: int,
    request: VNPlayScriptActionRequest,
    service: VNPlayService = Depends(_service),
) -> VNPlayTurnResponse | VNPlayScriptActionResponse:
    try:
        response = await service.start_story(
            session_id,
            client_scene_version=request.client_scene_version,
            idempotency_key=request.idempotency_key,
        )
        if isinstance(response, ServiceTurnResponse):
            return _turn_response(service, session_id, response)
        return VNPlayScriptActionResponse.model_validate(response)
    except (VNPlayConflictError, VNPlayNotFoundError, VNPlayTurnError) as exc:
        raise _http_error_for_service_exception(exc) from exc


@router.post(
    "/sessions/{session_id}/script/advance",
    response_model=VNPlayScriptActionResponse,
)
def advance_scripted_story(
    session_id: int,
    request: VNPlayScriptActionRequest,
    service: VNPlayService = Depends(_service),
) -> VNPlayScriptActionResponse:
    try:
        return VNPlayScriptActionResponse.model_validate(
            service.advance_script(
                session_id,
                client_scene_version=request.client_scene_version,
                idempotency_key=request.idempotency_key,
            )
        )
    except (VNPlayConflictError, VNPlayNotFoundError, VNPlayTurnError) as exc:
        raise _http_error_for_service_exception(exc) from exc


@router.post(
    "/sessions/{session_id}/script/regenerate",
    response_model=VNPlayScriptActionResponse,
)
def regenerate_scripted_story_expansion(
    session_id: int,
    request: VNPlayScriptActionRequest,
    service: VNPlayService = Depends(_service),
) -> VNPlayScriptActionResponse:
    try:
        return VNPlayScriptActionResponse.model_validate(
            service.regenerate_script_expansion(
                session_id,
                client_scene_version=request.client_scene_version,
                idempotency_key=request.idempotency_key,
            )
        )
    except (VNPlayConflictError, VNPlayNotFoundError, VNPlayTurnError) as exc:
        raise _http_error_for_service_exception(exc) from exc


@router.get(
    "/sessions/{session_id}/script/state",
    response_model=VNPlayScriptStateResponse,
)
def get_scripted_story_state(
    session_id: int,
    service: VNPlayService = Depends(_service),
) -> VNPlayScriptStateResponse:
    try:
        return VNPlayScriptStateResponse.model_validate(service.get_script_state(session_id))
    except (VNPlayConflictError, VNPlayNotFoundError, VNPlayTurnError) as exc:
        raise _http_error_for_service_exception(exc) from exc


@router.get(
    "/sessions/{session_id}/script/debug-state",
    response_model=VNPlayScriptDebugStateResponse,
)
def get_scripted_story_debug_state(
    session_id: int,
    service: VNPlayService = Depends(_service),
) -> VNPlayScriptDebugStateResponse:
    try:
        return VNPlayScriptDebugStateResponse.model_validate(
            service.get_script_debug_state(session_id)
        )
    except (VNPlayConflictError, VNPlayNotFoundError, VNPlayTurnError) as exc:
        raise _http_error_for_service_exception(exc) from exc


@router.post(
    "/sessions/{session_id}/script/choices/{choice_id}",
    response_model=VNPlayScriptActionResponse,
)
def choose_scripted_story_option(
    session_id: int,
    choice_id: str,
    request: VNPlayScriptActionRequest,
    service: VNPlayService = Depends(_service),
) -> VNPlayScriptActionResponse:
    try:
        return VNPlayScriptActionResponse.model_validate(
            service.choose_script_option(
                session_id,
                choice_id=choice_id,
                client_scene_version=request.client_scene_version,
                idempotency_key=request.idempotency_key,
            )
        )
    except (VNPlayConflictError, VNPlayNotFoundError, VNPlayTurnError) as exc:
        raise _http_error_for_service_exception(exc) from exc


def _session_response(
    service: VNPlayService,
    session: VNPlaySession,
) -> VNPlaySessionResponse:
    payload = service.public_session_payload(session)
    scene_state = _scene_state(service, session.id)
    payload["current_scene"] = scene_state
    payload["scene_state"] = scene_state
    return VNPlaySessionResponse.model_validate(payload)


def _checkpoint_restore_session_response(
    restore_response: dict[str, Any],
) -> VNPlaySessionResponse:
    payload = dict(restore_response["session"])
    payload["deleted"] = False
    scene_state = restore_response.get("current_scene")
    if scene_state is not None:
        scene_state_payload = dict(scene_state)
        payload["current_scene"] = scene_state_payload
        payload["scene_state"] = scene_state_payload
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
        if detail == ERROR_BRANCH_NOT_FOUND:
            return _vn_play_http_error(status.HTTP_404_NOT_FOUND, detail)
        return _vn_play_http_error(status.HTTP_404_NOT_FOUND, "not_found")
    if isinstance(exc, VNPlayConflictError) or detail in CONFLICT_ERROR_CODES:
        if detail in BAD_REQUEST_ERROR_CODES:
            return _vn_play_http_error(status.HTTP_400_BAD_REQUEST, detail)
        return _vn_play_http_error(status.HTTP_409_CONFLICT, detail)
    if detail in {TURN_STATUS_MODEL_FAILED, TURN_STATUS_PARSE_FAILED}:
        return _vn_play_http_error(status.HTTP_502_BAD_GATEWAY, detail, retryable=True)
    if detail in BAD_REQUEST_ERROR_CODES:
        return _vn_play_http_error(status.HTTP_400_BAD_REQUEST, detail)
    return _vn_play_http_error(status.HTTP_400_BAD_REQUEST, detail)


def _vn_play_http_error(
    status_code: int,
    code: str,
    *,
    retryable: bool = False,
) -> HTTPException:
    return HTTPException(
        status_code=status_code,
        detail=vn_error_detail(code, code, retryable=retryable),
    )
