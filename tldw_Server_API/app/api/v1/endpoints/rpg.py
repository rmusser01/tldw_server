from __future__ import annotations

from dataclasses import asdict
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, status
from fastapi.encoders import jsonable_encoder

from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import (
    RequirePermission,
    TokenScopeGuard,
    User,
    get_request_user,
    rbac_rate_limit,
)
from tldw_Server_API.app.api.v1.schemas.rpg_schemas import (
    RPGCampaignCreateRequest,
    RPGCampaignResponse,
    RPGProposalApplyRequest,
    RPGProposalRejectRequest,
    RPGRecordEventsRequest,
    RPGRecordEventsResponse,
    RPGSessionCreateRequest,
    RPGSessionResponse,
)
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.RPG_DB import RPGRepository
from tldw_Server_API.app.core.RPG.errors import RPGConflictError, RPGNotFoundError, RPGValidationError
from tldw_Server_API.app.core.RPG.service import RPGService

router = APIRouter(prefix="/rpg", tags=["rpg"])

RPG_RULES_READ = "rpg.rules.read"
RPG_CAMPAIGNS_MANAGE = "rpg.campaigns.manage"
RPG_SESSIONS_MANAGE = "rpg.sessions.manage"
RPG_PROPOSALS_REVIEW = "rpg.proposals.review"


def _read_dependencies(scope: str):
    return [
        Depends(rbac_rate_limit(scope)),
        Depends(RequirePermission(scope)),
        Depends(TokenScopeGuard("rpg", require_if_present=True, endpoint_id=scope, count_as="call")),
    ]


def _write_dependencies(scope: str):
    return _read_dependencies(scope)


def _owner_user_id(current_user: User) -> int:
    if current_user.id_int is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="invalid_user_id")
    return int(current_user.id_int)


def _service(
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    current_user: User = Depends(get_request_user),
) -> RPGService:
    return RPGService(repo=RPGRepository.initialized(db), owner_user_id=_owner_user_id(current_user))


def _map_error(exc: Exception) -> HTTPException:
    if isinstance(exc, RPGNotFoundError):
        return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    if isinstance(exc, RPGConflictError):
        return HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))
    if isinstance(exc, (RPGValidationError, ValueError)):
        return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    return HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="rpg_internal_error")


def _event_payload(event: Any) -> dict[str, Any]:
    return jsonable_encoder(asdict(event))


def _proposal_payload(proposal: Any | None) -> dict[str, Any] | None:
    if proposal is None:
        return None
    return jsonable_encoder(asdict(proposal))


@router.get(
    "/rules/adapters",
    dependencies=_read_dependencies(RPG_RULES_READ),
)
def list_adapters(service: RPGService = Depends(_service)) -> dict[str, Any]:
    return {"adapters": [asdict(adapter) for adapter in service.adapter_registry.list_infos()]}


@router.get(
    "/rules/adapters/{adapter_key}",
    dependencies=_read_dependencies(RPG_RULES_READ),
)
def get_adapter(adapter_key: str, service: RPGService = Depends(_service)) -> dict[str, Any]:
    try:
        return {"adapter": asdict(service.adapter_registry.get(adapter_key).info())}
    except Exception as exc:
        raise _map_error(exc) from exc


@router.post(
    "/campaigns",
    response_model=RPGCampaignResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=_write_dependencies(RPG_CAMPAIGNS_MANAGE),
)
def create_campaign(
    request: RPGCampaignCreateRequest,
    idempotency_key: str = Header(alias="Idempotency-Key"),
    service: RPGService = Depends(_service),
) -> RPGCampaignResponse:
    try:
        campaign = service.create_campaign(
            request.title,
            request.description,
            request.default_adapter_key,
            idempotency_key=idempotency_key,
        )
        return RPGCampaignResponse.model_validate(asdict(campaign))
    except Exception as exc:
        raise _map_error(exc) from exc


@router.post(
    "/campaigns/{campaign_id}/sessions",
    response_model=RPGSessionResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=_write_dependencies(RPG_SESSIONS_MANAGE),
)
def create_session(
    campaign_id: int,
    request: RPGSessionCreateRequest,
    idempotency_key: str = Header(alias="Idempotency-Key"),
    service: RPGService = Depends(_service),
) -> RPGSessionResponse:
    try:
        session = service.create_session(
            campaign_id=campaign_id,
            title=request.title,
            adapter_key=request.adapter_key,
            idempotency_key=idempotency_key,
        )
        return RPGSessionResponse.model_validate(asdict(session))
    except Exception as exc:
        raise _map_error(exc) from exc


@router.post(
    "/sessions/{session_id}/events",
    response_model=RPGRecordEventsResponse,
    dependencies=_write_dependencies(RPG_SESSIONS_MANAGE),
)
def record_events(
    session_id: int,
    request: RPGRecordEventsRequest,
    idempotency_key: str = Header(alias="Idempotency-Key"),
    service: RPGService = Depends(_service),
) -> RPGRecordEventsResponse:
    try:
        result = service.record_events(
            session_id=session_id,
            events=[event.model_dump() for event in request.events],
            source_type="user",
            expected_last_event_sequence=request.expected_last_event_sequence,
            idempotency_key=idempotency_key,
        )
        return RPGRecordEventsResponse(
            committed_events=[_event_payload(event) for event in result.committed_events],
            proposal=_proposal_payload(result.proposal),
        )
    except Exception as exc:
        raise _map_error(exc) from exc


@router.post(
    "/sessions/{session_id}/proposals/{proposal_id}/apply",
    response_model=RPGRecordEventsResponse,
    dependencies=_write_dependencies(RPG_PROPOSALS_REVIEW),
)
def apply_proposal(
    session_id: int,
    proposal_id: int,
    request: RPGProposalApplyRequest,
    idempotency_key: str = Header(alias="Idempotency-Key"),
    service: RPGService = Depends(_service),
) -> RPGRecordEventsResponse:
    try:
        result = service.apply_proposal(
            session_id=session_id,
            proposal_id=proposal_id,
            expected_last_event_sequence=request.expected_last_event_sequence,
            idempotency_key=idempotency_key,
            review_notes=request.review_notes,
        )
        return RPGRecordEventsResponse(
            committed_events=[_event_payload(event) for event in result.committed_events],
            proposal=None,
        )
    except Exception as exc:
        raise _map_error(exc) from exc


@router.post(
    "/sessions/{session_id}/proposals/{proposal_id}/reject",
    dependencies=_write_dependencies(RPG_PROPOSALS_REVIEW),
)
def reject_proposal(
    session_id: int,
    proposal_id: int,
    request: RPGProposalRejectRequest,
    idempotency_key: str = Header(alias="Idempotency-Key"),
    service: RPGService = Depends(_service),
) -> dict[str, Any]:
    try:
        proposal = service.reject_proposal(
            session_id=session_id,
            proposal_id=proposal_id,
            idempotency_key=idempotency_key,
            review_notes=request.review_notes,
        )
        return {"proposal": _proposal_payload(proposal)}
    except Exception as exc:
        raise _map_error(exc) from exc
