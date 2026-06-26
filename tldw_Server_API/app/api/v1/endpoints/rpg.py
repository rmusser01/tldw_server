from __future__ import annotations

from dataclasses import asdict
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.security import HTTPAuthorizationCredentials

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import (
    RequirePermission,
    TokenScopeGuard,
    User,
    enforce_rbac_rate_limit,
    get_auth_principal,
    get_request_user,
    rbac_rate_limit,
)
from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.api.v1.API_Deps.Collections_DB_Deps import get_collections_db_for_user
from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user, get_media_db_path_for_rag
from tldw_Server_API.app.api.v1.schemas.rpg_schemas import (
    RPGCampaignCreateRequest,
    RPGCampaignResponse,
    RPGContextBuildRequest,
    RPGContextResponse,
    RPGProposalApplyRequest,
    RPGProposalRejectRequest,
    RPGRecordEventsRequest,
    RPGRecordEventsResponse,
    RPGRulesLookupRequest,
    RPGRulesLookupResponse,
    RPGRulesPackRefsReplaceRequest,
    RPGRulesPackRefsResponse,
    RPGSessionCreateRequest,
    RPGSessionResponse,
)
from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
from tldw_Server_API.app.core.AuthNZ.llm_budget_guard import enforce_llm_budget
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.Collections_DB import CollectionsDatabase
from tldw_Server_API.app.core.DB_Management.RPG_DB import RPGRepository
from tldw_Server_API.app.core.RAG.rag_service.database_retrievers import MultiDatabaseRetriever
from tldw_Server_API.app.core.RPG.errors import RPGConflictError, RPGNotFoundError, RPGValidationError
from tldw_Server_API.app.core.RPG.rules.answering import ChatRulesAnswerGenerator, RulesAnswerOptions
from tldw_Server_API.app.core.RPG.rules.lookup import RulesLookupService
from tldw_Server_API.app.core.RPG.rules.retrieval import RulesRetrievalAdapter
from tldw_Server_API.app.core.RPG.rules.source_validation import RPGRulesSourceValidator
from tldw_Server_API.app.core.RPG.service import RPGService

router = APIRouter(prefix="/rpg", tags=["rpg"])

RPG_RULES_READ = "rpg.rules.read"
RPG_CAMPAIGNS_READ = "rpg.campaigns.read"
RPG_CAMPAIGNS_MANAGE = "rpg.campaigns.manage"
RPG_SESSIONS_READ = "rpg.sessions.read"
RPG_SESSIONS_MANAGE = "rpg.sessions.manage"
RPG_PROPOSALS_REVIEW = "rpg.proposals.review"
MEDIA_READ = "media.read"
CHAT_COMPLETIONS = "chat.completions"
CHAT_CREATE_RATE_LIMIT = "chat.create"

_answer_mode_permission_guard = RequirePermission(CHAT_COMPLETIONS)
_answer_mode_scope_guard = TokenScopeGuard(
    "any",
    require_if_present=True,
    endpoint_id=CHAT_COMPLETIONS,
    count_as="call",
)


def _read_dependencies(*scopes: str):
    primary_scope = scopes[0]
    return [
        Depends(rbac_rate_limit(primary_scope)),
        Depends(RequirePermission(*scopes)),
        Depends(TokenScopeGuard("rpg", require_if_present=True, endpoint_id=primary_scope, count_as="call")),
    ]


def _write_dependencies(*scopes: str):
    return _read_dependencies(*scopes)


async def _enforce_answer_mode_generation_controls(
    request: Request,
    principal: AuthPrincipal,
) -> None:
    await _answer_mode_permission_guard(principal)
    await _answer_mode_scope_guard(request, credentials=_bearer_credentials_from_request(request))
    await enforce_llm_budget(request)
    await enforce_rbac_rate_limit(request, CHAT_CREATE_RATE_LIMIT, await get_db_pool())


def _bearer_credentials_from_request(request: Request) -> HTTPAuthorizationCredentials | None:
    authorization = request.headers.get("Authorization")
    if not authorization:
        return None
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        return None
    return HTTPAuthorizationCredentials(scheme=scheme, credentials=token.strip())


def _owner_user_id(current_user: User) -> int:
    if current_user.id_int is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="invalid_user_id")
    return int(current_user.id_int)


def _service(
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    current_user: User = Depends(get_request_user),
) -> RPGService:
    return RPGService(repo=RPGRepository.initialized(db), owner_user_id=_owner_user_id(current_user))


def _rules_service(
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    media_db: Any = Depends(get_media_db_for_user),
    collections_db: CollectionsDatabase = Depends(get_collections_db_for_user),
    current_user: User = Depends(get_request_user),
) -> RPGService:
    owner_user_id = _owner_user_id(current_user)
    validator = RPGRulesSourceValidator(media_db=media_db, collections_db=collections_db)
    media_db_path = get_media_db_path_for_rag(media_db)
    rag_retriever = MultiDatabaseRetriever(
        {"media_db": media_db_path} if media_db_path else {},
        user_id=str(owner_user_id),
        media_db=media_db,
    )
    return RPGService(
        repo=RPGRepository.initialized(db),
        owner_user_id=owner_user_id,
        rules_source_validator=validator,
        rules_lookup_service=RulesLookupService(
            retriever=RulesRetrievalAdapter(
                source_validator=validator,
                rag_retriever=rag_retriever,
            ),
            answer_generator=ChatRulesAnswerGenerator(),
        ),
    )


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


def _rules_pack_refs_response(result: Any) -> RPGRulesPackRefsResponse:
    return RPGRulesPackRefsResponse.model_validate(jsonable_encoder(asdict(result)))


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
    "/sessions/{session_id}/rules/lookup",
    response_model=RPGRulesLookupResponse,
    dependencies=_read_dependencies(RPG_RULES_READ, MEDIA_READ),
)
async def lookup_rules(
    session_id: int,
    request: RPGRulesLookupRequest,
    http_request: Request,
    principal: AuthPrincipal = Depends(get_auth_principal),
    service: RPGService = Depends(_rules_service),
) -> RPGRulesLookupResponse:
    if request.mode == "answer":
        await _enforce_answer_mode_generation_controls(http_request, principal)
    try:
        return RPGRulesLookupResponse.model_validate(
            jsonable_encoder(
                asdict(
                    await service.lookup_rules(
                        session_id=session_id,
                        query=request.query,
                        mode=request.mode,
                        answer_options=RulesAnswerOptions(
                            provider=request.provider,
                            model=request.model,
                            temperature=request.temperature,
                            max_tokens=request.max_tokens,
                        ),
                    )
                )
            )
        )
    except Exception as exc:
        raise _map_error(exc) from exc


@router.get(
    "/campaigns/{campaign_id}/rules-packs",
    response_model=RPGRulesPackRefsResponse,
    dependencies=_read_dependencies(RPG_CAMPAIGNS_READ, MEDIA_READ),
)
def list_campaign_rules_pack_refs(
    campaign_id: int,
    service: RPGService = Depends(_service),
) -> RPGRulesPackRefsResponse:
    try:
        return _rules_pack_refs_response(service.list_campaign_rules_pack_refs(campaign_id))
    except Exception as exc:
        raise _map_error(exc) from exc


@router.put(
    "/campaigns/{campaign_id}/rules-packs",
    response_model=RPGRulesPackRefsResponse,
    dependencies=_write_dependencies(RPG_CAMPAIGNS_MANAGE, MEDIA_READ),
)
async def replace_campaign_rules_pack_refs(
    campaign_id: int,
    request: RPGRulesPackRefsReplaceRequest,
    service: RPGService = Depends(_rules_service),
) -> RPGRulesPackRefsResponse:
    try:
        result = await service.replace_campaign_rules_pack_refs(
            campaign_id=campaign_id,
            refs=[ref.model_dump() for ref in request.refs],
            expected_version=request.expected_version,
            idempotency_key=request.idempotency_key,
        )
        return _rules_pack_refs_response(result)
    except Exception as exc:
        raise _map_error(exc) from exc


@router.get(
    "/sessions/{session_id}/rules-packs",
    response_model=RPGRulesPackRefsResponse,
    dependencies=_read_dependencies(RPG_SESSIONS_READ, MEDIA_READ),
)
def list_session_rules_pack_refs(
    session_id: int,
    service: RPGService = Depends(_service),
) -> RPGRulesPackRefsResponse:
    try:
        return _rules_pack_refs_response(service.list_session_rules_pack_refs(session_id))
    except Exception as exc:
        raise _map_error(exc) from exc


@router.put(
    "/sessions/{session_id}/rules-packs",
    response_model=RPGRulesPackRefsResponse,
    dependencies=_write_dependencies(RPG_SESSIONS_MANAGE, MEDIA_READ),
)
async def replace_session_rules_pack_refs(
    session_id: int,
    request: RPGRulesPackRefsReplaceRequest,
    service: RPGService = Depends(_rules_service),
) -> RPGRulesPackRefsResponse:
    try:
        result = await service.replace_session_rules_pack_refs(
            session_id=session_id,
            refs=[ref.model_dump() for ref in request.refs],
            expected_version=request.expected_version,
            idempotency_key=request.idempotency_key,
        )
        return _rules_pack_refs_response(result)
    except Exception as exc:
        raise _map_error(exc) from exc


@router.post(
    "/sessions/{session_id}/context",
    response_model=RPGContextResponse,
    dependencies=_read_dependencies(RPG_SESSIONS_READ, MEDIA_READ),
)
async def build_context(
    session_id: int,
    request: RPGContextBuildRequest,
    service: RPGService = Depends(_rules_service),
) -> RPGContextResponse:
    try:
        return RPGContextResponse.model_validate(
            jsonable_encoder(
                asdict(
                    await service.build_context(
                        session_id=session_id,
                        query=request.query,
                        max_chars=request.max_chars,
                    )
                )
            )
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
