from __future__ import annotations

from dataclasses import asdict
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, status
from fastapi.encoders import jsonable_encoder

from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.api.v1.API_Deps.Collections_DB_Deps import get_collections_db_for_user
from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user
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
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.Collections_DB import CollectionsDatabase
from tldw_Server_API.app.core.DB_Management.RPG_DB import RPGRepository
from tldw_Server_API.app.core.RPG.errors import RPGConflictError, RPGNotFoundError, RPGValidationError
from tldw_Server_API.app.core.RPG.rules.refs import RulesPackSourceValidation
from tldw_Server_API.app.core.RPG.service import RPGService

router = APIRouter(prefix="/rpg", tags=["rpg"])

RPG_RULES_READ = "rpg.rules.read"
RPG_CAMPAIGNS_READ = "rpg.campaigns.read"
RPG_CAMPAIGNS_MANAGE = "rpg.campaigns.manage"
RPG_SESSIONS_READ = "rpg.sessions.read"
RPG_SESSIONS_MANAGE = "rpg.sessions.manage"
RPG_PROPOSALS_REVIEW = "rpg.proposals.review"
MEDIA_READ = "media.read"
_READY_COLLECTION_ITEM_STATUSES = {"completed", "skipped_existing"}


class RPGRulesSourceValidator:
    def __init__(self, media_db: Any, collections_db: CollectionsDatabase) -> None:
        self.media_db = media_db
        self.collections_db = collections_db

    async def validate_media_item(self, owner_user_id: int, media_id: int) -> RulesPackSourceValidation:
        media = self._readable_media_by_id(owner_user_id=owner_user_id, media_id=media_id)
        if not media:
            return RulesPackSourceValidation(
                ref_id=f"media_item:{media_id}",
                readable=False,
                display_name=None,
            )
        return RulesPackSourceValidation(
            ref_id=f"media_item:{media_id}",
            readable=True,
            display_name=_media_display_name(media),
            ready_media_ids=[int(media_id)],
        )

    async def validate_media_collection(
        self,
        owner_user_id: int,
        collection_id: int,
    ) -> RulesPackSourceValidation:
        try:
            collection = self.collections_db.get_media_collection(collection_id)
        except KeyError:
            return RulesPackSourceValidation(
                ref_id=f"media_collection:{collection_id}",
                readable=False,
                display_name=None,
            )

        ready_media_ids = [
            int(item.media_id)
            for item in collection.items
            if item.media_id is not None
            and item.status in _READY_COLLECTION_ITEM_STATUSES
            and self._readable_media_by_id(owner_user_id=owner_user_id, media_id=int(item.media_id)) is not None
        ]
        return RulesPackSourceValidation(
            ref_id=f"media_collection:{collection_id}",
            readable=True,
            display_name=collection.name,
            ready_media_ids=ready_media_ids,
        )

    def _readable_media_by_id(self, *, owner_user_id: int, media_id: int) -> dict[str, Any] | None:
        media = self.media_db.get_media_by_id(media_id, include_deleted=False, include_trash=False)
        if not media:
            return None
        if not _media_belongs_to_owner(media, owner_user_id=owner_user_id, media_db=self.media_db):
            return None
        return media


def _read_dependencies(*scopes: str):
    primary_scope = scopes[0]
    return [
        Depends(rbac_rate_limit(primary_scope)),
        Depends(RequirePermission(*scopes)),
        Depends(TokenScopeGuard("rpg", require_if_present=True, endpoint_id=primary_scope, count_as="call")),
    ]


def _write_dependencies(*scopes: str):
    return _read_dependencies(*scopes)


def _owner_user_id(current_user: User) -> int:
    if current_user.id_int is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="invalid_user_id")
    return int(current_user.id_int)


def _media_display_name(media: dict[str, Any]) -> str | None:
    for key in ("title", "name", "filename", "url"):
        value = str(media.get(key) or "").strip()
        if value:
            return value
    return None


def _media_belongs_to_owner(media: dict[str, Any], *, owner_user_id: int, media_db: Any) -> bool:
    allowed_owner_ids = {str(owner_user_id)}
    client_id = getattr(media_db, "client_id", None)
    if client_id is not None:
        allowed_owner_ids.add(str(client_id))

    owner_value = media.get("owner_user_id")
    if owner_value is not None:
        owner_text = str(owner_value).strip()
        return not owner_text or owner_text in allowed_owner_ids

    client_value = media.get("client_id")
    if client_value is not None:
        client_text = str(client_value).strip()
        return not client_text or client_text in allowed_owner_ids

    return True


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
    return RPGService(
        repo=RPGRepository.initialized(db),
        owner_user_id=_owner_user_id(current_user),
        rules_source_validator=RPGRulesSourceValidator(media_db=media_db, collections_db=collections_db),
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
def lookup_rules(
    session_id: int,
    request: RPGRulesLookupRequest,
    service: RPGService = Depends(_service),
) -> RPGRulesLookupResponse:
    try:
        return RPGRulesLookupResponse.model_validate(
            jsonable_encoder(asdict(service.lookup_rules(session_id=session_id, query=request.query)))
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
    dependencies=_read_dependencies(RPG_SESSIONS_READ),
)
def build_context(
    session_id: int,
    request: RPGContextBuildRequest,
    service: RPGService = Depends(_service),
) -> RPGContextResponse:
    try:
        return RPGContextResponse.model_validate(
            jsonable_encoder(
                asdict(
                    service.build_context(
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
