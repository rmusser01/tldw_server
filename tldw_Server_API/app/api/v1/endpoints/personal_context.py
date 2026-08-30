"""Authenticated canonical Personal Context API."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Path, Query, status
from pydantic import ValidationError
from tldw_profile_core import ProfileManifest, ProfileProposal, ProfileRecord, ProfileScope

from tldw_Server_API.app.api.v1.API_Deps.personal_context_deps import (
    get_personal_context_service,
)
from tldw_Server_API.app.api.v1.schemas.personal_context import (
    ExpectedVersionRequest,
    ExportRequest,
    ExportResponse,
    ProfileCreateRequest,
    ProposalListResponse,
    ProposalReviewRequest,
    ProposalReviewResponse,
    PurgeRequest,
    RecordCreateRequest,
    RecordListResponse,
    RecordUpdateRequest,
    RuntimeUpdateRequest,
    ScopeListResponse,
    WorkspaceScopeCreateRequest,
)
from tldw_Server_API.app.core.Personalization.personal_context_repository_models import (
    ProfileIntegrityError,
    ProfileQuotaExceededError,
    ProfileStorageLockedError,
    ProfileUnsupportedSchemaError,
)
from tldw_Server_API.app.core.Personalization.personal_context_runtime_policy import (
    RuntimePolicyVersion,
)
from tldw_Server_API.app.core.Personalization.personal_context_service import (
    PersonalContextService,
    ProfileConflictError,
    ProfileKeyCollisionError,
    ProfileStatus,
    ProfileUnsupportedOperationError,
    RecordMutation,
)

router = APIRouter()
OpaquePath = Annotated[str, Path(min_length=1, max_length=128)]


@contextmanager
def _profile_api_errors(not_found: str = "Personal context profile not found"):
    """Translate service errors without returning profile content or identifiers."""

    try:
        yield
    except HTTPException:
        raise
    except KeyError:
        raise HTTPException(status_code=404, detail=not_found) from None
    except ProfileConflictError:
        raise HTTPException(
            status_code=409,
            detail={"code": "profile_version_conflict"},
        ) from None
    except ProfileKeyCollisionError:
        raise HTTPException(
            status_code=409,
            detail={"code": "profile_semantic_key_collision"},
        ) from None
    except ProfileQuotaExceededError:
        raise HTTPException(
            status_code=429,
            detail={"code": "profile_quota_exceeded"},
        ) from None
    except ProfileUnsupportedOperationError as exc:
        raise HTTPException(
            status_code=409,
            detail={"code": exc.code},
        ) from None
    except ProfileUnsupportedSchemaError:
        raise HTTPException(
            status_code=409,
            detail={"code": "profile_schema_unsupported"},
        ) from None
    except (ProfileIntegrityError, ProfileStorageLockedError):
        raise HTTPException(
            status_code=423,
            detail={"code": "profile_locked"},
        ) from None
    except (TypeError, ValueError, ValidationError):
        raise HTTPException(
            status_code=422,
            detail={"code": "invalid_personal_context_request"},
        ) from None


@router.get("/status", response_model=ProfileStatus)
def get_status(
    service: PersonalContextService = Depends(get_personal_context_service),
) -> ProfileStatus:
    """Return the authenticated user's content-free profile status."""

    return service.status()


@router.post(
    "/manifest",
    response_model=ProfileManifest,
    status_code=status.HTTP_201_CREATED,
)
def create_manifest(
    request: ProfileCreateRequest,
    service: PersonalContextService = Depends(get_personal_context_service),
) -> ProfileManifest:
    """Create one canonical profile for the authenticated user."""

    with _profile_api_errors():
        return service.create_profile(runtime_enabled=request.runtime_enabled)


@router.get("/manifest", response_model=ProfileManifest)
def get_manifest(
    service: PersonalContextService = Depends(get_personal_context_service),
) -> ProfileManifest:
    """Return the authenticated user's canonical manifest."""

    with _profile_api_errors():
        return service.get_manifest()


@router.get("/scopes", response_model=ScopeListResponse)
def list_scopes(
    service: PersonalContextService = Depends(get_personal_context_service),
) -> ScopeListResponse:
    """List the authenticated user's global and workspace scopes."""

    with _profile_api_errors():
        return ScopeListResponse(items=service.list_scopes())


@router.post(
    "/scopes/workspace",
    response_model=ProfileScope,
    status_code=status.HTTP_201_CREATED,
)
def create_workspace_scope(
    request: WorkspaceScopeCreateRequest,
    service: PersonalContextService = Depends(get_personal_context_service),
) -> ProfileScope:
    """Create a scope after workspace ownership is proven."""

    with _profile_api_errors("Workspace not found"):
        return service.create_workspace_scope(request.workspace_id, request.label)


@router.get("/records", response_model=RecordListResponse)
def list_records(
    q: str | None = Query(default=None, min_length=1, max_length=16_384),
    limit: int = Query(default=5, ge=1, le=20),
    include_archived: bool = Query(default=False),
    service: PersonalContextService = Depends(get_personal_context_service),
) -> RecordListResponse:
    """List or search a bounded authenticated record projection."""

    with _profile_api_errors():
        if q is None:
            items = service.list_records(include_archived=include_archived)[:limit]
        else:
            items = service.search_records(q, limit=limit)
        return RecordListResponse(items=items, limit=limit)


@router.post(
    "/records",
    response_model=ProfileRecord,
    status_code=status.HTTP_201_CREATED,
)
def create_record(
    request: RecordCreateRequest,
    service: PersonalContextService = Depends(get_personal_context_service),
) -> ProfileRecord:
    """Create one explicit user-authored canonical record."""

    with _profile_api_errors("Personal context scope not found"):
        return service.create_manual_record(**request.model_dump(mode="python"))


@router.get("/records/{record_id}", response_model=ProfileRecord)
def get_record(
    record_id: OpaquePath,
    service: PersonalContextService = Depends(get_personal_context_service),
) -> ProfileRecord:
    """Return one exact-user record with uniform not-found behavior."""

    with _profile_api_errors("Personal context record not found"):
        return service.get_record(record_id)


@router.patch("/records/{record_id}", response_model=ProfileRecord)
def update_record(
    record_id: OpaquePath,
    request: RecordUpdateRequest,
    service: PersonalContextService = Depends(get_personal_context_service),
) -> ProfileRecord:
    """Update one record using its immutable expected version."""

    mutation_values: dict[str, Any] = {}
    for field_name in (
        "payload",
        "semantic_key",
        "controls",
        "expires_at",
        "no_expiry",
    ):
        if field_name in request.model_fields_set:
            mutation_values[field_name] = getattr(request, field_name)
    with _profile_api_errors("Personal context record not found"):
        return service.update_record(
            record_id,
            RecordMutation(**mutation_values),
            expected_version_id=request.expected_version_id,
        )


@router.post("/records/{record_id}/archive", response_model=ProfileRecord)
def archive_record(
    record_id: OpaquePath,
    request: ExpectedVersionRequest,
    service: PersonalContextService = Depends(get_personal_context_service),
) -> ProfileRecord:
    """Archive one record with an optimistic version check."""

    with _profile_api_errors("Personal context record not found"):
        return service.archive_record(
            record_id,
            expected_version_id=request.expected_version_id,
        )


@router.post("/records/{record_id}/restore", response_model=ProfileRecord)
def restore_record(
    record_id: OpaquePath,
    request: ExpectedVersionRequest,
    service: PersonalContextService = Depends(get_personal_context_service),
) -> ProfileRecord:
    """Restore one archived record with collision protection."""

    with _profile_api_errors("Personal context record not found"):
        return service.restore_record(
            record_id,
            expected_version_id=request.expected_version_id,
        )


@router.delete("/records/{record_id}", response_model=ProfileRecord)
def delete_record(
    record_id: OpaquePath,
    request: ExpectedVersionRequest,
    service: PersonalContextService = Depends(get_personal_context_service),
) -> ProfileRecord:
    """Replace one canonical record with a content-free tombstone."""

    with _profile_api_errors("Personal context record not found"):
        return service.delete_record(
            record_id,
            expected_version_id=request.expected_version_id,
        )


@router.get("/proposals", response_model=ProposalListResponse)
def list_proposals(
    pending_only: bool = Query(default=True),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0, le=1_000),
    service: PersonalContextService = Depends(get_personal_context_service),
) -> ProposalListResponse:
    """List pending proposal bodies or content-free terminal receipts."""

    with _profile_api_errors():
        return ProposalListResponse(
            items=service.list_proposals(
                pending_only=pending_only,
                limit=limit,
                offset=offset,
            ),
            limit=limit,
            offset=offset,
        )


@router.post(
    "/proposals",
    response_model=ProfileProposal,
    status_code=status.HTTP_201_CREATED,
)
def create_proposal(
    proposal: ProfileProposal,
    service: PersonalContextService = Depends(get_personal_context_service),
) -> ProfileProposal:
    """Submit a strict canonical agent proposal for user review."""

    with _profile_api_errors("Personal context scope not found"):
        return service.create_proposal(proposal)


@router.post("/proposals/{proposal_id}/review", response_model=ProposalReviewResponse)
def review_proposal(
    proposal_id: OpaquePath,
    request: ProposalReviewRequest,
    service: PersonalContextService = Depends(get_personal_context_service),
) -> ProposalReviewResponse:
    """Accept or reject one proposal and shred its content body."""

    with _profile_api_errors("Personal context proposal not found"):
        receipt, record = service.review_proposal(proposal_id, action=request.action)
        return ProposalReviewResponse(receipt=receipt, record=record)


@router.get("/runtime", response_model=RuntimePolicyVersion)
def get_runtime(
    service: PersonalContextService = Depends(get_personal_context_service),
) -> RuntimePolicyVersion:
    """Return server-local runtime enablement and its CAS version."""

    with _profile_api_errors():
        return service.get_runtime_policy()


@router.patch("/runtime", response_model=RuntimePolicyVersion)
def update_runtime(
    request: RuntimeUpdateRequest,
    service: PersonalContextService = Depends(get_personal_context_service),
) -> RuntimePolicyVersion:
    """Update server-local runtime enablement with optimistic concurrency."""

    with _profile_api_errors():
        return service.set_runtime_enabled(
            request.enabled,
            expected_version_id=request.expected_version_id,
        )


@router.post("/export", response_model=ExportResponse)
def export_profile(
    request: ExportRequest,
    service: PersonalContextService = Depends(get_personal_context_service),
) -> ExportResponse:
    """Create an explicitly confirmed plaintext or encrypted profile export."""

    with _profile_api_errors("Personal context scope not found"):
        if request.mode == "plaintext":
            data = service.export_plaintext(
                confirmation=request.confirmation,
                scope_ids=request.scope_ids,
            )
        else:
            data = service.export_recovery(
                confirmation=request.confirmation,
                passphrase=request.passphrase or "",
            )
        return ExportResponse(mode=request.mode, data=data)


@router.post("/purge", response_model=ProfileManifest)
def purge_profile(
    request: PurgeRequest,
    service: PersonalContextService = Depends(get_personal_context_service),
) -> ProfileManifest:
    """Globally purge canonical content or refuse server-local-copy removal."""

    with _profile_api_errors():
        return service.purge_profile(**request.model_dump())
