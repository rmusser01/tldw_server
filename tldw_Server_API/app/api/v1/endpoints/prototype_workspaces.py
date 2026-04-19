"""Prototype workspace collaboration endpoints."""
from __future__ import annotations

import inspect
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status

from ....core.AuthNZ.User_DB_Handling import User, get_request_user
from ..schemas.prototype_workspace_schemas import (
    PrototypeCollaboratorSessionCreateRequest,
    PrototypeWorkspaceDetailResponse,
    PrototypePreviewGrantResponse,
    PrototypePreviewRenewRequest,
    PrototypePromotionCreateRequest,
    PrototypePromotionRequestResponse,
    PrototypePromotionReviewRequest,
    PrototypePromotionReviewResponse,
    PrototypeSessionJobResponse,
    PrototypeWorkspaceCreateRequest,
    PrototypeWorkspaceResponse,
    PrototypeWorkspaceSessionCreateRequest,
)

router = APIRouter(tags=["prototype-workspaces"])


def _get_repo():
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
    from tldw_Server_API.app.core.AuthNZ.repos.prototype_workspaces_repo import (
        PrototypeWorkspacesRepo,
    )

    async def _build():
        return PrototypeWorkspacesRepo(db_pool=await get_db_pool())

    return _build()


def _get_preview_broker():
    from tldw_Server_API.app.core.Prototype_Workspaces.preview_broker import (
        PrototypePreviewBroker,
    )

    async def _build():
        repo = await _maybe_await(_get_repo())
        return PrototypePreviewBroker(repo=repo)

    return _build()


def _get_service():
    from tldw_Server_API.app.core.Prototype_Workspaces.service import PrototypeWorkspaceService

    async def _build():
        repo = await _maybe_await(_get_repo())
        broker = await _maybe_await(_get_preview_broker())
        return PrototypeWorkspaceService(repo=repo, preview_broker=broker)

    return _build()


def _get_jobs_service():
    from tldw_Server_API.app.core.Prototype_Workspaces.jobs import PrototypeWorkspaceJobs

    async def _build():
        repo = await _maybe_await(_get_repo())
        return PrototypeWorkspaceJobs(repo=repo)

    return _build()


def _get_access_service():
    from tldw_Server_API.app.core.Prototype_Workspaces.access import PrototypeAccessService

    async def _build():
        repo = await _maybe_await(_get_repo())
        return PrototypeAccessService(repo)

    return _build()


async def _maybe_await(value):
    if inspect.isawaitable(value):
        return await value
    return value


def _request_nonce_or_new(value: str | None) -> str:
    candidate = str(value or "").strip()
    return candidate or f"req_{uuid.uuid4().hex}"


def _coerce_user_id(user: User) -> int:
    try:
        return int(user.id)
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Authenticated user id is not compatible with prototype workspaces",
        ) from exc


def _epoch_to_iso8601(epoch: int | None) -> str | None:
    if epoch is None:
        return None
    return datetime.fromtimestamp(int(epoch), timezone.utc).isoformat()


async def _build_workspace_detail_response(repo, workspace: dict) -> PrototypeWorkspaceDetailResponse:
    sessions = await repo.list_sessions_for_workspace(str(workspace["id"]))
    snapshots = await repo.list_snapshots_for_workspace(str(workspace["id"]))
    canonical_snapshot_id = str(workspace.get("canonical_snapshot_id") or "")
    last_known_good_snapshot_id = str(workspace.get("last_known_good_snapshot_id") or "")

    snapshot_records = [
        {
            **snapshot,
            "is_canonical": str(snapshot.get("snapshot_id") or "") == canonical_snapshot_id,
            "is_last_known_good": str(snapshot.get("snapshot_id") or "") == last_known_good_snapshot_id,
        }
        for snapshot in snapshots
    ]

    return PrototypeWorkspaceDetailResponse.model_validate(
        {
            **workspace,
            "viewer_role": "owner",
            "sessions": sessions,
            "snapshots": snapshot_records,
        }
    )


@router.post(
    "/prototype-workspaces",
    response_model=PrototypeWorkspaceResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a prototype workspace",
)
async def create_prototype_workspace(
    body: PrototypeWorkspaceCreateRequest,
    user: User = Depends(get_request_user),
):
    service = await _maybe_await(_get_service())
    workspace = await service.create_workspace(
        owner_user_id=_coerce_user_id(user),
        title=body.title,
        creation_source=body.creation_source,
        description=body.description,
        prompt=body.prompt,
        preview_policy=body.preview_policy,
        share_policy=body.share_policy,
        runtime_policy=body.runtime_policy,
        designated_promoter_ids=body.designated_promoter_ids,
    )
    return PrototypeWorkspaceResponse.model_validate(workspace)


@router.get(
    "/prototype-workspaces/{prototype_workspace_id}",
    response_model=PrototypeWorkspaceDetailResponse,
    status_code=status.HTTP_200_OK,
    summary="Get prototype workspace detail for the owner workspace view",
)
async def get_prototype_workspace(
    prototype_workspace_id: str,
    user: User = Depends(get_request_user),
):
    repo = await _maybe_await(_get_repo())
    workspace = await repo.get_workspace(prototype_workspace_id)
    if not workspace:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Prototype workspace not found")

    owner_user_id = int(workspace["owner_user_id"])
    if _coerce_user_id(user) != owner_user_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Only the owner can view prototype workspace detail")

    return await _build_workspace_detail_response(repo, workspace)


@router.post(
    "/prototype-workspaces/{prototype_workspace_id}/sessions",
    response_model=PrototypeSessionJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Create or reuse an owner branch session",
)
async def create_owner_branch_session(
    prototype_workspace_id: str,
    body: PrototypeWorkspaceSessionCreateRequest,
    user: User = Depends(get_request_user),
):
    repo = await _maybe_await(_get_repo())
    workspace = await repo.get_workspace(prototype_workspace_id)
    if not workspace:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Prototype workspace not found")

    owner_user_id = int(workspace["owner_user_id"])
    if _coerce_user_id(user) != owner_user_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Only the owner can create branch sessions")

    service = await _maybe_await(_get_service())
    request_nonce = _request_nonce_or_new(body.request_nonce)
    created = await service.create_or_reuse_branch_session(
        prototype_workspace_id=prototype_workspace_id,
        actor_type="owner",
        actor_user_id=owner_user_id,
        request_nonce=request_nonce,
    )
    session = created["session"]

    jobs = await _maybe_await(_get_jobs_service())
    job = await jobs.enqueue_branch_session_bootstrap(
        prototype_workspace_id=prototype_workspace_id,
        actor_type="owner",
        actor_user_id=owner_user_id,
        request_nonce=request_nonce,
    )

    return PrototypeSessionJobResponse(
        job_id=str(job["id"]),
        status=str(job.get("status") or "pending"),
        prototype_workspace_id=prototype_workspace_id,
        prototype_session_id=str(session["id"]),
        actor_type="owner",
        idempotency_key=job.get("idempotency_key"),
    )


@router.post(
    "/prototype-sessions",
    response_model=PrototypeSessionJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Create or reuse an external collaborator branch session",
)
async def create_external_branch_session(
    body: PrototypeCollaboratorSessionCreateRequest,
):
    access_service = await _maybe_await(_get_access_service())
    token_payload = access_service.decode_session_token(body.session_token)
    if not token_payload:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid prototype session token")

    prototype_workspace_id = str(token_payload["prototype_workspace_id"])
    shared_actor_id = str(token_payload["shared_actor_id"])
    share_link_id = int(token_payload["share_link_id"])
    request_nonce = _request_nonce_or_new(body.request_nonce)

    service = await _maybe_await(_get_service())
    created = await service.create_or_reuse_branch_session(
        prototype_workspace_id=prototype_workspace_id,
        actor_type="external_collaborator",
        actor_shared_actor_id=shared_actor_id,
        request_nonce=request_nonce,
        share_link_id=share_link_id,
        expires_at=_epoch_to_iso8601(token_payload.get("exp")),
    )
    session = created["session"]

    jobs = await _maybe_await(_get_jobs_service())
    job = await jobs.enqueue_branch_session_bootstrap(
        prototype_workspace_id=prototype_workspace_id,
        actor_type="external_collaborator",
        actor_shared_actor_id=shared_actor_id,
        request_nonce=request_nonce,
        share_link_id=share_link_id,
        expires_at=_epoch_to_iso8601(token_payload.get("exp")),
    )

    return PrototypeSessionJobResponse(
        job_id=str(job["id"]),
        status=str(job.get("status") or "pending"),
        prototype_workspace_id=prototype_workspace_id,
        prototype_session_id=str(session["id"]),
        actor_type="external_collaborator",
        shared_actor_id=shared_actor_id,
        idempotency_key=job.get("idempotency_key"),
    )


@router.post(
    "/prototype-promotions",
    response_model=PrototypePromotionRequestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a prototype promotion request",
)
async def create_promotion_request(
    body: PrototypePromotionCreateRequest,
):
    access_service = await _maybe_await(_get_access_service())
    token_payload = access_service.decode_session_token(body.session_token)
    if not token_payload:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid prototype session token")
    if str(token_payload["prototype_workspace_id"]) != body.prototype_workspace_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Session token does not match prototype workspace")

    repo = await _maybe_await(_get_repo())
    session = await repo.get_session(body.prototype_session_id)
    if not session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Prototype session not found")
    if str(session.get("prototype_workspace_id")) != body.prototype_workspace_id:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Prototype session does not belong to the requested workspace",
        )
    if str(session.get("actor_shared_actor_id") or "") != str(token_payload["shared_actor_id"]):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Session token does not authorize promotion requests for this branch session",
        )

    candidate_snapshot = await repo.get_snapshot(body.candidate_snapshot_id)
    if not candidate_snapshot:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Prototype snapshot not found")
    if str(candidate_snapshot.get("prototype_workspace_id")) != body.prototype_workspace_id:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Prototype snapshot does not belong to the requested workspace",
        )
    if str(candidate_snapshot.get("created_from_session_id") or "") != body.prototype_session_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Prototype snapshot does not belong to the requested branch session",
        )
    if str(candidate_snapshot.get("author_shared_actor_id") or "") != str(token_payload["shared_actor_id"]):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Session token does not authorize promotion requests for this snapshot",
        )

    promotion_request = await repo.create_promotion_request(
        prototype_workspace_id=body.prototype_workspace_id,
        prototype_session_id=body.prototype_session_id,
        candidate_snapshot_id=body.candidate_snapshot_id,
        requested_by_shared_actor_id=str(token_payload["shared_actor_id"]),
    )
    return PrototypePromotionRequestResponse.model_validate(promotion_request)


@router.post(
    "/prototype-promotions/{promotion_request_id}/review",
    response_model=PrototypePromotionReviewResponse,
    status_code=status.HTTP_200_OK,
    summary="Review a prototype promotion request",
)
async def review_promotion_request(
    promotion_request_id: str,
    body: PrototypePromotionReviewRequest,
    user: User = Depends(get_request_user),
):
    repo = await _maybe_await(_get_repo())
    promotion_request = await repo.get_promotion_request(promotion_request_id)
    if not promotion_request:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Prototype promotion request not found")

    workspace = await repo.get_workspace(str(promotion_request["prototype_workspace_id"]))
    if not workspace:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Prototype workspace not found")
    reviewer_user_id = _coerce_user_id(user)
    if reviewer_user_id != int(workspace["owner_user_id"]):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Only the owner can review promotion requests")

    if body.decision == "reject":
        updated = await repo.update_promotion_request(
            promotion_request_id,
            status="rejected",
            reviewed_by_user_id=reviewer_user_id,
            review_notes=body.review_notes,
        )
        if not updated:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Prototype promotion request not found")
        return PrototypePromotionReviewResponse(
            status="rejected",
            prototype_workspace_id=str(updated["prototype_workspace_id"]),
            candidate_snapshot_id=str(updated["candidate_snapshot_id"]),
            canonical_snapshot_id=workspace.get("canonical_snapshot_id"),
            details={"review_notes": updated.get("review_notes")},
        )

    service = await _maybe_await(_get_service())
    result = await service.promote_candidate(
        prototype_workspace_id=str(promotion_request["prototype_workspace_id"]),
        candidate_snapshot_id=str(promotion_request["candidate_snapshot_id"]),
        reviewer_user_id=reviewer_user_id,
        review_baseline_snapshot_id=body.review_baseline_snapshot_id,
        promotion_request_id=promotion_request_id,
        review_notes=body.review_notes,
    )
    return PrototypePromotionReviewResponse.model_validate(result)


@router.post(
    "/prototype-previews/{preview_handle}/renew",
    response_model=PrototypePreviewGrantResponse,
    status_code=status.HTTP_200_OK,
    summary="Renew a prototype preview grant",
)
async def renew_preview_grant(
    preview_handle: str,
    _body: PrototypePreviewRenewRequest,
    user: User = Depends(get_request_user),
):
    preview_broker = await _maybe_await(_get_preview_broker())
    record = preview_broker.get_preview_record(preview_handle)
    if not record:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Prototype preview handle not found")

    repo = await _maybe_await(_get_repo())
    workspace = await repo.get_workspace(str(record["prototype_workspace_id"]))
    if not workspace:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Prototype workspace not found")
    if _coerce_user_id(user) != int(workspace["owner_user_id"]):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Only the owner can renew prototype previews")

    try:
        renewed = await preview_broker.renew_preview_grant(preview_handle)
    except RuntimeError as exc:
        detail = str(exc)
        if "not found" in detail:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=detail) from exc
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=detail) from exc
    return PrototypePreviewGrantResponse.model_validate(renewed)
