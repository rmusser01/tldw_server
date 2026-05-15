"""Prototype workspace collaboration endpoints."""
from __future__ import annotations

import inspect
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status

from ....core.AuthNZ.User_DB_Handling import User, get_request_user
from ....core.AuthNZ.repos.prototype_workspaces_repo import InactivePrototypeSharedActorError
from ..schemas.prototype_workspace_schemas import (
    PrototypeCollaboratorSessionCreateRequest,
    PrototypePreviewGrantResponse,
    PrototypePreviewRenewRequest,
    PrototypePromotionCreateRequest,
    PrototypePromotionRequestResponse,
    PrototypePromotionReviewRequest,
    PrototypePromotionReviewResponse,
    PrototypeSessionJobResponse,
    PrototypeWorkspaceCreateRequest,
    PrototypeWorkspaceDetailResponse,
    PrototypeWorkspaceResponse,
    PrototypeWorkspaceSessionCreateRequest,
)

router = APIRouter(tags=["prototype-workspaces"])


def _get_repo() -> Any:
    """Build the prototype workspace repository from the AuthNZ database pool."""
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
    from tldw_Server_API.app.core.AuthNZ.repos.prototype_workspaces_repo import (
        PrototypeWorkspacesRepo,
    )

    async def _build() -> Any:
        return PrototypeWorkspacesRepo(db_pool=await get_db_pool())

    return _build()


def _get_preview_broker() -> Any:
    """Build the preview broker dependency with the shared prototype repo."""
    from tldw_Server_API.app.core.Prototype_Workspaces.preview_broker import (
        PrototypePreviewBroker,
    )

    async def _build() -> Any:
        repo = await _maybe_await(_get_repo())
        return PrototypePreviewBroker(repo=repo)

    return _build()


def _get_service() -> Any:
    """Build the orchestration service dependency for workspace operations."""
    from tldw_Server_API.app.core.Prototype_Workspaces.service import PrototypeWorkspaceService

    async def _build() -> Any:
        repo = await _maybe_await(_get_repo())
        broker = await _maybe_await(_get_preview_broker())
        return PrototypeWorkspaceService(repo=repo, preview_broker=broker)

    return _build()


def _get_jobs_service() -> Any:
    """Build the runtime jobs dependency for prototype branch sessions."""
    from tldw_Server_API.app.core.Prototype_Workspaces.jobs import PrototypeWorkspaceJobs

    async def _build() -> Any:
        repo = await _maybe_await(_get_repo())
        return PrototypeWorkspaceJobs(repo=repo)

    return _build()


def _get_access_service() -> Any:
    """Build the external collaborator access dependency."""
    from tldw_Server_API.app.core.Prototype_Workspaces.access import PrototypeAccessService

    async def _build() -> Any:
        repo = await _maybe_await(_get_repo())
        return PrototypeAccessService(repo)

    return _build()


async def _repo_dependency() -> Any:
    """Resolve the repository factory through FastAPI dependency injection."""
    return await _maybe_await(_get_repo())


async def _preview_broker_dependency() -> Any:
    """Resolve the preview broker factory through FastAPI dependency injection."""
    return await _maybe_await(_get_preview_broker())


async def _service_dependency() -> Any:
    """Resolve the orchestration service factory through FastAPI dependency injection."""
    return await _maybe_await(_get_service())


async def _jobs_service_dependency() -> Any:
    """Resolve the jobs service factory through FastAPI dependency injection."""
    return await _maybe_await(_get_jobs_service())


async def _access_service_dependency() -> Any:
    """Resolve the access service factory through FastAPI dependency injection."""
    return await _maybe_await(_get_access_service())


async def _maybe_await(value: Any) -> Any:
    """Await factory results that are coroutines while preserving sync test doubles."""
    if inspect.isawaitable(value):
        return await value
    return value


def _request_nonce_or_new(value: str | None) -> str:
    """Return a provided idempotency nonce or generate a request-scoped nonce."""
    candidate = str(value or "").strip()
    return candidate or f"req_{uuid.uuid4().hex}"


def _coerce_user_id(user: User) -> int:
    """Convert the authenticated user id into the integer form used by AuthNZ repos."""
    try:
        return int(user.id)
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Authenticated user id is not compatible with prototype workspaces",
        ) from exc


def _epoch_to_iso8601(epoch: int | None) -> str | None:
    """Convert a JWT epoch timestamp into an ISO-8601 string for persisted sessions."""
    if epoch is None:
        return None
    return datetime.fromtimestamp(int(epoch), timezone.utc).isoformat()


def _parse_optional_datetime(value: Any) -> datetime | None:
    """Parse optional timestamps from repo rows without raising on bad data."""
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            return None
    else:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def _is_inactive_prototype_record(record: dict[str, Any] | None) -> bool:
    """Return true for missing, revoked, or expired prototype session records."""
    if not record:
        return True
    if record.get("is_revoked") or record.get("revoked_at"):
        return True
    raw_expires_at = record.get("expires_at")
    expires_at = _parse_optional_datetime(raw_expires_at)
    if raw_expires_at not in (None, "") and expires_at is None:
        return True
    return bool(expires_at and expires_at <= datetime.now(timezone.utc))


def _coerce_optional_int(value: Any) -> int | None:
    """Coerce integer-ish values without leaking malformed identity details."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _inactive_prototype_session_error() -> HTTPException:
    """Build the stable response for inactive external collaborator sessions."""
    return HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Prototype session token is no longer active",
    )


async def _assert_external_promotion_actor_active(
    repo: Any,
    *,
    session: dict[str, Any],
    token_payload: dict[str, Any],
) -> None:
    """Verify a collaborator session token still maps to an active shared actor."""
    if _is_inactive_prototype_record(session):
        raise _inactive_prototype_session_error()

    shared_actor_id = str(token_payload.get("shared_actor_id") or "").strip()
    actor = await repo.get_shared_actor(shared_actor_id)
    if _is_inactive_prototype_record(actor):
        raise _inactive_prototype_session_error()

    token_workspace_id = str(token_payload.get("prototype_workspace_id") or "").strip()
    token_share_link_id = _coerce_optional_int(token_payload.get("share_link_id"))
    session_share_link_id = _coerce_optional_int(session.get("share_link_id"))
    actor_share_link_id = _coerce_optional_int(actor.get("share_link_id"))
    if (
        str(actor.get("prototype_workspace_id") or "") != token_workspace_id
        or token_share_link_id is None
        or session_share_link_id != token_share_link_id
        or actor_share_link_id != token_share_link_id
    ):
        raise _inactive_prototype_session_error()


def _branch_session_http_error(exc: ValueError | RuntimeError) -> HTTPException:
    """Map expected branch-session domain failures to stable HTTP responses."""
    detail = str(exc).lower()
    if "not found" in detail:
        return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Prototype workspace not found")
    if "archived" in detail:
        return HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Prototype workspace is archived")
    if "revoked" in detail or "expired" in detail:
        return HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Prototype session token is no longer active",
        )
    if "canonical snapshot" in detail or "base_snapshot_id" in detail:
        return HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Prototype workspace is not ready for branch sessions",
        )
    if isinstance(exc, ValueError):
        return HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Invalid prototype branch session request",
        )
    return HTTPException(
        status_code=status.HTTP_409_CONFLICT,
        detail="Prototype branch session could not be created",
    )


async def _build_workspace_detail_response(repo: Any, workspace: dict[str, Any]) -> PrototypeWorkspaceDetailResponse:
    """Build an owner-facing workspace detail response with sessions and snapshots."""
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
    service: Any = Depends(_service_dependency),
) -> PrototypeWorkspaceResponse:
    """Create a prototype workspace and seed its canonical snapshot for the owner."""
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
    repo: Any = Depends(_repo_dependency),
) -> PrototypeWorkspaceDetailResponse:
    """Return owner-visible prototype workspace detail and branch inventory."""
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
    repo: Any = Depends(_repo_dependency),
    service: Any = Depends(_service_dependency),
    jobs: Any = Depends(_jobs_service_dependency),
) -> PrototypeSessionJobResponse:
    """Create or reuse an owner branch session and enqueue its bootstrap job."""
    workspace = await repo.get_workspace(prototype_workspace_id)
    if not workspace:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Prototype workspace not found")

    owner_user_id = int(workspace["owner_user_id"])
    if _coerce_user_id(user) != owner_user_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Only the owner can create branch sessions")

    request_nonce = _request_nonce_or_new(body.request_nonce)
    try:
        created = await service.create_or_reuse_branch_session(
            prototype_workspace_id=prototype_workspace_id,
            actor_type="owner",
            actor_user_id=owner_user_id,
            request_nonce=request_nonce,
        )
    except (RuntimeError, ValueError) as exc:
        raise _branch_session_http_error(exc) from exc
    session = created["session"]

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
    access_service: Any = Depends(_access_service_dependency),
    service: Any = Depends(_service_dependency),
    jobs: Any = Depends(_jobs_service_dependency),
) -> PrototypeSessionJobResponse:
    """Create or reuse an external collaborator branch session from a session token."""
    token_payload = access_service.decode_session_token(body.session_token)
    if not token_payload:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid prototype session token")

    prototype_workspace_id = str(token_payload["prototype_workspace_id"])
    shared_actor_id = str(token_payload["shared_actor_id"])
    share_link_id = int(token_payload["share_link_id"])
    request_nonce = _request_nonce_or_new(body.request_nonce)

    try:
        created = await service.create_or_reuse_branch_session(
            prototype_workspace_id=prototype_workspace_id,
            actor_type="external_collaborator",
            actor_shared_actor_id=shared_actor_id,
            request_nonce=request_nonce,
            share_link_id=share_link_id,
            expires_at=_epoch_to_iso8601(token_payload.get("exp")),
        )
    except (RuntimeError, ValueError) as exc:
        raise _branch_session_http_error(exc) from exc
    session = created["session"]

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
    access_service: Any = Depends(_access_service_dependency),
    repo: Any = Depends(_repo_dependency),
) -> PrototypePromotionRequestResponse:
    """Create a promotion request for an external collaborator-owned snapshot."""
    token_payload = access_service.decode_session_token(body.session_token)
    if not token_payload:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid prototype session token")
    if str(token_payload["prototype_workspace_id"]) != body.prototype_workspace_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Session token does not match prototype workspace")

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
    await _assert_external_promotion_actor_active(
        repo,
        session=session,
        token_payload=token_payload,
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

    try:
        promotion_request = await repo.create_promotion_request(
            prototype_workspace_id=body.prototype_workspace_id,
            prototype_session_id=body.prototype_session_id,
            candidate_snapshot_id=body.candidate_snapshot_id,
            requested_by_shared_actor_id=str(token_payload["shared_actor_id"]),
        )
    except InactivePrototypeSharedActorError as exc:
        raise _inactive_prototype_session_error() from exc
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
    repo: Any = Depends(_repo_dependency),
    service: Any = Depends(_service_dependency),
) -> PrototypePromotionReviewResponse:
    """Review a promotion request and promote or reject the candidate snapshot."""
    promotion_request = await repo.get_promotion_request(promotion_request_id)
    if not promotion_request:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Prototype promotion request not found")

    workspace = await repo.get_workspace(str(promotion_request["prototype_workspace_id"]))
    if not workspace:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Prototype workspace not found")
    reviewer_user_id = _coerce_user_id(user)
    if not service._is_promoter(workspace, reviewer_user_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Reviewer does not have promotion permissions",
        )

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
    body: PrototypePreviewRenewRequest,
    user: User = Depends(get_request_user),
    preview_broker: Any = Depends(_preview_broker_dependency),
    repo: Any = Depends(_repo_dependency),
) -> PrototypePreviewGrantResponse:
    """Renew an owner-authorized prototype preview grant."""
    body.model_dump()
    record = preview_broker.get_preview_record(preview_handle)
    if not record:
        record = await preview_broker.get_preview_record_async(preview_handle)
    if not record:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Prototype preview handle not found")

    workspace = await repo.get_workspace(str(record["prototype_workspace_id"]))
    if not workspace:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Prototype workspace not found")
    if _coerce_user_id(user) != int(workspace["owner_user_id"]):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Only the owner can renew prototype previews")

    from tldw_Server_API.app.core.Prototype_Workspaces.preview_broker import (
        PrototypePreviewHandleNotFound,
    )

    try:
        renewed = await preview_broker.renew_preview_grant(preview_handle)
    except PrototypePreviewHandleNotFound as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except RuntimeError as exc:
        detail = str(exc)
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=detail) from exc
    return PrototypePreviewGrantResponse.model_validate(renewed)
