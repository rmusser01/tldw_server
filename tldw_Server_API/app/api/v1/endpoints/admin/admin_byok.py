"""Admin BYOK endpoints for managing user and shared provider keys."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Annotated, Any, Literal, Protocol

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import (
    RequireRole,
    check_rate_limit,
    get_auth_principal,
)
from tldw_Server_API.app.api.v1.endpoints._pagination_utils import (
    build_offset_pagination_meta,
)
from tldw_Server_API.app.api.v1.schemas.user_keys import (
    AdminUserKeysResponse,
    ByokValidationRunCreateRequest,
    ByokValidationRunItem,
    ByokValidationRunListResponse,
    SharedProviderKeyResponse,
    SharedProviderKeysResponse,
    SharedProviderKeyTestRequest,
    SharedProviderKeyTestResponse,
    SharedProviderKeyUpsertRequest,
)
from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.AuthNZ.repos.byok_validation_runs_repo import (
    AuthnzByokValidationRunsRepo,
)
from tldw_Server_API.app.core.exceptions import (
    ByokValidationActiveRunError,
    ByokValidationDisabledError,
    ByokValidationRunNotFoundError,
)
from tldw_Server_API.app.services import admin_byok_service
from tldw_Server_API.app.services.admin_byok_validation_service import (
    AdminByokValidationService,
)

router = APIRouter()


class AdminByokValidationServiceProtocol(Protocol):
    """Protocol for admin BYOK validation service dependency overrides."""

    async def create_run(
        self,
        principal: AuthPrincipal,
        *,
        org_id: int | None,
        provider: str | None,
    ) -> dict[str, object]:
        ...

    async def list_runs(self, *, limit: int, offset: int) -> tuple[list[dict[str, object]], int]:
        ...

    async def get_run(self, run_id: str) -> dict[str, object]:
        ...


def _get_ensure_sqlite_authnz_ready_if_test_mode() -> Callable[[], Awaitable[None]]:
    """Return the AuthNZ test-mode readiness hook."""
    from tldw_Server_API.app.api.v1.endpoints import admin as admin_mod

    return admin_mod._ensure_sqlite_authnz_ready_if_test_mode


async def _get_byok_validation_runs_repo() -> AuthnzByokValidationRunsRepo:
    """Build the BYOK validation runs repository from the shared AuthNZ pool."""
    pool = await get_db_pool()
    repo = AuthnzByokValidationRunsRepo(pool)
    await repo.ensure_schema()
    return repo


async def get_admin_byok_validation_service(
    repo: AuthnzByokValidationRunsRepo = Depends(_get_byok_validation_runs_repo),
) -> AdminByokValidationServiceProtocol:
    """Build the admin BYOK validation service for dependency injection."""
    return AdminByokValidationService(repo=repo)


def get_byok_validation_job_enqueuer() -> Callable[[dict[str, Any]], Awaitable[str]]:
    """Return the callable used to enqueue BYOK validation Jobs."""
    from tldw_Server_API.app.services.admin_byok_validation_jobs_worker import (
        enqueue_byok_validation_run,
    )

    return enqueue_byok_validation_run


@router.get(
    "/keys/users/{user_id}",
    response_model=AdminUserKeysResponse,
    dependencies=[Depends(RequireRole("admin")), Depends(check_rate_limit)],
)
async def admin_list_user_byok_keys(
    user_id: int,
    principal: Annotated[AuthPrincipal, Depends(get_auth_principal)],
) -> AdminUserKeysResponse:
    """List BYOK keys for a given user."""
    return await admin_byok_service.list_user_keys(principal, user_id)


@router.delete(
    "/keys/users/{user_id}/{provider}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    dependencies=[Depends(RequireRole("admin")), Depends(check_rate_limit)],
)
async def admin_revoke_user_byok_key(
    user_id: int,
    provider: str,
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> Response:
    """Revoke a specific BYOK key for a user."""
    await admin_byok_service.revoke_user_key(principal, user_id, provider)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post(
    "/keys/shared",
    response_model=SharedProviderKeyResponse,
    dependencies=[Depends(RequireRole("admin")), Depends(check_rate_limit)],
)
async def admin_upsert_shared_byok_key(
    payload: SharedProviderKeyUpsertRequest,
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> SharedProviderKeyResponse:
    """Create or update a shared BYOK provider key."""
    return await admin_byok_service.upsert_shared_key(principal, payload)


@router.post(
    "/keys/shared/test",
    response_model=SharedProviderKeyTestResponse,
    dependencies=[Depends(RequireRole("admin")), Depends(check_rate_limit)],
)
async def admin_test_shared_byok_key(
    payload: SharedProviderKeyTestRequest,
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> SharedProviderKeyTestResponse:
    """Test a shared BYOK provider key and return connectivity results."""
    return await admin_byok_service.test_shared_key(principal, payload)


@router.get(
    "/keys/shared",
    response_model=SharedProviderKeysResponse,
    dependencies=[Depends(RequireRole("admin")), Depends(check_rate_limit)],
)
async def admin_list_shared_byok_keys(
    principal: AuthPrincipal = Depends(get_auth_principal),
    scope_type: Literal["org", "team"] | None = Query(None),
    scope_id: int | None = Query(None),
    provider: str | None = Query(None),
) -> SharedProviderKeysResponse:
    """List shared BYOK keys filtered by scope or provider."""
    return await admin_byok_service.list_shared_keys(
        principal,
        scope_type=scope_type,
        scope_id=scope_id,
        provider=provider,
    )


@router.delete(
    "/keys/shared/{scope_type}/{scope_id}/{provider}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    dependencies=[Depends(RequireRole("admin")), Depends(check_rate_limit)],
)
async def admin_delete_shared_byok_key(
    scope_type: Literal["org", "team"],
    scope_id: int,
    provider: str,
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> Response:
    """Delete a shared BYOK key for a given scope and provider."""
    await admin_byok_service.delete_shared_key(principal, scope_type, scope_id, provider)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post(
    "/byok/validation-runs",
    response_model=ByokValidationRunItem,
    dependencies=[Depends(RequireRole("admin")), Depends(check_rate_limit)],
)
async def admin_create_byok_validation_run(
    payload: ByokValidationRunCreateRequest,
    principal: AuthPrincipal = Depends(get_auth_principal),
    repo: AuthnzByokValidationRunsRepo = Depends(_get_byok_validation_runs_repo),
    service: AdminByokValidationServiceProtocol = Depends(get_admin_byok_validation_service),
    enqueue_run: Callable[[dict[str, Any]], Awaitable[str]] = Depends(get_byok_validation_job_enqueuer),
) -> ByokValidationRunItem:
    """Create an authoritative BYOK validation run."""
    from tldw_Server_API.app.services.admin_byok_validation_jobs_worker import (
        byok_validation_worker_enabled,
    )

    await _get_ensure_sqlite_authnz_ready_if_test_mode()()
    if not byok_validation_worker_enabled():
        raise HTTPException(status_code=503, detail="byok_validation_worker_unavailable")
    try:
        item = await service.create_run(
            principal,
            org_id=payload.org_id,
            provider=payload.provider,
        )
    except ByokValidationDisabledError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ByokValidationActiveRunError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    try:
        await enqueue_run(item)
    except Exception as exc:
        await repo.mark_failed(str(item["id"]), error_message="enqueue_failed")
        raise HTTPException(status_code=503, detail="byok_validation_enqueue_failed") from exc
    return ByokValidationRunItem(**item)


@router.get(
    "/byok/validation-runs",
    response_model=ByokValidationRunListResponse,
    dependencies=[Depends(RequireRole("admin")), Depends(check_rate_limit)],
)
async def admin_list_byok_validation_runs(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    service: AdminByokValidationServiceProtocol = Depends(get_admin_byok_validation_service),
) -> ByokValidationRunListResponse:
    """List authoritative BYOK validation runs newest-first."""
    await _get_ensure_sqlite_authnz_ready_if_test_mode()()
    items, total = await service.list_runs(limit=limit, offset=offset)
    return ByokValidationRunListResponse(
        items=[ByokValidationRunItem(**item) for item in items],
        total=total,
        limit=limit,
        offset=offset,
        pagination=build_offset_pagination_meta(
            total=total,
            limit=limit,
            offset=offset,
            count=len(items),
        ),
    )


@router.get(
    "/byok/validation-runs/{run_id}",
    response_model=ByokValidationRunItem,
    dependencies=[Depends(RequireRole("admin")), Depends(check_rate_limit)],
)
async def admin_get_byok_validation_run(
    run_id: str,
    service: AdminByokValidationServiceProtocol = Depends(get_admin_byok_validation_service),
) -> ByokValidationRunItem:
    """Return one authoritative BYOK validation run by id."""
    await _get_ensure_sqlite_authnz_ready_if_test_mode()()
    try:
        item = await service.get_run(run_id)
    except ByokValidationRunNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return ByokValidationRunItem(**item)
