"""Nested Notes Graph routes for semantic-index management."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Annotated, Any, Callable, TypeVar
from uuid import UUID

from fastapi import APIRouter, Depends, Header, HTTPException, Query, status
from fastapi.exceptions import RequestValidationError
from fastapi.routing import APIRoute
from loguru import logger
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import Response

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import (
    RequirePermission,
    TokenScopeGuard,
    User,
    get_request_user,
    principal_has_admin_bypass_claims,
    rbac_rate_limit,
)
from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.api.v1.API_Deps.jobs_deps import try_get_job_manager
from tldw_Server_API.app.api.v1.schemas.notes_semantic_index import (
    SemanticCapabilitiesResponse,
    SemanticDisableRequest,
    SemanticEnableRequest,
    SemanticHTTPErrorResponse,
    SemanticIndexMutationResponse,
    SemanticIndexStatusResponse,
    SemanticRunCancelRequest,
    SemanticRunCreateRequest,
    SemanticRunResponse,
)
from tldw_Server_API.app.core.AuthNZ.permissions import (
    NOTES_GRAPH_READ,
    NOTES_GRAPH_SEMANTIC_MANAGE,
)
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.Notes_Graph.semantic_api import (
    SemanticAPIError,
    build_notes_semantic_api,
)
from tldw_Server_API.app.core.Sync.v2.notes_link_coordinator import (
    NotesLinkDatasetConflictError,
    NotesLinkSyncInactiveDatasetError,
    resolve_notes_link_dataset_authority,
)

SEMANTIC_ERROR_MESSAGES = {
    "notes_semantic_active_generation_required": "An active semantic generation is required.",
    "notes_semantic_backend_change_requires_delete": "Delete the existing semantic index before changing vector storage backends.",
    "notes_semantic_capability_revision_conflict": "Semantic capabilities changed; refresh and retry.",
    "notes_semantic_configuration_revision_conflict": "The semantic index changed; refresh and retry.",
    "notes_semantic_dataset_authority_unavailable": "Semantic dataset authority is temporarily unavailable.",
    "notes_semantic_dataset_not_found": "The semantic dataset was not found.",
    "notes_semantic_idempotency_conflict": "The idempotency key was reused for another request.",
    "notes_semantic_invalid_request": "The semantic index request is invalid.",
    "notes_semantic_jobs_unavailable": "Semantic indexing is temporarily unavailable.",
    "notes_semantic_permission_denied": "Permission to access the Notes semantic index is required.",
    "notes_semantic_provider_unavailable": "Semantic indexing is temporarily unavailable.",
    "notes_semantic_quota_exceeded": "The semantic indexing quota has been reached.",
    "notes_semantic_run_not_found": "The requested semantic run was not found.",
    "notes_semantic_run_revision_conflict": "The semantic run changed; refresh and retry.",
    "notes_semantic_writer_conflict": "Another semantic index operation is already active.",
}
_ERROR_RESPONSES = {
    code: {"model": SemanticHTTPErrorResponse}
    for code in (403, 404, 409, 422, 429, 503)
}


def _invalid_request() -> HTTPException:
    return HTTPException(
        status_code=422,
        detail={
            "error_code": "notes_semantic_invalid_request",
            "message": SEMANTIC_ERROR_MESSAGES["notes_semantic_invalid_request"],
        },
    )


class _SemanticAPIRoute(APIRoute):
    """Translate framework validation and permission failures to feature errors."""

    def get_route_handler(self) -> Callable[[Request], Any]:
        original = super().get_route_handler()

        async def handler(request: Request) -> Response:
            try:
                return await original(request)
            except RequestValidationError as exc:
                raise _invalid_request() from exc
            except HTTPException as exc:
                if exc.status_code != status.HTTP_403_FORBIDDEN:
                    raise
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail={
                        "error_code": "notes_semantic_permission_denied",
                        "message": SEMANTIC_ERROR_MESSAGES[
                            "notes_semantic_permission_denied"
                        ],
                    },
                    headers=exc.headers,
                ) from exc

        return handler


router = APIRouter(
    tags=["notes", "notes-semantic-index"],
    responses=_ERROR_RESPONSES,
    route_class=_SemanticAPIRoute,
)

require_semantic_read = RequirePermission(NOTES_GRAPH_READ)
require_semantic_manage = RequirePermission(
    NOTES_GRAPH_READ,
    NOTES_GRAPH_SEMANTIC_MANAGE,
)

IdempotencyKeyHeader = Annotated[
    str | None,
    Header(alias="Idempotency-Key", min_length=1, max_length=256),
]
_T = TypeVar("_T")


def _dataset_key(*, owner_user_id: str, dataset_id: str | None) -> str:
    try:
        authority = resolve_notes_link_dataset_authority(
            user_id=owner_user_id,
            dataset_id=dataset_id,
        )
    except (NotesLinkDatasetConflictError, NotesLinkSyncInactiveDatasetError) as exc:
        raise SemanticAPIError(404, "notes_semantic_dataset_not_found") from exc
    except Exception as exc:  # noqa: BLE001 - resolver details stay internal
        raise SemanticAPIError(
            503,
            "notes_semantic_dataset_authority_unavailable",
        ) from exc
    if authority is None:
        raise SemanticAPIError(404, "notes_semantic_dataset_not_found")
    return authority[1].dataset_id


async def get_semantic_api(
    dataset_id: str | None = Query(default=None, min_length=1, max_length=256),
    user: User = Depends(get_request_user),
    db: Any = Depends(get_chacha_db_for_user),
    jobs: Any = Depends(try_get_job_manager),
) -> AsyncIterator[Any]:
    """Yield one owner/dataset application service and release request DB state."""

    try:
        owner = str(user.id_str)
        api = build_notes_semantic_api(
            note_db=db,
            jobs=jobs,
            owner_user_id=owner,
            dataset_id=_dataset_key(owner_user_id=owner, dataset_id=dataset_id),
        )
        yield api
    except SemanticAPIError as exc:
        raise _http_error(exc) from None
    finally:
        release = getattr(db, "release_context_connection", None)
        close = release if callable(release) else getattr(db, "close_connection", None)
        if callable(close):
            try:
                await run_in_threadpool(close)
            except Exception:  # noqa: BLE001 - release details stay internal
                logger.warning("Notes semantic request database release failed")


def _required_idempotency_key(value: str | None) -> str:
    normalized = value.strip() if isinstance(value, str) else ""
    if not normalized or len(normalized.encode("utf-8")) > 256:
        raise _invalid_request()
    return normalized


def _http_error(exc: SemanticAPIError) -> HTTPException:
    code = (
        exc.code
        if exc.code in SEMANTIC_ERROR_MESSAGES
        else "notes_semantic_provider_unavailable"
    )
    status_code = exc.status_code if code == exc.code else 503
    return HTTPException(
        status_code=status_code,
        detail={"error_code": code, "message": SEMANTIC_ERROR_MESSAGES[code]},
    )


async def _call(operation: Callable[[], _T]) -> _T:
    try:
        return await run_in_threadpool(operation)
    except SemanticAPIError as exc:
        raise _http_error(exc) from exc


@router.get(
    "/graph/semantic-index/capabilities",
    response_model=SemanticCapabilitiesResponse,
)
async def get_semantic_capabilities(
    api: Any = Depends(get_semantic_api),
    principal: AuthPrincipal = Depends(require_semantic_read),
    _rate: None = Depends(rbac_rate_limit("notes.graph.read")),
    _scope: None = Depends(
        TokenScopeGuard(
            "notes",
            require_if_present=True,
            endpoint_id="notes.graph.read",
        )
    ),
) -> SemanticCapabilitiesResponse:
    capabilities = await _call(api.capabilities)
    response = SemanticCapabilitiesResponse.model_validate(
        capabilities,
        from_attributes=True,
    )
    permissions = set(principal.permissions)
    return response.model_copy(
        update={
            "manage_authorized": principal_has_admin_bypass_claims(principal)
            or NOTES_GRAPH_SEMANTIC_MANAGE in permissions
        }
    )


@router.get(
    "/graph/semantic-index",
    response_model=SemanticIndexStatusResponse,
)
async def get_semantic_status(
    api: Any = Depends(get_semantic_api),
    _principal: Any = Depends(require_semantic_read),
    _rate: None = Depends(rbac_rate_limit("notes.graph.read")),
    _scope: None = Depends(
        TokenScopeGuard(
            "notes",
            require_if_present=True,
            endpoint_id="notes.graph.read",
        )
    ),
) -> SemanticIndexStatusResponse:
    resource = await _call(api.status)
    return SemanticIndexStatusResponse.model_validate(resource, from_attributes=True)


@router.put(
    "/graph/semantic-index",
    response_model=SemanticIndexMutationResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def enable_semantic_index(
    body: SemanticEnableRequest,
    idempotency_key: IdempotencyKeyHeader = None,
    api: Any = Depends(get_semantic_api),
    _principal: Any = Depends(require_semantic_manage),
    _rate: None = Depends(rbac_rate_limit("notes.graph.write")),
    _scope: None = Depends(
        TokenScopeGuard(
            "notes",
            require_if_present=True,
            endpoint_id="notes.graph.write",
        )
    ),
) -> SemanticIndexMutationResponse:
    mutation = await _call(
        lambda: api.enable(
            expected_revision=body.expected_revision,
            capability_revision=body.capability_revision,
            idempotency_key=_required_idempotency_key(idempotency_key),
        )
    )
    return SemanticIndexMutationResponse.model_validate(mutation, from_attributes=True)


@router.delete(
    "/graph/semantic-index",
    response_model=SemanticIndexMutationResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def disable_semantic_index(
    body: SemanticDisableRequest,
    idempotency_key: IdempotencyKeyHeader = None,
    api: Any = Depends(get_semantic_api),
    _principal: Any = Depends(require_semantic_manage),
    _rate: None = Depends(rbac_rate_limit("notes.graph.write")),
    _scope: None = Depends(
        TokenScopeGuard(
            "notes",
            require_if_present=True,
            endpoint_id="notes.graph.write",
        )
    ),
) -> SemanticIndexMutationResponse:
    mutation = await _call(
        lambda: api.disable(
            expected_revision=body.expected_revision,
            idempotency_key=_required_idempotency_key(idempotency_key),
        )
    )
    return SemanticIndexMutationResponse.model_validate(mutation, from_attributes=True)


@router.post(
    "/graph/semantic-index/runs",
    response_model=SemanticRunResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def create_semantic_run(
    body: SemanticRunCreateRequest,
    idempotency_key: IdempotencyKeyHeader = None,
    api: Any = Depends(get_semantic_api),
    _principal: Any = Depends(require_semantic_manage),
    _rate: None = Depends(rbac_rate_limit("notes.graph.write")),
    _scope: None = Depends(
        TokenScopeGuard(
            "notes",
            require_if_present=True,
            endpoint_id="notes.graph.write",
        )
    ),
) -> SemanticRunResponse:
    run = await _call(
        lambda: api.create_run(
            mode=body.mode,
            expected_revision=body.expected_revision,
            idempotency_key=_required_idempotency_key(idempotency_key),
        )
    )
    return SemanticRunResponse.model_validate(run, from_attributes=True)


@router.get(
    "/graph/semantic-index/runs/{run_id}",
    response_model=SemanticRunResponse,
)
async def get_semantic_run(
    run_id: UUID,
    api: Any = Depends(get_semantic_api),
    _principal: Any = Depends(require_semantic_read),
    _rate: None = Depends(rbac_rate_limit("notes.graph.read")),
    _scope: None = Depends(
        TokenScopeGuard(
            "notes",
            require_if_present=True,
            endpoint_id="notes.graph.read",
        )
    ),
) -> SemanticRunResponse:
    run = await _call(lambda: api.get_run(run_id=run_id))
    return SemanticRunResponse.model_validate(run, from_attributes=True)


@router.post(
    "/graph/semantic-index/runs/{run_id}/cancel",
    response_model=SemanticIndexMutationResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def cancel_semantic_run(
    run_id: UUID,
    body: SemanticRunCancelRequest,
    idempotency_key: IdempotencyKeyHeader = None,
    api: Any = Depends(get_semantic_api),
    _principal: Any = Depends(require_semantic_manage),
    _rate: None = Depends(rbac_rate_limit("notes.graph.write")),
    _scope: None = Depends(
        TokenScopeGuard(
            "notes",
            require_if_present=True,
            endpoint_id="notes.graph.write",
        )
    ),
) -> SemanticIndexMutationResponse:
    mutation = await _call(
        lambda: api.cancel_run(
            run_id=run_id,
            expected_revision=body.expected_revision,
            idempotency_key=_required_idempotency_key(idempotency_key),
        )
    )
    return SemanticIndexMutationResponse.model_validate(mutation, from_attributes=True)


__all__ = [
    "get_semantic_api",
    "require_semantic_manage",
    "require_semantic_read",
    "router",
]
