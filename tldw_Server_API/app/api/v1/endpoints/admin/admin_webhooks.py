"""Authorized, redacted HTTP transport for canonical admin webhooks."""

from __future__ import annotations

from collections.abc import Callable, Coroutine
from datetime import datetime, timezone
from typing import Annotated, Any

from fastapi import (
    APIRouter,
    Depends,
    Header,
    HTTPException,
    Path,
    Query,
    Request,
    Response,
)
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from loguru import logger

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_auth_principal
from tldw_Server_API.app.api.v1.schemas.admin_webhooks import (
    AdminWebhookRegistrationResponse,
    AdminWebhookStatusResponse,
    WebhookCatalogItemResponse,
    WebhookCatalogResponse,
    WebhookCreateRequest,
    WebhookDeleteResponse,
    WebhookErrorDetail,
    WebhookErrorResponse,
    WebhookLimitsResponse,
    WebhookListResponse,
    WebhookMigrationStatusResponse,
    WebhookPatchRequest,
    WebhookSecretResponse,
)
from tldw_Server_API.app.core.Admin_Webhooks.audit import (
    MutationAudit,
    MutationAuditSink,
    emit_mandatory_webhook_audit,
    validate_actor_kind,
    validate_actor_principal_id,
    validate_actor_roles,
)
from tldw_Server_API.app.core.Admin_Webhooks.control_plane import (
    CreateRegistrationCommand,
    DeleteRegistrationCommand,
    PatchRegistrationCommand,
    RegistrationChanges,
    RotateSecretCommand,
    get_admin_webhook_control_plane,
)
from tldw_Server_API.app.core.Admin_Webhooks.domain import (
    WebhookError,
    WebhookErrorCode,
    WebhookRegistration,
    WebhookStatus,
    build_registration_etag,
    normalize_request_id,
)
from tldw_Server_API.app.core.Audit.unified_audit_service import (
    MandatoryAuditWriteError,
)
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal

_DOMAIN_ERRORS: dict[WebhookErrorCode, tuple[int, str]] = {
    WebhookErrorCode.VALIDATION_FAILED: (422, "Webhook request validation failed"),
    WebhookErrorCode.REQUEST_REJECTED: (400, "Webhook request was rejected"),
    WebhookErrorCode.EVENT_UNSUPPORTED: (422, "Webhook event type is not supported"),
    WebhookErrorCode.IDEMPOTENCY_KEY_INVALID: (422, "Idempotency key is invalid"),
    WebhookErrorCode.IDEMPOTENCY_CONFLICT: (409, "Idempotency key conflicts with an existing request"),
    WebhookErrorCode.IDEMPOTENCY_IN_PROGRESS: (409, "Webhook request is already in progress"),
    WebhookErrorCode.IDEMPOTENCY_RESULT_SUPERSEDED: (409, "Webhook idempotent result is no longer available"),
    WebhookErrorCode.PRECONDITION_REQUIRED: (428, "Current webhook ETag is required"),
    WebhookErrorCode.PRECONDITION_FAILED: (412, "Webhook precondition failed"),
    WebhookErrorCode.TARGET_REJECTED: (422, "Webhook target was rejected"),
    WebhookErrorCode.NOT_FOUND: (404, "Webhook registration was not found"),
    WebhookErrorCode.DISABLED: (503, "Admin webhooks are disabled"),
    WebhookErrorCode.MIGRATION_PENDING: (503, "Admin webhook migration is not complete"),
    WebhookErrorCode.REGISTRATION_LIMIT: (409, "Webhook registration limit reached"),
    WebhookErrorCode.ACTIVE_LIMIT: (409, "Active webhook registration limit reached"),
    WebhookErrorCode.SECRET_ROTATION_REQUIRED: (409, "Webhook signing-secret rotation is required"),
    WebhookErrorCode.REGISTRATION_ACTIVE: (409, "Webhook registration must be inactive"),
    WebhookErrorCode.KEY_UNAVAILABLE: (503, "Webhook encryption key is unavailable"),
    WebhookErrorCode.KEY_CONFIGURATION_MISMATCH: (
        503,
        "Webhook encryption key configuration does not match durable state",
    ),
    WebhookErrorCode.KEY_ROTATION_IN_PROGRESS: (503, "Webhook encryption-key rotation is in progress"),
    WebhookErrorCode.DATABASE_BUSY: (503, "Webhook database is temporarily unavailable"),
    WebhookErrorCode.AUDIT_UNAVAILABLE: (503, "Webhook audit persistence is unavailable"),
    WebhookErrorCode.OPERATION_FAILED: (503, "Webhook operation is temporarily unavailable"),
    WebhookErrorCode.USER_PRINCIPAL_REQUIRED: (403, "A user-backed platform administrator is required"),
    WebhookErrorCode.DELIVERY_UNAVAILABLE: (503, "Webhook delivery capability is unavailable"),
}

_AUTH_ERRORS = {
    401: ("authentication_required", "Authentication is required"),
    403: ("platform_admin_required", "Platform administrator access is required"),
    429: ("authentication_rate_limited", "Authentication is rate limited"),
    503: ("authentication_unavailable", "Authentication is temporarily unavailable"),
}
_REQUEST_REJECTED_MESSAGE = "Webhook request was rejected"
_VALIDATION_MESSAGE = "Webhook request validation failed"
_PUBLIC_ERROR_CODES = frozenset(
    {code.value for code in _DOMAIN_ERRORS}
    | {code for code, _message in _AUTH_ERRORS.values()}
    | {WebhookErrorCode.REQUEST_REJECTED.value}
)
_ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    status: {
        "model": WebhookErrorResponse,
        "description": "Bounded canonical webhook error",
    }
    for status in (401, 403, 404, 409, 412, 422, 428, 429, 500, 503)
}


def _request_id(request: Request) -> str:
    """Return the normalized request correlation identifier."""

    return normalize_request_id(getattr(request.state, "request_id", None))


def _webhook_error_response(
    *,
    code: str,
    status_code: int,
    request_id: str,
    message: str,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    if code not in _PUBLIC_ERROR_CODES:
        raise RuntimeError("unregistered canonical webhook error code")
    payload = WebhookErrorResponse(
        error=WebhookErrorDetail(
            code=code,
            message=message,
            request_id=request_id,
        )
    )
    response = JSONResponse(
        status_code=status_code,
        content=payload.model_dump(mode="json"),
        headers=headers,
    )
    response.headers["X-Request-ID"] = request_id
    response.headers["Cache-Control"] = "no-store"
    return response


def _filtered_http_exception_headers(exc: HTTPException) -> dict[str, str]:
    if not isinstance(exc.headers, dict):
        return {}
    if exc.status_code == 401 and exc.headers.get("WWW-Authenticate") == "Bearer":
        return {"WWW-Authenticate": "Bearer"}
    if exc.status_code != 429:
        return {}
    raw = exc.headers.get("Retry-After")
    if not isinstance(raw, str) or not raw.isascii() or not raw.isdecimal():
        return {}
    seconds = int(raw)
    if not 0 <= seconds <= 86_400:
        return {}
    return {"Retry-After": str(seconds)}


class AdminWebhookRoute(APIRoute):
    """Map expected failures without reflecting rejected request data."""

    def get_route_handler(
        self,
    ) -> Callable[[Request], Coroutine[Any, Any, Response]]:
        """Return a route handler that maps failures to bounded responses."""

        original = super().get_route_handler()

        async def redacted_handler(request: Request) -> Response:
            """Execute the route while redacting expected failure details."""

            try:
                return await original(request)
            except RequestValidationError:
                return _webhook_error_response(
                    code=WebhookErrorCode.VALIDATION_FAILED.value,
                    status_code=422,
                    request_id=_request_id(request),
                    message=_VALIDATION_MESSAGE,
                )
            except WebhookError as exc:
                status_code, message = _DOMAIN_ERRORS[exc.code]
                return _webhook_error_response(
                    code=exc.code.value,
                    status_code=status_code,
                    request_id=_request_id(request),
                    message=message,
                )
            except HTTPException as exc:
                raw_status = exc.status_code
                status_code = (
                    raw_status
                    if isinstance(raw_status, int) and not isinstance(raw_status, bool) and 400 <= raw_status <= 599
                    else 500
                )
                if status_code in _AUTH_ERRORS:
                    code, message = _AUTH_ERRORS[status_code]
                else:
                    code = WebhookErrorCode.REQUEST_REJECTED.value
                    message = _REQUEST_REJECTED_MESSAGE
                return _webhook_error_response(
                    code=code,
                    status_code=status_code,
                    request_id=_request_id(request),
                    message=message,
                    headers=_filtered_http_exception_headers(exc),
                )

        return redacted_handler


status_router = APIRouter(
    route_class=AdminWebhookRoute,
    responses=_ERROR_RESPONSES,
)
canonical_router = APIRouter(
    route_class=AdminWebhookRoute,
    responses=_ERROR_RESPONSES,
)


def _require_platform_admin(principal: AuthPrincipal) -> None:
    from tldw_Server_API.app.api.v1.endpoints import admin as admin_module

    admin_module._require_platform_admin(principal)


async def _emit_admin_audit_event(
    request: Request,
    principal: AuthPrincipal,
    *,
    event_type: str,
    category: str,
    resource_type: str,
    resource_id: str | None,
    action: str,
    metadata: dict[str, Any],
) -> None:
    from tldw_Server_API.app.api.v1.endpoints import admin as admin_module

    await admin_module._emit_admin_audit_event(
        request,
        principal,
        event_type=event_type,
        category=category,
        resource_type=resource_type,
        resource_id=resource_id,
        action=action,
        metadata=metadata,
    )


def _require_webhook_mutation_actor(principal: AuthPrincipal) -> int:
    _require_platform_admin(principal)
    actor_id = principal.user_id
    if isinstance(actor_id, bool) or not isinstance(actor_id, int) or actor_id < 1:
        raise WebhookError(WebhookErrorCode.USER_PRINCIPAL_REQUIRED)
    return actor_id


def _build_webhook_audit_sink(
    *,
    request_id: str,
    principal: AuthPrincipal,
    actor_id: int,
) -> MutationAuditSink:
    principal_id = validate_actor_principal_id(principal.principal_id)
    actor_kind = validate_actor_kind(principal.kind)
    actor_roles = validate_actor_roles(tuple(principal.roles))

    async def sink(record: MutationAudit) -> None:
        if record.actor_id != actor_id or record.request_id != request_id:
            raise MandatoryAuditWriteError("Mandatory audit identity mismatch")
        await emit_mandatory_webhook_audit(
            record,
            actor_principal_id=principal_id,
            actor_kind=actor_kind,
            actor_roles=actor_roles,
        )

    return sink


async def _emit_read_audit(
    *,
    request: Request,
    principal: AuthPrincipal,
    request_id: str,
    action: str,
    resource_id: int | None = None,
    outcome: str,
    reason_code: str | None = None,
    result_count: int | None = None,
    target_hostname: str | None = None,
) -> None:
    metadata: dict[str, Any] = {
        "outcome": outcome,
        "request_id": request_id,
    }
    if reason_code is not None:
        metadata["reason_code"] = reason_code
    if result_count is not None:
        metadata["result_count"] = result_count
    if target_hostname is not None:
        metadata["target_hostname"] = target_hostname
    try:
        await _emit_admin_audit_event(
            request,
            principal,
            event_type="data.read",
            category="data_access",
            resource_type="admin_webhook",
            resource_id=str(resource_id) if resource_id is not None else None,
            action=action,
            metadata=metadata,
        )
    except Exception as exc:  # noqa: BLE001 - reads retain best-effort audit semantics
        logger.warning(
            "Admin webhook read audit failed action={} request_id={} error_type={}",
            action,
            request_id,
            type(exc).__name__,
        )
        return


async def _audit_read_failure(
    *,
    request: Request,
    principal: AuthPrincipal,
    request_id: str,
    action: str,
    resource_id: int | None,
    exc: Exception,
) -> None:
    if isinstance(exc, WebhookError):
        outcome = "failed" if _DOMAIN_ERRORS[exc.code][0] >= 500 else "denied"
        reason_code = exc.code.value
    else:
        outcome = "failed"
        reason_code = WebhookErrorCode.OPERATION_FAILED.value
    await _emit_read_audit(
        request=request,
        principal=principal,
        request_id=request_id,
        action=action,
        resource_id=resource_id,
        outcome=outcome,
        reason_code=reason_code,
    )


def _registration_response(registration: WebhookRegistration) -> AdminWebhookRegistrationResponse:
    return AdminWebhookRegistrationResponse(
        id=registration.id,
        description=registration.description,
        target_display=registration.target_display,
        target_hostname=registration.target_hostname,
        event_types=list(registration.event_types),
        active=registration.active,
        timeout_seconds=registration.timeout_seconds,
        revision=registration.revision,
        delivery_config_version=registration.delivery_config_version,
        secret_version=registration.secret_version,
        secret_rotation_required=registration.secret_rotation_required,
        created_by=registration.created_by_user_id,
        updated_by=registration.updated_by_user_id,
        created_at=registration.created_at,
        updated_at=registration.updated_at,
    )


def _status_response(status: WebhookStatus) -> AdminWebhookStatusResponse:
    return AdminWebhookStatusResponse.model_validate(
        {
            "mode": status.mode,
            "route_selection": status.route_selection,
            "schema_ready": status.schema_ready,
            "key_state": status.key_state,
            "delivery_capability_ready": status.delivery_capability_ready,
            "limits": WebhookLimitsResponse.model_validate(status.limits),
            "migration": WebhookMigrationStatusResponse(
                phase=status.migration.phase,
                imported_count=status.migration.imported_count,
                unresolved_count=status.migration.unresolved_count,
                rejected_count=status.migration.rejected_count,
                secret_rotation_required_count=status.migration.secret_rotation_required_count,
                legacy_file_restore_permitted=status.migration.legacy_file_restore_permitted,
                rollback_window_expires_at=status.migration.rollback_expires_at,
            ),
        }
    )


@status_router.get("/webhooks/status", response_model=AdminWebhookStatusResponse)
async def get_webhook_status(
    request: Request,
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> AdminWebhookStatusResponse:
    _require_platform_admin(principal)
    request_id = _request_id(request)
    try:
        service = await get_admin_webhook_control_plane()
        status = await service.status()
    except Exception as exc:
        await _audit_read_failure(
            request=request,
            principal=principal,
            request_id=request_id,
            action="admin_webhook.status.read",
            resource_id=None,
            exc=exc,
        )
        raise
    await _emit_read_audit(
        request=request,
        principal=principal,
        request_id=request_id,
        action="admin_webhook.status.read",
        outcome="succeeded",
    )
    return _status_response(status)


@canonical_router.get("/webhooks/catalog", response_model=WebhookCatalogResponse)
async def get_webhook_catalog(
    request: Request,
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> WebhookCatalogResponse:
    _require_platform_admin(principal)
    request_id = _request_id(request)
    try:
        service = await get_admin_webhook_control_plane()
        catalog = await service.catalog()
    except Exception as exc:
        await _audit_read_failure(
            request=request,
            principal=principal,
            request_id=request_id,
            action="admin_webhook.catalog.read",
            resource_id=None,
            exc=exc,
        )
        raise
    await _emit_read_audit(
        request=request,
        principal=principal,
        request_id=request_id,
        action="admin_webhook.catalog.read",
        outcome="succeeded",
    )
    return WebhookCatalogResponse(
        api_version=catalog.api_version,
        events=[WebhookCatalogItemResponse.model_validate(item) for item in catalog.events],
        registration_limit=catalog.registration_limit,
        active_limit=catalog.active_limit,
    )


@canonical_router.get("/webhooks", response_model=WebhookListResponse)
async def list_webhooks(
    request: Request,
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0, le=1_000),
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> WebhookListResponse:
    _require_platform_admin(principal)
    request_id = _request_id(request)
    try:
        service = await get_admin_webhook_control_plane()
        page = await service.list_page(limit=limit, offset=offset)
    except Exception as exc:
        await _audit_read_failure(
            request=request,
            principal=principal,
            request_id=request_id,
            action="admin_webhook.list",
            resource_id=None,
            exc=exc,
        )
        raise
    await _emit_read_audit(
        request=request,
        principal=principal,
        request_id=request_id,
        action="admin_webhook.list",
        outcome="succeeded",
        result_count=len(page.items),
    )
    return WebhookListResponse(
        items=[_registration_response(item) for item in page.items],
        total=page.total,
        limit=page.limit,
        offset=page.offset,
    )


@canonical_router.post(
    "/webhooks",
    response_model=WebhookSecretResponse,
    status_code=201,
)
async def create_webhook(
    payload: WebhookCreateRequest,
    request: Request,
    response: Response,
    idempotency_key: Annotated[str, Header(alias="Idempotency-Key")],
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> WebhookSecretResponse:
    actor_id = _require_webhook_mutation_actor(principal)
    request_id = _request_id(request)
    audit_sink = _build_webhook_audit_sink(
        request_id=request_id,
        principal=principal,
        actor_id=actor_id,
    )
    service = await get_admin_webhook_control_plane()
    result = await service.create(
        CreateRegistrationCommand(
            actor_id=actor_id,
            idempotency_key=idempotency_key,
            url=payload.url,
            event_types=tuple(payload.event_types),
            description=payload.description,
            timeout_seconds=payload.timeout_seconds,
            request_id=request_id,
            now=datetime.now(timezone.utc),
        ),
        audit_sink=audit_sink,
    )
    response.headers["ETag"] = build_registration_etag(
        webhook_id=result.registration.id,
        revision=result.registration.revision,
    )
    response.headers["Cache-Control"] = "no-store"
    response.headers["Pragma"] = "no-cache"
    return WebhookSecretResponse(
        registration=_registration_response(result.registration),
        signing_secret=result.secret,
        replayed=result.replayed,
    )


@canonical_router.post(
    "/webhooks/{webhook_id}/rotate-secret",
    response_model=WebhookSecretResponse,
)
async def rotate_webhook_secret(
    request: Request,
    response: Response,
    idempotency_key: Annotated[str, Header(alias="Idempotency-Key")],
    webhook_id: int = Path(ge=1),
    if_match: Annotated[str | None, Header(alias="If-Match")] = None,
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> WebhookSecretResponse:
    actor_id = _require_webhook_mutation_actor(principal)
    request_id = _request_id(request)
    audit_sink = _build_webhook_audit_sink(
        request_id=request_id,
        principal=principal,
        actor_id=actor_id,
    )
    service = await get_admin_webhook_control_plane()
    result = await service.rotate_secret(
        RotateSecretCommand(
            actor_id=actor_id,
            webhook_id=webhook_id,
            if_match=if_match,
            idempotency_key=idempotency_key,
            request_id=request_id,
            now=datetime.now(timezone.utc),
        ),
        audit_sink=audit_sink,
    )
    response.headers["ETag"] = build_registration_etag(
        webhook_id=result.registration.id,
        revision=result.registration.revision,
    )
    response.headers["Cache-Control"] = "no-store"
    response.headers["Pragma"] = "no-cache"
    return WebhookSecretResponse(
        registration=_registration_response(result.registration),
        signing_secret=result.secret,
        replayed=result.replayed,
    )


@canonical_router.get("/webhooks/{webhook_id}", response_model=AdminWebhookRegistrationResponse)
async def get_webhook(
    request: Request,
    response: Response,
    webhook_id: int = Path(ge=1),
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> AdminWebhookRegistrationResponse:
    _require_platform_admin(principal)
    request_id = _request_id(request)
    try:
        service = await get_admin_webhook_control_plane()
        registration = await service.get(webhook_id)
    except Exception as exc:
        await _audit_read_failure(
            request=request,
            principal=principal,
            request_id=request_id,
            action="admin_webhook.get",
            resource_id=webhook_id,
            exc=exc,
        )
        raise
    await _emit_read_audit(
        request=request,
        principal=principal,
        request_id=request_id,
        action="admin_webhook.get",
        resource_id=webhook_id,
        outcome="succeeded",
        target_hostname=registration.target_hostname,
    )
    response.headers["ETag"] = build_registration_etag(
        webhook_id=registration.id,
        revision=registration.revision,
    )
    return _registration_response(registration)


@canonical_router.patch("/webhooks/{webhook_id}", response_model=AdminWebhookRegistrationResponse)
async def patch_webhook(
    payload: WebhookPatchRequest,
    request: Request,
    response: Response,
    webhook_id: int = Path(ge=1),
    if_match: Annotated[str | None, Header(alias="If-Match")] = None,
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> AdminWebhookRegistrationResponse:
    actor_id = _require_webhook_mutation_actor(principal)
    request_id = _request_id(request)
    values = payload.model_dump(exclude_unset=True)
    if "event_types" in values:
        values["event_types"] = tuple(values["event_types"])
    audit_sink = _build_webhook_audit_sink(
        request_id=request_id,
        principal=principal,
        actor_id=actor_id,
    )
    service = await get_admin_webhook_control_plane()
    result = await service.patch(
        PatchRegistrationCommand(
            actor_id=actor_id,
            webhook_id=webhook_id,
            if_match=if_match,
            changes=RegistrationChanges(**values),
            request_id=request_id,
            now=datetime.now(timezone.utc),
        ),
        audit_sink=audit_sink,
    )
    response.headers["ETag"] = build_registration_etag(
        webhook_id=result.registration.id,
        revision=result.registration.revision,
    )
    return _registration_response(result.registration)


@canonical_router.delete("/webhooks/{webhook_id}", response_model=WebhookDeleteResponse)
async def delete_webhook(
    request: Request,
    webhook_id: int = Path(ge=1),
    if_match: Annotated[str | None, Header(alias="If-Match")] = None,
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> WebhookDeleteResponse:
    actor_id = _require_webhook_mutation_actor(principal)
    request_id = _request_id(request)
    audit_sink = _build_webhook_audit_sink(
        request_id=request_id,
        principal=principal,
        actor_id=actor_id,
    )
    service = await get_admin_webhook_control_plane()
    result = await service.delete(
        DeleteRegistrationCommand(
            actor_id=actor_id,
            webhook_id=webhook_id,
            if_match=if_match,
            request_id=request_id,
            now=datetime.now(timezone.utc),
        ),
        audit_sink=audit_sink,
    )
    return WebhookDeleteResponse(deleted=True, id=result.registration.id)
