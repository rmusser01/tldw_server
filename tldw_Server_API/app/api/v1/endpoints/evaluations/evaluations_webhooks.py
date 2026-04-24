"""
Webhook management endpoints extracted from evaluations_unified.
"""

import inspect
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from loguru import logger

from tldw_Server_API.app.api.v1.endpoints.evaluations.evaluations_auth import (
    get_evaluation_identity,
    get_eval_request_user,
    sanitize_error_message,
    verify_api_key,
)
from tldw_Server_API.app.api.v1.schemas.evaluation_schemas_unified import (
    WebhookRegistrationRequest,
    WebhookRegistrationResponse,
    WebhookStatusResponse,
    WebhookTestRequest,
    WebhookTestResponse,
)
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User
from tldw_Server_API.app.core.Evaluations.unified_evaluation_service import (
    get_unified_evaluation_service_for_user,
)
from tldw_Server_API.app.core.Evaluations.webhook_manager import (
    WebhookEvent,
    WebhookManager,
    webhook_manager,
)

webhooks_router = APIRouter()

_WEBHOOK_TESTMODE_EXCEPTIONS = (
    AttributeError,
    ImportError,
    RuntimeError,
    TypeError,
    ValueError,
)
_WEBHOOK_AWAIT_FALLBACK_EXCEPTIONS = (RuntimeError, TypeError, ValueError)
_WEBHOOK_ENDPOINT_EXCEPTIONS = (
    AttributeError,
    ConnectionError,
    HTTPException,
    KeyError,
    LookupError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)


def _get_webhook_manager_for_user(user_id: int | str) -> WebhookManager:
    # In tests, always route through the lazy proxy so patched methods
    # are honored and no real DB access is attempted.
    try:
        from tldw_Server_API.app.core.testing import is_test_mode as _is_test_mode
        if _is_test_mode():
            svc = get_unified_evaluation_service_for_user(user_id)
            svc.webhook_manager = webhook_manager
            return webhook_manager
    except _WEBHOOK_TESTMODE_EXCEPTIONS as e:
        logger.debug(f"Test mode detection skipped: {e}")
    service = get_unified_evaluation_service_for_user(user_id)
    manager = getattr(service, "webhook_manager", None)
    if manager is None:
        service.webhook_manager = webhook_manager
        return webhook_manager
    return manager


def _normalize_webhook_status_record(record: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(record)
    status = normalized.get("status")
    if "active" not in normalized:
        normalized["active"] = True if status is None else status == "active"

    stats = normalized.get("statistics")
    if not isinstance(stats, dict):
        stats = {}

    total = stats.get("total_deliveries", normalized.pop("total_deliveries", 0)) or 0
    failed = stats.get("failed_deliveries", normalized.pop("failure_count", 0)) or 0
    success = stats.get("successful_deliveries", total - failed)
    stats.setdefault("total_deliveries", total)
    stats.setdefault("failed_deliveries", failed)
    stats.setdefault("successful_deliveries", success)
    stats.setdefault("success_rate", (success / total) if total else 0.0)
    normalized["statistics"] = stats

    if "created_at" not in normalized or normalized["created_at"] is None:
        normalized["created_at"] = datetime.now(timezone.utc)

    return normalized


@webhooks_router.post("/webhooks", response_model=WebhookRegistrationResponse)
async def register_webhook(
    request: WebhookRegistrationRequest,
    _user_ctx: str = Depends(verify_api_key),
    current_user: User = Depends(get_eval_request_user),
):
    try:
        identity = get_evaluation_identity(current_user)
        wm = _get_webhook_manager_for_user(identity.user_scope)
        webhook_user_id = identity.webhook_user_id
        url = str(request.url)
        events = [WebhookEvent(e.value) if not isinstance(e, WebhookEvent) else e for e in request.events]
        _res = wm.register_webhook(
            user_id=webhook_user_id,
            url=url,
            secret=request.secret,
            events=events,
            retry_count=request.retry_count or 3,
            timeout_seconds=request.timeout_seconds or 30,
        )
        try:
            result = await _res if inspect.isawaitable(_res) else _res
        except _WEBHOOK_AWAIT_FALLBACK_EXCEPTIONS:
            result = _res
        return WebhookRegistrationResponse(**result)
    except _WEBHOOK_ENDPOINT_EXCEPTIONS as e:
        logger.error(f"Failed to register webhook: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to register webhook: {sanitize_error_message(e, 'webhook registration')}"
        ) from e


@webhooks_router.get("/webhooks", response_model=list[WebhookStatusResponse])
async def list_webhooks(
    _user_ctx: str = Depends(verify_api_key),
    current_user: User = Depends(get_eval_request_user),
):
    try:
        identity = get_evaluation_identity(current_user)
        wm = _get_webhook_manager_for_user(identity.user_scope)
        webhook_user_id = identity.webhook_user_id
        _res = wm.get_webhook_status(user_id=webhook_user_id)
        try:
            records = await _res if inspect.isawaitable(_res) else _res
        except _WEBHOOK_AWAIT_FALLBACK_EXCEPTIONS:
            records = _res
        normalized = [_normalize_webhook_status_record(w) for w in records]
        return [WebhookStatusResponse(**w) for w in normalized]
    except _WEBHOOK_ENDPOINT_EXCEPTIONS as e:
        logger.error(f"Failed to list webhooks: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list webhooks: {sanitize_error_message(e, 'listing webhooks')}"
        ) from e


@webhooks_router.delete("/webhooks")
async def unregister_webhook(
    url: str,
    _user_ctx: str = Depends(verify_api_key),
    current_user: User = Depends(get_eval_request_user),
):
    try:
        identity = get_evaluation_identity(current_user)
        wm = _get_webhook_manager_for_user(identity.user_scope)
        webhook_user_id = identity.webhook_user_id
        _res = wm.unregister_webhook(webhook_user_id, url)
        try:
            if inspect.isawaitable(_res):
                await _res
        except _WEBHOOK_AWAIT_FALLBACK_EXCEPTIONS:
            pass
        return {"status": "unregistered", "url": url}
    except _WEBHOOK_ENDPOINT_EXCEPTIONS as e:
        logger.error(f"Failed to unregister webhook: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to unregister webhook: {sanitize_error_message(e, 'webhook removal')}"
        ) from e


@webhooks_router.post("/webhooks/test", response_model=WebhookTestResponse)
async def test_webhook(
    payload: WebhookTestRequest,
    _user_ctx: str = Depends(verify_api_key),
    current_user: User = Depends(get_eval_request_user),
):
    try:
        identity = get_evaluation_identity(current_user)
        wm = _get_webhook_manager_for_user(identity.user_scope)
        webhook_user_id = identity.webhook_user_id
        _res = wm.test_webhook(user_id=webhook_user_id, url=str(payload.url))
        try:
            result = await _res if inspect.isawaitable(_res) else _res
        except _WEBHOOK_AWAIT_FALLBACK_EXCEPTIONS:
            result = _res
        if isinstance(result, WebhookTestResponse):
            return result
        if isinstance(result, dict):
            return WebhookTestResponse(**result)
        return WebhookTestResponse(success=bool(result))
    except _WEBHOOK_ENDPOINT_EXCEPTIONS as e:
        logger.error(f"Failed to test webhook: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to test webhook: {sanitize_error_message(e, 'webhook testing')}"
        ) from e
