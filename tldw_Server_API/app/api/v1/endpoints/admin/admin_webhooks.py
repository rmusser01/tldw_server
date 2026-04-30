"""Admin webhook management endpoints — CRUD + delivery log + test ping."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from loguru import logger

from tldw_Server_API.app.api.v1.endpoints._pagination_utils import build_offset_pagination_meta
from tldw_Server_API.app.api.v1.schemas.admin_schemas import (
    AdminWebhookCreateRequest,
    AdminWebhookDeleteResponse,
    AdminWebhookDeliveryLogListResponse,
    AdminWebhookListResponse,
    AdminWebhookResponse,
    AdminWebhookTestResponse,
    AdminWebhookUpdateRequest,
)
from tldw_Server_API.app.services.admin_webhooks_service import get_admin_webhooks_service

router = APIRouter()


@router.post("/webhooks", response_model=AdminWebhookResponse)
async def create_webhook(payload: AdminWebhookCreateRequest) -> AdminWebhookResponse:
    """Register a new admin webhook."""
    svc = get_admin_webhooks_service()
    record = await svc.create_webhook(
        url=payload.url,
        event_types=payload.event_types,
        description=payload.description,
        secret=payload.secret,
        active=payload.active,
        retry_count=payload.retry_count,
        timeout_seconds=payload.timeout_seconds,
    )
    logger.info("Created admin webhook id={}", record.id)
    return AdminWebhookResponse(
        id=record.id,
        url=record.url,
        event_types=record.event_types,
        description=record.description,
        active=record.active,
        retry_count=record.retry_count,
        timeout_seconds=record.timeout_seconds,
        created_by=record.created_by,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


@router.get("/webhooks", response_model=AdminWebhookListResponse)
async def list_webhooks(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    active_only: bool = Query(False),
) -> AdminWebhookListResponse:
    """List all admin webhooks."""
    svc = get_admin_webhooks_service()
    items, total = await svc.list_webhooks(limit=limit, offset=offset, active_only=active_only)
    return AdminWebhookListResponse(
        items=[
            AdminWebhookResponse(
                id=r.id, url=r.url, event_types=r.event_types,
                description=r.description, active=r.active,
                retry_count=r.retry_count, timeout_seconds=r.timeout_seconds,
                created_by=r.created_by, created_at=r.created_at,
                updated_at=r.updated_at,
            )
            for r in items
        ],
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


@router.get("/webhooks/{webhook_id}", response_model=AdminWebhookResponse)
async def get_webhook(webhook_id: int) -> AdminWebhookResponse:
    """Get a single admin webhook by ID."""
    svc = get_admin_webhooks_service()
    record = await svc.get_webhook(webhook_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Webhook not found")
    return AdminWebhookResponse(
        id=record.id, url=record.url, event_types=record.event_types,
        description=record.description, active=record.active,
        retry_count=record.retry_count, timeout_seconds=record.timeout_seconds,
        created_by=record.created_by, created_at=record.created_at,
        updated_at=record.updated_at,
    )


@router.patch("/webhooks/{webhook_id}", response_model=AdminWebhookResponse)
async def update_webhook(
    webhook_id: int,
    payload: AdminWebhookUpdateRequest,
) -> AdminWebhookResponse:
    """Update an existing admin webhook."""
    svc = get_admin_webhooks_service()
    existing = await svc.get_webhook(webhook_id)
    if existing is None:
        raise HTTPException(status_code=404, detail="Webhook not found")

    record = await svc.update_webhook(
        webhook_id,
        url=payload.url,
        event_types=payload.event_types,
        description=payload.description,
        secret=payload.secret,
        active=payload.active,
        retry_count=payload.retry_count,
        timeout_seconds=payload.timeout_seconds,
    )
    if record is None:
        raise HTTPException(status_code=404, detail="Webhook not found")
    logger.info("Updated admin webhook id={}", webhook_id)
    return AdminWebhookResponse(
        id=record.id, url=record.url, event_types=record.event_types,
        description=record.description, active=record.active,
        retry_count=record.retry_count, timeout_seconds=record.timeout_seconds,
        created_by=record.created_by, created_at=record.created_at,
        updated_at=record.updated_at,
    )


@router.delete("/webhooks/{webhook_id}", response_model=AdminWebhookDeleteResponse)
async def delete_webhook(webhook_id: int) -> AdminWebhookDeleteResponse:
    """Delete an admin webhook."""
    svc = get_admin_webhooks_service()
    existing = await svc.get_webhook(webhook_id)
    if existing is None:
        raise HTTPException(status_code=404, detail="Webhook not found")
    await svc.delete_webhook(webhook_id)
    logger.info("Deleted admin webhook id={}", webhook_id)
    return AdminWebhookDeleteResponse(deleted=True, id=webhook_id)


@router.post("/webhooks/{webhook_id}/test", response_model=AdminWebhookTestResponse)
async def test_webhook(webhook_id: int) -> AdminWebhookTestResponse:
    """Send a test ping to a webhook."""
    svc = get_admin_webhooks_service()
    existing = await svc.get_webhook(webhook_id)
    if existing is None:
        raise HTTPException(status_code=404, detail="Webhook not found")
    result = await svc.test_webhook(webhook_id)
    return AdminWebhookTestResponse(**result)


@router.get(
    "/webhooks/{webhook_id}/deliveries",
    response_model=AdminWebhookDeliveryLogListResponse,
)
async def list_webhook_deliveries(
    webhook_id: int,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> AdminWebhookDeliveryLogListResponse:
    """List delivery log entries for a webhook."""
    svc = get_admin_webhooks_service()
    existing = await svc.get_webhook(webhook_id)
    if existing is None:
        raise HTTPException(status_code=404, detail="Webhook not found")
    items, total = await svc.list_delivery_log(webhook_id, limit=limit, offset=offset)
    return AdminWebhookDeliveryLogListResponse(
        items=[
            {
                "id": e.id,
                "webhook_id": e.webhook_id,
                "event_type": e.event_type,
                "status_code": e.status_code,
                "latency_ms": e.latency_ms,
                "retry_attempt": e.retry_attempt,
                "error_message": e.error_message,
                "delivered_at": e.delivered_at,
                "created_at": e.created_at,
            }
            for e in items
        ],
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
