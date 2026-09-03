from __future__ import annotations

import hmac
import os

from fastapi import APIRouter, Depends, Header, HTTPException, status

from tldw_Server_API.app.api.v1.schemas.test_support_schemas import (
    AdminE2EBootstrapJwtSessionRequest,
    AdminE2EBootstrapJwtSessionResponse,
    AdminE2EPrepareAdminWebhooksResponse,
    AdminE2EResetResponse,
    AdminE2ERunDueBackupSchedulesResponse,
    AdminE2ESeedRequest,
    AdminE2ESeedResponse,
)
from tldw_Server_API.app.services.admin_e2e_support_service import (
    bootstrap_admin_e2e_jwt_session,
    prepare_admin_webhooks_for_admin_e2e,
    reset_admin_e2e_state,
    run_due_backup_schedules_for_admin_e2e,
    seed_admin_e2e_scenario,
)

ADMIN_E2E_SUPPORT_HEADER = "X-TLDW-Admin-E2E-Key"
ADMIN_E2E_SUPPORT_KEY_ENV = "TLDW_ADMIN_E2E_SUPPORT_KEY"


def require_admin_e2e_support_access(
    support_key: str | None = Header(None, alias=ADMIN_E2E_SUPPORT_HEADER),
) -> None:
    """Fail closed unless the caller presents the explicit admin e2e support key."""
    configured_support_key = os.getenv(ADMIN_E2E_SUPPORT_KEY_ENV, "").strip()
    if not configured_support_key:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="admin_e2e_support_key_not_configured",
        )

    if not support_key or not hmac.compare_digest(support_key, configured_support_key):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="admin_e2e_support_access_denied",
        )


router = APIRouter(dependencies=[Depends(require_admin_e2e_support_access)])


@router.post("/reset", response_model=AdminE2EResetResponse)
async def reset_admin_e2e() -> AdminE2EResetResponse:
    """Reset transient admin e2e seed state."""
    return AdminE2EResetResponse(**(await reset_admin_e2e_state()))


@router.post("/seed", response_model=AdminE2ESeedResponse)
async def seed_admin_e2e(request: AdminE2ESeedRequest) -> AdminE2ESeedResponse:
    """Seed deterministic AuthNZ fixtures for admin-ui real-backend browser tests."""
    payload = await seed_admin_e2e_scenario(request.scenario)
    return AdminE2ESeedResponse(**payload)


@router.post("/bootstrap-jwt-session", response_model=AdminE2EBootstrapJwtSessionResponse)
async def bootstrap_jwt_session(
    request: AdminE2EBootstrapJwtSessionRequest,
) -> AdminE2EBootstrapJwtSessionResponse:
    """Return browser cookie payloads for a seeded multi-user admin session."""
    payload = await bootstrap_admin_e2e_jwt_session(request.principal_key)
    return AdminE2EBootstrapJwtSessionResponse(**payload)


@router.post("/run-due-backup-schedules", response_model=AdminE2ERunDueBackupSchedulesResponse)
async def run_due_backup_schedules() -> AdminE2ERunDueBackupSchedulesResponse:
    """Trigger a deterministic scheduler tick for backup schedule browser tests."""
    return AdminE2ERunDueBackupSchedulesResponse(**(await run_due_backup_schedules_for_admin_e2e()))


@router.post(
    "/prepare-admin-webhooks",
    response_model=AdminE2EPrepareAdminWebhooksResponse,
)
async def prepare_admin_webhooks() -> AdminE2EPrepareAdminWebhooksResponse:
    """Prepare a fresh e2e database for the canonical webhook runtime."""
    return AdminE2EPrepareAdminWebhooksResponse(
        **(await prepare_admin_webhooks_for_admin_e2e())
    )
