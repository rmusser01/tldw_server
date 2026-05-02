# admin/__init__.py
# Description: Admin endpoint aggregation + compatibility shims.
from __future__ import annotations

import asyncio
import os
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from loguru import logger

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import RequireRole
from tldw_Server_API.app.core.AuthNZ.alerting import (
    get_security_alert_dispatcher as _core_get_security_alert_dispatcher,
)
from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
from tldw_Server_API.app.core.AuthNZ.exceptions import DuplicateRoleError
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.AuthNZ.repos.rbac_repo import AuthnzRbacRepo
from tldw_Server_API.app.core.DB_Management.Kanban_DB import InputError, KanbanDBError
from tldw_Server_API.app.core.exceptions import ResourceNotFoundError
from tldw_Server_API.app.core.testing import is_test_mode
from tldw_Server_API.app.services import admin_profiles_service, admin_scope_service

from . import admin_acp_agents as admin_acp_agents_endpoints
from . import admin_api_keys as admin_api_keys_endpoints
from . import admin_budgets as admin_budgets_endpoints
from . import admin_bundle_ops as admin_bundle_ops_endpoints
from . import admin_byok as admin_byok_endpoints
from . import admin_circuit_breakers as admin_circuit_breakers_endpoints
from . import admin_data_ops as admin_data_ops_endpoints
from . import admin_events_stream as admin_events_stream_endpoints
from . import admin_identity_providers as admin_identity_providers_endpoints
from . import admin_impersonation as admin_impersonation_endpoints
from . import admin_llm_providers as admin_llm_providers_endpoints
from . import admin_monitoring as admin_monitoring_endpoints
from . import admin_network as admin_network_endpoints
from . import admin_ops as admin_ops_endpoints
from . import admin_orgs as admin_orgs_endpoints
from . import admin_personalization as admin_personalization_endpoints
from . import admin_profiles as admin_profiles_endpoints
from . import admin_rate_limits as admin_rate_limits_endpoints
from . import admin_rbac as admin_rbac_endpoints
from . import admin_registration as admin_registration_endpoints
from . import admin_router_analytics as admin_router_analytics_endpoints
from . import admin_sessions_mfa as admin_sessions_mfa_endpoints
from . import admin_settings as admin_settings_endpoints
from . import admin_storage_quotas as admin_storage_quotas_endpoints
from . import admin_system as admin_system_endpoints
from . import admin_tenant_provisioning as admin_tenant_provisioning_endpoints
from . import admin_tools as admin_tools_endpoints
from . import admin_usage as admin_usage_endpoints
from . import admin_user as admin_user_endpoints
from . import startup_warnings as startup_warnings_endpoints

_ADMIN_NONCRITICAL_EXCEPTIONS = (
    asyncio.CancelledError,
    asyncio.TimeoutError,
    AssertionError,
    AttributeError,
    ConnectionError,
    FileNotFoundError,
    ImportError,
    IndexError,
    KeyError,
    LookupError,
    OSError,
    PermissionError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
    UnicodeDecodeError,
    HTTPException,
    DuplicateRoleError,
    InputError,
    KanbanDBError,
    ResourceNotFoundError,
)


async def _is_postgres_backend() -> bool:
    """Return True when AuthNZ is backed by PostgreSQL."""
    try:
        pool = await get_db_pool()
    except _ADMIN_NONCRITICAL_EXCEPTIONS:
        logger.debug("Admin backend detection falling back to SQLite due to pool error")
        return False
    return bool(getattr(pool, "pool", None) is not None)


def _get_rbac_repo() -> AuthnzRbacRepo:
    """Factory for AuthnzRbacRepo used by admin RBAC endpoints."""
    return AuthnzRbacRepo()


def get_security_alert_dispatcher():
    """Compatibility shim re-export used by legacy unit tests."""
    return _core_get_security_alert_dispatcher()


# Best-effort coordination for test-time SQLite migrations
_authnz_migration_lock = asyncio.Lock()

#######################################################################################################################
# Router Configuration

router = APIRouter(
    prefix="/admin",
    tags=["admin"],
    dependencies=[Depends(RequireRole("admin"))],  # All endpoints require admin role
    responses={403: {"description": "Not authorized"}},
)

router.include_router(admin_profiles_endpoints.router)
router.include_router(admin_sessions_mfa_endpoints.router)
router.include_router(admin_byok_endpoints.router)
router.include_router(admin_llm_providers_endpoints.router)
router.include_router(admin_monitoring_endpoints.router)
router.include_router(admin_orgs_endpoints.router)
router.include_router(admin_settings_endpoints.router)
router.include_router(admin_registration_endpoints.router)
router.include_router(admin_rbac_endpoints.router)
router.include_router(admin_rate_limits_endpoints.router)
router.include_router(admin_data_ops_endpoints.router)
router.include_router(admin_ops_endpoints.router)
router.include_router(admin_system_endpoints.router)
router.include_router(admin_usage_endpoints.router)
router.include_router(admin_router_analytics_endpoints.router)
router.include_router(admin_budgets_endpoints.router)
router.include_router(admin_user_endpoints.router)
router.include_router(admin_api_keys_endpoints.router)
router.include_router(admin_tools_endpoints.router)
router.include_router(admin_personalization_endpoints.router)
router.include_router(admin_bundle_ops_endpoints.router)
router.include_router(admin_network_endpoints.router)
router.include_router(admin_circuit_breakers_endpoints.router)
router.include_router(admin_acp_agents_endpoints.router)
router.include_router(admin_events_stream_endpoints.router)
router.include_router(admin_identity_providers_endpoints.router)
router.include_router(admin_storage_quotas_endpoints.router)
router.include_router(admin_tenant_provisioning_endpoints.router)
router.include_router(admin_impersonation_endpoints.router)
router.include_router(startup_warnings_endpoints.router)


# Backend detection now standardized via core AuthNZ database helper
async def _ensure_sqlite_authnz_ready_if_test_mode() -> None:
    """Best-effort: ensure AuthNZ SQLite schema/migrations before admin ops in tests.

    In CI/pytest with SQLite, the pool can reinitialize while migrations are
    pending. This helper checks for a core table and, if missing, runs
    migrations. A module-level asyncio.Lock coordinates concurrent requests to
    avoid parallel migration attempts.
    """
    try:
        # Only act in obvious test contexts to avoid production overhead
        is_test = is_test_mode() or os.getenv("PYTEST_CURRENT_TEST") is not None
        if not is_test:
            return

        from pathlib import Path as _Path

        from tldw_Server_API.app.core.AuthNZ.database import get_db_pool as _get_db_pool

        pool = await _get_db_pool()

        # Skip if Postgres
        if getattr(pool, "pool", None) is not None:
            return

        # Acquire coordination lock to avoid concurrent migration attempts
        async with _authnz_migration_lock:
            # Re-check existence of a core table after acquiring the lock in case
            # another coroutine completed migrations while we waited
            try:
                async with pool.acquire() as conn:
                    cur = await conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='organizations'")
                    row = await cur.fetchone()
                    if row:
                        return
            except _ADMIN_NONCRITICAL_EXCEPTIONS:
                # Proceed to ensure migrations (best-effort check)
                logger.debug("AuthNZ test ensure table check failed")

            from tldw_Server_API.app.core.AuthNZ.migrations import ensure_authnz_tables as _ensure

            db_path = getattr(pool, "_sqlite_fs_path", None) or getattr(pool, "db_path", None)
            if isinstance(db_path, str) and db_path:
                path_obj = _Path(db_path)
                # Best-effort: ensure parent directories exist to avoid path issues in CI
                try:
                    path_obj.parent.mkdir(parents=True, exist_ok=True)
                except _ADMIN_NONCRITICAL_EXCEPTIONS:
                    logger.debug("AuthNZ test ensure mkdir failed")
                await asyncio.to_thread(_ensure, path_obj)
    except _ADMIN_NONCRITICAL_EXCEPTIONS:
        # Best-effort only; do not interfere with request handling
        logger.debug("AuthNZ test ensure skipped/failed")


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
    """Best-effort audit emission for admin actions."""
    try:
        actor_id_raw = getattr(request.state, "user_id", None) or principal.user_id
        try:
            actor_id = int(actor_id_raw) if actor_id_raw is not None else None
        except (TypeError, ValueError):
            actor_id = None
        if actor_id is None:
            return
        from tldw_Server_API.app.api.v1.API_Deps.Audit_DB_Deps import (
            get_or_create_audit_service_for_user_id,
        )
        from tldw_Server_API.app.core.Audit.unified_audit_service import (
            AuditContext,
            AuditEventCategory,
            AuditEventType,
        )

        audit_service = await get_or_create_audit_service_for_user_id(actor_id)
        correlation_id = request.headers.get("X-Correlation-ID") or getattr(request.state, "correlation_id", None)
        request_id = request.headers.get("X-Request-ID") or getattr(request.state, "request_id", None) or ""
        ctx = AuditContext(
            user_id=str(actor_id),
            correlation_id=correlation_id,
            request_id=request_id,
            ip_address=(request.client.host if request.client else None),
            user_agent=request.headers.get("user-agent"),
            endpoint=str(request.url.path),
            method=request.method,
        )
        await audit_service.log_event(
            event_type=AuditEventType(event_type),
            category=AuditEventCategory(category),
            context=ctx,
            resource_type=resource_type,
            resource_id=resource_id,
            action=action,
            metadata=metadata,
        )
    except _ADMIN_NONCRITICAL_EXCEPTIONS:
        logger.warning("Admin audit emission failed")


# Test shim: budgets tests monkeypatch this symbol via the admin module path.
async def emit_budget_audit_event(
    request: Request,
    principal: AuthPrincipal,
    *,
    org_id: int,
    budget_updates: dict[str, Any] | None,
    audit_changes: list[dict[str, Any]] | None,
    clear_budgets: bool,
    actor_role: str | None,
) -> None:
    from tldw_Server_API.app.services.budget_audit_service import emit_budget_audit_event as svc_emit

    await svc_emit(
        request,
        principal,
        org_id=org_id,
        budget_updates=budget_updates,
        audit_changes=audit_changes,
        clear_budgets=clear_budgets,
        actor_role=actor_role,
    )


async def _enforce_admin_user_scope(
    principal: AuthPrincipal,
    target_user_id: int,
    *,
    require_hierarchy: bool,
) -> None:
    """Enforce shared org/team membership and optional role hierarchy for admin actions."""
    await admin_scope_service.enforce_admin_user_scope(
        principal,
        target_user_id,
        require_hierarchy=require_hierarchy,
    )


def _is_platform_admin(principal: AuthPrincipal) -> bool:
    return admin_scope_service.is_platform_admin(principal)


def _require_platform_admin(principal: AuthPrincipal) -> None:
    return admin_scope_service.require_platform_admin(principal)


async def _get_admin_org_ids(principal: AuthPrincipal) -> list[int] | None:
    return await admin_scope_service.get_admin_org_ids(principal)


# Compat shim: tests import this helper from the admin module.
async def _load_bulk_user_candidates(
    *,
    principal: AuthPrincipal,
    org_id: int | None,
    team_id: int | None,
    role: str | None,
    is_active: bool | None,
    search: str | None,
    user_ids: list[int] | None,
) -> list[int]:
    return await admin_profiles_service._load_bulk_user_candidates(
        principal=principal,
        org_id=org_id,
        team_id=team_id,
        role=role,
        is_active=is_active,
        search=search,
        user_ids=user_ids,
    )
