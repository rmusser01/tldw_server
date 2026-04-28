"""
billing_deps.py

FastAPI dependencies for billing and limit enforcement.
Provides guards that check subscription limits before allowing operations.
"""
from __future__ import annotations

from typing import Any

from fastapi import Depends, Header, HTTPException, Query, Response, status
from loguru import logger

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_auth_principal
from tldw_Server_API.app.api.v1.API_Deps.org_deps import _is_membership_active
from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.AuthNZ.repos.orgs_teams_repo import AuthnzOrgsTeamsRepo
from tldw_Server_API.app.core.Billing.enforcement import (
    EnforcementAction,
    LimitCategory,
    LimitCheckResult,
    enforcement_enabled,
    get_billing_enforcer,
)
from tldw_Server_API.app.core.Resource_Governance import cost_units

# Warning header name for soft limit notifications
BILLING_WARNING_HEADER = "X-Billing-Warning"
BILLING_LIMIT_HEADER = "X-Billing-Limit"
BILLING_USAGE_HEADER = "X-Billing-Usage"
_BILLING_HEADERS = (BILLING_LIMIT_HEADER, BILLING_USAGE_HEADER, BILLING_WARNING_HEADER)
_ADMIN_CLAIM_PERMISSIONS = frozenset({"*", "system.configure"})


def propagate_billing_headers(source: Response, target: Response) -> None:
    """Copy billing headers written by ``require_within_limit`` onto an explicit JSONResponse."""
    for name in _BILLING_HEADERS:
        value = source.headers.get(name)
        if value is not None:
            target.headers[name] = value


def _principal_has_admin_claims(principal: AuthPrincipal) -> bool:
    roles = {
        str(role).strip().lower()
        for role in (principal.roles or [])
        if str(role).strip()
    }
    if "admin" in roles:
        return True
    permissions = {
        str(permission).strip().lower()
        for permission in (principal.permissions or [])
        if str(permission).strip()
    }
    return bool(permissions & _ADMIN_CLAIM_PERMISSIONS)


def _allow_orgless_billing_access() -> bool:
    """
    Return True when org-less billing checks are acceptable.

    Single-user mode has no organization context by design and should remain
    permissive. Multi-user mode must require explicit org context.
    """
    try:
        from tldw_Server_API.app.core.AuthNZ.settings import get_settings
        from tldw_Server_API.app.core.testing import is_explicit_pytest_runtime, is_test_mode

        settings = get_settings()
        auth_mode = str(getattr(settings, "AUTH_MODE", "") or "").strip().lower()
        if auth_mode == "single_user":
            return True
        # Keep auth/claims/quota-focused test suites independent from org billing
        # context setup unless they explicitly monkeypatch this guard.
        if is_test_mode() or is_explicit_pytest_runtime():
            return True
        return False
    except Exception:
        # Fail closed when settings resolution fails.
        return False


async def _resolve_org_id(
    principal: AuthPrincipal,
    org_id: int | None = None,
    x_tldw_org_id: int | None = None,
) -> int | None:
    """
    Resolve the organization ID for billing purposes.

    Priority:
    1. org_id query parameter
    2. X-TLDW-Org-Id header
    3. First org in user's membership list
    4. None (user has no orgs)
    """
    try:
        pool = await get_db_pool()
        repo = AuthnzOrgsTeamsRepo(db_pool=pool)
        if org_id is not None:
            if _principal_has_admin_claims(principal):
                return org_id
            membership = await repo.get_org_member(org_id, principal.user_id)
            if not membership or not _is_membership_active(membership):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="You do not have active access to the specified organization",
                )
            return org_id

        if x_tldw_org_id is not None:
            if _principal_has_admin_claims(principal):
                return x_tldw_org_id
            membership = await repo.get_org_member(x_tldw_org_id, principal.user_id)
            if membership and _is_membership_active(membership):
                return x_tldw_org_id
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You do not have active access to the specified organization",
            )

        memberships = await repo.list_org_memberships_for_user(principal.user_id)
        if memberships:
            for membership in memberships:
                if _is_membership_active(membership):
                    return membership.get("org_id")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You do not have an active organization membership",
            )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to resolve org_id for billing enforcement")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Unable to resolve organization for billing enforcement",
        ) from exc

    return None


async def get_billing_org_id(
    principal: AuthPrincipal = Depends(get_auth_principal),
    x_tldw_org_id: int | None = Header(None, alias="X-TLDW-Org-Id"),
    org_id: int | None = Query(None, description="Organization ID"),
) -> int | None:
    """
    Resolve the billing org ID without enforcing any limit.

    Returns None when enforcement is disabled, the user has no org,
    or single-user mode is active.  Intended for handler bodies that
    need an org_id to create a ``LimitEnforcer`` context manager.
    """
    if not enforcement_enabled():
        return None
    try:
        resolved = await _resolve_org_id(principal, org_id, x_tldw_org_id)
    except HTTPException:
        if _allow_orgless_billing_access():
            return None
        raise
    return resolved


async def resolve_org_id_for_principal(principal: AuthPrincipal) -> int | None:
    """
    Resolve the billing org ID from an ``AuthPrincipal`` without FastAPI DI.

    Useful in WebSocket handlers where ``Depends()`` is unavailable.
    Returns None when enforcement is disabled or org context is absent.
    """
    if not enforcement_enabled():
        return None
    try:
        return await _resolve_org_id(principal, None, None)
    except HTTPException:
        if _allow_orgless_billing_access():
            return None
        raise
    except Exception:
        logger.debug("resolve_org_id_for_principal failed")
        return None


def require_within_limit(category: LimitCategory, units: int = 1):
    """
    Dependency factory that enforces a billing limit.

    Blocks requests that would exceed the organization's limit.
    Adds warning headers when approaching limits.

    Args:
        category: The limit category to check
        units: Number of units this operation will consume

    Usage:
        @router.post("/chat")
        async def chat(
            _: LimitCheckResult = Depends(require_within_limit(LimitCategory.LLM_TOKENS_MONTH, 1000))
        ):
            ...
    """
    async def _check_limit(
        response: Response,
        principal: AuthPrincipal = Depends(get_auth_principal),
        x_tldw_org_id: int | None = Header(None, alias="X-TLDW-Org-Id"),
        org_id: int | None = Query(None, description="Organization ID"),
    ) -> LimitCheckResult:
        # Skip enforcement if disabled
        if not enforcement_enabled():
            return LimitCheckResult(
                category=category.value,
                action=EnforcementAction.ALLOW,
                current=0,
                limit=-1,
                percent_used=0,
                unlimited=True,
            )

        # Resolve org_id
        org_id = await _resolve_org_id(principal, org_id, x_tldw_org_id)

        if org_id is None:
            if _allow_orgless_billing_access():
                # No org is expected in single-user mode.
                return LimitCheckResult(
                    category=category.value,
                    action=EnforcementAction.ALLOW,
                    current=0,
                    limit=-1,
                    percent_used=0,
                    unlimited=True,
                )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="An active organization context is required for billing enforcement",
            )

        # Check the limit
        enforcer = get_billing_enforcer()
        result = await enforcer.check_limit(org_id, category, requested_units=units)

        # Add headers
        response.headers[BILLING_LIMIT_HEADER] = str(result.limit) if not result.unlimited else "unlimited"
        response.headers[BILLING_USAGE_HEADER] = str(result.current)

        if result.should_warn and result.message:
            response.headers[BILLING_WARNING_HEADER] = result.message

        # Block if limit exceeded
        if result.should_block:
            raise HTTPException(
                status_code=status.HTTP_402_PAYMENT_REQUIRED
                if result.action == EnforcementAction.SOFT_BLOCK
                else status.HTTP_429_TOO_MANY_REQUESTS,
                detail={
                    "error": "limit_exceeded",
                    "category": category.value,
                    "current": result.current,
                    "limit": result.limit,
                    "message": result.message or f"Limit exceeded for {category.value}",
                    "upgrade_url": "/billing/plans",  # Frontend can use this
                },
                headers={
                    "Retry-After": str(result.retry_after) if result.retry_after else "3600",
                },
            )

        return result

    return _check_limit


def require_feature(feature: str):
    """
    Dependency factory that checks feature access.

    Blocks requests if the organization doesn't have access to the feature.

    Args:
        feature: Feature name (e.g., "advanced_analytics", "sso_enabled")

    Usage:
        @router.get("/analytics")
        async def get_analytics(
            _: bool = Depends(require_feature("advanced_analytics"))
        ):
            ...
    """
    async def _check_feature(
        principal: AuthPrincipal = Depends(get_auth_principal),
        x_tldw_org_id: int | None = Header(None, alias="X-TLDW-Org-Id"),
        org_id: int | None = Query(None, description="Organization ID"),
    ) -> bool:
        # Skip enforcement if disabled
        if not enforcement_enabled():
            return True

        # Resolve org_id
        org_id = await _resolve_org_id(principal, org_id, x_tldw_org_id)

        if org_id is None:
            if _allow_orgless_billing_access():
                # No org is expected in single-user mode.
                return True
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="An active organization context is required for billing feature checks",
            )

        # Check feature access
        enforcer = get_billing_enforcer()
        has_access = await enforcer.check_feature_access(org_id, feature)

        if not has_access:
            raise HTTPException(
                status_code=status.HTTP_402_PAYMENT_REQUIRED,
                detail={
                    "error": "feature_not_available",
                    "feature": feature,
                    "message": f"Your subscription plan does not include {feature.replace('_', ' ')}",
                    "upgrade_url": "/billing/plans",
                },
            )

        return True

    return _check_feature


async def get_org_limits(
    principal: AuthPrincipal = Depends(get_auth_principal),
    x_tldw_org_id: int | None = Header(None, alias="X-TLDW-Org-Id"),
    org_id: int | None = Query(None, description="Organization ID"),
) -> dict[str, Any]:
    """
    Dependency that returns the current org's subscription limits.

    Use this when you need to access limits for informational purposes
    without enforcing them.
    """
    org_id = await _resolve_org_id(principal, org_id, x_tldw_org_id)

    if org_id is None:
        if _allow_orgless_billing_access():
            # Return permissive defaults in single-user mode.
            return {
                "api_calls_day": -1,
                "llm_tokens_month": -1,
                "storage_mb": -1,
                "team_members": -1,
                "unlimited": True,
            }
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="An active organization context is required to fetch billing limits",
        )

    enforcer = get_billing_enforcer()
    return await enforcer.get_org_limits(org_id)


async def add_billing_headers(
    response: Response,
    principal: AuthPrincipal = Depends(get_auth_principal),
    x_tldw_org_id: int | None = Header(None, alias="X-TLDW-Org-Id"),
    org_id: int | None = Query(None, description="Organization ID"),
) -> None:
    """
    Dependency that adds billing info headers to the response.

    Useful for endpoints that don't enforce limits but want to inform
    clients about their usage.
    """
    if not enforcement_enabled():
        return

    org_id = await _resolve_org_id(principal, org_id, x_tldw_org_id)
    if org_id is None:
        return

    try:
        enforcer = get_billing_enforcer()
        limits = await enforcer.get_org_limits(org_id)
        usage = await enforcer.get_org_usage(org_id)

        # Add summary headers
        response.headers["X-Billing-Plan-Api-Limit"] = str(limits.get("api_calls_day", "unlimited"))
        response.headers["X-Billing-Api-Usage-Today"] = str(usage.api_calls_today)

    except Exception:
        logger.debug("Failed to add billing headers")


class LimitEnforcer:
    """
    Context manager for limit enforcement with automatic recording.

    Usage:
        async with LimitEnforcer(org_id, LimitCategory.LLM_TOKENS_MONTH, estimated=1000) as enforcer:
            # Do the operation
            actual_tokens = await call_llm(...)
            enforcer.record_actual(actual_tokens)
    """

    def __init__(
        self,
        org_id: int,
        category: LimitCategory,
        estimated_units: int = 1,
    ):
        self.org_id = org_id
        self.category = category
        self.estimated_units = estimated_units
        self.actual_units: int | None = None
        self._enforcer = get_billing_enforcer()
        self._check_result: LimitCheckResult | None = None

    async def __aenter__(self) -> LimitEnforcer:
        """Check limit on entry."""
        if enforcement_enabled():
            self._check_result = await self._enforcer.check_limit(
                self.org_id,
                self.category,
                requested_units=self.estimated_units,
            )

            if self._check_result.should_block:
                status_code = (
                    status.HTTP_402_PAYMENT_REQUIRED
                    if self._check_result.action == EnforcementAction.SOFT_BLOCK
                    else status.HTTP_429_TOO_MANY_REQUESTS
                )
                raise HTTPException(
                    status_code=status_code,
                    detail={
                        "error": "limit_exceeded",
                        "category": self.category.value,
                        "message": self._check_result.message,
                    },
                )

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Record actual usage on exit (if operation succeeded)."""
        cache_updated = False
        if exc_type is None and self.actual_units is not None and enforcement_enabled():
            try:
                units = int(self.actual_units)
            except Exception:
                units = 0

            if units > 0:
                # Best-effort in-memory cache delta for billing checks
                try:
                    cache_updated = self._enforcer.apply_usage_delta(self.org_id, self.category, units)
                except Exception:
                    logger.debug("LimitEnforcer usage delta recording failed")

                # Mirror usage into the generic cost-units ledger so that
                # cross-category budgets can reason about org-level usage.
                try:
                    tokens = 0
                    minutes = 0.0
                    requests = 0

                    if self.category == LimitCategory.LLM_TOKENS_MONTH:
                        tokens = units
                    elif self.category in (LimitCategory.API_CALLS_DAY, LimitCategory.RAG_QUERIES_DAY):
                        requests = units
                    elif self.category == LimitCategory.TRANSCRIPTION_MINUTES_MONTH:
                        minutes = float(units)

                    if tokens or minutes or requests:
                        await cost_units.record_cost_units_for_entity(
                            entity_scope="org",
                            entity_value=str(self.org_id),
                            tokens=tokens,
                            minutes=minutes,
                            requests=requests,
                        )
                except Exception:
                    logger.debug("LimitEnforcer cost-units ledger write failed")

        # Invalidate cache so next request gets fresh data
        if self.actual_units is not None and not cache_updated:
            self._enforcer.invalidate_cache(self.org_id)

    def record_actual(self, units: int) -> None:
        """Record the actual units consumed by the operation."""
        self.actual_units = units

    @property
    def check_result(self) -> LimitCheckResult | None:
        """Get the limit check result."""
        return self._check_result
