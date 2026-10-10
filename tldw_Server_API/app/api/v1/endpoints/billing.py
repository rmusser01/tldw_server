"""
billing.py

Admin billing endpoints for subscription management and analytics.

Mounted under /api/v1/admin/billing (admin-guarded). The legacy public
/api/v1/billing mount stays removed per the OSS billing API removal; no
deprecation alias resurrects it.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, Query
from loguru import logger

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import (
    RequireRole,
    get_auth_principal,
)
from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.AuthNZ.repos.billing_repo import AuthnzBillingRepo

router = APIRouter(
    prefix="/billing",
    tags=["billing"],
    dependencies=[Depends(RequireRole("admin"))],
)


def _parse_iso_datetime(value: Any) -> datetime | None:
    """Parse an ISO datetime string or return None."""
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value
    try:
        dt = datetime.fromisoformat(str(value))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        return None


def _compute_at_risk_flags(sub: dict[str, Any], now: datetime) -> dict[str, Any]:
    """Compute lifecycle and at-risk indicator fields for a subscription.

    Returns a dict with computed fields to merge into the subscription response.
    """
    status = str(sub.get("status") or "").lower()
    created_at = _parse_iso_datetime(sub.get("created_at"))
    current_period_end = _parse_iso_datetime(sub.get("current_period_end"))
    cancel_at_period_end = bool(sub.get("cancel_at_period_end"))

    # Days since created
    days_since_created = (now - created_at).days if created_at else None

    # Days until period end
    days_until_period_end: int | None = None
    if current_period_end:
        delta = (current_period_end - now).days
        days_until_period_end = max(delta, 0)

    # Days past due: how many days the subscription has been in past_due status.
    # Since we don't store the exact date it went past_due, we use the period
    # end date as an approximation (payment was due at period end).
    days_past_due = 0
    if status == "past_due" and current_period_end:
        days_past_due = max((now - current_period_end).days, 0)

    # Usage percentage against plan token limit
    effective_limits = sub.get("effective_limits") or sub.get("plan_limits") or {}
    token_limit = effective_limits.get("llm_tokens_month")
    usage_pct: float | None = None
    # Note: actual token usage is not stored in the subscription table.
    # The usage_pct will be None unless we have usage data.

    # At-risk determination:
    #   1. past_due for more than 7 days
    #   2. cancel_at_period_end is True (subscription is cancelling)
    #   3. status is "canceled"
    at_risk = False
    at_risk_reasons: list[str] = []

    if status == "past_due" and days_past_due > 7:
        at_risk = True
        at_risk_reasons.append("past_due_extended")

    if cancel_at_period_end and status not in ("canceled",):
        at_risk = True
        at_risk_reasons.append("cancelling")

    if status == "canceled":
        at_risk = True
        at_risk_reasons.append("canceled")

    return {
        "days_since_created": days_since_created,
        "days_past_due": days_past_due,
        "days_until_period_end": days_until_period_end,
        "usage_pct": usage_pct,
        "at_risk": at_risk,
        "at_risk_reasons": at_risk_reasons,
        "cancel_at_period_end": cancel_at_period_end,
    }


def _subscription_to_response_item(sub: dict[str, Any], now: datetime) -> dict[str, Any]:
    """Build a subscriptions-list response item from a repo subscription row.

    The row is expected to carry org_name (resolved by the repo's SQL JOIN).
    """
    computed = _compute_at_risk_flags(sub, now)
    org_id = sub.get("org_id")
    item: dict[str, Any] = {
        "id": sub.get("id"),
        "org_id": org_id,
        "org_name": sub.get("org_name"),
        "plan_id": sub.get("plan_id"),
        "plan": {
            "id": sub.get("plan_id"),
            "name": sub.get("plan_display_name") or sub.get("plan_name"),
            "tier": sub.get("plan_name", "free"),
            "stripe_product_id": None,
            "stripe_price_id": None,
            "monthly_price_cents": int((sub.get("price_usd_monthly") or 0) * 100),
            "included_token_credits": (sub.get("effective_limits") or {}).get(
                "llm_tokens_month", 0
            ),
            "overage_rate_per_1k_tokens_cents": 0,
            "features": [],
            "is_default": sub.get("plan_name") == "free",
            "created_at": sub.get("created_at"),
            "updated_at": sub.get("created_at"),
        },
        "stripe_subscription_id": sub.get("stripe_subscription_id"),
        "status": sub.get("status"),
        "current_period_start": sub.get("current_period_start"),
        "current_period_end": sub.get("current_period_end"),
        "trial_end": sub.get("trial_end"),
        "cancel_at": sub.get("current_period_end") if computed["cancel_at_period_end"] else None,
        "created_at": sub.get("created_at"),
        "updated_at": sub.get("updated_at"),
        # Computed lifecycle fields
        "days_since_created": computed["days_since_created"],
        "days_past_due": computed["days_past_due"],
        "days_until_period_end": computed["days_until_period_end"],
        "usage_pct": computed["usage_pct"],
        "at_risk": computed["at_risk"],
        "at_risk_reasons": computed["at_risk_reasons"],
        "cancel_at_period_end": computed["cancel_at_period_end"],
        "billing_cycle": sub.get("billing_cycle"),
    }
    return item


@router.get("/overview")
async def get_billing_overview(
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> dict[str, Any]:
    """Subscription counts by status plus monthly recurring revenue.

    Response shape (consumed by the admin BillingDashboardPage):
    - mrr: sum of active subscriptions' plan monthly price (0 when no cost column applies)
    - active_subscriptions / canceled_subscriptions / past_due_subscriptions: counts
    """
    try:
        pool = await get_db_pool()
        billing_repo = AuthnzBillingRepo(db_pool=pool)
        return await billing_repo.get_overview()
    except Exception:
        logger.error("get_billing_overview failed")
        raise


@router.get("/subscriptions")
async def list_subscriptions(
    status: str | None = Query(None, description="Filter by subscription status"),
    limit: int = Query(100, ge=1, le=500, description="Page size"),
    offset: int = Query(0, ge=0, description="Page offset"),
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> dict[str, Any]:
    """Page subscriptions with computed lifecycle and at-risk indicators.

    Status filtering, ordering and paging all happen in SQL; org names are
    resolved by the repo's LEFT JOIN. Returns ``{"items", "total"}`` where
    ``total`` is the truthful COUNT(*) of matching rows.

    Each item is enriched with:
    - org_name: resolved organization name
    - days_since_created: days since subscription was created
    - days_past_due: days the subscription has been past due
    - days_until_period_end: days until the current billing period ends
    - at_risk: boolean flag indicating the subscription needs attention
    - at_risk_reasons: list of reasons (past_due_extended, cancelling, canceled)
    - cancel_at_period_end: whether the subscription is set to cancel
    """
    try:
        pool = await get_db_pool()
        billing_repo = AuthnzBillingRepo(db_pool=pool)

        subscriptions, total = await billing_repo.list_subscriptions(
            status=status, limit=limit, offset=offset
        )

        now = datetime.now(timezone.utc)
        result = [
            _subscription_to_response_item(sub, now) for sub in subscriptions
        ]

        return {"items": result, "total": total}
    except Exception:
        logger.error("list_subscriptions failed")
        raise


@router.get("/events")
async def list_billing_events(
    limit: int = Query(100, ge=1, le=500, description="Page size"),
    offset: int = Query(0, ge=0, description="Page offset"),
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> dict[str, Any]:
    """Page the billing events ledger (billing audit log).

    Returns ``{"items", "total"}`` where each item carries ``event_type``
    (the audit action), ``user_id``, ``description`` (audit details) and
    ``created_at``; ``amount`` is not part of this source and stays absent.
    """
    try:
        pool = await get_db_pool()
        billing_repo = AuthnzBillingRepo(db_pool=pool)
        events, total = await billing_repo.list_billing_events(
            limit=limit, offset=offset
        )
        items = [
            {
                "id": event.get("id"),
                "org_id": event.get("org_id"),
                "org_name": event.get("org_name"),
                "user_id": event.get("user_id"),
                "event_type": event.get("action"),
                "description": event.get("details"),
                "ip_address": event.get("ip_address"),
                "created_at": event.get("created_at"),
            }
            for event in events
        ]
        return {"items": items, "total": total}
    except Exception:
        logger.error("list_billing_events failed")
        raise
