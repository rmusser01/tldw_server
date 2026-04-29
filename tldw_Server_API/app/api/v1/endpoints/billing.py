"""
billing.py

Admin billing endpoints for subscription management and analytics.
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
from tldw_Server_API.app.core.AuthNZ.repos.orgs_teams_repo import AuthnzOrgsTeamsRepo

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


@router.get("/subscriptions")
async def list_subscriptions(
    status: str | None = Query(None, description="Filter by subscription status"),
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> list[dict[str, Any]]:
    """List all subscriptions with computed lifecycle and at-risk indicators.

    Returns subscription data enriched with:
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
        orgs_repo = AuthnzOrgsTeamsRepo(db_pool=pool)

        # Fetch all subscriptions
        subscriptions = await billing_repo.list_all_subscriptions()

        # Build org name lookup
        org_ids = list({sub["org_id"] for sub in subscriptions if sub.get("org_id")})
        org_names: dict[int, str] = {}
        if org_ids:
            try:
                orgs_list, _total = await orgs_repo.list_organizations(limit=len(org_ids) + 50)
                for org in orgs_list:
                    org_id = org.get("id")
                    name = org.get("name")
                    if org_id is not None and name:
                        org_names[int(org_id)] = str(name)
            except Exception:
                logger.warning("Failed to resolve org names for subscriptions")

        now = datetime.now(timezone.utc)
        result = []

        for sub in subscriptions:
            # Apply status filter
            sub_status = str(sub.get("status") or "").lower()
            if status and sub_status != status.lower():
                continue

            # Compute at-risk fields
            computed = _compute_at_risk_flags(sub, now)

            # Build response item
            org_id = sub.get("org_id")
            item: dict[str, Any] = {
                "id": sub.get("id"),
                "org_id": org_id,
                "org_name": org_names.get(int(org_id)) if org_id else None,
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
                "updated_at": sub.get("created_at"),
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
            result.append(item)

        return result
    except Exception:
        logger.error("list_subscriptions failed")
        raise
