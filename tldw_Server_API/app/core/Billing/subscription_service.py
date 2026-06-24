"""
subscription_service.py

Service for subscription and billing management.
Coordinates between the billing repository and Stripe client.
"""
from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

from loguru import logger

from tldw_Server_API.app.core.AuthNZ.database import DatabasePool
from tldw_Server_API.app.core.Billing.plan_limits import check_limit, get_plan_limits
from tldw_Server_API.app.core.Billing.runtime_flags import is_billing_enabled

_LOCAL_REDIRECT_HOSTS = {"localhost", "127.0.0.1", "::1"}
_BILLING_CYCLES = {"monthly", "yearly"}


@dataclass
class CheckoutSession:
    """Historical checkout session shape retained for internal compatibility."""
    id: str
    url: str


@dataclass
class PortalSession:
    """Historical portal session shape retained for internal compatibility."""
    id: str
    url: str


@dataclass
class UsageStatus:
    """Current usage status for an organization."""
    org_id: int
    plan_name: str
    limits: dict[str, Any]
    usage: dict[str, int]
    limit_checks: dict[str, dict[str, Any]]
    has_warnings: bool
    has_exceeded: bool


@dataclass
class SubscriptionStatus:
    """Subscription status details."""
    org_id: int
    plan_name: str
    plan_display_name: str
    status: str
    billing_cycle: str | None
    current_period_end: str | None
    trial_end: str | None
    cancel_at_period_end: bool
    limits: dict[str, Any]


def _safe_exception_label(exc: BaseException) -> str:
    """Return a sanitized exception label suitable for billing audit state."""
    return exc.__class__.__name__


def _get_public_web_base_url() -> str | None:
    """Resolve the public web base URL without requiring settings at import time."""
    env_value = os.getenv("PUBLIC_WEB_BASE_URL")
    if env_value:
        return env_value.strip()
    try:
        from tldw_Server_API.app.core.AuthNZ.settings import get_settings

        value = getattr(get_settings(), "PUBLIC_WEB_BASE_URL", None)
    except Exception as exc:
        logger.warning(
            "Unable to resolve PUBLIC_WEB_BASE_URL from settings; error_type={}",
            _safe_exception_label(exc),
        )
        return None
    return str(value).strip() if value else None


def _url_origin_tuple(url: str) -> tuple[str, str, int | None]:
    parsed = urlparse(url)
    scheme = parsed.scheme.lower()
    hostname = (parsed.hostname or "").strip().lower()
    port = parsed.port
    if (scheme == "https" and port == 443) or (scheme == "http" and port == 80):
        port = None
    return scheme, hostname, port


def _validate_checkout_redirect_url(url: str, label: str) -> str:
    """Validate a checkout or portal redirect URL against the configured public origin."""
    value = str(url or "").strip()
    parsed = urlparse(value)
    hostname = (parsed.hostname or "").strip().lower()
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc or not hostname:
        raise ValueError(f"Redirect URL for {label} must be an absolute HTTP(S) URL")
    if parsed.username or parsed.password:
        raise ValueError(f"Redirect URL for {label} must not include credentials")
    if parsed.scheme.lower() != "https" and hostname not in _LOCAL_REDIRECT_HOSTS:
        raise ValueError(f"Redirect URL for {label} must use HTTPS")

    public_base_url = _get_public_web_base_url()
    if public_base_url:
        allowed_origin = _url_origin_tuple(public_base_url)
        if _url_origin_tuple(value) != allowed_origin:
            raise ValueError(f"Redirect URL for {label} is not allowed")
    return value


def _normalize_billing_cycle(billing_cycle: str) -> str:
    normalized = str(billing_cycle or "").strip().lower()
    if normalized not in _BILLING_CYCLES:
        raise ValueError("billing_cycle must be 'monthly' or 'yearly'")
    return normalized


class SubscriptionService:
    """
    Service for subscription and billing management.

    Provides high-level operations for:
    - Listing available plans
    - Getting/creating subscriptions
    - Creating checkout and portal sessions
    - Checking usage against limits
    """

    def __init__(
        self,
        db_pool: DatabasePool | None = None,
        billing_repo: Any | None = None,
        stripe_client: Any | None = None,
    ):
        self._db_pool = db_pool
        self._billing_repo = billing_repo
        self._stripe_client = stripe_client

    @staticmethod
    def _free_plan_record() -> dict[str, Any]:
        return {
            "name": "free",
            "display_name": "Free",
            "description": "Internal/self-host default plan",
            "price_usd_monthly": 0,
            "price_usd_yearly": 0,
            "limits": get_plan_limits("free"),
            "is_active": True,
            "is_public": False,
        }

    async def _require_billing_repo(self) -> Any:
        if self._billing_repo is None:
            raise RuntimeError("Legacy billing repository runtime is not available in OSS")
        return self._billing_repo

    def _get_stripe_client(self) -> Any:
        if self._stripe_client is None:
            raise RuntimeError("Stripe payment runtime is not available in OSS")
        return self._stripe_client

    # =========================================================================
    # Plans
    # =========================================================================

    async def list_available_plans(self) -> list[dict[str, Any]]:
        """
        List all publicly available subscription plans.

        Returns plans from the database. OSS no longer synthesizes a public
        paid fallback catalog when the database is empty, but it does
        synthesize the neutral free/self-host tier so the public endpoint
        never returns an empty catalog on a fresh install.
        """
        if self._billing_repo is not None and hasattr(self._billing_repo, "list_plans"):
            plans = await self._billing_repo.list_plans(active_only=True, public_only=True)
            if plans:
                return plans
        return [self._free_plan_record()]

    async def get_plan(self, plan_name: str) -> dict[str, Any] | None:
        """Get a specific plan by name.

        Only the neutral free/self-host fallback is synthesized when the
        database does not contain a matching plan row.
        """
        normalized_name = str(plan_name).strip().lower()
        if self._billing_repo is not None and hasattr(self._billing_repo, "get_plan_by_name"):
            plan = await self._billing_repo.get_plan_by_name(normalized_name)
            if plan and (normalized_name == "free" or plan.get("is_public") is not False):
                return plan
        if normalized_name != "free":
            return None
        return self._free_plan_record()

    async def get_plan_for_checkout(self, plan_name: str) -> dict[str, Any] | None:
        """
        Resolve a plan for checkout by name.

        Plans must exist in the subscription_plans table, be active, and be
        publicly purchasable.
        """
        repo = await self._require_billing_repo()
        plan = await repo.get_plan_by_name(plan_name)
        if not plan:
            return None
        if plan.get("is_active") is False:
            return None
        if plan.get("is_public") is False:
            return None
        return plan

    # =========================================================================
    # Subscriptions
    # =========================================================================

    async def get_subscription(self, org_id: int) -> SubscriptionStatus:
        """
        Get the subscription status for an organization.

        Organizations without an explicit subscription are treated as being
        on the implicit free tier.
        """
        repo = self._billing_repo
        sub = None
        if repo is not None and hasattr(repo, "get_org_subscription"):
            sub = await repo.get_org_subscription(org_id)

        limits = await self.get_org_limits(org_id)
        if sub:
            plan_name = str(sub.get("plan_name") or "free")
            return SubscriptionStatus(
                org_id=org_id,
                plan_name=plan_name,
                plan_display_name=str(sub.get("plan_display_name") or plan_name.title()),
                status=str(sub.get("status") or "active"),
                billing_cycle=sub.get("billing_cycle"),
                current_period_end=sub.get("current_period_end"),
                trial_end=sub.get("trial_end"),
                cancel_at_period_end=bool(sub.get("cancel_at_period_end")),
                limits=limits,
            )

        return SubscriptionStatus(
            org_id=org_id,
            plan_name="free",
            plan_display_name="Free",
            status="active",
            billing_cycle=None,
            current_period_end=None,
            trial_end=None,
            cancel_at_period_end=False,
            limits=limits,
        )

    async def create_subscription(
        self,
        *,
        org_id: int,
        plan_name: str,
        billing_cycle: str = "monthly",
        trial_days: int | None = None,
    ) -> dict[str, Any]:
        """
        Create or update a subscription for an organization.

        This creates the database record. For paid plans, a checkout session
        should be created separately.
        """
        repo = await self._require_billing_repo()

        # Get plan ID
        plan = await repo.get_plan_by_name(plan_name)
        if not plan:
            # Unknown plan names are treated as errors rather than silently
            # downgrading to the free tier. Callers should validate plan_name
            # against the available plans before invoking this method.
            raise ValueError(f"Plan '{plan_name}' not found")

        sub = await repo.create_org_subscription(
            org_id=org_id,
            plan_id=plan["id"],
            billing_cycle=billing_cycle,
            status="active" if plan_name == "free" else "pending",
            trial_days=trial_days,
        )

        # Log the action
        await repo.log_billing_action(
            org_id=org_id,
            action="subscription.created",
            details={
                "plan_name": plan_name,
                "billing_cycle": billing_cycle,
                "trial_days": trial_days,
            },
        )

        logger.info(f"Created subscription for org {org_id}: plan={plan_name}")
        return sub

    # =========================================================================
    # Stripe Integration
    # =========================================================================

    async def create_checkout_session(
        self,
        *,
        org_id: int,
        plan_name: str,
        billing_cycle: str = "monthly",
        success_url: str,
        cancel_url: str,
        org_email: str,
        org_name: str | None = None,
    ) -> CheckoutSession:
        """
        Create a Stripe checkout session for a plan upgrade.

        Args:
            org_id: Organization ID
            plan_name: Target plan name
            billing_cycle: monthly or yearly
            success_url: Redirect URL on success
            cancel_url: Redirect URL on cancel
            org_email: Organization billing email
            org_name: Organization name for customer record

        Returns:
            CheckoutSession with id and url
        """
        if not is_billing_enabled():
            raise RuntimeError("Billing is not enabled")

        success_url = _validate_checkout_redirect_url(success_url, "success_url")
        cancel_url = _validate_checkout_redirect_url(cancel_url, "cancel_url")
        normalized_cycle = _normalize_billing_cycle(billing_cycle)

        stripe = self._get_stripe_client()
        if not stripe.is_available:
            raise RuntimeError("Stripe is not configured")

        repo = await self._require_billing_repo()
        plan = await repo.get_plan_by_name(plan_name)
        if not plan:
            raise ValueError(f"Plan '{plan_name}' not found")
        if plan.get("is_active") is False:
            raise ValueError(f"Plan '{plan_name}' is not active")
        if plan.get("is_public") is False:
            raise ValueError(f"Plan '{plan_name}' is not public")

        # Get price ID
        price_id = stripe.get_price_id(plan_name, normalized_cycle)
        if not price_id:
            if normalized_cycle == "yearly":
                price_id = plan.get("stripe_price_id_yearly")
            else:
                price_id = plan.get("stripe_price_id")

        if not price_id:
            raise ValueError(f"No Stripe price configured for plan '{plan_name}'")

        # Get or create Stripe customer only after all local preconditions pass.
        sub = await repo.get_org_subscription(org_id)
        customer_id = sub.get("stripe_customer_id") if sub else None

        if not customer_id:
            customer_id = await stripe.create_customer(
                email=org_email,
                name=org_name,
                metadata={"org_id": str(org_id)},
            )
            if sub:
                await repo.update_org_subscription(org_id, stripe_customer_id=customer_id)
            else:
                await repo.create_org_subscription(
                    org_id=org_id,
                    plan_id=plan["id"],
                    stripe_customer_id=customer_id,
                    billing_cycle=normalized_cycle,
                    status="pending",
                )

        # Create checkout session
        session = await stripe.create_checkout_session(
            customer_id=customer_id,
            price_id=price_id,
            success_url=success_url,
            cancel_url=cancel_url,
            metadata={
                "org_id": str(org_id),
                "plan_name": plan_name,
                "billing_cycle": normalized_cycle,
            },
        )

        # Log the action
        await repo.log_billing_action(
            org_id=org_id,
            action="checkout.initiated",
            details={
                "plan_name": plan_name,
                "billing_cycle": normalized_cycle,
                "checkout_session_id": session.id,
            },
        )

        return session

    async def create_portal_session(
        self,
        *,
        org_id: int,
        return_url: str,
    ) -> PortalSession:
        """
        Create a Stripe billing portal session.

        Allows customers to manage their subscription, payment methods, etc.
        """
        if not is_billing_enabled():
            raise RuntimeError("Billing is not enabled")

        return_url = _validate_checkout_redirect_url(return_url, "return_url")

        stripe = self._get_stripe_client()
        if not stripe.is_available:
            raise RuntimeError("Stripe is not configured")

        repo = await self._require_billing_repo()
        sub = await repo.get_org_subscription(org_id)

        if not sub or not sub.get("stripe_customer_id"):
            raise ValueError("Organization does not have a billing account")

        session = await stripe.create_portal_session(
            customer_id=sub["stripe_customer_id"],
            return_url=return_url,
        )

        return session

    async def cancel_subscription(
        self,
        org_id: int,
        *,
        at_period_end: bool = True,
        user_id: int | None = None,
        ip_address: str | None = None,
    ) -> dict[str, Any]:
        """Cancel an organization's subscription."""
        repo = await self._require_billing_repo()
        sub = await repo.get_org_subscription(org_id)

        if not sub:
            raise ValueError("Organization does not have an active subscription")

        result = {"canceled": True}

        # Cancel in Stripe if applicable
        if sub.get("stripe_subscription_id") and is_billing_enabled():
            stripe = self._get_stripe_client()
            if not stripe.is_available:
                raise RuntimeError(
                    "Stripe is not configured; refusing to change local subscription state."
                )
            try:
                result = await stripe.cancel_subscription(
                    sub["stripe_subscription_id"],
                    at_period_end=at_period_end,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to cancel subscription in Stripe: {exc}"
                ) from exc

        # Update local status to mirror Stripe semantics:
        # - at_period_end=True  -> keep status "active", set cancel_at_period_end=True
        # - at_period_end=False -> set status "canceled", cancel_at_period_end=False
        if at_period_end:
            await repo.update_org_subscription(
                org_id,
                cancel_at_period_end=True,
            )
        else:
            await repo.update_org_subscription(
                org_id,
                status="canceled",
                cancel_at_period_end=False,
            )

        # Log the action
        await repo.log_billing_action(
            org_id=org_id,
            user_id=user_id,
            action="subscription.canceled",
            details={
                "at_period_end": at_period_end,
                "previous_status": sub.get("status"),
            },
            ip_address=ip_address,
        )

        logger.info(f"Canceled subscription for org {org_id} (at_period_end={at_period_end})")
        return result

    async def resume_subscription(
        self,
        org_id: int,
        *,
        user_id: int | None = None,
    ) -> dict[str, Any]:
        """Resume a subscription that was set to cancel."""
        repo = await self._require_billing_repo()
        sub = await repo.get_org_subscription(org_id)

        if not sub:
            raise ValueError("Organization does not have a subscription")

        result = {"resumed": True}

        # Resume in Stripe if applicable
        if sub.get("stripe_subscription_id") and is_billing_enabled():
            stripe = self._get_stripe_client()
            if not stripe.is_available:
                raise RuntimeError(
                    "Stripe is not configured; refusing to change local subscription state."
                )
            try:
                result = await stripe.resume_subscription(sub["stripe_subscription_id"])
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to resume subscription in Stripe: {exc}"
                ) from exc

        # Update local status
        await repo.update_org_subscription(org_id, status="active", cancel_at_period_end=False)

        # Log the action
        await repo.log_billing_action(
            org_id=org_id,
            user_id=user_id,
            action="subscription.resumed",
        )

        logger.info(f"Resumed subscription for org {org_id}")
        return result

    # =========================================================================
    # Usage & Limits
    # =========================================================================

    async def get_org_limits(self, org_id: int) -> dict[str, Any]:
        """Get the effective limits for an organization."""
        if self._billing_repo is not None and hasattr(self._billing_repo, "get_org_limits"):
            return await self._billing_repo.get_org_limits(org_id)
        return get_plan_limits("free")

    async def check_usage(
        self,
        org_id: int,
        *,
        current_usage: dict[str, int],
    ) -> UsageStatus:
        """
        Check current usage against limits.

        Args:
            org_id: Organization ID
            current_usage: Dict of current usage values by limit name

        Returns:
            UsageStatus with warnings and exceeded flags
        """
        limits = await self.get_org_limits(org_id)
        sub = await self.get_subscription(org_id)

        limit_checks = {}
        has_warnings = False
        has_exceeded = False

        # Check each provided usage value
        for limit_name, current_value in current_usage.items():
            limit_value = limits.get(limit_name)
            if limit_value is not None:
                check = check_limit(current_value, limit_value, limit_name)
                limit_checks[limit_name] = check
                if check["warning"]:
                    has_warnings = True
                if check["exceeded"]:
                    has_exceeded = True

        return UsageStatus(
            org_id=org_id,
            plan_name=sub.plan_name,
            limits=limits,
            usage=current_usage,
            limit_checks=limit_checks,
            has_warnings=has_warnings,
            has_exceeded=has_exceeded,
        )

    # =========================================================================
    # Webhook Handling
    # =========================================================================

    async def handle_webhook_event(
        self,
        event_type: str,
        event_data: dict[str, Any],
        *,
        stripe_event_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Handle a Stripe webhook event.

        Args:
            event_type: Stripe event type
            event_data: Event payload

        Returns:
            Processing result
        """
        repo = await self._require_billing_repo()

        handlers = {
            "checkout.session.completed": self._handle_checkout_completed,
            "customer.subscription.created": self._handle_subscription_updated,
            "customer.subscription.updated": self._handle_subscription_updated,
            "customer.subscription.deleted": self._handle_subscription_deleted,
            "invoice.paid": self._handle_invoice_paid,
            "invoice.payment_failed": self._handle_payment_failed,
        }

        handler = handlers.get(event_type)
        if handler:
            if stripe_event_id:
                return await self._handle_idempotent_webhook_event(
                    stripe_event_id=stripe_event_id,
                    event_type=event_type,
                    event_data=event_data,
                    repo=repo,
                    handler=handler,
                )
            return await handler(event_data, repo)

        logger.debug(f"Unhandled webhook event type: {event_type}")
        return {"handled": False, "event_type": event_type, "retryable": False}

    async def _handle_idempotent_webhook_event(
        self,
        *,
        stripe_event_id: str,
        event_type: str,
        event_data: dict[str, Any],
        repo: Any,
        handler: Any,
    ) -> dict[str, Any]:
        """Claim a Stripe event ID before running mutating webhook handlers."""
        record_event = getattr(repo, "record_webhook_event", None)
        try_claim = getattr(repo, "try_claim_webhook_event", None)
        mark_processed = getattr(repo, "mark_webhook_processed", None)

        if record_event is not None:
            is_new_event = await record_event(stripe_event_id, event_type, event_data)
            if is_new_event is False and try_claim is None:
                return {
                    "handled": True,
                    "event_type": event_type,
                    "duplicate": True,
                    "retryable": False,
                }

        if try_claim is not None:
            claimed = await try_claim(stripe_event_id)
            if not claimed:
                return {
                    "handled": True,
                    "event_type": event_type,
                    "duplicate": True,
                    "retryable": False,
                }

        try:
            result = await handler(event_data, repo)
        except Exception as exc:
            if mark_processed is not None:
                await mark_processed(stripe_event_id, error_message=_safe_exception_label(exc))
            raise

        if mark_processed is not None:
            await mark_processed(stripe_event_id)
        return result

    async def _handle_checkout_completed(
        self,
        event_data: dict[str, Any],
        repo: Any,
    ) -> dict[str, Any]:
        """Handle checkout.session.completed event."""
        session = event_data.get("object", {})
        metadata = session.get("metadata", {})
        org_id_str = metadata.get("org_id")

        if not org_id_str:
            logger.warning("Checkout completed without org_id in metadata")
            return {"handled": False, "reason": "missing_org_id", "retryable": False}

        try:
            org_id = int(org_id_str)
        except (ValueError, TypeError) as e:
            logger.warning(f"Checkout completed with invalid org_id '{org_id_str}': {e}")
            return {"handled": False, "reason": "invalid_org_id", "retryable": False}
        subscription_id = session.get("subscription")
        customer_id = session.get("customer")

        plan_updates: dict[str, Any] = {}
        plan_name = metadata.get("plan_name")
        if plan_name:
            try:
                plan = await repo.get_plan_by_name(str(plan_name))
                if plan:
                    plan_updates["plan_id"] = plan["id"]
                else:
                    logger.warning(f"Checkout completed with unknown plan_name: {plan_name}")
            except Exception as exc:
                logger.warning(
                    f"Checkout completed: failed to resolve plan_name {plan_name}: {exc}"
                )

        billing_cycle = metadata.get("billing_cycle")
        if billing_cycle:
            cycle_norm = str(billing_cycle).strip().lower()
            if cycle_norm in {"monthly", "yearly"}:
                plan_updates["billing_cycle"] = cycle_norm

        # Update or create subscription record. Some legacy datasets may not have
        # an org_subscriptions row yet, so update-only semantics can silently no-op.
        existing_sub = await repo.get_org_subscription(org_id)
        if existing_sub:
            await repo.update_org_subscription(
                org_id,
                stripe_subscription_id=subscription_id,
                stripe_customer_id=customer_id,
                status="active",
                **plan_updates,
            )
        else:
            plan_id = plan_updates.get("plan_id")
            if plan_id is None:
                free_plan = await repo.get_plan_by_name("free")
                if not free_plan:
                    logger.error(
                        "Checkout completed for org {} but no subscription row exists and free plan is missing",
                        org_id,
                    )
                    return {"handled": False, "reason": "missing_free_plan", "retryable": True}
                plan_id = free_plan["id"]

            billing_cycle_value = str(plan_updates.get("billing_cycle") or "monthly").strip().lower()
            if billing_cycle_value not in {"monthly", "yearly"}:
                billing_cycle_value = "monthly"

            await repo.create_org_subscription(
                org_id=org_id,
                plan_id=int(plan_id),
                stripe_customer_id=customer_id,
                stripe_subscription_id=subscription_id,
                billing_cycle=billing_cycle_value,
                status="active",
            )

        persisted = await repo.get_org_subscription(org_id)
        if not persisted:
            logger.error("Checkout completed for org {} but subscription state was not persisted", org_id)
            return {"handled": False, "reason": "subscription_not_persisted", "retryable": True}

        await repo.log_billing_action(
            org_id=org_id,
            action="checkout.completed",
            details={
                "subscription_id": subscription_id,
                "session_id": session.get("id"),
            },
        )

        logger.info(f"Checkout completed for org {org_id}")
        return {"handled": True, "org_id": org_id}

    async def _handle_subscription_updated(
        self,
        event_data: dict[str, Any],
        repo: Any,
    ) -> dict[str, Any]:
        """Handle customer.subscription.updated event."""
        subscription = event_data.get("object", {})
        customer_id = subscription.get("customer")

        sub = await repo.get_subscription_by_stripe_customer(customer_id)
        if not sub:
            logger.warning(f"Subscription update for unknown customer {customer_id}")
            return {"handled": False, "reason": "unknown_customer", "retryable": True}

        org_id = sub["org_id"]

        # Try to determine the active plan from the subscription items so that
        # local plan_id / limits stay in sync with Stripe when upgrades or
        # downgrades are initiated from the Billing Portal.
        plan_updates: dict[str, Any] = {}
        items = (subscription.get("items") or {}).get("data") or []
        if items:
            item0 = items[0] or {}
            price_obj = item0.get("price") or {}
            plan_obj = item0.get("plan") or {}

            price_id = price_obj.get("id") or item0.get("price_id")
            product_id = price_obj.get("product") or plan_obj.get("product")
            recurring = price_obj.get("recurring") or {}
            interval = recurring.get("interval") or plan_obj.get("interval")

            new_plan: dict[str, Any] | None = None
            if price_id:
                new_plan = await repo.get_plan_by_stripe_price_id(price_id)
            if not new_plan and product_id:
                new_plan = await repo.get_plan_by_stripe_product_id(product_id)

            if new_plan:
                plan_updates["plan_id"] = new_plan["id"]

            if interval == "year":
                plan_updates["billing_cycle"] = "yearly"
            elif interval == "month":
                plan_updates["billing_cycle"] = "monthly"

        # Always update status and period timestamps; merge any plan updates.
        stripe_status = subscription.get("status")
        update_fields: dict[str, Any] = {
            "stripe_subscription_status": stripe_status,
            "current_period_start": datetime.fromtimestamp(
                subscription.get("current_period_start", 0),
                tz=timezone.utc,
            ).isoformat() if subscription.get("current_period_start") else None,
            "current_period_end": datetime.fromtimestamp(
                subscription.get("current_period_end", 0),
                tz=timezone.utc,
            ).isoformat() if subscription.get("current_period_end") else None,
            "trial_end": datetime.fromtimestamp(
                subscription.get("trial_end", 0),
                tz=timezone.utc,
            ).isoformat() if subscription.get("trial_end") else None,
            **plan_updates,
        }
        if stripe_status is not None:
            update_fields["status"] = stripe_status
        if subscription.get("cancel_at_period_end") is not None:
            update_fields["cancel_at_period_end"] = bool(subscription.get("cancel_at_period_end"))

        await repo.update_org_subscription(org_id, **update_fields)

        logger.info(
            'Subscription updated for org {}: status={}, plan_updated={}',
            org_id,
            subscription.get("status"),
            bool(plan_updates),
        )
        return {"handled": True, "org_id": org_id}

    async def _handle_subscription_deleted(
        self,
        event_data: dict[str, Any],
        repo: Any,
    ) -> dict[str, Any]:
        """Handle customer.subscription.deleted event."""
        subscription = event_data.get("object", {})
        customer_id = subscription.get("customer")

        sub = await repo.get_subscription_by_stripe_customer(customer_id)
        if not sub:
            return {"handled": False, "reason": "unknown_customer", "retryable": True}

        org_id = sub["org_id"]

        # Downgrade to free plan.
        free_plan = await repo.get_plan_by_name("free")
        downgraded_to = "free"
        if free_plan:
            await repo.update_org_subscription(
                org_id,
                plan_id=free_plan["id"],
                status="active",
                stripe_subscription_id=None,
                stripe_subscription_status=None,
                cancel_at_period_end=False,
            )
        else:
            # Fail-safe when the free plan row is missing: remove Stripe linkage
            # and mark canceled so paid entitlements are not retained.
            logger.error(
                "Missing free plan during subscription deletion for org {}; marking canceled",
                org_id,
            )
            await repo.update_org_subscription(
                org_id,
                status="canceled",
                stripe_subscription_id=None,
                stripe_subscription_status=None,
                cancel_at_period_end=False,
            )
            downgraded_to = "canceled_no_free_plan"

        await repo.log_billing_action(
            org_id=org_id,
            action="subscription.deleted",
            details={"downgraded_to": downgraded_to},
        )

        logger.info(f"Subscription deleted for org {org_id}, downgraded_to={downgraded_to}")
        return {"handled": True, "org_id": org_id, "downgraded_to": downgraded_to}

    async def _handle_invoice_paid(
        self,
        event_data: dict[str, Any],
        repo: Any,
    ) -> dict[str, Any]:
        """Handle invoice.paid event."""
        invoice = event_data.get("object", {})
        customer_id = invoice.get("customer")

        sub = await repo.get_subscription_by_stripe_customer(customer_id)
        if not sub:
            return {"handled": False, "reason": "unknown_customer", "retryable": True}

        org_id = sub["org_id"]

        # Record payment
        await repo.add_payment(
            org_id=org_id,
            stripe_invoice_id=invoice.get("id"),
            amount_cents=invoice.get("amount_paid", 0),
            currency=invoice.get("currency", "usd"),
            status="succeeded",
            description=invoice.get("description"),
            invoice_pdf_url=invoice.get("invoice_pdf"),
        )

        logger.info(f"Invoice paid for org {org_id}: ${invoice.get('amount_paid', 0) / 100:.2f}")
        return {"handled": True, "org_id": org_id}

    async def _handle_payment_failed(
        self,
        event_data: dict[str, Any],
        repo: Any,
    ) -> dict[str, Any]:
        """Handle invoice.payment_failed event."""
        invoice = event_data.get("object", {})
        customer_id = invoice.get("customer")

        sub = await repo.get_subscription_by_stripe_customer(customer_id)
        if not sub:
            return {"handled": False, "reason": "unknown_customer", "retryable": True}

        org_id = sub["org_id"]

        # Record failed payment
        await repo.add_payment(
            org_id=org_id,
            stripe_invoice_id=invoice.get("id"),
            amount_cents=invoice.get("amount_due", 0),
            currency=invoice.get("currency", "usd"),
            status="failed",
            description="Payment failed",
            invoice_pdf_url=invoice.get("invoice_pdf"),
        )

        # Update subscription status to past_due
        await repo.update_org_subscription(org_id, status="past_due")

        await repo.log_billing_action(
            org_id=org_id,
            action="payment.failed",
            details={
                "invoice_id": invoice.get("id"),
                "amount_due": invoice.get("amount_due", 0),
            },
        )

        logger.warning(f"Payment failed for org {org_id}")
        return {"handled": True, "org_id": org_id}

    # =========================================================================
    # Payment History
    # =========================================================================

    async def list_invoices(
        self,
        org_id: int,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[dict[str, Any]], int]:
        """List payment/invoice history for an organization."""
        repo = await self._require_billing_repo()
        return await repo.list_payments(org_id, limit=limit, offset=offset)


# Singleton instance with async-safe initialization
_subscription_service: SubscriptionService | None = None
_subscription_service_lock = asyncio.Lock()


async def get_subscription_service() -> SubscriptionService:
    """Get or create the subscription service singleton (async-safe)."""
    global _subscription_service
    if _subscription_service is None:
        async with _subscription_service_lock:
            # Double-check pattern for async safety
            if _subscription_service is None:
                _subscription_service = SubscriptionService()
    return _subscription_service


async def reset_subscription_service() -> None:
    """Reset the subscription service singleton (primarily for tests)."""
    global _subscription_service
    _subscription_service = None
