"""
Unit tests for SubscriptionService.

These tests focus on subscription creation semantics, specifically:
- Unknown plan names should raise a ValueError rather than silently
    downgrading to the free tier.
- Valid plan names should delegate to the billing repo with the expected
    arguments and status values.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import pytest

from tldw_Server_API.app.core.Billing.subscription_service import SubscriptionService


class _FakeBillingRepo:
    def __init__(self) -> None:
        self.last_create_args: Optional[Dict[str, Any]] = None
        self.last_log_action: Optional[Dict[str, Any]] = None
        self.last_updated_subscription: Optional[Dict[str, Any]] = None
        self.last_payment: Optional[Dict[str, Any]] = None
        self._org_subscriptions: Dict[int, Dict[str, Any]] = {}

    async def get_plan_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        if name == "free":
            return {"id": 1, "name": "free"}
        if name == "pro":
            return {"id": 2, "name": "pro"}
        return None

    async def create_org_subscription(
        self,
        *,
        org_id: int,
        plan_id: int,
        stripe_customer_id: Optional[str] = None,
        stripe_subscription_id: Optional[str] = None,
        billing_cycle: str = "monthly",
        status: str = "active",
        trial_days: Optional[int] = None,
    ) -> Dict[str, Any]:
        self.last_create_args = {
            "org_id": org_id,
            "plan_id": plan_id,
            "stripe_customer_id": stripe_customer_id,
            "stripe_subscription_id": stripe_subscription_id,
            "billing_cycle": billing_cycle,
            "status": status,
            "trial_days": trial_days,
        }
        created = {
            "org_id": org_id,
            "plan_id": plan_id,
            "stripe_customer_id": stripe_customer_id,
            "stripe_subscription_id": stripe_subscription_id,
            "billing_cycle": billing_cycle,
            "status": status,
        }
        self._org_subscriptions[org_id] = dict(created)
        return created

    async def log_billing_action(
        self,
        *,
        org_id: int,
        action: str,
        user_id: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None,
    ) -> Dict[str, Any]:
        self.last_log_action = {
            "org_id": org_id,
            "action": action,
            "details": details,
        }
        return {
            "org_id": org_id,
            "action": action,
            "details": details,
        }

    async def get_org_subscription(self, org_id: int) -> Optional[Dict[str, Any]]:
        sub = self._org_subscriptions.get(org_id)
        return dict(sub) if sub else None

    async def get_subscription_by_stripe_customer(self, stripe_customer_id: str) -> Optional[Dict[str, Any]]:
        for sub in self._org_subscriptions.values():
            if sub.get("stripe_customer_id") == stripe_customer_id:
                return dict(sub)
        # Default: pretend we have a pro subscription for org 1
        if stripe_customer_id == "cus_pro_1":
            return {
                "org_id": 1,
                "plan_id": 2,
                "plan_name": "pro",
                "plan_display_name": "Pro",
                "plan_limits_json": '{"api_calls_day": 5000}',
                "custom_limits_json": None,
            }
        return None

    async def get_plan_by_stripe_price_id(self, price_id: str) -> Optional[Dict[str, Any]]:
        if price_id == "price_pro":
            return {"id": 2, "name": "pro"}
        if price_id == "price_enterprise":
            return {"id": 3, "name": "enterprise"}
        return None

    async def get_plan_by_stripe_product_id(self, product_id: str) -> Optional[Dict[str, Any]]:
        if product_id == "prod_pro":
            return {"id": 2, "name": "pro"}
        if product_id == "prod_enterprise":
            return {"id": 3, "name": "enterprise"}
        return None

    async def update_org_subscription(self, org_id: int, **updates: Any) -> None:
        # Record last update for assertions in webhook tests.
        self.last_updated_subscription = {"org_id": org_id, **updates}
        existing = self._org_subscriptions.get(org_id)
        if existing:
            existing.update(updates)

    async def add_payment(
        self,
        *,
        org_id: int,
        stripe_invoice_id: Optional[str] = None,
        amount_cents: int,
        currency: str = "usd",
        status: str = "succeeded",
        description: Optional[str] = None,
        invoice_pdf_url: Optional[str] = None,
    ) -> Dict[str, Any]:
        self.last_payment = {
            "org_id": org_id,
            "stripe_invoice_id": stripe_invoice_id,
            "amount_cents": amount_cents,
            "currency": currency,
            "status": status,
            "description": description,
            "invoice_pdf_url": invoice_pdf_url,
        }
        return dict(self.last_payment)


class _EmptyPlansRepo:
    async def list_plans(self, *, active_only: bool = True, public_only: bool = True):
        return []

    async def get_plan_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        return None


class _FakeStripeClient:
    """Minimal Stripe client stub for checkout tests."""

    def __init__(self) -> None:

        self.is_available = True
        self.created_customers: list[Dict[str, Any]] = []
        self.created_sessions: list[Dict[str, Any]] = []

    async def create_customer(self, *, email: str, name: Optional[str] = None, metadata: Optional[Dict[str, str]] = None) -> str:
        self.created_customers.append({"email": email, "name": name, "metadata": metadata})
        return "cus_test_123"

    def get_price_id(self, plan_name: str, billing_cycle: str = "monthly") -> Optional[str]:
        # Price lookup should never be reached in the unknown-plan test.
        return "price_test_123"

    async def create_checkout_session(
        self,
        *,
        customer_id: str,
        price_id: str,
        success_url: str,
        cancel_url: str,
        metadata: Optional[Dict[str, str]] = None,
    ):
        from tldw_Server_API.app.core.Billing.subscription_service import CheckoutSession

        self.created_sessions.append(
            {
                "customer_id": customer_id,
                "price_id": price_id,
                "success_url": success_url,
                "cancel_url": cancel_url,
                "metadata": metadata,
            }
        )
        return CheckoutSession(id="sess_test_123", url="https://example.com/checkout")


class _RepoForCancelResume:
    def __init__(self) -> None:
        self.updated_rows: list[Dict[str, Any]] = []
        self.logged_actions: list[Dict[str, Any]] = []

    async def get_org_subscription(self, org_id: int) -> Optional[Dict[str, Any]]:
        return {
            "org_id": org_id,
            "stripe_subscription_id": "sub_live_123",
            "status": "active",
            "cancel_at_period_end": False,
        }

    async def update_org_subscription(self, org_id: int, **updates: Any) -> Dict[str, Any]:
        row = {"org_id": org_id, **updates}
        self.updated_rows.append(row)
        return row

    async def log_billing_action(self, **kwargs: Any) -> Dict[str, Any]:
        self.logged_actions.append(dict(kwargs))
        return dict(kwargs)


class _StripeUnavailableClient:
    is_available = False


class _StripeFailingClient:
    is_available = True

    async def cancel_subscription(self, subscription_id: str, *, at_period_end: bool = True) -> Dict[str, Any]:
        raise RuntimeError("stripe cancel failed")

    async def resume_subscription(self, subscription_id: str) -> Dict[str, Any]:
        raise RuntimeError("stripe resume failed")


@pytest.mark.asyncio
async def test_cancel_subscription_fails_closed_when_stripe_unavailable(monkeypatch) -> None:
    """Stripe-backed cancellations should fail closed when Stripe is unavailable."""

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Billing.subscription_service.is_billing_enabled",
        lambda: True,
    )

    repo = _RepoForCancelResume()
    service = SubscriptionService(billing_repo=repo, stripe_client=_StripeUnavailableClient())

    with pytest.raises(RuntimeError, match="Stripe is not configured"):
        await service.cancel_subscription(9, at_period_end=False, user_id=1)

    assert repo.updated_rows == []
    assert repo.logged_actions == []


@pytest.mark.asyncio
async def test_resume_subscription_fails_closed_when_stripe_unavailable(monkeypatch) -> None:
    """Stripe-backed resumes should fail closed when Stripe is unavailable."""

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Billing.subscription_service.is_billing_enabled",
        lambda: True,
    )

    repo = _RepoForCancelResume()
    service = SubscriptionService(billing_repo=repo, stripe_client=_StripeUnavailableClient())

    with pytest.raises(RuntimeError, match="Stripe is not configured"):
        await service.resume_subscription(9, user_id=1)

    assert repo.updated_rows == []
    assert repo.logged_actions == []


@pytest.mark.asyncio
async def test_cancel_subscription_propagates_runtime_error_without_local_mutation(monkeypatch) -> None:
    """Stripe cancellation failures should bubble as RuntimeError and avoid local updates."""

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Billing.subscription_service.is_billing_enabled",
        lambda: True,
    )

    repo = _RepoForCancelResume()
    service = SubscriptionService(billing_repo=repo, stripe_client=_StripeFailingClient())

    with pytest.raises(RuntimeError, match="Failed to cancel subscription in Stripe"):
        await service.cancel_subscription(11, at_period_end=True, user_id=1)

    assert repo.updated_rows == []
    assert repo.logged_actions == []


@pytest.mark.asyncio
async def test_resume_subscription_propagates_runtime_error_without_local_mutation(monkeypatch) -> None:
    """Stripe resume failures should bubble as RuntimeError and avoid local updates."""

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Billing.subscription_service.is_billing_enabled",
        lambda: True,
    )

    repo = _RepoForCancelResume()
    service = SubscriptionService(billing_repo=repo, stripe_client=_StripeFailingClient())

    with pytest.raises(RuntimeError, match="Failed to resume subscription in Stripe"):
        await service.resume_subscription(11, user_id=1)

    assert repo.updated_rows == []
    assert repo.logged_actions == []


@pytest.mark.asyncio
async def test_list_available_plans_returns_neutral_free_fallback_when_repository_has_no_public_plans() -> None:
    """OSS should expose the neutral free tier when no public plans exist."""

    service = SubscriptionService(billing_repo=_EmptyPlansRepo())

    plans = await service.list_available_plans()

    assert len(plans) == 1
    assert plans[0]["name"] == "free"
    assert plans[0]["display_name"] == "Free"
    assert plans[0]["price_usd_monthly"] == 0
    assert plans[0]["price_usd_yearly"] == 0
    assert plans[0]["is_public"] is False


@pytest.mark.asyncio
async def test_get_plan_only_synthesizes_neutral_free_fallback() -> None:
    """Only the neutral free plan should be synthesized without a database row."""

    service = SubscriptionService(billing_repo=_EmptyPlansRepo())

    free_plan = await service.get_plan("free")
    paid_plan = await service.get_plan("pro")

    assert free_plan is not None
    assert free_plan["name"] == "free"
    assert free_plan["display_name"] == "Free"
    assert free_plan["price_usd_monthly"] == 0
    assert free_plan["price_usd_yearly"] == 0
    assert free_plan["limits"]["storage_mb"] > 0
    assert paid_plan is None


@pytest.mark.asyncio
async def test_get_plan_for_checkout_requires_active_public_plan() -> None:
    """get_plan_for_checkout should return None for inactive or hidden plans."""

    class _PlanRepo:
        async def get_plan_by_name(self, name: str) -> Optional[Dict[str, Any]]:
            if name == "inactive":
                return {"id": 99, "name": name, "is_active": False}
            if name == "hidden":
                return {"id": 100, "name": name, "is_active": True, "is_public": False}
            if name == "active":
                return {"id": 101, "name": name, "is_active": True, "is_public": True}
            return None

    service = SubscriptionService(billing_repo=_PlanRepo())

    assert await service.get_plan_for_checkout("active") is not None
    assert await service.get_plan_for_checkout("inactive") is None
    assert await service.get_plan_for_checkout("hidden") is None
    assert await service.get_plan_for_checkout("missing") is None


@pytest.mark.asyncio
async def test_create_subscription_unknown_plan_raises_value_error() -> None:
    """create_subscription should raise when plan_name is unknown."""
    repo = _FakeBillingRepo()
    service = SubscriptionService(billing_repo=repo)

    with pytest.raises(ValueError) as exc_info:
        await service.create_subscription(org_id=42, plan_name="unknown_plan")

    assert "unknown_plan" in str(exc_info.value)


@pytest.mark.asyncio
async def test_create_subscription_for_known_plan_uses_repo_and_logs() -> None:
    """create_subscription should delegate to billing repo for known plans."""
    repo = _FakeBillingRepo()
    service = SubscriptionService(billing_repo=repo)

    result = await service.create_subscription(
        org_id=7,
        plan_name="pro",
        billing_cycle="yearly",
        trial_days=14,
    )

    # Repo create call should reflect the requested plan and parameters.
    assert repo.last_create_args is not None
    assert repo.last_create_args["org_id"] == 7
    assert repo.last_create_args["plan_id"] == 2  # pro plan
    assert repo.last_create_args["billing_cycle"] == "yearly"
    # Non-free plans start as pending
    assert repo.last_create_args["status"] == "pending"
    assert repo.last_create_args["trial_days"] == 14

    # Returned subscription mirrors repo output.
    assert result["org_id"] == 7
    assert result["plan_id"] == 2
    assert result["billing_cycle"] == "yearly"
    assert result["status"] == "pending"

    # Billing action should be logged.
    assert repo.last_log_action is not None
    assert repo.last_log_action["org_id"] == 7
    assert repo.last_log_action["action"] == "subscription.created"
    details = repo.last_log_action["details"] or {}
    assert details.get("plan_name") == "pro"
    assert details.get("billing_cycle") == "yearly"
    assert details.get("trial_days") == 14


@pytest.mark.asyncio
async def test_create_checkout_session_unknown_plan_raises_value_error(monkeypatch) -> None:
    """create_checkout_session should raise when the requested plan does not exist."""

    # Ensure billing checks pass for the unit test.
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Billing.subscription_service.is_billing_enabled",
        lambda: True,
    )

    class _RepoNoPlan(_FakeBillingRepo):
        async def get_plan_by_name(self, name: str) -> Optional[Dict[str, Any]]:
            # Simulate missing plan in DB for all names.
            return None

    repo = _RepoNoPlan()
    stripe_client = _FakeStripeClient()
    service = SubscriptionService(billing_repo=repo, stripe_client=stripe_client)

    with pytest.raises(ValueError) as exc_info:
        await service.create_checkout_session(
            org_id=42,
            plan_name="pro",
            billing_cycle="monthly",
            success_url="https://example.com/success",
            cancel_url="https://example.com/cancel",
            org_email="owner@example.com",
            org_name="Example Org",
        )

    msg = str(exc_info.value)
    assert "Plan 'pro' not found" in msg
    assert stripe_client.created_customers == []


@pytest.mark.asyncio
async def test_create_checkout_session_rejects_disallowed_redirect_before_stripe_side_effect(monkeypatch) -> None:
    """Checkout redirects should be constrained to the configured public web origin."""

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Billing.subscription_service.is_billing_enabled",
        lambda: True,
    )
    monkeypatch.setenv("PUBLIC_WEB_BASE_URL", "https://app.example.test")

    repo = _FakeBillingRepo()
    stripe_client = _FakeStripeClient()
    service = SubscriptionService(billing_repo=repo, stripe_client=stripe_client)

    with pytest.raises(ValueError, match="Redirect URL"):
        await service.create_checkout_session(
            org_id=42,
            plan_name="pro",
            billing_cycle="monthly",
            success_url="https://evil.example.test/success",
            cancel_url="https://app.example.test/cancel",
            org_email="owner@example.com",
            org_name="Example Org",
        )

    assert stripe_client.created_customers == []
    assert stripe_client.created_sessions == []


class _RepoForSubscriptionRead:
    async def get_org_subscription(self, org_id: int) -> Dict[str, Any]:
        assert org_id == 7
        return {
            "org_id": org_id,
            "plan_name": "pro",
            "plan_display_name": "Pro",
            "status": "trialing",
            "billing_cycle": "yearly",
            "current_period_end": "2026-07-01T00:00:00+00:00",
            "trial_end": "2026-06-30T00:00:00+00:00",
            "cancel_at_period_end": True,
            "effective_limits": {"api_calls_day": 5000, "llm_tokens_month": 5_000_000},
        }

    async def get_org_limits(self, org_id: int) -> Dict[str, Any]:
        assert org_id == 7
        return {"api_calls_day": 5000, "llm_tokens_month": 5_000_000}


@pytest.mark.asyncio
async def test_injected_subscription_repo_reads_return_persisted_subscription_and_limits() -> None:
    """Injected-client compatibility should read back persisted subscription state."""

    service = SubscriptionService(billing_repo=_RepoForSubscriptionRead())

    subscription = await service.get_subscription(7)
    limits = await service.get_org_limits(7)

    assert subscription.plan_name == "pro"
    assert subscription.plan_display_name == "Pro"
    assert subscription.status == "trialing"
    assert subscription.cancel_at_period_end is True
    assert subscription.limits["api_calls_day"] == 5000
    assert limits["llm_tokens_month"] == 5_000_000


class _IdempotentWebhookRepo:
    def __init__(self) -> None:
        self.events: dict[str, str] = {}
        self.payments: list[Dict[str, Any]] = []
        self.marked: list[tuple[str, str | None]] = []

    async def record_webhook_event(self, stripe_event_id: str, event_type: str, event_data: Dict[str, Any]) -> bool:
        if stripe_event_id in self.events:
            return False
        self.events[stripe_event_id] = "pending"
        return True

    async def try_claim_webhook_event(self, stripe_event_id: str, *, processing_timeout_seconds: int | None = None) -> bool:
        if self.events.get(stripe_event_id) == "processed":
            return False
        self.events[stripe_event_id] = "processing"
        return True

    async def mark_webhook_processed(self, stripe_event_id: str, *, error_message: str | None = None) -> None:
        self.events[stripe_event_id] = "failed" if error_message else "processed"
        self.marked.append((stripe_event_id, error_message))

    async def get_subscription_by_stripe_customer(self, stripe_customer_id: str) -> Dict[str, Any]:
        assert stripe_customer_id == "cus_idem"
        return {"org_id": 88}

    async def add_payment(self, **kwargs: Any) -> Dict[str, Any]:
        self.payments.append(dict(kwargs))
        return dict(kwargs)


@pytest.mark.asyncio
async def test_handle_webhook_event_claims_event_id_before_mutating_payment_history() -> None:
    """Duplicate webhook event IDs should not create duplicate invoice payments."""

    repo = _IdempotentWebhookRepo()
    service = SubscriptionService(billing_repo=repo)
    event_data = {
        "object": {
            "id": "in_test_123",
            "customer": "cus_idem",
            "amount_paid": 1200,
            "currency": "usd",
            "description": "Monthly plan",
            "invoice_pdf": "https://stripe.example/invoice.pdf",
        }
    }

    first = await service.handle_webhook_event(
        "invoice.paid",
        event_data,
        stripe_event_id="evt_idem_1",
    )
    duplicate = await service.handle_webhook_event(
        "invoice.paid",
        event_data,
        stripe_event_id="evt_idem_1",
    )

    assert first["handled"] is True
    assert duplicate == {
        "handled": True,
        "event_type": "invoice.paid",
        "duplicate": True,
        "retryable": False,
    }
    assert len(repo.payments) == 1
    assert repo.marked == [("evt_idem_1", None)]


@pytest.mark.asyncio
async def test_handle_checkout_completed_updates_plan_and_cycle() -> None:
    """checkout.session.completed should persist plan_id and billing_cycle from metadata."""
    repo = _FakeBillingRepo()
    repo._org_subscriptions[7] = {
        "org_id": 7,
        "plan_id": 1,
        "stripe_customer_id": "cus_456",
        "billing_cycle": "monthly",
        "status": "pending",
    }
    service = SubscriptionService(billing_repo=repo)

    event_data = {
        "object": {
            "id": "cs_test_1",
            "subscription": "sub_123",
            "customer": "cus_456",
            "metadata": {
                "org_id": "7",
                "plan_name": "pro",
                "billing_cycle": "yearly",
            },
        }
    }

    result = await service._handle_checkout_completed(event_data, repo)  # type: ignore[attr-defined]

    assert result["handled"] is True
    assert repo.last_updated_subscription is not None
    assert repo.last_updated_subscription["org_id"] == 7
    assert repo.last_updated_subscription["plan_id"] == 2
    assert repo.last_updated_subscription["billing_cycle"] == "yearly"
    assert repo.last_updated_subscription["status"] == "active"
    assert repo.last_updated_subscription["stripe_subscription_id"] == "sub_123"
    assert repo.last_updated_subscription["stripe_customer_id"] == "cus_456"


@pytest.mark.asyncio
async def test_handle_checkout_completed_creates_subscription_when_missing() -> None:
    """checkout.session.completed should create an org subscription when none exists."""
    repo = _FakeBillingRepo()
    service = SubscriptionService(billing_repo=repo)

    event_data = {
        "object": {
            "id": "cs_test_new",
            "subscription": "sub_new_123",
            "customer": "cus_new_456",
            "metadata": {
                "org_id": "9",
                "plan_name": "pro",
                "billing_cycle": "yearly",
            },
        }
    }

    result = await service._handle_checkout_completed(event_data, repo)  # type: ignore[attr-defined]

    assert result["handled"] is True
    assert repo.last_create_args is not None
    assert repo.last_create_args["org_id"] == 9
    assert repo.last_create_args["plan_id"] == 2
    assert repo.last_create_args["status"] == "active"
    created = await repo.get_org_subscription(9)
    assert created is not None
    assert created["stripe_subscription_id"] == "sub_new_123"
    assert created["billing_cycle"] == "yearly"


@pytest.mark.asyncio
async def test_handle_subscription_updated_updates_plan_from_price(monkeypatch) -> None:
    """_handle_subscription_updated should update plan_id when price_id maps to a new plan."""

    repo = _FakeBillingRepo()
    service = SubscriptionService(billing_repo=repo)

    event_data = {
        "object": {
            "customer": "cus_pro_1",
            "status": "active",
            "current_period_start": 1700000000,
            "current_period_end": 1702592000,
            "items": {
                "data": [
                    {
                        "price": {
                            "id": "price_enterprise",
                            "product": "prod_enterprise",
                        }
                    }
                ]
            },
        }
    }

    result = await service._handle_subscription_updated(event_data, repo)  # type: ignore[attr-defined]

    assert result["handled"] is True
    assert repo.last_updated_subscription is not None
    assert repo.last_updated_subscription["org_id"] == 1
    # Plan should be updated to enterprise plan (id=3 in fake repo)
    assert repo.last_updated_subscription["plan_id"] == 3
    assert repo.last_updated_subscription["stripe_subscription_status"] == "active"


@pytest.mark.asyncio
async def test_handle_subscription_updated_updates_status_when_plan_unknown(monkeypatch) -> None:
    """_handle_subscription_updated should still update status even when plan cannot be resolved."""

    class _RepoNoPlanMapping(_FakeBillingRepo):
        async def get_plan_by_stripe_price_id(self, price_id: str) -> Optional[Dict[str, Any]]:
            return None

        async def get_plan_by_stripe_product_id(self, product_id: str) -> Optional[Dict[str, Any]]:
            return None

    repo = _RepoNoPlanMapping()
    service = SubscriptionService(billing_repo=repo)

    event_data = {
        "object": {
            "customer": "cus_pro_1",
            "status": "past_due",
            "current_period_start": 1700000000,
            "current_period_end": 1702592000,
            "items": {
                "data": [
                    {
                        "price": {
                            "id": "price_unknown",
                            "product": "prod_unknown",
                        }
                    }
                ]
            },
        }
    }

    result = await service._handle_subscription_updated(event_data, repo)  # type: ignore[attr-defined]

    assert result["handled"] is True
    assert repo.last_updated_subscription is not None
    assert repo.last_updated_subscription["org_id"] == 1
    # Plan_id should not be present when mapping fails
    assert "plan_id" not in repo.last_updated_subscription
    assert repo.last_updated_subscription["stripe_subscription_status"] == "past_due"


@pytest.mark.asyncio
async def test_handle_payment_failed_records_currency_and_pdf(monkeypatch) -> None:
    """_handle_payment_failed should record invoice currency and PDF URL and mark subscription past_due."""

    repo = _FakeBillingRepo()
    service = SubscriptionService(billing_repo=repo)

    event_data = {
        "object": {
            "customer": "cus_pro_1",
            "id": "in_test_123",
            "amount_due": 4321,
            "currency": "eur",
            "invoice_pdf": "https://example.com/invoice.pdf",
        }
    }

    result = await service._handle_payment_failed(event_data, repo)  # type: ignore[attr-defined]

    assert result["handled"] is True

    # Payment record should reflect invoice fields.
    assert repo.last_payment is not None
    assert repo.last_payment["org_id"] == 1
    assert repo.last_payment["stripe_invoice_id"] == "in_test_123"
    assert repo.last_payment["amount_cents"] == 4321
    assert repo.last_payment["currency"] == "eur"
    assert repo.last_payment["status"] == "failed"
    assert repo.last_payment["description"] == "Payment failed"
    assert repo.last_payment["invoice_pdf_url"] == "https://example.com/invoice.pdf"

    # Subscription should be marked past_due.
    assert repo.last_updated_subscription is not None
    assert repo.last_updated_subscription["org_id"] == 1
    assert repo.last_updated_subscription["status"] == "past_due"


@pytest.mark.asyncio
async def test_handle_payment_failed_uses_default_currency_when_missing(monkeypatch) -> None:
    """_handle_payment_failed should default currency to usd when invoice currency is missing."""

    repo = _FakeBillingRepo()
    service = SubscriptionService(billing_repo=repo)

    event_data = {
        "object": {
            "customer": "cus_pro_1",
            "id": "in_test_456",
            "amount_due": 1000,
            # no currency field
            # no invoice_pdf field
        }
    }

    result = await service._handle_payment_failed(event_data, repo)  # type: ignore[attr-defined]

    assert result["handled"] is True

    assert repo.last_payment is not None
    assert repo.last_payment["currency"] == "usd"
    assert repo.last_payment["invoice_pdf_url"] is None


@pytest.mark.asyncio
async def test_handle_subscription_deleted_downgrades_to_free_plan() -> None:
    """customer.subscription.deleted should downgrade to the free plan when available."""
    repo = _FakeBillingRepo()
    repo._org_subscriptions[1] = {
        "org_id": 1,
        "plan_id": 2,
        "stripe_customer_id": "cus_pro_1",
        "status": "active",
    }
    service = SubscriptionService(billing_repo=repo)

    event_data = {
        "object": {
            "customer": "cus_pro_1",
        }
    }

    result = await service._handle_subscription_deleted(event_data, repo)  # type: ignore[attr-defined]

    assert result["handled"] is True
    assert result["downgraded_to"] == "free"
    assert repo.last_updated_subscription is not None
    assert repo.last_updated_subscription["org_id"] == 1
    assert repo.last_updated_subscription["plan_id"] == 1
    assert repo.last_updated_subscription["status"] == "active"


@pytest.mark.asyncio
async def test_handle_subscription_deleted_without_free_plan_marks_canceled() -> None:
    """Missing free plan should not fall back to hardcoded plan IDs."""

    class _RepoNoFreePlan(_FakeBillingRepo):
        async def get_plan_by_name(self, name: str) -> Optional[Dict[str, Any]]:
            if name == "free":
                return None
            return await super().get_plan_by_name(name)

    repo = _RepoNoFreePlan()
    repo._org_subscriptions[1] = {
        "org_id": 1,
        "plan_id": 2,
        "stripe_customer_id": "cus_pro_1",
        "status": "active",
    }
    service = SubscriptionService(billing_repo=repo)

    event_data = {
        "object": {
            "customer": "cus_pro_1",
        }
    }

    result = await service._handle_subscription_deleted(event_data, repo)  # type: ignore[attr-defined]

    assert result["handled"] is True
    assert result["downgraded_to"] == "canceled_no_free_plan"
    assert repo.last_updated_subscription is not None
    assert repo.last_updated_subscription["org_id"] == 1
    assert repo.last_updated_subscription["status"] == "canceled"
    assert "plan_id" not in repo.last_updated_subscription
