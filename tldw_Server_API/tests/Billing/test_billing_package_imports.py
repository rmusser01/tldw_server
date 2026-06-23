from __future__ import annotations

import importlib
from pathlib import Path

import pytest


def test_subscription_service_module_no_longer_imports_billing_repo_eagerly() -> None:
    """OSS runtime should not depend on billing_repo at module import time."""
    module = importlib.import_module("tldw_Server_API.app.core.Billing.subscription_service")
    source = Path(module.__file__).read_text(encoding="utf-8")

    assert "core.AuthNZ.repos.billing_repo" not in source


def test_billing_package_lazy_subscription_exports() -> None:
    """Billing package should lazily expose subscription service symbols."""
    module = importlib.import_module("tldw_Server_API.app.core.Billing")
    assert callable(getattr(module, "get_subscription_service"))


@pytest.mark.parametrize(
    "module_name",
    [
        # Note: endpoints.billing (billing-status router) is intentionally retained
        # in OSS builds (billing is disabled at runtime, but the status endpoint and
        # its router registration still exist — see admin.py and the AuthNZ/Services
        # router-group contract tests). Only the payment-processing modules are removed.
        "tldw_Server_API.app.api.v1.endpoints.billing_webhooks",
        "tldw_Server_API.app.core.Billing.stripe_client",
        "tldw_Server_API.app.services.stripe_metering_service",
    ],
)
def test_oss_runtime_removes_public_billing_modules(module_name: str) -> None:
    """OSS should not ship Stripe / payment-processing runtime modules."""
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module_name)


def test_pg_billing_table_ensure_includes_runtime_tables() -> None:
    """Postgres OSS compatibility helper should only retain budget storage DDL."""
    module = importlib.import_module("tldw_Server_API.app.core.AuthNZ.pg_migrations_extra")
    ddl_statements = " ".join(
        sql.lower()
        for sql, _params in getattr(module, "_CREATE_BILLING_TABLES")
    )
    assert "create table if not exists org_budgets" in ddl_statements
    assert "create table if not exists subscription_plans" not in ddl_statements
    assert "create table if not exists org_subscriptions" not in ddl_statements
    assert "create table if not exists stripe_webhook_events" not in ddl_statements
    assert "create table if not exists payment_history" not in ddl_statements
    assert "create table if not exists billing_audit_log" not in ddl_statements
    assert not hasattr(module, "_backfill_org_budgets_pg")
