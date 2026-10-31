"""Billing-plan checks run only with quotas on and a billing repository wired (spec 2 §6)."""

from types import SimpleNamespace

import pytest
from fastapi import Response
from loguru import logger

from tldw_Server_API.app.api.v1.API_Deps import billing_deps
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.Billing import enforcement, subscription_service
from tldw_Server_API.app.core.Billing.enforcement import LimitCategory
from tldw_Server_API.app.core.RAG.rag_service import transport

pytestmark = pytest.mark.unit


async def _wired() -> bool:
    """A billing_repo_configured stand-in reporting a billing repository is wired."""
    return True


async def _not_wired() -> bool:
    """A billing_repo_configured stand-in reporting no billing repository is wired."""
    return False


async def _must_not_resolve(*_args: object, **_kwargs: object) -> int:
    """An org-resolution stand-in that fails the test if it is ever called."""
    raise AssertionError("org resolution must not run")


def _principal() -> AuthPrincipal:
    """A plain non-admin user principal for billing-gate checks."""
    return AuthPrincipal(kind="user", user_id=7, is_admin=False)


@pytest.fixture(autouse=True)
def _quotas_on(monkeypatch: pytest.MonkeyPatch) -> None:
    """Quotas switched on explicitly; each test picks whether a billing repo is wired."""
    monkeypatch.delenv("LIMIT_ENFORCEMENT_ENABLED", raising=False)
    monkeypatch.setenv("USAGE_QUOTAS_ENABLED", "1")


async def test_oss_skips_org_resolution_even_with_quotas_on(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no billing repository wired, org resolution is skipped even though quotas are on."""
    monkeypatch.setattr(subscription_service, "billing_repo_configured", _not_wired)
    calls: list[tuple[object, ...]] = []

    async def _recording_resolve(*args: object, **kwargs: object) -> int:
        """An org-resolution stand-in that records its calls and returns a fixed org id."""
        calls.append((args, kwargs))
        return 42

    monkeypatch.setattr(billing_deps, "_resolve_org_id", _recording_resolve)
    assert await billing_deps.get_billing_org_id(principal=_principal(), x_tldw_org_id=None, org_id=None) is None
    assert await billing_deps.resolve_org_id_for_principal(_principal()) is None
    await billing_deps.add_billing_headers(response=Response(), principal=_principal(), x_tldw_org_id=None, org_id=None)
    assert calls == []


async def test_orgless_multi_user_account_gets_no_403(monkeypatch: pytest.MonkeyPatch) -> None:
    """A multi-user account with no org and no billing repository passes limit and feature checks unlimited, without resolving an org."""
    monkeypatch.setattr(subscription_service, "billing_repo_configured", _not_wired)
    monkeypatch.setattr(billing_deps, "_allow_orgless_billing_access", lambda: False)
    monkeypatch.setattr(billing_deps, "_resolve_org_id", _must_not_resolve)
    check = billing_deps.require_within_limit(LimitCategory.RAG_QUERIES_DAY)
    result = await check(response=Response(), principal=_principal(), x_tldw_org_id=None, org_id=None)
    assert result.unlimited is True
    feature_check = billing_deps.require_feature("advanced_analytics")
    assert await feature_check(principal=_principal(), x_tldw_org_id=None, org_id=None) is True


async def test_hosted_path_still_resolves_the_org(monkeypatch: pytest.MonkeyPatch) -> None:
    """With a billing repository wired, the org id is still resolved."""
    monkeypatch.setattr(subscription_service, "billing_repo_configured", _wired)

    async def _org(*_args: object, **_kwargs: object) -> int:
        """An org-resolution stand-in returning a fixed org id."""
        return 42

    monkeypatch.setattr(billing_deps, "_resolve_org_id", _org)
    assert await billing_deps.get_billing_org_id(principal=_principal(), x_tldw_org_id=None, org_id=None) == 42


async def test_hosted_with_quotas_off_warns_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """A hosted install with a billing repo wired but quotas off reports plan limits unenforced and logs that warning exactly once across repeated calls."""
    monkeypatch.setenv("USAGE_QUOTAS_ENABLED", "0")
    monkeypatch.setattr(subscription_service, "billing_repo_configured", _wired)
    monkeypatch.setattr(enforcement, "_PLAN_LIMITS_UNENFORCED_WARNED", False)
    messages: list[str] = []
    handler_id = logger.add(lambda m: messages.append(str(m)), level="WARNING")
    try:
        assert await enforcement.billing_checks_active() is False
        assert await enforcement.billing_checks_active() is False
    finally:
        logger.remove(handler_id)
    assert sum("plan limits are NOT enforced" in m for m in messages) == 1


async def test_rag_transport_check_skips_without_billing_repo(monkeypatch: pytest.MonkeyPatch) -> None:
    """The RAG transport's org-context limit check skips org resolution entirely when no billing repository is wired."""
    monkeypatch.setattr(subscription_service, "billing_repo_configured", _not_wired)
    monkeypatch.setattr(transport, "resolve_org_id_for_rag_context", _must_not_resolve)
    await transport.enforce_rag_query_limit_for_org_context(current_user=SimpleNamespace(id=7), units=1)
