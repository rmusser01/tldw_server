"""
Unit tests for get_billing_org_id and resolve_org_id_for_principal helpers.
"""
from __future__ import annotations

import pytest
from fastapi import HTTPException, Response

from tldw_Server_API.app.api.v1.API_Deps import billing_deps
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal


# ---------------------------------------------------------------------------
# get_billing_org_id
# ---------------------------------------------------------------------------

class TestGetBillingOrgId:
    """Tests for the get_billing_org_id FastAPI dependency."""

    @pytest.mark.asyncio
    async def test_returns_none_when_enforcement_disabled(self, monkeypatch):
        """When LIMIT_ENFORCEMENT_ENABLED=false, returns None immediately."""
        monkeypatch.setenv("LIMIT_ENFORCEMENT_ENABLED", "false")
        principal = AuthPrincipal(kind="user", user_id=1, is_admin=False)
        result = await billing_deps.get_billing_org_id(
            principal=principal, x_tldw_org_id=None, org_id=None,
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_resolved_org_id(self, monkeypatch):
        """When enforcement is on and org resolves, returns the org_id."""
        monkeypatch.setenv("LIMIT_ENFORCEMENT_ENABLED", "true")

        async def _fake_resolve(principal, org_id=None, x_tldw_org_id=None):
            return 42

        monkeypatch.setattr(billing_deps, "_resolve_org_id", _fake_resolve, raising=False)
        principal = AuthPrincipal(kind="user", user_id=1, is_admin=False)
        result = await billing_deps.get_billing_org_id(
            principal=principal, x_tldw_org_id=None, org_id=None,
        )
        assert result == 42

    @pytest.mark.asyncio
    async def test_returns_none_on_http_error_when_orgless_allowed(self, monkeypatch):
        """When org resolution fails and orgless access is allowed, returns None."""
        monkeypatch.setenv("LIMIT_ENFORCEMENT_ENABLED", "true")
        monkeypatch.setattr(billing_deps, "_allow_orgless_billing_access", lambda: True, raising=False)

        async def _fail_resolve(principal, org_id=None, x_tldw_org_id=None):
            raise HTTPException(status_code=403, detail="No org")

        monkeypatch.setattr(billing_deps, "_resolve_org_id", _fail_resolve, raising=False)
        principal = AuthPrincipal(kind="user", user_id=1, is_admin=False)
        result = await billing_deps.get_billing_org_id(
            principal=principal, x_tldw_org_id=None, org_id=None,
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_raises_when_orgless_not_allowed(self, monkeypatch):
        """When org resolution fails and orgless access is NOT allowed, raises."""
        monkeypatch.setenv("LIMIT_ENFORCEMENT_ENABLED", "true")
        monkeypatch.setattr(billing_deps, "_allow_orgless_billing_access", lambda: False, raising=False)

        async def _fail_resolve(principal, org_id=None, x_tldw_org_id=None):
            raise HTTPException(status_code=403, detail="No org")

        monkeypatch.setattr(billing_deps, "_resolve_org_id", _fail_resolve, raising=False)
        principal = AuthPrincipal(kind="user", user_id=1, is_admin=False)

        with pytest.raises(HTTPException) as exc_info:
            await billing_deps.get_billing_org_id(
                principal=principal, x_tldw_org_id=None, org_id=None,
            )
        assert exc_info.value.status_code == 403


# ---------------------------------------------------------------------------
# resolve_org_id_for_principal
# ---------------------------------------------------------------------------

class TestResolveOrgIdForPrincipal:
    """Tests for the WebSocket-compatible resolve_org_id_for_principal helper."""

    @pytest.mark.asyncio
    async def test_returns_none_when_enforcement_disabled(self, monkeypatch):
        monkeypatch.setenv("LIMIT_ENFORCEMENT_ENABLED", "false")
        principal = AuthPrincipal(kind="user", user_id=1, is_admin=False)
        result = await billing_deps.resolve_org_id_for_principal(principal)
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_resolved_org_id(self, monkeypatch):
        monkeypatch.setenv("LIMIT_ENFORCEMENT_ENABLED", "true")

        async def _fake_resolve(principal, org_id=None, x_tldw_org_id=None):
            return 99

        monkeypatch.setattr(billing_deps, "_resolve_org_id", _fake_resolve, raising=False)
        principal = AuthPrincipal(kind="user", user_id=1, is_admin=False)
        result = await billing_deps.resolve_org_id_for_principal(principal)
        assert result == 99

    @pytest.mark.asyncio
    async def test_returns_none_on_generic_exception(self, monkeypatch):
        """Generic exceptions are caught and return None (fail-open)."""
        monkeypatch.setenv("LIMIT_ENFORCEMENT_ENABLED", "true")
        monkeypatch.setattr(billing_deps, "_allow_orgless_billing_access", lambda: False, raising=False)

        async def _fail_resolve(principal, org_id=None, x_tldw_org_id=None):
            raise RuntimeError("DB unavailable")

        monkeypatch.setattr(billing_deps, "_resolve_org_id", _fail_resolve, raising=False)
        principal = AuthPrincipal(kind="user", user_id=1, is_admin=False)
        result = await billing_deps.resolve_org_id_for_principal(principal)
        assert result is None

    @pytest.mark.asyncio
    async def test_generic_exception_log_omits_raw_backend_details(self, monkeypatch):
        """Generic fallback logs must not expose backend exception text."""
        monkeypatch.setenv("LIMIT_ENFORCEMENT_ENABLED", "true")
        sensitive_detail = "sqlite:///private/billing.db?token=super-secret"
        messages: list[str] = []
        sink_id = billing_deps.logger.add(messages.append, level="DEBUG", format="{message}")

        async def _fail_resolve(principal, org_id=None, x_tldw_org_id=None):
            raise RuntimeError(sensitive_detail)

        monkeypatch.setattr(billing_deps, "_resolve_org_id", _fail_resolve, raising=False)
        principal = AuthPrincipal(kind="user", user_id=1, is_admin=False)

        try:
            result = await billing_deps.resolve_org_id_for_principal(principal)
        finally:
            billing_deps.logger.remove(sink_id)

        assert result is None
        rendered = "\n".join(messages)
        assert "resolve_org_id_for_principal failed" in rendered
        assert sensitive_detail not in rendered
        assert "super-secret" not in rendered
        assert "/private/billing.db" not in rendered


# ---------------------------------------------------------------------------
# add_billing_headers
# ---------------------------------------------------------------------------

class TestAddBillingHeaders:
    """Tests for the best-effort billing header helper."""

    @pytest.mark.asyncio
    async def test_generic_header_exception_log_omits_raw_backend_details(self, monkeypatch):
        """Header fallback logs must not expose backend exception text."""
        monkeypatch.setenv("LIMIT_ENFORCEMENT_ENABLED", "true")
        sensitive_detail = "failed reading /private/plans.db with api_key=super-secret"
        messages: list[str] = []
        sink_id = billing_deps.logger.add(messages.append, level="DEBUG", format="{message}")

        async def _fake_resolve(principal, org_id=None, x_tldw_org_id=None):
            return 17

        class _FailingEnforcer:
            async def get_org_limits(self, org_id):
                raise RuntimeError(sensitive_detail)

            async def get_org_usage(self, org_id):
                raise AssertionError("usage should not be loaded after limits fail")

        monkeypatch.setattr(billing_deps, "_resolve_org_id", _fake_resolve, raising=False)
        monkeypatch.setattr(billing_deps, "get_billing_enforcer", lambda: _FailingEnforcer(), raising=False)
        response = Response()
        principal = AuthPrincipal(kind="user", user_id=1, is_admin=False)

        try:
            result = await billing_deps.add_billing_headers(
                response=response,
                principal=principal,
                x_tldw_org_id=None,
                org_id=None,
            )
        finally:
            billing_deps.logger.remove(sink_id)

        assert result is None
        rendered = "\n".join(messages)
        assert "Failed to add billing headers" in rendered
        assert sensitive_detail not in rendered
        assert "super-secret" not in rendered
        assert "/private/plans.db" not in rendered
