"""
Unit tests for billing dependency org resolution logic.
"""
from __future__ import annotations

import pytest
from fastapi import HTTPException
from fastapi import Response

from tldw_Server_API.app.api.v1.API_Deps import billing_deps
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.Billing.enforcement import LimitCategory


class _FakeRepoIgnoreHeader:
    def __init__(self, *args, **kwargs) -> None:
        pass

    async def get_org_member(self, org_id: int, user_id: int):
        return None

    async def list_org_memberships_for_user(self, user_id: int):
        return [{"org_id": 42, "status": "active"}]


class _FakeRepoAdminHeader:
    def __init__(self, *args, **kwargs) -> None:
        pass

    async def get_org_member(self, org_id: int, user_id: int):
        return None

    async def list_org_memberships_for_user(self, user_id: int):
        return [{"org_id": 11, "status": "active"}]


class _FakeRepoMemberHeader:
    def __init__(self, *args, **kwargs) -> None:
        pass

    async def get_org_member(self, org_id: int, user_id: int):
        if org_id == 999:
            return {"org_id": org_id, "user_id": user_id, "status": "active"}
        return None

    async def list_org_memberships_for_user(self, user_id: int):
        return [{"org_id": 42, "status": "active"}]


class _FakeRepoInactiveMember:
    def __init__(self, *args, **kwargs) -> None:
        pass

    async def get_org_member(self, org_id: int, user_id: int):
        return {"org_id": org_id, "user_id": user_id, "status": "inactive"}

    async def list_org_memberships_for_user(self, user_id: int):
        return [{"org_id": 5, "status": "inactive"}]


class _FakeRepoNoMembership:
    def __init__(self, *args, **kwargs) -> None:
        pass

    async def get_org_member(self, org_id: int, user_id: int):
        return None

    async def list_org_memberships_for_user(self, user_id: int):
        return []


@pytest.mark.asyncio
async def test_resolve_org_id_rejects_header_for_non_member(monkeypatch) -> None:
    """Non-members should not be able to force org selection via header."""

    async def _fake_get_db_pool():
        return object()

    monkeypatch.setattr(billing_deps, "get_db_pool", _fake_get_db_pool, raising=False)
    monkeypatch.setattr(billing_deps, "AuthnzOrgsTeamsRepo", _FakeRepoIgnoreHeader, raising=False)

    principal = AuthPrincipal(kind="user", user_id=7, is_admin=False)
    with pytest.raises(HTTPException) as exc_info:
        await billing_deps._resolve_org_id(principal, x_tldw_org_id=999)

    assert exc_info.value.status_code == 403


@pytest.mark.asyncio
async def test_resolve_org_id_allows_header_for_admin(monkeypatch) -> None:
    """Admins should be allowed to specify org via header."""

    async def _fake_get_db_pool():
        return object()

    monkeypatch.setattr(billing_deps, "get_db_pool", _fake_get_db_pool, raising=False)
    monkeypatch.setattr(billing_deps, "AuthnzOrgsTeamsRepo", _FakeRepoAdminHeader, raising=False)

    principal = AuthPrincipal(kind="user", user_id=8, is_admin=False, roles=["admin"], permissions=[])
    org_id = await billing_deps._resolve_org_id(principal, x_tldw_org_id=999)

    assert org_id == 999


@pytest.mark.asyncio
async def test_resolve_org_id_rejects_header_for_boolean_admin_without_claims(monkeypatch) -> None:
    async def _fake_get_db_pool():
        return object()

    monkeypatch.setattr(billing_deps, "get_db_pool", _fake_get_db_pool, raising=False)
    monkeypatch.setattr(billing_deps, "AuthnzOrgsTeamsRepo", _FakeRepoIgnoreHeader, raising=False)

    principal = AuthPrincipal(kind="user", user_id=8, is_admin=True, roles=["user"], permissions=[])
    with pytest.raises(HTTPException) as exc_info:
        await billing_deps._resolve_org_id(principal, x_tldw_org_id=999)

    assert exc_info.value.status_code == 403


@pytest.mark.asyncio
async def test_resolve_org_id_allows_header_for_member(monkeypatch) -> None:
    """Members should be able to select their org via header."""

    async def _fake_get_db_pool():
        return object()

    monkeypatch.setattr(billing_deps, "get_db_pool", _fake_get_db_pool, raising=False)
    monkeypatch.setattr(billing_deps, "AuthnzOrgsTeamsRepo", _FakeRepoMemberHeader, raising=False)

    principal = AuthPrincipal(kind="user", user_id=9, is_admin=False)
    org_id = await billing_deps._resolve_org_id(principal, x_tldw_org_id=999)

    assert org_id == 999


@pytest.mark.asyncio
async def test_resolve_org_id_allows_query_param_for_member(monkeypatch) -> None:
    """Members should be able to select their org via query param."""

    async def _fake_get_db_pool():
        return object()

    monkeypatch.setattr(billing_deps, "get_db_pool", _fake_get_db_pool, raising=False)
    monkeypatch.setattr(billing_deps, "AuthnzOrgsTeamsRepo", _FakeRepoMemberHeader, raising=False)

    principal = AuthPrincipal(kind="user", user_id=10, is_admin=False)
    org_id = await billing_deps._resolve_org_id(principal, org_id=999)

    assert org_id == 999


@pytest.mark.asyncio
async def test_resolve_org_id_rejects_inactive_membership(monkeypatch) -> None:
    """Inactive org memberships should not be considered valid access."""

    async def _fake_get_db_pool():
        return object()

    monkeypatch.setattr(billing_deps, "get_db_pool", _fake_get_db_pool, raising=False)
    monkeypatch.setattr(billing_deps, "AuthnzOrgsTeamsRepo", _FakeRepoInactiveMember, raising=False)

    principal = AuthPrincipal(kind="user", user_id=11, is_admin=False)
    with pytest.raises(HTTPException) as exc_info:
        await billing_deps._resolve_org_id(principal, x_tldw_org_id=5)

    assert exc_info.value.status_code == 403


@pytest.mark.asyncio
async def test_resolve_org_id_rejects_when_no_active_memberships(monkeypatch) -> None:
    """Fallback selection should reject when user has memberships but none are active."""

    async def _fake_get_db_pool():
        return object()

    monkeypatch.setattr(billing_deps, "get_db_pool", _fake_get_db_pool, raising=False)
    monkeypatch.setattr(billing_deps, "AuthnzOrgsTeamsRepo", _FakeRepoInactiveMember, raising=False)

    principal = AuthPrincipal(kind="user", user_id=12, is_admin=False)
    with pytest.raises(HTTPException) as exc_info:
        await billing_deps._resolve_org_id(principal)

    assert exc_info.value.status_code == 403


@pytest.mark.asyncio
async def test_resolve_org_id_fails_closed_on_repo_errors(monkeypatch) -> None:
    """Unexpected resolver errors should return 503 instead of silently bypassing enforcement."""

    async def _fake_get_db_pool():
        raise RuntimeError("db unavailable")

    monkeypatch.setattr(billing_deps, "get_db_pool", _fake_get_db_pool, raising=False)

    principal = AuthPrincipal(kind="user", user_id=13, is_admin=False)
    with pytest.raises(HTTPException) as exc_info:
        await billing_deps._resolve_org_id(principal)

    assert exc_info.value.status_code == 503


@pytest.mark.asyncio
async def test_resolve_org_id_failure_log_omits_backend_details(monkeypatch) -> None:
    """Unexpected resolver errors should not log raw backend details."""
    leaked_secret = "sk-billing-secret"
    leaked_path = "/private/billing/users.db"
    messages: list[str] = []

    async def _fake_get_db_pool():
        raise RuntimeError(f"db unavailable token={leaked_secret} path={leaked_path}")

    monkeypatch.setattr(billing_deps, "get_db_pool", _fake_get_db_pool, raising=False)
    monkeypatch.setattr(billing_deps.logger, "error", messages.append)

    principal = AuthPrincipal(kind="user", user_id=13, is_admin=False)
    with pytest.raises(HTTPException) as exc_info:
        await billing_deps._resolve_org_id(principal)

    assert exc_info.value.status_code == 503
    assert exc_info.value.detail == "Unable to resolve organization for billing enforcement"
    assert messages == ["Failed to resolve org_id for billing enforcement"]
    joined = "\n".join(messages)
    assert leaked_secret not in joined
    assert leaked_path not in joined
    assert "db unavailable" not in joined
    assert "13" not in joined


@pytest.mark.asyncio
async def test_resolve_org_id_returns_none_when_user_has_no_memberships(monkeypatch) -> None:
    """Single-user/no-org contexts should still resolve to None."""

    async def _fake_get_db_pool():
        return object()

    monkeypatch.setattr(billing_deps, "get_db_pool", _fake_get_db_pool, raising=False)
    monkeypatch.setattr(billing_deps, "AuthnzOrgsTeamsRepo", _FakeRepoNoMembership, raising=False)

    principal = AuthPrincipal(kind="user", user_id=14, is_admin=False)
    org_id = await billing_deps._resolve_org_id(principal)

    assert org_id is None


@pytest.mark.asyncio
async def test_require_within_limit_rejects_orgless_multi_user(monkeypatch) -> None:
    """Limit enforcement should fail closed without org context in multi-user mode."""

    async def _fake_resolve_org_id(principal, org_id=None, x_tldw_org_id=None):
        return None

    monkeypatch.setattr(billing_deps, "_resolve_org_id", _fake_resolve_org_id, raising=False)
    monkeypatch.setattr(billing_deps, "enforcement_enabled", lambda: True, raising=False)
    monkeypatch.setattr(billing_deps, "_allow_orgless_billing_access", lambda: False, raising=False)

    dependency = billing_deps.require_within_limit(LimitCategory.API_CALLS_DAY, units=1)
    principal = AuthPrincipal(kind="user", user_id=21, is_admin=False)

    with pytest.raises(HTTPException) as exc_info:
        await dependency(response=Response(), principal=principal, x_tldw_org_id=None, org_id=None)

    assert exc_info.value.status_code == 403


@pytest.mark.asyncio
async def test_require_within_limit_allows_orgless_single_user(monkeypatch) -> None:
    """Limit enforcement should remain permissive without org context in single-user mode."""

    async def _fake_resolve_org_id(principal, org_id=None, x_tldw_org_id=None):
        return None

    monkeypatch.setattr(billing_deps, "_resolve_org_id", _fake_resolve_org_id, raising=False)
    monkeypatch.setattr(billing_deps, "enforcement_enabled", lambda: True, raising=False)
    monkeypatch.setattr(billing_deps, "_allow_orgless_billing_access", lambda: True, raising=False)

    dependency = billing_deps.require_within_limit(LimitCategory.API_CALLS_DAY, units=1)
    principal = AuthPrincipal(kind="user", user_id=22, is_admin=False)
    result = await dependency(response=Response(), principal=principal, x_tldw_org_id=None, org_id=None)

    assert result.unlimited is True
    assert result.action.value == "allow"


@pytest.mark.asyncio
async def test_require_feature_rejects_orgless_multi_user(monkeypatch) -> None:
    """Feature checks should fail closed without org context in multi-user mode."""

    async def _fake_resolve_org_id(principal, org_id=None, x_tldw_org_id=None):
        return None

    monkeypatch.setattr(billing_deps, "_resolve_org_id", _fake_resolve_org_id, raising=False)
    monkeypatch.setattr(billing_deps, "enforcement_enabled", lambda: True, raising=False)
    monkeypatch.setattr(billing_deps, "_allow_orgless_billing_access", lambda: False, raising=False)

    dependency = billing_deps.require_feature("advanced_analytics")
    principal = AuthPrincipal(kind="user", user_id=23, is_admin=False)

    with pytest.raises(HTTPException) as exc_info:
        await dependency(principal=principal, x_tldw_org_id=None, org_id=None)

    assert exc_info.value.status_code == 403
