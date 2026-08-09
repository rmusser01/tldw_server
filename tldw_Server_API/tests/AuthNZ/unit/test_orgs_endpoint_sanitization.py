from types import SimpleNamespace

import pytest


class _LoggerStub:
    def __init__(self) -> None:
        self.debugs: list[str] = []
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def debug(self, message: str, *args, **kwargs) -> None:
        if args or kwargs:
            message = message.format(*args, **kwargs)
        self.debugs.append(message)

    def error(self, message: str, *args, **kwargs) -> None:
        if args or kwargs:
            message = message.format(*args, **kwargs)
        self.errors.append(message)

    def warning(self, message: str, *args, **kwargs) -> None:
        if args or kwargs:
            message = message.format(*args, **kwargs)
        self.warnings.append(message)

    def info(self, message: str, *args, **kwargs) -> None:
        pass


def _request(path: str, method: str) -> SimpleNamespace:
    return SimpleNamespace(
        headers={},
        state=SimpleNamespace(),
        client=None,
        url=SimpleNamespace(path=path),
        method=method,
    )


def test_membership_context_uses_platform_authority_for_platform_org_context():
    from tldw_Server_API.app.api.v1.API_Deps.org_deps import OrgContext
    from tldw_Server_API.app.api.v1.endpoints import orgs
    from tldw_Server_API.app.core.AuthNZ.membership_writer import MembershipAuthority

    principal = SimpleNamespace(user_id=17)

    scoped = orgs._membership_context(  # noqa: SLF001
        principal,
        OrgContext(org_id=9, role="admin"),
    )
    platform = orgs._membership_context(  # noqa: SLF001
        principal,
        OrgContext(org_id=9, role="admin", is_platform_admin=True),
    )

    assert scoped.required_authority is MembershipAuthority.SCOPED_MEMBERSHIP
    assert platform.required_authority is MembershipAuthority.PLATFORM_ADMIN


@pytest.mark.asyncio
async def test_platform_admin_membership_route_requests_persisted_platform_authority(
    monkeypatch,
):
    from tldw_Server_API.app.api.v1.API_Deps.org_deps import OrgContext
    from tldw_Server_API.app.api.v1.endpoints import orgs
    from tldw_Server_API.app.api.v1.schemas.org_team_schemas import OrgMemberAddRequest
    from tldw_Server_API.app.core.AuthNZ.membership_writer import MembershipAuthority

    captured: dict[str, object] = {}

    class _Repo:
        def __init__(self, db_pool) -> None:
            captured["db_pool"] = db_pool

        async def add_org_member(self, **kwargs):
            captured.update(kwargs)
            return {
                "org_id": kwargs["org_id"],
                "user_id": kwargs["user_id"],
                "role": kwargs["role"],
            }

    async def _pool():
        return object()

    monkeypatch.setattr(orgs, "get_db_pool", _pool)
    monkeypatch.setattr(orgs, "AuthnzOrgsTeamsRepo", _Repo)

    response = await orgs.add_org_member(
        body=OrgMemberAddRequest(user_id=23, role="member"),
        ctx=OrgContext(org_id=9, role="admin", is_platform_admin=True),
        principal=SimpleNamespace(user_id=17),
    )

    assert response.org_id == 9
    assert captured["context"].required_authority is MembershipAuthority.PLATFORM_ADMIN


def test_membership_route_openapi_contract_has_no_new_parameters():
    from fastapi import FastAPI

    from tldw_Server_API.app.api.v1.endpoints.orgs import router

    app = FastAPI()
    app.include_router(router)
    paths = app.openapi()["paths"]
    expected = {
        ("post", "/orgs/{org_id}/members"): (["org_id"], True),
        ("patch", "/orgs/{org_id}/members/{user_id}"): (
            ["user_id", "org_id"],
            True,
        ),
        ("delete", "/orgs/{org_id}/members/{user_id}"): (
            ["user_id", "org_id"],
            False,
        ),
        ("post", "/orgs/{org_id}/teams/{team_id}/members"): (
            ["team_id", "org_id"],
            True,
        ),
        ("delete", "/orgs/{org_id}/teams/{team_id}/members/{user_id}"): (
            ["team_id", "user_id", "org_id"],
            False,
        ),
    }

    for (method, path), (parameter_names, has_body) in expected.items():
        operation = paths[path][method]
        assert [item["name"] for item in operation.get("parameters", [])] == parameter_names
        assert ("requestBody" in operation) is has_body


@pytest.mark.asyncio
async def test_update_org_budgets_upsert_failure_log_is_sanitized(monkeypatch):
    from fastapi import HTTPException

    from tldw_Server_API.app.api.v1.API_Deps.org_deps import OrgContext
    from tldw_Server_API.app.api.v1.endpoints import orgs
    from tldw_Server_API.app.api.v1.schemas.admin_schemas import OrgBudgetSelfUpdateRequest

    async def _failing_upsert_org_budget(*_args, **_kwargs):
        raise RuntimeError("budget upsert exploded at /private/org-budget.db")

    logger_stub = _LoggerStub()
    monkeypatch.setattr(orgs, "svc_upsert_org_budget", _failing_upsert_org_budget)
    monkeypatch.setattr(orgs, "logger", logger_stub)

    with pytest.raises(HTTPException) as exc_info:
        await orgs.update_org_budgets(
            payload=OrgBudgetSelfUpdateRequest(clear_budgets=True),
            request=object(),
            ctx=OrgContext(org_id=9, role="admin"),
            principal=object(),
            db=object(),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to upsert org budget"
    assert logger_stub.errors == ["Failed to upsert org budget"]
    assert "budget upsert exploded" not in str(logger_stub.errors)
    assert "/private/org-budget.db" not in str(logger_stub.errors)


@pytest.mark.asyncio
async def test_update_org_budgets_audit_failure_log_is_sanitized(monkeypatch):
    from fastapi import HTTPException

    from tldw_Server_API.app.api.v1.API_Deps.org_deps import OrgContext
    from tldw_Server_API.app.api.v1.endpoints import orgs
    from tldw_Server_API.app.api.v1.schemas.admin_schemas import OrgBudgetSelfUpdateRequest

    async def _successful_upsert_org_budget(*_args, **_kwargs):
        return (
            {
                "org_id": 9,
                "org_name": "Example",
                "org_slug": "example",
                "plan_name": "free",
                "plan_display_name": "Free",
                "budgets": {},
                "custom_limits": {},
                "effective_limits": {},
            },
            {"budgets": "changed"},
        )

    async def _failing_emit_budget_audit_event(*_args, **_kwargs):
        raise RuntimeError("budget audit exploded at /private/org-audit.db")

    logger_stub = _LoggerStub()
    monkeypatch.setattr(orgs, "svc_upsert_org_budget", _successful_upsert_org_budget)
    monkeypatch.setattr(orgs, "emit_budget_audit_event", _failing_emit_budget_audit_event)
    monkeypatch.setattr(orgs, "logger", logger_stub)

    with pytest.raises(HTTPException) as exc_info:
        await orgs.update_org_budgets(
            payload=OrgBudgetSelfUpdateRequest(clear_budgets=True),
            request=object(),
            ctx=OrgContext(org_id=9, role="admin"),
            principal=object(),
            db=object(),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "audit_failed"
    assert logger_stub.errors == ["Budget audit failed"]
    assert "budget audit exploded" not in str(logger_stub.errors)
    assert "/private/org-audit.db" not in str(logger_stub.errors)


@pytest.mark.asyncio
async def test_delete_org_subscription_lookup_warning_log_is_sanitized(monkeypatch):
    from tldw_Server_API.app.api.v1.API_Deps.org_deps import OrgContext
    from tldw_Server_API.app.api.v1.endpoints import orgs

    class _FailingSubscriptionService:
        async def get_subscription(self, org_id: int):  # noqa: ARG002
            raise RuntimeError("subscription backend exploded at /private/subscriptions.db")

    class _FakeOrgsRepo:
        deleted_org_id: int | None = None

        def __init__(self, db_pool) -> None:  # noqa: ARG002
            pass

        async def delete_organization_with_provider_secrets(self, org_id: int) -> None:
            type(self).deleted_org_id = org_id

    async def _fake_get_subscription_service():
        return _FailingSubscriptionService()

    async def _fake_get_db_pool():
        return object()

    logger_stub = _LoggerStub()
    monkeypatch.setattr(orgs, "get_subscription_service", _fake_get_subscription_service)
    monkeypatch.setattr(orgs, "get_db_pool", _fake_get_db_pool)
    monkeypatch.setattr(orgs, "AuthnzOrgsTeamsRepo", _FakeOrgsRepo)
    monkeypatch.setattr(orgs, "logger", logger_stub)

    response = await orgs.delete_org(ctx=OrgContext(org_id=9, role="owner"))

    assert response.status_code == 204
    assert _FakeOrgsRepo.deleted_org_id == 9
    assert logger_stub.warnings == ["delete_org: failed to load subscription"]
    assert "subscription backend exploded" not in str(logger_stub.warnings)
    assert "/private/subscriptions.db" not in str(logger_stub.warnings)
    assert "org 9" not in str(logger_stub.warnings)


@pytest.mark.asyncio
async def test_create_invite_audit_failure_log_is_sanitized(monkeypatch):
    from tldw_Server_API.app.api.v1.API_Deps.org_deps import OrgContext
    from tldw_Server_API.app.api.v1.endpoints import orgs
    from tldw_Server_API.app.api.v1.schemas.org_team_schemas import OrgInviteCreateRequest

    class _InviteService:
        async def create_invite(self, **kwargs):
            assert kwargs["org_id"] == 9
            return {
                "id": 42,
                "code": "secret-code-123",
                "org_id": 9,
                "org_name": "Example",
                "team_id": None,
                "team_name": None,
                "role_to_grant": "member",
                "max_uses": 1,
                "uses_count": 0,
                "is_active": True,
                "expires_at": None,
                "created_at": None,
                "created_by": 7,
                "description": None,
                "allowed_email_domain": None,
            }

    async def _fake_get_invite_service():
        return _InviteService()

    async def _raise_audit_service(_user_id):
        raise RuntimeError("org invite audit exploded at /private/org-audit.db")

    logger_stub = _LoggerStub()
    monkeypatch.setattr(orgs, "get_invite_service", _fake_get_invite_service)
    monkeypatch.setattr(orgs, "get_or_create_audit_service_for_user_id", _raise_audit_service)
    monkeypatch.setattr(orgs, "logger", logger_stub)

    response = await orgs.create_invite(
        body=OrgInviteCreateRequest(),
        http_request=_request(path="/api/v1/orgs/9/invites", method="POST"),
        ctx=OrgContext(org_id=9, role="admin"),
        principal=SimpleNamespace(user_id=7),
    )

    assert response.id == 42
    assert logger_stub.debugs == ["Org invite audit failed"]
    assert "org invite audit exploded" not in str(logger_stub.debugs)
    assert "/private/org-audit.db" not in str(logger_stub.debugs)


@pytest.mark.asyncio
async def test_revoke_invite_audit_failure_log_is_sanitized(monkeypatch):
    from tldw_Server_API.app.api.v1.API_Deps.org_deps import OrgContext
    from tldw_Server_API.app.api.v1.endpoints import orgs

    class _InviteService:
        async def revoke_invite(self, invite_id, org_id):
            assert invite_id == 42
            assert org_id == 9
            return True

    async def _fake_get_invite_service():
        return _InviteService()

    async def _raise_audit_service(_user_id):
        raise RuntimeError("org invite revoke audit exploded at /private/org-audit.db")

    logger_stub = _LoggerStub()
    monkeypatch.setattr(orgs, "get_invite_service", _fake_get_invite_service)
    monkeypatch.setattr(orgs, "get_or_create_audit_service_for_user_id", _raise_audit_service)
    monkeypatch.setattr(orgs, "logger", logger_stub)

    response = await orgs.revoke_invite(
        invite_id=42,
        http_request=_request(path="/api/v1/orgs/9/invites/42", method="DELETE"),
        ctx=OrgContext(org_id=9, role="admin"),
        principal=SimpleNamespace(user_id=7),
    )

    assert response.status_code == 204
    assert logger_stub.debugs == ["Org invite audit failed"]
    assert "org invite revoke audit exploded" not in str(logger_stub.debugs)
    assert "/private/org-audit.db" not in str(logger_stub.debugs)


@pytest.mark.asyncio
async def test_list_invites_includes_canonical_pagination(monkeypatch):
    from tldw_Server_API.app.api.v1.API_Deps.org_deps import OrgContext
    from tldw_Server_API.app.api.v1.endpoints import orgs

    class _InviteService:
        async def list_org_invites(self, org_id, **kwargs):
            assert org_id == 9
            assert kwargs["include_expired"] is False
            assert kwargs["include_inactive"] is False
            assert kwargs["limit"] == 1
            assert kwargs["offset"] == 0
            return (
                [
                    {
                        "id": 42,
                        "code": "secret-code-123",
                        "org_id": 9,
                        "org_name": "Example",
                        "team_id": None,
                        "team_name": None,
                        "role_to_grant": "member",
                        "max_uses": 1,
                        "uses_count": 0,
                        "is_active": True,
                        "expires_at": None,
                        "created_at": None,
                        "created_by": 7,
                        "description": None,
                        "allowed_email_domain": None,
                    }
                ],
                2,
            )

    async def _fake_get_invite_service():
        return _InviteService()

    monkeypatch.setattr(orgs, "get_invite_service", _fake_get_invite_service)

    response = await orgs.list_invites(
        ctx=OrgContext(org_id=9, role="admin"),
        include_expired=False,
        include_inactive=False,
        limit=1,
        offset=0,
    )

    assert response.total == 2
    assert response.limit == 1
    assert response.offset == 0
    assert response.pagination.model_dump() == {
        "mode": "offset",
        "limit": 1,
        "offset": 0,
        "total": 2,
        "has_more": True,
        "next_offset": 1,
    }
    assert response.has_more is True
    assert response.next_offset == 1
    assert len(response.items) == 1
    assert response.items[0].id == 42


@pytest.mark.asyncio
async def test_list_my_orgs_includes_canonical_pagination(monkeypatch):
    from tldw_Server_API.app.api.v1.endpoints import orgs

    class _Repo:
        def __init__(self, db_pool) -> None:  # noqa: ARG002
            pass

        async def list_organizations_for_user(self, user_id, **kwargs):
            assert user_id == 7
            assert kwargs["with_total"] is True
            assert kwargs["limit"] == 1
            assert kwargs["offset"] == 0
            return (
                [
                    {"id": 11, "name": "Alpha", "slug": "alpha", "owner_user_id": 7},
                ],
                2,
            )

    async def _fake_get_db_pool():
        return object()

    monkeypatch.setattr(orgs, "get_db_pool", _fake_get_db_pool)
    monkeypatch.setattr(orgs, "AuthnzOrgsTeamsRepo", _Repo)

    response = await orgs.list_my_orgs(
        principal=SimpleNamespace(user_id=7),
        limit=1,
        offset=0,
    )

    assert response.total == 2
    assert response.limit == 1
    assert response.offset == 0
    assert response.has_more is True
    assert response.next_offset == 1
    assert response.pagination.model_dump() == {
        "mode": "offset",
        "limit": 1,
        "offset": 0,
        "total": 2,
        "has_more": True,
        "next_offset": 1,
    }
    assert len(response.items) == 1
    assert response.items[0].id == 11


@pytest.mark.asyncio
async def test_accept_invite_audit_failure_log_is_sanitized(monkeypatch):
    from tldw_Server_API.app.api.v1.endpoints import orgs
    from tldw_Server_API.app.api.v1.schemas.org_team_schemas import OrgInviteAcceptRequest

    class _RegistrationService:
        async def accept_org_invite_code(self, *, code, user_id):
            assert code == "secret-code"
            assert user_id == 7
            return {
                "registration_code_id": 123,
                "org_id": 9,
                "team_id": None,
                "org_role": "member",
                "was_already_member": False,
            }

    async def _raise_audit_service(_user_id):
        raise RuntimeError("org invite accept audit exploded at /private/org-audit.db")

    logger_stub = _LoggerStub()
    monkeypatch.setattr(orgs, "get_or_create_audit_service_for_user_id", _raise_audit_service)
    monkeypatch.setattr(orgs, "logger", logger_stub)

    response = await orgs.accept_org_invite(
        body=OrgInviteAcceptRequest(code="secret-code"),
        http_request=_request(path="/api/v1/orgs/invites/accept", method="POST"),
        principal=SimpleNamespace(user_id=7),
        registration_service=_RegistrationService(),
    )

    assert response.success is True
    assert response.org_id == 9
    assert logger_stub.debugs == ["Org invite audit failed"]
    assert "org invite accept audit exploded" not in str(logger_stub.debugs)
    assert "/private/org-audit.db" not in str(logger_stub.debugs)


@pytest.mark.asyncio
async def test_redeem_invite_audit_failure_log_is_sanitized(monkeypatch):
    from tldw_Server_API.app.api.v1.endpoints import org_invites
    from tldw_Server_API.app.api.v1.schemas.org_team_schemas import OrgInviteRedeemRequest

    class _InviteService:
        async def redeem_invite(self, **kwargs):
            assert kwargs["code"] == "secret-code-123"
            assert kwargs["user_id"] == 7
            return SimpleNamespace(
                success=True,
                was_already_member=False,
                org_id=9,
                org_name="Example",
                team_id=None,
                team_name=None,
                role="member",
                message="joined",
                invite_id=42,
            )

    async def _fake_get_invite_service():
        return _InviteService()

    async def _fake_fetch_active_user_by_id(_db, user_id):
        assert user_id == 7
        return {"email": "user@example.test"}

    async def _raise_audit_service(_user_id):
        raise RuntimeError("org invite redeem audit exploded at /private/org-audit.db")

    logger_stub = _LoggerStub()
    monkeypatch.setattr(org_invites, "get_invite_service", _fake_get_invite_service)
    monkeypatch.setattr(org_invites, "fetch_active_user_by_id", _fake_fetch_active_user_by_id)
    monkeypatch.setattr(org_invites, "get_or_create_audit_service_for_user_id", _raise_audit_service)
    monkeypatch.setattr(org_invites, "logger", logger_stub)

    response = await org_invites.redeem_invite(
        body=OrgInviteRedeemRequest(code="secret-code-123"),
        request=_request(path="/api/v1/invites/redeem", method="POST"),
        principal=SimpleNamespace(user_id=7),
        db=object(),
    )

    assert response.success is True
    assert response.org_id == 9
    assert logger_stub.debugs == ["Org invite audit failed"]
    assert "org invite redeem audit exploded" not in str(logger_stub.debugs)
    assert "/private/org-audit.db" not in str(logger_stub.debugs)
