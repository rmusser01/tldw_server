"""Endpoint outcomes must agree with committed membership/profile state."""

import secrets
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import HTTPException, Response
from loguru import logger

from tldw_Server_API.app.api.v1.API_Deps.org_deps import OrgContext
from tldw_Server_API.app.api.v1.endpoints import auth, org_invites, orgs
from tldw_Server_API.app.api.v1.endpoints.admin import admin_tenant_provisioning
from tldw_Server_API.app.api.v1.schemas.auth_schemas import RegisterRequest
from tldw_Server_API.app.api.v1.schemas.org_team_schemas import (
    OrgInviteAcceptRequest,
    OrgInviteRedeemRequest,
    OrgMemberAddRequest,
)
from tldw_Server_API.app.api.v1.schemas.user_profile_schemas import UserProfileBulkUpdateRequest
from tldw_Server_API.app.core.AuthNZ import database
from tldw_Server_API.app.core.AuthNZ.exceptions import DatabaseConcurrencyConflict
from tldw_Server_API.app.core.AuthNZ.membership_writer import (
    TrustedMembershipReason,
    TrustedMembershipWriteContext,
)
from tldw_Server_API.app.core.AuthNZ.migrations import ensure_authnz_tables
from tldw_Server_API.app.core.AuthNZ.password_service import get_password_service
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.AuthNZ.repos.org_invites_repo import AuthnzOrgInvitesRepo
from tldw_Server_API.app.core.AuthNZ.repos.orgs_teams_repo import AuthnzOrgsTeamsRepo
from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
from tldw_Server_API.app.core.exceptions import RollbackSignal
from tldw_Server_API.app.services import admin_profiles_service
from tldw_Server_API.app.services.org_invite_service import OrgInviteService
from tldw_Server_API.app.services.registration_service import RegistrationService
from tldw_Server_API.tests.AuthNZ_SQLite._user_fixtures import create_authnz_test_user

pytestmark = [pytest.mark.asyncio, pytest.mark.integration]
_BOOTSTRAP = TrustedMembershipWriteContext(trusted_reason=TrustedMembershipReason.BOOTSTRAP)


@pytest.fixture
def failure_logs():
    """Observe public-path logs without replacing their sanitizing code."""
    messages: list[str] = []
    sink = logger.add(lambda message: messages.append(str(message)))
    try:
        yield messages
    finally:
        logger.remove(sink)


@pytest.fixture
async def records(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[tuple[Any, ...]]:
    """Create an isolated migrated SQLite database and real invite records."""
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    monkeypatch.setenv("JWT_SECRET_KEY", secrets.token_hex(32))
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'users.db'}")
    monkeypatch.setenv("AUTHNZ_DB_POOL_ACQUIRE_TIMEOUT_SECONDS", "13")
    monkeypatch.setenv("AUTHNZ_SQLITE_LOCK_RETRY_AFTER_SECONDS", "17")
    reset_settings()
    await database.reset_db_pool()
    pool = await database.get_db_pool()
    ensure_authnz_tables(Path(pool.db_path))
    try:
        ids = {}
        for name in ("owner", "first", "second"):
            ids[name] = await create_authnz_test_user(
                pool,
                username=name,
                email=f"{name}@example.test",
                role="admin" if name == "owner" else "user",
            )
        repo = AuthnzOrgsTeamsRepo(pool)
        org = await repo.create_organization_with_owner_membership(
            name="Inviting organization",
            owner_user_id=ids["owner"],
            context=_BOOTSTRAP,
        )
        ids["org"] = org["id"]
        team = await repo.create_team(org_id=org["id"], name="Invite team")
        ids["team"] = team["id"]
        await pool.execute(
            "INSERT INTO registration_codes "
            "(code, max_uses, expires_at, created_by, org_id, org_role, team_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                "registration-invite",
                5,
                (datetime.now() + timedelta(days=1)).isoformat(),
                ids["owner"],
                ids["org"],
                "member",
                ids["team"],
            ),
        )
        invite = await AuthnzOrgInvitesRepo(pool).create_invite(
            org_id=ids["org"],
            created_by=ids["owner"],
            team_id=ids["team"],
        )
        yield pool, repo, ids, invite
    finally:
        await database.reset_db_pool()
        reset_settings()


async def _snapshot(pool: database.DatabasePool) -> list[list[dict[str, Any]]]:
    """Include anchors, bookkeeping and dependent rows in rollback assertions."""
    queries = (
        "SELECT * FROM users ORDER BY id",
        "SELECT * FROM organizations ORDER BY id",
        "SELECT * FROM teams ORDER BY id",
        "SELECT * FROM org_members ORDER BY org_id, user_id",
        "SELECT * FROM team_members ORDER BY team_id, user_id",
        "SELECT * FROM registration_codes ORDER BY id",
        "SELECT * FROM org_invites ORDER BY id",
        "SELECT * FROM org_invite_redemptions ORDER BY id",
        "SELECT * FROM user_config_overrides ORDER BY user_id, key",
        "SELECT * FROM user_roles ORDER BY user_id, role_id",
        "SELECT * FROM password_history ORDER BY id",
        "SELECT * FROM audit_logs ORDER BY id",
    )
    return [[dict(row) for row in await pool.fetchall(query)] for query in queries]


def _principal(user_id: int) -> AuthPrincipal:
    return AuthPrincipal(kind="user", user_id=user_id, roles=["admin"], is_admin=True)


def _request() -> SimpleNamespace:
    return SimpleNamespace(
        client=None,
        headers={},
        state=SimpleNamespace(),
        method="POST",
        url=SimpleNamespace(path="/test"),
    )


def _registration(pool: Any, monkeypatch: pytest.MonkeyPatch) -> RegistrationService:
    service = RegistrationService(
        db_pool=pool,
        settings=SimpleNamespace(
            ENABLE_REGISTRATION=True,
            REQUIRE_REGISTRATION_CODE=False,
            ENABLE_ORG_SCOPED_REGISTRATION_CODES=True,
            DEFAULT_USER_ROLE="user",
            DEFAULT_STORAGE_QUOTA_MB=100,
        ),
        password_service=SimpleNamespace(
            validate_password_strength=lambda *_args: None,
            hash_password=lambda _value: "test-hash",
        ),
    )
    # Avoid filesystem work and hashing; all database writes stay real.
    monkeypatch.setattr(service, "_create_user_directories", lambda _id: True)
    monkeypatch.setattr(service, "_cleanup_user_directories", lambda _id: None)
    return service


class _ContentionPool:
    """Model a translated transaction failure after real SQLite rollback.

    PostgreSQL produces DatabaseConcurrencyConflict at the pool boundary, not
    inside a SQLite transaction. Use its rollback control before publishing
    the equivalent translated outcome. Unbounded schema-readiness transactions
    are not the bounded writer acquisition being exercised.
    """

    def __init__(
        self,
        delegate: database.DatabasePool,
        failure: Exception,
        *,
        timing: str = "exit",
        fail_at: int = 1,
    ) -> None:
        self.delegate = delegate
        self.failure = failure
        self.timing = timing
        self.fail_at = fail_at
        self.timeouts = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self.delegate, name)

    @asynccontextmanager
    async def transaction(self, *, acquire_timeout_seconds: float | None = None) -> AsyncIterator[Any]:
        if acquire_timeout_seconds is None:
            async with self.delegate.transaction() as conn:
                yield conn
            return
        self.timeouts.append(acquire_timeout_seconds)
        fail = len(self.timeouts) == self.fail_at
        if fail and self.timing == "enter":
            raise self.failure
        try:
            async with self.delegate.transaction(acquire_timeout_seconds=acquire_timeout_seconds) as conn:
                yield conn
                if fail:
                    raise RollbackSignal()
        except RollbackSignal:
            raise self.failure from None


async def _unexpected_audit(*_args: Any, **_kwargs: Any) -> None:
    pytest.fail("Rolled-back operations must not emit a success audit")


@pytest.mark.parametrize("invalid_scope", ["inactive", "wrong_parent", "missing_target"])
async def test_invalid_acceptance_rolls_back_code_usage_and_memberships(
    records,
    monkeypatch: pytest.MonkeyPatch,
    invalid_scope: str,
) -> None:
    pool, repo, ids, _invite = records
    user_id = ids["first"]
    if invalid_scope == "inactive":
        await pool.execute("UPDATE teams SET is_active = 0 WHERE id = ?", (ids["team"],))
    elif invalid_scope == "wrong_parent":
        other_org = await repo.create_organization_with_owner_membership(
            name="Other organization",
            owner_user_id=ids["owner"],
            context=_BOOTSTRAP,
        )
        await pool.execute("UPDATE registration_codes SET org_id = ?", (other_org["id"],))
    else:
        user_id = 1_000_000
    before = await _snapshot(pool)
    monkeypatch.setattr(orgs, "get_or_create_audit_service_for_user_id", _unexpected_audit)

    with pytest.raises(HTTPException) as raised:
        await orgs.accept_org_invite(
            body=OrgInviteAcceptRequest(code="registration-invite"),
            http_request=_request(),
            principal=_principal(user_id),
            registration_service=_registration(pool, monkeypatch),
        )

    assert raised.value.status_code == 400
    assert raised.value.detail == "Invalid invite code"
    assert await _snapshot(pool) == before


@pytest.mark.parametrize("operation", ["register", "accept", "redeem_service", "redeem", "tenant", "org_member"])
@pytest.mark.parametrize("timing", ["enter", "exit"])
@pytest.mark.parametrize("failure_type", [DatabaseConcurrencyConflict, TimeoutError])
async def test_contention_has_retryable_outcome_without_hidden_writes(
    records,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    timing: str,
    failure_type: type[Exception],
    failure_logs,
) -> None:
    pool, _repo, ids, invite = records
    failure = (
        TimeoutError("private backend path /private/auth.db")
        if failure_type is TimeoutError
        else DatabaseConcurrencyConflict()
    )
    busy_pool = _ContentionPool(pool, failure, timing=timing)
    before = await _snapshot(pool)

    async def _pool():
        return busy_pool

    if operation in {"tenant", "org_member"}:
        pool_owner = database if operation == "tenant" else orgs
        monkeypatch.setattr(pool_owner, "get_db_pool", _pool)
    credential = get_password_service().generate_secure_password(length=32, username="new-tenant")
    audit_modules = {
        "register": (auth,),
        "accept": (orgs,),
        "redeem": (org_invites,),
        "org_member": (orgs,),
    }
    for module in audit_modules.get(operation, ()):
        monkeypatch.setattr(module, "get_or_create_audit_service_for_user_id", _unexpected_audit)
    service = OrgInviteService(
        db_pool=busy_pool,
        invites_repo=AuthnzOrgInvitesRepo(pool),
        orgs_repo=AuthnzOrgsTeamsRepo(busy_pool),
    )

    async def _invite_service():
        return service

    if operation == "redeem":
        monkeypatch.setattr(org_invites, "get_invite_service", _invite_service)
    expected_error = failure_type if operation == "redeem_service" else HTTPException
    with pytest.raises(expected_error) as raised:
        if operation == "register":
            await auth.register(
                payload=RegisterRequest(username="new-user", email="new@example.com", password=credential),
                http_request=_request(),
                response=Response(),
                _diag=None,
                registration_service=_registration(busy_pool, monkeypatch),
            )
        elif operation == "accept":
            await orgs.accept_org_invite(
                body=OrgInviteAcceptRequest(code="registration-invite"),
                http_request=_request(),
                principal=_principal(ids["first"]),
                registration_service=_registration(busy_pool, monkeypatch),
            )
        elif operation == "redeem_service":
            await service.redeem_invite(code=invite["code"], user_id=ids["first"])
        elif operation == "redeem":
            await org_invites.redeem_invite(
                body=OrgInviteRedeemRequest(code=invite["code"]),
                request=_request(),
                principal=_principal(ids["first"]),
                db=pool,
            )
        elif operation == "tenant":
            await admin_tenant_provisioning.provision_tenant(
                payload=admin_tenant_provisioning.TenantProvisionRequest(
                    username="new-tenant",
                    email="tenant@example.test",
                    password=credential,
                    org_name="Tenant",
                ),
                principal=_principal(ids["owner"]),
            )
        else:
            await orgs.add_org_member(
                body=OrgMemberAddRequest(user_id=ids["first"], role="member"),
                ctx=OrgContext(org_id=ids["org"], role="owner"),
                principal=_principal(ids["owner"]),
            )

    if operation == "redeem_service":
        assert raised.value is failure
    else:
        assert raised.value.status_code == 503, repr(raised.value.__cause__)
        assert raised.value.detail == "Authentication database is busy. Please retry shortly."
        assert raised.value.headers == {"Retry-After": "17"}
    assert busy_pool.timeouts == [13.0]
    assert await _snapshot(pool) == before
    assert "/private/auth.db" not in "".join(failure_logs)


@pytest.mark.parametrize("fail_at", [1, 2])
async def test_bulk_timeout_preserves_explicit_partial_success_contract(
    records,
    monkeypatch: pytest.MonkeyPatch,
    fail_at: int,
) -> None:
    pool, repo, ids, _invite = records
    targets = [ids["first"], ids["second"]]
    for user_id in targets:
        await repo.add_org_member(org_id=ids["org"], user_id=user_id, context=_BOOTSTRAP)
    versions = {
        user_id: await pool.fetchval("SELECT profile_version FROM users WHERE id = ?", (user_id,))
        for user_id in targets
    }
    busy_pool = _ContentionPool(
        pool,
        TimeoutError("private backend path /private/auth.db"),
        timing="enter",
        fail_at=fail_at,
    )

    async def _pool():
        return busy_pool

    monkeypatch.setattr(admin_profiles_service, "get_db_pool", _pool)
    response, audit = await admin_profiles_service.bulk_update_user_profiles(
        payload=UserProfileBulkUpdateRequest(
            user_ids=targets,
            confirm=True,
            updates=[{"key": "memberships.teams.member", "value": {"team_id": ids["team"], "action": "add"}}],
        ),
        principal=_principal(ids["owner"]),
    )

    assert (response.total_targets, response.updated, response.failed, response.skipped) == (2, 1, 1, 0)
    assert [result.user_id for result in response.results] == targets
    for index, user_id in enumerate(targets, start=1):
        result = response.results[index - 1]
        version = await pool.fetchval("SELECT profile_version FROM users WHERE id = ?", (user_id,))
        member = await repo.get_team_member(ids["team"], user_id)
        if index == fail_at:
            assert result.error == "update_failed"
            assert result.applied == []
            assert version == versions[user_id]
            assert member is None
        else:
            assert result.error is None
            assert result.applied == ["memberships.teams.member"]
            assert version != versions[user_id]
            assert member["role"] == "member"
    assert busy_pool.timeouts == [13.0, 13.0]
    assert audit["metadata"]["updated"] == 1
    assert audit["metadata"]["failed"] == 1
    assert "private" not in response.model_dump_json()
