"""Profile role validation against real SQLite transactions and membership writes."""

from __future__ import annotations

import json
import secrets
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio

from tldw_Server_API.app.core.AuthNZ.database import get_db_pool, reset_db_pool
from tldw_Server_API.app.core.AuthNZ.membership_writer import (
    TrustedMembershipReason,
    TrustedMembershipWriteContext,
)
from tldw_Server_API.app.core.AuthNZ.migrations import ensure_authnz_tables
from tldw_Server_API.app.core.AuthNZ.profile_version import VersionedUserWriteGateway
from tldw_Server_API.app.core.AuthNZ.repos.orgs_teams_repo import AuthnzOrgsTeamsRepo
from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
from tldw_Server_API.app.core.UserProfiles.command_service import ProfileCommandService
from tldw_Server_API.app.core.UserProfiles.contracts import ProfileUpdateCommand
from tldw_Server_API.app.core.UserProfiles.update_service import (
    ProfileUpdateScope,
    UserProfileUpdateService,
)
from tldw_Server_API.app.core.UserProfiles.version_gateway import ProfileVersionGateway


@pytest_asyncio.fixture
async def membership_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[tuple[Any, AuthnzOrgsTeamsRepo, dict[str, int]]]:
    """Build an isolated owner, target, organization and explicit child team."""
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'profiles.db'}")
    reset_settings()
    await reset_db_pool()
    pool = await get_db_pool()
    ensure_authnz_tables(Path(pool.db_path))
    bootstrap = TrustedMembershipWriteContext(trusted_reason=TrustedMembershipReason.BOOTSTRAP)
    ids: dict[str, int] = {}
    async with pool.transaction() as conn:
        gateway = VersionedUserWriteGateway("sqlite")
        for label in ("actor", "target"):
            result = await gateway.insert_user(
                conn,
                values={
                    "username": label,
                    "email": f"{label}@example.com",
                    "password_hash": secrets.token_hex(32),
                    "is_active": True,
                },
            )
            ids[label] = result.affected_user_ids[0]
    repo = AuthnzOrgsTeamsRepo(pool)
    org = await repo.create_organization_with_owner_membership(
        name="Role validation",
        owner_user_id=ids["actor"],
        context=bootstrap,
    )
    ids["org"] = int(org["id"])
    await repo.add_org_member(org_id=ids["org"], user_id=ids["target"], role="member", context=bootstrap)
    team = await repo.create_team(org_id=ids["org"], name="Role validation team")
    ids["team"] = int(team["id"])
    await repo.add_team_member(team_id=ids["team"], user_id=ids["target"], role="member", context=bootstrap)
    try:
        yield pool, repo, ids
    finally:
        await reset_db_pool()
        reset_settings()


@pytest.mark.asyncio
@pytest.mark.parametrize("expected_version", [False, True])
async def test_optional_actor_membership_command_does_not_invent_authority(
    membership_profile,
    expected_version: bool,
) -> None:
    pool, repo, ids = membership_profile
    before = await ProfileVersionGateway(pool).read(ids["target"])
    command = ProfileUpdateCommand(
        actor_user_id=None,
        target_user_id=ids["target"],
        updates=(("memberships.orgs.role", {"org_id": ids["org"], "role": "admin"}),),
        roles=frozenset({"platform_admin"}),
        dry_run=False,
        expected_profile_version=before if expected_version else None,
    )

    async with pool.transaction() as conn:
        with pytest.raises(ValueError, match="^membership_context_unavailable$"):
            await ProfileCommandService(db_pool=pool).apply(command, db_conn=conn, scope=None)

    assert (await repo.get_org_member(ids["org"], ids["target"]))["role"] == "member"
    assert await ProfileVersionGateway(pool).read(ids["target"]) == before


@pytest.mark.asyncio
@pytest.mark.parametrize("scope_type", ["org", "team"])
@pytest.mark.parametrize("dry_run", [False, True])
@pytest.mark.parametrize(
    ("role", "message"),
    [
        (None, "invalid_value"),
        (7, "invalid_value"),
        ({}, "invalid_value"),
        ([], "invalid_value"),
        ("", "enum_violation"),
        ("superadmin", "enum_violation"),
    ],
)
async def test_invalid_dictionary_role_returns_validation_without_partial_writes(
    membership_profile,
    scope_type: str,
    dry_run: bool,
    role: Any,
    message: str,
) -> None:
    pool, repo, ids = membership_profile
    key = "memberships.orgs.role" if scope_type == "org" else "memberships.teams.role"
    before = await ProfileVersionGateway(pool).read(ids["target"])
    command = ProfileUpdateCommand(
        actor_user_id=ids["actor"],
        target_user_id=ids["target"],
        updates=(
            ("preferences.ui.theme", "paper"),
            ("memberships.teams.role", {"team_id": ids["team"], "role": "lead"}),
            (key, {f"{scope_type}_id": ids[scope_type], "role": role}),
        ),
        roles=frozenset({"org_admin", "user"}),
        dry_run=dry_run,
    )

    async with pool.transaction() as conn:
        result = await ProfileCommandService(db_pool=pool).apply(
            command,
            db_conn=conn,
            scope=ProfileUpdateScope(actor_user_id=ids["actor"]),
        )

    assert result.status_code == 422
    assert result.error_code == "profile_update_invalid"
    assert result.skipped == ({"key": key, "message": message},)
    assert (await repo.get_org_member(ids["org"], ids["target"]))["role"] == "member"
    assert (await repo.get_team_member(ids["team"], ids["target"]))["role"] == "member"
    assert await pool.fetchval("SELECT COUNT(*) FROM user_config_overrides WHERE user_id = ?", (ids["target"],)) == 0
    assert await ProfileVersionGateway(pool).read(ids["target"]) == before


@pytest.mark.asyncio
@pytest.mark.parametrize("dry_run", [False, True])
@pytest.mark.parametrize("scope_type", ["org", "team"])
@pytest.mark.parametrize("dictionary", [False, True])
@pytest.mark.parametrize(
    ("role", "canonical"),
    [("owner", "owner"), ("admin", "admin"), ("lead", "lead"), ("member", "member"), (" AdMiN ", "admin")],
)
async def test_role_update_normalizes_before_closed_writer(
    membership_profile,
    dry_run: bool,
    scope_type: str,
    dictionary: bool,
    role: str,
    canonical: str,
) -> None:
    pool, repo, ids = membership_profile
    key = "memberships.orgs.role" if scope_type == "org" else "memberships.teams.role"
    value = {f"{scope_type}_id": ids[scope_type], "role": role} if dictionary else role
    before = await ProfileVersionGateway(pool).read(ids["target"])

    async with pool.transaction() as conn:
        result = await UserProfileUpdateService(pool).apply_updates(
            user_id=ids["target"],
            updates=((key, value),),
            roles={"org_admin"},
            dry_run=dry_run,
            db_conn=conn,
            updated_by=ids["actor"],
            scope=ProfileUpdateScope(
                actor_user_id=ids["actor"],
                active_org_id=ids["org"],
                active_team_id=ids["team"],
            ),
        )

    assert result.applied == [key]
    assert result.skipped == []
    member = await (
        repo.get_org_member(ids["org"], ids["target"])
        if scope_type == "org"
        else repo.get_team_member(ids["team"], ids["target"])
    )
    assert member["role"] == ("member" if dry_run else canonical)
    after = await ProfileVersionGateway(pool).read(ids["target"])
    assert after == before if dry_run else after > before


@pytest.mark.asyncio
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("scope_type", ["org", "team"])
async def test_direct_invalid_role_batch_does_not_write_valid_memberships_or_overrides(
    membership_profile,
    reverse: bool,
    scope_type: str,
) -> None:
    pool, repo, ids = membership_profile
    key = "memberships.orgs.role" if scope_type == "org" else "memberships.teams.role"
    before = await ProfileVersionGateway(pool).read(ids["target"])
    updates = (
        ("preferences.ui.theme", "paper"),
        ("memberships.teams.role", {"team_id": ids["team"], "role": "lead"}),
        (key, {f"{scope_type}_id": ids[scope_type], "role": "superadmin"}),
    )

    async with pool.transaction() as conn:
        result = await UserProfileUpdateService(pool).apply_updates(
            user_id=ids["target"],
            updates=updates[::-1] if reverse else updates,
            roles={"org_admin", "user"},
            dry_run=False,
            db_conn=conn,
            updated_by=ids["actor"],
            scope=ProfileUpdateScope(actor_user_id=ids["actor"]),
        )

    assert (await repo.get_org_member(ids["org"], ids["target"]))["role"] == "member"
    assert (await repo.get_team_member(ids["team"], ids["target"]))["role"] == "member"
    assert await pool.fetchval("SELECT COUNT(*) FROM user_config_overrides WHERE user_id = ?", (ids["target"],)) == 0
    assert await ProfileVersionGateway(pool).read(ids["target"]) == before
    assert result.skipped == [{"key": key, "message": "enum_violation"}]
    assert result.applied == []


@pytest.mark.asyncio
@pytest.mark.parametrize("bulk", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("scope_type", ["org", "team"])
async def test_invalid_scalar_role_preserves_legacy_partial_success(
    membership_profile,
    monkeypatch: pytest.MonkeyPatch,
    bulk: bool,
    reverse: bool,
    scope_type: str,
) -> None:
    from unittest.mock import AsyncMock

    from tldw_Server_API.app.api.v1.schemas.user_profile_schemas import UserProfileBulkUpdateRequest
    from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
    from tldw_Server_API.app.services import admin_profiles_service

    pool, repo, ids = membership_profile
    before = await ProfileVersionGateway(pool).read(ids["target"])
    key = "memberships.orgs.role" if scope_type == "org" else "memberships.teams.role"
    updates = (("preferences.ui.theme", "paper"), (key, 7))
    if reverse:
        updates = updates[::-1]
    if bulk:
        monkeypatch.setattr(
            admin_profiles_service,
            "_load_bulk_user_candidates",
            AsyncMock(return_value=[ids["target"]]),
        )
        monkeypatch.setattr(
            admin_profiles_service.admin_scope_service,
            "enforce_admin_user_scope",
            AsyncMock(return_value=None),
        )
        payload = UserProfileBulkUpdateRequest(
            user_ids=[ids["target"]],
            updates=[{"key": name, "value": value} for name, value in updates],
            dry_run=False,
            confirm=True,
        )
        response, _audit = await admin_profiles_service.bulk_update_user_profiles(
            payload=payload,
            principal=AuthPrincipal(
                kind="user",
                user_id=ids["actor"],
                roles=["admin", "user"],
                is_admin=True,
                active_org_id=ids["org"],
                active_team_id=ids["team"],
            ),
        )
        result = response.results[0]
        skipped = [item.model_dump() for item in result.skipped]
        assert (response.updated, response.failed) == (1, 0)
    else:
        async with pool.transaction() as conn:
            result = await UserProfileUpdateService(pool).apply_updates(
                user_id=ids["target"],
                updates=updates,
                roles={"org_admin", "user"},
                dry_run=False,
                db_conn=conn,
                updated_by=ids["actor"],
                scope=ProfileUpdateScope(
                    actor_user_id=ids["actor"],
                    active_org_id=ids["org"],
                    active_team_id=ids["team"],
                ),
            )
        skipped = result.skipped

    assert result.applied == ["preferences.ui.theme"]
    assert skipped == [{"key": key, "message": "invalid_value"}]
    stored = await pool.fetchval(
        "SELECT value_json FROM user_config_overrides WHERE user_id = ? AND key = ?",
        (ids["target"], "preferences.ui.theme"),
    )
    assert json.loads(stored) == "paper"
    assert await ProfileVersionGateway(pool).read(ids["target"]) > before
    assert (await repo.get_org_member(ids["org"], ids["target"]))["role"] == "member"
    assert (await repo.get_team_member(ids["team"], ids["target"]))["role"] == "member"
