from __future__ import annotations

import asyncio
import uuid
from contextlib import asynccontextmanager
from typing import Any

import asyncpg
import pytest

from tldw_Server_API.app.api.v1.schemas.user_profile_schemas import (
    UserProfileBulkUpdateRequest,
    UserProfileUpdateEntry,
)
from tldw_Server_API.app.core.AuthNZ.membership_writer import (
    TrustedMembershipReason,
    TrustedMembershipWriteContext,
)
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.AuthNZ.profile_version import VersionedUserWriteGateway
from tldw_Server_API.app.core.AuthNZ.repos.orgs_teams_repo import AuthnzOrgsTeamsRepo
from tldw_Server_API.app.core.UserProfiles import update_service as update_service_module
from tldw_Server_API.app.core.UserProfiles.command_service import ProfileCommandService
from tldw_Server_API.app.core.UserProfiles.contracts import ProfileUpdateCommand
from tldw_Server_API.app.core.UserProfiles.service import UserProfileService
from tldw_Server_API.app.core.UserProfiles.update_service import ProfileUpdateScope
from tldw_Server_API.app.core.UserProfiles.version_gateway import ProfileVersionGateway
from tldw_Server_API.app.services import admin_profiles_service

pytestmark = pytest.mark.postgres

_BOOTSTRAP = TrustedMembershipWriteContext(
    trusted_reason=TrustedMembershipReason.BOOTSTRAP,
)


class _TwoPartyGate:
    def __init__(self) -> None:
        self._arrivals = 0
        self._ready = asyncio.Event()

    async def arrive(self) -> None:
        self._arrivals += 1
        if self._arrivals == 2:
            self._ready.set()
        await self._ready.wait()


class _GatedUserProfileService(UserProfileService):
    def __init__(self, db_pool: Any, gate: _TwoPartyGate) -> None:
        super().__init__(db_pool)
        self._gate = gate
        self.lock_calls: list[tuple[int, ...]] = []

    async def get_profile_version(
        self,
        *,
        user_id: int,
        user_updated_at: Any | None = None,
        db_conn: Any | None = None,
        lock_user: bool = False,
    ):
        if lock_user:
            await self._gate.arrive()
        return await super().get_profile_version(
            user_id=user_id,
            user_updated_at=user_updated_at,
            db_conn=db_conn,
            lock_user=lock_user,
        )

    async def lock_profile_users(
        self,
        *,
        user_ids: tuple[int, ...],
        db_conn: Any,
    ):
        self.lock_calls.append(user_ids)
        await self._gate.arrive()
        return await super().lock_profile_users(user_ids=user_ids, db_conn=db_conn)


async def _create_user(test_db_pool: Any, label: str, *, role: str) -> int:
    async with test_db_pool.transaction() as conn:
        result = await VersionedUserWriteGateway("postgres").insert_user(
            conn,
            values={
                "uuid": str(uuid.uuid4()),
                "username": label,
                "email": f"{label}@example.com",
                "password_hash": "hash",
                "role": role,
                "is_active": True,
                "is_verified": True,
            },
        )
    return result.affected_user_ids[0]


@pytest.mark.asyncio
async def test_profile_version_lock_uses_transaction_connection(test_db_pool):
    user_id = await test_db_pool.fetchval(
        """
        INSERT INTO users (uuid, username, email, password_hash, is_active)
        VALUES ($1, $2, $3, $4, TRUE)
        RETURNING id
        """,
        str(uuid.uuid4()),
        "pg-profile-version-lock",
        "pg-profile-version-lock@example.com",
        "hash",
    )
    gateway = ProfileVersionGateway(test_db_pool)
    backend_pool = test_db_pool.pool
    assert backend_pool is not None
    lock_failure_observed = False

    async with backend_pool.acquire(timeout=5.0) as lock_conn:
        async with lock_conn.transaction():
            version = await gateway.read_in_transaction(
                lock_conn,
                int(user_id),
                lock_user=True,
            )

            async with backend_pool.acquire(timeout=5.0) as competing_conn:
                with pytest.raises(asyncpg.exceptions.LockNotAvailableError) as raised:
                    async with competing_conn.transaction():
                        await competing_conn.execute(
                            "SET LOCAL lock_timeout = '500ms'"
                        )
                        await competing_conn.execute(
                            "UPDATE users SET profile_version = profile_version "
                            "WHERE id = $1",
                            int(user_id),
                        )
                lock_failure_observed = True
                assert raised.value.sqlstate == "55P03"

    assert version is not None
    assert lock_failure_observed is True


@pytest.mark.asyncio
@pytest.mark.parametrize("with_expected_version", (False, True))
async def test_reciprocal_admin_profile_commands_lock_users_without_deadlock(
    test_db_pool,
    monkeypatch,
    with_expected_version: bool,
) -> None:
    suffix = uuid.uuid4().hex[:8]
    owner_id = await _create_user(test_db_pool, f"owner_{suffix}", role="user")
    first_id = await _create_user(test_db_pool, f"first_{suffix}", role="admin")
    second_id = await _create_user(test_db_pool, f"second_{suffix}", role="admin")
    repo = AuthnzOrgsTeamsRepo(test_db_pool)
    organization = await repo.create_organization_with_owner_membership(
        name=f"Reciprocal command org {suffix}",
        owner_user_id=owner_id,
        context=_BOOTSTRAP,
    )
    org_id = int(organization["id"])
    for user_id, role in (
        (first_id, "member"),
        (second_id, "member"),
    ):
        await repo.add_org_member(
            org_id=org_id,
            user_id=user_id,
            role=role,
            context=_BOOTSTRAP,
        )

    monkeypatch.setattr(
        update_service_module,
        "list_org_memberships_for_user",
        repo.list_org_memberships_for_user,
    )
    monkeypatch.setattr(
        update_service_module,
        "list_memberships_for_user",
        repo.list_memberships_for_user,
    )

    profile_service = _GatedUserProfileService(test_db_pool, _TwoPartyGate())
    command_service = ProfileCommandService(
        db_pool=test_db_pool,
        profile_service=profile_service,
    )
    expected_versions = {
        first_id: await profile_service.get_profile_version(user_id=first_id),
        second_id: await profile_service.get_profile_version(user_id=second_id),
    }

    commands = (
        ProfileUpdateCommand(
            actor_user_id=first_id,
            target_user_id=second_id,
            updates=(("memberships.orgs.role", {"org_id": org_id, "role": "admin"}),),
            roles=frozenset({"admin"}),
            dry_run=False,
            expected_profile_version=(
                expected_versions[second_id] if with_expected_version else None
            ),
        ),
        ProfileUpdateCommand(
            actor_user_id=second_id,
            target_user_id=first_id,
            updates=(("memberships.orgs.role", {"org_id": org_id, "role": "admin"}),),
            roles=frozenset({"admin"}),
            dry_run=False,
            expected_profile_version=(
                expected_versions[first_id] if with_expected_version else None
            ),
        ),
    )

    async def _apply(command: ProfileUpdateCommand):
        async with test_db_pool.transaction(acquire_timeout_seconds=5.0) as conn:
            await conn.execute("SET LOCAL lock_timeout = '5s'")
            return await command_service.apply(
                command,
                db_conn=conn,
                scope=ProfileUpdateScope(actor_user_id=command.actor_user_id),
            )

    results = await asyncio.wait_for(
        asyncio.gather(*(_apply(command) for command in commands)),
        timeout=12.0,
    )

    assert [result.status_code for result in results] == [200, 200]
    assert [result.applied for result in results] == [
        ("memberships.orgs.role",),
        ("memberships.orgs.role",),
    ]
    expected_lock_order = tuple(sorted((first_id, second_id)))
    assert profile_service.lock_calls == [expected_lock_order, expected_lock_order]


@pytest.mark.asyncio
async def test_reciprocal_admin_bulk_membership_updates_lock_users_without_deadlock(
    test_db_pool,
    monkeypatch,
) -> None:
    suffix = uuid.uuid4().hex[:8]
    owner_id = await _create_user(test_db_pool, f"bulk_owner_{suffix}", role="user")
    first_id = await _create_user(test_db_pool, f"bulk_first_{suffix}", role="admin")
    second_id = await _create_user(test_db_pool, f"bulk_second_{suffix}", role="admin")
    repo = AuthnzOrgsTeamsRepo(test_db_pool)
    organization = await repo.create_organization_with_owner_membership(
        name=f"Reciprocal bulk org {suffix}",
        owner_user_id=owner_id,
        context=_BOOTSTRAP,
    )
    org_id = int(organization["id"])
    for user_id, role in (
        (first_id, "member"),
        (second_id, "member"),
    ):
        await repo.add_org_member(
            org_id=org_id,
            user_id=user_id,
            role=role,
            context=_BOOTSTRAP,
        )

    monkeypatch.setattr(
        update_service_module,
        "list_org_memberships_for_user",
        repo.list_org_memberships_for_user,
    )
    monkeypatch.setattr(
        update_service_module,
        "list_memberships_for_user",
        repo.list_memberships_for_user,
    )

    class _BoundedPool:
        pool = test_db_pool.pool

        @asynccontextmanager
        async def transaction(self):
            async with test_db_pool.transaction(acquire_timeout_seconds=5.0) as conn:
                await conn.execute("SET LOCAL lock_timeout = '5s'")
                yield conn

    bounded_pool = _BoundedPool()
    profile_service = _GatedUserProfileService(
        test_db_pool,
        _TwoPartyGate(),
    )

    async def _get_pool():
        return bounded_pool

    async def _candidates(**kwargs):
        return list(kwargs["user_ids"] or [])

    async def _allow_scope(*_args, **_kwargs):
        return None

    async def _repo_from_pool():
        return object()

    async def _before_values(**_kwargs):
        return {}

    monkeypatch.setattr(admin_profiles_service, "get_db_pool", _get_pool)
    monkeypatch.setattr(
        admin_profiles_service,
        "_load_bulk_user_candidates",
        _candidates,
    )
    monkeypatch.setattr(
        admin_profiles_service.admin_scope_service,
        "enforce_admin_user_scope",
        _allow_scope,
    )
    monkeypatch.setattr(
        admin_profiles_service,
        "UserProfileService",
        lambda _db_pool: profile_service,
    )
    monkeypatch.setattr(
        admin_profiles_service.AuthnzUsersRepo,
        "from_pool",
        _repo_from_pool,
    )
    monkeypatch.setattr(
        admin_profiles_service,
        "_build_bulk_update_before_values",
        _before_values,
    )

    def _principal(user_id: int) -> AuthPrincipal:
        return AuthPrincipal(
            kind="user",
            user_id=user_id,
            username=f"bulk-admin-{user_id}",
            roles=["admin"],
            permissions=["*"],
            is_admin=True,
            org_ids=[org_id],
            active_org_id=org_id,
        )

    def _payload(target_user_id: int) -> UserProfileBulkUpdateRequest:
        return UserProfileBulkUpdateRequest(
            updates=[
                UserProfileUpdateEntry(
                    key="memberships.orgs.role",
                    value={"org_id": org_id, "role": "admin"},
                )
            ],
            confirm=True,
            user_ids=[target_user_id],
        )

    responses = await asyncio.wait_for(
        asyncio.gather(
            admin_profiles_service.bulk_update_user_profiles(
                payload=_payload(second_id),
                principal=_principal(first_id),
            ),
            admin_profiles_service.bulk_update_user_profiles(
                payload=_payload(first_id),
                principal=_principal(second_id),
            ),
        ),
        timeout=12.0,
    )

    assert [(response.updated, response.failed) for response, _audit in responses] == [
        (1, 0),
        (1, 0),
    ]
    expected_lock_order = tuple(sorted((first_id, second_id)))
    assert profile_service.lock_calls == [expected_lock_order, expected_lock_order]
