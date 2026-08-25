from __future__ import annotations

import asyncio
import io
import json
import uuid
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient
from loguru import logger

from tldw_Server_API.app.api.v1.schemas.user_profile_schemas import (
    UserProfileUpdateEntry,
    UserProfileUpdateRequest,
)
from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
from tldw_Server_API.app.core.AuthNZ.membership_writer import (
    ActorMembershipWriteContext,
    AnchorOwnership,
    MembershipAuthority,
    MembershipAuthorizationError,
    MembershipMutationResult,
    MembershipScopeType,
    MembershipUserVersionFloor,
    MembershipWriteResult,
    TrustedMembershipReason,
    TrustedMembershipWriteContext,
)
from tldw_Server_API.app.core.AuthNZ.orgs_teams import (
    add_org_member,
    add_team_member,
    create_organization,
    create_team,
    list_memberships_for_user,
    list_org_memberships_for_user,
)
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.AuthNZ.rate_limiter import get_rate_limiter
from tldw_Server_API.app.core.AuthNZ.repos.mfa_repo import AuthnzMfaRepo
from tldw_Server_API.app.core.DB_Management.Users_DB import UsersDB
from tldw_Server_API.app.core.UserProfiles import update_service as update_service_module
from tldw_Server_API.app.core.UserProfiles.command_service import ProfileCommandService
from tldw_Server_API.app.core.UserProfiles.contracts import ProfileContractMode, ProfileUpdateCommand
from tldw_Server_API.app.core.UserProfiles.response_mappers import (
    LegacyProfileCommandResult,
)
from tldw_Server_API.app.core.UserProfiles.version_gateway import ProfileVersionGateway
from tldw_Server_API.app.main import app
from tldw_Server_API.app.services.storage_quota_service import StorageQuotaService

_BOOTSTRAP_MEMBERSHIP_CONTEXT = TrustedMembershipWriteContext(
    trusted_reason=TrustedMembershipReason.BOOTSTRAP,
)


def _run_async(coro):
    return asyncio.run(coro)


def _get_user_id(client: TestClient, auth_headers) -> int:
    resp = client.get("/api/v1/users/me/profile", headers=auth_headers)
    assert resp.status_code == 200
    return resp.json()["user"]["id"]


def _get_profile_version(client: TestClient, auth_headers) -> datetime:
    resp = client.get("/api/v1/users/me/profile", headers=auth_headers)
    assert resp.status_code == 200
    return datetime.fromisoformat(resp.json()["profile_version"].replace("Z", "+00:00"))


@pytest.mark.asyncio
async def test_legacy_users_db_last_login_uses_utc_aware_timestamp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    users = UsersDB(db_pool=object())
    captured: dict[str, object] = {}

    async def _capture_update(user_id: int, **updates: object) -> None:
        captured["user_id"] = user_id
        captured.update(updates)

    monkeypatch.setattr(users, "update_user", _capture_update)

    await users.update_last_login(17)

    assert captured["user_id"] == 17
    assert isinstance(captured["last_login"], datetime)
    assert captured["last_login"].tzinfo is timezone.utc


@pytest.mark.asyncio
async def test_user_field_mutation_failure_log_is_sanitized() -> None:
    secret = "email=private@example.com database=/private/authnz.db"

    class _Anchor:
        async def capture(self) -> None:
            raise RuntimeError(secret)

    output = io.StringIO()
    sink = logger.add(output, format="{message} {extra}")
    try:
        with pytest.raises(RuntimeError, match="private@example.com"):
            await update_service_module._update_user_field(  # noqa: SLF001
                object(),
                7,
                "email",
                "new@example.com",
                anchor=_Anchor(),
                is_postgres_backend=False,
            )
    finally:
        logger.remove(sink)

    assert secret not in output.getvalue()


def test_committed_identity_update_strictly_advances_profile_version(auth_headers) -> None:
    with TestClient(app) as client:
        before = _get_profile_version(client, auth_headers)
        email = f"profile-version-{uuid.uuid4().hex[:8]}@example.com"

        response = client.patch(
            "/api/v1/users/me/profile",
            headers=auth_headers,
            json={"updates": [{"key": "identity.email", "value": email}]},
        )

        assert response.status_code == 200
        assert _get_profile_version(client, auth_headers) > before


def test_committed_quota_status_and_mfa_updates_strictly_advance_profile_version(
    auth_headers,
) -> None:
    with TestClient(app) as client:
        user_id = _get_user_id(client, auth_headers)

        async def _exercise_writers() -> tuple[datetime, ...]:
            pool = await get_db_pool()
            reader = ProfileVersionGateway(pool)
            users = UsersDB(db_pool=pool)
            await users.initialize()
            original = await users.get_user_by_id(user_id)
            assert original is not None
            original_quota = int(original.get("storage_quota_mb") or 5120)
            original_active = bool(original.get("is_active", True))

            versions = [await reader.read(user_id)]
            quota_service = StorageQuotaService(db_pool=pool)
            await quota_service.set_user_quota(user_id, original_quota + 1)
            versions.append(await reader.read(user_id))

            await users.update_user(user_id, is_active=not original_active)
            versions.append(await reader.read(user_id))
            await users.update_user(user_id, is_active=original_active)

            mfa_repo = AuthnzMfaRepo(pool)
            await mfa_repo.set_mfa_config(
                user_id=user_id,
                encrypted_secret="encrypted-test-secret",
                backup_codes_json="[]",
                updated_at=datetime.now(timezone.utc),
            )
            versions.append(await reader.read(user_id))

            await mfa_repo.clear_mfa_config(
                user_id=user_id,
                updated_at=datetime.now(timezone.utc),
            )
            await quota_service.set_user_quota(user_id, original_quota)
            return tuple(versions)

        before, after_quota, after_status, after_mfa = _run_async(
            _exercise_writers()
        )

        assert after_quota > before
        assert after_status > after_quota
        assert after_mfa > after_status


def test_deprecated_profile_update_strictly_advances_profile_version(auth_headers) -> None:
    with TestClient(app) as client:
        before = _get_profile_version(client, auth_headers)
        email = f"legacy-version-{uuid.uuid4().hex[:8]}@example.com"

        response = client.put(
            "/api/v1/users/me",
            headers=auth_headers,
            json={"email": email},
        )

        assert response.status_code == 200
        assert _get_profile_version(client, auth_headers) > before


@pytest.mark.asyncio
async def test_user_profile_update_accepts_valid_identity_email_with_pydantic_v2(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    updates: list[tuple[int, str, str]] = []

    async def _capture_update(
        _db_conn,
        user_id: int,
        field: str,
        value: str,
        **_kwargs,
    ) -> None:
        updates.append((user_id, field, value))

    monkeypatch.setattr(update_service_module, "_update_user_field", _capture_update)

    result = await update_service_module.UserProfileUpdateService(db_pool=object()).apply_updates(
        user_id=7,
        updates=(("identity.email", "Restored.Profile@example.com"),),
        roles={"user"},
        dry_run=False,
        db_conn=object(),
        updated_by=7,
    )

    assert result.applied == ["identity.email"]
    assert result.skipped == []
    assert updates == [(7, "email", "restored.profile@example.com")]


@pytest.mark.asyncio
async def test_invalid_identity_email_log_excludes_validation_details(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    debug_calls: list[tuple[str, tuple[object, ...]]] = []

    def _capture_debug(message: str, *args: object) -> None:
        debug_calls.append((message, args))

    monkeypatch.setattr(update_service_module.logger, "debug", _capture_debug)
    service = update_service_module.UserProfileUpdateService(db_pool=object())
    rejected_email = "private-rejected-value"

    with pytest.raises(ValueError, match="^invalid_email$"):
        await service._apply_key_update(
            user_id=7,
            key="identity.email",
            value=rejected_email,
            dry_run=True,
            db_conn=object(),
            repo_holder={"repo": None},
            updated_by=7,
            is_postgres_backend=False,
            anchor=object(),
        )

    assert debug_calls == [("Invalid email update for user {}", (7,))]
    assert rejected_email not in repr(debug_calls)
    assert "ValidationError" not in repr(debug_calls)


def test_user_profile_update_preferences(auth_headers) -> None:
    with TestClient(app) as client:
        resp = client.patch(
            "/api/v1/users/me/profile",
            headers=auth_headers,
            json={
                "updates": [
                    {"key": "preferences.ui.theme", "value": "paper"},
                ]
            },
        )
        assert resp.status_code == 200
        payload = resp.json()
        assert "preferences.ui.theme" in payload["applied"]

        profile_resp = client.get(
            "/api/v1/users/me/profile",
            params={"sections": "preferences"},
            headers=auth_headers,
        )
        assert profile_resp.status_code == 200
        profile = profile_resp.json()

        effective_resp = client.get(
            "/api/v1/users/me/profile",
            params={"sections": "effective_config"},
            headers=auth_headers,
        )
        assert effective_resp.status_code == 200
        effective = effective_resp.json()

    assert profile.get("preferences", {}).get("preferences.ui.theme") == "paper"
    assert effective.get("effective_config", {}).get("preferences.ui.theme") == "paper"


@pytest.mark.asyncio
async def test_user_profile_override_writes_use_transaction_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Cursor:
        rowcount = 1

        def __init__(self, rows=None) -> None:
            self._rows = rows or []

        async def fetchall(self):
            return self._rows

    class _DbConn:
        async def execute(self, query, params):
            if str(query).lstrip().lower().startswith("with target_user as"):
                return _Cursor(
                    [("user", int(params[0]), "2026-07-26T12:00:00.000000Z")]
                )
            return _Cursor()

    db_conn = _DbConn()
    calls: list[tuple[str, str, object | None]] = []

    class FakeOverridesRepo:
        def __init__(self, db_pool) -> None:
            self.db_pool = db_pool

        async def ensure_tables(self) -> None:
            return None

        async def upsert_override(
            self,
            *,
            user_id: int,
            key: str,
            value,
            updated_by: int | None,
            db_conn=None,
        ) -> None:
            calls.append(("upsert", key, db_conn))

        async def delete_override(
            self,
            *,
            user_id: int,
            key: str,
            db_conn=None,
        ) -> None:
            calls.append(("delete", key, db_conn))

    monkeypatch.setattr(
        update_service_module,
        "UserProfileOverridesRepo",
        FakeOverridesRepo,
    )

    result = await update_service_module.UserProfileUpdateService(
        db_pool=object(),
    ).apply_updates(
        user_id=7,
        updates=(
            ("preferences.ui.theme", "paper"),
            ("preferences.chat.default_character_id", None),
        ),
        roles={"user"},
        dry_run=False,
        db_conn=db_conn,
        updated_by=7,
    )

    assert result.applied == [
        "preferences.ui.theme",
        "preferences.chat.default_character_id",
    ]
    assert result.skipped == []
    assert calls == [
        ("upsert", "preferences.ui.theme", db_conn),
        ("delete", "preferences.chat.default_character_id", db_conn),
    ]


@pytest.mark.asyncio
async def test_membership_updates_use_caller_transaction_and_one_outer_touch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_pool = object()
    db_conn = object()
    floor_one = datetime(2026, 7, 26, 12, 0, tzinfo=timezone.utc)
    floor_two = datetime(2026, 7, 26, 12, 1, tzinfo=timezone.utc)
    writer_calls: list[dict[str, object]] = []
    final_touches: list[tuple[object, int, datetime]] = []
    membership_context_connections: list[object | None] = []

    async def _membership_context(*_args, **_kwargs):
        membership_context_connections.append(_kwargs.get("db_conn"))
        return update_service_module._MembershipContext(  # noqa: SLF001
            target_org_roles={11: "member"},
            target_team_roles={22: "member"},
            target_team_orgs={22: 11},
        )

    class _MembershipWriter:
        def __init__(self, supplied_pool: object) -> None:
            assert supplied_pool is db_pool

        async def apply_membership_mutations(self, **kwargs) -> MembershipWriteResult:
            writer_calls.append(kwargs)
            mutations = kwargs["mutations"]
            return MembershipWriteResult(
                mutation_results=tuple(
                    MembershipMutationResult(
                        mutation=mutation,
                        changed=True,
                        found=True,
                        role=mutation.role,
                        organization_id=(
                            11
                            if mutation.scope_type is MembershipScopeType.TEAM
                            else None
                        ),
                    )
                    for mutation in mutations
                ),
                affected_user_ids=(7,),
                version_floors=(
                    MembershipUserVersionFloor(
                        user_id=7,
                        pre_mutation_floor=floor_one,
                        post_mutation_floor=floor_two,
                    ),
                ),
            )

    class _VersionGateway:
        def __init__(self, _backend: str, *, clock) -> None:
            self._clock = clock

        async def capture_floor(self, *_args, **_kwargs):
            raise AssertionError("membership updates must use the writer's floor")

        async def final_touch(self, conn, *, user_id: int, version_floor: datetime):
            final_touches.append((conn, user_id, version_floor))

    async def _legacy_membership_call(**_kwargs):
        raise AssertionError("membership update opened a nested transaction")

    async def _legacy_touch(*_args, **_kwargs):
        raise AssertionError("membership update issued a separate users touch")

    monkeypatch.setattr(
        update_service_module.UserProfileUpdateService,
        "_build_membership_context",
        _membership_context,
    )
    monkeypatch.setattr(
        update_service_module,
        "MembershipWriter",
        _MembershipWriter,
        raising=False,
    )
    monkeypatch.setattr(
        update_service_module,
        "VersionedUserWriteGateway",
        _VersionGateway,
    )
    monkeypatch.setattr(
        update_service_module,
        "update_org_member_role",
        _legacy_membership_call,
        raising=False,
    )
    monkeypatch.setattr(
        update_service_module,
        "update_team_member_role",
        _legacy_membership_call,
        raising=False,
    )
    monkeypatch.setattr(
        update_service_module,
        "_touch_user_updated_at",
        _legacy_touch,
    )

    result = await update_service_module.UserProfileUpdateService(
        db_pool=db_pool,
    ).apply_updates(
        user_id=7,
        updates=(
            ("memberships.orgs.role", {"org_id": 11, "role": "admin"}),
            ("memberships.teams.role", {"team_id": 22, "role": "lead"}),
        ),
        roles={"platform_admin"},
        dry_run=False,
        db_conn=db_conn,
        updated_by=99,
        scope=update_service_module.ProfileUpdateScope(actor_user_id=99),
    )

    assert result.applied == ["memberships.orgs.role", "memberships.teams.role"]
    assert result.skipped == []
    assert membership_context_connections == [db_conn]
    assert len(writer_calls) == 1
    assert writer_calls[0]["conn"] is db_conn
    assert (
        writer_calls[0]["anchor_ownership"]
        is AnchorOwnership.CALLER_OWNS_ANCHOR
    )
    assert writer_calls[0]["context"] == ActorMembershipWriteContext(
        actor_user_id=99,
        required_authority=MembershipAuthority.PLATFORM_ADMIN,
    )
    assert [
        (mutation.scope_type, mutation.scope_id)
        for mutation in writer_calls[0]["mutations"]
    ] == [
        (MembershipScopeType.ORGANIZATION, 11),
        (MembershipScopeType.TEAM, 22),
    ]
    assert final_touches == [(db_conn, 7, floor_two)]


@pytest.mark.asyncio
async def test_membership_context_forwards_owned_connection_to_all_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_conn = object()
    calls: list[tuple[str, int, object | None]] = []

    async def _org_memberships(user_id: int, *, db_conn=None):
        calls.append(("org", user_id, db_conn))
        return [{"org_id": 11, "role": "admin", "status": "active"}]

    async def _team_memberships(user_id: int, *, db_conn=None):
        calls.append(("team", user_id, db_conn))
        return [{"team_id": 22, "org_id": 11, "role": "lead"}]

    monkeypatch.setattr(
        update_service_module,
        "list_org_memberships_for_user",
        _org_memberships,
    )
    monkeypatch.setattr(
        update_service_module,
        "list_memberships_for_user",
        _team_memberships,
    )

    context = await update_service_module.UserProfileUpdateService(
        db_pool=object(),
    )._build_membership_context(  # noqa: SLF001
        user_id=7,
        scope=update_service_module.ProfileUpdateScope(actor_user_id=9),
        is_platform_admin=False,
        db_conn=db_conn,
    )

    assert calls == [
        ("org", 7, db_conn),
        ("team", 7, db_conn),
        ("org", 9, db_conn),
        ("team", 9, db_conn),
    ]
    assert context.target_org_roles == {11: "admin"}
    assert context.actor_team_roles == {22: "lead"}


@pytest.mark.asyncio
async def test_membership_authorization_drift_precedes_all_mixed_side_effects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    side_effects: list[str] = []

    async def _membership_context(*_args, **_kwargs):
        return update_service_module._MembershipContext(  # noqa: SLF001
            target_org_roles={11: "member"},
        )

    class _RejectingMembershipWriter:
        def __init__(self, _pool: object) -> None:
            pass

        async def apply_membership_mutations(self, **_kwargs) -> MembershipWriteResult:
            side_effects.append("membership_writer")
            raise MembershipAuthorizationError()

    async def _update_user_field(*_args, **_kwargs) -> None:
        side_effects.append("user_write")

    async def _fetch_username(*_args, **_kwargs) -> str:
        side_effects.append("username_read")
        return "target-user"

    async def _touch_user(*_args, **_kwargs) -> None:
        side_effects.append("user_touch")

    class _LoginLimiter:
        async def record_failed_attempt(self, **_kwargs) -> None:
            side_effects.append("login_limiter")

    class _OverridesRepo:
        def __init__(self, _pool: object) -> None:
            pass

        async def ensure_tables(self) -> None:
            side_effects.append("overrides_ensure")

        async def upsert_override(self, **_kwargs) -> None:
            side_effects.append("override_write")

    class _EvaluationConfig:
        evaluations_per_minute = 10
        batch_evaluations_per_minute = 10
        evaluations_per_day = 100
        total_tokens_per_day = 1000
        burst_size = 2
        max_cost_per_day = 1.0
        max_cost_per_month = 10.0

    class _EvaluationLimiter:
        async def _get_user_config(self, _user_id: str) -> _EvaluationConfig:
            side_effects.append("evaluation_config")
            return _EvaluationConfig()

        async def upgrade_user_tier(self, *_args, **_kwargs) -> bool:
            side_effects.append("evaluation_limiter")
            return True

    class _ProfileService:
        async def get_profile_version(self, **_kwargs):
            return datetime(2026, 7, 26, 12, 0, tzinfo=timezone.utc)

        async def lock_profile_users(self, *, user_ids, db_conn):
            assert user_ids == (7, 99)
            return dict.fromkeys(
                user_ids,
                datetime(2026, 7, 26, 12, 0, tzinfo=timezone.utc),
            )

        @staticmethod
        def versions_match(current, expected) -> bool:
            return current == expected

    class _VersionGateway:
        def __init__(self, _backend: str, *, clock) -> None:
            del clock

        async def capture_floor(self, *_args, **_kwargs):
            side_effects.append("profile_floor")
            return datetime(2026, 7, 26, 12, 0, tzinfo=timezone.utc)

        async def final_touch(self, *_args, **_kwargs) -> None:
            side_effects.append("final_touch")

    class _Planner:
        async def plan(self, command, **_kwargs):
            return update_service_module.UpdateResult(
                applied=[key for key, _value in command.updates]
            )

    class _Connection:
        pass

    from tldw_Server_API.app.core.Evaluations import user_rate_limiter

    monkeypatch.setattr(
        update_service_module.UserProfileUpdateService,
        "_build_membership_context",
        _membership_context,
    )
    monkeypatch.setattr(
        update_service_module,
        "MembershipWriter",
        _RejectingMembershipWriter,
    )
    monkeypatch.setattr(update_service_module, "_update_user_field", _update_user_field)
    monkeypatch.setattr(update_service_module, "_fetch_username", _fetch_username)
    monkeypatch.setattr(update_service_module, "_touch_user_updated_at", _touch_user)
    monkeypatch.setattr(update_service_module, "get_rate_limiter", _LoginLimiter)
    monkeypatch.setattr(update_service_module, "UserProfileOverridesRepo", _OverridesRepo)
    monkeypatch.setattr(
        update_service_module,
        "VersionedUserWriteGateway",
        _VersionGateway,
    )
    monkeypatch.setattr(
        user_rate_limiter,
        "get_user_rate_limiter_for_user",
        lambda _user_id: _EvaluationLimiter(),
    )

    command = ProfileUpdateCommand(
        actor_user_id=99,
        target_user_id=7,
        updates=(
            ("identity.email", "target@example.com"),
            ("identity.is_locked", True),
            ("limits.evaluations_per_minute", 42),
            ("memberships.orgs.role", {"org_id": 11, "role": "admin"}),
        ),
        roles=frozenset({"platform_admin", "user"}),
        dry_run=False,
    )
    result = await ProfileCommandService(
        db_pool=type("Pool", (), {"pool": None})(),
        profile_service=_ProfileService(),
        planner=_Planner(),
    ).apply(
        command,
        db_conn=_Connection(),
        scope=update_service_module.ProfileUpdateScope(actor_user_id=99),
    )

    assert result.status_code == 403
    assert result.error_code == "profile_update_forbidden"
    assert result.applied == ()
    assert result.skipped == (
        {"key": "memberships.orgs.role", "message": "forbidden"},
    )
    assert side_effects == ["membership_writer"]


@pytest.mark.asyncio
async def test_idempotent_membership_batch_is_applied_and_touches_profile_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_conn = object()
    floor = datetime(2026, 7, 26, 12, 0, tzinfo=timezone.utc)
    writer_calls: list[dict[str, object]] = []
    touches: list[datetime] = []

    async def _membership_context(*_args, **_kwargs):
        return update_service_module._MembershipContext(  # noqa: SLF001
            target_team_roles={22: "member"},
            target_team_orgs={22: 11},
        )

    class _MembershipWriter:
        def __init__(self, _pool: object) -> None:
            pass

        async def apply_membership_mutations(self, **kwargs) -> MembershipWriteResult:
            writer_calls.append(kwargs)
            mutation = kwargs["mutations"][0]
            return MembershipWriteResult(
                mutation_results=(
                    MembershipMutationResult(
                        mutation=mutation,
                        changed=False,
                        found=True,
                        role="member",
                        organization_id=11,
                    ),
                ),
                affected_user_ids=(),
                version_floors=(),
            )

    class _VersionGateway:
        def __init__(self, _backend: str, *, clock) -> None:
            del clock

        async def capture_floor(self, conn, *, user_id: int):
            assert conn is db_conn
            assert user_id == 7
            return floor

        async def final_touch(self, conn, *, user_id: int, version_floor: datetime):
            assert conn is db_conn
            assert user_id == 7
            touches.append(version_floor)

    monkeypatch.setattr(
        update_service_module.UserProfileUpdateService,
        "_build_membership_context",
        _membership_context,
    )
    monkeypatch.setattr(update_service_module, "MembershipWriter", _MembershipWriter)
    monkeypatch.setattr(
        update_service_module,
        "VersionedUserWriteGateway",
        _VersionGateway,
    )

    result = await update_service_module.UserProfileUpdateService(
        db_pool=type("Pool", (), {"pool": None})(),
    ).apply_updates(
        user_id=7,
        updates=(("memberships.teams.member", {"team_id": 22, "action": "add"}),),
        roles={"platform_admin"},
        dry_run=False,
        db_conn=db_conn,
        updated_by=99,
        scope=update_service_module.ProfileUpdateScope(actor_user_id=99),
    )

    assert result.applied == ["memberships.teams.member"]
    assert result.skipped == []
    assert len(writer_calls) == 1
    assert touches == [floor]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("actor_team_role", "expected_applied", "expected_skipped", "writer_call_count"),
    (
        ("lead", ["memberships.teams.role"], [], 1),
        (
            "member",
            [],
            [{"key": "memberships.teams.role", "message": "forbidden_scope"}],
            0,
        ),
    ),
)
async def test_user_profile_team_mutation_requires_scoped_team_authority(
    monkeypatch: pytest.MonkeyPatch,
    actor_team_role: str,
    expected_applied: list[str],
    expected_skipped: list[dict[str, str]],
    writer_call_count: int,
) -> None:
    writer_calls: list[dict[str, object]] = []

    async def _membership_context(*_args, **_kwargs):
        return update_service_module._MembershipContext(  # noqa: SLF001
            target_team_roles={22: "member"},
            target_team_orgs={22: 11},
            actor_team_roles={22: actor_team_role},
        )

    class _MembershipWriter:
        def __init__(self, _pool: object) -> None:
            pass

        async def apply_membership_mutations(self, **kwargs) -> MembershipWriteResult:
            writer_calls.append(kwargs)
            mutation = kwargs["mutations"][0]
            return MembershipWriteResult(
                mutation_results=(
                    MembershipMutationResult(
                        mutation=mutation,
                        changed=True,
                        found=True,
                        role=mutation.role,
                        organization_id=11,
                    ),
                ),
                affected_user_ids=(7,),
                version_floors=(
                    MembershipUserVersionFloor(
                        user_id=7,
                        pre_mutation_floor=datetime(
                            2026, 7, 26, 12, 0, tzinfo=timezone.utc
                        ),
                        post_mutation_floor=datetime(
                            2026, 7, 26, 12, 1, tzinfo=timezone.utc
                        ),
                    ),
                ),
            )

    class _VersionGateway:
        def __init__(self, _backend: str, *, clock) -> None:
            del clock

        async def final_touch(self, *_args, **_kwargs) -> None:
            return None

    monkeypatch.setattr(
        update_service_module.UserProfileUpdateService,
        "_build_membership_context",
        _membership_context,
    )
    monkeypatch.setattr(update_service_module, "MembershipWriter", _MembershipWriter)
    monkeypatch.setattr(
        update_service_module,
        "VersionedUserWriteGateway",
        _VersionGateway,
    )

    result = await update_service_module.UserProfileUpdateService(
        db_pool=type("Pool", (), {"pool": None})(),
    ).apply_updates(
        user_id=7,
        updates=(
            (
                "memberships.teams.role",
                {"team_id": 22, "role": "admin"},
            ),
        ),
        roles={"team_admin"},
        dry_run=False,
        db_conn=object(),
        updated_by=99,
        scope=update_service_module.ProfileUpdateScope(actor_user_id=99),
    )

    assert result.applied == expected_applied
    assert result.skipped == expected_skipped
    assert len(writer_calls) == writer_call_count
    if writer_calls:
        assert writer_calls[0]["context"] == ActorMembershipWriteContext(
            actor_user_id=99,
            required_authority=MembershipAuthority.SCOPED_MEMBERSHIP,
        )


def test_user_profile_org_mutation_requires_scoped_org_admin_authority() -> None:
    service = update_service_module.UserProfileUpdateService(db_pool=object())
    member_context = update_service_module._MembershipContext(  # noqa: SLF001
        target_org_roles={11: "member"},
        actor_org_roles={11: "member"},
    )

    with pytest.raises(ValueError, match="^forbidden_scope$"):
        service._prepare_membership_mutation(  # noqa: SLF001
            user_id=7,
            key="memberships.orgs.role",
            value={"org_id": 11, "role": "admin"},
            scope=update_service_module.ProfileUpdateScope(actor_user_id=99),
            is_platform_admin=False,
            membership_context=member_context,
        )

    admin_context = update_service_module._MembershipContext(  # noqa: SLF001
        target_org_roles={11: "member"},
        actor_org_roles={11: "admin"},
    )
    mutation = service._prepare_membership_mutation(  # noqa: SLF001
        user_id=7,
        key="memberships.orgs.role",
        value={"org_id": 11, "role": "admin"},
        scope=update_service_module.ProfileUpdateScope(actor_user_id=99),
        is_platform_admin=False,
        membership_context=admin_context,
    )

    assert mutation.scope_id == 11


@pytest.mark.asyncio
async def test_membership_batch_preserves_opposite_scope_order_and_one_mixed_touch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_conn = object()
    floor = datetime(2026, 7, 26, 12, 1, tzinfo=timezone.utc)
    writer_calls: list[dict[str, object]] = []
    execution_order: list[str] = []
    capture_calls: list[int] = []
    touches: list[datetime] = []

    async def _membership_context(*_args, **_kwargs):
        return update_service_module._MembershipContext(  # noqa: SLF001
            target_org_roles={11: "member"},
            target_team_roles={22: "member"},
            target_team_orgs={22: 11},
        )

    class _MembershipWriter:
        def __init__(self, _pool: object) -> None:
            pass

        async def apply_membership_mutations(self, **kwargs) -> MembershipWriteResult:
            execution_order.append("membership_writer")
            writer_calls.append(kwargs)
            team_mutation, org_mutation = kwargs["mutations"]
            return MembershipWriteResult(
                mutation_results=(
                    MembershipMutationResult(
                        mutation=team_mutation,
                        changed=False,
                        found=True,
                        role="member",
                        organization_id=11,
                    ),
                    MembershipMutationResult(
                        mutation=org_mutation,
                        changed=True,
                        found=True,
                        role="admin",
                    ),
                ),
                affected_user_ids=(7,),
                version_floors=(
                    MembershipUserVersionFloor(
                        user_id=7,
                        pre_mutation_floor=datetime(
                            2026, 7, 26, 12, 0, tzinfo=timezone.utc
                        ),
                        post_mutation_floor=floor,
                    ),
                ),
            )

    class _VersionGateway:
        def __init__(self, _backend: str, *, clock) -> None:
            del clock

        async def capture_floor(self, *_args, **_kwargs):
            capture_calls.append(1)
            return floor

        async def final_touch(self, conn, *, user_id: int, version_floor: datetime):
            assert conn is db_conn
            assert user_id == 7
            touches.append(version_floor)
            execution_order.append("final_touch")

    class _OverridesRepo:
        def __init__(self, _pool: object) -> None:
            pass

        async def ensure_tables(self) -> None:
            return None

        async def upsert_override(self, **_kwargs) -> None:
            execution_order.append("preference_write")

    monkeypatch.setattr(
        update_service_module.UserProfileUpdateService,
        "_build_membership_context",
        _membership_context,
    )
    monkeypatch.setattr(update_service_module, "MembershipWriter", _MembershipWriter)
    monkeypatch.setattr(
        update_service_module,
        "VersionedUserWriteGateway",
        _VersionGateway,
    )
    monkeypatch.setattr(update_service_module, "UserProfileOverridesRepo", _OverridesRepo)

    result = await update_service_module.UserProfileUpdateService(
        db_pool=type("Pool", (), {"pool": None})(),
    ).apply_updates(
        user_id=7,
        updates=(
            ("preferences.ui.theme", "paper"),
            (
                "memberships.teams.role",
                {"team_id": 22, "role": "member"},
            ),
            (
                "memberships.orgs.role",
                {"org_id": 11, "role": "admin"},
            ),
        ),
        roles={"platform_admin", "user"},
        dry_run=False,
        db_conn=db_conn,
        updated_by=99,
        scope=update_service_module.ProfileUpdateScope(actor_user_id=99),
    )

    assert result.applied == [
        "preferences.ui.theme",
        "memberships.teams.role",
        "memberships.orgs.role",
    ]
    assert result.skipped == []
    assert len(writer_calls) == 1
    assert [
        (mutation.scope_type, mutation.scope_id)
        for mutation in writer_calls[0]["mutations"]
    ] == [
        (MembershipScopeType.TEAM, 22),
        (MembershipScopeType.ORGANIZATION, 11),
    ]
    assert capture_calls == []
    assert touches == [floor]
    assert execution_order == [
        "membership_writer",
        "preference_write",
        "final_touch",
    ]


def test_user_profile_preferences_include_sources(auth_headers) -> None:
    with TestClient(app) as client:
        resp = client.patch(
            "/api/v1/users/me/profile",
            headers=auth_headers,
            json={
                "updates": [
                    {"key": "preferences.ui.theme", "value": "paper"},
                ]
            },
        )
        assert resp.status_code == 200

        profile_resp = client.get(
            "/api/v1/users/me/profile",
            params={"sections": "preferences", "include_sources": "true"},
            headers=auth_headers,
        )
        assert profile_resp.status_code == 200
        preferences = profile_resp.json().get("preferences", {})

    entry = preferences.get("preferences.ui.theme")
    assert entry.get("value") == "paper"
    assert entry.get("source") == "user"


def test_user_profile_update_default_character_preference_set_and_clear(
    auth_headers,
) -> None:
    with TestClient(app) as client:
        set_resp = client.patch(
            "/api/v1/users/me/profile",
            headers=auth_headers,
            json={
                "updates": [
                    {
                        "key": "preferences.chat.default_character_id",
                        "value": "char-123",
                    }
                ]
            },
        )
        assert set_resp.status_code == 200
        assert (
            "preferences.chat.default_character_id"
            in set_resp.json().get("applied", [])
        )

        profile_resp = client.get(
            "/api/v1/users/me/profile",
            params={"sections": "preferences"},
            headers=auth_headers,
        )
        assert profile_resp.status_code == 200
        preferences = profile_resp.json().get("preferences", {})
        assert preferences.get("preferences.chat.default_character_id") == "char-123"

        clear_resp = client.patch(
            "/api/v1/users/me/profile",
            headers=auth_headers,
            json={
                "updates": [
                    {
                        "key": "preferences.chat.default_character_id",
                        "value": None,
                    }
                ]
            },
        )
        assert clear_resp.status_code == 200
        assert (
            "preferences.chat.default_character_id"
            in clear_resp.json().get("applied", [])
        )

        profile_after_clear = client.get(
            "/api/v1/users/me/profile",
            params={"sections": "preferences"},
            headers=auth_headers,
        )
        assert profile_after_clear.status_code == 200
        cleared_preferences = profile_after_clear.json().get("preferences", {})

    assert "preferences.chat.default_character_id" not in cleared_preferences


def test_user_profile_update_default_character_preference_type_validation(
    auth_headers,
) -> None:
    with TestClient(app) as client:
        resp = client.patch(
            "/api/v1/users/me/profile",
            headers=auth_headers,
            json={
                "updates": [
                    {
                        "key": "preferences.chat.default_character_id",
                        "value": 123,
                    }
                ]
            },
        )
        assert resp.status_code == 422
        payload = resp.json()
        assert payload.get("error_code") == "profile_update_invalid"
        assert payload.get("errors")


def test_admin_profile_update_storage_quota(auth_headers) -> None:
    with TestClient(app) as client:
        user_id = _get_user_id(client, auth_headers)
        warm_resp = client.get(
            f"/api/v1/admin/users/{user_id}/profile",
            params={"sections": "quotas"},
            headers=auth_headers,
        )
        assert warm_resp.status_code == 200

        resp = client.patch(
            f"/api/v1/admin/users/{user_id}/profile",
            headers=auth_headers,
            json={
                "updates": [
                    {"key": "limits.storage_quota_mb", "value": 4096},
                ]
            },
        )
        assert resp.status_code == 200
        payload = resp.json()
        assert "limits.storage_quota_mb" in payload["applied"]

        profile_resp = client.get(
            f"/api/v1/admin/users/{user_id}/profile",
            params={"sections": "quotas"},
            headers=auth_headers,
        )
        assert profile_resp.status_code == 200
        profile = profile_resp.json()

    assert profile.get("quotas", {}).get("storage_quota_mb") == 4096


def test_user_profile_update_no_updates(auth_headers) -> None:
    with TestClient(app) as client:
        resp = client.patch(
            "/api/v1/users/me/profile",
            headers=auth_headers,
            json={"updates": []},
        )
        assert resp.status_code == 400
        payload = resp.json()
        assert payload.get("error_code") == "profile_update_invalid"
        assert payload.get("errors")


def test_user_profile_update_version_conflict(auth_headers) -> None:
    with TestClient(app) as client:
        resp = client.patch(
            "/api/v1/users/me/profile",
            headers=auth_headers,
            json={
                "profile_version": "2000-01-01T00:00:00Z",
                "updates": [
                    {"key": "preferences.ui.theme", "value": "midnight"},
                ],
            },
        )
        assert resp.status_code == 409
        payload = resp.json()
        assert payload.get("error_code") == "profile_version_mismatch"
        assert payload.get("errors")


def test_user_profile_update_unknown_key(auth_headers) -> None:
    with TestClient(app) as client:
        resp = client.patch(
            "/api/v1/users/me/profile",
            headers=auth_headers,
            json={
                "updates": [
                    {"key": "preferences.ui.unknown", "value": "oops"},
                ],
            },
        )
        assert resp.status_code == 400
        payload = resp.json()
        assert payload.get("error_code") == "profile_update_unknown_key"
        assert payload.get("errors")


def test_user_profile_update_forbidden_key(auth_headers) -> None:
    with TestClient(app) as client:
        resp = client.patch(
            "/api/v1/users/me/profile",
            headers=auth_headers,
            json={
                "updates": [
                    {"key": "limits.storage_quota_mb", "value": 1024},
                ],
            },
        )
        assert resp.status_code == 403
        payload = resp.json()
        assert payload.get("error_code") == "profile_update_forbidden"
        assert payload.get("errors")


def test_user_profile_update_invalid_value(auth_headers) -> None:
    with TestClient(app) as client:
        resp = client.patch(
            "/api/v1/users/me/profile",
            headers=auth_headers,
            json={
                "updates": [
                    {"key": "preferences.ui.theme", "value": 123},
                ],
            },
        )
        assert resp.status_code == 422
        payload = resp.json()
        assert payload.get("error_code") == "profile_update_invalid"
        assert payload.get("errors")


def test_admin_profile_update_org_role(auth_headers) -> None:
    with TestClient(app) as client:
        user_id = _get_user_id(client, auth_headers)
        suffix = uuid.uuid4().hex[:8]

        async def _setup():
            org = await create_organization(name=f"Profile Org {suffix}", owner_user_id=None)
            await add_org_member(
                org_id=int(org["id"]),
                user_id=user_id,
                role="member",
                context=_BOOTSTRAP_MEMBERSHIP_CONTEXT,
            )
            return int(org["id"])

        org_id = _run_async(_setup())

        resp = client.patch(
            f"/api/v1/admin/users/{user_id}/profile",
            headers=auth_headers,
            json={
                "updates": [
                    {"key": "memberships.orgs.role", "value": {"org_id": org_id, "role": "admin"}}
                ]
            },
        )
        assert resp.status_code == 200
        assert "memberships.orgs.role" in resp.json().get("applied", [])

        async def _fetch_roles():
            return await list_org_memberships_for_user(user_id)

        orgs = _run_async(_fetch_roles())
        target = next(item for item in orgs if int(item.get("org_id")) == org_id)
        assert target.get("role") == "admin"


def test_admin_profile_update_team_role(auth_headers) -> None:
    with TestClient(app) as client:
        user_id = _get_user_id(client, auth_headers)
        suffix = uuid.uuid4().hex[:8]

        async def _setup():
            org = await create_organization(name=f"Profile Team Org {suffix}", owner_user_id=None)
            await add_org_member(
                org_id=int(org["id"]),
                user_id=user_id,
                role="member",
                context=_BOOTSTRAP_MEMBERSHIP_CONTEXT,
            )
            team = await create_team(org_id=int(org["id"]), name=f"Team {suffix}")
            await add_team_member(
                team_id=int(team["id"]),
                user_id=user_id,
                role="member",
                context=_BOOTSTRAP_MEMBERSHIP_CONTEXT,
            )
            return int(team["id"])

        team_id = _run_async(_setup())

        resp = client.patch(
            f"/api/v1/admin/users/{user_id}/profile",
            headers=auth_headers,
            json={
                "updates": [
                    {"key": "memberships.teams.role", "value": {"team_id": team_id, "role": "lead"}}
                ]
            },
        )
        assert resp.status_code == 200
        assert "memberships.teams.role" in resp.json().get("applied", [])

        async def _fetch_team_roles():
            return await list_memberships_for_user(user_id)

        teams = _run_async(_fetch_team_roles())
        target = next(item for item in teams if int(item.get("team_id")) == team_id)
        assert target.get("role") == "lead"


def test_admin_profile_update_team_member_add_remove(auth_headers) -> None:
    with TestClient(app) as client:
        user_id = _get_user_id(client, auth_headers)
        suffix = uuid.uuid4().hex[:8]

        async def _setup():
            org = await create_organization(name=f"Profile Team Org 2 {suffix}", owner_user_id=None)
            await add_org_member(
                org_id=int(org["id"]),
                user_id=user_id,
                role="member",
                context=_BOOTSTRAP_MEMBERSHIP_CONTEXT,
            )
            team_a = await create_team(org_id=int(org["id"]), name=f"Team A {suffix}")
            team_b = await create_team(org_id=int(org["id"]), name=f"Team B {suffix}")
            await add_team_member(
                team_id=int(team_a["id"]),
                user_id=user_id,
                role="member",
                context=_BOOTSTRAP_MEMBERSHIP_CONTEXT,
            )
            return int(team_a["id"]), int(team_b["id"])

        team_a_id, team_b_id = _run_async(_setup())

        resp = client.patch(
            f"/api/v1/admin/users/{user_id}/profile",
            headers=auth_headers,
            json={
                "updates": [
                    {
                        "key": "memberships.teams.member",
                        "value": {"team_id": team_b_id, "action": "add", "role": "member"},
                    }
                ]
            },
        )
        assert resp.status_code == 200
        assert "memberships.teams.member" in resp.json().get("applied", [])

        async def _list_memberships():
            return await list_memberships_for_user(user_id)

        memberships = _run_async(_list_memberships())
        team_ids = {int(item.get("team_id")) for item in memberships}
        assert team_b_id in team_ids
        before_remove = _get_profile_version(client, auth_headers)

        resp = client.patch(
            f"/api/v1/admin/users/{user_id}/profile",
            headers=auth_headers,
            json={
                "updates": [
                    {
                        "key": "memberships.teams.member",
                        "value": {"team_id": team_a_id, "action": "remove"},
                    }
                ]
            },
        )
        assert resp.status_code == 200
        assert "memberships.teams.member" in resp.json().get("applied", [])

        memberships = _run_async(_list_memberships())
        team_ids = {int(item.get("team_id")) for item in memberships}
        assert team_a_id not in team_ids
        assert _get_profile_version(client, auth_headers) > before_remove


def test_user_profile_update_rejects_inactive_user(auth_headers, monkeypatch) -> None:
    from tldw_Server_API.app.api.v1.endpoints import users as users_endpoints

    async def _fake_resolve_user_context(_principal, *, allow_missing: bool = False):
        del allow_missing
        return {
            "id": 1,
            "username": "inactive-user",
            "email": "inactive@example.invalid",
            "role": "user",
            "is_active": False,
            "is_verified": True,
            "storage_quota_mb": 5120,
            "storage_used_mb": 0.0,
            "created_at": datetime.utcnow(),
            "last_login": None,
        }

    monkeypatch.setattr(users_endpoints, "_resolve_user_context", _fake_resolve_user_context)

    with TestClient(app) as client:
        resp = client.patch(
            "/api/v1/users/me/profile",
            headers=auth_headers,
            json={
                "updates": [
                    {"key": "preferences.ui.theme", "value": "paper"},
                ]
            },
        )

    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_user_profile_update_delegates_to_command_service(monkeypatch) -> None:
    from tldw_Server_API.app.api.v1.endpoints import users as users_endpoints

    version = datetime(2026, 1, 1, tzinfo=timezone.utc)
    write_conn = object()
    captured: dict[str, object] = {}

    async def _fake_resolve_user_context(_principal, *, allow_missing: bool = False):
        del allow_missing
        return {
            "id": 7,
            "username": "delegated-user",
            "email": "delegated@example.invalid",
            "role": "user",
            "is_active": True,
            "is_verified": True,
            "storage_quota_mb": 5120,
            "storage_used_mb": 0.0,
            "created_at": version,
            "last_login": None,
        }

    async def _fake_pool():
        return object()

    class _CommandService:
        def __init__(self, *, db_pool) -> None:
            captured["db_pool"] = db_pool

        async def apply(self, command, *, db_conn, scope):
            captured["command"] = command
            captured["db_conn"] = db_conn
            captured["scope"] = scope
            return LegacyProfileCommandResult(
                profile_version=version,
                applied=("preferences.ui.theme",),
            )

    class _DirectUpdateServiceTrap:
        def __init__(self, *_args, **_kwargs) -> None:
            raise AssertionError("route must delegate to ProfileCommandService")

    async def _audit(*_args, **_kwargs) -> None:
        captured["audit"] = _kwargs

    monkeypatch.setattr(users_endpoints, "_resolve_user_context", _fake_resolve_user_context)
    monkeypatch.setattr(users_endpoints, "get_db_pool", _fake_pool)
    monkeypatch.setattr(users_endpoints, "ProfileCommandService", _CommandService, raising=False)
    monkeypatch.setattr(
        users_endpoints,
        "UserProfileUpdateService",
        _DirectUpdateServiceTrap,
        raising=False,
    )
    monkeypatch.setattr(users_endpoints, "_emit_user_profile_audit_event", _audit)

    principal = AuthPrincipal(
        kind="user",
        user_id=7,
        username="delegated-user",
        roles=["user"],
        permissions=[],
        is_admin=False,
        org_ids=[],
        team_ids=[],
        active_org_id=None,
        active_team_id=None,
    )
    payload = UserProfileUpdateRequest(
        profile_version=version,
        updates=[UserProfileUpdateEntry(key="preferences.ui.theme", value="paper")],
    )

    response = await users_endpoints.update_current_user_profile(
        payload,
        http_request=object(),
        principal=principal,
        db=write_conn,
    )

    command = captured["command"]
    assert command.actor_user_id == 7
    assert command.target_user_id == 7
    assert command.updates == (("preferences.ui.theme", "paper"),)
    assert command.roles == frozenset({"user"})
    assert command.dry_run is False
    assert command.expected_profile_version == version
    assert command.contract_mode == ProfileContractMode.LEGACY_V1
    assert captured["db_conn"] is write_conn
    assert captured["scope"] is None
    assert response.profile_version == version
    assert response.applied == ["preferences.ui.theme"]
    assert response.skipped == []
    assert captured["audit"]["applied_count"] == 1


@pytest.mark.asyncio
async def test_user_profile_v1_forbidden_command_result_has_no_audit(monkeypatch) -> None:
    from tldw_Server_API.app.api.v1.endpoints import users as users_endpoints

    audit_calls: list[dict[str, object]] = []

    async def _fake_resolve_user_context(_principal, *, allow_missing: bool = False):
        del allow_missing
        return {"id": 7, "is_active": True, "is_verified": True}

    async def _fake_pool():
        return object()

    class _CommandService:
        def __init__(self, *, db_pool) -> None:
            del db_pool

        async def apply(self, command, *, db_conn, scope):
            del command, db_conn, scope
            return LegacyProfileCommandResult(
                status_code=403,
                error_code="profile_update_forbidden",
                detail="Caller cannot edit one or more fields",
                skipped=(
                    {"key": "memberships.orgs.role", "message": "forbidden"},
                ),
            )

    async def _audit(*_args, **kwargs) -> None:
        audit_calls.append(kwargs)

    monkeypatch.setattr(users_endpoints, "_resolve_user_context", _fake_resolve_user_context)
    monkeypatch.setattr(users_endpoints, "get_db_pool", _fake_pool)
    monkeypatch.setattr(users_endpoints, "ProfileCommandService", _CommandService)
    monkeypatch.setattr(users_endpoints, "_emit_user_profile_audit_event", _audit)

    response = await users_endpoints.update_current_user_profile(
        UserProfileUpdateRequest(
            updates=[
                UserProfileUpdateEntry(
                    key="memberships.orgs.role",
                    value={"org_id": 3, "role": "admin"},
                )
            ]
        ),
        http_request=object(),
        principal=AuthPrincipal(
            kind="user",
            user_id=7,
            username="profile-user",
            roles=["user"],
            permissions=[],
            is_admin=False,
            org_ids=[],
            team_ids=[],
        ),
        db=object(),
    )

    assert response.status_code == 403
    assert json.loads(response.body) == {
        "error_code": "profile_update_forbidden",
        "detail": "Caller cannot edit one or more fields",
        "errors": [{"key": "memberships.orgs.role", "message": "forbidden"}],
    }
    assert audit_calls == []


def test_admin_profile_update_audio_limits(auth_headers) -> None:
    with TestClient(app) as client:
        user_id = _get_user_id(client, auth_headers)
        resp = client.patch(
            f"/api/v1/admin/users/{user_id}/profile",
            headers=auth_headers,
            json={
                "updates": [
                    {"key": "limits.audio_daily_minutes", "value": 120},
                    {"key": "limits.audio_concurrent_jobs", "value": 4},
                ]
            },
        )
        assert resp.status_code == 200
        payload = resp.json()
        assert "limits.audio_daily_minutes" in payload.get("applied", [])
        assert "limits.audio_concurrent_jobs" in payload.get("applied", [])

        profile_resp = client.get(
            f"/api/v1/admin/users/{user_id}/profile",
            params={"sections": "quotas"},
            headers=auth_headers,
        )
        assert profile_resp.status_code == 200
        quotas = profile_resp.json().get("quotas", {})
        audio = quotas.get("audio", {})
        assert audio.get("daily_minutes_limit") == 120
        assert audio.get("concurrent_jobs_limit") == 4


def test_admin_profile_update_evaluations_limits(auth_headers) -> None:
    with TestClient(app) as client:
        user_id = _get_user_id(client, auth_headers)
        resp = client.patch(
            f"/api/v1/admin/users/{user_id}/profile",
            headers=auth_headers,
            json={
                "updates": [
                    {"key": "limits.evaluations_per_minute", "value": 42},
                    {"key": "limits.evaluations_per_day", "value": 900},
                ]
            },
        )
        assert resp.status_code == 200
        payload = resp.json()
        assert "limits.evaluations_per_minute" in payload.get("applied", [])
        assert "limits.evaluations_per_day" in payload.get("applied", [])

        profile_resp = client.get(
            f"/api/v1/admin/users/{user_id}/profile",
            params={"sections": "quotas"},
            headers=auth_headers,
        )
        assert profile_resp.status_code == 200
        quotas = profile_resp.json().get("quotas", {})
        evaluations = quotas.get("evaluations", {})
        limits = evaluations.get("limits", {})
        assert limits.get("per_minute", {}).get("evaluations") == 42
        assert limits.get("daily", {}).get("evaluations") == 900


def test_admin_profile_update_identity_locked(auth_headers) -> None:
    with TestClient(app) as client:
        profile_resp = client.get("/api/v1/users/me/profile", headers=auth_headers)
        assert profile_resp.status_code == 200
        user = profile_resp.json().get("user", {})
        user_id = int(user.get("id"))
        username = user.get("username")
        assert username

        resp = client.patch(
            f"/api/v1/admin/users/{user_id}/profile",
            headers=auth_headers,
            json={"updates": [{"key": "identity.is_locked", "value": True}]},
        )
        assert resp.status_code == 200
        assert "identity.is_locked" in resp.json().get("applied", [])

        limiter = get_rate_limiter()
        is_locked, _ = _run_async(limiter.check_lockout(str(username), attempt_type="login"))
        assert is_locked is True

        resp = client.patch(
            f"/api/v1/admin/users/{user_id}/profile",
            headers=auth_headers,
            json={"updates": [{"key": "identity.is_locked", "value": False}]},
        )
        assert resp.status_code == 200
        assert "identity.is_locked" in resp.json().get("applied", [])

        is_locked, _ = _run_async(limiter.check_lockout(str(username), attempt_type="login"))
        assert is_locked is False
