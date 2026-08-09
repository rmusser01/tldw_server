from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest

from tldw_Server_API.app.core.AuthNZ.membership_writer import (
    MembershipAuthorizationError,
)
from tldw_Server_API.app.core.UserProfiles.command_service import ProfileCommandService
from tldw_Server_API.app.core.UserProfiles.contracts import ProfileUpdateCommand
from tldw_Server_API.app.core.UserProfiles.response_mappers import (
    LegacyProfileCommandResult,
)
from tldw_Server_API.app.core.UserProfiles.update_service import UpdateResult


class _ProfileService:
    def __init__(self) -> None:
        self.calls: list[tuple[Any, bool]] = []
        self.lock_calls: list[tuple[Any, tuple[int, ...]]] = []
        self.initial = datetime(2026, 1, 1, tzinfo=timezone.utc)
        self.locked = datetime(2026, 1, 2, tzinfo=timezone.utc)
        self.after_write = datetime(2026, 1, 3, tzinfo=timezone.utc)
        self._unlocked_calls = 0

    async def get_profile_version(self, *, user_id: int, db_conn=None, lock_user: bool = False):
        self.calls.append((db_conn, lock_user))
        if lock_user:
            return self.locked
        self._unlocked_calls += 1
        return self.initial if self._unlocked_calls == 1 else self.after_write

    async def lock_profile_users(self, *, user_ids: tuple[int, ...], db_conn):
        self.lock_calls.append((db_conn, user_ids))
        return dict.fromkeys(user_ids, self.locked)

    def versions_match(self, current, expected) -> bool:
        return current == expected


class _Planner:
    def __init__(self, result: UpdateResult | None = None) -> None:
        self.result = result
        self.calls: list[dict[str, Any]] = []

    async def plan(self, command, *, db_conn, scope):
        self.calls.append({"command": command, "db_conn": db_conn, "scope": scope})
        if self.result is not None:
            return self.result
        return UpdateResult(applied=[key for key, _value in command.updates])


class _Executor:
    def __init__(self) -> None:
        self.called = False
        self.calls: list[dict[str, Any]] = []

    async def apply_updates(self, **kwargs):
        self.called = True
        self.calls.append(kwargs)
        return UpdateResult(applied=["preferences.ui.theme"])


class _RecordingConnection:
    def __init__(self) -> None:
        self.events: list[str] = []


@pytest.mark.asyncio
async def test_command_service_rechecks_version_before_apply() -> None:
    profile_service = _ProfileService()
    executor = _Executor()
    command_service = ProfileCommandService(
        db_pool=object(),
        profile_service=profile_service,
        planner=_Planner(),
        executor=executor,
    )
    write_conn = object()
    command = ProfileUpdateCommand(
        actor_user_id=7,
        target_user_id=7,
        updates=(("preferences.ui.theme", "paper"),),
        roles=frozenset({"user"}),
        dry_run=False,
        expected_profile_version=profile_service.initial,
    )

    result = await command_service.apply(command, db_conn=write_conn, scope=None)

    assert result.status_code == 409
    assert result.error_code == "profile_version_mismatch"
    assert executor.called is False


@pytest.mark.asyncio
async def test_command_service_locks_actor_and_target_once_in_canonical_order() -> None:
    profile_service = _ProfileService()
    profile_service.locked = profile_service.initial
    executor = _Executor()
    command_service = ProfileCommandService(
        db_pool=object(),
        profile_service=profile_service,
        planner=_Planner(),
        executor=executor,
    )
    write_conn = _RecordingConnection()
    command = ProfileUpdateCommand(
        actor_user_id=9,
        target_user_id=7,
        updates=(("memberships.orgs.role", {"org_id": 3, "role": "admin"}),),
        roles=frozenset({"admin"}),
        dry_run=False,
        expected_profile_version=profile_service.initial,
    )

    result = await command_service.apply(command, db_conn=write_conn, scope=object())

    assert result.status_code == 200
    assert profile_service.lock_calls == [(write_conn, (7, 9))]
    assert all(lock_user is False for _conn, lock_user in profile_service.calls)
    assert "prelocked_user_ids" not in executor.calls[0]
    assert write_conn.events == []


@pytest.mark.asyncio
async def test_versionless_mixed_membership_command_locks_before_executor() -> None:
    events: list[str] = []

    class _OrderedProfileService(_ProfileService):
        async def lock_profile_users(self, *, user_ids: tuple[int, ...], db_conn):
            events.append(f"lock:{user_ids}")
            return await super().lock_profile_users(user_ids=user_ids, db_conn=db_conn)

    class _OrderedExecutor(_Executor):
        async def apply_updates(self, **kwargs):
            events.append("executor")
            return await super().apply_updates(**kwargs)

    profile_service = _OrderedProfileService()
    executor = _OrderedExecutor()
    command_service = ProfileCommandService(
        db_pool=object(),
        profile_service=profile_service,
        planner=_Planner(),
        executor=executor,
    )
    write_conn = _RecordingConnection()
    command = ProfileUpdateCommand(
        actor_user_id=9,
        target_user_id=7,
        updates=(
            ("preferences.ui.theme", "paper"),
            ("memberships.orgs.role", {"org_id": 3, "role": "admin"}),
        ),
        roles=frozenset({"admin"}),
        dry_run=False,
    )

    result = await command_service.apply(command, db_conn=write_conn, scope=object())

    assert result.status_code == 200
    assert profile_service.lock_calls == [(write_conn, (7, 9))]
    assert events == ["lock:(7, 9)", "executor"]
    assert "prelocked_user_ids" not in executor.calls[0]
    assert write_conn.events == []


@pytest.mark.asyncio
async def test_command_service_dry_run_returns_preflight_without_executor() -> None:
    profile_service = _ProfileService()
    executor = _Executor()
    command_service = ProfileCommandService(
        db_pool=object(),
        profile_service=profile_service,
        planner=_Planner(),
        executor=executor,
    )
    command = ProfileUpdateCommand(
        actor_user_id=7,
        target_user_id=7,
        updates=(("preferences.ui.theme", "paper"),),
        roles=frozenset({"user"}),
        dry_run=True,
    )

    result = await command_service.apply(command, db_conn=object(), scope=None)

    assert result.status_code == 200
    assert result.profile_version == profile_service.initial
    assert result.applied == ("preferences.ui.theme",)
    assert result.skipped == ()
    assert executor.called is False


@pytest.mark.asyncio
async def test_command_service_preflight_skip_returns_error_without_executor() -> None:
    executor = _Executor()
    command_service = ProfileCommandService(
        db_pool=object(),
        profile_service=_ProfileService(),
        planner=_Planner(
            UpdateResult(
                applied=["preferences.ui.theme"],
                skipped=[{"key": "identity.email", "message": "invalid_email"}],
            )
        ),
        executor=executor,
    )
    command = ProfileUpdateCommand(
        actor_user_id=7,
        target_user_id=7,
        updates=(("identity.email", "not-an-email"),),
        roles=frozenset({"user"}),
        dry_run=False,
    )

    result = await command_service.apply(command, db_conn=object(), scope=None)

    assert result.status_code == 422
    assert result.error_code == "profile_update_invalid"
    assert result.applied == ("preferences.ui.theme",)
    assert result.skipped == ({"key": "identity.email", "message": "invalid_email"},)
    assert executor.called is False


@pytest.mark.asyncio
async def test_command_service_unknown_key_skip_maps_to_legacy_bad_request() -> None:
    executor = _Executor()
    command_service = ProfileCommandService(
        db_pool=object(),
        profile_service=_ProfileService(),
        planner=_Planner(
            UpdateResult(
                skipped=[{"key": "preferences.ui.unknown", "message": "unknown_key"}],
            )
        ),
        executor=executor,
    )
    command = ProfileUpdateCommand(
        actor_user_id=7,
        target_user_id=7,
        updates=(("preferences.ui.unknown", "oops"),),
        roles=frozenset({"user"}),
        dry_run=False,
    )

    result = await command_service.apply(command, db_conn=object(), scope=None)

    assert result.status_code == 400
    assert result.error_code == "profile_update_unknown_key"
    assert result.detail == "One or more keys are not recognized"
    assert result.skipped == (
        {"key": "preferences.ui.unknown", "message": "unknown_key"},
    )
    assert executor.called is False


@pytest.mark.asyncio
async def test_command_service_forbidden_skip_maps_to_legacy_forbidden() -> None:
    executor = _Executor()
    command_service = ProfileCommandService(
        db_pool=object(),
        profile_service=_ProfileService(),
        planner=_Planner(
            UpdateResult(
                skipped=[{"key": "limits.storage_quota_mb", "message": "forbidden"}],
            )
        ),
        executor=executor,
    )
    command = ProfileUpdateCommand(
        actor_user_id=7,
        target_user_id=7,
        updates=(("limits.storage_quota_mb", 1024),),
        roles=frozenset({"user"}),
        dry_run=False,
    )

    result = await command_service.apply(command, db_conn=object(), scope=None)

    assert result.status_code == 403
    assert result.error_code == "profile_update_forbidden"
    assert result.detail == "Caller cannot edit one or more fields"
    assert result.skipped == (
        {"key": "limits.storage_quota_mb", "message": "forbidden"},
    )
    assert executor.called is False


@pytest.mark.asyncio
async def test_command_service_maps_writer_authorization_drift_without_savepoint() -> None:
    class _AuthorizationDriftExecutor:
        async def apply_updates(self, **kwargs):
            kwargs["db_conn"].events.append("executor")
            raise MembershipAuthorizationError()

    write_conn = _RecordingConnection()
    command_service = ProfileCommandService(
        db_pool=object(),
        profile_service=_ProfileService(),
        planner=_Planner(),
        executor=_AuthorizationDriftExecutor(),
    )
    command = ProfileUpdateCommand(
        actor_user_id=9,
        target_user_id=7,
        updates=(
            ("preferences.ui.theme", "paper"),
            ("memberships.orgs.role", {"org_id": 3, "role": "admin"}),
        ),
        roles=frozenset({"admin"}),
        dry_run=False,
    )

    result = await command_service.apply(command, db_conn=write_conn, scope=object())

    assert result.status_code == 403
    assert result.error_code == "profile_update_forbidden"
    assert result.detail == "Caller cannot edit one or more fields"
    assert result.applied == ()
    assert result.skipped == (
        {"key": "memberships.orgs.role", "message": "forbidden"},
    )
    assert write_conn.events == ["executor"]


@pytest.mark.asyncio
async def test_command_service_preserves_non_authorization_executor_errors() -> None:
    class _FailingExecutor:
        async def apply_updates(self, **kwargs):
            kwargs["db_conn"].events.append("executor")
            raise RuntimeError("database failure")

    write_conn = _RecordingConnection()
    command_service = ProfileCommandService(
        db_pool=object(),
        profile_service=_ProfileService(),
        planner=_Planner(),
        executor=_FailingExecutor(),
    )
    command = ProfileUpdateCommand(
        actor_user_id=9,
        target_user_id=7,
        updates=(
            ("preferences.ui.theme", "paper"),
            ("memberships.orgs.role", {"org_id": 3, "role": "admin"}),
        ),
        roles=frozenset({"admin"}),
        dry_run=False,
    )

    with pytest.raises(RuntimeError, match="database failure"):
        await command_service.apply(command, db_conn=write_conn, scope=object())

    assert write_conn.events == ["executor"]


@pytest.mark.asyncio
async def test_command_service_generic_preflight_skip_stays_legacy_invalid() -> None:
    executor = _Executor()
    command_service = ProfileCommandService(
        db_pool=object(),
        profile_service=_ProfileService(),
        planner=_Planner(
            UpdateResult(
                skipped=[{"key": "preferences.ui.theme", "message": "type_mismatch"}],
            )
        ),
        executor=executor,
    )
    command = ProfileUpdateCommand(
        actor_user_id=7,
        target_user_id=7,
        updates=(("preferences.ui.theme", 123),),
        roles=frozenset({"user"}),
        dry_run=False,
    )

    result = await command_service.apply(command, db_conn=object(), scope=None)

    assert result.status_code == 422
    assert result.error_code == "profile_update_invalid"
    assert result.detail == "One or more updates failed validation"
    assert executor.called is False


@pytest.mark.asyncio
async def test_command_service_team_not_found_skip_maps_without_executor() -> None:
    executor = _Executor()
    command_service = ProfileCommandService(
        db_pool=object(),
        profile_service=_ProfileService(),
        planner=_Planner(
            UpdateResult(
                skipped=[
                    {"key": "memberships.teams.role", "message": "team_not_found"}
                ],
            )
        ),
        executor=executor,
    )
    command = ProfileUpdateCommand(
        actor_user_id=7,
        target_user_id=9,
        updates=(("memberships.teams.role", {"team_id": 42, "role": "member"}),),
        roles=frozenset({"admin"}),
        dry_run=False,
    )

    result = await command_service.apply(command, db_conn=object(), scope=None)

    assert result.status_code == 404
    assert result.error_code == "profile_update_not_found"
    assert result.detail == "Target resource not found"
    assert result.skipped == (
        {"key": "memberships.teams.role", "message": "team_not_found"},
    )
    assert executor.called is False


@pytest.mark.asyncio
async def test_command_service_invalid_payload_skip_maps_without_executor() -> None:
    executor = _Executor()
    command_service = ProfileCommandService(
        db_pool=object(),
        profile_service=_ProfileService(),
        planner=_Planner(
            UpdateResult(
                skipped=[
                    {"key": "memberships.teams.role", "message": "invalid_payload"}
                ],
            )
        ),
        executor=executor,
    )
    command = ProfileUpdateCommand(
        actor_user_id=7,
        target_user_id=9,
        updates=(("memberships.teams.role", "not-a-payload"),),
        roles=frozenset({"admin"}),
        dry_run=False,
    )

    result = await command_service.apply(command, db_conn=object(), scope=None)

    assert result.status_code == 400
    assert result.error_code == "profile_update_invalid"
    assert result.detail == "Invalid profile update payload"
    assert result.skipped == (
        {"key": "memberships.teams.role", "message": "invalid_payload"},
    )
    assert executor.called is False


@pytest.mark.asyncio
async def test_command_service_dry_run_rejects_stale_version_before_planning() -> None:
    profile_service = _ProfileService()
    planner = _Planner()
    executor = _Executor()
    command_service = ProfileCommandService(
        db_pool=object(),
        profile_service=profile_service,
        planner=planner,
        executor=executor,
    )
    command = ProfileUpdateCommand(
        actor_user_id=7,
        target_user_id=7,
        updates=(("preferences.ui.theme", "paper"),),
        roles=frozenset({"user"}),
        dry_run=True,
        expected_profile_version=datetime(2000, 1, 1, tzinfo=timezone.utc),
    )

    result = await command_service.apply(command, db_conn=object(), scope=None)

    assert result.status_code == 409
    assert result.profile_version == profile_service.initial
    assert result.error_code == "profile_version_mismatch"
    assert result.skipped == ({"key": "profile_version", "message": "mismatch"},)
    assert planner.calls == []
    assert executor.called is False


@pytest.mark.asyncio
async def test_command_service_stale_version_wins_over_invalid_preflight_candidate() -> None:
    profile_service = _ProfileService()
    planner = _Planner(
        UpdateResult(
            skipped=[{"key": "preferences.ui.unknown", "message": "unknown_key"}],
        )
    )
    executor = _Executor()
    command_service = ProfileCommandService(
        db_pool=object(),
        profile_service=profile_service,
        planner=planner,
        executor=executor,
    )
    command = ProfileUpdateCommand(
        actor_user_id=7,
        target_user_id=7,
        updates=(("preferences.ui.unknown", "oops"),),
        roles=frozenset({"user"}),
        dry_run=False,
        expected_profile_version=datetime(2000, 1, 1, tzinfo=timezone.utc),
    )

    result = await command_service.apply(command, db_conn=object(), scope=None)

    assert result.status_code == 409
    assert result.profile_version == profile_service.initial
    assert result.error_code == "profile_version_mismatch"
    assert planner.calls == []
    assert executor.called is False


def test_legacy_command_result_error_version_and_skipped_are_defensive() -> None:
    skipped = {"key": "identity.email", "message": "invalid_email"}
    result = LegacyProfileCommandResult(
        status_code=422,
        skipped=(skipped,),
        error_code="profile_update_invalid",
        detail="One or more updates failed validation",
    )

    skipped["message"] = "mutated"

    assert result.profile_version is None
    assert result.skipped == ({"key": "identity.email", "message": "invalid_email"},)
    with pytest.raises(TypeError):
        result.skipped[0]["message"] = "mutated"


@pytest.mark.asyncio
async def test_command_service_successful_write_returns_executor_result_and_version() -> None:
    profile_service = _ProfileService()
    executor = _Executor()
    scope = object()
    write_conn = object()
    command_service = ProfileCommandService(
        db_pool=object(),
        profile_service=profile_service,
        planner=_Planner(),
        executor=executor,
    )
    command = ProfileUpdateCommand(
        actor_user_id=7,
        target_user_id=9,
        updates=(("preferences.ui.theme", "paper"),),
        roles=frozenset({"user"}),
        dry_run=False,
    )

    result = await command_service.apply(command, db_conn=write_conn, scope=scope)

    assert result.status_code == 200
    assert result.profile_version == profile_service.after_write
    assert result.applied == ("preferences.ui.theme",)
    assert result.skipped == ()
    assert profile_service.calls == [(write_conn, False), (write_conn, False)]
    assert executor.calls == [
        {
            "user_id": 9,
            "updates": (("preferences.ui.theme", "paper"),),
            "roles": {"user"},
            "dry_run": False,
            "db_conn": write_conn,
            "updated_by": 7,
            "scope": scope,
        }
    ]
