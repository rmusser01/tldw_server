from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest

from tldw_Server_API.app.core.UserProfiles.command_service import ProfileCommandService
from tldw_Server_API.app.core.UserProfiles.contracts import ProfileUpdateCommand
from tldw_Server_API.app.core.UserProfiles.response_mappers import (
    LegacyProfileCommandResult,
)
from tldw_Server_API.app.core.UserProfiles.update_service import UpdateResult


class _ProfileService:
    def __init__(self) -> None:
        self.calls: list[tuple[Any, bool]] = []
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
