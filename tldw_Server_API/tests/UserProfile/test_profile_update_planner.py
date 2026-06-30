from __future__ import annotations

from typing import Any

import pytest

from tldw_Server_API.app.core.UserProfiles.contracts import (
    ProfileContractMode,
    ProfileUpdateCommand,
)
from tldw_Server_API.app.core.UserProfiles.planner import ProfileUpdatePlanner
from tldw_Server_API.app.core.UserProfiles.update_service import (
    ProfileUpdateScope,
    UpdateResult,
)


@pytest.mark.asyncio
async def test_planner_rejects_unknown_key_without_mutation() -> None:
    planner = ProfileUpdatePlanner(db_pool=object())
    command = ProfileUpdateCommand(
        actor_user_id=7,
        target_user_id=7,
        updates=(("preferences.ui.missing", "paper"),),
        roles=frozenset({"user"}),
        dry_run=True,
        contract_mode=ProfileContractMode.LEGACY_V1,
    )

    result = await planner.plan(
        command,
        db_conn=object(),
        scope=ProfileUpdateScope(actor_user_id=7),
    )

    assert result.applied == []
    assert result.skipped == [
        {"key": "preferences.ui.missing", "message": "unknown_key"}
    ]


@pytest.mark.asyncio
async def test_planner_accepts_preference_update_without_executing_write() -> None:
    planner = ProfileUpdatePlanner(db_pool=object())
    command = ProfileUpdateCommand(
        actor_user_id=7,
        target_user_id=7,
        updates=(("preferences.ui.theme", "paper"),),
        roles=frozenset({"user"}),
        dry_run=True,
        contract_mode=ProfileContractMode.LEGACY_V1,
    )

    result = await planner.plan(
        command,
        db_conn=object(),
        scope=ProfileUpdateScope(actor_user_id=7),
    )

    assert result.applied == ["preferences.ui.theme"]
    assert result.skipped == []


@pytest.mark.asyncio
async def test_planner_forces_dry_run_when_command_requests_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core.UserProfiles import planner as planner_module

    db_pool = object()
    db_conn = object()
    scope = ProfileUpdateScope(actor_user_id=7)
    calls: list[dict[str, Any]] = []

    class FakeUpdateService:
        def __init__(self, db_pool: Any) -> None:
            self._db_pool = db_pool

        async def apply_updates(
            self,
            *,
            user_id: int,
            updates: tuple[tuple[str, Any], ...],
            roles: set[str],
            dry_run: bool,
            db_conn: Any,
            updated_by: int | None,
            scope: ProfileUpdateScope | None,
        ) -> UpdateResult:
            calls.append(
                {
                    "db_pool": self._db_pool,
                    "user_id": user_id,
                    "updates": updates,
                    "roles": roles,
                    "dry_run": dry_run,
                    "db_conn": db_conn,
                    "updated_by": updated_by,
                    "scope": scope,
                }
            )
            return UpdateResult(applied=["preferences.ui.theme"])

    monkeypatch.setattr(planner_module, "UserProfileUpdateService", FakeUpdateService)

    command = ProfileUpdateCommand(
        actor_user_id=7,
        target_user_id=9,
        updates=(("preferences.ui.theme", "paper"),),
        roles=frozenset({"user"}),
        dry_run=False,
        contract_mode=ProfileContractMode.LEGACY_V1,
    )

    result = await ProfileUpdatePlanner(db_pool=db_pool).plan(
        command,
        db_conn=db_conn,
        scope=scope,
    )

    assert result.applied == ["preferences.ui.theme"]
    assert calls == [
        {
            "db_pool": db_pool,
            "user_id": 9,
            "updates": (("preferences.ui.theme", "paper"),),
            "roles": {"user"},
            "dry_run": True,
            "db_conn": db_conn,
            "updated_by": 7,
            "scope": scope,
        }
    ]
