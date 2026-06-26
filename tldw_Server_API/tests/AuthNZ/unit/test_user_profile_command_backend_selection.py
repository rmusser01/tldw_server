from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest

from tldw_Server_API.app.core.UserProfiles.command_service import ProfileCommandService
from tldw_Server_API.app.core.UserProfiles.contracts import ProfileUpdateCommand
from tldw_Server_API.app.core.UserProfiles.update_service import UpdateResult


class _ProfileService:
    def __init__(self) -> None:
        self.version = datetime(2026, 1, 1, tzinfo=timezone.utc)

    async def get_profile_version(self, **_kwargs: Any) -> datetime:
        return self.version


class _Planner:
    async def plan(self, command, *, db_conn, scope):
        del db_conn, scope
        return UpdateResult(applied=[key for key, _value in command.updates])


class _Executor:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def apply_updates(self, **kwargs: Any) -> UpdateResult:
        self.calls.append(kwargs)
        return UpdateResult(applied=["preferences.ui.theme"])


@pytest.mark.asyncio
async def test_profile_command_service_passes_supplied_transaction_connection_to_executor() -> None:
    executor = _Executor()
    write_conn = object()
    command_service = ProfileCommandService(
        db_pool=object(),
        profile_service=_ProfileService(),
        planner=_Planner(),
        executor=executor,
    )
    command = ProfileUpdateCommand(
        actor_user_id=7,
        target_user_id=7,
        updates=(("preferences.ui.theme", "paper"),),
        roles=frozenset({"user"}),
        dry_run=False,
    )

    await command_service.apply(command, db_conn=write_conn, scope=None)

    assert executor.calls
    assert executor.calls[0]["db_conn"] is write_conn
