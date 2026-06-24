"""
UserProfiles update planning.
"""

from __future__ import annotations

from typing import Any

from tldw_Server_API.app.core.UserProfiles.contracts import ProfileUpdateCommand
from tldw_Server_API.app.core.UserProfiles.update_service import (
    ProfileUpdateScope,
    UpdateResult,
    UserProfileUpdateService,
)


class ProfileUpdatePlanner:
    """Build an update plan using current catalog validation without mutating state."""

    def __init__(self, db_pool: Any) -> None:
        self._db_pool = db_pool

    async def plan(
        self,
        command: ProfileUpdateCommand,
        *,
        db_conn: Any,
        scope: ProfileUpdateScope | None,
    ) -> UpdateResult:
        service = UserProfileUpdateService(self._db_pool)
        return await service.apply_updates(
            user_id=command.target_user_id,
            updates=command.updates,
            roles=set(command.roles),
            dry_run=True,
            db_conn=db_conn,
            updated_by=command.actor_user_id,
            scope=scope,
        )
