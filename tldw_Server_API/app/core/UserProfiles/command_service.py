"""
Single-update UserProfiles command orchestration.
"""

from __future__ import annotations

from typing import Any

from tldw_Server_API.app.core.AuthNZ.membership_writer import (
    MembershipAuthorizationError,
)
from tldw_Server_API.app.core.UserProfiles.contracts import ProfileUpdateCommand
from tldw_Server_API.app.core.UserProfiles.effects import ProfileEffectDispatcher
from tldw_Server_API.app.core.UserProfiles.error_mapping import (
    classify_legacy_profile_update_skips,
)
from tldw_Server_API.app.core.UserProfiles.planner import ProfileUpdatePlanner
from tldw_Server_API.app.core.UserProfiles.response_mappers import LegacyProfileCommandResult
from tldw_Server_API.app.core.UserProfiles.service import UserProfileService
from tldw_Server_API.app.core.UserProfiles.update_service import (
    ProfileUpdateScope,
    UserProfileUpdateService,
)


class ProfileCommandService:
    def __init__(
        self,
        *,
        db_pool: Any,
        profile_service: Any | None = None,
        planner: Any | None = None,
        executor: Any | None = None,
        effects: ProfileEffectDispatcher | None = None,
    ) -> None:
        self._db_pool = db_pool
        self._profile_service = profile_service or UserProfileService(db_pool)
        self._planner = planner or ProfileUpdatePlanner(db_pool)
        self._executor = executor or UserProfileUpdateService(db_pool)
        self._effects = effects or ProfileEffectDispatcher()

    async def apply(
        self,
        command: ProfileUpdateCommand,
        *,
        db_conn: Any,
        scope: ProfileUpdateScope | None,
    ) -> LegacyProfileCommandResult:
        membership_keys = tuple(
            key for key, _value in command.updates if key.startswith("memberships.")
        )
        current_version = None
        if command.expected_profile_version is not None:
            current_version = await self._profile_service.get_profile_version(
                user_id=command.target_user_id,
                db_conn=db_conn,
            )
            if not self._profile_service.versions_match(
                current_version,
                command.expected_profile_version,
            ):
                return LegacyProfileCommandResult(
                    status_code=409,
                    profile_version=current_version,
                    error_code="profile_version_mismatch",
                    detail="profile_version_mismatch",
                    skipped=({"key": "profile_version", "message": "mismatch"},),
                )

        preflight = await self._planner.plan(command, db_conn=db_conn, scope=scope)
        if preflight.skipped:
            mapped = classify_legacy_profile_update_skips(tuple(preflight.skipped))
            if mapped is None:
                raise RuntimeError(
                    "preflight skipped result could not be classified"
                )
            return LegacyProfileCommandResult(
                status_code=mapped.status_code,
                applied=tuple(preflight.applied),
                skipped=tuple(preflight.skipped),
                error_code=mapped.error_code,
                detail=mapped.detail,
            )

        if current_version is None:
            current_version = await self._profile_service.get_profile_version(
                user_id=command.target_user_id,
                db_conn=db_conn,
            )
        if command.dry_run:
            return LegacyProfileCommandResult(
                profile_version=current_version,
                applied=tuple(preflight.applied),
                skipped=(),
            )

        if command.expected_profile_version is not None or membership_keys:
            lock_user_ids = tuple(
                sorted({command.actor_user_id, command.target_user_id})
            )
            locked_versions = await self._profile_service.lock_profile_users(
                user_ids=lock_user_ids,
                db_conn=db_conn,
            )
            if command.expected_profile_version is not None:
                locked_version = locked_versions[command.target_user_id]
                if not self._profile_service.versions_match(
                    locked_version,
                    command.expected_profile_version,
                ):
                    return LegacyProfileCommandResult(
                        status_code=409,
                        profile_version=locked_version,
                        error_code="profile_version_mismatch",
                        detail="profile_version_mismatch",
                        skipped=({"key": "profile_version", "message": "mismatch"},),
                    )

        executor_kwargs = {
            "user_id": command.target_user_id,
            "updates": command.updates,
            "roles": set(command.roles),
            "dry_run": False,
            "db_conn": db_conn,
            "updated_by": command.actor_user_id,
            "scope": scope,
        }
        if membership_keys:
            try:
                result = await self._executor.apply_updates(**executor_kwargs)
            except MembershipAuthorizationError:
                return LegacyProfileCommandResult(
                    status_code=403,
                    applied=(),
                    skipped=tuple(
                        {"key": key, "message": "forbidden"}
                        for key in membership_keys
                    ),
                    error_code="profile_update_forbidden",
                    detail="Caller cannot edit one or more fields",
                )
        else:
            result = await self._executor.apply_updates(**executor_kwargs)
        current_version = await self._profile_service.get_profile_version(
            user_id=command.target_user_id,
            db_conn=db_conn,
        )
        return LegacyProfileCommandResult(
            profile_version=current_version,
            applied=tuple(result.applied),
            skipped=tuple(result.skipped),
        )
