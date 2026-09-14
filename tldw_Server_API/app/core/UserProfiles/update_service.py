"""
User profile update helpers and validation.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from loguru import logger
from pydantic import EmailStr, TypeAdapter, ValidationError

from tldw_Server_API.app.core.AuthNZ.orgs_teams import (
    add_team_member,
    get_team,
    list_memberships_for_user,
    list_org_memberships_for_user,
    remove_team_member,
    update_org_member_role,
    update_team_member_role,
)
from tldw_Server_API.app.core.AuthNZ.profile_version import (
    UserVersionOwnership,
    VersionedUserWriteGateway,
)
from tldw_Server_API.app.core.AuthNZ.rate_limiter import get_rate_limiter
from tldw_Server_API.app.core.UserProfiles.overrides_repo import UserProfileOverridesRepo
from tldw_Server_API.app.core.UserProfiles.user_profile_catalog import (
    UserProfileCatalogEntry,
    load_user_profile_catalog,
)


@dataclass
class UpdateResult:
    applied: list[str] = field(default_factory=list)
    skipped: list[dict[str, str]] = field(default_factory=list)


@dataclass
class ProfileUpdateScope:
    """Scope context for admin profile updates."""

    actor_user_id: int | None = None
    active_org_id: int | None = None
    active_team_id: int | None = None


@dataclass
class _MembershipContext:
    target_org_roles: dict[int, str] = field(default_factory=dict)
    target_team_roles: dict[int, str] = field(default_factory=dict)
    target_team_orgs: dict[int, int] = field(default_factory=dict)
    actor_org_roles: dict[int, str] = field(default_factory=dict)
    actor_team_roles: dict[int, str] = field(default_factory=dict)


@dataclass
class _CallerOwnedAnchor:
    gateway: VersionedUserWriteGateway
    db_conn: Any
    user_id: int
    version_floor: Any | None = None
    changed: bool = False

    async def capture(self) -> None:
        if self.version_floor is None:
            self.version_floor = await self.gateway.capture_floor(
                self.db_conn,
                user_id=self.user_id,
            )

    def mark_changed(self) -> None:
        self.changed = True

    async def finalize(self) -> None:
        if self.changed and self.version_floor is not None:
            await self.gateway.final_touch(
                self.db_conn,
                user_id=self.user_id,
                version_floor=self.version_floor,
            )


def _is_postgres_backend_for_pool(db_pool: Any) -> bool:
    """Derive backend from DatabasePool state without probing connection methods."""
    return bool(getattr(db_pool, "pool", None))


class UserProfileUpdateService:
    """Apply profile updates with catalog-driven validation."""

    def __init__(self, db_pool):
        self._db_pool = db_pool

    async def apply_updates(
        self,
        *,
        user_id: int,
        updates: Iterable[tuple[str, Any]],
        roles: set[str],
        dry_run: bool,
        db_conn: Any,
        updated_by: int | None,
        scope: ProfileUpdateScope | None = None,
    ) -> UpdateResult:
        catalog = load_user_profile_catalog()
        catalog_map = {entry.key: entry for entry in catalog.entries}
        result = UpdateResult()
        repo_holder: dict[str, UserProfileOverridesRepo | None] = {"repo": None}
        normalized_roles = {str(role).strip().lower() for role in roles if role}
        if "admin" in normalized_roles:
            normalized_roles.update({"org_admin", "team_admin", "platform_admin"})
        is_platform_admin = "platform_admin" in normalized_roles
        updates_list = list(updates)
        membership_context: _MembershipContext | None = None
        if any(str(key).startswith("memberships.") for key, _ in updates_list):
            membership_context = await self._build_membership_context(
                user_id=user_id,
                scope=scope,
                is_platform_admin=is_platform_admin,
            )
        is_postgres_backend = _is_postgres_backend_for_pool(self._db_pool)
        anchor = _CallerOwnedAnchor(
            gateway=VersionedUserWriteGateway(
                "postgres" if is_postgres_backend else "sqlite"
            ),
            db_conn=db_conn,
            user_id=user_id,
        )

        for key, value in updates_list:
            entry = catalog_map.get(key)
            if not entry:
                result.skipped.append({"key": key, "message": "unknown_key"})
                continue

            if not _can_edit(entry, normalized_roles):
                result.skipped.append({"key": key, "message": "forbidden"})
                continue

            if value is None:
                if key.startswith("preferences."):
                    if not dry_run:
                        repo = repo_holder.get("repo")
                        if repo is None:
                            repo = UserProfileOverridesRepo(self._db_pool)
                            await repo.ensure_tables()
                            repo_holder["repo"] = repo
                        await anchor.capture()
                        await repo.delete_override(user_id=user_id, key=key, db_conn=db_conn)
                        anchor.mark_changed()
                    result.applied.append(key)
                    continue
                result.skipped.append({"key": key, "message": "null_not_allowed"})
                continue

            if key in {"memberships.orgs.role", "memberships.teams.role"} and isinstance(value, dict):
                ok, normalized, err = True, value, None
            else:
                ok, normalized, err = _validate_value(entry, value)
            if not ok:
                result.skipped.append({"key": key, "message": err or "invalid_value"})
                continue

            try:
                handled = await self._apply_key_update(
                    user_id=user_id,
                    key=key,
                    value=normalized,
                    dry_run=dry_run,
                    db_conn=db_conn,
                    repo_holder=repo_holder,
                    updated_by=updated_by,
                    scope=scope,
                    is_platform_admin=is_platform_admin,
                    membership_context=membership_context,
                    is_postgres_backend=is_postgres_backend,
                    anchor=anchor,
                )
            except ValueError as exc:
                result.skipped.append({"key": key, "message": str(exc)})
                continue

            if handled:
                result.applied.append(key)
            else:
                result.skipped.append({"key": key, "message": "unsupported_key"})

        if not dry_run:
            await anchor.finalize()
        return result

    async def _apply_key_update(
        self,
        *,
        user_id: int,
        key: str,
        value: Any,
        dry_run: bool,
        db_conn: Any,
        repo_holder: dict[str, UserProfileOverridesRepo | None],
        updated_by: int | None,
        scope: ProfileUpdateScope | None,
        is_platform_admin: bool,
        membership_context: _MembershipContext | None,
        is_postgres_backend: bool,
        anchor: _CallerOwnedAnchor,
    ) -> bool:
        if key == "identity.email":
            try:
                email = TypeAdapter(EmailStr).validate_python(value)
            except ValidationError as exc:
                logger.debug("Invalid email update for user {}", user_id)
                raise ValueError("invalid_email") from exc
            if not dry_run:
                await _update_user_field(
                    db_conn,
                    user_id,
                    "email",
                    str(email).lower(),
                    anchor=anchor,
                    is_postgres_backend=is_postgres_backend,
                )
            return True

        if key == "identity.role":
            if not dry_run:
                await _update_user_field(
                    db_conn,
                    user_id,
                    "role",
                    str(value),
                    anchor=anchor,
                    is_postgres_backend=is_postgres_backend,
                )
            return True

        if key == "identity.is_active":
            if not dry_run:
                await _update_user_field(
                    db_conn,
                    user_id,
                    "is_active",
                    int(bool(value)),
                    anchor=anchor,
                    is_postgres_backend=is_postgres_backend,
                )
            return True

        if key == "identity.is_verified":
            if not dry_run:
                await _update_user_field(
                    db_conn,
                    user_id,
                    "is_verified",
                    int(bool(value)),
                    anchor=anchor,
                    is_postgres_backend=is_postgres_backend,
                )
            return True

        if key == "identity.is_locked":
            if not dry_run:
                await anchor.capture()
                username = await _fetch_username(
                    db_conn,
                    user_id,
                    is_postgres_backend=is_postgres_backend,
                )
                if not username:
                    raise ValueError("user_not_found")
                limiter = get_rate_limiter()
                if bool(value):
                    await limiter.record_failed_attempt(
                        identifier=username,
                        attempt_type="login",
                        lockout_threshold=1,
                    )
                else:
                    await limiter.reset_failed_attempts(identifier=username, attempt_type="login")
                await _touch_user_updated_at(db_conn, user_id)
                anchor.mark_changed()
            return True

        if key == "limits.storage_quota_mb":
            if not dry_run:
                await _update_user_field(
                    db_conn,
                    user_id,
                    "storage_quota_mb",
                    int(value),
                    anchor=anchor,
                    is_postgres_backend=is_postgres_backend,
                )
                try:
                    from tldw_Server_API.app.services.storage_quota_service import (
                        invalidate_storage_cache_for_user,
                    )

                    invalidate_storage_cache_for_user(int(user_id))
                except Exception as exc:
                    logger.debug(
                        "Failed to invalidate storage quota cache for user {}: {}",
                        user_id,
                        exc,
                    )
            return True

        if key in {"limits.audio_daily_minutes", "limits.audio_concurrent_jobs"}:
            if not dry_run:
                repo = repo_holder.get("repo")
                if repo is None:
                    repo = UserProfileOverridesRepo(self._db_pool)
                    await repo.ensure_tables()
                    repo_holder["repo"] = repo
                await anchor.capture()
                await repo.upsert_override(
                    user_id=user_id,
                    key=key,
                    value=value,
                    updated_by=updated_by,
                    db_conn=db_conn,
                )
                anchor.mark_changed()
            return True

        if key in {"limits.evaluations_per_minute", "limits.evaluations_per_day"}:
            if not dry_run:
                repo = repo_holder.get("repo")
                if repo is None:
                    repo = UserProfileOverridesRepo(self._db_pool)
                    await repo.ensure_tables()
                    repo_holder["repo"] = repo
                await anchor.capture()
                await repo.upsert_override(
                    user_id=user_id,
                    key=key,
                    value=value,
                    updated_by=updated_by,
                    db_conn=db_conn,
                )
                anchor.mark_changed()
                try:
                    from tldw_Server_API.app.core.Evaluations.user_rate_limiter import (
                        UserTier,
                        get_user_rate_limiter_for_user,
                    )

                    limiter = get_user_rate_limiter_for_user(user_id)
                    config = await limiter._get_user_config(str(user_id))
                    custom_limits = {
                        "evaluations_per_minute": config.evaluations_per_minute,
                        "batch_evaluations_per_minute": config.batch_evaluations_per_minute,
                        "evaluations_per_day": config.evaluations_per_day,
                        "total_tokens_per_day": config.total_tokens_per_day,
                        "burst_size": config.burst_size,
                        "max_cost_per_day": config.max_cost_per_day,
                        "max_cost_per_month": config.max_cost_per_month,
                    }
                    if key == "limits.evaluations_per_minute":
                        custom_limits["evaluations_per_minute"] = int(value)
                    else:
                        custom_limits["evaluations_per_day"] = int(value)
                    updated = await limiter.upgrade_user_tier(
                        str(user_id),
                        UserTier.CUSTOM,
                        custom_limits=custom_limits,
                    )
                    if not updated:
                        raise ValueError("evaluations_limit_update_failed")
                except ValueError:
                    raise
                except Exception as exc:
                    logger.debug("Evaluations limit update failed for user {}: {}", user_id, exc)
                    raise ValueError("evaluations_limit_update_failed") from exc
            return True

        if key.startswith("preferences."):
            if not dry_run:
                repo = repo_holder.get("repo")
                if repo is None:
                    repo = UserProfileOverridesRepo(self._db_pool)
                    await repo.ensure_tables()
                    repo_holder["repo"] = repo
                await anchor.capture()
                await repo.upsert_override(
                    user_id=user_id,
                    key=key,
                    value=value,
                    updated_by=updated_by,
                    db_conn=db_conn,
                )
                anchor.mark_changed()
            return True

        if key == "memberships.orgs.role":
            if membership_context is None:
                raise ValueError("membership_context_unavailable")
            org_id_override = None
            role_value = value
            if isinstance(value, dict):
                if "org_id" not in value or "role" not in value:
                    raise ValueError("invalid_membership_payload")
                try:
                    org_id_override = int(value.get("org_id"))
                except (TypeError, ValueError) as exc:
                    raise ValueError("invalid_org_id") from exc
                role_value = value.get("role")
            if org_id_override is not None:
                if org_id_override not in membership_context.target_org_roles:
                    raise ValueError("membership_not_found")
                if not is_platform_admin and org_id_override not in membership_context.actor_org_roles:
                    raise ValueError("forbidden_scope")
                org_id = org_id_override
            else:
                org_id = self._resolve_org_id(
                    membership_context,
                    scope=scope,
                    is_platform_admin=is_platform_admin,
                )
            if not dry_run:
                await anchor.capture()
                result = await update_org_member_role(
                    org_id=org_id,
                    user_id=user_id,
                    role=str(role_value),
                )
                if not result:
                    raise ValueError("membership_not_found")
                if result.get("error") == "owner_required":
                    raise ValueError("owner_required")
                await _touch_user_updated_at(db_conn, user_id)
                anchor.mark_changed()
            return True

        if key == "memberships.teams.role":
            if membership_context is None:
                raise ValueError("membership_context_unavailable")
            team_id_override = None
            role_value = value
            if isinstance(value, dict):
                if "team_id" not in value or "role" not in value:
                    raise ValueError("invalid_membership_payload")
                try:
                    team_id_override = int(value.get("team_id"))
                except (TypeError, ValueError) as exc:
                    raise ValueError("invalid_team_id") from exc
                role_value = value.get("role")
            if team_id_override is not None:
                if team_id_override not in membership_context.target_team_roles:
                    raise ValueError("membership_not_found")
                if not is_platform_admin and not self._actor_can_access_team(
                    membership_context,
                    team_id=team_id_override,
                ):
                    raise ValueError("forbidden_scope")
                team_id = team_id_override
            else:
                team_id = self._resolve_team_id(
                    membership_context,
                    scope=scope,
                    is_platform_admin=is_platform_admin,
                )
            if not dry_run:
                await anchor.capture()
                result = await update_team_member_role(
                    team_id=team_id,
                    user_id=user_id,
                    role=str(role_value),
                )
                if not result:
                    raise ValueError("membership_not_found")
                await _touch_user_updated_at(db_conn, user_id)
                anchor.mark_changed()
            return True

        if key == "memberships.teams.member":
            if membership_context is None:
                raise ValueError("membership_context_unavailable")
            team_id, action, role = _parse_team_membership_payload(value)
            if not dry_run:
                await anchor.capture()
                team = await get_team(team_id)
                if not team:
                    raise ValueError("team_not_found")
                if not is_platform_admin and not self._actor_can_access_team(
                    membership_context,
                    team_id=team_id,
                    team_org_id=int(team.get("org_id")) if team.get("org_id") is not None else None,
                ):
                    raise ValueError("forbidden_scope")
                if action == "add":
                    if team.get("org_id") is not None:
                        org_id = int(team.get("org_id"))
                        if org_id not in membership_context.target_org_roles:
                            raise ValueError("org_membership_required")
                    await add_team_member(team_id=team_id, user_id=user_id, role=role or "member")
                else:
                    res = await remove_team_member(team_id=team_id, user_id=user_id)
                    if not res.get("removed"):
                        raise ValueError("membership_not_found")
                await _touch_user_updated_at(db_conn, user_id)
                anchor.mark_changed()
            return True

        return False

    async def _build_membership_context(
        self,
        *,
        user_id: int,
        scope: ProfileUpdateScope | None,
        is_platform_admin: bool,
    ) -> _MembershipContext:
        target_org_rows = await list_org_memberships_for_user(user_id)
        target_team_rows = await list_memberships_for_user(user_id)
        target_org_roles = {
            int(row.get("org_id")): str(row.get("role") or "member").lower()
            for row in target_org_rows
            if row.get("org_id") is not None
        }
        target_team_roles = {
            int(row.get("team_id")): str(row.get("role") or "member").lower()
            for row in target_team_rows
            if row.get("team_id") is not None
        }
        target_team_orgs = {
            int(row.get("team_id")): int(row.get("org_id"))
            for row in target_team_rows
            if row.get("team_id") is not None and row.get("org_id") is not None
        }
        actor_org_roles: dict[int, str] = {}
        actor_team_roles: dict[int, str] = {}
        actor_user_id = scope.actor_user_id if scope else None
        if actor_user_id is not None and not is_platform_admin:
            actor_org_rows = await list_org_memberships_for_user(int(actor_user_id))
            actor_team_rows = await list_memberships_for_user(int(actor_user_id))
            actor_org_roles = {
                int(row.get("org_id")): str(row.get("role") or "member").lower()
                for row in actor_org_rows
                if row.get("org_id") is not None
            }
            actor_team_roles = {
                int(row.get("team_id")): str(row.get("role") or "member").lower()
                for row in actor_team_rows
                if row.get("team_id") is not None
            }
        return _MembershipContext(
            target_org_roles=target_org_roles,
            target_team_roles=target_team_roles,
            target_team_orgs=target_team_orgs,
            actor_org_roles=actor_org_roles,
            actor_team_roles=actor_team_roles,
        )

    @staticmethod
    def _resolve_org_id(
        context: _MembershipContext,
        *,
        scope: ProfileUpdateScope | None,
        is_platform_admin: bool,
    ) -> int:
        target_org_ids = set(context.target_org_roles)
        if not target_org_ids:
            raise ValueError("membership_not_found")
        active_org_id = scope.active_org_id if scope else None
        if active_org_id is not None:
            org_id = int(active_org_id)
            if org_id not in target_org_ids:
                raise ValueError("membership_not_found")
            if not is_platform_admin and org_id not in context.actor_org_roles:
                raise ValueError("forbidden_scope")
            return org_id
        shared_orgs = target_org_ids & set(context.actor_org_roles)
        if shared_orgs:
            if len(shared_orgs) == 1:
                return next(iter(shared_orgs))
            raise ValueError("ambiguous_org_membership")
        if len(target_org_ids) == 1:
            return next(iter(target_org_ids))
        raise ValueError("ambiguous_org_membership")

    def _resolve_team_id(
        self,
        context: _MembershipContext,
        *,
        scope: ProfileUpdateScope | None,
        is_platform_admin: bool,
    ) -> int:
        target_team_ids = set(context.target_team_roles)
        if not target_team_ids:
            raise ValueError("membership_not_found")
        active_team_id = scope.active_team_id if scope else None
        if active_team_id is not None:
            team_id = int(active_team_id)
            if team_id not in target_team_ids:
                raise ValueError("membership_not_found")
            if not is_platform_admin and not self._actor_can_access_team(context, team_id=team_id):
                raise ValueError("forbidden_scope")
            return team_id
        shared_teams = target_team_ids & set(context.actor_team_roles)
        if shared_teams:
            if len(shared_teams) == 1:
                return next(iter(shared_teams))
            raise ValueError("ambiguous_team_membership")
        if len(target_team_ids) == 1:
            team_id = next(iter(target_team_ids))
            if not is_platform_admin and not self._actor_can_access_team(context, team_id=team_id):
                raise ValueError("forbidden_scope")
            return team_id
        raise ValueError("ambiguous_team_membership")

    @staticmethod
    def _actor_can_access_team(
        context: _MembershipContext,
        *,
        team_id: int,
        team_org_id: int | None = None,
    ) -> bool:
        if team_id in context.actor_team_roles:
            return True
        org_id = team_org_id if team_org_id is not None else context.target_team_orgs.get(team_id)
        if org_id is None:
            return False
        return org_id in context.actor_org_roles


def _can_edit(entry: UserProfileCatalogEntry, roles: set[str]) -> bool:
    entry_roles = {str(role).strip() for role in (entry.editable_by or []) if role}
    return bool(entry_roles & roles)


def _validate_value(entry: UserProfileCatalogEntry, value: Any) -> tuple[bool, Any, str | None]:
    if entry.enum and value not in entry.enum:
        return False, None, "enum_violation"

    if entry.type == "string":
        if not isinstance(value, str):
            return False, None, "type_mismatch"
        return True, value, None
    if entry.type == "integer":
        if isinstance(value, bool) or not isinstance(value, int):
            return False, None, "type_mismatch"
        return _validate_numeric(entry, value)
    if entry.type == "number":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return False, None, "type_mismatch"
        return _validate_numeric(entry, float(value))
    if entry.type == "boolean":
        if not isinstance(value, bool):
            return False, None, "type_mismatch"
        return True, value, None
    if entry.type == "json":
        if not isinstance(value, (dict, list)):
            return False, None, "type_mismatch"
        return True, value, None
    return False, None, "unsupported_type"


def _validate_numeric(entry: UserProfileCatalogEntry, value: float) -> tuple[bool, Any, str | None]:
    if entry.minimum is not None and value < entry.minimum:
        return False, None, "min_violation"
    if entry.maximum is not None and value > entry.maximum:
        return False, None, "max_violation"
    if entry.type == "integer":
        return True, int(value), None
    return True, value, None


async def _update_user_field(
    db_conn: Any,
    user_id: int,
    column: str,
    value: Any,
    *,
    anchor: _CallerOwnedAnchor,
    is_postgres_backend: bool,
) -> None:
    try:
        await anchor.capture()
        placeholders = ("$1", "$2") if is_postgres_backend else ("?", "?")
        update_user_sql_template = (
            "UPDATE users SET {column} = {placeholders[0]}, "
            "updated_at = CURRENT_TIMESTAMP WHERE id = {placeholders[1]}"
        )
        update_user_sql = update_user_sql_template.format_map(locals())  # nosec B608
        write_result = await anchor.gateway.execute_update(
            db_conn,
            user_id=user_id,
            profile_visible_fields=(column,),
            statement=update_user_sql,
            parameters=(value, user_id),
            ownership=UserVersionOwnership.CALLER_OWNS_ANCHOR,
        )
        if write_result.affected_user_ids:
            anchor.mark_changed()
    except Exception as exc:
        logger.bind(
            operation="update_user_field",
            field=column,
            user_id=user_id,
            exception_type=type(exc).__name__,
        ).error("Failed to update user field")
        raise


async def _touch_user_updated_at(db_conn: Any, user_id: int) -> None:
    try:
        await db_conn.execute(
            "UPDATE users SET updated_at = CURRENT_TIMESTAMP WHERE id = $1",
            user_id,
        )
    except Exception as exc:
        logger.bind(
            operation="update_user_timestamp",
            user_id=user_id,
            exception_type=type(exc).__name__,
        ).error("Failed to update user timestamp")
        raise


async def _fetch_username(
    db_conn: Any,
    user_id: int,
    *,
    is_postgres_backend: bool,
) -> str | None:
    try:
        if is_postgres_backend:
            value = await db_conn.fetchval(
                "SELECT username FROM users WHERE id = $1",
                user_id,
            )
            return str(value) if value is not None else None
        cursor = await db_conn.execute(
            "SELECT username FROM users WHERE id = ?",
            (user_id,),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        if isinstance(row, dict):
            raw = row.get("username")
        else:
            try:
                raw = row["username"]  # sqlite3.Row / aiosqlite.Row mapping access
            except (TypeError, KeyError, IndexError):
                raw = row[0]
        return str(raw) if raw is not None else None
    except Exception as exc:
        logger.bind(
            operation="fetch_username",
            user_id=user_id,
            exception_type=type(exc).__name__,
        ).error("Failed to fetch username")
        raise


def _parse_team_membership_payload(value: Any) -> tuple[int, str, str | None]:
    if not isinstance(value, dict):
        raise ValueError("invalid_team_membership")
    raw_team_id = value.get("team_id")
    try:
        team_id = int(raw_team_id)
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid_team_id") from exc
    action = str(value.get("action") or "").strip().lower()
    if action not in {"add", "remove"}:
        raise ValueError("invalid_team_action")
    role = value.get("role")
    role_value = str(role).strip().lower() if role is not None else None
    return team_id, action, role_value
