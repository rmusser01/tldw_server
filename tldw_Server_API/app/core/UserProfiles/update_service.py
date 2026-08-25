"""
User profile update helpers and validation.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from loguru import logger
from pydantic import EmailStr, TypeAdapter, ValidationError

from tldw_Server_API.app.core.AuthNZ.membership_writer import (
    ActorMembershipWriteContext,
    AnchorOwnership,
    MembershipAuthority,
    MembershipMutation,
    MembershipMutationKind,
    MembershipScopeType,
    MembershipWriter,
    MembershipWriteResult,
)
from tldw_Server_API.app.core.AuthNZ.orgs_teams import (
    list_memberships_for_user,
    list_org_memberships_for_user,
)
from tldw_Server_API.app.core.AuthNZ.profile_version import (
    UserVersionOwnership,
    VersionedUserWriteGateway,
)
from tldw_Server_API.app.core.AuthNZ.rate_limiter import get_rate_limiter
from tldw_Server_API.app.core.AuthNZ.repos.orgs_teams_repo import AuthnzOrgsTeamsRepo
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


@dataclass(frozen=True)
class _PreparedMembershipUpdate:
    update_index: int
    mutation: MembershipMutation


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

    def include_floor(self, version_floor: Any) -> None:
        if self.version_floor is None:
            self.version_floor = version_floor
        else:
            self.version_floor = max(self.version_floor, version_floor)
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
                db_conn=None if dry_run else db_conn,
            )
        is_postgres_backend = _is_postgres_backend_for_pool(self._db_pool)
        operation_time = datetime.now(timezone.utc)
        anchor = _CallerOwnedAnchor(
            gateway=VersionedUserWriteGateway(
                "postgres" if is_postgres_backend else "sqlite",
                clock=lambda: operation_time,
            ),
            db_conn=db_conn,
            user_id=user_id,
        )
        prepared_memberships: dict[int, _PreparedMembershipUpdate] = {}
        membership_prepare_errors: dict[int, str] = {}
        if membership_context is not None:
            for update_index, (key, value) in enumerate(updates_list):
                if key not in {
                    "memberships.orgs.role",
                    "memberships.teams.role",
                    "memberships.teams.member",
                }:
                    continue
                entry = catalog_map.get(key)
                if entry is None or not _can_edit(entry, normalized_roles) or value is None:
                    continue
                if key in {
                    "memberships.orgs.role",
                    "memberships.teams.role",
                } and isinstance(value, dict):
                    ok, normalized = True, value
                else:
                    ok, normalized, _ = _validate_value(entry, value)
                if not ok:
                    continue
                try:
                    mutation = self._prepare_membership_mutation(
                        user_id=user_id,
                        key=key,
                        value=normalized,
                        scope=scope,
                        is_platform_admin=is_platform_admin,
                        membership_context=membership_context,
                    )
                except ValueError as exc:
                    membership_prepare_errors[update_index] = str(exc)
                else:
                    prepared_memberships[update_index] = _PreparedMembershipUpdate(
                        update_index=update_index,
                        mutation=mutation,
                    )
        membership_results: dict[int, Any] = {}
        if prepared_memberships and not dry_run:
            membership_results, write_result = await self._apply_membership_batch(
                db_conn=db_conn,
                prepared=tuple(prepared_memberships.values()),
                scope=scope,
                is_platform_admin=is_platform_admin,
                operation_time=operation_time,
            )
            if user_id in write_result.affected_user_ids:
                anchor.include_floor(write_result.floor_for(user_id))

        for update_index, (key, value) in enumerate(updates_list):
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

            if key.startswith("memberships."):
                prepare_error = membership_prepare_errors.get(update_index)
                if prepare_error is not None:
                    result.skipped.append({"key": key, "message": prepare_error})
                    continue
                prepared = prepared_memberships.get(update_index)
                if prepared is None:
                    result.skipped.append({"key": key, "message": "unsupported_key"})
                    continue
                if dry_run:
                    result.applied.append(key)
                    continue
                mutation_result = membership_results[update_index]
                result_error = self._membership_result_error(mutation_result)
                if result_error is not None:
                    result.skipped.append({"key": key, "message": result_error})
                    continue
                if not mutation_result.changed:
                    await anchor.capture()
                    anchor.mark_changed()
                result.applied.append(key)
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

        return False

    def _prepare_membership_mutation(
        self,
        *,
        user_id: int,
        key: str,
        value: Any,
        scope: ProfileUpdateScope | None,
        is_platform_admin: bool,
        membership_context: _MembershipContext,
    ) -> MembershipMutation:
        if key == "memberships.orgs.role":
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
                if (
                    not is_platform_admin
                    and not self._actor_can_access_org(
                        membership_context,
                        org_id=org_id_override,
                    )
                ):
                    raise ValueError("forbidden_scope")
                org_id = org_id_override
            else:
                org_id = self._resolve_org_id(
                    membership_context,
                    scope=scope,
                    is_platform_admin=is_platform_admin,
                )
            return MembershipMutation(
                scope_type=MembershipScopeType.ORGANIZATION,
                scope_id=org_id,
                user_id=user_id,
                kind=MembershipMutationKind.UPDATE_ROLE,
                role=str(role_value),
            )

        if key == "memberships.teams.role":
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
            return MembershipMutation(
                scope_type=MembershipScopeType.TEAM,
                scope_id=team_id,
                user_id=user_id,
                kind=MembershipMutationKind.UPDATE_ROLE,
                role=str(role_value),
            )

        if key == "memberships.teams.member":
            team_id, action, role = _parse_team_membership_payload(value)
            if not is_platform_admin and not self._actor_can_access_team(
                membership_context,
                team_id=team_id,
            ):
                raise ValueError("forbidden_scope")
            return MembershipMutation(
                scope_type=MembershipScopeType.TEAM,
                scope_id=team_id,
                user_id=user_id,
                kind=(
                    MembershipMutationKind.ADD
                    if action == "add"
                    else MembershipMutationKind.REMOVE
                ),
                role=(role or "member") if action == "add" else None,
            )
        raise ValueError("unsupported_key")

    async def _apply_membership_batch(
        self,
        *,
        db_conn: Any,
        prepared: tuple[_PreparedMembershipUpdate, ...],
        scope: ProfileUpdateScope | None,
        is_platform_admin: bool,
        operation_time: datetime,
    ) -> tuple[dict[int, Any], MembershipWriteResult]:
        actor_user_id = scope.actor_user_id if scope else None
        if type(actor_user_id) is not int or actor_user_id <= 0:
            raise ValueError("membership_context_unavailable")
        context = ActorMembershipWriteContext(
            actor_user_id=actor_user_id,
            required_authority=(
                MembershipAuthority.PLATFORM_ADMIN
                if is_platform_admin
                else MembershipAuthority.SCOPED_MEMBERSHIP
            ),
        )
        write_result = await MembershipWriter(self._db_pool).apply_membership_mutations(
            conn=db_conn,
            context=context,
            mutations=tuple(item.mutation for item in prepared),
            anchor_ownership=AnchorOwnership.CALLER_OWNS_ANCHOR,
            operation_time=operation_time,
        )
        if len(write_result.mutation_results) != len(prepared):
            raise RuntimeError("Membership writer returned an incomplete result batch")
        return (
            {
                item.update_index: mutation_result
                for item, mutation_result in zip(
                    prepared,
                    write_result.mutation_results,
                    strict=True,
                )
            },
            write_result,
        )

    @staticmethod
    def _membership_result_error(mutation_result: Any) -> str | None:
        if mutation_result.error is not None:
            return str(mutation_result.error)
        if mutation_result.mutation.kind is MembershipMutationKind.UPDATE_ROLE:
            return None if mutation_result.found else "membership_not_found"
        if mutation_result.mutation.kind is MembershipMutationKind.REMOVE:
            return None if mutation_result.changed else "membership_not_found"
        return None

    async def _build_membership_context(
        self,
        *,
        user_id: int,
        scope: ProfileUpdateScope | None,
        is_platform_admin: bool,
        db_conn: Any | None = None,
    ) -> _MembershipContext:
        if db_conn is None:
            async def _list_orgs(member_user_id: int) -> list[dict[str, Any]]:
                return await list_org_memberships_for_user(member_user_id)

            async def _list_teams(member_user_id: int) -> list[dict[str, Any]]:
                return await list_memberships_for_user(member_user_id)
        else:
            membership_repo = AuthnzOrgsTeamsRepo(self._db_pool)

            async def _list_orgs(member_user_id: int) -> list[dict[str, Any]]:
                return await membership_repo.list_org_memberships_for_user(
                    member_user_id,
                    conn=db_conn,
                )

            async def _list_teams(member_user_id: int) -> list[dict[str, Any]]:
                return await membership_repo.list_memberships_for_user(
                    member_user_id,
                    conn=db_conn,
                )

        target_org_rows = await _list_orgs(user_id)
        target_team_rows = await _list_teams(user_id)
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
            actor_org_rows = await _list_orgs(int(actor_user_id))
            actor_team_rows = await _list_teams(int(actor_user_id))
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
            if not is_platform_admin and not UserProfileUpdateService._actor_can_access_org(
                context,
                org_id=org_id,
            ):
                raise ValueError("forbidden_scope")
            return org_id
        shared_orgs = {
            org_id
            for org_id in target_org_ids
            if UserProfileUpdateService._actor_can_access_org(
                context,
                org_id=org_id,
            )
        }
        if shared_orgs:
            if len(shared_orgs) == 1:
                return next(iter(shared_orgs))
            raise ValueError("ambiguous_org_membership")
        if len(target_org_ids) == 1:
            if not is_platform_admin:
                raise ValueError("forbidden_scope")
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
        team_role = context.actor_team_roles.get(team_id, "").strip().lower()
        if team_role in {"owner", "admin", "lead"}:
            return True
        org_id = team_org_id if team_org_id is not None else context.target_team_orgs.get(team_id)
        if org_id is None:
            return False
        org_role = context.actor_org_roles.get(org_id, "").strip().lower()
        return org_role in {"owner", "admin"}

    @staticmethod
    def _actor_can_access_org(
        context: _MembershipContext,
        *,
        org_id: int,
    ) -> bool:
        role = context.actor_org_roles.get(org_id, "").strip().lower()
        return role in {"owner", "admin"}


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
