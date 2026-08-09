"""
User profile service for assembling profile sections.
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterable
from datetime import datetime, timezone
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.AuthNZ.database import DatabasePool
from tldw_Server_API.app.core.AuthNZ.exceptions import StorageError, UserNotFoundError
from tldw_Server_API.app.core.AuthNZ.mfa_service import get_mfa_service
from tldw_Server_API.app.core.AuthNZ.repos.orgs_teams_repo import AuthnzOrgsTeamsRepo
from tldw_Server_API.app.core.AuthNZ.repos.user_provider_secrets_repo import (
    AuthnzUserProviderSecretsRepo,
)
from tldw_Server_API.app.core.UserProfiles.overrides_repo import (
    OrgProfileOverridesRepo,
    TeamProfileOverridesRepo,
    UserProfileOverridesRepo,
)
from tldw_Server_API.app.core.UserProfiles.user_profile_catalog import load_user_profile_catalog
from tldw_Server_API.app.core.UserProfiles.version_gateway import (
    ProfileVersionGateway,
    ProfileVersionGatewayProtocol,
)

KNOWN_SECTIONS: set[str] = {
    "identity",
    "memberships",
    "security",
    "quotas",
    "preferences",
    "effective_config",
}
SUPPORTED_SECTIONS: set[str] = {
    "identity",
    "memberships",
    "security",
    "quotas",
    "preferences",
    "effective_config",
}
DEFAULT_SECTIONS: set[str] = {
    "identity",
    "memberships",
    "security",
    "quotas",
    "preferences",
    "effective_config",
}

_PROFILE_METRICS_REGISTERED = False

_PROFILE_NONCRITICAL_EXCEPTIONS = (
    AssertionError,
    AttributeError,
    ConnectionError,
    FileNotFoundError,
    ImportError,
    IndexError,
    KeyError,
    LookupError,
    OSError,
    PermissionError,
    RuntimeError,
    StorageError,
    TimeoutError,
    TypeError,
    UnicodeDecodeError,
    UserNotFoundError,
    ValueError,
)


class UserProfileService:
    """Service that assembles user profile data from multiple sources."""

    def __init__(
        self,
        db_pool: DatabasePool,
        *,
        profile_version_gateway: ProfileVersionGatewayProtocol | None = None,
    ):
        self._db_pool = db_pool
        self._orgs_repo = AuthnzOrgsTeamsRepo(db_pool)
        self._profile_version_gateway = (
            profile_version_gateway or ProfileVersionGateway(db_pool)
        )

    @staticmethod
    def parse_sections(raw: Iterable[str] | None | str | None) -> set[str] | None:
        """Normalize a sections query parameter into a set."""
        if raw is None:
            return None
        if isinstance(raw, str):
            parts = [p.strip() for p in raw.split(",") if p.strip()]
            return set(parts) if parts else None
        parts: list[str] = []
        for entry in raw:
            if entry is None:
                continue
            for piece in str(entry).split(","):
                piece = piece.strip()
                if piece:
                    parts.append(piece)
        return set(parts) if parts else None

    @staticmethod
    def _catalog_entry_map(catalog: Any) -> dict[str, Any]:
        return {str(entry.key): entry for entry in getattr(catalog, "entries", []) or []}

    @staticmethod
    def _mask_value(
        key: str,
        value: Any,
        *,
        catalog_map: dict[str, Any],
        mask_secrets: bool,
    ) -> tuple[Any, bool, str | None]:
        entry = catalog_map.get(str(key))
        sensitivity = str(getattr(entry, "sensitivity", "internal")).strip().lower()
        if sensitivity != "secret":
            return value, False, None
        if value is None:
            return None, False, None
        hint = "configured" if not mask_secrets else None
        return "[REDACTED]", True, hint

    def _format_value(
        self,
        key: str,
        value: Any,
        *,
        include_sources: bool,
        source: str | None,
        catalog_map: dict[str, Any],
        mask_secrets: bool,
    ) -> Any:
        masked_value, masked, hint = self._mask_value(
            key,
            value,
            catalog_map=catalog_map,
            mask_secrets=mask_secrets,
        )
        if not include_sources:
            return masked_value
        payload: dict[str, Any] = {"value": masked_value, "source": source}
        if masked:
            payload["masked"] = True
        if hint:
            payload["hint"] = hint
        return payload

    def _get_metrics_registry(self):
        global _PROFILE_METRICS_REGISTERED
        try:
            from tldw_Server_API.app.core.Metrics.metrics_manager import (
                MetricDefinition,
                MetricType,
                get_metrics_registry,
            )
        except _PROFILE_NONCRITICAL_EXCEPTIONS as exc:
            logger.debug("Profile metrics unavailable: {}", exc)
            return None

        registry = get_metrics_registry()
        if _PROFILE_METRICS_REGISTERED:
            return registry

        registry.register_metric(
            MetricDefinition(
                name="profile_fetch_latency_ms",
                type=MetricType.HISTOGRAM,
                description="User profile fetch latency",
                unit="ms",
                labels=["scope"],
                buckets=[5, 10, 25, 50, 100, 250, 500, 1000, 2500],
            )
        )
        registry.register_metric(
            MetricDefinition(
                name="profile_section_latency_ms",
                type=MetricType.HISTOGRAM,
                description="User profile section build latency",
                unit="ms",
                labels=["scope", "section"],
                buckets=[2, 5, 10, 25, 50, 100, 250, 500, 1000],
            )
        )
        registry.register_metric(
            MetricDefinition(
                name="profile_errors_total",
                type=MetricType.COUNTER,
                description="User profile section errors",
                labels=["scope", "section"],
            )
        )
        registry.register_metric(
            MetricDefinition(
                name="profile_bulk_update_total",
                type=MetricType.COUNTER,
                description="Total users targeted by profile bulk updates",
                labels=["dry_run"],
            )
        )
        registry.register_metric(
            MetricDefinition(
                name="profile_sla_breach_total",
                type=MetricType.COUNTER,
                description="Profile fetch SLA breaches",
                labels=["scope"],
            )
        )
        registry.register_metric(
            MetricDefinition(
                name="profile_batch_latency_ms",
                type=MetricType.HISTOGRAM,
                description="Profile batch request latency",
                unit="ms",
                labels=["page_size"],
                buckets=[50, 100, 250, 500, 1000, 2500, 5000, 10000],
            )
        )
        registry.register_metric(
            MetricDefinition(
                name="profile_batch_sla_breach_total",
                type=MetricType.COUNTER,
                description="Profile batch SLA breaches",
                labels=["page_size"],
            )
        )
        registry.register_metric(
            MetricDefinition(
                name="profile_batch_timeout_total",
                type=MetricType.COUNTER,
                description="Profile batch timeout threshold breaches",
                labels=["page_size"],
            )
        )
        _PROFILE_METRICS_REGISTERED = True
        return registry

    @staticmethod
    def _read_int_env(name: str, default: int) -> int:
        raw = os.getenv(name)
        if raw is None or raw == "":
            return default
        try:
            return int(raw)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _read_float_env(name: str, default: float) -> float:
        raw = os.getenv(name)
        if raw is None or raw == "":
            return default
        try:
            return float(raw)
        except (TypeError, ValueError):
            return default

    @classmethod
    def profile_sla_threshold_ms(cls, scope: str) -> int | None:
        scope_key = str(scope or "").strip().lower()
        if scope_key not in {"self", "admin"}:
            return None
        base = cls._read_int_env("PROFILE_SLA_MS", 300)
        if scope_key == "self":
            return cls._read_int_env("PROFILE_SLA_MS_SELF", base)
        if scope_key == "admin":
            return cls._read_int_env("PROFILE_SLA_MS_ADMIN", base)
        return base

    @classmethod
    def batch_sla_threshold_ms(cls, page_size: int) -> int:
        base_ms = cls._read_int_env("PROFILE_BATCH_BASE_MS", 800)
        base_size = max(1, cls._read_int_env("PROFILE_BATCH_BASE_SIZE", 50))
        scale = max(1.0, float(page_size) / float(base_size))
        return int(base_ms * scale)

    @classmethod
    def batch_timeout_ms(cls) -> int:
        seconds = cls._read_float_env("PROFILE_BATCH_TIMEOUT_SECONDS", 10.0)
        if seconds < 0:
            seconds = 10.0
        return int(seconds * 1000.0)

    @staticmethod
    def _normalize_timestamp(value: Any) -> datetime | None:
        if value is None:
            return None
        if isinstance(value, datetime):
            ts = value
        else:
            try:
                if isinstance(value, str):
                    candidate = value.replace("Z", "+00:00") if value.endswith("Z") else value
                    ts = datetime.fromisoformat(candidate)
                else:
                    return None
            except _PROFILE_NONCRITICAL_EXCEPTIONS:
                return None
        if ts.tzinfo is None:
            return ts.replace(tzinfo=timezone.utc)
        return ts.astimezone(timezone.utc)

    @classmethod
    def versions_match(cls, current: Any, expected: Any) -> bool:
        current_ts = cls._normalize_timestamp(current)
        expected_ts = cls._normalize_timestamp(expected)
        if current_ts is None or expected_ts is None:
            return False
        return current_ts == expected_ts

    def _build_identity(self, user: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": user["id"],
            "uuid": str(user.get("uuid")) if user.get("uuid") else None,
            "username": user.get("username") or "",
            "email": user.get("email") or "",
            "role": user.get("role") or "user",
            "is_active": bool(user.get("is_active", True)),
            "is_verified": bool(user.get("is_verified", False)),
            "is_locked": user.get("is_locked"),
            "created_at": user.get("created_at") or datetime.now(timezone.utc),
            "last_login": user.get("last_login"),
        }

    async def _attach_lockout_status(self, identity: dict[str, Any]) -> None:
        username = identity.get("username")
        if not username:
            return
        try:
            from tldw_Server_API.app.core.AuthNZ.rate_limiter import get_rate_limiter

            limiter = get_rate_limiter()
            is_locked, _ = await limiter.check_lockout(str(username), attempt_type="login")
            identity["is_locked"] = bool(is_locked)
        except _PROFILE_NONCRITICAL_EXCEPTIONS as exc:
            logger.debug("Lockout lookup failed for user {}: {}", identity.get("id"), exc)

    async def _build_memberships(self, user_id: int) -> dict[str, Any]:
        orgs = await self._orgs_repo.list_org_memberships_for_user(user_id)
        teams = await self._orgs_repo.list_memberships_for_user(user_id)
        org_ids = [
            int(row["org_id"])
            for row in orgs
            if row.get("org_id") is not None
        ]
        policy_summaries = await self._build_org_policy_summaries(org_ids)
        return {
            "orgs": [
                {
                    "org_id": int(row["org_id"]),
                    "role": str(row["role"]),
                    "policy_summary": policy_summaries.get(int(row["org_id"])),
                }
                for row in orgs
            ],
            "teams": [
                {
                    "team_id": int(row["team_id"]),
                    "role": str(row["role"]),
                    "org_id": int(row["org_id"]) if row.get("org_id") is not None else None,
                    "policy_summary": policy_summaries.get(int(row["org_id"]))
                    if row.get("org_id") is not None
                    else None,
                }
                for row in teams
            ],
        }

    async def _build_org_policy_summaries(self, org_ids: list[int]) -> dict[int, dict[str, Any]]:
        summaries: dict[int, dict[str, Any]] = {}
        if not org_ids:
            return summaries
        try:
            from tldw_Server_API.app.core.External_Sources.connectors_service import get_policy
            from tldw_Server_API.app.core.External_Sources.policy import (
                get_default_policy_from_env,
            )
        except _PROFILE_NONCRITICAL_EXCEPTIONS as exc:
            logger.debug("Org policy helpers unavailable: {}", exc)
            return summaries

        async def _load_with_conn(conn) -> None:
            for org_id in sorted(set(org_ids)):
                try:
                    policy = await get_policy(conn, int(org_id))
                    source = "db" if policy else "env_default"
                    if not policy:
                        policy = get_default_policy_from_env(int(org_id))
                    summaries[int(org_id)] = {
                        "connectors": policy,
                        "source": source,
                    }
                except _PROFILE_NONCRITICAL_EXCEPTIONS as exc:
                    logger.debug("Org policy summary failed for org_id {}: {}", org_id, exc)

        try:
            acquire = getattr(self._db_pool, "acquire", None)
            if callable(acquire):
                async with self._db_pool.acquire() as conn:
                    await _load_with_conn(conn)
            else:
                await _load_with_conn(self._db_pool)
        except _PROFILE_NONCRITICAL_EXCEPTIONS as exc:
            logger.debug("Org policy summary connection failure: {}", exc)
        return summaries

    async def _get_membership_ids(self, user_id: int) -> tuple[list[int], list[int]]:
        orgs = await self._orgs_repo.list_org_memberships_for_user(user_id)
        teams = await self._orgs_repo.list_active_team_memberships_for_user(user_id)
        org_ids = sorted(
            {
                int(row["org_id"])
                for row in orgs
                if row.get("org_id") is not None
                and row.get("status") in (None, "active")
            }
        )
        team_ids = sorted(
            {
                int(row["team_id"])
                for row in teams
                if row.get("team_id") is not None
            }
        )
        return org_ids, team_ids

    async def _build_quotas(self, user: dict[str, Any]) -> dict[str, Any]:
        quotas: dict[str, Any] = {
            "storage_quota_mb": int(user.get("storage_quota_mb", 0) or 0),
            "storage_used_mb": float(user.get("storage_used_mb", 0.0) or 0.0),
        }
        user_id = int(user.get("id"))
        try:
            from tldw_Server_API.app.services.storage_quota_service import (
                get_storage_service,
            )

            storage_service = await get_storage_service()
            storage_info = await storage_service.calculate_user_storage(
                user_id=user_id,
                update_database=False,
            )
            live_quota = storage_info.get("quota_mb")
            live_used = storage_info.get("total_mb")
            if live_quota is not None:
                quotas["storage_quota_mb"] = int(live_quota)
            if live_used is not None:
                quotas["storage_used_mb"] = float(live_used)
        except _PROFILE_NONCRITICAL_EXCEPTIONS as exc:
            logger.debug("Live storage quotas unavailable for user {}: {}", user_id, exc)

        try:
            from tldw_Server_API.app.core.Usage.audio_quota import (
                active_streams_count,
                get_daily_minutes_used,
                get_limits_for_user,
            )

            limits = await get_limits_for_user(user_id)
            daily_limit = limits.get("daily_minutes")
            used = await get_daily_minutes_used(user_id)
            remaining = None
            if daily_limit is not None:
                remaining = max(0.0, float(daily_limit) - float(used))
            active_streams = await active_streams_count(user_id)
            quotas["audio"] = {
                "daily_minutes_limit": daily_limit,
                "daily_minutes_used": float(used),
                "daily_minutes_remaining": remaining,
                "concurrent_streams_limit": (
                    int(limits["concurrent_streams"])
                    if limits.get("concurrent_streams") is not None
                    else None
                ),
                "concurrent_streams_active": int(active_streams),
                "concurrent_jobs_limit": (
                    int(limits["concurrent_jobs"])
                    if limits.get("concurrent_jobs") is not None
                    else None
                ),
                "max_file_size_mb": (
                    int(limits["max_file_size_mb"])
                    if limits.get("max_file_size_mb") is not None
                    else None
                ),
            }
        except _PROFILE_NONCRITICAL_EXCEPTIONS as exc:
            logger.debug("Audio quotas unavailable for user {}: {}", user_id, exc)

        try:
            from tldw_Server_API.app.core.Evaluations.user_rate_limiter import (
                get_user_rate_limiter_for_user,
            )

            limiter = get_user_rate_limiter_for_user(user_id)
            summary = await limiter.get_usage_summary(str(user_id))
            quotas["evaluations"] = {
                "tier": summary.get("tier"),
                "limits": summary.get("limits", {}),
                "usage": summary.get("usage", {}),
                "remaining": summary.get("remaining", {}),
            }
        except _PROFILE_NONCRITICAL_EXCEPTIONS as exc:
            logger.debug("Evaluations quotas unavailable for user {}: {}", user_id, exc)

        try:
            effective = await self._build_effective_config(
                user_id,
                include_sources=False,
                mask_secrets=True,
            )
            quotas["prompt_studio"] = {
                "limits": {
                    "max_concurrent_jobs": effective.get("limits.prompt_studio_max_concurrent_jobs"),
                    "max_queued_jobs": effective.get("limits.prompt_studio_max_queued_jobs"),
                    "submits_per_min": effective.get("limits.prompt_studio_submits_per_min"),
                }
            }
        except _PROFILE_NONCRITICAL_EXCEPTIONS as exc:
            logger.debug("Prompt Studio quotas unavailable for user {}: {}", user_id, exc)

        return quotas

    async def _build_preferences(
        self,
        user_id: int,
        *,
        include_sources: bool,
        mask_secrets: bool,
    ) -> dict[str, Any]:
        effective = await self._build_effective_config(
            user_id,
            include_sources=include_sources,
            mask_secrets=mask_secrets,
        )
        return {
            key: value
            for key, value in effective.items()
            if str(key).startswith("preferences.")
        }

    async def _build_effective_config(
        self,
        user_id: int,
        *,
        include_sources: bool,
        mask_secrets: bool,
    ) -> dict[str, Any]:
        catalog = load_user_profile_catalog()
        catalog_map = self._catalog_entry_map(catalog)
        entries = [
            entry
            for entry in catalog.entries
            if not str(entry.key).startswith(("identity.", "memberships."))
        ]
        repo = UserProfileOverridesRepo(self._db_pool)
        await repo.ensure_tables()
        rows = await repo.list_overrides_for_user(user_id)
        overrides = {
            str(row["key"]): row.get("value")
            for row in rows
            if row.get("value") is not None
        }

        org_overrides: dict[str, dict[str, Any]] = {}
        team_overrides: dict[str, dict[str, Any]] = {}
        try:
            org_ids, team_ids = await self._get_membership_ids(user_id)
            if org_ids:
                org_repo = OrgProfileOverridesRepo(self._db_pool)
                await org_repo.ensure_tables()
                org_rows = await org_repo.list_overrides_for_orgs(org_ids)
                org_overrides = self._select_lowest_id_overrides(org_rows, id_field="org_id")
            if team_ids:
                team_repo = TeamProfileOverridesRepo(self._db_pool)
                await team_repo.ensure_tables()
                team_rows = await team_repo.list_overrides_for_teams(team_ids)
                team_overrides = self._select_lowest_id_overrides(team_rows, id_field="team_id")
        except _PROFILE_NONCRITICAL_EXCEPTIONS as exc:
            logger.debug("Org/team overrides unavailable for user {}: {}", user_id, exc)

        effective: dict[str, Any] = {}
        for entry in entries:
            key = str(entry.key)
            if key in overrides:
                value = overrides.get(key)
                source = "user"
            elif key in team_overrides:
                value = team_overrides[key]["value"]
                source = "team"
            elif key in org_overrides:
                value = org_overrides[key]["value"]
                source = "org"
            else:
                if entry.default is None:
                    continue
                value = entry.default
                source = "default"
            effective[key] = self._format_value(
                key,
                value,
                include_sources=include_sources,
                source=source,
                catalog_map=catalog_map,
                mask_secrets=mask_secrets,
            )
        return effective

    async def _build_raw_overrides(
        self,
        user_id: int,
        *,
        mask_secrets: bool,
    ) -> dict[str, Any]:
        catalog = load_user_profile_catalog()
        catalog_map = self._catalog_entry_map(catalog)
        raw: dict[str, Any] = {"user": [], "orgs": [], "teams": []}

        repo = UserProfileOverridesRepo(self._db_pool)
        await repo.ensure_tables()
        user_rows = await repo.list_overrides_for_user(user_id)
        for row in user_rows:
            key = str(row.get("key") or "")
            if not key:
                continue
            raw["user"].append(self._build_raw_entry(row, key, catalog_map, mask_secrets=mask_secrets))

        try:
            org_ids, team_ids = await self._get_membership_ids(user_id)
            if org_ids:
                org_repo = OrgProfileOverridesRepo(self._db_pool)
                await org_repo.ensure_tables()
                org_rows = await org_repo.list_overrides_for_orgs(org_ids)
                for row in org_rows:
                    key = str(row.get("key") or "")
                    if not key:
                        continue
                    org_id = row.get("org_id")
                    raw["orgs"].append(
                        self._build_raw_entry(
                            row,
                            key,
                            catalog_map,
                            mask_secrets=mask_secrets,
                            org_id=int(org_id) if org_id is not None else None,
                        )
                    )
            if team_ids:
                team_repo = TeamProfileOverridesRepo(self._db_pool)
                await team_repo.ensure_tables()
                team_rows = await team_repo.list_overrides_for_teams(team_ids)
                for row in team_rows:
                    key = str(row.get("key") or "")
                    if not key:
                        continue
                    team_id = row.get("team_id")
                    raw["teams"].append(
                        self._build_raw_entry(
                            row,
                            key,
                            catalog_map,
                            mask_secrets=mask_secrets,
                            team_id=int(team_id) if team_id is not None else None,
                        )
                    )
        except _PROFILE_NONCRITICAL_EXCEPTIONS as exc:
            logger.debug("Raw override load failed for user {}: {}", user_id, exc)

        return raw

    def _build_raw_entry(
        self,
        row: dict[str, Any],
        key: str,
        catalog_map: dict[str, Any],
        *,
        mask_secrets: bool,
        org_id: int | None = None,
        team_id: int | None = None,
    ) -> dict[str, Any]:
        masked_value, masked, hint = self._mask_value(
            key,
            row.get("value"),
            catalog_map=catalog_map,
            mask_secrets=mask_secrets,
        )
        entry: dict[str, Any] = {
            "key": key,
            "value": masked_value,
            "updated_at": row.get("updated_at"),
            "updated_by": row.get("updated_by"),
        }
        if masked:
            entry["masked"] = True
        if hint:
            entry["hint"] = hint
        if org_id is not None:
            entry["org_id"] = org_id
        if team_id is not None:
            entry["team_id"] = team_id
        return entry

    @staticmethod
    def _select_lowest_id_overrides(
        rows: list[dict[str, Any]],
        *,
        id_field: str,
    ) -> dict[str, dict[str, Any]]:
        selected: dict[str, dict[str, Any]] = {}
        for row in rows:
            key = row.get("key")
            if not key:
                continue
            value = row.get("value")
            if value is None:
                continue
            try:
                entity_id = int(row.get(id_field))
            except _PROFILE_NONCRITICAL_EXCEPTIONS:
                continue
            current = selected.get(str(key))
            if current is None or entity_id < current["id"]:
                selected[str(key)] = {"value": value, "id": entity_id}
        return selected

    async def get_profile_version(
        self,
        *,
        user_id: int,
        user_updated_at: Any | None = None,
        db_conn: Any | None = None,
        lock_user: bool = False,
    ) -> datetime:
        del user_updated_at
        if db_conn is not None:
            return await self._profile_version_gateway.read_in_transaction(
                db_conn,
                user_id,
                lock_user=lock_user,
            )
        return await self._profile_version_gateway.read(user_id)

    async def lock_profile_users(
        self,
        *,
        user_ids: tuple[int, ...],
        db_conn: Any,
    ) -> dict[int, datetime]:
        """Lock profile user rows once in caller-supplied canonical order."""
        if user_ids != tuple(sorted(set(user_ids))) or any(
            type(user_id) is not int or user_id <= 0 for user_id in user_ids
        ):
            raise ValueError("Profile user locks require sorted unique positive IDs")
        return {
            user_id: await self._profile_version_gateway.read_in_transaction(
                db_conn,
                user_id,
                lock_user=True,
            )
            for user_id in user_ids
        }

    async def _get_byok_repo(self) -> AuthnzUserProviderSecretsRepo:
        repo = AuthnzUserProviderSecretsRepo(self._db_pool)
        await repo.ensure_tables()
        return repo

    async def build_security(
        self,
        *,
        user_id: int,
        session_manager: Any,
        api_key_manager: Any,
    ) -> dict[str, Any]:
        mfa_service = get_mfa_service()
        mfa_status = await mfa_service.get_user_mfa_status(user_id)
        sessions = await session_manager.get_user_sessions(user_id)
        api_key_rows = await api_key_manager.list_user_keys(
            user_id=user_id,
            include_revoked=False,
        )

        byok_keys: list[dict[str, Any]] = []
        try:
            repo = await self._get_byok_repo()
            secrets = await repo.list_secrets_for_user(user_id)
            byok_keys = [
                {"provider": str(row.get("provider") or ""), "has_key": True}
                for row in secrets
            ]
        except _PROFILE_NONCRITICAL_EXCEPTIONS as exc:
            logger.debug("BYOK metadata unavailable for user {}: {}", user_id, exc)

        return {
            "mfa_enabled": bool(mfa_status.get("enabled", False)),
            "active_sessions": len(sessions),
            "api_keys": [
                {
                    "id": int(row.get("id")),
                    "name": row.get("name"),
                    "last_used_at": row.get("last_used_at"),
                }
                for row in api_key_rows
            ],
            "byok_keys": byok_keys,
        }

    async def build_profile(
        self,
        *,
        user: dict[str, Any],
        sections: set[str] | None = None,
        security: dict[str, Any] | None = None,
        include_sources: bool = False,
        include_raw: bool = False,
        mask_secrets: bool = True,
        metrics_scope: str | None = None,
    ) -> dict[str, Any]:
        catalog = load_user_profile_catalog()
        registry = self._get_metrics_registry()
        scope_label = metrics_scope or "unknown"
        overall_start = time.perf_counter()
        requested = sections or set(DEFAULT_SECTIONS)
        section_errors: dict[str, str] = {}
        response: dict[str, Any] = {
            "catalog_version": catalog.version,
        }
        response["profile_version"] = await self.get_profile_version(
            user_id=int(user["id"]),
            user_updated_at=user.get("updated_at"),
        )

        unknown_sections = requested - KNOWN_SECTIONS
        for section in sorted(unknown_sections):
            section_errors[section] = "unknown_section"
        requested = requested - unknown_sections

        unsupported_sections = requested - SUPPORTED_SECTIONS
        for section in sorted(unsupported_sections):
            section_errors[section] = "section_not_supported"
        requested = requested - unsupported_sections

        if "identity" in requested:
            section_start = time.perf_counter()
            try:
                identity = self._build_identity(user)
                await self._attach_lockout_status(identity)
                response["user"] = identity
            except _PROFILE_NONCRITICAL_EXCEPTIONS as exc:
                logger.error("Failed to build identity section: {}", exc)
                section_errors["identity"] = "failed_to_build"
                if registry:
                    registry.increment(
                        "profile_errors_total",
                        1,
                        labels={"scope": scope_label, "section": "identity"},
                    )
            if registry:
                registry.observe(
                    "profile_section_latency_ms",
                    (time.perf_counter() - section_start) * 1000.0,
                    labels={"scope": scope_label, "section": "identity"},
                )

        if "memberships" in requested:
            section_start = time.perf_counter()
            try:
                response["memberships"] = await self._build_memberships(int(user["id"]))
            except _PROFILE_NONCRITICAL_EXCEPTIONS as exc:
                logger.error("Failed to build memberships section: {}", exc)
                section_errors["memberships"] = "failed_to_build"
                if registry:
                    registry.increment(
                        "profile_errors_total",
                        1,
                        labels={"scope": scope_label, "section": "memberships"},
                    )
            if registry:
                registry.observe(
                    "profile_section_latency_ms",
                    (time.perf_counter() - section_start) * 1000.0,
                    labels={"scope": scope_label, "section": "memberships"},
                )

        if "quotas" in requested:
            section_start = time.perf_counter()
            try:
                response["quotas"] = await self._build_quotas(user)
            except _PROFILE_NONCRITICAL_EXCEPTIONS as exc:
                logger.error("Failed to build quotas section: {}", exc)
                section_errors["quotas"] = "failed_to_build"
                if registry:
                    registry.increment(
                        "profile_errors_total",
                        1,
                        labels={"scope": scope_label, "section": "quotas"},
                    )
            if registry:
                registry.observe(
                    "profile_section_latency_ms",
                    (time.perf_counter() - section_start) * 1000.0,
                    labels={"scope": scope_label, "section": "quotas"},
                )

        effective_cache: dict[str, Any] | None = None
        effective_error: Exception | None = None

        async def _get_effective_config() -> dict[str, Any]:
            nonlocal effective_cache, effective_error
            if effective_cache is not None:
                return effective_cache
            if effective_error is not None:
                raise effective_error
            try:
                effective_cache = await self._build_effective_config(
                    int(user["id"]),
                    include_sources=include_sources,
                    mask_secrets=mask_secrets,
                )
                return effective_cache
            except _PROFILE_NONCRITICAL_EXCEPTIONS as exc:
                effective_error = exc
                raise

        if "preferences" in requested:
            section_start = time.perf_counter()
            try:
                effective = await _get_effective_config()
                response["preferences"] = {
                    key: value
                    for key, value in effective.items()
                    if str(key).startswith("preferences.")
                }
            except _PROFILE_NONCRITICAL_EXCEPTIONS as exc:
                logger.error("Failed to build preferences section: {}", exc)
                section_errors["preferences"] = "failed_to_build"
                if registry:
                    registry.increment(
                        "profile_errors_total",
                        1,
                        labels={"scope": scope_label, "section": "preferences"},
                    )
            if registry:
                registry.observe(
                    "profile_section_latency_ms",
                    (time.perf_counter() - section_start) * 1000.0,
                    labels={"scope": scope_label, "section": "preferences"},
                )

        if "effective_config" in requested:
            section_start = time.perf_counter()
            try:
                response["effective_config"] = await _get_effective_config()
            except _PROFILE_NONCRITICAL_EXCEPTIONS as exc:
                logger.error("Failed to build effective config section: {}", exc)
                section_errors["effective_config"] = "failed_to_build"
                if registry:
                    registry.increment(
                        "profile_errors_total",
                        1,
                        labels={"scope": scope_label, "section": "effective_config"},
                    )
            if registry:
                registry.observe(
                    "profile_section_latency_ms",
                    (time.perf_counter() - section_start) * 1000.0,
                    labels={"scope": scope_label, "section": "effective_config"},
                )

        if "security" in requested:
            section_start = time.perf_counter()
            if security is None:
                section_errors["security"] = "security_unavailable"
                if registry:
                    registry.increment(
                        "profile_errors_total",
                        1,
                        labels={"scope": scope_label, "section": "security"},
                    )
            else:
                response["security"] = security
            if registry:
                registry.observe(
                    "profile_section_latency_ms",
                    (time.perf_counter() - section_start) * 1000.0,
                    labels={"scope": scope_label, "section": "security"},
                )

        if include_raw:
            try:
                response["raw_overrides"] = await self._build_raw_overrides(
                    int(user["id"]),
                    mask_secrets=mask_secrets,
                )
            except _PROFILE_NONCRITICAL_EXCEPTIONS as exc:
                logger.error("Failed to build raw overrides section: {}", exc)
                section_errors["raw_overrides"] = "failed_to_build"

        if section_errors:
            response["section_errors"] = section_errors

        latency_ms = (time.perf_counter() - overall_start) * 1000.0
        if registry:
            registry.observe(
                "profile_fetch_latency_ms",
                latency_ms,
                labels={"scope": scope_label},
            )
        threshold_ms = self.profile_sla_threshold_ms(scope_label)
        if threshold_ms is not None and latency_ms > threshold_ms:
            logger.warning(
                "Profile SLA exceeded for scope {}: {:.2f}ms (threshold {}ms)",
                scope_label,
                latency_ms,
                threshold_ms,
            )
            if registry:
                registry.increment(
                    "profile_sla_breach_total",
                    1,
                    labels={"scope": scope_label},
                )
        return response
