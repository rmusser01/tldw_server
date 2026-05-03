from __future__ import annotations

import hashlib
import json
import os
from collections import defaultdict
from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from typing import Any

from loguru import logger

from tldw_Server_API.app.api.v1.utils.pagination import build_page_pagination_meta
from tldw_Server_API.app.core.AuthNZ.database import DatabasePool, get_db_pool
from tldw_Server_API.app.core.AuthNZ.privilege_catalog import PrivilegeCatalog, ScopeEntry, load_catalog
from tldw_Server_API.app.core.AuthNZ.settings import is_single_user_profile_mode
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_single_user_instance
from tldw_Server_API.app.core.PrivilegeMaps.cache import get_privilege_cache
from tldw_Server_API.app.core.PrivilegeMaps.introspection import (
    RouteMetadata,
    collect_privilege_route_registry,
)
from tldw_Server_API.app.core.PrivilegeMaps.trends import PrivilegeTrendStore, get_privilege_trend_store

RESOURCE_FALLBACK = "uncategorized"
MAX_DETAIL_ROWS = 50_000


class PaginationLimitExceeded(Exception):
    """Raised when a client requests pagination beyond the allowed result window."""

_PRIVILEGE_MAP_COERCE_EXCEPTIONS = (
    AttributeError,
    TypeError,
    ValueError,
)

_PRIVILEGE_MAP_NONCRITICAL_EXCEPTIONS = (
    AssertionError,
    AttributeError,
    ConnectionError,
    ImportError,
    KeyError,
    LookupError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
)


class PrivilegeMapService:
    """Aggregates privilege information for organization, team, and user views."""

    def __init__(
        self,
        *,
        route_registry: dict[str, list[RouteMetadata]] | None = None,
        catalog: PrivilegeCatalog | None = None,
        trend_store: PrivilegeTrendStore | None = None,
    ) -> None:
        self._catalog: PrivilegeCatalog = catalog or load_catalog()
        self._route_registry: dict[str, list[RouteMetadata]] = route_registry or self._collect_route_registry()
        self._placeholder_routes: dict[str, list[RouteMetadata]] = {}
        self._role_scope_map: dict[str, list[ScopeEntry]] = self._build_role_scope_map()
        self._admin_roles: set[str] = {"admin", "owner", "platform_admin"}
        self._feature_flag_map = {flag.id: flag for flag in self.catalog.feature_flags}
        self._scope_lookup = {scope.id: scope for scope in self.catalog.scopes}
        self._summary_cache = get_privilege_cache()
        self._cache_ttl_seconds = self._resolve_cache_ttl()
        self._trend_store = trend_store or get_privilege_trend_store()
        self._route_signature = self._compute_route_signature()

    @property
    def catalog(self) -> PrivilegeCatalog:
        return self._catalog

    def invalidate_cache(self) -> None:
        """Manually clear cached summaries."""
        self._summary_cache.invalidate()

    def _resolve_cache_ttl(self) -> int:
        try:
            ttl = int(os.getenv("PRIVILEGE_MAP_CACHE_TTL_SECONDS", "900") or "900")
        except _PRIVILEGE_MAP_COERCE_EXCEPTIONS:
            ttl = 900
        # Enforce a sensible floor to avoid thrashing.
        return max(ttl, 10)

    def _compute_route_signature(self) -> str:
        try:
            serialized: list[tuple[Any, ...]] = []
            for scope_id, routes in sorted(self._route_registry.items()):
                for route in sorted(routes, key=lambda r: (r.path, r.methods_or_any)):
                    dependencies_signature = tuple(
                        (dep.id, dep.type, dep.module or "")
                        for dep in (route.dependencies or ())
                    )
                    serialized.append(
                        (
                            scope_id,
                            route.path,
                            tuple(route.methods_or_any),
                            route.endpoint or "",
                            tuple(route.tags or ()),
                            dependencies_signature,
                            tuple(route.rate_limit_resources or ()),
                        )
                    )
            payload = json.dumps(serialized, sort_keys=True, default=list)
            return hashlib.sha256(payload.encode("utf-8")).hexdigest()
        except _PRIVILEGE_MAP_COERCE_EXCEPTIONS:
            return "static"

    def _compute_user_signature(self, users: Sequence[dict[str, Any]]) -> str:
        serializable: list[dict[str, Any]] = []
        for user in sorted(users, key=lambda u: str(u.get("id"))):
            serializable.append(
                {
                    "id": str(user.get("id")),
                    "roles": sorted(user.get("roles", [])),
                    "permissions": sorted(user.get("permissions", [])),
                    "scopes": sorted(user.get("allowed_scopes", [])),
                    "feature_flags": sorted(user.get("feature_flags", [])),
                }
            )
        payload = json.dumps(serializable, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _summary_cache_key(
        self,
        *,
        scope: str,
        group_by: str,
        team_id: str | None,
        users_signature: str,
    ) -> str:
        team_component = str(team_id or "__none__")
        return "::".join(
            [
                scope,
                group_by,
                team_component,
                users_signature,
                self._route_signature,
                self.catalog.version,
                str(self._summary_cache.generation),
            ]
        )

    @staticmethod
    def _normalize_timestamp(value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    @staticmethod
    def _normalize_role_name(role: str | None) -> str:
        return (role or "user").strip().lower()

    def _coerce_summary_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not payload:
            return payload
        base_payload = dict(payload)
        generated_at = base_payload.get("generated_at")
        if isinstance(generated_at, str):
            try:
                base_payload["generated_at"] = datetime.fromisoformat(generated_at.replace("Z", "+00:00")).astimezone(timezone.utc)
            except _PRIVILEGE_MAP_COERCE_EXCEPTIONS:
                base_payload["generated_at"] = self._normalize_timestamp(datetime.now(timezone.utc))
        return base_payload

    async def get_org_summary(
        self,
        *,
        group_by: str,
        include_trends: bool,
        since: datetime | None,
        org_id: str | None = None,
    ) -> dict[str, Any]:
        users = await self._fetch_users()
        if org_id:
            users = await self._filter_users_for_org(users, org_id=org_id)
        users_signature = self._compute_user_signature(users)
        cache_key = self._summary_cache_key(
            scope="org",
            group_by=group_by,
            team_id=None,
            users_signature=users_signature,
        )
        cached = self._summary_cache.get(cache_key)
        if cached:
            base_payload = self._coerce_summary_payload(cached)
        else:
            generated_at = self._normalize_timestamp(datetime.now(timezone.utc))
            if group_by == "resource":
                buckets = self._group_by_resource(users)
            elif group_by == "team":
                buckets = await self._group_by_team(users)
            else:
                buckets = self._group_by_role(users)
            base_payload = {
                "generated_at": generated_at,
                "buckets": buckets,
            }
            self._summary_cache.set(cache_key, base_payload, ttl_sec=self._cache_ttl_seconds)
            try:
                await self._record_trend_snapshot(
                    scope="org",
                    group_by=group_by,
                    buckets=buckets,
                    generated_at=generated_at,
                    team_id=None,
                )
            except _PRIVILEGE_MAP_NONCRITICAL_EXCEPTIONS as exc:
                logger.debug("Unable to record privilege trend snapshot: {}", exc)

        generated_at = base_payload["generated_at"]
        buckets = base_payload["buckets"]
        trends: list[dict[str, Any]] = []
        if include_trends:
            trends = await self._build_trends(
                scope="org",
                group_by=group_by,
                buckets=buckets,
                generated_at=generated_at,
                since=since,
                team_id=None,
            )

        return {
            "catalog_version": self.catalog.version,
            "generated_at": generated_at,
            "group_by": group_by,
            "buckets": buckets,
            "trends": trends,
            "metadata": {
                "filters": {
                    "group_by": group_by,
                    "include_trends": include_trends,
                    "since": since.isoformat() if since else None,
                }
            },
        }

    async def get_org_detail(
        self,
        *,
        page: int,
        page_size: int,
        resource: str | None,
        dependency: str | None,
        role_filter: str | None,
        org_id: str | None = None,
    ) -> dict[str, Any]:
        users = await self._fetch_users()
        if org_id:
            users = await self._filter_users_for_org(users, org_id=org_id)
        items = self._build_detail_items(
            users,
            resource_filter=resource,
            dependency_filter=dependency,
            role_filter=role_filter,
        )
        return self._paginate_detail(items, page=page, page_size=page_size)

    async def get_team_summary(
        self,
        *, team_id: str, group_by: str, include_trends: bool, since: datetime | None
    ) -> dict[str, Any]:
        users = await self._fetch_users()
        team_users = await self._filter_users_for_team(users, team_id=team_id)
        users_signature = self._compute_user_signature(team_users)
        cache_key = self._summary_cache_key(
            scope="team",
            group_by=group_by,
            team_id=team_id,
            users_signature=users_signature,
        )
        cached = self._summary_cache.get(cache_key)
        if cached:
            base_payload = self._coerce_summary_payload(cached)
        else:
            generated_at = self._normalize_timestamp(datetime.now(timezone.utc))
            if group_by == "resource":
                buckets = self._group_by_resource(team_users)
            else:
                buckets = self._group_by_member(team_users)
            base_payload = {
                "generated_at": generated_at,
                "buckets": buckets,
            }
            self._summary_cache.set(cache_key, base_payload, ttl_sec=self._cache_ttl_seconds)
            try:
                await self._record_trend_snapshot(
                    scope="team",
                    group_by=group_by,
                    buckets=buckets,
                    generated_at=generated_at,
                    team_id=team_id,
                )
            except _PRIVILEGE_MAP_NONCRITICAL_EXCEPTIONS as exc:
                logger.debug("Unable to record team privilege trend snapshot: {}", exc)

        generated_at = base_payload["generated_at"]
        buckets = base_payload["buckets"]
        trends: list[dict[str, Any]] = []
        if include_trends:
            trends = await self._build_trends(
                scope="team",
                group_by=group_by,
                buckets=buckets,
                generated_at=generated_at,
                since=since,
                team_id=team_id,
            )

        return {
            "catalog_version": self.catalog.version,
            "generated_at": generated_at,
            "group_by": group_by,
            "buckets": buckets,
            "trends": trends,
            "metadata": {
                "filters": {
                    "team_id": team_id,
                    "group_by": group_by,
                    "include_trends": include_trends,
                    "since": since.isoformat() if since else None,
                }
            },
        }

    async def get_team_detail(
        self,
        *,
        team_id: str,
        page: int,
        page_size: int,
        resource: str | None,
        dependency: str | None,
        role_filter: str | None,
    ) -> dict[str, Any]:
        users = await self._fetch_users()
        team_users = await self._filter_users_for_team(users, team_id=team_id)
        items = self._build_detail_items(
            team_users,
            resource_filter=resource,
            dependency_filter=dependency,
            role_filter=role_filter,
            restrict_to_team=team_id,
        )
        return self._paginate_detail(items, page=page, page_size=page_size)

    async def get_user_detail(
        self,
        *,
        user_id: str,
        page: int,
        page_size: int,
        resource: str | None,
    ) -> dict[str, Any]:
        users = await self._fetch_users()
        filtered = [u for u in users if str(u["id"]) == str(user_id)]
        if not filtered:
            logger.debug("Privilege detail requested for unknown user_id={}", user_id)
        items = self._build_detail_items(filtered, resource_filter=resource)
        return self._paginate_detail(items, page=page, page_size=page_size)

    async def get_self_map(
        self,
        *,
        user_id: str,
        resource: str | None,
    ) -> dict[str, Any]:
        users = await self._fetch_users()
        filtered = [u for u in users if str(u["id"]) == str(user_id)]
        items = self._build_detail_items(filtered, resource_filter=resource)
        condensed_items = [
            {
                "endpoint": item["endpoint"],
                "method": item["method"],
                "privilege_scope_id": item["privilege_scope_id"],
                "feature_flag_id": item["feature_flag_id"],
                "sensitivity_tier": item["sensitivity_tier"],
                "ownership_predicates": item["ownership_predicates"],
                "status": item["status"],
                "blocked_reason": item["blocked_reason"],
                "dependencies": item.get("dependencies", []),
                "dependency_sources": item.get("dependency_sources", []),
                "rate_limit_class": item.get("rate_limit_class"),
                "rate_limit_resources": item.get("rate_limit_resources", []),
                "source_module": item.get("source_module"),
                "summary": item.get("summary"),
                "tags": item.get("tags", []),
            }
            for item in items
        ]
        recommended_actions = self._build_recommended_actions(items)
        return {
            "catalog_version": self.catalog.version,
            "generated_at": datetime.now(timezone.utc),
            "items": condensed_items,
            "recommended_actions": recommended_actions,
        }

    async def build_snapshot_summary(
        self,
        *,
        target_scope: str,
        org_id: str | None,
        team_id: str | None,
        user_ids: Sequence[str] | None,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        users = await self._fetch_users()
        if target_scope == "team":
            if not team_id:
                raise ValueError("team_id required for team scope snapshot")
            users = await self._filter_users_for_team(users, team_id=team_id)
        elif target_scope == "org":
            if not org_id:
                raise ValueError("org_id required for org scope snapshot")
            users = await self._filter_users_for_org(users, org_id=org_id)
        elif target_scope == "user":
            target_ids = {str(uid) for uid in (user_ids or []) if uid}
            if not target_ids:
                raise ValueError("user_ids required for user scope snapshot")
            users = [user for user in users if str(user.get("id")) in target_ids]

        scope_ids: set[str] = set()
        sensitivity_breakdown: dict[str, int] = defaultdict(int)
        endpoint_keys: set[tuple[str, str]] = set()
        for user in users:
            for scope_id in user.get("allowed_scopes", set()):
                scope_ids.add(scope_id)

        for scope_id in scope_ids:
            scope = self._scope_lookup.get(scope_id)
            if scope:
                sensitivity_breakdown[scope.sensitivity_tier] += 1
                for route_meta in self._routes_for_scope(scope):
                    for method in route_meta.methods_or_any:
                        endpoint_keys.add((route_meta.path, method))

        summary = {
            "users": len(users),
            "scopes": len(scope_ids),
            "endpoints": len(endpoint_keys) if endpoint_keys else len(scope_ids),
            "scope_ids": sorted(scope_ids),
            "sensitivity_breakdown": dict(sensitivity_breakdown),
            "endpoint_paths": sorted({path for (path, _method) in endpoint_keys}),
        }
        return summary, users

    def build_snapshot_detail(
        self,
        users: Sequence[dict[str, Any]],
        *,
        resource: str | None = None,
        role_filter: str | None = None,
        restrict_to_team: str | None = None,
    ) -> list[dict[str, Any]]:
        return self._build_detail_items(
            users,
            resource_filter=resource,
            role_filter=role_filter,
            restrict_to_team=restrict_to_team,
        )

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #

    def _build_role_scope_map(self) -> dict[str, list[ScopeEntry]]:
        mapping: dict[str, dict[str, ScopeEntry]] = defaultdict(dict)
        for scope in self.catalog.scopes:
            for role in scope.default_roles or []:
                normalized_role = role.strip().lower()
                if not normalized_role:
                    continue
                mapping[normalized_role][scope.id] = scope

        # Admin roles automatically get access to all scopes
        for admin_role in ("admin", "owner", "platform_admin"):
            mapping[admin_role] = {scope.id: scope for scope in self.catalog.scopes}
        return {role: list(scopes.values()) for role, scopes in mapping.items()}

    def _collect_route_registry(self) -> dict[str, list[RouteMetadata]]:
        try:
            from tldw_Server_API.app.main import app as fastapi_app

            return collect_privilege_route_registry(fastapi_app, self.catalog)
        except _PRIVILEGE_MAP_NONCRITICAL_EXCEPTIONS as exc:
            logger.warning("Privilege route introspection failed: {}", exc)
            return {}

    def _routes_for_scope(self, scope: ScopeEntry) -> list[RouteMetadata]:
        routes = self._route_registry.get(scope.id)
        if routes:
            return routes
        cached = self._placeholder_routes.get(scope.id)
        if cached:
            return cached
        placeholder = self._create_placeholder_route(scope)
        self._placeholder_routes[scope.id] = [placeholder]
        return [placeholder]

    def _create_placeholder_route(self, scope: ScopeEntry) -> RouteMetadata:
        return RouteMetadata(
            path=self._scope_to_endpoint(scope),
            methods=("ANY",),
            name=scope.id,
            tags=tuple(scope.resource_tags or [RESOURCE_FALLBACK]),
            endpoint="",
            dependencies=(),
            dependency_sources=(),
            rate_limit_resources=(),
            summary=scope.description,
            description=scope.description,
        )

    async def _fetch_users(self) -> list[dict[str, Any]]:
        try:
            pool: DatabasePool = await get_db_pool()
            rows = await pool.fetchall(
                "SELECT id, username, role, is_active FROM users ORDER BY id"
            )
            if not rows:
                raise ValueError("No users returned")

            base_users: dict[str, dict[str, Any]] = {}
            for row in rows:
                record = self._row_to_dict(row)
                if not record or not record.get("is_active", 1):
                    continue
                user_id = str(record.get("id"))
                base_users[user_id] = {
                    "id": record.get("id"),
                    "username": record.get("username", ""),
                    "primary_role": (record.get("role") or "user"),
                    "roles": [],
                    "permissions": set(),
                }

            if not base_users:
                raise ValueError("No active users found")

            await self._hydrate_roles_and_permissions(pool, base_users)
            users: list[dict[str, Any]] = []
            for payload in base_users.values():
                roles = payload.get("roles") or [payload.get("primary_role")]
                roles = sorted({role for role in roles if role})
                permissions = {perm for perm in payload.get("permissions", set()) if perm}
                feature_flags = self._feature_flags_for_user(roles, permissions)
                allowed_scopes = self._resolve_scopes_for_user(roles, permissions)
                primary_role = roles[0] if roles else payload.get("primary_role", "user")
                users.append(
                    {
                        "id": payload["id"],
                        "username": payload["username"],
                        "primary_role": primary_role,
                        "roles": roles,
                        "permissions": sorted(permissions),
                        "feature_flags": feature_flags,
                        "allowed_scopes": allowed_scopes,
                    }
                )
            return users
        except _PRIVILEGE_MAP_NONCRITICAL_EXCEPTIONS as exc:
            logger.debug("Falling back to single-user privilege dataset: {}", exc)

        # Fallback: single-user-style profile or empty DB
        single_user = get_single_user_instance()
        default_role = (
            "admin"
            if is_single_user_profile_mode()
            else (single_user.roles[0] if single_user.roles else "admin")
        )
        feature_flags = self._feature_flags_for_user([default_role], set())
        allowed_scopes = self._resolve_scopes_for_user([default_role], [])
        if not allowed_scopes:
            allowed_scopes = {scope.id for scope in self.catalog.scopes}
        return [
            {
                "id": single_user.id,
                "username": single_user.username,
                "primary_role": default_role or "admin",
                "roles": [default_role or "admin"],
                "permissions": [],
                "feature_flags": feature_flags,
                "allowed_scopes": allowed_scopes,
            }
        ]

    async def _fetch_team_memberships(self) -> list[dict[str, Any]]:
        try:
            pool: DatabasePool = await get_db_pool()
            rows = await pool.fetchall(
                """
                SELECT tm.team_id, tm.user_id, tm.role AS membership_role,
                       t.name AS team_name, t.org_id
                FROM team_members tm
                JOIN teams t ON tm.team_id = t.id
                """
            )
            memberships: list[dict[str, Any]] = []
            for row in rows:
                record = self._row_to_dict(row)
                if record:
                    memberships.append(record)
            return memberships
        except _PRIVILEGE_MAP_NONCRITICAL_EXCEPTIONS as exc:
            logger.debug("Unable to load team memberships: {}", exc)
            return []

    async def _fetch_org_memberships(self) -> list[dict[str, Any]]:
        try:
            pool: DatabasePool = await get_db_pool()
            rows = await pool.fetchall(
                """
                SELECT om.org_id, om.user_id, om.role AS membership_role
                FROM org_members om
                """
            )
            memberships: list[dict[str, Any]] = []
            for row in rows:
                record = self._row_to_dict(row)
                if record:
                    memberships.append(record)
            return memberships
        except _PRIVILEGE_MAP_NONCRITICAL_EXCEPTIONS as exc:
            logger.debug("Unable to load org memberships: {}", exc)
            return []

    async def _filter_users_for_team(
        self,
        users: list[dict[str, Any]],
        *,
        team_id: str,
    ) -> list[dict[str, Any]]:
        memberships = await self._fetch_team_memberships()
        user_ids = {
            str(m["user_id"])
            for m in memberships
            if str(m.get("team_id")) == str(team_id)
        }
        if not user_ids:
            return []
        return [user for user in users if str(user["id"]) in user_ids]

    async def _filter_users_for_org(
        self,
        users: list[dict[str, Any]],
        *,
        org_id: str,
    ) -> list[dict[str, Any]]:
        memberships = await self._fetch_org_memberships()
        user_ids = {
            str(m["user_id"])
            for m in memberships
            if str(m.get("org_id")) == str(org_id)
        }
        if not user_ids:
            return []
        return [user for user in users if str(user["id"]) in user_ids]

    async def _hydrate_roles_and_permissions(
        self,
        pool: DatabasePool,
        base_users: dict[str, dict[str, Any]],
    ) -> None:
        role_assignments: dict[str, set[str]] = defaultdict(set)
        try:
            rows = await pool.fetchall(
                """
                SELECT ur.user_id, r.name AS role_name
                FROM user_roles ur
                JOIN roles r ON ur.role_id = r.id
                """
            )
            for row in rows:
                record = self._row_to_dict(row)
                if not record:
                    continue
                user_id = str(record.get("user_id"))
                role_name = record.get("role_name")
                if user_id in base_users and role_name:
                    role_assignments[user_id].add(role_name)
        except _PRIVILEGE_MAP_NONCRITICAL_EXCEPTIONS as exc:
            logger.debug("Unable to load role assignments: {}", exc)

        role_permissions: dict[str, set[str]] = defaultdict(set)
        try:
            rows = await pool.fetchall(
                """
                SELECT r.name AS role_name, p.name AS permission_name
                FROM role_permissions rp
                JOIN roles r ON rp.role_id = r.id
                JOIN permissions p ON rp.permission_id = p.id
                """
            )
            for row in rows:
                record = self._row_to_dict(row)
                if not record:
                    continue
                role_name = record.get("role_name")
                permission_name = record.get("permission_name")
                if role_name and permission_name:
                    role_permissions[role_name].add(permission_name)
        except _PRIVILEGE_MAP_NONCRITICAL_EXCEPTIONS as exc:
            logger.debug("Unable to load role permissions: {}", exc)

        user_permissions_direct: dict[str, set[str]] = defaultdict(set)
        try:
            rows = await pool.fetchall(
                """
                SELECT up.user_id, p.name AS permission_name, up.granted
                FROM user_permissions up
                JOIN permissions p ON up.permission_id = p.id
                """
            )
            for row in rows:
                record = self._row_to_dict(row)
                if not record:
                    continue
                granted = record.get("granted")
                if granted is not None and not bool(granted):
                    continue
                user_id = str(record.get("user_id"))
                permission_name = record.get("permission_name")
                if user_id in base_users and permission_name:
                    user_permissions_direct[user_id].add(permission_name)
        except _PRIVILEGE_MAP_NONCRITICAL_EXCEPTIONS as exc:
            logger.debug("Unable to load direct user permissions: {}", exc)

        for user_id, payload in base_users.items():
            roles = sorted(role_assignments.get(user_id, set())) or [payload.get("primary_role")]
            payload["roles"] = roles
            perms: set[str] = set(user_permissions_direct.get(user_id, set()))
            for role in roles:
                perms.update(role_permissions.get(role, set()))
            payload["permissions"] = perms

    def _feature_flags_for_user(
        self,
        roles: Sequence[str],
        permissions: Sequence[str],
    ) -> set[str]:
        enabled: set[str] = set()
        role_set = {role.lower() for role in roles if role}
        perm_set = {perm.lower() for perm in permissions if perm}
        is_admin = bool(role_set & self._admin_roles)

        for flag_id, flag in self._feature_flag_map.items():
            if is_admin:
                enabled.add(flag_id)
                continue
            if getattr(flag, "default_state", "disabled") == "enabled":
                enabled.add(flag_id)
                continue
            allowed_roles = {r.lower() for r in getattr(flag, "allowed_roles", []) or []}
            if role_set & allowed_roles:
                enabled.add(flag_id)
                continue
            candidate_permissions = {
                flag_id.lower(),
                f"feature_flag:{flag_id}".lower(),
                f"feature_flag.{flag_id}".lower(),
            }
            if perm_set & candidate_permissions:
                enabled.add(flag_id)
        return enabled

    def _resolve_scopes_for_user(
        self,
        roles: Sequence[str],
        permissions: Sequence[str],
    ) -> set[str]:
        normalized_roles = [role.lower() for role in roles if role]
        role_set = set(normalized_roles)
        perm_set = {perm.lower() for perm in permissions if perm}

        if role_set & self._admin_roles:
            return {scope.id for scope in self.catalog.scopes}

        allowed: set[str] = set()
        for scope in self.catalog.scopes:
            granted = False
            for role in normalized_roles:
                for mapped_scope in self._role_scope_map.get(role, []):
                    if mapped_scope.id == scope.id:
                        granted = True
                        break
                if granted:
                    break

            if not granted:
                candidate_permissions = {
                    scope.id.lower(),
                    scope.id.replace(".", "_").lower(),
                    f"scope:{scope.id}".lower(),
                    f"scope.{scope.id}".lower(),
                    f"{scope.id}:read".lower(),
                    f"{scope.id}:write".lower(),
                }
                if perm_set & candidate_permissions:
                    granted = True

            if granted:
                allowed.add(scope.id)
        return allowed

    def _scopes_from_ids(self, scope_ids: Sequence[str]) -> list[ScopeEntry]:
        return [self._scope_lookup[sid] for sid in scope_ids if sid in self._scope_lookup]

    def _group_by_role(self, users: list[dict[str, Any]]) -> list[dict[str, Any]]:
        buckets: dict[str, dict[str, Any]] = {}
        for user in users:
            role = user.get("primary_role") or "user"
            role_key = self._normalize_role_name(role)
            scopes = set(user.get("allowed_scopes", set()))
            bucket = buckets.setdefault(
                role_key,
                {"key": role_key, "users": 0, "scopes": set(), "endpoints": set()},
            )
            bucket["users"] += 1
            bucket["scopes"].update(scopes)
            for scope_id in scopes:
                scope_entry = self._scope_lookup.get(scope_id)
                if not scope_entry:
                    continue
                for route_meta in self._routes_for_scope(scope_entry):
                    for method in route_meta.methods_or_any:
                        bucket["endpoints"].add((route_meta.path, method))

        results: list[dict[str, Any]] = []
        for role, payload in buckets.items():
            scope_count = len(payload["scopes"])
            endpoint_count = len(payload["endpoints"]) if payload["endpoints"] else scope_count
            results.append(
                {
                    "key": role,
                    "users": payload["users"],
                    "endpoints": endpoint_count,
                    "scopes": scope_count,
                }
            )
        return results

    async def _group_by_team(self, users: list[dict[str, Any]]) -> list[dict[str, Any]]:
        memberships = await self._fetch_team_memberships()
        by_team: dict[str, dict[str, Any]] = {}
        for entry in memberships:
            team_id = str(entry.get("team_id"))
            team = by_team.setdefault(
                team_id,
                {
                    "key": team_id,
                    "team_name": entry.get("team_name"),
                    "org_id": entry.get("org_id"),
                    "users": set(),
                    "scopes": set(),
                    "endpoints": set(),
                },
            )
            team["users"].add(str(entry.get("user_id")))
            user = next((u for u in users if str(u["id"]) == str(entry.get("user_id"))), None)
            if user:
                team_scopes = set(user.get("allowed_scopes", set()))
                team["scopes"].update(team_scopes)
                for scope_id in team_scopes:
                    scope_entry = self._scope_lookup.get(scope_id)
                    if not scope_entry:
                        continue
                    for route_meta in self._routes_for_scope(scope_entry):
                        for method in route_meta.methods_or_any:
                            team["endpoints"].add((route_meta.path, method))

        buckets: list[dict[str, Any]] = []
        for team in by_team.values():
            endpoint_count = len(team["endpoints"]) if team["endpoints"] else len(team["scopes"])
            buckets.append(
                {
                    "key": team["key"],
                    "users": len(team["users"]),
                    "endpoints": endpoint_count,
                    "scopes": len(team["scopes"]),
                    "metadata": {
                        "team_name": team.get("team_name"),
                        "org_id": team.get("org_id"),
                    },
                }
            )
        return buckets

    def _group_by_member(self, users: list[dict[str, Any]]) -> list[dict[str, Any]]:
        buckets: list[dict[str, Any]] = []
        for user in users:
            scopes = set(user.get("allowed_scopes", set()))
            endpoints: set[tuple[str, str]] = set()
            for scope_id in scopes:
                scope_entry = self._scope_lookup.get(scope_id)
                if not scope_entry:
                    continue
                for route_meta in self._routes_for_scope(scope_entry):
                    for method in route_meta.methods_or_any:
                        endpoints.add((route_meta.path, method))
            buckets.append(
                {
                    "key": str(user.get("id")),
                    "users": 1,
                    "endpoints": len(endpoints) if endpoints else len(scopes),
                    "scopes": len(scopes),
                    "metadata": {
                        "username": user.get("username"),
                        "primary_role": user.get("primary_role"),
                        "roles": user.get("roles"),
                    },
                }
            )
        return buckets

    def _group_by_resource(self, users: list[dict[str, Any]]) -> list[dict[str, Any]]:
        resource_access: dict[str, dict[str, Any]] = {}
        for user in users:
            for scope in self._scopes_from_ids(user.get("allowed_scopes", set())):
                tags = scope.resource_tags or [RESOURCE_FALLBACK]
                for tag in tags:
                    bucket = resource_access.setdefault(
                        tag,
                        {"key": tag, "users": set(), "scopes": set(), "endpoints": set()},
                    )
                    bucket["users"].add(str(user["id"]))
                    bucket["scopes"].add(scope.id)
                    for route_meta in self._routes_for_scope(scope):
                        for method in route_meta.methods_or_any:
                            bucket["endpoints"].add((route_meta.path, method))

        results: list[dict[str, Any]] = []
        for tag, payload in resource_access.items():
            endpoint_count = len(payload["endpoints"]) if payload["endpoints"] else len(payload["scopes"])
            results.append(
                {
                    "key": tag,
                    "users": len(payload["users"]),
                    "endpoints": endpoint_count,
                    "scopes": len(payload["scopes"]),
                }
            )
        return results

    def _scopes_for_role(self, role: str | None) -> list[ScopeEntry]:
        normalized = (role or "user").lower()
        if normalized in self._admin_roles:
            return list(self.catalog.scopes)
        if normalized in self._role_scope_map:
            return self._role_scope_map[normalized]
        return []

    def _build_detail_items(
        self,
        users: Sequence[dict[str, Any]],
        *,
        resource_filter: str | None,
        dependency_filter: str | None = None,
        role_filter: str | None = None,
        restrict_to_team: str | None = None,
    ) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        resource_filter_lower = resource_filter.lower() if resource_filter else None
        dependency_filter_lower = dependency_filter.lower() if dependency_filter else None
        normalized_role_filter = self._normalize_role_name(role_filter) if role_filter else None
        for user in users:
            role = user.get("primary_role") or "user"
            if normalized_role_filter and self._normalize_role_name(role) != normalized_role_filter:
                continue
            allowed_scopes = user.get("allowed_scopes", set())
            feature_flags = user.get("feature_flags", set())
            for scope in self.catalog.scopes:
                route_entries = self._routes_for_scope(scope)
                for route_meta in route_entries:
                    tag_values: list[str] = list(scope.resource_tags or [])
                    tag_values.extend(list(route_meta.tags or ()))
                    if not tag_values:
                        tag_values = [RESOURCE_FALLBACK]
                    if resource_filter_lower and resource_filter_lower not in {t.lower() for t in tag_values}:
                        continue
                    normalized_tag_values = list(dict.fromkeys(tag_values))
                    for method in route_meta.methods_or_any:
                        status = "allowed"
                        blocked_reason = None
                        if scope.feature_flag_id and scope.feature_flag_id not in feature_flags:
                            status = "blocked"
                            blocked_reason = "feature_flag_disabled"
                        elif scope.id not in allowed_scopes:
                            status = "blocked"
                            blocked_reason = "missing_scope"

                        dependency_entries = [
                            {
                                "id": dep.id,
                                "type": dep.type,
                                "module": dep.module,
                            }
                            for dep in route_meta.dependencies
                        ]
                        rate_limit_entries = [
                            {
                                "id": f"ratelimit.{resource}",
                                "type": "rate_limit",
                                "module": None,
                            }
                            for resource in route_meta.rate_limit_resources
                        ]
                        if dependency_filter_lower:
                            candidate_ids = {entry["id"].lower() for entry in dependency_entries}
                            candidate_ids.update(entry["id"].lower() for entry in rate_limit_entries)
                            if dependency_filter_lower not in candidate_ids:
                                continue

                        dependency_payload = dependency_entries + rate_limit_entries
                        source_module = None
                        if route_meta.endpoint:
                            source_module = route_meta.endpoint.rsplit(".", 1)[0]

                        items.append(
                            {
                                "user_id": str(user.get("id")),
                                "user_name": user.get("username") or "",
                                "role": role,
                                "endpoint": route_meta.path,
                                "method": method,
                                "privilege_scope_id": scope.id,
                                "feature_flag_id": scope.feature_flag_id,
                                "sensitivity_tier": scope.sensitivity_tier,
                                "ownership_predicates": scope.ownership_predicates or [],
                                "status": status,
                                "blocked_reason": blocked_reason,
                                "dependencies": dependency_payload,
                                "dependency_sources": list(route_meta.dependency_sources),
                                "rate_limit_class": scope.rate_limit_class,
                                "rate_limit_resources": list(route_meta.rate_limit_resources),
                                "source_module": source_module,
                                "summary": route_meta.summary or route_meta.description or scope.description,
                                "tags": normalized_tag_values,
                            }
                        )
        return items

    def _paginate_detail(
        self,
        items: Sequence[dict[str, Any]],
        *,
        page: int,
        page_size: int,
    ) -> dict[str, Any]:
        total_items = len(items)
        if page_size > MAX_DETAIL_ROWS:
            raise PaginationLimitExceeded("Requested page_size exceeds maximum allowed rows.")
        if page_size <= 0:
            page_size = 1
        if page <= 0:
            page = 1
        start = (page - 1) * page_size
        if start >= MAX_DETAIL_ROWS:
            raise PaginationLimitExceeded("Requested pagination exceeds allowed result window.")
        end = min(start + page_size, len(items))
        paginated = items[start:end]
        total_pages = (total_items + page_size - 1) // page_size
        return {
            "catalog_version": self.catalog.version,
            "generated_at": datetime.now(timezone.utc),
            "page": page,
            "page_size": page_size,
            "total_items": total_items,
            "pagination": build_page_pagination_meta(
                page=page,
                per_page=page_size,
                total=total_items,
                total_pages=total_pages,
            ),
            "items": paginated,
        }

    async def _record_trend_snapshot(
        self,
        *,
        scope: str,
        group_by: str,
        buckets: Sequence[dict[str, Any]],
        generated_at: datetime,
        team_id: str | None,
    ) -> None:
        if not buckets:
            return
        normalized_ts = self._normalize_timestamp(generated_at)
        await self._trend_store.record_snapshot(
            scope=scope,
            group_by=group_by,
            catalog_version=self.catalog.version,
            generated_at=normalized_ts,
            buckets=buckets,
            team_id=team_id,
        )

    async def _build_trends(
        self,
        *,
        scope: str,
        group_by: str,
        buckets: Sequence[dict[str, Any]],
        generated_at: datetime,
        since: datetime | None,
        team_id: str | None,
    ) -> list[dict[str, Any]]:
        if not buckets:
            return []
        window_end = self._normalize_timestamp(generated_at)
        window_start = self._normalize_timestamp(since) if since else window_end - timedelta(days=30)
        if window_start > window_end:
            window_start = window_end - timedelta(days=30)
        bucket_counts = {}
        for bucket in buckets:
            key = bucket.get("key")
            if key is None:
                continue
            bucket_counts[str(key)] = {
                "users": int(bucket.get("users") or 0),
                "endpoints": int(bucket.get("endpoints") or 0),
                "scopes": int(bucket.get("scopes") or 0),
            }
        if not bucket_counts:
            return []
        try:
            return await self._trend_store.compute_trends(
                scope=scope,
                group_by=group_by,
                bucket_counts=bucket_counts,
                window_start=window_start,
                window_end=window_end,
                team_id=team_id,
            )
        except _PRIVILEGE_MAP_NONCRITICAL_EXCEPTIONS as exc:
            logger.debug("Privilege trend computation failed: {}", exc)
            return []

    def _build_recommended_actions(
        self, items: Sequence[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        actions: dict[str, dict[str, Any]] = {}
        for item in items:
            if item.get("status") != "blocked":
                continue
            scope_id = item.get("privilege_scope_id")
            reason = item.get("blocked_reason")
            if reason == "feature_flag_disabled":
                action_text = "Request org upgrade"
                reason_text = "Feature flag disabled"
            elif reason == "missing_scope":
                action_text = "Request scope assignment"
                reason_text = "Scope not assigned"
            else:
                continue
            key = f"{scope_id}:{reason}"
            actions[key] = {
                "privilege_scope_id": scope_id,
                "action": action_text,
                "reason": reason_text,
            }
        return list(actions.values())

    @staticmethod
    def _scope_to_endpoint(scope: ScopeEntry) -> str:
        # Translate scope identifier to a placeholder endpoint path.
        path = scope.id.replace(".", "/")
        if not path.startswith("/"):
            path = f"/api/v1/{path}"
        return path

    @staticmethod
    def _row_to_dict(row: Any) -> dict[str, Any] | None:
        if row is None:
            return None
        if isinstance(row, dict):
            return row
        if hasattr(row, "_mapping"):
            return dict(row._mapping)  # type: ignore[attr-defined]
        if hasattr(row, "keys"):
            return {key: row[key] for key in row.keys()}
        return None


@lru_cache
def get_privilege_map_service() -> PrivilegeMapService:
    """Return a singleton privilege map service instance."""
    return PrivilegeMapService()
