from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.AuthNZ.database import DatabasePool
from tldw_Server_API.app.core.AuthNZ.membership_writer import (
    AnchorOwnership,
    TrustedMembershipReason,
    TrustedMembershipWriteContext,
)
from tldw_Server_API.app.core.AuthNZ.repos.federated_identity_repo import FederatedIdentityRepo
from tldw_Server_API.app.core.AuthNZ.repos.federated_managed_grant_repo import FederatedManagedGrantRepo
from tldw_Server_API.app.core.AuthNZ.repos.orgs_teams_repo import (
    DEFAULT_BASE_TEAM_NAME,
    AuthnzOrgsTeamsRepo,
)
from tldw_Server_API.app.core.AuthNZ.repos.users_repo import AuthnzUsersRepo
from tldw_Server_API.app.core.AuthNZ.transaction_policy import (
    get_authnz_transaction_policy,
)

_FEDERATION_MEMBERSHIP_CONTEXT = TrustedMembershipWriteContext(
    trusted_reason=TrustedMembershipReason.BOOTSTRAP,
)


def _claims_hash(claims: dict[str, Any]) -> str:
    serialized = json.dumps(claims, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _coerce_distinct_ints(values: Any) -> list[int]:
    result: set[int] = set()
    if not isinstance(values, (list, tuple, set)):
        return []
    for raw_value in values:
        try:
            result.add(int(raw_value))
        except (TypeError, ValueError):
            continue
    return sorted(result)


def _coerce_distinct_strings(values: Any) -> list[str]:
    result: set[str] = set()
    if not isinstance(values, (list, tuple, set)):
        return []
    for raw_value in values:
        if raw_value is None:
            continue
        normalized = str(raw_value).strip()
        if normalized:
            result.add(normalized)
    return sorted(result)


_FEDERATED_GRANT_MODES = frozenset(
    {
        "jit_grant_only",
        "jit_grant_and_revoke",
        "sync_managed_only",
    }
)
_FEDERATED_REVOKE_MODES = frozenset({"jit_grant_and_revoke", "sync_managed_only"})


def _grant_key(grant_kind: str, target_ref: str) -> tuple[str, str]:
    return str(grant_kind).strip().lower(), str(target_ref).strip()


@dataclass
class FederationProvisioningService:
    """Resolve local users for federated OIDC logins."""

    db_pool: DatabasePool

    async def resolve_user_for_login(
        self,
        *,
        provider: dict[str, Any],
        mapped_claims: dict[str, Any],
    ) -> dict[str, Any] | None:
        subject = str(mapped_claims.get("subject") or "").strip()
        email = str(mapped_claims.get("email") or "").strip().lower()

        users_repo = AuthnzUsersRepo(db_pool=self.db_pool)
        links_repo = FederatedIdentityRepo(db_pool=self.db_pool)
        await links_repo.ensure_tables()

        if subject:
            existing_link = await links_repo.get_by_provider_subject(
                identity_provider_id=int(provider["id"]),
                external_subject=subject,
            )
            if existing_link:
                user = await users_repo.get_user_by_id(int(existing_link["user_id"]))
                if user:
                    return user

        policy = provider.get("provisioning_policy") or {}
        if bool(policy.get("allow_email_account_linking")) and email:
            return await users_repo.get_user_by_email(email)

        return None

    async def dry_run_login_resolution(
        self,
        *,
        provider: dict[str, Any],
        mapped_claims: dict[str, Any],
    ) -> dict[str, Any]:
        subject = str(mapped_claims.get("subject") or "").strip()
        email = str(mapped_claims.get("email") or "").strip().lower()
        warnings: list[str] = []

        users_repo = AuthnzUsersRepo(db_pool=self.db_pool)
        links_repo = FederatedIdentityRepo(db_pool=self.db_pool)
        await links_repo.ensure_tables()

        if not subject:
            warnings.append("OIDC claims did not resolve a subject")
            return {
                "provisioning_action": "deny_missing_subject",
                "matched_user_id": None,
                "identity_link_found": False,
                "email_match_found": False,
                "warnings": warnings,
            }

        policy = provider.get("provisioning_policy") or {}
        allow_email_account_linking = bool(policy.get("allow_email_account_linking"))
        jit_create = bool(policy.get("jit_create"))

        existing_link = await links_repo.get_by_provider_subject(
            identity_provider_id=int(provider.get("id") or 0),
            external_subject=subject,
        )
        if existing_link:
            user = await users_repo.get_user_by_id(int(existing_link["user_id"]))
            if user:
                action = "subject_already_linked"
                if not bool(user.get("is_active", True)):
                    warnings.append("Linked local user is inactive")
                    action = "deny_inactive_user"
                return {
                    "provisioning_action": action,
                    "matched_user_id": int(user["id"]),
                    "identity_link_found": True,
                    "email_match_found": bool(email and str(user.get("email") or "").strip().lower() == email),
                    "warnings": warnings,
                }
            warnings.append("Federated identity link points to a missing local user")

        email_user = await users_repo.get_user_by_email(email) if email else None
        email_match_found = email_user is not None
        matched_user_id = int(email_user["id"]) if email_user else None

        if email_user and allow_email_account_linking:
            action = "link_existing_user"
            if not bool(email_user.get("is_active", True)):
                warnings.append("Matched local user is inactive")
                action = "deny_inactive_user"
            return {
                "provisioning_action": action,
                "matched_user_id": matched_user_id,
                "identity_link_found": False,
                "email_match_found": True,
                "warnings": warnings,
            }

        if jit_create:
            if not email:
                warnings.append("OIDC claims did not resolve an email for JIT provisioning")
                return {
                    "provisioning_action": "deny_missing_email_for_jit_create",
                    "matched_user_id": None,
                    "identity_link_found": False,
                    "email_match_found": False,
                    "warnings": warnings,
                }
            if email_user:
                warnings.append("Federated email already belongs to a local user")
                return {
                    "provisioning_action": "deny_email_collision",
                    "matched_user_id": matched_user_id,
                    "identity_link_found": False,
                    "email_match_found": True,
                    "warnings": warnings,
                }
            return {
                "provisioning_action": "create_new_user",
                "matched_user_id": None,
                "identity_link_found": False,
                "email_match_found": False,
                "warnings": warnings,
            }

        if email_user and not allow_email_account_linking:
            warnings.append("Existing email match found but automatic account linking is disabled")

        return {
            "provisioning_action": "deny_unlinked_user",
            "matched_user_id": matched_user_id,
            "identity_link_found": False,
            "email_match_found": email_match_found,
            "warnings": warnings,
        }

    async def upsert_identity_link(
        self,
        *,
        provider: dict[str, Any],
        user_id: int,
        mapped_claims: dict[str, Any],
        raw_claims: dict[str, Any],
    ) -> dict[str, Any]:
        links_repo = FederatedIdentityRepo(db_pool=self.db_pool)
        await links_repo.ensure_tables()
        return await links_repo.upsert_identity(
            identity_provider_id=int(provider["id"]),
            external_subject=str(mapped_claims.get("subject") or "").strip(),
            user_id=int(user_id),
            external_username=(str(mapped_claims.get("username")).strip() if mapped_claims.get("username") else None),
            external_email=(str(mapped_claims.get("email")).strip().lower() if mapped_claims.get("email") else None),
            last_claims_hash=_claims_hash(raw_claims),
            last_seen_at=datetime.now(timezone.utc),
            status="active",
        )

    async def _resolve_desired_grants(
        self,
        *,
        provider_id: int | None,
        user_id: int | None,
        mapped_claims: dict[str, Any],
    ) -> tuple[set[int], set[int], set[str], list[str]]:
        warnings: list[str] = []
        orgs_repo = AuthnzOrgsTeamsRepo(db_pool=self.db_pool)

        desired_org_ids = set(_coerce_distinct_ints(mapped_claims.get("derived_org_ids")))
        desired_team_ids: set[int] = set()
        for team_id in _coerce_distinct_ints(mapped_claims.get("derived_team_ids")):
            try:
                team = await orgs_repo.get_team(int(team_id))
            except Exception as exc:  # pragma: no cover - defensive runtime hardening
                logger.warning(
                    "Failed to resolve federated team grant provider_id={} user_id={} team_id={}: {}",
                    provider_id,
                    user_id,
                    team_id,
                    exc,
                )
                warnings.append(f"Failed to resolve mapped team {team_id}")
                continue

            if not team or not bool(team.get("is_active", True)):
                logger.warning(
                    "Skipping federated team grant for missing or inactive team provider_id={} user_id={} team_id={}",
                    provider_id,
                    user_id,
                    team_id,
                )
                warnings.append(f"Mapped team {team_id} is missing or inactive")
                continue

            desired_team_ids.add(int(team_id))
            if team.get("org_id") is not None:
                desired_org_ids.add(int(team["org_id"]))

        desired_role_names = set(_coerce_distinct_strings(mapped_claims.get("derived_roles")))
        return desired_org_ids, desired_team_ids, desired_role_names, warnings

    async def preview_mapped_grants(
        self,
        *,
        provider: dict[str, Any],
        user_id: int | None,
        mapped_claims: dict[str, Any],
    ) -> dict[str, Any]:
        policy = provider.get("provisioning_policy") or {}
        mode = str(policy.get("mode") or "").strip().lower()
        if mode not in _FEDERATED_GRANT_MODES:
            return {
                "mode": mode,
                "would_change": False,
                "grant_org_ids": [],
                "grant_team_ids": [],
                "grant_roles": [],
                "revoke_org_ids": [],
                "revoke_team_ids": [],
                "revoke_roles": [],
                "warnings": [],
            }

        provider_id_raw = provider.get("id")
        provider_id = int(provider_id_raw) if provider_id_raw not in (None, "", 0, "0") else None
        desired_org_ids, desired_team_ids, desired_role_names, warnings = await self._resolve_desired_grants(
            provider_id=provider_id,
            user_id=user_id,
            mapped_claims=mapped_claims,
        )

        if user_id is None:
            grant_org_ids = sorted(desired_org_ids)
            grant_team_ids = sorted(desired_team_ids)
            grant_roles = sorted(desired_role_names)
            return {
                "mode": mode,
                "would_change": bool(grant_org_ids or grant_team_ids or grant_roles),
                "grant_org_ids": grant_org_ids,
                "grant_team_ids": grant_team_ids,
                "grant_roles": grant_roles,
                "revoke_org_ids": [],
                "revoke_team_ids": [],
                "revoke_roles": [],
                "warnings": warnings,
            }

        orgs_repo = AuthnzOrgsTeamsRepo(db_pool=self.db_pool)
        users_repo = AuthnzUsersRepo(db_pool=self.db_pool)
        existing_managed: list[dict[str, Any]] = []
        if provider_id is not None:
            managed_repo = FederatedManagedGrantRepo(db_pool=self.db_pool)
            await managed_repo.ensure_tables()
            existing_managed = await managed_repo.list_for_provider_user(
                identity_provider_id=provider_id,
                user_id=int(user_id),
            )
        elif mode in _FEDERATED_REVOKE_MODES:
            warnings.append("Revocation preview requires provider_id context for existing managed grants")

        grant_org_ids: list[int] = []
        for org_id in sorted(desired_org_ids):
            if await orgs_repo.get_org_member(int(org_id), int(user_id)) is None:
                grant_org_ids.append(int(org_id))

        grant_team_ids: list[int] = []
        for team_id in sorted(desired_team_ids):
            if await orgs_repo.get_team_member(int(team_id), int(user_id)) is None:
                grant_team_ids.append(int(team_id))

        grant_roles: list[str] = []
        for role_name in sorted(desired_role_names):
            has_role = await users_repo.has_role_assignment(
                user_id=int(user_id),
                role_name=role_name,
            )
            if not has_role:
                grant_roles.append(role_name)

        revoke_org_ids: list[int] = []
        revoke_team_ids: list[int] = []
        revoke_roles: list[str] = []
        if mode in _FEDERATED_REVOKE_MODES and provider_id is not None:
            desired_keys = {
                *(_grant_key("org", str(org_id)) for org_id in desired_org_ids),
                *(_grant_key("team", str(team_id)) for team_id in desired_team_ids),
                *(_grant_key("role", role_name) for role_name in desired_role_names),
            }
            stale_rows = sorted(
                (
                    row
                    for row in existing_managed
                    if _grant_key(row.get("grant_kind", ""), row.get("target_ref", "")) not in desired_keys
                ),
                key=lambda row: {"team": 0, "role": 1, "org": 2}.get(str(row.get("grant_kind") or ""), 99),
            )
            for row in stale_rows:
                grant_kind = str(row.get("grant_kind") or "").strip().lower()
                target_ref = str(row.get("target_ref") or "").strip()
                if not target_ref:
                    continue

                if grant_kind == "team":
                    current = await orgs_repo.get_team_member(int(target_ref), int(user_id))
                    if current is not None and str(current.get("role") or "member").strip().lower() == "member":
                        revoke_team_ids.append(int(target_ref))
                    continue

                if grant_kind == "role":
                    if await users_repo.has_role_assignment(
                        user_id=int(user_id),
                        role_name=target_ref,
                    ):
                        revoke_roles.append(target_ref)
                    continue

                if grant_kind != "org":
                    continue

                current = await orgs_repo.get_org_member(int(target_ref), int(user_id))
                if current is None or str(current.get("role") or "member").strip().lower() != "member":
                    continue
                remaining_team_memberships = await orgs_repo.list_memberships_for_user(int(user_id))
                if any(
                    membership.get("org_id") is not None
                    and int(membership["org_id"]) == int(target_ref)
                    and membership.get("team_id") is not None
                    and int(membership["team_id"]) not in revoke_team_ids
                    and str(membership.get("team_name") or "").strip() != DEFAULT_BASE_TEAM_NAME
                    for membership in remaining_team_memberships
                ):
                    continue
                revoke_org_ids.append(int(target_ref))

        return {
            "mode": mode,
            "would_change": bool(
                grant_org_ids
                or grant_team_ids
                or grant_roles
                or revoke_org_ids
                or revoke_team_ids
                or revoke_roles
            ),
            "grant_org_ids": grant_org_ids,
            "grant_team_ids": grant_team_ids,
            "grant_roles": grant_roles,
            "revoke_org_ids": revoke_org_ids,
            "revoke_team_ids": revoke_team_ids,
            "revoke_roles": revoke_roles,
            "warnings": warnings,
        }

    async def apply_mapped_grants(
        self,
        *,
        provider: dict[str, Any],
        user_id: int,
        mapped_claims: dict[str, Any],
    ) -> dict[str, Any]:
        policy = provider.get("provisioning_policy") or {}
        mode = str(policy.get("mode") or "").strip().lower()
        if mode not in _FEDERATED_GRANT_MODES:
            return {
                "mode": mode,
                "applied": False,
                "org_ids": [],
                "team_ids": [],
                "roles": [],
                "revoked_org_ids": [],
                "revoked_team_ids": [],
                "revoked_roles": [],
            }

        orgs_repo = AuthnzOrgsTeamsRepo(db_pool=self.db_pool)
        users_repo = AuthnzUsersRepo(db_pool=self.db_pool)
        managed_repo = FederatedManagedGrantRepo(db_pool=self.db_pool)
        await managed_repo.ensure_tables()
        acquire_timeout_seconds = (
            get_authnz_transaction_policy().db_pool_acquire_timeout_seconds
        )

        provider_id = int(provider["id"])
        desired_org_ids, desired_team_ids, desired_role_names, _ = await self._resolve_desired_grants(
            provider_id=provider_id,
            user_id=user_id,
            mapped_claims=mapped_claims,
        )
        existing_managed = await managed_repo.list_for_provider_user(
            identity_provider_id=provider_id,
            user_id=int(user_id),
        )
        existing_managed_keys = {
            _grant_key(row.get("grant_kind", ""), row.get("target_ref", ""))
            for row in existing_managed
        }

        applied_org_ids: list[int] = []
        for org_id in sorted(desired_org_ids):
            grant_key = _grant_key("org", str(org_id))
            try:
                current = await orgs_repo.get_org_member(int(org_id), int(user_id))
                if current is None or grant_key in existing_managed_keys:
                    if current is None:
                        operation_time = datetime.now(timezone.utc)
                        async with self.db_pool.transaction(
                            acquire_timeout_seconds=acquire_timeout_seconds,
                        ) as conn:
                            await orgs_repo.provision_org_membership_on_connection(
                                conn=conn,
                                org_id=int(org_id),
                                user_id=int(user_id),
                                org_role="member",
                                team_id=None,
                                team_role=None,
                                team_failure_is_best_effort=False,
                                context=_FEDERATION_MEMBERSHIP_CONTEXT,
                                anchor_ownership=(
                                    AnchorOwnership.WRITER_OWNS_ANCHOR
                                ),
                                operation_time=operation_time,
                            )
                    await managed_repo.upsert_grant(
                        identity_provider_id=provider_id,
                        user_id=int(user_id),
                        grant_kind="org",
                        target_ref=str(org_id),
                    )
                    applied_org_ids.append(int(org_id))
            except Exception as exc:  # pragma: no cover - defensive runtime hardening
                logger.bind(error_type=type(exc).__name__).warning(
                    "Failed to apply federated org grant provider_id={} user_id={} org_id={}",
                    provider_id,
                    user_id,
                    org_id,
                )

        applied_team_ids: list[int] = []
        for team_id in sorted(desired_team_ids):
            grant_key = _grant_key("team", str(team_id))
            try:
                current = await orgs_repo.get_team_member(int(team_id), int(user_id))
                if current is None or grant_key in existing_managed_keys:
                    if current is None:
                        operation_time = datetime.now(timezone.utc)
                        async with self.db_pool.transaction(
                            acquire_timeout_seconds=acquire_timeout_seconds,
                        ) as conn:
                            await orgs_repo.add_team_member_on_connection(
                                conn=conn,
                                team_id=int(team_id),
                                user_id=int(user_id),
                                role="member",
                                context=_FEDERATION_MEMBERSHIP_CONTEXT,
                                anchor_ownership=(
                                    AnchorOwnership.WRITER_OWNS_ANCHOR
                                ),
                                operation_time=operation_time,
                            )
                    await managed_repo.upsert_grant(
                        identity_provider_id=provider_id,
                        user_id=int(user_id),
                        grant_kind="team",
                        target_ref=str(team_id),
                    )
                    applied_team_ids.append(int(team_id))
            except Exception as exc:  # pragma: no cover - defensive runtime hardening
                logger.bind(error_type=type(exc).__name__).warning(
                    "Failed to apply federated team grant provider_id={} user_id={} team_id={}",
                    provider_id,
                    user_id,
                    team_id,
                )

        applied_roles: list[str] = []
        for role_name in sorted(desired_role_names):
            grant_key = _grant_key("role", role_name)
            try:
                has_role = await users_repo.has_role_assignment(
                    user_id=int(user_id),
                    role_name=role_name,
                )
                if not has_role or grant_key in existing_managed_keys:
                    if not has_role:
                        await users_repo.assign_role_if_missing(
                            user_id=int(user_id),
                            role_name=role_name,
                        )
                        has_role = await users_repo.has_role_assignment(
                            user_id=int(user_id),
                            role_name=role_name,
                        )
                    if not has_role:
                        continue
                    await managed_repo.upsert_grant(
                        identity_provider_id=provider_id,
                        user_id=int(user_id),
                        grant_kind="role",
                        target_ref=role_name,
                    )
                    applied_roles.append(role_name)
            except Exception as exc:  # pragma: no cover - defensive runtime hardening
                logger.bind(error_type=type(exc).__name__).warning(
                    "Failed to apply federated role grant provider_id={} user_id={} role={}",
                    provider_id,
                    user_id,
                    role_name,
                )

        revoked_org_ids: list[int] = []
        revoked_team_ids: list[int] = []
        revoked_roles: list[str] = []
        if mode in _FEDERATED_REVOKE_MODES:
            desired_keys = {
                *(_grant_key("org", str(org_id)) for org_id in desired_org_ids),
                *(_grant_key("team", str(team_id)) for team_id in desired_team_ids),
                *(_grant_key("role", role_name) for role_name in desired_role_names),
            }
            stale_rows = sorted(
                (
                    row
                    for row in existing_managed
                    if _grant_key(row.get("grant_kind", ""), row.get("target_ref", "")) not in desired_keys
                ),
                key=lambda row: {"team": 0, "role": 1, "org": 2}.get(str(row.get("grant_kind") or ""), 99),
            )
            for row in stale_rows:
                grant_kind = str(row.get("grant_kind") or "").strip().lower()
                target_ref = str(row.get("target_ref") or "").strip()
                if not target_ref:
                    continue

                try:
                    if grant_kind == "team":
                        team_id = int(target_ref)
                        current = await orgs_repo.get_team_member(team_id, int(user_id))
                        if current is None:
                            await managed_repo.delete_grant(
                                identity_provider_id=provider_id,
                                user_id=int(user_id),
                                grant_kind="team",
                                target_ref=target_ref,
                            )
                            continue
                        if str(current.get("role") or "member").strip().lower() != "member":
                            await managed_repo.delete_grant(
                                identity_provider_id=provider_id,
                                user_id=int(user_id),
                                grant_kind="team",
                                target_ref=target_ref,
                            )
                            continue
                        operation_time = datetime.now(timezone.utc)
                        async with self.db_pool.transaction(
                            acquire_timeout_seconds=acquire_timeout_seconds,
                        ) as conn:
                            removal = (
                                await orgs_repo.remove_team_member_on_connection(
                                    conn=conn,
                                    team_id=team_id,
                                    user_id=int(user_id),
                                    context=_FEDERATION_MEMBERSHIP_CONTEXT,
                                    anchor_ownership=(
                                        AnchorOwnership.WRITER_OWNS_ANCHOR
                                    ),
                                    operation_time=operation_time,
                                )
                            )
                        await managed_repo.delete_grant(
                            identity_provider_id=provider_id,
                            user_id=int(user_id),
                            grant_kind="team",
                            target_ref=target_ref,
                        )
                        if removal.get("removed"):
                            revoked_team_ids.append(team_id)
                        continue

                    if grant_kind == "role":
                        if await users_repo.has_role_assignment(
                            user_id=int(user_id),
                            role_name=target_ref,
                        ):
                            removed = await users_repo.remove_role_if_present(
                                user_id=int(user_id),
                                role_name=target_ref,
                            )
                            if removed:
                                revoked_roles.append(target_ref)
                        await managed_repo.delete_grant(
                            identity_provider_id=provider_id,
                            user_id=int(user_id),
                            grant_kind="role",
                            target_ref=target_ref,
                        )
                        continue

                    if grant_kind == "org":
                        org_id = int(target_ref)
                        current = await orgs_repo.get_org_member(org_id, int(user_id))
                        if current is None:
                            await managed_repo.delete_grant(
                                identity_provider_id=provider_id,
                                user_id=int(user_id),
                                grant_kind="org",
                                target_ref=target_ref,
                            )
                            continue
                        if str(current.get("role") or "member").strip().lower() != "member":
                            await managed_repo.delete_grant(
                                identity_provider_id=provider_id,
                                user_id=int(user_id),
                                grant_kind="org",
                                target_ref=target_ref,
                            )
                            continue
                        remaining_team_memberships = await orgs_repo.list_memberships_for_user(int(user_id))
                        if any(
                            membership.get("org_id") is not None
                            and int(membership["org_id"]) == org_id
                            and str(membership.get("team_name") or "").strip() != DEFAULT_BASE_TEAM_NAME
                            for membership in remaining_team_memberships
                        ):
                            await managed_repo.delete_grant(
                                identity_provider_id=provider_id,
                                user_id=int(user_id),
                                grant_kind="org",
                                target_ref=target_ref,
                            )
                            continue
                        operation_time = datetime.now(timezone.utc)
                        async with self.db_pool.transaction(
                            acquire_timeout_seconds=acquire_timeout_seconds,
                        ) as conn:
                            removal = (
                                await orgs_repo.remove_org_member_on_connection(
                                    conn=conn,
                                    org_id=org_id,
                                    user_id=int(user_id),
                                    context=_FEDERATION_MEMBERSHIP_CONTEXT,
                                    anchor_ownership=(
                                        AnchorOwnership.WRITER_OWNS_ANCHOR
                                    ),
                                    operation_time=operation_time,
                                )
                            )
                        await managed_repo.delete_grant(
                            identity_provider_id=provider_id,
                            user_id=int(user_id),
                            grant_kind="org",
                            target_ref=target_ref,
                        )
                        if removal.get("removed"):
                            revoked_org_ids.append(org_id)
                except Exception as exc:  # pragma: no cover - defensive runtime hardening
                    logger.bind(error_type=type(exc).__name__).warning(
                        "Failed to reconcile stale federated grant provider_id={} user_id={} grant_kind={} target_ref={}",
                        provider_id,
                        user_id,
                        grant_kind,
                        target_ref,
                    )

        return {
            "mode": mode,
            "applied": bool(applied_org_ids or applied_team_ids or applied_roles or revoked_org_ids or revoked_team_ids or revoked_roles),
            "org_ids": applied_org_ids,
            "team_ids": applied_team_ids,
            "roles": applied_roles,
            "revoked_org_ids": revoked_org_ids,
            "revoked_team_ids": revoked_team_ids,
            "revoked_roles": revoked_roles,
        }
