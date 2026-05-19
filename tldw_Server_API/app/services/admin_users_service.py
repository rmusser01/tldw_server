from __future__ import annotations

import json
import secrets
import string
from collections.abc import Awaitable
from datetime import datetime, timezone
from typing import Any, Callable

from fastapi import HTTPException, status
from loguru import logger

from tldw_Server_API.app.api.v1.schemas.admin_schemas import (
    AdminMfaRequirementRequest,
    AdminPasswordResetRequest,
    AdminUserCreateRequest,
    UserUpdateRequest,
    unwrap_optional_secret,
)
from tldw_Server_API.app.core.AuthNZ.exceptions import (
    DuplicateUserError,
    RegistrationDisabledError,
    RegistrationError,
    UserNotFoundError,
    WeakPasswordError,
)
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.AuthNZ.repos.users_repo import AuthnzUsersRepo
from tldw_Server_API.app.core.AuthNZ.settings import get_profile
from tldw_Server_API.app.core.AuthNZ.password_service import hash_password
from tldw_Server_API.app.services import admin_scope_service
from tldw_Server_API.app.services.admin_audit_service import (
    emit_admin_account_audit_event as _emit_admin_account_audit_event,
)
from tldw_Server_API.app.services.admin_guardrails_service import verify_privileged_action
from tldw_Server_API.app.services.admin_data_ops_service import (
    build_users_csv as svc_build_users_csv,
)
from tldw_Server_API.app.services.admin_data_ops_service import (
    build_users_json as svc_build_users_json,
)
from tldw_Server_API.app.core.Audit.unified_audit_service import (
    AuditEventCategory,
    AuditEventType,
)


def _generate_temporary_password(length: int = 20) -> str:
    alphabet = string.ascii_letters + string.digits + "!@#$%^&*()-_=+"
    while True:
        password = ''.join(secrets.choice(alphabet) for _ in range(length))
        has_upper = any(char.isupper() for char in password)
        has_lower = any(char.islower() for char in password)
        has_digit = any(char.isdigit() for char in password)
        has_symbol = any(char in "!@#$%^&*()-_=+" for char in password)
        if has_upper and has_lower and has_digit and has_symbol:
            return password


def _parse_user_metadata(raw_metadata: Any) -> dict[str, Any]:
    if raw_metadata is None:
        return {}
    if isinstance(raw_metadata, dict):
        return dict(raw_metadata)
    if isinstance(raw_metadata, str):
        try:
            parsed = json.loads(raw_metadata)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            return {}
    return {}


async def create_user(
    payload: AdminUserCreateRequest,
    principal: AuthPrincipal,
    registration_service,
):
    profile = get_profile()
    if isinstance(profile, str) and profile.strip().lower() in {"local-single-user", "single_user"}:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User creation is not allowed in local-single-user profile",
        )

    created_by = int(principal.user_id) if principal.user_id is not None else None
    try:
        user_info = await registration_service.register_user(
            username=payload.username,
            email=payload.email,
            password=payload.password,
            created_by=created_by,
            role_override=payload.role,
            is_active_override=payload.is_active,
            is_verified_override=payload.is_verified,
            storage_quota_override=payload.storage_quota_mb,
        )
        repo = await AuthnzUsersRepo.from_pool()
        user = await repo.get_user_by_id(int(user_info["user_id"]))
        if not user:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to load created user",
            )
        logger.info("Admin created user {} (id={})", payload.username, user_info["user_id"])
        return user
    except DuplicateUserError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="User already exists.") from exc
    except WeakPasswordError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password does not meet requirements.",
        ) from exc
    except RegistrationDisabledError as exc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Registration is currently disabled.",
        ) from exc
    except RegistrationError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Registration failed.") from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to create user")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create user",
        ) from exc


async def list_users(
    principal: AuthPrincipal,
    *,
    page: int,
    limit: int,
    role: str | None,
    admin_capable: bool,
    is_active: bool | None,
    mfa_enabled: bool | None = None,
    search: str | None,
    org_id: int | None,
) -> tuple[list[dict[str, Any]], int]:
    try:
        offset = (page - 1) * limit
        repo = await AuthnzUsersRepo.from_pool()
        org_ids = await admin_scope_service.get_admin_org_ids(principal)
        if org_id is not None:
            org_ids = [org_id] if org_ids is None else [org_id] if org_id in org_ids else []
        users, total = await repo.list_users(
            offset=offset,
            limit=limit,
            role=role,
            admin_capable=admin_capable,
            is_active=is_active,
            mfa_enabled=mfa_enabled,
            search=search,
            org_ids=org_ids,
        )
        return users, total
    except Exception as e:
        logger.error("Failed to list users")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve users",
        ) from e


async def export_users(
    principal: AuthPrincipal,
    *,
    role: str | None,
    is_active: bool | None,
    search: str | None,
    org_id: int | None,
    limit: int,
    offset: int,
    format: str,
) -> tuple[str, str, str]:
    try:
        org_ids = await admin_scope_service.get_admin_org_ids(principal)
        if org_id is not None:
            org_ids = [org_id] if org_ids is None else [org_id] if org_id in org_ids else []
        repo = await AuthnzUsersRepo.from_pool()
        users, total = await repo.list_users(
            offset=offset,
            limit=limit,
            role=role,
            is_active=is_active,
            search=search,
            org_ids=org_ids,
        )
        if format == "json":
            content = svc_build_users_json(users, total=total, limit=limit, offset=offset)
            media_type = "application/json"
            default_name = "users.json"
        else:
            content = svc_build_users_csv(users)
            media_type = "text/csv"
            default_name = "users.csv"
        return content, media_type, default_name
    except Exception as exc:
        logger.error("Failed to export users")
        raise HTTPException(status_code=500, detail="Failed to export users") from exc


async def get_user_details(
    principal: AuthPrincipal,
    user_id: int,
) -> dict[str, Any]:
    try:
        await admin_scope_service.enforce_admin_user_scope(
            principal,
            user_id,
            require_hierarchy=False,
        )

        repo = await AuthnzUsersRepo.from_pool()
        user = await repo.get_user_by_id(user_id)
        if not user:
            raise UserNotFoundError(f"User {user_id}")

        user_dict: dict[str, Any] = dict(user)
        user_dict.pop("password_hash", None)

        return user_dict
    except UserNotFoundError as err:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User {user_id} not found",
        ) from err
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get user")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve user details",
        ) from e


async def update_user(
    principal: AuthPrincipal,
    user_id: int,
    request: UserUpdateRequest,
    db,
    password_service,
    *,
    is_pg_fn: Callable[[], Awaitable[bool]],
) -> dict[str, str]:
    try:
        await admin_scope_service.enforce_admin_user_scope(
            principal,
            user_id,
            require_hierarchy=True,
        )
        reason: str | None = None
        if request.role is not None or request.is_active is not None:
            reason = await verify_privileged_action(
                principal,
                db,
                password_service,
                reason=request.reason,
                admin_password=unwrap_optional_secret(request.admin_password),
                admin_reauth_token=unwrap_optional_secret(request.admin_reauth_token),
            )

        is_pg = await is_pg_fn()
        updates = []
        params = []
        param_count = 0

        if request.email is not None:
            param_count += 1
            updates.append(f"email = ${param_count}" if is_pg else "email = ?")
            params.append(request.email)

        if request.role is not None:
            param_count += 1
            updates.append(f"role = ${param_count}" if is_pg else "role = ?")
            params.append(request.role)

        if request.is_active is not None:
            param_count += 1
            updates.append(f"is_active = ${param_count}" if is_pg else "is_active = ?")
            params.append(request.is_active)

        if request.is_verified is not None:
            param_count += 1
            updates.append(f"is_verified = ${param_count}" if is_pg else "is_verified = ?")
            params.append(request.is_verified)

        if request.is_locked is not None:
            param_count += 1
            updates.append(f"is_locked = ${param_count}" if is_pg else "is_locked = ?")
            params.append(request.is_locked)

            if not request.is_locked:
                param_count += 1
                updates.append(f"failed_login_attempts = ${param_count}" if is_pg else "failed_login_attempts = ?")
                params.append(0)
                updates.append("locked_until = NULL")

        if request.storage_quota_mb is not None:
            param_count += 1
            updates.append(f"storage_quota_mb = ${param_count}" if is_pg else "storage_quota_mb = ?")
            params.append(request.storage_quota_mb)

        if not updates:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No fields to update",
            )

        param_count += 1
        if is_pg:
            updates.append("updated_at = CURRENT_TIMESTAMP")
            params.append(user_id)
            set_clause = ", ".join(updates)
            update_user_sql_template = "UPDATE users SET {set_clause} WHERE id = ${param_count}"
            query = update_user_sql_template.format_map(locals())  # nosec B608
        else:
            updates.append("updated_at = datetime('now')")
            params.append(user_id)
            set_clause = ", ".join(updates)
            update_user_sql_template = "UPDATE users SET {set_clause} WHERE id = ?"
            query = update_user_sql_template.format_map(locals())  # nosec B608

        if is_pg:
            row = await db.fetchrow(query + " RETURNING id", *params)
            if not row:
                raise UserNotFoundError(f"User {user_id}")
        else:
            cursor = await db.execute(query, params)
            affected = int(getattr(cursor, "rowcount", 0) or 0)
            if affected == 0:
                cursor = await db.execute("SELECT 1 FROM users WHERE id = ?", (user_id,))
                if not await cursor.fetchone():
                    raise UserNotFoundError(f"User {user_id}")
            await db.commit()

        if reason is not None:
            metadata: dict[str, Any] = {"reason": reason}
            if request.role is not None:
                metadata["role"] = request.role
            if request.is_active is not None:
                metadata["is_active"] = request.is_active
            await _emit_admin_account_audit_event(
                actor_id=principal.user_id,
                target_user_id=user_id,
                event_type=AuditEventType.USER_UPDATED,
                category=AuditEventCategory.AUTHORIZATION,
                resource_type="user_account",
                resource_id=str(user_id),
                action="admin.user.update",
                metadata=metadata,
            )

        logger.info(f"Admin updated user {user_id}")

        return {"message": f"User {user_id} updated successfully"}

    except UserNotFoundError as err:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User {user_id} not found",
        ) from err
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to update user")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update user",
        ) from e


async def reset_user_password(
    principal: AuthPrincipal,
    user_id: int,
    request: AdminPasswordResetRequest,
    db,
    password_service,
    *,
    is_pg_fn: Callable[[], Awaitable[bool]],
) -> dict[str, Any]:
    try:
        await admin_scope_service.enforce_admin_user_scope(
            principal,
            user_id,
            require_hierarchy=True,
        )
        reason = await verify_privileged_action(
            principal,
            db,
            password_service,
            reason=request.reason,
            admin_password=unwrap_optional_secret(request.admin_password),
            admin_reauth_token=unwrap_optional_secret(request.admin_reauth_token),
        )

        temporary_password = request.temporary_password
        password_hash = hash_password(temporary_password)
        force_password_change = bool(request.force_password_change)

        is_pg = await is_pg_fn()
        if is_pg:
            row = await db.fetchrow("SELECT metadata FROM users WHERE id = $1", user_id)
            if not row:
                raise UserNotFoundError(f"User {user_id}")
            existing_metadata = _parse_user_metadata(row.get("metadata"))
        else:
            cursor = await db.execute("SELECT metadata FROM users WHERE id = ?", (user_id,))
            row = await cursor.fetchone()
            if not row:
                raise UserNotFoundError(f"User {user_id}")
            if isinstance(row, (tuple, list)):
                metadata_raw = row[0] if row else None
            elif hasattr(row, "keys"):
                metadata_raw = dict(row).get("metadata")
            else:
                metadata_raw = None
            existing_metadata = _parse_user_metadata(metadata_raw)

        existing_metadata["force_password_change"] = force_password_change
        existing_metadata["password_reset_at"] = datetime.now(timezone.utc).isoformat()
        metadata_json = json.dumps(existing_metadata)

        if is_pg:
            await db.execute(
                """
                UPDATE users
                SET password_hash = $1,
                    metadata = $2::jsonb,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = $3
                """,
                password_hash,
                metadata_json,
                user_id,
            )
        else:
            await db.execute(
                """
                UPDATE users
                SET password_hash = ?,
                    metadata = ?,
                    updated_at = datetime('now')
                WHERE id = ?
                """,
                (password_hash, metadata_json, user_id),
            )
            await db.commit()

        await _emit_admin_account_audit_event(
            actor_id=principal.user_id,
            target_user_id=user_id,
            event_type=AuditEventType.USER_PASSWORD_RESET,
            category=AuditEventCategory.AUTHENTICATION,
            resource_type="user_account",
            resource_id=str(user_id),
            action="admin.user.password_reset",
            metadata={
                "reason": reason,
                "force_password_change": force_password_change,
                "credential_provided_by_admin": True,
            },
        )

        logger.info("Admin reset password for user {}", user_id)

        return {
            "user_id": user_id,
            "temporary_password": temporary_password,
            "force_password_change": force_password_change,
            "message": "Password reset successfully",
        }

    except UserNotFoundError as err:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User {user_id} not found",
        ) from err
    except WeakPasswordError as err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(err),
        ) from err
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to reset password")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to reset password",
        ) from exc


async def set_user_mfa_requirement(
    principal: AuthPrincipal,
    user_id: int,
    request: AdminMfaRequirementRequest,
    db,
    password_service,
    *,
    is_pg_fn: Callable[[], Awaitable[bool]],
) -> dict[str, Any]:
    try:
        await admin_scope_service.enforce_admin_user_scope(
            principal,
            user_id,
            require_hierarchy=True,
        )
        reason = await verify_privileged_action(
            principal,
            db,
            password_service,
            reason=request.reason,
            admin_password=unwrap_optional_secret(request.admin_password),
            admin_reauth_token=unwrap_optional_secret(request.admin_reauth_token),
        )

        require_mfa = bool(request.require_mfa)
        is_pg = await is_pg_fn()
        if is_pg:
            row = await db.fetchrow("SELECT metadata FROM users WHERE id = $1", user_id)
            if not row:
                raise UserNotFoundError(f"User {user_id}")
            existing_metadata = _parse_user_metadata(row.get("metadata"))
        else:
            cursor = await db.execute("SELECT metadata FROM users WHERE id = ?", (user_id,))
            row = await cursor.fetchone()
            if not row:
                raise UserNotFoundError(f"User {user_id}")
            if isinstance(row, (tuple, list)):
                metadata_raw = row[0] if row else None
            elif hasattr(row, "keys"):
                metadata_raw = dict(row).get("metadata")
            else:
                metadata_raw = None
            existing_metadata = _parse_user_metadata(metadata_raw)

        existing_metadata["require_mfa"] = require_mfa
        existing_metadata["mfa_requirement_updated_at"] = datetime.now(timezone.utc).isoformat()
        metadata_json = json.dumps(existing_metadata)

        if is_pg:
            await db.execute(
                """
                UPDATE users
                SET metadata = $1::jsonb,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = $2
                """,
                metadata_json,
                user_id,
            )
        else:
            await db.execute(
                """
                UPDATE users
                SET metadata = ?,
                    updated_at = datetime('now')
                WHERE id = ?
                """,
                (metadata_json, user_id),
            )
            await db.commit()

        await _emit_admin_account_audit_event(
            actor_id=principal.user_id,
            target_user_id=user_id,
            event_type=AuditEventType.CONFIG_CHANGED,
            category=AuditEventCategory.SECURITY,
            resource_type="user_mfa",
            resource_id=str(user_id),
            action="admin.user.mfa_requirement.update",
            metadata={
                "reason": reason,
                "require_mfa": require_mfa,
            },
        )

        logger.info("Admin updated MFA requirement for user {} to {}", user_id, require_mfa)

        return {
            "user_id": user_id,
            "require_mfa": require_mfa,
            "message": "MFA requirement updated successfully",
        }

    except UserNotFoundError as err:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User {user_id} not found",
        ) from err
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to update MFA requirement")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update MFA requirement",
        ) from exc


async def delete_user(
    principal: AuthPrincipal,
    user_id: int,
    request,
    db,
    password_service,
    *,
    is_pg_fn: Callable[[], Awaitable[bool]],
) -> dict[str, str]:
    try:
        if principal.user_id is not None and str(user_id) == str(principal.user_id):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot delete your own account",
            )
        await admin_scope_service.enforce_admin_user_scope(
            principal,
            user_id,
            require_hierarchy=True,
        )
        reason = await verify_privileged_action(
            principal,
            db,
            password_service,
            reason=getattr(request, "reason", None),
            admin_password=unwrap_optional_secret(getattr(request, "admin_password", None)),
            admin_reauth_token=unwrap_optional_secret(getattr(request, "admin_reauth_token", None)),
        )

        is_pg = await is_pg_fn()
        if is_pg:
            await db.execute(
                "UPDATE users SET is_active = FALSE, updated_at = CURRENT_TIMESTAMP WHERE id = $1",
                user_id,
            )
        else:
            await db.execute(
                "UPDATE users SET is_active = 0, updated_at = datetime('now') WHERE id = ?",
                (user_id,),
            )
            await db.commit()

        await _emit_admin_account_audit_event(
            actor_id=principal.user_id,
            target_user_id=user_id,
            event_type=AuditEventType.USER_DEACTIVATED,
            category=AuditEventCategory.AUTHORIZATION,
            resource_type="user_account",
            resource_id=str(user_id),
            action="admin.user.deactivate",
            metadata={"reason": reason},
        )

        logger.info(f"Admin soft-deleted user {user_id}")

        return {"message": f"User {user_id} has been deactivated"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to delete user")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete user",
        ) from e
