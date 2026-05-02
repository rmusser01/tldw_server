from __future__ import annotations

import json
import secrets
import string
from datetime import datetime, timedelta
from typing import Any

from fastapi import HTTPException, status
from loguru import logger

from tldw_Server_API.app.api.v1.schemas.admin_schemas import (
    RegistrationCodeListResponse,
    RegistrationCodeRequest,
    RegistrationCodeResponse,
    RegistrationSettingsResponse,
    RegistrationSettingsUpdateRequest,
)
from tldw_Server_API.app.core.AuthNZ.database import DatabasePool
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.AuthNZ.settings import get_profile, get_settings, reset_settings
from tldw_Server_API.app.core.Setup import setup_manager
from tldw_Server_API.app.services.registration_service import reset_registration_service


def _is_db_pool_object(db: Any) -> bool:
    return isinstance(db, DatabasePool)


def _is_postgres_connection(db: Any) -> bool:
    """Resolve backend mode from connection/adapter shape without global probes."""
    if _is_db_pool_object(db):
        return getattr(db, "pool", None) is not None

    sqlite_hint = getattr(db, "_is_sqlite", None)
    if isinstance(sqlite_hint, bool):
        return not sqlite_hint

    if getattr(db, "_c", None) is not None:
        return False

    module_name = getattr(type(db), "__module__", "")
    if isinstance(module_name, str) and module_name.startswith("asyncpg"):
        return True

    return callable(getattr(db, "fetchrow", None))


async def get_registration_settings() -> RegistrationSettingsResponse:
    """Return current registration settings."""
    settings = get_settings()
    profile = get_profile()
    self_allowed = bool(settings.ENABLE_REGISTRATION)
    if isinstance(profile, str) and profile.strip().lower() in {"local-single-user", "single_user"}:
        self_allowed = False

    return RegistrationSettingsResponse(
        enable_registration=bool(settings.ENABLE_REGISTRATION),
        require_registration_code=bool(settings.REQUIRE_REGISTRATION_CODE),
        auth_mode=str(settings.AUTH_MODE) if getattr(settings, "AUTH_MODE", None) is not None else None,
        profile=str(profile) if profile is not None else None,
        self_registration_allowed=self_allowed,
    )


async def update_registration_settings(
    payload: RegistrationSettingsUpdateRequest,
) -> RegistrationSettingsResponse:
    """Update registration settings and refresh cached config."""
    updates: dict[str, Any] = {}
    if payload.enable_registration is not None:
        updates["enable_registration"] = payload.enable_registration
    if payload.require_registration_code is not None:
        updates["require_registration_code"] = payload.require_registration_code

    if not updates:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No registration settings provided")

    try:
        setup_manager.update_config({"AuthNZ": updates})
        reset_settings()
        try:
            await reset_registration_service()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Registration service reset failed")
    except ValueError as exc:
        logger.warning("Invalid registration settings update")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid registration settings",
        ) from exc
    except Exception as exc:
        logger.error("Failed to update registration settings")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update registration settings",
        ) from exc

    return await get_registration_settings()


async def create_registration_code(
    request: RegistrationCodeRequest,
    principal: AuthPrincipal,
    db,
):
    """Create a new registration code."""
    try:
        # Generate secure code
        code = "".join(secrets.choice(string.ascii_uppercase + string.digits) for _ in range(24))

        # Calculate expiration
        expires_at = datetime.utcnow() + timedelta(days=request.expiry_days)

        is_pg = _is_postgres_connection(db)
        creator_id = int(principal.user_id) if principal.user_id is not None else None
        org_id = request.org_id
        org_role = request.org_role or ("member" if org_id is not None else None)
        team_id = request.team_id
        settings = get_settings()

        if (org_id is not None or team_id is not None or request.org_role is not None) and not (
            settings.ENABLE_ORG_SCOPED_REGISTRATION_CODES
        ):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Org-scoped registration codes are disabled",
            )

        allowed_email_domain = request.allowed_email_domain
        if allowed_email_domain is not None:
            normalized = allowed_email_domain.strip().lower()
            if normalized.startswith("@"):
                normalized = normalized[1:]
            if "@" in normalized or not normalized:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="allowed_email_domain must be a domain like example.com",
                )
            allowed_email_domain = normalized

        if team_id is not None and org_id is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="org_id is required when team_id is provided",
            )

        org_name = None
        if org_id is not None:
            if is_pg:
                org_row = await db.fetchrow("SELECT id, name FROM organizations WHERE id = $1", org_id)
            else:
                cursor = await db.execute("SELECT id, name FROM organizations WHERE id = ?", (org_id,))
                org_row = await cursor.fetchone()
            if not org_row:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Organization not found",
                )
            org_name = org_row["name"] if isinstance(org_row, dict) else org_row[1]

        if team_id is not None:
            if is_pg:
                team_row = await db.fetchrow(
                    "SELECT id, org_id FROM teams WHERE id = $1",
                    team_id,
                )
                team_org_id = team_row["org_id"] if team_row else None
            else:
                cursor = await db.execute(
                    "SELECT id, org_id FROM teams WHERE id = ?",
                    (team_id,),
                )
                team_row = await cursor.fetchone()
                team_org_id = team_row[1] if team_row else None
            if not team_row:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Team not found",
                )
            if org_id is not None and team_org_id != org_id:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Team does not belong to the specified organization",
                )

        if is_pg:
            # PostgreSQL
            result = await db.fetchrow(
                """
                INSERT INTO registration_codes
                (code, max_uses, expires_at, created_by, role_to_grant, allowed_email_domain, metadata, org_id, org_role, team_id)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                RETURNING id, code, max_uses, times_used, expires_at, created_at, created_by, role_to_grant,
                          org_id, org_role, team_id, metadata, is_active, allowed_email_domain
            """,
                code,
                request.max_uses,
                expires_at,
                creator_id,
                request.role_to_grant,
                allowed_email_domain,
                json.dumps(request.metadata or {}),
                org_id,
                org_role,
                team_id,
            )
        else:
            # SQLite
            cursor = await db.execute(
                """
                INSERT INTO registration_codes
                (code, max_uses, expires_at, created_by, role_to_grant, allowed_email_domain, metadata, org_id, org_role, team_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    code,
                    request.max_uses,
                    expires_at.isoformat(),
                    creator_id,
                    request.role_to_grant,
                    allowed_email_domain,
                    json.dumps(request.metadata or {}),
                    org_id,
                    org_role,
                    team_id,
                ),
            )

            code_id = cursor.lastrowid
            await db.commit()

            # Fetch the created code
            cursor = await db.execute(
                """
                SELECT id, code, max_uses, times_used, expires_at, created_at, created_by, role_to_grant,
                       org_id, org_role, team_id, metadata, is_active, allowed_email_domain
                FROM registration_codes
                WHERE id = ?
                """,
                (code_id,),
            )
            result = await cursor.fetchone()

        logger.info(f"Admin created registration code: {code[:8]}...")

        if isinstance(result, tuple):
            created_at = result[5]
            metadata_value = result[11]
            created_by = result[6]
            is_active = result[12]
            allowed_email_domain = result[13]
            code_id = result[0]
        else:
            created_at = result["created_at"]
            metadata_value = result["metadata"]
            created_by = result.get("created_by")
            is_active = result.get("is_active")
            allowed_email_domain = result.get("allowed_email_domain")
            code_id = result["id"]
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)

        if isinstance(metadata_value, str):
            try:
                metadata_value = json.loads(metadata_value)
            except json.JSONDecodeError:
                metadata_value = None

        response = RegistrationCodeResponse(
            id=code_id,
            code=code,
            max_uses=request.max_uses,
            times_used=0,
            expires_at=expires_at,
            created_at=created_at,
            created_by=created_by,
            role_to_grant=request.role_to_grant,
            allowed_email_domain=allowed_email_domain,
            org_id=result[8] if isinstance(result, tuple) else result["org_id"],
            org_role=result[9] if isinstance(result, tuple) else result["org_role"],
            team_id=result[10] if isinstance(result, tuple) else result["team_id"],
            org_name=org_name,
            metadata=metadata_value if metadata_value is not None else request.metadata,
            is_active=is_active,
        )

        audit_info = {
            "event_type": "data.write",
            "category": "data_modification",
            "resource_type": "registration_code",
            "resource_id": str(code_id),
            "action": "registration_code.create",
            "metadata": {
                "code_prefix": code[:8],
                "max_uses": request.max_uses,
                "expires_at": expires_at.isoformat(),
                "role_to_grant": request.role_to_grant,
                "allowed_email_domain": allowed_email_domain,
                "org_id": org_id,
                "org_role": org_role,
                "team_id": team_id,
            },
        }

        return response, audit_info

    except Exception as exc:
        logger.error("Failed to create registration code")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create registration code",
        ) from exc


async def list_registration_codes(
    include_expired: bool,
    db,
) -> RegistrationCodeListResponse:
    """List all registration codes."""
    try:
        is_pg = _is_postgres_connection(db)
        if is_pg:
            # PostgreSQL
            if include_expired:
                query = """
                    SELECT rc.id, rc.code, rc.max_uses, rc.times_used, rc.expires_at,
                           rc.created_at, rc.created_by, rc.role_to_grant,
                           rc.org_id, rc.org_role, rc.team_id, rc.metadata, rc.is_active,
                           rc.allowed_email_domain, o.name AS org_name
                    FROM registration_codes rc
                    LEFT JOIN organizations o ON rc.org_id = o.id
                    ORDER BY rc.created_at DESC
                """
            else:
                query = """
                    SELECT rc.id, rc.code, rc.max_uses, rc.times_used, rc.expires_at,
                           rc.created_at, rc.created_by, rc.role_to_grant,
                           rc.org_id, rc.org_role, rc.team_id, rc.metadata, rc.is_active,
                           rc.allowed_email_domain, o.name AS org_name
                    FROM registration_codes rc
                    LEFT JOIN organizations o ON rc.org_id = o.id
                    WHERE rc.is_active = TRUE
                      AND rc.times_used < rc.max_uses
                      AND rc.expires_at > CURRENT_TIMESTAMP
                    ORDER BY rc.created_at DESC
                """
            rows = await db.fetch(query)
        else:
            # SQLite
            if include_expired:
                query = """
                    SELECT rc.id, rc.code, rc.max_uses, rc.times_used, rc.expires_at,
                           rc.created_at, rc.created_by, rc.role_to_grant,
                           rc.org_id, rc.org_role, rc.team_id, rc.metadata, rc.is_active,
                           rc.allowed_email_domain, o.name AS org_name
                    FROM registration_codes rc
                    LEFT JOIN organizations o ON rc.org_id = o.id
                    ORDER BY rc.created_at DESC
                """
            else:
                query = """
                    SELECT rc.id, rc.code, rc.max_uses, rc.times_used, rc.expires_at,
                           rc.created_at, rc.created_by, rc.role_to_grant,
                           rc.org_id, rc.org_role, rc.team_id, rc.metadata, rc.is_active,
                           rc.allowed_email_domain, o.name AS org_name
                    FROM registration_codes rc
                    LEFT JOIN organizations o ON rc.org_id = o.id
                    WHERE rc.is_active = 1
                      AND rc.times_used < rc.max_uses
                      AND datetime(rc.expires_at) > datetime('now')
                    ORDER BY rc.created_at DESC
                """
            cursor = await db.execute(query)
            rows = await cursor.fetchall()

        codes = []

        def _get_value(row, key: str, index: int):
            try:
                return row[key]
            except Exception:
                return row[index]

        for row in rows:
            metadata_value = _get_value(row, "metadata", 11)
            if isinstance(metadata_value, str):
                try:
                    metadata_value = json.loads(metadata_value)
                except json.JSONDecodeError:
                    metadata_value = None

            expires_at_value = _get_value(row, "expires_at", 4)
            if isinstance(expires_at_value, str):
                expires_at_dt = datetime.fromisoformat(expires_at_value)
            else:
                expires_at_dt = expires_at_value

            times_used = _get_value(row, "times_used", 3)
            max_uses = _get_value(row, "max_uses", 2)
            is_active = _get_value(row, "is_active", 12)
            is_active_value = True if is_active is None else bool(is_active)

            code_dict = {
                "id": _get_value(row, "id", 0),
                "code": _get_value(row, "code", 1),
                "max_uses": max_uses,
                "times_used": times_used,
                "expires_at": expires_at_value,
                "created_at": _get_value(row, "created_at", 5),
                "created_by": _get_value(row, "created_by", 6),
                "role_to_grant": _get_value(row, "role_to_grant", 7),
                "allowed_email_domain": _get_value(row, "allowed_email_domain", 13),
                "org_id": _get_value(row, "org_id", 8),
                "org_role": _get_value(row, "org_role", 9),
                "team_id": _get_value(row, "team_id", 10),
                "org_name": _get_value(row, "org_name", 14),
                "metadata": metadata_value,
                "is_active": is_active,
                "is_valid": is_active_value and times_used < max_uses and expires_at_dt > datetime.utcnow(),
            }
            codes.append(code_dict)

        return RegistrationCodeListResponse(codes=codes)

    except Exception as exc:
        logger.error("Failed to list registration codes")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve registration codes",
        ) from exc


async def delete_registration_code(
    code_id: int,
    db,
):
    """Delete (revoke) a registration code."""
    try:
        is_pg = _is_postgres_connection(db)
        if is_pg:
            # PostgreSQL
            await db.execute(
                "UPDATE registration_codes SET is_active = FALSE WHERE id = $1",
                code_id,
            )
        else:
            # SQLite
            await db.execute(
                "UPDATE registration_codes SET is_active = 0 WHERE id = ?",
                (code_id,),
            )
            await db.commit()

        logger.info(f"Admin revoked registration code {code_id}")

        audit_info = {
            "event_type": "data.update",
            "category": "data_modification",
            "resource_type": "registration_code",
            "resource_id": str(code_id),
            "action": "registration_code.revoke",
            "metadata": {},
        }

        return {"message": f"Registration code {code_id} revoked"}, audit_info

    except Exception as exc:
        logger.error("Failed to delete registration code")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete registration code",
        ) from exc
