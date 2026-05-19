# registration_service.py
# Description: User registration service with transaction safety and directory management
#
# Imports
import asyncio
import json
import os
import secrets
import shutil
import stat
import string
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

#
# 3rd-party imports
from loguru import logger

from tldw_Server_API.app.core.AuthNZ.database import DatabasePool, get_db_pool
from tldw_Server_API.app.core.AuthNZ.exceptions import (
    DirectoryCreationError,
    DuplicateUserError,
    InvalidRegistrationCodeError,
    RegistrationCodeExhaustedError,
    RegistrationCodeExpiredError,
    RegistrationDisabledError,
    RegistrationError,
)
from tldw_Server_API.app.core.AuthNZ.password_service import PasswordService, get_password_service

#
# Local imports
from tldw_Server_API.app.core.AuthNZ.settings import Settings, get_settings
from tldw_Server_API.app.core.testing import is_test_mode

_OWNER_ONLY_DIR_MODE = stat.S_IRWXU

#######################################################################################################################
#
# Registration Service Class

class RegistrationService:
    """Service for user registration with full transaction safety"""

    def __init__(
        self,
        db_pool: Optional[DatabasePool] = None,
        password_service: Optional[PasswordService] = None,
        settings: Optional[Settings] = None
    ):
        """Initialize registration service"""
        self.settings = settings or get_settings()
        self.db_pool = db_pool
        self.password_service = password_service or get_password_service()

        # Check if registration is enabled
        self.registration_enabled = self.settings.ENABLE_REGISTRATION
        self.require_code = self.settings.REQUIRE_REGISTRATION_CODE

        logger.info(
            f"RegistrationService initialized (enabled={self.registration_enabled}, "
            f"require_code={self.require_code})"
        )

    def _is_postgres_backend(self) -> bool:
        """
        Return True when this service is operating on a PostgreSQL backend.

        Backend selection is derived from DatabasePool state instead of probing
        connection method presence, because shim connections can expose methods
        that do not reflect the active backend.
        """
        if not self.db_pool:
            return False
        if getattr(self.db_pool, "pool", None):
            return True
        backend = getattr(self.db_pool, "backend", None)
        if isinstance(backend, str):
            return backend.strip().lower() in {"postgres", "postgresql", "pg"}
        return False

    async def initialize(self):
        """Initialize the registration service"""
        if not self.db_pool:
            self.db_pool = await get_db_pool()

    def generate_registration_code(self, length: int = 24) -> str:
        """
        Generate a secure registration code

        Args:
            length: Length of the code

        Returns:
            Secure random alphanumeric code
        """
        alphabet = string.ascii_letters + string.digits
        return ''.join(secrets.choice(alphabet) for _ in range(length))

    def _create_user_directories(self, user_id: int) -> bool:
        """
        Create user directories (runs in thread pool)

        Args:
            user_id: User's database ID

        Returns:
            True if successful, False otherwise
        """
        try:
            base_path = Path(self.settings.USER_DATA_BASE_PATH)
            user_dir = base_path / str(user_id)

            # Create main user directory
            user_dir.mkdir(parents=True, exist_ok=True)

            # Create subdirectories
            subdirs = ["media", "notes", "embeddings", "exports", "temp"]
            for subdir in subdirs:
                (user_dir / subdir).mkdir(exist_ok=True)

            # Create user-specific ChromaDB directory if configured
            if self.settings.CHROMADB_BASE_PATH:
                chroma_path = Path(self.settings.CHROMADB_BASE_PATH) / str(user_id)
                chroma_path.mkdir(parents=True, exist_ok=True)

            # Set permissions on Unix-like systems
            if os.name != 'nt':
                os.chmod(user_dir, _OWNER_ONLY_DIR_MODE)
                for subdir in subdirs:
                    os.chmod(user_dir / subdir, _OWNER_ONLY_DIR_MODE)

            logger.debug(f"Created directories for user {user_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to create directories for user {user_id}: {e}")
            return False

    def _cleanup_user_directories(self, user_id: int):
        """
        Clean up user directories (for rollback)

        Args:
            user_id: User's database ID
        """
        try:
            # Remove user data directory
            user_dir = Path(self.settings.USER_DATA_BASE_PATH) / str(user_id)
            if user_dir.exists():
                shutil.rmtree(user_dir, ignore_errors=True)

            # Remove ChromaDB directory
            if self.settings.CHROMADB_BASE_PATH:
                chroma_dir = Path(self.settings.CHROMADB_BASE_PATH) / str(user_id)
                if chroma_dir.exists():
                    shutil.rmtree(chroma_dir, ignore_errors=True)

            logger.debug(f"Cleaned up directories for user {user_id}")

        except Exception as e:
            logger.error(f"Error cleaning up directories for user {user_id}: {e}")

    def _merge_org_scope_from_metadata(self, code_info: dict[str, Any]) -> dict[str, Any]:
        metadata = code_info.get("metadata")
        if metadata:
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except json.JSONDecodeError:
                    metadata = {}
            elif not isinstance(metadata, dict):
                metadata = {}
        else:
            metadata = {}

        for key in ("org_id", "org_role", "team_id"):
            if code_info.get(key) is None and metadata.get(key) is not None:
                code_info[key] = metadata.get(key)

        for key in ("org_id", "team_id"):
            if code_info.get(key) is not None:
                try:
                    code_info[key] = int(code_info[key])
                except (TypeError, ValueError):
                    code_info[key] = None

        if code_info.get("org_role") is not None:
            code_info["org_role"] = str(code_info["org_role"]).lower()

        return code_info

    async def _ensure_org_membership(
        self,
        conn,
        *,
        user_id: int,
        org_id: int,
        org_role: Optional[str],
        team_id: Optional[int],
    ) -> bool:
        org_role_value = (org_role or "member").lower()
        if org_role_value not in {"owner", "admin", "lead", "member"}:
            org_role_value = "member"

        was_already_member = False
        if self._is_postgres_backend():
            existing = await conn.fetchrow(
                "SELECT 1 FROM org_members WHERE org_id = $1 AND user_id = $2",
                org_id,
                user_id,
            )
            if existing:
                was_already_member = True
            else:
                await conn.execute(
                    """
                    INSERT INTO org_members (org_id, user_id, role)
                    VALUES ($1, $2, $3)
                    ON CONFLICT (org_id, user_id) DO NOTHING
                    """,
                    org_id,
                    user_id,
                    org_role_value,
                )
            if team_id is not None:
                await conn.execute(
                    """
                    INSERT INTO team_members (team_id, user_id, role)
                    VALUES ($1, $2, $3)
                    ON CONFLICT (team_id, user_id) DO NOTHING
                    """,
                    team_id,
                    user_id,
                    "member",
                )
        else:
            cursor = await conn.execute(
                "SELECT 1 FROM org_members WHERE org_id = ? AND user_id = ?",
                (org_id, user_id),
            )
            existing = await cursor.fetchone()
            if existing:
                was_already_member = True
            else:
                await conn.execute(
                    "INSERT OR IGNORE INTO org_members (org_id, user_id, role) VALUES (?, ?, ?)",
                    (org_id, user_id, org_role_value),
                )
            if team_id is not None:
                await conn.execute(
                    "INSERT OR IGNORE INTO team_members (team_id, user_id, role) VALUES (?, ?, ?)",
                    (team_id, user_id, "member"),
                )

        return was_already_member

    async def register_user(
        self,
        username: str,
        email: str,
        password: str,
        registration_code: Optional[str] = None,
        created_by: Optional[int] = None,
        role_override: Optional[str] = None,
        is_active_override: Optional[bool] = None,
        is_verified_override: Optional[bool] = None,
        storage_quota_override: Optional[int] = None,
        system_provisioning: bool = False,
    ) -> dict[str, Any]:
        """
        Register a new user with full transaction safety

        Args:
            username: Username (must be unique)
            email: Email address (must be unique)
            password: Plain text password
            registration_code: Optional registration code
            created_by: ID of user creating this account (for admin creation)
            role_override: Optional admin-specified role for the new user
            is_active_override: Optional admin override for active status
            is_verified_override: Optional admin override for verification status
            storage_quota_override: Optional admin override for storage quota
            system_provisioning: Internal provisioning bypass for IdP/JIT-created users

        Returns:
            Dictionary with user information

        Raises:
            Various registration exceptions
        """
        privileged_creation = created_by is not None or system_provisioning

        # Check if registration is enabled
        if not self.registration_enabled and not privileged_creation:
            raise RegistrationDisabledError()

        # Initialize if needed
        if not self.db_pool:
            await self.initialize()

        # Validate password strength
        self.password_service.validate_password_strength(password, username)

        user_id = None
        directories_created = False
        code_info: Optional[dict[str, Any]] = None

        try:
            async with self.db_pool.transaction() as conn:
                # Check for duplicate username/email
                if self._is_postgres_backend():
                    # PostgreSQL
                    existing = await conn.fetchrow(
                        """
                        SELECT username, email
                        FROM users
                        WHERE username = $1 OR email = $2
                        """,
                        username.lower(), email.lower()
                    )
                else:
                    # SQLite
                    cursor = await conn.execute(
                        """
                        SELECT username, email
                        FROM users
                        WHERE lower(username) = ? OR lower(email) = ?
                        """,
                        (username.lower(), email.lower())
                    )
                    existing = await cursor.fetchone()

                if existing:
                    if existing[0] and existing[0].lower() == username.lower():
                        raise DuplicateUserError("username")
                    else:
                        raise DuplicateUserError("email")

                # Validate registration code if required
                role = self.settings.DEFAULT_USER_ROLE
                storage_quota = self.settings.DEFAULT_STORAGE_QUOTA_MB

                if privileged_creation:
                    if role_override:
                        role = role_override
                    if storage_quota_override is not None:
                        if int(storage_quota_override) < 0:
                            raise ValueError("storage_quota_override must be non-negative")
                        storage_quota = storage_quota_override

                if registration_code and not privileged_creation:
                    # Validate and use registration code (optional when not required)
                    code_info = await self._validate_and_use_registration_code(
                        registration_code,
                        conn,
                        user_email=email,
                    )
                    role = code_info.get('role_to_grant', role)

                    # Check if code specifies storage quota
                    if 'storage_quota_mb' in code_info:
                        storage_quota = code_info['storage_quota_mb']
                elif self.require_code and not privileged_creation:
                    raise InvalidRegistrationCodeError("Registration code required")

                # Hash the password
                password_hash = self.password_service.hash_password(password)

                # Generate UUID
                user_uuid = str(uuid4())

                # Create user
                is_active = True
                if privileged_creation and is_active_override is not None:
                    is_active = bool(is_active_override)

                is_verified = not self.require_code
                if privileged_creation:
                    is_verified = bool(is_verified_override) if is_verified_override is not None else True

                if self._is_postgres_backend():
                    # PostgreSQL
                    user_id = await conn.fetchval(
                        """
                        INSERT INTO users (
                            uuid, username, email, password_hash, role,
                            is_active, is_verified, created_by, storage_quota_mb
                        )
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                        RETURNING id
                        """,
                        user_uuid, username, email, password_hash, role,
                        is_active, is_verified, created_by, storage_quota
                    )
                else:
                    # SQLite
                    cursor = await conn.execute(
                        """
                        INSERT INTO users (
                            uuid, username, email, password_hash, role,
                            is_active, is_verified, created_by, storage_quota_mb
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (user_uuid, username, email, password_hash, role,
                         int(is_active), int(is_verified), created_by, storage_quota)
                    )
                    user_id = cursor.lastrowid

                # Add password to history
                await self._add_password_to_history(user_id, password_hash, conn)

                if code_info and code_info.get("org_id") is not None:
                    await self._ensure_org_membership(
                        conn,
                        user_id=user_id,
                        org_id=int(code_info["org_id"]),
                        org_role=code_info.get("org_role"),
                        team_id=code_info.get("team_id"),
                    )

                # Create user directories (before committing transaction)
                directories_created = await asyncio.to_thread(
                    self._create_user_directories,
                    user_id
                )

                if not directories_created:
                    if is_test_mode():
                        logger.warning(f"TEST_MODE: Skipping directory creation failure for user {user_id}")
                    else:
                        raise DirectoryCreationError(
                            f"user_databases/{user_id}",
                            "Failed to create user directories"
                        )

                # Log registration in audit log
                await self._log_registration(
                    user_id, username, email, role,
                    created_by,
                    registration_code is not None,
                    conn,
                    registration_code_id=code_info.get("id") if code_info else None,
                    registration_code_org_id=code_info.get("org_id") if code_info else None,
                    registration_code_org_role=code_info.get("org_role") if code_info else None,
                    registration_code_team_id=code_info.get("team_id") if code_info else None,
                )

                # If we get here, everything succeeded
                try:
                    _s = get_settings()
                    if getattr(_s, 'PII_REDACT_LOGS', False):
                        logger.info("User registration successful [redacted]")
                    else:
                        logger.info(
                            f"Successfully registered user: {username} (ID: {user_id}, Role: {role})"
                        )
                except Exception:
                    logger.info(
                        f"Successfully registered user: {username} (ID: {user_id}, Role: {role})"
                    )

                return {
                    "user_id": user_id,
                    "uuid": user_uuid,
                    "username": username,
                    "email": email,
                    "role": role,
                    "is_active": is_active,
                    "is_verified": is_verified,
                    "storage_quota_mb": storage_quota,
                    "registration_code_id": code_info.get("id") if code_info else None,
                    "registration_code_org_id": code_info.get("org_id") if code_info else None,
                    "registration_code_org_role": code_info.get("org_role") if code_info else None,
                    "registration_code_team_id": code_info.get("team_id") if code_info else None,
                }

        except Exception as e:
            # Rollback: Clean up directories if they were created
            if user_id and directories_created:
                await asyncio.to_thread(self._cleanup_user_directories, user_id)

            # Log the error
            try:
                _s = get_settings()
                if getattr(_s, 'PII_REDACT_LOGS', False):
                    logger.error(f"Registration failed [redacted]: {e}")
                else:
                    logger.error(f"Registration failed for {username}: {e}")
            except Exception:
                logger.error(f"Registration failed for {username}: {e}")

            # Re-raise the exception
            raise

    async def rollback_user_registration(self, user_id: int) -> bool:
        """Compensate a partially completed registration by tombstoning the user row."""
        if not self.db_pool:
            await self.initialize()

        tombstone = f"registration-rollback-{user_id}-{uuid4().hex[:12]}"
        tombstone_email = f"{tombstone}@invalid.local"

        try:
            async with self.db_pool.transaction() as conn:
                if self._is_postgres_backend():
                    existing = await conn.fetchrow("SELECT id FROM users WHERE id = $1", user_id)
                    if not existing:
                        return False
                    await conn.execute(
                        """
                        UPDATE users
                        SET username = $2,
                            email = $3,
                            is_active = FALSE,
                            is_verified = FALSE,
                            updated_at = CURRENT_TIMESTAMP
                        WHERE id = $1
                        """,
                        user_id,
                        tombstone,
                        tombstone_email,
                    )
                else:
                    cursor = await conn.execute("SELECT id FROM users WHERE id = ?", (user_id,))
                    existing = await cursor.fetchone()
                    if not existing:
                        return False
                    await conn.execute(
                        """
                        UPDATE users
                        SET username = ?,
                            email = ?,
                            is_active = 0,
                            is_verified = 0,
                            updated_at = datetime('now')
                        WHERE id = ?
                        """,
                        (tombstone, tombstone_email, user_id),
                    )

            await asyncio.to_thread(self._cleanup_user_directories, user_id)
            logger.warning(
                "Rolled back partially registered user {} after downstream provisioning failure",
                user_id,
            )
            return True
        except Exception as exc:
            logger.opt(exception=exc).error(
                "Failed to roll back partially registered user {}",
                user_id,
            )
            return False

    async def accept_org_invite_code(
        self,
        *,
        code: str,
        user_id: int,
    ) -> dict[str, Any]:
        """Accept an org-scoped registration code for an existing user."""
        if not self.registration_enabled:
            raise RegistrationDisabledError()

        if not self.db_pool:
            await self.initialize()

        async with self.db_pool.transaction() as conn:
            user_email = None
            if self._is_postgres_backend():
                user_row = await conn.fetchrow("SELECT email FROM users WHERE id = $1", user_id)
                if user_row:
                    user_email = user_row["email"]
            else:
                cursor = await conn.execute("SELECT email FROM users WHERE id = ?", (user_id,))
                row = await cursor.fetchone()
                if row:
                    user_email = row[0]

            code_info = await self._validate_and_use_registration_code(
                code,
                conn,
                user_email=user_email,
            )
            org_id = code_info.get("org_id")
            if org_id is None:
                raise InvalidRegistrationCodeError("Invite code does not grant organization access")

            was_already_member = await self._ensure_org_membership(
                conn,
                user_id=user_id,
                org_id=int(org_id),
                org_role=code_info.get("org_role"),
                team_id=code_info.get("team_id"),
            )

            return {
                "org_id": int(org_id),
                "org_role": code_info.get("org_role") or "member",
                "team_id": code_info.get("team_id"),
                "was_already_member": was_already_member,
                "registration_code_id": code_info.get("id"),
            }

    async def _validate_and_use_registration_code(
        self,
        code: str,
        conn,
        *,
        user_email: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Validate and consume a registration code

        Args:
            code: Registration code
            conn: Database connection (in transaction)

        Returns:
            Code information

        Raises:
            InvalidRegistrationCodeError: If code is invalid
        """
        if self._is_postgres_backend():
            # PostgreSQL - Use SELECT FOR UPDATE to lock the row
            code_row = await conn.fetchrow(
                """
                SELECT id, role_to_grant, times_used, max_uses,
                       expires_at, is_active, description, allowed_email_domain,
                       org_id, org_role, team_id, metadata
                FROM registration_codes
                WHERE code = $1
                FOR UPDATE
                """,
                code
            )

            if not code_row:
                raise InvalidRegistrationCodeError("Invalid registration code")

            # Check if active
            if not code_row['is_active']:
                raise InvalidRegistrationCodeError("Registration code is inactive")

            # Check expiration
            if code_row['expires_at'] < datetime.utcnow():
                raise RegistrationCodeExpiredError()

            # Check usage limit
            if code_row['times_used'] >= code_row['max_uses']:
                raise RegistrationCodeExhaustedError()

            code_info = dict(code_row)
            code_info = self._merge_org_scope_from_metadata(code_info)

            if code_info.get("org_id") is not None and not self.settings.ENABLE_ORG_SCOPED_REGISTRATION_CODES:
                raise InvalidRegistrationCodeError("Org-scoped registration codes are disabled")

            allowed_domain = code_info.get("allowed_email_domain")
            if allowed_domain:
                normalized = str(allowed_domain).strip().lower().lstrip("@")
                if not user_email:
                    raise InvalidRegistrationCodeError(
                        "Registration code is restricted to allowed email domain"
                    )
                email_domain = user_email.split("@")[-1].lower()
                if email_domain != normalized:
                    raise InvalidRegistrationCodeError(
                        "Registration code is restricted to allowed email domain"
                    )

            # Update usage count
            await conn.execute(
                """
                UPDATE registration_codes
                SET times_used = times_used + 1
                WHERE id = $1
                """,
                code_row['id']
            )

            return code_info

        else:
            # SQLite - Manual locking with transaction
            cursor = await conn.execute(
                """
                SELECT id, role_to_grant, times_used, max_uses,
                       expires_at, is_active, description, allowed_email_domain,
                       org_id, org_role, team_id, metadata
                FROM registration_codes
                WHERE code = ?
                """,
                (code,)
            )
            code_row = await cursor.fetchone()

            if not code_row:
                raise InvalidRegistrationCodeError("Invalid registration code")

            # Convert to dict for easier access
            code_info = {
                "id": code_row[0],
                "role_to_grant": code_row[1],
                "times_used": code_row[2],
                "max_uses": code_row[3],
                "expires_at": datetime.fromisoformat(code_row[4]),
                "is_active": code_row[5],
                "description": code_row[6],
                "allowed_email_domain": code_row[7],
                "org_id": code_row[8],
                "org_role": code_row[9],
                "team_id": code_row[10],
                "metadata": code_row[11],
            }

            # Validate
            if not code_info['is_active']:
                raise InvalidRegistrationCodeError("Registration code is inactive")

            if code_info['expires_at'] < datetime.utcnow():
                raise RegistrationCodeExpiredError()

            if code_info['times_used'] >= code_info['max_uses']:
                raise RegistrationCodeExhaustedError()

            code_info = self._merge_org_scope_from_metadata(code_info)

            if code_info.get("org_id") is not None and not self.settings.ENABLE_ORG_SCOPED_REGISTRATION_CODES:
                raise InvalidRegistrationCodeError("Org-scoped registration codes are disabled")

            allowed_domain = code_info.get("allowed_email_domain")
            if allowed_domain:
                normalized = str(allowed_domain).strip().lower().lstrip("@")
                if not user_email:
                    raise InvalidRegistrationCodeError(
                        "Registration code is restricted to allowed email domain"
                    )
                email_domain = user_email.split("@")[-1].lower()
                if email_domain != normalized:
                    raise InvalidRegistrationCodeError(
                        "Registration code is restricted to allowed email domain"
                    )

            # Update usage
            await conn.execute(
                """
                UPDATE registration_codes
                SET times_used = times_used + 1
                WHERE id = ?
                """,
                (code_info['id'],)
            )

            return code_info

    async def _add_password_to_history(
        self,
        user_id: int,
        password_hash: str,
        conn
    ):
        """Add password to user's password history"""
        try:
            if self._is_postgres_backend():
                # PostgreSQL
                await conn.execute(
                    """
                    INSERT INTO password_history (user_id, password_hash)
                    VALUES ($1, $2)
                    """,
                    user_id, password_hash
                )
            else:
                # SQLite
                await conn.execute(
                    """
                    INSERT INTO password_history (user_id, password_hash)
                    VALUES (?, ?)
                    """,
                    (user_id, password_hash)
                )
        except Exception as e:
            logger.error(f"Failed to add password to history: {e}")
            # Don't fail registration for this

    async def _log_registration(
        self,
        user_id: int,
        username: str,
        email: str,
        role: str,
        created_by: Optional[int],
        used_code: bool,
        conn,
        registration_code_id: Optional[int] = None,
        registration_code_org_id: Optional[int] = None,
        registration_code_org_role: Optional[str] = None,
        registration_code_team_id: Optional[int] = None,
    ):
        """Log registration in audit log"""
        try:
            details = {
                "username": username,
                "email": email,
                "role": role,
                "used_registration_code": used_code,
                "created_by": created_by,
                "registration_code_id": registration_code_id,
                "registration_code_org_id": registration_code_org_id,
                "registration_code_org_role": registration_code_org_role,
                "registration_code_team_id": registration_code_team_id,
            }

            if self._is_postgres_backend():
                # PostgreSQL
                await conn.execute(
                    """
                    INSERT INTO audit_log (
                        user_id, action, target_type, target_id,
                        success, details
                    )
                    VALUES ($1, $2, $3, $4, $5, $6)
                    """,
                    user_id, 'user_registered', 'user', user_id,
                    True, json.dumps(details)
                )
            else:
                # SQLite
                await conn.execute(
                    """
                    INSERT INTO audit_log (
                        user_id, action, target_type, target_id,
                        success, details
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (user_id, 'user_registered', 'user', user_id,
                     1, json.dumps(details))
                )
        except Exception as e:
            logger.error(f"Failed to log registration: {e}")
            # Don't fail registration for this

    async def create_registration_code(
        self,
        created_by: int,
        max_uses: int = 1,
        expires_in_days: Optional[int] = None,
        role_to_grant: str = "user",
        description: Optional[str] = None,
        allowed_email_domain: Optional[str] = None
    ) -> dict[str, Any]:
        """
        Create a new registration code

        Args:
            created_by: ID of user creating the code
            max_uses: Maximum number of uses
            expires_in_days: Days until expiration
            role_to_grant: Role to grant to users
            description: Optional description
            allowed_email_domain: Restrict to email domain

        Returns:
            Registration code information
        """
        if not self.db_pool:
            await self.initialize()

        code = self.generate_registration_code()
        expires_at = datetime.utcnow() + timedelta(
            days=expires_in_days or self.settings.REGISTRATION_CODE_DEFAULT_EXPIRY_DAYS
        )

        try:
            async with self.db_pool.transaction() as conn:
                if self._is_postgres_backend():
                    # PostgreSQL
                    code_id = await conn.fetchval(
                        """
                        INSERT INTO registration_codes (
                            code, max_uses, expires_at, created_by,
                            role_to_grant, description, allowed_email_domain
                        )
                        VALUES ($1, $2, $3, $4, $5, $6, $7)
                        RETURNING id
                        """,
                        code, max_uses, expires_at, created_by,
                        role_to_grant, description, allowed_email_domain
                    )
                else:
                    # SQLite
                    cursor = await conn.execute(
                        """
                        INSERT INTO registration_codes (
                            code, max_uses, expires_at, created_by,
                            role_to_grant, description, allowed_email_domain
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (code, max_uses, expires_at.isoformat(), created_by,
                         role_to_grant, description, allowed_email_domain)
                    )
                    code_id = cursor.lastrowid
                    await conn.commit()

                logger.info(
                    f"Created registration code {code[:8]}... "
                    f"(max_uses={max_uses}, role={role_to_grant})"
                )

                return {
                    "id": code_id,
                    "code": code,
                    "max_uses": max_uses,
                    "expires_at": expires_at.isoformat(),
                    "role_to_grant": role_to_grant,
                    "description": description
                }

        except Exception as e:
            logger.error(f"Failed to create registration code: {e}")
            raise RegistrationError(f"Failed to create registration code: {e}") from e

    async def list_registration_codes(
        self,
        active_only: bool = True,
        created_by: Optional[int] = None
    ) -> list[dict[str, Any]]:
        """
        List registration codes

        Args:
            active_only: Only show active codes
            created_by: Filter by creator

        Returns:
            List of registration codes
        """
        if not self.db_pool:
            await self.initialize()

        query_parts = ["SELECT * FROM registration_codes WHERE 1=1"]
        params = []

        if active_only:
            query_parts.append("AND is_active = ?")
            params.append(True if self._is_postgres_backend() else 1)
            query_parts.append("AND expires_at > ?")
            params.append(datetime.utcnow())

        if created_by:
            query_parts.append("AND created_by = ?")
            params.append(created_by)

        query_parts.append("ORDER BY created_at DESC")
        query = " ".join(query_parts)

        codes = await self.db_pool.fetchall(query, *params)
        return codes

    async def revoke_registration_code(self, code_id: int):
        """Revoke a registration code"""
        if not self.db_pool:
            await self.initialize()

        await self.db_pool.execute(
            "UPDATE registration_codes SET is_active = ? WHERE id = ?",
            False, code_id
        )
        logger.info(f"Revoked registration code {code_id}")

    async def shutdown(self):
        """Shutdown the registration service"""
        logger.info("RegistrationService shutdown complete (no dedicated executor)")


#######################################################################################################################
#
# Module Functions

# Global instance
_registration_service: Optional[RegistrationService] = None


async def get_registration_service() -> RegistrationService:
    """Get registration service singleton"""
    global _registration_service
    if not _registration_service:
        _registration_service = RegistrationService()
        await _registration_service.initialize()
    return _registration_service


async def reset_registration_service():
    """Reset registration service singleton (for testing)"""
    global _registration_service
    if _registration_service:
        await _registration_service.shutdown()
        _registration_service = None


#
# End of registration_service.py
#######################################################################################################################
