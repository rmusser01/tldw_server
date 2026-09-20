# UserDatabase_v2.py
# Description: User and authentication database management using DatabaseBackend interface
# This version uses the existing DatabaseBackend interface for database-agnostic operations
#
# Handles user management, RBAC, registration codes, and authentication for the tldw_server
# with support for both SQLite and PostgreSQL backends.
#
########################################################################################################################

import contextlib
import json
import secrets
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from loguru import logger

from tldw_Server_API.app.core.AuthNZ.postgres_profile_version_schema import (
    ensure_postgres_profile_version_sync,
)
from tldw_Server_API.app.core.AuthNZ.profile_candidate_schema import (
    PROFILE_CANDIDATE_TABLES,
    profile_candidate_schema_is_valid,
)
from tldw_Server_API.app.core.AuthNZ.profile_user_sync_boundary import (
    _guard_authnz_sync_backend,
)
from tldw_Server_API.app.core.AuthNZ.profile_version import (
    PROFILE_VISIBLE_USER_FIELDS,
    ProfileVersionNotFound,
    VersionedUserWriteGateway,
)
from tldw_Server_API.app.core.AuthNZ.sqlite_profile_version_schema import (
    remediate_sqlite_profile_version_schema,
    sqlite_profile_version_connection_invalid,
)

# Local imports
from tldw_Server_API.app.core.DB_Management.backends.base import (
    BackendType,
    DatabaseBackend,
    DatabaseConfig,
    DatabaseError,
)
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.sql_utils import split_sql_statements

_USERDB_NONCRITICAL_EXCEPTIONS: tuple[type[BaseException], ...] = (
    AttributeError,
    DatabaseError,
    LookupError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
    json.JSONDecodeError,
)

########################################################################################################################
# Custom Exceptions
########################################################################################################################

class UserDatabaseError(DatabaseError):
    """Base exception for user database related errors."""
    pass

class UserNotFoundError(UserDatabaseError):
    """User not found in database."""
    pass

class DuplicateUserError(UserDatabaseError):
    """User already exists."""
    pass

class InvalidPermissionError(UserDatabaseError):
    """Invalid permission or role."""
    pass

class RegistrationCodeError(UserDatabaseError):
    """Registration code related errors."""
    pass

class AuthenticationError(UserDatabaseError):
    """Authentication related errors."""
    pass

########################################################################################################################
# UserDatabase Class
########################################################################################################################

class UserDatabase:
    """
    Manages user authentication and authorization using the DatabaseBackend interface,
    supporting both SQLite and PostgreSQL backends.
    """

    _CURRENT_SCHEMA_VERSION = 1

    def __init__(self, backend: Optional[DatabaseBackend] = None,
                 config: Optional[DatabaseConfig] = None,
                 client_id: str = "auth_service"):
        """
        Initialize UserDatabase instance.

        Args:
            backend: Pre-configured DatabaseBackend instance
            config: DatabaseConfig for creating a new backend
            client_id: Identifier for the client/instance making changes
        """
        self.client_id = client_id

        # Use provided backend or create from config
        if backend:
            raw_backend = backend
        elif config:
            raw_backend = DatabaseBackendFactory.create_backend(config)
        else:
            # Default to SQLite with Users.db
            default_sqlite_path = (
                Path(__file__).resolve()
                .parent.parent.parent.parent.parent
                / "Databases"
                / "users.db"
            )
            config = DatabaseConfig(
                backend_type=BackendType.SQLITE,
                sqlite_path=str(default_sqlite_path)
            )
            raw_backend = DatabaseBackendFactory.create_backend(config)

        # Schema/bootstrap is a narrow, non-serving maintenance phase. The
        # managed runtime boundary is installed only after it succeeds.
        self.backend = raw_backend
        self._initialize_schema()
        self.backend = _guard_authnz_sync_backend(raw_backend)

        logger.info(
            "UserDatabase initialized backend={} target={} client_id={}",
            self.backend.backend_type.value,
            self._describe_backend_target(),
            self.client_id,
        )

    def _describe_backend_target(self) -> str:
        config = getattr(self.backend, "config", None)
        if not config:
            return "<no-config>"

        backend_type = getattr(self.backend, "backend_type", None)
        if backend_type == BackendType.SQLITE:
            raw_path = (config.sqlite_path or "").strip()
            if not raw_path:
                return "<sqlite-default>"
            if raw_path == ":memory:":
                return raw_path
            if raw_path.lower().startswith("file:"):
                return raw_path
            return str(Path(raw_path).resolve())
        elif backend_type == BackendType.POSTGRESQL:
            host = config.pg_host or "localhost"
            port = config.pg_port or 5432
            database = config.pg_database or "<postgres>"
            return f"{host}:{port}/{database}"
        return "<unknown>"

    def _initialize_schema(self):
        """Initialize database schema if needed."""
        # Determine schema file based on backend type
        schema_name = "users_auth_schema.sql"
        # Go up to the project root (tldw_server/)
        base_path = Path(__file__).parent.parent.parent.parent.parent

        if self.backend.backend_type == BackendType.SQLITE:
            schema_path = base_path / "Databases" / "SQLite" / "Schema" / schema_name
        elif self.backend.backend_type == BackendType.POSTGRESQL:
            schema_path = base_path / "Databases" / "Postgres" / "Schema" / schema_name
        else:
            backend_label = getattr(self.backend.backend_type, "value", str(self.backend.backend_type))
            raise UserDatabaseError(
                f"Unsupported backend type for schema initialization: {backend_label}"
            )

        schema_statements: Optional[list[str]] = None
        loaded_from_file = False

        if schema_path.exists():
            try:
                with open(schema_path, encoding='utf-8') as f:
                    schema_sql = f.read()
                schema_statements = self._split_sql_statements(schema_sql)
                logger.info("Database schema loaded from packaged schema file")
                loaded_from_file = True
            except Exception as exc:  # noqa: BLE001
                logger.bind(
                    operation="schema_file_read",
                    backend=self.backend.backend_type.value,
                    exception_type=type(exc).__name__,
                ).error("Failed to read user database schema file")

        if not schema_statements:
            logger.warning(
                'Schema file not available for {} backend, using embedded defaults',
                self.backend.backend_type.value,
            )
            schema_statements = self._default_schema_statements()

        try:
            self._apply_schema_statements(schema_statements)
        except Exception as exc:  # noqa: BLE001
            if loaded_from_file:
                fallback_statements = self._default_schema_statements()
                logger.info("Retrying schema initialization with embedded defaults")
                try:
                    self._apply_schema_statements(fallback_statements)
                except Exception as fallback_exc:  # noqa: BLE001
                    logger.bind(
                        operation="schema_fallback_apply",
                        backend=self.backend.backend_type.value,
                        exception_type=type(fallback_exc).__name__,
                    ).error("Required user database schema fallback failed")
                    raise UserDatabaseError(
                        "Required schema initialization failed after fallback"
                    ) from None
            else:
                logger.bind(
                    operation="schema_apply",
                    backend=self.backend.backend_type.value,
                    exception_type=type(exc).__name__,
                ).error("Required user database schema initialization failed")
                raise UserDatabaseError(
                    "Required schema initialization failed"
                ) from None

        if self.backend.backend_type == BackendType.SQLITE:
            self._ensure_sqlite_profile_version_schema()
        elif self.backend.backend_type == BackendType.POSTGRESQL:
            with self.backend.transaction() as connection:
                ensure_postgres_profile_version_sync(
                    self.backend,
                    connection=connection,
                )
        self._ensure_core_columns()
        self._seed_default_data()

    ########################################################################################################################
    # User Management Methods
    ########################################################################################################################

    def create_user(self, username: str, email: str, password_hash: str,
                   role: str = 'user', **kwargs) -> int:
        """
        Create a new user.

        Args:
            username: Unique username
            email: User email address
            password_hash: Hashed password
            role: Initial role (default: 'user')
            **kwargs: Additional user fields

        Returns:
            int: User ID of created user

        Raises:
            DuplicateUserError: If username or email already exists
        """
        try:
            # Basic validation
            if not isinstance(username, str) or not username.strip():
                raise ValueError("Username cannot be empty")
            if not isinstance(email, str) or not email.strip():
                raise ValueError("Email cannot be empty")
            # Enforce max lengths similar to typical DB constraints
            if username is not None and len(username) > 255:
                username = username[:255]
            if email is not None and len(email) > 255:
                email = email[:255]
            extra_fields = dict(kwargs) if kwargs else {}
            user_uuid = extra_fields.pop("uuid", str(uuid4()))
            metadata = json.dumps(extra_fields) if extra_fields else None

            with self.backend.transaction() as conn:
                # Check for duplicates
                existing = self.backend.execute(
                    "SELECT id FROM users WHERE username = ? OR email = ?",
                    (username, email),
                    connection=conn,
                )

                if existing.rows:
                    raise DuplicateUserError("Username or email already exists")

                gateway = VersionedUserWriteGateway(
                    "postgres"
                    if self.backend.backend_type == BackendType.POSTGRESQL
                    else "sqlite"
                )
                gateway.insert_user_sync(
                    self.backend,
                    conn,
                    values={
                        "uuid": user_uuid,
                        "username": username,
                        "email": email,
                        "password_hash": password_hash,
                        "metadata": metadata,
                    },
                )
                # Retrieve ID using UUID to support backends without lastrowid
                user_lookup = self.backend.execute(
                    "SELECT id FROM users WHERE uuid = ?",
                    (user_uuid,),
                    connection=conn,
                )
                if not user_lookup.rows:
                    raise UserDatabaseError("Failed to locate newly created user record")
                user_id = user_lookup.rows[0]['id']

                # Assign default role
                role_result = self.backend.execute(
                    "SELECT id FROM roles WHERE name = ?",
                    (role,),
                    connection=conn,
                )

                if role_result.rows:
                    role_id = role_result.rows[0]['id']
                    self.backend.execute(
                        "INSERT INTO user_roles (user_id, role_id) VALUES (?, ?)",
                        (user_id, role_id),
                        connection=conn,
                    )

                # Log the creation
                self._audit_log(
                    'user_created',
                    user_id,
                    None,
                    {'username': username, 'email': email, 'role': role},
                    connection=conn,
                )

                logger.info(f"Created user {username} with ID {user_id}")
                return user_id

        except Exception as e:
            # Preserve explicit duplicate signal
            if isinstance(e, DuplicateUserError):
                raise
            emsg = str(e).lower()
            if ("duplicate" in emsg) or ("unique" in emsg) or ("already exists" in emsg):
                raise DuplicateUserError("Username or email already exists") from e
            logger.bind(
                operation="create_user",
                backend=self.backend.backend_type.value,
                exception_type=type(e).__name__,
            ).error("Failed to create user")
            raise UserDatabaseError("Failed to create user") from None

    def get_user(self, user_id: Optional[int] = None, username: Optional[str] = None,
                 email: Optional[str] = None) -> Optional[dict[str, Any]]:
        """
        Get user by ID, username, or email.

        Args:
            user_id: User ID
            username: Username
            email: Email address

        Returns:
            Dict containing user data or None if not found
        """
        if user_id:
            result = self.backend.execute(
                "SELECT * FROM users WHERE id = ?", (user_id,)
            )
        elif username:
            result = self.backend.execute(
                "SELECT * FROM users WHERE username = ?", (username,)
            )
        elif email:
            result = self.backend.execute(
                "SELECT * FROM users WHERE email = ?", (email,)
            )
        else:
            return None

        if result.rows:
            user_dict = result.rows[0]
            # Normalize metadata to dict
            try:
                meta = user_dict.get('metadata')
                if isinstance(meta, str) and meta:
                    user_dict['metadata'] = json.loads(meta)
                elif meta is None:
                    user_dict['metadata'] = {}
            except _USERDB_NONCRITICAL_EXCEPTIONS:
                user_dict['metadata'] = {}
            # Normalize boolean-ish flags for cross-backend consistency
            for _flag in ("is_active", "is_verified", "is_superuser"):
                try:
                    if _flag in user_dict:
                        user_dict[_flag] = bool(user_dict[_flag])
                except _USERDB_NONCRITICAL_EXCEPTIONS:
                    pass
            # Add roles
            user_dict['roles'] = self.get_user_roles(user_dict['id'])
            return user_dict
        return None

    def update_user(self, user_id: int, **updates) -> bool:
        """
        Update user information.

        Args:
            user_id: User ID to update
            **updates: Fields to update

        Returns:
            bool: True if update successful
        """
        with self.backend.transaction() as conn:
            # Build update query
            allowed_fields = ['email', 'is_active', 'is_verified', 'metadata']
            set_clause = []
            values = []
            accepted_fields: list[str] = []

            for field, value in updates.items():
                if field in allowed_fields:
                    set_clause.append(f"{field} = ?")
                    values.append(value if field != 'metadata' else json.dumps(value))
                    accepted_fields.append(field)

            if not set_clause:
                return False

            values.append(user_id)
            set_clause_sql = ", ".join(set_clause)
            query_template = "UPDATE users SET {set_clause_sql}, updated_at = CURRENT_TIMESTAMP WHERE id = ?"
            query = query_template.format_map(locals())  # nosec B608

            backend_name = (
                "postgres"
                if self.backend.backend_type == BackendType.POSTGRESQL
                else "sqlite"
            )
            if backend_name == "postgres":
                query = query.replace("?", "%s")
            visible_fields = tuple(
                field for field in accepted_fields if field in PROFILE_VISIBLE_USER_FIELDS
            )
            try:
                write_result = VersionedUserWriteGateway(
                    backend_name
                ).execute_update_sync(
                    self.backend,
                    conn,
                    user_id=user_id,
                    profile_visible_fields=visible_fields,
                    statement=query,
                    parameters=tuple(values),
                )
            except ProfileVersionNotFound:
                return False
            if visible_fields:
                success = bool(write_result.affected_user_ids)
            elif backend_name == "sqlite":
                change_result = self.backend.execute(
                    "SELECT changes() AS changes",
                    connection=conn,
                )
                changes = (
                    change_result.rows[0].get("changes", 0)
                    if change_result.rows
                    else 0
                )
                success = bool(changes)
            else:
                existing = self.backend.execute(
                    "SELECT 1 FROM users WHERE id = %s",
                    (user_id,),
                    connection=conn,
                )
                success = bool(existing.rows)

            if success:
                self._audit_log('user_updated', user_id, None, updates, connection=conn)
            return success

    def delete_user(self, user_id: int) -> bool:
        """
        Delete a user (soft delete by setting is_active = 0).

        Args:
            user_id: User ID to delete

        Returns:
            bool: True if deletion successful
        """
        return self.update_user(user_id, is_active=False)

    ########################################################################################################################
    # Role and Permission Management
    ########################################################################################################################

    def get_user_roles(self, user_id: int) -> list[str]:
        """
        Get all roles assigned to a user.

        Args:
            user_id: User ID

        Returns:
            List of role names
        """
        result = self.backend.execute(
            """
            SELECT r.name
            FROM roles r
            JOIN user_roles ur ON r.id = ur.role_id
            WHERE ur.user_id = ? AND (ur.expires_at IS NULL OR ur.expires_at > CURRENT_TIMESTAMP)
            """,
            (user_id,)
        )

        return [row['name'] for row in result.rows]

    def assign_role(self, user_id: int, role_name: str, granted_by: Optional[int] = None,
                   expires_at: Optional[datetime] = None) -> bool:
        """
        Assign a role to a user.

        Args:
            user_id: User ID
            role_name: Name of role to assign
            granted_by: ID of user granting the role
            expires_at: Optional expiration datetime

        Returns:
            bool: True if assignment successful
        """
        with self.backend.transaction() as conn:
            # Get role ID
            role_result = self.backend.execute(
                "SELECT id FROM roles WHERE name = ?", (role_name,),
                connection=conn,
            )

            if not role_result.rows:
                # Gracefully handle unknown roles per tests
                return False

            role_id = role_result.rows[0]['id']

            try:
                # Use REPLACE for SQLite, ON CONFLICT for PostgreSQL
                if self.backend.backend_type == BackendType.SQLITE:
                    query = """
                        INSERT OR REPLACE INTO user_roles (user_id, role_id, granted_by, expires_at)
                        VALUES (?, ?, ?, ?)
                    """
                else:
                    query = """
                        INSERT INTO user_roles (user_id, role_id, granted_by, expires_at)
                        VALUES (?, ?, ?, ?)
                        ON CONFLICT (user_id, role_id)
                        DO UPDATE SET granted_by = EXCLUDED.granted_by, expires_at = EXCLUDED.expires_at
                    """

                self.backend.execute(
                    query,
                    (user_id, role_id, granted_by, expires_at),
                    connection=conn,
                )

                self._audit_log('role_assigned', user_id, granted_by,
                              {'role': role_name, 'expires_at': expires_at.isoformat() if expires_at else None},
                              connection=conn)
                return True

            except _USERDB_NONCRITICAL_EXCEPTIONS as e:
                logger.bind(
                    operation="assign_role",
                    exception_type=type(e).__name__,
                ).error("Failed to assign role")
                return False

    def revoke_role(self, user_id: int, role_name: str, revoked_by: Optional[int] = None) -> bool:
        """
        Revoke a role from a user.

        Args:
            user_id: User ID
            role_name: Name of role to revoke
            revoked_by: ID of user revoking the role

        Returns:
            bool: True if revocation successful
        """
        with self.backend.transaction() as conn:
            # Get role ID
            role_result = self.backend.execute(
                "SELECT id FROM roles WHERE name = ?", (role_name,),
                connection=conn,
            )

            if not role_result.rows:
                return False

            role_id = role_result.rows[0]['id']

            result = self.backend.execute(
                "DELETE FROM user_roles WHERE user_id = ? AND role_id = ?",
                (user_id, role_id),
                connection=conn,
            )

            if result.rowcount > 0:
                self._audit_log('role_revoked', user_id, revoked_by, {'role': role_name}, connection=conn)
                return True
            return False

    def get_user_permissions(self, user_id: int) -> list[str]:
        """
        Get all permissions for a user (from roles and direct assignments).

        Args:
            user_id: User ID

        Returns:
            List of permission names
        """
        # Get permissions from roles
        role_perms = self.backend.execute(
            """
            SELECT DISTINCT p.name
            FROM permissions p
            JOIN role_permissions rp ON p.id = rp.permission_id
            JOIN user_roles ur ON rp.role_id = ur.role_id
            WHERE ur.user_id = ? AND (ur.expires_at IS NULL OR ur.expires_at > CURRENT_TIMESTAMP)
            """,
            (user_id,)
        )

        permissions = {row['name'] for row in role_perms.rows}

        # Get direct permissions (add granted, remove revoked)
        direct_perms = self.backend.execute(
            """
            SELECT p.name, up.granted
            FROM permissions p
            JOIN user_permissions up ON p.id = up.permission_id
            WHERE up.user_id = ? AND (up.expires_at IS NULL OR up.expires_at > CURRENT_TIMESTAMP)
            """,
            (user_id,)
        )

        for row in direct_perms.rows:
            if row['granted']:
                permissions.add(row['name'])
            else:
                permissions.discard(row['name'])

        return list(permissions)

    def has_permission(self, user_id: int, permission: str) -> bool:
        """Check if user has a specific permission."""
        permissions = self.get_user_permissions(user_id)
        return permission in permissions

    def has_role(self, user_id: int, role: str) -> bool:
        """Check if user has a specific role."""
        roles = self.get_user_roles(user_id)
        return role in roles

    ########################################################################################################################
    # Registration Code Management
    ########################################################################################################################

    def create_registration_code(self, created_by: Optional[int] = None,
                                expires_in_days: int = 7,
                                max_uses: int = 1,
                                role: str = 'user') -> str:
        """
        Create a new registration code.

        Args:
            created_by: User ID who created the code
            expires_in_days: Days until code expires
            max_uses: Maximum number of times code can be used
            role: Default role to assign to users who register with this code

        Returns:
            str: The generated registration code
        """
        code = secrets.token_urlsafe(32)
        expires_at = datetime.now(timezone.utc) + timedelta(days=expires_in_days)

        with self.backend.transaction() as conn:
            # Get role ID
            role_result = self.backend.execute(
                "SELECT id FROM roles WHERE name = ?", (role,),
                connection=conn,
            )
            role_id = role_result.rows[0]['id'] if role_result.rows else None

            self.backend.execute(
                """
                INSERT INTO registration_codes (code, created_by, expires_at, max_uses, role_id)
                VALUES (?, ?, ?, ?, ?)
                """,
                (code, created_by, expires_at, max_uses, role_id),
                connection=conn,
            )

            self._audit_log('registration_code_created', None, created_by,
                          {'code': code[:8] + '...', 'max_uses': max_uses, 'role': role},
                          connection=conn)

            logger.info(f"Created registration code {code[:8]}... with {max_uses} uses")
            return code

    def validate_registration_code(self, code: str) -> Optional[dict[str, Any]]:
        """
        Validate a registration code.

        Args:
            code: Registration code to validate

        Returns:
            Dict with code info if valid, None if invalid
        """
        result = self.backend.execute(
            """
            SELECT rc.*, r.name as role_name
            FROM registration_codes rc
            LEFT JOIN roles r ON rc.role_id = r.id
            WHERE rc.code = ?
            AND rc.is_active = ?
            AND rc.expires_at > CURRENT_TIMESTAMP
            AND rc.times_used < rc.max_uses
            """,
            (code, True if self.backend.backend_type == BackendType.POSTGRESQL else 1)
        )

        return result.rows[0] if result.rows else None

    def use_registration_code(self, code: str, user_id: int, ip_address: Optional[str] = None,
                             user_agent: Optional[str] = None) -> bool:
        """
        Mark a registration code as used.

        Args:
            code: Registration code
            user_id: User ID who used the code
            ip_address: IP address of registration
            user_agent: User agent string

        Returns:
            bool: True if code was successfully used
        """
        with self.backend.transaction() as conn:
            # Get code info
            code_result = self.backend.execute(
                """
                SELECT id, times_used FROM registration_codes
                WHERE code = ? AND is_active = ?
                """,
                (code, True if self.backend.backend_type == BackendType.POSTGRESQL else 1),
                connection=conn,
            )

            if not code_result.rows:
                return False

            code_id = code_result.rows[0]['id']

            active_value = True if self.backend.backend_type == BackendType.POSTGRESQL else 1
            update_params = (code_id, active_value)

            if self.backend.backend_type == BackendType.POSTGRESQL:
                update_result = self.backend.execute(
                    """
                    UPDATE registration_codes
                    SET times_used = times_used + 1,
                        is_active = CASE WHEN times_used + 1 >= max_uses THEN FALSE ELSE is_active END
                    WHERE id = ?
                      AND is_active = ?
                      AND times_used < max_uses
                      AND expires_at > CURRENT_TIMESTAMP
                    RETURNING times_used, max_uses, is_active
                    """,
                    update_params,
                    connection=conn,
                )
                if not update_result.rows:
                    return False
                new_times_used = update_result.rows[0]['times_used']
                max_uses = update_result.rows[0]['max_uses']
            else:
                self.backend.execute(
                    """
                    UPDATE registration_codes
                    SET times_used = times_used + 1
                    WHERE id = ?
                      AND is_active = ?
                      AND times_used < max_uses
                      AND expires_at > CURRENT_TIMESTAMP
                    """,
                    update_params,
                    connection=conn,
                )
                change_result = self.backend.execute(
                    "SELECT changes() AS changes",
                    connection=conn,
                )
                if not change_result.rows or change_result.rows[0].get("changes", 0) == 0:
                    return False
                fetch_result = self.backend.execute(
                    "SELECT times_used, max_uses FROM registration_codes WHERE id = ?",
                    (code_id,),
                    connection=conn,
                )
                if not fetch_result.rows:
                    return False
                new_times_used = fetch_result.rows[0]['times_used']
                max_uses = fetch_result.rows[0]['max_uses']

                if new_times_used >= max_uses:
                    self.backend.execute(
                        "UPDATE registration_codes SET is_active = 0 WHERE id = ?",
                        (code_id,),
                        connection=conn,
                    )

            if new_times_used >= max_uses and self.backend.backend_type == BackendType.POSTGRESQL:
                self.backend.execute(
                    "UPDATE registration_codes SET is_active = FALSE WHERE id = ?",
                    (code_id,),
                    connection=conn,
                )

            # Record usage
            self.backend.execute(
                """
                INSERT INTO registration_code_usage (code_id, user_id, ip_address, user_agent)
                VALUES (?, ?, ?, ?)
                """,
                (code_id, user_id, ip_address, user_agent),
                connection=conn,
            )

            self._audit_log('registration_code_used', user_id, None,
                          {'code': code[:8] + '...', 'ip': ip_address},
                          connection=conn)

            return True

    ########################################################################################################################
    # Authentication Methods
    ########################################################################################################################

    def record_login(self, user_id: int, ip_address: Optional[str] = None,
                    user_agent: Optional[str] = None) -> bool:
        """Record a successful login."""
        with self.backend.transaction() as conn:
            backend_name = (
                "postgres"
                if self.backend.backend_type == BackendType.POSTGRESQL
                else "sqlite"
            )
            placeholder = "%s" if backend_name == "postgres" else "?"
            VersionedUserWriteGateway(backend_name).execute_update_sync(
                self.backend,
                conn,
                user_id=user_id,
                profile_visible_fields=("last_login",),
                statement=(
                    "UPDATE users SET last_login = CURRENT_TIMESTAMP, "
                    "failed_login_attempts = 0, locked_until = NULL "
                    # Placeholder is closed over the detected backend; values stay bound.
                    f"WHERE id = {placeholder}"  # nosec B608
                ),
                parameters=(user_id,),
            )

            self._audit_log('login_success', user_id, None,
                          {'ip': ip_address, 'user_agent': user_agent},
                          connection=conn)
            return True

    def record_failed_login(self, username: str, ip_address: Optional[str] = None) -> int:
        """Record a failed login attempt."""
        with self.backend.transaction() as conn:
            # Different syntax for SQLite vs PostgreSQL
            if self.backend.backend_type == BackendType.SQLITE:
                self.backend.execute(
                    """
                    UPDATE users
                    SET failed_login_attempts = failed_login_attempts + 1
                    WHERE username = ?
                    """,
                    (username,),
                    connection=conn,
                )

                result = self.backend.execute(
                    "SELECT failed_login_attempts, id FROM users WHERE username = ?",
                    (username,),
                    connection=conn,
                )
            else:
                result = self.backend.execute(
                    """
                    UPDATE users
                    SET failed_login_attempts = failed_login_attempts + 1
                    WHERE username = ?
                    RETURNING failed_login_attempts, id
                    """,
                    (username,),
                    connection=conn,
                )

            if result.rows:
                attempts = result.rows[0]['failed_login_attempts']
                user_id = result.rows[0]['id']

                # Lock account after 5 attempts
                if attempts >= 5:
                    lock_until = datetime.now(timezone.utc) + timedelta(minutes=15)
                    self.backend.execute(
                        "UPDATE users SET locked_until = ? WHERE id = ?",
                        (lock_until, user_id),
                        connection=conn,
                    )

                self._audit_log('login_failed', None, None,
                              {'username': username, 'ip': ip_address, 'attempts': attempts},
                              connection=conn)

                return attempts
            return 0

    def is_account_locked(self, user_id: int) -> bool:
        """Check if user account is locked."""
        result = self.backend.execute(
            """
            SELECT locked_until FROM users
            WHERE id = ? AND locked_until > CURRENT_TIMESTAMP
            """,
            (user_id,)
        )

        return len(result.rows) > 0

    ########################################################################################################################
    # Helper Methods
    ########################################################################################################################

    def _audit_log(
        self,
        event_type: str,
        user_id: Optional[int],
        target_user_id: Optional[int],
        details: Optional[dict[str, Any]] = None,
        connection: Optional[Any] = None,
    ):
        """Create an audit log entry."""
        try:
            self.backend.execute(
                """
                INSERT INTO auth_audit_log (event_type, user_id, target_user_id, details)
                VALUES (?, ?, ?, ?)
                """,
                (event_type, user_id, target_user_id,
                 json.dumps(details) if details else None),
                connection=connection,
            )
        except _USERDB_NONCRITICAL_EXCEPTIONS as e:
            logger.bind(
                operation="create_audit_log",
                exception_type=type(e).__name__,
            ).error("Failed to create audit log")

    # ------------------------------------------------------------------------------------------------------------------
    # Internal helpers for schema/bootstrap
    # ------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def _split_sql_statements(sql: str) -> list[str]:
        return split_sql_statements(sql)

    def _apply_schema_statements(self, statements: list[str]) -> None:
        if not statements:
            return

        with self.backend.transaction() as conn:
            if self.backend.backend_type == BackendType.SQLITE:
                for stmt in statements:
                    conn.execute(stmt)
            else:
                cursor = conn.cursor()
                try:
                    for stmt in statements:
                        cursor.execute(stmt)
                finally:
                    cursor.close()

    def _ensure_sqlite_profile_version_schema(self) -> None:
        """Own canonical anchor remediation during raw SQLite bootstrap."""
        pool = self.backend.get_pool()
        try:
            connection = pool.get_connection()
            remediate_sqlite_profile_version_schema(connection)
        except BaseException as exc:
            if sqlite_profile_version_connection_invalid(exc):
                with contextlib.suppress(Exception):
                    pool.clear_thread_local_connection()
            if not isinstance(exc, Exception):
                raise
            logger.bind(
                operation="profile_version_schema_readiness",
                backend="sqlite",
                exception_type=type(exc).__name__,
            ).error("Required users.profile_version schema validation failed")
            raise UserDatabaseError(
                "Required users.profile_version schema validation failed"
            ) from None

    def _default_schema_statements(self) -> list[str]:
        if self.backend.backend_type == BackendType.POSTGRESQL:
            return self._default_schema_statements_postgres()
        return self._default_schema_statements_sqlite()

    @staticmethod
    def _default_schema_statements_sqlite() -> list[str]:
        return [
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                uuid TEXT UNIQUE NOT NULL DEFAULT (lower(hex(randomblob(16)))),
                username TEXT UNIQUE NOT NULL CHECK (length(username) <= 255),
                email TEXT UNIQUE NOT NULL CHECK (length(email) <= 255),
                password_hash TEXT NOT NULL,
                metadata TEXT,
                is_active INTEGER NOT NULL DEFAULT 1,
                is_verified INTEGER NOT NULL DEFAULT 0,
                is_superuser INTEGER NOT NULL DEFAULT 0,
                email_verified INTEGER NOT NULL DEFAULT 0,
                two_factor_enabled INTEGER NOT NULL DEFAULT 0,
                role TEXT NOT NULL DEFAULT 'user',
                storage_quota_mb INTEGER NOT NULL DEFAULT 5120,
                storage_used_mb REAL NOT NULL DEFAULT 0,
                failed_login_attempts INTEGER NOT NULL DEFAULT 0,
                locked_until TIMESTAMP,
                last_login TIMESTAMP,
                email_verified_at TIMESTAMP,
                two_factor_secret TEXT,
                totp_secret TEXT,
                backup_codes TEXT,
                created_by INTEGER REFERENCES users(id) ON DELETE SET NULL,
                password_changed_at TIMESTAMP,
                profile_version TEXT NOT NULL DEFAULT (STRFTIME('%Y-%m-%dT%H:%M:%f000Z', 'now')),
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS organizations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                uuid TEXT UNIQUE,
                name TEXT UNIQUE NOT NULL,
                slug TEXT UNIQUE,
                owner_user_id INTEGER,
                is_active INTEGER DEFAULT 1,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (owner_user_id) REFERENCES users(id) ON DELETE SET NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS teams (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                org_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                slug TEXT,
                description TEXT,
                is_active INTEGER DEFAULT 1,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE (org_id, name),
                FOREIGN KEY (org_id) REFERENCES organizations(id) ON DELETE CASCADE
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS org_members (
                org_id INTEGER NOT NULL,
                user_id INTEGER NOT NULL,
                role TEXT DEFAULT 'member',
                status TEXT DEFAULT 'active',
                added_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (org_id, user_id),
                FOREIGN KEY (org_id) REFERENCES organizations(id) ON DELETE CASCADE,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS team_members (
                team_id INTEGER NOT NULL,
                user_id INTEGER NOT NULL,
                role TEXT DEFAULT 'member',
                status TEXT DEFAULT 'active',
                added_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (team_id, user_id),
                FOREIGN KEY (team_id) REFERENCES teams(id) ON DELETE CASCADE,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS user_config_overrides (
                user_id INTEGER NOT NULL,
                key TEXT NOT NULL,
                value_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                created_by INTEGER,
                updated_by INTEGER,
                PRIMARY KEY (user_id, key),
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS org_config_overrides (
                org_id INTEGER NOT NULL,
                key TEXT NOT NULL,
                value_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                created_by INTEGER,
                updated_by INTEGER,
                PRIMARY KEY (org_id, key),
                FOREIGN KEY (org_id) REFERENCES organizations(id) ON DELETE CASCADE
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS team_config_overrides (
                team_id INTEGER NOT NULL,
                key TEXT NOT NULL,
                value_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                created_by INTEGER,
                updated_by INTEGER,
                PRIMARY KEY (team_id, key),
                FOREIGN KEY (team_id) REFERENCES teams(id) ON DELETE CASCADE
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_users_username ON users(username)",
            "CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)",
            """
            CREATE TABLE IF NOT EXISTS roles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                description TEXT,
                is_system INTEGER NOT NULL DEFAULT 0
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS permissions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                description TEXT,
                category TEXT
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS role_permissions (
                role_id INTEGER NOT NULL,
                permission_id INTEGER NOT NULL,
                granted_by INTEGER,
                granted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (role_id, permission_id),
                FOREIGN KEY (role_id) REFERENCES roles(id) ON DELETE CASCADE,
                FOREIGN KEY (permission_id) REFERENCES permissions(id) ON DELETE CASCADE
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS user_roles (
                user_id INTEGER NOT NULL,
                role_id INTEGER NOT NULL,
                granted_by INTEGER,
                expires_at TIMESTAMP,
                PRIMARY KEY (user_id, role_id),
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                FOREIGN KEY (role_id) REFERENCES roles(id) ON DELETE CASCADE
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS user_permissions (
                user_id INTEGER NOT NULL,
                permission_id INTEGER NOT NULL,
                granted INTEGER NOT NULL DEFAULT 1,
                granted_by INTEGER,
                expires_at TIMESTAMP,
                PRIMARY KEY (user_id, permission_id),
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                FOREIGN KEY (permission_id) REFERENCES permissions(id) ON DELETE CASCADE
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS registration_codes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                code TEXT UNIQUE NOT NULL,
                created_by INTEGER,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP,
                max_uses INTEGER NOT NULL DEFAULT 1,
                times_used INTEGER NOT NULL DEFAULT 0,
                is_active INTEGER NOT NULL DEFAULT 1,
                role_id INTEGER,
                FOREIGN KEY (created_by) REFERENCES users(id) ON DELETE SET NULL,
                FOREIGN KEY (role_id) REFERENCES roles(id) ON DELETE SET NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS registration_code_usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                code_id INTEGER NOT NULL,
                user_id INTEGER NOT NULL,
                ip_address TEXT,
                user_agent TEXT,
                used_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (code_id) REFERENCES registration_codes(id) ON DELETE CASCADE,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS auth_audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL,
                user_id INTEGER,
                target_user_id INTEGER,
                details TEXT,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL,
                FOREIGN KEY (target_user_id) REFERENCES users(id) ON DELETE SET NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS audio_studio_media_tickets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                token_hash TEXT UNIQUE NOT NULL,
                user_id INTEGER NOT NULL,
                project_id TEXT NOT NULL,
                artifact_id TEXT NOT NULL,
                purpose TEXT NOT NULL CHECK (purpose IN ('playback', 'download')),
                expires_at TEXT NOT NULL,
                consumed_at TEXT,
                revoked_at TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                created_by_auth_mode TEXT,
                last_redeemed_at TEXT
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_audio_studio_media_tickets_hash ON audio_studio_media_tickets(token_hash)",
            "CREATE INDEX IF NOT EXISTS idx_audio_studio_media_tickets_expiry ON audio_studio_media_tickets(expires_at)",
            "CREATE INDEX IF NOT EXISTS idx_audio_studio_media_tickets_artifact ON audio_studio_media_tickets(user_id, project_id, artifact_id)",
        ]

    @staticmethod
    def _default_schema_statements_postgres() -> list[str]:
        return [
            """
            CREATE TABLE IF NOT EXISTS public.users (
                id BIGSERIAL PRIMARY KEY,
                uuid UUID UNIQUE NOT NULL DEFAULT gen_random_uuid(),
                username VARCHAR(255) UNIQUE NOT NULL,
                email VARCHAR(255) UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                metadata JSONB,
                is_active BOOLEAN NOT NULL DEFAULT TRUE,
                is_verified BOOLEAN NOT NULL DEFAULT FALSE,
                is_superuser BOOLEAN NOT NULL DEFAULT FALSE,
                email_verified BOOLEAN NOT NULL DEFAULT FALSE,
                two_factor_enabled BOOLEAN NOT NULL DEFAULT FALSE,
                role TEXT NOT NULL DEFAULT 'user',
                storage_quota_mb INTEGER NOT NULL DEFAULT 5120,
                storage_used_mb DOUBLE PRECISION NOT NULL DEFAULT 0,
                failed_login_attempts INTEGER NOT NULL DEFAULT 0,
                locked_until TIMESTAMPTZ,
                last_login TIMESTAMPTZ,
                email_verified_at TIMESTAMPTZ,
                two_factor_secret TEXT,
                totp_secret TEXT,
                backup_codes TEXT,
                created_by BIGINT REFERENCES public.users(id) ON DELETE SET NULL,
                password_changed_at TIMESTAMPTZ,
                profile_version TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS public.organizations (
                id BIGSERIAL PRIMARY KEY,
                uuid TEXT UNIQUE,
                name TEXT UNIQUE NOT NULL,
                slug TEXT UNIQUE,
                owner_user_id BIGINT REFERENCES public.users(id) ON DELETE SET NULL,
                is_active BOOLEAN DEFAULT TRUE,
                metadata JSONB,
                created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS public.teams (
                id BIGSERIAL PRIMARY KEY,
                org_id BIGINT NOT NULL REFERENCES public.organizations(id) ON DELETE CASCADE,
                name TEXT NOT NULL,
                slug TEXT,
                description TEXT,
                is_active BOOLEAN DEFAULT TRUE,
                metadata JSONB,
                created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE (org_id, name)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS public.org_members (
                org_id BIGINT NOT NULL,
                user_id BIGINT NOT NULL,
                role TEXT DEFAULT 'member',
                status TEXT DEFAULT 'active',
                added_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (org_id, user_id),
                FOREIGN KEY (org_id) REFERENCES public.organizations(id) ON DELETE CASCADE,
                FOREIGN KEY (user_id) REFERENCES public.users(id) ON DELETE CASCADE
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS public.team_members (
                team_id BIGINT NOT NULL,
                user_id BIGINT NOT NULL,
                role TEXT DEFAULT 'member',
                status TEXT DEFAULT 'active',
                added_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (team_id, user_id),
                FOREIGN KEY (team_id) REFERENCES public.teams(id) ON DELETE CASCADE,
                FOREIGN KEY (user_id) REFERENCES public.users(id) ON DELETE CASCADE
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS public.user_config_overrides (
                user_id BIGINT NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
                key TEXT NOT NULL,
                value_json TEXT,
                created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                created_by BIGINT,
                updated_by BIGINT,
                PRIMARY KEY (user_id, key)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS public.org_config_overrides (
                org_id BIGINT NOT NULL REFERENCES public.organizations(id) ON DELETE CASCADE,
                key TEXT NOT NULL,
                value_json TEXT,
                created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                created_by BIGINT,
                updated_by BIGINT,
                PRIMARY KEY (org_id, key)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS public.team_config_overrides (
                team_id BIGINT NOT NULL REFERENCES public.teams(id) ON DELETE CASCADE,
                key TEXT NOT NULL,
                value_json TEXT,
                created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                created_by BIGINT,
                updated_by BIGINT,
                PRIMARY KEY (team_id, key)
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_users_username ON public.users (username)",
            "CREATE INDEX IF NOT EXISTS idx_users_email ON public.users (email)",
            """
            CREATE TABLE IF NOT EXISTS roles (
                id BIGSERIAL PRIMARY KEY,
                name TEXT UNIQUE NOT NULL,
                description TEXT,
                is_system BOOLEAN NOT NULL DEFAULT FALSE
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS permissions (
                id BIGSERIAL PRIMARY KEY,
                name TEXT UNIQUE NOT NULL,
                description TEXT,
                category TEXT
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS role_permissions (
                role_id BIGINT NOT NULL REFERENCES roles(id) ON DELETE CASCADE,
                permission_id BIGINT NOT NULL REFERENCES permissions(id) ON DELETE CASCADE,
                granted_by BIGINT,
                granted_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (role_id, permission_id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS user_roles (
                user_id BIGINT NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
                role_id BIGINT NOT NULL REFERENCES roles(id) ON DELETE CASCADE,
                granted_by BIGINT,
                expires_at TIMESTAMPTZ,
                PRIMARY KEY (user_id, role_id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS user_permissions (
                user_id BIGINT NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
                permission_id BIGINT NOT NULL REFERENCES permissions(id) ON DELETE CASCADE,
                granted BOOLEAN NOT NULL DEFAULT TRUE,
                granted_by BIGINT,
                expires_at TIMESTAMPTZ,
                PRIMARY KEY (user_id, permission_id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS registration_codes (
                id BIGSERIAL PRIMARY KEY,
                code TEXT UNIQUE NOT NULL,
                created_by BIGINT REFERENCES public.users(id) ON DELETE SET NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMPTZ,
                max_uses INTEGER NOT NULL DEFAULT 1,
                times_used INTEGER NOT NULL DEFAULT 0,
                is_active BOOLEAN NOT NULL DEFAULT TRUE,
                role_id BIGINT REFERENCES roles(id) ON DELETE SET NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS registration_code_usage (
                id BIGSERIAL PRIMARY KEY,
                code_id BIGINT NOT NULL REFERENCES registration_codes(id) ON DELETE CASCADE,
                user_id BIGINT NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
                ip_address TEXT,
                user_agent TEXT,
                used_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS auth_audit_log (
                id BIGSERIAL PRIMARY KEY,
                event_type TEXT NOT NULL,
                user_id BIGINT REFERENCES public.users(id) ON DELETE SET NULL,
                target_user_id BIGINT REFERENCES public.users(id) ON DELETE SET NULL,
                details JSONB,
                created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS audio_studio_media_tickets (
                id BIGSERIAL PRIMARY KEY,
                token_hash TEXT UNIQUE NOT NULL,
                user_id BIGINT NOT NULL,
                project_id TEXT NOT NULL,
                artifact_id TEXT NOT NULL,
                purpose TEXT NOT NULL CHECK (purpose IN ('playback', 'download')),
                expires_at TIMESTAMPTZ NOT NULL,
                consumed_at TIMESTAMPTZ,
                revoked_at TIMESTAMPTZ,
                created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                created_by_auth_mode TEXT,
                last_redeemed_at TIMESTAMPTZ
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_audio_studio_media_tickets_hash ON audio_studio_media_tickets(token_hash)",
            "CREATE INDEX IF NOT EXISTS idx_audio_studio_media_tickets_expiry ON audio_studio_media_tickets(expires_at)",
            "CREATE INDEX IF NOT EXISTS idx_audio_studio_media_tickets_artifact ON audio_studio_media_tickets(user_id, project_id, artifact_id)",
        ]

    def _seed_default_data(self) -> None:
        required_roles = ("admin", "user", "viewer")
        required_permissions = (
            "media.read",
            "media.create",
            "media.delete",
            "sql.read",
            "sql.target:media_db",
            "system.configure",
            "users.manage_roles",
        )
        required_role_permissions = {
            "user": ("media.read", "media.create", "sql.read", "sql.target:media_db"),
            "viewer": ("media.read",),
            "admin": required_permissions,
        }

        # Seed roles
        default_roles = [
            ("admin", "Administrator", True),
            ("user", "Standard User", True),
            ("viewer", "Read-only User", True),
            ("custom", "Custom role (no default permissions)", False),
        ]

        if self.backend.backend_type == BackendType.POSTGRESQL:
            role_sql = (
                "INSERT INTO roles (name, description, is_system) VALUES (%s, %s, %s) "
                "ON CONFLICT (name) DO NOTHING"
            )
            perm_sql = (
                "INSERT INTO permissions (name, description, category) VALUES (%s, %s, %s) "
                "ON CONFLICT (name) DO NOTHING"
            )
            rp_sql = (
                "INSERT INTO role_permissions (role_id, permission_id) VALUES (%s, %s) "
                "ON CONFLICT DO NOTHING"
            )
            sel_role_id = "SELECT id FROM roles WHERE name = %s"
            sel_perm_id = "SELECT id FROM permissions WHERE name = %s"
            sel_role_perm = "SELECT 1 FROM role_permissions WHERE role_id = %s AND permission_id = %s"
        else:
            role_sql = "INSERT OR IGNORE INTO roles (name, description, is_system) VALUES (?, ?, ?)"
            perm_sql = "INSERT OR IGNORE INTO permissions (name, description, category) VALUES (?, ?, ?)"
            rp_sql = "INSERT OR IGNORE INTO role_permissions (role_id, permission_id) VALUES (?, ?)"
            sel_role_id = "SELECT id FROM roles WHERE name = ?"
            sel_perm_id = "SELECT id FROM permissions WHERE name = ?"
            sel_role_perm = "SELECT 1 FROM role_permissions WHERE role_id = ? AND permission_id = ?"

        for name, description, is_system in default_roles:
            try:
                self.backend.execute(role_sql, (name, description, is_system))
            except Exception as exc:  # noqa: BLE001
                logger.debug("Skipping role seed for {}: {}", name, exc)

        # Seed baseline permissions
        default_perms = [
            ("media.read", "Read media", "media"),
            ("media.create", "Create media", "media"),
            ("media.delete", "Delete media", "media"),
            ("sql.read", "Run read-only SQL retrieval", "sql"),
            ("sql.target:media_db", "Allow SQL retrieval against media_db target", "sql"),
            ("system.configure", "Configure system", "system"),
            ("users.manage_roles", "Manage user roles", "users"),
        ]
        for name, desc, cat in default_perms:
            try:
                self.backend.execute(perm_sql, (name, desc, cat))
            except Exception as exc:  # noqa: BLE001
                logger.debug("Skipping permission seed for {}: {}", name, exc)

        # Map permissions to roles
        def _get_id(query: str, value: str) -> Optional[int]:
            res = self.backend.execute(query, (value,))
            return res.rows[0]['id'] if res.rows else None

        try:
            role_ids = {name: _get_id(sel_role_id, name) for name in required_roles}
            perm_ids = {name: _get_id(sel_perm_id, name) for name in required_permissions}

            missing_roles = [name for name, role_id in role_ids.items() if role_id is None]
            missing_permissions = [name for name, perm_id in perm_ids.items() if perm_id is None]
            if missing_roles or missing_permissions:
                raise UserDatabaseError(
                    "Required RBAC seed state missing: "
                    f"roles={missing_roles}, permissions={missing_permissions}"
                )

            for role_name, permission_names in required_role_permissions.items():
                role_id = role_ids[role_name]
                for permission_name in permission_names:
                    perm_id = perm_ids[permission_name]
                    if role_id and perm_id:
                        with contextlib.suppress(_USERDB_NONCRITICAL_EXCEPTIONS):
                            self.backend.execute(rp_sql, (role_id, perm_id))

            missing_links: list[str] = []
            for role_name, permission_names in required_role_permissions.items():
                role_id = role_ids[role_name]
                for permission_name in permission_names:
                    perm_id = perm_ids[permission_name]
                    result = self.backend.execute(sel_role_perm, (role_id, perm_id))
                    if not result.rows:
                        missing_links.append(f"{role_name}:{permission_name}")

            if missing_links:
                raise UserDatabaseError(
                    f"Required RBAC seed links missing: {missing_links}"
                )
        except UserDatabaseError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.bind(
                operation="rbac_seed_verification",
                exception_type=type(exc).__name__,
            ).error("Required RBAC seed verification failed")
            raise UserDatabaseError(
                "Required RBAC seed verification failed"
            ) from None


#
# End of UserDatabase_v2.py
########################################################################################################################
    def _ensure_core_columns(self) -> None:
        """Ensure essential columns and defaults exist across backends."""
        user_step = "users table inspection"
        try:
            if self.backend.backend_type == BackendType.SQLITE:
                user_step = "users table inspection"
                result = self.backend.execute("PRAGMA table_info(users)")
                column_names = {row['name'] if isinstance(row, dict) else row[1] for row in result.rows}
                if 'uuid' not in column_names:
                    user_step = "users.uuid"
                    self.backend.execute("ALTER TABLE users ADD COLUMN uuid TEXT")
                if 'metadata' not in column_names:
                    user_step = "users.metadata"
                    self.backend.execute("ALTER TABLE users ADD COLUMN metadata TEXT")
                if 'failed_login_attempts' not in column_names:
                    user_step = "users.failed_login_attempts"
                    self.backend.execute("ALTER TABLE users ADD COLUMN failed_login_attempts INTEGER DEFAULT 0")
                if 'locked_until' not in column_names:
                    user_step = "users.locked_until"
                    self.backend.execute("ALTER TABLE users ADD COLUMN locked_until TIMESTAMP")
                if 'is_superuser' not in column_names:
                    user_step = "users.is_superuser"
                    self.backend.execute("ALTER TABLE users ADD COLUMN is_superuser INTEGER DEFAULT 0")
                if 'profile_version' not in column_names:
                    raise UserDatabaseError(
                        "Required users.profile_version schema validation failed"
                    )
                user_step = "profile candidate source tables"
                with self.backend.transaction() as conn:
                    for statement in self._profile_candidate_table_statements_sqlite():
                        self.backend.execute(statement, connection=conn)
                    self._validate_profile_candidate_tables_sqlite(
                        connection=conn,
                    )
                user_step = "users.uuid backfill"
                missing_uuid_rows = self.backend.execute(
                    "SELECT id FROM users WHERE uuid IS NULL OR uuid = ''"
                ).rows
                gateway = VersionedUserWriteGateway("sqlite")
                maintenance_executor = _guard_authnz_sync_backend(self.backend)
                with self.backend.transaction() as conn:
                    for row in missing_uuid_rows:
                        user_id = int(row["id"] if isinstance(row, dict) else row[0])
                        gateway.execute_update_sync(
                            maintenance_executor,
                            conn,
                            user_id=user_id,
                            profile_visible_fields=("uuid",),
                            statement="UPDATE users SET uuid = ? WHERE id = ?",
                            parameters=(str(uuid4()), user_id),
                        )
                user_step = "users.uuid unique index"
                self.backend.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_users_uuid ON users(uuid)")
                user_step = "users.failed_login_attempts backfill"
                self.backend.execute(
                    "UPDATE users SET failed_login_attempts = 0 WHERE failed_login_attempts IS NULL"
                )
                user_step = "users.locked_until backfill"
                self.backend.execute(
                    "UPDATE users SET locked_until = NULL WHERE locked_until IS NULL"
                )
            elif self.backend.backend_type == BackendType.POSTGRESQL:
                user_step = "users.uuid"
                self.backend.execute("ALTER TABLE public.users ADD COLUMN IF NOT EXISTS uuid UUID")
                user_step = "users.metadata"
                self.backend.execute("ALTER TABLE public.users ADD COLUMN IF NOT EXISTS metadata JSONB")
                user_step = "users.failed_login_attempts"
                self.backend.execute("ALTER TABLE public.users ADD COLUMN IF NOT EXISTS failed_login_attempts INTEGER DEFAULT 0")
                user_step = "users.locked_until"
                self.backend.execute("ALTER TABLE public.users ADD COLUMN IF NOT EXISTS locked_until TIMESTAMPTZ")
                user_step = "users.is_superuser"
                self.backend.execute("ALTER TABLE public.users ADD COLUMN IF NOT EXISTS is_superuser BOOLEAN DEFAULT FALSE")
                user_step = "profile candidate source tables"
                with self.backend.transaction() as conn:
                    for statement in self._profile_candidate_table_statements_postgres():
                        self.backend.execute(statement, connection=conn)
                    self._validate_profile_candidate_tables_postgres(
                        connection=conn,
                    )
                user_step = "users.uuid backfill"
                missing_uuid_rows = self.backend.execute(
                    "SELECT id FROM public.users WHERE uuid IS NULL"
                ).rows
                gateway = VersionedUserWriteGateway("postgres")
                maintenance_executor = _guard_authnz_sync_backend(self.backend)
                with self.backend.transaction() as conn:
                    for row in missing_uuid_rows:
                        user_id = int(row["id"] if isinstance(row, dict) else row[0])
                        gateway.execute_update_sync(
                            maintenance_executor,
                            conn,
                            user_id=user_id,
                            profile_visible_fields=("uuid",),
                            statement="UPDATE public.users SET uuid = %s WHERE id = %s",
                            parameters=(str(uuid4()), user_id),
                        )
                user_step = "users.uuid not null"
                self.backend.execute("ALTER TABLE public.users ALTER COLUMN uuid SET NOT NULL")
                try:
                    user_step = "users.uuid default"
                    self.backend.execute("ALTER TABLE public.users ALTER COLUMN uuid SET DEFAULT gen_random_uuid()")
                except _USERDB_NONCRITICAL_EXCEPTIONS:
                    user_step = "users.uuid text default"
                    self.backend.execute("ALTER TABLE public.users ALTER COLUMN uuid SET DEFAULT (gen_random_uuid()::text)")
                user_step = "users.failed_login_attempts backfill"
                self.backend.execute(
                    "UPDATE public.users SET failed_login_attempts = 0 WHERE failed_login_attempts IS NULL"
                )
                user_step = "users.locked_until backfill"
                self.backend.execute(
                    "UPDATE public.users SET locked_until = NULL WHERE locked_until IS NULL"
                )
        except Exception as exc:  # noqa: BLE001
            logger.bind(
                operation="user_schema_normalization",
                step=user_step,
                exception_type=type(exc).__name__,
            ).error("Required user schema normalization failed")
            raise UserDatabaseError(
                f"Required user schema normalization failed at {user_step}"
            ) from None

        registration_step = "registration_codes table inspection"
        try:
            if self.backend.backend_type == BackendType.SQLITE:
                registration_step = "registration_codes table inspection"
                reg_info = self.backend.execute("PRAGMA table_info(registration_codes)")
                reg_columns = {row['name'] if isinstance(row, dict) else row[1] for row in reg_info.rows}
                if 'role_id' not in reg_columns:
                    registration_step = "registration_codes.role_id"
                    self.backend.execute("ALTER TABLE registration_codes ADD COLUMN role_id INTEGER REFERENCES roles(id)")
            elif self.backend.backend_type == BackendType.POSTGRESQL:
                registration_step = "registration_codes.role_id"
                self.backend.execute(
                    """
                    ALTER TABLE registration_codes
                    ADD COLUMN IF NOT EXISTS role_id BIGINT REFERENCES roles(id) ON DELETE SET NULL
                    """
                )
        except Exception as exc:  # noqa: BLE001
            logger.bind(
                operation="registration_schema_normalization",
                step=registration_step,
                exception_type=type(exc).__name__,
            ).error("Required registration_codes normalization failed")
            raise UserDatabaseError(
                "Required registration_codes normalization failed at "
                f"{registration_step}"
            ) from None

    @staticmethod
    def _profile_candidate_table_statements_sqlite() -> tuple[str, ...]:
        statements = UserDatabase._default_schema_statements_sqlite()
        return tuple(
            statement
            for table_name in PROFILE_CANDIDATE_TABLES
            for statement in statements
            if f"CREATE TABLE IF NOT EXISTS {table_name} " in statement
        )

    @staticmethod
    def _profile_candidate_table_statements_postgres() -> tuple[str, ...]:
        statements = UserDatabase._default_schema_statements_postgres()
        return tuple(
            statement
            for table_name in PROFILE_CANDIDATE_TABLES
            for statement in statements
            if f"CREATE TABLE IF NOT EXISTS public.{table_name} " in statement
        )

    def _validate_profile_candidate_tables_sqlite(
        self,
        *,
        connection: Any | None = None,
    ) -> None:
        def execute(query: str, params: Any = None) -> Any:
            if connection is None:
                return self.backend.execute(query, params)
            return self.backend.execute(query, params, connection=connection)

        columns_by_table: dict[str, dict[str, dict[str, Any]]] = {}
        primary_keys_by_table: dict[str, tuple[str, ...]] = {}
        unique_keys_by_table: dict[str, set[tuple[str, ...]]] = {}
        foreign_keys_by_table: dict[
            str,
            set[tuple[str, str, str, str, str]],
        ] = {}
        for table_name in PROFILE_CANDIDATE_TABLES:
            table_info = execute(
                f'PRAGMA table_info("{table_name}")'  # nosec B608
            ).rows
            columns_by_table[table_name] = {
                str(row["name"] if isinstance(row, dict) else row[1]): {
                    "data_type": row["type"] if isinstance(row, dict) else row[2],
                    "not_null": bool(
                        row["notnull"] if isinstance(row, dict) else row[3]
                    )
                    or int(row["pk"] if isinstance(row, dict) else row[5]) > 0,
                    "default": (
                        row["dflt_value"] if isinstance(row, dict) else row[4]
                    ),
                }
                for row in table_info
            }
            primary_keys_by_table[table_name] = tuple(
                str(row["name"] if isinstance(row, dict) else row[1])
                for row in sorted(
                    table_info,
                    key=lambda row: int(
                        row["pk"] if isinstance(row, dict) else row[5]
                    ),
                )
                if int(row["pk"] if isinstance(row, dict) else row[5]) > 0
            )
            unique_rows = execute(
                "SELECT index_list.name AS index_name, "
                "index_info.name AS column_name, index_info.seqno "
                "FROM pragma_index_list(?) AS index_list "
                "JOIN pragma_index_info(index_list.name) AS index_info "
                "WHERE index_list.[unique] = 1 AND index_list.origin <> 'pk' "
                "ORDER BY index_list.name, index_info.seqno",
                (table_name,),
            ).rows
            unique_columns: dict[str, list[tuple[int, str]]] = {}
            for row in unique_rows:
                index_name = str(
                    row["index_name"] if isinstance(row, dict) else row[0]
                )
                unique_columns.setdefault(index_name, []).append(
                    (
                        int(row["seqno"] if isinstance(row, dict) else row[2]),
                        str(
                            row["column_name"]
                            if isinstance(row, dict)
                            else row[1]
                        ),
                    )
                )
            unique_keys_by_table[table_name] = {
                tuple(column for _position, column in sorted(index_columns))
                for index_columns in unique_columns.values()
            }
            foreign_keys = execute(
                f'PRAGMA foreign_key_list("{table_name}")'  # nosec B608
            ).rows
            foreign_keys_by_table[table_name] = {
                (
                    str(row["from"] if isinstance(row, dict) else row[3]),
                    "main",
                    str(row["table"] if isinstance(row, dict) else row[2]),
                    str(row["to"] if isinstance(row, dict) else row[4]),
                    str(row["on_delete"] if isinstance(row, dict) else row[6]),
                )
                for row in foreign_keys
            }

        if not profile_candidate_schema_is_valid(
            backend="sqlite",
            columns=columns_by_table,
            primary_keys=primary_keys_by_table,
            unique_keys=unique_keys_by_table,
            foreign_keys=foreign_keys_by_table,
        ):
            raise UserDatabaseError(
                "Required profile candidate schema validation failed"
            )

    def _validate_profile_candidate_tables_postgres(
        self,
        *,
        connection: Any | None = None,
    ) -> None:
        def execute(query: str, params: Any = None) -> Any:
            if connection is None:
                return self.backend.execute(query, params)
            return self.backend.execute(query, params, connection=connection)

        placeholders = ", ".join("%s" for _ in PROFILE_CANDIDATE_TABLES)
        table_filter = f"({placeholders})"
        columns = execute(
            "SELECT table_name, column_name, data_type, is_nullable, "
            "column_default, is_identity, identity_generation "
            "FROM information_schema.columns "
            "WHERE table_schema = 'public' AND table_name IN "
            + table_filter,  # nosec B608 -- only fixed-count placeholders are appended.
            PROFILE_CANDIDATE_TABLES,
        ).rows
        columns_by_table: dict[str, dict[str, dict[str, Any]]] = {
            table_name: {} for table_name in PROFILE_CANDIDATE_TABLES
        }
        for row in columns:
            columns_by_table[str(row["table_name"])][str(row["column_name"])] = {
                "data_type": row["data_type"],
                "not_null": str(row["is_nullable"]).upper() == "NO",
                "default": row["column_default"],
                "is_identity": row["is_identity"],
                "identity_generation": row["identity_generation"],
            }

        primary_keys = execute(
            "SELECT tc.table_name, kcu.column_name, kcu.ordinal_position "
            "FROM information_schema.table_constraints AS tc "
            "JOIN information_schema.key_column_usage AS kcu "
            "ON tc.constraint_name = kcu.constraint_name "
            "AND tc.constraint_schema = kcu.constraint_schema "
            "WHERE tc.table_schema = 'public' "
            "AND tc.constraint_type = 'PRIMARY KEY' "
            "AND tc.table_name IN "
            + table_filter  # nosec B608 -- only fixed-count placeholders are appended.
            + " "
            "ORDER BY tc.table_name, kcu.ordinal_position",
            PROFILE_CANDIDATE_TABLES,
        ).rows
        primary_keys_by_table: dict[str, list[tuple[int, str]]] = {
            table_name: [] for table_name in PROFILE_CANDIDATE_TABLES
        }
        for row in primary_keys:
            primary_keys_by_table[str(row["table_name"])].append(
                (int(row["ordinal_position"]), str(row["column_name"]))
            )

        unique_rows = execute(
            "SELECT tc.table_name, tc.constraint_name, kcu.column_name, "
            "kcu.ordinal_position FROM information_schema.table_constraints AS tc "
            "JOIN information_schema.key_column_usage AS kcu "
            "ON tc.constraint_name = kcu.constraint_name "
            "AND tc.constraint_schema = kcu.constraint_schema "
            "WHERE tc.table_schema = 'public' "
            "AND tc.constraint_type = 'UNIQUE' AND tc.table_name IN "
            + table_filter,  # nosec B608 -- only fixed-count placeholders are appended.
            PROFILE_CANDIDATE_TABLES,
        ).rows
        unique_columns: dict[tuple[str, str], list[tuple[int, str]]] = {}
        for row in unique_rows:
            key = (str(row["table_name"]), str(row["constraint_name"]))
            unique_columns.setdefault(key, []).append(
                (int(row["ordinal_position"]), str(row["column_name"]))
            )
        unique_keys_by_table: dict[str, set[tuple[str, ...]]] = {
            table_name: set() for table_name in PROFILE_CANDIDATE_TABLES
        }
        for (table_name, _constraint_name), index_columns in unique_columns.items():
            unique_keys_by_table[table_name].add(
                tuple(column for _position, column in sorted(index_columns))
            )

        foreign_keys = execute(
            "SELECT tc.table_name, kcu.column_name, "
            "ccu.table_schema AS foreign_table_schema, "
            "ccu.table_name AS foreign_table_name, "
            "ccu.column_name AS foreign_column_name, rc.delete_rule "
            "FROM information_schema.table_constraints AS tc "
            "JOIN information_schema.key_column_usage AS kcu "
            "ON tc.constraint_name = kcu.constraint_name "
            "AND tc.constraint_schema = kcu.constraint_schema "
            "JOIN information_schema.referential_constraints AS rc "
            "ON tc.constraint_name = rc.constraint_name "
            "AND tc.constraint_schema = rc.constraint_schema "
            "JOIN information_schema.constraint_column_usage AS ccu "
            "ON rc.unique_constraint_name = ccu.constraint_name "
            "AND rc.unique_constraint_schema = ccu.constraint_schema "
            "WHERE tc.table_schema = 'public' "
            "AND tc.constraint_type = 'FOREIGN KEY' "
            "AND tc.table_name IN "
            + table_filter,  # nosec B608 -- only fixed-count placeholders are appended.
            PROFILE_CANDIDATE_TABLES,
        ).rows
        foreign_keys_by_table: dict[
            str,
            set[tuple[str, str, str, str, str]],
        ] = {table_name: set() for table_name in PROFILE_CANDIDATE_TABLES}
        for row in foreign_keys:
            foreign_keys_by_table[str(row["table_name"])].add(
                (
                    str(row["column_name"]),
                    str(row["foreign_table_schema"]),
                    str(row["foreign_table_name"]),
                    str(row["foreign_column_name"]),
                    str(row["delete_rule"]),
                )
            )

        normalized_primary_keys = {
            table_name: tuple(
                column
                for _position, column in sorted(primary_keys_by_table[table_name])
            )
            for table_name in PROFILE_CANDIDATE_TABLES
        }
        if not profile_candidate_schema_is_valid(
            backend="postgres",
            columns=columns_by_table,
            primary_keys=normalized_primary_keys,
            unique_keys=unique_keys_by_table,
            foreign_keys=foreign_keys_by_table,
        ):
            raise UserDatabaseError(
                "Required profile candidate schema validation failed"
            )
