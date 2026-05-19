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
            self.backend = backend
        elif config:
            self.backend = DatabaseBackendFactory.create_backend(config)
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
            self.backend = DatabaseBackendFactory.create_backend(config)

        # Initialize schema if needed
        self._initialize_schema()

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
                logger.info(f"Database schema loaded from {schema_path}")
                loaded_from_file = True
            except Exception as exc:  # noqa: BLE001
                logger.error(f"Failed to read schema file {schema_path}: {exc}")

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
                    raise UserDatabaseError(
                        f"Required schema initialization failed after fallback: {fallback_exc}"
                    ) from fallback_exc
            else:
                raise UserDatabaseError(
                    f"Required schema initialization failed: {exc}"
                ) from exc

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

                # Insert user
                self.backend.execute(
                    """
                    INSERT INTO users (uuid, username, email, password_hash, metadata)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (user_uuid, username, email, password_hash, metadata),
                    connection=conn,
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
            raise UserDatabaseError(f"Failed to create user: {e}") from e

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

            for field, value in updates.items():
                if field in allowed_fields:
                    set_clause.append(f"{field} = ?")
                    values.append(value if field != 'metadata' else json.dumps(value))

            if not set_clause:
                return False

            values.append(user_id)
            set_clause_sql = ", ".join(set_clause)
            query_template = "UPDATE users SET {set_clause_sql}, updated_at = CURRENT_TIMESTAMP WHERE id = ?"
            query = query_template.format_map(locals())  # nosec B608

            result = self.backend.execute(query, tuple(values), connection=conn)

            success = False
            if self.backend.backend_type == BackendType.SQLITE:
                try:
                    change_result = self.backend.execute(
                        "SELECT changes() AS changes",
                        connection=conn,
                    )
                    changes = change_result.rows[0].get("changes", 0) if change_result.rows else 0
                    success = bool(changes)
                except _USERDB_NONCRITICAL_EXCEPTIONS:
                    success = False
            else:
                success = result.rowcount > 0

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
                logger.error(f"Failed to assign role: {e}")
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
            self.backend.execute(
                """
                UPDATE users
                SET last_login = CURRENT_TIMESTAMP,
                    failed_login_attempts = 0,
                    locked_until = NULL
                WHERE id = ?
                """,
                (user_id,),
                connection=conn,
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
            logger.error(f"Failed to create audit log: {e}")

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
                uuid TEXT UNIQUE NOT NULL,
                username TEXT UNIQUE NOT NULL CHECK (length(username) <= 255),
                email TEXT UNIQUE NOT NULL CHECK (length(email) <= 255),
                password_hash TEXT NOT NULL,
                metadata TEXT,
                is_active INTEGER NOT NULL DEFAULT 1,
                is_verified INTEGER NOT NULL DEFAULT 0,
                is_superuser INTEGER NOT NULL DEFAULT 0,
                failed_login_attempts INTEGER NOT NULL DEFAULT 0,
                locked_until TIMESTAMP,
                last_login TIMESTAMP,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
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
        ]

    @staticmethod
    def _default_schema_statements_postgres() -> list[str]:
        return [
            "CREATE EXTENSION IF NOT EXISTS pgcrypto;",
            """
            CREATE TABLE IF NOT EXISTS users (
                id BIGSERIAL PRIMARY KEY,
                uuid UUID UNIQUE NOT NULL DEFAULT gen_random_uuid(),
                username VARCHAR(255) UNIQUE NOT NULL,
                email VARCHAR(255) UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                metadata JSONB,
                is_active BOOLEAN NOT NULL DEFAULT TRUE,
                is_verified BOOLEAN NOT NULL DEFAULT FALSE,
                is_superuser BOOLEAN NOT NULL DEFAULT FALSE,
                failed_login_attempts INTEGER NOT NULL DEFAULT 0,
                locked_until TIMESTAMPTZ,
                last_login TIMESTAMPTZ,
                created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_users_username ON users (username)",
            "CREATE INDEX IF NOT EXISTS idx_users_email ON users (email)",
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
                user_id BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                role_id BIGINT NOT NULL REFERENCES roles(id) ON DELETE CASCADE,
                granted_by BIGINT,
                expires_at TIMESTAMPTZ,
                PRIMARY KEY (user_id, role_id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS user_permissions (
                user_id BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
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
                created_by BIGINT REFERENCES users(id) ON DELETE SET NULL,
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
                user_id BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                ip_address TEXT,
                user_agent TEXT,
                used_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS auth_audit_log (
                id BIGSERIAL PRIMARY KEY,
                event_type TEXT NOT NULL,
                user_id BIGINT REFERENCES users(id) ON DELETE SET NULL,
                target_user_id BIGINT REFERENCES users(id) ON DELETE SET NULL,
                details JSONB,
                created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """,
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
            raise UserDatabaseError(
                f"Required RBAC seed verification failed: {exc}"
            ) from exc


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
                user_step = "users.uuid backfill"
                self.backend.execute(
                    """
                    UPDATE users
                    SET uuid = lower(hex(randomblob(4))) || '-' ||
                               lower(hex(randomblob(2))) || '-' ||
                               lower(hex(randomblob(2))) || '-' ||
                               lower(hex(randomblob(2))) || '-' ||
                               lower(hex(randomblob(6)))
                    WHERE uuid IS NULL OR uuid = ''
                    """
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
                user_step = "pgcrypto extension"
                self.backend.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto;")
                user_step = "users.uuid"
                self.backend.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS uuid UUID")
                user_step = "users.metadata"
                self.backend.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS metadata JSONB")
                user_step = "users.failed_login_attempts"
                self.backend.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS failed_login_attempts INTEGER DEFAULT 0")
                user_step = "users.locked_until"
                self.backend.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS locked_until TIMESTAMPTZ")
                user_step = "users.is_superuser"
                self.backend.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS is_superuser BOOLEAN DEFAULT FALSE")
                try:
                    user_step = "users.uuid backfill"
                    self.backend.execute("UPDATE users SET uuid = gen_random_uuid() WHERE uuid IS NULL")
                except _USERDB_NONCRITICAL_EXCEPTIONS:
                    user_step = "users.uuid text backfill"
                    self.backend.execute("UPDATE users SET uuid = gen_random_uuid()::text WHERE uuid IS NULL")
                user_step = "users.uuid not null"
                self.backend.execute("ALTER TABLE users ALTER COLUMN uuid SET NOT NULL")
                try:
                    user_step = "users.uuid default"
                    self.backend.execute("ALTER TABLE users ALTER COLUMN uuid SET DEFAULT gen_random_uuid()")
                except _USERDB_NONCRITICAL_EXCEPTIONS:
                    user_step = "users.uuid text default"
                    self.backend.execute("ALTER TABLE users ALTER COLUMN uuid SET DEFAULT (gen_random_uuid()::text)")
                user_step = "users.failed_login_attempts backfill"
                self.backend.execute(
                    "UPDATE users SET failed_login_attempts = 0 WHERE failed_login_attempts IS NULL"
                )
                user_step = "users.locked_until backfill"
                self.backend.execute(
                    "UPDATE users SET locked_until = NULL WHERE locked_until IS NULL"
                )
        except Exception as exc:  # noqa: BLE001
            raise UserDatabaseError(
                f"Required user schema normalization failed at {user_step}: {exc}"
            ) from exc

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
            raise UserDatabaseError(
                f"Required registration_codes normalization failed at {registration_step}: {exc}"
            ) from exc
