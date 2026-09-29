# auth_permissions.py
# Authentication and permissions system for Prompt Studio (Deprecated)

import hashlib
import json
import secrets
import warnings
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Optional

from tldw_Server_API.app.core.Utils import jwt_compat as jwt
from tldw_Server_API.app.core.Utils.jwt_compat import JWTError
from loguru import logger

from tldw_Server_API.app.core.DB_Management.backends.base import BackendType
from tldw_Server_API.app.core.DB_Management.PromptStudioDatabase import PromptStudioDatabase

########################################################################################################################
# Permission Types

class Permission(str, Enum):
    """Prompt Studio permissions."""

    # Project permissions
    PROJECT_CREATE = "project.create"
    PROJECT_READ = "project.read"
    PROJECT_UPDATE = "project.update"
    PROJECT_DELETE = "project.delete"
    PROJECT_SHARE = "project.share"

    # Prompt permissions
    PROMPT_CREATE = "prompt.create"
    PROMPT_READ = "prompt.read"
    PROMPT_UPDATE = "prompt.update"
    PROMPT_DELETE = "prompt.delete"
    PROMPT_EXECUTE = "prompt.execute"

    # Test permissions
    TEST_CREATE = "test.create"
    TEST_READ = "test.read"
    TEST_UPDATE = "test.update"
    TEST_DELETE = "test.delete"
    TEST_EXECUTE = "test.execute"

    # Evaluation permissions
    EVALUATION_CREATE = "evaluation.create"
    EVALUATION_READ = "evaluation.read"
    EVALUATION_CANCEL = "evaluation.cancel"
    EVALUATION_DELETE = "evaluation.delete"

    # Optimization permissions
    OPTIMIZATION_CREATE = "optimization.create"
    OPTIMIZATION_READ = "optimization.read"
    OPTIMIZATION_CANCEL = "optimization.cancel"
    OPTIMIZATION_DELETE = "optimization.delete"

    # Admin permissions
    ADMIN_ACCESS = "admin.access"
    ADMIN_USERS = "admin.users"
    ADMIN_SETTINGS = "admin.settings"
    ADMIN_EXPORT = "admin.export"

class Role(str, Enum):
    """User roles."""

    VIEWER = "viewer"
    EDITOR = "editor"
    ADMIN = "admin"
    OWNER = "owner"

# Role permission mappings
# Define base permissions first
VIEWER_PERMISSIONS = {
    Permission.PROJECT_READ,
    Permission.PROMPT_READ,
    Permission.TEST_READ,
    Permission.EVALUATION_READ,
    Permission.OPTIMIZATION_READ
}

EDITOR_PERMISSIONS = {
    Permission.PROJECT_CREATE,
    Permission.PROJECT_READ,
    Permission.PROJECT_UPDATE,
    Permission.PROMPT_CREATE,
    Permission.PROMPT_READ,
    Permission.PROMPT_UPDATE,
    Permission.PROMPT_EXECUTE,
    Permission.TEST_CREATE,
    Permission.TEST_READ,
    Permission.TEST_UPDATE,
    Permission.TEST_EXECUTE,
    Permission.EVALUATION_CREATE,
    Permission.EVALUATION_READ,
    Permission.EVALUATION_CANCEL,
    Permission.OPTIMIZATION_CREATE,
    Permission.OPTIMIZATION_READ,
    Permission.OPTIMIZATION_CANCEL
}

ADMIN_PERMISSIONS = {
    *EDITOR_PERMISSIONS,
    Permission.PROJECT_DELETE,
    Permission.PROJECT_SHARE,
    Permission.PROMPT_DELETE,
    Permission.TEST_DELETE,
    Permission.EVALUATION_DELETE,
    Permission.OPTIMIZATION_DELETE,
    Permission.ADMIN_ACCESS,
    Permission.ADMIN_SETTINGS,
    Permission.ADMIN_EXPORT
}

ROLE_PERMISSIONS = {
    Role.VIEWER: VIEWER_PERMISSIONS,
    Role.EDITOR: EDITOR_PERMISSIONS,
    Role.ADMIN: ADMIN_PERMISSIONS,
    Role.OWNER: {
        # All permissions
        *list(Permission)
    }
}

########################################################################################################################
# Authentication Manager

class AuthenticationManager:
    """Manages user authentication for Prompt Studio."""

    def __init__(self, db: PromptStudioDatabase, secret_key: str):
        """
        Initialize authentication manager.

        Args:
            db: Database instance
            secret_key: Secret key for JWT tokens
        """
        # Deprecation notice: use core AuthNZ/JWT and RBAC instead
        warnings.warn(
            "Prompt Studio AuthenticationManager is deprecated. "
            "Use core AuthNZ (JWT/API key) and RBAC instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.db = db
        self.secret_key = secret_key
        self.algorithm = "HS256"
        self.token_expiry_hours = 24

        # Initialize user tables if needed
        self._init_auth_tables()

    def _init_auth_tables(self):
        """Initialize authentication tables."""
        with self.db.transaction() as conn:
            cursor = conn.cursor()

            # Users table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS prompt_studio_users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    uuid TEXT UNIQUE NOT NULL,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    role TEXT NOT NULL DEFAULT 'editor',
                    is_active BOOLEAN DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_login TIMESTAMP,
                    settings TEXT DEFAULT '{}',
                    api_key TEXT UNIQUE,
                    client_id TEXT NOT NULL
                )
            """)

            # Sessions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS prompt_studio_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    token TEXT UNIQUE NOT NULL,
                    expires_at TIMESTAMP NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    ip_address TEXT,
                    user_agent TEXT,
                    FOREIGN KEY (user_id) REFERENCES prompt_studio_users(id)
                )
            """)

            # Project permissions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS prompt_studio_project_permissions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_id INTEGER NOT NULL,
                    user_id INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    permissions TEXT NOT NULL DEFAULT '[]',
                    granted_by INTEGER,
                    granted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES prompt_studio_users(id),
                    UNIQUE(project_id, user_id)
                )
            """)

            # Audit log table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS prompt_studio_audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    action TEXT NOT NULL,
                    resource_type TEXT,
                    resource_id INTEGER,
                    details TEXT,
                    ip_address TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES prompt_studio_users(id)
                )
            """)

    ####################################################################################################################
    # User Management

    def create_user(self, username: str, email: str, password: str,
                   role: Role = Role.EDITOR) -> dict[str, Any]:
        """
        Create a new user.

        Args:
            username: Username
            email: Email address
            password: Plain text password
            role: User role

        Returns:
            Created user details
        """
        with self.db.transaction() as conn:
            cursor = conn.cursor()

            # Check if user exists
            cursor.execute(
                "SELECT id FROM prompt_studio_users WHERE username = ? OR email = ?",
                (username, email)
            )
            if cursor.fetchone():
                raise ValueError("User with this username or email already exists")

            # Hash password
            password_hash = self._hash_password(password)

            # Generate API key
            api_key = self._generate_api_key()

            # Create user
            cursor.execute("""
                INSERT INTO prompt_studio_users (
                    uuid, username, email, password_hash, role, api_key, client_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                f"user-{secrets.token_urlsafe(16)}",
                username,
                email,
                password_hash,
                role.value,
                api_key,
                self.db.client_id
            ))

            user_id = cursor.lastrowid

        logger.info(f"Created user: {username} (ID: {user_id})")

        return {
            "id": user_id,
            "username": username,
            "email": email,
            "role": role.value,
            "api_key": api_key
        }

    def authenticate_user(self, username: str, password: str) -> Optional[dict[str, Any]]:
        """
        Authenticate a user with username and password.

        Args:
            username: Username or email
            password: Plain text password

        Returns:
            User details if authenticated, None otherwise
        """
        with self.db.transaction() as conn:
            cursor = conn.cursor()

            # Get user
            cursor.execute("""
                SELECT id, username, email, password_hash, role, is_active
                FROM prompt_studio_users
                WHERE (username = ? OR email = ?) AND is_active = 1
            """, (username, username))

            row = cursor.fetchone()
            if not row:
                return None

            user_id, db_username, email, password_hash, role, is_active = row

            # Verify password
            if not self._verify_password(password, password_hash):
                return None

            # Update last login
            cursor.execute(
                "UPDATE prompt_studio_users SET last_login = CURRENT_TIMESTAMP WHERE id = ?",
                (user_id,)
            )

        # Create session token (outside transaction to allow nested transaction)
        token = self.create_session(user_id)

        return {
            "id": user_id,
            "username": db_username,
            "email": email,
            "role": role,
            "token": token
        }

    def authenticate_api_key(self, api_key: str) -> Optional[dict[str, Any]]:
        """
        Authenticate using API key.

        Args:
            api_key: API key

        Returns:
            User details if authenticated
        """
        with self.db.transaction() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT id, username, email, role
                FROM prompt_studio_users
                WHERE api_key = ? AND is_active = 1
            """, (api_key,))

            row = cursor.fetchone()
            if row:
                return {
                    "id": row[0],
                    "username": row[1],
                    "email": row[2],
                    "role": row[3]
                }

        return None

    ####################################################################################################################
    # Session Management

    def create_session(self, user_id: int, ip_address: Optional[str] = None,
                      user_agent: Optional[str] = None) -> str:
        """
        Create a session for a user.

        Args:
            user_id: User ID
            ip_address: Client IP address
            user_agent: Client user agent

        Returns:
            JWT token
        """
        # Create JWT token
        payload = {
            "user_id": user_id,
            "exp": datetime.utcnow() + timedelta(hours=self.token_expiry_hours),
            "iat": datetime.utcnow(),
            "jti": secrets.token_urlsafe(16)
        }

        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)

        # Store session
        with self.db.transaction() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO prompt_studio_sessions (
                    user_id, token, expires_at, ip_address, user_agent
                ) VALUES (?, ?, ?, ?, ?)
            """, (
                user_id,
                token,
                payload["exp"],
                ip_address,
                user_agent
            ))

        return token

    def validate_session(self, token: str) -> Optional[dict[str, Any]]:
        """
        Validate a session token.

        Args:
            token: JWT token

        Returns:
            User details if valid
        """
        try:
            # Decode token
            jwt.decode(token, self.secret_key, algorithms=[self.algorithm])

            # Check if session exists and not expired
            with self.db.transaction() as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    SELECT u.id, u.username, u.email, u.role
                    FROM prompt_studio_sessions s
                    JOIN prompt_studio_users u ON s.user_id = u.id
                    WHERE s.token = ? AND s.expires_at > CURRENT_TIMESTAMP
                        AND u.is_active = 1
                """, (token,))

                row = cursor.fetchone()
                if row:
                    return {
                        "id": row[0],
                        "username": row[1],
                        "email": row[2],
                        "role": row[3]
                    }

        except JWTError:
            pass

        return None

    def revoke_session(self, token: str):
        """Revoke a session."""
        with self.db.transaction() as conn:
            cursor = conn.cursor()

            cursor.execute(
                "DELETE FROM prompt_studio_sessions WHERE token = ?",
                (token,)
            )

    ####################################################################################################################
    # Helper Methods

    def _hash_password(self, password: str) -> str:
        """Hash a password using PBKDF2."""
        salt = secrets.token_bytes(32)
        pwd_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 100000)
        return salt.hex() + pwd_hash.hex()

    def _verify_password(self, password: str, password_hash: str) -> bool:
        """Verify a password against hash."""
        salt = bytes.fromhex(password_hash[:64])
        stored_hash = password_hash[64:]
        pwd_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 100000)
        return pwd_hash.hex() == stored_hash

    def _generate_api_key(self) -> str:
        """Generate a unique API key."""
        return f"pstudio_{secrets.token_urlsafe(32)}"

########################################################################################################################
# Permission Manager

class PermissionManager:
    """Manages permissions for Prompt Studio resources."""

    def __init__(self, db: PromptStudioDatabase):
        """
        Initialize permission manager.

        Args:
            db: Database instance
        """
        warnings.warn(
            "Prompt Studio PermissionManager is deprecated. "
            "Use core AuthNZ RBAC and endpoint dependencies instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.db = db

    def check_permission(self, user_id: int, permission: Permission,
                        resource_type: Optional[str] = None,
                        resource_id: Optional[int] = None) -> bool:
        """
        Check if user has permission.

        Args:
            user_id: User ID
            permission: Required permission
            resource_type: Type of resource (project, prompt, etc.)
            resource_id: Resource ID

        Returns:
            True if user has permission
        """
        # Get user role
        user_role = self._get_user_role(user_id)
        if not user_role:
            return False

        # Check role-based permissions
        role_perms = ROLE_PERMISSIONS.get(Role(user_role), set())
        if permission in role_perms:
            return True

        # Check resource-specific permissions
        if resource_type == "project" and resource_id:
            return self._check_project_permission(user_id, resource_id, permission)

        return False

    def grant_project_access(self, project_id: int, user_id: int,
                           role: Role, granted_by: int):
        """
        Grant user access to a project.

        Args:
            project_id: Project ID
            user_id: User ID to grant access
            role: Role for the project
            granted_by: User ID granting access
        """
        with self.db.transaction() as conn:
            cursor = conn.cursor()

            # Get role permissions
            permissions = list(ROLE_PERMISSIONS.get(role, set()))

            params = (
                project_id,
                user_id,
                role.value,
                json.dumps([p.value for p in permissions]),
                granted_by,
            )

            if getattr(self.db, "backend_type", BackendType.SQLITE) == BackendType.POSTGRESQL:
                cursor.execute(
                    """
                    INSERT INTO prompt_studio_project_permissions (
                        project_id, user_id, role, permissions, granted_by
                    ) VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT (project_id, user_id) DO UPDATE
                        SET role = EXCLUDED.role,
                            permissions = EXCLUDED.permissions,
                            granted_by = EXCLUDED.granted_by,
                            granted_at = CURRENT_TIMESTAMP
                    """,
                    params,
                )
            else:
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO prompt_studio_project_permissions (
                        project_id, user_id, role, permissions, granted_by
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    params,
                )

        # Log action
        self.log_action(
            user_id=granted_by,
            action="grant_access",
            resource_type="project",
            resource_id=project_id,
            details=f"Granted {role.value} to user {user_id}"
        )

    def revoke_project_access(self, project_id: int, user_id: int,
                             revoked_by: int):
        """Revoke user access to a project."""
        with self.db.transaction() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                DELETE FROM prompt_studio_project_permissions
                WHERE project_id = ? AND user_id = ?
            """, (project_id, user_id))

        # Log action
        self.log_action(
            user_id=revoked_by,
            action="revoke_access",
            resource_type="project",
            resource_id=project_id,
            details=f"Revoked access for user {user_id}"
        )

    def get_user_projects(self, user_id: int) -> list[dict[str, Any]]:
        """Get projects user has access to."""
        with self.db.transaction() as conn:
            cursor = conn.cursor()

            # Get owned projects
            cursor.execute("""
                SELECT p.*, 'owner' as role
                FROM prompt_studio_projects p
                WHERE p.created_by = ? AND p.deleted = 0
            """, (user_id,))

            projects = []
            for row in cursor.fetchall():
                project = self.db._row_to_dict(cursor, row)
                projects.append(project)

            # Get shared projects
            cursor.execute("""
                SELECT p.*, pp.role
                FROM prompt_studio_projects p
                JOIN prompt_studio_project_permissions pp ON p.id = pp.project_id
                WHERE pp.user_id = ? AND p.deleted = 0
            """, (user_id,))

            for row in cursor.fetchall():
                project = self.db._row_to_dict(cursor, row)
                projects.append(project)

            return projects

    def log_action(self, user_id: int, action: str,
                  resource_type: Optional[str] = None,
                  resource_id: Optional[int] = None,
                  details: Optional[str] = None,
                  ip_address: Optional[str] = None):
        """Log an action for audit trail."""
        with self.db.transaction() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO prompt_studio_audit_log (
                    user_id, action, resource_type, resource_id,
                    details, ip_address
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                user_id,
                action,
                resource_type,
                resource_id,
                details,
                ip_address
            ))

    def _get_user_role(self, user_id: int) -> Optional[str]:
        """Get user's global role."""
        with self.db.transaction() as conn:
            cursor = conn.cursor()

            cursor.execute(
                "SELECT role FROM prompt_studio_users WHERE id = ?",
                (user_id,)
            )

            row = cursor.fetchone()
            return row[0] if row else None

    def _check_project_permission(self, user_id: int, project_id: int,
                                 permission: Permission) -> bool:
        """Check project-specific permission."""
        with self.db.transaction() as conn:
            cursor = conn.cursor()

            # Check if user owns the project
            cursor.execute("""
                SELECT created_by FROM prompt_studio_projects
                WHERE id = ? AND deleted = 0
            """, (project_id,))

            row = cursor.fetchone()
            if row and row[0] == user_id:
                return True  # Owner has all permissions

            # Check granted permissions
            cursor.execute("""
                SELECT permissions FROM prompt_studio_project_permissions
                WHERE project_id = ? AND user_id = ?
            """, (project_id, user_id))

            row = cursor.fetchone()
            if row:
                raw_permissions = row[0]
                if isinstance(raw_permissions, list):
                    granted_permissions = raw_permissions
                else:
                    try:
                        granted_permissions = json.loads(raw_permissions) if raw_permissions else []
                    except (json.JSONDecodeError, TypeError):
                        granted_permissions = []
                return permission.value in granted_permissions

            return False

########################################################################################################################
# Resource Access Control

class ResourceAccessControl:
    """Controls access to Prompt Studio resources."""

    def __init__(self, auth_manager: AuthenticationManager,
                 perm_manager: PermissionManager):
        """
        Initialize resource access control.

        Args:
            auth_manager: Authentication manager
            perm_manager: Permission manager
        """
        self.auth = auth_manager
        self.perms = perm_manager

    def require_auth(self, token: Optional[str] = None,
                     api_key: Optional[str] = None) -> dict[str, Any]:
        """
        Require authentication.

        Args:
            token: JWT token
            api_key: API key

        Returns:
            User details

        Raises:
            ValueError: If not authenticated
        """
        user = None

        if token:
            user = self.auth.validate_session(token)
        elif api_key:
            user = self.auth.authenticate_api_key(api_key)

        if not user:
            raise ValueError("Authentication required")

        return user

    def require_permission(self, user: dict[str, Any], permission: Permission,
                         resource_type: Optional[str] = None,
                         resource_id: Optional[int] = None):
        """
        Require specific permission.

        Args:
            user: User details
            permission: Required permission
            resource_type: Resource type
            resource_id: Resource ID

        Raises:
            ValueError: If permission denied
        """
        if not self.perms.check_permission(
            user["id"], permission, resource_type, resource_id
        ):
            raise ValueError(f"Permission denied: {permission.value}")

    def filter_by_access(self, user: dict[str, Any],
                        resources: list[dict[str, Any]],
                        resource_type: str) -> list[dict[str, Any]]:
        """
        Filter resources by user access.

        Args:
            user: User details
            resources: List of resources
            resource_type: Type of resources

        Returns:
            Filtered list of accessible resources
        """
        if resource_type == "project":
            # Get user's accessible projects
            user_projects = self.perms.get_user_projects(user["id"])
            project_ids = {p["id"] for p in user_projects}

            return [r for r in resources if r.get("id") in project_ids]

        # For other resources, check project access
        filtered = []
        for resource in resources:
            project_id = resource.get("project_id")
            if project_id and self.perms.check_permission(
                user["id"], Permission.PROJECT_READ, "project", project_id
            ):
                filtered.append(resource)

        return filtered
