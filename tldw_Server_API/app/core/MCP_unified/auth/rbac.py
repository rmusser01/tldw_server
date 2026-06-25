"""
Role-Based Access Control (RBAC) for unified MCP module

Implements fine-grained permission management with role inheritance.
"""

import contextlib
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from typing import Any, Optional

from loguru import logger


class UserRole(str, Enum):
    """System-defined user roles"""
    ADMIN = "admin"
    MODERATOR = "moderator"
    USER = "user"
    API_CLIENT = "api_client"
    GUEST = "guest"


class Resource(str, Enum):
    """Resource types in the system"""
    MODULE = "module"
    TOOL = "tool"
    RESOURCE = "resource"
    PROMPT = "prompt"
    MEDIA = "media"
    NOTE = "note"
    CONVERSATION = "conversation"
    TRANSCRIPT = "transcript"
    USER = "user"
    SETTINGS = "settings"


class Action(str, Enum):
    """Actions that can be performed on resources"""
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    EXECUTE = "execute"
    ADMIN = "admin"


@dataclass
class Permission:
    """
    A permission defines an action on a resource.

    Supports wildcards for flexible permission definitions.
    """
    resource: Resource
    action: Action
    resource_id: Optional[str] = None  # None means all resources of this type
    conditions: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        resource_part = f"{self.resource.value}:{self.resource_id or '*'}"
        return f"{resource_part}:{self.action.value}"

    def __hash__(self) -> int:
        return hash(str(self))

    def matches(self, resource: Resource, action: Action, resource_id: Optional[str] = None) -> bool:
        """Check if this permission matches a request"""
        # Check resource type
        if self.resource != resource:
            return False

        # Check action
        if self.action != Action.ADMIN and self.action != action:
            return False

        # Check resource ID (None means all)
        if self.resource_id is not None and self.resource_id != resource_id:
            return False

        # Check conditions if any
        if self.conditions:
            # Implement condition checking logic here
            # For now, we'll skip complex conditions
            pass

        return True


@dataclass
class Role:
    """A role is a collection of permissions with inheritance support"""
    name: str
    description: str
    permissions: set[Permission] = field(default_factory=set)
    inherits_from: set[str] = field(default_factory=set)

    def add_permission(self, permission: Permission):
        """Add a permission to this role"""
        self.permissions.add(permission)

    def remove_permission(self, permission: Permission):
        """Remove a permission from this role"""
        self.permissions.discard(permission)

    def has_permission(self, resource: Resource, action: Action, resource_id: Optional[str] = None) -> bool:
        """Check if role has a specific permission"""
        return any(perm.matches(resource, action, resource_id) for perm in self.permissions)


class RBACPolicy:
    """
    RBAC Policy manager with role inheritance and caching.

    Provides efficient permission checking with LRU cache.
    """

    def __init__(self):
        self.roles: dict[str, Role] = {}
        self.user_roles: dict[str, set[str]] = {}
        self.user_permissions: dict[str, set[Permission]] = {}

        # Initialize default roles
        self._init_default_roles()

        logger.info("RBAC Policy initialized with default roles")

    def _init_default_roles(self):
        """Initialize default system roles with secure defaults"""

        # Admin role - full access
        admin_role = Role(
            name=UserRole.ADMIN.value,
            description="Administrator with full system access",
            permissions={
                Permission(resource, Action.ADMIN)
                for resource in Resource
            }
        )

        # Moderator role - manage content and users
        moderator_role = Role(
            name=UserRole.MODERATOR.value,
            description="Moderator with content management access",
            permissions={
                # Can manage media and content
                Permission(Resource.MEDIA, Action.CREATE),
                Permission(Resource.MEDIA, Action.READ),
                Permission(Resource.MEDIA, Action.UPDATE),
                Permission(Resource.MEDIA, Action.DELETE),
                # Can manage notes
                Permission(Resource.NOTE, Action.READ),
                Permission(Resource.NOTE, Action.UPDATE),
                Permission(Resource.NOTE, Action.DELETE),
                # Can view conversations
                Permission(Resource.CONVERSATION, Action.READ),
                Permission(Resource.CONVERSATION, Action.DELETE),
            },
            inherits_from={UserRole.USER.value}
        )

        # User role - standard access
        user_role = Role(
            name=UserRole.USER.value,
            description="Standard user with read/write access to own content",
            permissions={
                # Can execute explicitly safe tools
                Permission(Resource.TOOL, Action.EXECUTE, "search_media"),
                Permission(Resource.TOOL, Action.EXECUTE, "get_transcript"),
                Permission(Resource.TOOL, Action.EXECUTE, "chat_completion"),
                # Can read resources
                Permission(Resource.RESOURCE, Action.READ),
                # Can use prompts
                Permission(Resource.PROMPT, Action.READ),
                Permission(Resource.PROMPT, Action.EXECUTE),
                # Can manage own media
                Permission(Resource.MEDIA, Action.CREATE),
                Permission(Resource.MEDIA, Action.READ),
                Permission(Resource.MEDIA, Action.UPDATE),
                # Can manage own notes
                Permission(Resource.NOTE, Action.CREATE),
                Permission(Resource.NOTE, Action.READ),
                Permission(Resource.NOTE, Action.UPDATE),
                Permission(Resource.NOTE, Action.DELETE),
                # Can manage own conversations
                Permission(Resource.CONVERSATION, Action.CREATE),
                Permission(Resource.CONVERSATION, Action.READ),
                Permission(Resource.CONVERSATION, Action.UPDATE),
                Permission(Resource.CONVERSATION, Action.DELETE),
                # Can read transcripts
                Permission(Resource.TRANSCRIPT, Action.READ),
            }
        )

        # API Client role - programmatic access
        api_client_role = Role(
            name=UserRole.API_CLIENT.value,
            description="API client with limited programmatic access",
            permissions={
                # Can execute specific tools
                Permission(Resource.TOOL, Action.EXECUTE, "search_media"),
                Permission(Resource.TOOL, Action.EXECUTE, "get_transcript"),
                Permission(Resource.TOOL, Action.EXECUTE, "chat_completion"),
                # Read-only access to resources
                Permission(Resource.RESOURCE, Action.READ),
                Permission(Resource.MEDIA, Action.READ),
                Permission(Resource.PROMPT, Action.READ),
                Permission(Resource.TRANSCRIPT, Action.READ),
            }
        )

        # Guest role - minimal access
        guest_role = Role(
            name=UserRole.GUEST.value,
            description="Guest with minimal read-only access",
            permissions={
                # Can only read public resources
                Permission(Resource.RESOURCE, Action.READ),
                # Can search media
                Permission(Resource.TOOL, Action.EXECUTE, "search_media"),
                # Can view media metadata
                Permission(Resource.MEDIA, Action.READ),
            }
        )

        # Register all roles
        self.roles = {
            admin_role.name: admin_role,
            moderator_role.name: moderator_role,
            user_role.name: user_role,
            api_client_role.name: api_client_role,
            guest_role.name: guest_role,
        }

    def create_role(self, role: Role):
        """Create a new role"""
        if role.name in self.roles:
            raise ValueError(f"Role {role.name} already exists")

        self.roles[role.name] = role
        logger.info(f"Created role: {role.name}")

    def delete_role(self, role_name: str):
        """Delete a role"""
        if role_name not in self.roles:
            raise ValueError(f"Role {role_name} does not exist")

        # Don't allow deletion of system roles
        if role_name in [r.value for r in UserRole]:
            raise ValueError(f"Cannot delete system role: {role_name}")

        del self.roles[role_name]

        # Remove role from all users
        for user_roles in self.user_roles.values():
            user_roles.discard(role_name)

        logger.info(f"Deleted role: {role_name}")

    def assign_role(self, user_id: str, role_name: str):
        """Assign a role to a user"""
        if role_name not in self.roles:
            raise ValueError(f"Role {role_name} does not exist")

        if user_id not in self.user_roles:
            self.user_roles[user_id] = set()

        self.user_roles[user_id].add(role_name)

        # Clear permission cache for user
        self._clear_user_cache(user_id)

        logger.info(f"Assigned role {role_name} to user {user_id}", extra={"audit": True})

    def revoke_role(self, user_id: str, role_name: str):
        """Revoke a role from a user"""
        if user_id in self.user_roles:
            self.user_roles[user_id].discard(role_name)

            # Clear permission cache for user
            self._clear_user_cache(user_id)

            logger.info(f"Revoked role {role_name} from user {user_id}", extra={"audit": True})

    def grant_permission(self, user_id: str, permission: Permission):
        """Grant a specific permission to a user"""
        if user_id not in self.user_permissions:
            self.user_permissions[user_id] = set()

        self.user_permissions[user_id].add(permission)

        # Clear permission cache for user
        self._clear_user_cache(user_id)

        logger.info(
            f"Granted permission {permission} to user {user_id}",
            extra={"audit": True}
        )

    def revoke_permission(self, user_id: str, permission: Permission):
        """Revoke a specific permission from a user"""
        if user_id in self.user_permissions:
            self.user_permissions[user_id].discard(permission)

            # Clear permission cache for user
            self._clear_user_cache(user_id)

            logger.info(
                f"Revoked permission {permission} from user {user_id}",
                extra={"audit": True}
            )

    @lru_cache(maxsize=1000)  # noqa: B019
    def _get_user_permissions(self, user_id: str) -> set[Permission]:
        """
        Get all permissions for a user (cached).

        Includes permissions from roles and direct grants.
        """
        permissions = set()

        # Get direct permissions
        if user_id in self.user_permissions:
            permissions.update(self.user_permissions[user_id])

        # Get permissions from roles (with inheritance)
        if user_id in self.user_roles:
            visited_roles = set()
            roles_to_check = list(self.user_roles[user_id])

            while roles_to_check:
                role_name = roles_to_check.pop()

                if role_name in visited_roles:
                    continue

                visited_roles.add(role_name)

                if role_name not in self.roles:
                    continue

                role = self.roles[role_name]
                permissions.update(role.permissions)

                # Add inherited roles to check
                roles_to_check.extend(role.inherits_from)

        return permissions

    def check_permission(
        self,
        user_id: str,
        resource: Resource,
        action: Action,
        resource_id: Optional[str] = None
    ) -> bool:
        """
        Check if a user has permission to perform an action on a resource.

        Uses caching for efficient permission checks.
        """
        # Get user's permissions (cached)
        permissions = self._get_user_permissions(user_id)

        # Check each permission
        for perm in permissions:
            if perm.matches(resource, action, resource_id):
                logger.debug(
                    f"Permission granted: user={user_id}, "
                    f"resource={resource.value}:{resource_id}, "
                    f"action={action.value}"
                )
                return True

        logger.debug(
            f"Permission denied: user={user_id}, "
            f"resource={resource.value}:{resource_id}, "
            f"action={action.value}"
        )
        return False

    def get_user_roles(self, user_id: str) -> set[str]:
        """Get all roles for a user"""
        return self.user_roles.get(user_id, set())

    def get_user_permissions_list(self, user_id: str) -> list[str]:
        """Get a list of all permissions for a user (for display)"""
        permissions = self._get_user_permissions(user_id)
        return [str(p) for p in permissions]

    def _clear_user_cache(self, user_id: str):
        """Clear cached permissions for a user"""
        # lru_cache has no per-key invalidation; clear entire cache.
        with contextlib.suppress(Exception):
            self._get_user_permissions.cache_clear()

    def can_access_module(self, user_id: str, module_id: str) -> bool:
        """Check if user can access a module"""
        return self.check_permission(
            user_id,
            Resource.MODULE,
            Action.READ,
            module_id
        )

    def can_execute_tool(self, user_id: str, tool_name: str) -> bool:
        """Check if user can execute a tool"""
        return self.check_permission(
            user_id,
            Resource.TOOL,
            Action.EXECUTE,
            tool_name
        )

    def can_read_resource(self, user_id: str, resource_uri: str) -> bool:
        """Check if user can read a resource"""
        return self.check_permission(
            user_id,
            Resource.RESOURCE,
            Action.READ,
            resource_uri
        )


# Singleton instance
_rbac_policy = None


def get_rbac_policy() -> RBACPolicy:
    """Get or create RBAC policy singleton"""
    global _rbac_policy
    if _rbac_policy is None:
        _rbac_policy = RBACPolicy()
    return _rbac_policy
