"""
Task authorization and permission management for the scheduler.

This module provides authorization checks to ensure users can only
submit and manage tasks they have permissions for.
"""

import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

from loguru import logger


class TaskPermission(Enum):
    """Task permissions"""
    SUBMIT = "submit"
    CANCEL = "cancel"
    VIEW = "view"
    ADMIN = "admin"


@dataclass
class AuthContext:
    """Authorization context for a request"""
    user_id: Optional[str] = None
    api_key: Optional[str] = None
    roles: list[str] = None
    permissions: set[str] = None

    def __post_init__(self):
        if self.roles is None:
            self.roles = []
        if self.permissions is None:
            self.permissions = set()

    @property
    def is_authenticated(self) -> bool:
        """Check if context is authenticated"""
        return self.user_id is not None or self.api_key is not None

    @property
    def is_admin(self) -> bool:
        """Check if context has admin privileges"""
        return 'admin' in self.roles or TaskPermission.ADMIN.value in self.permissions


class TaskAuthorizer:
    """
    Manages task authorization and permission checks.

    This class handles:
    - Handler-level permission requirements
    - User authorization checks
    - Rate limiting per user
    - Queue access control
    """

    def __init__(self):
        """Initialize the authorizer"""
        # Handler permissions: handler_name -> required permissions
        self._handler_permissions: dict[str, set[TaskPermission]] = {}

        # Queue permissions: queue_name -> required permissions
        self._queue_permissions: dict[str, set[TaskPermission]] = {}

        # User rate limits: user_id -> max tasks per minute
        self._user_rate_limits: dict[str, int] = {}
        self._user_rate_windows: dict[str, list[float]] = {}

        # Default permissions for unauthenticated users
        self._allow_anonymous = False
        self._anonymous_handlers: set[str] = set()

        # Admin-only handlers
        self._admin_handlers: set[str] = set()

        logger.info("Task authorizer initialized")

    def register_handler_permissions(self,
                                    handler: str,
                                    permissions: list[TaskPermission],
                                    admin_only: bool = False) -> None:
        """
        Register permission requirements for a handler.

        Args:
            handler: Handler name
            permissions: Required permissions
            admin_only: Whether handler requires admin access
        """
        self._handler_permissions[handler] = set(permissions)

        if admin_only:
            self._admin_handlers.add(handler)
            logger.info(f"Registered admin-only handler: {handler}")
        else:
            logger.debug(f"Registered handler permissions: {handler} -> {permissions}")

    def register_queue_permissions(self,
                                  queue: str,
                                  permissions: list[TaskPermission]) -> None:
        """
        Register permission requirements for a queue.

        Args:
            queue: Queue name
            permissions: Required permissions
        """
        self._queue_permissions[queue] = set(permissions)
        logger.debug(f"Registered queue permissions: {queue} -> {permissions}")

    def allow_anonymous_handler(self, handler: str) -> None:
        """
        Allow anonymous access to a specific handler.

        Args:
            handler: Handler name
        """
        self._anonymous_handlers.add(handler)
        logger.info(f"Enabled anonymous access for handler: {handler}")

    def set_user_rate_limit(self, user_id: str, max_per_minute: int) -> None:
        """
        Set rate limit for a specific user.

        Args:
            user_id: User ID
            max_per_minute: Maximum tasks per minute
        """
        if not isinstance(max_per_minute, int) or max_per_minute < 0:
            raise ValueError("max_per_minute must be a non-negative integer")

        self._user_rate_limits[user_id] = max_per_minute
        self._user_rate_windows.pop(user_id, None)
        logger.debug(f"Set rate limit for user {user_id}: {max_per_minute}/min")

    def can_submit_task(self,
                       handler: str,
                       queue: str,
                       context: AuthContext) -> tuple[bool, Optional[str]]:
        """
        Check if user can submit a task.

        Args:
            handler: Handler name
            queue: Queue name
            context: Authorization context

        Returns:
            Tuple of (allowed, reason_if_denied)
        """
        # Check if handler allows anonymous access
        if handler in self._anonymous_handlers:
            return True, None

        # Require authentication for non-anonymous handlers
        if not context.is_authenticated:
            return False, "Authentication required"

        # Check configured per-user rate limits before permission bypasses
        if context.user_id in self._user_rate_limits:
            allowed, reason = self._check_user_rate_limit(context.user_id)
            if not allowed:
                return False, reason

        # Check admin-only handlers
        if handler in self._admin_handlers and not context.is_admin:
            return False, f"Handler '{handler}' requires admin privileges"

        # Admins bypass further permission checks
        if context.is_admin:
            return True, None

        # Check handler-specific permissions
        required_perms = self._handler_permissions.get(handler, set())
        if required_perms:
            # Convert permission enums to strings for comparison
            required_perm_values = {p.value if isinstance(p, TaskPermission) else p for p in required_perms}
            user_perms = context.permissions
            missing_perms = required_perm_values - user_perms
            if missing_perms:
                return False, f"Missing required permissions: {missing_perms}"

        # Check queue-specific permissions
        queue_perms = self._queue_permissions.get(queue, set())
        if queue_perms:
            # Convert permission enums to strings for comparison
            queue_perm_values = {p.value if isinstance(p, TaskPermission) else p for p in queue_perms}
            user_perms = context.permissions
            missing_perms = queue_perm_values - user_perms
            if missing_perms:
                return False, f"Missing queue permissions: {missing_perms}"

        return True, None

    def _check_user_rate_limit(self, user_id: str) -> tuple[bool, Optional[str]]:
        """Apply the configured per-minute submission limit for a user."""
        max_per_minute = self._user_rate_limits.get(user_id)
        if max_per_minute is None:
            return True, None

        now = time.monotonic()
        window_start = now - 60
        timestamps = [
            timestamp
            for timestamp in self._user_rate_windows.get(user_id, [])
            if timestamp > window_start
        ]

        if len(timestamps) >= max_per_minute:
            self._user_rate_windows[user_id] = timestamps
            return False, f"Rate limit exceeded: {max_per_minute} tasks per minute"

        timestamps.append(now)
        self._user_rate_windows[user_id] = timestamps
        return True, None

    def can_cancel_task(self,
                       task_owner: Optional[str],
                       context: AuthContext) -> tuple[bool, Optional[str]]:
        """
        Check if user can cancel a task.

        Args:
            task_owner: Task owner user ID
            context: Authorization context

        Returns:
            Tuple of (allowed, reason_if_denied)
        """
        if not context.is_authenticated:
            return False, "Authentication required"

        # Admins can cancel any task
        if context.is_admin:
            return True, None

        # Users can cancel their own tasks
        if task_owner == context.user_id:
            return True, None

        # Check if user has cancel permission
        if TaskPermission.CANCEL.value in context.permissions:
            return True, None

        return False, "Not authorized to cancel this task"

    def can_view_task(self,
                      task_owner: Optional[str],
                      context: AuthContext) -> tuple[bool, Optional[str]]:
        """
        Check if user can view a task.

        Args:
            task_owner: Task owner user ID
            context: Authorization context

        Returns:
            Tuple of (allowed, reason_if_denied)
        """
        # Anonymous tasks can be viewed by anyone
        if task_owner is None:
            return True, None

        if not context.is_authenticated:
            return False, "Authentication required"

        # Admins can view any task
        if context.is_admin:
            return True, None

        # Users can view their own tasks
        if task_owner == context.user_id:
            return True, None

        # Check if user has view permission
        if TaskPermission.VIEW.value in context.permissions:
            return True, None

        return False, "Not authorized to view this task"

    def filter_handlers_for_user(self,
                                handlers: list[str],
                                context: AuthContext) -> list[str]:
        """
        Filter list of handlers to only those user can access.

        Args:
            handlers: List of all handlers
            context: Authorization context

        Returns:
            Filtered list of handlers
        """
        allowed = []

        for handler in handlers:
            # Check anonymous handlers
            if handler in self._anonymous_handlers:
                allowed.append(handler)
                continue

            # Skip if not authenticated
            if not context.is_authenticated:
                continue

            # Check admin handlers
            if handler in self._admin_handlers:
                if context.is_admin:
                    allowed.append(handler)
                continue

            # Check handler permissions
            required_perms = self._handler_permissions.get(handler, set())
            if not required_perms:
                # No specific permissions required
                allowed.append(handler)
            elif required_perms.issubset(context.permissions):
                # User has all required permissions
                allowed.append(handler)

        return allowed

    def validate_payload_for_handler(self,
                                    handler: str,
                                    payload: Any,
                                    context: AuthContext) -> tuple[bool, Optional[str]]:
        """
        Validate that payload is appropriate for handler and user.

        This can be extended to check:
        - Payload size limits per user/handler
        - Required fields
        - Forbidden fields based on permissions

        Args:
            handler: Handler name
            payload: Task payload
            context: Authorization context

        Returns:
            Tuple of (valid, error_message)
        """
        # Example: Check payload size for non-admin users
        if not context.is_admin:
            import json
            try:
                payload_size = len(json.dumps(payload))
                # Non-admin users limited to 100KB payloads
                if payload_size > 102400:
                    return False, f"Payload too large for non-admin user: {payload_size} bytes"
            except Exception as e:
                logger.debug(f"Payload size computation failed in scheduler authorization: error={e}")

        # Example: Some handlers might require specific fields
        # This would be configured per handler
        # For now, just return valid

        return True, None


# Global authorizer instance
_authorizer: Optional[TaskAuthorizer] = None


def get_authorizer() -> TaskAuthorizer:
    """Get the global task authorizer"""
    global _authorizer
    if _authorizer is None:
        _authorizer = TaskAuthorizer()
    return _authorizer


def require_auth(permissions: list[TaskPermission] = None,
                admin_only: bool = False):
    """
    Decorator to mark a handler as requiring authentication.

    Args:
        permissions: Required permissions
        admin_only: Whether handler requires admin access

    Example:
        @require_auth(permissions=[TaskPermission.SUBMIT])
        async def process_sensitive_data(data):
            # Process data
            return {"status": "completed"}
    """
    def decorator(func):
        # Register permissions for this handler
        handler_name = f"{func.__module__}.{func.__name__}"
        authorizer = get_authorizer()
        authorizer.register_handler_permissions(
            handler_name,
            permissions or [],
            admin_only=admin_only
        )
        return func
    return decorator


def allow_anonymous(func):
    """
    Decorator to mark a handler as allowing anonymous access.

    Example:
        @allow_anonymous
        async def public_task(data):
            # Process public data
            return {"status": "completed"}
    """
    handler_name = f"{func.__module__}.{func.__name__}"
    authorizer = get_authorizer()
    authorizer.allow_anonymous_handler(handler_name)
    return func
