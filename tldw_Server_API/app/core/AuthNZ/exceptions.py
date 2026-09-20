# exceptions.py
# Description: Custom exception classes for the user registration system
#
# Imports
from datetime import datetime, timezone
from typing import Optional

#######################################################################################################################
#
# Base Exceptions

class UserRegistrationException(Exception):
    """Base exception for user registration system"""
    pass


#######################################################################################################################
#
# Authentication Exceptions

class AuthenticationError(UserRegistrationException):
    """Base authentication exception"""
    pass


class InvalidCredentialsError(AuthenticationError):
    """Invalid username or password"""
    def __init__(self, message: str = "Invalid credentials"):
        super().__init__(message)


class AccountLockedException(AuthenticationError):
    """Account is locked due to failed attempts"""
    def __init__(self, locked_until: datetime, username: Optional[str] = None):
        self.locked_until = locked_until
        self.username = username

        # Calculate remaining lock time
        remaining = (locked_until - datetime.now(timezone.utc)).total_seconds()
        if remaining > 0:
            minutes = int(remaining / 60)
            message = f"Account locked for {minutes} more minutes"
        else:
            message = "Account lock expired, please try again"

        super().__init__(message)


class AccountInactiveError(AuthenticationError):
    """Account has been deactivated"""
    def __init__(self, username: Optional[str] = None):
        self.username = username
        super().__init__("Account is inactive. Please contact an administrator.")


class SessionExpiredError(AuthenticationError):
    """Session has expired"""
    def __init__(self):
        super().__init__("Session expired. Please login again.")


class InvalidTokenError(AuthenticationError):
    """Invalid or malformed token"""
    def __init__(self, detail: Optional[str] = None):
        message = "Invalid token"
        if detail:
            message = f"{message}: {detail}"
        super().__init__(message)


class TokenExpiredError(AuthenticationError):
    """Token has expired"""
    def __init__(self):
        super().__init__("Token has expired. Please refresh or login again.")


#######################################################################################################################
#
# Registration Exceptions

class RegistrationError(UserRegistrationException):
    """Base registration exception"""
    pass


class InvalidRegistrationCodeError(RegistrationError):
    """Invalid or expired registration code"""
    def __init__(self, detail: Optional[str] = None):
        # Generic message for security (don't reveal specifics)
        message = "Invalid registration code"
        if detail:
            # Only include detail in logs, not in user-facing message
            self.detail = detail
        super().__init__(message)


class RegistrationCodeExpiredError(RegistrationError):
    """Registration code has expired"""
    def __init__(self):
        super().__init__("Registration code has expired")


class RegistrationCodeExhaustedError(RegistrationError):
    """Registration code has been fully used"""
    def __init__(self):
        super().__init__("Registration code has been fully used")


class DuplicateUserError(RegistrationError):
    """Username or email already exists"""
    def __init__(self, field: str):
        self.field = field
        if field == "username":
            message = "Username already exists"
        elif field == "email":
            message = "Email already registered"
        else:
            message = f"{field} already exists"
        super().__init__(message)


class RegistrationDisabledError(RegistrationError):
    """Registration is currently disabled"""
    def __init__(self):
        super().__init__("Registration is currently disabled")


class WeakPasswordError(RegistrationError):
    """Password does not meet requirements"""
    def __init__(self, requirements: Optional[str] = None):
        message = "Password does not meet security requirements"
        if requirements:
            message = f"{message}: {requirements}"
        super().__init__(message)


#######################################################################################################################
#
# User Management Exceptions

class UserNotFoundError(UserRegistrationException):
    """User not found in database"""
    def __init__(self, identifier: Optional[str] = None):
        message = f"User not found: {identifier}" if identifier else "User not found"
        super().__init__(message)


class UserAlreadyExistsError(UserRegistrationException):
    """User already exists"""
    def __init__(self, field: str, value: str):
        self.field = field
        self.value = value
        super().__init__(f"User with {field} '{value}' already exists")


class InsufficientPermissionsError(UserRegistrationException):
    """User lacks required permissions"""
    def __init__(self, required_permission: Optional[str] = None):
        if required_permission:
            message = f"Insufficient permissions. Required: {required_permission}"
        else:
            message = "Insufficient permissions for this operation"
        super().__init__(message)


#######################################################################################################################
#
# Storage Exceptions

class StorageError(UserRegistrationException):
    """Base storage exception"""
    pass


class QuotaExceededError(StorageError):
    """Storage quota exceeded"""
    def __init__(self, used_mb: float, quota_mb: int):
        self.used_mb = used_mb
        self.quota_mb = quota_mb
        super().__init__(
            f"Storage quota exceeded: {used_mb:.2f}MB / {quota_mb}MB"
        )


class DirectoryCreationError(StorageError):
    """Failed to create user directory"""
    def __init__(self, path: str, detail: Optional[str] = None):
        self.path = path
        message = f"Failed to create directory: {path}"
        if detail:
            message = f"{message} - {detail}"
        super().__init__(message)


#######################################################################################################################
#
# Database Exceptions

class DatabaseError(UserRegistrationException):
    """Base database exception"""
    pass


class RollbackSignal(Exception):
    """Trusted control-flow signal that requires transaction rollback."""


class DatabaseConcurrencyConflict(DatabaseError):
    """Sanitized deadlock or serialization conflict."""

    def __init__(self):
        super().__init__("Database transaction conflicted. Please retry.")


class ConnectionPoolExhaustedError(DatabaseError):
    """Database connection pool is exhausted"""
    def __init__(self):
        super().__init__("Database connection pool exhausted. Please try again.")


class TransactionError(DatabaseError):
    """Database transaction failed"""
    def __init__(self, operation: str, detail: Optional[str] = None):
        message = f"Transaction failed during: {operation}"
        if detail:
            message = f"{message} - {detail}"
        super().__init__(message)


#######################################################################################################################
#
# Organization/Team Exceptions

class DuplicateOrganizationError(DatabaseError):
    """Raised when attempting to create an organization that already exists."""
    def __init__(self, field: str, value: str):
        self.field = field
        self.value = value
        super().__init__(f"Organization with {field} '{value}' already exists")


class DuplicateTeamError(DatabaseError):
    """Raised when attempting to create a team that already exists within an org."""
    def __init__(self, org_id: int, field: str, value: str):
        self.org_id = org_id
        self.field = field
        self.value = value
        super().__init__(f"Team with {field} '{value}' already exists in org {org_id}")


class DuplicateRoleError(DatabaseError):
    """Raised when attempting to create a role with a duplicate unique name."""
    def __init__(self, name: str):
        self.name = name
        super().__init__(f"Role '{name}' already exists")


class DuplicatePermissionError(DatabaseError):
    """Raised when attempting to create a permission with a duplicate unique name."""
    def __init__(self, name: str):
        self.name = name
        super().__init__(f"Permission '{name}' already exists")


class DatabaseLockError(DatabaseError):
    """Database is locked (SQLite specific)"""
    def __init__(self):
        super().__init__("Database is temporarily locked. Please try again.")


class AuthnzMonitoringError(DatabaseError):
    """Monitoring-related database failure in AuthNZ."""

    def __init__(self, operation: str, detail: Optional[str] = None):
        message = f"AuthNZ monitoring operation failed: {operation}"
        if detail:
            message = f"{message} - {detail}"
        super().__init__(message)
        self.operation = operation


class MigrationError(DatabaseError):
    """Database migration failed"""
    def __init__(self, version: Optional[str] = None, detail: Optional[str] = None):
        message = "Database migration failed"
        if version:
            message = f"{message} at version {version}"
        if detail:
            message = f"{message}: {detail}"
        super().__init__(message)


#######################################################################################################################
#
# Rate Limiting Exceptions

class RateLimitError(UserRegistrationException):
    """Rate limit exceeded"""
    def __init__(self, retry_after: Optional[int] = None):
        self.retry_after = retry_after
        message = "Rate limit exceeded"
        if retry_after:
            message = f"{message}. Retry after {retry_after} seconds"
        super().__init__(message)


#######################################################################################################################
#
# Configuration Exceptions

class ConfigurationError(UserRegistrationException):
    """Configuration error"""
    pass


class InvalidConfigurationError(ConfigurationError):
    """Invalid configuration value"""
    def __init__(self, setting: str, detail: Optional[str] = None):
        message = f"Invalid configuration: {setting}"
        if detail:
            message = f"{message} - {detail}"
        super().__init__(message)


class MissingConfigurationError(ConfigurationError):
    """Required configuration is missing"""
    def __init__(self, setting: str):
        super().__init__(f"Missing required configuration: {setting}")


#######################################################################################################################
#
# Session Management Exceptions

class SessionError(UserRegistrationException):
    """Base session exception"""
    pass


class InvalidSessionError(SessionError):
    """Session is invalid or not found"""
    def __init__(self):
        super().__init__("Invalid or expired session")


class SessionRevokedException(SessionError):
    """Session has been revoked"""
    def __init__(self):
        super().__init__("Session has been revoked")


#######################################################################################################################
#
# Service Exceptions

class ServiceUnavailableError(UserRegistrationException):
    """Service is temporarily unavailable"""
    def __init__(self, service: str, detail: Optional[str] = None):
        message = f"{service} service is temporarily unavailable"
        if detail:
            message = f"{message}: {detail}"
        super().__init__(message)


class ExternalServiceError(UserRegistrationException):
    """External service error"""
    def __init__(self, service: str, detail: Optional[str] = None):
        message = f"External service error: {service}"
        if detail:
            message = f"{message} - {detail}"
        super().__init__(message)


#
# End of exceptions.py
#######################################################################################################################
