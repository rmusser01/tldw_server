# auth_schemas.py
# Description: Pydantic schemas for authentication endpoints
#
# Imports
from datetime import datetime
from typing import Any, Optional
from uuid import UUID

#
# 3rd-party imports
from pydantic import BaseModel, EmailStr, Field, field_validator

#
# Local imports

#######################################################################################################################
#
# Request Schemas

class LoginRequest(BaseModel):
    """Login request with username and password"""
    username: str = Field(..., min_length=1, max_length=50, description="Username or email")
    password: str = Field(..., min_length=1, description="User password")

    model_config = {
        "json_schema_extra": {
            "example": {
                "username": "johndoe",
                "password": "SecurePass123!"
            }
        }
    }


class RegisterRequest(BaseModel):
    """User registration request"""
    username: str = Field(
        ...,
        min_length=3,
        max_length=50,
        pattern="^[a-zA-Z0-9_-]+$",
        description="Username (alphanumeric, underscore, hyphen only)"
    )
    email: EmailStr = Field(..., description="Valid email address")
    password: str = Field(
        ...,
        min_length=10,
        description="Password (min 10 chars, must include upper, lower, number, special)"
    )
    registration_code: Optional[str] = Field(
        None,
        description="Registration code if required"
    )

    @field_validator('username')
    @classmethod
    def username_valid(cls, v):
        """Validate username format"""
        if v.lower() in ['admin', 'root', 'system', 'api']:
            raise ValueError('Reserved username')
        return v.lower()

    @field_validator('email')
    @classmethod
    def email_lowercase(cls, v):
        """Normalize email to lowercase"""
        return v.lower()

    model_config = {
        "json_schema_extra": {
            "example": {
                "username": "johndoe",
                "email": "john@example.com",
                "password": "SecurePass123!",
                "registration_code": "ABCD1234EFGH5678"
            }
        }
    }


class RefreshTokenRequest(BaseModel):
    """Token refresh request"""
    refresh_token: str = Field(..., description="Valid refresh token")

    model_config = {
        "json_schema_extra": {
            "example": {
                "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
            }
        }
    }


class UpdateProfileRequest(BaseModel):
    """Request model for updating user profile."""
    email: Optional[EmailStr] = Field(None, description="New email address")

    @field_validator('email')
    @classmethod
    def email_lowercase(cls, v):
        """Normalize email to lowercase"""
        if v:
            return v.lower()
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "email": "newemail@example.com"
            }
        }
    }


class PasswordChangeRequest(BaseModel):
    """Password change request"""
    current_password: str = Field(..., description="Current password")
    new_password: str = Field(
        ...,
        min_length=10,
        description="New password"
    )

    @field_validator('new_password')
    @classmethod
    def passwords_different(cls, v, info):
        """Ensure new password is different"""
        if 'current_password' in info.data and v == info.data['current_password']:
            raise ValueError('New password must be different from current password')
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "current_password": "OldPass123!",
                "new_password": "NewSecurePass456!"
            }
        }
    }


class PasswordResetRequest(BaseModel):
    """Password reset request"""
    email: EmailStr = Field(..., description="Email address for reset link")

    @field_validator('email')
    @classmethod
    def email_lowercase(cls, v):
        """Normalize email to lowercase"""
        return v.lower()

    model_config = {
        "json_schema_extra": {
            "example": {
                "email": "john@example.com"
            }
        }
    }


class PasswordResetConfirm(BaseModel):
    """Password reset confirmation"""
    token: str = Field(..., description="Password reset token")
    new_password: str = Field(
        ...,
        min_length=10,
        description="New password"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "token": "reset-token-here",
                "new_password": "NewSecurePass123!"
            }
        }
    }


class MagicLinkRequest(BaseModel):
    """Request a magic link sign-in email."""
    email: EmailStr = Field(..., description="Email address to send sign-in link")

    @field_validator('email')
    @classmethod
    def email_lowercase(cls, v):
        """Normalize email to lowercase"""
        return v.lower()

    model_config = {
        "json_schema_extra": {
            "example": {
                "email": "user@example.com"
            }
        }
    }


class MagicLinkVerifyRequest(BaseModel):
    """Verify a magic link token."""
    token: str = Field(..., description="Magic link token")

    model_config = {
        "json_schema_extra": {
            "example": {
                "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
            }
        }
    }


#######################################################################################################################
#
# Response Schemas

class SingleUserSessionResponse(BaseModel):
    """Opaque single-user cookie-session status returned after minting."""

    authenticated: bool = Field(..., description="Whether the session is authenticated")
    expires_at: datetime = Field(..., description="Opaque session expiry timestamp")


class SingleUserSessionLogoutResponse(BaseModel):
    """Single-user cookie-session logout result."""

    authenticated: bool = Field(..., description="Whether a session remains authenticated")

class TokenResponse(BaseModel):
    """JWT token response"""
    access_token: str = Field(..., description="JWT access token")
    refresh_token: str = Field(..., description="JWT refresh token")
    token_type: str = Field(default="bearer", description="Token type")
    expires_in: int = Field(..., description="Access token expiration in seconds")

    model_config = {
        "json_schema_extra": {
            "example": {
                "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                "token_type": "bearer",
                "expires_in": 1800
            }
        }
    }


class MFAChallengeResponse(BaseModel):
    """MFA challenge response for login flows."""
    session_token: str = Field(..., description="Temporary session token for MFA completion")
    mfa_required: bool = Field(default=True, description="Whether MFA completion is required")
    expires_in: int = Field(..., description="MFA session expiration in seconds")
    message: Optional[str] = Field(default=None, description="Optional MFA instructions")

    model_config = {
        "json_schema_extra": {
            "example": {
                "session_token": "mfa-session-token",
                "mfa_required": True,
                "expires_in": 600,
                "message": "MFA required. Submit your TOTP or backup code.",
            }
        }
    }


class UserResponse(BaseModel):
    """User information response"""
    id: int = Field(..., description="User ID")
    uuid: Optional[UUID] = Field(None, description="User UUID")
    username: str = Field(..., description="Username")
    email: str = Field(..., description="Email address")
    role: str = Field(..., description="User role")
    is_active: bool = Field(..., description="Account active status")
    is_verified: bool = Field(..., description="Email verification status")
    created_at: datetime = Field(..., description="Account creation time")
    last_login: Optional[datetime] = Field(None, description="Last login time")
    storage_quota_mb: int = Field(..., description="Storage quota in MB")
    storage_used_mb: float = Field(..., description="Storage used in MB")

    model_config = {
        "from_attributes": True,
        "json_schema_extra": {
            "example": {
                "id": 1,
                "uuid": "123e4567-e89b-12d3-a456-426614174000",
                "username": "johndoe",
                "email": "john@example.com",
                "role": "user",
                "is_active": True,
                "is_verified": True,
                "created_at": "2025-01-14T10:00:00",
                "last_login": "2025-01-14T12:00:00",
                "storage_quota_mb": 5120,
                "storage_used_mb": 1024.5
            }
        }
    }


class DeprecatedUserResponse(UserResponse):
    """User response with deprecation metadata."""

    warning: Optional[str] = Field(
        None,
        description="Deprecation warning identifier",
    )
    successor: Optional[str] = Field(
        None,
        description="Successor endpoint to migrate to",
    )

    model_config = {
        "from_attributes": True,
        "json_schema_extra": {
            "example": {
                "warning": "deprecated_endpoint",
                "successor": "/api/v1/users/me/profile",
                "id": 1,
                "uuid": "123e4567-e89b-12d3-a456-426614174000",
                "username": "johndoe",
                "email": "john@example.com",
                "role": "user",
                "is_active": True,
                "is_verified": True,
                "created_at": "2025-01-14T10:00:00",
                "last_login": "2025-01-14T12:00:00",
                "storage_quota_mb": 5120,
                "storage_used_mb": 1024.5
            }
        }
    }


class SessionResponse(BaseModel):
    """Session information response"""
    id: int = Field(..., description="Session ID")
    ip_address: Optional[str] = Field(None, description="IP address")
    user_agent: Optional[str] = Field(None, description="User agent")
    created_at: datetime = Field(..., description="Session creation time")
    last_activity: datetime = Field(..., description="Last activity time")
    expires_at: datetime = Field(..., description="Session expiration time")

    model_config = {
        "from_attributes": True,
        "json_schema_extra": {
            "example": {
                "id": 1,
                "ip_address": "192.168.1.1",
                "user_agent": "Mozilla/5.0...",
                "created_at": "2025-01-14T10:00:00",
                "last_activity": "2025-01-14T11:30:00",
                "expires_at": "2025-01-14T12:00:00"
            }
        }
    }


class MessageResponse(BaseModel):
    """Simple message response"""
    message: str = Field(..., description="Response message")
    details: Optional[dict[str, Any]] = Field(None, description="Additional details")

    model_config = {
        "json_schema_extra": {
            "example": {
                "message": "Operation successful",
                "details": {"user_id": 1}
            }
        }
    }


class ErrorResponse(BaseModel):
    """Error response"""
    detail: str = Field(..., description="Error message")
    status_code: int = Field(..., description="HTTP status code")
    error_type: Optional[str] = Field(None, description="Error type")

    model_config = {
        "json_schema_extra": {
            "example": {
                "detail": "Invalid credentials",
                "status_code": 401,
                "error_type": "authentication_error"
            }
        }
    }


class RegistrationResponse(BaseModel):
    """Registration success response"""
    message: str = Field(default="Registration successful", description="Success message")
    user_id: int = Field(..., description="New user ID")
    username: str = Field(..., description="Username")
    email: str = Field(..., description="Email address")
    requires_verification: bool = Field(..., description="Email verification required")
    api_key: Optional[str] = Field(
        default=None,
        description="X-API-KEY for this user (shown once, SQLite multi-user)."
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "message": "Registration successful",
                "user_id": 1,
                "username": "johndoe",
                "email": "john@example.com",
                "requires_verification": True,
                "api_key": "tldw_abcd1234..."
            }
        }
    }


class StorageQuotaResponse(BaseModel):
    """Storage quota information"""
    user_id: int = Field(..., description="User ID")
    storage_used_mb: float = Field(..., description="Storage used in MB")
    storage_quota_mb: int = Field(..., description="Storage quota in MB")
    available_mb: float = Field(..., description="Available storage in MB")
    usage_percentage: float = Field(..., description="Usage percentage")

    model_config = {
        "json_schema_extra": {
            "example": {
                "user_id": 1,
                "storage_used_mb": 1024.5,
                "storage_quota_mb": 5120,
                "available_mb": 4095.5,
                "usage_percentage": 20.0
            }
        }
    }


#
# End of auth_schemas.py
#######################################################################################################################
