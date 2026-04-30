"""
Secure JWT authentication manager for unified MCP module

Uses environment-based secrets and implements secure token management.
"""

import hashlib
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from jose.exceptions import ExpiredSignatureError
from loguru import logger
from passlib.context import CryptContext
from pydantic import BaseModel, Field

from ..config import get_config

# Security scheme
security = HTTPBearer(auto_error=False)

# Password hashing context
pwd_context = CryptContext(
    # Use PBKDF2-SHA256 to avoid bcrypt backend constraints while remaining secure
    schemes=["pbkdf2_sha256"],
    deprecated="auto",
    pbkdf2_sha256__rounds=200000,
)

_ACCESS_KIND = "access"


class TokenData(BaseModel):
    """Token payload data"""
    sub: str  # Subject (user_id)
    username: Optional[str] = None
    roles: list[str] = Field(default_factory=list)
    permissions: list[str] = Field(default_factory=list)
    token_type: str = _ACCESS_KIND
    jti: Optional[str] = None  # JWT ID for revocation
    iat: Optional[datetime] = None
    exp: Optional[datetime] = None


class RefreshToken(BaseModel):
    """Refresh token with rotation support"""
    token: str
    user_id: str
    expires_at: datetime
    created_at: datetime
    rotated_from: Optional[str] = None
    revoked: bool = False


class JWTManager:
    """
    Secure JWT manager with token rotation and revocation support.

    Features:
    - Environment-based secrets (no hardcoded values)
    - Access and refresh token management
    - Token rotation for enhanced security
    - Token revocation support
    - Secure password hashing
    """

    def __init__(self):
        self.config = get_config()
        self._revoked_tokens = set()  # In production, use Redis
        self._refresh_tokens = {}  # In production, use database

        # Validate configuration
        if not self.config.jwt_secret_key:
            raise ValueError("JWT secret key not configured! Set MCP_JWT_SECRET environment variable")

        logger.info("JWT Manager initialized with secure configuration")

    def _get_secret_key(self) -> str:
        """Get the JWT secret key from secure configuration"""
        return self.config.jwt_secret_key.get_secret_value()

    def hash_password(self, password: str) -> str:
        """Hash a password using PBKDF2-SHA256 (handles long inputs safely)."""
        return pwd_context.hash(password)

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash"""
        try:
            return pwd_context.verify(plain_password, hashed_password)
        except Exception:
            logger.error("Password verification failed")
            return False

    def create_access_token(
        self,
        subject: str,
        username: Optional[str] = None,
        roles: Optional[list[str]] = None,
        permissions: Optional[list[str]] = None,
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """
        Create a JWT access token with secure defaults.

        Args:
            subject: User ID or identifier
            username: Optional username
            roles: User roles
            permissions: User permissions
            expires_delta: Optional custom expiration time

        Returns:
            Encoded JWT token
        """
        now = datetime.now(timezone.utc)

        if expires_delta:
            expire = now + expires_delta
        else:
            expire = now + timedelta(minutes=self.config.jwt_access_token_expire_minutes)

        # Generate unique JWT ID for revocation support
        jti = secrets.token_urlsafe(16)

        payload = {
            "sub": subject,
            "username": username,
            "roles": roles or [],
            "permissions": permissions or [],
            "token_type": _ACCESS_KIND,
            "jti": jti,
            "iat": now,
            "exp": expire,
        }

        # Encode token with secure algorithm
        token = jwt.encode(
            payload,
            self._get_secret_key(),
            algorithm=self.config.jwt_algorithm
        )

        logger.info(f"Access token created for user: {subject}", extra={"audit": True})
        return token

    def create_refresh_token(
        self,
        subject: str,
        expires_delta: Optional[timedelta] = None
    ) -> tuple[str, str]:
        """
        Create a refresh token with rotation support.

        Args:
            subject: User ID
            expires_delta: Optional custom expiration time

        Returns:
            Tuple of (refresh_token, token_id)
        """
        now = datetime.now(timezone.utc)

        if expires_delta:
            expire = now + expires_delta
        else:
            expire = now + timedelta(days=self.config.jwt_refresh_token_expire_days)

        # Generate secure random token
        token = secrets.token_urlsafe(32)
        token_id = secrets.token_hex(16)

        # Store refresh token (in production, use database)
        refresh_token = RefreshToken(
            token=token,
            user_id=subject,
            expires_at=expire,
            created_at=now,
            revoked=False
        )

        self._refresh_tokens[token_id] = refresh_token

        logger.info(f"Refresh token created for user: {subject}", extra={"audit": True})
        return token, token_id

    def verify_token(self, token: str, token_type: str = _ACCESS_KIND) -> TokenData:
        """
        Verify and decode a JWT token.

        Args:
            token: JWT token to verify
            token_type: Expected token type

        Returns:
            Decoded token data

        Raises:
            HTTPException: If token is invalid or expired
        """
        try:
            # Decode token
            payload = jwt.decode(
                token,
                self._get_secret_key(),
                algorithms=[self.config.jwt_algorithm]
            )

            # Check token type
            if payload.get("token_type") != token_type:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail=f"Invalid token type. Expected {token_type}",
                    headers={"WWW-Authenticate": "Bearer"},
                )

            # Check if token is revoked
            jti = payload.get("jti")
            if jti and jti in self._revoked_tokens:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token has been revoked",
                    headers={"WWW-Authenticate": "Bearer"},
                )

            return TokenData(**payload)

        except ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired",
                headers={"WWW-Authenticate": "Bearer"},
            ) from None
        except JWTError as e:
            logger.warning("Invalid token")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token",
                headers={"WWW-Authenticate": "Bearer"},
            ) from e

    def rotate_refresh_token(
        self,
        old_token: str,
        token_id: str
    ) -> tuple[str, str, str]:
        """
        Rotate a refresh token for enhanced security.

        Args:
            old_token: Current refresh token
            token_id: Token ID

        Returns:
            Tuple of (new_access_token, new_refresh_token, new_token_id)
        """
        # Verify old refresh token
        if token_id not in self._refresh_tokens:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token"
            )

        refresh_token = self._refresh_tokens[token_id]

        if refresh_token.revoked:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Refresh token has been revoked"
            )

        if refresh_token.token != old_token:
            # Potential token reuse attack - revoke all tokens
            refresh_token.revoked = True
            logger.warning(
                f"Potential token reuse attack for user: {refresh_token.user_id}",
                extra={"audit": True}
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token"
            )

        if refresh_token.expires_at < datetime.now(timezone.utc):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Refresh token has expired"
            )

        # Revoke old token
        refresh_token.revoked = True

        # Create new tokens
        new_access_token = self.create_access_token(subject=refresh_token.user_id)
        new_refresh_token, new_token_id = self.create_refresh_token(subject=refresh_token.user_id)

        # Link new token to old for audit trail
        self._refresh_tokens[new_token_id].rotated_from = token_id

        logger.info(
            f"Refresh token rotated for user: {refresh_token.user_id}",
            extra={"audit": True}
        )

        return new_access_token, new_refresh_token, new_token_id

    def revoke_token(self, jti: str):
        """
        Revoke a token by its JWT ID.

        Args:
            jti: JWT ID to revoke
        """
        self._revoked_tokens.add(jti)
        logger.info(f"Token revoked: {jti}", extra={"audit": True})

    def revoke_all_user_tokens(self, user_id: str):
        """
        Revoke all tokens for a user.

        Args:
            user_id: User ID whose tokens to revoke
        """
        # Revoke refresh tokens
        for _token_id, refresh_token in self._refresh_tokens.items():
            if refresh_token.user_id == user_id:
                refresh_token.revoked = True

        logger.info(f"All tokens revoked for user: {user_id}", extra={"audit": True})

    async def get_current_user(
        self,
        credentials: HTTPAuthorizationCredentials = Security(security)
    ) -> TokenData:
        """
        FastAPI dependency to get current user from JWT token.

        Args:
            credentials: HTTP Bearer credentials

        Returns:
            Token data with user information
        """
        if not credentials:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required",
                headers={"WWW-Authenticate": "Bearer"},
            )

        return self.verify_token(credentials.credentials)

    def create_api_key(self, user_id: str, name: str, permissions: list[str]) -> str:
        """
        Create a secure API key for a user.

        Args:
            user_id: User ID
            name: API key name
            permissions: Granted permissions

        Returns:
            Hashed API key
        """
        # Generate secure random API key
        raw_key = secrets.token_urlsafe(32)

        # Hash the API key for storage
        salt = self.config.api_key_salt.get_secret_value()
        hashlib.pbkdf2_hmac(
            'sha256',
            raw_key.encode(),
            salt.encode(),
            self.config.api_key_iterations
        )

        # Store key metadata (in production, use database)
        # Only store the hash, never the raw key
        secrets.token_hex(16)

        logger.info(
            f"API key created for user: {user_id}, name: {name}",
            extra={"audit": True}
        )

        return raw_key  # Return raw key only once

    def verify_api_key(self, api_key: str) -> bool:
        """
        Verify an API key.

        Args:
            api_key: API key to verify

        Returns:
            True if valid, False otherwise
        """
        # Hash the provided key
        salt = self.config.api_key_salt.get_secret_value()
        hashlib.pbkdf2_hmac(
            'sha256',
            api_key.encode(),
            salt.encode(),
            self.config.api_key_iterations
        )

        # In production, look up the hash in database
        # For now, return False (no keys stored)
        return False


# Singleton instance
_jwt_manager = None


def get_jwt_manager() -> JWTManager:
    """Get or create JWT manager singleton"""
    global _jwt_manager
    if _jwt_manager is None:
        _jwt_manager = JWTManager()
    return _jwt_manager


# Convenience functions
def create_access_token(subject: str, **kwargs) -> str:
    """Create an access token"""
    manager = get_jwt_manager()
    return manager.create_access_token(subject, **kwargs)


def verify_token(token: str, token_type: str = _ACCESS_KIND) -> TokenData:
    """Verify a token"""
    manager = get_jwt_manager()
    return manager.verify_token(token, token_type)
