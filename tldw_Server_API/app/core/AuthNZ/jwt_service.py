# jwt_service.py
# Description: JWT token service with persistent secret management
#
# Imports
import hashlib
import hmac
import threading
from datetime import datetime, timedelta, timezone
from typing import Any, Optional
from uuid import uuid4

#
# 3rd-party imports
from jose import JWTError, jwt
from jose.exceptions import ExpiredSignatureError, JWTClaimsError
from loguru import logger

from tldw_Server_API.app.core.AuthNZ.crypto_utils import (
    derive_hmac_key,
    derive_hmac_key_candidates,
)
from tldw_Server_API.app.core.AuthNZ.exceptions import ConfigurationError, InvalidTokenError, TokenExpiredError

_JWT_SERVICE_NONCRITICAL_EXCEPTIONS = (
    OSError,
    ValueError,
    TypeError,
    KeyError,
    RuntimeError,
    AttributeError,
    ConnectionError,
    TimeoutError,
    JWTError,
    JWTClaimsError,
    ExpiredSignatureError,
    ConfigurationError,
    InvalidTokenError,
    TokenExpiredError,
)

#
# Local imports
from tldw_Server_API.app.core.AuthNZ.settings import Settings, get_settings

#######################################################################################################################
#
# JWT Service Class

class JWTService:
    """Service for creating and verifying JWT tokens with persistent secret management"""

    # Valid token types - reject any token with an unknown type
    VALID_TOKEN_TYPES = frozenset({
        "access",
        "refresh",
        "password_reset",
        "email_verification",
        "magic_link",
        "admin_reauth",
        "service",
    })

    @staticmethod
    def _filter_additional_claims(
        additional_claims: Optional[dict[str, Any]],
        *,
        reserved: set[str],
    ) -> Optional[dict[str, Any]]:
        """Strip reserved claim keys from additional_claims to prevent overrides."""
        if not additional_claims:
            return None
        filtered = {
            key: value
            for key, value in additional_claims.items()
            if key not in reserved
        }
        return filtered or None

    @staticmethod
    def _extract_refresh_rotation_metadata(payload: dict[str, Any]) -> tuple[str, datetime]:
        """Return the prior refresh token JTI and expiry needed for revocation."""
        old_jti = payload.get("jti")
        old_exp = payload.get("exp")
        if not old_jti or not isinstance(old_exp, (int, float)):
            raise InvalidTokenError("Refresh token rotation failed: missing prior token metadata")
        return str(old_jti), datetime.fromtimestamp(old_exp, tz=timezone.utc)

    async def _revoke_rotated_refresh_token(self, *, payload: dict[str, Any], user_id: int) -> None:
        """Persist revocation for the prior refresh token before returning rotated credentials."""
        try:
            old_jti, exp_dt = self._extract_refresh_rotation_metadata(payload)
            from tldw_Server_API.app.core.AuthNZ.token_blacklist import get_token_blacklist as _get_bl

            bl = _get_bl()
            try:
                bl.hint_blacklisted(old_jti, exp_dt)
            except _JWT_SERVICE_NONCRITICAL_EXCEPTIONS as hint_exc:
                logger.debug(f"Refresh rotation: hint cache failed: {hint_exc}")

            revoked = await bl.revoke_token(
                jti=old_jti,
                expires_at=exp_dt,
                user_id=user_id,
                token_type="refresh",
                reason="refresh-rotated",
                revoked_by=None,
                ip_address=None,
            )
        except InvalidTokenError:
            raise
        except _JWT_SERVICE_NONCRITICAL_EXCEPTIONS as exc:
            raise InvalidTokenError(f"Refresh token rotation failed: {exc}") from exc

        if not revoked:
            raise InvalidTokenError("Refresh token rotation failed: revocation persistence was not confirmed")

    def __init__(self, settings: Optional[Settings] = None):
        """Initialize JWT service"""
        self.settings = settings or get_settings()

        # Validate we're in multi-user mode
        if self.settings.AUTH_MODE != "multi_user":
            logger.warning("JWTService initialized in single-user mode - JWT features may not work correctly")

        # JWT configuration
        self.algorithm = self.settings.JWT_ALGORITHM
        # Note: We don't cache timedeltas to allow dynamic configuration changes during testing

        # Select signing/verification keys according to algorithm
        alg_upper = (self.algorithm or "").upper()
        if alg_upper.startswith("HS"):
            # Symmetric HMAC
            self._encode_key = self.settings.JWT_SECRET_KEY
            self._decode_key = self.settings.JWT_SECRET_KEY
            self._secondary_decode_key = self.settings.JWT_SECONDARY_SECRET
            if not self._encode_key:
                # In single-user mode, allow initialization for hashing by deriving a per-instance key
                if (self.settings.AUTH_MODE == "single_user" and getattr(self.settings, "SINGLE_USER_API_KEY", None)):
                    try:
                        # Derive a stable surrogate secret from the API key
                        derived = derive_hmac_key(self.settings)
                        # jose accepts str/bytes; use hex string for readability
                        self._encode_key = derived.hex()
                        self._decode_key = self._encode_key
                        logger.debug("JWTService: using derived single-user surrogate key for HS algorithms")
                    except _JWT_SERVICE_NONCRITICAL_EXCEPTIONS:
                        raise ConfigurationError("JWT_SECRET_KEY", "JWT secret key not configured for HS algorithms") from None
                else:
                    raise ConfigurationError("JWT_SECRET_KEY", "JWT secret key not configured for HS algorithms")
        elif alg_upper.startswith("RS") or alg_upper.startswith("ES"):
            # Asymmetric (RSA/ECDSA)
            self._encode_key = self.settings.JWT_PRIVATE_KEY
            # Allow verifying with public if provided, else private
            if self.settings.JWT_PUBLIC_KEY:
                self._decode_key = self.settings.JWT_PUBLIC_KEY
            else:
                # Warn when using private key for verification - unusual configuration
                self._decode_key = self.settings.JWT_PRIVATE_KEY
                if self._decode_key:
                    logger.warning(
                        "JWTService: No JWT_PUBLIC_KEY configured; using private key for verification. "
                        "Consider configuring JWT_PUBLIC_KEY for asymmetric algorithms."
                    )
            self._secondary_decode_key = self.settings.JWT_SECONDARY_PUBLIC_KEY
            if not self._encode_key:
                raise ConfigurationError("JWT_PRIVATE_KEY", f"Private key required for {self.algorithm}")
            if not self._decode_key:
                raise ConfigurationError("JWT_PUBLIC_KEY", f"Public key (or private) required for {self.algorithm}")
        else:
            raise ConfigurationError("JWT_ALGORITHM", f"Unsupported algorithm: {self.algorithm}")

        logger.debug(f"JWTService initialized with algorithm: {self.algorithm}")

    def create_access_token(
        self,
        user_id: int,
        username: str,
        role: str,
        additional_claims: Optional[dict[str, Any]] = None
    ) -> str:
        """
        Create an access token for a user

        Args:
            user_id: User's database ID
            username: User's username
            role: User's role
            additional_claims: Additional claims to include in token

        Returns:
            Encoded JWT access token
        """
        # Calculate expiration dynamically from settings
        expire = datetime.now(timezone.utc) + timedelta(minutes=self.settings.ACCESS_TOKEN_EXPIRE_MINUTES)

        # Build token payload
        payload = {
            "sub": str(user_id),  # Subject (user ID)
            "username": username,
            "role": role,
            "exp": expire,
            "iat": datetime.now(timezone.utc),
            "jti": str(uuid4()),  # JWT ID for tracking
            "type": "access"
        }
        # Optional issuer/audience
        if self.settings.JWT_ISSUER:
            payload["iss"] = self.settings.JWT_ISSUER
        if self.settings.JWT_AUDIENCE:
            payload["aud"] = self.settings.JWT_AUDIENCE

        # Add any additional claims (excluding reserved JWT fields)
        extra_claims = self._filter_additional_claims(
            additional_claims,
            reserved=set(payload.keys()) | {"iss", "aud"},
        )
        if extra_claims:
            payload.update(extra_claims)

        # Encode the token
        try:
            token = jwt.encode(payload, self._encode_key, algorithm=self.algorithm)
            if self.settings.PII_REDACT_LOGS:
                logger.debug("Created access token for authenticated user (details redacted)")
            else:
                logger.debug(f"Created access token for user {username} (ID: {user_id})")
            return token

        except _JWT_SERVICE_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Failed to create access token: {e}")
            raise InvalidTokenError(f"Failed to create token: {e}") from e

    def create_refresh_token(
        self,
        user_id: int,
        username: str,
        additional_claims: Optional[dict[str, Any]] = None
    ) -> str:
        """
        Create a refresh token for a user

        Args:
            user_id: User's database ID
            username: User's username
            additional_claims: Additional claims to include in token

        Returns:
            Encoded JWT refresh token
        """
        # Calculate expiration dynamically from settings
        expire = datetime.now(timezone.utc) + timedelta(days=self.settings.REFRESH_TOKEN_EXPIRE_DAYS)

        # Build token payload (minimal claims for refresh token)
        payload = {
            "sub": str(user_id),
            "username": username,
            "exp": expire,
            "iat": datetime.now(timezone.utc),
            "jti": str(uuid4()),
            "type": "refresh"
        }
        if self.settings.JWT_ISSUER:
            payload["iss"] = self.settings.JWT_ISSUER
        if self.settings.JWT_AUDIENCE:
            payload["aud"] = self.settings.JWT_AUDIENCE

        # Add any additional claims (excluding reserved JWT fields)
        extra_claims = self._filter_additional_claims(
            additional_claims,
            reserved=set(payload.keys()) | {"iss", "aud"},
        )
        if extra_claims:
            payload.update(extra_claims)

        # Encode the token
        try:
            token = jwt.encode(payload, self._encode_key, algorithm=self.algorithm)
            if self.settings.PII_REDACT_LOGS:
                logger.debug("Created refresh token for authenticated user (details redacted)")
            else:
                logger.debug(f"Created refresh token for user {username} (ID: {user_id})")
            return token

        except _JWT_SERVICE_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Failed to create refresh token: {e}")
            raise InvalidTokenError(f"Failed to create token: {e}") from e

    def _create_short_lived_email_token(
        self,
        *,
        token_type: str,
        email: str,
        user_id: Optional[int] = None,
        expires_in_minutes: Optional[int] = None,
        base_claims: Optional[dict[str, Any]] = None,
        additional_claims: Optional[dict[str, Any]] = None,
    ) -> str:
        """Create a short-lived email-delivered token such as a magic link or admin step-up token."""
        if token_type not in self.VALID_TOKEN_TYPES:
            raise InvalidTokenError(f"Unsupported token type: {token_type}")

        ttl_minutes = expires_in_minutes or int(getattr(self.settings, "MAGIC_LINK_EXPIRE_MINUTES", 15))
        expire = datetime.now(timezone.utc) + timedelta(minutes=ttl_minutes)

        payload: dict[str, Any] = {
            "sub": str(user_id) if user_id is not None else str(email),
            "email": str(email).strip().lower(),
            "exp": expire,
            "iat": datetime.now(timezone.utc),
            "jti": str(uuid4()),
            "type": token_type,
        }
        if user_id is not None:
            payload["user_id"] = int(user_id)
        if base_claims:
            payload.update(base_claims)
        if self.settings.JWT_ISSUER:
            payload["iss"] = self.settings.JWT_ISSUER
        if self.settings.JWT_AUDIENCE:
            payload["aud"] = self.settings.JWT_AUDIENCE

        extra_claims = self._filter_additional_claims(
            additional_claims,
            reserved=set(payload.keys()) | {"iss", "aud"},
        )
        if extra_claims:
            payload.update(extra_claims)

        try:
            token = jwt.encode(payload, self._encode_key, algorithm=self.algorithm)
            if self.settings.PII_REDACT_LOGS:
                logger.debug("Created {} token for authenticated user (details redacted)", token_type)
            else:
                safe_identifier = (
                    f"user_id={int(user_id)}"
                    if user_id is not None
                    else "email=redacted"
                )
                logger.debug("Created {} token for {}", token_type, safe_identifier)
            return token
        except _JWT_SERVICE_NONCRITICAL_EXCEPTIONS as e:
            logger.error("Failed to create {} token: {}", token_type, e)
            raise InvalidTokenError(f"Failed to create token: {e}") from e

    def create_magic_link_token(
        self,
        *,
        email: str,
        user_id: Optional[int] = None,
        expires_in_minutes: Optional[int] = None,
        additional_claims: Optional[dict[str, Any]] = None
    ) -> str:
        """
        Create a short-lived magic link token for passwordless login.

        Args:
            email: User email address
            user_id: Optional existing user ID
            expires_in_minutes: Optional override for token expiry
            additional_claims: Additional claims to include in token

        Returns:
            Encoded JWT magic link token
        """
        return self._create_short_lived_email_token(
            token_type="magic_link",
            email=email,
            user_id=user_id,
            expires_in_minutes=expires_in_minutes,
            additional_claims=additional_claims,
        )

    def create_admin_reauth_token(
        self,
        *,
        email: str,
        user_id: Optional[int] = None,
        expires_in_minutes: Optional[int] = None,
        additional_claims: Optional[dict[str, Any]] = None,
    ) -> str:
        """Create a short-lived admin step-up token delivered by email."""
        return self._create_short_lived_email_token(
            token_type="admin_reauth",
            email=email,
            user_id=user_id,
            expires_in_minutes=expires_in_minutes,
            base_claims={"purpose": "admin_reauth"},
            additional_claims=additional_claims,
        )

    async def verify_token_async(self, token: str, token_type: Optional[str] = None) -> dict[str, Any]:
        """
        Verify and decode a JWT token with blacklist checking (async version)

        Args:
            token: JWT token to verify
            token_type: Expected token type ('access' or 'refresh')

        Returns:
            Decoded token payload

        Raises:
            InvalidTokenError: If token is invalid, malformed, or blacklisted
            TokenExpiredError: If token has expired
        """
        try:
            # Decode the token
            # Enforce optional issuer/audience when configured
            decode_kwargs = {
                "algorithms": [self.algorithm],
            }
            if self.settings.JWT_AUDIENCE:
                decode_kwargs["audience"] = self.settings.JWT_AUDIENCE
            if self.settings.JWT_ISSUER:
                decode_kwargs["issuer"] = self.settings.JWT_ISSUER
            try:
                payload = jwt.decode(token, self._decode_key, **decode_kwargs)
            except JWTError as primary_err:
                # Dual-key fallback during rotations
                if getattr(self, "_secondary_decode_key", None):
                    try:
                        payload = jwt.decode(token, self._secondary_decode_key, **decode_kwargs)
                        # SECURITY: Log when secondary key is used - indicates key rotation in progress
                        logger.info(
                            "JWT verified using secondary key - key rotation may be in progress. "
                            "Primary key verification failed, falling back to secondary."
                        )
                    except JWTError as secondary_err:
                        # Both keys failed - chain exceptions to preserve both error contexts
                        logger.debug(f"Token verification failed with both keys: primary={primary_err}, secondary={secondary_err}")
                        raise primary_err from secondary_err
                else:
                    raise

            # Check if token is blacklisted
            jti = payload.get("jti")
            if jti:
                from tldw_Server_API.app.core.AuthNZ.token_blacklist import is_token_blacklisted
                if await is_token_blacklisted(jti):
                    raise InvalidTokenError("Token has been revoked")

            # Validate token type against whitelist
            actual_type = payload.get("type")
            if actual_type and actual_type not in self.VALID_TOKEN_TYPES:
                logger.warning(f"Token has unknown type: {actual_type}")
                raise InvalidTokenError(f"Unknown token type: {actual_type}")

            # Verify token type if specified
            if token_type and actual_type != token_type:
                raise InvalidTokenError(f"Invalid token type. Expected {token_type}, got {actual_type}")

            if self.settings.PII_REDACT_LOGS:
                logger.debug("Token verified successfully for authenticated user (details redacted)")
            else:
                logger.debug(f"Token verified successfully for user ID: {payload.get('sub')}")
            return payload

        except ExpiredSignatureError:
            logger.debug("Token has expired")
            raise TokenExpiredError() from None

        except JWTClaimsError as e:
            logger.warning(f"JWT claims error: {e}")
            raise InvalidTokenError(f"Invalid token claims: {e}") from e

        except JWTError as e:
            logger.warning(f"JWT verification error: {e}")
            raise InvalidTokenError(f"Invalid token: {e}") from e

        except _JWT_SERVICE_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Unexpected error verifying token: {e}")
            raise InvalidTokenError(f"Token verification failed: {e}") from e

    def verify_token(self, token: str, token_type: Optional[str] = None) -> dict[str, Any]:
        """
        Verify and decode a JWT token (sync, stateless; no blacklist checks).

        SECURITY WARNING: This method performs ONLY cryptographic signature
        validation and expiry checks. It does NOT check if the token has been
        revoked/blacklisted. Revoked tokens will still pass validation here.

        For security-critical paths that require blacklist enforcement, use
        `verify_token_async()` instead, which checks the token blacklist.

        Use cases for this sync method:
        - Performance-critical read operations where blacklist lag is acceptable
        - Contexts where async is not available and revocation is handled elsewhere
        - Token introspection/debugging (not for authorization decisions)

        Args:
            token: JWT token to verify
            token_type: Expected token type ('access' or 'refresh')

        Returns:
            Decoded token payload

        Raises:
            InvalidTokenError: If token is invalid or malformed
            TokenExpiredError: If token has expired
        """
        try:
            # Decode the token
            decode_kwargs = {
                "algorithms": [self.algorithm],
            }
            if self.settings.JWT_AUDIENCE:
                decode_kwargs["audience"] = self.settings.JWT_AUDIENCE
            if self.settings.JWT_ISSUER:
                decode_kwargs["issuer"] = self.settings.JWT_ISSUER
            try:
                payload = jwt.decode(token, self._decode_key, **decode_kwargs)
            except JWTError as primary_err:
                if getattr(self, "_secondary_decode_key", None):
                    try:
                        payload = jwt.decode(token, self._secondary_decode_key, **decode_kwargs)
                    except JWTError as secondary_err:
                        # Both keys failed - chain exceptions to preserve both error contexts
                        logger.debug(f"Token verification failed with both keys: primary={primary_err}, secondary={secondary_err}")
                        raise primary_err from secondary_err
                else:
                    raise

            # Validate token type against whitelist
            actual_type = payload.get("type")
            if actual_type and actual_type not in self.VALID_TOKEN_TYPES:
                logger.warning(f"Token has unknown type: {actual_type}")
                raise InvalidTokenError(f"Unknown token type: {actual_type}")

            # Verify token type if specified
            if token_type and actual_type != token_type:
                raise InvalidTokenError(f"Invalid token type. Expected {token_type}, got {actual_type}")

            # Note: blacklist enforcement is only supported in verify_token_async()

            if self.settings.PII_REDACT_LOGS:
                logger.debug("Token verified successfully for authenticated user (details redacted)")
            else:
                logger.debug(f"Token verified successfully for user ID: {payload.get('sub')}")
            return payload

        except ExpiredSignatureError:
            logger.debug("Token has expired")
            raise TokenExpiredError() from None

        except JWTClaimsError as e:
            logger.warning(f"JWT claims error: {e}")
            raise InvalidTokenError(f"Invalid token claims: {e}") from e

        except JWTError as e:
            logger.warning(f"JWT verification error: {e}")
            raise InvalidTokenError(f"Invalid token: {e}") from e

        except _JWT_SERVICE_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Unexpected error verifying token: {e}")
            raise InvalidTokenError(f"Token verification failed: {e}") from e

    def decode_access_token(self, token: str) -> dict[str, Any]:
        """
        Decode and verify an access token (SYNC, NO BLACKLIST CHECK).

        SECURITY NOTE: This uses the sync verify_token() which does NOT check
        the token blacklist. Use verify_token_async() for authorization decisions
        where blacklist enforcement is required.

        Args:
            token: Access token to decode

        Returns:
            Decoded token payload

        Raises:
            InvalidTokenError: If token is invalid
            ExpiredTokenError: If token has expired
        """
        return self.verify_token(token, token_type="access")

    def decode_refresh_token(self, token: str) -> dict[str, Any]:
        """
        Decode and verify a refresh token (SYNC, NO BLACKLIST CHECK).

        SECURITY NOTE: This uses the sync verify_token() which does NOT check
        the token blacklist. For refresh token flows, prefer refresh_access_token_async()
        or the /auth/refresh endpoint which enforce blacklist checks.

        Args:
            token: Refresh token to decode

        Returns:
            Decoded token payload

        Raises:
            InvalidTokenError: If token is invalid
            ExpiredTokenError: If token has expired
        """
        return self.verify_token(token, token_type="refresh")

    def verify_password_reset_token(self, token: str) -> dict[str, Any]:
        """
        Verify and decode a password reset token (SYNC).

        NOTE: Uses sync verify_token() without blacklist check. This is acceptable
        for password reset tokens because they are single-use tokens with additional
        database-level validation (token hash lookup, used_at check) that prevents reuse.

        Args:
            token: Password reset token to verify

        Returns:
            Decoded token payload containing user_id and email

        Raises:
            InvalidTokenError: If token is invalid or wrong type
            TokenExpiredError: If token has expired
        """
        return self.verify_token(token, token_type="password_reset")

    def verify_email_verification_token(self, token: str) -> dict[str, Any]:
        """
        Verify and decode an email verification token (SYNC).

        NOTE: Uses sync verify_token() without blacklist check. This is acceptable
        for email verification tokens because they are single-use tokens - the user's
        is_verified flag is set on first use, making subsequent uses ineffective.

        Args:
            token: Email verification token to verify

        Returns:
            Decoded token payload containing user_id and email

        Raises:
            InvalidTokenError: If token is invalid or wrong type
            TokenExpiredError: If token has expired
        """
        return self.verify_token(token, token_type="email_verification")

    def verify_service_token(self, token: str) -> dict[str, Any]:
        """
        Verify and decode a service account token (SYNC, NO BLACKLIST CHECK).

        SECURITY NOTE: This uses the sync verify_token() which does NOT check
        the token blacklist. Service tokens are typically long-lived. If service
        token revocation is required, use verify_token_async() instead or implement
        additional validation at the service level.

        Args:
            token: Service account token to verify

        Returns:
            Decoded token payload containing service name and permissions

        Raises:
            InvalidTokenError: If token is invalid or wrong type
            TokenExpiredError: If token has expired
        """
        return self.verify_token(token, token_type="service")

    def create_virtual_access_token(
        self,
        *,
        user_id: int,
        username: str,
        role: str,
        scope: str = "workflows",
        ttl_minutes: int = 60,
        schedule_id: Optional[str] = None,
        additional_claims: Optional[dict[str, Any]] = None,
    ) -> str:
        """
        Create a short-lived, scoped access token ("virtual key").

        Intended for internal automation (e.g., Workflows scheduler) and
        constrained integrations. Tokens include a required `scope` claim and
        optional `schedule_id` claim for further restriction.

        Args:
            user_id: Subject user id
            username: Display username for diagnostics
            role: User role (admin/user)
            scope: Logical scope string (e.g., 'workflows')
            ttl_minutes: Time-to-live in minutes
            schedule_id: Optional associated schedule id
            additional_claims: Extra claims to include

        Returns:
            Encoded JWT access token (scoped)
        """
        expire = datetime.now(timezone.utc) + timedelta(minutes=max(1, int(ttl_minutes)))
        payload: dict[str, Any] = {
            "sub": str(user_id),
            "username": username,
            "role": role,
            "exp": expire,
            "iat": datetime.now(timezone.utc),
            "jti": str(uuid4()),
            "type": "access",
            "scope": str(scope or "workflows"),
        }
        if schedule_id:
            payload["schedule_id"] = str(schedule_id)
        if self.settings.JWT_ISSUER:
            payload["iss"] = self.settings.JWT_ISSUER
        if self.settings.JWT_AUDIENCE:
            payload["aud"] = self.settings.JWT_AUDIENCE
        extra_claims = self._filter_additional_claims(
            additional_claims,
            reserved=set(payload.keys()) | {"iss", "aud"},
        )
        if extra_claims:
            payload.update(extra_claims)
        try:
            token = jwt.encode(payload, self._encode_key, algorithm=self.algorithm)
            logger.debug(
                f"Created virtual access token scope={payload.get('scope')} user={username} ttl={ttl_minutes}m"
            )
            return token
        except _JWT_SERVICE_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Failed to create virtual access token: {e}")
            raise InvalidTokenError(f"Failed to create token: {e}") from e

    def hash_token_candidates(self, token: str) -> list[str]:
        """Return ordered HMAC-SHA256 hashes derived from current and legacy secrets."""
        hashes: list[str] = []
        try:
            key_candidates = derive_hmac_key_candidates(self.settings)
        except _JWT_SERVICE_NONCRITICAL_EXCEPTIONS:
            key_candidates = [derive_hmac_key(self.settings)]
        for key in key_candidates:
            digest = hmac.new(key, token.encode("utf-8"), hashlib.sha256).hexdigest()
            if digest not in hashes:
                hashes.append(digest)
        return hashes

    def hash_token(self, token: str) -> str:
        """
        Create a secure hash of a token for storage using HMAC-SHA256.

        This provides better security than plain SHA256 by using a secret key,
        preventing length extension attacks and providing authentication.

        Note: We use HMAC-SHA256 instead of Argon2 because:
        - Tokens are already high-entropy (cryptographically random)
        - This hash is used for fast lookups on every request
        - Argon2 would add unnecessary latency without security benefit

        Args:
            token: Token to hash

        Returns:
            HMAC-SHA256 hash of the token
        """
        candidates = self.hash_token_candidates(token)
        if not candidates:
            raise ValueError("Unable to derive HMAC hash for token")
        return candidates[0]

    def hash_password_reset_token(self, token: str) -> str:
        """
        Create a computationally expensive hash of a password reset token.

        This uses PBKDF2-HMAC with SHA256 and a high iteration count so that
        brute-forcing reset tokens is significantly more expensive than a
        single SHA256 evaluation.

        Args:
            token: Password reset token to hash

        Returns:
            Hex-encoded PBKDF2-HMAC-SHA256 hash of the token
        """
        # Derive a stable salt from the existing HMAC key material
        base_key = derive_hmac_key(self.settings)
        salt = hmac.new(base_key, b"password-reset-token-salt", hashlib.sha256).digest()
        dk = hashlib.pbkdf2_hmac(
            "sha256",
            token.encode("utf-8"),
            salt,
            200_000,
        )
        return dk.hex()

    def hash_password_reset_token_candidates(self, token: str) -> list[str]:
        """Return ordered PBKDF2-HMAC hashes for a password reset token."""
        hashes: list[str] = []
        try:
            key_candidates = derive_hmac_key_candidates(self.settings)
        except _JWT_SERVICE_NONCRITICAL_EXCEPTIONS:
            key_candidates = [derive_hmac_key(self.settings)]
        for key in key_candidates:
            salt = hmac.new(key, b"password-reset-token-salt", hashlib.sha256).digest()
            dk = hashlib.pbkdf2_hmac(
                "sha256",
                token.encode("utf-8"),
                salt,
                200_000,
            )
            digest = dk.hex()
            if digest not in hashes:
                hashes.append(digest)
        return hashes

    def extract_jti(self, token: str) -> Optional[str]:
        """
        Extract the JTI (JWT ID) from a token without full verification.

        SECURITY NOTE: This method decodes the token WITHOUT signature verification.
        It should ONLY be used for:
        - Logging/debugging purposes
        - Token revocation operations (where the JTI is needed to blacklist)
        - Cases where full verification happens separately

        DO NOT use the extracted JTI for authorization decisions without
        first calling verify_token() or verify_token_async() to validate
        the token's signature and expiry.

        Args:
            token: JWT token

        Returns:
            JTI if present, None otherwise
        """
        try:
            # Decode without verification to get JTI
            unverified = jwt.get_unverified_claims(token)
            jti = unverified.get("jti")
            # Log usage at debug level for security auditing
            if jti:
                logger.debug(
                    "extract_jti called (unverified token decode) - "
                    "ensure token is verified separately for authorization"
                )
            return jti
        except _JWT_SERVICE_NONCRITICAL_EXCEPTIONS as e:
            logger.debug(f"extract_jti failed to decode token: {e}")
            return None


    def create_password_reset_token(
        self,
        user_id: int,
        email: str,
        expires_in_hours: int = 1
    ) -> str:
        """
        Create a password reset token

        Args:
            user_id: User's database ID
            email: User's email
            expires_in_hours: Token validity in hours

        Returns:
            Encoded password reset token
        """
        expire = datetime.now(timezone.utc) + timedelta(hours=expires_in_hours)

        payload = {
            "sub": str(user_id),
            "email": email,
            "exp": expire,
            "iat": datetime.now(timezone.utc),
            "jti": str(uuid4()),
            "type": "password_reset"
        }
        if self.settings.JWT_ISSUER:
            payload["iss"] = self.settings.JWT_ISSUER
        if self.settings.JWT_AUDIENCE:
            payload["aud"] = self.settings.JWT_AUDIENCE

        try:
            token = jwt.encode(payload, self._encode_key, algorithm=self.algorithm)
            if self.settings.PII_REDACT_LOGS:
                logger.info("Created password reset token for authenticated user (details redacted)")
            else:
                logger.info(f"Created password reset token for user {user_id}")
            return token

        except _JWT_SERVICE_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Failed to create password reset token: {e}")
            raise InvalidTokenError(f"Failed to create token: {e}") from e

    def create_email_verification_token(
        self,
        user_id: int,
        email: str,
        expires_in_hours: int = 24
    ) -> str:
        """
        Create an email verification token

        Args:
            user_id: User's database ID
            email: Email to verify
            expires_in_hours: Token validity in hours

        Returns:
            Encoded email verification token
        """
        expire = datetime.now(timezone.utc) + timedelta(hours=expires_in_hours)

        payload = {
            "sub": str(user_id),
            "email": email,
            "exp": expire,
            "iat": datetime.now(timezone.utc),
            "jti": str(uuid4()),
            "type": "email_verification"
        }
        if self.settings.JWT_ISSUER:
            payload["iss"] = self.settings.JWT_ISSUER
        if self.settings.JWT_AUDIENCE:
            payload["aud"] = self.settings.JWT_AUDIENCE

        try:
            token = jwt.encode(payload, self._encode_key, algorithm=self.algorithm)
            if self.settings.PII_REDACT_LOGS:
                logger.info("Created email verification token for authenticated user (details redacted)")
            else:
                logger.info(f"Created email verification token for user {user_id}")
            return token

        except _JWT_SERVICE_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Failed to create email verification token: {e}")
            raise InvalidTokenError(f"Failed to create token: {e}") from e

    def create_service_account_token(
        self,
        service_name: str,
        permissions: list,
        expires_in_days: int = 365
    ) -> str:
        """
        Create a long-lived token for service accounts

        Args:
            service_name: Name of the service
            permissions: List of permissions granted
            expires_in_days: Token validity in days

        Returns:
            Encoded service account token
        """
        expire = datetime.now(timezone.utc) + timedelta(days=expires_in_days)

        payload = {
            "sub": f"service:{service_name}",
            "service": service_name,
            "permissions": permissions,
            "exp": expire,
            "iat": datetime.now(timezone.utc),
            "jti": str(uuid4()),
            "type": "service"
        }
        if self.settings.JWT_ISSUER:
            payload["iss"] = self.settings.JWT_ISSUER
        if self.settings.JWT_AUDIENCE:
            payload["aud"] = self.settings.JWT_AUDIENCE

        try:
            token = jwt.encode(payload, self._encode_key, algorithm=self.algorithm)
            logger.info(f"Created service account token for {service_name}")
            return token

        except _JWT_SERVICE_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Failed to create service account token: {e}")
            raise InvalidTokenError(f"Failed to create token: {e}") from e

    def refresh_access_token(self, refresh_token: str) -> tuple[str, str]:
        """
        Create a new access token from a refresh token (helper).

        Args:
            refresh_token: Valid refresh token

        Returns:
            Tuple of (new_access_token, refresh_token_out)

        Raises:
            InvalidTokenError: If refresh token is invalid
        """
        # NOTE:
        # This helper performs cryptographic validation and minting only.
        # It does NOT update session records or reliably enforce blacklist checks.
        # For production flows, prefer the /auth/refresh endpoint logic which
        # calls SessionManager.refresh_session() to persist rotation and ensure
        # revocation of previous refresh tokens.

        # Verify the refresh token (stateless by default; see note above)
        payload = self.verify_token(refresh_token, token_type="refresh")

        # Optional guard: enforce blacklist check for the presented refresh token.
        # Fail closed if this sync helper is invoked inside a running event loop.
        try:
            import asyncio as _asyncio

            try:
                _asyncio.get_running_loop()
                # If we're already in an event loop, we cannot safely run the async blacklist check.
                raise InvalidTokenError(
                    "refresh_access_token cannot run inside an event loop; "
                    "use refresh_access_token_async or /auth/refresh instead"
                )
            except RuntimeError:
                pass

            jti = payload.get("jti")
            if jti:
                from tldw_Server_API.app.core.AuthNZ.token_blacklist import is_token_blacklisted as _is_bl
                if _asyncio.run(_is_bl(jti)):
                    raise InvalidTokenError("Token has been revoked")
        except InvalidTokenError:
            raise
        except _JWT_SERVICE_NONCRITICAL_EXCEPTIONS as _guard_e:
            logger.debug(f"Refresh helper blacklist guard skipped: {_guard_e}")

        # Extract user information
        user_id = int(payload["sub"])
        username = payload.get("username", "")

        # Fetch role from the configured user database for correctness
        role = payload.get("role", "user")
        try:
            from tldw_Server_API.app.core.AuthNZ.db_config import get_configured_user_database
            user_db = get_configured_user_database(client_id="jwt_service")
            roles = []
            try:
                roles = user_db.get_user_roles(user_id)
            except _JWT_SERVICE_NONCRITICAL_EXCEPTIONS:
                # Fallback to full user fetch if roles accessor differs
                user = user_db.get_user(user_id=user_id)
                roles = (user or {}).get("roles", []) if isinstance(user, dict) else []
            if roles:
                role = "admin" if "admin" in roles else roles[0]
        except _JWT_SERVICE_NONCRITICAL_EXCEPTIONS as e:
            logger.debug(f"JWTService: failed to fetch user role on refresh; using fallback: {e}")

        # Create new access token
        new_access_token = self.create_access_token(
            user_id=user_id,
            username=username,
            role=role
        )

        # Handle optional refresh rotation based on settings
        refresh_out = refresh_token
        try:
            if getattr(self.settings, "ROTATE_REFRESH_TOKENS", False):
                refresh_out = self.create_refresh_token(
                    user_id=user_id,
                    username=username,
                    additional_claims={k: v for k, v in (payload.items()) if k in {"session_id"}}
                )
                import asyncio as _asyncio

                _asyncio.run(self._revoke_rotated_refresh_token(payload=payload, user_id=user_id))
        except InvalidTokenError:
            raise
        except _JWT_SERVICE_NONCRITICAL_EXCEPTIONS as _rot_err:
            raise InvalidTokenError(f"Refresh token rotation failed: {_rot_err}") from _rot_err

        if self.settings.PII_REDACT_LOGS:
            logger.info("Refreshed access token for authenticated user (details redacted)")
        else:
            logger.info(f"Refreshed access token for user {username}")
        return new_access_token, refresh_out

    async def refresh_access_token_async(self, refresh_token: str) -> tuple[str, str]:
        """
        Async-safe refresh helper with blacklist enforcement.

        This method performs cryptographic validation, blacklist checks, and
        token rotation in an async context. Prefer this helper (or the
        /auth/refresh endpoint) when running inside an event loop.
        """
        # Verify the refresh token and enforce blacklist via async path
        payload = await self.verify_token_async(refresh_token, token_type="refresh")

        # Extract user information
        user_id = int(payload["sub"])
        username = payload.get("username", "")

        # Fetch role from the configured user database for correctness
        role = payload.get("role", "user")
        try:
            from tldw_Server_API.app.core.AuthNZ.db_config import get_configured_user_database
            user_db = get_configured_user_database(client_id="jwt_service")
            roles = []
            try:
                roles = user_db.get_user_roles(user_id)
            except _JWT_SERVICE_NONCRITICAL_EXCEPTIONS:
                user = user_db.get_user(user_id=user_id)
                roles = (user or {}).get("roles", []) if isinstance(user, dict) else []
            if roles:
                role = "admin" if "admin" in roles else roles[0]
        except _JWT_SERVICE_NONCRITICAL_EXCEPTIONS as e:
            logger.debug(f"JWTService: failed to fetch user role on refresh (async); using fallback: {e}")

        # Create new access token
        new_access_token = self.create_access_token(
            user_id=user_id,
            username=username,
            role=role
        )

        # Handle optional refresh rotation based on settings
        refresh_out = refresh_token
        try:
            if getattr(self.settings, "ROTATE_REFRESH_TOKENS", False):
                refresh_out = self.create_refresh_token(
                    user_id=user_id,
                    username=username,
                    additional_claims={k: v for k, v in (payload.items()) if k in {"session_id"}}
                )
                await self._revoke_rotated_refresh_token(payload=payload, user_id=user_id)
        except InvalidTokenError:
            raise
        except _JWT_SERVICE_NONCRITICAL_EXCEPTIONS as _rot_err:
            raise InvalidTokenError(f"Refresh token rotation failed: {_rot_err}") from _rot_err

        if self.settings.PII_REDACT_LOGS:
            logger.info("Refreshed access token for authenticated user (details redacted)")
        else:
            logger.info(f"Refreshed access token for user {username}")
        return new_access_token, refresh_out

    def get_token_remaining_time(self, token: str) -> Optional[int]:
        """
        Get remaining time before token expires

        Args:
            token: JWT token

        Returns:
            Remaining seconds until expiration, None if invalid
        """
        try:
            payload = self.verify_token(token)
            exp = payload.get("exp")
            if exp:
                remaining = exp - datetime.now(timezone.utc).timestamp()
                return max(0, int(remaining))
            return None

        except (InvalidTokenError, TokenExpiredError):
            return None


#######################################################################################################################
#
# Module Functions for convenience

# Global instance with thread-safe initialization
_jwt_service: Optional[JWTService] = None
_jwt_service_lock = threading.Lock()


def get_jwt_service() -> JWTService:
    """Get JWT service singleton instance (thread-safe)"""
    global _jwt_service
    # Fast path - no lock if already initialized
    if _jwt_service is not None:
        return _jwt_service

    # Slow path - acquire lock for initialization
    with _jwt_service_lock:
        # Double-check after acquiring lock
        if _jwt_service is None:
            _jwt_service = JWTService()
        return _jwt_service


def reset_jwt_service() -> None:
    """Reset the cached JWTService singleton (used in tests to pick up new settings)."""
    global _jwt_service
    with _jwt_service_lock:
        _jwt_service = None


def create_access_token(user_id: int, username: str, role: str) -> str:
    """Convenience function to create an access token"""
    return get_jwt_service().create_access_token(user_id, username, role)


def create_refresh_token(user_id: int, username: str) -> str:
    """Convenience function to create a refresh token"""
    return get_jwt_service().create_refresh_token(user_id, username)


def verify_token(token: str, token_type: Optional[str] = None) -> dict[str, Any]:
    """
    Convenience function to verify a token (SYNC, NO BLACKLIST CHECK).

    SECURITY NOTE: This calls the sync verify_token() method which does NOT
    check the token blacklist. Revoked tokens will still pass validation.
    For security-critical authorization decisions, use verify_token_async()
    or the async verify methods in auth dependencies instead.

    Appropriate uses:
    - Token introspection/debugging
    - Extracting claims for non-authorization purposes (e.g., logout cleanup)
    - One-time tokens (password_reset, email_verification) with additional validation
    """
    return get_jwt_service().verify_token(token, token_type)


def hash_token(token: str) -> str:
    """Convenience function to hash a token"""
    return get_jwt_service().hash_token(token)


def hash_token_candidates(token: str) -> list[str]:
    """Convenience function to retrieve hash candidates for a token."""
    return get_jwt_service().hash_token_candidates(token)


#
# End of jwt_service.py
#######################################################################################################################
