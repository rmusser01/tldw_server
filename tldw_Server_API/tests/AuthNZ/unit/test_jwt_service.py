"""
Unit tests for JWT service.
"""

import pytest
from jose import jwt

from tldw_Server_API.app.core.AuthNZ.jwt_service import JWTService
from tldw_Server_API.app.core.AuthNZ.settings import Settings
from tldw_Server_API.app.core.AuthNZ.exceptions import (
    InvalidTokenError,
    TokenExpiredError
)


class TestJWTServiceUnit:
    """Unit tests for JWT service."""

    def test_create_access_token(self, jwt_service):

        """Test creating an access token."""
        token = jwt_service.create_access_token(
            user_id=1,
            username="testuser",
            role="user"
        )

        assert token is not None
        assert isinstance(token, str)

        # Decode and verify
        payload = jwt.decode(
            token,
            jwt_service.settings.JWT_SECRET_KEY,
            algorithms=[jwt_service.settings.JWT_ALGORITHM]
        )

        assert payload["sub"] == "1"  # JWT service stores user_id as string in 'sub'
        assert payload["username"] == "testuser"
        assert payload["role"] == "user"
        assert payload["type"] == "access"
        assert "exp" in payload
        assert "iat" in payload
        assert "jti" in payload

    def test_create_refresh_token(self, jwt_service):

        """Test creating a refresh token."""
        token = jwt_service.create_refresh_token(user_id=1, username="testuser")

        assert token is not None
        assert isinstance(token, str)

        # Decode and verify
        payload = jwt.decode(
            token,
            jwt_service.settings.JWT_SECRET_KEY,
            algorithms=[jwt_service.settings.JWT_ALGORITHM]
        )

        assert payload["sub"] == "1"  # JWT service stores user_id as string in 'sub'
        assert payload["type"] == "refresh"
        assert "exp" in payload
        assert "iat" in payload
        assert "jti" in payload

    def test_create_admin_reauth_token(self, jwt_service):

        """Test creating a dedicated admin reauthentication token."""
        token = jwt_service.create_admin_reauth_token(
            email="admin@example.com",
            user_id=7,
            expires_in_minutes=5,
        )

        assert token is not None
        assert isinstance(token, str)

        payload = jwt_service.verify_token(token, token_type="admin_reauth")

        assert payload["sub"] == "7"
        assert payload["user_id"] == 7
        assert payload["email"] == "admin@example.com"
        assert payload["type"] == "admin_reauth"
        assert "exp" in payload
        assert "iat" in payload
        assert "jti" in payload

    def test_create_magic_link_token_does_not_log_raw_email(self, jwt_service, monkeypatch):

        """Token creation debug logs should not emit the raw recipient email."""
        jwt_service.settings.PII_REDACT_LOGS = False
        debug_calls: list[tuple[object, ...]] = []

        def _capture_debug(*args):  # noqa: ANN002
            debug_calls.append(args)

        monkeypatch.setattr(
            "tldw_Server_API.app.core.AuthNZ.jwt_service.logger.debug",
            _capture_debug,
        )

        token = jwt_service.create_magic_link_token(
            email="person@example.com",
            user_id=42,
            expires_in_minutes=5,
        )

        assert isinstance(token, str)
        assert debug_calls
        combined = " ".join(str(part) for call in debug_calls for part in call)
        assert "person@example.com" not in combined
        assert "42" in combined or "redacted" in combined.lower()

    def test_decode_access_token_valid(self, jwt_service):

        """Test decoding a valid access token."""
        token = jwt_service.create_access_token(
            user_id=1,
            username="testuser",
            role="user"
        )

        payload = jwt_service.decode_access_token(token)

        assert payload["sub"] == "1"  # JWT service stores user_id as string in 'sub'
        assert payload["username"] == "testuser"
        assert payload["role"] == "user"
        assert payload["type"] == "access"

    def test_decode_access_token_expired(self, jwt_service):

        """Test decoding an expired access token."""
        # Create expired token
        original_expire = jwt_service.settings.ACCESS_TOKEN_EXPIRE_MINUTES
        jwt_service.settings.ACCESS_TOKEN_EXPIRE_MINUTES = -1

        token = jwt_service.create_access_token(
            user_id=1,
            username="testuser",
            role="user"
        )

        jwt_service.settings.ACCESS_TOKEN_EXPIRE_MINUTES = original_expire

        with pytest.raises(TokenExpiredError):
            jwt_service.decode_access_token(token)

    def test_decode_access_token_invalid(self, jwt_service):

        """Test decoding an invalid access token."""
        with pytest.raises(InvalidTokenError):
            jwt_service.decode_access_token("invalid.token.here")

    def test_decode_refresh_token_as_access(self, jwt_service):

        """Test that refresh tokens cannot be used as access tokens."""
        refresh_token = jwt_service.create_refresh_token(user_id=1, username="testuser")

        with pytest.raises(InvalidTokenError, match="Invalid token type"):
            jwt_service.decode_access_token(refresh_token)

    def test_decode_refresh_token_valid(self, jwt_service):

        """Test decoding a valid refresh token."""
        token = jwt_service.create_refresh_token(user_id=1, username="testuser")

        payload = jwt_service.decode_refresh_token(token)

        assert payload["sub"] == "1"  # JWT service stores user_id as string in 'sub'
        assert payload["type"] == "refresh"

    def test_refresh_access_token_raises_when_rotation_blacklist_persistence_fails(self, monkeypatch):

        class _FailingBlacklist:
            async def is_blacklisted(self, *_args, **_kwargs):
                return False

            def hint_blacklisted(self, *_args, **_kwargs):
                return None

            async def revoke_token(self, **_kwargs):
                raise RuntimeError("simulated blacklist persistence failure")

        class _StubUserDB:
            def get_user_roles(self, _user_id):
                return ["user"]

        settings = Settings(
            AUTH_MODE="multi_user",
            JWT_SECRET_KEY="test-key-that-is-at-least-32-characters-long",
            JWT_ALGORITHM="HS256",
            ACCESS_TOKEN_EXPIRE_MINUTES=5,
            REFRESH_TOKEN_EXPIRE_DAYS=7,
            ROTATE_REFRESH_TOKENS=True,
        )
        svc = JWTService(settings=settings)
        refresh = svc.create_refresh_token(user_id=1, username="testuser")

        monkeypatch.setattr(
            "tldw_Server_API.app.core.AuthNZ.token_blacklist.get_token_blacklist",
            lambda: _FailingBlacklist(),
        )
        monkeypatch.setattr(
            "tldw_Server_API.app.core.AuthNZ.db_config.get_configured_user_database",
            lambda client_id=None: _StubUserDB(),
        )

        with pytest.raises(InvalidTokenError, match="rotation"):
            svc.refresh_access_token(refresh)

    def test_token_with_additional_claims(self, jwt_service):

        """Test creating tokens with additional claims."""
        token = jwt_service.create_access_token(
            user_id=1,
            username="testuser",
            role="user",
            additional_claims={
                "session_id": "session-123",
                "custom_claim": "custom_value"
            }
        )

        payload = jwt_service.decode_access_token(token)

        assert payload["session_id"] == "session-123"
        assert payload["custom_claim"] == "custom_value"

    def test_additional_claims_cannot_override_reserved_access_claims(self, jwt_service):

        token = jwt_service.create_access_token(
            user_id=1,
            username="testuser",
            role="user",
            additional_claims={
                "sub": "99",
                "username": "evil",
                "role": "admin",
                "type": "refresh",
                "exp": 0,
                "iat": 0,
                "jti": "override",
                "iss": "evil-issuer",
                "aud": "evil-aud",
                "session_id": "session-123",
            },
        )

        payload = jwt_service.decode_access_token(token)

        assert payload["sub"] == "1"
        assert payload["username"] == "testuser"
        assert payload["role"] == "user"
        assert payload["type"] == "access"
        assert payload["session_id"] == "session-123"
        assert payload.get("iss") is None
        assert payload.get("aud") is None

    def test_additional_claims_cannot_override_reserved_refresh_claims(self, jwt_service):

        token = jwt_service.create_refresh_token(
            user_id=1,
            username="testuser",
            additional_claims={
                "sub": "99",
                "type": "access",
                "exp": 0,
                "iat": 0,
                "jti": "override",
                "iss": "evil-issuer",
                "aud": "evil-aud",
                "session_id": "session-999",
            },
        )

        payload = jwt_service.decode_refresh_token(token)

        assert payload["sub"] == "1"
        assert payload["type"] == "refresh"
        assert payload["session_id"] == "session-999"
        assert payload.get("iss") is None
        assert payload.get("aud") is None

    def test_virtual_access_token_ignores_reserved_claim_overrides(self, jwt_service):

        token = jwt_service.create_virtual_access_token(
            user_id=1,
            username="testuser",
            role="user",
            scope="workflows",
            additional_claims={
                "sub": "99",
                "type": "refresh",
                "scope": "override",
            },
        )

        payload = jwt_service.decode_access_token(token)

        assert payload["sub"] == "1"
        assert payload["type"] == "access"
        assert payload["scope"] == "workflows"

    def test_issuer_audience_enforced(self):

        """Ensure tokens with wrong/missing iss/aud fail and correct ones pass (HS)."""
        settings = Settings(
            AUTH_MODE="multi_user",
            JWT_SECRET_KEY="test-key-that-is-at-least-32-characters-long",
            JWT_ALGORITHM="HS256",
            JWT_ISSUER="tldw.test",
            JWT_AUDIENCE="tldw.clients",
            ACCESS_TOKEN_EXPIRE_MINUTES=5,
            REFRESH_TOKEN_EXPIRE_DAYS=7,
            SESSION_CLEANUP_INTERVAL_HOURS=24,
            SESSION_MAX_AGE_DAYS=30,
            PASSWORD_MIN_LENGTH=8,
            PASSWORD_REQUIRE_UPPERCASE=True,
            PASSWORD_REQUIRE_LOWERCASE=True,
            PASSWORD_REQUIRE_DIGIT=True,
            PASSWORD_REQUIRE_SPECIAL=False,
            REGISTRATION_ENABLED=True,
            REGISTRATION_REQUIRE_CODE=False,
            REGISTRATION_CODES=[],
            DEFAULT_USER_ROLE="user",
            DEFAULT_STORAGE_QUOTA_MB=1000,
            EMAIL_VERIFICATION_REQUIRED=False,
            CORS_ORIGINS=["*"],
            API_PREFIX="/api/v1",
        )
        svc = JWTService(settings=settings)
        good = svc.create_access_token(1, "good", "user")
        # Verify good token passes
        assert svc.decode_access_token(good)["sub"] == "1"
        # Mint a token with wrong audience by temporarily overriding settings
        bad_settings = Settings(
            AUTH_MODE="multi_user",
            JWT_SECRET_KEY=settings.JWT_SECRET_KEY,
            JWT_ALGORITHM="HS256",
            JWT_ISSUER="tldw.test",
            JWT_AUDIENCE="wrong.aud",
            ACCESS_TOKEN_EXPIRE_MINUTES=5,
            REFRESH_TOKEN_EXPIRE_DAYS=7,
            SESSION_CLEANUP_INTERVAL_HOURS=24,
            SESSION_MAX_AGE_DAYS=30,
            PASSWORD_MIN_LENGTH=8,
            PASSWORD_REQUIRE_UPPERCASE=True,
            PASSWORD_REQUIRE_LOWERCASE=True,
            PASSWORD_REQUIRE_DIGIT=True,
            PASSWORD_REQUIRE_SPECIAL=False,
            REGISTRATION_ENABLED=True,
            REGISTRATION_REQUIRE_CODE=False,
            REGISTRATION_CODES=[],
            DEFAULT_USER_ROLE="user",
            DEFAULT_STORAGE_QUOTA_MB=1000,
            EMAIL_VERIFICATION_REQUIRED=False,
            CORS_ORIGINS=["*"],
            API_PREFIX="/api/v1",
        )
        svc_bad = JWTService(settings=bad_settings)
        bad = svc_bad.create_access_token(1, "bad", "user")
        with pytest.raises(InvalidTokenError):
            svc.decode_access_token(bad)
