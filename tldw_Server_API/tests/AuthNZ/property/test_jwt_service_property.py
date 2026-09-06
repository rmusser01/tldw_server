"""
Property-based tests for the JWT service.
"""

from datetime import datetime, timedelta

import jwt
from hypothesis import given, strategies as st, settings as hypothesis_settings, HealthCheck
from hypothesis.strategies import text, integers

from tldw_Server_API.app.core.AuthNZ.jwt_service import JWTService
from tldw_Server_API.app.core.AuthNZ.settings import Settings


class TestJWTServiceProperty:
    """Property-based tests for JWT service."""

    @given(
        user_id=integers(min_value=1, max_value=2**31 - 1),
        username=text(min_size=1, max_size=50).filter(lambda x: x.strip()),
        role=st.sampled_from(["user", "admin", "moderator"]),
    )
    @hypothesis_settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_access_token_roundtrip(self, jwt_service, user_id, username, role):
        """Tokens created by the service should decode back to the original payload."""
        token = jwt_service.create_access_token(
            user_id=user_id,
            username=username,
            role=role,
        )

        payload = jwt_service.decode_access_token(token)

        assert payload["sub"] == str(user_id)  # Stored as string
        assert payload["username"] == username
        assert payload["role"] == role
        assert payload["type"] == "access"

    @given(user_id=integers(min_value=1, max_value=2**31 - 1))
    @hypothesis_settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_refresh_token_roundtrip(self, jwt_service, user_id):
        """Refresh tokens should decode correctly for arbitrary user IDs."""
        token = jwt_service.create_refresh_token(user_id=user_id, username="testuser")

        payload = jwt_service.decode_refresh_token(token)

        assert payload["sub"] == str(user_id)  # Stored as string
        assert payload["type"] == "refresh"

    @given(
        secret_key=text(min_size=32, max_size=64),
        algorithm=st.sampled_from(["HS256", "HS384", "HS512"]),
    )
    @hypothesis_settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture],
        deadline=None,
    )
    def test_jwt_settings_configuration(self, secret_key, algorithm):
        """Service should operate for a range of symmetric key lengths and algorithms."""
        settings = Settings(
            AUTH_MODE="multi_user",
            JWT_SECRET_KEY=secret_key,
            JWT_ALGORITHM=algorithm,
            ACCESS_TOKEN_EXPIRE_MINUTES=30,
            REFRESH_TOKEN_EXPIRE_DAYS=7,
            SESSION_CLEANUP_INTERVAL_HOURS=24,
            SESSION_MAX_AGE_DAYS=30,
            RATE_LIMIT_MAX_REQUESTS=100,
            RATE_LIMIT_WINDOW_SECONDS=60,
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

        service = JWTService(settings=settings)

        token = service.create_access_token(
            user_id=1,
            username="testuser",
            role="user",
        )

        payload = service.decode_access_token(token)

        assert payload["sub"] == "1"  # jwt service stores user_id as string in 'sub'
        assert payload["username"] == "testuser"

    @given(
        expire_minutes=integers(min_value=1, max_value=60 * 24),  # 1 minute to 1 day
        expire_days=integers(min_value=1, max_value=365),
    )
    @hypothesis_settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_token_expiry_settings(self, expire_minutes, expire_days):
        """Token expirations should track configured lifetimes across varied inputs."""
        settings = Settings(
            AUTH_MODE="multi_user",
            JWT_SECRET_KEY="test-key-that-is-at-least-32-characters-long",
            JWT_ALGORITHM="HS256",
            ACCESS_TOKEN_EXPIRE_MINUTES=expire_minutes,
            REFRESH_TOKEN_EXPIRE_DAYS=expire_days,
            SESSION_CLEANUP_INTERVAL_HOURS=24,
            SESSION_MAX_AGE_DAYS=30,
            RATE_LIMIT_MAX_REQUESTS=100,
            RATE_LIMIT_WINDOW_SECONDS=60,
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

        service = JWTService(settings=settings)

        # Capture time before creating tokens
        now = datetime.utcnow()

        access_token = service.create_access_token(
            user_id=1,
            username="testuser",
            role="user",
        )
        refresh_token = service.create_refresh_token(user_id=1, username="testuser")

        # Decode to check expiry
        access_payload = jwt.decode(
            access_token,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM],
        )
        refresh_payload = jwt.decode(
            refresh_token,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM],
        )

        # Check expiry times are set correctly (use utcfromtimestamp for UTC)
        access_exp = datetime.utcfromtimestamp(access_payload["exp"])
        refresh_exp = datetime.utcfromtimestamp(refresh_payload["exp"])

        # Allow 30 seconds tolerance for test execution time (more lenient)
        expected_access_exp = now + timedelta(minutes=expire_minutes)
        expected_refresh_exp = now + timedelta(days=expire_days)

        # Check access token expiry (within 30 seconds)
        assert abs((access_exp - expected_access_exp).total_seconds()) <= 30, (
            f"Access token expiry mismatch: {access_exp} vs {expected_access_exp}"
        )

        # Check refresh token expiry (within 30 seconds)
        assert abs((refresh_exp - expected_refresh_exp).total_seconds()) <= 30, (
            f"Refresh token expiry mismatch: {refresh_exp} vs {expected_refresh_exp}"
        )
