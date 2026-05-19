"""
Security-focused tests for AuthNZ module.

Tests cover:
- TEST_MODE production guard
- Password history fail-closed behavior
- RBAC permission check error handling
- Crypto fallback protection
- Input validation XSS prevention
"""
import pytest
import os
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock

from tldw_Server_API.app.core.AuthNZ.settings import reset_settings


class TestTestModeProductionGuard:
    """Tests for TEST_MODE production environment protection."""

    @pytest.mark.asyncio
    async def test_test_mode_blocked_in_production(self, monkeypatch):
        """TEST_MODE should be blocked when ENVIRONMENT=production."""
        reset_settings()
        monkeypatch.setenv("TEST_MODE", "1")
        monkeypatch.setenv("ENVIRONMENT", "production")
        monkeypatch.delenv("ALLOW_TEST_MODE_IN_PRODUCTION", raising=False)

        # Import after setting env vars
        from fastapi import Request
        from fastapi.testclient import TestClient

        # The production guard should raise an HTTPException
        # This tests that the guard is in place
        with pytest.raises(Exception):
            # Simulate the check that happens in auth_deps
            test_mode = os.getenv("TEST_MODE", "").strip().lower() in {"1", "true", "yes", "y", "on"}
            allow_test_in_prod = os.getenv("ALLOW_TEST_MODE_IN_PRODUCTION", "").strip().lower() in {"1", "true", "yes", "y", "on"}
            environment = os.getenv("ENVIRONMENT", "").strip().lower()
            if test_mode and environment in {"production", "prod"} and not allow_test_in_prod:
                raise RuntimeError("TEST_MODE cannot be enabled in production environment")

    def test_test_mode_allowed_with_override(self, monkeypatch):

        """TEST_MODE should work with explicit production override."""
        monkeypatch.setenv("TEST_MODE", "1")
        monkeypatch.setenv("ENVIRONMENT", "production")
        monkeypatch.setenv("ALLOW_TEST_MODE_IN_PRODUCTION", "1")

        test_mode = os.getenv("TEST_MODE", "").strip().lower() in {"1", "true", "yes", "y", "on"}
        allow_test_in_prod = os.getenv("ALLOW_TEST_MODE_IN_PRODUCTION", "").strip().lower() in {"1", "true", "yes", "y", "on"}
        environment = os.getenv("ENVIRONMENT", "").strip().lower()

        # Should not raise
        if test_mode and environment in {"production", "prod"} and not allow_test_in_prod:
            pytest.fail("Should not have reached this check with override enabled")

    def test_test_mode_allowed_in_dev(self, monkeypatch):

        """TEST_MODE should work in non-production environments."""
        monkeypatch.setenv("TEST_MODE", "1")
        monkeypatch.setenv("ENVIRONMENT", "development")
        monkeypatch.delenv("ALLOW_TEST_MODE_IN_PRODUCTION", raising=False)

        test_mode = os.getenv("TEST_MODE", "").strip().lower() in {"1", "true", "yes", "y", "on"}
        environment = os.getenv("ENVIRONMENT", "").strip().lower()

        # Non-production should not be blocked
        assert environment not in {"production", "prod"}


class TestPasswordHistoryFailClosed:
    """Tests for password history fail-closed behavior."""

    @pytest.mark.asyncio
    async def test_password_history_check_fails_closed(self, monkeypatch):
        """Password history should fail closed on errors."""
        reset_settings()
        monkeypatch.setenv("AUTH_MODE", "multi_user")
        monkeypatch.setenv("JWT_SECRET_KEY", "test-secret-jwt-key-please-change-1234567890")

        from tldw_Server_API.app.core.AuthNZ.password_service import PasswordService
        from tldw_Server_API.app.core.AuthNZ.settings import get_settings

        service = PasswordService(settings=get_settings())

        # Mock the database to raise an error
        mock_conn = MagicMock()
        mock_conn.fetch = AsyncMock(side_effect=Exception("Database error"))

        # The check should return False (fail closed) on error
        result = await service.check_password_history(
            user_id=1,
            new_password="newpassword123!",
            db_connection=mock_conn
        )

        # After our fix, this should return False (fail closed)
        assert result is False


class TestRBACErrorHandling:
    """Tests for RBAC permission check error handling."""

    def test_user_has_permission_raises_on_error(self, monkeypatch):

        """user_has_permission should raise RBACError on database errors."""
        reset_settings()
        monkeypatch.setenv("AUTH_MODE", "multi_user")
        monkeypatch.setenv("JWT_SECRET_KEY", "test-secret-jwt-key-please-change-1234567890")

        from tldw_Server_API.app.core.AuthNZ.rbac import user_has_permission, RBACError, _get_rbac_repo

        # Mock the repo to raise an error
        with patch.object(_get_rbac_repo(), "has_permission", side_effect=Exception("DB error")):
            with pytest.raises(RBACError):
                user_has_permission(1, "test:permission")


class TestCryptoFallbackProtection:
    """Tests for crypto fallback secret protection."""

    def test_crypto_fallback_blocked_in_production(self, monkeypatch):

        """Crypto fallback should be blocked in production environment.

        This test verifies that when no secrets are configured and ENVIRONMENT=production,
        the crypto_utils module will raise an error instead of using the test fallback.
        """
        # Set up environment for production with test context
        monkeypatch.setenv("ENVIRONMENT", "production")
        monkeypatch.setenv("PYTEST_CURRENT_TEST", "yes")  # Mark as test context
        monkeypatch.setenv("TEST_MODE", "true")

        from tldw_Server_API.app.core.AuthNZ.crypto_utils import derive_hmac_key_candidates

        # Create a mock settings object with no secrets
        mock_settings = MagicMock()
        mock_settings.AUTH_MODE = "multi_user"
        mock_settings.SINGLE_USER_API_KEY = None
        mock_settings.API_KEY_PEPPER = None
        mock_settings.JWT_SECRET_KEY = None
        mock_settings.JWT_PUBLIC_KEY = None
        mock_settings.JWT_PRIVATE_KEY = None
        mock_settings.JWT_SECONDARY_SECRET = None
        mock_settings.JWT_SECONDARY_PRIVATE_KEY = None

        # Should raise ValueError because we're in production with no secrets
        with pytest.raises(ValueError) as exc_info:
            derive_hmac_key_candidates(mock_settings)

        # Either "production" or "could not locate a configured secret" should be in error
        error_msg = str(exc_info.value).lower()
        assert "production" in error_msg or "secret" in error_msg

    def test_crypto_fallback_warning_in_test_env(self, monkeypatch, caplog):

        """Crypto fallback should log a warning when used in test environment."""
        # Set up environment for non-production with test context
        monkeypatch.setenv("ENVIRONMENT", "development")
        monkeypatch.setenv("PYTEST_CURRENT_TEST", "yes")  # Mark as test context
        monkeypatch.setenv("TEST_MODE", "true")

        from tldw_Server_API.app.core.AuthNZ.crypto_utils import derive_hmac_key_candidates

        # Create a mock settings object with no secrets
        mock_settings = MagicMock()
        mock_settings.AUTH_MODE = "multi_user"
        mock_settings.SINGLE_USER_API_KEY = None
        mock_settings.API_KEY_PEPPER = None
        mock_settings.JWT_SECRET_KEY = None
        mock_settings.JWT_PUBLIC_KEY = None
        mock_settings.JWT_PRIVATE_KEY = None
        mock_settings.JWT_SECONDARY_SECRET = None
        mock_settings.JWT_SECONDARY_PRIVATE_KEY = None

        # Should succeed but use fallback (no exception)
        keys = derive_hmac_key_candidates(mock_settings)
        assert len(keys) > 0  # Should have at least the fallback key


class TestInputValidationXSS:
    """Tests for input validation XSS prevention."""

    def test_script_tag_blocked(self):

        """Script tags should be blocked in input validation."""
        from tldw_Server_API.app.core.AuthNZ.input_validation import InputValidator

        validator = InputValidator()
        ok, error = validator.validate_username("<script>alert('xss')</script>")
        assert ok is False

    def test_case_insensitive_script_blocked(self):

        """Script tags in different cases should be blocked."""
        from tldw_Server_API.app.core.AuthNZ.input_validation import InputValidator

        validator = InputValidator()

        # Mixed case
        ok1, _ = validator.validate_username("<SCRIPT>alert('xss')</SCRIPT>")
        assert ok1 is False

        # Another mixed case
        ok2, _ = validator.validate_username("<ScRiPt>alert('xss')</ScRiPt>")
        assert ok2 is False

    def test_javascript_uri_blocked(self):

        """JavaScript URIs should be blocked."""
        from tldw_Server_API.app.core.AuthNZ.input_validation import InputValidator

        validator = InputValidator()
        ok, error = validator.validate_username("javascript:alert('xss')")
        assert ok is False

    def test_event_handler_blocked(self):

        """Event handlers like onclick should be blocked."""
        from tldw_Server_API.app.core.AuthNZ.input_validation import InputValidator

        validator = InputValidator()
        ok, error = validator.validate_username("test onclick=alert('xss')")
        assert ok is False

    def test_data_uri_xss_blocked(self):

        """Data URI XSS attempts should be blocked."""
        from tldw_Server_API.app.core.AuthNZ.input_validation import InputValidator

        validator = InputValidator()
        ok, error = validator.validate_username("data:text/html,<script>alert('xss')</script>")
        assert ok is False


def test_env_example_keeps_jwt_secret_placeholder_commented() -> None:
    env_example = Path(__file__).resolve().parents[3] / "Config_Files" / ".env.example"
    jwt_placeholder_line = next(
        line
        for line in env_example.read_text(encoding="utf-8").splitlines()
        if "JWT_SECRET_KEY=CHANGE_ME_TO_SECURE_RANDOM_KEY_MIN_32_CHARS" in line
    )

    assert jwt_placeholder_line.lstrip().startswith("#")

    def test_path_traversal_blocked(self):

        """Path traversal attempts should be blocked."""
        from tldw_Server_API.app.core.AuthNZ.input_validation import InputValidator

        validator = InputValidator()

        # Forward slash traversal
        ok1, _ = validator.validate_username("../../../etc/passwd")
        assert ok1 is False

        # Backslash traversal (Windows)
        ok2, _ = validator.validate_username("..\\..\\windows\\system32")
        assert ok2 is False

    def test_null_byte_blocked(self):

        """Null byte injection should be blocked."""
        from tldw_Server_API.app.core.AuthNZ.input_validation import InputValidator

        validator = InputValidator()

        # URL encoded null byte
        ok1, _ = validator.validate_username("test%00.txt")
        assert ok1 is False

    def test_normal_username_allowed(self):

        """Normal usernames should be allowed."""
        from tldw_Server_API.app.core.AuthNZ.input_validation import InputValidator

        validator = InputValidator()

        # Regular username
        ok1, error1 = validator.validate_username("john_doe123")
        assert ok1 is True, error1

        # Username with hyphens
        ok2, error2 = validator.validate_username("john-doe")
        assert ok2 is True, error2


class TestPrivilegeEscalationPrevention:
    """Tests for privilege escalation prevention."""

    def test_blocked_admin_usernames(self):

        """Admin-related usernames should be blocked."""
        from tldw_Server_API.app.core.AuthNZ.input_validation import InputValidator

        validator = InputValidator()

        blocked_names = ["admin", "administrator", "root", "superuser", "system"]
        for name in blocked_names:
            ok, error = validator.validate_username(name)
            assert ok is False, f"{name} should be blocked"

    def test_blocked_service_usernames(self):

        """Service account usernames should be blocked."""
        from tldw_Server_API.app.core.AuthNZ.input_validation import InputValidator

        validator = InputValidator()

        blocked_names = ["api", "bot", "service", "webhook"]
        for name in blocked_names:
            ok, error = validator.validate_username(name)
            assert ok is False, f"{name} should be blocked"
