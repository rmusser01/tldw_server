"""
Centralized test configuration for tldw_server tests.

This module provides secure test configuration without hardcoded secrets.
"""

import os
import secrets
from typing import Dict, Any


class TestConfig:
    """Test configuration with secure defaults."""

    # Generate unique test API key for this test session
    # Can be overridden with TEST_API_KEY environment variable
    # Use a fixed test key for consistency across all tests
    TEST_API_KEY = os.environ.get("TEST_API_KEY", "test-api-key-for-authentication-testing")

    # Generate unique SK key for OpenAI compatibility testing
    TEST_SK_KEY = os.environ.get("TEST_SK_KEY", f"sk-test-{secrets.token_hex(16)}")

    # Test database path
    TEST_DB_PATH = os.environ.get("TEST_DB_PATH", ":memory:")

    # Test authentication mode
    AUTH_MODE = "single_user"

    # Disable CSRF for testing
    CSRF_ENABLED = False

    @classmethod
    def setup_test_environment(cls) -> None:
        """Set up environment variables for testing."""
        os.environ["AUTH_MODE"] = cls.AUTH_MODE
        # Don't set API_BEARER - it causes the wrong authentication path in single-user mode
        # os.environ["API_BEARER"] = cls.TEST_API_KEY
        os.environ["SINGLE_USER_API_KEY"] = cls.TEST_API_KEY
        # Keep SINGLE_USER_TEST_API_KEY aligned for fixtures that reset env between tests.
        os.environ["SINGLE_USER_TEST_API_KEY"] = cls.TEST_API_KEY
        # Bypass global rate limiting during test runs
        os.environ["TEST_MODE"] = "true"
        # Ensure evaluations routes are available for tests that import the full app
        os.environ.setdefault("MINIMAL_TEST_APP", "1")
        os.environ.setdefault("ULTRA_MINIMAL_APP", "0")
        try:
            existing_enable = os.getenv("ROUTES_ENABLE", "")
            enable_parts = [p for p in existing_enable.replace(" ", ",").split(",") if p]
            if "evaluations" not in [p.lower() for p in enable_parts]:
                enable_parts.append("evaluations")
            os.environ["ROUTES_ENABLE"] = ",".join(dict.fromkeys(enable_parts))
        except Exception:
            _ = None
        try:
            existing_disable = os.getenv("ROUTES_DISABLE", "")
            disable_parts = [p for p in existing_disable.replace(" ", ",").split(",") if p]
            disable_parts = [p for p in disable_parts if p.lower() != "evaluations"]
            os.environ["ROUTES_DISABLE"] = ",".join(dict.fromkeys(disable_parts))
        except Exception:
            _ = None

        # Disable CSRF for testing
        try:
            from tldw_Server_API.app.core.AuthNZ.csrf_protection import global_settings
            global_settings['CSRF_ENABLED'] = cls.CSRF_ENABLED
        except ImportError:
            _ = None

    @classmethod
    def get_auth_headers(cls) -> Dict[str, str]:
        """Get authentication headers for API testing."""
        return {"Authorization": f"Bearer {cls.TEST_API_KEY}"}

    @classmethod
    def get_sk_auth_headers(cls) -> Dict[str, str]:
        """Get OpenAI-style authentication headers for API testing."""
        return {"Authorization": f"Bearer {cls.TEST_SK_KEY}"}

    @classmethod
    def reset_settings(cls) -> None:
        """Reset authentication settings."""
        try:
            from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
            reset_settings()
        except ImportError:
            _ = None


# Singleton instance
test_config = TestConfig()
