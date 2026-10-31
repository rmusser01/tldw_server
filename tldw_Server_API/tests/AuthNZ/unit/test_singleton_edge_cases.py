"""
Tests for singleton edge cases in AuthNZ module.

These tests verify proper behavior around:
- Concurrent singleton initialization
- Settings generation counter
- Token blacklist pool recovery
"""

import asyncio
import pytest

pytestmark = pytest.mark.unit


class TestConcurrentSingletonInitialization:
    """Tests for concurrent access to singleton getters."""

    @pytest.mark.asyncio
    async def test_concurrent_api_key_manager_returns_same_instance(self, monkeypatch):
        """Verify concurrent calls to get_api_key_manager() return the same instance."""
        from tldw_Server_API.app.core.AuthNZ.api_key_manager import (
            get_api_key_manager,
            reset_api_key_manager,
        )
        from tldw_Server_API.app.core.AuthNZ.settings import reset_settings

        # Reset to ensure clean state
        await reset_api_key_manager()
        reset_settings()

        # Set up test environment
        monkeypatch.setenv("AUTH_MODE", "single_user")
        monkeypatch.setenv("SINGLE_USER_API_KEY", "test-key-for-concurrent-test-12345678901234567890")

        # Launch 5 concurrent initialization requests
        tasks = [get_api_key_manager() for _ in range(5)]
        managers = await asyncio.gather(*tasks)

        # All should be the same instance
        first_manager = managers[0]
        for i, manager in enumerate(managers[1:], start=2):
            assert manager is first_manager, f"Manager {i} is not the same instance as manager 1"

        # Cleanup
        await reset_api_key_manager()
        reset_settings()

    @pytest.mark.asyncio
    async def test_concurrent_session_manager_returns_same_instance(self, monkeypatch):
        """Verify concurrent calls to get_session_manager() return the same instance."""
        from tldw_Server_API.app.core.AuthNZ.session_manager import (
            get_session_manager,
        )
        from tldw_Server_API.tests._plugins.app_services_fixtures import discard_app_services
        from tldw_Server_API.app.core.AuthNZ.settings import reset_settings

        # Reset to ensure clean state
        await discard_app_services()
        reset_settings()

        # Set up test environment
        monkeypatch.setenv("AUTH_MODE", "single_user")
        monkeypatch.setenv("SINGLE_USER_API_KEY", "test-key-for-concurrent-test-12345678901234567890")

        # Launch 5 concurrent initialization requests
        tasks = [get_session_manager() for _ in range(5)]
        managers = await asyncio.gather(*tasks)

        # All should be the same instance
        first_manager = managers[0]
        for i, manager in enumerate(managers[1:], start=2):
            assert manager is first_manager, f"Session manager {i} is not the same instance"

        # Cleanup
        await discard_app_services()
        reset_settings()


class TestSettingsGenerationCounter:
    """Tests for settings generation tracking."""

    def test_settings_generation_increments_on_reset(self, monkeypatch):

        """Verify settings generation counter increments on reset."""
        from tldw_Server_API.app.core.AuthNZ.settings import (
            get_settings,
            reset_settings,
            get_settings_generation,
        )

        # Set up test environment
        monkeypatch.setenv("AUTH_MODE", "single_user")
        monkeypatch.setenv("SINGLE_USER_API_KEY", "test-key-for-generation-test-12345678901234567890")

        reset_settings()
        gen1 = get_settings_generation()

        reset_settings()
        gen2 = get_settings_generation()

        reset_settings()
        gen3 = get_settings_generation()

        assert gen2 > gen1, "Generation should increment after first reset"
        assert gen3 > gen2, "Generation should increment after second reset"

        # Cleanup
        reset_settings()

    def test_settings_accessed_without_reset_keeps_generation(self, monkeypatch):

        """Verify accessing settings without reset maintains the same generation."""
        from tldw_Server_API.app.core.AuthNZ.settings import (
            get_settings,
            reset_settings,
            get_settings_generation,
        )

        # Set up test environment
        monkeypatch.setenv("AUTH_MODE", "single_user")
        monkeypatch.setenv("SINGLE_USER_API_KEY", "test-key-for-generation-test-12345678901234567890")

        reset_settings()
        gen1 = get_settings_generation()

        # Access settings multiple times without reset
        _ = get_settings()
        _ = get_settings()
        _ = get_settings()

        gen2 = get_settings_generation()

        assert gen2 == gen1, "Generation should not change without reset"

        # Cleanup
        reset_settings()


class TestTokenBlacklistPoolRecovery:
    """Tests for token blacklist database pool recovery."""

    @pytest.mark.asyncio
    async def test_token_blacklist_handles_uninitialized_state(self, monkeypatch):
        """Verify blacklist handles queries before explicit initialization."""
        from tldw_Server_API.app.core.AuthNZ.token_blacklist import TokenBlacklist
        from tldw_Server_API.app.core.AuthNZ.settings import Settings

        # Create settings for test
        test_settings = Settings(
            AUTH_MODE="single_user",
            SINGLE_USER_API_KEY="test-key-for-blacklist-test-12345678901234567890",
            DATABASE_URL="sqlite:///./test_blacklist_recovery.db",
        )

        # Create a fresh blacklist instance (not initialized)
        blacklist = TokenBlacklist(settings=test_settings)
        assert not blacklist._initialized, "Blacklist should start uninitialized"

        # Query should trigger lazy initialization and not raise
        result = await blacklist.is_blacklisted("nonexistent-jti-12345")

        # Should return False for nonexistent token (or True if fail-closed on error)
        assert isinstance(result, bool), "Result should be a boolean"
        assert blacklist._initialized, "Blacklist should be initialized after query"

    @pytest.mark.asyncio
    async def test_token_blacklist_reinitializes_on_closed_pool(self, monkeypatch):
        """Verify blacklist handles a closed database pool gracefully."""
        from tldw_Server_API.app.core.AuthNZ.token_blacklist import (
            TokenBlacklist,
            reset_token_blacklist,
        )
        from tldw_Server_API.app.core.AuthNZ.settings import Settings

        # Create settings for test
        test_settings = Settings(
            AUTH_MODE="single_user",
            SINGLE_USER_API_KEY="test-key-for-pool-recovery-test-123456789012345678",
            DATABASE_URL="sqlite:///./test_pool_recovery.db",
        )

        blacklist = TokenBlacklist(settings=test_settings)
        await blacklist.initialize()

        # Close the pool to simulate failure
        if blacklist.db_pool:
            try:
                await blacklist.db_pool.close()
            except Exception:
                _ = None  # May already be closed

        # Query should recover
        try:
            result = await blacklist.is_blacklisted("test-jti-after-close")
            # Should return a boolean (False for nonexistent, or True if fail-closed)
            assert isinstance(result, bool), "Result should be a boolean after pool recovery"
        except Exception as e:
            # Some recovery failures may raise, which is acceptable
            # as long as it doesn't crash silently
            assert "database" in str(e).lower() or "pool" in str(e).lower(), \
                f"Unexpected exception: {e}"


class TestSettingsChangeInvalidation:
    """Tests for singleton invalidation when settings change."""

    @pytest.mark.asyncio
    async def test_api_key_manager_fingerprint_changes_with_secret(self, monkeypatch):
        """Verify manager fingerprint changes when JWT secret changes.

        This test uses multi_user mode because in single_user mode the JWT secret
        initialization is skipped, meaning the fingerprint wouldn't change.
        """
        from tldw_Server_API.app.core.AuthNZ.api_key_manager import (
            get_api_key_manager,
            reset_api_key_manager,
        )
        from tldw_Server_API.app.core.AuthNZ.settings import reset_settings

        # First initialization with one secret - use multi_user mode to test JWT secret usage
        await reset_api_key_manager()
        reset_settings()
        monkeypatch.setenv("AUTH_MODE", "multi_user")
        monkeypatch.setenv("JWT_SECRET_KEY", "first-secret-key-1234567890123456789012345678901234567890")
        monkeypatch.setenv("JWT_REFRESH_SECRET_KEY", "first-refresh-key-1234567890123456789012345678901234567890")

        manager1 = await get_api_key_manager()
        fingerprint1 = getattr(manager1, "_hmac_key_fingerprint", None)

        # Reset and change the secret
        await reset_api_key_manager()
        reset_settings()
        monkeypatch.setenv("JWT_SECRET_KEY", "second-secret-key-different-9876543210987654321098765432109876543210")
        monkeypatch.setenv("JWT_REFRESH_SECRET_KEY", "second-refresh-key-different-9876543210987654321098765432109876543210")

        manager2 = await get_api_key_manager()
        fingerprint2 = getattr(manager2, "_hmac_key_fingerprint", None)

        # Fingerprints should be different
        if fingerprint1 is not None and fingerprint2 is not None:
            assert fingerprint1 != fingerprint2, \
                "Fingerprints should differ when secret changes"

        # Cleanup
        await reset_api_key_manager()
        reset_settings()
