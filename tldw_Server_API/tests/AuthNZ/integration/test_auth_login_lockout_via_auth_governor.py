"""
Integration tests for login lockouts routed via AuthGovernor.

These tests focus on the HTTP surface of `/api/v1/auth/login` while
exercising the AuthGovernor lockout facade through a stubbed rate
limiter dependency. The underlying database remains Postgres via the
`isolated_test_environment` fixture.
"""

from datetime import datetime, timedelta, timezone
import re

import asyncpg
import pytest
import pytest_asyncio

from tldw_Server_API.tests.helpers.pg_env import get_pg_env
from tldw_Server_API.app.core.AuthNZ.password_service import PasswordService

pytestmark = pytest.mark.integration


_PG = get_pg_env()
TEST_DB_HOST = _PG.host
TEST_DB_PORT = _PG.port
TEST_DB_USER = _PG.user
TEST_DB_PASSWORD = _PG.password


class _StubLimiter:
    """
    Minimal in-memory rate limiter used to drive lockout scenarios.

    The shape of `record_failed_attempt` mirrors the real limiter result:
    `{"attempt_count": int, "remaining_attempts": int, "is_locked": bool}`.
    """

    def __init__(self, threshold: int = 3) -> None:
        self.enabled = True
        self.threshold = threshold
        self._attempts: dict[str, int] = {}
        self._locked_ids: set[str] = set()
        self.checked_identifiers: list[str] = []
        self.failed_identifiers: list[str] = []
        self.reset_identifiers: list[str] = []

    async def check_lockout(self, identifier: str):
        self.checked_identifiers.append(identifier)
        if identifier in self._locked_ids:
            # Provide a synthetic expiry time for HTTP 429 headers
            return True, datetime.now(timezone.utc) + timedelta(minutes=15)
        return False, None

    async def record_failed_attempt(self, *, identifier: str, attempt_type: str):
        # Track attempts per identifier; lock when threshold reached.
        _ = attempt_type
        self.failed_identifiers.append(identifier)
        count = self._attempts.get(identifier, 0) + 1
        self._attempts[identifier] = count
        is_locked = count >= self.threshold
        if is_locked:
            self._locked_ids.add(identifier)
        remaining = 0 if is_locked else max(self.threshold - count, 0)
        return {
            "attempt_count": count,
            "remaining_attempts": remaining,
            "is_locked": is_locked,
        }

    async def reset_failed_attempts(self, identifier: str, attempt_type: str):
        _ = attempt_type
        self.reset_identifiers.append(identifier)
        self._attempts.pop(identifier, None)
        self._locked_ids.discard(identifier)

    def clear_all_locks(self) -> None:

        """Reset lockout state to simulate expiry."""
        self._attempts.clear()
        self._locked_ids.clear()


class TestAuthLoginLockoutViaAuthGovernor:
    """PostgreSQL-backed login lockout flows using AuthGovernor + stub limiter."""

    @pytest_asyncio.fixture(autouse=True)
    async def setup_user_and_limiter(self, isolated_test_environment, monkeypatch):
        # Arrange per-test DB + client from the shared AuthNZ fixture
        client, db_name = isolated_test_environment
        self.client = client

        # Insert a test user directly into the isolated Postgres DB
        conn = await asyncpg.connect(
            host=TEST_DB_HOST,
            port=TEST_DB_PORT,
            user=TEST_DB_USER,
            password=TEST_DB_PASSWORD,
            database=db_name,
        )
        try:
            password_service = PasswordService()
            password = "Lockout@Pass#2025"
            password_hash = password_service.hash_password(password)
            await conn.execute(
                """
                INSERT INTO users (
                    uuid,
                    username,
                    email,
                    password_hash,
                    role,
                    is_active,
                    is_verified,
                    storage_quota_mb,
                    storage_used_mb
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                """,
                "00000000-0000-0000-0000-000000000001",
                "lockout_user",
                "lockout@example.com",
                password_hash,
                "user",
                True,
                True,
                5120,
                0.0,
            )
            self.username = "lockout_user"
            self.password = password
        finally:
            await conn.close()

        # Override the rate limiter dependency to provide a deterministic stub
        from tldw_Server_API.app.main import app as _app
        from tldw_Server_API.app.api.v1.API_Deps import auth_deps

        limiter = _StubLimiter(threshold=3)

        async def _get_stub_limiter():
            return limiter

        _app.dependency_overrides[auth_deps.get_rate_limiter_dep] = _get_stub_limiter
        # Expose limiter for assertions and lockout expiry simulation
        self.limiter = limiter

        # Ensure settings see rate limiting as enabled if consulted elsewhere
        def _fake_settings():
            from tldw_Server_API.app.core.AuthNZ.settings import Settings

            return Settings()

        monkeypatch.setattr(
            "tldw_Server_API.app.core.AuthNZ.settings.get_settings",
            lambda: _fake_settings(),
        )

        yield

        # Cleanup: remove the dependency override
        _app.dependency_overrides.pop(auth_deps.get_rate_limiter_dep, None)

    def test_repeated_invalid_logins_lead_to_lockout(self):

             # First attempt: invalid password, should be 401 (no lockout yet)
        r1 = self.client.post(
            "/api/v1/auth/login",
            data={"username": self.username, "password": "WrongPassword1!"},
        )
        assert r1.status_code == 401

        # Second attempt: still under threshold, expect another 401
        r2 = self.client.post(
            "/api/v1/auth/login",
            data={"username": self.username, "password": "WrongPassword1!"},
        )
        assert r2.status_code == 401

        # Third attempt: threshold reached, AuthGovernor + stub limiter should
        # respond with HTTP 429 and a Retry-After header.
        r3 = self.client.post(
            "/api/v1/auth/login",
            data={"username": self.username, "password": "WrongPassword1!"},
        )
        assert r3.status_code == 429
        body = r3.json()
        assert "Too many failed login attempts" in body.get("detail", "")
        retry_after = int(r3.headers.get("Retry-After", "0"))
        assert retry_after > 0, "Expected positive Retry-After when locked out"

    def test_successful_login_after_lockout_expires(self):

             # Drive the identifier into a locked state via repeated failures.
        for _ in range(3):
            resp = self.client.post(
                "/api/v1/auth/login",
                data={"username": self.username, "password": "WrongPassword1!"},
            )
        assert resp.status_code == 429

        # Simulate lockout expiry by clearing the stub limiter state.
        self.limiter.clear_all_locks()

        # With locks cleared and correct credentials, login should now succeed.
        success = self.client.post(
            "/api/v1/auth/login",
            data={"username": self.username, "password": self.password},
        )
        assert success.status_code == 200
        payload = success.json()
        assert payload.get("token_type") == "bearer"
        assert "access_token" in payload

    def test_client_login_lockout_does_not_cross_login_identifiers(self):
        for _ in range(3):
            response = self.client.post(
                "/api/v1/auth/login",
                data={"username": self.username, "password": "WrongPassword1!"},
            )
        assert response.status_code == 429

        other_identifier = self.client.post(
            "/api/v1/auth/login",
            data={"username": "other_identifier", "password": "WrongPassword1!"},
        )

        assert other_identifier.status_code == 401
        client_identifiers = self.limiter.checked_identifiers + self.limiter.failed_identifiers
        assert client_identifiers
        assert all(
            re.fullmatch(r"login-client-v1:[0-9a-f]{64}", identifier)
            for identifier in client_identifiers
            if identifier != self.username
        )
        assert "testclient" not in client_identifiers
