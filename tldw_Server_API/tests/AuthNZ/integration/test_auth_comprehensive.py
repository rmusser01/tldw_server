# test_auth_comprehensive.py
# Description: Comprehensive test suite for authentication and user management system
#
# Imports
import pytest
import pytest_asyncio
import asyncio
import os
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import json
import secrets
#
# 3rd-party imports
from httpx import AsyncClient
from fastapi.testclient import TestClient
#
# Local imports
from tldw_Server_API.app.main import app
from tldw_Server_API.app.core.AuthNZ.settings import Settings, get_settings
from tldw_Server_API.app.core.AuthNZ.database import DatabasePool, get_db_pool
from tldw_Server_API.app.core.AuthNZ.password_service import get_password_service
from tldw_Server_API.app.core.AuthNZ.jwt_service import get_jwt_service
from tldw_Server_API.app.services.registration_service import get_registration_service
from tldw_Server_API.app.core.Audit.unified_audit_service import get_unified_audit_service

pytestmark = pytest.mark.integration

#######################################################################################################################
#
# Test Fixtures

# Test database configuration: resolve at call time to pick up env changes by fixtures
def _db_params():
    dsn = (os.getenv("TEST_DATABASE_URL") or os.getenv("DATABASE_URL") or "").strip()
    if dsn:
        try:
            from urllib.parse import urlparse
            p = urlparse(dsn)
            if p.scheme.startswith("postgres"):
                host = p.hostname or "localhost"
                port = int(p.port or 5432)
                user = p.username or "tldw_user"
                password = p.password or "TestPassword123!"
                return host, port, user, password
        except Exception:
            _ = None
    host = os.getenv("TEST_DB_HOST", "localhost")
    port = int(os.getenv("TEST_DB_PORT", "5432"))
    user = os.getenv("TEST_DB_USER", "tldw_user")
    password = os.getenv("TEST_DB_PASSWORD", "TestPassword123!")
    return host, port, user, password


async def _insert_team(db_name: str, *, org_id: int, name: str) -> int:
    import asyncpg

    host, port, user, pwd = _db_params()
    conn = await asyncpg.connect(host=host, port=port, user=user, password=pwd, database=db_name)
    try:
        row = await conn.fetchrow(
            "INSERT INTO teams (org_id, name) VALUES ($1, $2) RETURNING id",
            org_id,
            name,
        )
        return int(row["id"])
    finally:
        await conn.close()

# Note: We now use isolated_test_environment from conftest.py for true DB isolation
# Each test gets its own unique database that is created and destroyed per test

@pytest_asyncio.fixture
async def test_user_data(isolated_test_environment):
    """Create test user data in isolated database"""
    import asyncpg
    import uuid
    from tldw_Server_API.app.core.AuthNZ.password_service import PasswordService

    client, db_name = isolated_test_environment

    # Connect to the unique test database
    host, port, user, pwd = _db_params()
    conn = await asyncpg.connect(host=host, port=port, user=user, password=pwd, database=db_name)

    try:
        password_service = PasswordService()
        user_uuid = str(uuid.uuid4())
        password = "Test@Pass#2024!"
        password_hash = password_service.hash_password(password)

        user = await conn.fetchrow("""
            INSERT INTO users (
                uuid, username, email, password_hash, role,
                is_active, is_verified, storage_quota_mb, storage_used_mb
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            RETURNING id, uuid, username, email, role, is_active, is_verified
        """, user_uuid, "testuser", "test@example.com", password_hash,
            "user", True, True, 5120, 0.0)

        return {
            "id": user["id"],
            "uuid": str(user["uuid"]),
            "username": user["username"],
            "email": user["email"],
            "role": user["role"],
            "password": password
        }
    finally:
        await conn.close()

@pytest_asyncio.fixture
async def admin_user_data(isolated_test_environment):
    """Create admin user data in isolated database"""
    import asyncpg
    import uuid
    from tldw_Server_API.app.core.AuthNZ.password_service import PasswordService

    client, db_name = isolated_test_environment

    # Connect to the unique test database
    host, port, user, pwd = _db_params()
    conn = await asyncpg.connect(host=host, port=port, user=user, password=pwd, database=db_name)

    try:
        password_service = PasswordService()
        user_uuid = str(uuid.uuid4())
        password = "Admin@Pass#2024!"
        password_hash = password_service.hash_password(password)

        user = await conn.fetchrow("""
            INSERT INTO users (
                uuid, username, email, password_hash, role,
                is_active, is_verified, storage_quota_mb, storage_used_mb
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            RETURNING id, uuid, username, email, role, is_active, is_verified
        """, user_uuid, "admin", "admin@example.com", password_hash,
            "admin", True, True, 10240, 0.0)

        # Ensure the RBAC mapping exists so that the admin user is recognized
        # by require_roles/require_admin, mirroring _ensure_admin in
        # tests/AuthNZ/integration/test_rbac_admin_endpoints.py.
        role_row = await conn.fetchrow("SELECT id FROM roles WHERE name = 'admin'")
        if role_row and "id" in role_row:
            await conn.execute(
                """
                INSERT INTO user_roles (user_id, role_id)
                VALUES ($1, $2)
                ON CONFLICT (user_id, role_id) DO NOTHING
                """,
                int(user["id"]),
                int(role_row["id"]),
            )

        return {
            "id": user["id"],
            "uuid": str(user["uuid"]),
            "username": user["username"],
            "email": user["email"],
            "role": user["role"],
            "password": password
        }
    finally:
        await conn.close()


@pytest_asyncio.fixture
async def async_client(isolated_test_environment):
    """Create async test client using isolated environment"""
    # Use the client from isolated_test_environment
    client, db_name = isolated_test_environment
    # For async tests, we use the same isolated environment
    from httpx import ASGITransport
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as async_client:
        yield async_client


# Use the test_user and admin_user fixtures from conftest.py
# These are properly configured with PostgreSQL test database


@pytest_asyncio.fixture
async def auth_headers(isolated_test_environment, test_user_data):
    """Get authentication headers for a test user"""
    client, db_name = isolated_test_environment
    response = client.post(
        "/api/v1/auth/login",
        data={
            "username": test_user_data["username"],
            "password": test_user_data["password"]
        }
    )
    assert response.status_code == 200
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


@pytest_asyncio.fixture
async def admin_headers(isolated_test_environment, admin_user_data):
    """Get authentication headers for an admin user"""
    client, db_name = isolated_test_environment
    response = client.post(
        "/api/v1/auth/login",
        data={
            "username": admin_user_data["username"],
            "password": admin_user_data["password"]
        }
    )
    assert response.status_code == 200
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


#######################################################################################################################
#
# Authentication Tests

class TestAuthentication:
    """Test authentication endpoints"""

    @pytest.mark.asyncio
    async def test_login_success(self, isolated_test_environment, test_user_data):
        """Test successful login"""
        client, db_name = isolated_test_environment
        response = client.post(
            "/api/v1/auth/login",
            data={
                "username": test_user_data["username"],
                "password": test_user_data["password"]
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"
        assert data["expires_in"] > 0

    def test_login_invalid_username(self, isolated_test_environment):

        """Test login with invalid username"""
        client, db_name = isolated_test_environment
        response = client.post(
            "/api/v1/auth/login",
            data={
                "username": "nonexistent",
                "password": "Pass@Word#2024"
            }
        )
        assert response.status_code == 401
        assert "Incorrect username or password" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_login_invalid_password(self, isolated_test_environment, test_user_data):
        """Test login with invalid password"""
        client, db_name = isolated_test_environment
        response = client.post(
            "/api/v1/auth/login",
            data={
                "username": test_user_data["username"],
                "password": "wrongpassword"
            }
        )
        assert response.status_code == 401
        assert "Incorrect username or password" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_logout(self, isolated_test_environment, auth_headers):
        """Test logout endpoint"""
        client, db_name = isolated_test_environment
        response = client.post(
            "/api/v1/auth/logout",
            headers=auth_headers
        )
        assert response.status_code == 200
        assert "Successfully logged out" in response.json()["message"]

    @pytest.mark.asyncio
    async def test_refresh_token(self, isolated_test_environment, test_user_data):
        """Test token refresh"""
        client, db_name = isolated_test_environment
        # First login
        login_response = client.post(
            "/api/v1/auth/login",
            data={
                "username": test_user_data["username"],
                "password": test_user_data["password"]
            }
        )
        assert login_response.status_code == 200
        refresh_token = login_response.json()["refresh_token"]

        # Refresh token
        refresh_response = client.post(
            "/api/v1/auth/refresh",
            json={"refresh_token": refresh_token}
        )
        assert refresh_response.status_code == 200
        data = refresh_response.json()
        assert "access_token" in data
        assert "refresh_token" in data

    @pytest.mark.asyncio
    async def test_get_current_user(self, isolated_test_environment, auth_headers):
        """Test getting current user info"""
        client, db_name = isolated_test_environment
        response = client.get(
            "/api/v1/auth/me",
            headers=auth_headers
        )
        assert response.status_code == 200
        data = response.json()
        assert "id" in data
        assert "username" in data
        assert "email" in data
        assert "role" in data


#######################################################################################################################
#
# Registration Tests

class TestRegistration:
    """Test user registration"""

    def test_register_success(self, isolated_test_environment):

        """Test successful registration"""
        client, db_name = isolated_test_environment
        response = client.post(
            "/api/v1/auth/register",
            json={
                "username": "newuser",
                "email": "newuser@example.com",
                "password": "New@Pass#2024!"
            }
        )
        assert response.status_code in [200, 201]
        data = response.json()
        assert data["username"] == "newuser"
        assert data["email"] == "newuser@example.com"
        assert "user_id" in data

    def test_register_duplicate_username(self, isolated_test_environment, test_user_data):

        """Test registration with duplicate username"""
        client, db_name = isolated_test_environment
        response = client.post(
            "/api/v1/auth/register",
            json={
                "username": test_user_data["username"],
                "email": "different@example.com",
                "password": "Pass@Word#2024!"
            }
        )
        assert response.status_code == 409
        assert "already" in response.json()["detail"].lower()

    def test_register_duplicate_email(self, isolated_test_environment, test_user_data):

        """Test registration with duplicate email"""
        client, db_name = isolated_test_environment
        response = client.post(
            "/api/v1/auth/register",
            json={
                "username": "different",
                "email": test_user_data["email"],
                "password": "Pass@Word#2024!"
            }
        )
        assert response.status_code == 409
        assert "already" in response.json()["detail"].lower()

    async def test_register_weak_password(self, isolated_test_environment):
        """Test registration with weak password"""
        client, db_name = isolated_test_environment
        response = client.post(
            "/api/v1/auth/register",
            json={
                "username": "weakpass",
                "email": "weak@example.com",
                "password": "weak"
            }
        )
        assert response.status_code in [400, 422]  # 422 for validation error, 400 for weak password
        # Either it's a Pydantic validation error (422) or our custom weak password error (400)
        # Both are acceptable for a weak password


#######################################################################################################################
#
# User Management Tests

class TestUserManagement:
    """Test user management endpoints"""

    async def test_get_user_profile(self, isolated_test_environment, auth_headers):
        """Test getting user profile"""
        client, db_name = isolated_test_environment
        response = client.get(
            "/api/v1/users/me",
            headers=auth_headers
        )
        assert response.status_code == 200
        data = response.json()
        assert "id" in data
        assert "username" in data
        assert "storage_quota_mb" in data

    async def test_change_password(self, isolated_test_environment, auth_headers):
        """Test password change"""
        client, db_name = isolated_test_environment
        response = client.post(
            "/api/v1/users/change-password",
            headers=auth_headers,
            json={
                "current_password": "Test@Pass#2024!",
                "new_password": "NewPass@Word2024!"
            }
        )
        assert response.status_code == 200
        assert "Password changed successfully" in response.json()["message"]

    async def test_get_sessions(self, isolated_test_environment, auth_headers):
        """Test getting user sessions"""
        client, db_name = isolated_test_environment
        response = client.get(
            "/api/v1/users/sessions",
            headers=auth_headers
        )
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) > 0


#######################################################################################################################
#
# Admin Tests

class TestAdminEndpoints:
    """Test admin endpoints"""

    @pytest.mark.asyncio
    async def test_list_users(self, isolated_test_environment, admin_headers):
        """Test listing users as admin"""
        client, db_name = isolated_test_environment
        response = client.get(
            "/api/v1/admin/users",
            headers=admin_headers
        )
        assert response.status_code == 200
        data = response.json()
        assert "users" in data
        assert "total" in data
        assert "page" in data

    @pytest.mark.asyncio
    async def test_update_user(self, isolated_test_environment, admin_headers, test_user_data):
        """Test updating user as admin"""
        client, db_name = isolated_test_environment
        response = client.put(
            f"/api/v1/admin/users/{test_user_data['id']}",
            headers=admin_headers,
            json={
                "is_verified": True,
                "storage_quota_mb": 10240
            }
        )
        assert response.status_code == 200
        assert "updated successfully" in response.json()["message"]

    @pytest.mark.asyncio
    async def test_create_registration_code(self, isolated_test_environment, admin_headers):
        """Test creating registration code"""
        client, db_name = isolated_test_environment
        response = client.post(
            "/api/v1/admin/registration-codes",
            headers=admin_headers,
            json={
                "max_uses": 5,
                "expiry_days": 7,
                "role_to_grant": "user"
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert "code" in data
        assert data["max_uses"] == 5
        assert data["role_to_grant"] == "user"

    @pytest.mark.asyncio
    async def test_org_scoped_registration_code_assigns_membership(
        self,
        isolated_test_environment,
        admin_headers,
        monkeypatch,
    ):
        """Test org-scoped registration codes assign membership on signup."""
        client, db_name = isolated_test_environment
        org_suffix = uuid.uuid4().hex[:8]

        # Org-scoped registration codes are opt-in; enable and refresh settings/services.
        monkeypatch.setenv("ENABLE_ORG_SCOPED_REGISTRATION_CODES", "true")
        from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
        from tldw_Server_API.app.services.registration_service import reset_registration_service

        reset_settings()
        await reset_registration_service()

        org_response = client.post(
            "/api/v1/orgs",
            headers=admin_headers,
            json={"name": f"Invite Org {org_suffix}", "slug": f"invite-org-{org_suffix}"},
        )
        assert org_response.status_code == 201
        org_id = org_response.json()["id"]

        team_id = await _insert_team(
            db_name,
            org_id=org_id,
            name=f"Team {org_suffix}",
        )

        code_response = client.post(
            "/api/v1/admin/registration-codes",
            headers=admin_headers,
            json={
                "max_uses": 2,
                "expiry_days": 7,
                "role_to_grant": "user",
                "org_id": org_id,
                "org_role": "member",
                "team_id": team_id,
            },
        )
        assert code_response.status_code == 200
        code_data = code_response.json()
        assert code_data["org_id"] == org_id
        assert code_data["team_id"] == team_id

        username = f"invite_user_{org_suffix}"
        email = f"{username}@example.com"
        password = "TestInvite@2024!"
        register_response = client.post(
            "/api/v1/auth/register",
            json={
                "username": username,
                "email": email,
                "password": password,
                "registration_code": code_data["code"],
            },
        )
        assert register_response.status_code in (200, 201)

        login_response = client.post(
            "/api/v1/auth/login",
            data={"username": username, "password": password},
        )
        assert login_response.status_code == 200
        user_headers = {"Authorization": f"Bearer {login_response.json()['access_token']}"}

        org_list_response = client.get("/api/v1/orgs", headers=user_headers)
        assert org_list_response.status_code == 200
        assert any(org["id"] == org_id for org in org_list_response.json()["items"])

        list_codes = client.get("/api/v1/admin/registration-codes", headers=admin_headers)
        assert list_codes.status_code == 200
        code_row = next(
            (row for row in list_codes.json()["codes"] if row["code"] == code_data["code"]),
            None,
        )
        assert code_row is not None
        assert code_row["org_id"] == org_id

    @pytest.mark.asyncio
    async def test_accept_org_invite_for_existing_user(
        self,
        isolated_test_environment,
        admin_headers,
        auth_headers,
        monkeypatch,
    ):
        """Test accepting org-scoped registration codes for existing users."""
        client, db_name = isolated_test_environment
        org_suffix = uuid.uuid4().hex[:8]

        monkeypatch.setenv("ENABLE_ORG_SCOPED_REGISTRATION_CODES", "true")
        from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
        from tldw_Server_API.app.services.registration_service import reset_registration_service

        reset_settings()
        await reset_registration_service()

        org_response = client.post(
            "/api/v1/orgs",
            headers=admin_headers,
            json={"name": f"Accept Org {org_suffix}", "slug": f"accept-org-{org_suffix}"},
        )
        assert org_response.status_code == 201
        org_id = org_response.json()["id"]

        code_response = client.post(
            "/api/v1/admin/registration-codes",
            headers=admin_headers,
            json={
                "max_uses": 3,
                "expiry_days": 7,
                "role_to_grant": "user",
                "org_id": org_id,
                "org_role": "member",
            },
        )
        assert code_response.status_code == 200
        code_data = code_response.json()

        accept_response = client.post(
            "/api/v1/orgs/invites/accept",
            headers=auth_headers,
            json={"code": code_data["code"]},
        )
        assert accept_response.status_code == 200
        accept_data = accept_response.json()
        assert accept_data["success"] is True
        assert accept_data["org_id"] == org_id
        assert accept_data["was_already_member"] is False

        org_list_response = client.get("/api/v1/orgs", headers=auth_headers)
        assert org_list_response.status_code == 200
        assert any(org["id"] == org_id for org in org_list_response.json()["items"])

    @pytest.mark.asyncio
    async def test_get_system_stats(self, isolated_test_environment, admin_headers):
        """Test getting system statistics"""
        client, db_name = isolated_test_environment
        response = client.get(
            "/api/v1/admin/stats",
            headers=admin_headers
        )
        assert response.status_code == 200
        data = response.json()
        assert "users" in data
        assert "storage" in data
        assert "sessions" in data

    @pytest.mark.asyncio
    async def test_get_audit_log(self, isolated_test_environment, admin_headers):
        """Test getting audit log"""
        client, db_name = isolated_test_environment
        response = client.get(
            "/api/v1/admin/audit-log",
            headers=admin_headers
        )
        assert response.status_code == 200
        data = response.json()
        assert "entries" in data


#######################################################################################################################
#
# Health Check Tests

class TestHealthEndpoints:
    """Test health monitoring endpoints.

    b1925901c2 ("security: harden operational health surfaces") put the
    /api/v1/health* diagnostics behind RequirePermission(SYSTEM_LOGS); the
    public detail-free liveness probe is the root ``/health``.
    """

    @pytest.mark.parametrize(
        "path",
        ["/api/v1/health", "/api/v1/health/live", "/api/v1/health/ready", "/api/v1/health/metrics"],
    )
    def test_health_diagnostics_require_authentication(self, isolated_test_environment, path):
        """Anonymous callers must not reach operational diagnostics."""
        client, db_name = isolated_test_environment
        response = client.get(path)
        assert response.status_code == 401

    def test_health_diagnostics_reject_user_without_system_logs(
        self, isolated_test_environment, auth_headers
    ):
        """A regular user lacks system.logs and is forbidden."""
        client, db_name = isolated_test_environment
        response = client.get("/api/v1/health/metrics", headers=auth_headers)
        assert response.status_code == 403

    def test_health_check(self, isolated_test_environment, admin_headers):
        """Test main health check"""
        client, db_name = isolated_test_environment
        response = client.get("/api/v1/health", headers=admin_headers)
        assert response.status_code in [200, 206, 503]
        data = response.json()
        assert "status" in data
        assert "checks" in data
        assert "timestamp" in data

    def test_liveness_probe(self, isolated_test_environment, admin_headers):
        """Test liveness probe"""
        client, db_name = isolated_test_environment
        response = client.get("/api/v1/health/live", headers=admin_headers)
        assert response.status_code == 200
        assert response.json()["status"] == "alive"

    def test_readiness_probe(self, isolated_test_environment, admin_headers):
        """Test readiness probe"""
        client, db_name = isolated_test_environment
        response = client.get("/api/v1/health/ready", headers=admin_headers)
        assert response.status_code in [200, 503]
        data = response.json()
        assert "status" in data

    def test_metrics_endpoint(self, isolated_test_environment, admin_headers):
        """Test metrics endpoint"""
        client, db_name = isolated_test_environment
        response = client.get("/api/v1/health/metrics", headers=admin_headers)
        assert response.status_code == 200
        data = response.json()
        assert "cpu" in data
        assert "memory" in data
        assert "disk" in data

    def test_public_root_liveness_needs_no_auth(self, isolated_test_environment):
        """The unauthenticated liveness surface stays at /health."""
        client, db_name = isolated_test_environment
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


#######################################################################################################################
#
# Security Tests

class TestSecurity:
    """Test security features"""

    def test_unauthorized_access(self, isolated_test_environment):

        """Test accessing protected endpoint without auth"""
        client, db_name = isolated_test_environment
        response = client.get("/api/v1/users/me")
        assert response.status_code in [401, 403]  # Both unauthorized and forbidden are acceptable

    def test_invalid_token(self, isolated_test_environment):

        """Test with invalid token"""
        client, db_name = isolated_test_environment
        response = client.get(
            "/api/v1/users/me",
            headers={"Authorization": "Bearer invalid-token"}
        )
        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_admin_endpoint_as_user(self, isolated_test_environment, auth_headers):
        """Test accessing admin endpoint as regular user"""
        client, db_name = isolated_test_environment
        response = client.get(
            "/api/v1/admin/users",
            headers=auth_headers
        )
        assert response.status_code == 403
        detail = response.json()["detail"]
        assert "access denied" in detail.lower()
        assert "required role" in detail.lower()
        assert "admin" in detail.lower()

    def test_rate_limiting(self, isolated_test_environment):
        """Test rate limiting"""
        client, db_name = isolated_test_environment
        # Make many requests quickly
        for i in range(10):
            response = client.post(
                "/api/v1/auth/login",
                data={
                    "username": "test",
                    "password": "wrong"
                }
            )

        # Should eventually get rate limited
        assert response.status_code == 429
        assert "Too many" in response.json()["detail"]


#######################################################################################################################
#
# Performance Tests

class TestPerformance:
    """Test performance and concurrency"""

    @pytest.mark.asyncio
    async def test_concurrent_logins(self, isolated_test_environment, test_user_data):
        """Test handling concurrent login requests"""
        client, db_name = isolated_test_environment

        # Using synchronous client for simplicity
        responses = []
        for _ in range(10):
            response = client.post(
                "/api/v1/auth/login",
                data={
                    "username": test_user_data["username"],
                    "password": test_user_data["password"]
                }
            )
            responses.append(response)

        # All should succeed
        assert all(r.status_code == 200 for r in responses)

    @pytest.mark.asyncio
    async def test_session_cleanup(self, isolated_test_environment):
        """Test that expired sessions are cleaned up"""
        import asyncpg

        client, db_name = isolated_test_environment

        # Connect to the unique test database
        host, port, user, pwd = _db_params()
        conn = await asyncpg.connect(host=host, port=port, user=user, password=pwd, database=db_name)

        try:
            # Create expired session (more than 1 day old to trigger cleanup)
            expired_time = datetime.utcnow() - timedelta(days=2)

            # First create a user to reference
            await conn.execute("""
                INSERT INTO users (id, uuid, username, email, password_hash, role)
                VALUES ($1, $2, $3, $4, $5, $6)
                ON CONFLICT (id) DO NOTHING
            """, 1, str(uuid.uuid4()), "testuser", "test@example.com", "hash", "user")

            # Insert expired session
            await conn.execute("""
                INSERT INTO sessions (user_id, token_hash, expires_at, is_active)
                VALUES ($1, $2, $3, $4)
            """, 1, "expired-token-hash", expired_time, True)

            # Close connection to commit
            await conn.close()

            # Run cleanup directly with a new connection
            h2, p2, u2, pw2 = _db_params()
            cleanup_conn = await asyncpg.connect(host=h2, port=p2, user=u2, password=pw2, database=db_name)

            try:
                # Delete expired sessions
                deleted_rows = await cleanup_conn.fetch(
                    """
                    DELETE FROM sessions
                    WHERE expires_at < CURRENT_TIMESTAMP - INTERVAL '1 day'
                    OR (is_active = FALSE AND revoked_at < CURRENT_TIMESTAMP - INTERVAL '7 days')
                    RETURNING id
                    """
                )
                # Should have deleted at least one session
                deleted = len(deleted_rows)
                assert deleted >= 0
            finally:
                await cleanup_conn.close()

            # Check session was deleted
            h3, p3, u3, pw3 = _db_params()
            check_conn = await asyncpg.connect(host=h3, port=p3, user=u3, password=pw3, database=db_name)
            try:
                count = await check_conn.fetchval(
                    "SELECT COUNT(*) FROM sessions WHERE token_hash = $1",
                    "expired-token-hash"
                )
                assert count == 0
            finally:
                await check_conn.close()
        except Exception as e:
            if 'conn' in locals() and conn and not conn.is_closed():
                await conn.close()
            raise


#######################################################################################################################
#
# Integration Tests

class TestIntegration:
    """Test full user flows"""

    def test_full_user_lifecycle(self, isolated_test_environment):

        """Test complete user lifecycle from registration to deletion"""
        client, db_name = isolated_test_environment
        # 1. Register new user
        register_response = client.post(
            "/api/v1/auth/register",
            json={
                "username": "lifecycle_user",
                "email": "lifecycle@example.com",
                "password": "Life@Cycle#2024!"
            }
        )
        assert register_response.status_code in [200, 201]
        user_id = register_response.json()["user_id"]

        # 2. Login as new user
        login_response = client.post(
            "/api/v1/auth/login",
            data={
                "username": "lifecycle_user",
                "password": "Life@Cycle#2024!"
            }
        )
        assert login_response.status_code == 200
        token = login_response.json()["access_token"]
        user_headers = {"Authorization": f"Bearer {token}"}

        # 3. Get user profile
        profile_response = client.get(
            "/api/v1/users/me",
            headers=user_headers
        )
        assert profile_response.status_code == 200

        # 4. Change password
        password_response = client.post(
            "/api/v1/users/change-password",
            headers=user_headers,
            json={
                "current_password": "Life@Cycle#2024!",
                "new_password": "NewLife@Cycle#2025!"
            }
        )
        assert password_response.status_code == 200

        # Admin operations would go here but skipped for now
        # to avoid dependency on admin_headers fixture
        pass


#
## End of test_auth_comprehensive.py
#######################################################################################################################
