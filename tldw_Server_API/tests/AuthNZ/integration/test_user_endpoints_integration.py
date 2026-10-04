"""
Integration tests for user management endpoints using real database.
"""

import pytest
pytestmark = pytest.mark.integration
import uuid
from datetime import datetime, timedelta
from tldw_Server_API.app.core.AuthNZ.password_service import PasswordService


class TestUserEndpointsIntegration:
    """Integration tests for user management endpoints with real database."""

    @pytest.mark.asyncio
    async def test_get_user_profile(self, isolated_test_environment):
        """Test getting user profile."""
        client, db_name = isolated_test_environment

        # Create a test user
        response = client.post(
            "/api/v1/auth/register",
            json={
                "username": "testuser",
                "email": "test@example.com",
                "password": "Test@Pass#2024!"
            }
        )
        assert response.status_code in [200, 201]

        # Login to get token
        login_response = client.post(
            "/api/v1/auth/login",
            data={
                "username": "testuser",
                "password": "Test@Pass#2024!"
            }
        )
        assert login_response.status_code == 200
        token = login_response.json()["access_token"]

        # Get user profile
        profile_response = client.get(
            "/api/v1/users/me",
            headers={"Authorization": f"Bearer {token}"}
        )

        assert profile_response.status_code == 200
        data = profile_response.json()
        assert data["username"] == "testuser"
        assert data["email"] == "test@example.com"
        assert data["role"] == "user"
        assert "storage_quota_mb" in data
        assert "storage_used_mb" in data

    @pytest.mark.asyncio
    async def test_update_user_profile(self, isolated_test_environment):
        """Test updating user profile."""
        client, db_name = isolated_test_environment

        # Create and login user
        client.post(
            "/api/v1/auth/register",
            json={
                "username": "updateuser",
                "email": "update@example.com",
                "password": "Update@Pass#2024!"
            }
        )

        login_response = client.post(
            "/api/v1/auth/login",
            data={
                "username": "updateuser",
                "password": "Update@Pass#2024!"
            }
        )
        token = login_response.json()["access_token"]

        # Update profile
        update_response = client.put(
            "/api/v1/users/me",
            headers={"Authorization": f"Bearer {token}"},
            json={"email": "newemail@example.com"}
        )

        assert update_response.status_code == 200
        data = update_response.json()
        assert data["email"] == "newemail@example.com"
        assert data["username"] == "updateuser"

    @pytest.mark.asyncio
    async def test_change_password(self, isolated_test_environment):
        """Test changing user password."""
        client, db_name = isolated_test_environment

        # Create and login user
        client.post(
            "/api/v1/auth/register",
            json={
                "username": "passworduser",
                "email": "password@example.com",
                "password": "Old@Pass#2024!"
            }
        )

        login_response = client.post(
            "/api/v1/auth/login",
            data={
                "username": "passworduser",
                "password": "Old@Pass#2024!"
            }
        )
        token = login_response.json()["access_token"]

        # Change password
        change_response = client.post(
            "/api/v1/users/change-password",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "current_password": "Old@Pass#2024!",
                "new_password": "New@Pass#2024!"
            }
        )

        assert change_response.status_code == 200
        assert "Password changed successfully" in change_response.json()["message"]

        # Verify can login with new password
        new_login = client.post(
            "/api/v1/auth/login",
            data={
                "username": "passworduser",
                "password": "New@Pass#2024!"
            }
        )
        assert new_login.status_code == 200

    @pytest.mark.asyncio
    async def test_change_password_wrong_current(self, isolated_test_environment):
        """Test changing password with wrong current password."""
        client, db_name = isolated_test_environment

        # Create and login user
        client.post(
            "/api/v1/auth/register",
            json={
                "username": "wrongpassuser",
                "email": "wrongpass@example.com",
                "password": "Correct@Pass#2024!"
            }
        )

        login_response = client.post(
            "/api/v1/auth/login",
            data={
                "username": "wrongpassuser",
                "password": "Correct@Pass#2024!"
            }
        )
        token = login_response.json()["access_token"]

        # Try to change password with wrong current password
        change_response = client.post(
            "/api/v1/users/change-password",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "current_password": "Wrong@Pass#2024!",
                "new_password": "New@Pass#2024!"
            }
        )

        assert change_response.status_code == 401
        assert "Current password is incorrect" in change_response.json()["detail"]

    @pytest.mark.asyncio
    async def test_get_user_sessions(self, isolated_test_environment):
        """Test getting user sessions."""
        client, db_name = isolated_test_environment

        # Create and login user
        client.post(
            "/api/v1/auth/register",
            json={
                "username": "sessionuser",
                "email": "session@example.com",
                "password": "Session@Pass#2024!"
            }
        )

        # Create multiple sessions by logging in multiple times
        login1 = client.post(
            "/api/v1/auth/login",
            data={
                "username": "sessionuser",
                "password": "Session@Pass#2024!"
            }
        )
        token1 = login1.json()["access_token"]

        login2 = client.post(
            "/api/v1/auth/login",
            data={
                "username": "sessionuser",
                "password": "Session@Pass#2024!"
            }
        )

        # Get sessions
        sessions_response = client.get(
            "/api/v1/users/sessions",
            headers={"Authorization": f"Bearer {token1}"}
        )

        assert sessions_response.status_code == 200
        data = sessions_response.json()
        assert isinstance(data, list)
        assert len(data) >= 1  # At least one session
        assert all("id" in session for session in data)
        assert all("ip_address" in session for session in data)

    @pytest.mark.asyncio
    async def test_revoke_session(self, isolated_test_environment):
        """Test revoking a user session."""
        client, db_name = isolated_test_environment

        # Create and login user
        client.post(
            "/api/v1/auth/register",
            json={
                "username": "revokeuser",
                "email": "revoke@example.com",
                "password": "Revoke@Pass#2024!"
            }
        )

        login_response = client.post(
            "/api/v1/auth/login",
            data={
                "username": "revokeuser",
                "password": "Revoke@Pass#2024!"
            }
        )
        token = login_response.json()["access_token"]

        # Get sessions to find a session ID
        sessions = client.get(
            "/api/v1/users/sessions",
            headers={"Authorization": f"Bearer {token}"}
        )
        session_id = sessions.json()[0]["id"]

        # Revoke the session
        revoke_response = client.delete(
            f"/api/v1/users/sessions/{session_id}",
            headers={"Authorization": f"Bearer {token}"}
        )

        assert revoke_response.status_code == 200
        assert "Session revoked successfully" in revoke_response.json()["message"]

    @pytest.mark.asyncio
    async def test_get_storage_quota(self, isolated_test_environment):
        """Test getting storage quota information."""
        client, db_name = isolated_test_environment

        # Create and login user
        client.post(
            "/api/v1/auth/register",
            json={
                "username": "storageuser",
                "email": "storage@example.com",
                "password": "Storage@Pass#2024!"
            }
        )

        login_response = client.post(
            "/api/v1/auth/login",
            data={
                "username": "storageuser",
                "password": "Storage@Pass#2024!"
            }
        )
        token = login_response.json()["access_token"]

        # Get storage quota
        storage_response = client.get(
            "/api/v1/users/storage",
            headers={"Authorization": f"Bearer {token}"}
        )

        assert storage_response.status_code == 200
        data = storage_response.json()
        assert "storage_quota_mb" in data
        assert "storage_used_mb" in data
        assert "available_mb" in data
        assert "usage_percentage" in data
        # A fresh user has no limits.storage_quota_mb override, so the quota comes
        # back null (unlimited) from the resolver (spec 2 Sec. 5), not a positive
        # legacy column value.
        assert data["storage_quota_mb"] is None
        assert data["available_mb"] is None
        assert data["usage_percentage"] is None
        assert data["storage_used_mb"] >= 0


class TestUserEndpointsEdgeCases:
    """Edge case tests for user endpoints."""

    @pytest.mark.asyncio
    async def test_update_profile_no_changes(self, isolated_test_environment):
        """Test updating profile without providing any changes."""
        client, db_name = isolated_test_environment

        # Create and login user
        client.post(
            "/api/v1/auth/register",
            json={
                "username": "nochangeuser",
                "email": "nochange@example.com",
                "password": "NoChange@Pass#2024!"
            }
        )

        login_response = client.post(
            "/api/v1/auth/login",
            data={
                "username": "nochangeuser",
                "password": "NoChange@Pass#2024!"
            }
        )
        token = login_response.json()["access_token"]

        # Try to update with same email
        update_response = client.put(
            "/api/v1/users/me",
            headers={"Authorization": f"Bearer {token}"},
            json={"email": "nochange@example.com"}  # Same email
        )

        # Should return 400 for no updates
        assert update_response.status_code == 400
        assert "No updates provided" in update_response.json()["detail"]

    @pytest.mark.asyncio
    async def test_weak_password_change(self, isolated_test_environment):
        """Test changing to a weak password."""
        client, db_name = isolated_test_environment

        # Create and login user
        client.post(
            "/api/v1/auth/register",
            json={
                "username": "weakpassuser",
                "email": "weakpass@example.com",
                "password": "Strong@Pass#2024!"
            }
        )

        login_response = client.post(
            "/api/v1/auth/login",
            data={
                "username": "weakpassuser",
                "password": "Strong@Pass#2024!"
            }
        )
        token = login_response.json()["access_token"]

        # Try to change to weak password
        change_response = client.post(
            "/api/v1/users/change-password",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "current_password": "Strong@Pass#2024!",
                "new_password": "weak"
            }
        )

        assert change_response.status_code in [400, 422]
        detail = str(change_response.json().get("detail", ""))
        assert "password" in detail.lower() or "at least" in detail.lower()

    @pytest.mark.asyncio
    async def test_revoke_nonexistent_session(self, isolated_test_environment):
        """Test revoking a session that doesn't exist."""
        client, db_name = isolated_test_environment

        # Create and login user
        client.post(
            "/api/v1/auth/register",
            json={
                "username": "nosessionuser",
                "email": "nosession@example.com",
                "password": "NoSession@Pass#2024!"
            }
        )

        login_response = client.post(
            "/api/v1/auth/login",
            data={
                "username": "nosessionuser",
                "password": "NoSession@Pass#2024!"
            }
        )
        token = login_response.json()["access_token"]

        # Try to revoke non-existent session
        revoke_response = client.delete(
            "/api/v1/users/sessions/999999",
            headers={"Authorization": f"Bearer {token}"}
        )

        # Should still return 200 (idempotent operation)
        assert revoke_response.status_code == 200

    @pytest.mark.asyncio
    async def test_unauthorized_access(self, isolated_test_environment):
        """Test accessing user endpoints without authentication."""
        client, db_name = isolated_test_environment

        # Try to access profile without token
        response = client.get("/api/v1/users/me")
        assert response.status_code in [401, 403]

        # Try to update profile without token
        response = client.put(
            "/api/v1/users/me",
            json={"email": "new@example.com"}
        )
        assert response.status_code in [401, 403]

        # Try to change password without token
        response = client.post(
            "/api/v1/users/change-password",
            json={
                "current_password": "old",
                "new_password": "new"
            }
        )
        assert response.status_code in [401, 403]
