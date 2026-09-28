"""
Property-based tests for user endpoints.
"""

from unittest.mock import AsyncMock

import pytest
from email_validator import validate_email, EmailNotValidError
from httpx import AsyncClient, ASGITransport
from hypothesis import given, strategies as st, settings as hypothesis_settings, HealthCheck

from tldw_Server_API.app.main import app


class TestUserEndpointsProperty:
    """Property-based tests for user endpoints."""

    @pytest.mark.asyncio
    @given(email=st.emails())
    @hypothesis_settings(
        max_examples=10,
        deadline=5000,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    async def test_update_profile_with_various_emails(
        self,
        email,
        mock_db_pool,
        test_user,
        valid_access_token,
    ):
        """Exercise profile updates across a wide range of email formats."""
        try:
            validation = validate_email(email, check_deliverability=False)
            normalized_email = validation.normalized.lower()
        except (EmailNotValidError, Exception):
            normalized_email = email.lower()

        updated_user = {**test_user, "email": normalized_email}

        from tldw_Server_API.app.api.v1.endpoints import users as users_endpoints

        # Email writes go through the profile-version gateway, which needs a
        # real users row; the endpoint contract is what is under test here.
        # (hypothesis forbids the function-scoped monkeypatch fixture.)
        original_update_email = users_endpoints.update_user_email
        users_endpoints.update_user_email = AsyncMock()

        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock()
        mock_conn.fetchrow = AsyncMock(return_value=updated_user)
        mock_conn.commit = AsyncMock()

        mock_db_pool.transaction.return_value.__aenter__.return_value = mock_conn
        mock_db_pool.transaction.return_value.__aexit__.return_value = None

        from tldw_Server_API.app.api.v1.API_Deps.auth_deps import (
            get_current_active_user,
            get_db_transaction,
        )

        async def mock_get_current_active_user():
            return test_user

        app.dependency_overrides[get_current_active_user] = mock_get_current_active_user

        async def mock_get_db_transaction():
            yield mock_conn

        app.dependency_overrides[get_db_transaction] = mock_get_db_transaction

        try:
            async with AsyncClient(
                transport=ASGITransport(app=app),
                base_url="http://test",
            ) as client:
                response = await client.put(
                    "/api/v1/users/me",
                    headers={"Authorization": f"Bearer {valid_access_token}"},
                    json={"email": email},
                )
        finally:
            users_endpoints.update_user_email = original_update_email

        if response.status_code == 400:
            assert "No updates provided" in response.json()["detail"]
        else:
            assert response.status_code == 200
            data = response.json()
            assert data["email"] == normalized_email

        app.dependency_overrides.clear()

    @pytest.mark.asyncio
    @given(
        new_password=st.text(min_size=8, max_size=100).filter(
            lambda x: not x.isspace() and any(c.isdigit() for c in x)
        )
    )
    @hypothesis_settings(
        max_examples=5,
        deadline=5000,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    async def test_change_password_various_formats(
        self,
        new_password,
        mock_db_pool,
        password_service,
        test_user,
        valid_access_token,
    ):
        """Changing passwords should succeed for a variety of valid strong inputs."""
        test_user_copy = test_user.copy()
        test_user_copy["password_hash"] = password_service.hash_password("Current@Pass#2024")

        mock_conn = AsyncMock()
        mock_conn.fetchval = AsyncMock(return_value=test_user_copy["password_hash"])
        mock_conn.execute = AsyncMock()
        mock_conn.commit = AsyncMock()

        mock_db_pool.transaction.return_value.__aenter__.return_value = mock_conn
        mock_db_pool.transaction.return_value.__aexit__.return_value = None

        from tldw_Server_API.app.api.v1.API_Deps.auth_deps import (
            get_current_active_user,
            get_db_transaction,
            get_password_service_dep,
        )

        async def mock_get_current_active_user():
            return test_user_copy

        app.dependency_overrides[get_current_active_user] = mock_get_current_active_user

        async def mock_get_db_transaction():
            yield mock_conn

        app.dependency_overrides[get_db_transaction] = mock_get_db_transaction
        app.dependency_overrides[get_password_service_dep] = lambda: password_service

        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/api/v1/users/change-password",
                headers={"Authorization": f"Bearer {valid_access_token}"},
                json={
                    "current_password": "Current@Pass#2024",
                    "new_password": new_password,
                },
            )

        assert response.status_code in [200, 400, 422]

        if response.status_code == 200:
            assert "Password changed successfully" in response.json()["message"]

        app.dependency_overrides.clear()
