"""
Property-based tests for authentication endpoints.
"""

from unittest.mock import AsyncMock

import pytest
from httpx import AsyncClient, ASGITransport
from hypothesis import given, strategies as st, settings as hypothesis_settings, HealthCheck

from tldw_Server_API.app.main import app


class TestAuthEndpointsProperty:
    """Property-based tests for authentication endpoints."""

    @pytest.mark.asyncio
    @given(
        username=st.text(
            alphabet=st.characters(whitelist_categories=("Ll", "Lu", "Nd")),
            min_size=3,
            max_size=50,
        ),
        email=st.emails(),
        password=st.text(min_size=8, max_size=100).filter(lambda x: len(x.strip()) >= 8),
    )
    @hypothesis_settings(
        max_examples=10,
        deadline=5000,
        # st.emails() is slow to build on a cold, loaded runner (26.9s first draw in CI).
        suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.filter_too_much, HealthCheck.too_slow],
    )
    async def test_register_with_various_inputs(
        self,
        username,
        email,
        password,
        registration_service,
    ):
        """Registration should handle a range of valid credentials."""
        registration_service.register_user = AsyncMock(
            return_value={
                "user_id": 1,
                "username": username,
                "email": email,
                "is_verified": False,
            }
        )

        from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_registration_service_dep

        app.dependency_overrides[get_registration_service_dep] = lambda: registration_service

        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/api/v1/auth/register",
                json={"username": username, "email": email, "password": password},
            )

        assert response.status_code in [200, 400, 403, 409, 422]

        if response.status_code == 200:
            data = response.json()
            assert data["username"] == username
            assert data["email"] == email

        app.dependency_overrides.clear()

    @pytest.mark.asyncio
    @given(token_length=st.integers(min_value=10, max_value=1000))
    @hypothesis_settings(
        max_examples=10,
        deadline=5000,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    async def test_refresh_with_various_token_lengths(
        self,
        isolated_test_environment,
        token_length,
        jwt_service,
    ):
        """Refresh should reject arbitrary token-like inputs."""
        client, _ = isolated_test_environment
        from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_jwt_service_dep

        app.dependency_overrides[get_jwt_service_dep] = lambda: jwt_service

        fake_token = "a" * token_length

        response = client.post(
            "/api/v1/auth/refresh",
            json={"refresh_token": fake_token},
        )

        assert response.status_code == 401

        app.dependency_overrides.clear()
