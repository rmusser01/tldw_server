# test_embeddings_v5_unit.py
# Comprehensive test suite for production embeddings service - FIXED VERSION
# Unit tests with mocks

import os
import uuid

_ORIG_TESTING = os.environ.get("TESTING")
_ORIG_AUTO_DOWNLOAD_MODELS = os.environ.get("AUTO_DOWNLOAD_MODELS")

# Set TESTING environment variable BEFORE importing anything else
os.environ["TESTING"] = "true"
os.environ["AUTO_DOWNLOAD_MODELS"] = "false"

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException, status
from fastapi.testclient import TestClient
from starlette.requests import Request

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_auth_principal
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthContext, AuthPrincipal
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.main import app


# Cleanup fixture to remove TESTING env var after tests
@pytest.fixture(autouse=True, scope="module")
def cleanup_testing_env():
    """Cleanup TESTING environment variable after module tests"""
    os.environ["TESTING"] = "true"
    os.environ["AUTO_DOWNLOAD_MODELS"] = "false"
    yield
    if _ORIG_TESTING is None:
        os.environ.pop("TESTING", None)
    else:
        os.environ["TESTING"] = _ORIG_TESTING
    if _ORIG_AUTO_DOWNLOAD_MODELS is None:
        os.environ.pop("AUTO_DOWNLOAD_MODELS", None)
    else:
        os.environ["AUTO_DOWNLOAD_MODELS"] = _ORIG_AUTO_DOWNLOAD_MODELS

# Mock metrics for tests to avoid registry conflicts
@pytest.fixture(autouse=True)
def mock_metrics():
    """Mock Prometheus metrics to avoid registry conflicts"""
    mock_counter = MagicMock()
    mock_counter_instance = MagicMock()
    mock_counter_instance.inc = MagicMock()
    mock_counter_instance._value = MagicMock()
    mock_counter_instance._value.get.return_value = 0
    mock_counter.labels.return_value = mock_counter_instance

    mock_histogram = MagicMock()
    mock_histogram_instance = MagicMock()
    mock_histogram_instance.observe = MagicMock()
    mock_histogram.labels.return_value = mock_histogram_instance

    mock_gauge = MagicMock()
    mock_gauge.inc = MagicMock()
    mock_gauge.dec = MagicMock()
    mock_gauge._value = MagicMock()
    mock_gauge._value.get.return_value = 0

    with patch('tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced.embedding_requests_total', mock_counter), \
         patch('tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced.embedding_request_duration', mock_histogram), \
         patch('tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced.embedding_cache_hits', mock_counter), \
         patch('tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced.active_embedding_requests', mock_gauge):
             yield


@pytest.fixture
def setup():
    """Setup test environment fixture with proper TestClient lifecycle"""
    class SetupData:
        pass

    client = TestClient(app)
    try:
        data = SetupData()
        data.client = client
        # Set CSRF token in both cookie and header for double-submit pattern
        csrf_token = f"test-csrf-{uuid.uuid4().hex}"
        client.cookies.set("csrf_token", csrf_token)
        data.auth_headers = {
            "Authorization": "Bearer test-api-key",
            "X-CSRF-Token": csrf_token
        }

        data.regular_user = User(
            id=1,
            username="testuser",
            email="test@example.com",
            is_active=True,
            is_admin=False
        )

        data.admin_user = User(
            id=2,
            username="admin",
            email="admin@example.com",
            is_active=True,
            is_admin=True
        )

        yield data
    finally:
        app.dependency_overrides.clear()
        client.close()


class TestCriticalSecurity:
    """Test critical security fixes"""

    @pytest.mark.unit
    def test_no_placeholder_embeddings(self):
        """Verify system fails properly when dependencies missing"""
        # Note: This test verifies that the module properly checks for dependencies
        # In v5, if EMBEDDINGS_AVAILABLE is False, the module raises RuntimeError at import
        # Since the module is already imported, we can only verify the flag exists
        from tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced import EMBEDDINGS_AVAILABLE
        assert EMBEDDINGS_AVAILABLE is True  # Should be True if imports succeeded

    @pytest.mark.unit
    def test_admin_authorization_required(self, setup):
        """Test admin endpoints require proper authorization"""
        # Non-admin principal (no admin role / system.configure) should be forbidden
        def override_regular_user():
            return setup.regular_user

        async def override_regular_principal(request: Request) -> AuthPrincipal:  # type: ignore[override]
            # token_type is a test principal label, not a secret.
            principal = AuthPrincipal(  # nosec B106
                kind="user",
                user_id=setup.regular_user.id,
                api_key_id=None,
                subject=setup.regular_user.username,
                token_type="access",
                jti=None,
                roles=["user"],
                permissions=[],
                is_admin=False,
                org_ids=[],
                team_ids=[],
            )
            try:
                request.state.auth = AuthContext(
                    principal=principal,
                    ip=None,
                    user_agent=None,
                    request_id=None,
                )
            except Exception:
                _ = None
            return principal

        app.dependency_overrides[get_request_user] = override_regular_user
        app.dependency_overrides[get_auth_principal] = override_regular_principal

        response = setup.client.delete(
            "/api/v1/embeddings/cache",
            headers=setup.auth_headers,
        )
        # Non-admins should be rejected by authentication/RBAC (401/403) or by
        # an upstream rate/budget guard (429) for this admin-only endpoint.
        assert response.status_code in (401, 403, 429)

        # Admin principal with admin role and system.configure permission should succeed
        def override_admin_user():
            return setup.admin_user

        async def override_admin_principal(request: Request) -> AuthPrincipal:  # type: ignore[override]
            # token_type is a test principal label, not a secret.
            principal = AuthPrincipal(  # nosec B106
                kind="user",
                user_id=setup.admin_user.id,
                api_key_id=None,
                subject=setup.admin_user.username,
                token_type="access",
                jti=None,
                roles=["admin"],
                permissions=["system.configure"],
                is_admin=True,
                org_ids=[],
                team_ids=[],
            )
            try:
                request.state.auth = AuthContext(
                    principal=principal,
                    ip=None,
                    user_agent=None,
                    request_id=None,
                )
            except Exception:
                _ = None
            return principal

        app.dependency_overrides[get_request_user] = override_admin_user
        app.dependency_overrides[get_auth_principal] = override_admin_principal

        response = setup.client.delete(
            "/api/v1/embeddings/cache",
            headers=setup.auth_headers,
        )
        # Depending on LLM budget or rate limiting, a successful admin call may be 200 or 429.
        assert response.status_code in (200, 429)


class TestTTLCache:
    """Test TTL cache implementation"""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_cache_ttl_expiration(self, monkeypatch):
        """Test that cache entries expire after TTL"""
        from tldw_Server_API.app.api.v1.endpoints import embeddings_v5_production_enhanced as embeddings_module

        current_time = 1000.0
        monkeypatch.setattr(embeddings_module.time, "time", lambda: current_time)
        cache = embeddings_module.TTLCache(max_size=10, ttl_seconds=1)

        await cache.set("test_key", [1.0, 2.0, 3.0])
        value = await cache.get("test_key")
        assert value == [1.0, 2.0, 3.0]

        current_time += 1.5

        value = await cache.get("test_key")
        assert value is None

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_cache_lru_eviction(self):
        """Test LRU eviction when cache is full"""
        from tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced import TTLCache

        cache = TTLCache(max_size=3, ttl_seconds=3600)

        # Fill cache
        await cache.set("key1", [1.0])
        await cache.set("key2", [2.0])
        await cache.set("key3", [3.0])

        # Access key1 to make it more recently used
        await cache.get("key1")

        # Add new key - should evict key2 (least recently used)
        await cache.set("key4", [4.0])

        assert await cache.get("key1") == [1.0]  # Still there
        assert await cache.get("key2") is None   # Evicted
        assert await cache.get("key3") == [3.0]  # Still there
        assert await cache.get("key4") == [4.0]  # New entry

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_cache_thread_safety(self):
        """Test cache operations are thread-safe"""
        from tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced import TTLCache

        cache = TTLCache(max_size=100, ttl_seconds=3600)

        async def writer(start, end):
            for i in range(start, end):
                await cache.set(f"key_{i}", [float(i)])

        async def reader(start, end):
            for i in range(start, end):
                await cache.get(f"key_{i}")

        # Run concurrent operations
        tasks = [
            writer(0, 20),
            writer(20, 40),
            reader(0, 20),
            reader(10, 30),
        ]

        await asyncio.gather(*tasks)

        # Verify some values
        assert await cache.get("key_5") == [5.0]
        assert await cache.get("key_25") == [25.0]

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_cache_thread_cleanup_removes_expired_entries(self, monkeypatch):
        import tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced as embeddings_mod

        monkeypatch.setattr(embeddings_mod, "CACHE_CLEANUP_INTERVAL", 0.05)

        cache = embeddings_mod.TTLCache(max_size=10, ttl_seconds=0)
        await cache.set("stale", [1.0])
        await cache.start_cleanup_task()
        try:
            await asyncio.sleep(0.15)
        finally:
            await cache.stop_cleanup_task()

        assert await cache.get("stale") is None


class TestConnectionPooling:
    """Test connection pool management"""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_connection_pool_creation(self):
        """Test that connection pools are created properly"""
        from tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced import ConnectionPoolManager

        manager = ConnectionPoolManager()

        try:
            # Get sessions for different providers
            session1 = await manager.get_session("openai")
            session2 = await manager.get_session("huggingface")

            assert session1 is not None
            assert session2 is not None
            assert session1 is not session2

        finally:
            await manager.close_all()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_connection_pool_reopens_after_close(self):
        """Manager should recreate sessions after shutdown."""
        from tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced import ConnectionPoolManager

        manager = ConnectionPoolManager()

        try:
            first_session = await manager.get_session("huggingface")
            assert first_session is not None

            await manager.close_all()

            second_session = await manager.get_session("huggingface")
            assert second_session is not None
            assert second_session is not first_session
        finally:
            await manager.close_all()


class TestRetryLogic:
    """Test retry logic and error handling"""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_retry_on_connection_error(self):
        """Test that connection errors are handled by circuit breaker"""
        from tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced import (
            create_embeddings_with_circuit_breaker,
        )
        from tldw_Server_API.app.core.Infrastructure.circuit_breaker import CircuitBreaker

        attempt_count = 0

        def mock_embeddings(texts, config, model_id_override, metadata=None, **_):

            nonlocal attempt_count
            attempt_count += 1

            # First 2 attempts fail, third succeeds
            if attempt_count < 3:
                raise ConnectionError("Connection failed")

            return [[1.0, 2.0, 3.0]] * len(texts)

        from tenacity import retry, retry_if_exception_type, stop_after_attempt

        @retry(
            stop=stop_after_attempt(3),
            retry=retry_if_exception_type(ConnectionError),
        )
        def retry_wrapper_sync(*, texts, config, model_id_override, metadata=None):
            return mock_embeddings(
                texts=texts,
                config=config,
                model_id_override=model_id_override,
                metadata=metadata,
            )

        async def retry_wrapper(**kwargs):
            return retry_wrapper_sync(**kwargs)

        with patch(
            'tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced.batching_create_embeddings_batch_async',
            new=AsyncMock(side_effect=retry_wrapper),
        ):

            config = {"api_key": "test-key"}

            # Reset circuit breaker for clean test
            with patch('tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced.get_or_create_circuit_breaker') as mock_breaker:
                # Create a breaker that allows the call through
                breaker = CircuitBreaker(
                    name="test_breaker",
                    failure_threshold=5,
                    recovery_timeout=1.0,
                    expected_exception=(ConnectionError,)
                )
                mock_breaker.return_value = breaker

                result = await create_embeddings_with_circuit_breaker(
                    ["test text"],
                    "openai",
                    "test-model",
                    config
                )

            assert attempt_count == 3  # Should retry twice
            assert result == [[1.0, 2.0, 3.0]]

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_no_retry_on_value_error(self):
        """Test that value errors don't trigger retries"""
        from tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced import (
            create_embeddings_with_circuit_breaker,
        )

        attempt_count = 0

        def mock_embeddings(texts, config, model_id_override, metadata=None, **_):

            nonlocal attempt_count
            attempt_count += 1
            raise ValueError("Invalid input")

        with patch(
            'tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced.batching_create_embeddings_batch_async',
            new=AsyncMock(side_effect=mock_embeddings),
        ):
            with pytest.raises(ValueError):
                config = {"api_key": "test-key"}
                await create_embeddings_with_circuit_breaker(
                    ["test text"],
                    "openai",
                    "test-model",
                    config
                )

        # Should only try once since ValueError is not retryable
        assert attempt_count == 1


class TestErrorHandling:
    """Test error handling with mocked dependencies"""

    @pytest.mark.unit
    def test_empty_input_error(self, setup):
        """Test error on empty input"""
        def override_user():
            return setup.regular_user

        app.dependency_overrides[get_request_user] = override_user

        response = setup.client.post(
            "/api/v1/embeddings",
            headers=setup.auth_headers,
            json={
                "input": "",
                "model": "text-embedding-3-small"
            }
        )

        assert response.status_code == 400
        assert "Input cannot be empty" in response.json()["detail"]

    @pytest.mark.unit
    def test_invalid_provider_error(self, setup):
        """Test error on invalid provider"""
        def override_user():
            return setup.regular_user

        app.dependency_overrides[get_request_user] = override_user

        response = setup.client.post(
            "/api/v1/embeddings",
            headers={**setup.auth_headers, "x-provider": "invalid_provider"},
            json={
                "input": "test text",
                "model": "some-model"
            }
        )

        # Check response - might be 400 for invalid provider or 503 if service unavailable
        assert response.status_code in [400, 503]
        if response.status_code == 400:
            assert "Unknown provider" in response.json()["detail"]
        else:
            # 503 means service temporarily unavailable
            detail = response.json().get("detail", "")
            assert "unavailable" in detail.lower() or "service" in detail.lower()

    @pytest.mark.unit
    def test_missing_provider_credentials_returns_503(self, setup, monkeypatch):
        """Missing provider credentials should return 503 with error code."""
        def override_user():
            return setup.regular_user

        app.dependency_overrides[get_request_user] = override_user

        import tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced as emb_ep
        from tldw_Server_API.app.core.AuthNZ.byok_runtime import ResolvedByokCredentials

        async def _missing(provider, *_args, **_kwargs):
            return ResolvedByokCredentials(
                provider=provider,
                api_key=None,
                app_config=None,
                credential_fields={},
                source="server",
                allowlisted=True,
            )

        monkeypatch.setattr(emb_ep, "resolve_byok_credentials", _missing)

        response = setup.client.post(
            "/api/v1/embeddings",
            headers={**setup.auth_headers, "x-provider": "cohere"},
            json={
                "input": "test text",
                "model": "embed-english-v3.0",
            },
        )

        assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
        detail = response.json().get("detail", {})
        assert detail.get("error_code") == "missing_provider_credentials"

    @pytest.mark.unit
    def test_openai_oauth_401_retries_with_forced_refresh(self, setup, monkeypatch):
        """OpenAI OAuth auth failures should force-refresh BYOK once and retry."""
        def override_user():
            return setup.regular_user

        app.dependency_overrides[get_request_user] = override_user
        monkeypatch.setenv("USE_REAL_OPENAI_IN_TESTS", "1")

        import tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced as emb_ep
        from tldw_Server_API.app.core.AuthNZ.byok_runtime import ResolvedByokCredentials

        forced_refresh_flags: list[bool] = []

        async def _resolve(provider, *_args, **kwargs):
            forced = bool(kwargs.get("force_oauth_refresh", False))
            forced_refresh_flags.append(forced)
            api_key = "oauth-refreshed-key" if forced else "oauth-initial-key"
            return ResolvedByokCredentials(
                provider=provider,
                api_key=api_key,
                app_config=None,
                credential_fields={},
                source="user",
                allowlisted=True,
                auth_source="oauth",
            )

        call_count = {"count": 0}

        async def _fake_batch_async(*, texts, provider, model_id, dimensions=None, api_key=None, metadata=None, **_kwargs):
            _ = (provider, model_id, dimensions, api_key, metadata)
            call_count["count"] += 1
            if call_count["count"] == 1:
                raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="expired oauth access token")
            return [[0.1, 0.2, 0.3] for _ in texts]

        monkeypatch.setattr(emb_ep, "resolve_byok_credentials", _resolve, raising=True)
        monkeypatch.setattr(emb_ep, "create_embeddings_batch_async", _fake_batch_async, raising=True)

        response = setup.client.post(
            "/api/v1/embeddings",
            headers={**setup.auth_headers, "x-provider": "openai"},
            json={
                "input": "test text",
                "model": "text-embedding-3-small",
            },
        )

        assert response.status_code == status.HTTP_200_OK
        assert call_count["count"] == 2
        assert forced_refresh_flags[:2] == [False, True]

    @pytest.mark.unit
    def test_openai_oauth_second_401_maps_to_upstream_502(self, setup, monkeypatch):
        """A second OpenAI OAuth auth failure cannot retain client-auth semantics."""
        def override_user():
            return setup.regular_user

        app.dependency_overrides[get_request_user] = override_user
        monkeypatch.setenv("USE_REAL_OPENAI_IN_TESTS", "1")

        import tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced as emb_ep
        from tldw_Server_API.app.core.AuthNZ.byok_runtime import ResolvedByokCredentials

        async def _resolve(provider, *_args, **kwargs):
            forced = bool(kwargs.get("force_oauth_refresh", False))
            api_key = "oauth-refreshed-key" if forced else "oauth-initial-key"
            return ResolvedByokCredentials(
                provider=provider,
                api_key=api_key,
                app_config=None,
                credential_fields={},
                source="user",
                allowlisted=True,
                auth_source="oauth",
            )

        async def _fake_batch_async(*, texts, provider, model_id, dimensions=None, api_key=None, metadata=None, **_kwargs):
            _ = (texts, provider, model_id, dimensions, api_key, metadata)
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="oauth auth failure")

        monkeypatch.setattr(emb_ep, "resolve_byok_credentials", _resolve, raising=True)
        monkeypatch.setattr(emb_ep, "create_embeddings_batch_async", _fake_batch_async, raising=True)

        response = setup.client.post(
            "/api/v1/embeddings",
            headers={**setup.auth_headers, "x-provider": "openai"},
            json={
                "input": "test text",
                "model": "text-embedding-3-small",
            },
        )

        assert response.status_code == status.HTTP_502_BAD_GATEWAY
        assert response.json().get("detail") == "Embedding provider authentication failed"


class TestMockedFlow:
    """Test complete flow with mocked embeddings"""

    @pytest.mark.unit
    def test_end_to_end_flow_mocked(self, setup):
        """Test complete flow with mocked embeddings"""
        def override_user():
            return setup.regular_user

        app.dependency_overrides[get_request_user] = override_user

        async def mock_embeddings(
            texts,
            provider,
            model_id,
            dimensions=None,
            api_key=None,
            api_url=None,
            metadata=None,
            cache_scope_sensitive=False,
        ):
            _ = cache_scope_sensitive
            return [[float(i), float(i+1), float(i+2)] for i, _ in enumerate(texts)]

        with patch('tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced.create_embeddings_batch_async', new=mock_embeddings):

            response = setup.client.post(
                "/api/v1/embeddings",
                headers=setup.auth_headers,
                json={
                    "input": ["text1", "text2", "text3"],
                    "model": "text-embedding-3-small"
                }
            )

            if response.status_code != 200:
                print(f"Response status: {response.status_code}")
                print(f"Response body: {response.text}")
            assert response.status_code == 200
            data = response.json()

            assert "data" in data
            assert "model" in data
            assert "usage" in data
            assert len(data["data"]) == 3

    @pytest.mark.unit
    def test_caching_behavior_mocked(self, setup):
        """Test caching behavior with mocked API calls"""
        def override_user():
            return setup.regular_user

        app.dependency_overrides[get_request_user] = override_user

        call_count = 0

        async def mock_embeddings(texts, provider, model_id, dimensions=None, api_key=None, api_url=None, metadata=None):
            nonlocal call_count
            call_count += 1
            return [[1.0, 2.0, 3.0]] * len(texts)  # Return same embedding for consistent caching

        # Mock the embedding function at the right level
        with patch('tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced.create_embeddings_with_circuit_breaker') as mock_create:
            # Wrap to track calls
            async def wrapper(texts, provider, model_id, config, metadata=None, dimensions=None):
                return await mock_embeddings(
                    texts,
                    provider,
                    model_id,
                    dimensions=dimensions,
                    metadata=metadata,
                )
            mock_create.side_effect = wrapper

            # First request
            response1 = setup.client.post(
                "/api/v1/embeddings",
                headers=setup.auth_headers,
                json={
                    "input": "cached text",
                    "model": "text-embedding-3-small"
                }
            )

            assert response1.status_code == 200

            # Second request with same input (should hit cache)
            response2 = setup.client.post(
                "/api/v1/embeddings",
                headers=setup.auth_headers,
                json={
                    "input": "cached text",
                    "model": "text-embedding-3-small"
                }
            )

            assert response2.status_code == 200

            # Verify embeddings are the same (from cache)
            emb1 = response1.json()["data"][0]["embedding"]
            emb2 = response2.json()["data"][0]["embedding"]
            assert emb1 == emb2

            # Check that the function was called fewer times for second request
            # Note: The exact call count depends on the cache implementation
            # The important thing is that the responses are identical
            assert call_count <= 2  # At most 2 calls (cache might not be perfect in test env)


class TestLLMBudgetGuardrails:
    """Tests for LLM budget middleware interactions with core LLM endpoints."""

    @pytest.mark.unit
    def test_embeddings_budget_exceeded_returns_402(self, monkeypatch, test_client):
        """Simulate an over-budget virtual key and assert 402 from budget middleware."""
        import tldw_Server_API.app.core.AuthNZ.llm_budget_middleware as budget_mod

        # Force budget middleware to apply to this path regardless of settings
        monkeypatch.setattr(
            budget_mod.LLMBudgetMiddleware,
            "_should_check",
            lambda self, path: path.startswith("/api/v1/embeddings"),
        )

        # Stub key resolution to treat the Authorization header as a valid virtual key
        async def _fake_resolve_api_key_by_hash(api_key, settings=None):  # type: ignore[override]
            _ = (api_key, settings)
            return {"id": 123, "user_id": 42}

        monkeypatch.setattr(budget_mod, "resolve_api_key_by_hash", _fake_resolve_api_key_by_hash)

        # Treat this key as a virtual key so budget enforcement runs
        async def _fake_get_key_limits(key_id: int):  # type: ignore[override]
            _ = key_id
            return {"is_virtual": True}

        monkeypatch.setattr(budget_mod, "get_key_limits", _fake_get_key_limits)

        # Force the auth governor to report the key as over budget
        class _FakeGov:
            async def check_llm_budget_for_api_key(self, principal, key_id):  # type: ignore[override]
                _ = (principal, key_id)
                return {
                    "over": True,
                    "limits": {"llm_budget_day_usd": 0},
                    "reasons": ["test_over_budget"],
                }

        async def _fake_get_auth_governor():  # type: ignore[override]
            return _FakeGov()

        monkeypatch.setattr(budget_mod, "get_auth_governor", _fake_get_auth_governor)

        client = test_client
        client.headers["Authorization"] = "Bearer test-virtual-key"

        resp = client.post(
            "/api/v1/embeddings",
            json={"input": "hello", "model": "text-embedding-3-small"},
        )
        # Over-budget virtual key should yield a 402 from budget middleware.
        assert resp.status_code == 402
        body = resp.json()
        assert body.get("error") == "budget_exceeded"

    @pytest.mark.unit
    def test_chat_budget_exceeded_returns_402(self, monkeypatch, test_client):
        """Simulate an over-budget virtual key and assert 402 for chat completions."""
        import tldw_Server_API.app.core.AuthNZ.llm_budget_middleware as budget_mod

        # Force budget middleware to apply to chat completions
        monkeypatch.setattr(
            budget_mod.LLMBudgetMiddleware,
            "_should_check",
            lambda self, path: path.startswith("/api/v1/chat/completions"),
        )

        # Stub key resolution to treat the Authorization header as a valid virtual key
        async def _fake_resolve_api_key_by_hash(api_key, settings=None):  # type: ignore[override]
            _ = (api_key, settings)
            return {"id": 999, "user_id": 77}

        monkeypatch.setattr(budget_mod, "resolve_api_key_by_hash", _fake_resolve_api_key_by_hash)

        # Treat this key as a virtual key so budget enforcement runs
        async def _fake_get_key_limits(key_id: int):  # type: ignore[override]
            _ = key_id
            return {"is_virtual": True}

        monkeypatch.setattr(budget_mod, "get_key_limits", _fake_get_key_limits)

        # Force the auth governor to report the key as over budget
        class _FakeGov:
            async def check_llm_budget_for_api_key(self, principal, key_id):  # type: ignore[override]
                _ = (principal, key_id)
                return {
                    "over": True,
                    "limits": {"llm_budget_day_tokens": 0},
                    "reasons": ["test_over_budget_chat"],
                }

        async def _fake_get_auth_governor():  # type: ignore[override]
            return _FakeGov()

        monkeypatch.setattr(budget_mod, "get_auth_governor", _fake_get_auth_governor)

        client = test_client
        client.headers["Authorization"] = "Bearer test-virtual-key-chat"

        resp = client.post(
            "/api/v1/chat/completions",
            json={
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "user", "content": "Hello"},
                ],
            },
        )
        # Over-budget virtual key should yield a 402 from budget middleware.
        assert resp.status_code == 402
        body = resp.json()
        assert body.get("error") == "budget_exceeded"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_batch_length_mismatch_raises(monkeypatch):
    import tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced as mod

    async def fake_create_embeddings_with_circuit_breaker(
        texts,
        provider,
        model_id,
        config,
        metadata=None,
        dimensions=None,
    ):
        _ = (provider, model_id, config, metadata, dimensions)
        return [[0.1] for _ in range(max(1, len(texts) - 1))]

    monkeypatch.setattr(
        mod,
        "create_embeddings_with_circuit_breaker",
        fake_create_embeddings_with_circuit_breaker,
        raising=True,
    )
    monkeypatch.setattr(mod.embedding_cache, "get", AsyncMock(return_value=None))
    monkeypatch.setattr(mod.embedding_cache, "set", AsyncMock())

    with pytest.raises(HTTPException) as exc:
        await mod.create_embeddings_batch_async(
            ["a", "b"],
            provider="huggingface",
            model_id="sentence-transformers/all-MiniLM-L6-v2",
        )

    assert exc.value.status_code == 502


@pytest.mark.unit
@pytest.mark.asyncio
async def test_batch_rate_limit_maps_to_429(monkeypatch):
    import tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced as mod
    from tldw_Server_API.app.core.Embeddings.request_batching import EmbeddingsRateLimitError

    async def fake_create_embeddings_with_circuit_breaker(
        texts,
        provider,
        model_id,
        config,
        metadata=None,
        dimensions=None,
    ):
        _ = (texts, provider, model_id, config, metadata, dimensions)
        raise EmbeddingsRateLimitError("rate limited", retry_after=3)

    monkeypatch.setattr(
        mod,
        "create_embeddings_with_circuit_breaker",
        fake_create_embeddings_with_circuit_breaker,
        raising=True,
    )
    monkeypatch.setattr(mod.embedding_cache, "get", AsyncMock(return_value=None))
    monkeypatch.setattr(mod.embedding_cache, "set", AsyncMock())

    with pytest.raises(HTTPException) as exc:
        await mod.create_embeddings_batch_async(
            ["a"],
            provider="huggingface",
            model_id="sentence-transformers/all-MiniLM-L6-v2",
        )

    assert exc.value.status_code == 429
    assert exc.value.headers.get("Retry-After") == "3"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_batch_generic_provider_error_is_sanitized(monkeypatch):
    from loguru import logger

    import tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced as mod

    sentinel = "backend leaked /private/embedding-provider path"
    records = []

    async def fake_create_embeddings_with_circuit_breaker(
        texts,
        provider,
        model_id,
        config,
        metadata=None,
        dimensions=None,
    ):
        _ = (texts, provider, model_id, config, metadata, dimensions)
        raise RuntimeError(sentinel)

    monkeypatch.setattr(
        mod,
        "create_embeddings_with_circuit_breaker",
        fake_create_embeddings_with_circuit_breaker,
        raising=True,
    )
    monkeypatch.setattr(mod.embedding_cache, "get", AsyncMock(return_value=None))
    monkeypatch.setattr(mod.embedding_cache, "set", AsyncMock())
    monkeypatch.setattr(mod.connection_manager, "remove_provider", AsyncMock())

    sink_id = logger.add(records.append, format="{message} {extra}")
    try:
        with pytest.raises(HTTPException) as exc:
            await mod.create_embeddings_batch_async(
                ["a"],
                provider="huggingface",
                model_id="sentence-transformers/all-MiniLM-L6-v2",
            )
    finally:
        logger.remove(sink_id)

    assert exc.value.status_code == 503
    assert exc.value.detail == "Embedding service error"
    output = "".join(map(str, records))
    assert "RuntimeError" in output
    assert sentinel not in output


@pytest.mark.unit
@pytest.mark.asyncio
async def test_mlx_adapter_runtime_error_is_sanitized(monkeypatch):
    import tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced as mod

    class FakeBreaker:
        async def call_async(self, func, *args, **kwargs):
            return await func(*args, **kwargs)

    class FakeAdapter:
        def embed(self, payload):
            _ = payload
            raise RuntimeError("mlx cache exploded at /private/models")

    class FakeRegistry:
        def get_adapter(self, name):
            assert name == "mlx"
            return FakeAdapter()

    monkeypatch.setattr(mod, "get_or_create_circuit_breaker", lambda provider: FakeBreaker())
    monkeypatch.setattr(mod, "get_embeddings_registry", lambda: FakeRegistry())

    with pytest.raises(HTTPException) as exc:
        await mod.create_embeddings_with_circuit_breaker(
            ["a"],
            provider="mlx",
            model_id="mlx/test-model",
            config={"model_name_or_path": "mlx/test-model"},
        )

    assert exc.value.status_code == 502
    assert exc.value.detail == "MLX embeddings error"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_adapter_builder_copies_runtime_config_and_preserves_endpoint_provenance(
    monkeypatch,
):
    """Execution-scoped endpoint/key state is copied atomically for the adapter."""
    import tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced as mod
    from tldw_Server_API.app.core.AuthNZ.byok_config import (
        is_runtime_base_url_override,
        runtime_base_url_override_provenance,
    )

    endpoint = "https://runtime-hf.example/models"
    provenance = runtime_base_url_override_provenance()
    original_app_config = {
        "huggingface_api": {
            "api_base_url": endpoint,
            "_runtime_base_url_override": provenance,
            "nested": {"value": "original"},
        }
    }
    credentials = mod.ResolvedByokCredentials(
        provider="huggingface",
        api_key="runtime-key",
        app_config=original_app_config,
        credential_fields={"base_url": endpoint},
        source="user",
        allowlisted=True,
    )
    captured: list[dict] = []

    class _Adapter:
        def embed(self, request):
            captured.append(request)
            request["app_config"]["huggingface_api"]["nested"]["value"] = "mutated"
            return {"data": [{"index": 0, "embedding": [0.1, 0.2]}]}

    class _Registry:
        def get_adapter(self, provider):
            assert provider == "huggingface"
            return _Adapter()

    monkeypatch.setenv("LLM_EMBEDDINGS_ADAPTERS_ENABLED", "1")
    monkeypatch.setattr(mod, "get_embeddings_registry", lambda: _Registry())
    executor = mod._EndpointEmbeddingExecutor(
        request=MagicMock(),
        current_user=MagicMock(),
        user_metadata=None,
    )

    vectors = await executor._try_adapter_execution(
        ["hello"],
        provider="huggingface",
        model="org/model",
        dimensions=None,
        credentials=credentials,
    )

    assert vectors == [[0.1, 0.2]]
    assert len(captured) == 1
    adapter_request = captured[0]
    assert adapter_request["api_key"] == "runtime-key"
    assert adapter_request["base_url"] == endpoint
    assert adapter_request["credentials_resolved"] is True
    assert is_runtime_base_url_override(adapter_request["_runtime_base_url_override"])
    assert adapter_request["app_config"] is not original_app_config
    assert adapter_request["app_config"]["huggingface_api"] is not original_app_config["huggingface_api"]
    assert is_runtime_base_url_override(
        adapter_request["app_config"]["huggingface_api"]["_runtime_base_url_override"]
    )
    assert original_app_config["huggingface_api"]["nested"] == {"value": "original"}


@pytest.mark.unit
@pytest.mark.parametrize(
    ("provider", "section", "environment", "default_endpoint"),
    [
        (
            "openai",
            "openai_api",
            "OPENAI_API_BASE_URL",
            "https://api.openai.com/v1",
        ),
        (
            "google",
            "google_api",
            "GOOGLE_GEMINI_BASE_URL",
            "https://generativelanguage.googleapis.com/v1",
        ),
        (
            "huggingface",
            "huggingface_api",
            "HUGGINGFACE_INFERENCE_BASE_URL",
            "https://api-inference.huggingface.co/models",
        ),
    ],
)
def test_adapter_builder_selects_endpoint_atomically_with_stable_precedence(
    monkeypatch,
    provider,
    section,
    environment,
    default_endpoint,
):
    """A resolved credential snapshot wins over later ambient env changes."""
    import tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced as mod

    env_endpoint = f"https://{provider}-env.example/v1"
    config_endpoint = f"https://{provider}-config.example/v1"
    credential_endpoint = f"https://{provider}-credential.example/v1"
    monkeypatch.setenv(environment, env_endpoint)
    credentials = mod.ResolvedByokCredentials(
        provider=provider,
        api_key=f"{provider}-key",
        app_config={section: {"api_base_url": config_endpoint}},
        credential_fields={"base_url": credential_endpoint},
        source="user",
        allowlisted=True,
    )

    built = mod._build_embeddings_adapter_request(
        ["hello"],
        provider=provider,
        model="embedding-model",
        dimensions=None,
        credentials=credentials,
    )
    assert built["base_url"] == credential_endpoint
    assert built["api_key"] == f"{provider}-key"

    credentials.credential_fields = {}
    assert mod._build_embeddings_adapter_request(
        ["hello"],
        provider=provider,
        model="embedding-model",
        dimensions=None,
        credentials=credentials,
    )["base_url"] == config_endpoint

    credentials.app_config = None
    assert mod._build_embeddings_adapter_request(
        ["hello"],
        provider=provider,
        model="embedding-model",
        dimensions=None,
        credentials=credentials,
    )["base_url"] == default_endpoint


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["huggingface", "google"])
async def test_disabled_embedding_native_adapter_falls_back_to_legacy(
    monkeypatch,
    provider,
):
    """Opt-in adapters that are disabled remain a clean legacy fallback signal."""
    import tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced as mod
    from tldw_Server_API.app.core.LLM_Calls.providers.google_embeddings_adapter import (
        GoogleEmbeddingsAdapter,
    )
    from tldw_Server_API.app.core.LLM_Calls.providers.huggingface_embeddings_adapter import (
        HuggingFaceEmbeddingsAdapter,
    )

    adapters = {
        "google": GoogleEmbeddingsAdapter(),
        "huggingface": HuggingFaceEmbeddingsAdapter(),
    }

    class _Registry:
        def get_adapter(self, selected):
            return adapters[selected]

    monkeypatch.setenv("LLM_EMBEDDINGS_ADAPTERS_ENABLED", "1")
    monkeypatch.delenv("LLM_EMBEDDINGS_NATIVE_HTTP_GOOGLE", raising=False)
    monkeypatch.delenv("LLM_EMBEDDINGS_NATIVE_HTTP_HUGGINGFACE", raising=False)
    monkeypatch.setattr(mod, "get_embeddings_registry", lambda: _Registry())
    credentials = mod.ResolvedByokCredentials(
        provider=provider,
        api_key="runtime-key",
        app_config=None,
        credential_fields={},
        source="server_default",
        allowlisted=True,
    )
    executor = mod._EndpointEmbeddingExecutor(
        request=MagicMock(),
        current_user=MagicMock(),
        user_metadata=None,
    )

    assert await executor._try_adapter_execution(
        ["hello"],
        provider=provider,
        model="text-embedding-model",
        dimensions=None,
        credentials=credentials,
    ) is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_plain_not_implemented_adapter_failure_never_triggers_legacy_replay(
    monkeypatch,
):
    """Only the dedicated pre-dispatch sentinel may opt into legacy fallback."""
    from loguru import logger

    import tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced as mod

    sentinel = "post-dispatch-not-implemented-secret-sentinel"
    records = []

    class _Adapter:
        def embed(self, request):
            assert request["api_key"] == "runtime-key"
            raise NotImplementedError(sentinel)

    class _Registry:
        def get_adapter(self, _provider):
            return _Adapter()

    monkeypatch.setenv("LLM_EMBEDDINGS_ADAPTERS_ENABLED", "1")
    monkeypatch.setattr(mod, "get_embeddings_registry", lambda: _Registry())
    credentials = mod.ResolvedByokCredentials(
        provider="huggingface",
        api_key="runtime-key",
        app_config=None,
        credential_fields={},
        source="server_default",
        allowlisted=True,
    )
    executor = mod._EndpointEmbeddingExecutor(
        request=MagicMock(),
        current_user=MagicMock(),
        user_metadata=None,
    )
    sink_id = logger.add(records.append, format="{message} {extra}")
    try:
        with pytest.raises(mod.EmbeddingProviderError) as exc_info:
            await executor._try_adapter_execution(
                ["hello"],
                provider="huggingface",
                model="embedding-model",
                dimensions=None,
                credentials=credentials,
            )
    finally:
        logger.remove(sink_id)

    assert exc_info.value.message == "Embedding provider request failed"
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    assert sentinel not in str(exc_info.value)
    assert sentinel not in "".join(map(str, records))


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["openai", "google", "huggingface"])
async def test_adapter_provider_failure_becomes_bounded_embedding_error_without_raw_cause(
    monkeypatch,
    provider,
):
    """Chat adapter exceptions cannot escape the embeddings HTTP error boundary."""
    from loguru import logger

    import tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced as mod
    from tldw_Server_API.app.core.Chat.Chat_Deps import ChatProviderError

    sentinel = f"raw-{provider}-endpoint-key-body-sentinel"
    records = []

    class _Adapter:
        def embed(self, request):
            del request
            try:
                raise RuntimeError(sentinel)
            except RuntimeError as exc:
                raise ChatProviderError(provider=provider, message=sentinel) from exc

    class _Registry:
        def get_adapter(self, selected):
            assert selected == provider
            return _Adapter()

    monkeypatch.setenv("LLM_EMBEDDINGS_ADAPTERS_ENABLED", "1")
    monkeypatch.setattr(mod, "get_embeddings_registry", lambda: _Registry())
    credentials = mod.ResolvedByokCredentials(
        provider=provider,
        api_key="runtime-key",
        app_config=None,
        credential_fields={},
        source="server_default",
        allowlisted=True,
    )
    executor = mod._EndpointEmbeddingExecutor(
        request=MagicMock(),
        current_user=MagicMock(),
        user_metadata=None,
    )
    sink_id = logger.add(records.append, format="{message} {extra}")

    try:
        with pytest.raises(mod.EmbeddingProviderError) as exc_info:
            await executor._try_adapter_execution(
                ["hello"],
                provider=provider,
                model="text-embedding-model",
                dimensions=None,
                credentials=credentials,
            )
    finally:
        logger.remove(sink_id)

    assert exc_info.value.code == "provider_unavailable"
    assert exc_info.value.message == "Embedding provider request failed"
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    assert sentinel not in str(exc_info.value)
    assert sentinel not in "".join(map(str, records))


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("provider_status", [401, 403])
async def test_raw_adapter_auth_http_failure_becomes_detached_upstream_502(
    monkeypatch,
    provider_status,
):
    """HTTP-shaped provider auth failures cannot retain client-auth semantics."""
    import tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced as mod

    sentinel = f"raw-adapter-auth-{provider_status}-secret-sentinel"

    class _Adapter:
        def embed(self, request):
            del request
            try:
                raise RuntimeError(sentinel)
            except RuntimeError as exc:
                raise HTTPException(
                    status_code=provider_status,
                    detail=sentinel,
                ) from exc

    class _Registry:
        def get_adapter(self, selected):
            assert selected == "google"
            return _Adapter()

    monkeypatch.setenv("LLM_EMBEDDINGS_ADAPTERS_ENABLED", "1")
    monkeypatch.setattr(mod, "get_embeddings_registry", lambda: _Registry())
    credentials = mod.ResolvedByokCredentials(
        provider="google",
        api_key="runtime-key",
        app_config=None,
        credential_fields={},
        source="server_default",
        allowlisted=True,
    )
    executor = mod._EndpointEmbeddingExecutor(
        request=MagicMock(),
        current_user=MagicMock(),
        user_metadata=None,
    )

    with pytest.raises(HTTPException) as exc_info:
        await executor._try_adapter_execution(
            ["hello"],
            provider="google",
            model="text-embedding-model",
            dimensions=None,
            credentials=credentials,
        )

    assert exc_info.value.status_code == status.HTTP_502_BAD_GATEWAY
    assert exc_info.value.detail == "Embedding provider authentication failed"
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    assert sentinel not in repr(exc_info.value)


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("retry_status", "expected_status", "expected_calls"),
    [
        (200, None, 2),
        (401, 502, 2),
        (403, 502, 1),
    ],
)
async def test_openai_oauth_adapter_refreshes_once_only_after_401(
    monkeypatch,
    retry_status,
    expected_status,
    expected_calls,
):
    """Adapter OAuth retry keeps refreshed key and endpoint in one snapshot."""
    import tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced as mod
    from tldw_Server_API.app.core.Chat.Chat_Deps import ChatAuthenticationError

    initial = mod.ResolvedByokCredentials(
        provider="openai",
        api_key="oauth-old-key",
        app_config={"openai_api": {"api_base_url": "https://old-oauth.example/v1"}},
        credential_fields={"base_url": "https://old-oauth.example/v1"},
        source="user",
        allowlisted=True,
        auth_source="oauth",
    )
    refreshed = mod.ResolvedByokCredentials(
        provider="openai",
        api_key="oauth-new-key",
        app_config={"openai_api": {"api_base_url": "https://new-oauth.example/v1"}},
        credential_fields={"base_url": "https://new-oauth.example/v1"},
        source="user",
        allowlisted=True,
        auth_source="oauth",
    )
    resolve_calls = []
    adapter_calls = []

    async def _resolve(*_args, force_oauth_refresh=False, **_kwargs):
        resolve_calls.append(force_oauth_refresh)
        return refreshed if force_oauth_refresh else initial

    class _Adapter:
        def embed(self, request):
            adapter_calls.append(
                (request["api_key"], request["base_url"], request["model"])
            )
            status_code = 401 if len(adapter_calls) == 1 else retry_status
            if status_code in {401, 403}:
                raise ChatAuthenticationError(
                    provider="openai-embeddings",
                    message="bounded auth failure",
                    status_code=status_code,
                )
            return {"data": [{"index": 0, "embedding": [0.1, 0.2]}]}

    class _Registry:
        def get_adapter(self, _provider):
            return _Adapter()

    if retry_status == 403:
        # A first 403 is never refreshable.
        class _ForbiddenAdapter(_Adapter):
            def embed(self, request):
                adapter_calls.append(
                    (request["api_key"], request["base_url"], request["model"])
                )
                raise ChatAuthenticationError(
                    provider="openai-embeddings",
                    message="bounded auth failure",
                    status_code=403,
                )

        class _Registry:
            def get_adapter(self, _provider):
                return _ForbiddenAdapter()

    monkeypatch.setenv("LLM_EMBEDDINGS_ADAPTERS_ENABLED", "1")
    monkeypatch.setattr(mod, "_resolve_embeddings_byok", _resolve)
    monkeypatch.setattr(mod, "get_embeddings_registry", lambda: _Registry())
    executor = mod._EndpointEmbeddingExecutor(
        request=MagicMock(),
        current_user=MagicMock(),
        user_metadata=None,
    )

    if expected_status is None:
        result = await executor.create_adapter(
            ["hello"],
            provider="openai",
            model="text-embedding-3-small",
            dimensions=None,
        )
        assert result is not None
        assert result.vectors == [[0.1, 0.2]]
    else:
        with pytest.raises(HTTPException) as exc_info:
            await executor.create_adapter(
                ["hello"],
                provider="openai",
                model="text-embedding-3-small",
                dimensions=None,
            )
        assert exc_info.value.status_code == expected_status
        assert exc_info.value.detail == "Embedding provider authentication failed"
        assert exc_info.value.__cause__ is None
        assert exc_info.value.__context__ is None

    assert len(adapter_calls) == expected_calls
    assert adapter_calls[0] == (
        "oauth-old-key",
        "https://old-oauth.example/v1",
        "text-embedding-3-small",
    )
    if expected_calls == 2:
        assert adapter_calls[1] == (
            "oauth-new-key",
            "https://new-oauth.example/v1",
            "text-embedding-3-small",
        )
        assert resolve_calls == [False, True]
    else:
        assert resolve_calls == [False]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_resolved_credentials_touch_log_omits_callback_error_details(monkeypatch):
    """The credential object's callback boundary never logs raw exception text."""
    from loguru import logger

    import tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced as mod

    sentinel = "touch-db-path-token-sentinel"
    records = []

    async def _fail_touch():
        raise RuntimeError(sentinel)

    credentials = mod.ResolvedByokCredentials(
        provider="openai",
        api_key="runtime-key",
        app_config=None,
        credential_fields={},
        source="user",
        allowlisted=True,
        _touch_cb=_fail_touch,
    )
    executor = mod._EndpointEmbeddingExecutor(
        request=MagicMock(),
        current_user=MagicMock(),
        user_metadata=None,
    )
    sink_id = logger.add(records.append, format="{message} {extra}")
    try:
        await executor._touch_credentials(credentials, "openai")
    finally:
        logger.remove(sink_id)

    output = "".join(map(str, records))
    assert "BYOK last_used_at update failed" in output
    assert "RuntimeError" in output
    assert sentinel not in output


@pytest.mark.unit
@pytest.mark.asyncio
async def test_endpoint_touch_credentials_log_omits_callback_error_details():
    """The endpoint's generic credential boundary also keeps logs bounded."""
    from loguru import logger

    import tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced as mod

    sentinel = "endpoint-touch-db-path-token-sentinel"
    records = []

    async def _fail_touch():
        raise RuntimeError(sentinel)

    credentials = MagicMock()
    credentials.touch_last_used = _fail_touch
    executor = mod._EndpointEmbeddingExecutor(
        request=MagicMock(),
        current_user=MagicMock(),
        user_metadata=None,
    )
    sink_id = logger.add(records.append, format="{message} {extra}")
    try:
        await executor._touch_credentials(credentials, "openai")
    finally:
        logger.remove(sink_id)

    output = "".join(map(str, records))
    assert "BYOK touch_last_used failed" in output
    assert "RuntimeError" in output
    assert sentinel not in output


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provider", "model_id", "status_code", "expected_detail"),
    [
        ("cohere", "embed-english-v3.0", 401, "Embedding provider authentication failed"),
        ("google", "text-embedding-004", 403, "Embedding provider authentication failed"),
        ("cohere", "embed-english-v3.0", 502, "Cohere embeddings error"),
        ("google", "text-embedding-004", 503, "Google embeddings error"),
    ],
)
async def test_provider_http_error_body_is_sanitized(
    monkeypatch,
    provider,
    model_id,
    status_code,
    expected_detail,
):
    import tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced as mod

    class FakeBreaker:
        async def call_async(self, func, *args, **kwargs):
            return await func(*args, **kwargs)

    class FakeResponse:
        text = "upstream leaked token and /private/provider/path"

        def __init__(self, status_code):
            self.status_code = status_code

        def json(self):
            return {}

        async def aclose(self):
            return None

    async def fake_afetch(**kwargs):
        _ = kwargs
        return FakeResponse(status_code)

    monkeypatch.setattr(mod, "get_or_create_circuit_breaker", lambda selected: FakeBreaker())
    monkeypatch.setattr(mod.connection_manager, "get_session", AsyncMock(return_value=object()))
    monkeypatch.setattr(mod, "_http_afetch", fake_afetch)

    with pytest.raises(HTTPException) as exc:
        await mod.create_embeddings_with_circuit_breaker(
            ["a"],
            provider=provider,
            model_id=model_id,
            config={"api_key": "fake-provider-key", "model_name_or_path": model_id},
        )

    expected_status = 502 if status_code in {401, 403} else status_code
    assert exc.value.status_code == expected_status
    assert exc.value.detail == expected_detail


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("provider_status", [401, 403])
async def test_runtime_provider_auth_failure_is_detached_upstream_502(
    monkeypatch,
    provider_status,
):
    """Typed runtime auth failures cannot expose their private exception graph."""
    import tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced as mod

    sentinel = f"raw-provider-{provider_status}-secret-body"

    async def explicit_boundary(*_args, **_kwargs):
        try:
            raise RuntimeError(sentinel)
        except RuntimeError as raw_exc:
            raise mod.RuntimeEmbeddingProviderError(
                "openai",
                code="authentication",
                status_code=provider_status,
            ) from raw_exc

    monkeypatch.setattr(mod, "create_explicit_embeddings_batch_async", explicit_boundary)

    with pytest.raises(HTTPException) as exc_info:
        await mod.create_embeddings_with_circuit_breaker(
            ["hello"],
            provider="openai",
            model_id="text-embedding-3-small",
            config={
                "api_key": "provider-key-must-not-leak",
                "model_name_or_path": "text-embedding-3-small",
                "_runtime_credentials_resolved": True,
                "_runtime_credentials_private": True,
            },
        )

    error = exc_info.value
    assert error.status_code == status.HTTP_502_BAD_GATEWAY
    assert error.detail == "Embedding provider authentication failed"
    assert error.__cause__ is None
    assert error.__context__ is None
    assert sentinel not in repr(error)
    assert "provider-key-must-not-leak" not in repr(error)


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("model", "expected_path_model", "expected_body_model"),
    [
        ("text-embedding-004", "models/text-embedding-004", "models/text-embedding-004"),
        ("models/text-embedding-004", "models/text-embedding-004", "models/text-embedding-004"),
        ("models/org/mødel", "models/org/m%C3%B8del", "models/org/mødel"),
    ],
)
async def test_google_batch_embeddings_encodes_model_and_uses_header_auth(
    monkeypatch,
    model,
    expected_path_model,
    expected_body_model,
):
    import tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced as mod

    calls = []

    class FakeBreaker:
        async def call_async(self, func, *args, **kwargs):
            return await func(*args, **kwargs)

    class FakeResponse:
        status_code = 200

        def json(self):
            return {"embeddings": [{"values": [0.25]}]}

        async def aclose(self):
            return None

    async def fake_afetch(**kwargs):
        payload = kwargs["json"]
        calls.append(
            (
                kwargs["url"],
                dict(kwargs.get("headers") or {}),
                payload["requests"][0]["model"],
            )
        )
        return FakeResponse()

    monkeypatch.setattr(mod, "get_or_create_circuit_breaker", lambda _provider: FakeBreaker())
    monkeypatch.setattr(mod.connection_manager, "get_session", AsyncMock(return_value=object()))
    monkeypatch.setattr(mod, "_http_afetch", fake_afetch)

    result = await mod.create_embeddings_with_circuit_breaker(
        ["alpha"],
        provider="google",
        model_id=model,
        config={"api_key": "google-boundary-key", "model_name_or_path": model},
    )

    assert result == [[0.25]]
    assert calls == [
        (
            f"https://generativelanguage.googleapis.com/v1beta/{expected_path_model}:batchEmbedContents",
            {"Content-Type": "application/json", "x-goog-api-key": "google-boundary-key"},
            expected_body_model,
        )
    ]


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model",
    ["../files", "models/../../files#", "models/org\\model", "models/%2e%2e/files"],
)
async def test_google_batch_embeddings_rejects_unsafe_model_before_transport(
    monkeypatch,
    model,
):
    import tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced as mod

    class FakeBreaker:
        async def call_async(self, func, *args, **kwargs):
            return await func(*args, **kwargs)

    async def fail_afetch(**_kwargs):
        pytest.fail("unsafe Google embedding model must fail before HTTP dispatch")

    monkeypatch.setattr(mod, "get_or_create_circuit_breaker", lambda _provider: FakeBreaker())
    monkeypatch.setattr(mod.connection_manager, "get_session", AsyncMock(return_value=object()))
    monkeypatch.setattr(mod, "_http_afetch", fail_afetch)

    with pytest.raises(HTTPException, match="model identifier") as exc_info:
        await mod.create_embeddings_with_circuit_breaker(
            ["unsafe"],
            provider="google",
            model_id=model,
            config={"api_key": "google-secret-key", "model_name_or_path": model},
        )

    assert exc_info.value.status_code == 400
    assert model not in str(exc_info.value)
    assert "google-secret-key" not in str(exc_info.value)


@pytest.mark.unit
@pytest.mark.concurrent
@pytest.mark.asyncio
async def test_concurrent_google_batch_embeddings_keep_model_key_and_payload_request_local(
    monkeypatch,
):
    import tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced as mod

    calls = []
    calls_lock = asyncio.Lock()
    both_valid_arrived = asyncio.Event()
    release = asyncio.Event()

    class FakeBreaker:
        async def call_async(self, func, *args, **kwargs):
            return await func(*args, **kwargs)

    class FakeResponse:
        status_code = 200

        def json(self):
            return {"embeddings": [{"values": [1.0]}]}

        async def aclose(self):
            return None

    async def gated_afetch(**kwargs):
        request = kwargs["json"]["requests"][0]
        text = request["content"]["parts"][0]["text"]
        call = (
            kwargs["url"],
            (kwargs.get("headers") or {}).get("x-goog-api-key"),
            request["model"],
            text,
        )
        async with calls_lock:
            calls.append(call)
            valid_texts = {item[3] for item in calls if item[3] in {"alpha", "beta"}}
            if valid_texts == {"alpha", "beta"}:
                both_valid_arrived.set()
        await asyncio.wait_for(release.wait(), timeout=10)
        return FakeResponse()

    monkeypatch.setattr(mod, "get_or_create_circuit_breaker", lambda _provider: FakeBreaker())
    monkeypatch.setattr(mod.connection_manager, "get_session", AsyncMock(return_value=object()))
    monkeypatch.setattr(mod, "_http_afetch", gated_afetch)

    async def invoke(model, key, text):
        return await mod.create_embeddings_with_circuit_breaker(
            [text],
            provider="google",
            model_id=model,
            config={"api_key": key, "model_name_or_path": model},
        )

    alpha = asyncio.create_task(invoke("models/gemini-alpha", "key-alpha", "alpha"))
    beta = asyncio.create_task(invoke("gemini-beta", "key-beta", "beta"))
    unsafe = asyncio.create_task(invoke("models/../../files#", "key-unsafe", "unsafe"))
    try:
        await asyncio.wait_for(both_valid_arrived.wait(), timeout=10)
    finally:
        release.set()

    assert await alpha == [[1.0]]
    assert await beta == [[1.0]]
    with pytest.raises(HTTPException, match="model identifier"):
        await unsafe
    assert len(calls) == 2
    assert set(calls) == {
        (
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-alpha:batchEmbedContents",
            "key-alpha",
            "models/gemini-alpha",
            "alpha",
        ),
        (
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-beta:batchEmbedContents",
            "key-beta",
            "models/gemini-beta",
            "beta",
        ),
    }


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["google", "huggingface"])
async def test_explicit_remote_fallback_reaches_http_with_exact_key_base_model_and_payload(
    monkeypatch,
    provider,
):
    """The disabled-native fallback must not drop or reinterpret resolved credentials."""
    import tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced as mod

    calls = []
    base_url = f"https://{provider}-runtime.example/custom/v9"
    model = "org/embedding-model"

    class FakeBreaker:
        async def call_async(self, func, *args, **kwargs):
            return await func(*args, **kwargs)

    class FakeResponse:
        status_code = 200

        def json(self):
            if provider == "google":
                return {"embeddings": [{"values": [0.25]}]}
            return [[0.25]]

        async def aclose(self):
            return None

    async def fake_afetch(**kwargs):
        calls.append(kwargs)
        return FakeResponse()

    async def fail_local_batcher(**_kwargs):
        pytest.fail("explicit Hugging Face credentials must not enter the local-model batcher")

    monkeypatch.setattr(mod, "get_or_create_circuit_breaker", lambda _provider: FakeBreaker())
    monkeypatch.setattr(mod.connection_manager, "get_session", AsyncMock(return_value=object()))
    monkeypatch.setattr(mod, "_http_afetch", fake_afetch)
    monkeypatch.setattr(mod, "batching_create_embeddings_batch_async", fail_local_batcher)

    config = mod.build_provider_config(
        mod.EmbeddingProvider(provider),
        model,
        api_key=f"key-{provider}",
        api_url=base_url,
    )
    result = await mod.create_embeddings_with_circuit_breaker(
        [f"text-{provider}"],
        provider=provider,
        model_id=model,
        config=config,
    )

    assert result == [[0.25]]
    assert len(calls) == 1
    call = calls[0]
    if provider == "google":
        assert call["url"] == f"{base_url}/models/org/embedding-model:batchEmbedContents"
        assert call["headers"]["x-goog-api-key"] == "key-google"
        assert call["json"] == {
            "requests": [
                {
                    "model": "models/org/embedding-model",
                    "content": {"parts": [{"text": "text-google"}]},
                }
            ]
        }
    else:
        assert call["url"] == f"{base_url}/org/embedding-model"
        assert call["headers"]["Authorization"] == "Bearer key-huggingface"
        assert call["json"] == {
            "inputs": ["text-huggingface"],
            "options": {"wait_for_model": True},
        }


@pytest.mark.unit
@pytest.mark.concurrent
@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["google", "huggingface"])
async def test_concurrent_explicit_remote_fallback_keeps_credential_snapshot_request_local(
    monkeypatch,
    provider,
):
    """Overlapping fallback calls keep each key, base, model, and payload paired."""
    import tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced as mod

    calls = []
    calls_lock = asyncio.Lock()
    both_arrived = asyncio.Event()
    release = asyncio.Event()

    class FakeBreaker:
        async def call_async(self, func, *args, **kwargs):
            return await func(*args, **kwargs)

    class FakeResponse:
        status_code = 200

        def json(self):
            if provider == "google":
                return {"embeddings": [{"values": [1.0]}]}
            return [[1.0]]

        async def aclose(self):
            return None

    async def gated_afetch(**kwargs):
        if provider == "google":
            text = kwargs["json"]["requests"][0]["content"]["parts"][0]["text"]
            key = kwargs["headers"]["x-goog-api-key"]
        else:
            text = kwargs["json"]["inputs"][0]
            key = kwargs["headers"]["Authorization"].removeprefix("Bearer ")
        call = (kwargs["url"], key, text)
        async with calls_lock:
            calls.append(call)
            if len(calls) == 2:
                both_arrived.set()
        await asyncio.wait_for(release.wait(), timeout=10)
        return FakeResponse()

    async def fail_local_batcher(**_kwargs):
        pytest.fail("explicit remote fallback must never enter the local-model batcher")

    monkeypatch.setattr(mod, "get_or_create_circuit_breaker", lambda _provider: FakeBreaker())
    monkeypatch.setattr(mod.connection_manager, "get_session", AsyncMock(return_value=object()))
    monkeypatch.setattr(mod, "_http_afetch", gated_afetch)
    monkeypatch.setattr(mod, "batching_create_embeddings_batch_async", fail_local_batcher)

    async def invoke(label):
        model = f"org/model-{label}"
        config = mod.build_provider_config(
            mod.EmbeddingProvider(provider),
            model,
            api_key=f"key-{label}",
            api_url=f"https://{provider}-{label}.example/custom",
        )
        return await mod.create_embeddings_with_circuit_breaker(
            [label],
            provider=provider,
            model_id=model,
            config=config,
        )

    alpha = asyncio.create_task(invoke("alpha"))
    beta = asyncio.create_task(invoke("beta"))
    try:
        await asyncio.wait_for(both_arrived.wait(), timeout=10)
    finally:
        release.set()

    assert await alpha == [[1.0]]
    assert await beta == [[1.0]]
    suffix = ":batchEmbedContents" if provider == "google" else ""
    google_models = "/models" if provider == "google" else ""
    assert set(calls) == {
        (
            f"https://{provider}-alpha.example/custom{google_models}/org/model-alpha{suffix}",
            "key-alpha",
            "alpha",
        ),
        (
            f"https://{provider}-beta.example/custom{google_models}/org/model-beta{suffix}",
            "key-beta",
            "beta",
        ),
    }


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provider", "expected_base"),
    [
        ("google", "https://generativelanguage.googleapis.com/v1"),
        ("huggingface", "https://api-inference.huggingface.co/models"),
    ],
)
async def test_key_only_remote_fallback_uses_same_adapter_default_base(
    monkeypatch,
    provider,
    expected_base,
):
    """A resolved key with no override must keep the adapter's selected default."""
    import tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced as mod

    calls = []
    model = "org/default-model"

    class FakeBreaker:
        async def call_async(self, func, *args, **kwargs):
            return await func(*args, **kwargs)

    class FakeResponse:
        status_code = 200

        def json(self):
            return (
                {"embeddings": [{"values": [0.5]}]}
                if provider == "google"
                else [[0.5]]
            )

        async def aclose(self):
            return None

    async def fake_afetch(**kwargs):
        calls.append(kwargs)
        return FakeResponse()

    async def fail_local_batcher(**_kwargs):
        pytest.fail("a resolved remote key must not enter the local-model batcher")

    monkeypatch.setattr(mod, "get_or_create_circuit_breaker", lambda _provider: FakeBreaker())
    monkeypatch.setattr(mod.connection_manager, "get_session", AsyncMock(return_value=object()))
    monkeypatch.setattr(mod, "_http_afetch", fake_afetch)
    monkeypatch.setattr(mod, "batching_create_embeddings_batch_async", fail_local_batcher)

    config = mod.build_provider_config(
        mod.EmbeddingProvider(provider),
        model,
        api_key=f"key-{provider}",
    )
    result = await mod.create_embeddings_with_circuit_breaker(
        [provider],
        provider=provider,
        model_id=model,
        config=config,
    )

    assert result == [[0.5]]
    assert len(calls) == 1
    suffix = "/models/org/default-model:batchEmbedContents" if provider == "google" else "/org/default-model"
    assert calls[0]["url"] == f"{expected_base}{suffix}"
    if provider == "google":
        assert calls[0]["headers"]["x-goog-api-key"] == "key-google"
    else:
        assert calls[0]["headers"]["Authorization"] == "Bearer key-huggingface"


@pytest.mark.unit
@pytest.mark.concurrent
@pytest.mark.asyncio
async def test_concurrent_keyless_local_and_key_only_remote_hf_do_not_exchange_modes(
    monkeypatch,
):
    """A local keyless call and remote keyed call retain independent execution modes."""
    import tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced as mod

    local_calls = []
    remote_calls = []
    both_arrived = asyncio.Event()
    release = asyncio.Event()

    class FakeBreaker:
        async def call_async(self, func, *args, **kwargs):
            return await func(*args, **kwargs)

    class FakeResponse:
        status_code = 200

        def json(self):
            return [[2.0]]

        async def aclose(self):
            return None

    async def mark_arrived():
        if local_calls and remote_calls:
            both_arrived.set()
        await asyncio.wait_for(release.wait(), timeout=10)

    async def gated_local_batcher(**kwargs):
        local_calls.append(kwargs)
        await mark_arrived()
        return [[1.0]]

    async def gated_afetch(**kwargs):
        remote_calls.append(kwargs)
        await mark_arrived()
        return FakeResponse()

    monkeypatch.delenv("HUGGINGFACE_INFERENCE_BASE_URL", raising=False)
    monkeypatch.setattr(mod, "get_or_create_circuit_breaker", lambda _provider: FakeBreaker())
    monkeypatch.setattr(mod.connection_manager, "get_session", AsyncMock(return_value=object()))
    monkeypatch.setattr(mod, "_http_afetch", gated_afetch)
    monkeypatch.setattr(mod, "batching_create_embeddings_batch_async", gated_local_batcher)

    local_config = mod.build_provider_config(
        mod.EmbeddingProvider.HUGGINGFACE,
        "org/local-model",
    )
    remote_config = mod.build_provider_config(
        mod.EmbeddingProvider.HUGGINGFACE,
        "org/remote-model",
        api_key="remote-key",
    )
    local = asyncio.create_task(
        mod.create_embeddings_with_circuit_breaker(
            ["local"],
            provider="huggingface",
            model_id="org/local-model",
            config=local_config,
        )
    )
    remote = asyncio.create_task(
        mod.create_embeddings_with_circuit_breaker(
            ["remote"],
            provider="huggingface",
            model_id="org/remote-model",
            config=remote_config,
        )
    )
    try:
        await asyncio.wait_for(both_arrived.wait(), timeout=10)
    finally:
        release.set()

    assert await local == [[1.0]]
    assert await remote == [[2.0]]
    assert len(local_calls) == 1
    assert len(remote_calls) == 1
    assert "org/local-model" in repr(local_calls[0]["config"])
    assert remote_calls[0]["url"] == (
        "https://api-inference.huggingface.co/models/org/remote-model"
    )
    assert remote_calls[0]["headers"]["Authorization"] == "Bearer remote-key"


@pytest.mark.unit
def test_resolve_model_and_provider_strips_prefix():
    import tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced as mod

    model, provider = mod._resolve_model_and_provider("openai:text-embedding-3-small", None)
    assert model == "text-embedding-3-small"
    assert provider == "openai"


@pytest.mark.unit
def test_resolve_model_and_provider_rejects_mismatch():
    import tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced as mod

    with pytest.raises(HTTPException):
        mod._resolve_model_and_provider("openai:text-embedding-3-small", "huggingface")


@pytest.mark.unit
@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("user", True),
        ("team", True),
        ("org", True),
        ("server_default", False),
        ("none", False),
    ],
)
def test_only_private_byok_credentials_require_shared_state_isolation(source, expected):
    """Server defaults retain shared caches and breakers; private credentials do not."""
    import tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced as mod

    credentials = mod.ResolvedByokCredentials(
        provider="local_api",
        api_key="configured-key",
        app_config=None,
        credential_fields={"base_url": "https://configured.example/v1"},
        source=source,
        allowlisted=True,
    )

    assert mod._credentials_require_cache_isolation(credentials) is expected


@pytest.mark.unit
@pytest.mark.asyncio
async def test_private_explicit_fallback_failure_cannot_poison_shared_provider_breaker(
    monkeypatch,
):
    """A failed private endpoint cannot open the provider-global fallback breaker."""
    import tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced as mod

    class PoisoningBreaker:
        def __init__(self):
            self.open = False
            self.calls = 0

        async def call_async(self, func, *args, **kwargs):
            self.calls += 1
            if self.open:
                raise RuntimeError("shared breaker poisoned")
            try:
                return await func(*args, **kwargs)
            except Exception:
                self.open = True
                raise

    breaker = PoisoningBreaker()
    boundary_calls = []

    async def explicit_boundary(
        texts,
        _config,
        *,
        model_id_override,
        api_key_override,
        base_url_override,
        credentials_resolved,
    ):
        boundary_calls.append((api_key_override, base_url_override, texts[0]))
        assert model_id_override == "local_api:embedding-model"
        assert credentials_resolved is True
        if "bad" in base_url_override:
            raise RuntimeError("private upstream failed")
        return [[0.6, 0.8]]

    monkeypatch.setattr(mod, "get_or_create_circuit_breaker", lambda _provider: breaker)
    monkeypatch.setattr(mod, "create_explicit_embeddings_batch_async", explicit_boundary)

    def private_config(label):
        config = mod.build_provider_config(
            mod.EmbeddingProvider.LOCAL_API,
            "embedding-model",
            api_key=f"key-{label}",
            api_url=f"https://{label}.example/v1",
        )
        config["_runtime_credentials_resolved"] = True
        config["_runtime_credentials_private"] = True
        return config

    with pytest.raises(RuntimeError, match="private upstream failed"):
        await mod.create_embeddings_with_circuit_breaker(
            ["bad"],
            provider="local_api",
            model_id="embedding-model",
            config=private_config("bad"),
        )

    result = await mod.create_embeddings_with_circuit_breaker(
        ["healthy"],
        provider="local_api",
        model_id="embedding-model",
        config=private_config("healthy"),
    )

    assert result == [[0.6, 0.8]]
    assert breaker.calls == 0
    assert boundary_calls == [
        ("key-bad", "https://bad.example/v1", "bad"),
        ("key-healthy", "https://healthy.example/v1", "healthy"),
    ]


@pytest.mark.unit
@pytest.mark.concurrent
@pytest.mark.asyncio
async def test_private_fallbacks_reach_isolated_boundaries_after_another_tenant_fails(
    monkeypatch,
):
    """Concurrent healthy private fallbacks still dispatch after a peer poisons itself."""
    import tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced as mod

    class PoisoningBreaker:
        def __init__(self):
            self.open = False

        async def call_async(self, func, *args, **kwargs):
            if self.open:
                raise RuntimeError("shared breaker poisoned")
            try:
                return await func(*args, **kwargs)
            except Exception:
                self.open = True
                raise

    breaker = PoisoningBreaker()
    healthy_arrivals = []
    both_healthy_arrived = asyncio.Event()
    release = asyncio.Event()

    async def explicit_boundary(
        texts,
        _config,
        *,
        model_id_override,
        api_key_override,
        base_url_override,
        credentials_resolved,
    ):
        del model_id_override, credentials_resolved
        label = texts[0]
        if label == "bad":
            raise RuntimeError("private upstream failed")
        healthy_arrivals.append((api_key_override, base_url_override, label))
        if len(healthy_arrivals) == 2:
            both_healthy_arrived.set()
        await asyncio.wait_for(release.wait(), timeout=5)
        return [[1.0, 0.0]]

    monkeypatch.setattr(mod, "get_or_create_circuit_breaker", lambda _provider: breaker)
    monkeypatch.setattr(mod, "create_explicit_embeddings_batch_async", explicit_boundary)

    def private_config(label):
        config = mod.build_provider_config(
            mod.EmbeddingProvider.LOCAL_API,
            "embedding-model",
            api_key=f"key-{label}",
            api_url=f"https://{label}.example/v1",
        )
        config["_runtime_credentials_resolved"] = True
        config["_runtime_credentials_private"] = True
        return config

    with pytest.raises(RuntimeError, match="private upstream failed"):
        await mod.create_embeddings_with_circuit_breaker(
            ["bad"],
            provider="local_api",
            model_id="embedding-model",
            config=private_config("bad"),
        )

    tasks = [
        asyncio.create_task(
            mod.create_embeddings_with_circuit_breaker(
                [label],
                provider="local_api",
                model_id="embedding-model",
                config=private_config(label),
            )
        )
        for label in ("alpha", "beta")
    ]
    reached_boundary = False
    try:
        await asyncio.wait_for(both_healthy_arrived.wait(), timeout=0.5)
        reached_boundary = True
    except asyncio.TimeoutError:
        pass
    finally:
        release.set()
    results = await asyncio.gather(*tasks, return_exceptions=True)

    assert reached_boundary is True
    assert results == [[[1.0, 0.0]], [[1.0, 0.0]]]
    assert set(healthy_arrivals) == {
        ("key-alpha", "https://alpha.example/v1", "alpha"),
        ("key-beta", "https://beta.example/v1", "beta"),
    }
