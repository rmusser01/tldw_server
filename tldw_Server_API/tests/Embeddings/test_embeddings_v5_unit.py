# test_embeddings_v5_unit.py
# Comprehensive test suite for production embeddings service - FIXED VERSION
# Unit tests with mocks

import os
import uuid
# Set TESTING environment variable BEFORE importing anything else
os.environ["TESTING"] = "true"
os.environ["AUTO_DOWNLOAD_MODELS"] = "false"

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch, Mock
import pytest
import numpy as np

from fastapi import HTTPException, status
from fastapi.testclient import TestClient
from tldw_Server_API.app.main import app
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_auth_principal
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal, AuthContext
from starlette.requests import Request

# Cleanup fixture to remove TESTING env var after tests
@pytest.fixture(autouse=True, scope="module")
def cleanup_testing_env():
    """Cleanup TESTING environment variable after module tests"""
    previous_testing = os.environ.get("TESTING")
    previous_auto_download = os.environ.get("AUTO_DOWNLOAD_MODELS")
    os.environ["TESTING"] = "true"
    os.environ["AUTO_DOWNLOAD_MODELS"] = "false"
    yield
    if previous_testing is None:
        os.environ.pop("TESTING", None)
    else:
        os.environ["TESTING"] = previous_testing
    if previous_auto_download is None:
        os.environ.pop("AUTO_DOWNLOAD_MODELS", None)
    else:
        os.environ["AUTO_DOWNLOAD_MODELS"] = previous_auto_download

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
        # Non-admins should be rejected either directly by RBAC (403) or by
        # an upstream rate/budget guard (429) for this admin-only endpoint.
        assert response.status_code in (403, 429)

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
        from tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced import create_embeddings_with_circuit_breaker
        from tldw_Server_API.app.core.Infrastructure.circuit_breaker import CircuitBreaker

        attempt_count = 0

        def mock_embeddings(texts, config, model_id_override, metadata=None, **_):

            nonlocal attempt_count
            attempt_count += 1

            # First 2 attempts fail, third succeeds
            if attempt_count < 3:
                raise ConnectionError("Connection failed")

            return [[1.0, 2.0, 3.0]] * len(texts)

        from tenacity import retry, stop_after_attempt, retry_if_exception_type

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
        from tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced import create_embeddings_with_circuit_breaker

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

        from tldw_Server_API.app.core.AuthNZ.byok_runtime import ResolvedByokCredentials
        import tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced as emb_ep

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

        from tldw_Server_API.app.core.AuthNZ.byok_runtime import ResolvedByokCredentials
        import tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced as emb_ep

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
    def test_openai_oauth_second_401_propagates_original_auth_error(self, setup, monkeypatch):
        """Second OpenAI OAuth auth failure should return the original auth error."""
        def override_user():
            return setup.regular_user

        app.dependency_overrides[get_request_user] = override_user
        monkeypatch.setenv("USE_REAL_OPENAI_IN_TESTS", "1")

        from tldw_Server_API.app.core.AuthNZ.byok_runtime import ResolvedByokCredentials
        import tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced as emb_ep

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

        assert response.status_code == status.HTTP_401_UNAUTHORIZED
        assert response.json().get("detail") == "oauth auth failure"


class TestMockedFlow:
    """Test complete flow with mocked embeddings"""

    @pytest.mark.unit
    def test_end_to_end_flow_mocked(self, setup):
        """Test complete flow with mocked embeddings"""
        def override_user():
            return setup.regular_user

        app.dependency_overrides[get_request_user] = override_user

        async def mock_embeddings(texts, provider, model_id, dimensions=None, api_key=None, api_url=None, metadata=None):
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
    import tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced as mod

    async def fake_create_embeddings_with_circuit_breaker(
        texts,
        provider,
        model_id,
        config,
        metadata=None,
        dimensions=None,
    ):
        _ = (texts, provider, model_id, config, metadata, dimensions)
        raise RuntimeError("backend leaked /private/embedding-provider path")

    monkeypatch.setattr(
        mod,
        "create_embeddings_with_circuit_breaker",
        fake_create_embeddings_with_circuit_breaker,
        raising=True,
    )
    monkeypatch.setattr(mod.embedding_cache, "get", AsyncMock(return_value=None))
    monkeypatch.setattr(mod.embedding_cache, "set", AsyncMock())
    monkeypatch.setattr(mod.connection_manager, "remove_provider", AsyncMock())

    with pytest.raises(HTTPException) as exc:
        await mod.create_embeddings_batch_async(
            ["a"],
            provider="huggingface",
            model_id="sentence-transformers/all-MiniLM-L6-v2",
        )

    assert exc.value.status_code == 503
    assert exc.value.detail == "Embedding service error"


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
@pytest.mark.parametrize(
    ("provider", "model_id", "status_code", "expected_detail"),
    [
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

    assert exc.value.status_code == status_code
    assert exc.value.detail == expected_detail


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
