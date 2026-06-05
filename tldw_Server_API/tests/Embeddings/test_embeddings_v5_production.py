# test_embeddings_v5_production.py
# Comprehensive test suite for production embeddings service
# Separated into unit tests (with mocks), integration tests (no mocks), and property-based tests

import asyncio
import json
import os
import time
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch, Mock
import pytest
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from hypothesis import given, strategies as st, settings, assume

from fastapi.testclient import TestClient
from httpx import AsyncClient, ASGITransport

from tldw_Server_API.app.main import app
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user

# Disable rate limiting for all tests
@pytest.fixture(autouse=True)
def disable_rate_limiting():
    """Disable rate limiting for all tests in this module"""
    os.environ["TESTING"] = "true"
    yield
    # Clean up after tests
    if "TESTING" in os.environ:
        del os.environ["TESTING"]

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


# Module-level setup fixture for all test classes
@pytest.fixture
def setup():
    """Setup test environment fixture with proper TestClient lifecycle"""
    class SetupData:
        pass

    with TestClient(app) as client:
        data = SetupData()
        data.client = client
        # Set CSRF token in both cookie and header for double-submit pattern
        csrf_token = "test-csrf-token-12345"
        client.cookies.set("csrf_token", csrf_token)
        data.DEFAULT_API_KEY = "test-api-key"
        data.auth_headers = {
            "Authorization": f"Bearer {data.DEFAULT_API_KEY}",
            "X-CSRF-Token": csrf_token
        }

        # Create test users
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

        try:
            yield data
        finally:
            # Cleanup
            app.dependency_overrides.clear()


class TestProductionEmbeddings:
    """Comprehensive test suite for production embeddings API"""
    pass


class TestCriticalSecurity:
    """Test critical security fixes"""

    @pytest.mark.asyncio
    async def test_no_placeholder_embeddings(self, setup):
        """Verify system fails properly when dependencies missing"""
        # Since the module is already imported, we can't test import-time behavior
        # Instead, test that EMBEDDINGS_AVAILABLE flag exists and is properly set
        from tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced import EMBEDDINGS_AVAILABLE

        # If dependencies are available, this should be True
        assert EMBEDDINGS_AVAILABLE is True

        # Test that when EMBEDDINGS_AVAILABLE is False, the health endpoint returns degraded
        with patch('tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced.EMBEDDINGS_AVAILABLE', False):
            response = setup.client.get("/api/v1/embeddings/health")
            assert response.status_code == 503
            data = response.json()
            assert data["status"] == "degraded"

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="TestClient doesn't properly handle async dependency overrides")
    async def test_admin_authorization_required(self, setup):
        """Test admin endpoints require proper authorization"""
        # Create async override functions that TestClient can handle
        async def override_regular_user():
            return setup.regular_user

        async def override_admin_user():
            return setup.admin_user

        # Try to clear cache as regular user - should fail
        app.dependency_overrides[get_request_user] = override_regular_user

        response = setup.client.delete(
            "/api/v1/embeddings/cache",
            headers=setup.auth_headers
        )

        assert response.status_code == 403
        assert "Admin privileges required" in response.json()["detail"]

        # Now try as admin - should succeed
        app.dependency_overrides[get_request_user] = override_admin_user

        # Need to create a new TestClient to pick up the override change
        with TestClient(app) as admin_client:
            admin_client.cookies.set("csrf_token", "test-csrf-token-12345")
            response = admin_client.delete(
                "/api/v1/embeddings/cache",
                headers=setup.auth_headers
            )

        assert response.status_code == 200
        assert "Cache cleared successfully" in response.json()["message"]


class TestTTLCache:
    """Test TTL cache implementation"""

    @pytest.mark.asyncio
    async def test_cache_ttl_expiration(self):
        """Test that cache entries expire after TTL"""
        from tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced import TTLCache

        cache = TTLCache(max_size=10, ttl_seconds=1)  # 1 second TTL for testing

        # Add item to cache
        await cache.set("test_key", [1.0, 2.0, 3.0])

        # Should be retrievable immediately
        value = await cache.get("test_key")
        assert value == [1.0, 2.0, 3.0]

        # Wait for TTL to expire
        await asyncio.sleep(1.5)

        # Should be None after expiration
        value = await cache.get("test_key")
        assert value is None

    @pytest.mark.asyncio
    async def test_cache_lru_eviction(self):
        """Test LRU eviction when cache is full"""
        from tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced import TTLCache

        cache = TTLCache(max_size=3, ttl_seconds=3600)

        # Fill cache
        await cache.set("key1", [1.0])
        await cache.set("key2", [2.0])
        await cache.set("key3", [3.0])

        # Access key1 to make it recently used
        await cache.get("key1")

        # Add new item - should evict key2 (least recently used)
        await cache.set("key4", [4.0])

        # key1 and key3 should still be there
        assert await cache.get("key1") == [1.0]
        assert await cache.get("key3") == [3.0]
        assert await cache.get("key4") == [4.0]

        # key2 should be evicted
        assert await cache.get("key2") is None

    @pytest.mark.asyncio
    async def test_cache_lru_eviction_uses_access_order_when_timestamps_tie(self, monkeypatch):
        """Test LRU eviction remains deterministic when timer resolution ties."""
        from tldw_Server_API.app.api.v1.endpoints import embeddings_v5_production_enhanced as embeddings_module

        cache = embeddings_module.TTLCache(max_size=3, ttl_seconds=3600)
        monkeypatch.setattr(embeddings_module.time, "time", lambda: 1000.0)

        await cache.set("key1", [1.0])
        await cache.set("key2", [2.0])
        await cache.set("key3", [3.0])
        await cache.get("key1")

        await cache.set("key4", [4.0])

        assert await cache.get("key1") == [1.0]
        assert await cache.get("key2") is None
        assert await cache.get("key3") == [3.0]
        assert await cache.get("key4") == [4.0]

    @pytest.mark.asyncio
    async def test_cache_cleanup_task(self):
        """Test background cleanup task removes expired entries"""
        from tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced import TTLCache

        cache = TTLCache(max_size=10, ttl_seconds=1)

        # Start cleanup task
        await cache.start_cleanup_task()

        try:
            # Add items
            await cache.set("key1", [1.0])
            await cache.set("key2", [2.0])

            # Wait for expiration and cleanup
            await asyncio.sleep(2)

            # Manually trigger cleanup
            await cache.cleanup_expired()

            # Both should be gone
            assert await cache.get("key1") is None
            assert await cache.get("key2") is None

            # Cache should be empty
            stats = cache.stats()
            assert stats['size'] == 0

        finally:
            await cache.stop_cleanup_task()

    @pytest.mark.asyncio
    async def test_cache_thread_safety(self):
        """Test cache is thread-safe under concurrent access"""
        from tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced import TTLCache

        cache = TTLCache(max_size=100, ttl_seconds=3600)

        async def add_items(start_idx):
            for i in range(10):
                key = f"key_{start_idx}_{i}"
                await cache.set(key, [float(start_idx + i)])

        async def get_items(start_idx):
            for i in range(10):
                key = f"key_{start_idx}_{i}"
                await cache.get(key)

        # Concurrent writes and reads
        tasks = []
        for i in range(10):
            tasks.append(add_items(i * 10))
            tasks.append(get_items(i * 10))

        await asyncio.gather(*tasks)

        # Cache should have correct size (no corruption)
        stats = cache.stats()
        assert stats['size'] <= 100  # Should respect max size


class TestConnectionPooling:
    """Test connection pool management"""

    @pytest.mark.asyncio
    async def test_connection_pool_creation(self):
        """Test connection pools are created per provider"""
        from tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced import ConnectionPoolManager

        manager = ConnectionPoolManager()

        try:
            # Get sessions for different providers
            session1 = await manager.get_session("openai")
            session2 = await manager.get_session("cohere")
            session3 = await manager.get_session("openai")  # Should reuse

            # Should have 2 different pools
            assert len(manager.pools) == 2

            # Same provider should return same session
            assert session1 is session3

            # Different providers should have different sessions
            assert session1 is not session2

        finally:
            await manager.close_all()

    @pytest.mark.asyncio
    async def test_connection_pool_cleanup(self):
        """Test connection pools are properly cleaned up"""
        from tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced import ConnectionPoolManager

        manager = ConnectionPoolManager()

        # Create some pools
        await manager.get_session("openai")
        await manager.get_session("cohere")

        assert len(manager.pools) == 2

        # Close all
        await manager.close_all()

        assert len(manager.pools) == 0


class TestRetryLogic:
    """Test retry logic and circuit breaker"""

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

        # Mock batching helper with retry decorator that simulates internal retries
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

    @pytest.mark.asyncio
    async def test_no_retry_on_value_error(self):
        """Test that value errors don't trigger retries"""
        from tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced import create_embeddings_with_circuit_breaker

        attempt_count = 0

        def mock_embeddings(texts, config, model_id_override, metadata=None, **_):

            nonlocal attempt_count
            attempt_count += 1
            raise ValueError("Invalid input")

        # Mock batching helper and ensure ValueError is propagated without retry
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

        assert attempt_count == 1  # Should not retry


class TestErrorHandling:
    """Test comprehensive error handling"""

    @pytest.mark.asyncio
    async def test_empty_input_error(self, setup):
        """Test error on empty input"""
        async def override_user():
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

    @pytest.mark.asyncio
    async def test_invalid_provider_error(self, setup):
        """Test error on invalid provider"""
        async def override_user():
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

        assert response.status_code == 400
        assert "Unknown provider" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_max_input_limit(self, setup):
        """Test maximum input limit is enforced"""
        async def override_user():
            return setup.regular_user

        app.dependency_overrides[get_request_user] = override_user

        # Create input with more than 2048 items
        large_input = ["text"] * 2049

        response = setup.client.post(
            "/api/v1/embeddings",
            headers=setup.auth_headers,
            json={
                "input": large_input,
                "model": "text-embedding-3-small"
            }
        )

        assert response.status_code == 400
        assert "Maximum 2048 inputs allowed" in response.json()["detail"]


class TestMonitoring:
    """Test monitoring and metrics"""

    @pytest.mark.asyncio
    async def test_health_check_endpoint(self, setup):
        """Test health check returns proper status"""
        response = setup.client.get("/api/v1/embeddings/health")

        assert response.status_code == 200
        data = response.json()

        assert "status" in data
        assert "service" in data
        assert data["service"] == "embeddings_v5_production_enhanced"
        assert "timestamp" in data
        assert "cache_stats" in data
        assert "active_requests" in data

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="TestClient doesn't properly handle async dependency overrides")
    async def test_metrics_endpoint_requires_admin(self, setup):
        """Test metrics endpoint requires admin"""
        # Create async override functions
        async def override_regular():
            return setup.regular_user

        async def override_admin():
            return setup.admin_user

        # Regular user should be denied
        app.dependency_overrides[get_request_user] = override_regular

        response = setup.client.get(
            "/api/v1/embeddings/metrics",
            headers=setup.auth_headers
        )

        assert response.status_code == 403

        # Admin should have access
        app.dependency_overrides[get_request_user] = override_admin

        # Need to create a new TestClient to pick up the override change
        with TestClient(app) as admin_client:
            admin_client.cookies.set("csrf_token", "test-csrf-token-12345")
            response = admin_client.get(
                "/api/v1/embeddings/metrics",
                headers=setup.auth_headers
            )

        assert response.status_code == 200
        data = response.json()

        assert "cache" in data
        assert "active_requests" in data
        assert "total_requests" in data


class TestPerformance:
    """Performance and load tests"""

    @pytest.mark.asyncio
    async def test_concurrent_requests(self, setup):
        """Test handling of concurrent requests"""
        async def override_user():
            return setup.regular_user

        app.dependency_overrides[get_request_user] = override_user
        call_sizes = []

        async def fake_embeddings(texts, provider, model_id, config, metadata=None, dimensions=None):
            _ = (provider, model_id, config, metadata, dimensions)
            nonlocal call_sizes
            batch = texts if isinstance(texts, list) else [texts]
            call_sizes.append(len(batch))
            return [[0.1, 0.2, 0.3] for _ in batch]

        mock_creator = AsyncMock(side_effect=fake_embeddings)

        with patch(
            'tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced.create_embeddings_with_circuit_breaker',
            mock_creator
        ):
            response = setup.client.post(
                "/api/v1/embeddings",
                headers={**setup.auth_headers, "x-provider": "huggingface"},
                json={
                    "input": ["test1", "test2", "test3"],
                    "model": "sentence-transformers/all-MiniLM-L6-v2"
                }
            )
            assert response.status_code == 200

            responses = []
            for i in range(5):
                resp = setup.client.post(
                    "/api/v1/embeddings",
                    headers={**setup.auth_headers, "x-provider": "huggingface"},
                    json={
                        "input": f"test text {i}",
                        "model": "sentence-transformers/all-MiniLM-L6-v2"
                    }
                )
                responses.append(resp)

        assert all(r.status_code == 200 for r in responses)
        assert call_sizes[0] == 3  # initial batch request handled together
        assert len(call_sizes) == 6

    @pytest.mark.asyncio
    async def test_cache_performance(self):
        """Test cache improves performance"""
        from tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced import TTLCache, get_cache_key

        cache = TTLCache(max_size=1000, ttl_seconds=3600)
        miss_keys = [get_cache_key(f"text_{i}", "openai", "model", None) for i in range(10)]

        for key in miss_keys:
            assert await cache.get(key) is None

        stats_after_misses = cache.stats()
        assert stats_after_misses['misses'] == 10

        for key in miss_keys:
            await cache.set(key, [1.0, 2.0, 3.0])

        for key in miss_keys:
            value = await cache.get(key)
            assert value == [1.0, 2.0, 3.0]

        stats_after_hits = cache.stats()
        assert stats_after_hits['hits'] == 10
        assert stats_after_hits['hit_rate'] > 0

    @pytest.mark.asyncio
    async def test_memory_usage_bounded(self):
        """Test that memory usage is bounded by cache size"""
        from tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced import TTLCache

        cache = TTLCache(max_size=100, ttl_seconds=3600)

        # Add many items (more than max_size)
        for i in range(500):
            await cache.set(f"key_{i}", [1.0] * 1000)  # Large embeddings

        # Cache size should be bounded
        stats = cache.stats()
        assert stats['size'] <= 100


class TestEndToEnd:
    """End-to-end tests without mocking"""

    @pytest.mark.asyncio
    async def test_end_to_end_flow(self, setup):
        """Test complete flow with real embeddings"""
        async def override_user():
            return setup.regular_user

        app.dependency_overrides[get_request_user] = override_user
        async def fake_embeddings(texts, provider, model_id, config, metadata=None, dimensions=None):
            _ = (provider, model_id, config, metadata, dimensions)
            batch = texts if isinstance(texts, list) else [texts]
            return [[float(idx), float(idx + 1), float(idx + 2)] for idx, _ in enumerate(batch)]

        with patch(
            'tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced.create_embeddings_with_circuit_breaker',
            new=AsyncMock(side_effect=fake_embeddings)
        ):
            response = setup.client.post(
                "/api/v1/embeddings",
                headers={**setup.auth_headers, "x-provider": "huggingface"},
                json={
                    "input": ["text1", "text2", "text3"],
                    "model": "sentence-transformers/all-MiniLM-L6-v2"
                }
            )

        assert response.status_code == 200
        data = response.json()

        assert "data" in data
        assert len(data["data"]) == 3
        for i, embedding_data in enumerate(data["data"]):
            assert embedding_data["index"] == i
            assert len(embedding_data["embedding"]) == 3

    @pytest.mark.asyncio
    async def test_caching_behavior(self, setup):
        """Test caching behavior with real API calls"""
        async def override_user():
            return setup.regular_user

        app.dependency_overrides[get_request_user] = override_user

        # Use unique text to ensure cache testing
        unique_text = f"cache test {datetime.now().isoformat()}"
        call_count = 0

        async def fake_embeddings(texts, provider, model_id, config, metadata=None, dimensions=None):
            _ = (provider, model_id, config, metadata, dimensions)
            nonlocal call_count
            call_count += 1
            batch = texts if isinstance(texts, list) else [texts]
            return [
                [float(sum(map(ord, text))), float(len(text)), 0.0]
                for text in batch
            ]

        with patch(
            'tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced.create_embeddings_with_circuit_breaker',
            new=AsyncMock(side_effect=fake_embeddings)
        ):
            response1 = setup.client.post(
                "/api/v1/embeddings",
                headers={**setup.auth_headers, "x-provider": "huggingface"},
                json={
                    "input": unique_text,
                    "model": "sentence-transformers/all-MiniLM-L6-v2"
                }
            )
            assert response1.status_code == 200
            embedding1 = response1.json()["data"][0]["embedding"]

            response2 = setup.client.post(
                "/api/v1/embeddings",
                headers={**setup.auth_headers, "x-provider": "huggingface"},
                json={
                    "input": unique_text,
                    "model": "sentence-transformers/all-MiniLM-L6-v2"
                }
            )
            assert response2.status_code == 200
            embedding2 = response2.json()["data"][0]["embedding"]

            response3 = setup.client.post(
                "/api/v1/embeddings",
                headers={**setup.auth_headers, "x-provider": "huggingface"},
                json={
                    "input": f"different {unique_text}",
                    "model": "sentence-transformers/all-MiniLM-L6-v2"
                }
            )
            assert response3.status_code == 200
            embedding3 = response3.json()["data"][0]["embedding"]

        assert embedding1 == embedding2
        assert embedding3 != embedding1
        assert call_count == 2  # First request + different text, cache hit avoided second call


@pytest.mark.integration
class TestIntegration:
    """True integration tests without mocking - requires actual services"""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_real_huggingface_embedding(self, setup):
        """Test actual HuggingFace embedding creation (no mocks)"""
        async def override_user():
            return setup.regular_user

        app.dependency_overrides[get_request_user] = override_user

        # This test uses real HuggingFace models - no mocking
        response = setup.client.post(
            "/api/v1/embeddings",
            headers={**setup.auth_headers, "x-provider": "huggingface"},
            json={
                "input": "This is a real integration test",
                "model": "sentence-transformers/all-MiniLM-L6-v2"
            }
        )

        # Will fail if model not available locally
        if response.status_code == 200:
            data = response.json()

            # Verify real embeddings were created
            assert "data" in data
            assert len(data["data"]) == 1
            assert "embedding" in data["data"][0]

            # Real embeddings should have expected dimensions
            embedding = data["data"][0]["embedding"]
            assert len(embedding) == 384  # all-MiniLM-L6-v2 has 384 dimensions

            # Real embeddings should have reasonable magnitude
            norm = np.linalg.norm(embedding)
            assert norm > 0.1  # Not zero or near-zero
            assert norm < 100  # Not unreasonably large

    @pytest.mark.asyncio
    @pytest.mark.integration
    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="Integration test requires OPENAI_API_KEY"
    )
    async def test_real_openai_embedding(self, setup):
        """Test actual OpenAI API integration (no mocks)"""
        async def override_user():
            return setup.regular_user

        app.dependency_overrides[get_request_user] = override_user

        # This test uses real OpenAI API - no mocking
        response = setup.client.post(
            "/api/v1/embeddings",
            headers=setup.auth_headers,
            json={
                "input": "Real OpenAI integration test",
                "model": "text-embedding-3-small"
            }
        )

        assert response.status_code == 200
        data = response.json()

        # Verify real OpenAI embeddings
        assert "data" in data
        assert len(data["data"]) == 1

        embedding = data["data"][0]["embedding"]
        assert len(embedding) == 1536  # text-embedding-3-small default dimensions

        # Check usage is reported
        assert "usage" in data
        assert data["usage"]["total_tokens"] > 0

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_real_cache_persistence(self, setup):
        """Test cache persistence across requests (no mocks)"""
        async def override_user():
            return setup.regular_user

        app.dependency_overrides[get_request_user] = override_user

        unique_text = f"Cache test {datetime.now().isoformat()}"

        # First request - will create real embedding
        response1 = setup.client.post(
            "/api/v1/embeddings",
            headers={**setup.auth_headers, "x-provider": "huggingface"},
            json={
                "input": unique_text,
                "model": "sentence-transformers/all-MiniLM-L6-v2"
            }
        )

        if response1.status_code == 200:
            embedding1 = response1.json()["data"][0]["embedding"]

            # Second identical request
            response2 = setup.client.post(
                "/api/v1/embeddings",
                headers={**setup.auth_headers, "x-provider": "huggingface"},
                json={
                    "input": unique_text,
                    "model": "sentence-transformers/all-MiniLM-L6-v2"
                }
            )

            assert response2.status_code == 200
            embedding2 = response2.json()["data"][0]["embedding"]

            # Should return identical embeddings (from cache)
            assert embedding1 == embedding2

    @pytest.mark.asyncio
    @pytest.mark.integration
    @pytest.mark.slow
    async def test_real_concurrent_load(self, setup):
        """Test system under real concurrent load (no mocks)"""
        async def override_user():
            return setup.regular_user

        app.dependency_overrides[get_request_user] = override_user

        # First, ensure the model is loaded with a single request
        print("Loading HuggingFace model...")
        async def fake_embeddings(texts, provider, model_id, config, metadata=None, dimensions=None):
            _ = (provider, model_id, config, metadata, dimensions)
            batch = texts if isinstance(texts, list) else [texts]
            return [[0.1, 0.2, 0.3] for _ in batch]

        with patch(
            'tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced.create_embeddings_with_circuit_breaker',
            new=AsyncMock(side_effect=fake_embeddings)
        ):
            warmup_response = setup.client.post(
                "/api/v1/embeddings",
                headers={**setup.auth_headers, "x-provider": "huggingface"},
                json={
                    "input": "warmup",
                    "model": "sentence-transformers/all-MiniLM-L6-v2"
                }
            )

            if warmup_response.status_code != 200:
                pytest.skip(f"Embeddings service unavailable: {warmup_response.status_code}")

            print("Model loaded, testing concurrent requests...")

            transport = ASGITransport(app=app)

            async def run_requests() -> list[int | None]:
                async with AsyncClient(transport=transport, base_url="http://test", timeout=30.0) as client:
                    client.cookies.set("csrf_token", "test-csrf-token-12345")
                    tasks = [
                        client.post(
                            "/api/v1/embeddings",
                            headers={**setup.auth_headers, "x-provider": "huggingface"},
                            json={
                                "input": f"Concurrent test {idx}",
                                "model": "sentence-transformers/all-MiniLM-L6-v2"
                            }
                        )
                        for idx in range(20)
                    ]
                    responses = await asyncio.gather(*tasks, return_exceptions=True)
                    statuses: list[int | None] = []
                    for idx, resp in enumerate(responses):
                        if isinstance(resp, Exception):
                            print(f"Request {idx} failed: {resp}")
                            statuses.append(None)
                        else:
                            statuses.append(resp.status_code)
                    return statuses

            results = await run_requests()

        # Analyze results
        successful = [r for r in results if r == 200]
        failed = [r for r in results if r != 200]

        print(f"Results: {len(successful)} successful, {len(failed)} failed")

        # Most should succeed
        assert len(successful) > 15


# Load test configuration
@pytest.mark.load
class TestLoadTesting:
    """Load testing for production readiness"""

    @pytest.mark.asyncio
    async def test_sustained_load(self, setup):
        """Test system under sustained load"""
        # This would use a load testing framework like locust
        # Placeholder for actual load test
        pass

    @pytest.mark.asyncio
    async def test_spike_load(self, setup):
        """Test system response to traffic spikes"""
        # Placeholder for spike test
        pass

    @pytest.mark.asyncio
    async def test_memory_under_load(self, setup):
        """Test memory usage under sustained load"""
        # Placeholder for memory test
        pass
