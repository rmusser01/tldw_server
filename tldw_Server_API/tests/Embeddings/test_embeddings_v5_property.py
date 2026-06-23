# test_embeddings_v5_property.py
# Property-based tests for production embeddings service

import os
import uuid
import pytest
from hypothesis import given, strategies as st, settings, assume, HealthCheck
from hypothesis.stateful import RuleBasedStateMachine, rule, invariant, Bundle
import numpy as np

from fastapi.testclient import TestClient
from tldw_Server_API.app.main import app
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from unittest.mock import AsyncMock, patch

IN_CI = os.getenv("CI", "").lower() == "true"
PROPERTY_MAX_EXAMPLES = 10 if IN_CI else 50
STATE_MACHINE_MAX_EXAMPLES = 10 if IN_CI else 50
STATE_MACHINE_STEP_COUNT = 8 if IN_CI else 20


async def fake_embeddings(texts, provider=None, model_id=None, config=None, metadata=None, dimensions=None, **kwargs):
    _ = (provider, model_id, config, metadata, dimensions, kwargs)
    batch = texts if isinstance(texts, list) else [texts]
    return [[float(len(text)), 0.0, 0.0] for text in batch]


# Disable rate limiting for all tests
@pytest.fixture(autouse=True)
def disable_rate_limiting():
    """Disable rate limiting for all tests in this module"""
    previous_testing = os.environ.get("TESTING")
    previous_auto_download = os.environ.get("AUTO_DOWNLOAD_MODELS")
    os.environ["TESTING"] = "true"
    os.environ["AUTO_DOWNLOAD_MODELS"] = "false"
    try:
        yield
    finally:
        if previous_testing is None:
            os.environ.pop("TESTING", None)
        else:
            os.environ["TESTING"] = previous_testing
        if previous_auto_download is None:
            os.environ.pop("AUTO_DOWNLOAD_MODELS", None)
        else:
            os.environ["AUTO_DOWNLOAD_MODELS"] = previous_auto_download


@pytest.fixture(autouse=True)
def mock_embedding_creation():
    """Keep property tests on fake embeddings and away from network/model loading."""
    with patch(
        "tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced.create_embeddings_with_circuit_breaker",
        new=AsyncMock(side_effect=fake_embeddings),
    ), patch(
        "tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced.create_embeddings_batch_async",
        new=AsyncMock(side_effect=fake_embeddings),
    ), patch(
        "tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced.batching_create_embeddings_batch_async",
        new=AsyncMock(side_effect=fake_embeddings),
    ), patch(
        "tldw_Server_API.app.core.Embeddings.Embeddings_Server.Embeddings_Create.create_embeddings_batch",
        side_effect=lambda texts, *args, **kwargs: [[float(len(text)), 0.0, 0.0] for text in texts],
    ):
        yield


# Use the shared fixtures from conftest.py
@pytest.fixture
def setup(test_client, regular_user, auth_headers):
    """Setup fixture for property tests"""
    class SetupData:
        def __init__(self):
            self.client = test_client
            self.auth_headers = auth_headers
            self.test_user = regular_user

    async def override_user():
        return regular_user

    app.dependency_overrides[get_request_user] = override_user

    yield SetupData()
    app.dependency_overrides.clear()


@pytest.mark.property
class TestEmbeddingsProperties:
    """Property-based tests for embeddings service"""
    pass


class TestCacheProperties:
    """Property tests for cache implementation"""

    @pytest.mark.property
    @given(
        max_size=st.integers(min_value=1, max_value=100),
        ttl_seconds=st.integers(min_value=1, max_value=10),
        num_items=st.integers(min_value=0, max_value=200)
    )
    @settings(max_examples=PROPERTY_MAX_EXAMPLES, deadline=5000)
    @pytest.mark.asyncio
    async def test_cache_never_exceeds_max_size(self, max_size, ttl_seconds, num_items):
        """Property: Cache size never exceeds max_size"""
        from tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced import TTLCache

        cache = TTLCache(max_size=max_size, ttl_seconds=ttl_seconds)

        # Add items
        for i in range(num_items):
            await cache.set(f"key_{i}", [float(i)])

        # Cache size should never exceed max_size
        stats = cache.stats()
        assert stats['size'] <= max_size

    @pytest.mark.property
    @given(
        keys=st.lists(
            st.text(min_size=1, max_size=10),
            min_size=1,
            max_size=20,
            unique=True
        )
    )
    @settings(max_examples=PROPERTY_MAX_EXAMPLES, deadline=5000)
    @pytest.mark.asyncio
    async def test_cache_get_returns_set_value(self, keys):
        """Property: Cache get returns exactly what was set"""
        from tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced import TTLCache

        cache = TTLCache(max_size=100, ttl_seconds=3600)

        # Store values
        values = {}
        for key in keys:
            value = [float(hash(key) % 100)]
            await cache.set(key, value)
            values[key] = value

        # Retrieve and verify
        for key in keys:
            retrieved = await cache.get(key)
            if retrieved is not None:  # May be evicted if cache full
                assert retrieved == values[key]

    @pytest.mark.property
    @given(
        ttl_seconds=st.floats(min_value=0.1, max_value=1.0)
    )
    @settings(max_examples=PROPERTY_MAX_EXAMPLES, deadline=10000)
    @pytest.mark.asyncio
    async def test_cache_ttl_expiration_property(self, ttl_seconds, monkeypatch):
        """Property: Items expire after TTL"""
        from tldw_Server_API.app.api.v1.endpoints import embeddings_v5_production_enhanced as embeddings_module

        current_time = 1000.0
        monkeypatch.setattr(embeddings_module.time, "time", lambda: current_time)
        cache = embeddings_module.TTLCache(max_size=10, ttl_seconds=ttl_seconds)

        await cache.set("test_key", [1.0, 2.0, 3.0])

        # Should exist immediately
        assert await cache.get("test_key") is not None

        current_time += ttl_seconds + 0.1

        # Should be expired
        assert await cache.get("test_key") is None


class TestInputValidationProperties:
    """Property tests for input validation"""

    @pytest.mark.property
    @given(
        input_text=st.text(min_size=0, max_size=10000)
    )
    @settings(max_examples=10, deadline=10000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_non_empty_input_accepted(self, setup, input_text):
        """Property: Non-empty strings are accepted, empty strings rejected"""
        response = setup.client.post(
            "/api/v1/embeddings",
            headers=setup.auth_headers,
            json={
                "input": input_text,
                "model": "text-embedding-3-small"
            }
        )

        if input_text.strip():
            assert response.status_code in (200, 400)
            if response.status_code == 400:
                detail = response.json().get("detail", "").lower()
                assert "empty" in detail or "long" in detail
        else:
            assert response.status_code == 400
            assert "empty" in response.json()["detail"].lower()

    @pytest.mark.property
    @given(
        num_inputs=st.integers(min_value=0, max_value=3000)
    )
    @settings(max_examples=5, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_input_list_size_limit(self, setup, num_inputs):
        """Property: Input lists > 2048 items are rejected"""
        inputs = [f"text_{i}" for i in range(num_inputs)]

        response = setup.client.post(
            "/api/v1/embeddings",
            headers=setup.auth_headers,
            json={
                "input": inputs,
                "model": "text-embedding-3-small"
            }
        )

        if num_inputs == 0:
            assert response.status_code == 400
            assert "empty" in response.json()["detail"].lower()
        elif num_inputs <= 2048:
            assert response.status_code == 200
        else:
            assert response.status_code == 400
            assert "2048" in response.json()["detail"]

    @pytest.mark.property
    @given(
        provider=st.sampled_from(["openai", "huggingface", "cohere", "invalid_provider", ""])
    )
    @settings(max_examples=5, deadline=10000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_provider_validation(self, setup, provider):
        """Property: Only valid providers are accepted"""
        valid_providers = ["openai", "huggingface", "cohere", "voyage", "google", "mistral", "onnx", "local_api"]

        headers = setup.auth_headers.copy()
        if provider:
            headers["x-provider"] = provider

        response = setup.client.post(
            "/api/v1/embeddings",
            headers=headers,
            json={
                "input": "test text",
                "model": "some-model"
            }
        )

        if not provider or provider in valid_providers:
            # Valid provider or default
            assert response.status_code in (200, 404)
        else:
            assert response.status_code == 400
            assert "unknown provider" in response.json()["detail"].lower()


class TestEmbeddingOutputProperties:
    """Property tests for embedding outputs"""

    @pytest.mark.property
    @given(
        num_texts=st.integers(min_value=1, max_value=10)
    )
    @settings(max_examples=5, deadline=10000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_output_count_matches_input(self, setup, num_texts):
        """Property: Number of embeddings matches number of inputs"""
        from unittest.mock import patch

        texts = [f"text_{i}" for i in range(num_texts)]

        # Mock to avoid actual API calls
        async def mock_embeddings(*args, **kwargs):
            return [[1.0, 2.0, 3.0]] * len(texts)

        # Also patch the underlying create_embeddings_batch to avoid config issues
        with patch('tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced.create_embeddings_batch_async', mock_embeddings), \
             patch('tldw_Server_API.app.core.Embeddings.Embeddings_Server.Embeddings_Create.create_embeddings_batch', mock_embeddings):
            response = setup.client.post(
                "/api/v1/embeddings",
                headers=setup.auth_headers,
                json={
                    "input": texts,
                    "model": "text-embedding-3-small"
                }
            )

            if response.status_code == 200:
                data = response.json()
                assert len(data["data"]) == num_texts

                # Check indices are correct
                for i in range(num_texts):
                    assert data["data"][i]["index"] == i

    @pytest.mark.property
    @given(
        encoding_format=st.sampled_from(["float", "base64", None])
    )
    @settings(max_examples=5, deadline=10000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_encoding_format_property(self, setup, encoding_format):
        """Property: Encoding format is respected"""
        from unittest.mock import patch
        import base64

        async def mock_embeddings(*args, **kwargs):
            return [[1.0, 2.0, 3.0]]

        # Also patch the underlying create_embeddings_batch to avoid config issues
        with patch('tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced.create_embeddings_batch_async', mock_embeddings), \
             patch('tldw_Server_API.app.core.Embeddings.Embeddings_Server.Embeddings_Create.create_embeddings_batch', mock_embeddings):
            request_data = {
                "input": "test text",
                "model": "text-embedding-3-small"
            }

            if encoding_format:
                request_data["encoding_format"] = encoding_format

            response = setup.client.post(
                "/api/v1/embeddings",
                headers=setup.auth_headers,
                json=request_data
            )

            if response.status_code == 200:
                data = response.json()
                embedding = data["data"][0]["embedding"]

                if encoding_format == "base64":
                    # Should be base64 string
                    assert isinstance(embedding, str)
                    # Should be valid base64
                    try:
                        base64.b64decode(embedding)
                    except Exception:
                        pytest.fail("Invalid base64 encoding")
                else:
                    # Should be list of floats
                    assert isinstance(embedding, list)
                    assert all(isinstance(x, (int, float)) for x in embedding)


class EmbeddingStateMachine(RuleBasedStateMachine):
    """Stateful testing for embeddings service"""

    def __init__(self):

        super().__init__()
        # Mock metrics to avoid issues
        from unittest.mock import MagicMock
        import tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced as emb_module
        mock_metrics = MagicMock()
        mock_metrics.labels.return_value.inc = MagicMock()
        mock_metrics.labels.return_value.observe = MagicMock()
        mock_metrics.inc = MagicMock()
        mock_metrics.dec = MagicMock()
        emb_module.embedding_requests_total = mock_metrics
        emb_module.embedding_request_duration = mock_metrics
        emb_module.embedding_cache_hits = mock_metrics
        emb_module.active_embedding_requests = mock_metrics

        self.client = TestClient(app)
        # Set CSRF token
        csrf_token = f"test-csrf-{uuid.uuid4().hex}"
        self.client.cookies.set("csrf_token", csrf_token)
        self.auth_headers = {
            "Authorization": "Bearer test-api-key",
            "X-CSRF-Token": csrf_token
        }
        self.cached_texts = set()
        self.request_count = 0

        # Setup user override
        self.test_user = User(
            id=1,
            username="testuser",
            email="test@example.com",
            is_active=True,
            is_admin=False
        )

        async def override_user():
            return self.test_user

        app.dependency_overrides[get_request_user] = override_user

    def __del__(self):

        try:
            if hasattr(self, 'client') and self.client is not None:
                self.client.close()
        except Exception:
            _ = None
        try:
            app.dependency_overrides.clear()
        except Exception:
            _ = None

    texts = Bundle("texts")

    @rule(target=texts, text=st.text(min_size=1, max_size=100))
    def add_text(self, text):
        """Add a text to be embedded"""
        return text

    @rule(text=texts)
    def embed_text(self, text):
        """Embed a text"""
        from unittest.mock import patch

        async def mock_embeddings(*args, **kwargs):
            # Return consistent embedding for same text
            return [[float(hash(text) % 100), 1.0, 2.0]]

        # Also patch the underlying create_embeddings_batch to avoid config issues
        with patch('tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced.create_embeddings_batch_async', new=AsyncMock(side_effect=mock_embeddings)), \
             patch('tldw_Server_API.app.core.Embeddings.Embeddings_Server.Embeddings_Create.create_embeddings_batch', side_effect=lambda texts, *args, **kwargs: [[1.0, 2.0, 3.0]] * len(texts)):
            response = self.client.post(
                "/api/v1/embeddings",
                headers=self.auth_headers,
                json={
                    "input": text,
                    "model": "text-embedding-3-small"
                }
            )

            self.request_count += 1

            if response.status_code == 200:
                data = response.json()

                # Should return valid response
                assert "data" in data
                assert len(data["data"]) == 1
                assert "embedding" in data["data"][0]

                # If we've embedded this text before, it might be cached
                if text in self.cached_texts:
                    # Could be from cache (faster response expected)
                    pass

                self.cached_texts.add(text)

    @rule()
    def check_health(self):
        """Check health endpoint"""
        response = self.client.get("/api/v1/embeddings/health")
        assert response.status_code in [200, 503]
        data = response.json()
        assert "status" in data
        assert "cache_stats" in data

    @invariant()
    def cache_size_reasonable(self):
        """Cache size should be reasonable"""
        # This is a simple invariant - in production you'd check actual cache
        assert len(self.cached_texts) <= 10000

    @invariant()
    def request_count_reasonable(self):
        """Request count should be reasonable"""
        assert self.request_count <= 1000


# Run the state machine test
TestEmbeddingStateMachine = EmbeddingStateMachine.TestCase
TestEmbeddingStateMachine.settings = settings(
    max_examples=STATE_MACHINE_MAX_EXAMPLES,
    stateful_step_count=STATE_MACHINE_STEP_COUNT,
    suppress_health_check=[HealthCheck.function_scoped_fixture]
)
