# test_embeddings_v5_integration.py
# Integration tests for production embeddings service (no mocking)

import asyncio
import os
import importlib.util
import subprocess  # nosec B404
import sys
import uuid
from datetime import datetime
import pytest
pytestmark = pytest.mark.integration
import numpy as np

IN_CI = os.getenv("CI", "").lower() == "true"
RUN_REAL_HF_EMBEDDING_TESTS = (
    os.getenv("RUN_REAL_HF_EMBEDDING_TESTS", "").lower() in {"1", "true", "yes", "on"}
    or not IN_CI
)

try:
    import redis
except Exception:  # pragma: no cover - optional dependency
    redis = None  # type: ignore

from fastapi.testclient import TestClient
from httpx import AsyncClient, ASGITransport
from tldw_Server_API.app.core.config import settings as app_settings
from tldw_Server_API.app.main import app
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user


def _redis_available() -> bool:


    """Detect whether a Redis instance is reachable for cache tests."""
    if redis is None:
        return False

    try:
        settings_url = str(app_settings.get("REDIS_URL", "")) or None
        redis_enabled = bool(app_settings.get("REDIS_ENABLED", False))
    except Exception:
        settings_url = None
        redis_enabled = True

    if not redis_enabled:
        return False

    env_url = os.getenv("REDIS_URL")
    url = env_url or settings_url or "redis://localhost:6379/0"
    try:
        client = redis.from_url(url)
        try:
            client.ping()
        finally:
            client.close()
        return True
    except Exception:
        return False


REDIS_AVAILABLE = _redis_available()


def _huggingface_deps_available() -> bool:


    """Return True when optional HuggingFace dependencies are present."""
    try:
        required = ("torch", "transformers", "sentence_transformers")
        if not all(importlib.util.find_spec(name) is not None for name in required):
            return False
        # Validate imports in an isolated process so native crashes do not kill pytest collection.
        probe = subprocess.run(
            [sys.executable, "-c", "import torch, transformers, sentence_transformers"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=30,
        )  # nosec B603
        return probe.returncode == 0
    except Exception:
        return False


HF_DEPS_AVAILABLE = RUN_REAL_HF_EMBEDDING_TESTS and _huggingface_deps_available()


def _skip_if_hf_service_unavailable(resp) -> None:
    """Skip strict HF integration assertions when provider is unavailable in this environment."""
    if resp.status_code != 503:
        return
    detail = None
    try:
        payload = resp.json()
        detail = payload.get("detail")
    except Exception:
        detail = resp.text
    pytest.skip(f"HuggingFace embeddings unavailable in this runtime: {detail}")


# Disable rate limiting for all tests
@pytest.fixture(autouse=True)
def disable_rate_limiting():
    """Disable rate limiting for all tests in this module"""
    previous_auto_download = os.environ.get("AUTO_DOWNLOAD_MODELS")
    os.environ["TESTING"] = "true"
    os.environ["AUTO_DOWNLOAD_MODELS"] = "false"
    yield
    # Clean up after tests
    if "TESTING" in os.environ:
        del os.environ["TESTING"]
    if previous_auto_download is None:
        os.environ.pop("AUTO_DOWNLOAD_MODELS", None)
    else:
        os.environ["AUTO_DOWNLOAD_MODELS"] = previous_auto_download

# Module-level setup fixture for integration tests
@pytest.fixture
def setup():
    """Setup test environment fixture with proper TestClient lifecycle"""
    class SetupData:
        pass

    with TestClient(app) as client:
        data = SetupData()
        data.client = client
        # Set CSRF token in both cookie and header
        csrf_token = f"test-csrf-{uuid.uuid4().hex}"
        client.cookies.set("csrf_token", csrf_token)
        data.auth_headers = {
            "Authorization": "Bearer test-api-key",
            "X-CSRF-Token": csrf_token
        }

        data.test_user = User(
            id=1,
            username="testuser",
            email="test@example.com",
            is_active=True,
            is_admin=False
        )

        try:
            yield data
        finally:
            app.dependency_overrides.clear()


@pytest.mark.integration
class TestEmbeddingsIntegration:
    """Integration tests without mocking - requires actual services"""

    @pytest.mark.integration
    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not HF_DEPS_AVAILABLE,
        reason="HuggingFace dependencies unavailable",
    )
    async def test_real_huggingface_embedding(self, setup):
        """Test actual HuggingFace embedding creation (no mocks)"""
        # Ensure authenticated user context
        async def override_user():
            return setup.test_user
        app.dependency_overrides[get_request_user] = override_user

        app.dependency_overrides[get_request_user] = override_user

        # This test uses real HuggingFace models - no mocking
        response = setup.client.post(
            "/api/v1/embeddings",
            headers={**setup.auth_headers, "x-provider": "huggingface"},
            json={
                "input": "This is a real integration test with HuggingFace",
                "model": "sentence-transformers/all-MiniLM-L6-v2"
            }
        )

        _skip_if_hf_service_unavailable(response)
        assert response.status_code == 200

        # Provider headers and no-fallback guarantees for real HF path
        provider_header = (response.headers.get("X-Embeddings-Provider") or "").lower()
        assert provider_header == "huggingface"
        assert response.headers.get("X-Embeddings-Fallback-From") in (None, "")

        # Dimensions policy header is optional; if present it must be valid
        dim_policy = (response.headers.get("X-Embeddings-Dimensions-Policy") or "").lower()
        if dim_policy:
            assert dim_policy in {"reduce", "pad", "ignore"}

        data = response.json()

        # Response envelope and model id
        assert data.get("object") == "list"
        assert data.get("model") == "huggingface:sentence-transformers/all-MiniLM-L6-v2"
        assert "usage" in data and data["usage"]["total_tokens"] > 0 and data["usage"]["prompt_tokens"] > 0

        # Verify real embeddings were created
        assert "data" in data and len(data["data"]) == 1
        item = data["data"][0]
        assert item.get("object") == "embedding"
        assert item.get("index") == 0
        assert "embedding" in item

        # Embedding numeric shape and normalization
        embedding = np.asarray(item["embedding"], dtype=float)
        assert embedding.shape[0] == 384
        norm = np.linalg.norm(embedding)
        # Normalized by API for float format
        assert 0.95 < norm < 1.05
        # Reasonable numeric range
        assert np.all(np.isfinite(embedding))
        assert np.max(np.abs(embedding)) < 10

        # Optional rate limit headers: if present, should be integers
        lim = response.headers.get("X-RateLimit-Limit")
        rem = response.headers.get("X-RateLimit-Remaining")
        if lim is not None and rem is not None:
            assert str(lim).isdigit() and str(rem).isdigit()

    @pytest.mark.integration
    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not HF_DEPS_AVAILABLE,
        reason="HuggingFace dependencies unavailable",
    )
    async def test_huggingface_embedding_base64_format(self, setup):
        """Verify base64 encoding path returns decodable float32 bytes with expected length."""
        async def override_user():
            return setup.test_user
        app.dependency_overrides[get_request_user] = override_user
        resp = setup.client.post(
            "/api/v1/embeddings",
            headers={**setup.auth_headers, "x-provider": "huggingface"},
            json={
                "input": "Base64 encoding check",
                "model": "sentence-transformers/all-MiniLM-L6-v2",
                "encoding_format": "base64"
            }
        )
        _skip_if_hf_service_unavailable(resp)
        assert resp.status_code == 200
        out = resp.json()
        blob = out["data"][0]["embedding"]
        # Decode and parse as float32 array
        import base64
        raw = base64.b64decode(blob)
        arr = np.frombuffer(raw, dtype=np.float32)
        assert arr.shape[0] == 384
        assert np.linalg.norm(arr) > 0

    @pytest.mark.integration
    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not HF_DEPS_AVAILABLE,
        reason="HuggingFace dependencies unavailable",
    )
    async def test_huggingface_embedding_dimension_override_reduce(self, setup):
        """Request a smaller dimension and verify adjust_dimensions(policy=reduce) and header are applied."""
        # Explicitly enforce reduce policy to avoid leakage from other tests
        prev_policy = os.environ.get("EMBEDDINGS_DIMENSION_POLICY")
        os.environ["EMBEDDINGS_DIMENSION_POLICY"] = "reduce"
        try:
            async def override_user():
                return setup.test_user
            app.dependency_overrides[get_request_user] = override_user
            resp = setup.client.post(
                "/api/v1/embeddings",
                headers={**setup.auth_headers, "x-provider": "huggingface"},
                json={
                    "input": "Dimension override reduce policy",
                    "model": "sentence-transformers/all-MiniLM-L6-v2",
                    "dimensions": 128
                }
            )
            _skip_if_hf_service_unavailable(resp)
            assert resp.status_code == 200
            dim_policy = (resp.headers.get("X-Embeddings-Dimensions-Policy") or "").lower()
            if dim_policy:
                assert dim_policy in {"reduce", "pad", "ignore"}
            vec = np.asarray(resp.json()["data"][0]["embedding"], dtype=float)
            assert vec.shape[0] == 128
            # API normalizes float output after adjustment
            assert 0.95 < np.linalg.norm(vec) < 1.05
        finally:
            # Restore prior policy to prevent cross-test interference
            if prev_policy is None:
                os.environ.pop("EMBEDDINGS_DIMENSION_POLICY", None)
            else:
                os.environ["EMBEDDINGS_DIMENSION_POLICY"] = prev_policy

    @pytest.mark.integration
    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="Integration test requires OPENAI_API_KEY environment variable"
    )
    async def test_real_openai_embedding(self, setup):
        """Test actual OpenAI API integration (no mocks)"""
        async def override_user():
            return setup.test_user

        app.dependency_overrides[get_request_user] = override_user

        # This test uses real OpenAI API - no mocking
        response = setup.client.post(
            "/api/v1/embeddings",
            headers=setup.auth_headers,
            json={
                "input": "Real OpenAI integration test with actual API",
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

        # Check usage is reported correctly
        assert "usage" in data
        assert data["usage"]["total_tokens"] > 0
        assert data["usage"]["prompt_tokens"] > 0

        # OpenAI embeddings should be normalized
        norm = np.linalg.norm(embedding)
        assert 0.95 < norm < 1.05

    @pytest.mark.integration
    @pytest.mark.asyncio
    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not HF_DEPS_AVAILABLE,
        reason="HuggingFace dependencies unavailable",
    )
    async def test_real_cache_persistence(self, setup):
        """Test cache persistence across requests (no mocks)"""
        async def override_user():
            return setup.test_user

        app.dependency_overrides[get_request_user] = override_user

        # Use unique text to avoid conflicts with other tests
        unique_text = f"Cache test {datetime.now().isoformat()} {os.getpid()}"

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
            provider_header = (response1.headers.get("X-Embeddings-Provider") or "").lower()
            if provider_header and provider_header != "huggingface":
                pytest.skip(f"Embedding provider fell back to {provider_header}")

            embedding1 = response1.json()["data"][0]["embedding"]
            if len(embedding1) != 384:
                pytest.skip(f"HuggingFace embedding returned unexpected dimension {len(embedding1)} (likely fallback)")

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
            provider_header2 = (response2.headers.get("X-Embeddings-Provider") or "").lower()
            if provider_header2 and provider_header2 != "huggingface":
                pytest.skip(f"Embedding provider fell back to {provider_header2}")
            embedding2 = response2.json()["data"][0]["embedding"]
            if len(embedding2) != 384:
                pytest.skip(f"HuggingFace embedding returned unexpected dimension {len(embedding2)} (likely fallback)")

            # Should return identical embeddings (from cache)
            assert embedding1 == embedding2

            # Verify embeddings are valid
            assert len(embedding1) == 384
            assert len(embedding2) == 384

    @pytest.mark.integration
    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not HF_DEPS_AVAILABLE,
        reason="HuggingFace dependencies unavailable",
    )
    async def test_different_providers_produce_different_embeddings(self, setup):
        """Test that different providers produce different embeddings for same text"""
        async def override_user():
            return setup.test_user

        app.dependency_overrides[get_request_user] = override_user

        test_text = "Compare embeddings across providers"

        # Get HuggingFace embedding
        response_hf = setup.client.post(
            "/api/v1/embeddings",
            headers={**setup.auth_headers, "x-provider": "huggingface"},
            json={
                "input": test_text,
                "model": "sentence-transformers/all-MiniLM-L6-v2"
            }
        )

        if response_hf.status_code == 200 and os.getenv("OPENAI_API_KEY"):
            provider_header = (response_hf.headers.get("X-Embeddings-Provider") or "").lower()
            if provider_header and provider_header != "huggingface":
                pytest.skip(f"Embedding provider fell back to {provider_header}")

            embedding_hf = response_hf.json()["data"][0]["embedding"]

            # Get OpenAI embedding
            response_openai = setup.client.post(
                "/api/v1/embeddings",
                headers=setup.auth_headers,
                json={
                    "input": test_text,
                    "model": "text-embedding-3-small"
                }
            )

            if response_openai.status_code == 200:
                embedding_openai = response_openai.json()["data"][0]["embedding"]

                # Should have different dimensions
                assert len(embedding_hf) != len(embedding_openai)
                assert len(embedding_hf) == 384
                assert len(embedding_openai) == 1536

                # Both should be normalized
                assert 0.95 < np.linalg.norm(embedding_hf) < 1.05
                assert 0.95 < np.linalg.norm(embedding_openai) < 1.05

    @pytest.mark.integration
    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.getenv("RUN_STRESS_TESTS"),
        reason="Stress tests require RUN_STRESS_TESTS=true"
    )
    @pytest.mark.skipif(
        not HF_DEPS_AVAILABLE,
        reason="HuggingFace dependencies unavailable",
    )
    async def test_real_concurrent_load(self, setup):
        """Test system under real concurrent load (no mocks)"""
        async def override_user():
            return setup.test_user

        app.dependency_overrides[get_request_user] = override_user

        async def make_real_request(idx):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test", timeout=30.0) as ac:
                # Set CSRF token in cookie
                ac.cookies.set("csrf_token", "test-csrf-token-12345")
                response = await ac.post(
                    "/api/v1/embeddings",
                    headers={**setup.auth_headers, "x-provider": "huggingface"},
                    json={
                        "input": f"Concurrent test {idx} at {datetime.now()}",
                        "model": "sentence-transformers/all-MiniLM-L6-v2"
                    }
                )
                return response

        # Make real concurrent requests
        tasks = [make_real_request(i) for i in range(20)]
        # Apply an overall timeout to prevent hangs in CI
        results = await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=120.0)

        # Count successful requests
        responses = [r for r in results if hasattr(r, "status_code")]
        successful = sum(1 for r in responses if r.status_code == 200)
        failed = len(results) - successful

        # Most should succeed
        assert successful > 15, f"Only {successful}/20 requests succeeded"

        # Log any failures for debugging
        if failed > 0:
            print(f"Failed requests: {failed}/20")
            for i, r in enumerate(results):
                if not hasattr(r, "status_code") or r.status_code != 200:
                    print(f"Request {i}: {r}")

        for resp in responses:
            provider_header = (resp.headers.get("X-Embeddings-Provider") or "").lower()
            if provider_header and provider_header != "huggingface":
                pytest.skip(f"Embedding provider fell back to {provider_header}")

    @pytest.mark.integration
    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not HF_DEPS_AVAILABLE,
        reason="HuggingFace dependencies unavailable",
    )
    async def test_batch_processing(self, setup):
        """Test batch processing with real embeddings"""
        async def override_user():
            return setup.test_user

        app.dependency_overrides[get_request_user] = override_user

        # Create batch of unique texts
        batch_texts = [f"Batch text {i}" for i in range(10)]

        response = setup.client.post(
            "/api/v1/embeddings",
            headers={**setup.auth_headers, "x-provider": "huggingface"},
            json={
                "input": batch_texts,
                "model": "sentence-transformers/all-MiniLM-L6-v2"
            }
        )

        if response.status_code == 200:
            provider_header = (response.headers.get("X-Embeddings-Provider") or "").lower()
            if provider_header and provider_header != "huggingface":
                pytest.skip(f"Embedding provider fell back to {provider_header}")
            data = response.json()

            # Should return embeddings for all texts
            assert len(data["data"]) == 10

            # Each embedding should be valid
            for i, embedding_data in enumerate(data["data"]):
                assert embedding_data["index"] == i
                embedding = embedding_data["embedding"]
                if len(embedding) != 384:
                    pytest.skip(f"HuggingFace embedding returned unexpected dimension {len(embedding)} (likely fallback)")

                # Should have reasonable magnitude
                norm = np.linalg.norm(embedding)
                assert norm > 0.1  # Not zero or near-zero
                assert norm < 100  # Not unreasonably large

            # Different texts should produce different embeddings
            embeddings = [d["embedding"] for d in data["data"]]
            for i in range(len(embeddings) - 1):
                # Calculate proper cosine similarity
                vec1 = np.array(embeddings[i])
                vec2 = np.array(embeddings[i + 1])
                cos_sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                assert cos_sim < 0.99, f"Different texts produced too similar embeddings (similarity={cos_sim})"

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_health_check_with_real_service(self, setup):
        """Test health check endpoint with real service"""
        response = setup.client.get("/api/v1/embeddings/health")

        assert response.status_code in [200, 503]  # May be degraded if dependencies missing
        data = response.json()

        assert "status" in data
        assert "service" in data
        assert data["service"] == "embeddings_v5_production_enhanced"
        assert "timestamp" in data
        assert "cache_stats" in data
        assert "active_requests" in data

        # Cache stats should be valid
        cache_stats = data["cache_stats"]
        assert "size" in cache_stats
        assert "max_size" in cache_stats
        assert "ttl_seconds" in cache_stats
        assert cache_stats["size"] >= 0
        assert cache_stats["max_size"] > 0
