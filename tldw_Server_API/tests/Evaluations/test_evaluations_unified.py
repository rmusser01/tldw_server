# test_evaluations_unified.py - Comprehensive test suite for unified evaluations API
"""
Test suite for the unified evaluations API.

Tests include:
- OpenAI-compatible CRUD operations
- tldw-specific evaluation endpoints (G-Eval, RAG, Response Quality)
- Unified service functionality
- Webhooks and rate limiting
- Health checks and metrics
"""

import pytest
import os
from pathlib import Path
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient
from fastapi import FastAPI, status
from httpx import ASGITransport, AsyncClient

# Import centralized test configuration
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from test_config import test_config

# Set up test environment before any app imports
test_config.setup_test_environment()
test_config.reset_settings()

from tldw_Server_API.app.api.v1.endpoints.evaluations.evaluations_unified import router as evaluations_router
from tldw_Server_API.app.core.Evaluations.unified_evaluation_service import (
    UnifiedEvaluationService, get_unified_evaluation_service
)
from tldw_Server_API.app.core.Evaluations.evaluation_manager import EvaluationManager
from tldw_Server_API.app.core.Evaluations.eval_runner import EvaluationRunner
from tldw_Server_API.app.api.v1.schemas.evaluation_schemas_unified import EvaluationListResponse, RunListResponse
from tldw_Server_API.app.api.v1.schemas.pagination import CursorPaginationMeta

# Use configuration from test_config
DEFAULT_API_KEY = test_config.TEST_API_KEY
TEST_SK_KEY = test_config.TEST_SK_KEY


def test_cursor_list_responses_backfill_has_more_from_pagination() -> None:
    """Evaluation cursor list aliases should reflect canonical pagination metadata."""
    pagination = CursorPaginationMeta(limit=10, cursor=None, next_cursor="next", has_more=True)

    evaluations = EvaluationListResponse(data=[], pagination=pagination)
    runs = RunListResponse(data=[], pagination=pagination)

    assert evaluations.has_more is True
    assert runs.has_more is True


@pytest.fixture(scope="function")
def evaluations_test_app():
    app = FastAPI()
    app.include_router(evaluations_router, prefix="/api/v1")
    return app


@pytest.fixture(scope="function")
def client(evaluations_test_app):
    """Create a test client"""
    with TestClient(evaluations_test_app) as c:
        yield c


import pytest_asyncio
@pytest_asyncio.fixture(scope="function")
async def async_client(evaluations_test_app):
    """Create an async test client"""
    async with AsyncClient(transport=ASGITransport(app=evaluations_test_app), base_url="http://test") as ac:
        yield ac


@pytest.fixture(autouse=True)
def use_temp_evaluations_db(temp_db_path, monkeypatch, event_loop, evaluations_test_app):
    """Route all evaluation storage to the per-test temporary database."""

    from tldw_Server_API.app.core.Evaluations import unified_evaluation_service as _svc_module
    from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths as _DP

    def _get_db_path(_self, explicit_path=None, **_ignored):

        if explicit_path:
            try:
                return Path(explicit_path)
            except Exception:
                return Path(temp_db_path)
        return Path(temp_db_path)

    monkeypatch.setattr(EvaluationManager, "_get_db_path", _get_db_path, raising=False)
    monkeypatch.setenv("EVALUATIONS_TEST_DB_PATH", str(temp_db_path))
    monkeypatch.setenv("OPENAI_API_KEY", TEST_SK_KEY)

    service = UnifiedEvaluationService(
        db_path=str(temp_db_path),
        enable_webhooks=False,
        enable_caching=True
    )

    # Ensure per-user service cache points at the test-scoped instance
    try:
        _svc_module._service_instance = service
    except Exception:
        _ = None

    try:
        cache = getattr(_svc_module, "_service_instances_by_user")
        cache.clear()
    except Exception:
        cache = None

    try:
        user_id = _DP.get_single_user_id()
    except Exception:
        user_id = 1

    if cache is not None:
        cache[user_id] = service
    else:
        try:
            from collections import OrderedDict  # type: ignore
            _svc_module._service_instances_by_user = OrderedDict(((user_id, service),))
        except Exception:
            _svc_module._service_instances_by_user = {user_id: service}  # type: ignore[assignment]

    event_loop.run_until_complete(service.initialize())

    async def _dependency_override():
        return service

    evaluations_test_app.dependency_overrides[get_unified_evaluation_service] = _dependency_override

    try:
        yield
    finally:
        try:
            cache = getattr(_svc_module, "_service_instances_by_user")
            cache.pop(user_id, None)
        except Exception:
            _ = None
        try:
            _svc_module._service_instance = None
        except Exception:
            _ = None
        event_loop.run_until_complete(service.shutdown())
        evaluations_test_app.dependency_overrides.pop(get_unified_evaluation_service, None)


@pytest.fixture
def auth_headers():
    """Get authentication headers with default API key"""
    return test_config.get_auth_headers()


@pytest.fixture
def sk_auth_headers():
    """Get authentication headers with OpenAI-style sk- key"""
    return test_config.get_sk_auth_headers()


@pytest.fixture
def sample_evaluation_request():
    """Create a sample evaluation request"""
    return {
        "name": "test_evaluation",
        "description": "Test evaluation for unit tests",
        "eval_type": "model_graded",
        "eval_spec": {
            "metrics": ["accuracy", "relevance"],
            "thresholds": {"pass": 0.7, "excellent": 0.9},
            "model": "gpt-3.5-turbo",
            "temperature": 0.3
        },
        "dataset": [
            {
                "input": {"text": "Test input 1"},
                "expected": {"score": 0.8}
            },
            {
                "input": {"text": "Test input 2"},
                "expected": {"score": 0.9}
            }
        ],
        "metadata": {
            "project": "test",
            "tags": ["unit", "test"],
            "version": "1.0.0"
        }
    }


@pytest.fixture
def sample_run_request():
    """Create a sample run request"""
    return {
        "target_model": "gpt-3.5-turbo",
        "config": {
            "temperature": 0.7,
            "max_workers": 4,
            "timeout_seconds": 300
        }
    }


@pytest.fixture
def sample_dataset_request():
    """Create a sample dataset request"""
    return {
        "name": "test_dataset",
        "description": "Test dataset for unit tests",
        "samples": [
            {
                "input": {"question": "What is AI?"},
                "expected": {"answer": "Artificial Intelligence"}
            },
            {
                "input": {"question": "What is ML?"},
                "expected": {"answer": "Machine Learning"}
            }
        ]
    }


@pytest.fixture
def sample_geval_request():
    """Create a sample G-Eval request"""
    return {
        "source_text": "Artificial intelligence is a branch of computer science that aims to create intelligent machines. It has become an essential part of the technology industry.",
        "summary": "AI is a field of computer science focused on creating intelligent machines.",
        "metrics": ["fluency", "consistency", "relevance", "coherence"],
        "api_name": "openai",
        "save_results": False
    }


@pytest.fixture
def sample_rag_request():
    """Create a sample RAG evaluation request"""
    return {
        "query": "What is machine learning?",
        "retrieved_contexts": [
            "Machine learning is a subset of artificial intelligence.",
            "ML algorithms learn patterns from data."
        ],
        "generated_response": "Machine learning is a branch of AI that enables computers to learn from data.",
        "ground_truth": "Machine learning is a subset of AI that uses algorithms to learn from data.",
        "metrics": ["relevance", "faithfulness", "answer_similarity"],
        "api_name": "openai"
    }


class TestUnifiedEvaluationCRUD:
    """Test CRUD operations for unified evaluations API"""

    def test_create_evaluation(self, client, auth_headers, sample_evaluation_request):

        """Test creating a new evaluation"""
        response = client.post(
            "/api/v1/evaluations",
            json=sample_evaluation_request,
            headers=auth_headers
        )
        assert response.status_code == 201
        data = response.json()
        assert data["name"] == sample_evaluation_request["name"]
        assert data["eval_type"] == sample_evaluation_request["eval_type"]
        assert data["id"].startswith("eval_")
        assert "created_at" in data

    def test_get_evaluation(self, client, auth_headers, sample_evaluation_request):

        """Test getting an evaluation by ID"""
        # First create an evaluation
        create_response = client.post(
            "/api/v1/evaluations",
            json=sample_evaluation_request,
            headers=auth_headers
        )
        eval_id = create_response.json()["id"]

        # Then retrieve it
        response = client.get(f"/api/v1/evaluations/{eval_id}", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == eval_id
        assert data["name"] == sample_evaluation_request["name"]

    def test_update_evaluation(self, client, auth_headers, sample_evaluation_request):

        """Test updating an evaluation"""
        # Create evaluation
        create_response = client.post(
            "/api/v1/evaluations",
            json=sample_evaluation_request,
            headers=auth_headers
        )
        eval_id = create_response.json()["id"]

        # Update it
        update_data = {
            "description": "Updated description",
            "metadata": {"custom": {"project": "updated", "updated": True}}
        }
        response = client.patch(
            f"/api/v1/evaluations/{eval_id}",
            json=update_data,
            headers=auth_headers
        )
        assert response.status_code == 200
        data = response.json()
        assert data["description"] == "Updated description"
        assert data["metadata"]["custom"]["updated"] is True

    def test_delete_evaluation(self, client, auth_headers, sample_evaluation_request):

        """Test deleting an evaluation"""
        # Create evaluation
        create_response = client.post(
            "/api/v1/evaluations",
            json=sample_evaluation_request,
            headers=auth_headers
        )
        eval_id = create_response.json()["id"]

        # Delete it
        response = client.delete(f"/api/v1/evaluations/{eval_id}", headers=auth_headers)
        assert response.status_code == 204

        # Verify it's deleted (soft delete, so might still return but with deleted flag)
        get_response = client.get(f"/api/v1/evaluations/{eval_id}", headers=auth_headers)
        # Should return 404 or have deleted flag
        assert get_response.status_code in [404, 200]

    def test_list_evaluations(self, client, auth_headers, sample_evaluation_request):

        """Test listing evaluations with pagination"""
        # Create multiple evaluations
        for i in range(3):
            request = sample_evaluation_request.copy()
            request["name"] = f"test_eval_{i}"
            client.post("/api/v1/evaluations", json=request, headers=auth_headers)

        # List evaluations
        response = client.get("/api/v1/evaluations?limit=2", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        assert len(data["data"]) <= 2
        assert "has_more" in data


class TestTldwSpecificEndpoints:
    """Test tldw-specific evaluation endpoints"""

    def test_geval_missing_provider_credentials_returns_503(
        self,
        client,
        auth_headers,
        sample_geval_request,
        monkeypatch,
    ):

        """Missing provider credentials should return 503 with error code."""
        from tldw_Server_API.app.api.v1.endpoints.evaluations import evaluations_unified as eval_ep
        from tldw_Server_API.app.core.AuthNZ.byok_runtime import ResolvedByokCredentials

        async def _missing(provider, *args, **kwargs):
            return ResolvedByokCredentials(
                provider=provider,
                api_key=None,
                app_config=None,
                credential_fields={},
                source="server",
                allowlisted=True,
            )

        monkeypatch.setattr(eval_ep, "_is_eval_test_mode", lambda: False)
        monkeypatch.setattr(eval_ep, "resolve_byok_credentials", _missing)

        response = client.post(
            "/api/v1/evaluations/geval",
            json=sample_geval_request,
            headers=auth_headers,
        )

        assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
        detail = response.json().get("detail", {})
        assert detail.get("error_code") == "missing_provider_credentials"

    def test_geval_endpoint(self, client, auth_headers, sample_geval_request):

        """Test G-Eval summarization endpoint"""
        # Mock multiple potential service paths
        with patch('tldw_Server_API.app.core.Evaluations.ms_g_eval.run_geval') as mock_run_geval:
            # Mock the actual run_geval function that the endpoint uses
            mock_run_geval.return_value = {
                "metrics": {
                    "fluency": {
                        "name": "fluency",
                        "score": 0.85,
                        "raw_score": 2.55,
                        "explanation": "Text flows naturally"
                    },
                    "consistency": {
                        "name": "consistency",
                        "score": 0.90,
                        "raw_score": 4.5,
                        "explanation": "Information is consistent"
                    }
                },
                "average_score": 0.875,
                "summary_assessment": "High quality summary",
                "evaluation_time": 1.5,
                "metadata": {
                    "evaluation_id": "eval_123"
                }
            }

            response = client.post(
                "/api/v1/evaluations/geval",
                json=sample_geval_request,
                headers=auth_headers
            )
            if response.status_code != 200:
                print(f"Response status: {response.status_code}")
                print(f"Response text: {response.text}")
            assert response.status_code == 200
            data = response.json()
            assert "metrics" in data
            # G-Eval response uses 'average_score' or it might be in the response
            assert "average_score" in data or "overall_score" in data or "summary_assessment" in data
            assert "evaluation_time" in data or "metadata" in data

    async def test_rag_endpoint(self, client, auth_headers, sample_rag_request):
        """Test RAG evaluation endpoint"""
        with patch(
            'tldw_Server_API.app.core.Evaluations.unified_evaluation_service.UnifiedEvaluationService.evaluate_rag',
            new_callable=AsyncMock,
        ) as mock_evaluate:
            # Mock the RAG evaluation service method
            mock_evaluate.return_value = {
                "evaluation_id": "rag_123",
                "results": {
                    "metrics": {
                        "relevance": {
                            "name": "relevance",
                            "score": 0.9,
                            "explanation": "Highly relevant"
                        },
                        "faithfulness": {
                            "name": "faithfulness",
                            "score": 0.85,
                            "explanation": "Well grounded"
                        }
                    },
                    "overall_score": 0.875,
                    "retrieval_quality": 0.9,
                    "generation_quality": 0.85,
                    "suggestions": ["Consider adding more context"]
                },
                "evaluation_time": 2.1
            }

            response = client.post(
                "/api/v1/evaluations/rag",
                json=sample_rag_request,
                headers=auth_headers
            )
            assert response.status_code == 200
            data = response.json()
            assert "metrics" in data
            assert "overall_score" in data
            assert "retrieval_quality" in data
            assert data["metrics"]["relevance"]["score"] == pytest.approx(0.9)
            assert data["metrics"]["faithfulness"]["score"] == pytest.approx(0.85)

    async def test_response_quality_endpoint(self, client, auth_headers):
        """Test response quality evaluation endpoint"""
        with patch(
            'tldw_Server_API.app.core.Evaluations.unified_evaluation_service.UnifiedEvaluationService.evaluate_response_quality',
            new_callable=AsyncMock,
        ) as mock_evaluate:
            # Mock the response quality evaluation service method
            mock_evaluate.return_value = {
                "evaluation_id": "quality_123",
                "results": {
                    "metrics": {
                        "relevance": {
                            "name": "relevance",
                            "score": 0.95,
                            "explanation": "Highly relevant response"
                        },
                        "completeness": {
                            "name": "completeness",
                            "score": 0.88,
                            "explanation": "Mostly complete answer"
                        }
                    },
                    "overall_quality": 0.915,
                    "format_compliance": {"format_ok": True},
                    "issues": [],
                    "improvements": ["Could be more concise"]
                },
                "evaluation_time": 1.3
            }

            request_data = {
                "prompt": "Explain quantum computing",
                "response": "Quantum computing uses quantum mechanics principles...",
                "expected_format": "explanation",
                "api_name": "openai"
            }

            response = client.post(
                "/api/v1/evaluations/response-quality",
                json=request_data,
                headers=auth_headers
            )
            assert response.status_code == 200
            data = response.json()
            assert "overall_quality" in data
            assert "format_compliance" in data


class TestRunManagement:
    """Test evaluation run management"""

    def test_create_run(self, client, auth_headers, sample_evaluation_request, sample_run_request):

        """Test creating an evaluation run"""
        # First create an evaluation
        eval_response = client.post(
            "/api/v1/evaluations",
            json=sample_evaluation_request,
            headers=auth_headers
        )
        eval_id = eval_response.json()["id"]

        # Create a run
        with patch('tldw_Server_API.app.core.Evaluations.eval_runner.EvaluationRunner.run_evaluation'):
            response = client.post(
                f"/api/v1/evaluations/{eval_id}/runs",
                json=sample_run_request,
                headers=auth_headers
            )
            assert response.status_code == 202  # Accepted for async processing
            data = response.json()
            assert data["eval_id"] == eval_id
            assert data["status"] in ["pending", "running"]
            assert data["target_model"] == sample_run_request["target_model"]

    def test_get_run_status(self, client, auth_headers, sample_evaluation_request, sample_run_request):

        """Test getting run status"""
        # Create evaluation and run
        eval_response = client.post(
            "/api/v1/evaluations",
            json=sample_evaluation_request,
            headers=auth_headers
        )
        eval_id = eval_response.json()["id"]

        with patch('tldw_Server_API.app.core.Evaluations.eval_runner.EvaluationRunner.run_evaluation'):
            run_response = client.post(
                f"/api/v1/evaluations/{eval_id}/runs",
                json=sample_run_request,
                headers=auth_headers
            )
            run_id = run_response.json()["id"]

        # Get run status
        response = client.get(f"/api/v1/evaluations/runs/{run_id}", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == run_id
        assert "status" in data

    def test_cancel_run(self, client, auth_headers, sample_evaluation_request, sample_run_request):

        """Test cancelling a run"""
        # Create evaluation and run
        eval_response = client.post(
            "/api/v1/evaluations",
            json=sample_evaluation_request,
            headers=auth_headers
        )
        eval_id = eval_response.json()["id"]

        with patch('tldw_Server_API.app.core.Evaluations.eval_runner.EvaluationRunner.run_evaluation'):
            run_response = client.post(
                f"/api/v1/evaluations/{eval_id}/runs",
                json=sample_run_request,
                headers=auth_headers
            )
            run_id = run_response.json()["id"]

        # Cancel the run
        with patch('tldw_Server_API.app.core.Evaluations.eval_runner.EvaluationRunner.cancel_run', return_value=True):
            response = client.post(f"/api/v1/evaluations/runs/{run_id}/cancel", headers=auth_headers)
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "cancelled"


class TestDatasetManagement:
    """Test dataset management endpoints"""

    def test_create_dataset(self, client, auth_headers, sample_dataset_request):

        """Test creating a dataset"""
        response = client.post(
            "/api/v1/evaluations/datasets",
            json=sample_dataset_request,
            headers=auth_headers
        )
        assert response.status_code == 201
        data = response.json()
        assert data["name"] == sample_dataset_request["name"]
        assert data["sample_count"] == len(sample_dataset_request["samples"])

    def test_get_dataset(self, client, auth_headers, sample_dataset_request):

        """Test getting a dataset"""
        # Create dataset
        create_response = client.post(
            "/api/v1/evaluations/datasets",
            json=sample_dataset_request,
            headers=auth_headers
        )
        dataset_id = create_response.json()["id"]

        # Get dataset
        response = client.get(f"/api/v1/evaluations/datasets/{dataset_id}", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == dataset_id
        assert data["name"] == sample_dataset_request["name"]

    def test_get_dataset_supports_sample_pagination(
        self,
        client,
        auth_headers,
        sample_dataset_request,
    ):
        """Test getting a paginated slice of dataset samples."""
        create_response = client.post(
            "/api/v1/evaluations/datasets",
            json=sample_dataset_request,
            headers=auth_headers
        )
        dataset_id = create_response.json()["id"]

        response = client.get(
            f"/api/v1/evaluations/datasets/{dataset_id}?include_samples=true&limit=1&offset=1",
            headers=auth_headers,
        )

        assert response.status_code == 200
        data = response.json()
        assert data["sample_count"] == len(sample_dataset_request["samples"])
        assert len(data["samples"]) == 1
        assert data["samples"][0]["input"]["question"] == "What is ML?"
        assert data["pagination"] == {
            "mode": "offset",
            "limit": 1,
            "offset": 1,
            "total": len(sample_dataset_request["samples"]),
            "has_more": False,
            "next_offset": None,
        }
        assert data["has_more"] is False
        assert data["next_offset"] is None

    def test_list_datasets(self, client, auth_headers, sample_dataset_request):

        """Test listing datasets"""
        # Create multiple datasets
        for i in range(3):
            request = sample_dataset_request.copy()
            request["name"] = f"dataset_{i}"
            client.post("/api/v1/evaluations/datasets", json=request, headers=auth_headers)

        # List datasets
        response = client.get("/api/v1/evaluations/datasets?limit=2", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        assert len(data["data"]) <= 2
        assert data["total"] == 3
        assert data["pagination"] == {
            "mode": "offset",
            "limit": 2,
            "offset": 0,
            "total": 3,
            "has_more": True,
            "next_offset": 2,
        }

    def test_delete_dataset(self, client, auth_headers, sample_dataset_request):

        """Test deleting a dataset"""
        # Create dataset
        create_response = client.post(
            "/api/v1/evaluations/datasets",
            json=sample_dataset_request,
            headers=auth_headers
        )
        dataset_id = create_response.json()["id"]

        # Delete dataset
        response = client.delete(f"/api/v1/evaluations/datasets/{dataset_id}", headers=auth_headers)
        assert response.status_code == 204


class TestWebhooks:
    """Test webhook functionality"""

    def test_register_webhook(self, client, auth_headers):

        """Test webhook registration"""
        with patch('tldw_Server_API.app.core.Evaluations.webhook_manager.webhook_manager.register_webhook') as mock_register:
            mock_register.return_value = {
                "webhook_id": 1,
                "url": "https://example.com/webhook",
                "events": ["evaluation.completed"],
                "secret": "test_secret_that_is_at_least_32_characters_long",
                "created_at": "2024-01-01T00:00:00",
                "status": "active"
            }

            request_data = {
                "url": "https://example.com/webhook",
                "events": ["evaluation.completed", "evaluation.failed"],
                "secret": "test_secret_that_is_at_least_32_characters_long"
            }

            response = client.post(
                "/api/v1/evaluations/webhooks",
                json=request_data,
                headers=auth_headers
            )
            assert response.status_code == 200
            data = response.json()
            assert data["webhook_id"] == 1
            assert data["url"] == request_data["url"]

    def test_list_webhooks(self, client, auth_headers):

        """Test listing webhooks"""
        with patch('tldw_Server_API.app.core.Evaluations.webhook_manager.webhook_manager.get_webhook_status') as mock_status:
            mock_status.return_value = [
                {
                    "webhook_id": 1,
                    "url": "https://example.com/webhook",
                    "events": ["evaluation.completed"],
                    "status": "active",
                    "created_at": "2024-01-01T00:00:00",
                    "failure_count": 0
                }
            ]

            response = client.get("/api/v1/evaluations/webhooks", headers=auth_headers)
            assert response.status_code == 200
            data = response.json()
            assert len(data) == 1
            assert data[0]["webhook_id"] == 1

    def test_test_webhook(self, client, auth_headers):

        """Test webhook testing endpoint"""
        with patch('tldw_Server_API.app.core.Evaluations.webhook_manager.webhook_manager.test_webhook') as mock_test:
            mock_test.return_value = {
                "success": True,
                "status_code": 200,
                "response_time_ms": 123.45
            }

            request_data = {
                "url": "https://example.com/webhook"
            }

            response = client.post(
                "/api/v1/evaluations/webhooks/test",
                json=request_data,
                headers=auth_headers
            )
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["status_code"] == 200


class TestHealthAndMetrics:
    """Test health check and metrics endpoints"""

    def test_health_check(self, client, auth_headers):

        """Test health check endpoint"""
        response = client.get("/api/v1/evaluations/health", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] in ["healthy", "degraded", "unhealthy"]
        assert "version" in data
        assert "database" in data

    def test_metrics_endpoint(self, client, auth_headers):

        """Test metrics endpoint"""
        response = client.get("/api/v1/evaluations/metrics", headers=auth_headers)
        assert response.status_code == 200
        # Response can be JSON or Prometheus text format
        if response.headers.get("content-type", "").startswith("application/json"):
            data = response.json()
            assert isinstance(data, dict)
        else:
            # Prometheus text format
            assert response.text.startswith("#")


class TestRateLimiting:
    """Test rate limiting functionality"""

    def test_rate_limit_status(self, client, auth_headers):

        """Test getting rate limit status"""
        from datetime import datetime, timezone
        with patch('tldw_Server_API.app.core.Evaluations.user_rate_limiter.user_rate_limiter.get_usage_summary') as mock_summary:
            mock_summary.return_value = {
                "tier": "free",
                "limits": {"requests_per_minute": 10, "tokens_per_minute": 10000},
                "usage": {"requests": 5, "tokens": 5000},
                "remaining": {"requests": 5, "tokens": 5000},
                "reset_at": datetime(2024, 1, 1, 0, 1, 0, tzinfo=timezone.utc)
            }

            response = client.get("/api/v1/evaluations/rate-limits", headers=auth_headers)
            if response.status_code != 200:
                print(f"Response status: {response.status_code}")
                print(f"Response text: {response.text}")
            assert response.status_code == 200
            data = response.json()
            assert data["tier"] == "free"
            assert "limits" in data
            assert "usage" in data

    def test_rate_limit_exceeded(self, client, auth_headers, sample_geval_request):

        """Test rate limit exceeded response"""
        # Mock evaluations user limiter to always reject
        mock_limiter = AsyncMock()
        mock_limiter.check_rate_limit = AsyncMock(
            return_value=(False, {"retry_after": 60, "error": "Rate limit exceeded"})
        )
        with patch(
            'tldw_Server_API.app.api.v1.endpoints.evaluations.evaluations_unified.get_user_rate_limiter_for_user',
            return_value=mock_limiter,
        ):

            response = client.post(
                "/api/v1/evaluations/geval",
                json=sample_geval_request,
                headers=auth_headers
            )
            assert response.status_code == 429
            assert "Retry-After" in response.headers


class TestUnifiedService:
    """Test the UnifiedEvaluationService directly"""

    @pytest.mark.asyncio
    async def test_service_initialization(self):
        """Test unified service initialization"""
        service = get_unified_evaluation_service()
        assert service is not None
        assert hasattr(service, 'db')
        assert hasattr(service, 'runner')
        assert hasattr(service, 'circuit_breaker')
        assert hasattr(service, 'audit_logger')

    @pytest.mark.asyncio
    async def test_service_health_check(self):
        """Test service health check"""
        service = get_unified_evaluation_service()
        health = await service.health_check()
        assert "status" in health
        assert "database" in health
        assert "circuit_breaker" in health
        assert "version" in health

    @pytest.mark.asyncio
    async def test_service_evaluation_creation(self):
        """Test creating evaluation via service"""
        service = get_unified_evaluation_service()

        with patch.object(service.db, 'create_evaluation', return_value='eval_123'):
            with patch.object(service.db, 'get_evaluation', return_value={
                "id": "eval_123",
                "name": "test",
                "eval_type": "model_graded",
                "created_at": 1234567890
            }):
                evaluation = await service.create_evaluation(
                    name="test",
                    eval_type="model_graded",
                    eval_spec={"metrics": ["accuracy"]},
                    created_by="test_user"
                )

                assert evaluation["id"] == "eval_123"
                assert evaluation["name"] == "test"

    @pytest.mark.asyncio
    async def test_history_and_count_use_created_by(self, unified_service):
        """Ensure history and count respect creator field."""
        eval_spec = {"metrics": ["exact_match"], "threshold": 0.5}
        await unified_service.create_evaluation(
            name="eval_user_a_1",
            eval_type="exact_match",
            eval_spec=eval_spec,
            created_by="user_a"
        )
        await unified_service.create_evaluation(
            name="eval_user_a_2",
            eval_type="exact_match",
            eval_spec=eval_spec,
            created_by="user_a"
        )
        await unified_service.create_evaluation(
            name="eval_user_b_1",
            eval_type="exact_match",
            eval_spec=eval_spec,
            created_by="user_b"
        )

        history_a = await unified_service.get_evaluation_history(user_id="user_a", limit=10)
        count_a = await unified_service.count_evaluations(user_id="user_a")

        assert len(history_a) == 2
        assert count_a == 2
        assert all(item.get("created_by") == "user_a" for item in history_a)


class TestErrorHandling:
    """Test error handling"""

    def test_missing_auth(self, client):

        """Test request without authentication"""
        response = client.get("/api/v1/evaluations")
        assert response.status_code == 401
        detail = response.json().get("detail")
        assert isinstance(detail, dict)
        error = detail.get("error")
        assert isinstance(error, dict)
        assert "message" in error

    def test_invalid_evaluation_type(self, client, auth_headers):

        """Test creating evaluation with invalid type"""
        request_data = {
            "name": "test",
            "eval_type": "invalid_type",
            "eval_spec": {}
        }
        response = client.post(
            "/api/v1/evaluations",
            json=request_data,
            headers=auth_headers
        )
        assert response.status_code == 422  # Validation error

    def test_not_found(self, client, auth_headers):

        """Test getting non-existent evaluation"""
        response = client.get(
            "/api/v1/evaluations/eval_nonexistent",
            headers=auth_headers
        )
        assert response.status_code == 404
        data = response.json()
        assert "error" in data["detail"]


class TestAuthentication:
    """Test authentication modes"""

    def test_bearer_token_auth(self, client, auth_headers):

        """Test authentication with Bearer token"""
        response = client.get("/api/v1/evaluations", headers=auth_headers)
        assert response.status_code == 200

    def test_x_api_key_auth(self, client):

        """Test authentication with X-API-KEY header"""
        headers = {"X-API-KEY": DEFAULT_API_KEY}
        response = client.get("/api/v1/evaluations", headers=headers)
        assert response.status_code == 200

    @patch.dict(os.environ, {"SINGLE_USER_API_KEY": TEST_SK_KEY})
    def test_openai_style_auth(self, client, sk_auth_headers):
        """Test authentication with OpenAI-style sk- key"""
        response = client.get("/api/v1/evaluations", headers=sk_auth_headers)
        # Should work if environment is set up correctly
        assert response.status_code in [200, 401]  # Depends on env setup


class TestEvaluationRunnerBatching:
    """Direct tests for EvaluationRunner behavior."""

    @pytest.mark.asyncio
    async def test_sample_ids_unique_across_batches(self, temp_db_path):
        runner = EvaluationRunner(str(temp_db_path), max_concurrent_evals=2, eval_timeout=5)

        samples = [
            {"id": "s1", "input": {"output": "alpha"}, "expected": {"output": "alpha"}},
            {"id": "s2", "input": {"output": "beta"}, "expected": {"output": "beta"}},
            {"id": "s3", "input": {"output": "gamma"}, "expected": {"output": "gamma"}},
        ]

        dataset_id = runner.db.create_dataset(
            name="ds",
            samples=samples,
            created_by="tester"
        )
        eval_spec = {"metrics": ["exact_match"], "threshold": 0.0}
        eval_id = runner.db.create_evaluation(
            name="exact",
            eval_type="exact_match",
            eval_spec=eval_spec,
            dataset_id=dataset_id,
            created_by="tester",
        )
        run_id = runner.db.create_run(
            eval_id=eval_id,
            target_model="unit",
            config={"batch_size": 2},
        )
        eval_config = {
            "eval_type": "exact_match",
            "eval_spec": eval_spec,
            "dataset_id": dataset_id,
            "config": {"batch_size": 2},
        }

        results = await runner.run_evaluation(run_id, eval_id, eval_config, background=False)
        sample_results = results["sample_results"]
        sample_ids = [r["sample_id"] for r in sample_results]

        assert len(sample_ids) == len(sample_results)
        assert len(set(sample_ids)) == len(sample_ids)
        # Ensure original dataset IDs are reflected
        assert any("s1" in sid for sid in sample_ids)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
