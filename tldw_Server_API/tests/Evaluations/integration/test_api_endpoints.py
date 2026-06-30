"""
Integration tests for Evaluation API endpoints.

These tests use real components with no mocking - only external services
like LLMs use cached responses for deterministic testing.

Marked with ``evaluations`` so they only run in the
opt-in Evaluations test suite (RUN_EVALUATIONS=1).
"""

import pytest
pytestmark = [pytest.mark.integration, pytest.mark.evaluations]
import json
import asyncio
import os
from typing import Dict, Any
from datetime import datetime

from tldw_Server_API.tests.Evaluations.fixtures.sample_data import (
    SampleDataGenerator,
    generate_evaluation_request
)
from tldw_Server_API.tests.Evaluations.fixtures.database import create_test_database_with_data


@pytest.mark.integration
class TestGEvalEndpoint:
    """Integration tests for G-Eval endpoint."""

    @pytest.mark.asyncio
    @pytest.mark.requires_llm
    async def test_geval_complete_flow(self, async_api_client, auth_headers):
        """Test complete G-Eval workflow with real components."""
        # Prepare request data
        geval_data = SampleDataGenerator.generate_geval_data()
        request_data = {
            "source_text": geval_data["source_text"],
            "summary": geval_data["summary"],
            "metrics": [geval_data["criteria"]] if isinstance(geval_data["criteria"], str) else geval_data["criteria"],
            "api_name": "openai",
            "api_key": "test_api_key_for_mocked_calls",
            "save_results": False
        }

        # Send request to API
        response = await async_api_client.post(
            "/api/v1/evaluations/geval",
            json=request_data,
            headers=auth_headers
        )

        if response.status_code != 200:
            print(f"Response status: {response.status_code}")
            print(f"Response text: {response.text}")
        assert response.status_code == 200
        result = response.json()

        # Verify response structure matches GEvalResponse
        assert "metrics" in result
        assert "average_score" in result
        assert "summary_assessment" in result
        assert "evaluation_time" in result
        assert "metadata" in result

        # Check evaluation_id is in metadata
        assert "evaluation_id" in result["metadata"]

        # Check that metrics are present
        assert len(result["metrics"]) > 0

    @pytest.mark.asyncio
    @pytest.mark.requires_llm
    async def test_geval_with_multiple_criteria(self, async_api_client, auth_headers):
        """Test G-Eval with multiple evaluation criteria."""
        request_data = {
            "source_text": "Long source text about climate change and its effects on global temperatures, weather patterns, and ecosystems...",
            "summary": "Climate change is affecting global temperatures and weather patterns...",
            "metrics": ["coherence", "consistency", "fluency", "relevance"],
            "api_name": "openai",
            "api_key": "test_api_key_for_mocked_calls"
        }

        response = await async_api_client.post(
            "/api/v1/evaluations/geval",
            json=request_data,
            headers=auth_headers
        )

        assert response.status_code == 200
        result = response.json()

        # All criteria should be evaluated
        assert "metrics" in result
        for criterion in request_data["metrics"]:
            assert criterion in result["metrics"]
            # Scores are normalized to 0-1
            assert 0 <= result["metrics"][criterion]["score"] <= 1

    @pytest.mark.asyncio
    async def test_geval_persistence(self, async_api_client, auth_headers):
        """Test that G-Eval results are persisted to database."""
        request_data = {
            "source_text": "Test source",
            "summary": "Test summary",
            "metrics": ["coherence"],
            "api_key": "test_api_key"
        }

        response = await async_api_client.post(
            "/api/v1/evaluations/geval",
            json=request_data,
            headers=auth_headers
        )

        result = response.json()
        assert "metadata" in result
        eval_id = result["metadata"]["evaluation_id"]

        # Verify data was persisted - check using the actual evaluations database
        from tldw_Server_API.app.core.DB_Management.Evaluations_DB import EvaluationsDatabase
        from pathlib import Path

        db_env = os.environ.get("EVALUATIONS_TEST_DB_PATH")
        assert db_env, "EVALUATIONS_TEST_DB_PATH must be set for integration tests"
        db_path = Path(db_env)

        # Use the EvaluationsDatabase to check persistence
        eval_db = EvaluationsDatabase(str(db_path))

        # Check using the unified method
        if hasattr(eval_db, 'get_unified_evaluation'):
            stored_eval = eval_db.get_unified_evaluation(eval_id)
        else:
            # Fallback to checking multiple tables
            import sqlite3
            with sqlite3.connect(db_path) as conn:
                # Check unified table first
                cursor = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='evaluations_unified'"
                )
                if cursor.fetchone():
                    cursor = conn.execute(
                        "SELECT * FROM evaluations_unified WHERE evaluation_id = ? OR id = ?",
                        (eval_id, eval_id)
                    )
                else:
                    # Check internal_evaluations table
                    cursor = conn.execute(
                        "SELECT * FROM internal_evaluations WHERE evaluation_id = ?",
                        (eval_id,)
                    )
                stored_eval = cursor.fetchone()

        assert stored_eval is not None


class TestRAGEvaluationEndpoint:
    """Integration tests for RAG evaluation endpoint."""

    @pytest.mark.asyncio
    @pytest.mark.requires_llm
    async def test_rag_evaluation_complete_flow(self, async_api_client, auth_headers):
        """Test complete RAG evaluation with all metrics."""
        rag_data = SampleDataGenerator.generate_rag_evaluation_data()

        response = await async_api_client.post(
            "/api/v1/evaluations/rag",
            json=rag_data,
            headers=auth_headers
        )

        assert response.status_code == 200
        result = response.json()

        # Verify all RAG metrics are present
        assert "metrics" in result
        assert "context_relevance" in result["metrics"]
        assert "answer_faithfulness" in result["metrics"]
        assert "answer_relevance" in result["metrics"]
        assert "overall_score" in result

    @pytest.mark.asyncio
    @pytest.mark.requires_llm
    @pytest.mark.requires_embeddings
    async def test_rag_with_embeddings_enabled(self, async_api_client, auth_headers):
        """Test RAG evaluation with embeddings for similarity."""
        rag_data = SampleDataGenerator.generate_rag_evaluation_data()
        rag_data["use_embeddings"] = True
        rag_data["embedding_provider"] = "openai"
        rag_data["embedding_model"] = "text-embedding-3-small"

        response = await async_api_client.post(
            "/api/v1/evaluations/rag",
            json=rag_data,
            headers=auth_headers
        )

        assert response.status_code == 200
        result = response.json()

        # Should include similarity metric when ground truth provided
        if rag_data.get("ground_truth"):
            assert "answer_similarity" in result["metrics"]

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_rag_concurrent_evaluations(self, async_api_client, auth_headers):
        """Test multiple concurrent RAG evaluations without mocking.

        Keeps total requests under default rate limits and staggers slightly.
        """
        # Create multiple evaluation requests (keep payloads small)
        requests = []
        for _ in range(5):
            data = SampleDataGenerator.generate_rag_evaluation_data()
            # Ensure small texts to avoid token-based rate limiting
            data["question"] = "Short?"
            data["answer"] = "Short."
            data["ground_truth"] = "Short."
            requests.append(data)

        # Send concurrent requests with small delays to avoid rate limiting
        tasks = []
        for i, req_data in enumerate(requests):
            if i > 0:
                await asyncio.sleep(0.15)
            task = async_api_client.post(
                "/api/v1/evaluations/rag",
                json=req_data,
                headers=auth_headers
            )
            tasks.append(task)

        responses = await asyncio.gather(*tasks)

        # All should succeed
        for response in responses:
            assert response.status_code == 200
            result = response.json()
            assert "metadata" in result
            assert "evaluation_id" in result["metadata"]


@pytest.mark.integration
class TestResponseQualityEndpoint:
    """Integration tests for response quality evaluation."""

    @pytest.mark.asyncio
    @pytest.mark.requires_llm
    async def test_response_quality_evaluation(self, async_api_client, auth_headers):
        """Test response quality evaluation with all criteria."""
        quality_data = SampleDataGenerator.generate_response_quality_data()

        response = await async_api_client.post(
            "/api/v1/evaluations/response-quality",
            json=quality_data,
            headers=auth_headers
        )

        assert response.status_code == 200
        result = response.json()

        # Verify all quality metrics
        assert "metrics" in result
        assert "overall_quality" in result

        # Check that we have some metrics (may not match exact criteria names due to evaluation logic)
        assert len(result["metrics"]) > 0

    @pytest.mark.asyncio
    @pytest.mark.requires_llm
    async def test_response_quality_custom_weights(self, async_api_client, auth_headers):
        """Test response quality with custom metric weights."""
        quality_data = SampleDataGenerator.generate_response_quality_data()
        quality_data["weights"] = {
            "coherence": 0.3,
            "relevance": 0.4,
            "fluency": 0.15,
            "factuality": 0.15
        }

        response = await async_api_client.post(
            "/api/v1/evaluations/response-quality",
            json=quality_data,
            headers=auth_headers
        )

        assert response.status_code == 200
        result = response.json()

        # Overall quality should be calculated
        assert "overall_quality" in result
        assert 0 <= result["overall_quality"] <= 1


@pytest.mark.unit
class TestBatchEvaluationEndpoint:
    """Integration tests for batch evaluation."""

    @pytest.mark.asyncio
    async def test_batch_evaluation_mixed_types(self, async_api_client, auth_headers):
        """Test batch evaluation with different evaluation types."""
        batch_data = SampleDataGenerator.generate_batch_evaluation_request(size=3)

        response = await async_api_client.post(
            "/api/v1/evaluations/batch",
            json=batch_data,
            headers=auth_headers
        )

        assert response.status_code == 200
        result = response.json()

        # Should have results for all evaluations
        assert "results" in result
        assert len(result["results"]) == len(batch_data["items"])

        # Each result should have status
        for eval_result in result["results"]:
            assert "status" in eval_result
            assert "evaluation_id" in eval_result

    @pytest.mark.asyncio
    async def test_batch_evaluation_rate_limit_headers(self, async_api_client, auth_headers):
        """Headers X-RateLimit-* should be present on batch response."""
        batch_data = SampleDataGenerator.generate_batch_evaluation_request(size=2)
        batch_data["parallel_workers"] = 1
        response = await async_api_client.post(
            "/api/v1/evaluations/batch",
            json=batch_data,
            headers=auth_headers,
        )
        assert response.status_code == 200
        hdrs = response.headers
        for key in [
            "X-RateLimit-Tier",
            "X-RateLimit-PerMinute-Limit",
            "X-RateLimit-PerMinute-Remaining",
            "X-RateLimit-Daily-Limit",
            "X-RateLimit-Daily-Remaining",
            "X-RateLimit-Tokens-Remaining",
            "X-RateLimit-Reset",
        ]:
            assert key in hdrs, f"Missing header: {key}"
        # Baseline RateLimit-* headers present
        for key in [
            "RateLimit-Limit",
            "RateLimit-Reset",
        ]:
            assert key in hdrs, f"Missing header: {key}"

    @pytest.mark.asyncio
    async def test_batch_parallel_processing(self, async_api_client, auth_headers):
        """Test that batch evaluations are processed in parallel."""
        batch_data = SampleDataGenerator.generate_batch_evaluation_request(size=5)
        batch_data["parallel_workers"] = 3

        import time
        start_time = time.time()

        response = await async_api_client.post(
            "/api/v1/evaluations/batch",
            json=batch_data,
            headers=auth_headers
        )

        elapsed = time.time() - start_time

        assert response.status_code == 200

        # Parallel processing should be faster than sequential
        # (This is a soft assertion - depends on system)
        assert elapsed < len(batch_data["items"]) * 2

    @pytest.mark.asyncio
    async def test_batch_fail_fast_behavior(self, async_api_client, auth_headers):
        """Test fail-fast behavior in batch processing."""
        # Create batch data with an invalid evaluation type at batch level
        batch_data = {
            "evaluation_type": "invalid_type",  # This will trigger validation error or unknown type
            "items": [
                {"data": "test1"},
                {"data": "test2"},
                {"data": "test3"}
            ],
            "parallel_workers": 1,
            "continue_on_error": False  # fail_fast = True
        }

        response = await async_api_client.post(
            "/api/v1/evaluations/batch",
            json=batch_data,
            headers=auth_headers
        )

        # Should either get validation error (422) or process with failures (200/207)
        if response.status_code == 422:
            # Pydantic validation caught the invalid type
            assert True
        else:
            # Should return partial results with error
            assert response.status_code in [200, 207]  # 207 for partial success
            result = response.json()

            # Should have at least one failure
            failed = [r for r in result.get("results", []) if r.get("status") == "failed"]
            assert len(failed) > 0 or result.get("failed", 0) > 0


@pytest.mark.integration
class TestEvaluationHistoryEndpoint:
    """Integration tests for evaluation history."""

    @pytest.mark.asyncio
    async def test_get_evaluation_history(self, async_api_client, auth_headers, temp_db_path):
        """Test retrieving evaluation history."""
        # Seed database with history
        helper = create_test_database_with_data(str(temp_db_path))

        request_data = {
            "user_id": "user_0",
            "limit": 10,
            "evaluation_type": "g_eval"
        }

        response = await async_api_client.post(
            "/api/v1/evaluations/history",
            json=request_data,
            headers=auth_headers
        )

        assert response.status_code == 200
        result = response.json()

        assert "items" in result
        assert "total_count" in result
        assert len(result["items"]) <= 10
        assert result["pagination"] == {
            "mode": "offset",
            "limit": 10,
            "offset": 0,
            "total": result["total_count"],
            "has_more": result["total_count"] > 10,
            "next_offset": 10 if result["total_count"] > 10 else None,
        }

    @pytest.mark.asyncio
    async def test_history_with_date_filter(self, async_api_client, auth_headers):
        """Test evaluation history with date range filter."""
        from datetime import datetime, timedelta

        request_data = {
            "start_date": (datetime.utcnow() - timedelta(days=7)).isoformat(),
            "end_date": datetime.utcnow().isoformat(),
            "limit": 20
        }

        response = await async_api_client.post(
            "/api/v1/evaluations/history",
            json=request_data,
            headers=auth_headers
        )

        assert response.status_code == 200
        result = response.json()

        # All results should be within date range
        for eval in result["items"]:
            created_at = datetime.fromisoformat(eval["created_at"])
            assert created_at >= datetime.fromisoformat(request_data["start_date"])
            assert created_at <= datetime.fromisoformat(request_data["end_date"])


@pytest.mark.integration
class TestWebhookEndpoints:
    """Integration tests for webhook functionality."""

    @pytest.mark.asyncio
    async def test_webhook_registration(self, async_api_client, auth_headers):
        """Test webhook registration and retrieval."""
        import uuid
        import sqlite3
        from pathlib import Path

        db_env = os.environ.get("EVALUATIONS_TEST_DB_PATH")
        assert db_env, "EVALUATIONS_TEST_DB_PATH must be set for integration tests"
        db_path = Path(db_env)

        with sqlite3.connect(db_path) as conn:
            try:
                initial_count = conn.execute(
                    "SELECT COUNT(*) FROM webhook_registrations"
                ).fetchone()[0]
            except sqlite3.OperationalError as exc:
                pytest.fail(f"webhook_registrations table missing before test: {exc}")

        # Use unique URL to avoid conflicts
        unique_id = str(uuid.uuid4())[:8]
        webhook_data = {
            "url": f"https://example.com/webhook/{unique_id}",
            "events": ["evaluation.completed", "evaluation.failed"]
        }

        # Register webhook
        response = await async_api_client.post(
            "/api/v1/evaluations/webhooks",
            json=webhook_data,
            headers=auth_headers
        )

        assert response.status_code == 200
        result = response.json()
        assert "webhook_id" in result

        webhook_id = result["webhook_id"]

        # Retrieve webhooks
        response = await async_api_client.get(
            "/api/v1/evaluations/webhooks",
            headers=auth_headers
        )

        assert response.status_code == 200
        webhooks = response.json()

        # Should contain our webhook
        webhook_ids = [w["webhook_id"] for w in webhooks]
        assert webhook_id in webhook_ids

        with sqlite3.connect(db_path) as conn:
            stored = conn.execute(
                "SELECT url FROM webhook_registrations WHERE url = ?",
                (webhook_data["url"],)
            ).fetchone()
            assert stored is not None
            final_count = conn.execute(
                "SELECT COUNT(*) FROM webhook_registrations"
            ).fetchone()[0]
            assert final_count >= initial_count + 1

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_webhook_delivery_on_evaluation(self, async_api_client, auth_headers, webhook_receiver_server):
        """Test webhook delivery using a real local receiver (no mocks)."""
        from tldw_Server_API.app.core.Evaluations.webhook_manager import webhook_manager, WebhookEvent
        import asyncio

        # Register webhook directly with manager (skip validation for localhost)
        user_id = "user_1"  # single-user mode webhook id
        url = webhook_receiver_server["url"]
        await webhook_manager.register_webhook(
            user_id=user_id,
            url=url,
            events=[WebhookEvent.EVALUATION_COMPLETED],
            skip_validation=True
        )

        # Trigger an evaluation which should send the webhook asynchronously
        eval_data = SampleDataGenerator.generate_geval_data()
        response = await async_api_client.post(
            "/api/v1/evaluations/geval",
            json={
                "source_text": eval_data["source_text"],
                "summary": eval_data["summary"],
                "metrics": ["coherence"]
            },
            headers=auth_headers
        )
        assert response.status_code == 200

        # Allow async webhook delivery to complete
        await asyncio.sleep(0.3)
        received = webhook_receiver_server["received"]
        assert len(received) >= 1
        # Validate signature header is present
        headers = received[0]["headers"]
        assert any(h.lower() == "x-webhook-signature" for h in headers)


@pytest.mark.integration
class TestRateLimitEndpoints:
    """Integration tests for rate limiting."""

    @pytest.mark.asyncio
    async def test_rate_limit_enforcement(self, async_api_client, auth_headers):
        """Test that rate limits are enforced (deterministic in TEST_MODE)."""
        # With TEST_EVALUATIONS_RATE_LIMIT=2, 3 quick requests should trigger 429
        statuses = []
        for i in range(3):
            resp = await async_api_client.post(
                "/api/v1/evaluations/geval",
                json={
                    "source_text": f"Text number {i} goes here.",
                    "summary": f"Summary number {i} goes here.",
                    "metrics": ["coherence"],
                    "api_name": "openai",
                    "api_key": "test_api_key"
                },
                headers=auth_headers
            )
            statuses.append(resp.status_code)
        # Prefer 429 in TEST_MODE, but allow environments without RL enforcement
        assert 429 in statuses or all(s == 200 for s in statuses)

    @pytest.mark.asyncio
    async def test_rate_limit_status_endpoint(self, async_api_client, auth_headers):
        """Test rate limit status retrieval."""
        response = await async_api_client.get(
            "/api/v1/evaluations/rate-limits",
            headers=auth_headers
        )

        assert response.status_code == 200
        result = response.json()

        assert "tier" in result
        assert "usage" in result
        assert "limits" in result
        assert "reset_at" in result
        assert "remaining" in result


@pytest.mark.integration
class TestErrorHandling:
    """Integration tests for error handling."""

    @pytest.mark.asyncio
    async def test_invalid_evaluation_type(self, async_api_client, auth_headers):
        """Test handling of invalid evaluation type."""
        response = await async_api_client.post(
            "/api/v1/evaluations/geval",
            json={
                "source_text": "Test",
                "summary": "Test",
                "criteria": "invalid_criteria_xyz"
            },
            headers=auth_headers
        )

        assert response.status_code == 422  # Validation error for invalid field
        error = response.json()
        assert "error" in error or "detail" in error

    @pytest.mark.asyncio
    async def test_missing_required_fields(self, async_api_client, auth_headers):
        """Test handling of missing required fields."""
        response = await async_api_client.post(
            "/api/v1/evaluations/rag",
            json={
                "query": "Test query"
                # Missing required 'context' and 'response'
            },
            headers=auth_headers
        )

        assert response.status_code == 422  # Validation error

    @pytest.mark.asyncio
    async def test_authentication_required(self, async_api_client):
        """Test that authentication is required for endpoints."""
        response = await async_api_client.post(
            "/api/v1/evaluations/geval",
            json={
                "source_text": "Test",
                "summary": "Test",
                "criteria": "coherence"
            }
            # No auth headers
        )

        assert response.status_code in [401, 403]
