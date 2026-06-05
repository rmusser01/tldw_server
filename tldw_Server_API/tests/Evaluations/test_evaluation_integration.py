"""
Comprehensive integration tests for the evaluations module.

Tests the complete evaluation pipeline including:
- Embeddings integration
- Error handling
- Database operations
- Authentication
- Rate limiting
"""

import pytest
pytestmark = pytest.mark.unit
import asyncio
import os
from pathlib import Path
from unittest.mock import Mock, patch
import sqlite3

from tldw_Server_API.app.core.Evaluations.rag_evaluator import RAGEvaluator
from tldw_Server_API.app.core.Evaluations.evaluation_manager import EvaluationManager
from tldw_Server_API.app.core.DB_Management.migrations import migrate_evaluations_database


class TestEvaluationIntegration:
    """Integration tests for the complete evaluation pipeline."""

    @pytest.fixture
    def temp_db_path(self, tmp_path: Path):
        """Create a temporary database for testing."""
        return tmp_path / "evaluations.db"


    @pytest.fixture
    def evaluation_manager(self, temp_db_path):
        """Create an evaluation manager with temporary database."""
        # Patch the db_path after initialization since EvaluationManager doesn't take db_path arg
        manager = EvaluationManager()
        manager.db_path = temp_db_path
        manager._init_database()  # Re-initialize with new path
        return manager

    @pytest.mark.asyncio
    async def test_full_evaluation_pipeline(self, evaluation_manager):
        """Test the complete evaluation pipeline from input to storage."""
        # Create evaluator without mocking - will fall back to LLM if embeddings unavailable
        evaluator = RAGEvaluator()
        evaluator.embedding_available = False  # Force LLM path for deterministic testing

        # Mock LLM call for predictable results
        with patch('tldw_Server_API.app.core.Evaluations.rag_evaluator.asyncio.to_thread') as mock_thread:
            mock_thread.return_value = "4"  # Good score

            # Run evaluation
            results = await evaluator.evaluate(
                query="What is machine learning?",
                contexts=["ML is a subset of AI", "It learns from data"],
                response="Machine learning is AI that learns from data",
                ground_truth="ML is artificial intelligence that learns patterns from data",
                metrics=["answer_similarity"]
            )

            # Verify results structure
            assert "metrics" in results
            assert "answer_similarity" in results["metrics"]
            assert results["metrics"]["answer_similarity"]["method"] == "llm"

            # Store evaluation
            eval_id = await evaluation_manager.store_evaluation(
                evaluation_type="rag_evaluation",
                input_data={"query": "What is machine learning?"},
                results=results
            )

            assert eval_id is not None

            # Retrieve and verify
            history = await evaluation_manager.get_history(
                evaluation_type="rag_evaluation",
                limit=1
            )

            # get_history returns a dict with 'items' key
            assert "items" in history
            assert len(history["items"]) == 1
            assert history["items"][0]["evaluation_id"] == eval_id

            # Cleanup
            evaluator.close()

    @pytest.mark.asyncio
    async def test_error_propagation(self):
        """Test that errors are properly propagated instead of returning 0.0."""
        # Create evaluator without embeddings
        evaluator = RAGEvaluator()
        evaluator.embedding_available = False  # Force LLM path

        # Mock LLM to fail
        with patch('tldw_Server_API.app.core.Evaluations.rag_evaluator.asyncio.to_thread') as mock_thread:
            mock_thread.side_effect = RuntimeError("LLM error")

            # Should raise error instead of returning 0.0
            with pytest.raises(ValueError) as exc_info:
                await evaluator._evaluate_answer_similarity(
                    "response", "ground_truth"
                )

            assert "Answer similarity evaluation failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_partial_failure_handling(self):
        """Test that partial failures are handled gracefully."""
        evaluator = RAGEvaluator()
        evaluator.embedding_available = False  # Force LLM path

        # Mock one metric to fail
        with patch.object(evaluator, '_evaluate_relevance') as mock_relevance:
            with patch.object(evaluator, '_evaluate_faithfulness') as mock_faithfulness:
                mock_relevance.side_effect = ValueError("Relevance failed")
                mock_faithfulness.return_value = ("faithfulness", {
                    "score": 0.8,
                    "name": "faithfulness"
                })

                results = await evaluator.evaluate(
                    query="test",
                    contexts=["context"],
                    response="response",
                    metrics=["relevance", "faithfulness"]
                )

                # Should have partial results
                assert "metrics" in results
                # faithfulness is stored as answer_faithfulness
                assert "answer_faithfulness" in results["metrics"]
                assert "failed_metrics" in results
                assert "relevance" in results["failed_metrics"]
                assert results.get("partial_results") is True

    def test_database_migration(self, temp_db_path):

        """Test that database migrations work correctly."""
        # Apply migrations
        migrate_evaluations_database(temp_db_path)

        # Check that tables exist with correct schema
        with sqlite3.connect(temp_db_path) as conn:
            cursor = conn.cursor()

            # Check internal_evaluations table (not evaluations)
            cursor.execute("PRAGMA table_info(internal_evaluations)")
            columns = {col[1] for col in cursor.fetchall()}

            # Should have all columns from migrations
            expected_columns = {
                'id', 'evaluation_id', 'evaluation_type', 'created_at',
                'input_data', 'results', 'metadata', 'user_id',
                'status', 'error_message', 'completed_at',
                'embedding_provider', 'embedding_model'
            }

            assert expected_columns.issubset(columns)

            # Check migration tracking
            cursor.execute("SELECT version FROM schema_migrations ORDER BY version")
            versions = [row[0] for row in cursor.fetchall()]
            assert len(versions) >= 4  # Should have at least 4 migrations

    @pytest.mark.asyncio
    async def test_embedding_fallback(self):
        """Test that evaluation falls back to LLM when embeddings unavailable."""
        # Create evaluator without embeddings
        with patch('tldw_Server_API.app.core.Evaluations.rag_evaluator.create_embedding') as mock_create:
            mock_create.side_effect = RuntimeError("No API key")

            evaluator = RAGEvaluator()
            assert evaluator.embedding_available is False

            # Mock LLM response
            with patch('tldw_Server_API.app.core.Evaluations.rag_evaluator.asyncio.to_thread') as mock_thread:
                mock_thread.return_value = "4"

                _, result = await evaluator._evaluate_answer_similarity(
                    "response", "ground_truth"
                )

                assert result["method"] == "llm"
                assert result["score"] == 0.8  # 4/5

    @pytest.mark.asyncio
    async def test_concurrent_evaluations(self, evaluation_manager):
        """Test that multiple evaluations can run concurrently."""
        evaluator = RAGEvaluator()
        evaluator.embedding_available = False  # Force LLM path

        # Mock LLM for predictable results
        with patch('tldw_Server_API.app.core.Evaluations.rag_evaluator.asyncio.to_thread') as mock_thread:
            mock_thread.return_value = "3"  # Average score

            # Create multiple evaluation tasks
            tasks = []
            for i in range(5):
                task = evaluator.evaluate(
                    query=f"Query {i}",
                    contexts=[f"Context {i}"],
                    response=f"Response {i}",
                    metrics=["relevance"]
                )
                tasks.append(task)

            # Run concurrently
            results = await asyncio.gather(*tasks)

            # All should complete
            assert len(results) == 5
            for result in results:
                assert "metrics" in result

            evaluator.close()

    @pytest.mark.asyncio
    async def test_evaluation_history_tracking(self, evaluation_manager):
        """Test that evaluation history is properly tracked."""
        # Store multiple evaluations
        eval_ids = []
        for i in range(3):
            eval_id = await evaluation_manager.store_evaluation(
                evaluation_type="test_eval",
                input_data={"index": i},
                results={"score": i * 0.1},
                metadata={"test": True}
            )
            eval_ids.append(eval_id)
            await asyncio.sleep(0.01)  # Small delay to ensure different timestamps

        # Get history
        history = await evaluation_manager.get_history(
            evaluation_type="test_eval",
            limit=10
        )

        # Should have all evaluations
        assert "items" in history
        assert len(history["items"]) == 3

        # Should be in reverse chronological order
        retrieved_ids = [h["evaluation_id"] for h in history["items"]]
        assert retrieved_ids == list(reversed(eval_ids))

    @pytest.mark.asyncio
    async def test_evaluation_comparison(self, evaluation_manager):
        """Test comparing multiple evaluations."""
        # Store evaluations with metrics
        eval_ids = []
        for i in range(2):
            results = {
                "metrics": {
                    "relevance": {"score": 0.5 + i * 0.2},
                    "faithfulness": {"score": 0.6 + i * 0.1}
                }
            }
            eval_id = await evaluation_manager.store_evaluation(
                evaluation_type="comparison_test",
                input_data={"version": i},
                results=results
            )
            eval_ids.append(eval_id)

        # Compare evaluations
        comparison = await evaluation_manager.compare_evaluations(eval_ids)

        assert "comparison_summary" in comparison
        assert "metric_comparisons" in comparison
        assert "best_performing" in comparison
        assert "relevance" in comparison["best_performing"]


class TestAuthentication:
    """Test authentication improvements."""

    @pytest.fixture
    def mock_settings(self):
        """Create mock settings."""
        settings = Mock()
        settings.AUTH_MODE = "single_user"
        settings.SINGLE_USER_ALLOWED_IPS = []
        settings.AUTH_TRUST_X_FORWARDED_FOR = False
        settings.AUTH_TRUSTED_PROXY_IPS = []
        return settings

    @pytest.mark.asyncio
    async def test_single_user_auth(self, mock_settings, setup_auth_db):
        """Test single-user authentication without hardcoded keys."""
        from tldw_Server_API.app.api.v1.endpoints.evaluations.evaluations_unified import verify_api_key
        from fastapi import HTTPException
        from fastapi.security import HTTPAuthorizationCredentials
        from starlette.requests import Request

        with patch('tldw_Server_API.app.api.v1.endpoints.evaluations.evaluations_auth.get_settings') as mock_get_settings:
            mock_get_settings.return_value = mock_settings

            # Test with valid API key
            with patch.dict(os.environ, {'API_BEARER': 'test-api-key', 'SINGLE_USER_API_KEY': 'test-api-key', 'TEST_MODE': 'false'}):
                creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials="test-api-key")
                request = Request({"type": "http", "client": ("127.0.0.1", 12345), "headers": []})
                result = await verify_api_key(creds, request=request)
                assert result == "test-api-key"

            # Test with invalid API key
            with patch.dict(os.environ, {'API_BEARER': 'correct-key', 'SINGLE_USER_API_KEY': 'correct-key', 'TEST_MODE': 'false'}):
                creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials="wrong-key")
                with pytest.raises(HTTPException) as exc_info:
                    request = Request({"type": "http", "client": ("127.0.0.1", 12345), "headers": []})
                    await verify_api_key(creds, request=request)
                assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_multi_user_jwt_auth(self, mock_settings, setup_auth_db):
        """Test multi-user JWT authentication."""
        from tldw_Server_API.app.api.v1.endpoints.evaluations.evaluations_unified import verify_api_key
        from fastapi.security import HTTPAuthorizationCredentials
        from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User
        from starlette.requests import Request

        mock_settings.AUTH_MODE = "multi_user"

        with patch('tldw_Server_API.app.api.v1.endpoints.evaluations.evaluations_auth.get_settings') as mock_get_settings:
            mock_get_settings.return_value = mock_settings

            async def _fake_verify_jwt(_request, token: str) -> User:
                assert token == "valid-jwt-token"  # nosec B105
                return User(id=123, username="testuser", is_active=True)

            class _JwtService:
                def decode_access_token(self, token: str):
                    assert token == "valid-jwt-token"  # nosec B105
                    return {"sub": "123"}

            with patch('tldw_Server_API.app.api.v1.endpoints.evaluations.evaluations_auth.get_jwt_service', lambda: _JwtService()):
                with patch('tldw_Server_API.app.api.v1.endpoints.evaluations.evaluations_auth.verify_jwt_and_fetch_user', _fake_verify_jwt):
                    with patch.dict(os.environ, {'TEST_MODE': 'false'}):
                        creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials="valid-jwt-token")
                        request = Request({"type": "http", "client": ("127.0.0.1", 12345), "headers": []})
                        result = await verify_api_key(creds, request=request)
                    assert result == "user_123"

    @pytest.mark.asyncio
    async def test_multi_user_jwt_expired_token_maps_error(self, mock_settings, setup_auth_db):
        """Expired JWTs should map to token_expired in evals auth."""
        from tldw_Server_API.app.api.v1.endpoints.evaluations.evaluations_unified import verify_api_key
        from fastapi.security import HTTPAuthorizationCredentials
        from fastapi import HTTPException
        from tldw_Server_API.app.core.AuthNZ.exceptions import TokenExpiredError
        from starlette.requests import Request

        mock_settings.AUTH_MODE = "multi_user"

        with patch('tldw_Server_API.app.api.v1.endpoints.evaluations.evaluations_auth.get_settings') as mock_get_settings:
            mock_get_settings.return_value = mock_settings

            class _JwtService:
                def decode_access_token(self, token: str):
                    raise TokenExpiredError()

            with patch('tldw_Server_API.app.api.v1.endpoints.evaluations.evaluations_auth.get_jwt_service', lambda: _JwtService()):
                creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials="expired-token")
                request = Request({"type": "http", "client": ("127.0.0.1", 12345), "headers": []})
                with pytest.raises(HTTPException) as exc_info:
                    await verify_api_key(creds, request=request)

            assert exc_info.value.status_code == 401
            assert exc_info.value.detail["error"]["code"] == "token_expired"

    @pytest.mark.asyncio
    async def test_multi_user_jwt_invalid_token_maps_error(self, mock_settings, setup_auth_db):
        """Invalid JWTs should map to invalid_token in evals auth."""
        from tldw_Server_API.app.api.v1.endpoints.evaluations.evaluations_unified import verify_api_key
        from fastapi.security import HTTPAuthorizationCredentials
        from fastapi import HTTPException
        from tldw_Server_API.app.core.AuthNZ.exceptions import InvalidTokenError
        from starlette.requests import Request

        mock_settings.AUTH_MODE = "multi_user"

        with patch('tldw_Server_API.app.api.v1.endpoints.evaluations.evaluations_auth.get_settings') as mock_get_settings:
            mock_get_settings.return_value = mock_settings

            class _JwtService:
                def decode_access_token(self, token: str):
                    raise InvalidTokenError("invalid")

            with patch('tldw_Server_API.app.api.v1.endpoints.evaluations.evaluations_auth.get_jwt_service', lambda: _JwtService()):
                creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials="invalid-token")
                request = Request({"type": "http", "client": ("127.0.0.1", 12345), "headers": []})
                with pytest.raises(HTTPException) as exc_info:
                    await verify_api_key(creds, request=request)

            assert exc_info.value.status_code == 401
            assert exc_info.value.detail["error"]["code"] == "invalid_token"

    @pytest.mark.asyncio
    async def test_multi_user_jwt_inactive_user_maps_error(self, mock_settings, setup_auth_db):
        """Inactive users should map to inactive_user in evals auth."""
        from tldw_Server_API.app.api.v1.endpoints.evaluations.evaluations_unified import verify_api_key
        from fastapi.security import HTTPAuthorizationCredentials
        from fastapi import HTTPException
        from starlette.requests import Request
        from tldw_Server_API.app.core.exceptions import InactiveUserError

        mock_settings.AUTH_MODE = "multi_user"

        with patch('tldw_Server_API.app.api.v1.endpoints.evaluations.evaluations_auth.get_settings') as mock_get_settings:
            mock_get_settings.return_value = mock_settings

            class _JwtService:
                def decode_access_token(self, token: str):
                    return {"sub": "123"}

            async def _inactive_user(_request, _token: str):
                raise InactiveUserError("Inactive user")

            with patch('tldw_Server_API.app.api.v1.endpoints.evaluations.evaluations_auth.get_jwt_service', lambda: _JwtService()):
                with patch('tldw_Server_API.app.api.v1.endpoints.evaluations.evaluations_auth.verify_jwt_and_fetch_user', _inactive_user):
                    creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials="inactive-user-token")
                    request = Request({"type": "http", "client": ("127.0.0.1", 12345), "headers": []})
                    with pytest.raises(HTTPException) as exc_info:
                        await verify_api_key(creds, request=request)

            assert exc_info.value.status_code == 400
            assert exc_info.value.detail["error"]["code"] == "inactive_user"

    @pytest.mark.asyncio
    async def test_single_user_auth_respects_ip_allowlist(self, mock_settings, setup_auth_db):
        """Single-user auth should enforce SINGLE_USER_ALLOWED_IPS."""
        from tldw_Server_API.app.api.v1.endpoints.evaluations.evaluations_unified import verify_api_key
        from starlette.requests import Request
        from fastapi import HTTPException

        mock_settings.AUTH_MODE = "single_user"
        mock_settings.SINGLE_USER_API_KEY = "test-api-key-abcdefghijklmnopqrstuvwxyz"
        mock_settings.SINGLE_USER_ALLOWED_IPS = ["203.0.113.10"]

        with patch('tldw_Server_API.app.api.v1.endpoints.evaluations.evaluations_auth.get_settings') as mock_get_settings:
            mock_get_settings.return_value = mock_settings

            request_denied = Request({"type": "http", "client": ("198.51.100.5", 12345), "headers": []})
            with pytest.raises(HTTPException) as exc_info:
                await verify_api_key(credentials=None, x_api_key="test-api-key-abcdefghijklmnopqrstuvwxyz", request=request_denied)
            assert exc_info.value.status_code == 401

            request_allowed = Request({"type": "http", "client": ("203.0.113.10", 12345), "headers": []})
            result = await verify_api_key(credentials=None, x_api_key="test-api-key-abcdefghijklmnopqrstuvwxyz", request=request_allowed)
            assert result == "test-api-key-abcdefghijklmnopqrstuvwxyz"


class TestRateLimiting:
    """Legacy rate-limiting tests removed; unified endpoints handle limits internally."""
    pass


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
