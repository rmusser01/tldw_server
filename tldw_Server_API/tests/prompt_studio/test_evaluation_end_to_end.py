# test_evaluation_end_to_end.py
# Comprehensive end-to-end test for Prompt Studio evaluation system

import pytest
import json
import tempfile
import os
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime
from typing import Dict, List, Any, Optional

from tldw_Server_API.app.core.DB_Management.PromptStudioDatabase import PromptStudioDatabase
from tldw_Server_API.app.core.Prompt_Management.prompt_studio.evaluation_manager import EvaluationManager
from tldw_Server_API.app.core.Prompt_Management.prompt_studio.test_runner import TestRunner
from tldw_Server_API.app.core.Prompt_Management.prompt_studio.evaluation_metrics import EvaluationMetrics
from loguru import logger

########################################################################################################################
# Test Fixtures

@pytest.fixture
def test_db():
    """Create a test database with Prompt Studio schema."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        db_path = tmp.name

    try:
        # Initialize database
        db = PromptStudioDatabase(db_path, client_id="test_client")
        yield db
    finally:
        # Cleanup
        try:
            os.unlink(db_path)
        except:
            _ = None

@pytest.fixture
def mock_llm_responses():
    """Predefined LLM responses for testing."""
    return {
        "test_case_1": "The capital of France is Paris.",
        "test_case_2": "Python is a high-level programming language.",
        "test_case_3": "Machine learning is a subset of artificial intelligence.",
        "default": "This is a test response."
    }

@pytest.fixture
def mock_chat_api_call(mock_llm_responses):
    """Mock the LLM adapter call."""
    def _mock_call(*args, **kwargs):
        # Extract messages to determine which response to return
        messages = kwargs.get('messages_payload', kwargs.get('messages', []))

        # Simple logic to return different responses based on input
        if messages:
            user_message = next((m for m in messages if m.get('role') == 'user'), None)
            if user_message:
                content = user_message.get('content', '')
                if 'capital' in content.lower() and 'france' in content.lower():
                    return mock_llm_responses["test_case_1"]
                elif 'python' in content.lower():
                    return mock_llm_responses["test_case_2"]
                elif 'machine learning' in content.lower():
                    return mock_llm_responses["test_case_3"]

        return mock_llm_responses["default"]

    return _mock_call

########################################################################################################################
# Test Data Helpers

def create_test_project(db: PromptStudioDatabase, name: str = "Test Project") -> int:
    """Create a test project and return its ID."""
    result = db.create_project(
        name=name,
        description="Test project for evaluation testing"
    )
    # create_project returns a dict, extract the ID
    return result["id"] if isinstance(result, dict) else result

def create_test_prompt(db: PromptStudioDatabase, project_id: int, name: str,
                      system_prompt: str, user_prompt: str) -> int:
    """Create a test prompt and return its ID."""
    conn = db.get_connection()
    cursor = conn.cursor()

    # Check for existing versions of this prompt name
    cursor.execute("""
        SELECT MAX(version_number) FROM prompt_studio_prompts
        WHERE project_id = ? AND name = ?
    """, (project_id, name))

    max_version = cursor.fetchone()[0]
    version_number = (max_version + 1) if max_version else 1

    # The modules_config JSON can hold model settings
    modules_config = json.dumps({
        "model": "gpt-3.5-turbo",
        "temperature": 0.7,
        "max_tokens": 1000
    })

    cursor.execute("""
        INSERT INTO prompt_studio_prompts (
            project_id, name, system_prompt, user_prompt,
            version_number, modules_config, client_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        project_id, name, system_prompt, user_prompt,
        version_number, modules_config, db.client_id
    ))

    prompt_id = cursor.lastrowid
    conn.commit()
    return prompt_id

def create_test_case(
    db: PromptStudioDatabase,
    project_id: int,
    name: str,
    inputs: Dict[str, Any],
    expected_outputs: Dict[str, Any],
) -> int:
    """Create a test case and return its ID."""
    conn = db.get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO prompt_studio_test_cases (
            project_id, name, inputs, expected_outputs,
            is_golden, client_id
        ) VALUES (?, ?, ?, ?, ?, ?)
    """, (
        project_id, name, json.dumps(inputs), json.dumps(expected_outputs),
        0, db.client_id
    ))

    test_case_id = cursor.lastrowid
    conn.commit()
    return test_case_id

########################################################################################################################
# Main Test Class

class TestPromptStudioEvaluationEndToEnd:
    """Comprehensive end-to-end test for Prompt Studio evaluation system."""

    def test_complete_evaluation_workflow(self, test_db, mock_chat_api_call):

        """Test the complete evaluation workflow from start to finish."""
        # 1. Create project
        project_id = create_test_project(test_db, "Evaluation Test Project")
        assert project_id > 0
        logger.info(f"Created project with ID: {project_id}")

        # 2. Create prompts
        prompt1_id = create_test_prompt(
            test_db,
            project_id,
            "Geography Prompt",
            "You are a helpful geography assistant.",
            "What is the capital of {country}?"
        )

        prompt2_id = create_test_prompt(
            test_db,
            project_id,
            "Programming Prompt",
            "You are a programming expert.",
            "Explain what {language} is."
        )

        assert prompt1_id > 0 and prompt2_id > 0
        logger.info(f"Created prompts with IDs: {prompt1_id}, {prompt2_id}")

        # 3. Create test cases
        test_case1_id = create_test_case(
            test_db,
            project_id,
            "France Capital Test",
            {"country": "France"},
            {"response": "The capital of France is Paris."}
        )

        test_case2_id = create_test_case(
            test_db,
            project_id,
            "Python Language Test",
            {"language": "Python"},
            {"response": "Python is a high-level programming language."}
        )

        test_case3_id = create_test_case(
            test_db,
            project_id,
            "ML Definition Test",
            {"topic": "machine learning"},
            {"response": "Machine learning is a subset of artificial intelligence."}
        )

        assert all([test_case1_id > 0, test_case2_id > 0, test_case3_id > 0])
        logger.info(f"Created test cases with IDs: {test_case1_id}, {test_case2_id}, {test_case3_id}")

        # 4. Run evaluation with mocked LLM
        with patch(
            'tldw_Server_API.app.core.Prompt_Management.prompt_studio.evaluation_manager.EvaluationManager._call_adapter_text',
            mock_chat_api_call,
        ):
            eval_manager = EvaluationManager(test_db)

            # Run evaluation for first prompt
            result = eval_manager.run_evaluation(
                prompt_id=prompt1_id,
                test_case_ids=[test_case1_id],
                model="gpt-3.5-turbo",
                temperature=0.7,
                max_tokens=100
            )

            # 5. Verify results
            assert result is not None
            assert result["status"] == "completed"
            assert result["prompt_id"] == prompt1_id
            assert "metrics" in result
            assert "results" in result

            # Check metrics
            metrics = result["metrics"]
            assert metrics["total_tests"] == 1
            assert metrics["passed"] == 1  # Should pass with exact match
            assert metrics["average_score"] == 1.0

            # Check individual results
            results = result["results"]
            assert len(results) == 1
            assert results[0]["test_case_id"] == test_case1_id
            assert results[0]["passed"] == True
            assert results[0]["score"] == 1.0

            logger.info(f"Evaluation completed successfully with metrics: {metrics}")

    def test_multiple_test_cases_evaluation(self, test_db, mock_chat_api_call):

        """Test evaluation with multiple test cases."""
        # Setup
        project_id = create_test_project(test_db, "Multi Test Project")

        # Create a versatile prompt
        prompt_id = create_test_prompt(
            test_db,
            project_id,
            "Versatile Prompt",
            "You are a knowledgeable assistant.",
            "Answer this question: {question}"
        )

        # Create multiple test cases
        test_case_ids = []
        test_cases_data = [
            ("Capital Test", {"question": "What is the capital of France?"},
            {"response": "The capital of France is Paris."}),
            ("Programming Test", {"question": "What is Python?"},
            {"response": "Python is a high-level programming language."}),
            ("ML Test", {"question": "What is machine learning?"},
            {"response": "Machine learning is a subset of artificial intelligence."})
        ]

        for name, inputs, outputs in test_cases_data:
            test_case_id = create_test_case(test_db, project_id, name, inputs, outputs)
            test_case_ids.append(test_case_id)

        # Run evaluation with all test cases
        with patch(
            'tldw_Server_API.app.core.Prompt_Management.prompt_studio.evaluation_manager.EvaluationManager._call_adapter_text',
            mock_chat_api_call,
        ):
            eval_manager = EvaluationManager(test_db)

            result = eval_manager.run_evaluation(
                prompt_id=prompt_id,
                test_case_ids=test_case_ids,
                model="gpt-3.5-turbo"
            )

            # Verify all test cases were evaluated
            assert result["status"] == "completed"
            assert len(result["results"]) == 3
            assert result["metrics"]["total_tests"] == 3

            # All should pass with our mocked responses
            assert result["metrics"]["passed"] == 3
            assert result["metrics"]["pass_rate"] == 1.0

            logger.info(f"Multi-test evaluation passed with {result['metrics']['passed']}/{result['metrics']['total_tests']} tests")

    def test_evaluation_records_real_test_run_ids(self, test_db, mock_chat_api_call):
        """Evaluation metadata should store test run IDs, not test case IDs."""
        project_id = create_test_project(test_db, "Run ID Project")
        prompt_id = create_test_prompt(
            test_db,
            project_id,
            "Run ID Prompt",
            "You are concise.",
            "Answer: {question}"
        )
        tc1_id = create_test_case(
            test_db,
            project_id,
            "Case 1",
            {"question": "What is Python?"},
            {"response": "Python is a high-level programming language."},
        )
        tc2_id = create_test_case(
            test_db,
            project_id,
            "Case 2",
            {"question": "What is ML?"},
            {"response": "Machine learning is a subset of artificial intelligence."},
        )
        assert tc2_id > tc1_id

        with patch(
            'tldw_Server_API.app.core.Prompt_Management.prompt_studio.evaluation_manager.EvaluationManager._call_adapter_text',
            mock_chat_api_call,
        ):
            eval_manager = EvaluationManager(test_db)
            result = eval_manager.run_evaluation(
                prompt_id=prompt_id,
                test_case_ids=[tc2_id],
                model="gpt-3.5-turbo",
            )

        stored_eval = test_db.get_evaluation(result["id"])
        assert stored_eval is not None
        run_ids = stored_eval.get("test_run_ids") or []
        assert len(run_ids) == 1
        # In a fresh DB this run id should differ from the second test case id.
        assert run_ids[0] != tc2_id

    def test_evaluation_metrics_calculation(self, test_db):

        """Test various evaluation metrics calculations."""
        metrics = EvaluationMetrics()

        # Test exact match
        score = metrics.calculate_exact_match("Hello World", "Hello World")
        assert score == 1.0

        score = metrics.calculate_exact_match("Hello World", "hello world")
        assert score == 0.0  # Case sensitive

        # Test similarity
        score = metrics.calculate_similarity("Hello World", "Hello World")
        assert score == 1.0

        score = metrics.calculate_similarity("Hello World", "Hello Earth")
        assert 0.5 < score < 0.8  # Partial similarity

        # Test word overlap
        score = metrics.calculate_word_overlap("the quick brown fox", "the lazy brown dog")
        assert score == 0.5  # 2 out of 4 words match

        # Test contains score
        score = metrics.calculate_contains_score("Paris", "The capital is Paris, France")
        assert score == 1.0

        score = metrics.calculate_contains_score("London", "The capital is Paris, France")
        assert score == 0.0

        # Test composite score
        expected = {"response": "The answer is 42"}
        actual = {"response": "The answer is 42"}
        score = metrics.calculate_composite_score(expected, actual)
        assert score == 1.0  # Perfect match

        logger.info("All metrics calculations tested successfully")

    def test_evaluation_comparison(self, test_db, mock_chat_api_call):

        """Test comparing multiple evaluations."""
        # Setup
        project_id = create_test_project(test_db, "Comparison Project")
        prompt_id = create_test_prompt(
            test_db,
            project_id,
            "Test Prompt",
            "You are helpful.",
            "Answer: {question}"
        )

        test_case_id = create_test_case(
            test_db,
            project_id,
            "Test Case",
            {"question": "What is the capital of France?"},
            {"response": "The capital of France is Paris."}
        )

        with patch(
            'tldw_Server_API.app.core.Prompt_Management.prompt_studio.evaluation_manager.EvaluationManager._call_adapter_text',
            mock_chat_api_call,
        ):
            eval_manager = EvaluationManager(test_db)

            # Run two evaluations with different parameters
            eval1 = eval_manager.run_evaluation(
                prompt_id=prompt_id,
                test_case_ids=[test_case_id],
                model="gpt-3.5-turbo",
                temperature=0.7
            )

            eval2 = eval_manager.run_evaluation(
                prompt_id=prompt_id,
                test_case_ids=[test_case_id],
                model="gpt-4",
                temperature=0.3
            )

            # Compare evaluations
            comparison = eval_manager.compare_evaluations([eval1["id"], eval2["id"]])

            assert "evaluations" in comparison
            assert len(comparison["evaluations"]) == 2
            assert "metrics_comparison" in comparison
            assert "best_performer" in comparison["metrics_comparison"]

            logger.info(f"Evaluation comparison completed: best performer is evaluation {comparison['metrics_comparison']['best_performer']}")

    def test_error_recovery(self, test_db):

        """Test error handling and recovery."""
        # Setup
        project_id = create_test_project(test_db, "Error Test Project")
        prompt_id = create_test_prompt(
            test_db,
            project_id,
            "Error Prompt",
            "System",
            "User"
        )

        # Create test case with ID that doesn't exist
        eval_manager = EvaluationManager(test_db)

        # Mock adapter call to raise an error
        def failing_llm(*args, **kwargs):
            raise Exception("LLM API Error")

        with patch(
            'tldw_Server_API.app.core.Prompt_Management.prompt_studio.evaluation_manager.EvaluationManager._call_adapter_text',
            failing_llm,
        ):

            # Create a valid test case
            test_case_id = create_test_case(
                test_db,
                project_id,
                "Failing Test",
                {"input": "test"},
                {"output": "expected"}
            )

            # Run evaluation - should handle error gracefully
            result = eval_manager.run_evaluation(
                prompt_id=prompt_id,
                test_case_ids=[test_case_id],
                model="gpt-3.5-turbo"
            )

            # Should complete but with failed test
            assert result["status"] == "completed"
            assert result["metrics"]["passed"] == 0
            assert result["metrics"]["failed"] == 1
            assert result["results"][0]["passed"] == False
            assert "error" in result["results"][0]["actual"]

            logger.info("Error recovery test passed - evaluation handled LLM failure gracefully")

    @pytest.mark.asyncio
    async def test_async_test_runner(self, test_db, mock_chat_api_call):
        """Test asynchronous test runner functionality."""
        # Setup
        project_id = create_test_project(test_db, "Async Test Project")
        prompt_id = create_test_prompt(
            test_db,
            project_id,
            "Async Prompt",
            "You are an assistant.",
            "Answer: {question}"
        )

        # Create multiple test cases
        test_case_ids = []
        for i in range(3):
            test_case_id = create_test_case(
                test_db,
                project_id,
                f"Async Test {i}",
                {"question": f"Question {i}"},
                {"response": f"Answer {i}"}
            )
            test_case_ids.append(test_case_id)

        with patch(
            'tldw_Server_API.app.core.Prompt_Management.prompt_studio.test_runner.TestRunner._call_adapter',
            mock_chat_api_call,
        ):
            test_runner = TestRunner(test_db)

            # Run single test case
            result = await test_runner.run_test_case(
                prompt_id=prompt_id,
                test_case_id=test_case_ids[0],
                model="gpt-3.5-turbo"
            )

            assert result is not None
            assert result["prompt_id"] == prompt_id
            assert result["test_case_id"] == test_case_ids[0]
            assert "actual" in result

            # Run multiple test cases in parallel
            results = await test_runner.run_multiple_tests(
                prompt_id=prompt_id,
                test_case_ids=test_case_ids,
                model="gpt-3.5-turbo",
                parallel=True
            )

            assert len(results) == 3
            for i, result in enumerate(results):
                assert result["test_case_id"] == test_case_ids[i]
                assert "actual" in result

            logger.info(f"Async test runner completed {len(results)} tests successfully")

    def test_database_integrity(self, test_db):

        """Test database integrity and constraints."""
        # Test foreign key constraints
        with pytest.raises(Exception):
            # Try to create prompt with non-existent project
            test_db.create_prompt(
                project_id=99999,
                name="Invalid Prompt",
                system_prompt="System",
                user_prompt="User"
            )

        # Test unique constraints
        project_id = create_test_project(test_db, "Integrity Test")

        # Create first prompt
        prompt_id = create_test_prompt(
            test_db,
            project_id,
            "Unique Prompt",
            "System",
            "User"
        )

        # Try to create duplicate (should create new version instead)
        prompt_id2 = create_test_prompt(
            test_db,
            project_id,
            "Unique Prompt",
            "System Modified",
            "User Modified"
        )

        assert prompt_id2 != prompt_id  # Should be different IDs

        # Verify both exist by trying to query them
        conn = test_db.get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT id, name FROM prompt_studio_prompts WHERE id = ?", (prompt_id,))
        prompt1 = cursor.fetchone()

        cursor.execute("SELECT id, name FROM prompt_studio_prompts WHERE id = ?", (prompt_id2,))
        prompt2 = cursor.fetchone()

        assert prompt1 is not None
        assert prompt2 is not None
        assert prompt1[1] == prompt2[1]  # Same name

        logger.info("Database integrity tests passed")

    def test_evaluation_persistence(self, test_db, mock_chat_api_call):

        """Test that evaluation results are properly persisted."""
        # Setup
        project_id = create_test_project(test_db, "Persistence Test")
        prompt_id = create_test_prompt(
            test_db,
            project_id,
            "Persist Prompt",
            "System",
            "User: {input}"
        )

        test_case_id = create_test_case(
            test_db,
            project_id,
            "Persist Test",
            {"input": "test"},
            {"response": "expected"}
        )

        with patch(
            'tldw_Server_API.app.core.Prompt_Management.prompt_studio.evaluation_manager.EvaluationManager._call_adapter_text',
            mock_chat_api_call,
        ):
            eval_manager = EvaluationManager(test_db)

            # Run evaluation
            result = eval_manager.run_evaluation(
                prompt_id=prompt_id,
                test_case_ids=[test_case_id]
            )

            eval_id = result["id"]

            # Retrieve evaluation from database
            retrieved_eval = eval_manager.get_evaluation(eval_id)

            assert retrieved_eval is not None
            assert retrieved_eval["id"] == eval_id
            assert retrieved_eval["prompt_id"] == prompt_id
            assert retrieved_eval["status"] == "completed"

            # List evaluations
            eval_list = eval_manager.list_evaluations(
                prompt_id=prompt_id,
                page=1,
                per_page=10
            )

            assert eval_list["evaluations"]
            assert len(eval_list["evaluations"]) >= 1
            assert any(e["id"] == eval_id for e in eval_list["evaluations"])

            logger.info(f"Evaluation {eval_id} successfully persisted and retrieved")

########################################################################################################################
# Run Tests

if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
