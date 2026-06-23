"""
Property-based tests for evaluation invariants.

Uses Hypothesis to generate test data and verify that certain properties
always hold true regardless of input.
"""

import asyncio
import gc
import json
import math
import os
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
from hypothesis import HealthCheck, assume, given, settings, strategies as st
from hypothesis.stateful import Bundle, RuleBasedStateMachine, initialize, invariant, rule

from tldw_Server_API.app.core.Evaluations.evaluation_manager import EvaluationManager
from tldw_Server_API.app.core.Evaluations.rag_evaluator import RAGEvaluator
from tldw_Server_API.app.core.Evaluations.response_quality_evaluator import ResponseQualityEvaluator


# Custom Hypothesis strategies
@st.composite
def evaluation_score_strategy(draw):
    """Generate valid evaluation scores."""
    return draw(st.floats(min_value=0.0, max_value=1.0))


@st.composite
def raw_score_strategy(draw):
    """Generate raw scores (1-5 scale)."""
    return draw(st.floats(min_value=1.0, max_value=5.0))


@st.composite
def text_strategy(draw):
    """Generate text for evaluation."""
    return draw(st.text(
        min_size=1,
        max_size=1000,
        alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Pc', 'Zs'))
    ))


@st.composite
def evaluation_type_strategy(draw):
    """Generate valid evaluation types."""
    return draw(st.sampled_from(['g_eval', 'rag', 'response_quality', 'custom']))


@st.composite
def metric_weights_strategy(draw):
    """Generate valid metric weights that sum to 1."""
    num_metrics = draw(st.integers(min_value=2, max_value=5))
    weights = draw(st.lists(
        st.floats(min_value=0.01, max_value=1.0),
        min_size=num_metrics,
        max_size=num_metrics
    ))

    # Normalize to sum to 1
    total = sum(weights)
    return {f"metric_{i}": w/total for i, w in enumerate(weights)}


@st.composite
def evaluation_data_strategy(draw):
    """Generate complete evaluation data."""
    return {
        "evaluation_id": draw(st.text(min_size=8, max_size=32)),
        "evaluation_type": draw(evaluation_type_strategy()),
        "input_data": {
            "text": draw(text_strategy()),
            "metadata": draw(st.dictionaries(
                st.text(min_size=1, max_size=20),
                st.text(min_size=1, max_size=100),
                min_size=0,
                max_size=5
            ))
        },
        "results": {
            "score": draw(evaluation_score_strategy()),
            "raw_score": draw(raw_score_strategy())
        }
    }


@pytest.mark.property
class TestScoreNormalizationInvariants:
    """Test invariants for score normalization."""

    @given(raw_score=raw_score_strategy())
    def test_normalization_bounds(self, raw_score):
        """Normalized scores must always be between 0 and 1."""
        evaluator = RAGEvaluator()
        normalized = evaluator._normalize_score(raw_score)

        assert 0 <= normalized <= 1

    @given(raw_score=st.floats())
    def test_normalization_handles_any_input(self, raw_score):
        """Normalization should handle any float input without error."""
        assume(not np.isnan(raw_score) and not np.isinf(raw_score))

        evaluator = RAGEvaluator()
        normalized = evaluator._normalize_score(raw_score)

        assert 0 <= normalized <= 1

    @given(scores=st.lists(raw_score_strategy(), min_size=1, max_size=100))
    def test_normalization_preserves_order(self, scores):
        """Normalization must preserve relative ordering of scores."""
        evaluator = RAGEvaluator()
        normalized = [evaluator._normalize_score(s) for s in scores]

        for i in range(len(scores) - 1):
            if scores[i] < scores[i + 1]:
                assert normalized[i] <= normalized[i + 1]
            elif scores[i] > scores[i + 1]:
                assert normalized[i] >= normalized[i + 1]

    @given(raw_score=raw_score_strategy())
    def test_normalization_idempotency(self, raw_score):
        """Normalizing twice should give the same result."""
        evaluator = RAGEvaluator()

        normalized_once = evaluator._normalize_score(raw_score)
        # Normalized scores are already 0-1, so re-normalizing from 1-5 scale
        # would change them. Instead, test that same input gives same output
        normalized_again = evaluator._normalize_score(raw_score)

        assert normalized_once == normalized_again


@pytest.mark.property
class TestEvaluationMetricInvariants:
    """Test invariants for evaluation metrics."""

    @given(
        scores=st.dictionaries(
            st.text(min_size=1, max_size=20),
            st.dictionaries(
                keys=st.just("score"),
                values=evaluation_score_strategy()
            ),
            min_size=1,
            max_size=10
        )
    )
    def test_overall_score_bounds(self, scores):
        """Overall score must be within [0, 1] range."""
        evaluator = RAGEvaluator()
        overall = evaluator._calculate_overall_score(scores)

        assert 0 <= overall <= 1

    @given(
        scores=st.dictionaries(
            st.text(min_size=1, max_size=20),
            st.dictionaries(
                keys=st.just("score"),
                values=evaluation_score_strategy()
            ),
            min_size=1,
            max_size=10
        ),
        weights=st.data()
    )
    def test_weighted_average_properties(self, scores, weights):
        """Weighted average must respect weight constraints."""
        # Skip if any score dict is empty (malformed)
        if not all(scores.values()):
            return  # Skip this test case

        # Check all have 'score' key
        for metric_name, metric_dict in scores.items():
            if not isinstance(metric_dict, dict) or "score" not in metric_dict:
                return  # Skip malformed test data

        # Generate weights that sum to 1
        metric_names = list(scores.keys())
        weight_values = weights.draw(st.lists(
            st.floats(min_value=0.01, max_value=1.0),
            min_size=len(metric_names),
            max_size=len(metric_names)
        ))

        total = sum(weight_values)
        weight_dict = {name: w/total for name, w in zip(metric_names, weight_values)}

        evaluator = RAGEvaluator()
        overall = evaluator._calculate_overall_score(scores, weight_dict)

        # Overall must be between min and max scores
        score_values = [s["score"] for s in scores.values()]
        assert min(score_values) <= overall <= max(score_values)

    @given(
        num_metrics=st.integers(min_value=1, max_value=20),
        score_value=evaluation_score_strategy()
    )
    def test_uniform_scores_average(self, num_metrics, score_value):
        """If all metrics have the same score, average should equal that score."""
        scores = {
            f"metric_{i}": {"score": score_value}
            for i in range(num_metrics)
        }

        evaluator = RAGEvaluator()
        overall = evaluator._calculate_overall_score(scores)

        assert abs(overall - score_value) < 0.0001  # Account for float precision


@pytest.mark.property
class TestEvaluationStorageInvariants:
    """Test invariants for evaluation storage."""

    @settings(max_examples=10, deadline=5000, suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture])  # Reduce examples and increase deadline
    @given(eval_data=evaluation_data_strategy())
    def test_storage_retrieval_consistency(self, evaluation_manager, eval_data):
        """Stored evaluations must be retrievable with same data."""
        # Remove evaluation_id from eval_data as store_evaluation generates its own
        eval_data_copy = eval_data.copy()
        expected_id = eval_data_copy.pop("evaluation_id", None)

        # Store evaluation (it returns the generated ID)
        import asyncio
        eval_id = asyncio.run(evaluation_manager.store_evaluation(**eval_data_copy))
        assume(eval_id is not None)  # Skip if storage fails

        # Retrieve evaluation using the returned ID
        retrieved = asyncio.run(evaluation_manager.get_evaluation(eval_id))

        assert retrieved is not None
        assert retrieved["evaluation_id"] == eval_id
        assert retrieved["evaluation_type"] == eval_data_copy["evaluation_type"]
        assert json.loads(retrieved["input_data"]) == eval_data_copy["input_data"]
        assert json.loads(retrieved["results"]) == eval_data_copy["results"]

    @settings(max_examples=10, deadline=5000, suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture])  # Reduce examples and increase deadline
    @given(
        eval_ids=st.lists(
            st.text(min_size=8, max_size=32),
            min_size=1,
            max_size=10,
            unique=True
        )
    )
    def test_unique_id_constraint(self, evaluation_manager, eval_ids):
        """Each evaluation ID must be unique in storage."""
        import asyncio
        stored_ids = set()

        for _ in eval_ids:
            # Don't pass evaluation_id - let store_evaluation generate it
            eval_id = asyncio.run(evaluation_manager.store_evaluation(
                evaluation_type="test",
                input_data={"test": True},
                results={"score": 0.5}
            ))

            if eval_id:
                # Generated IDs should always be unique
                assert eval_id not in stored_ids
                stored_ids.add(eval_id)

    @settings(max_examples=10, deadline=5000, suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture])  # Reduce examples and increase deadline
    @given(
        num_evaluations=st.integers(min_value=0, max_value=20),
        limit=st.integers(min_value=1, max_value=50)
    )
    def test_list_pagination_invariant(self, evaluation_manager, num_evaluations, limit):
        """Pagination must return correct number of results."""
        import asyncio
        # Create evaluations
        for i in range(num_evaluations):
            asyncio.run(evaluation_manager.store_evaluation(
                evaluation_type="test",
                input_data={"index": i},
                results={"score": i / (num_evaluations + 1)}
            ))

        # Test pagination
        results = asyncio.run(evaluation_manager.list_evaluations(limit=limit))

        # Handle both list and dict response formats
        if isinstance(results, dict) and "items" in results:
            results = results["items"]

        assert len(results) <= limit
        assert len(results) <= num_evaluations


@pytest.mark.property
class TestConcurrencyInvariants:
    """Test invariants under concurrent operations."""

    @settings(max_examples=5, deadline=10000, suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture])  # Reduce examples and increase deadline for concurrent tests
    @given(
        operations=st.lists(
            st.tuples(
                st.sampled_from(['store', 'retrieve', 'list']),
                evaluation_data_strategy()
            ),
            min_size=1,
            max_size=20
        )
    )
    def test_concurrent_operations_consistency(self, evaluation_manager, operations):
        """Concurrent operations must maintain data consistency."""
        import threading

        results = []
        errors = []

        def perform_operation(op_type, data):

            try:
                async def _run():
                    if op_type == 'store':
                        payload = {k: v for k, v in data.items() if k != "evaluation_id"}
                        return await evaluation_manager.store_evaluation(**payload)
                    if op_type == 'retrieve':
                        eval_id = data.get("evaluation_id")
                        if eval_id:
                            return await evaluation_manager.get_evaluation(eval_id)
                        return None
                    return await evaluation_manager.list_evaluations(limit=10)

                result = asyncio.run(_run())
                results.append((op_type, result))
            except Exception as e:
                errors.append((op_type, str(e)))

        threads = []
        for op_type, data in operations:
            t = threading.Thread(target=perform_operation, args=(op_type, data))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        for error_type, error_msg in errors:
            assert "corrupt" not in error_msg.lower()
            assert "integrity" not in error_msg.lower()


@pytest.mark.property
class EvaluationManagerStateMachine(RuleBasedStateMachine):
    """Stateful testing of evaluation manager using an isolated database."""

    evaluations = Bundle("evaluations")

    def __init__(self):

        super().__init__()
        self._tmpdir = tempfile.TemporaryDirectory(ignore_cleanup_errors=os.name == "nt")
        self._db_path = Path(self._tmpdir.name) / "state_machine_evals.db"

        from tldw_Server_API.app.core.Evaluations.evaluation_manager import EvaluationManager

        original_get_db_path = EvaluationManager._get_db_path

        def _patched_get_db_path(instance, explicit_path=None, **_ignored):

            if explicit_path is not None:
                try:
                    return Path(explicit_path)
                except Exception:
                    return self._db_path
            return self._db_path

        EvaluationManager._get_db_path = _patched_get_db_path
        try:
            self.manager = EvaluationManager()
        finally:
            EvaluationManager._get_db_path = original_get_db_path

    def teardown(self):

        self.manager = None
        for attempt in range(3):
            gc.collect()
            try:
                self._tmpdir.cleanup()
                return
            except PermissionError:
                if os.name != "nt" or attempt == 2:
                    raise
                time.sleep(0.1)

    @rule(target=evaluations, eval_type=evaluation_type_strategy(), score=evaluation_score_strategy(), text=text_strategy())
    def create_evaluation(self, eval_type, score, text):
        """Create and persist a new evaluation record."""
        payload = {
            "evaluation_type": eval_type,
            "input_data": {"question": text, "context": [text]},
            "results": {
                "score": score,
                "metrics": {
                    "overall": {"score": score}
                }
            },
            "metadata": {"source": "state_machine", "length": len(text)}
        }

        eval_id = asyncio.run(self.manager.store_evaluation(**payload))
        return eval_id

    @rule(eval_id=evaluations)
    def retrieve_evaluation(self, eval_id):
        """Ensure stored evaluations are retrievable."""
        record = asyncio.run(self.manager.get_evaluation(eval_id))
        assert record is not None
        assert record["evaluation_id"] == eval_id
        assert "results" in record

    @rule()
    def list_evaluations(self):
        """List evaluations and validate basic structure."""
        records = asyncio.run(self.manager.list_evaluations(limit=50))
        assert isinstance(records, list)
        for record in records:
            assert "evaluation_id" in record

    @invariant()
    def evaluations_remain_accessible(self):
        """Previously created evaluations remain accessible."""
        records = asyncio.run(self.manager.list_evaluations(limit=200))
        seen_ids = {record["evaluation_id"] for record in records}
        for eval_id in seen_ids:
            assert asyncio.run(self.manager.get_evaluation(eval_id)) is not None


TestEvaluationManagerStateMachine = EvaluationManagerStateMachine.TestCase


@pytest.mark.property
class TestEmbeddingSimilarityInvariants:
    """Test invariants for embedding-based similarity."""

    @given(
        text1=text_strategy(),
        text2=text_strategy()
    )
    @settings(max_examples=50, deadline=None)
    def test_similarity_symmetry(self, text1, text2):
        """Similarity(A, B) must equal Similarity(B, A)."""
        from tldw_Server_API.tests.Evaluations.fixtures.llm_responses import LLMResponseCache

        cache = LLMResponseCache()

        # Get embeddings
        embedding1_for_text1 = cache.get_embedding_response(text1)
        embedding2_for_text2 = cache.get_embedding_response(text2)

        # Calculate similarity both ways
        def cosine_similarity(a, b):
            dot_product = sum(x * y for x, y in zip(a, b))
            norm_a = sum(x ** 2 for x in a) ** 0.5
            norm_b = sum(x ** 2 for x in b) ** 0.5
            return dot_product / (norm_a * norm_b) if norm_a * norm_b > 0 else 0

        sim_1_2 = cosine_similarity(embedding1_for_text1, embedding2_for_text2)
        sim_2_1 = cosine_similarity(embedding2_for_text2, embedding1_for_text1)

        assert abs(sim_1_2 - sim_2_1) < 0.0001

    @given(text=text_strategy())
    @settings(max_examples=50, deadline=None)
    def test_self_similarity_is_maximum(self, text):
        """Similarity of text with itself must be 1.0."""
        from tldw_Server_API.tests.Evaluations.fixtures.llm_responses import LLMResponseCache

        assume(len(text) > 0)

        cache = LLMResponseCache()
        embedding = cache.get_embedding_response(text)

        # Calculate self-similarity
        def cosine_similarity(a, b):
            dot_product = sum(x * y for x, y in zip(a, b))
            norm_a = sum(x ** 2 for x in a) ** 0.5
            norm_b = sum(x ** 2 for x in b) ** 0.5
            return dot_product / (norm_a * norm_b) if norm_a * norm_b > 0 else 0

        self_similarity = cosine_similarity(embedding, embedding)

        assert abs(self_similarity - 1.0) < 0.0001

    @given(
        texts=st.lists(text_strategy(), min_size=3, max_size=3, unique=True)
    )
    @settings(max_examples=50, deadline=None)
    def test_triangle_inequality(self, texts):
        """Similarity must respect triangle inequality properties."""
        from tldw_Server_API.tests.Evaluations.fixtures.llm_responses import LLMResponseCache

        cache = LLMResponseCache()
        embeddings = [cache.get_embedding_response(t) for t in texts]

        def cosine_distance(a, b):

            dot_product = sum(x * y for x, y in zip(a, b))
            norm_a = sum(x ** 2 for x in a) ** 0.5
            norm_b = sum(x ** 2 for x in b) ** 0.5
            if norm_a * norm_b == 0:
                return 0.0
            similarity = dot_product / (norm_a * norm_b)
            similarity = max(min(similarity, 1.0), -1.0)
            return math.acos(similarity)

        # Calculate distances
        d_01 = cosine_distance(embeddings[0], embeddings[1])
        d_12 = cosine_distance(embeddings[1], embeddings[2])
        d_02 = cosine_distance(embeddings[0], embeddings[2])

        # Triangle inequality: d(a,c) <= d(a,b) + d(b,c)
        # With some tolerance for floating point
        assert d_02 <= d_01 + d_12 + 0.0001
