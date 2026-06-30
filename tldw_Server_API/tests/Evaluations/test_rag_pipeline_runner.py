import asyncio
from typing import Any, Dict, List

import pytest

from tldw_Server_API.app.api.v1.schemas.evaluation_schemas_unified import EvaluationSpec, RAGPipelineEvalSpec, ChunkingSweepConfig, RetrieverSweepConfig, RerankerSweepConfig, GenerationSweepConfig
from tldw_Server_API.app.core.Evaluations.eval_runner import EvaluationRunner


def test_rag_pipeline_schema_normalization():


    spec = EvaluationSpec(
        sub_type="rag_pipeline",
        rag_pipeline=RAGPipelineEvalSpec(
            dataset=[{"input": {"question": "Q?"}, "expected": {"answer": "A"}}],
            chunking=ChunkingSweepConfig(method="sentences", chunk_size=512, overlap=64),
            retrievers=[RetrieverSweepConfig(search_mode="hybrid", top_k=8)],
            rerankers=[RerankerSweepConfig(strategy="flashrank", top_k=5)],
            rag=GenerationSweepConfig(model="gpt-4o-mini", temperature=0.1, max_tokens=256),
            search_strategy="grid",
            max_trials=4,
        )
    )
    # Validators normalize scalars to lists for sweeps
    assert isinstance(spec.rag_pipeline.chunking.method, list)  # type: ignore
    assert isinstance(spec.rag_pipeline.chunking.chunk_size, list)  # type: ignore
    assert isinstance(spec.rag_pipeline.rag.model, list)  # type: ignore


@pytest.mark.asyncio
async def test_rag_pipeline_runner_basic(monkeypatch, tmp_path):
    runner = EvaluationRunner(db_path=str(tmp_path / "runner-evals.db"))

    # Stub DB methods to avoid sqlite writes
    class _StubDB:
        def update_run_progress(self, *args, **kwargs):
            return True

        def store_run_results(self, *args, **kwargs):

            self.last = (args, kwargs)
            return True

    runner.db = _StubDB()  # type: ignore

    # Mock unified_rag_pipeline to return deterministic minimal result
    async def _fake_unified_rag_pipeline(query: str, **kwargs):
        return {
            "documents": [{"content": f"ctx for {query}"}],
            "generated_answer": f"ans for {query}",
            "timings": {"total": 0.005},
        }

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Evaluations.eval_runner.unified_rag_pipeline",
        _fake_unified_rag_pipeline,
    )

    # Mock rag evaluator to return simple metric objects with .score
    class _M:
        def __init__(self, s: float):
            self.score = s

    async def _fake_rag_eval(**kwargs):
        return {
            "metrics": {
                "faithfulness": _M(0.8),
                "relevance": _M(0.9),
                "answer_similarity": _M(0.7),
            },
            "overall_score": 0.8,
        }

    class _E:
        async def evaluate(self, **kwargs):
            return await _fake_rag_eval(**kwargs)

    runner._rag_evaluator = _E()  # type: ignore

    eval_spec: Dict[str, Any] = {
        "sub_type": "rag_pipeline",
        "rag_pipeline": {
            "dataset": [
                {"input": {"question": "What is X?"}, "expected": {"answer": "X is ..."}},
                {"input": {"question": "What is Y?"}, "expected": {"answer": "Y is ..."}},
            ],
            "retrievers": [{"search_mode": ["hybrid"], "hybrid_alpha": [0.3, 0.7], "top_k": [5]}],
            "rerankers": [{"strategy": ["flashrank", "cross_encoder"], "top_k": [5]}],
            "rag": {"model": ["gpt-4o-mini"], "temperature": [0.1]},
            "search_strategy": "grid",
        },
        "metrics": ["relevance", "faithfulness", "answer_similarity"],
        "evaluator_model": "openai",
    }

    # Build minimal samples from spec
    samples = eval_spec["rag_pipeline"]["dataset"]

    results, usage = await runner._execute_rag_pipeline_run(
        run_id="run_test",
        samples=samples,
        eval_spec=eval_spec,
        eval_config={},
    )

    assert "leaderboard" in results
    assert len(results["leaderboard"]) >= 2  # two retriever alphas * two rerankers
    assert results["best_config"] is not None
