import asyncio
from io import StringIO

import pytest
from loguru import logger

from tldw_Server_API.app.core.Evaluations.eval_runner import EvaluationRunner


@pytest.mark.asyncio
async def test_eval_summarization_parses_geval_dict(monkeypatch, tmp_path):
    runner = EvaluationRunner(db_path=str(tmp_path / "evals.db"))

    def mock_run_geval(*args, **kwargs):

        assert kwargs.get("api_name") == "openai"
        assert kwargs.get("api_key") is None
        assert kwargs.get("model") == "gpt-4o-mini"
        return {
            "metrics": {
                "coherence": 4.0,
                "fluency": 3.0,
            }
        }

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Evaluations.eval_runner.run_geval",
        mock_run_geval,
    )

    sample = {"input": {"source_text": "source", "summary": "summary"}}
    eval_spec = {
        "metrics": ["coherence", "fluency"],
        "threshold": 0.7,
        "evaluator_model": "openai",
        "model": "gpt-4o-mini",
    }

    result = await runner._eval_summarization(sample, eval_spec, {}, "sample_000001")
    assert result["scores"]["coherence"] == pytest.approx(0.8)
    assert result["scores"]["fluency"] == pytest.approx(1.0)
    assert result["passed"] is True


@pytest.mark.asyncio
async def test_eval_summarization_respects_thresholds_dict(monkeypatch, tmp_path):
    runner = EvaluationRunner(db_path=str(tmp_path / "evals.db"))

    def mock_run_geval(*args, **kwargs):
        return {
            "metrics": {
                "coherence": 4.0,
                "fluency": 3.0,
            }
        }

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Evaluations.eval_runner.run_geval",
        mock_run_geval,
    )

    sample = {"input": {"source_text": "source", "summary": "summary"}}
    eval_spec = {
        "metrics": ["coherence", "fluency"],
        "thresholds": {"coherence": 0.85, "fluency": 0.9},
        "evaluator_model": "openai",
        "model": "gpt-4o-mini",
    }

    result = await runner._eval_summarization(sample, eval_spec, {}, "sample_000001")
    assert result["scores"]["coherence"] == pytest.approx(0.8)
    assert result["scores"]["fluency"] == pytest.approx(1.0)
    assert result["passed"] is False


def test_evaluate_passed_missing_metric_threshold_logs_warning(tmp_path):
    runner = EvaluationRunner(db_path=str(tmp_path / "evals.db"))
    scores = {"coherence": 0.9}
    eval_spec = {"thresholds": {"coherence": 0.8, "fluency": 0.9}}
    log_buffer = StringIO()
    handler_id = logger.add(log_buffer, level="WARNING")
    try:
        passed = runner._evaluate_passed(scores, avg_score=0.9, eval_spec=eval_spec, default_threshold=0.7)
    finally:
        logger.remove(handler_id)

    assert passed is False
    assert "missing scores" in log_buffer.getvalue()


@pytest.mark.asyncio
async def test_eval_rag_parses_metric_dicts(tmp_path):
    runner = EvaluationRunner(db_path=str(tmp_path / "evals.db"))

    class _FakeRagEval:
        async def evaluate(self, **kwargs):
            return {
                "metrics": {
                    "relevance": {"score": 0.9},
                    "faithfulness": {"score": 0.8},
                }
            }

    runner._rag_evaluator = _FakeRagEval()  # type: ignore

    sample = {
        "input": {"query": "q", "contexts": ["c"], "response": "r"},
        "expected": {"answer": "a"},
    }
    eval_spec = {"metrics": ["relevance", "faithfulness"], "threshold": 0.7}

    result = await runner._eval_rag(sample, eval_spec, {}, "sample_000002")
    assert result["scores"]["relevance"] == pytest.approx(0.9)
    assert result["scores"]["faithfulness"] == pytest.approx(0.8)


@pytest.mark.asyncio
async def test_eval_rag_accepts_string_expected(tmp_path):
    runner = EvaluationRunner(db_path=str(tmp_path / "evals.db"))

    class _FakeRagEval:
        async def evaluate(self, **kwargs):
            assert kwargs.get("ground_truth") == "expected answer"
            return {
                "metrics": {
                    "relevance": {"score": 0.7},
                }
            }

    runner._rag_evaluator = _FakeRagEval()  # type: ignore

    sample = {
        "input": {"query": "q", "contexts": ["c"], "response": "r"},
        "expected": "expected answer",
    }
    eval_spec = {"metrics": ["relevance"], "threshold": 0.7}

    result = await runner._eval_rag(sample, eval_spec, {}, "sample_000004")
    assert result["scores"]["relevance"] == pytest.approx(0.7)


@pytest.mark.asyncio
async def test_eval_response_quality_parses_metric_dicts(tmp_path):
    runner = EvaluationRunner(db_path=str(tmp_path / "evals.db"))

    class _FakeQualityEval:
        async def evaluate(self, **kwargs):
            return {
                "metrics": {
                    "relevance": {"score": 0.9},
                    "clarity": {"score": 0.8},
                },
                "overall_quality": 0.85,
                "format_compliance": True,
            }

    runner._quality_evaluator = _FakeQualityEval()  # type: ignore

    sample = {"input": {"prompt": "p", "response": "r"}}
    eval_spec = {"metrics": ["relevance", "clarity"], "threshold": 0.7}

    result = await runner._eval_response_quality(sample, eval_spec, {}, "sample_000003")
    assert result["scores"]["relevance"] == pytest.approx(0.9)
    assert result["scores"]["clarity"] == pytest.approx(0.8)
    assert result["scores"]["overall_quality"] == pytest.approx(0.85)


@pytest.mark.asyncio
async def test_rag_pipeline_extracts_metric_scores(monkeypatch, tmp_path):
    runner = EvaluationRunner(db_path=str(tmp_path / "evals.db"))

    async def fake_unified_rag_pipeline(*args, **kwargs):
        return {
            "documents": [{"content": "ctx", "id": "doc1"}],
            "generated_answer": "answer",
            "timings": {"total": 0.01},
        }

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Evaluations.eval_runner.unified_rag_pipeline",
        fake_unified_rag_pipeline,
    )

    class _FakeRagEval:
        async def evaluate(self, **kwargs):
            return {
                "metrics": {
                    "relevance": {"score": 0.9},
                },
                "overall_score": 0.9,
            }

    runner._rag_evaluator = _FakeRagEval()  # type: ignore

    eval_spec = {
        "metrics": ["relevance"],
        "rag_pipeline": {"custom_metrics": False},
    }
    samples = [{"input": {"question": "q"}, "expected": {"answer": "a"}}]

    results, usage = await runner._execute_rag_pipeline_run(
        run_id="run_000001",
        samples=samples,
        eval_spec=eval_spec,
        eval_config={},
    )

    assert usage == {"total_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0}
    per_sample = results["by_config"][0]["per_sample"][0]
    assert per_sample["scores"]["relevance"] == pytest.approx(0.9)


@pytest.mark.asyncio
async def test_exact_match_accepts_string_expected(tmp_path):
    runner = EvaluationRunner(db_path=str(tmp_path / "evals.db"))
    sample = {"input": {"output": "Hello"}, "expected": "hello"}
    eval_spec = {"threshold": 1.0}

    result = await runner._eval_exact_match(sample, eval_spec, {}, "sample_000005")
    assert result["scores"]["exact_match"] == pytest.approx(1.0)
    assert result["passed"] is True


@pytest.mark.asyncio
async def test_includes_accepts_list_expected(tmp_path):
    runner = EvaluationRunner(db_path=str(tmp_path / "evals.db"))
    sample = {"input": {"output": "alpha beta"}, "expected": ["alpha", "gamma"]}
    eval_spec = {"threshold": 0.5}

    result = await runner._eval_includes(sample, eval_spec, {}, "sample_000006")
    assert result["scores"]["includes"] == pytest.approx(0.5)
    assert result["passed"] is True


@pytest.mark.asyncio
async def test_includes_respects_thresholds_dict(tmp_path):
    runner = EvaluationRunner(db_path=str(tmp_path / "evals.db"))
    sample = {"input": {"output": "alpha beta"}, "expected": ["alpha", "gamma"]}
    eval_spec = {"thresholds": {"includes": 0.6}}

    result = await runner._eval_includes(sample, eval_spec, {}, "sample_000008")
    assert result["scores"]["includes"] == pytest.approx(0.5)
    assert result["passed"] is False


@pytest.mark.asyncio
async def test_fuzzy_match_accepts_string_expected(tmp_path):
    runner = EvaluationRunner(db_path=str(tmp_path / "evals.db"))
    sample = {"input": {"output": "hello world"}, "expected": "hello world!"}
    eval_spec = {"threshold": 0.5}

    result = await runner._eval_fuzzy_match(sample, eval_spec, {}, "sample_000007")
    assert result["scores"]["fuzzy_match"] >= 0.5


@pytest.mark.asyncio
async def test_process_batch_honors_timeout(tmp_path):
    runner = EvaluationRunner(db_path=str(tmp_path / "evals.db"), eval_timeout=10)

    async def slow_eval(*, sample, eval_spec, config, sample_id):
        await asyncio.sleep(0.05)
        return {"sample_id": sample_id, "avg_score": 1.0}

    batch = [{"input": {}}, {"input": {}}]
    results = await runner._process_batch(
        batch=batch,
        eval_fn=slow_eval,
        eval_spec={},
        eval_config={},
        max_workers=2,
        start_index=0,
        timeout_seconds=0.01,
    )
    assert all("error" in r and "Timeout" in r["error"] for r in results)


@pytest.mark.asyncio
async def test_process_batch_honors_max_workers(tmp_path):
    runner = EvaluationRunner(db_path=str(tmp_path / "evals.db"))
    current = 0
    max_seen = 0
    lock = asyncio.Lock()

    async def tracking_eval(*, sample, eval_spec, config, sample_id):
        nonlocal current, max_seen
        async with lock:
            current += 1
            max_seen = max(max_seen, current)
        await asyncio.sleep(0.05)
        async with lock:
            current -= 1
        return {"sample_id": sample_id, "avg_score": 1.0}

    batch = [{"input": {}}, {"input": {}}, {"input": {}}, {"input": {}}]
    await runner._process_batch(
        batch=batch,
        eval_fn=tracking_eval,
        eval_spec={},
        eval_config={},
        max_workers=2,
        start_index=0,
        timeout_seconds=1.0,
    )
    assert max_seen <= 2


def test_cancel_run_cleans_up_local_task_without_overwriting_completed_status(monkeypatch, tmp_path):
    runner = EvaluationRunner(db_path=str(tmp_path / "evals.db"))

    class _Task:
        def __init__(self) -> None:
            self.cancel_called = False

        def cancel(self) -> None:
            self.cancel_called = True

    task = _Task()
    runner.running_tasks["run_terminal"] = task

    monkeypatch.setattr(
        runner.db,
        "get_run",
        lambda run_id: {"id": run_id, "status": "completed"},
    )

    status_updates: list[tuple[str, str, str | None]] = []

    def _record_status(run_id: str, status: str, error_message: str | None = None) -> bool:
        status_updates.append((run_id, status, error_message))
        return True

    monkeypatch.setattr(runner.db, "update_run_status", _record_status)

    assert runner.cancel_run("run_terminal") is False
    assert status_updates == []
    assert task.cancel_called is True
    assert "run_terminal" not in runner.running_tasks


def test_cancel_run_keeps_task_registered_when_status_update_fails(monkeypatch, tmp_path):
    runner = EvaluationRunner(db_path=str(tmp_path / "evals.db"))

    class _Task:
        def __init__(self) -> None:
            self.cancel_called = False

        def cancel(self) -> None:
            self.cancel_called = True

    task = _Task()
    runner.running_tasks["run_db_error"] = task

    monkeypatch.setattr(
        runner.db,
        "get_run",
        lambda run_id: {"id": run_id, "status": "running"},
    )

    def _raise_status_error(run_id: str, status: str, error_message: str | None = None) -> bool:
        raise RuntimeError("db write failed")

    monkeypatch.setattr(runner.db, "update_run_status", _raise_status_error)

    with pytest.raises(RuntimeError, match="db write failed"):
        runner.cancel_run("run_db_error")

    assert task.cancel_called is True
    assert runner.running_tasks["run_db_error"] is task
