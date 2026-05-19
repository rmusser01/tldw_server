from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from tldw_Server_API.app.api.v1.endpoints import benchmark_api as benchmark_module


class _LoggerStub:
    def __init__(self):
        self.errors = []

    def error(self, message, *args, **kwargs):
        self.errors.append(str(message))


def _assert_sanitized_error_log(logger_stub, expected_message):
    assert logger_stub.errors == [expected_message]
    assert "/private/" not in logger_stub.errors[0]
    assert "exploded" not in logger_stub.errors[0]


@pytest.mark.asyncio
async def test_list_benchmarks_sanitizes_generic_failure(monkeypatch):
    logger_stub = _LoggerStub()

    class _BrokenRegistry:
        def list_benchmarks(self):
            raise RuntimeError("benchmark registry exploded at /private/benchmarks.db")

    monkeypatch.setattr(benchmark_module, "logger", logger_stub)
    monkeypatch.setattr(benchmark_module, "get_registry", lambda: _BrokenRegistry())

    with pytest.raises(HTTPException) as exc_info:
        await benchmark_module.list_benchmarks()

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to list benchmarks"
    _assert_sanitized_error_log(logger_stub, "Failed to list benchmarks")


@pytest.mark.asyncio
async def test_get_benchmark_info_sanitizes_generic_failure(monkeypatch):
    logger_stub = _LoggerStub()

    class _BrokenRegistry:
        def get(self, name):
            return object()

        def get_benchmark_info(self, name):
            raise RuntimeError("benchmark registry exploded at /private/benchmarks.db")

    monkeypatch.setattr(benchmark_module, "logger", logger_stub)
    monkeypatch.setattr(benchmark_module, "get_registry", lambda: _BrokenRegistry())

    with pytest.raises(HTTPException) as exc_info:
        await benchmark_module.get_benchmark_info("demo-benchmark")

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to get benchmark info"
    _assert_sanitized_error_log(logger_stub, "Failed to get benchmark info")


@pytest.mark.asyncio
async def test_get_benchmark_samples_sanitizes_generic_failure(monkeypatch):
    logger_stub = _LoggerStub()

    class _Registry:
        def get(self, name):
            return object()

    def _raise_dataset_error(*args, **kwargs):
        raise RuntimeError("benchmark dataset exploded at /private/benchmarks.json")

    monkeypatch.setattr(benchmark_module, "logger", logger_stub)
    monkeypatch.setattr(benchmark_module, "get_registry", lambda: _Registry())
    monkeypatch.setattr(benchmark_module, "load_benchmark_dataset", _raise_dataset_error)

    with pytest.raises(HTTPException) as exc_info:
        await benchmark_module.get_benchmark_samples("demo-benchmark", limit=5)

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to get samples"
    _assert_sanitized_error_log(logger_stub, "Failed to get benchmark samples")


@pytest.mark.asyncio
async def test_run_benchmark_sanitizes_generic_failure(monkeypatch):
    logger_stub = _LoggerStub()

    class _Registry:
        def get(self, name):
            return SimpleNamespace(evaluation_type="demo-eval")

        def create_evaluator(self, name):
            class _Evaluator:
                def format_for_custom_metric(self, item):
                    return {
                        "name": "demo-metric",
                        "description": "demo",
                        "evaluation_prompt": "{question}",
                        "input_data": {"question": item["question"]},
                        "scoring_criteria": {"accuracy": "high"},
                    }

            return _Evaluator()

    monkeypatch.setattr(benchmark_module, "get_registry", lambda: _Registry())
    monkeypatch.setattr(
        benchmark_module,
        "load_benchmark_dataset",
        lambda *args, **kwargs: [{"question": "What is 2+2?", "category": "math"}],
    )
    monkeypatch.setattr(
        benchmark_module.evaluation_manager,
        "evaluate_custom_metric",
        AsyncMock(return_value={"score": 1.0, "explanation": "ok"}),
    )
    monkeypatch.setattr(
        benchmark_module.evaluation_manager,
        "store_evaluation",
        AsyncMock(side_effect=RuntimeError("benchmark storage exploded at /private/evals.db")),
    )
    monkeypatch.setattr(benchmark_module, "logger", logger_stub)

    request = benchmark_module.BenchmarkRunRequest(
        limit=1,
        api_name="openai",
        parallel=1,
        save_results=True,
    )

    with pytest.raises(HTTPException) as exc_info:
        await benchmark_module.run_benchmark(
            "demo-benchmark",
            request,
            user_id="user-1",
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to run benchmark"
    _assert_sanitized_error_log(logger_stub, "Failed to run benchmark")


@pytest.mark.asyncio
async def test_run_benchmark_sanitizes_item_evaluation_failure_log(monkeypatch):
    logger_stub = _LoggerStub()

    class _Registry:
        def get(self, name):
            return SimpleNamespace(evaluation_type="demo-eval")

        def create_evaluator(self, name):
            class _Evaluator:
                def format_for_custom_metric(self, item):
                    return {
                        "name": "demo-metric",
                        "description": "demo",
                        "evaluation_prompt": "{question}",
                        "input_data": {"question": item["question"]},
                        "scoring_criteria": {"accuracy": "high"},
                    }

            return _Evaluator()

    monkeypatch.setattr(benchmark_module, "get_registry", lambda: _Registry())
    monkeypatch.setattr(
        benchmark_module,
        "load_benchmark_dataset",
        lambda *args, **kwargs: [{"question": "What is 2+2?", "category": "math"}],
    )
    monkeypatch.setattr(
        benchmark_module.evaluation_manager,
        "evaluate_custom_metric",
        AsyncMock(side_effect=RuntimeError("benchmark item exploded at /private/evals.db")),
    )
    monkeypatch.setattr(benchmark_module, "logger", logger_stub)

    request = benchmark_module.BenchmarkRunRequest(
        limit=1,
        api_name="openai",
        parallel=1,
        save_results=False,
    )

    response = await benchmark_module.run_benchmark(
        "demo-benchmark",
        request,
        user_id="user-1",
    )

    assert response.total_samples == 1
    assert response.results_summary["successful"] == 0
    assert response.results_summary["failed"] == 1
    _assert_sanitized_error_log(logger_stub, "Benchmark item evaluation failed")


@pytest.mark.asyncio
async def test_evaluate_simpleqa_sanitizes_generic_failure(monkeypatch):
    from tldw_Server_API.app.core.Evaluations import simpleqa_eval

    logger_stub = _LoggerStub()

    def _raise_simpleqa_error(*args, **kwargs):
        raise RuntimeError("simpleqa backend exploded at /private/simpleqa")

    monkeypatch.setattr(benchmark_module, "logger", logger_stub)
    monkeypatch.setattr(simpleqa_eval, "SimpleQAEvaluation", _raise_simpleqa_error)

    with pytest.raises(HTTPException) as exc_info:
        await benchmark_module.evaluate_simpleqa(
            question="What is 2+2?",
            api_name="openai",
            strict_grading=True,
            user_id="user-1",
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Evaluation failed"
    _assert_sanitized_error_log(logger_stub, "SimpleQA evaluation failed")
