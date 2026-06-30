"""Sanitizer coverage for RAG custom metrics fallback logs."""

import pytest

from tldw_Server_API.app.core.RAG import rag_custom_metrics
from tldw_Server_API.app.core.RAG.rag_custom_metrics import (
    CustomMetricResult,
    MetricType,
    RAGCustomMetrics,
)


def _capture_logs(level: str = "ERROR") -> tuple[list[str], int]:
    messages: list[str] = []
    sink_id = rag_custom_metrics.logger.add(lambda message: messages.append(str(message)), level=level)
    return messages, sink_id


def _assert_not_leaked(messages: list[str], *secrets: str) -> None:
    joined = "\n".join(messages)
    for secret in secrets:
        assert secret not in joined


@pytest.mark.asyncio
async def test_response_completeness_heuristic_fallback_log_is_sanitized(monkeypatch):
    """Completeness fallback should not log raw LLM exception details."""

    secret = "/private/rag-completeness.db?token=secret-completeness-token"
    query = f"Summarize confidential query from {secret}"
    response = "short fallback response"

    async def broken_analyze(_prompt, _system_prompt):
        raise RuntimeError(f"LLM failed for query={query!r} response={response!r}")

    monkeypatch.setattr(rag_custom_metrics, "analyze", broken_analyze)

    evaluator = RAGCustomMetrics(embedding_config={"embedding_config": {"default_model_id": ""}})
    messages, sink_id = _capture_logs()
    try:
        result = await evaluator.evaluate_response_completeness(query, response)
    finally:
        rag_custom_metrics.logger.remove(sink_id)

    assert result.metric_type == MetricType.RESPONSE_COMPLETENESS
    assert result.score == pytest.approx(3 / 200)
    assert result.confidence == 0.5
    assert result.details == {"word_count": 3, "method": "heuristic"}
    assert result.suggestions == ["Could not perform deep evaluation"]

    joined = "\n".join(messages)
    assert "Failed to evaluate completeness" in joined
    _assert_not_leaked(
        messages,
        "rag-completeness.db",
        "secret-completeness-token",
        query,
        response,
    )


@pytest.mark.asyncio
async def test_evaluate_all_metrics_skipped_metric_log_is_sanitized(monkeypatch):
    """Per-metric failures should skip only that metric without logging raw details."""

    secret = "/private/rag-metric.db?token=secret-metric-token"
    query = f"Find information from {secret}"
    response = f"Do not leak this response tied to {secret}"

    evaluator = RAGCustomMetrics(embedding_config={"embedding_config": {"default_model_id": ""}})

    async def broken_coverage(_query, _retrieved_contexts):
        raise RuntimeError(f"coverage failed for query={query!r} response={response!r}")

    async def diversity(_retrieved_contexts, _sources=None):
        return CustomMetricResult(MetricType.RETRIEVAL_DIVERSITY, 0.8, 0.9)

    async def completeness(_query, _response):
        return CustomMetricResult(MetricType.RESPONSE_COMPLETENESS, 0.7, 0.8)

    async def coherence(_response):
        return CustomMetricResult(MetricType.RESPONSE_COHERENCE, 0.6, 0.7)

    monkeypatch.setattr(evaluator, "evaluate_retrieval_coverage", broken_coverage)
    monkeypatch.setattr(evaluator, "evaluate_retrieval_diversity", diversity)
    monkeypatch.setattr(evaluator, "evaluate_response_completeness", completeness)
    monkeypatch.setattr(evaluator, "evaluate_response_coherence", coherence)

    messages, sink_id = _capture_logs()
    try:
        results = await evaluator.evaluate_all_metrics(query, ["context"], response)
    finally:
        rag_custom_metrics.logger.remove(sink_id)

    assert "coverage" not in results
    assert set(results) == {"diversity", "completeness", "coherence"}
    assert results["diversity"].metric_type == MetricType.RETRIEVAL_DIVERSITY
    assert results["completeness"].metric_type == MetricType.RESPONSE_COMPLETENESS
    assert results["coherence"].metric_type == MetricType.RESPONSE_COHERENCE

    joined = "\n".join(messages)
    assert "Failed to evaluate coverage" in joined
    _assert_not_leaked(
        messages,
        "rag-metric.db",
        "secret-metric-token",
        query,
        response,
    )
