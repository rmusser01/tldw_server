"""Sanitizer coverage for RAG metrics collector fallback logs."""

import asyncio

import pytest

from tldw_Server_API.app.core.RAG.rag_service import metrics_collector
from tldw_Server_API.app.core.RAG.rag_service.metrics_collector import MetricsCollector


@pytest.mark.asyncio
async def test_aggregation_loop_sanitizes_aggregate_failure_log(monkeypatch):
    collector = MetricsCollector(aggregation_interval=30)
    secret = "/private/rag-metrics.db?token=secret-metrics-token"
    sleep_calls = 0

    async def sleep_then_cancel(_delay):
        nonlocal sleep_calls
        sleep_calls += 1
        if sleep_calls == 2:
            raise asyncio.CancelledError()

    def broken_aggregate(_start_time, _end_time):
        raise RuntimeError(f"aggregation failed at {secret}")

    monkeypatch.setattr(metrics_collector.asyncio, "sleep", sleep_then_cancel)
    monkeypatch.setattr(collector, "aggregate_metrics", broken_aggregate)

    messages: list[str] = []
    sink_id = metrics_collector.logger.add(
        lambda message: messages.append(str(message)),
        level="ERROR",
    )
    try:
        with pytest.raises(asyncio.CancelledError):
            await collector._aggregation_loop()
    finally:
        metrics_collector.logger.remove(sink_id)

    joined = "\n".join(messages)
    assert "Error in metrics aggregation" in joined
    assert "rag-metrics.db" not in joined
    assert "secret-metrics-token" not in joined
    assert sleep_calls == 2
