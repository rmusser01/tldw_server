"""Sanitizer coverage for RAG performance monitor fallbacks."""

from tldw_Server_API.app.core.RAG.rag_service import performance_monitor
from tldw_Server_API.app.core.RAG.rag_service.performance_monitor import PerformanceMonitor


def test_record_query_sanitizes_resource_usage_fallback_log():
    """Resource usage fallback logs should not expose backend exception details."""

    class BrokenProcess:
        def memory_info(self):
            raise RuntimeError(
                "psutil failed for /private/rag-private.db?token=secret-token"
            )

        def cpu_percent(self):  # pragma: no cover - memory_info fails first
            return 99.0

    monitor = PerformanceMonitor()
    monitor._process = BrokenProcess()

    messages: list[str] = []
    sink_id = performance_monitor.logger.add(
        lambda message: messages.append(str(message)),
        level="DEBUG",
    )
    try:
        monitor.record_query(
            query="safe query",
            total_duration=0.25,
            component_timings={"retrieval": 0.1},
            cache_hit=True,
        )
    finally:
        performance_monitor.logger.remove(sink_id)

    joined = "\n".join(messages)
    assert "Failed to get resource usage" in joined
    assert "rag-private.db" not in joined
    assert "secret-token" not in joined

    metrics = monitor._query_metrics[-1]
    assert metrics.memory_usage_mb == 0.0
    assert metrics.cpu_percent == 0.0
    assert metrics.query_count == 1
    assert monitor.total_queries == 1
