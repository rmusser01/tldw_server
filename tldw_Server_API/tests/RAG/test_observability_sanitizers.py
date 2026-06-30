"""Sanitizer coverage for RAG observability helpers."""

import sys

import pytest

import tldw_Server_API.app.core.RAG.rag_service.observability as observability
from tldw_Server_API.app.core.RAG.rag_service.observability import (
    AlertManager,
    AlertSeverity,
    PerformanceMonitor,
    TraceSpan,
    Tracer,
)


def test_performance_monitor_sanitizes_memory_usage_errors(monkeypatch):
    """Memory usage fallback should not expose psutil/backend exception details."""

    class FakeMemoryError(Exception):
        pass

    class BrokenProcess:
        def memory_info(self):
            raise RuntimeError("psutil failed for /private/rag-observability-proc")

        def memory_percent(self):
            return 0.0

    class FakePsutil:
        Error = FakeMemoryError

        @staticmethod
        def Process():
            return BrokenProcess()

    monkeypatch.setitem(sys.modules, "psutil", FakePsutil)

    result = PerformanceMonitor().get_memory_usage()

    assert result == {"error": "memory_usage_unavailable"}


def test_tracer_span_sanitizes_exception_attributes(monkeypatch):
    """Trace span error attributes should not expose raw exception details."""

    monkeypatch.setattr(observability, "OTEL_AVAILABLE", False)
    tracer = Tracer()
    captured: dict[str, TraceSpan] = {}

    with pytest.raises(RuntimeError):
        with tracer.span("rag_step") as span:
            captured["span"] = span
            raise RuntimeError("trace failed at /private/rag-trace.db")

    span = captured["span"]
    assert span.status == "error"
    assert span.attributes["error"] == "span_failed"
    assert span.attributes["error_type"] == "RuntimeError"
    assert "rag-trace.db" not in str(span.attributes)


def test_tracer_sanitizes_opentelemetry_failure_logs(monkeypatch):
    """OpenTelemetry fallback logs should not expose backend exception details."""

    class BrokenTraceModule:
        def set_tracer_provider(self, _provider):
            raise RuntimeError("otel init failed at /private/otel-init.db")

        def get_tracer(self, _name):  # pragma: no cover - init failure stops before this
            return None

    class BrokenOtelTracer:
        def start_span(self, _operation):
            raise RuntimeError("otel start failed at /private/otel-start.db")

    class BrokenOtelSpan:
        def end(self):
            raise RuntimeError("otel end failed at /private/otel-end.db")

    monkeypatch.setattr(observability, "OTEL_AVAILABLE", True)
    monkeypatch.setattr(observability, "trace", BrokenTraceModule())

    messages: list[str] = []
    sink_id = observability.logger.add(lambda message: messages.append(str(message)), level="DEBUG")
    try:
        tracer = Tracer()
        tracer.otel_tracer = BrokenOtelTracer()
        span = tracer.start_span("rag_step")
        span.attributes["otel_span"] = BrokenOtelSpan()
        tracer.end_span(span)
    finally:
        observability.logger.remove(sink_id)

    joined = "\n".join(messages)
    assert "Failed to initialize OpenTelemetry tracer" in joined
    assert "Failed to create OpenTelemetry span" in joined
    assert "Failed to end OpenTelemetry span" in joined
    assert "otel-init.db" not in joined
    assert "otel-start.db" not in joined
    assert "otel-end.db" not in joined


def test_alert_manager_sanitizes_handler_failure_logs():
    """Alert handler failures should not log raw backend exception details."""

    manager = AlertManager()
    manager.add_rule(
        metric_name="rag_error_rate",
        condition=">",
        threshold=0.1,
        severity=AlertSeverity.ERROR,
        message_template="RAG error rate high",
        cooldown_seconds=0,
    )

    def broken_handler(_alert):
        raise RuntimeError("alert handler failed at /private/alert-handler.db")

    manager.add_handler(broken_handler)

    messages: list[str] = []
    sink_id = observability.logger.add(lambda message: messages.append(str(message)), level="ERROR")
    try:
        manager.check_rules({"rag_error_rate": 0.5})
    finally:
        observability.logger.remove(sink_id)

    joined = "\n".join(messages)
    assert "Alert handler failed" in joined
    assert "alert-handler.db" not in joined
