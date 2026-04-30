import os
from datetime import datetime, timedelta
from io import StringIO

from loguru import logger
import pytest


def test_jobs_metrics_no_registry_noop(monkeypatch):


     # Simulate environment where metrics registry is unavailable
    from tldw_Server_API.app.core.Jobs import metrics as met

    # Force import-time registry symbol to None
    monkeypatch.setattr(met, "get_metrics_registry", None, raising=False)

    # Ensure registration path does not blow up without registry
    met.ensure_jobs_metrics_registered()

    # All metric helpers should short-circuit and not raise
    met.set_queue_gauges("d", "q", "t", queued=1, processing=0, backlog=1, scheduled=0)
    met.increment_created({"domain": "d", "queue": "q", "job_type": "t"})
    met.increment_completed({"domain": "d", "queue": "q", "job_type": "t"})
    met.increment_cancelled({"domain": "d", "queue": "q", "job_type": "t"})
    met.increment_json_truncated({"domain": "d", "queue": "q", "job_type": "t"}, "payload")
    met.increment_sla_breach({"domain": "d", "queue": "q", "job_type": "t"}, "duration")
    met.observe_queue_latency({"domain": "d", "queue": "q", "job_type": "t"}, None, None)
    met.observe_duration({"domain": "d", "queue": "q", "job_type": "t"}, None, None)


def test_jobs_metrics_registration_failure_log_omits_raw_exception_details(monkeypatch):
    from tldw_Server_API.app.core.Jobs import metrics as met

    class RegistryWithFailingRegister:
        metrics = {}

        def normalize_metric_name(self, name):
            return name.replace(".", "_")

        def register_metric(self, definition):
            raise RuntimeError("secret-token leaked from /tmp/private/jobs.db")

    stream = StringIO()
    sink_id = logger.add(stream, level="DEBUG", format="{message}")
    try:
        monkeypatch.setattr(met, "JOBS_METRICS_REGISTERED", False)
        monkeypatch.setattr(met, "get_metrics_registry", lambda: RegistryWithFailingRegister())

        met.ensure_jobs_metrics_registered()
    finally:
        logger.remove(sink_id)

    logs = stream.getvalue()
    assert "Jobs metrics registration skipped for jobs.queued" in logs
    assert "secret-token" not in logs
    assert "/tmp/private/jobs.db" not in logs


class _RegistryWithObserve:
    metrics = {"jobs_queue_latency_seconds": object()}

    def __init__(self):
        self.observed = []

    def normalize_metric_name(self, name):
        return name.replace(".", "_")

    def observe(self, name, value, labels):
        self.observed.append((name, value, labels))


def _configure_observe_enrichment_failure(monkeypatch, met, registry):
    monkeypatch.setenv("JOBS_METRICS_EXEMPLARS", "true")
    monkeypatch.setenv("JOBS_METRICS_EXEMPLAR_SAMPLING", "1")
    monkeypatch.setattr(met, "JOBS_METRICS_REGISTERED", True)
    monkeypatch.setattr(met, "get_metrics_registry", lambda: registry)

    def fail_sample(_rate):
        raise RuntimeError("secret-token leaked from /tmp/private/jobs.db")

    monkeypatch.setattr(met, "_sample_exemplar", fail_sample)


def test_observe_queue_latency_sanitizes_label_enrichment_failure_log(monkeypatch):
    from tldw_Server_API.app.core.Jobs import metrics as met

    registry = _RegistryWithObserve()
    _configure_observe_enrichment_failure(monkeypatch, met, registry)

    stream = StringIO()
    sink_id = logger.add(stream, level="DEBUG", format="{message} {extra}")
    try:
        created_at = datetime(2026, 1, 1, 12, 0, 0)
        acquired_at = created_at + timedelta(seconds=2)

        met.observe_queue_latency(
            {
                "domain": "d",
                "queue": "q",
                "job_type": "t",
                "trace_id": "trace-1",
                "request_id": "request-1",
            },
            acquired_at,
            created_at,
        )
    finally:
        logger.remove(sink_id)

    assert registry.observed == [
        (
            "jobs.queue_latency_seconds",
            2.0,
            {"domain": "d", "queue": "q", "job_type": "t"},
        )
    ]
    logs = stream.getvalue()
    assert "Failed to enrich queue latency metric labels" in logs
    assert "secret-token" not in logs
    assert "/tmp/private/jobs.db" not in logs


def test_observe_duration_sanitizes_label_enrichment_failure_log(monkeypatch):
    from tldw_Server_API.app.core.Jobs import metrics as met

    registry = _RegistryWithObserve()
    _configure_observe_enrichment_failure(monkeypatch, met, registry)

    stream = StringIO()
    sink_id = logger.add(stream, level="DEBUG", format="{message} {extra}")
    try:
        started_at = datetime(2026, 1, 1, 12, 0, 0)
        completed_at = started_at + timedelta(seconds=3)

        met.observe_duration(
            {
                "domain": "d",
                "queue": "q",
                "job_type": "t",
                "trace_id": "trace-1",
                "request_id": "request-1",
            },
            started_at,
            completed_at,
        )
    finally:
        logger.remove(sink_id)

    assert registry.observed == [
        (
            "jobs.duration_seconds",
            3.0,
            {"domain": "d", "queue": "q", "job_type": "t"},
        )
    ]
    logs = stream.getvalue()
    assert "Failed to enrich job duration metric labels" in logs
    assert "secret-token" not in logs
    assert "/tmp/private/jobs.db" not in logs
