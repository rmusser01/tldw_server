import os
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
