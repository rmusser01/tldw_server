import asyncio
from typing import Any

import pytest

from tldw_Server_API.app.core.Jobs import metrics as jobs_metrics
from tldw_Server_API.app.core.Metrics import metrics_manager
from tldw_Server_API.app.services import jobs_metrics_service as service


pytestmark = pytest.mark.unit


class _LoggerStub:
    def __init__(self) -> None:
        self.debugs: list[str] = []
        self.infos: list[str] = []
        self.warnings: list[str] = []
        self.binds: list[dict[str, Any]] = []

    def bind(self, **kwargs: Any):
        self.binds.append(kwargs)
        return self

    def debug(self, message: str, *args: Any, **kwargs: Any) -> None:
        self.debugs.append(message.format(*args) if args else message)

    def info(self, message: str, *args: Any, **kwargs: Any) -> None:
        self.infos.append(message.format(*args) if args else message)

    def warning(self, message: str, *args: Any, **kwargs: Any) -> None:
        self.warnings.append(message.format(*args) if args else message)


def test_run_forever_reconcile_failure_log_is_sanitized(monkeypatch):
    logger = _LoggerStub()
    svc = object.__new__(service.JobsMetricsService)
    svc.interval = 0.0

    def _fail_reconcile_once() -> int:
        raise RuntimeError("blocking reconcile leaked /tmp/jobs-metrics.db sk-live-metrics")

    def _stop_after_tick(_interval: float) -> None:
        raise KeyboardInterrupt

    svc.reconcile_once = _fail_reconcile_once
    monkeypatch.setenv("JOBS_METRICS_RECONCILE_ENABLE", "true")
    monkeypatch.setattr(service, "logger", logger)
    monkeypatch.setattr(service.time, "sleep", _stop_after_tick)

    with pytest.raises(KeyboardInterrupt):
        svc.run_forever()

    assert logger.warnings == ["Jobs metrics reconcile error"]
    assert logger.binds[-1] == {"error_type": "RuntimeError"}
    rendered = "\n".join(logger.debugs + logger.infos + logger.warnings)
    assert "/tmp/jobs-metrics.db" not in rendered
    assert "sk-live-metrics" not in rendered
    assert "blocking reconcile leaked" not in rendered


@pytest.mark.asyncio
async def test_async_reconcile_loop_failure_log_is_sanitized(monkeypatch):
    stop_event = asyncio.Event()
    logger = _LoggerStub()

    class _FakeService:
        interval = 60.0

        def reconcile_once(self) -> int:
            raise RuntimeError("async reconcile leaked /tmp/jobs-reconcile.db sk-live-reconcile")

    async def _stop_wait(_stop_event: asyncio.Event, _timeout: float) -> None:
        stop_event.set()

    monkeypatch.setenv("JOBS_METRICS_RECONCILE_ENABLE", "true")
    monkeypatch.setattr(service, "logger", logger)
    monkeypatch.setattr(service, "JobsMetricsService", _FakeService)
    monkeypatch.setattr(service, "_wait_for_stop_or_timeout", _stop_wait)

    await service.run_jobs_metrics_reconcile(stop_event)

    assert logger.debugs == ["Jobs reconcile loop error"]
    assert logger.binds[-1] == {"error_type": "RuntimeError"}
    rendered = "\n".join(logger.debugs + logger.infos + logger.warnings)
    assert "/tmp/jobs-reconcile.db" not in rendered
    assert "sk-live-reconcile" not in rendered
    assert "async reconcile leaked" not in rendered


@pytest.mark.asyncio
async def test_slo_gauges_loop_failure_log_is_sanitized(monkeypatch):
    stop_event = asyncio.Event()
    logger = _LoggerStub()

    class _FakeJobManager:
        backend = None

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def _connect(self) -> object:
            raise RuntimeError("slo gauges leaked /tmp/jobs-slo.db sk-live-slo")

    async def _stop_wait(_stop_event: asyncio.Event, _timeout: float) -> None:
        stop_event.set()

    monkeypatch.setenv("JOBS_SLO_ENABLE", "true")
    monkeypatch.setattr(service, "logger", logger)
    monkeypatch.setattr(service, "JobManager", _FakeJobManager)
    monkeypatch.setattr(service, "_wait_for_stop_or_timeout", _stop_wait)
    monkeypatch.setattr(jobs_metrics, "ensure_jobs_metrics_registered", lambda: None)
    monkeypatch.setattr(metrics_manager, "get_metrics_registry", lambda: object())

    await service.run_jobs_metrics_gauges(stop_event)

    assert logger.debugs == ["Jobs SLO gauges loop error"]
    assert logger.binds[-1] == {"error_type": "RuntimeError"}
    rendered = "\n".join(logger.debugs + logger.infos + logger.warnings)
    assert "/tmp/jobs-slo.db" not in rendered
    assert "sk-live-slo" not in rendered
    assert "slo gauges leaked" not in rendered
