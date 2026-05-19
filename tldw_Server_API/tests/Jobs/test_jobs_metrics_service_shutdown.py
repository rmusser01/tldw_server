import asyncio

import pytest

from tldw_Server_API.app.services import jobs_metrics_service


pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_run_jobs_metrics_reconcile_stop_event_interrupts_interval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stop_event = asyncio.Event()

    class _FakeService:
        def __init__(self) -> None:
            self.interval = 60.0

        def reconcile_once(self) -> int:
            stop_event.set()
            return 1

    monkeypatch.setenv("JOBS_METRICS_RECONCILE_ENABLE", "1")
    monkeypatch.setattr(jobs_metrics_service, "JobsMetricsService", _FakeService)

    task = asyncio.create_task(jobs_metrics_service.run_jobs_metrics_reconcile(stop_event))
    await asyncio.wait_for(task, timeout=0.2)

    assert stop_event.is_set() is True
