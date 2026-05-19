from __future__ import annotations

import asyncio

import pytest

from tldw_Server_API.app.services import jobs_notifications_service as service

pytestmark = pytest.mark.unit

_LEAK = "jobs bridge loop leaked token at /tmp/jobs-secret-token"


class _LoopFailingJobsNotificationsService(service.JobsNotificationsService):
    def __init__(self, *, stop_event: asyncio.Event) -> None:
        super().__init__(poll_interval_seconds=0.01)
        self._stop_event = stop_event

    async def run_once(self) -> dict[str, int | bool]:
        self._stop_event.set()
        raise RuntimeError(_LEAK)


class _ProcessEventFailingJobsNotificationsService(service.JobsNotificationsService):
    def _fetch_events_after(self, *, after_id: int, limit: int):  # noqa: ARG002
        return [
            {
                "id": 99,
                "event_type": "job.failed",
                "owner_user_id": "1",
            }
        ]

    async def process_event(self, event):
        raise RuntimeError(_LEAK)


class _BridgeState:
    last_event_id = 0


class _StateDb:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):  # noqa: ANN001, ARG002
        return False

    def try_claim_notification_bridge_lease(self, **_kwargs):
        return True

    def get_notification_bridge_state(self, **_kwargs):
        return _BridgeState()

    def update_notification_bridge_state(self, **_kwargs):
        return None


def _assert_safe_warning(rendered: str) -> None:
    assert "RuntimeError" in rendered
    assert "jobs bridge loop leaked token" not in rendered
    assert "/tmp/jobs-secret-token" not in rendered
    assert "exc_info" not in rendered


@pytest.mark.asyncio
async def test_jobs_notifications_process_event_warning_omits_raw_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(service.CollectionsDatabase, "for_user", lambda **_kwargs: _StateDb())
    monkeypatch.setattr(service.JobManager, "set_rls_context", lambda **_kwargs: None)
    monkeypatch.setattr(service.JobManager, "clear_rls_context", lambda: None)

    bridge = _ProcessEventFailingJobsNotificationsService()
    messages: list[str] = []
    sink_id = service.logger.add(lambda message: messages.append(str(message)), level="WARNING")
    try:
        summary = await bridge.run_once()
    finally:
        service.logger.remove(sink_id)

    assert summary["failed"] == 1
    _assert_safe_warning("\n".join(messages))


@pytest.mark.asyncio
async def test_jobs_notifications_bridge_loop_warning_omits_raw_exception() -> None:
    stop_event = asyncio.Event()
    bridge = _LoopFailingJobsNotificationsService(stop_event=stop_event)
    messages: list[str] = []
    sink_id = service.logger.add(lambda message: messages.append(str(message)), level="WARNING")
    try:
        await bridge.run_forever(stop_event=stop_event)
    finally:
        service.logger.remove(sink_id)

    rendered = "\n".join(messages)
    _assert_safe_warning(rendered)
