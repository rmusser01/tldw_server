from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.tests.helpers.app_main_state import (
    clear_app_main,
    import_app_main,
    restore_app_main,
    snapshot_app_main,
)


pytestmark = pytest.mark.unit


def test_notifications_services_start_when_enabled(monkeypatch):
    monkeypatch.setenv("MINIMAL_TEST_APP", "1")
    monkeypatch.setenv("ULTRA_MINIMAL_APP", "1")
    monkeypatch.setenv("ROUTES_DISABLE", "media,audio,audio-websocket")
    monkeypatch.setenv("REMINDERS_SCHEDULER_ENABLED", "true")
    monkeypatch.setenv("REMINDER_JOBS_WORKER_ENABLED", "true")
    monkeypatch.setenv("JOBS_NOTIFICATIONS_BRIDGE_ENABLED", "true")
    monkeypatch.setenv("TLDW_WORKERS_SIDECAR_MODE", "0")

    called = {
        "reminders_scheduler_start": 0,
        "reminders_scheduler_stop": 0,
        "reminder_jobs_worker_start": 0,
        "jobs_notifications_bridge_start": 0,
    }

    async def _fake_start_reminders_scheduler(*_args, **_kwargs):
        called["reminders_scheduler_start"] += 1
        return asyncio.create_task(asyncio.sleep(0), name="fake_reminders_scheduler")

    async def _fake_stop_reminders_scheduler(*_args, **_kwargs):
        called["reminders_scheduler_stop"] += 1

    async def _fake_start_reminder_jobs_worker(*_args, **_kwargs):
        called["reminder_jobs_worker_start"] += 1
        return None

    async def _fake_run_reminder_jobs_worker(stop_event):
        called["reminder_jobs_worker_start"] += 1
        await stop_event.wait()

    async def _fake_start_jobs_notifications_service(*_args, **_kwargs):
        called["jobs_notifications_bridge_start"] += 1
        return asyncio.create_task(asyncio.sleep(0), name="fake_jobs_notifications_bridge")

    import tldw_Server_API.app.services.jobs_notifications_service as jobs_notifications_service
    import tldw_Server_API.app.services.reminder_jobs_worker as reminder_jobs_worker
    import tldw_Server_API.app.services.reminders_scheduler as reminders_scheduler

    previous_main = snapshot_app_main()
    clear_app_main()
    try:
        main_mod = import_app_main()
        app = main_mod.app

        monkeypatch.setattr(reminders_scheduler, "start_reminders_scheduler", _fake_start_reminders_scheduler)
        monkeypatch.setattr(reminders_scheduler, "stop_reminders_scheduler", _fake_stop_reminders_scheduler)
        monkeypatch.setattr(reminder_jobs_worker, "start_reminder_jobs_worker", _fake_start_reminder_jobs_worker)
        monkeypatch.setattr(reminder_jobs_worker, "run_reminder_jobs_worker", _fake_run_reminder_jobs_worker)
        monkeypatch.setattr(
            jobs_notifications_service,
            "start_jobs_notifications_service",
            _fake_start_jobs_notifications_service,
        )

        with TestClient(app):
            pass
    finally:
        restore_app_main(previous_main)

    assert called["reminders_scheduler_start"] == 1
    assert called["reminders_scheduler_stop"] == 1
    assert called["reminder_jobs_worker_start"] == 1
    assert called["jobs_notifications_bridge_start"] == 1
