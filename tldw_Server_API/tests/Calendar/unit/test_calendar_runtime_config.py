"""Pin Calendar's environment semantics at the runtime consumers."""

from __future__ import annotations

import asyncio
from contextlib import nullcontext
from dataclasses import asdict

import pytest

from tldw_Server_API.app.core.Calendar import calendar_sync_worker as worker
from tldw_Server_API.app.core.Calendar.errors import CalendarValidationError
from tldw_Server_API.app.core.Calendar.secret_store import CalendarSecretStore
from tldw_Server_API.app.services import calendar_sync_scheduler as scheduler

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "environment,expected",
    [
        ({}, ("calendar-sync-worker-4242", 120, 10, 0)),
        ({"JOBS_LEASE_SECONDS": "37"}, ("calendar-sync-worker-4242", 37, 10, 0)),
        ({"JOBS_LEASE_SECONDS": ""}, ("calendar-sync-worker-4242", 120, 10, 0)),
        ({"JOBS_LEASE_SECONDS": "37", "CALENDAR_SYNC_LEASE_SECONDS": ""},
         ("calendar-sync-worker-4242", 120, 10, 0)),
        ({"JOBS_LEASE_SECONDS": "bad", "CALENDAR_SYNC_LEASE_SECONDS": " 23 ",
          "CALENDAR_SYNC_WORKER_ID": " custom ", "CALENDAR_SYNC_RENEW_THRESHOLD_SECONDS": " 7 ",
          "CALENDAR_SYNC_RENEW_JITTER_SECONDS": " 2 "}, ("custom", 23, 7, 2)),
        ({"CALENDAR_SYNC_WORKER_ID": "", "CALENDAR_SYNC_RENEW_THRESHOLD_SECONDS": "",
          "CALENDAR_SYNC_RENEW_JITTER_SECONDS": ""}, ("calendar-sync-worker-4242", 120, 10, 0)),
        ({"CALENDAR_SYNC_WORKER_ID": "   ", "CALENDAR_SYNC_LEASE_SECONDS": "-1",
          "CALENDAR_SYNC_RENEW_THRESHOLD_SECONDS": "0", "CALENDAR_SYNC_RENEW_JITTER_SECONDS": "-2"},
         ("", -1, 0, -2)),
    ],
)
async def test_worker_passes_existing_settings_to_sdk_and_honors_stop(monkeypatch, environment, expected):
    for key in ("CALENDAR_SYNC_WORKER_ID", "CALENDAR_SYNC_LEASE_SECONDS", "JOBS_LEASE_SECONDS",
                "CALENDAR_SYNC_RENEW_THRESHOLD_SECONDS", "CALENDAR_SYNC_RENEW_JITTER_SECONDS"):
        monkeypatch.delenv(key, raising=False)
    for key, value in environment.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(worker.os, "getpid", lambda: 4242)
    observed = []
    handlers = []
    scopes = []
    stopped = asyncio.Event()
    stop_event = asyncio.Event()
    stop_event.set()

    class SDK:
        def __init__(self, manager, config):
            observed.append(asdict(config))

        async def run(self, *, handler):
            handlers.append(handler)
            await stopped.wait()

        def stop(self):
            stopped.set()

    class Manager:
        def scoped_rls_context(self, **kwargs):
            scopes.append(kwargs)
            return nullcontext()

    monkeypatch.setattr(worker, "WorkerSDK", SDK)
    monkeypatch.setattr(worker, "JobManager", Manager)
    await worker.run_calendar_sync_worker(stop_event)
    assert len(observed) == 1
    assert handlers == [worker.handle_calendar_sync_job]
    assert scopes == [{"is_admin": True, "domain_allowlist": "calendar", "owner_user_id": None}]
    config = observed[0]
    assert (config["domain"], config["queue"]) == ("calendar", "default")
    assert tuple(config[key] for key in (
        "worker_id", "lease_seconds", "renew_threshold_seconds", "renew_jitter_seconds",
    )) == expected
    await asyncio.sleep(0)
    assert not any(task.get_name() == "calendar_sync_worker_stop_waiter" for task in asyncio.all_tasks())


@pytest.mark.asyncio
@pytest.mark.parametrize("key", [
    "CALENDAR_SYNC_LEASE_SECONDS", "JOBS_LEASE_SECONDS",
    "CALENDAR_SYNC_RENEW_THRESHOLD_SECONDS", "CALENDAR_SYNC_RENEW_JITTER_SECONDS",
])
@pytest.mark.parametrize("value", ["   ", "invalid", "1.5"])
async def test_worker_keeps_invalid_integer_failure(monkeypatch, key, value):
    monkeypatch.delenv("CALENDAR_SYNC_LEASE_SECONDS", raising=False)
    monkeypatch.setenv(key, value)
    with pytest.raises(ValueError):
        await worker.run_calendar_sync_worker()


@pytest.mark.parametrize("raw,expected", [
    (None, 60.0), ("", 60.0), (" ", 60.0), ("bad", 60.0),
    ("-1", 5.0), ("0", 5.0), ("4.9", 5.0), (" 7.5 ", 7.5),
    ("nan", 5.0), ("inf", float("inf")),
])
def test_scheduler_keeps_interval_parsing(monkeypatch, raw, expected):
    if raw is None:
        monkeypatch.delenv("CALENDAR_SYNC_SCHEDULER_INTERVAL_SECONDS", raising=False)
    else:
        monkeypatch.setenv("CALENDAR_SYNC_SCHEDULER_INTERVAL_SECONDS", raw)
    assert scheduler._scheduler_interval_seconds() == expected


@pytest.mark.parametrize("raw", [None, "", "   "])
def test_secret_key_unset_or_blank_fails_closed(monkeypatch, raw):
    if raw is None:
        monkeypatch.delenv("CALENDAR_SECRET_ENCRYPTION_KEY", raising=False)
    else:
        monkeypatch.setenv("CALENDAR_SECRET_ENCRYPTION_KEY", raw)
    with pytest.raises(CalendarValidationError, match="CALENDAR_SECRET_ENCRYPTION_KEY"):
        CalendarSecretStore._required_encryption_key()


def test_secret_key_is_stripped_and_not_cached(monkeypatch):
    monkeypatch.setenv("CALENDAR_SECRET_ENCRYPTION_KEY", " first-test-value ")
    assert CalendarSecretStore._required_encryption_key() == "first-test-value"
    monkeypatch.setenv("CALENDAR_SECRET_ENCRYPTION_KEY", " second-test-value ")
    assert CalendarSecretStore._required_encryption_key() == "second-test-value"
