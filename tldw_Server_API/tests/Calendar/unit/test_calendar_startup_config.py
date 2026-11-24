"""Calendar opt-in flags retain lifecycle route policy in the Calendar CI shard."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.services import startup_content_jobs_pollers as startup
from tldw_Server_API.app.services.lifecycle_worker_specs import WorkerLifecycleContext

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("name,flag", [
    ("calendar_sync_jobs_task", "CALENDAR_SYNC_JOBS_WORKER_ENABLED"),
    ("calendar_sync_scheduler_task", "CALENDAR_SYNC_SCHEDULER_ENABLED"),
])
@pytest.mark.parametrize("raw,flag_enabled", [
    (None, False), ("", False), ("   ", False), ("invalid", False), ("false", False),
    ("1", True), (" TRUE ", True), ("yes", True), ("y", True), ("on", True), ("enabled", True),
])
@pytest.mark.parametrize("route_allowed,test_mode", [(True, False), (False, False), (True, True)])
def test_calendar_flags_require_route_admission(monkeypatch, name, flag, raw, flag_enabled, route_allowed, test_mode):
    if raw is None:
        monkeypatch.delenv(flag, raising=False)
    else:
        monkeypatch.setenv(flag, raw)
    calls = []

    def route_enabled(key, **kwargs):
        calls.append((key, kwargs))
        return route_allowed

    context = WorkerLifecycleContext(
        app=SimpleNamespace(), settings={}, test_mode=test_mode, route_enabled=route_enabled,
        logger=None, startup_guard_exceptions=(), import_exceptions=(),
    )
    spec = next(spec for spec in startup.provide_content_jobs_worker_specs() if spec.name == name)
    assert spec.enabled(context) is (flag_enabled and route_allowed)
    assert calls == ([("calendar", {"default_stable": False})] if flag_enabled else [])
    # These explicit flags have never depended on the sidecar ownership default.
    assert spec.enabled(replace(context, sidecar_mode=True)) is (flag_enabled and route_allowed)


@pytest.mark.parametrize("name,field", [
    ("calendar_sync_jobs_task", "jobs_worker_enabled"),
    ("calendar_sync_scheduler_task", "scheduler_enabled"),
])
def test_calendar_predicates_consume_typed_settings_live(monkeypatch, name, field):
    from tldw_Server_API.app.core.config_sections import calendar

    settings = calendar.CalendarSyncStartupConfig(jobs_worker_enabled=False, scheduler_enabled=False)
    monkeypatch.setattr(startup, "load_calendar_sync_startup_config", lambda: settings)
    context = WorkerLifecycleContext(
        app=SimpleNamespace(), settings={}, test_mode=True, route_enabled=lambda *_a, **_kw: True,
        logger=None, startup_guard_exceptions=(), import_exceptions=(),
    )
    spec = next(spec for spec in startup.provide_content_jobs_worker_specs() if spec.name == name)
    assert spec.enabled(context) is False
    settings = replace(settings, **{field: True})
    assert spec.enabled(context) is True
