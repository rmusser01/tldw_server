"""Tests for the jobs notification bridge enable/disable logic."""
from __future__ import annotations

import asyncio

import pytest

from tldw_Server_API.app.services.jobs_notifications_service import (
    start_jobs_notifications_service,
)

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Ensure bridge env vars are unset by default."""
    monkeypatch.delenv("JOBS_NOTIFICATIONS_BRIDGE_ENABLED", raising=False)
    monkeypatch.delenv("JOBS_NOTIFICATIONS_BRIDGE_DISABLED", raising=False)


def _run(coro):
    return asyncio.run(coro)


class _FakeService:
    """Replaces JobsNotificationsService to avoid real DB/task creation."""
    started = False

    def __init__(self, **_kwargs):
        pass

    async def run_forever(self):
        _FakeService.started = True


@pytest.fixture(autouse=True)
def _patch_service(monkeypatch):
    _FakeService.started = False
    monkeypatch.setattr(
        "tldw_Server_API.app.services.jobs_notifications_service.JobsNotificationsService",
        _FakeService,
    )


def test_bridge_on_by_default(monkeypatch):
    """Bridge starts when no env vars are set (default ON)."""
    result = _run(start_jobs_notifications_service())
    assert result is not None


def test_bridge_disabled_via_opt_out(monkeypatch):
    """Bridge skipped when JOBS_NOTIFICATIONS_BRIDGE_DISABLED=true."""
    monkeypatch.setenv("JOBS_NOTIFICATIONS_BRIDGE_DISABLED", "true")
    result = _run(start_jobs_notifications_service())
    assert result is None


def test_bridge_disabled_via_legacy_false(monkeypatch):
    """Bridge skipped when legacy JOBS_NOTIFICATIONS_BRIDGE_ENABLED=false."""
    monkeypatch.setenv("JOBS_NOTIFICATIONS_BRIDGE_ENABLED", "false")
    result = _run(start_jobs_notifications_service())
    assert result is None


def test_bridge_enabled_via_legacy_true(monkeypatch):
    """Bridge starts when legacy JOBS_NOTIFICATIONS_BRIDGE_ENABLED=true."""
    monkeypatch.setenv("JOBS_NOTIFICATIONS_BRIDGE_ENABLED", "true")
    result = _run(start_jobs_notifications_service())
    assert result is not None


def test_disabled_takes_precedence_over_enabled(monkeypatch):
    """DISABLED=true wins even when ENABLED=true is also set."""
    monkeypatch.setenv("JOBS_NOTIFICATIONS_BRIDGE_ENABLED", "true")
    monkeypatch.setenv("JOBS_NOTIFICATIONS_BRIDGE_DISABLED", "true")
    result = _run(start_jobs_notifications_service())
    assert result is None
