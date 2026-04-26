from __future__ import annotations

import importlib
import sys

import pytest


pytestmark = pytest.mark.unit


def _import_shutdown_recurring_schedulers():
    sys.modules.pop("tldw_Server_API.app.services.shutdown_recurring_schedulers", None)
    return importlib.import_module("tldw_Server_API.app.services.shutdown_recurring_schedulers")


@pytest.mark.asyncio
async def test_stop_recurring_schedulers_runs_helpers_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shutdown_recurring = _import_shutdown_recurring_schedulers()
    calls: list[tuple[str, object | None]] = []

    async def _fake_workflows(task):
        calls.append(("workflows", task))

    async def _fake_reading_digest(task):
        calls.append(("reading-digest", task))

    async def _fake_admin_backup(task):
        calls.append(("admin-backup", task))

    async def _fake_companion_reflection(task):
        calls.append(("companion-reflection", task))

    async def _fake_reminders(task):
        calls.append(("reminders", task))

    async def _fake_connectors_sync(task):
        calls.append(("connectors-sync", task))

    monkeypatch.setattr(shutdown_recurring, "_stop_workflows_scheduler", _fake_workflows)
    monkeypatch.setattr(shutdown_recurring, "_stop_reading_digest_scheduler", _fake_reading_digest)
    monkeypatch.setattr(shutdown_recurring, "_stop_admin_backup_scheduler", _fake_admin_backup)
    monkeypatch.setattr(shutdown_recurring, "_stop_companion_reflection_scheduler", _fake_companion_reflection)
    monkeypatch.setattr(shutdown_recurring, "_stop_reminders_scheduler", _fake_reminders)
    monkeypatch.setattr(shutdown_recurring, "_stop_connectors_sync_scheduler", _fake_connectors_sync)

    await shutdown_recurring.stop_recurring_schedulers(
        workflows_sched_task="workflows-task",
        reading_digest_sched_task="reading-digest-task",
        admin_backup_sched_task="admin-backup-task",
        companion_reflection_sched_task="companion-reflection-task",
        reminders_sched_task=None,
        connectors_sync_sched_task="connectors-sync-task",
    )

    assert calls == [
        ("workflows", "workflows-task"),
        ("reading-digest", "reading-digest-task"),
        ("admin-backup", "admin-backup-task"),
        ("companion-reflection", "companion-reflection-task"),
        ("reminders", None),
        ("connectors-sync", "connectors-sync-task"),
    ]


@pytest.mark.asyncio
async def test_stop_workflows_scheduler_skips_none_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shutdown_recurring = _import_shutdown_recurring_schedulers()
    called = False

    async def _fake_stop(_task):
        nonlocal called
        called = True

    monkeypatch.setattr(shutdown_recurring, "_stop_workflows_scheduler_service", _fake_stop)

    await shutdown_recurring._stop_workflows_scheduler(None)

    assert called is False


@pytest.mark.asyncio
async def test_stop_reminders_scheduler_forwards_none_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shutdown_recurring = _import_shutdown_recurring_schedulers()
    calls: list[object | None] = []

    async def _fake_stop(task):
        calls.append(task)

    monkeypatch.setattr(shutdown_recurring, "_stop_reminders_scheduler_service", _fake_stop)

    await shutdown_recurring._stop_reminders_scheduler(None)

    assert calls == [None]


@pytest.mark.asyncio
async def test_stop_companion_reflection_scheduler_cancels_task_on_guard_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shutdown_recurring = _import_shutdown_recurring_schedulers()

    class _FakeTask:
        def __init__(self) -> None:
            self.cancelled = False

        def cancel(self) -> None:
            self.cancelled = True

    async def _failing_stop(_task):
        raise RuntimeError("boom")

    task = _FakeTask()
    monkeypatch.setattr(
        shutdown_recurring,
        "_stop_companion_reflection_scheduler_service",
        _failing_stop,
    )

    await shutdown_recurring._stop_companion_reflection_scheduler(task)

    assert task.cancelled is True


@pytest.mark.asyncio
async def test_stop_workflows_scheduler_logs_cancel_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shutdown_recurring = _import_shutdown_recurring_schedulers()
    warning_messages: list[str] = []

    class _FakeTask:
        def cancel(self) -> None:
            raise RuntimeError("cancel boom")

    async def _failing_stop(_task):
        raise RuntimeError("stop boom")

    monkeypatch.setattr(
        shutdown_recurring,
        "_stop_workflows_scheduler_service",
        _failing_stop,
    )
    monkeypatch.setattr(
        shutdown_recurring.logger,
        "warning",
        lambda message, *args, **kwargs: warning_messages.append(str(message)),
    )

    await shutdown_recurring._stop_workflows_scheduler(_FakeTask())

    assert any("Workflow scheduler shutdown failed" in message for message in warning_messages)
