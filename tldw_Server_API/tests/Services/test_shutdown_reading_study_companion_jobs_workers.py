from __future__ import annotations

import importlib
import sys

import pytest


pytestmark = pytest.mark.unit


def _import_shutdown_reading_study_companion_jobs_workers():
    sys.modules.pop("tldw_Server_API.app.services.shutdown_reading_study_companion_jobs_workers", None)
    return importlib.import_module("tldw_Server_API.app.services.shutdown_reading_study_companion_jobs_workers")


class _FakeStopEvent:
    def __init__(self) -> None:
        self.is_set = False

    def set(self) -> None:
        self.is_set = True


class _FakeTask:
    def __init__(self) -> None:
        self.cancelled = False

    def cancel(self) -> None:
        self.cancelled = True


@pytest.mark.asyncio
async def test_shutdown_reading_study_companion_jobs_workers_runs_helpers_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shutdown_workers = _import_shutdown_reading_study_companion_jobs_workers()
    calls: list[str] = []

    async def _record_reading(**kwargs):
        del kwargs
        calls.append("reading")

    async def _record_study_pack(**kwargs):
        del kwargs
        calls.append("study-pack")

    async def _record_study_suggestions(**kwargs):
        del kwargs
        calls.append("study-suggestions")

    async def _record_companion(**kwargs):
        del kwargs
        calls.append("companion")

    monkeypatch.setattr(shutdown_workers, "_shutdown_reading_digest_jobs_worker", _record_reading)
    monkeypatch.setattr(shutdown_workers, "_shutdown_study_pack_jobs_worker", _record_study_pack)
    monkeypatch.setattr(shutdown_workers, "_shutdown_study_suggestions_jobs_worker", _record_study_suggestions)
    monkeypatch.setattr(shutdown_workers, "_shutdown_companion_reflection_jobs_worker", _record_companion)

    handles = await shutdown_workers.shutdown_reading_study_companion_jobs_workers(
        reading_digest_jobs_task="reading-task",
        reading_digest_jobs_stop_event="reading-stop",
        study_pack_jobs_task="study-pack-task",
        study_pack_jobs_stop_event="study-pack-stop",
        study_suggestions_jobs_task="study-suggestions-task",
        study_suggestions_jobs_stop_event="study-suggestions-stop",
        companion_reflection_jobs_task="companion-task",
        companion_reflection_jobs_stop_event="companion-stop",
        should_run_late_stop=lambda *args, **kwargs: True,
        guard_exceptions=(RuntimeError,),
    )

    assert calls == ["reading", "study-pack", "study-suggestions", "companion"]
    assert handles.reading_digest_jobs_task == "reading-task"
    assert handles.reading_digest_jobs_stop_event == "reading-stop"
    assert handles.study_pack_jobs_task == "study-pack-task"
    assert handles.study_pack_jobs_stop_event == "study-pack-stop"
    assert handles.study_suggestions_jobs_task == "study-suggestions-task"
    assert handles.study_suggestions_jobs_stop_event == "study-suggestions-stop"
    assert handles.companion_reflection_jobs_task == "companion-task"
    assert handles.companion_reflection_jobs_stop_event == "companion-stop"


@pytest.mark.asyncio
async def test_shutdown_reading_digest_jobs_worker_stops_via_stop_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shutdown_workers = _import_shutdown_reading_study_companion_jobs_workers()
    waits: list[tuple[object, float]] = []
    task = _FakeTask()
    stop_event = _FakeStopEvent()

    async def _fake_wait(task_obj, *, timeout):
        waits.append((task_obj, timeout))

    monkeypatch.setattr(shutdown_workers, "_wait_for_task", _fake_wait)

    await shutdown_workers._shutdown_reading_digest_jobs_worker(
        task=task,
        stop_event=stop_event,
        should_run_late_stop=lambda name, current_task: (name, current_task) == ("reading_digest_jobs_task", task),
        guard_exceptions=(RuntimeError,),
    )

    assert stop_event.is_set is True
    assert waits == [(task, 5.0)]
    assert task.cancelled is False


@pytest.mark.asyncio
async def test_shutdown_study_pack_jobs_worker_skips_when_late_stop_says_false() -> None:
    shutdown_workers = _import_shutdown_reading_study_companion_jobs_workers()
    task = _FakeTask()
    stop_event = _FakeStopEvent()

    await shutdown_workers._shutdown_study_pack_jobs_worker(
        task=task,
        stop_event=stop_event,
        should_run_late_stop=lambda name, current_task: False,
        guard_exceptions=(RuntimeError,),
    )

    assert stop_event.is_set is False
    assert task.cancelled is False


@pytest.mark.asyncio
async def test_shutdown_study_suggestions_jobs_worker_cancels_on_guard_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shutdown_workers = _import_shutdown_reading_study_companion_jobs_workers()
    task = _FakeTask()
    stop_event = _FakeStopEvent()

    async def _failing_wait(_task, *, timeout):
        del timeout
        raise RuntimeError("boom")

    monkeypatch.setattr(shutdown_workers, "_wait_for_task", _failing_wait)

    await shutdown_workers._shutdown_study_suggestions_jobs_worker(
        task=task,
        stop_event=stop_event,
        should_run_late_stop=lambda name, current_task: (
            name,
            current_task,
        ) == ("study_suggestions_jobs_task", task),
        guard_exceptions=(RuntimeError,),
    )

    assert stop_event.is_set is True
    assert task.cancelled is True


@pytest.mark.asyncio
async def test_shutdown_companion_reflection_jobs_worker_cancels_without_stop_event() -> None:
    shutdown_workers = _import_shutdown_reading_study_companion_jobs_workers()
    task = _FakeTask()

    await shutdown_workers._shutdown_companion_reflection_jobs_worker(
        task=task,
        stop_event=None,
        should_run_late_stop=lambda name, current_task: (
            name,
            current_task,
        ) == ("companion_reflection_jobs_task", task),
        guard_exceptions=(RuntimeError,),
    )

    assert task.cancelled is True
