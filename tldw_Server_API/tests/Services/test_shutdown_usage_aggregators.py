from __future__ import annotations

import importlib
import sys

import pytest


pytestmark = pytest.mark.unit


def _import_shutdown_usage_aggregators():
    sys.modules.pop("tldw_Server_API.app.services.shutdown_usage_aggregators", None)
    return importlib.import_module("tldw_Server_API.app.services.shutdown_usage_aggregators")


@pytest.mark.asyncio
async def test_stop_usage_aggregators_stops_both_in_order_and_clears_task_handles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shutdown_usage = _import_shutdown_usage_aggregators()
    calls: list[tuple[str, object]] = []

    async def _fake_stop_usage(task):
        calls.append(("usage", task))

    async def _fake_stop_llm(task):
        calls.append(("llm", task))

    monkeypatch.setattr(shutdown_usage, "_stop_usage_aggregator_service", _fake_stop_usage)
    monkeypatch.setattr(shutdown_usage, "_stop_llm_usage_aggregator_service", _fake_stop_llm)

    handles = await shutdown_usage.stop_usage_aggregators(
        coordinated_legacy_component_names=set(),
        usage_task="usage-task",
        llm_usage_task="llm-task",
        guard_exceptions=(RuntimeError,),
    )

    assert calls == [("usage", "usage-task"), ("llm", "llm-task")]
    assert handles.usage_task is None
    assert handles.llm_usage_task is None


@pytest.mark.asyncio
async def test_stop_usage_aggregators_skips_coordinated_components(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shutdown_usage = _import_shutdown_usage_aggregators()
    called = False

    async def _fake_stop_usage(_task):
        nonlocal called
        called = True

    monkeypatch.setattr(shutdown_usage, "_stop_usage_aggregator_service", _fake_stop_usage)

    handles = await shutdown_usage.stop_usage_aggregators(
        coordinated_legacy_component_names={"usage_aggregator"},
        usage_task="usage-task",
        llm_usage_task=None,
        guard_exceptions=(RuntimeError,),
    )

    assert called is False
    assert handles.usage_task == "usage-task"


@pytest.mark.asyncio
async def test_stop_usage_aggregators_cancels_usage_task_on_guard_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shutdown_usage = _import_shutdown_usage_aggregators()

    class _FakeTask:
        def __init__(self) -> None:
            self.cancelled = False

        def cancel(self) -> None:
            self.cancelled = True

    async def _failing_stop(_task):
        raise RuntimeError("boom")

    task = _FakeTask()
    monkeypatch.setattr(shutdown_usage, "_stop_usage_aggregator_service", _failing_stop)

    handles = await shutdown_usage.stop_usage_aggregators(
        coordinated_legacy_component_names=set(),
        usage_task=task,
        llm_usage_task=None,
        guard_exceptions=(RuntimeError,),
    )

    assert task.cancelled is True
    assert handles.usage_task is None


@pytest.mark.asyncio
async def test_stop_usage_aggregators_cancels_llm_usage_task_on_guard_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shutdown_usage = _import_shutdown_usage_aggregators()

    class _FakeTask:
        def __init__(self) -> None:
            self.cancelled = False

        def cancel(self) -> None:
            self.cancelled = True

    async def _failing_stop(_task):
        raise RuntimeError("boom")

    task = _FakeTask()
    monkeypatch.setattr(shutdown_usage, "_stop_llm_usage_aggregator_service", _failing_stop)

    handles = await shutdown_usage.stop_usage_aggregators(
        coordinated_legacy_component_names=set(),
        usage_task=None,
        llm_usage_task=task,
        guard_exceptions=(RuntimeError,),
    )

    assert task.cancelled is True
    assert handles.llm_usage_task is None
