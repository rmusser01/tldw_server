from __future__ import annotations

import importlib
import sys

import pytest


pytestmark = pytest.mark.unit


def _import_shutdown_evaluations_resources():
    sys.modules.pop("tldw_Server_API.app.services.shutdown_evaluations_resources", None)
    return importlib.import_module("tldw_Server_API.app.services.shutdown_evaluations_resources")


@pytest.mark.asyncio
async def test_shutdown_evaluations_resources_runs_steps_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shutdown_evals = _import_shutdown_evaluations_resources()
    calls: list[str] = []

    def _fake_shutdown_pool():
        calls.append("pool")

    def _fake_shutdown_webhooks():
        calls.append("webhooks")

    monkeypatch.setattr(shutdown_evals, "_shutdown_evaluations_pool_service", _fake_shutdown_pool)
    monkeypatch.setattr(shutdown_evals, "_shutdown_evaluations_webhook_manager_service", _fake_shutdown_webhooks)

    await shutdown_evals.shutdown_evaluations_resources(import_exceptions=(RuntimeError,))

    assert calls == ["pool", "webhooks"]


@pytest.mark.asyncio
async def test_shutdown_evaluations_pool_handles_guard_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shutdown_evals = _import_shutdown_evaluations_resources()

    def _failing_shutdown_pool():
        raise RuntimeError("boom")

    monkeypatch.setattr(shutdown_evals, "_shutdown_evaluations_pool_service", _failing_shutdown_pool)

    await shutdown_evals._shutdown_evaluations_pool(import_exceptions=(RuntimeError,))


@pytest.mark.asyncio
async def test_shutdown_evaluations_webhook_manager_handles_guard_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shutdown_evals = _import_shutdown_evaluations_resources()

    def _failing_shutdown_webhooks():
        raise RuntimeError("boom")

    monkeypatch.setattr(
        shutdown_evals,
        "_shutdown_evaluations_webhook_manager_service",
        _failing_shutdown_webhooks,
    )

    await shutdown_evals._shutdown_evaluations_webhook_manager(import_exceptions=(RuntimeError,))
