from __future__ import annotations

import importlib
import sys
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.unit


def _import_shutdown_cleanup_timed_segments():
    sys.modules.pop("tldw_Server_API.app.services.shutdown_cleanup_timed_segments", None)
    return importlib.import_module("tldw_Server_API.app.services.shutdown_cleanup_timed_segments")


@pytest.mark.asyncio
async def test_shutdown_cleanup_timed_segments_runs_steps_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cleanup_tail = _import_shutdown_cleanup_timed_segments()
    calls: list[str] = []
    logged_messages: list[str] = []
    app = SimpleNamespace(state=SimpleNamespace())

    async def _record_auth_db_pool(*, db_pool, in_pytest_for_db_pool_shutdown, guard_exceptions):
        assert db_pool == "db-pool"
        assert in_pytest_for_db_pool_shutdown is True
        assert guard_exceptions == (RuntimeError,)
        calls.append("auth-db-pool")

    async def _record_resource_cleanup(
        *,
        app,
        session_manager,
        heavy_startup_handles,
        in_pytest_for_tts_shutdown,
        import_exceptions,
        startup_guard_exceptions,
    ):
        assert session_manager == "session-manager"
        assert heavy_startup_handles == "heavy-handles"
        assert in_pytest_for_tts_shutdown is False
        assert import_exceptions == (LookupError,)
        assert startup_guard_exceptions == (RuntimeError,)
        calls.append("resource-cleanup")

    async def _record_evaluations_resources(*, import_exceptions):
        assert import_exceptions == (LookupError,)
        calls.append("evaluations")

    async def _record_unified_audit_services(*, startup_guard_exceptions, import_exceptions):
        assert startup_guard_exceptions == (RuntimeError,)
        assert import_exceptions == (LookupError,)
        calls.append("unified-audit")

    async def _record_executor_resources(*, startup_guard_exceptions, import_exceptions):
        assert startup_guard_exceptions == (RuntimeError,)
        assert import_exceptions == (LookupError,)
        calls.append("executor")

    async def _record_cpu_pools(*, guard_exceptions):
        assert guard_exceptions == (RuntimeError,)
        calls.append("cpu-pools")

    async def _record_telemetry_services(*, import_exceptions) -> None:
        assert import_exceptions == (LookupError,)
        calls.append("telemetry")

    @contextmanager
    def _fake_timed_shutdown_segment(seen_app, segment_name):
        assert seen_app is app
        calls.append(f"enter:{segment_name}")
        try:
            yield
        finally:
            calls.append(f"exit:{segment_name}")

    monkeypatch.setattr(cleanup_tail, "_shutdown_auth_db_pool", _record_auth_db_pool)
    monkeypatch.setattr(cleanup_tail, "_shutdown_resource_cleanup", _record_resource_cleanup)
    monkeypatch.setattr(cleanup_tail, "_shutdown_evaluations_resources", _record_evaluations_resources)
    monkeypatch.setattr(cleanup_tail, "_shutdown_unified_audit_services", _record_unified_audit_services)
    monkeypatch.setattr(cleanup_tail, "_shutdown_executor_resources", _record_executor_resources)
    monkeypatch.setattr(cleanup_tail, "_shutdown_cpu_pools", _record_cpu_pools)
    monkeypatch.setattr(cleanup_tail, "_shutdown_telemetry_services", _record_telemetry_services)
    monkeypatch.setattr(
        cleanup_tail.logger,
        "info",
        lambda message, *args, **kwargs: logged_messages.append(str(message)),
    )

    handles = await cleanup_tail.shutdown_cleanup_timed_segments(
        app=app,
        db_pool="db-pool",
        session_manager="session-manager",
        heavy_startup_handles="heavy-handles",
        in_pytest_for_db_pool_shutdown=True,
        in_pytest_for_tts_shutdown=False,
        import_exceptions=(LookupError,),
        startup_guard_exceptions=(RuntimeError,),
        timed_shutdown_segment=_fake_timed_shutdown_segment,
    )

    assert handles == cleanup_tail.CleanupTimedShutdownHandles()
    assert logged_messages == [
        "App Shutdown: Cleaning up resources...",
        "App Shutdown: Audit services cleanup handled by dependency injection",
    ]
    assert calls == [
        "auth-db-pool",
        "resource-cleanup",
        "enter:evaluations_pool_shutdown",
        "evaluations",
        "exit:evaluations_pool_shutdown",
        "enter:unified_audit_and_executor_shutdown",
        "unified-audit",
        "executor",
        "cpu-pools",
        "exit:unified_audit_and_executor_shutdown",
        "enter:telemetry_shutdown",
        "telemetry",
        "exit:telemetry_shutdown",
    ]
