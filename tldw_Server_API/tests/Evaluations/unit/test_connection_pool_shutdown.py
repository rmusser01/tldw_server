"""Focused tests for connection pool shutdown behavior."""

import time
import threading

import pytest

from tldw_Server_API.app.core.Evaluations.connection_pool import ConnectionPool


class _ShutdownAwareMaintenanceTask:
    """Test double that validates shutdown ordering."""

    def __init__(self, wakeup_event: threading.Event):
        self._wakeup_event = wakeup_event
        self._alive = True
        self.join_timeout = None

    def is_alive(self) -> bool:
        return self._alive

    def join(self, timeout: float | None = None) -> None:
        if not self._wakeup_event.is_set():
            pytest.fail("shutdown must wake maintenance before joining")
        self.join_timeout = timeout
        self._alive = False


class _TimeoutRecordingMaintenanceTask:
    """Test double that records the shutdown join timeout."""

    def __init__(self):
        self._alive = True
        self.join_timeout = None

    def is_alive(self) -> bool:
        return self._alive

    def join(self, timeout: float | None = None) -> None:
        self.join_timeout = timeout
        self._alive = False


class _StuckMaintenanceTask:
    """Test double that remains alive after the bounded join."""

    def __init__(self):
        self.join_timeouts: list[float | None] = []

    def is_alive(self) -> bool:
        return True

    def join(self, timeout: float | None = None) -> None:
        self.join_timeouts.append(timeout)


class _StopsAfterFinalJoinMaintenanceTask:
    """Test double that only exits after the post-cleanup join."""

    def __init__(self):
        self.join_timeouts: list[float | None] = []
        self._alive = True

    def is_alive(self) -> bool:
        return self._alive

    def join(self, timeout: float | None = None) -> None:
        self.join_timeouts.append(timeout)
        if len(self.join_timeouts) >= 2:
            self._alive = False


class _DiesDuringCleanupMaintenanceTask:
    """Test double that stops before the post-cleanup retry join."""

    def __init__(self):
        self.join_timeouts: list[float | None] = []
        self._alive_states = iter([True, True, False, False])

    def is_alive(self) -> bool:
        return next(self._alive_states)

    def join(self, timeout: float | None = None) -> None:
        self.join_timeouts.append(timeout)


@pytest.mark.unit
class TestConnectionPoolShutdown:
    """Shutdown behavior should wake maintenance and avoid long joins."""

    def test_shutdown_sets_maintenance_wakeup_before_join(self, temp_db_path):
        pool = ConnectionPool(db_path=str(temp_db_path), enable_monitoring=False)
        wakeup_event = threading.Event()
        maintenance_task = _ShutdownAwareMaintenanceTask(wakeup_event)
        pool._maintenance_shutdown_event = wakeup_event
        pool._maintenance_task = maintenance_task

        pool.shutdown()

        if not wakeup_event.is_set():
            pytest.fail("maintenance wakeup event was not set during shutdown")
        if maintenance_task.join_timeout != 1.0:
            pytest.fail(f"expected join timeout 1.0, got {maintenance_task.join_timeout!r}")

    def test_shutdown_uses_short_join_timeout_for_maintenance(self, temp_db_path):
        pool = ConnectionPool(db_path=str(temp_db_path), enable_monitoring=False)
        maintenance_task = _TimeoutRecordingMaintenanceTask()
        pool._maintenance_task = maintenance_task

        pool.shutdown()

        if maintenance_task.join_timeout is None:
            pytest.fail("maintenance thread was not joined during shutdown")
        if not pytest.approx(1.0, abs=0.05) == maintenance_task.join_timeout:
            pytest.fail(f"expected join timeout around 1.0, got {maintenance_task.join_timeout!r}")

    def test_shutdown_interrupts_real_maintenance_wait_promptly(self, temp_db_path):
        pool = ConnectionPool(db_path=str(temp_db_path), enable_monitoring=False)
        maintenance_started = threading.Event()
        original_perform_maintenance = pool._perform_maintenance

        def _recording_perform_maintenance() -> None:
            maintenance_started.set()
            original_perform_maintenance()

        pool._perform_maintenance = _recording_perform_maintenance  # type: ignore[method-assign]
        pool._start_maintenance()

        if not maintenance_started.wait(timeout=2.0):
            pytest.fail("maintenance worker did not start")

        start = time.perf_counter()
        pool.shutdown()
        shutdown_duration = time.perf_counter() - start

        if pool._maintenance_task is None or pool._maintenance_task.is_alive():
            pytest.fail("maintenance worker did not stop during shutdown")
        if shutdown_duration >= 1.5:
            pytest.fail(f"shutdown did not interrupt maintenance wait promptly: {shutdown_duration:.3f}s")

    def test_shutdown_warns_if_maintenance_thread_is_still_alive_after_join(
        self,
        monkeypatch: pytest.MonkeyPatch,
        temp_db_path,
    ):
        pool = ConnectionPool(db_path=str(temp_db_path), enable_monitoring=False)
        maintenance_task = _StuckMaintenanceTask()
        infos: list[tuple[object, ...]] = []
        warnings: list[tuple[object, ...]] = []
        pool._maintenance_task = maintenance_task

        monkeypatch.setattr(
            "tldw_Server_API.app.core.Evaluations.connection_pool.logger.info",
            lambda *args, **_kwargs: infos.append(args),
        )
        monkeypatch.setattr(
            "tldw_Server_API.app.core.Evaluations.connection_pool.logger.warning",
            lambda *args, **_kwargs: warnings.append(args),
        )

        pool.shutdown()

        if maintenance_task.join_timeouts != [1.0, 0.1]:
            pytest.fail(f"expected join timeouts [1.0, 0.1], got {maintenance_task.join_timeouts!r}")
        if not warnings:
            pytest.fail("expected shutdown to warn when maintenance thread remains alive")
        if any(args and args[0] == "Connection pool shutdown complete" for args in infos):
            pytest.fail("shutdown should not claim clean completion while maintenance thread is still alive")

    def test_shutdown_logs_breadcrumb_before_cleanup_when_maintenance_is_still_alive(
        self,
        monkeypatch: pytest.MonkeyPatch,
        temp_db_path,
    ):
        pool = ConnectionPool(db_path=str(temp_db_path), enable_monitoring=False)
        maintenance_task = _StuckMaintenanceTask()
        infos: list[tuple[object, ...]] = []
        pool._maintenance_task = maintenance_task

        monkeypatch.setattr(
            "tldw_Server_API.app.core.Evaluations.connection_pool.logger.info",
            lambda *args, **_kwargs: infos.append(args),
        )

        pool.shutdown()

        if not any(
            args and args[0] == "Connection pool maintenance thread still alive after 1.0s; proceeding with shutdown cleanup and will recheck"
            for args in infos
        ):
            pytest.fail("expected shutdown to log an intermediate breadcrumb before cleanup when maintenance is still alive")

    def test_shutdown_rechecks_maintenance_thread_after_cleanup_before_clean_completion(
        self,
        monkeypatch: pytest.MonkeyPatch,
        temp_db_path,
    ):
        pool = ConnectionPool(db_path=str(temp_db_path), enable_monitoring=False)
        maintenance_task = _StopsAfterFinalJoinMaintenanceTask()
        infos: list[tuple[object, ...]] = []
        warnings: list[tuple[object, ...]] = []
        pool._maintenance_task = maintenance_task

        monkeypatch.setattr(
            "tldw_Server_API.app.core.Evaluations.connection_pool.logger.info",
            lambda *args, **_kwargs: infos.append(args),
        )
        monkeypatch.setattr(
            "tldw_Server_API.app.core.Evaluations.connection_pool.logger.warning",
            lambda *args, **_kwargs: warnings.append(args),
        )

        pool.shutdown()

        if maintenance_task.join_timeouts != [1.0, 0.1]:
            pytest.fail(f"expected join timeouts [1.0, 0.1], got {maintenance_task.join_timeouts!r}")
        if warnings:
            pytest.fail(f"expected clean completion without warnings, got {warnings!r}")
        if not any(args and args[0] == "Connection pool shutdown complete" for args in infos):
            pytest.fail("expected shutdown to report clean completion after the maintenance thread exits")

    def test_shutdown_reports_clean_completion_when_maintenance_dies_during_cleanup(
        self,
        monkeypatch: pytest.MonkeyPatch,
        temp_db_path,
    ):
        pool = ConnectionPool(db_path=str(temp_db_path), enable_monitoring=False)
        maintenance_task = _DiesDuringCleanupMaintenanceTask()
        infos: list[tuple[object, ...]] = []
        warnings: list[tuple[object, ...]] = []
        pool._maintenance_task = maintenance_task

        monkeypatch.setattr(
            "tldw_Server_API.app.core.Evaluations.connection_pool.logger.info",
            lambda *args, **_kwargs: infos.append(args),
        )
        monkeypatch.setattr(
            "tldw_Server_API.app.core.Evaluations.connection_pool.logger.warning",
            lambda *args, **_kwargs: warnings.append(args),
        )

        pool.shutdown()

        if maintenance_task.join_timeouts != [1.0]:
            pytest.fail(f"expected only the initial bounded join, got {maintenance_task.join_timeouts!r}")
        if warnings:
            pytest.fail(f"expected clean completion without warnings, got {warnings!r}")
        if not any(args and args[0] == "Connection pool shutdown complete" for args in infos):
            pytest.fail("expected shutdown to report clean completion after the maintenance thread dies during cleanup")
