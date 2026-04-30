# tldw_Server_API/tests/kanban/test_kanban_deps_rate_limit.py
"""
Unit tests for Kanban dependency helpers.

Focused on:
- In-memory rate limiting behavior and cleanup
- Closing cached KanbanDB instances on shutdown
"""

from collections import deque
from types import SimpleNamespace

import pytest
from cachetools import LRUCache
from fastapi import HTTPException

from tldw_Server_API.app.api.v1.API_Deps import kanban_deps
from tldw_Server_API.app.core.DB_Management.Kanban_DB import (
    ConflictError,
    InputError,
    KanbanDBError,
    NotFoundError,
)


@pytest.fixture(autouse=True)
def _reset_rate_limit_state(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(kanban_deps, "_rate_limit_windows", {})
    monkeypatch.setattr(kanban_deps, "_rate_limit_last_cleanup_ts", 0.0)
    monkeypatch.setattr(kanban_deps, "_RATE_LIMIT_CLEANUP_INTERVAL_SECONDS", 0.0)
    yield


class TestKanbanRateLimiting:
    def test_blocks_after_limit(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setitem(kanban_deps.KANBAN_RATE_LIMITS, "test.action", 2)
        monkeypatch.setattr(kanban_deps.time, "time", lambda: 1000.0)

        allowed, _ = kanban_deps.check_kanban_rate_limit(user_id=1, action="test.action")
        assert allowed is True

        allowed, _ = kanban_deps.check_kanban_rate_limit(user_id=1, action="test.action")
        assert allowed is True

        allowed, info = kanban_deps.check_kanban_rate_limit(user_id=1, action="test.action")
        assert allowed is False
        assert info["limit"] == 2

    def test_resets_after_window(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setitem(kanban_deps.KANBAN_RATE_LIMITS, "test.action", 1)

        monkeypatch.setattr(kanban_deps.time, "time", lambda: 1000.0)
        allowed, _ = kanban_deps.check_kanban_rate_limit(user_id=1, action="test.action")
        assert allowed is True

        allowed, _ = kanban_deps.check_kanban_rate_limit(user_id=1, action="test.action")
        assert allowed is False

        monkeypatch.setattr(kanban_deps.time, "time", lambda: 1061.0)
        allowed, _ = kanban_deps.check_kanban_rate_limit(user_id=1, action="test.action")
        assert allowed is True

    def test_cleanup_removes_stale_keys(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(kanban_deps.time, "time", lambda: 1000.0)

        with kanban_deps._rate_limit_lock:
            kanban_deps._rate_limit_windows["stale:action"] = deque([0.0])

        allowed, _ = kanban_deps.check_kanban_rate_limit(user_id=1, action="test.action")
        assert allowed is True
        assert "stale:action" not in kanban_deps._rate_limit_windows

    async def test_dependency_bypasses_rate_limit_during_pytest_runtime(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setitem(kanban_deps.KANBAN_RATE_LIMITS, "test.action", 0)
        monkeypatch.setenv("PYTEST_CURRENT_TEST", "tests::test_dependency_bypasses_rate_limit_during_pytest_runtime")

        dependency = kanban_deps.kanban_rate_limit("test.action")
        current_user = SimpleNamespace(id=123)

        try:
            await dependency(current_user=current_user)
        except HTTPException as exc:  # pragma: no cover - this is the pre-fix failure mode
            pytest.fail(f"Rate limit should be bypassed during explicit pytest runtime, got {exc.status_code}")


class TestKanbanDbCacheShutdown:
    def test_close_all_kanban_db_instances_calls_close(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(kanban_deps, "_kanban_db_instances", LRUCache(maxsize=10))
        monkeypatch.setattr(kanban_deps, "_kanban_db_health_checks", {})

        class DummyDB:
            def __init__(self) -> None:
                self.closed = False

            def close(self) -> None:

                self.closed = True

        dummy = DummyDB()

        with kanban_deps._kanban_db_lock:
            kanban_deps._kanban_db_instances["kanban::1"] = dummy
            kanban_deps._kanban_db_health_checks["kanban::1"] = 123.0

        kanban_deps.close_all_kanban_db_instances()
        assert dummy.closed is True
        assert len(kanban_deps._kanban_db_instances) == 0
        assert kanban_deps._kanban_db_health_checks == {}


class TestKanbanDbErrorMapping:
    @pytest.mark.parametrize(
        ("exc", "expected_status", "expected_detail"),
        [
            (NotFoundError("missing board"), 404, "missing board"),
            (InputError("invalid card"), 400, "invalid card"),
            (ConflictError("stale card"), 409, "stale card"),
            (KanbanDBError("backend failed"), 500, "Kanban operation failed"),
        ],
    )
    def test_handle_kanban_db_error_maps_known_contracts(
        self,
        exc: Exception,
        expected_status: int,
        expected_detail: str,
    ):
        mapped = kanban_deps.handle_kanban_db_error(exc)

        assert mapped.status_code == expected_status
        assert mapped.detail == expected_detail

    def test_handle_kanban_db_error_preserves_unexpected_contract(self):
        mapped = kanban_deps.handle_kanban_db_error(RuntimeError("boom"))

        assert mapped.status_code == 500
        assert mapped.detail == "An unexpected error occurred"


class TestKanbanDbInitializationErrorMapping:
    @staticmethod
    def _patch_init_failure(monkeypatch: pytest.MonkeyPatch, exc: Exception) -> None:
        monkeypatch.setattr(kanban_deps, "_kanban_db_instances", LRUCache(maxsize=10))
        monkeypatch.setattr(kanban_deps, "_kanban_db_health_checks", {})
        monkeypatch.setattr(kanban_deps, "_KANBAN_INIT_TIMEOUT_SECS", 1.0)
        kanban_deps.shutdown_kanban_executor(wait=True)

        def fail_create(user_id: int):
            raise exc

        monkeypatch.setattr(kanban_deps, "_create_kanban_db", fail_create)

    @pytest.mark.asyncio
    async def test_get_or_init_db_instance_maps_base_kanban_db_error(self, monkeypatch: pytest.MonkeyPatch):
        self._patch_init_failure(monkeypatch, KanbanDBError("backend exploded"))

        try:
            with pytest.raises(HTTPException) as exc_info:
                await kanban_deps._get_or_init_db_instance(321)
        finally:
            kanban_deps.shutdown_kanban_executor(wait=True)

        assert exc_info.value.status_code == 500
        assert exc_info.value.detail == "Kanban DB unavailable"

    @pytest.mark.asyncio
    async def test_get_or_init_db_instance_keeps_input_init_errors_as_500(self, monkeypatch: pytest.MonkeyPatch):
        self._patch_init_failure(monkeypatch, InputError("invalid bootstrap path"))

        try:
            with pytest.raises(HTTPException) as exc_info:
                await kanban_deps._get_or_init_db_instance(322)
        finally:
            kanban_deps.shutdown_kanban_executor(wait=True)

        assert exc_info.value.status_code == 500
        assert exc_info.value.detail == "invalid bootstrap path"

    @pytest.mark.asyncio
    async def test_get_or_init_db_instance_keeps_conflict_init_errors_as_500(self, monkeypatch: pytest.MonkeyPatch):
        self._patch_init_failure(monkeypatch, ConflictError("duplicate bootstrap state"))

        try:
            with pytest.raises(HTTPException) as exc_info:
                await kanban_deps._get_or_init_db_instance(323)
        finally:
            kanban_deps.shutdown_kanban_executor(wait=True)

        assert exc_info.value.status_code == 500
        assert exc_info.value.detail == "Kanban DB unavailable"
