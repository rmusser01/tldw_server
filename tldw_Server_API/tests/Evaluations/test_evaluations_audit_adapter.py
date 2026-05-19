import asyncio

import pytest

from tldw_Server_API.app.core.Audit.unified_audit_service import AuditEventType, UnifiedAuditService
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
import tldw_Server_API.app.core.Evaluations.audit_adapter as eval_adapter
from tldw_Server_API.app.core.Evaluations.audit_adapter import (
    MandatoryAuditWriteError,
    log_evaluation_created,
    log_run_started_async,
    _in_test_mode,
    _parse_cache_size,
    shutdown_evaluations_audit_services,
)
from tldw_Server_API.app.core.config import settings


@pytest.mark.asyncio
async def test_evaluation_created_threadpool_fallback(tmp_path, monkeypatch):
    monkeypatch.setitem(settings, "USER_DB_BASE_DIR", str(tmp_path))
    user_id = 9090
    eval_id = "eval-sync-1"

    await asyncio.to_thread(
        log_evaluation_created,
        user_id=str(user_id),
        eval_id=eval_id,
        name="Threadpool Eval",
        eval_type="unit",
    )

    db_path = DatabasePaths.get_audit_db_path(user_id)
    svc = UnifiedAuditService(db_path=str(db_path))
    await svc.initialize()
    try:
        events = await svc.query_events(user_id=str(user_id))
        match = next(
            (
                e
                for e in events
                if e.get("event_type") == AuditEventType.DATA_WRITE.value
                and e.get("resource_id") == eval_id
                and e.get("action") == "evaluation_create"
            ),
            None,
        )
        assert match is not None, "Threadpool evaluation_create event not found"
    finally:
        await svc.stop()
        await shutdown_evaluations_audit_services()


@pytest.mark.asyncio
async def test_evaluation_adapter_wraps_mandatory_failures(monkeypatch):
    async def _boom(_user_id):
        raise RuntimeError("audit boom")

    monkeypatch.setattr(eval_adapter, "get_or_create_audit_service_for_user_id_optional", _boom)

    with pytest.raises(MandatoryAuditWriteError) as exc_info:
        await log_run_started_async(
            user_id="user-x",
            eval_id="eval-fail",
            run_id="run-fail",
            target_model="unit",
        )

    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert "audit boom" in str(exc_info.value.__cause__)


def test_evaluations_local_shutdown_helper_stops_loop(monkeypatch):
    calls = {"local": 0}

    def _local_shutdown() -> None:
        calls["local"] += 1

    monkeypatch.setattr(eval_adapter, "_stop_sync_loop", _local_shutdown)

    eval_adapter.shutdown_local_evaluations_audit_loop()

    assert calls == {"local": 1}


def test_evaluations_audit_cache_size_clamped(monkeypatch):
    monkeypatch.setenv("EVALUATIONS_AUDIT_MAX_CACHED_SERVICES", "0")
    assert _parse_cache_size("EVALUATIONS_AUDIT_MAX_CACHED_SERVICES", 20) == 1


def test_evaluations_audit_test_mode_parsing(monkeypatch):
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setenv("TEST_MODE", "0")
    monkeypatch.setenv("TLDW_TEST_MODE", "false")
    assert _in_test_mode() is False
    monkeypatch.setenv("TEST_MODE", "true")
    assert _in_test_mode() is True
