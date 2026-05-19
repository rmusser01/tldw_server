import asyncio
import os
from pathlib import Path

import pytest

import tldw_Server_API.app.core.Embeddings.audit_adapter as emb_adapter
from tldw_Server_API.app.core.Embeddings.audit_adapter import (
    emit_security_violation_async,
    emit_model_evicted_async,
    emit_memory_limit_exceeded_async,
    log_security_violation,
    _in_test_mode,
    _parse_cache_size,
    shutdown_audit_adapter_services,
)
from tldw_Server_API.app.core.Audit.unified_audit_service import UnifiedAuditService, AuditEventType
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths


@pytest.mark.asyncio
async def test_security_violation_maps_to_unified_per_user(tmp_path):
    # Use a test user id and ensure DB path
    user_id = 4242
    db_path = DatabasePaths.get_audit_db_path(user_id)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # Emit a security event
    await emit_security_violation_async(
        user_id=str(user_id),
        action="request_signature_invalid",
        metadata={"reason": "bad_sig"},
    )

    # Query events from the per-user audit DB
    svc = UnifiedAuditService(db_path=str(db_path))
    await svc.initialize()
    events = await svc.query_events(user_id=str(user_id))
    assert events, "Expected at least one audit event"
    # Find our event
    match = next(
        (
            e
            for e in events
            if e.get("event_type") == AuditEventType.SECURITY_VIOLATION.value
            and e.get("action") == "request_signature_invalid"
        ),
        None,
    )
    assert match is not None, "Security violation event not found"


@pytest.mark.asyncio
async def test_model_evicted_records_data_delete(tmp_path):
    # Default audit DB file path
    default_db = Path("./Databases/unified_audit.db")
    default_db.parent.mkdir(parents=True, exist_ok=True)

    model_id = "model-test-evict"
    await emit_model_evicted_async(model_id=model_id, memory_usage_gb=1.25, reason="lru_eviction")

    svc = UnifiedAuditService(db_path=str(default_db))
    await svc.initialize()
    events = await svc.query_events()
    assert events, "Expected events in default audit DB"
    match = next(
        (
            e
            for e in events
            if e.get("event_type") == AuditEventType.DATA_DELETE.value
            and e.get("resource_type") == "embedding_model"
            and e.get("resource_id") == model_id
            and e.get("action") == "model_evicted"
        ),
        None,
    )
    assert match is not None, "Model eviction event not found"


@pytest.mark.asyncio
async def test_memory_limit_exceeded_records_system_error(tmp_path):
    default_db = Path("./Databases/unified_audit.db")
    default_db.parent.mkdir(parents=True, exist_ok=True)

    model_id = "model-oom"
    await emit_memory_limit_exceeded_async(
        model_id=model_id,
        memory_usage_gb=2.5,
        current_usage_gb=6.0,
        limit_gb=8.0,
    )

    svc = UnifiedAuditService(db_path=str(default_db))
    await svc.initialize()
    events = await svc.query_events()
    assert events, "Expected events in default audit DB"
    match = next(
        (
            e
            for e in events
            if e.get("event_type") == AuditEventType.SYSTEM_ERROR.value
            and e.get("resource_type") == "embedding_model"
            and e.get("resource_id") == model_id
            and e.get("action") == "embeddings_memory_limit_exceeded"
        ),
        None,
    )
    assert match is not None, "Memory limit exceeded event not found"


@pytest.mark.asyncio
async def test_security_violation_threadpool_fallback(tmp_path, monkeypatch):
    from tldw_Server_API.app.core.config import settings

    monkeypatch.setitem(settings, "USER_DB_BASE_DIR", str(tmp_path))
    user_id = 8080
    action = "threadpool_security_violation"

    await asyncio.to_thread(
        log_security_violation,
        user_id=str(user_id),
        action=action,
        metadata={"reason": "sync_call"},
    )

    db_path = DatabasePaths.get_audit_db_path(user_id)
    svc = UnifiedAuditService(db_path=str(db_path))
    await svc.initialize()
    try:
        events = await svc.query_events(user_id=str(user_id))
        match = next(
            (e for e in events if e.get("event_type") == AuditEventType.SECURITY_VIOLATION.value and e.get("action") == action),
            None,
        )
        assert match is not None, "Threadpool security violation event not found"
    finally:
        await svc.stop()
        await shutdown_audit_adapter_services()


@pytest.mark.asyncio
async def test_embeddings_adapter_propagates_failures(monkeypatch):
    async def _boom(_user_id):
        raise RuntimeError("audit boom")

    monkeypatch.setattr(emb_adapter, "get_or_create_audit_service_for_user_id_optional", _boom)

    try:
        with pytest.raises(RuntimeError):
            await asyncio.to_thread(
                log_security_violation,
                user_id="user-x",
                action="fail-path",
                metadata={"reason": "boom"},
            )
    finally:
        await shutdown_audit_adapter_services()


def test_embeddings_local_shutdown_helper_stops_loop(monkeypatch):
    calls = {"local": 0}

    def _local_shutdown() -> None:
        calls["local"] += 1

    monkeypatch.setattr(emb_adapter, "_stop_sync_loop", _local_shutdown)

    emb_adapter.shutdown_local_audit_adapter_loop()

    assert calls == {"local": 1}


def test_embeddings_atexit_only_stops_local_loop(monkeypatch):
    calls = {"global": 0, "local": 0}

    async def _global_shutdown() -> None:
        calls["global"] += 1

    def _local_shutdown() -> None:
        calls["local"] += 1

    monkeypatch.setattr(emb_adapter, "shutdown_all_audit_services", _global_shutdown, raising=False)
    monkeypatch.setattr(emb_adapter, "_stop_sync_loop", _local_shutdown)
    monkeypatch.setattr(emb_adapter.logger, "disable", lambda *_args, **_kwargs: None)

    emb_adapter._shutdown_on_exit()

    assert calls == {"global": 0, "local": 1}


def test_embeddings_audit_cache_size_clamped(monkeypatch):
    monkeypatch.setenv("EMBEDDINGS_AUDIT_MAX_CACHED_SERVICES", "0")
    assert _parse_cache_size("EMBEDDINGS_AUDIT_MAX_CACHED_SERVICES", 20) == 1


def test_embeddings_audit_test_mode_parsing(monkeypatch):
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setenv("TEST_MODE", "0")
    monkeypatch.setenv("TLDW_TEST_MODE", "false")
    assert _in_test_mode() is False
    monkeypatch.setenv("TEST_MODE", "true")
    assert _in_test_mode() is True
