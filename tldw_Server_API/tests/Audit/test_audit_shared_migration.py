import importlib
from datetime import datetime, timezone

import aiosqlite
import pytest

from tldw_Server_API.app.core.DB_Management import sqlite_policy
from tldw_Server_API.app.core.Audit import audit_shared_migration as audit_migration
from tldw_Server_API.app.core.Audit.audit_shared_migration import (
    discover_audit_sources,
    migrate_to_shared_audit_db,
)
from tldw_Server_API.app.core.Audit.unified_audit_service import (
    UnifiedAuditService,
    AuditContext,
    AuditEvent,
    AuditEventCategory,
    AuditEventType,
)


@pytest.mark.asyncio
async def test_shared_audit_migration_uses_shared_sqlite_policy_helper(tmp_path):
    audit_module = importlib.import_module(
        "tldw_Server_API.app.core.Audit.audit_shared_migration"
    )
    calls: list[dict[str, object]] = []

    async def fake_configure(conn, **kwargs):
        calls.append(kwargs)

    class _FakeService:
        def __init__(self, *args, **kwargs) -> None:
            self._event_columns = ["event_id"]
            self._event_insert_sql = "INSERT INTO audit_events (event_id) VALUES (?)"

        async def initialize(self, start_background_tasks: bool = False) -> None:
            return None

        async def stop(self) -> None:
            return None

    async def fake_ensure_checkpoint_table(shared_db):
        return None

    async def fake_migrate_source(shared_db, source, **kwargs):
        return audit_module.AuditMigrationCounts(source=source)

    source_path = tmp_path / "user_dbs" / "101" / "audit" / "unified_audit.db"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.touch()

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(sqlite_policy, "configure_sqlite_connection_async", fake_configure)
        audit_module = importlib.reload(audit_module)
        mp.setattr(audit_module, "UnifiedAuditService", _FakeService)
        mp.setattr(audit_module, "_ensure_checkpoint_table", fake_ensure_checkpoint_table)
        mp.setattr(audit_module, "_migrate_source", fake_migrate_source)
        mp.setattr(
            audit_module,
            "discover_audit_sources",
            lambda **kwargs: [
                audit_module.AuditMigrationSource(
                    path=source_path.resolve(),
                    tenant_id="101",
                    label="user:101",
                )
            ],
        )
        await audit_module.migrate_to_shared_audit_db(
            shared_db_path=tmp_path / "Databases" / "audit_shared.db",
            user_db_base_dir=tmp_path / "user_dbs",
            chunk_size=10,
        )

    importlib.reload(audit_module)

    assert calls == [{
        "use_wal": True,
        "synchronous": "NORMAL",
        "busy_timeout_ms": 5000,
        "foreign_keys": True,
        "temp_store": "MEMORY",
    }]


@pytest.mark.asyncio
async def test_migrate_to_shared_db(tmp_path):
    user_base = tmp_path / "user_dbs"
    user_id = "101"
    user_db_path = user_base / user_id / "audit" / "unified_audit.db"
    shared_db_path = tmp_path / "Databases" / "audit_shared.db"
    default_db_path = tmp_path / "Databases" / "unified_audit.db"
    user_db_path.parent.mkdir(parents=True, exist_ok=True)
    default_db_path.parent.mkdir(parents=True, exist_ok=True)

    svc_user = UnifiedAuditService(
        db_path=str(user_db_path),
        storage_mode="per_user",
        enable_pii_detection=False,
        enable_risk_scoring=False,
        buffer_size=1,
        flush_interval=0.1,
    )
    await svc_user.initialize(start_background_tasks=False)
    await svc_user.log_event(
        event_type=AuditEventType.DATA_READ,
        context=AuditContext(user_id=user_id),
        resource_type="doc",
        resource_id="u1",
    )
    await svc_user.flush()
    await svc_user.stop()

    svc_default = UnifiedAuditService(
        db_path=str(default_db_path),
        storage_mode="per_user",
        enable_pii_detection=False,
        enable_risk_scoring=False,
        buffer_size=1,
        flush_interval=0.1,
    )
    await svc_default.initialize(start_background_tasks=False)
    await svc_default.log_event(event_type=AuditEventType.SYSTEM_START)
    await svc_default.log_event(
        event_type=AuditEventType.DATA_READ,
        resource_type="doc",
        resource_id="default-missing",
    )
    await svc_default.flush()
    await svc_default.stop()

    report = await migrate_to_shared_audit_db(
        shared_db_path=shared_db_path,
        user_db_base_dir=user_base,
        default_db_path=default_db_path,
        system_tenant_id="system",
        chunk_size=100,
    )
    assert report.total_events_inserted >= 2
    assert report.total_stats_read >= 2
    assert report.total_stats_inserted >= 2
    assert report.total_stats_skipped == 0

    async with aiosqlite.connect(shared_db_path) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT tenant_user_id FROM audit_events") as cur:
            rows = await cur.fetchall()
        tenants = {row["tenant_user_id"] for row in rows}
        assert user_id in tenants
        assert "system" in tenants
        assert "unidentified_user" in tenants

        async with db.execute("SELECT tenant_user_id FROM audit_daily_stats") as cur:
            stats_rows = await cur.fetchall()
        stats_tenants = {row["tenant_user_id"] for row in stats_rows}
        assert user_id in stats_tenants
        assert "system" in stats_tenants
        assert "unidentified_user" in stats_tenants

    report2 = await migrate_to_shared_audit_db(
        shared_db_path=shared_db_path,
        user_db_base_dir=user_base,
        default_db_path=default_db_path,
        system_tenant_id="system",
        chunk_size=100,
    )
    assert report2.total_events_inserted == 0


def test_discover_sources_with_subpath(tmp_path, monkeypatch):
    base = tmp_path / "user_dbs"
    subpath = "nested_users"
    user_id = "404"
    user_db_path = base / subpath / user_id / "audit" / "unified_audit.db"
    user_db_path.parent.mkdir(parents=True, exist_ok=True)
    user_db_path.touch()

    from tldw_Server_API.app.core.config import settings

    monkeypatch.setitem(settings, "AUDIT_ETL_USER_SUBPATH", subpath)
    sources = discover_audit_sources(user_db_base_dir=base)
    assert any(src.path == user_db_path.resolve() for src in sources)


@pytest.mark.asyncio
async def test_migration_resume_checkpoint(tmp_path):
    user_base = tmp_path / "user_dbs"
    user_id = "202"
    user_db_path = user_base / user_id / "audit" / "unified_audit.db"
    shared_db_path = tmp_path / "Databases" / "audit_shared.db"
    user_db_path.parent.mkdir(parents=True, exist_ok=True)

    svc_user = UnifiedAuditService(
        db_path=str(user_db_path),
        storage_mode="per_user",
        enable_pii_detection=False,
        enable_risk_scoring=False,
        buffer_size=1,
        flush_interval=0.1,
    )
    await svc_user.initialize(start_background_tasks=False)
    for idx in range(3):
        await svc_user.log_event(
            event_type=AuditEventType.DATA_READ,
            context=AuditContext(user_id=user_id),
            resource_type="doc",
            resource_id=f"doc-{idx}",
        )
    await svc_user.flush()
    await svc_user.stop()

    missing_default = tmp_path / "Databases" / "missing_unified_audit.db"
    report = await migrate_to_shared_audit_db(
        shared_db_path=shared_db_path,
        user_db_base_dir=user_base,
        default_db_path=missing_default,
        system_tenant_id="system",
        chunk_size=1,
    )
    source_counts = next(c for c in report.sources if c.source.label == f"user:{user_id}")
    assert source_counts.events_read == 3
    assert source_counts.stats_read == 3
    assert source_counts.stats_inserted == 3
    assert source_counts.stats_skipped == 0

    svc_user = UnifiedAuditService(
        db_path=str(user_db_path),
        storage_mode="per_user",
        enable_pii_detection=False,
        enable_risk_scoring=False,
        buffer_size=1,
        flush_interval=0.1,
    )
    await svc_user.initialize(start_background_tasks=False)
    await svc_user.log_event(
        event_type=AuditEventType.DATA_READ,
        context=AuditContext(user_id=user_id),
        resource_type="doc",
        resource_id="doc-new",
    )
    await svc_user.flush()
    await svc_user.stop()

    report2 = await migrate_to_shared_audit_db(
        shared_db_path=shared_db_path,
        user_db_base_dir=user_base,
        default_db_path=missing_default,
        system_tenant_id="system",
        chunk_size=1,
    )
    source_counts2 = next(c for c in report2.sources if c.source.label == f"user:{user_id}")
    assert source_counts2.events_read == 1
    assert source_counts2.stats_read == 1
    assert source_counts2.stats_inserted == 1
    assert source_counts2.stats_skipped == 0


@pytest.mark.asyncio
async def test_migration_checkpoint_handles_empty_timestamp_resume(tmp_path, monkeypatch):
    user_base = tmp_path / "user_dbs"
    user_id = "909"
    user_db_path = user_base / user_id / "audit" / "unified_audit.db"
    shared_db_path = tmp_path / "Databases" / "audit_shared.db"
    user_db_path.parent.mkdir(parents=True, exist_ok=True)

    svc_user = UnifiedAuditService(
        db_path=str(user_db_path),
        storage_mode="per_user",
        enable_pii_detection=False,
        enable_risk_scoring=False,
        buffer_size=1,
        flush_interval=0.1,
    )
    await svc_user.initialize(start_background_tasks=False)
    await svc_user.stop()

    ts_valid = datetime(2020, 1, 1, 0, 0, 0, tzinfo=timezone.utc).isoformat()
    async with aiosqlite.connect(user_db_path) as db:
        await db.execute(
            "INSERT INTO audit_events (event_id, timestamp, category, event_type, severity) "
            "VALUES (?, ?, ?, ?, ?)",
            ("evt-empty", "", "api_call", "api.request", "info"),
        )
        await db.execute(
            "INSERT INTO audit_events (event_id, timestamp, category, event_type, severity) "
            "VALUES (?, ?, ?, ?, ?)",
            ("evt-valid", ts_valid, "api_call", "api.request", "info"),
        )
        await db.commit()

    svc_shared = UnifiedAuditService(
        db_path=str(shared_db_path),
        storage_mode="shared",
        enable_pii_detection=False,
        enable_risk_scoring=False,
        buffer_size=1,
        flush_interval=0.1,
    )
    await svc_shared.initialize(start_background_tasks=False)
    await svc_shared.stop()

    async with aiosqlite.connect(shared_db_path) as shared_db:
        shared_db.row_factory = aiosqlite.Row
        await audit_migration._ensure_checkpoint_table(shared_db)
        await shared_db.commit()

        source = audit_migration.AuditMigrationSource(
            path=user_db_path.resolve(),
            tenant_id=user_id,
            label=f"user:{user_id}",
        )

        orig_commit = shared_db.commit
        call_count = {"count": 0}

        async def commit_with_interrupt():
            call_count["count"] += 1
            if call_count["count"] == 2:
                raise RuntimeError("interrupt")
            await orig_commit()

        monkeypatch.setattr(shared_db, "commit", commit_with_interrupt)
        counts = await audit_migration._migrate_source(
            shared_db,
            source,
            columns=list(svc_shared._event_columns),
            insert_sql=svc_shared._event_insert_sql,
            system_tenant_id="system",
            unidentified_tenant_id="unidentified_user",
            chunk_size=1,
        )
        assert counts.failed is True
        assert counts.events_inserted == 1
        assert counts.events_skipped == 0
        assert counts.stats_inserted == 1
        assert counts.stats_skipped == 0

        async with shared_db.execute("SELECT COUNT(*) AS n FROM audit_events") as cur:
            row = await cur.fetchone()
        assert row["n"] == 1

        monkeypatch.setattr(shared_db, "commit", orig_commit)
        counts2 = await audit_migration._migrate_source(
            shared_db,
            source,
            columns=list(svc_shared._event_columns),
            insert_sql=svc_shared._event_insert_sql,
            system_tenant_id="system",
            unidentified_tenant_id="unidentified_user",
            chunk_size=1,
        )
        assert counts2.events_inserted == 1

        async with shared_db.execute("SELECT event_id FROM audit_events") as cur:
            rows = await cur.fetchall()
        event_ids = {row["event_id"] for row in rows}
        assert {"evt-empty", "evt-valid"} <= event_ids


@pytest.mark.asyncio
async def test_migration_skips_corrupt_source(tmp_path):
    user_base = tmp_path / "user_dbs"
    user_id = "303"
    user_db_path = user_base / user_id / "audit" / "unified_audit.db"
    shared_db_path = tmp_path / "Databases" / "audit_shared.db"
    user_db_path.parent.mkdir(parents=True, exist_ok=True)
    user_db_path.write_text("not a sqlite db", encoding="utf-8")

    missing_default = tmp_path / "Databases" / "missing_unified_audit.db"
    report = await migrate_to_shared_audit_db(
        shared_db_path=shared_db_path,
        user_db_base_dir=user_base,
        default_db_path=missing_default,
        system_tenant_id="system",
        chunk_size=1,
    )
    source_counts = next(c for c in report.sources if c.source.label == f"user:{user_id}")
    assert source_counts.failed is True
    assert report.total_failures == 1


@pytest.mark.asyncio
async def test_migration_preserves_non_numeric_tenant_id(tmp_path):
    user_base = tmp_path / "user_dbs"
    user_id = "550e8400-e29b-41d4-a716-446655440000"
    user_db_path = user_base / user_id / "audit" / "unified_audit.db"
    shared_db_path = tmp_path / "Databases" / "audit_shared.db"
    user_db_path.parent.mkdir(parents=True, exist_ok=True)

    svc_user = UnifiedAuditService(
        db_path=str(user_db_path),
        storage_mode="per_user",
        enable_pii_detection=False,
        enable_risk_scoring=False,
        buffer_size=1,
        flush_interval=0.1,
    )
    await svc_user.initialize(start_background_tasks=False)
    await svc_user.log_event(
        event_type=AuditEventType.DATA_READ,
        context=AuditContext(user_id=user_id),
        resource_type="doc",
        resource_id="uuid-doc",
    )
    await svc_user.flush()
    await svc_user.stop()

    await migrate_to_shared_audit_db(
        shared_db_path=shared_db_path,
        user_db_base_dir=user_base,
        default_db_path=None,
        system_tenant_id="system",
        chunk_size=100,
    )

    async with aiosqlite.connect(shared_db_path) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT DISTINCT tenant_user_id FROM audit_events"
        ) as cur:
            rows = await cur.fetchall()
    tenants = {row["tenant_user_id"] for row in rows}
    assert user_id in tenants


@pytest.mark.asyncio
async def test_migration_keyset_checkpoint_handles_rowid_reuse(tmp_path):
    user_base = tmp_path / "user_dbs"
    user_id = "707"
    user_db_path = user_base / user_id / "audit" / "unified_audit.db"
    shared_db_path = tmp_path / "Databases" / "audit_shared.db"
    user_db_path.parent.mkdir(parents=True, exist_ok=True)

    svc_user = UnifiedAuditService(
        db_path=str(user_db_path),
        storage_mode="per_user",
        enable_pii_detection=False,
        enable_risk_scoring=False,
        buffer_size=1,
        flush_interval=0.1,
    )
    await svc_user.initialize(start_background_tasks=False)
    await svc_user.stop()

    ts1 = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc).isoformat()
    ts2 = datetime(2025, 1, 1, 0, 1, 0, tzinfo=timezone.utc).isoformat()
    async with aiosqlite.connect(user_db_path) as db:
        await db.execute(
            "INSERT INTO audit_events (rowid, event_id, timestamp, category, event_type, severity) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (1, "evt-1", ts1, "api_call", "api.request", "info"),
        )
        await db.commit()

    report1 = await migrate_to_shared_audit_db(
        shared_db_path=shared_db_path,
        user_db_base_dir=user_base,
        default_db_path=None,
        system_tenant_id="system",
        chunk_size=10,
    )
    assert report1.total_events_inserted == 1

    async with aiosqlite.connect(user_db_path) as db:
        await db.execute("DELETE FROM audit_events")
        await db.execute(
            "INSERT INTO audit_events (rowid, event_id, timestamp, category, event_type, severity) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (1, "evt-2", ts2, "api_call", "api.request", "info"),
        )
        await db.commit()
        async with db.execute("SELECT rowid FROM audit_events WHERE event_id = ?", ("evt-2",)) as cur:
            row = await cur.fetchone()
        assert row[0] == 1

    report2 = await migrate_to_shared_audit_db(
        shared_db_path=shared_db_path,
        user_db_base_dir=user_base,
        default_db_path=None,
        system_tenant_id="system",
        chunk_size=10,
    )
    assert report2.total_events_inserted == 1

    async with aiosqlite.connect(shared_db_path) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT event_id FROM audit_events") as cur:
            rows = await cur.fetchall()
    event_ids = {row["event_id"] for row in rows}
    assert {"evt-1", "evt-2"} <= event_ids


@pytest.mark.asyncio
async def test_migration_stats_skipped_tracks_duplicate_events(tmp_path):
    user_base = tmp_path / "user_dbs"
    user_id = "717"
    user_db_path = user_base / user_id / "audit" / "unified_audit.db"
    shared_db_path = tmp_path / "Databases" / "audit_shared.db"
    user_db_path.parent.mkdir(parents=True, exist_ok=True)

    svc_user = UnifiedAuditService(
        db_path=str(user_db_path),
        storage_mode="per_user",
        enable_pii_detection=False,
        enable_risk_scoring=False,
        buffer_size=1,
        flush_interval=0.1,
    )
    await svc_user.initialize(start_background_tasks=False)
    await svc_user.stop()

    ts = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc).isoformat()
    async with aiosqlite.connect(user_db_path) as db:
        await db.execute(
            "INSERT INTO audit_events (event_id, timestamp, category, event_type, severity) "
            "VALUES (?, ?, ?, ?, ?)",
            ("dup-stats-1", ts, "api_call", "api.request", "info"),
        )
        await db.commit()

    svc_shared = UnifiedAuditService(
        db_path=str(shared_db_path),
        storage_mode="shared",
        enable_pii_detection=False,
        enable_risk_scoring=False,
        buffer_size=1,
        flush_interval=0.1,
    )
    await svc_shared.initialize(start_background_tasks=False)
    await svc_shared.stop()

    source = audit_migration.AuditMigrationSource(
        path=user_db_path.resolve(),
        tenant_id=user_id,
        label=f"user:{user_id}",
    )

    async with aiosqlite.connect(shared_db_path) as shared_db:
        shared_db.row_factory = aiosqlite.Row
        await audit_migration._ensure_checkpoint_table(shared_db)
        await shared_db.commit()

        first = await audit_migration._migrate_source(
            shared_db,
            source,
            columns=list(svc_shared._event_columns),
            insert_sql=svc_shared._event_insert_sql,
            system_tenant_id="system",
            unidentified_tenant_id="unidentified_user",
            chunk_size=100,
        )
        assert first.events_read == 1
        assert first.events_inserted == 1
        assert first.events_skipped == 0
        assert first.stats_read == 1
        assert first.stats_inserted == 1
        assert first.stats_skipped == 0

        await audit_migration._save_checkpoint(
            shared_db,
            source.path.resolve(),
            last_rowid=0,
            last_event_id=None,
            last_timestamp=None,
        )
        await shared_db.commit()

        second = await audit_migration._migrate_source(
            shared_db,
            source,
            columns=list(svc_shared._event_columns),
            insert_sql=svc_shared._event_insert_sql,
            system_tenant_id="system",
            unidentified_tenant_id="unidentified_user",
            chunk_size=100,
        )
        assert second.events_read == 1
        assert second.events_inserted == 0
        assert second.events_skipped == 1
        assert second.stats_read == 1
        assert second.stats_inserted == 0
        assert second.stats_skipped == 1


@pytest.mark.asyncio
async def test_migration_updates_stats_incrementally(tmp_path):
    user_base = tmp_path / "user_dbs"
    user_id = "808"
    user_db_path = user_base / user_id / "audit" / "unified_audit.db"
    shared_db_path = tmp_path / "Databases" / "audit_shared.db"
    user_db_path.parent.mkdir(parents=True, exist_ok=True)

    svc_user = UnifiedAuditService(
        db_path=str(user_db_path),
        storage_mode="per_user",
        enable_pii_detection=False,
        enable_risk_scoring=False,
        buffer_size=10,
        flush_interval=0.1,
    )
    await svc_user.initialize(start_background_tasks=False)
    t1 = datetime(2025, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
    ev1 = AuditEvent(
        event_id="inc-1",
        timestamp=t1,
        category=AuditEventCategory.API_CALL,
        event_type=AuditEventType.API_REQUEST,
        context=AuditContext(user_id=user_id),
    )
    async with svc_user.buffer_lock:
        svc_user.event_buffer.append(ev1)
    await svc_user.flush()
    await svc_user.stop()

    report1 = await migrate_to_shared_audit_db(
        shared_db_path=shared_db_path,
        user_db_base_dir=user_base,
        default_db_path=None,
        system_tenant_id="system",
        chunk_size=10,
    )
    source_counts1 = next(c for c in report1.sources if c.source.label == f"user:{user_id}")
    assert source_counts1.stats_read == 1
    assert source_counts1.stats_inserted == 1
    assert source_counts1.stats_skipped == 0

    svc_user = UnifiedAuditService(
        db_path=str(user_db_path),
        storage_mode="per_user",
        enable_pii_detection=False,
        enable_risk_scoring=False,
        buffer_size=10,
        flush_interval=0.1,
    )
    await svc_user.initialize(start_background_tasks=False)
    t2 = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    ev2 = AuditEvent(
        event_id="inc-2",
        timestamp=t2,
        category=AuditEventCategory.API_CALL,
        event_type=AuditEventType.API_REQUEST,
        context=AuditContext(user_id=user_id),
    )
    async with svc_user.buffer_lock:
        svc_user.event_buffer.append(ev2)
    await svc_user.flush()
    await svc_user.stop()

    report2 = await migrate_to_shared_audit_db(
        shared_db_path=shared_db_path,
        user_db_base_dir=user_base,
        default_db_path=None,
        system_tenant_id="system",
        chunk_size=10,
    )
    source_counts2 = next(c for c in report2.sources if c.source.label == f"user:{user_id}")
    assert source_counts2.stats_read == 1
    assert source_counts2.stats_inserted == 1
    assert source_counts2.stats_skipped == 0

    async with aiosqlite.connect(shared_db_path) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT total_events FROM audit_daily_stats WHERE tenant_user_id = ? AND date = ? AND category = ?",
            (user_id, t1.date().isoformat(), AuditEventCategory.API_CALL.value),
        ) as cur:
            row = await cur.fetchone()
    assert row is not None
    assert row["total_events"] == 2


@pytest.mark.asyncio
async def test_migration_resume_realigns_normalized_checkpoint_timestamp(tmp_path):
    """Resume should not miss rows when stored checkpoint timestamp format differs."""
    user_base = tmp_path / "user_dbs"
    user_id = "1001"
    user_db_path = user_base / user_id / "audit" / "unified_audit.db"
    shared_db_path = tmp_path / "Databases" / "audit_shared.db"
    user_db_path.parent.mkdir(parents=True, exist_ok=True)

    svc_user = UnifiedAuditService(
        db_path=str(user_db_path),
        storage_mode="per_user",
        enable_pii_detection=False,
        enable_risk_scoring=False,
        buffer_size=1,
        flush_interval=0.1,
    )
    await svc_user.initialize(start_background_tasks=False)
    await svc_user.stop()

    ts1_raw = "2025-01-01T00:00:00-05:00"
    ts1_normalized = "2025-01-01T05:00:00+00:00"
    ts2_raw = "2025-01-01T01:00:00-05:00"

    async with aiosqlite.connect(user_db_path) as db:
        await db.execute(
            "INSERT INTO audit_events (event_id, timestamp, category, event_type, severity) VALUES (?, ?, ?, ?, ?)",
            ("evt-1", ts1_raw, "api_call", "api.request", "info"),
        )
        await db.execute(
            "INSERT INTO audit_events (event_id, timestamp, category, event_type, severity) VALUES (?, ?, ?, ?, ?)",
            ("evt-2", ts2_raw, "api_call", "api.request", "info"),
        )
        await db.commit()

    svc_shared = UnifiedAuditService(
        db_path=str(shared_db_path),
        storage_mode="shared",
        enable_pii_detection=False,
        enable_risk_scoring=False,
        buffer_size=10,
        flush_interval=0.1,
    )
    await svc_shared.initialize(start_background_tasks=False)
    await svc_shared.stop()

    source = audit_migration.AuditMigrationSource(
        path=user_db_path.resolve(),
        tenant_id=user_id,
        label=f"user:{user_id}",
    )

    async with aiosqlite.connect(shared_db_path) as shared_db:
        shared_db.row_factory = aiosqlite.Row
        await audit_migration._ensure_checkpoint_table(shared_db)
        await audit_migration._save_checkpoint(
            shared_db,
            source.path.resolve(),
            last_rowid=1,
            last_event_id="evt-1",
            last_timestamp=ts1_normalized,
        )
        await shared_db.commit()

        counts = await audit_migration._migrate_source(
            shared_db,
            source,
            columns=list(svc_shared._event_columns),
            insert_sql=svc_shared._event_insert_sql,
            system_tenant_id="system",
            unidentified_tenant_id="unidentified_user",
            chunk_size=100,
        )
        assert counts.failed is False
        assert counts.events_inserted == 1

        async with shared_db.execute("SELECT event_id FROM audit_events ORDER BY event_id") as cur:
            rows = await cur.fetchall()
        event_ids = [row["event_id"] for row in rows]
        assert event_ids == ["evt-2"]


def test_build_event_record_parses_falsey_pii_detected_string():
    row = {
        "event_id": "pii-bool-test",
        "timestamp": datetime(2025, 1, 1, tzinfo=timezone.utc).isoformat(),
        "category": "api_call",
        "event_type": "api.request",
        "severity": "info",
        "pii_detected": "false",
    }
    columns = [
        "event_id",
        "timestamp",
        "category",
        "event_type",
        "severity",
        "tenant_user_id",
    ]
    record = audit_migration._build_event_record(
        row,
        columns,
        tenant_override="123",
        system_tenant_id="system",
        unidentified_tenant_id="unidentified_user",
    )
    assert record["pii_detected"] is False

    row["pii_detected"] = "1"
    record_true = audit_migration._build_event_record(
        row,
        columns,
        tenant_override="123",
        system_tenant_id="system",
        unidentified_tenant_id="unidentified_user",
    )
    assert record_true["pii_detected"] is True
