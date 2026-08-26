from __future__ import annotations

import aiosqlite
import json
import pytest
import sqlite3


@pytest.mark.asyncio
@pytest.mark.unit
async def test_create_source_persists_state_row(tmp_path):
    from tldw_Server_API.app.core.Ingestion_Sources.service import (
        create_source,
        ensure_ingestion_sources_schema,
    )

    db_path = tmp_path / "ingestion_sources.sqlite3"
    async with aiosqlite.connect(str(db_path)) as db:
        db.row_factory = aiosqlite.Row

        await ensure_ingestion_sources_schema(db)
        row = await create_source(
            db,
            user_id=7,
            payload={
                "source_type": "local_directory",
                "sink_type": "media",
                "policy": "canonical",
                "config": {"path": "/allowed/project/docs"},
            },
        )

        assert row["user_id"] == 7
        assert row["source_type"] == "local_directory"
        assert row["sink_type"] == "media"

        state_cur = await db.execute(
            "SELECT source_id, active_job_id, last_successful_snapshot_id "
            "FROM ingestion_source_state WHERE source_id = ?",
            (row["id"],),
        )
        state_row = await state_cur.fetchone()

        assert state_row is not None
        assert state_row["source_id"] == row["id"]
        assert state_row["active_job_id"] is None
        assert state_row["last_successful_snapshot_id"] is None


@pytest.mark.asyncio
@pytest.mark.unit
async def test_create_source_works_through_guarded_authnz_transaction(tmp_path):
    from tldw_Server_API.app.core.AuthNZ.database import DatabasePool
    from tldw_Server_API.app.core.AuthNZ.profile_user_write_guard import (
        ProfileUserWriteRejected,
    )
    from tldw_Server_API.app.core.Ingestion_Sources.service import (
        create_source,
        ensure_ingestion_sources_schema,
    )

    db_path = tmp_path / "authnz.sqlite3"
    with sqlite3.connect(db_path) as setup:
        setup.execute(
            "CREATE TABLE users (id INTEGER PRIMARY KEY, email TEXT NOT NULL)"
        )
        setup.execute("INSERT INTO users (id, email) VALUES (1, 'before@example.test')")

    pool = DatabasePool.__new__(DatabasePool)
    pool.pool = None
    pool.db_path = str(db_path)
    pool._sqlite_uri = False
    pool._initialized = True

    async with pool.transaction() as db:
        await ensure_ingestion_sources_schema(db)
        source = await create_source(
            db,
            user_id=1,
            payload={
                "source_type": "local_directory",
                "sink_type": "media",
                "policy": "canonical",
                "config": {"path": "/allowed/project/docs"},
            },
        )
        with pytest.raises(ProfileUserWriteRejected):
            await db.execute("UPDATE users SET email = 'changed@example.test' WHERE id = 1")

    with sqlite3.connect(db_path) as verify:
        source_row = verify.execute(
            "SELECT user_id, source_type FROM ingestion_sources WHERE id = ?",
            (source["id"],),
        ).fetchone()
        user_email = verify.execute("SELECT email FROM users WHERE id = 1").fetchone()[0]

    assert source_row == (1, "local_directory")
    assert user_email == "before@example.test"


@pytest.mark.asyncio
@pytest.mark.unit
async def test_ensure_sqlite_column_rejects_unsafe_identifiers(tmp_path):
    from tldw_Server_API.app.core.Ingestion_Sources.service import _ensure_sqlite_column
    from tldw_Server_API.app.core.exceptions import IngestionSourceValidationError

    db_path = tmp_path / "ingestion_sources.sqlite3"
    async with aiosqlite.connect(str(db_path)) as db:
        with pytest.raises(IngestionSourceValidationError, match="unsafe SQL identifier"):
            await _ensure_sqlite_column(
                db,
                table_name="ingestion_source_items; DROP TABLE users;--",
                column_name="present_in_source",
                column_sql="INTEGER NOT NULL DEFAULT 1",
            )


@pytest.mark.asyncio
@pytest.mark.unit
async def test_ensure_ingestion_sources_schema_creates_query_indexes(tmp_path):
    from tldw_Server_API.app.core.Ingestion_Sources.service import ensure_ingestion_sources_schema

    db_path = tmp_path / "ingestion_sources.sqlite3"
    async with aiosqlite.connect(str(db_path)) as db:
        db.row_factory = aiosqlite.Row

        await ensure_ingestion_sources_schema(db)

        cursor = await db.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'index'
              AND name LIKE 'idx_ingestion_%'
            """
        )
        rows = await cursor.fetchall()

    index_names = {row["name"] for row in rows}
    assert {
        "idx_ingestion_sources_user_id",
        "idx_ingestion_sources_scheduler",
        "idx_ingestion_source_state_active_job",
        "idx_ingestion_source_snapshots_source_status_id",
        "idx_ingestion_source_artifacts_source_kind_status",
        "idx_ingestion_item_events_source_item",
    }.issubset(index_names)


@pytest.mark.asyncio
@pytest.mark.unit
async def test_finish_source_sync_job_raises_when_active_job_fence_mismatches(tmp_path):
    from tldw_Server_API.app.core.Ingestion_Sources.service import (
        create_source,
        ensure_ingestion_sources_schema,
        finish_source_sync_job,
        start_source_sync_job,
    )

    db_path = tmp_path / "ingestion_sources.sqlite3"
    async with aiosqlite.connect(str(db_path)) as db:
        db.row_factory = aiosqlite.Row

        await ensure_ingestion_sources_schema(db)
        row = await create_source(
            db,
            user_id=7,
            payload={
                "source_type": "local_directory",
                "sink_type": "media",
                "policy": "canonical",
                "config": {"path": "/allowed/project/docs"},
            },
        )
        await start_source_sync_job(db, source_id=int(row["id"]), job_id="job-1")

        with pytest.raises(RuntimeError, match="active sync job"):
            await finish_source_sync_job(
                db,
                source_id=int(row["id"]),
                job_id="job-2",
                outcome="success",
                snapshot_id=1,
            )


@pytest.mark.asyncio
@pytest.mark.unit
async def test_update_source_delegates_row_update_to_db_management(tmp_path, monkeypatch):
    import tldw_Server_API.app.core.Ingestion_Sources.service as service

    db_path = tmp_path / "ingestion_sources.sqlite3"
    async with aiosqlite.connect(str(db_path)) as db:
        db.row_factory = aiosqlite.Row
        await service.ensure_ingestion_sources_schema(db)
        created = await service.create_source(
            db,
            user_id=7,
            payload={
                "source_type": "local_directory",
                "sink_type": "notes",
                "policy": "canonical",
                "config": {"path": "/allowed/project/docs"},
            },
        )

        captured_calls: list[dict[str, object]] = []

        async def _fake_update_ingestion_source_record(
            db_conn,
            *,
            source_id: int,
            source_type: str,
            sink_type: str,
            policy: str,
            enabled: bool,
            schedule_enabled: bool,
            schedule_config: dict[str, object],
            config: dict[str, object],
            updated_at: str,
        ) -> None:
            captured_calls.append(
                {
                    "source_id": source_id,
                    "source_type": source_type,
                    "sink_type": sink_type,
                    "policy": policy,
                    "enabled": enabled,
                    "schedule_enabled": schedule_enabled,
                    "schedule_config": schedule_config,
                    "config": config,
                }
            )
            await db_conn.execute(
                """
                UPDATE ingestion_sources
                SET source_type = ?,
                    sink_type = ?,
                    policy = ?,
                    enabled = ?,
                    schedule_enabled = ?,
                    schedule_config_json = ?,
                    config_json = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (
                    source_type,
                    sink_type,
                    policy,
                    1 if enabled else 0,
                    1 if schedule_enabled else 0,
                    json.dumps(schedule_config, sort_keys=True),
                    json.dumps(config, sort_keys=True),
                    updated_at,
                    source_id,
                ),
            )

        monkeypatch.setattr(service, "update_ingestion_source_record", _fake_update_ingestion_source_record, raising=False)

        updated = await service.update_source(
            db,
            source_id=int(created["id"]),
            user_id=7,
            patch={
                "source_type": "git_repository",
                "sink_type": "notes",
                "config": {"mode": "local_repo", "path": "/allowed/project/repo"},
                "policy": "import_only",
                "enabled": False,
                "schedule_enabled": True,
                "schedule": {"interval_minutes": 15},
            },
        )

        assert captured_calls == [
            {
                "source_id": int(created["id"]),
                "source_type": "git_repository",
                "sink_type": "notes",
                "policy": "import_only",
                "enabled": False,
                "schedule_enabled": True,
                "schedule_config": {"interval_minutes": 15},
                "config": {"mode": "local_repo", "path": "/allowed/project/repo"},
            }
        ]
        assert updated["source_type"] == "git_repository"
        assert updated["sink_type"] == "notes"
        assert updated["policy"] == "import_only"
        assert updated["enabled"] is False
        assert updated["schedule_enabled"] is True
        assert updated["schedule_config"] == {"interval_minutes": 15}
        assert updated["config"] == {"mode": "local_repo", "path": "/allowed/project/repo"}
