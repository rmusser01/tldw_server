from __future__ import annotations

import os
import sqlite3
import socket
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.DB_Management import migrate_db as migrate_db_cli
from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase
from tldw_Server_API.app.core.DB_Management.Workflows_DB import WorkflowsDatabase
from tldw_Server_API.app.core.DB_Management import migration_tools
from tldw_Server_API.app.core.DB_Management.backends.base import BackendType, DatabaseConfig
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory

try:
    import psycopg as _psycopg_v3

    _PG_DRIVER = "psycopg"
except Exception:  # pragma: no cover - may be missing locally
    try:
        import psycopg2 as _psycopg2

        _PG_DRIVER = "psycopg2"
    except Exception:
        _PG_DRIVER = None

_required_env = [
    "POSTGRES_TEST_HOST",
    "POSTGRES_TEST_PORT",
    "POSTGRES_TEST_DB",
    "POSTGRES_TEST_USER",
    "POSTGRES_TEST_PASSWORD",
]

requires_postgres_driver = pytest.mark.skipif(
    _PG_DRIVER is None,
    reason="Postgres driver not installed",
)


def _require_postgres_driver() -> str:
    if _PG_DRIVER is None:
        pytest.skip("Postgres driver not installed")
    return _PG_DRIVER


def _postgres_server_reachable() -> bool:
    from tldw_Server_API.tests.helpers.pg_env import get_pg_env

    env = get_pg_env()
    try:
        with socket.create_connection((env.host, int(env.port)), timeout=1.0):
            return True
    except OSError:
        return False


def test_migrate_threads_create_backup_flag(monkeypatch):
    calls = {}
    placeholder_db_path = "task2_cli_placeholder.db"

    class _FakeMigrator:
        def __init__(self, db_path: str):
            calls["db_path"] = db_path

        def get_current_version(self):
            return 0

        def load_migrations(self):
            return [SimpleNamespace(version=5)]

        def migrate_to_version(self, target_version, create_backup=True):
            calls["target_version"] = target_version
            calls["create_backup"] = create_backup
            return {
                "status": "success",
                "previous_version": 0,
                "current_version": target_version,
                "target_version": target_version,
                "migrations_applied": [],
                "total_execution_time": 0.0,
                "backup_path": None,
            }

    monkeypatch.setattr(migrate_db_cli, "DatabaseMigrator", _FakeMigrator)
    monkeypatch.setattr(migrate_db_cli, "get_default_db_path", lambda: placeholder_db_path)
    monkeypatch.setattr(
        migrate_db_cli.sys,
        "argv",
        [
            "migrate_db.py",
            "--db-path",
            placeholder_db_path,
            "migrate",
            "--version",
            "5",
            "--no-backup",
        ],
    )

    migrate_db_cli.main()

    expected = {
        "db_path": placeholder_db_path,
        "target_version": 5,
        "create_backup": False,
    }
    if calls != expected:
        pytest.fail(f"Expected migrate() to thread create_backup; got {calls!r}")


def _base_postgres_config() -> DatabaseConfig:

    from tldw_Server_API.tests.helpers.pg_env import get_pg_env

    _pg = get_pg_env()
    return DatabaseConfig(
        backend_type=BackendType.POSTGRESQL,
        pg_host=_pg.host,
        pg_port=int(_pg.port),
        pg_database=_pg.database,
        pg_user=_pg.user,
        pg_password=_pg.password,
    )


def _create_temp_postgres_database(config: DatabaseConfig) -> DatabaseConfig:
    driver = _require_postgres_driver()
    db_name = f"tldw_test_{uuid.uuid4().hex[:8]}"
    if driver == "psycopg":
        admin = _psycopg_v3.connect(
            host=config.pg_host,
            port=config.pg_port,
            dbname="postgres",
            user=config.pg_user,
            password=config.pg_password,
        )
    else:
        admin = _psycopg2.connect(
            host=config.pg_host,
            port=config.pg_port,
            database="postgres",
            user=config.pg_user,
            password=config.pg_password,
        )
    admin.autocommit = True
    try:
        with admin.cursor() as cur:
            cur.execute(f"CREATE DATABASE {db_name} OWNER {config.pg_user};")
    finally:
        admin.close()

    return DatabaseConfig(
        backend_type=BackendType.POSTGRESQL,
        pg_host=config.pg_host,
        pg_port=config.pg_port,
        pg_database=db_name,
        pg_user=config.pg_user,
        pg_password=config.pg_password,
    )


def _drop_postgres_database(config: DatabaseConfig) -> None:
    driver = _require_postgres_driver()
    if driver == "psycopg":
        admin = _psycopg_v3.connect(
            host=config.pg_host,
            port=config.pg_port,
            dbname="postgres",
            user=config.pg_user,
            password=config.pg_password,
        )
    else:
        admin = _psycopg2.connect(
            host=config.pg_host,
            port=config.pg_port,
            database="postgres",
            user=config.pg_user,
            password=config.pg_password,
        )
    admin.autocommit = True
    try:
        with admin.cursor() as cur:
            cur.execute(
                "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = %s;",
                (config.pg_database,),
            )
            cur.execute(f"DROP DATABASE IF EXISTS {config.pg_database};")
    finally:
        admin.close()


@pytest.fixture()
def temp_postgres_config() -> DatabaseConfig:
    if not _postgres_server_reachable():
        pytest.skip("Postgres not reachable; skipping Postgres-backed tests")

    base = _base_postgres_config()
    cfg = _create_temp_postgres_database(base)
    try:
        return cfg
    finally:
        # pytest won't run this finally here; actual cleanup is done in tests' teardown where used
        ...


@pytest.fixture()
def sqlite_media_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "media_source.db"
    media_db = MediaDatabase(str(db_path), client_id="migration-test")
    now = media_db._get_current_utc_timestamp_str()
    media_uuid = str(uuid.uuid4())
    claim_uuid = str(uuid.uuid4())
    with media_db.transaction() as conn:
        conn.execute(
            """
            INSERT INTO Media (
                title, type, content, url, ingestion_date, transcription_model,
                chunking_status, vector_processing, content_hash, uuid,
                last_modified, version, client_id, deleted, is_trash
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 0)
            """,
            (
                "Migration Source Document",
                "text",
                "Original content from SQLite",
                None,
                now,
                None,
                "complete",
                0,
                "hash-123",
                media_uuid,
                now,
                1,
                "migration-test",
            ),
        )
        media_id = conn.execute(
            "SELECT id FROM Media WHERE uuid = ?",
            (media_uuid,),
        ).fetchone()[0]
        conn.execute(
            """
            INSERT INTO Claims (
                media_id, chunk_index, claim_text, chunk_hash, extractor,
                extractor_version, created_at, uuid, last_modified, version,
                client_id, deleted
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
            """,
            (
                media_id,
                0,
                "Claim copied via migration",
                "chunk-hash-1",
                "migration",
                "v1",
                now,
                claim_uuid,
                now,
                1,
                "migration-test",
            ),
        )
    media_db.close_connection()
    return db_path


@pytest.fixture()
def sqlite_workflows_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "workflows_source.db"
    wf_db = WorkflowsDatabase(str(db_path))

    definition_id = wf_db.create_definition(
        tenant_id="tenant-wf",
        name="Workflow Migration",
        version=1,
        owner_id="owner-wf",
        visibility="private",
        description="Migrated definition",
        tags=["wf"],
        definition={"name": "Workflow Migration", "version": 1, "steps": []},
    )

    run_id = "wf-run-source"
    wf_db.create_run(
        run_id=run_id,
        tenant_id="tenant-wf",
        user_id="runner",
        inputs={},
        workflow_id=definition_id,
        definition_version=1,
        definition_snapshot={"name": "Workflow Migration", "version": 1, "steps": []},
        idempotency_key="wf-idem",
        session_id="wf-session",
    )

    wf_db.append_event(
        tenant_id="tenant-wf",
        run_id=run_id,
        event_type="run_created",
        payload={"source": "sqlite"},
    )
    wf_db.create_step_run(
        step_run_id="wf-step",
        tenant_id="tenant-wf",
        run_id=run_id,
        step_id="step-1",
        name="Step",
        step_type="prompt",
        inputs={"config": {"prompt": "Hi"}},
    )
    wf_db.complete_step_run(
        step_run_id="wf-step",
        status="succeeded",
        outputs={"response": "Hi"},
    )
    wf_db.add_artifact(
        artifact_id="wf-artifact",
        tenant_id="tenant-wf",
        run_id=run_id,
        step_run_id="wf-step",
        type="text",
        uri="s3://bucket/wf.txt",
        metadata={"origin": "sqlite"},
    )
    wf_db.close_connection()
    return db_path


def _reset_postgres_database(config: DatabaseConfig) -> None:
    # No-op retained for backward compatibility (per-test DB is created instead)
    return None


def _postgres_counts(config: DatabaseConfig) -> tuple[int, int]:
    driver = _require_postgres_driver()
    if driver == "psycopg":
        conn = _psycopg_v3.connect(
            host=config.pg_host,
            port=config.pg_port,
            dbname=config.pg_database,
            user=config.pg_user,
            password=config.pg_password,
        )
    else:
        conn = _psycopg2.connect(
            host=config.pg_host,
            port=config.pg_port,
            database=config.pg_database,
            user=config.pg_user,
            password=config.pg_password,
        )
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM media")
            media_count = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM claims")
            claims_count = cur.fetchone()[0]
    finally:
        conn.close()
    return media_count, claims_count


@requires_postgres_driver
@pytest.mark.integration
def test_migration_cli_transfers_content_rows(sqlite_media_db: Path, temp_postgres_config: DatabaseConfig) -> None:
    backend = DatabaseBackendFactory.create_backend(temp_postgres_config)
    pg_media_db = MediaDatabase(db_path=":memory:", client_id="migration-target", backend=backend)
    pg_media_db.close_connection()

    migration_tools.migrate_sqlite_to_postgres(sqlite_media_db, temp_postgres_config, label="content", batch_size=16)

    with sqlite3.connect(sqlite_media_db) as conn:
        sqlite_media = conn.execute("SELECT COUNT(*) FROM Media").fetchone()[0]
        sqlite_claims = conn.execute("SELECT COUNT(*) FROM Claims").fetchone()[0]

    pg_media, pg_claims = _postgres_counts(temp_postgres_config)

    if pg_media != sqlite_media:
        pytest.fail(f"Expected migrated media count {sqlite_media}, got {pg_media}")
    if pg_claims != sqlite_claims:
        pytest.fail(f"Expected migrated claim count {sqlite_claims}, got {pg_claims}")

    if pg_media <= 0:
        pytest.fail(f"Expected migrated media rows to be > 0, got {pg_media}")
    if pg_claims <= 0:
        pytest.fail(f"Expected migrated claim rows to be > 0, got {pg_claims}")

    if _PG_DRIVER == "psycopg":
        conn = _psycopg_v3.connect(
            host=temp_postgres_config.pg_host,
            port=temp_postgres_config.pg_port,
            dbname=temp_postgres_config.pg_database,
            user=temp_postgres_config.pg_user,
            password=temp_postgres_config.pg_password,
        )
    else:
        conn = _psycopg2.connect(
            host=temp_postgres_config.pg_host,
            port=temp_postgres_config.pg_port,
            database=temp_postgres_config.pg_database,
            user=temp_postgres_config.pg_user,
            password=temp_postgres_config.pg_password,
        )
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT claim_text FROM claims ORDER BY id LIMIT 1")
            migrated_claim = cur.fetchone()[0]
    finally:
        conn.close()

    if migrated_claim != "Claim copied via migration":
        pytest.fail(
            f"Expected migrated claim text to round-trip, got {migrated_claim!r}"
        )
    # Cleanup temporary DB
    _drop_postgres_database(temp_postgres_config)


def _postgres_workflow_counts(config: DatabaseConfig) -> tuple[int, int, int, int]:
    driver = _require_postgres_driver()
    if driver == "psycopg":
        conn = _psycopg_v3.connect(
            host=config.pg_host,
            port=config.pg_port,
            dbname=config.pg_database,
            user=config.pg_user,
            password=config.pg_password,
        )
    else:
        conn = _psycopg2.connect(
            host=config.pg_host,
            port=config.pg_port,
            database=config.pg_database,
            user=config.pg_user,
            password=config.pg_password,
        )
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM workflows")
            defs = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM workflow_runs")
            runs = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM workflow_events")
            events = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM workflow_artifacts")
            artifacts = cur.fetchone()[0]
    finally:
        conn.close()
    return defs, runs, events, artifacts


def _sqlite_workflow_counts(path: Path) -> tuple[int, int, int, int]:
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    try:
        defs = conn.execute("SELECT COUNT(*) FROM workflows").fetchone()[0]
        runs = conn.execute("SELECT COUNT(*) FROM workflow_runs").fetchone()[0]
        events = conn.execute("SELECT COUNT(*) FROM workflow_events").fetchone()[0]
        artifacts = conn.execute("SELECT COUNT(*) FROM workflow_artifacts").fetchone()[0]
    finally:
        conn.close()
    return defs, runs, events, artifacts


@requires_postgres_driver
@pytest.mark.integration
def test_migration_cli_transfers_workflow_rows(sqlite_workflows_db: Path, temp_postgres_config: DatabaseConfig) -> None:
    backend = DatabaseBackendFactory.create_backend(temp_postgres_config)
    wf_db = WorkflowsDatabase(db_path=":memory:", backend=backend)
    wf_db.close_connection()

    migration_tools.migrate_workflows_sqlite_to_postgres(
        sqlite_workflows_db,
        temp_postgres_config,
        batch_size=32,
    )

    sqlite_counts = _sqlite_workflow_counts(sqlite_workflows_db)
    pg_counts = _postgres_workflow_counts(temp_postgres_config)

    if sqlite_counts != pg_counts:
        pytest.fail(
            f"Expected workflow row counts to match after migration: "
            f"sqlite={sqlite_counts!r}, postgres={pg_counts!r}"
        )

    defs, runs, _, _ = pg_counts
    if defs <= 0 or runs <= 0:
        pytest.fail(
            f"Expected migrated workflow definitions and runs to be > 0, got {(defs, runs)!r}"
        )
    # Cleanup temporary DB
    _drop_postgres_database(temp_postgres_config)
