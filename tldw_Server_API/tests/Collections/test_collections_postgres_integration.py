from __future__ import annotations

import os
from datetime import datetime, timezone

import pytest

from tldw_Server_API.app.core.DB_Management.Collections_DB import CollectionsDatabase
from tldw_Server_API.app.core.DB_Management.backends.base import BackendType, DatabaseConfig
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory

pytest_plugins = ["tldw_Server_API.tests._plugins.authnz_full_fixtures"]


def _pg_backend(db_name: str):
    return DatabaseBackendFactory.create_backend(
        DatabaseConfig(
            backend_type=BackendType.POSTGRESQL,
            pg_host=os.getenv("TEST_DB_HOST", "localhost"),
            pg_port=int(os.getenv("TEST_DB_PORT", "5432")),
            pg_database=db_name,
            pg_user=os.getenv("TEST_DB_USER", "tldw_user"),
            pg_password=os.getenv("TEST_DB_PASSWORD", "TestPassword123!"),
            pg_sslmode=os.getenv("TEST_DB_SSLMODE", "prefer"),
        )
    )


def test_collections_postgres_round_trip(request: pytest.FixtureRequest, monkeypatch, tmp_path):
    _client, db_name = request.getfixturevalue("isolated_test_environment")  # type: ignore[assignment]
    monkeypatch.setenv("USER_DB_BASE_DIR", str((tmp_path / "user_dbs").resolve()))

    backend = _pg_backend(db_name)
    db = CollectionsDatabase.from_backend(user_id="1", backend=backend)

    item = db.upsert_content_item(
        origin="reading",
        url="https://example.com/article",
        canonical_url="https://example.com/article",
        domain="example.com",
        title="Postgres Collections",
        summary="Testing collections on Postgres",
        notes=None,
        content_hash="hash-123",
        word_count=42,
        published_at=None,
        status="saved",
        favorite=True,
        metadata={"source": "test"},
        media_id=None,
        job_id=None,
        run_id=None,
        source_id=None,
        read_at=None,
        tags=["news", "postgres"],
    )
    assert item.id > 0
    assert item.is_new is True

    tpl = db.create_output_template(
        name="Default Summary",
        type_="summary",
        format_="markdown",
        body="Example body",
        description="test",
        is_default=True,
    )
    assert tpl.id > 0
    assert tpl.is_default is True
    items, total = db.list_output_templates(q="default summary", limit=10, offset=0)
    assert total >= 1
    assert any(item.name == "Default Summary" for item in items)

    expired_at = datetime(2000, 1, 1, tzinfo=timezone.utc).isoformat()
    output = db.create_output_artifact(
        type_="summary",
        title="Expired Output",
        format_="markdown",
        storage_path="expired.md",
        metadata_json=None,
        retention_until=expired_at,
        idempotency_key="postgres-output-v1",
    )
    assert output.id > 0
    replay = db.create_output_artifact(
        type_="summary",
        title="Expired Output",
        format_="markdown",
        storage_path="expired.md",
        metadata_json=None,
        retention_until=expired_at,
        idempotency_key="postgres-output-v1",
    )
    assert replay.id == output.id
    assert replay.idempotency_key == "postgres-output-v1"

    columns = {row["name"] for row in backend.get_table_info("outputs")}
    assert "idempotency_key" in columns
    index_names = {
        row["indexname"]
        for row in backend.execute(
            "SELECT indexname FROM pg_indexes WHERE tablename = ?",
            ("outputs",),
        ).rows
    }
    assert "ux_outputs_user_idempotency" in index_names

    purged = db.purge_expired_outputs()
    assert purged >= 1


def test_collections_postgres_backfills_notification_delivery_columns(
    request: pytest.FixtureRequest,
    monkeypatch,
    tmp_path,
):
    _client, db_name = request.getfixturevalue("isolated_test_environment")  # type: ignore[assignment]
    monkeypatch.setenv("USER_DB_BASE_DIR", str((tmp_path / "user_dbs").resolve()))

    backend = _pg_backend(db_name)
    backend.execute(
        """
        CREATE TABLE user_notifications (
            id BIGSERIAL PRIMARY KEY,
            user_id TEXT NOT NULL,
            kind TEXT NOT NULL,
            title TEXT NOT NULL,
            message TEXT NOT NULL,
            severity TEXT NOT NULL,
            source_task_id TEXT,
            source_task_run_id BIGINT,
            source_job_id TEXT,
            source_domain TEXT,
            source_job_type TEXT,
            link_type TEXT,
            link_id TEXT,
            link_url TEXT,
            dedupe_key TEXT,
            retention_until TEXT,
            archived_at TEXT,
            created_at TEXT NOT NULL,
            read_at TEXT,
            dismissed_at TEXT
        )
        """,
        (),
    )

    db = CollectionsDatabase.from_backend(user_id="1", backend=backend)

    columns = {row["name"] for row in backend.get_table_info("user_notifications")}
    assert "snooze_task_id" in columns
    assert "delivery_status" in columns
    assert "delivered_at" in columns

    notification = db.create_user_notification(
        kind="job_completed",
        title="Backfilled notification",
        message="Backfill should preserve defaults",
        severity="info",
    )
    assert notification.delivery_status == "pending"
