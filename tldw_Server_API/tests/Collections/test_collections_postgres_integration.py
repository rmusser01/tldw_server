from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import pytest

from tldw_Server_API.app.core.DB_Management.backends.base import BackendType, DatabaseConfig
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.Collections_DB import CollectionsDatabase

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
    indexes = {
        row["indexname"]: row["indexdef"]
        for row in backend.execute(
            "SELECT indexname, indexdef FROM pg_indexes WHERE tablename = ?",
            ("outputs",),
        ).rows
    }
    assert "ux_outputs_user_idempotency_active" in indexes
    assert " WHERE " in indexes["ux_outputs_user_idempotency_active"].upper()
    assert "deleted" in indexes["ux_outputs_user_idempotency_active"]
    assert "ux_outputs_user_idempotency" not in indexes

    assert db.delete_output_artifact(output.id) is True
    recreated = db.create_output_artifact(
        type_="summary",
        title="Expired Output",
        format_="markdown",
        storage_path="expired.md",
        metadata_json=None,
        retention_until=expired_at,
        idempotency_key="postgres-output-v1",
    )
    assert recreated.id != output.id
    tombstones = backend.execute(
        "SELECT id FROM outputs WHERE user_id = ? AND idempotency_key = ? AND deleted = TRUE",
        ("1", "postgres-output-v1"),
    ).rows
    assert [int(row["id"]) for row in tombstones] == [output.id]

    purged = db.purge_expired_outputs()
    assert purged >= 1


def test_playlist_collection_actions_are_cas_safe_on_postgres(
    request: pytest.FixtureRequest,
    monkeypatch,
    tmp_path,
):
    _client, db_name = request.getfixturevalue("isolated_test_environment")  # type: ignore[assignment]
    monkeypatch.setenv("USER_DB_BASE_DIR", str((tmp_path / "user_dbs").resolve()))

    backend = _pg_backend(db_name)
    db = CollectionsDatabase.from_backend(user_id="1", backend=backend)
    created = db.create_media_collection_with_items(
        name="Playlist CAS",
        kind="playlist_ingest",
        metadata={
            "playlist_ingest_run_id": "pg-run-cas",
            "playlist_ingest_initialization_token": "pg-token-a",
        },
        items=[
            {"source_url": "https://example.com/one", "ordinal": 1},
            {"source_url": "https://example.com/two", "ordinal": 2},
        ],
    )

    def resolve(media_id: int):
        try:
            return db.resolve_media_collection_item(created.items[0].id, media_id=media_id)
        except ValueError as exc:
            return exc

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(resolve, (17, 18)))

    resolved = db.get_media_collection_item(created.items[0].id)
    assert len([result for result in results if not isinstance(result, Exception)]) == 1
    assert len([result for result in results if isinstance(result, ValueError)]) == 1
    assert resolved.media_id in {17, 18}
    repeated = db.resolve_media_collection_item(
        resolved.id,
        media_id=resolved.media_id,
        status="completed",
    )
    assert repeated.updated_at == resolved.updated_at

    restored = db.restore_media_collection_item_plan(
        resolved.id,
        expected_media_id=resolved.media_id,
        expected_status="completed",
        expected_updated_at=resolved.updated_at,
    )
    repeated_restore = db.restore_media_collection_item_plan(
        resolved.id,
        expected_media_id=resolved.media_id,
        expected_status="completed",
        expected_updated_at=resolved.updated_at,
    )
    assert repeated_restore.updated_at == restored.updated_at

    claimed = db.claim_playlist_ingest_collection(
        created.id,
        run_id="pg-run-cas",
        initialization_token="pg-token-b",
        expected_item_ids=[item.id for item in created.items],
    )
    assert claimed.metadata["playlist_ingest_initialization_token"] == "pg-token-b"

    with pytest.raises(ValueError, match="media_collection_discard_mismatch"):
        db.discard_media_collection(
            created.id,
            expected_item_ids=[item.id for item in created.items],
            expected_run_id="pg-run-cas",
            expected_initialization_token="pg-token-a",
        )
    assert db.get_media_collection(created.id).id == created.id

    assert db.discard_media_collection(
        created.id,
        expected_item_ids=[item.id for item in created.items],
        expected_run_id="pg-run-cas",
        expected_initialization_token="pg-token-b",
    )


def test_playlist_collection_commit_then_error_recovery_attaches_run_on_postgres(
    request: pytest.FixtureRequest,
    monkeypatch,
    tmp_path,
):
    _client, db_name = request.getfixturevalue("isolated_test_environment")  # type: ignore[assignment]
    monkeypatch.setenv("USER_DB_BASE_DIR", str((tmp_path / "user_dbs").resolve()))
    monkeypatch.setenv("TEST_MODE", "true")

    from tldw_Server_API.app.core.Ingestion_Media_Processing.Video.playlist_ingest_service import (
        PlaylistIngestService,
    )
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Video.playlist_ingest_store import (
        PlaylistIngestStore,
    )
    from tldw_Server_API.app.core.Jobs.manager import JobManager

    backend = _pg_backend(db_name)
    db = CollectionsDatabase.from_backend(user_id="1", backend=backend)
    malformed = db.create_media_collection(
        name="Malformed legacy playlist",
        kind="playlist_ingest",
    )
    backend.execute(
        "UPDATE media_collections SET metadata_json = ? WHERE id = ? AND user_id = ?",
        ("{malformed", malformed.id, "1"),
    )

    original_create = db.create_media_collection_with_items
    committed = None

    def create_then_raise(**kwargs):
        nonlocal committed
        committed = original_create(**kwargs)
        raise RuntimeError("private post-commit detail")

    monkeypatch.setattr(db, "create_media_collection_with_items", create_then_raise)
    monkeypatch.setattr(db, "close", lambda: None)

    class _NoDuplicateMediaDB:
        @staticmethod
        def get_media_by_urls(_urls, **_kwargs):
            return []

        @staticmethod
        def close_connection():
            return None

    manager = JobManager(db_path=tmp_path / "playlist-recovery-jobs.db")
    service = PlaylistIngestService(
        manager,
        media_db_factory=lambda _owner: _NoDuplicateMediaDB(),
        collections_db_factory=lambda owner: db if owner == "1" else None,
    )
    created = service.create_run(
        "1",
        inputs=[
            {
                "input_kind": "direct_url",
                "occurrence_id": "pg-recovery-occurrence",
                "url": "https://example.com/postgres-recovery",
                "source_kind": "video",
                "display_metadata": {"title": "PostgreSQL recovery"},
            }
        ],
        review_overrides={},
        new_collection={"name": "Recovered playlist plan"},
    )

    assert committed is not None
    recovered = db.get_playlist_ingest_collection_for_run(created.run_id)
    item = PlaylistIngestStore(manager).list_run_items("1", created.run_id)[0]
    assert created.collection_id == committed.id == recovered.id
    assert item.planned_collection_item_id == committed.items[0].id
    other_owner = CollectionsDatabase.from_backend(user_id="2", backend=backend)
    with pytest.raises(KeyError, match="media_collection_not_found"):
        other_owner.get_playlist_ingest_collection_for_run(created.run_id)


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
