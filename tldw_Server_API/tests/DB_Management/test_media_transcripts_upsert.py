import os
import uuid

import pytest

from tldw_Server_API.app.core.DB_Management.media_db import legacy_transcripts
from tldw_Server_API.app.core.DB_Management.media_db.errors import DatabaseError
from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase
from tldw_Server_API.app.core.DB_Management.backends.base import BackendType, DatabaseConfig
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.media_db.legacy_reads import (
    get_latest_transcription,
)
from tldw_Server_API.app.core.DB_Management.media_db.legacy_transcripts import (
    upsert_transcript,
)


def _insert_minimal_media(db: MediaDatabase) -> int:
    now = db._get_current_utc_timestamp_str()
    media_uuid = str(uuid.uuid4())
    sql = (
        "INSERT INTO Media (url, title, type, content, content_hash, is_trash, chunking_status, vector_processing, uuid, last_modified, version, client_id, deleted) "
        "VALUES (?, ?, ?, ?, ?, 0, 'pending', 0, ?, ?, 1, ?, 0)"
    )
    with db.transaction() as conn:
        cur = db._execute_with_connection(
            conn,
            sql,
            (
                f"http://example.com/{media_uuid}",
                "Unit Test Media",
                "article",
                "lorem ipsum",
                media_uuid,
                media_uuid,
                now,
                db.client_id,
            ),
        )
        if db.backend_type == BackendType.POSTGRESQL:
            # No RETURNING clause, fetch id
            row = db._fetchone_with_connection(conn, "SELECT id FROM Media WHERE uuid = ?", (media_uuid,))
            return int(row["id"])  # type: ignore[index]
        return int(cur.lastrowid or 0)


@pytest.mark.unit
def test_sqlite_upsert_transcript_roundtrip(tmp_path):

    db = MediaDatabase(db_path=str(tmp_path / "media.db"), client_id="unit-sqlite")
    media_id = _insert_minimal_media(db)

    p1 = upsert_transcript(db, media_id, transcription="hello world", whisper_model="base")
    assert p1["media_id"] == media_id and p1["version"] >= 1 and p1["uuid"]
    assert p1["transcription_run_id"] == 1

    p2 = upsert_transcript(db, media_id, transcription="updated text", whisper_model="base")
    assert p2["uuid"] != p1["uuid"]
    assert p2["version"] == 1
    assert p2["transcription_run_id"] == 2

    latest_row = db.execute_query(
        "SELECT latest_transcription_run_id, next_transcription_run_id FROM Media WHERE id = ?",
        (media_id,),
    ).fetchone()
    assert latest_row is not None
    assert latest_row["latest_transcription_run_id"] == 2
    assert latest_row["next_transcription_run_id"] == 3

    latest = get_latest_transcription(db, media_id)
    assert latest == "updated text"

@pytest.mark.unit
def test_sqlite_upsert_transcript_records_superseded_run_id(tmp_path):

    db = MediaDatabase(db_path=str(tmp_path / "media.db"), client_id="unit-sqlite-supersedes")
    media_id = _insert_minimal_media(db)

    p1 = upsert_transcript(db, media_id, transcription="hello world", whisper_model="base")
    p2 = upsert_transcript(db, media_id, transcription="updated text", whisper_model="base")

    rows = list(
        db.execute_query(
            """
            SELECT transcription_run_id, supersedes_run_id
            FROM Transcripts
            WHERE media_id = ?
            ORDER BY transcription_run_id ASC
            """,
            (media_id,),
        )
        or []
    )

    assert p1["transcription_run_id"] == 1
    assert p1["supersedes_run_id"] is None
    assert p2["transcription_run_id"] == 2
    assert p2["supersedes_run_id"] == 1
    assert len(rows) == 2
    assert rows[0]["transcription_run_id"] == 1
    assert rows[0]["supersedes_run_id"] is None
    assert rows[1]["transcription_run_id"] == 2
    assert rows[1]["supersedes_run_id"] == 1


@pytest.mark.integration
def test_postgres_upsert_transcript_roundtrip_if_available(tmp_path, pg_eval_params):
    cfg = DatabaseConfig(
        backend_type=BackendType.POSTGRESQL,
        pg_host=pg_eval_params["host"],
        pg_port=int(pg_eval_params["port"]),
        pg_database=pg_eval_params["database"],
        pg_user=pg_eval_params["user"],
        pg_password=pg_eval_params.get("password"),
    )
    try:
        backend = DatabaseBackendFactory.create_backend(cfg)
    except Exception:
        pytest.skip("psycopg backend not available")

    db = MediaDatabase(db_path=":memory:", client_id="unit-pg", backend=backend)
    media_id = _insert_minimal_media(db)

    p1 = upsert_transcript(db, media_id, transcription="pg hello", whisper_model="large-v3")
    assert p1["media_id"] == media_id and p1["version"] >= 1 and p1["uuid"]
    assert p1["transcription_run_id"] == 1

    p2 = upsert_transcript(db, media_id, transcription="pg updated", whisper_model="large-v3")
    assert p2["uuid"] != p1["uuid"]
    assert p2["version"] == 1
    assert p2["transcription_run_id"] == 2

    latest_row = db.execute_query(
        "SELECT latest_transcription_run_id, next_transcription_run_id FROM Media WHERE id = ?",
        (media_id,),
    ).fetchone()
    assert latest_row is not None
    assert latest_row["latest_transcription_run_id"] == 2
    assert latest_row["next_transcription_run_id"] == 3

    latest = get_latest_transcription(db, media_id)
    assert latest == "pg updated"


@pytest.mark.unit
def test_sqlite_upsert_transcript_reuses_existing_run_for_idempotency_key(tmp_path):

    db = MediaDatabase(db_path=str(tmp_path / "media.db"), client_id="unit-sqlite-idem")
    media_id = _insert_minimal_media(db)

    p1 = upsert_transcript(
        db,
        media_id,
        transcription="hello world",
        whisper_model="base",
        idempotency_key="stream-1",
    )
    p2 = upsert_transcript(
        db,
        media_id,
        transcription="hello world final",
        whisper_model="base",
        idempotency_key="stream-1",
    )

    rows = list(
        db.execute_query(
            """
            SELECT uuid, transcription, version, transcription_run_id, idempotency_key
            FROM Transcripts
            WHERE media_id = ?
            ORDER BY id ASC
            """,
            (media_id,),
        )
        or []
    )

    assert p2["uuid"] == p1["uuid"]
    assert p2["version"] == p1["version"] + 1
    assert p2["transcription_run_id"] == p1["transcription_run_id"] == 1
    assert len(rows) == 1
    assert rows[0]["transcription"] == "hello world final"
    assert rows[0]["idempotency_key"] == "stream-1"


@pytest.mark.unit
def test_upsert_transcript_retries_wrapped_unique_conflict(monkeypatch, tmp_path):
    db = MediaDatabase(db_path=str(tmp_path / "media.db"), client_id="unit-sqlite-retry")
    media_id = _insert_minimal_media(db)

    attempts = {"count": 0}
    real_upsert_once = legacy_transcripts._upsert_transcript_once

    def _fake_upsert_once(*args, **kwargs):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise DatabaseError(
                "Backend execute failed: duplicate key value violates unique constraint idx_transcripts_media_idempotency_key"
            )
        return real_upsert_once(*args, **kwargs)

    monkeypatch.setattr(legacy_transcripts, "_upsert_transcript_once", _fake_upsert_once)

    payload = upsert_transcript(
        db,
        media_id,
        transcription="hello world",
        whisper_model="base",
        idempotency_key="stream-1",
    )

    assert attempts["count"] == 2
    assert payload["transcription_run_id"] == 1
    assert payload["idempotency_key"] == "stream-1"


@pytest.mark.integration
def test_postgres_transaction_context_commits_if_available(tmp_path, pg_eval_params):
    cfg = DatabaseConfig(
        backend_type=BackendType.POSTGRESQL,
        pg_host=pg_eval_params["host"],
        pg_port=int(pg_eval_params["port"]),
        pg_database=pg_eval_params["database"],
        pg_user=pg_eval_params["user"],
        pg_password=pg_eval_params.get("password"),
    )
    try:
        backend = DatabaseBackendFactory.create_backend(cfg)
    except Exception:
        pytest.skip("psycopg backend not available")

    db = MediaDatabase(db_path=":memory:", client_id="txn-pg", backend=backend)

    inserted_uuid = str(uuid.uuid4())
    timestamp = db._get_current_utc_timestamp_str()
    insert_sql = (
        "INSERT INTO Media (url, title, type, content, content_hash, is_trash, chunking_status, "
        "vector_processing, uuid, last_modified, version, client_id, deleted) "
        "VALUES (?, ?, ?, ?, ?, 0, 'pending', 0, ?, ?, 1, ?, 0)"
    )

    with db.transaction() as conn:
        db._execute_with_connection(
            conn,
            insert_sql,
            (
                f"http://example.com/{inserted_uuid}",
                "Txn Commit Media",
                "article",
                "content",
                inserted_uuid,
                inserted_uuid,
                timestamp,
                db.client_id,
            ),
        )

    db.close_connection()

    with db.transaction() as conn:
        row = db._fetchone_with_connection(
            conn,
            "SELECT uuid FROM Media WHERE uuid = ?",
            (inserted_uuid,),
        )
        assert row is not None and row["uuid"] == inserted_uuid  # type: ignore[index]
        db._execute_with_connection(conn, "DELETE FROM Media WHERE uuid = ?", (inserted_uuid,))

    db.close_connection()

    failed_uuid = str(uuid.uuid4())
    with pytest.raises(RuntimeError):
        with db.transaction() as conn:
            db._execute_with_connection(
                conn,
                insert_sql,
                (
                    f"http://example.com/{failed_uuid}",
                    "Txn Rollback Media",
                    "article",
                    "content",
                    failed_uuid,
                    failed_uuid,
                    timestamp,
                    db.client_id,
                ),
            )
            raise RuntimeError("force rollback")

    db.close_connection()

    with db.transaction() as conn:
        row = db._fetchone_with_connection(
            conn,
            "SELECT uuid FROM Media WHERE uuid = ?",
            (failed_uuid,),
        )
        assert row is None
