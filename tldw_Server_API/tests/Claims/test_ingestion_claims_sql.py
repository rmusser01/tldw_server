import os
import sqlite3
import tempfile

import psycopg

from tldw_Server_API.app.core.Chunking import chunk_for_embedding
from tldw_Server_API.app.core.Claims_Extraction import ingestion_claims
from tldw_Server_API.app.core.Claims_Extraction.ingestion_claims import (
    extract_claims_for_chunks,
    store_claims,
)
from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase


def test_ingestion_time_claims_extract_and_store_sql():


    # Setup temp DB
    temp_dir = tempfile.mkdtemp(prefix="claims_sql_")
    db_path = os.path.join(temp_dir, "media.db")
    db = MediaDatabase(db_path=db_path, client_id="test_client")
    db.initialize_db()
    try:
        # Create media row
        content = "Hello world. This is a test document. It contains a few sentences."
        media_id, media_uuid, _ = db.add_media_with_keywords(
            title="Doc",
            media_type="text",
            content=content,
            keywords=None,
        )
        assert media_id is not None

        # Chunk and extract
        chunks = chunk_for_embedding(content, file_name="doc.txt", max_size=50)
        claims = extract_claims_for_chunks(chunks, extractor_mode="heuristic", max_per_chunk=2)
        assert claims, "No claims extracted"

        # Build chunk_index->text map
        chunk_text_map = {int(ch.get("metadata", {}).get("chunk_index", 0)): ch.get("text", "") for ch in chunks}
        inserted = store_claims(db, media_id=media_id, chunk_texts_by_index=chunk_text_map, claims=claims)
        assert inserted == len(claims)

        # Verify fetch
        rows = db.get_claims_by_media(media_id)
        assert len(rows) == inserted
        assert any("Hello world" in r["claim_text"] or "test document" in r["claim_text"] for r in rows)
        assert any(r.get("span_start") is not None and r.get("span_end") is not None for r in rows)
    finally:
        try:
            db.close_connection()
        except Exception:
            _ = None


def test_ingestion_time_claims_extract_auto_multilingual():


    temp_dir = tempfile.mkdtemp(prefix="claims_sql_")
    db_path = os.path.join(temp_dir, "media.db")
    db = MediaDatabase(db_path=db_path, client_id="test_client")
    db.initialize_db()
    try:
        content = "这是一个比较长的中文句子。这里还有另一个比较长的句子。"
        media_id, media_uuid, _ = db.add_media_with_keywords(
            title="Doc",
            media_type="text",
            content=content,
            keywords=None,
        )
        assert media_id is not None

        chunks = [{"text": content, "metadata": {"chunk_index": 0}}]
        claims = extract_claims_for_chunks(chunks, extractor_mode="auto", max_per_chunk=2)
        assert claims, "No claims extracted for auto multilingual path"

        chunk_text_map = {0: content}
        inserted = store_claims(db, media_id=media_id, chunk_texts_by_index=chunk_text_map, claims=claims)
        assert inserted == len(claims)

        rows = db.get_claims_by_media(media_id)
        assert len(rows) == inserted
    finally:
        try:
            db.close_connection()
        except Exception:
            _ = None


def test_store_claims_review_assignment_notifications_use_jobs(monkeypatch):
    from tldw_Server_API.app.core.Claims_Extraction import claims_jobs, claims_notifications

    temp_dir = tempfile.mkdtemp(prefix="claims_assignment_jobs_")
    db_path = os.path.join(temp_dir, "media.db")
    db = MediaDatabase(db_path=db_path, client_id="1")
    db.initialize_db()
    enqueued: list[dict[str, object]] = []

    def _assign_review(*, db, claims):
        del db
        return [{**claim, "reviewer_id": 9} for claim in claims]

    monkeypatch.setattr(ingestion_claims, "apply_review_rules", _assign_review)
    monkeypatch.setattr(claims_jobs, "claims_jobs_enabled", lambda: True)
    monkeypatch.setattr(
        claims_jobs,
        "enqueue_claims_review_notification",
        lambda **kwargs: enqueued.append(kwargs) or {"id": 77},
    )
    monkeypatch.setattr(
        claims_notifications,
        "record_review_assignment_notifications",
        lambda **_kwargs: [101, 102],
    )
    monkeypatch.setattr(
        claims_notifications,
        "dispatch_claim_review_notifications",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("legacy dispatch should not run")),
    )

    try:
        media_id, _, _ = db.add_media_with_keywords(
            title="Doc",
            media_type="text",
            content="A. B.",
            keywords=None,
            owner_user_id=1,
        )
        inserted = store_claims(
            db,
            media_id=media_id,
            chunk_texts_by_index={0: "A. B."},
            claims=[{"chunk_index": 0, "claim_text": "A."}],
        )
    finally:
        db.close_connection()

    assert inserted == 1
    assert enqueued == [{"owner_user_id": "1", "notification_ids": [101, 102]}]


def test_store_claims_review_assignment_job_sqlite_enqueue_failure_does_not_rollback(monkeypatch):
    from tldw_Server_API.app.core.Claims_Extraction import claims_jobs, claims_notifications

    temp_dir = tempfile.mkdtemp(prefix="claims_assignment_jobs_failure_")
    db_path = os.path.join(temp_dir, "media.db")
    db = MediaDatabase(db_path=db_path, client_id="1")
    db.initialize_db()

    def _assign_review(*, db, claims):
        del db
        return [{**claim, "reviewer_id": 9} for claim in claims]

    monkeypatch.setattr(ingestion_claims, "apply_review_rules", _assign_review)
    monkeypatch.setattr(claims_jobs, "claims_jobs_enabled", lambda: True)
    monkeypatch.setattr(
        claims_jobs,
        "enqueue_claims_review_notification",
        lambda **_kwargs: (_ for _ in ()).throw(sqlite3.Error("jobs database unavailable")),
    )
    monkeypatch.setattr(
        claims_notifications,
        "record_review_assignment_notifications",
        lambda **_kwargs: [101],
    )
    legacy_dispatch_calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        claims_notifications,
        "dispatch_claim_review_notifications",
        lambda **kwargs: legacy_dispatch_calls.append(kwargs),
    )

    try:
        media_id, _, _ = db.add_media_with_keywords(
            title="Doc",
            media_type="text",
            content="A. B.",
            keywords=None,
            owner_user_id=1,
        )
        inserted = store_claims(
            db,
            media_id=media_id,
            chunk_texts_by_index={0: "A. B."},
            claims=[{"chunk_index": 0, "claim_text": "A."}],
        )
        rows = db.get_claims_by_media(media_id)
    finally:
        db.close_connection()

    assert inserted == 1
    assert len(rows) == 1
    assert legacy_dispatch_calls == [
        {
            "db_path": str(db.db_path_str),
            "owner_user_id": "1",
            "notification_ids": [101],
        }
    ]


def test_store_claims_review_assignment_job_pg_enqueue_failure_does_not_rollback(monkeypatch):
    from tldw_Server_API.app.core.Claims_Extraction import claims_jobs, claims_notifications

    temp_dir = tempfile.mkdtemp(prefix="claims_assignment_jobs_pg_failure_")
    db_path = os.path.join(temp_dir, "media.db")
    db = MediaDatabase(db_path=db_path, client_id="1")
    db.initialize_db()

    def _assign_review(*, db, claims):
        del db
        return [{**claim, "reviewer_id": 9} for claim in claims]

    monkeypatch.setattr(ingestion_claims, "apply_review_rules", _assign_review)
    monkeypatch.setattr(claims_jobs, "claims_jobs_enabled", lambda: True)
    monkeypatch.setattr(
        claims_jobs,
        "enqueue_claims_review_notification",
        lambda **_kwargs: (_ for _ in ()).throw(psycopg.Error("jobs database unavailable")),
    )
    monkeypatch.setattr(
        claims_notifications,
        "record_review_assignment_notifications",
        lambda **_kwargs: [101],
    )
    legacy_dispatch_calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        claims_notifications,
        "dispatch_claim_review_notifications",
        lambda **kwargs: legacy_dispatch_calls.append(kwargs),
    )

    try:
        media_id, _, _ = db.add_media_with_keywords(
            title="Doc",
            media_type="text",
            content="A. B.",
            keywords=None,
            owner_user_id=1,
        )
        inserted = store_claims(
            db,
            media_id=media_id,
            chunk_texts_by_index={0: "A. B."},
            claims=[{"chunk_index": 0, "claim_text": "A."}],
        )
        rows = db.get_claims_by_media(media_id)
    finally:
        db.close_connection()

    assert inserted == 1
    assert len(rows) == 1
    assert legacy_dispatch_calls == [
        {
            "db_path": str(db.db_path_str),
            "owner_user_id": "1",
            "notification_ids": [101],
        }
    ]
