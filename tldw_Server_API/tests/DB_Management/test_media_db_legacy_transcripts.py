import tldw_Server_API.app.core.Metrics.metrics_manager as metrics_manager
from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase
from tldw_Server_API.app.core.DB_Management.media_db import legacy_reads
from tldw_Server_API.app.core.DB_Management.media_db.legacy_reads import (
    get_latest_transcription,
    get_media_transcripts,
)
from tldw_Server_API.app.core.DB_Management.media_db.legacy_transcripts import (
    soft_delete_transcript,
    upsert_transcript,
)
from tldw_Server_API.app.core.DB_Management.media_db.repositories.media_repository import (
    MediaRepository,
)


def test_legacy_transcript_helpers_round_trip_run_history_and_soft_delete() -> None:
    db = MediaDatabase(db_path=":memory:", client_id="legacy-transcript-helpers")
    media_repo = MediaRepository.from_legacy_db(db)
    try:
        media_id, _media_uuid, _msg = media_repo.add_text_media(
            title="Transcripted doc",
            content="spoken words",
            media_type="audio",
        )

        created = upsert_transcript(
            db,
            media_id=media_id,
            transcription='{"text": "first"}',
            whisper_model="base",
        )
        updated = upsert_transcript(
            db,
            media_id=media_id,
            transcription='{"text": "second"}',
            whisper_model="base",
        )

        deleted = soft_delete_transcript(db, updated["uuid"])
        latest_after_delete = get_latest_transcription(db, media_id)
        transcripts = get_media_transcripts(db, media_id)

        transcript_row = db.execute_query(
            """
            SELECT deleted, version, client_id, transcription, transcription_run_id, idempotency_key
            FROM Transcripts
            WHERE uuid = ?
            """,
            (updated["uuid"],),
        ).fetchone()
        created_row = db.execute_query(
            """
            SELECT deleted, version, transcription_run_id
            FROM Transcripts
            WHERE uuid = ?
            """,
            (created["uuid"],),
        ).fetchone()
        media_row = db.execute_query(
            "SELECT latest_transcription_run_id FROM Media WHERE id = ?",
            (media_id,),
        ).fetchone()
        sync_row = db.execute_query(
            """
            SELECT entity, operation, version, client_id
            FROM sync_log
            WHERE entity_uuid = ?
            ORDER BY change_id DESC
            LIMIT 1
            """,
            (updated["uuid"],),
        ).fetchone()

        assert updated["uuid"] != created["uuid"]
        assert created["version"] == 1
        assert updated["version"] == 1
        assert created["transcription_run_id"] == 1
        assert updated["transcription_run_id"] == 2
        assert deleted is True
        assert latest_after_delete == "first"
        assert len(transcripts) == 1
        assert transcripts[0]["uuid"] == created["uuid"]
        assert transcript_row is not None
        assert transcript_row["deleted"] == 1
        assert transcript_row["version"] == 2
        assert transcript_row["client_id"] == "legacy-transcript-helpers"
        assert transcript_row["transcription_run_id"] == 2
        assert transcript_row["idempotency_key"] is None
        assert "second" in transcript_row["transcription"]
        assert created_row is not None
        assert created_row["deleted"] == 0
        assert created_row["version"] == 1
        assert media_row is not None
        assert media_row["latest_transcription_run_id"] == 2
        assert sync_row is not None
        assert sync_row["entity"] == "Transcripts"
        assert sync_row["operation"] == "delete"
        assert sync_row["version"] == transcript_row["version"]
        assert sync_row["client_id"] == "legacy-transcript-helpers"
    finally:
        db.close_connection()


def test_get_latest_transcription_prefers_media_latest_run_pointer() -> None:
    db = MediaDatabase(db_path=":memory:", client_id="legacy-transcript-latest")
    media_repo = MediaRepository.from_legacy_db(db)
    try:
        media_id, _media_uuid, _msg = media_repo.add_text_media(
            title="Transcript ordering doc",
            content="spoken words",
            media_type="audio",
        )
        first = upsert_transcript(
            db,
            media_id=media_id,
            transcription='{"text": "first"}',
            whisper_model="base",
        )
        second = upsert_transcript(
            db,
            media_id=media_id,
            transcription='{"text": "second"}',
            whisper_model="base",
        )

        media_row = db.execute_query(
            "SELECT version FROM Media WHERE id = ?",
            (media_id,),
        ).fetchone()
        assert media_row is not None
        db.execute_query(
            """
            UPDATE Media
            SET latest_transcription_run_id = ?, version = ?, client_id = ?, last_modified = ?
            WHERE id = ?
            """,
            (
                first["transcription_run_id"],
                int(media_row["version"]) + 1,
                db.client_id,
                db._get_current_utc_timestamp_str(),
                media_id,
            ),
        )

        latest = get_latest_transcription(db, media_id)
        transcripts = get_media_transcripts(db, media_id)

        assert latest == "first"
        assert [row["uuid"] for row in transcripts] == [first["uuid"], second["uuid"]]
    finally:
        db.close_connection()


def test_get_latest_transcription_bounds_fallback_telemetry(monkeypatch) -> None:
    db = MediaDatabase(db_path=":memory:", client_id="legacy-transcript-fallback")
    media_repo = MediaRepository.from_legacy_db(db)
    metric_calls: list[dict[str, str]] = []
    warning_calls: list[tuple[object, ...]] = []

    legacy_reads._latest_run_fallback_cache.clear()
    legacy_reads._latest_run_fallback_cache_order.clear()
    monkeypatch.setattr(
        legacy_reads,
        "increment_counter",
        lambda _metric_name, labels=None: metric_calls.append(dict(labels or {})),
    )
    monkeypatch.setattr(
        legacy_reads.logger,
        "warning",
        lambda *args, **kwargs: warning_calls.append(args),
    )
    try:
        media_id, _media_uuid, _msg = media_repo.add_text_media(
            title="Transcript fallback doc",
            content="spoken words",
            media_type="audio",
        )
        latest = upsert_transcript(
            db,
            media_id=media_id,
            transcription='{"text": "latest"}',
            whisper_model="base",
        )
        db.execute_query(
            """
            UPDATE Media
            SET latest_transcription_run_id = ?, version = version + 1, client_id = ?, last_modified = ?
            WHERE id = ?
            """,
            (
                int(latest["transcription_run_id"]) + 99,
                db.client_id,
                db._get_current_utc_timestamp_str(),
                media_id,
            ),
        )

        first = get_latest_transcription(db, media_id)
        second = get_latest_transcription(db, media_id)

        assert first == "latest"
        assert second == "latest"
        assert len(metric_calls) == 1
        assert metric_calls[0]["component"] == "media_db"
        assert metric_calls[0]["event"] == "latest_transcript_run_fallback"
        assert metric_calls[0]["reason"] == "dangling_pointer"
        assert len(warning_calls) == 1
    finally:
        db.close_connection()


def test_get_latest_transcription_emits_bounded_read_path_metrics() -> None:
    metrics_manager._metrics_registry = None
    registry = metrics_manager.get_metrics_registry()
    db = MediaDatabase(db_path=":memory:", client_id="legacy-transcript-read-path")
    media_repo = MediaRepository.from_legacy_db(db)
    try:
        media_id, _media_uuid, _msg = media_repo.add_text_media(
            title="Transcript read path doc",
            content="spoken words",
            media_type="audio",
        )
        latest = upsert_transcript(
            db,
            media_id=media_id,
            transcription='{"text": "latest"}',
            whisper_model="base",
        )

        assert get_latest_transcription(db, media_id) == "latest"
        assert registry.get_cumulative_counter(
            "audio_stt_transcript_read_path_total",
            {"path": "latest_run"},
        ) == 1

        db.execute_query(
            """
            UPDATE Media
            SET latest_transcription_run_id = ?, version = version + 1, client_id = ?, last_modified = ?
            WHERE id = ?
            """,
            (
                int(latest["transcription_run_id"]) + 99,
                db.client_id,
                db._get_current_utc_timestamp_str(),
                media_id,
            ),
        )

        assert get_latest_transcription(db, media_id) == "latest"
        assert registry.get_cumulative_counter(
            "audio_stt_transcript_read_path_total",
            {"path": "legacy_fallback"},
        ) == 1
    finally:
        db.close_connection()
        metrics_manager._metrics_registry = None


def test_soft_deleted_idempotent_transcript_is_not_resurrected() -> None:
    db = MediaDatabase(db_path=":memory:", client_id="legacy-transcript-delete-retry")
    media_repo = MediaRepository.from_legacy_db(db)
    try:
        media_id, _media_uuid, _msg = media_repo.add_text_media(
            title="Transcript retry doc",
            content="spoken words",
            media_type="audio",
        )
        original = upsert_transcript(
            db,
            media_id=media_id,
            transcription='{"text": "first"}',
            whisper_model="base",
            idempotency_key="stream-1",
        )
        assert soft_delete_transcript(db, original["uuid"]) is True

        retried = upsert_transcript(
            db,
            media_id=media_id,
            transcription='{"text": "retry"}',
            whisper_model="base",
            idempotency_key="stream-1",
        )

        rows = list(
            db.execute_query(
                """
                SELECT uuid, deleted, transcription_run_id, idempotency_key
                FROM Transcripts
                WHERE media_id = ?
                ORDER BY id ASC
                """,
                (media_id,),
            )
            or []
        )

        assert retried["uuid"] != original["uuid"]
        assert retried["transcription_run_id"] == 2
        assert len(rows) == 2
        assert rows[0]["uuid"] == original["uuid"]
        assert rows[0]["deleted"] == 1
        assert rows[0]["idempotency_key"] is None
        assert rows[1]["uuid"] == retried["uuid"]
        assert rows[1]["deleted"] == 0
        assert rows[1]["idempotency_key"] == "stream-1"
        assert get_latest_transcription(db, media_id) == "retry"
    finally:
        db.close_connection()
