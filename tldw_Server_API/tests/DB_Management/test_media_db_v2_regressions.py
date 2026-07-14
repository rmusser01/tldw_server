from pathlib import Path

from tldw_Server_API.app.core.DB_Management.media_db import media_database, native_class


def test_native_media_database_exports_resolve_same_class() -> None:
    assert native_class.MediaDatabase is media_database.MediaDatabase


def test_set_media_vector_embedding_versions_row_and_logs_sanitized_payload(
    tmp_path: Path,
) -> None:
    db = native_class.MediaDatabase(
        db_path=str(tmp_path / "media.db"),
        client_id="vector-contract-test",
    )
    try:
        media_id, media_uuid, _ = db.add_media_with_keywords(
            title="Vector contract",
            media_type="document",
            content="vector payload",
            keywords=[],
        )
        before = db.get_media_by_id(int(media_id))
        assert before is not None
        before_version = int(before["version"])

        result = db.set_media_vector_embedding(int(media_id), b"\x00fixture-vector\xff")

        after = db.get_media_by_id(int(media_id))
        assert after is not None
        assert bytes(after["vector_embedding"]) == b"\x00fixture-vector\xff"
        assert int(after["version"]) == before_version + 1
        assert result == {
            "media_id": int(media_id),
            "media_uuid": str(media_uuid),
            "version": before_version + 1,
        }
        sync_event = db.get_sync_log_entries()[-1]
        assert sync_event["entity"] == "Media"
        assert sync_event["entity_uuid"] == str(media_uuid)
        assert sync_event["operation"] == "update"
        assert sync_event["version"] == before_version + 1
        assert sync_event["payload"]["id"] == int(media_id)
        assert sync_event["payload"]["uuid"] == str(media_uuid)
        assert sync_event["payload"]["version"] == before_version + 1
        assert sync_event["payload"]["client_id"] == "vector-contract-test"
        assert "vector_embedding" not in sync_event["payload"]
    finally:
        db.close_connection()
