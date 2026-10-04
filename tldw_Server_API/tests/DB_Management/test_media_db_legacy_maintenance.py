from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase
from tldw_Server_API.app.core.DB_Management.media_db import legacy_maintenance
from tldw_Server_API.app.core.DB_Management.media_db.legacy_maintenance import (
    empty_trash,
    permanently_delete_item,
)
from tldw_Server_API.app.core.DB_Management.media_db.repositories.media_repository import (
    MediaRepository,
)


def test_legacy_maintenance_empty_trash_soft_deletes_eligible_items() -> None:
    db = MediaDatabase(db_path=":memory:", client_id="legacy-maintenance-trash")
    media_repo = MediaRepository.from_legacy_db(db)
    try:
        media_id, _media_uuid, _msg = media_repo.add_text_media(
            title="Trash candidate",
            content="discard me",
            media_type="text",
        )

        assert db.mark_as_trash(media_id) is True

        processed, remaining = empty_trash(db, 0)
        media_row = db.execute_query(
            "SELECT deleted, is_trash, client_id FROM Media WHERE id = ?",
            (media_id,),
        ).fetchone()

        assert processed == 1
        assert remaining == 0
        assert media_row is not None
        assert media_row["deleted"] == 1
        assert media_row["is_trash"] == 1
        assert media_row["client_id"] == "legacy-maintenance-trash"
    finally:
        db.close_connection()


def test_legacy_maintenance_permanent_delete_removes_media_and_keyword_links() -> None:
    db = MediaDatabase(db_path=":memory:", client_id="legacy-maintenance-delete")
    media_repo = MediaRepository.from_legacy_db(db)
    try:
        media_id, _media_uuid, _msg = media_repo.add_text_media(
            title="Hard delete candidate",
            content="erase me",
            media_type="text",
            keywords=["alpha", "beta"],
        )

        deleted = permanently_delete_item(db, media_id)
        media_row = db.execute_query(
            "SELECT COUNT(*) AS total FROM Media WHERE id = ?",
            (media_id,),
        ).fetchone()
        keyword_row = db.execute_query(
            "SELECT COUNT(*) AS total FROM MediaKeywords WHERE media_id = ?",
            (media_id,),
        ).fetchone()

        assert deleted is True
        assert media_row is not None
        assert media_row["total"] == 0
        assert keyword_row is not None
        assert keyword_row["total"] == 0
        assert not hasattr(legacy_maintenance, "check_media_and_whisper_model")
    finally:
        db.close_connection()
