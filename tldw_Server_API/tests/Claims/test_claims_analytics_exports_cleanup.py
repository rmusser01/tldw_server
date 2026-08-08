from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase
from tldw_Server_API.app.core.DB_Management.db_path_utils import get_user_media_db_path
from tldw_Server_API.app.core.config import settings


def test_claims_analytics_exports_cleanup_and_list(monkeypatch, tmp_path):


    base_dir = tmp_path / "user_dbs"
    monkeypatch.setenv("USER_DB_BASE_DIR", str(base_dir))
    settings["USER_DB_BASE_DIR"] = str(base_dir)

    db_path = get_user_media_db_path(1)
    db = MediaDatabase(db_path=db_path, client_id="test")
    db.initialize_db()
    try:
        db.create_claims_analytics_export(
            export_id="old_export",
            user_id="1",
            format="json",
            status="ready",
            payload_json='{"events":[]}',
        )
        db.create_claims_analytics_export(
            export_id="new_export",
            user_id="1",
            format="json",
            status="ready",
            payload_json='{"events":[]}',
        )
        db.execute_query(
            "UPDATE claims_analytics_exports SET created_at = ?, updated_at = ? WHERE export_id = ?",
            (
                "2000-01-01T00:00:00.000Z",
                "2000-01-01T00:00:00.000Z",
                "old_export",
            ),
            commit=True,
        )

        deleted = db.cleanup_claims_analytics_exports(user_id="1", retention_hours=1)
        assert deleted >= 1

        rows = db.list_claims_analytics_exports(user_id="1", limit=10, offset=0)
        export_ids = {row.get("export_id") for row in rows}
        assert "new_export" in export_ids
        assert "old_export" not in export_ids

        total = db.count_claims_analytics_exports(user_id="1")
        assert total == len(rows)
    finally:
        try:
            db.close_connection()
        except Exception:
            _ = None
