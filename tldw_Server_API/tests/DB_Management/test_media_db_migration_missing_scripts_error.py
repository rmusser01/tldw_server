"""Tests for MediaDatabase upgrade diagnostics when migration scripts are missing."""

import sqlite3

import pytest

from tldw_Server_API.app.core.DB_Management.media_db.errors import DatabaseError
from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase


@pytest.mark.unit
def test_media_db_upgrade_from_pre_v22_reports_unsupported_legacy_schema(tmp_path):
    db_path = tmp_path / "Media_DB_v2.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE schema_version (version INTEGER)")
        conn.execute("INSERT INTO schema_version (version) VALUES (8)")
        conn.commit()

    with pytest.raises(DatabaseError) as exc_info:
        MediaDatabase(db_path=str(db_path), client_id="legacy-version-test")

    msg = str(exc_info.value)
    assert "unsupported legacy Media DB schema version 8" in msg
    assert "supported automatic migration starts at version 22" in msg
    assert "backup" in msg.lower()
