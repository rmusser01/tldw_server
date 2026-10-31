"""Regression coverage for media ingested with a worker provenance client ID."""

import pytest

from tldw_Server_API.app.core.DB_Management.media_db.api import create_media_database
from tldw_Server_API.app.core.DB_Management.media_db.schema.sqlite_post_core_structures import (
    ensure_sqlite_visibility_columns,
)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("client_id", "owner_user_id", "expected_owner"),
    [
        ("media_ingest_worker:2", None, 2),
        ("media_ingest_worker:3", None, 3),
        ("media_ingest_worker:2", 3, 3),
        ("another_worker:2", None, None),
        ("media_ingest_worker:2-extra", None, None),
        ("media_ingest_worker:02", None, None),
        ("media_ingest_worker:0", None, None),
        ("media_ingest_worker:-2", None, None),
        ("media_ingest_worker:9223372036854775808", None, None),
    ],
)
def test_sqlite_bootstrap_recovers_only_unambiguous_worker_owners(
    tmp_path, client_id, owner_user_id, expected_owner
):
    db = create_media_database(client_id, db_path=str(tmp_path / "media.db"))
    try:
        media_id, _, _ = db.add_media_with_keywords(
            url="https://example.test/cedar",
            title="Project Cedar",
            media_type="document",
            content="Project Cedar launches in October.",
            keywords=[],
            owner_user_id=owner_user_id,
        )
        original_version = db.get_media_by_id(media_id)["version"]

        ensure_sqlite_visibility_columns(db, db.get_connection())
        first = db.get_media_by_id(media_id)
        ensure_sqlite_visibility_columns(db, db.get_connection())
        second = db.get_media_by_id(media_id)

        assert (first["owner_user_id"], second["owner_user_id"]) == (expected_owner, expected_owner)
        expected_version = original_version + (expected_owner != owner_user_id)
        assert (first["version"], second["version"]) == (expected_version, expected_version)
    finally:
        db.close_connection()
