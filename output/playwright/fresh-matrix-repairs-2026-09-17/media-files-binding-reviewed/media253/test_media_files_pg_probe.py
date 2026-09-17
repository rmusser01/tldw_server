"""Private official-fixture reproduction; no held native databases."""
import pytest

from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase
from tldw_Server_API.app.core.DB_Management.media_db.services.media_details_service import get_full_media_details_rich


@pytest.fixture
def media_db(pg_database_config):
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    db = MediaDatabase(":memory:", client_id="1", backend=backend)
    try:
        yield db
    finally:
        db.close_connection()
        backend.get_pool().close_all()


def test_absent_original_lookup_is_successful(media_db):
    assert media_db.get_media_file(1, "original") is None


def test_rich_detail_preserves_plaintext_without_original(media_db):
    media_id, _, _ = media_db.add_media_with_keywords(
        title="Synthetic Media253", content="Public synthetic source.",
        media_type="document", url="file:///synthetic-media253.txt", keywords=[]
    )
    detail = get_full_media_details_rich(media_db, media_id)
    assert detail["has_original_file"] is False


def test_register_original_is_successful(media_db):
    media_id, _, _ = media_db.add_media_with_keywords(
        title="Synthetic Media253 original", content="Public synthetic source.",
        media_type="document", url="file:///synthetic-media253-original.txt", keywords=[]
    )
    file_uuid = media_db.insert_media_file(media_id, "original", "synthetic/original.txt")
    assert file_uuid
