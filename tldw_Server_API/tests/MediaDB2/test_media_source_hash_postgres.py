"""PostgreSQL integration coverage for the local upload source-hash lookup."""

import pytest

from tldw_Server_API.app.core.DB_Management.backends.base import BackendType, DatabaseConfig
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase
from tldw_Server_API.app.core.DB_Management.media_db.repositories.media_lookup_repository import MediaLookupRepository
from tldw_Server_API.app.core.DB_Management.scope_context import scoped_context

pytest_plugins = ["tldw_Server_API.tests._plugins.authnz_full_fixtures"]


@pytest.mark.integration
@pytest.mark.postgres
@pytest.mark.usefixtures("setup_test_database")
def test_postgres_local_upload_source_hash_preserves_owner_and_stored_hash() -> None:
    """Check both lookup paths and owner filtering under unrestricted visibility."""
    config = DatabaseConfig.from_env()
    assert config.backend_type == BackendType.POSTGRESQL
    backend = DatabaseBackendFactory.create_backend(config)
    owner_db = MediaDatabase(db_path=":memory:", client_id="101", backend=backend)
    other_db = MediaDatabase(db_path=":memory:", client_id="202", backend=backend)
    historical_hash = "a" * 64
    indexed_hash = "b" * 64
    metadata = f'{{"source_hash": "{historical_hash}"}}'
    try:
        legacy_id, _, _ = owner_db.add_media_with_keywords(
            url="legacy.wav", media_type="audio", content="legacy transcript", safe_metadata=metadata,
        )
        indexed_id, _, _ = owner_db.add_media_with_keywords(
            url="indexed.wav", media_type="audio", content="indexed transcript", safe_metadata=metadata,
            source_hash=indexed_hash,
        )
        assert legacy_id is not None and indexed_id is not None
        owner_lookup = MediaLookupRepository(owner_db).local_upload_id_by_source_hash
        other_lookup = MediaLookupRepository(other_db).local_upload_id_by_source_hash
        assert owner_lookup("legacy.wav", historical_hash, source_hash_column_available=True) == legacy_id
        assert owner_lookup("indexed.wav", indexed_hash, source_hash_column_available=True) == indexed_id
        assert owner_lookup("indexed.wav", historical_hash, source_hash_column_available=True) is None
        with scoped_context(user_id=202, is_admin=True):
            visible = other_db.execute_query(
                "SELECT id FROM Media WHERE id IN (?, ?)", (legacy_id, indexed_id),
            ).fetchall()
            assert {row["id"] for row in visible} == {legacy_id, indexed_id}
            assert other_lookup("legacy.wav", historical_hash, source_hash_column_available=True) is None
            assert other_lookup("indexed.wav", indexed_hash, source_hash_column_available=True) is None
    finally:
        owner_db.close_connection()
        other_db.close_connection()
