"""Count only exportable media vectors under the normal owner boundary."""

from contextlib import ExitStack

import pytest

from tldw_Server_API.app.core.DB_Management.backends.base import BackendType, DatabaseConfig
from tldw_Server_API.app.core.DB_Management.backends.sqlite_backend import SQLiteBackend
from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase
from tldw_Server_API.app.core.DB_Management.scope_context import scoped_context

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("storage", ["sqlite", pytest.param("postgres", marks=pytest.mark.postgres)])
def test_media_vector_scope_counts_active_owned_payloads(storage, request, tmp_path):
    shared = request.getfixturevalue("pg_restricted_backend") if storage == "postgres" else None
    handles = {}
    with ExitStack() as cleanup:
        for owner, active_vectors in ((101, 1), (202, 2)):
            path = str(tmp_path / f"owner-{owner}.db")
            backend = shared or SQLiteBackend(DatabaseConfig(backend_type=BackendType.SQLITE, sqlite_path=path))
            if shared is None:
                cleanup.callback(backend.get_pool().close_all)
            with scoped_context(user_id=owner, is_admin=False):
                db = MediaDatabase(db_path=path, client_id=str(owner), backend=backend)
                cleanup.callback(db.close_connection)
                handles[owner] = db
                seen = set()
                for index in range(active_vectors + 3):
                    media_id, _, _ = db.add_media_with_keywords(
                        url=f"https://example.test/s02/{owner}/{index}",
                        title=f"Owner {owner} vector {index}",
                        media_type="document",
                        content=f"Owner {owner}, record {index}. Rowan Observatory uses Selene.",
                        keywords=[],
                        owner_user_id=owner,
                    )
                    assert media_id not in seen  # nosec B101 -- Independent fixture rows required.
                    seen.add(media_id)
                    if index != active_vectors:
                        db.set_media_vector_embedding(int(media_id), bytes.fromhex("0000803f00000000"))
                    if index == active_vectors + 1:
                        assert db.mark_as_trash(int(media_id))  # nosec B101 -- Pytest behavior assertion.
                    elif index == active_vectors + 2:
                        # Seed a tombstone through the admin scope; owner deletion is tested separately.
                        if shared is not None:
                            with scoped_context(user_id=owner, is_admin=True):
                                assert db.soft_delete_media(int(media_id))  # nosec B101 -- Fixture-state assertion.
                        else:
                            assert db.soft_delete_media(int(media_id))  # nosec B101 -- Fixture-state assertion.
                db.close_connection()
        for owner, expected in ((101, 1), (202, 2)):
            with scoped_context(user_id=owner, is_admin=False):
                assert handles[owner].count_chatbook_scope_category("media_vector_embeddings") == expected  # nosec B101 -- Pytest behavior assertion.
                handles[owner].close_connection()
        if shared is not None:
            with scoped_context(user_id=303, is_admin=False):
                assert handles[101].count_chatbook_scope_category("media_vector_embeddings") == 0  # nosec B101 -- Pytest behavior assertion.
                handles[101].close_connection()
