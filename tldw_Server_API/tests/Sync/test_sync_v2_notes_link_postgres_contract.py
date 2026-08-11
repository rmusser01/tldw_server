"""Server-free PostgreSQL contracts for Notes link schema v58."""

from __future__ import annotations

import inspect
import os
from uuid import uuid4

import pytest

from tldw_Server_API.app.core.DB_Management.backends.base import DatabaseConfig, DatabaseError
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.chacha.note_link_store import (
    NotesLink,
    NotesLinkStore,
)
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.Sync.v2.materializers.notes_link import NotesLinkMaterializer

_LIVE_POSTGRES_DSN = os.getenv("TEST_DATABASE_URL") or os.getenv("POSTGRES_TEST_DSN")


def test_postgres_initializer_serializes_v58_before_policy_install() -> None:
    source = inspect.getsource(CharactersRAGDB._initialize_schema_postgres)

    assert "_get_schema_version_postgres(conn, lock=True)" in source
    migration = source.index("_migrate_from_v57_to_v58_postgres(conn)")
    policy = source.index("_ensure_chacha_rls_postgres(conn)")
    assert migration < policy


def test_postgres_v58_schema_uses_owner_and_live_query_indexes() -> None:
    source = inspect.getsource(CharactersRAGDB._migrate_from_v57_to_v58_postgres)

    assert "idx_note_edges_owner_live" in source
    assert "user_id, deleted" in source
    assert "idx_note_edges_from_live" in source
    assert "idx_note_edges_to_live" in source
    assert "WHERE deleted = FALSE" in source


def test_notes_link_store_queries_require_owner_on_edge_and_both_endpoints() -> None:
    get_source = inspect.getsource(NotesLinkStore._get_locked)
    list_source = inspect.getsource(NotesLinkStore.list_for_notes)
    endpoint_source = inspect.getsource(NotesLinkStore._validate_endpoints_locked)

    assert "edge.user_id = ?" in get_source
    assert "source.client_id = ?" in get_source
    assert "target.client_id = ?" in get_source
    assert "edge.user_id = ?" in list_source
    assert "source.client_id = ?" in list_source
    assert "target.client_id = ?" in list_source
    assert "SELECT id, client_id, deleted FROM notes" in endpoint_source
    assert set(NotesLink.__dataclass_fields__) == {
        "edge_id",
        "owner_user_id",
        "source_note_id",
        "target_note_id",
        "type",
        "directed",
        "weight",
        "label",
        "properties",
        "created_at",
        "last_modified",
        "created_by",
        "version",
        "deleted",
        "deleted_at",
    }
    materializer_source = inspect.getsource(NotesLinkMaterializer.apply)
    assert "NotesLinkStore(self.note_db)" in materializer_source
    assert ".upsert(" in materializer_source
    assert ".tombstone(" in materializer_source
    assert ".restore(" in materializer_source


@pytest.mark.integration
@pytest.mark.skipif(
    not str(_LIVE_POSTGRES_DSN or "").lower().startswith("postgres"),
    reason="A PostgreSQL TEST_DATABASE_URL or POSTGRES_TEST_DSN is required",
)
def test_postgres_notes_link_same_edge_scope_is_tenant_isolated_by_rls(
    pg_database_config: DatabaseConfig,
) -> None:
    owner_a = "920001"
    owner_b = "920002"
    source_a, target_a = str(uuid4()), str(uuid4())
    source_b, target_b = str(uuid4()), str(uuid4())
    edge_a = str(uuid4())
    backend_a = DatabaseBackendFactory.create_backend(pg_database_config)
    backend_b = DatabaseBackendFactory.create_backend(pg_database_config)
    db_a = CharactersRAGDB(db_path=":memory:", client_id=owner_a, backend=backend_a)
    db_b = CharactersRAGDB(db_path=":memory:", client_id=owner_b, backend=backend_b)
    try:
        for db, note_ids in ((db_a, (source_a, target_a)), (db_b, (source_b, target_b))):
            for note_id in note_ids:
                db.add_note(title=note_id, content="Body", note_id=note_id)
        with db_a.transaction() as conn:
            conn.execute(
                """
                INSERT INTO note_edges(
                  edge_id, user_id, from_note_id, to_note_id, type, directed,
                  weight, label, properties, created_at, last_modified,
                  created_by, version, deleted
                ) VALUES (?, ?, ?, ?, 'manual', 0, 1.0, NULL, '{}',
                          CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, 'device-a', 1, FALSE)
                """,
                (edge_a, owner_a, min(source_a, target_a), max(source_a, target_a)),
            )
        with db_b.transaction() as conn:
            assert conn.execute("SELECT edge_id FROM note_edges WHERE edge_id = ?", (edge_a,)).fetchone() is None
        with pytest.raises(DatabaseError):
            with db_b.transaction() as conn:
                conn.execute(
                    """
                    INSERT INTO note_edges(
                      edge_id, user_id, from_note_id, to_note_id, type, directed,
                      weight, label, properties, created_at, last_modified,
                      created_by, version, deleted
                    ) VALUES (?, ?, ?, ?, 'manual', 0, 1.0, NULL, '{}',
                              CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, 'device-b', 1, FALSE)
                    """,
                    (str(uuid4()), owner_b, min(source_a, target_a), max(source_a, target_a)),
                )
    finally:
        db_a.close_connection()
        db_b.close_connection()
