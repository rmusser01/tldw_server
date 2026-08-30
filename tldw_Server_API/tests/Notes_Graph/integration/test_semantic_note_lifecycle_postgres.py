"""Live PostgreSQL contracts for canonical Note semantic lifecycle fences."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pytest

from tldw_Server_API.app.core.DB_Management.backends.base import DatabaseConfig
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.chacha.note_semantic_models import (
    SemanticDimensionState,
)
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.Notes_Graph.suggestion_content import content_fingerprint

pytestmark = [pytest.mark.integration, pytest.mark.timeout(30)]

OWNER_ID = "semantic-postgres-owner"
OTHER_OWNER_ID = "semantic-postgres-other-owner"
DATASET_ID = "semantic-postgres-dataset"
NOTE_ID = "semantic-postgres-note"
ROLLED_BACK_NOTE_ID = "semantic-postgres-rolled-back-note"
NOW = datetime(2026, 8, 29, 12, 0, tzinfo=timezone.utc)


def _activate_semantic_generation(db: CharactersRAGDB) -> str:
    config = db.note_semantic_store.create_configuration(
        dataset_id=DATASET_ID,
        capability_revision="capability-v1",
        disclosure_hash="disclosure-v1",
        provider="provider-a",
        model="model-a",
        endpoint_origin_revision="origin-v1",
        endpoint_origin_display="https://api.example.test",
        data_boundary="provider",
        vector_backend="chromadb",
        storage_boundary="server_local",
        storage_label="local semantic vectors",
        normalization_version="normalization-v1",
        chunker_version="chunker-v1",
        now=NOW,
    )
    enabled = db.note_semantic_store.enable_configuration(
        dataset_id=DATASET_ID,
        expected_configuration_revision=config.configuration_revision,
        capability_revision="capability-v1",
        now=NOW,
    )
    assert enabled is not None
    generation = db.note_semantic_store.create_generation(
        dataset_id=DATASET_ID,
        configuration_revision=enabled.configuration_revision,
        compatibility_hash=None,
        dimension_state=SemanticDimensionState.PENDING,
        dimensions=None,
        root_job_id="job-postgres",
        now=NOW,
    )
    resolved = db.note_semantic_store.resolve_generation_dimensions(
        dataset_id=DATASET_ID,
        generation_id=generation.id,
        expected_configuration_revision=enabled.configuration_revision,
        dimensions=768,
        compatibility_hash="compatibility-v1",
        now=NOW,
    )
    assert resolved is not None
    active = db.note_semantic_store.activate_generation(
        dataset_id=DATASET_ID,
        generation_id=generation.id,
        expected_configuration_revision=resolved.configuration_revision,
        publication_receipt="receipt-postgres",
        now=NOW,
    )
    assert active is not None
    return generation.id


def _scope(conn) -> None:
    conn.execute("SELECT set_config('app.current_dataset_id', ?, true)", (DATASET_ID,))


def _state_and_work(db: CharactersRAGDB, note_id: str):
    with db.transaction() as conn:
        _scope(conn)
        state = conn.execute(
            "SELECT content_version,dirty_generation,state FROM note_semantic_note_state "
            "WHERE owner_user_id=? AND dataset_id=? AND note_id=?",
            (OWNER_ID, DATASET_ID, note_id),
        ).fetchone()
        work = conn.execute(
            "SELECT kind,generation_id,dirty_generation FROM note_semantic_work "
            "WHERE owner_user_id=? AND dataset_id=? AND note_id=? ORDER BY kind",
            (OWNER_ID, DATASET_ID, note_id),
        ).fetchall()
    state_values = (
        None
        if state is None
        else (state["content_version"], state["dirty_generation"], state["state"])
    )
    work_values = [
        (row["kind"], row["generation_id"], row["dirty_generation"])
        for row in work
    ]
    return state_values, work_values


def test_postgres_note_lifecycle_is_transactional_rls_scoped_and_coalesced(
    pg_database_config: DatabaseConfig,
) -> None:
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    db = CharactersRAGDB(":memory:", client_id=OWNER_ID, backend=backend)
    ident = backend.escape_identifier  # type: ignore[attr-defined]
    role_name = f"semantic_note_rls_{uuid4().hex[:8]}"
    role_created = False
    try:
        generation_id = _activate_semantic_generation(db)

        assert db.note_store.add_note(
            "Original",
            "Body",
            note_id=NOTE_ID,
            semantic_dataset_id=DATASET_ID,
        ) == NOTE_ID
        state, work = _state_and_work(db, NOTE_ID)
        assert state == (1, 1, "pending")
        assert work == [("index_note", generation_id, 1)]

        assert db.note_store.update_note(
            NOTE_ID,
            {"title": "Revised"},
            expected_version=1,
            semantic_dataset_id=DATASET_ID,
        )
        state, work = _state_and_work(db, NOTE_ID)
        assert state == (2, 2, "pending")
        assert work == [("index_note", generation_id, 2)]

        assert db.note_store.soft_delete_note(
            NOTE_ID,
            expected_version=2,
            semantic_dataset_id=DATASET_ID,
        )
        state, work = _state_and_work(db, NOTE_ID)
        assert state == (3, 3, "tombstoned")
        assert work == [("delete_note_vectors", generation_id, 3)]

        assert db.note_store.restore_note(
            NOTE_ID,
            expected_version=3,
            semantic_dataset_id=DATASET_ID,
        )
        state, work = _state_and_work(db, NOTE_ID)
        assert state == (4, 4, "pending")
        assert work == [("index_note", generation_id, 4)]

        with backend.transaction() as conn:
            backend.execute(
                f"CREATE ROLE {ident(role_name)} NOLOGIN NOSUPERUSER NOBYPASSRLS",
                connection=conn,
            )
            backend.execute(
                f"GRANT USAGE ON SCHEMA public TO {ident(role_name)}",
                connection=conn,
            )
            backend.execute(
                "GRANT SELECT ON note_semantic_index_configs,note_semantic_generations,"
                "note_semantic_note_state,note_semantic_chunks,note_semantic_work "
                f"TO {ident(role_name)}",
                connection=conn,
            )
            backend.execute(f"GRANT {ident(role_name)} TO CURRENT_USER", connection=conn)
        role_created = True

        with backend.transaction() as conn:
            backend.execute(f"SET LOCAL ROLE {ident(role_name)}", connection=conn)
            backend.execute(
                "SELECT set_config('app.current_user_id', ?, true)",
                (OWNER_ID,),
                connection=conn,
            )
            backend.execute(
                "SELECT set_config('app.current_dataset_id', ?, true)",
                (DATASET_ID,),
                connection=conn,
            )
            principal = backend.execute(
                "SELECT rolsuper,rolbypassrls FROM pg_roles WHERE rolname=current_user",
                connection=conn,
            ).rows[0]
            assert principal == {"rolsuper": False, "rolbypassrls": False}
            assert backend.execute(
                "SELECT note_id FROM note_semantic_note_state",
                connection=conn,
            ).rows == [{"note_id": NOTE_ID}]
            backend.execute(
                "SELECT set_config('app.current_user_id', ?, true)",
                (OTHER_OWNER_ID,),
                connection=conn,
            )
            assert backend.execute(
                "SELECT note_id FROM note_semantic_note_state",
                connection=conn,
            ).rows == []

        with pytest.raises(RuntimeError, match="rollback-postgres"):
            with db.transaction() as conn:
                db.note_store.add_note(
                    "Rolled back",
                    "Body",
                    note_id=ROLLED_BACK_NOTE_ID,
                    semantic_dataset_id=DATASET_ID,
                    conn=conn,
                )
                scope = conn.execute(
                    "SELECT current_setting('app.current_user_id', true) AS owner_id, "
                    "current_setting('app.current_dataset_id', true) AS dataset_id"
                ).fetchone()
                assert scope["owner_id"] == OWNER_ID
                assert scope["dataset_id"] == DATASET_ID
                raise RuntimeError("rollback-postgres")
        assert db.get_note_by_id(ROLLED_BACK_NOTE_ID) is None
        state, work = _state_and_work(db, ROLLED_BACK_NOTE_ID)
        assert state is None
        assert work == []

        with db.transaction() as conn:
            _scope(conn)
            conn.execute(
                "INSERT INTO note_semantic_chunks("
                "chunk_id,owner_user_id,dataset_id,generation_id,note_id,content_version,"
                "ordinal,field,start_offset,end_offset,chunk_fingerprint,normalization_version,chunker_version"
                ") VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    "postgres-chunk",
                    OWNER_ID,
                    DATASET_ID,
                    generation_id,
                    NOTE_ID,
                    4,
                    0,
                    "content",
                    0,
                    4,
                    content_fingerprint("", "Body"),
                    "normalization-v1",
                    "chunker-v1",
                ),
            )
        assert db.note_store.delete_note(
            NOTE_ID,
            hard_delete=True,
            semantic_dataset_id=DATASET_ID,
        )
        assert db.get_note_by_id(NOTE_ID) is None
        state, work = _state_and_work(db, NOTE_ID)
        assert state is None
        assert work == [("delete_note_vectors", generation_id, 5)]
        with db.transaction() as conn:
            _scope(conn)
            assert conn.execute(
                "SELECT COUNT(*) AS row_count FROM note_semantic_chunks WHERE owner_user_id=? "
                "AND dataset_id=? AND note_id=?",
                (OWNER_ID, DATASET_ID, NOTE_ID),
            ).fetchone()["row_count"] == 0
        config = db.note_semantic_store.get_configuration(DATASET_ID)
        assert config is not None
        assert config.semantic_index_revision == 3
    finally:
        if role_created:
            with backend.transaction() as conn:
                backend.execute(f"REVOKE {ident(role_name)} FROM CURRENT_USER", connection=conn)
                backend.execute(f"DROP OWNED BY {ident(role_name)}", connection=conn)
                backend.execute(f"DROP ROLE {ident(role_name)}", connection=conn)
        db.close_all_connections()
        backend.get_pool().close_all()
