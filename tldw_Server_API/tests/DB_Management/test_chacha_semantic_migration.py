"""SQLite schema-v65 contracts for Notes semantic-index persistence."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDB,
    CharactersRAGDBError,
)

pytestmark = pytest.mark.unit


_TABLES = {
    "note_semantic_index_configs",
    "note_semantic_generations",
    "note_semantic_note_state",
    "note_semantic_chunks",
    "note_semantic_work",
}
_DIGEST = f"sha256:{'a' * 64}"


def _version(conn: sqlite3.Connection) -> int:
    return int(
        conn.execute(
            "SELECT version FROM db_schema_version WHERE schema_name = ?",
            (CharactersRAGDB._SCHEMA_NAME,),
        ).fetchone()[0]
    )


def _initialize(path: Path) -> CharactersRAGDB:
    return CharactersRAGDB(str(path), client_id="owner-a")


def _prepare_v64_database(path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(CharactersRAGDB, "_CURRENT_SCHEMA_VERSION", 64)
    db = _initialize(path)
    db.close_all_connections()


def test_sqlite_v65_fresh_schema_has_semantic_tables_constraints_and_indexes(tmp_path: Path) -> None:
    db_path = tmp_path / "chacha-v65-fresh.sqlite"
    db = _initialize(db_path)
    db.close_all_connections()

    with sqlite3.connect(db_path) as conn:
        tables = {
            str(row[0])
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
        config_columns = {
            str(row[1]) for row in conn.execute("PRAGMA table_info(note_semantic_index_configs)")
        }
        generation_indexes = {
            str(row[1]) for row in conn.execute("PRAGMA index_list(note_semantic_generations)")
        }
        work_indexes = {
            str(row[1]) for row in conn.execute("PRAGMA index_list(note_semantic_work)")
        }
        chunk_columns = {
            str(row[1]) for row in conn.execute("PRAGMA table_info(note_semantic_chunks)")
        }
        generation_fks = list(conn.execute("PRAGMA foreign_key_list(note_semantic_note_state)"))

        assert _version(conn) == 65
        assert tables >= _TABLES
        assert {
            "owner_user_id",
            "dataset_id",
            "desired_state",
            "configuration_revision",
            "semantic_index_revision",
            "endpoint_origin_display",
            "dimension_state",
            "dimensions",
        } <= config_columns
        assert {"chunk_id", "field", "start_offset", "end_offset", "chunk_fingerprint"} <= chunk_columns
        assert {
            "idx_note_semantic_generations_one_active",
            "idx_note_semantic_generations_one_staging",
        } <= generation_indexes
        assert {
            "idx_note_semantic_work_claimable",
            "idx_note_semantic_work_note_coalesce",
            "idx_note_semantic_work_generation_coalesce",
        } <= work_indexes
        assert any(str(row[2]) == "note_semantic_generations" for row in generation_fks)

        conn.execute(
            """
            INSERT INTO note_semantic_index_configs(
                owner_user_id, dataset_id, desired_state, configuration_revision,
                semantic_index_revision, dimension_state, metric,
                normalization_version, chunker_version, updated_at
            ) VALUES ('owner-a', 'dataset-a', 'disabled', 1, 0, 'pending', 'cosine', 'v1', 'v1', CURRENT_TIMESTAMP)
            """
        )
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "UPDATE note_semantic_index_configs SET desired_state = 'unknown' "
                "WHERE owner_user_id = 'owner-a'"
            )


def test_sqlite_v65_config_dimension_identity_check_preserves_disabled_states(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "chacha-v65-config-dimensions.sqlite"
    db = _initialize(db_path)
    db.close_all_connections()

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO note_semantic_index_configs(
                owner_user_id,dataset_id,desired_state,configuration_revision,
                semantic_index_revision,metric,dimension_state,dimensions,
                compatibility_hash,normalization_version,chunker_version,updated_at
            ) VALUES ('owner-a','dataset-a','disabled',1,0,'cosine','pending',
                      NULL,NULL,'v1','v1',CURRENT_TIMESTAMP)
            """
        )
        for dimension_state, dimensions, compatibility_hash in (
            ("pending", 768, None),
            ("pending", None, "compatibility-v1"),
            ("resolved", None, "compatibility-v1"),
            ("resolved", 768, None),
            ("resolved", 768, ""),
        ):
            with pytest.raises(sqlite3.IntegrityError):
                conn.execute(
                    "UPDATE note_semantic_index_configs SET dimension_state=?, dimensions=?, "
                    "compatibility_hash=? WHERE owner_user_id='owner-a' AND dataset_id='dataset-a'",
                    (dimension_state, dimensions, compatibility_hash),
                )
        conn.execute(
            "UPDATE note_semantic_index_configs SET dimension_state='resolved', dimensions=768, "
            "compatibility_hash='compatibility-v1' "
            "WHERE owner_user_id='owner-a' AND dataset_id='dataset-a'"
        )
        row = conn.execute(
            "SELECT desired_state,dimension_state,dimensions,compatibility_hash "
            "FROM note_semantic_index_configs WHERE owner_user_id='owner-a' AND dataset_id='dataset-a'"
        ).fetchone()
        assert tuple(row) == ("disabled", "resolved", 768, "compatibility-v1")


@pytest.mark.parametrize(
    ("dimension_state", "dimensions", "compatibility_hash"),
    (
        ("pending", 768, None),
        ("pending", None, "compatibility-v1"),
        ("resolved", None, "compatibility-v1"),
        ("resolved", 768, None),
        ("resolved", 768, ""),
    ),
)
def test_sqlite_v65_generation_dimension_identity_check(
    tmp_path: Path,
    dimension_state: str,
    dimensions: int | None,
    compatibility_hash: str | None,
) -> None:
    db_path = tmp_path / f"chacha-v65-generation-dimensions-{dimension_state}-{dimensions}.sqlite"
    db = _initialize(db_path)
    db.close_all_connections()

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO note_semantic_index_configs(
                owner_user_id,dataset_id,desired_state,configuration_revision,
                semantic_index_revision,metric,dimension_state,dimensions,
                compatibility_hash,normalization_version,chunker_version,updated_at
            ) VALUES ('owner-a','dataset-a','disabled',1,0,'cosine','resolved',
                      768,'compatibility-v1','v1','v1',CURRENT_TIMESTAMP)
            """
        )
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """
                INSERT INTO note_semantic_generations(
                    id,owner_user_id,dataset_id,configuration_revision,state,
                    compatibility_hash,dimension_state,dimensions,created_at
                ) VALUES ('generation-a','owner-a','dataset-a',1,'staging',?,?,?,CURRENT_TIMESTAMP)
                """,
                (compatibility_hash, dimension_state, dimensions),
            )


def test_sqlite_v65_rejects_raw_content_and_chunk_fingerprints(tmp_path: Path) -> None:
    db_path = tmp_path / "chacha-v65-fingerprints.sqlite"
    db = _initialize(db_path)
    db.close_all_connections()

    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute(
            "INSERT INTO notes(id,title,content,client_id) VALUES ('note-a','title','body','owner-a')"
        )
        conn.execute(
            """
            INSERT INTO note_semantic_index_configs(
                owner_user_id,dataset_id,desired_state,configuration_revision,
                semantic_index_revision,metric,dimension_state,dimensions,
                compatibility_hash,normalization_version,chunker_version,updated_at
            ) VALUES ('owner-a','dataset-a','enabled',1,0,'cosine','resolved',768,
                      'compatibility-v1','v1','v1',CURRENT_TIMESTAMP)
            """
        )
        conn.execute(
            """
            INSERT INTO note_semantic_generations(
                id,owner_user_id,dataset_id,configuration_revision,state,
                compatibility_hash,dimension_state,dimensions,created_at
            ) VALUES ('generation-a','owner-a','dataset-a',1,'staging',
                      'compatibility-v1','resolved',768,CURRENT_TIMESTAMP)
            """
        )
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """
                INSERT INTO note_semantic_note_state(
                    owner_user_id,dataset_id,generation_id,note_id,content_version,
                    content_fingerprint,dirty_generation,state
                ) VALUES ('owner-a','dataset-a','generation-a','note-a',1,
                          'raw Note body',1,'pending')
                """
            )
        conn.execute(
            """
            INSERT INTO note_semantic_note_state(
                owner_user_id,dataset_id,generation_id,note_id,content_version,
                content_fingerprint,dirty_generation,state
            ) VALUES ('owner-a','dataset-a','generation-a','note-a',1,?,1,'pending')
            """,
            (_DIGEST,),
        )
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """
                INSERT INTO note_semantic_chunks(
                    chunk_id,owner_user_id,dataset_id,generation_id,note_id,
                    content_version,ordinal,field,start_offset,end_offset,
                    chunk_fingerprint,normalization_version,chunker_version
                ) VALUES ('chunk-a','owner-a','dataset-a','generation-a','note-a',
                          1,0,'content',0,5,'raw Note body','v1','v1')
                """
            )


@pytest.mark.parametrize(
    ("kind", "note_id", "dirty_generation"),
    (
        ("index_note", "note-a", 1),
        ("delete_note_vectors", "note-a", 1),
        ("delete_generation", None, None),
    ),
)
def test_sqlite_v65_requires_generation_identity_for_all_work_kinds(
    tmp_path: Path,
    kind: str,
    note_id: str | None,
    dirty_generation: int | None,
) -> None:
    db_path = tmp_path / f"chacha-v65-work-{kind}.sqlite"
    db = _initialize(db_path)
    db.close_all_connections()

    with sqlite3.connect(db_path) as conn, pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            """
            INSERT INTO note_semantic_work(
                id,owner_user_id,dataset_id,kind,note_id,generation_id,
                dirty_generation,fencing_token,claim_state,attempt_count,
                next_eligible_at,created_at,updated_at
            ) VALUES (?,?,?,?,?,NULL,?,'fence','pending',0,
                      CURRENT_TIMESTAMP,CURRENT_TIMESTAMP,CURRENT_TIMESTAMP)
            """,
            (f"work-{kind}", "owner-a", "dataset-a", kind, note_id, dirty_generation),
        )


def test_sqlite_v64_to_v65_upgrade_creates_semantic_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "chacha-v64-upgrade.sqlite"
    _prepare_v64_database(db_path, monkeypatch)
    monkeypatch.setattr(CharactersRAGDB, "_CURRENT_SCHEMA_VERSION", 65)
    db = _initialize(db_path)
    db.close_all_connections()

    with sqlite3.connect(db_path) as conn:
        tables = {
            str(row[0])
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
        assert _version(conn) == 65
        assert tables >= _TABLES


def test_sqlite_v65_rolls_back_partial_ddl_and_preserves_v64(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "chacha-v65-rollback.sqlite"
    _prepare_v64_database(db_path, monkeypatch)
    injected_sql = (
        CharactersRAGDB._MIGRATION_SQL_V64_TO_V65
        + "\nTHIS IS AN INJECTED MIGRATION FAILURE;"
    )
    monkeypatch.setattr(CharactersRAGDB, "_CURRENT_SCHEMA_VERSION", 65)
    monkeypatch.setattr(CharactersRAGDB, "_MIGRATION_SQL_V64_TO_V65", injected_sql)

    with pytest.raises(CharactersRAGDBError):
        _initialize(db_path)

    with sqlite3.connect(db_path) as conn:
        tables = {
            str(row[0])
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
        assert _version(conn) == 64
        assert not (tables & _TABLES)
