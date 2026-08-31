"""SQLite schema-v67 contracts for semantic model and operation authority."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB

pytestmark = pytest.mark.unit

_RECEIPTS = "note_semantic_operation_receipts"


def _initialize(path: Path) -> None:
    db = CharactersRAGDB(str(path), client_id="owner-a")
    db.close_all_connections()


def _version(conn: sqlite3.Connection) -> int:
    return int(
        conn.execute(
            "SELECT version FROM db_schema_version WHERE schema_name=?",
            (CharactersRAGDB._SCHEMA_NAME,),
        ).fetchone()[0]
    )


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {
        str(row[1])
        for row in conn.execute(f"PRAGMA table_info({table})")  # nosec B608
    }


def test_sqlite_v67_fresh_schema_has_model_revision_and_operation_receipts(
    tmp_path: Path,
) -> None:
    path = tmp_path / "semantic-v67-fresh.sqlite"

    _initialize(path)

    with sqlite3.connect(path) as conn:
        assert _version(conn) == 67
        assert "model_revision" in _columns(conn, "note_semantic_index_configs")
        assert "model_revision" in _columns(conn, "note_semantic_generations")
        assert {
            "owner_user_id",
            "dataset_id",
            "key_digest",
            "action",
            "request_fingerprint",
            "run_id",
            "expected_revision",
            "state",
            "response_json",
            "expires_at",
        } <= _columns(conn, _RECEIPTS)
        indexes = {
            str(row[1])
            for row in conn.execute(f"PRAGMA index_list({_RECEIPTS})")  # nosec B608
        }
        assert "idx_note_semantic_operation_receipts_scope" in indexes


def test_sqlite_v66_to_v67_preserves_semantic_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "semantic-v66-upgrade.sqlite"
    monkeypatch.setattr(CharactersRAGDB, "_CURRENT_SCHEMA_VERSION", 66)
    _initialize(path)
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            INSERT INTO note_semantic_index_configs(
                owner_user_id,dataset_id,desired_state,configuration_revision,
                semantic_index_revision,metric,dimension_state,
                normalization_version,chunker_version,updated_at
            ) VALUES ('owner-a','dataset-a','disabled',1,0,'cosine','pending','v1','v1',CURRENT_TIMESTAMP)
            """
        )
        conn.commit()

    monkeypatch.setattr(CharactersRAGDB, "_CURRENT_SCHEMA_VERSION", 67)
    _initialize(path)

    with sqlite3.connect(path) as conn:
        assert _version(conn) == 67
        assert conn.execute(
            "SELECT desired_state,model_revision FROM note_semantic_index_configs "
            "WHERE owner_user_id='owner-a' AND dataset_id='dataset-a'"
        ).fetchone() == ("disabled", None)
        assert conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (_RECEIPTS,),
        ).fetchone() is not None
