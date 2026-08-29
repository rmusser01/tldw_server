"""SQLite schema-v64 contracts for Notes graph suggestion persistence."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB, SchemaError

pytestmark = pytest.mark.unit


_TABLES = {
    "note_graph_suggestion_runs",
    "note_graph_suggestion_operation_receipts",
    "note_graph_suggestion_rejection_sets",
    "note_graph_suggestions",
    "note_graph_suggestion_evidence",
}


def _initialize(path: Path) -> CharactersRAGDB:
    return CharactersRAGDB(str(path), client_id="owner-a")


def _version(conn: sqlite3.Connection) -> int:
    return int(
        conn.execute(
            "SELECT version FROM db_schema_version WHERE schema_name = ?",
            (CharactersRAGDB._SCHEMA_NAME,),
        ).fetchone()[0]
    )


def _create_note(conn: sqlite3.Connection, note_id: str, owner_user_id: str = "owner-a") -> None:
    conn.execute(
        "INSERT INTO notes(id, title, content, client_id) VALUES (?, ?, ?, ?)",
        (note_id, note_id, "content", owner_user_id),
    )


def _insert_receipt(
    conn: sqlite3.Connection,
    receipt_id: str,
    *,
    owner_user_id: str = "owner-a",
    dataset_id: str = "dataset-a",
    source_note_id: str = "source-note",
) -> None:
    conn.execute(
        """
        INSERT INTO note_graph_suggestion_operation_receipts(
            id, operation_kind, owner_user_id, dataset_id, source_note_id,
            resource_identity, idempotency_key_digest, request_fingerprint,
            state, expires_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """,
        (
            receipt_id,
            "run_admit",
            owner_user_id,
            dataset_id,
            source_note_id,
            f"resource-{receipt_id}",
            f"key-{receipt_id}",
            f"request-{receipt_id}",
            "completed",
        ),
    )


def _insert_run(
    conn: sqlite3.Connection,
    run_id: str,
    *,
    owner_user_id: str = "owner-a",
    dataset_id: str = "dataset-a",
    source_note_id: str = "source-note",
    admission_receipt_id: str | None = None,
) -> None:
    conn.execute(
        """
        INSERT INTO note_graph_suggestion_runs(
            id, owner_user_id, dataset_id, source_note_id, source_fingerprint,
            admission_receipt_id, provider, model, capability_revision,
            prompt_contract_version, state, revision, created_at, expires_at
        ) VALUES (?, ?, ?, ?, ?, ?, 'openai', 'model-a', 'cap-v1',
                  'prompt-v1', ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
        """,
        (
            run_id,
            owner_user_id,
            dataset_id,
            source_note_id,
            f"fingerprint-{source_note_id}",
            admission_receipt_id,
            "succeeded",
            1,
        ),
    )


@pytest.mark.parametrize(
    ("column", "value"),
    (
        ("provider", None),
        ("provider", ""),
        ("provider", "   "),
        ("model", None),
        ("model", ""),
        ("capability_revision", None),
        ("capability_revision", " "),
        ("prompt_contract_version", None),
        ("prompt_contract_version", ""),
    ),
)
def test_sqlite_v64_active_run_binding_fields_are_non_null_trimmed_and_non_empty(
    tmp_path: Path,
    column: str,
    value: str | None,
) -> None:
    db_path = tmp_path / f"chacha-v64-{column}-{value!r}.sqlite"
    db = _initialize(db_path)
    db.close_all_connections()
    fields = {
        "provider": "openai",
        "model": "model-a",
        "capability_revision": "cap-v1",
        "prompt_contract_version": "prompt-v1",
    }
    fields[column] = value
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        _create_note(conn, "binding-note")
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """
                INSERT INTO note_graph_suggestion_runs(
                    id,owner_user_id,dataset_id,source_note_id,source_fingerprint,
                    provider,model,capability_revision,prompt_contract_version,
                    state,revision,created_at,expires_at
                ) VALUES ('binding-run','owner-a','dataset-a','binding-note','fp',
                          ?,?,?,?,'queued',1,CURRENT_TIMESTAMP,CURRENT_TIMESTAMP)
                """,
                (
                    fields["provider"],
                    fields["model"],
                    fields["capability_revision"],
                    fields["prompt_contract_version"],
                ),
            )


def test_sqlite_v64_active_run_contract_fields_preserve_duplicate_equivalence(tmp_path: Path) -> None:
    db_path = tmp_path / "chacha-v64-active-equivalence.sqlite"
    db = _initialize(db_path)
    db.close_all_connections()
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        _create_note(conn, "equivalent-note")
        values = (
            "owner-a",
            "dataset-a",
            "equivalent-note",
            "fingerprint",
            "openai",
            "model-a",
            "cap-v1",
            "prompt-v1",
            "queued",
        )
        sql = """
            INSERT INTO note_graph_suggestion_runs(
                id,owner_user_id,dataset_id,source_note_id,source_fingerprint,
                provider,model,capability_revision,prompt_contract_version,
                state,revision,created_at,expires_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,1,CURRENT_TIMESTAMP,CURRENT_TIMESTAMP)
        """
        conn.execute(sql, ("equivalent-a", *values))
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(sql, ("equivalent-b", *values))


def _insert_related_suggestion(
    conn: sqlite3.Connection,
    suggestion_id: str,
    *,
    run_id: str,
    source_note_id: str,
    target_note_id: str,
    source_fingerprint: str,
    target_fingerprint: str,
    owner_user_id: str = "owner-a",
    dataset_id: str = "dataset-a",
    decision_receipt_id: str | None = None,
    state: str = "pending",
) -> None:
    conn.execute(
        """
        INSERT INTO note_graph_suggestions(
            id, run_id, owner_user_id, dataset_id, kind, source_note_id,
            source_fingerprint, target_note_id, target_fingerprint, state,
            revision, decision_receipt_id, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
        """,
        (
            suggestion_id,
            run_id,
            owner_user_id,
            dataset_id,
            "related_note",
            source_note_id,
            source_fingerprint,
            target_note_id,
            target_fingerprint,
            state,
            1,
            decision_receipt_id,
        ),
    )


def _insert_tag_suggestion(
    conn: sqlite3.Connection,
    suggestion_id: str,
    *,
    run_id: str,
    source_note_id: str = "source-note",
    source_fingerprint: str = "fingerprint-source-note",
    normalized_tag: str = "research",
    decision_receipt_id: str | None = None,
    state: str = "pending",
) -> None:
    conn.execute(
        """
        INSERT INTO note_graph_suggestions(
            id, run_id, owner_user_id, dataset_id, kind, source_note_id,
            source_fingerprint, normalized_tag, display_tag, state, revision,
            decision_receipt_id, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
        """,
        (
            suggestion_id,
            run_id,
            "owner-a",
            "dataset-a",
            "tag",
            source_note_id,
            source_fingerprint,
            normalized_tag,
            normalized_tag.title(),
            state,
            1,
            decision_receipt_id,
        ),
    )


def _prepare_v63_database(path: Path) -> CharactersRAGDB:
    db = _initialize(path)
    conn = db.get_connection()
    conn.execute("DROP TABLE note_graph_suggestion_evidence")
    conn.execute("DROP TABLE note_graph_suggestions")
    conn.execute("DROP TABLE note_graph_suggestion_rejection_sets")
    conn.execute("DROP TABLE note_graph_suggestion_runs")
    conn.execute("DROP TABLE note_graph_suggestion_operation_receipts")
    conn.execute(
        "UPDATE db_schema_version SET version = 63 WHERE schema_name = ?",
        (CharactersRAGDB._SCHEMA_NAME,),
    )
    conn.commit()
    return db


def test_sqlite_v64_fresh_schema_has_graph_suggestion_tables_constraints_and_indexes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "chacha-v64-fresh.sqlite"
    monkeypatch.setattr(CharactersRAGDB, "_CURRENT_SCHEMA_VERSION", 64)
    db = _initialize(db_path)
    db.close_all_connections()

    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        tables = {
            str(row[0])
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
        run_indexes = {
            str(row[1]) for row in conn.execute("PRAGMA index_list(note_graph_suggestion_runs)")
        }
        suggestion_indexes = {
            str(row[1]) for row in conn.execute("PRAGMA index_list(note_graph_suggestions)")
        }
        receipt_indexes = {
            str(row[1])
            for row in conn.execute(
                "PRAGMA index_list(note_graph_suggestion_operation_receipts)"
            )
        }
        evidence_fks = conn.execute(
            "PRAGMA foreign_key_list(note_graph_suggestion_evidence)"
        ).fetchall()
        suggestion_columns = {
            str(row[1]) for row in conn.execute("PRAGMA table_info(note_graph_suggestions)")
        }

        _create_note(conn, "source-note")
        conn.execute(
            """
            INSERT INTO note_graph_suggestion_runs(
                id, owner_user_id, dataset_id, source_note_id, source_fingerprint,
                provider, model, capability_revision, prompt_contract_version,
                state, revision, created_at, expires_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            """,
            (
                "run-active", "owner-a", "dataset-a", "source-note", "source-fingerprint",
                "openai", "model-a", "cap-v1", "prompt-v1", "queued", 1,
            ),
        )
        conn.execute(
            """
            INSERT INTO note_graph_suggestion_runs(
                id, owner_user_id, dataset_id, source_note_id, source_fingerprint,
                provider, model, capability_revision, prompt_contract_version,
                state, revision, created_at, expires_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            """,
            (
                "run-different-fingerprint", "owner-a", "dataset-a", "source-note",
                "new-fingerprint", "openai", "model-a", "cap-v1", "prompt-v1", "running", 1,
            ),
        )
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """
                INSERT INTO note_graph_suggestion_runs(
                    id, owner_user_id, dataset_id, source_note_id, source_fingerprint,
                    provider, model, capability_revision, prompt_contract_version,
                    state, revision, created_at, expires_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                """,
                (
                    "run-conflict", "owner-a", "dataset-a", "source-note", "source-fingerprint",
                    "openai", "model-a", "cap-v1", "prompt-v1", "running", 1,
                ),
            )
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """
                INSERT INTO note_graph_suggestion_runs(
                    id, owner_user_id, dataset_id, source_note_id, source_fingerprint,
                    provider,model,capability_revision,prompt_contract_version,
                    state, revision, created_at, expires_at
                ) VALUES (?, ?, ?, ?, ?, 'openai','model-a','cap-v1','prompt-v1',
                          ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                """,
                ("run-invalid", "owner-a", "dataset-a", "source-note", "source-fingerprint", "invalid", 1),
            )

    assert _version(sqlite3.connect(db_path)) == 64
    assert tables >= _TABLES
    assert "keyword_sync_id" in suggestion_columns
    assert "keyword_id" not in suggestion_columns
    assert {
        "idx_note_graph_suggestion_runs_owner_dataset_note_state",
        "idx_note_graph_suggestion_runs_active_source",
        "idx_note_graph_suggestion_runs_retention",
        "idx_note_graph_suggestion_runs_maintenance",
    } <= run_indexes
    assert {
        "idx_note_graph_suggestions_owner_dataset_source_state",
        "idx_note_graph_suggestions_acceptance_lease",
        "idx_note_graph_suggestions_retention",
        "idx_note_graph_suggestions_staged_related_identity",
        "idx_note_graph_suggestions_staged_tag_identity",
    } <= suggestion_indexes
    assert "idx_note_graph_suggestion_operation_receipts_retention" in receipt_indexes
    assert {
        (str(row[2]), str(row[3]), str(row[6]).upper()) for row in evidence_fks
    } >= {
        ("note_graph_suggestions", "suggestion_id", "CASCADE"),
        ("notes", "note_id", "CASCADE"),
    }


def test_sqlite_v64_fix_round_three_has_paired_maintenance_lease_and_scan_index(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "chacha-v64-maintenance-lease.sqlite"
    db = _initialize(db_path)
    db.close_all_connections()

    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        columns = {
            str(row[1]): row
            for row in conn.execute("PRAGMA table_info(note_graph_suggestion_runs)")
        }
        indexes = {
            str(row[1]) for row in conn.execute("PRAGMA index_list(note_graph_suggestion_runs)")
        }
        _create_note(conn, "maintenance-note")
        conn.execute(
            """
            INSERT INTO note_graph_suggestion_runs(
                id,owner_user_id,dataset_id,source_note_id,source_fingerprint,
                provider,model,capability_revision,prompt_contract_version,
                state,revision,created_at,expires_at
            ) VALUES ('maintenance-run','owner-a','dataset-a','maintenance-note','fp',
                      'openai','model-a','cap-v1','prompt-v1','queued',1,
                      CURRENT_TIMESTAMP,CURRENT_TIMESTAMP)
            """
        )
        lease = conn.execute(
            "SELECT maintenance_lease_token,maintenance_lease_expires_at "
            "FROM note_graph_suggestion_runs WHERE id='maintenance-run'"
        ).fetchone()

        assert columns["maintenance_lease_token"][3] == 0
        assert columns["maintenance_lease_token"][4] is None
        assert columns["maintenance_lease_expires_at"][3] == 0
        assert columns["maintenance_lease_expires_at"][4] is None
        assert lease == (None, None)
        assert "idx_note_graph_suggestion_runs_maintenance" in indexes
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "UPDATE note_graph_suggestion_runs SET maintenance_lease_token='orphan' "
                "WHERE id='maintenance-run'"
            )


def test_sqlite_v63_to_v64_upgrade_creates_graph_suggestion_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "chacha-v63-upgrade.sqlite"
    base = _prepare_v63_database(db_path)
    base.close_all_connections()
    upgraded = _initialize(db_path)
    upgraded.close_all_connections()

    with sqlite3.connect(db_path) as conn:
        tables = {
            str(row[0])
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
        run_columns = {
            str(row[1]) for row in conn.execute("PRAGMA table_info(note_graph_suggestion_runs)")
        }
        run_indexes = {
            str(row[1]) for row in conn.execute("PRAGMA index_list(note_graph_suggestion_runs)")
        }
        assert _version(conn) == 64
        assert tables >= _TABLES
        assert {"maintenance_lease_token", "maintenance_lease_expires_at"} <= run_columns
        assert "idx_note_graph_suggestion_runs_maintenance" in run_indexes


def test_sqlite_v64_failure_rolls_back_partial_ddl_and_preserves_v63(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "chacha-v64-interrupted.sqlite"
    db = _prepare_v63_database(db_path)
    original_sql = CharactersRAGDB._MIGRATION_SQL_V63_TO_V64
    injected_sql = original_sql.replace(
        "CREATE TABLE note_graph_suggestion_rejection_sets",
        "SELECT * FROM injected_v64_failure;\n"
        "CREATE TABLE note_graph_suggestion_rejection_sets",
        1,
    )

    try:
        monkeypatch.setattr(CharactersRAGDB, "_CURRENT_SCHEMA_VERSION", 64)
        monkeypatch.setattr(CharactersRAGDB, "_MIGRATION_SQL_V63_TO_V64", injected_sql)

        with pytest.raises(SchemaError, match="Notes graph suggestion v64 SQLite migration failed"):
            db._initialize_schema_sqlite()

        conn = db.get_connection()
        tables_after_failure = {
            str(row[0])
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
        assert _version(conn) == 63
        assert not _TABLES.intersection(tables_after_failure)
    finally:
        db.close_all_connections()


def test_sqlite_v64_rejects_cross_scope_parent_references(tmp_path: Path) -> None:
    db = _initialize(tmp_path / "chacha-v64-cross-scope.sqlite")
    try:
        conn = db.get_connection()
        _create_note(conn, "source-note")
        _create_note(conn, "owner-b-note", "owner-b")
        with pytest.raises(sqlite3.IntegrityError):
            _insert_run(conn, "wrong-note-owner", source_note_id="owner-b-note")

        _insert_receipt(conn, "receipt-a")
        _insert_receipt(
            conn,
            "receipt-b",
            owner_user_id="owner-b",
            dataset_id="dataset-b",
            source_note_id="owner-b-note",
        )
        with pytest.raises(sqlite3.IntegrityError):
            _insert_run(
                conn,
                "wrong-receipt-dataset",
                dataset_id="dataset-b",
                admission_receipt_id="receipt-a",
            )

        _insert_run(conn, "run-a", admission_receipt_id="receipt-a")
        with pytest.raises(sqlite3.IntegrityError):
            _insert_related_suggestion(
                conn,
                "wrong-run-dataset",
                run_id="run-a",
                source_note_id="source-note",
                target_note_id="source-note",
                source_fingerprint="source-fingerprint",
                target_fingerprint="target-fingerprint",
                dataset_id="dataset-b",
            )
        with pytest.raises(sqlite3.IntegrityError):
            _insert_related_suggestion(
                conn,
                "wrong-target-owner",
                run_id="run-a",
                source_note_id="source-note",
                target_note_id="owner-b-note",
                source_fingerprint="source-fingerprint",
                target_fingerprint="target-fingerprint",
            )

        _insert_related_suggestion(
            conn,
            "suggestion-a",
            run_id="run-a",
            source_note_id="source-note",
            target_note_id="source-note",
            source_fingerprint="source-fingerprint",
            target_fingerprint="target-fingerprint",
        )
        with pytest.raises(sqlite3.IntegrityError):
            _insert_related_suggestion(
                conn,
                "wrong-decision-receipt",
                run_id="run-a",
                source_note_id="source-note",
                target_note_id="source-note",
                source_fingerprint="other-source-fingerprint",
                target_fingerprint="other-target-fingerprint",
                decision_receipt_id="receipt-b",
            )
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """
                INSERT INTO note_graph_suggestion_evidence(
                    suggestion_id, owner_user_id, dataset_id, side, ordinal, note_id,
                    field, content_fingerprint, start_offset, end_offset
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "suggestion-a",
                    "owner-a",
                    "dataset-b",
                    "source",
                    0,
                    "source-note",
                    "content",
                    "source-fingerprint",
                    0,
                    1,
                ),
            )
    finally:
        db.close_all_connections()


def test_sqlite_v64_note_hard_delete_cascades_receipt_graph(tmp_path: Path) -> None:
    db = _initialize(tmp_path / "chacha-v64-receipt-cascade.sqlite")
    try:
        conn = db.get_connection()
        _create_note(conn, "source-note")
        _insert_receipt(conn, "receipt-a")
        _insert_run(conn, "run-a", admission_receipt_id="receipt-a")
        _insert_tag_suggestion(conn, "suggestion-a", run_id="run-a")
        conn.execute(
            """
            INSERT INTO note_graph_suggestion_evidence(
                suggestion_id, owner_user_id, dataset_id, side, ordinal, note_id,
                field, content_fingerprint, start_offset, end_offset
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "suggestion-a",
                "owner-a",
                "dataset-a",
                "source",
                0,
                "source-note",
                "content",
                "fingerprint-source-note",
                0,
                1,
            ),
        )

        conn.execute("DELETE FROM notes WHERE client_id = ? AND id = ?", ("owner-a", "source-note"))

        for table in (
            "note_graph_suggestion_operation_receipts",
            "note_graph_suggestion_runs",
            "note_graph_suggestions",
            "note_graph_suggestion_evidence",
        ):
            assert conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0  # nosec B608
    finally:
        db.close_all_connections()


def test_sqlite_v64_receipt_delete_clears_only_scoped_receipt_references(tmp_path: Path) -> None:
    db = _initialize(tmp_path / "chacha-v64-receipt-retention.sqlite")
    try:
        conn = db.get_connection()
        _create_note(conn, "source-note")
        _insert_receipt(conn, "admission-receipt")
        _insert_receipt(conn, "decision-receipt")
        _insert_run(conn, "run-a", admission_receipt_id="admission-receipt")
        _insert_tag_suggestion(
            conn,
            "suggestion-a",
            run_id="run-a",
            decision_receipt_id="decision-receipt",
            state="rejected",
        )
        _insert_tag_suggestion(
            conn,
            "suggestion-pending",
            run_id="run-a",
            normalized_tag="planning",
        )

        conn.execute(
            "DELETE FROM note_graph_suggestion_operation_receipts WHERE id = ?",
            ("admission-receipt",),
        )
        run = conn.execute(
            """
            SELECT owner_user_id, dataset_id, source_note_id, admission_receipt_id, state
              FROM note_graph_suggestion_runs
             WHERE id = ?
            """,
            ("run-a",),
        ).fetchone()
        assert tuple(run) == ("owner-a", "dataset-a", "source-note", None, "succeeded")
        assert tuple(
            conn.execute(
                "SELECT state FROM note_graph_suggestions WHERE id = ?",
                ("suggestion-a",),
            ).fetchone()
        ) == ("rejected",)
        assert tuple(
            conn.execute(
                "SELECT state FROM note_graph_suggestions WHERE id = ?",
                ("suggestion-pending",),
            ).fetchone()
        ) == ("pending",)

        conn.execute(
            "DELETE FROM note_graph_suggestion_operation_receipts WHERE id = ?",
            ("decision-receipt",),
        )
        suggestion = conn.execute(
            """
            SELECT owner_user_id, dataset_id, run_id, source_note_id, decision_receipt_id, state
              FROM note_graph_suggestions
             WHERE id = ?
            """,
            ("suggestion-a",),
        ).fetchone()
        assert tuple(suggestion) == (
            "owner-a",
            "dataset-a",
            "run-a",
            "source-note",
            None,
            "rejected",
        )
    finally:
        db.close_all_connections()


def test_sqlite_v64_allows_accepting_plus_pending_but_rejects_duplicate_pending(
    tmp_path: Path,
) -> None:
    db = _initialize(tmp_path / "chacha-v64-canonical-identity.sqlite")
    try:
        conn = db.get_connection()
        _create_note(conn, "alpha")
        _create_note(conn, "beta")
        _insert_run(conn, "run-alpha", source_note_id="alpha")
        _insert_run(conn, "run-beta", source_note_id="beta")
        _insert_related_suggestion(
            conn,
            "related-alpha-beta",
            run_id="run-alpha",
            source_note_id="alpha",
            target_note_id="beta",
            source_fingerprint="fingerprint-alpha",
            target_fingerprint="fingerprint-beta",
        )
        with pytest.raises(sqlite3.IntegrityError):
            _insert_related_suggestion(
                conn,
                "related-beta-alpha",
                run_id="run-beta",
                source_note_id="beta",
                target_note_id="alpha",
                source_fingerprint="fingerprint-beta",
                target_fingerprint="fingerprint-alpha",
            )
        conn.execute(
            "UPDATE note_graph_suggestions SET state='accepting' WHERE id='related-alpha-beta'"
        )
        _insert_related_suggestion(
            conn,
            "related-beta-alpha",
            run_id="run-beta",
            source_note_id="beta",
            target_note_id="alpha",
            source_fingerprint="fingerprint-beta",
            target_fingerprint="fingerprint-alpha",
        )

        _insert_tag_suggestion(
            conn,
            "tag-one",
            run_id="run-alpha",
            source_note_id="alpha",
            source_fingerprint="fingerprint-alpha",
        )
        with pytest.raises(sqlite3.IntegrityError):
            _insert_tag_suggestion(
                conn,
                "tag-two",
                run_id="run-alpha",
                source_note_id="alpha",
                source_fingerprint="fingerprint-alpha",
            )
        conn.execute("UPDATE note_graph_suggestions SET state='accepting' WHERE id='tag-one'")
        _insert_tag_suggestion(
            conn,
            "tag-two",
            run_id="run-alpha",
            source_note_id="alpha",
            source_fingerprint="fingerprint-alpha",
        )
    finally:
        db.close_all_connections()
