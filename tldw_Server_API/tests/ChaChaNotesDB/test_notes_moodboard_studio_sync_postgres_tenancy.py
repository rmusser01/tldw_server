"""Required-live PostgreSQL schema-v63 catalog and tenancy proofs."""

from __future__ import annotations

import hashlib
import inspect
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from uuid import uuid4

import pytest

from tldw_Server_API.app.core.DB_Management.backends.base import (
    DatabaseConfig,
    DatabaseError,
)
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDB,
    CharactersRAGDBError,
    ConflictError,
    SchemaError,
)
from tldw_Server_API.app.core.DB_Management.scope_context import scoped_context
from tldw_Server_API.app.core.Sync.v2.notes_moodboard_studio_contract import (
    diagram_render_hash,
    notes_studio_document_object_hash,
    parse_notes_studio_document_v1,
    studio_result_hash,
)

pytestmark = pytest.mark.integration

V61_TABLES = (
    "moodboards",
    "moodboard_notes",
    "note_studio_documents",
    "note_task_scope_authority",
)
V61_INDEXES = (
    "idx_moodboards_scope_page",
    "idx_moodboards_scope_sync_id",
    "idx_moodboard_notes_scope_board_page",
    "idx_moodboard_notes_scope_note",
    "idx_moodboard_notes_scope_placement",
    "idx_note_studio_documents_scope_page",
    "idx_note_studio_documents_scope_note",
    "idx_note_studio_documents_scope_source",
)
V61_EXACT_INDEXES = (
    "idx_moodboards_deleted",
    "idx_moodboards_last_modified",
    *V61_INDEXES,
    "idx_moodboard_notes_board",
    "idx_moodboard_notes_note",
    "idx_note_studio_documents_source_note_id",
    "moodboards_pkey",
    "moodboards_v61_scope_id_unique",
    "moodboards_v61_scope_sync_unique",
    "moodboard_notes_pkey",
    "moodboard_notes_v61_scope_placement_unique",
    "note_studio_documents_pkey",
    "note_task_scope_authority_pkey",
)
V61_CONSTRAINTS = (
    "moodboards_pkey",
    "moodboards_v61_canonical_hash_check",
    "moodboards_v61_canonical_revision_check",
    "moodboards_v61_dataset_check",
    "moodboards_v61_diagnostic_code_check",
    "moodboards_v61_diagnostic_hash_check",
    "moodboards_v61_owner_check",
    "moodboards_v61_scope_id_unique",
    "moodboards_v61_scope_sync_unique",
    "moodboards_v61_sync_id_check",
    "moodboard_notes_moodboard_id_fkey",
    "moodboard_notes_note_id_fkey",
    "moodboard_notes_pkey",
    "moodboard_notes_v61_board_fk",
    "moodboard_notes_v61_canonical_hash_check",
    "moodboard_notes_v61_canonical_revision_check",
    "moodboard_notes_v61_dataset_check",
    "moodboard_notes_v61_diagnostic_code_check",
    "moodboard_notes_v61_diagnostic_hash_check",
    "moodboard_notes_v61_height_check",
    "moodboard_notes_v61_note_fk",
    "moodboard_notes_v61_note_id_check",
    "moodboard_notes_v61_owner_check",
    "moodboard_notes_v61_placement_id_check",
    "moodboard_notes_v61_scope_placement_unique",
    "moodboard_notes_v61_version_check",
    "moodboard_notes_v61_width_check",
    "note_studio_documents_handwriting_mode_check",
    "note_studio_documents_note_id_fkey",
    "note_studio_documents_pkey",
    "note_studio_documents_render_version_check",
    "note_studio_documents_template_type_check",
    "note_studio_documents_v61_canonical_hash_check",
    "note_studio_documents_v61_canonical_revision_check",
    "note_studio_documents_v61_dataset_check",
    "note_studio_documents_v61_diagnostic_code_check",
    "note_studio_documents_v61_diagnostic_hash_check",
    "note_studio_documents_v61_note_fk",
    "note_studio_documents_v61_note_hash_check",
    "note_studio_documents_v61_note_id_check",
    "note_studio_documents_v61_note_revision_check",
    "note_studio_documents_v61_owner_check",
    "note_studio_documents_v61_version_check",
    "note_task_scope_authority_dataset_check",
    "note_task_scope_authority_owner_check",
    "note_task_scope_authority_pkey",
)


def _open_db(config: DatabaseConfig, *, owner: str = "960001") -> tuple[Any, CharactersRAGDB]:
    backend = DatabaseBackendFactory.create_backend(config)
    return backend, CharactersRAGDB(":memory:", client_id=owner, backend=backend)


def _postgres_force_rls_flags(backend: Any) -> dict[str, bool]:
    with backend.transaction() as conn:
        rows = backend.execute(
            "SELECT relname, relforcerowsecurity FROM pg_class "
            "WHERE relname IN (?,?,?,?)",
            V61_TABLES,
            connection=conn,
        ).rows
    return {str(row["relname"]): bool(row["relforcerowsecurity"]) for row in rows}


def _postgres_v61_scope_state(
    backend: Any,
    *,
    owner: str,
) -> dict[str, Any]:
    tables = (
        "moodboards",
        "moodboard_notes",
        "note_studio_documents",
        "note_task_scope_authority",
    )
    with backend.transaction() as conn:
        for table in tables:
            backend.execute(
                f"ALTER TABLE {table} NO FORCE ROW LEVEL SECURITY",  # nosec B608 - fixed test table names.
                connection=conn,
            )
        try:
            counts: dict[str, dict[str, int]] = {}
            for table in ("moodboards", "moodboard_notes", "note_studio_documents"):
                rows = backend.execute(
                    f"SELECT dataset_id, COUNT(*) AS count FROM {table} "  # nosec B608 - fixed test table names.
                    "WHERE owner_user_id=? GROUP BY dataset_id ORDER BY dataset_id",
                    (owner,),
                    connection=conn,
                ).rows
                counts[table] = {
                    str(row["dataset_id"]): int(row["count"])
                    for row in rows
                }
            authority = backend.execute(
                "SELECT dataset_id,task_graph_bound,moodboard_graph_bound,"
                "studio_graph_bound FROM note_task_scope_authority "
                "WHERE owner_user_id=? ORDER BY dataset_id",
                (owner,),
                connection=conn,
            ).rows
        finally:
            for table in reversed(tables):
                backend.execute(
                    f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY",  # nosec B608 - fixed test table names.
                    connection=conn,
                )
    return {
        **counts,
        "authority": [
            {
                "dataset_id": row["dataset_id"],
                "task_graph_bound": bool(row["task_graph_bound"]),
                "moodboard_graph_bound": bool(row["moodboard_graph_bound"]),
                "studio_graph_bound": bool(row["studio_graph_bound"]),
            }
            for row in authority
        ],
    }


def _seed_local_unbound_moodboard_studio_graph(
    db: CharactersRAGDB,
    *,
    note_id: str,
) -> int:
    db.add_note("Local unbound seed", "Body", note_id=note_id)
    moodboard_id = db.add_moodboard("Local board", "seed")
    assert isinstance(moodboard_id, int)
    assert db.link_note_to_moodboard(moodboard_id, note_id) is True
    studio = db.create_note_studio_document(
        note_id=note_id,
        payload_json={"sections": []},
        template_type="lined",
        handwriting_mode="off",
        render_version=1,
    )
    assert studio["dataset_id"] == "local-unbound"
    db.close_connection()
    return moodboard_id


def test_postgres_v63_migration_contract_is_bounded_and_version_last() -> None:
    source = inspect.getsource(CharactersRAGDB._migrate_from_v62_to_v63_postgres)
    fingerprint_source = inspect.getsource(
        CharactersRAGDB._postgres_v61_fingerprint_phase_page
    )
    begin_source = inspect.getsource(
        CharactersRAGDB._begin_notes_moodboard_studio_v61_postgres_transaction
    )
    configure_source = inspect.getsource(
        CharactersRAGDB._configure_notes_moodboard_studio_v61_postgres_transaction
    )
    schema_source = inspect.getsource(
        CharactersRAGDB._notes_moodboard_studio_v61_postgres_schema_sql
    )
    initializer_source = inspect.getsource(CharactersRAGDB._initialize_schema_postgres)

    assert CharactersRAGDB._POSTGRES_SCHEMA_VERSION == 63
    assert "lock_timeout" in configure_source
    assert "statement_timeout" in configure_source
    assert "lock=True" in begin_source
    assert "chacha_schema_migration_progress" in schema_source
    assert "_NOTES_MOODBOARD_STUDIO_V61_MIGRATION_PAGE_SIZE" in (
        source + fingerprint_source
    )
    assert source.index(
        "_begin_notes_moodboard_studio_v61_postgres_transaction"
    ) < source.index("LOCK TABLE")
    assert source.rindex("_set_schema_version_postgres") > source.rindex(
        "_verify_notes_moodboard_studio_schema_postgres"
    )
    assert source.index(
        'verification_phase = f"aggregate_verification:{source_phase}"'
    ) < source.index('if not begin_phase("RLS phase")')
    assert initializer_source.index(
        "_configure_notes_moodboard_studio_v61_postgres_transaction"
    ) < initializer_source.index("_get_schema_version_postgres(conn, lock=True)")


@pytest.mark.parametrize("predict", (True, False), ids=("source_prediction", "aggregate"))
def test_postgres_v61_fingerprint_phases_stop_at_deadline_and_resume_exactly(
    monkeypatch: pytest.MonkeyPatch,
    predict: bool,
) -> None:
    class _FakeTime:
        def __init__(self, *samples: float) -> None:
            self._samples = iter(samples)

        def monotonic(self) -> float:
            return next(self._samples)

    db = object.__new__(CharactersRAGDB)
    large_json = json.dumps(
        {"sections": [{"kind": "markdown", "content": "x" * 200_000}]},
        separators=(",", ":"),
    )
    rows = [
        {
            "owner_user_id": "deadline-owner",
            "note_id": f"00000000-0000-4000-8000-00000000000{number}",
            "payload_json": large_json,
            "excerpt_snapshot": "y" * 200_000,
        }
        for number in range(1, 4)
    ]

    def source_page(
        _conn: Any, *, phase: str, progress: dict[str, Any]
    ) -> tuple[list[dict[str, Any]], object | None]:
        assert phase == "note_studio_documents"
        cursor = (
            json.loads(str(progress["keyset_cursor"]))
            if progress.get("keyset_cursor")
            else ["", ""]
        )
        selected = [
            row
            for row in rows
            if (row["owner_user_id"], row["note_id"]) > tuple(cursor)
        ]
        return selected, (
            [selected[-1]["owner_user_id"], selected[-1]["note_id"]]
            if selected
            else cursor
        )

    def expected_row(
        _conn: Any, *, phase: str, row: dict[str, Any]
    ) -> dict[str, Any]:
        assert phase == "note_studio_documents"
        return {**row, "canonical_hash": "sha256:" + row["note_id"][-1] * 64}

    db._postgres_v61_source_page = source_page
    db._postgres_v61_expected_row = expected_row
    empty_fingerprint = "sha256:" + hashlib.sha256().hexdigest()
    progress: dict[str, Any] = {
        "keyset_cursor": None,
        "copied_count": 0,
        "aggregate_fingerprint": empty_fingerprint,
    }

    monkeypatch.setattr(
        "tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB.time",
        _FakeTime(0.0, 26.0),
    )
    first = db._postgres_v61_fingerprint_phase_page(
        None,
        source_phase="note_studio_documents",
        progress=progress,
        predict=predict,
    )
    assert first["processed_count"] == 1
    assert first["status"] == "running"
    assert first["cursor"] == [rows[0]["owner_user_id"], rows[0]["note_id"]]

    resumed_progress = {
        "keyset_cursor": json.dumps(first["cursor"], separators=(",", ":")),
        "copied_count": first["count"],
        "aggregate_fingerprint": first["fingerprint"],
    }
    monkeypatch.setattr(
        "tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB.time",
        _FakeTime(100.0, 100.0),
    )
    resumed = db._postgres_v61_fingerprint_phase_page(
        None,
        source_phase="note_studio_documents",
        progress=resumed_progress,
        predict=predict,
    )
    expected_rows = [expected_row(None, phase="note_studio_documents", row=row) for row in rows]
    if not predict:
        expected_rows = rows
    expected_count, expected_fingerprint = db._postgres_v61_progress_fingerprint(
        empty_fingerprint, expected_rows
    )
    assert resumed == {
        "cursor": [rows[-1]["owner_user_id"], rows[-1]["note_id"]],
        "count": expected_count,
        "fingerprint": expected_fingerprint,
        "status": "complete",
        "processed_count": 2,
    }


def test_fresh_postgres_schema_is_exact_v63_with_forced_rls(
    pg_database_config: DatabaseConfig,
) -> None:
    backend, db = _open_db(pg_database_config)
    try:
        with db.transaction() as conn:
            assert db._get_schema_version_postgres(conn) == 63
            db._verify_notes_moodboard_studio_schema_postgres(conn)
            rows = conn.execute(
                "SELECT c.relname,c.relrowsecurity,c.relforcerowsecurity,"
                "c.relowner=n.nspowner AS owner_matches_schema "
                "FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace "
                "WHERE n.nspname=current_schema() AND c.relname=ANY(?) "
                "ORDER BY c.relname",
                (list(V61_TABLES),),
            ).fetchall()
            constraints = conn.execute(
                "SELECT k.conname FROM pg_constraint k JOIN pg_class c "
                "ON c.oid=k.conrelid JOIN pg_namespace n ON n.oid=c.relnamespace "
                "WHERE n.nspname=current_schema() AND c.relname=ANY(?) "
                "AND k.contype IN ('p','f','u','c') ORDER BY k.conname",
                (list(V61_TABLES),),
            ).fetchall()
            indexes = conn.execute(
                "SELECT i.relname FROM pg_index x JOIN pg_class c ON c.oid=x.indrelid "
                "JOIN pg_class i ON i.oid=x.indexrelid JOIN pg_namespace n "
                "ON n.oid=c.relnamespace WHERE n.nspname=current_schema() "
                "AND c.relname=ANY(?) ORDER BY i.relname",
                (list(V61_TABLES),),
            ).fetchall()
        assert [row["relname"] for row in rows] == sorted(V61_TABLES)
        assert all(row["relrowsecurity"] and row["relforcerowsecurity"] for row in rows)
        assert all(row["owner_matches_schema"] for row in rows)
        assert [row["conname"] for row in constraints] == sorted(V61_CONSTRAINTS)
        assert [row["relname"] for row in indexes] == sorted(V61_EXACT_INDEXES)
    finally:
        db.close_all_connections()
        backend.get_pool().close_all()


def test_postgres_initializer_sets_timeouts_before_actual_first_version_lock(
    pg_database_config: DatabaseConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[tuple[str, str]] = []
    original = CharactersRAGDB._get_schema_version_postgres

    def observe_lock(self: CharactersRAGDB, conn: Any, *, lock: bool = False) -> int:
        if lock:
            settings = self.backend.execute(
                "SELECT current_setting('lock_timeout') AS lock_timeout,"
                "current_setting('statement_timeout') AS statement_timeout",
                connection=conn,
            ).rows[0]
            observed.append(
                (str(settings["lock_timeout"]), str(settings["statement_timeout"]))
            )
        return original(self, conn, lock=lock)

    monkeypatch.setattr(CharactersRAGDB, "_get_schema_version_postgres", observe_lock)
    backend, db = _open_db(pg_database_config, owner="960016")
    try:
        assert observed
        assert observed[0] == (
            CharactersRAGDB._NOTES_MOODBOARD_STUDIO_V61_POSTGRES_LOCK_TIMEOUT,
            CharactersRAGDB._NOTES_MOODBOARD_STUDIO_V61_POSTGRES_STATEMENT_TIMEOUT,
        )
    finally:
        db.close_all_connections()
        backend.get_pool().close_all()


def test_postgres_v61_progress_catalog_is_private_and_bounded(
    pg_database_config: DatabaseConfig,
) -> None:
    backend, db = _open_db(pg_database_config)
    try:
        with db.transaction() as conn:
            columns = conn.execute(
                "SELECT a.attname,format_type(a.atttypid,a.atttypmod) AS data_type,"
                "a.attnotnull,pg_get_expr(d.adbin,d.adrelid,false) AS default_expression "
                "FROM pg_attribute a JOIN pg_class c ON c.oid=a.attrelid "
                "JOIN pg_namespace n ON n.oid=c.relnamespace "
                "LEFT JOIN pg_attrdef d ON d.adrelid=c.oid AND d.adnum=a.attnum "
                "WHERE n.nspname=current_schema() "
                "AND c.relname='chacha_schema_migration_progress' "
                "AND a.attnum>0 AND NOT a.attisdropped ORDER BY a.attnum"
            ).fetchall()
            privilege = conn.execute(
                "SELECT has_table_privilege('public',"
                "'chacha_schema_migration_progress','SELECT') AS public_select,"
                "c.relowner=n.nspowner AS owner_matches_schema "
                "FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace "
                "WHERE n.nspname=current_schema() "
                "AND c.relname='chacha_schema_migration_progress'"
            ).fetchone()
        assert [row["attname"] for row in columns] == [
            "migration_id",
            "phase",
            "keyset_cursor",
            "copied_count",
            "aggregate_fingerprint",
            "status",
            "updated_at",
        ]
        assert [row["data_type"] for row in columns] == [
            "text", "text", "text", "bigint", "text", "text",
            "timestamp with time zone",
        ]
        assert [row["attnotnull"] for row in columns] == [
            True, True, False, True, True, True, True,
        ]
        assert [row["default_expression"] for row in columns] == [
            None, None, None, "0", None, None, "CURRENT_TIMESTAMP",
        ]
        assert privilege == {
            "public_select": False,
            "owner_matches_schema": True,
        }
    finally:
        db.close_all_connections()
        backend.get_pool().close_all()


def test_current_postgres_v61_rejects_extra_policy_without_repair(
    pg_database_config: DatabaseConfig,
) -> None:
    backend, db = _open_db(pg_database_config, owner="960002")
    second_backend = None
    try:
        with backend.transaction() as conn:
            backend.execute(
                "CREATE POLICY moodboards_v61_unexpected ON moodboards "
                "USING (true) WITH CHECK (true)",
                connection=conn,
            )

        second_backend = DatabaseBackendFactory.create_backend(pg_database_config)
        with pytest.raises(CharactersRAGDBError, match="policy catalog drifted"):
            CharactersRAGDB(":memory:", client_id="960002", backend=second_backend)

        with backend.transaction() as conn:
            policies = backend.execute(
                "SELECT policyname FROM pg_policies WHERE schemaname=current_schema() "
                "AND tablename='moodboards' ORDER BY policyname",
                connection=conn,
            ).rows
        assert [row["policyname"] for row in policies] == [
            "moodboards_tenant_isolation",
            "moodboards_v61_unexpected",
        ]
    finally:
        with backend.transaction() as conn:
            backend.execute(
                "DROP POLICY IF EXISTS moodboards_v61_unexpected ON moodboards",
                connection=conn,
            )
        db.close_all_connections()
        backend.get_pool().close_all()
        if second_backend is not None:
            second_backend.get_pool().close_all()


def test_current_postgres_v61_rejects_constraint_and_progress_drift_without_repair(
    pg_database_config: DatabaseConfig,
) -> None:
    backend, db = _open_db(pg_database_config, owner="960007")
    second_backend = None
    try:
        with backend.transaction() as conn:
            backend.execute(
                "ALTER TABLE moodboards DROP CONSTRAINT moodboards_v61_owner_check",
                connection=conn,
            )
            backend.execute(
                "ALTER TABLE moodboards ADD CONSTRAINT moodboards_v61_owner_check "
                "CHECK (true)",
                connection=conn,
            )

        second_backend = DatabaseBackendFactory.create_backend(pg_database_config)
        with pytest.raises(CharactersRAGDBError, match="constraint catalog drifted"):
            CharactersRAGDB(":memory:", client_id="960007", backend=second_backend)
        second_backend.get_pool().close_all()
        second_backend = None

        with backend.transaction() as conn:
            backend.execute(
                "ALTER TABLE moodboards DROP CONSTRAINT moodboards_v61_owner_check",
                connection=conn,
            )
            backend.execute(
                "ALTER TABLE moodboards ADD CONSTRAINT moodboards_v61_owner_check "
                "CHECK (char_length(btrim(owner_user_id))>0)",
                connection=conn,
            )
            backend.execute(
                "ALTER TABLE chacha_schema_migration_progress "
                "ADD COLUMN leaked_payload TEXT",
                connection=conn,
            )

        second_backend = DatabaseBackendFactory.create_backend(pg_database_config)
        with pytest.raises(CharactersRAGDBError, match="progress catalog drifted"):
            CharactersRAGDB(":memory:", client_id="960007", backend=second_backend)

        with backend.transaction() as conn:
            columns = backend.execute(
                "SELECT attname FROM pg_attribute a "
                "JOIN pg_class c ON c.oid=a.attrelid "
                "JOIN pg_namespace n ON n.oid=c.relnamespace "
                "WHERE n.nspname=current_schema() "
                "AND c.relname='chacha_schema_migration_progress' "
                "AND a.attnum>0 AND NOT a.attisdropped ORDER BY a.attnum",
                connection=conn,
            ).rows
        assert columns[-1] == {"attname": "leaked_payload"}
    finally:
        with backend.transaction() as conn:
            backend.execute(
                "ALTER TABLE moodboards DROP CONSTRAINT IF EXISTS "
                "moodboards_v61_owner_check",
                connection=conn,
            )
            backend.execute(
                "ALTER TABLE moodboards ADD CONSTRAINT moodboards_v61_owner_check "
                "CHECK (char_length(btrim(owner_user_id))>0)",
                connection=conn,
            )
            backend.execute(
                "ALTER TABLE chacha_schema_migration_progress "
                "DROP COLUMN IF EXISTS leaked_payload",
                connection=conn,
            )
        db.close_all_connections()
        backend.get_pool().close_all()
        if second_backend is not None:
            second_backend.get_pool().close_all()


@pytest.mark.parametrize(
    ("mutations", "cleanups", "error", "evidence_sql"),
    (
        (
            (
                "ALTER TABLE moodboards ADD CONSTRAINT "
                "moodboards_unexpected_check CHECK (true)",
            ),
            (
                "ALTER TABLE moodboards DROP CONSTRAINT IF EXISTS "
                "moodboards_unexpected_check",
            ),
            "constraint catalog drifted",
            "SELECT EXISTS(SELECT 1 FROM pg_constraint "
            "WHERE conname='moodboards_unexpected_check') AS drift_present",
        ),
        (
            (
                "ALTER TABLE moodboard_notes ADD CONSTRAINT "
                "moodboards_v61_owner_check "
                "CHECK(char_length(btrim(owner_user_id))>0)",
            ),
            (
                "ALTER TABLE moodboard_notes DROP CONSTRAINT IF EXISTS "
                "moodboards_v61_owner_check",
            ),
            "constraint catalog drifted",
            "SELECT EXISTS(SELECT 1 FROM pg_constraint k JOIN pg_class c "
            "ON c.oid=k.conrelid WHERE c.relname='moodboard_notes' "
            "AND k.conname='moodboards_v61_owner_check') AS drift_present",
        ),
        (
            ("CREATE INDEX idx_moodboards_unexpected ON moodboards(name)",),
            ("DROP INDEX IF EXISTS idx_moodboards_unexpected",),
            "index catalog drifted",
            "SELECT to_regclass('idx_moodboards_unexpected') IS NOT NULL "
            "AS drift_present",
        ),
        (
            ("DROP INDEX idx_moodboards_deleted",),
            ("CREATE INDEX idx_moodboards_deleted ON moodboards(deleted)",),
            "index catalog drifted",
            "SELECT to_regclass('idx_moodboards_deleted') IS NULL AS drift_present",
        ),
        (
            (
                "ALTER TABLE note_studio_documents DROP CONSTRAINT "
                "note_studio_documents_template_type_check",
                "ALTER TABLE note_studio_documents ADD CONSTRAINT "
                "note_studio_documents_template_type_check CHECK (true)",
            ),
            (
                "ALTER TABLE note_studio_documents DROP CONSTRAINT IF EXISTS "
                "note_studio_documents_template_type_check",
                "ALTER TABLE note_studio_documents ADD CONSTRAINT "
                "note_studio_documents_template_type_check "
                "CHECK(template_type IN ('lined','grid','cornell'))",
            ),
            "constraint catalog drifted",
            "SELECT pg_get_expr(conbin,conrelid,false)='true' AS drift_present "
            "FROM pg_constraint "
            "WHERE conname='note_studio_documents_template_type_check'",
        ),
    ),
    ids=(
        "unexpected-check",
        "expected-name-on-wrong-table",
        "extra-index",
        "missing-retained-index",
        "drifted-retained-check",
    ),
)
def test_current_postgres_v61_rejects_complete_constraint_and_index_drift(
    pg_database_config: DatabaseConfig,
    mutations: tuple[str, ...],
    cleanups: tuple[str, ...],
    error: str,
    evidence_sql: str,
) -> None:
    backend, db = _open_db(pg_database_config, owner="960018")
    drift_backend = None
    try:
        with backend.transaction() as conn:
            for statement in mutations:
                backend.execute(statement, connection=conn)
        drift_backend = DatabaseBackendFactory.create_backend(pg_database_config)
        with pytest.raises(CharactersRAGDBError, match=error):
            CharactersRAGDB(":memory:", client_id="960018", backend=drift_backend)
        with backend.transaction() as conn:
            evidence = backend.execute(evidence_sql, connection=conn).rows
        assert evidence == [{"drift_present": True}]
    finally:
        with backend.transaction() as conn:
            for statement in cleanups:
                backend.execute(statement, connection=conn)
        db.close_all_connections()
        backend.get_pool().close_all()
        if drift_backend is not None:
            drift_backend.get_pool().close_all()


@pytest.mark.parametrize(
    ("mutation", "cleanup"),
    (
        (
            "ALTER TABLE moodboards ALTER COLUMN description SET NOT NULL",
            "ALTER TABLE moodboards ALTER COLUMN description DROP NOT NULL",
        ),
        (
            "ALTER TABLE moodboards ALTER COLUMN source_diagnostic_code "
            "TYPE varchar(64)",
            "ALTER TABLE moodboards ALTER COLUMN source_diagnostic_code TYPE text",
        ),
        (
            "ALTER TABLE moodboards ALTER COLUMN description "
            "SET DEFAULT 'unexpected'",
            "ALTER TABLE moodboards ALTER COLUMN description DROP DEFAULT",
        ),
    ),
)
def test_current_postgres_v61_rejects_exact_column_metadata_drift_without_repair(
    pg_database_config: DatabaseConfig,
    mutation: str,
    cleanup: str,
) -> None:
    backend, db = _open_db(pg_database_config, owner="960012")
    drift_backend = None
    try:
        with backend.transaction() as conn:
            backend.execute(mutation, connection=conn)
        drift_backend = DatabaseBackendFactory.create_backend(pg_database_config)
        with pytest.raises(CharactersRAGDBError, match="column metadata drifted"):
            CharactersRAGDB(":memory:", client_id="960012", backend=drift_backend)
        with backend.transaction() as conn:
            metadata = backend.execute(
                "SELECT format_type(a.atttypid,a.atttypmod) AS data_type,"
                "a.attnotnull,pg_get_expr(d.adbin,d.adrelid,false) AS default_expression "
                "FROM pg_attribute a JOIN pg_class c ON c.oid=a.attrelid "
                "JOIN pg_namespace n ON n.oid=c.relnamespace "
                "LEFT JOIN pg_attrdef d ON d.adrelid=c.oid AND d.adnum=a.attnum "
                "WHERE n.nspname=current_schema() AND c.relname='moodboards' "
                "AND a.attname=CASE WHEN %s LIKE '%%source_diagnostic_code%%' "
                "THEN 'source_diagnostic_code' ELSE 'description' END",
                (mutation,),
                connection=conn,
            ).rows
        assert len(metadata) == 1
    finally:
        with backend.transaction() as conn:
            backend.execute(cleanup, connection=conn)
        db.close_all_connections()
        backend.get_pool().close_all()
        if drift_backend is not None:
            drift_backend.get_pool().close_all()


def test_current_postgres_v61_rejects_progress_acl_and_owner_drift(
    pg_database_config: DatabaseConfig,
) -> None:
    backend, db = _open_db(pg_database_config, owner="960013")
    ident = backend.escape_identifier  # type: ignore[attr-defined]
    role = f"v61_progress_drift_{uuid4().hex[:8]}"
    role_created = False
    original_schema_owner = ""
    current_login = ""
    product_relations = (
        "note_task_scope_authority",
        "moodboards",
        "moodboard_notes",
        "note_studio_documents",
    )
    try:
        with backend.transaction() as conn:
            original_schema_owner = str(
                backend.execute(
                    "SELECT r.rolname FROM pg_namespace n JOIN pg_roles r "
                    "ON r.oid=n.nspowner WHERE n.nspname=current_schema()",
                    connection=conn,
                ).rows[0]["rolname"]
            )
            current_login = str(
                backend.execute("SELECT current_user AS name", connection=conn).rows[0][
                    "name"
                ]
            )
            backend.execute(
                f"CREATE ROLE {ident(role)} NOLOGIN NOSUPERUSER NOBYPASSRLS",
                connection=conn,
            )
            backend.execute(
                f"GRANT SELECT,INSERT,UPDATE,DELETE,TRUNCATE,REFERENCES,TRIGGER "
                f"ON chacha_schema_migration_progress TO {ident(role)}",
                connection=conn,
            )
        role_created = True
        drift_backend = DatabaseBackendFactory.create_backend(pg_database_config)
        with pytest.raises(CharactersRAGDBError, match="progress ACL drifted"):
            CharactersRAGDB(":memory:", client_id="960013", backend=drift_backend)
        drift_backend.get_pool().close_all()

        with backend.transaction() as conn:
            backend.execute(
                f"REVOKE ALL ON chacha_schema_migration_progress FROM {ident(role)}",
                connection=conn,
            )
            backend.execute(
                "GRANT TRUNCATE,REFERENCES,TRIGGER ON "
                "chacha_schema_migration_progress TO PUBLIC",
                connection=conn,
            )
        drift_backend = DatabaseBackendFactory.create_backend(pg_database_config)
        with pytest.raises(CharactersRAGDBError, match="progress ACL drifted"):
            CharactersRAGDB(":memory:", client_id="960013", backend=drift_backend)
        drift_backend.get_pool().close_all()

        with backend.transaction() as conn:
            backend.execute(
                "REVOKE TRUNCATE,REFERENCES,TRIGGER ON "
                "chacha_schema_migration_progress FROM PUBLIC",
                connection=conn,
            )
            backend.execute(
                f"REVOKE TRIGGER ON chacha_schema_migration_progress "
                f"FROM {ident(original_schema_owner)}",
                connection=conn,
            )
        drift_backend = DatabaseBackendFactory.create_backend(pg_database_config)
        with pytest.raises(CharactersRAGDBError, match="progress ACL drifted"):
            CharactersRAGDB(":memory:", client_id="960013", backend=drift_backend)
        drift_backend.get_pool().close_all()

        with backend.transaction() as conn:
            backend.execute(
                f"GRANT TRIGGER ON chacha_schema_migration_progress "
                f"TO {ident(original_schema_owner)}",
                connection=conn,
            )
            backend.execute(
                f"GRANT {ident(role)} TO {ident(current_login)}", connection=conn
            )
            for relation in product_relations:
                backend.execute(
                    f"ALTER TABLE {ident(relation)} OWNER TO {ident(role)}",
                    connection=conn,
                )
            backend.execute(
                f"ALTER SCHEMA public OWNER TO {ident(role)}", connection=conn
            )
        drift_backend = DatabaseBackendFactory.create_backend(pg_database_config)
        with pytest.raises(CharactersRAGDBError, match="progress catalog drifted"):
            CharactersRAGDB(":memory:", client_id="960013", backend=drift_backend)
        drift_backend.get_pool().close_all()
    finally:
        with backend.transaction() as conn:
            if original_schema_owner:
                backend.execute(
                    f"ALTER SCHEMA public OWNER TO {ident(original_schema_owner)}",
                    connection=conn,
                )
                for relation in product_relations:
                    backend.execute(
                        f"ALTER TABLE {ident(relation)} OWNER TO "
                        f"{ident(original_schema_owner)}",
                        connection=conn,
                    )
            backend.execute(
                "REVOKE TRUNCATE,REFERENCES,TRIGGER ON "
                "chacha_schema_migration_progress FROM PUBLIC",
                connection=conn,
            )
            if original_schema_owner:
                backend.execute(
                    f"GRANT ALL PRIVILEGES ON chacha_schema_migration_progress "
                    f"TO {ident(original_schema_owner)}",
                    connection=conn,
                )
            if role_created:
                backend.execute(
                    f"REVOKE ALL ON chacha_schema_migration_progress FROM {ident(role)}",
                    connection=conn,
                )
                if current_login:
                    backend.execute(
                        f"REVOKE {ident(role)} FROM {ident(current_login)}",
                        connection=conn,
                    )
                backend.execute(f"DROP ROLE {ident(role)}", connection=conn)
        db.close_all_connections()
        backend.get_pool().close_all()


def test_current_postgres_v61_requires_product_owner_to_equal_schema_owner(
    pg_database_config: DatabaseConfig,
) -> None:
    backend, db = _open_db(pg_database_config, owner="960017")
    ident = backend.escape_identifier  # type: ignore[attr-defined]
    schema_owner = ""
    login_owner = ""
    drift_backend = None
    try:
        with backend.transaction() as conn:
            ownership = backend.execute(
                "SELECT schema_role.rolname AS schema_owner,"
                "product_role.rolname AS product_owner,current_user AS login_owner "
                "FROM pg_namespace n JOIN pg_roles schema_role "
                "ON schema_role.oid=n.nspowner JOIN pg_class product "
                "ON product.relnamespace=n.oid AND product.relname='moodboards' "
                "JOIN pg_roles product_role ON product_role.oid=product.relowner "
                "WHERE n.nspname=current_schema()",
                connection=conn,
            ).rows[0]
            schema_owner = str(ownership["schema_owner"])
            login_owner = str(ownership["login_owner"])
            assert ownership["product_owner"] == schema_owner
            assert login_owner != schema_owner
            backend.execute(
                f"ALTER TABLE moodboards OWNER TO {ident(login_owner)}",
                connection=conn,
            )

        drift_backend = DatabaseBackendFactory.create_backend(pg_database_config)
        with pytest.raises(CharactersRAGDBError, match="ownership or RLS drifted"):
            CharactersRAGDB(":memory:", client_id="960017", backend=drift_backend)

        with backend.transaction() as conn:
            ownership = backend.execute(
                "SELECT c.relowner=n.nspowner AS owner_matches_schema "
                "FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace "
                "WHERE n.nspname=current_schema() AND c.relname='moodboards'",
                connection=conn,
            ).rows
        assert ownership == [{"owner_matches_schema": False}]
    finally:
        with backend.transaction() as conn:
            if schema_owner:
                backend.execute(
                    f"ALTER TABLE moodboards OWNER TO {ident(schema_owner)}",
                    connection=conn,
                )
        db.close_all_connections()
        backend.get_pool().close_all()
        if drift_backend is not None:
            drift_backend.get_pool().close_all()


def test_postgres_v61_public_moodboard_crud_sets_dataset_guc_transaction_locally(
    pg_database_config: DatabaseConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend, db = _open_db(pg_database_config, owner="960020")
    ident = backend.escape_identifier  # type: ignore[attr-defined]
    role_name = f"v61_public_crud_{uuid4().hex[:8]}"
    role_created = False
    current_user = ""
    note_id = f"00000000-0000-4000-8000-{uuid4().hex[:12]}"
    bound_dataset = f"dataset-{uuid4()}"
    try:
        db.add_note("Public CRUD note", "Body", note_id=note_id)
        db.moodboard_sync_store.bind_local_moodboard_graph_to_dataset(
            owner_user_id=str(db.client_id),
            target_dataset_id=bound_dataset,
        )
        db.moodboard_sync_store.bind_local_studio_graph_to_dataset(
            owner_user_id=str(db.client_id),
            target_dataset_id=bound_dataset,
        )
        with backend.transaction() as conn:
            backend.execute(
                "INSERT INTO moodboards("
                "name,client_id,owner_user_id,dataset_id,sync_id,canvas_json,"
                "deleted,version,canonical_revision,canonical_hash"
                ") VALUES ('hidden dataset board',?,?,?,?,?,FALSE,1,1,?)",
                (
                    str(db.client_id),
                    str(db.client_id),
                    f"hidden-{uuid4()}",
                    str(uuid4()),
                    "{}",
                    "sha256:" + "9" * 64,
                ),
                connection=conn,
            )
        db.close_connection()

        with backend.transaction() as conn:
            current_user = backend.execute(
                "SELECT current_user AS name",
                connection=conn,
            ).rows[0]["name"]
            backend.execute(
                f"CREATE ROLE {ident(role_name)} NOLOGIN NOSUPERUSER NOBYPASSRLS",
                connection=conn,
            )
            role_created = True
            backend.execute(
                f"GRANT USAGE ON SCHEMA public TO {ident(role_name)}",
                connection=conn,
            )
            backend.execute(
                f"GRANT SELECT ON ALL TABLES IN SCHEMA public TO {ident(role_name)}",
                connection=conn,
            )
            backend.execute(
                f"GRANT INSERT,UPDATE,DELETE ON moodboards,moodboard_notes,"
                f"note_studio_documents TO {ident(role_name)}",
                connection=conn,
            )
            backend.execute(
                f"GRANT USAGE,SELECT ON ALL SEQUENCES IN SCHEMA public "
                f"TO {ident(role_name)}",
                connection=conn,
            )
            backend.execute(
                f"GRANT {ident(role_name)} TO {ident(current_user)}",
                connection=conn,
            )

        monkeypatch.setenv("TLDW_CONTENT_PG_ROLE_SWITCH", "1")
        monkeypatch.setenv("TLDW_CONTENT_PG_ROLE_WHITELIST", role_name)

        def assert_dataset_guc_cleared() -> None:
            with db.transaction() as conn:
                row = conn.execute(
                    "SELECT current_setting('app.current_dataset_id', true) "
                    "AS dataset"
                ).fetchone()
            assert row is not None
            assert row["dataset"] in ("", None)

        with scoped_context(
            user_id=int(str(db.client_id)),
            session_role=role_name,
        ):
            moodboard_id = db.add_moodboard("Bound board", "scoped")
            assert isinstance(moodboard_id, int)
            assert_dataset_guc_cleared()

            created = db.get_moodboard_by_id(moodboard_id)
            assert created is not None
            assert created["dataset_id"] == bound_dataset
            assert created["name"] == "Bound board"
            assert db.count_moodboards() == 1
            assert [row["id"] for row in db.list_moodboards()] == [moodboard_id]
            assert_dataset_guc_cleared()

            assert db.update_moodboard(
                moodboard_id,
                {"name": "Renamed bound board"},
                expected_version=int(created["version"]),
            )
            updated = db.get_moodboard_by_id(moodboard_id)
            assert updated is not None
            assert updated["name"] == "Renamed bound board"
            assert int(updated["version"]) == int(created["version"]) + 1

            studio = db.create_note_studio_document(
                note_id=note_id,
                payload_json={"sections": []},
                template_type="lined",
                handwriting_mode="off",
                render_version=1,
            )
            assert studio["dataset_id"] == bound_dataset
            assert db.get_note_studio_document(note_id)["dataset_id"] == bound_dataset
            assert_dataset_guc_cleared()

            assert db.link_note_to_moodboard(moodboard_id, note_id) is True
            assert db.link_note_to_moodboard(moodboard_id, note_id) is False
            assert db.count_moodboard_notes(moodboard_id) == 1
            listed_notes = db.list_moodboard_notes(moodboard_id)
            assert [row["id"] for row in listed_notes] == [note_id]
            assert listed_notes[0]["membership_source"] == "manual"

            assert db.unlink_note_from_moodboard(moodboard_id, note_id) is True
            assert db.unlink_note_from_moodboard(moodboard_id, note_id) is False
            assert db.count_moodboard_notes(moodboard_id) == 0
            assert db.list_moodboard_notes(moodboard_id) == []
            assert db.link_note_to_moodboard(moodboard_id, note_id) is True
            assert db.count_moodboard_notes(moodboard_id) == 1
            assert_dataset_guc_cleared()

            assert db.delete_moodboard(
                moodboard_id,
                expected_version=int(updated["version"]),
            )
            assert db.get_moodboard_by_id(moodboard_id) is None
            deleted = db.get_moodboard_by_id(moodboard_id, include_deleted=True)
            assert deleted is not None
            assert deleted["dataset_id"] == bound_dataset
            assert bool(deleted["deleted"]) is True
            assert db.count_moodboards() == 0
            assert db.count_moodboards(only_deleted=True) == 1
            assert db.list_moodboards() == []
            assert [row["id"] for row in db.list_moodboards(only_deleted=True)] == [
                moodboard_id
            ]
            assert_dataset_guc_cleared()
    finally:
        if role_created:
            with backend.transaction() as conn:
                backend.execute(f"DROP OWNED BY {ident(role_name)}", connection=conn)
                if current_user:
                    backend.execute(
                        f"REVOKE {ident(role_name)} FROM {ident(current_user)}",
                        connection=conn,
                    )
                backend.execute(f"DROP ROLE IF EXISTS {ident(role_name)}", connection=conn)
        db.close_all_connections()
        backend.get_pool().close_all()


def test_postgres_v61_moodboard_public_sql_uses_backend_boolean_values() -> None:
    sources = "\n".join(
        inspect.getsource(method)
        for method in (
            CharactersRAGDB._update_moodboard_v61,
            CharactersRAGDB.delete_moodboard,
            CharactersRAGDB.unlink_note_from_moodboard,
            CharactersRAGDB._build_moodboard_note_union_query,
        )
    )

    assert "deleted=0" not in sources
    assert "deleted = 0" not in sources
    assert "deleted=1" not in sources
    assert "deleted = 1" not in sources


def test_postgres_v61_binders_rekey_local_unbound_under_force_rls_without_dataset_guc(
    pg_database_config: DatabaseConfig,
) -> None:
    owner = "960021"
    target = f"dataset-{uuid4()}"
    backend, db = _open_db(pg_database_config, owner=owner)
    note_id = f"00000000-0000-4000-8000-{uuid4().hex[:12]}"
    try:
        _seed_local_unbound_moodboard_studio_graph(db, note_id=note_id)
        assert all(_postgres_force_rls_flags(backend).values())
        with db.transaction() as conn:
            row = conn.execute(
                "SELECT current_setting('app.current_dataset_id', true) AS dataset"
            ).fetchone()
        assert row is not None
        assert row["dataset"] in ("", None)

        moodboard_counts = db.moodboard_sync_store.bind_local_moodboard_graph_to_dataset(
            owner_user_id=owner,
            target_dataset_id=target,
        )
        studio_counts = db.moodboard_sync_store.bind_local_studio_graph_to_dataset(
            owner_user_id=owner,
            target_dataset_id=target,
        )

        assert moodboard_counts == {"moodboards": 1, "moodboard_notes": 1}
        assert studio_counts == {"note_studio_documents": 1}
        state = _postgres_v61_scope_state(backend, owner=owner)
        assert state["moodboards"] == {target: 1}
        assert state["moodboard_notes"] == {target: 1}
        assert state["note_studio_documents"] == {target: 1}
        assert state["authority"] == [
            {
                "dataset_id": target,
                "task_graph_bound": False,
                "moodboard_graph_bound": True,
                "studio_graph_bound": True,
            }
        ]
        assert all(_postgres_force_rls_flags(backend).values())
    finally:
        db.close_all_connections()
        backend.get_pool().close_all()


def test_postgres_v61_moodboard_bind_failure_rolls_back_rekey_and_force_rls(
    pg_database_config: DatabaseConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner = "960022"
    target = f"dataset-{uuid4()}"
    backend, db = _open_db(pg_database_config, owner=owner)
    note_id = f"00000000-0000-4000-8000-{uuid4().hex[:12]}"
    try:
        _seed_local_unbound_moodboard_studio_graph(db, note_id=note_id)
        original_prove = db.moodboard_sync_store._prove_moodboard_graph

        def fail_after_rekey(conn: Any, *, owner: str, dataset: str) -> None:
            original_prove(conn, owner=owner, dataset=dataset)
            if dataset == target:
                raise ConflictError(
                    "injected bind failure",
                    entity="moodboards",
                    entity_id=target,
                )

        monkeypatch.setattr(
            db.moodboard_sync_store,
            "_prove_moodboard_graph",
            fail_after_rekey,
        )
        with db.transaction() as conn:
            with pytest.raises(ConflictError, match="injected bind failure"):
                db.moodboard_sync_store.bind_local_moodboard_graph_to_dataset(
                    owner_user_id=owner,
                    target_dataset_id=target,
                    conn=conn,
                )
            db._verify_notes_moodboard_studio_schema_postgres(conn)

        state = _postgres_v61_scope_state(backend, owner=owner)
        assert state["moodboards"] == {"local-unbound": 1}
        assert state["moodboard_notes"] == {"local-unbound": 1}
        assert state["authority"] == []
        assert all(_postgres_force_rls_flags(backend).values())
    finally:
        db.close_all_connections()
        backend.get_pool().close_all()


def test_postgres_v61_bootstrap_pages_set_dataset_guc_for_nobypassrls_role(
    pg_database_config: DatabaseConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner = "960023"
    target = f"dataset-{uuid4()}"
    role_name = f"v61_bootstrap_{uuid4().hex[:8]}"
    backend, db = _open_db(pg_database_config, owner=owner)
    ident = backend.escape_identifier  # type: ignore[attr-defined]
    role_created = False
    current_user = ""
    note_id = f"00000000-0000-4000-8000-{uuid4().hex[:12]}"
    try:
        _seed_local_unbound_moodboard_studio_graph(db, note_id=note_id)
        db.moodboard_sync_store.bind_local_moodboard_graph_to_dataset(
            owner_user_id=owner,
            target_dataset_id=target,
        )
        db.moodboard_sync_store.bind_local_studio_graph_to_dataset(
            owner_user_id=owner,
            target_dataset_id=target,
        )
        db.close_connection()

        with backend.transaction() as conn:
            current_user = backend.execute(
                "SELECT current_user AS name",
                connection=conn,
            ).rows[0]["name"]
            backend.execute(
                f"CREATE ROLE {ident(role_name)} NOLOGIN NOSUPERUSER NOBYPASSRLS",
                connection=conn,
            )
            role_created = True
            backend.execute(
                f"GRANT USAGE ON SCHEMA public TO {ident(role_name)}",
                connection=conn,
            )
            backend.execute(
                f"GRANT SELECT ON ALL TABLES IN SCHEMA public TO {ident(role_name)}",
                connection=conn,
            )
            backend.execute(
                f"GRANT {ident(role_name)} TO {ident(current_user)}",
                connection=conn,
            )

        monkeypatch.setenv("TLDW_CONTENT_PG_ROLE_SWITCH", "1")
        monkeypatch.setenv("TLDW_CONTENT_PG_ROLE_WHITELIST", role_name)

        def assert_dataset_guc_cleared() -> None:
            with db.transaction() as conn:
                row = conn.execute(
                    "SELECT current_setting('app.current_dataset_id', true) "
                    "AS dataset"
                ).fetchone()
            assert row is not None
            assert row["dataset"] in ("", None)

        with scoped_context(user_id=int(owner), session_role=role_name):
            boards = db.moodboard_sync_store.page_moodboards_for_sync_bootstrap(
                owner_user_id=owner,
                dataset_id=target,
            )
            assert len(boards) == 1
            assert boards[0]["dataset_id"] == target
            assert_dataset_guc_cleared()

            placements = db.moodboard_sync_store.page_moodboard_placements_for_sync_bootstrap(
                owner_user_id=owner,
                dataset_id=target,
            )
            assert len(placements) == 1
            assert placements[0]["dataset_id"] == target
            assert placements[0]["note_id"] == note_id
            assert_dataset_guc_cleared()

            studio_documents = db.moodboard_sync_store.page_studio_documents_for_sync_bootstrap(
                owner_user_id=owner,
                dataset_id=target,
            )
            assert len(studio_documents) == 1
            assert studio_documents[0]["dataset_id"] == target
            assert studio_documents[0]["note_id"] == note_id
            assert_dataset_guc_cleared()
    finally:
        if role_created:
            with backend.transaction() as conn:
                backend.execute(f"DROP OWNED BY {ident(role_name)}", connection=conn)
                if current_user:
                    backend.execute(
                        f"REVOKE {ident(role_name)} FROM {ident(current_user)}",
                        connection=conn,
                    )
                backend.execute(f"DROP ROLE IF EXISTS {ident(role_name)}", connection=conn)
        db.close_all_connections()
        backend.get_pool().close_all()


def test_postgres_v61_old_authority_insert_defaults_and_first_enrollment_race(
    pg_database_config: DatabaseConfig,
) -> None:
    owner = "960003"
    dataset = f"dataset-{uuid4()}"
    backend_a, db_a = _open_db(pg_database_config, owner=owner)
    backend_b, db_b = _open_db(pg_database_config, owner=owner)
    try:
        with db_a.transaction() as conn:
            conn.execute(
                "INSERT INTO note_task_scope_authority(owner_user_id,dataset_id) VALUES (?,?)",
                (owner, dataset),
            )
            row = conn.execute(
                "SELECT task_graph_bound,moodboard_graph_bound,studio_graph_bound "
                "FROM note_task_scope_authority WHERE owner_user_id=?",
                (owner,),
            ).fetchone()
        assert row == {
            "task_graph_bound": True,
            "moodboard_graph_bound": False,
            "studio_graph_bound": False,
        }
        with db_a.transaction() as conn:
            conn.execute(
                "DELETE FROM note_task_scope_authority WHERE owner_user_id=?", (owner,)
            )

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = (
                pool.submit(
                    db_a.moodboard_sync_store.bind_local_moodboard_graph_to_dataset,
                    owner_user_id=owner,
                    target_dataset_id=dataset,
                ),
                pool.submit(
                    db_b.moodboard_sync_store.bind_local_studio_graph_to_dataset,
                    owner_user_id=owner,
                    target_dataset_id=dataset,
                ),
            )
            assert [future.result(timeout=20) for future in futures] == [
                {"moodboards": 0, "moodboard_notes": 0},
                {"note_studio_documents": 0},
            ]
        with db_a.transaction() as conn:
            authority = conn.execute(
                "SELECT dataset_id,task_graph_bound,moodboard_graph_bound,studio_graph_bound "
                "FROM note_task_scope_authority WHERE owner_user_id=?",
                (owner,),
            ).fetchone()
        assert authority == {
            "dataset_id": dataset,
            "task_graph_bound": False,
            "moodboard_graph_bound": True,
            "studio_graph_bound": True,
        }
    finally:
        db_a.close_all_connections()
        db_b.close_all_connections()
        backend_a.get_pool().close_all()
        backend_b.get_pool().close_all()


def test_postgres_studio_diagram_update_rejects_stale_writer_after_lock_wait(
    pg_database_config: DatabaseConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject a stale Studio writer after a concurrent update releases its row lock."""
    owner = "960021"
    note_id = f"00000000-0000-4000-8000-{uuid4().hex[:12]}"
    backend_a, db_a = _open_db(pg_database_config, owner=owner)
    backend_b, db_b = _open_db(pg_database_config, owner=owner)
    observer_backend = DatabaseBackendFactory.create_backend(pg_database_config)
    sections = [
        {
            "id": "notes-1",
            "kind": "notes",
            "title": "Notes",
            "content": "Accepted content",
        }
    ]
    source_graph = [
        {
            "id": "notes-1",
            "title": "Notes",
            "kind": "notes",
            "content": "Accepted content",
        }
    ]
    winner_diagram = "graph TD; A-->winner"
    loser_diagram = "graph TD; A-->loser"
    winner_manifest = {
        "diagram_type": "flowchart",
        "source_section_ids": ["notes-1"],
        "source_graph": source_graph,
        "diagram": winner_diagram,
        "format": "mermaid",
        "status": "ready",
        "render_hash": diagram_render_hash(
            diagram_type="flowchart",
            context="Notes\nAccepted content",
            diagram=winner_diagram,
        ),
    }
    loser_manifest = {
        **winner_manifest,
        "diagram": loser_diagram,
        "render_hash": diagram_render_hash(
            diagram_type="flowchart",
            context="Notes\nAccepted content",
            diagram=loser_diagram,
        ),
    }
    try:
        db_a.add_note("Studio race", "Accepted content", note_id=note_id)
        before = db_a.create_note_studio_document(
            note_id=note_id,
            payload_json={"sections": sections},
            template_type="lined",
            handwriting_mode="off",
            render_version=1,
        )
        monkeypatch.setattr(
            db_a,
            "_get_current_utc_timestamp_iso",
            lambda: "2099-01-01T00:00:01.000000+00:00",
        )

        with ThreadPoolExecutor(max_workers=1) as pool:
            with db_a.transaction() as winner_conn:
                winner = db_a.update_note_studio_diagram_manifest(
                    note_id=note_id,
                    diagram_manifest_json=winner_manifest,
                    expected_companion_content_hash=before["companion_content_hash"],
                    expected_render_version=int(before["render_version"]),
                    expected_last_modified=before["last_modified"],
                    conn=winner_conn,
                )
                loser_future = pool.submit(
                    db_b.update_note_studio_diagram_manifest,
                    note_id=note_id,
                    diagram_manifest_json=loser_manifest,
                    expected_companion_content_hash=before["companion_content_hash"],
                    expected_render_version=int(before["render_version"]),
                    expected_last_modified=before["last_modified"],
                )
                waiting_on_lock = False
                deadline = time.monotonic() + 10
                while time.monotonic() < deadline and not loser_future.done():
                    with observer_backend.transaction() as observer_conn:
                        row = observer_backend.execute(
                            "SELECT EXISTS(SELECT 1 FROM pg_stat_activity "
                            "WHERE datname=current_database() AND pid<>pg_backend_pid() "
                            "AND wait_event_type='Lock' "
                            "AND query ILIKE '%note_studio_documents%') AS waiting",
                            connection=observer_conn,
                        ).rows[0]
                    waiting_on_lock = bool(row["waiting"])
                    if waiting_on_lock:
                        break
                    time.sleep(0.02)
                assert waiting_on_lock is True

            with pytest.raises(ConflictError, match="changed concurrently"):
                loser_future.result(timeout=20)

        stored = db_a.get_note_studio_document(note_id)
        assert stored is not None
        assert stored["diagram_manifest_json"]["diagram"] == winner_diagram
        assert stored["canonical_revision"] == winner["canonical_revision"]
    finally:
        db_a.close_all_connections()
        db_b.close_all_connections()
        backend_a.get_pool().close_all()
        backend_b.get_pool().close_all()
        observer_backend.get_pool().close_all()


def test_postgres_v61_conflicting_first_enrollment_has_one_winner_and_rolls_back_loser(
    pg_database_config: DatabaseConfig,
) -> None:
    owner = "960015"
    dataset_a = f"dataset-a-{uuid4()}"
    dataset_b = f"dataset-b-{uuid4()}"
    backend_a, db_a = _open_db(pg_database_config, owner=owner)
    backend_b, db_b = _open_db(pg_database_config, owner=owner)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = {
                dataset_a: pool.submit(
                    db_a.moodboard_sync_store.bind_local_moodboard_graph_to_dataset,
                    owner_user_id=owner,
                    target_dataset_id=dataset_a,
                ),
                dataset_b: pool.submit(
                    db_b.moodboard_sync_store.bind_local_moodboard_graph_to_dataset,
                    owner_user_id=owner,
                    target_dataset_id=dataset_b,
                ),
            }
            outcomes: dict[str, object] = {}
            for dataset, future in futures.items():
                try:
                    outcomes[dataset] = future.result(timeout=20)
                except Exception as exc:  # noqa: BLE001 - concurrent loser is asserted below.
                    outcomes[dataset] = exc
        winners = [
            dataset for dataset, outcome in outcomes.items() if isinstance(outcome, dict)
        ]
        losers = [
            outcome for outcome in outcomes.values() if isinstance(outcome, Exception)
        ]
        assert len(winners) == len(losers) == 1
        assert isinstance(losers[0], CharactersRAGDBError)
        with backend_a.transaction() as conn:
            authority = backend_a.execute(
                "SELECT dataset_id,task_graph_bound,moodboard_graph_bound,"
                "studio_graph_bound FROM note_task_scope_authority "
                "WHERE owner_user_id=?",
                (owner,),
                connection=conn,
            ).rows
            wrong_scope_rows = backend_a.execute(
                "SELECT count(*) AS count FROM moodboards "
                "WHERE owner_user_id=? AND dataset_id<>?",
                (owner, winners[0]),
                connection=conn,
            ).rows[0]["count"]
        assert authority == [
            {
                "dataset_id": winners[0],
                "task_graph_bound": False,
                "moodboard_graph_bound": True,
                "studio_graph_bound": False,
            }
        ]
        assert wrong_scope_rows == 0
    finally:
        db_a.close_all_connections()
        db_b.close_all_connections()
        backend_a.get_pool().close_all()
        backend_b.get_pool().close_all()


def _restore_large_postgres_v62_fixture(
    db: CharactersRAGDB,
    *,
    owner: str,
    row_count: int,
) -> None:
    """Replace a fresh product graph with the exact-shape v62 predecessor."""
    board_columns = (
        "owner_user_id", "dataset_id", "sync_id", "canvas_json",
        "canonical_revision", "canonical_hash", "source_diagnostic_code",
        "source_diagnostic_hash",
    )
    placement_columns = (
        "owner_user_id", "dataset_id", "placement_id", "x", "y", "width",
        "height", "order_index", "display_json", "last_modified", "deleted",
        "version", "canonical_revision", "canonical_hash",
        "source_diagnostic_code", "source_diagnostic_hash",
    )
    studio_columns = (
        "owner_user_id", "dataset_id", "note_revision", "note_hash",
        "accepted_provenance_json", "deleted", "version", "canonical_revision",
        "canonical_hash", "source_diagnostic_code", "source_diagnostic_hash",
    )
    with db.transaction() as conn:
        for table in V61_TABLES:
            conn.execute(
                f"DROP POLICY IF EXISTS {table}_tenant_isolation ON {table}"  # nosec B608
            )
            conn.execute(f"ALTER TABLE {table} DISABLE ROW LEVEL SECURITY")  # nosec B608
        for index in V61_INDEXES:
            conn.execute(f"DROP INDEX IF EXISTS {index}")  # nosec B608
        constraints = conn.execute(
            "SELECT c.relname,k.conname FROM pg_constraint k "
            "JOIN pg_class c ON c.oid=k.conrelid "
            "JOIN pg_namespace n ON n.oid=c.relnamespace "
            "WHERE n.nspname=current_schema() AND k.conname LIKE '%\\_v61\\_%' ESCAPE '\\' "
            "ORDER BY CASE WHEN k.contype='f' THEN 0 ELSE 1 END,k.conname"
        ).fetchall()
        for row in constraints:
            conn.execute(
                f"ALTER TABLE {row['relname']} DROP CONSTRAINT {row['conname']}"  # nosec B608
            )
        for column in (
            "task_graph_bound", "moodboard_graph_bound", "studio_graph_bound",
        ):
            conn.execute(
                f"ALTER TABLE note_task_scope_authority DROP COLUMN {column}"  # nosec B608
            )
        for table, columns in (
            ("moodboards", board_columns),
            ("moodboard_notes", placement_columns),
            ("note_studio_documents", studio_columns),
        ):
            for column in columns:
                conn.execute(f"ALTER TABLE {table} DROP COLUMN {column}")  # nosec B608
        conn.execute("DROP TABLE chacha_schema_migration_progress")
        db._set_schema_version_postgres(conn, 62)

        conn.execute("SELECT set_config('app.current_user_id', ?, true)", (owner,))
        conn.execute(
            "INSERT INTO notes(id,title,content,client_id) "
            "SELECT 'legacy-note-' || lpad(i::text,4,'0'),"
            "'Legacy note ' || i::text,'Legacy body ' || i::text,? "
            "FROM generate_series(1,?) AS i",
            (owner, row_count),
        )
        conn.execute(
            "INSERT INTO moodboards(name,description,smart_rule_json,client_id) "
            "SELECT 'Legacy board ' || i::text,'Legacy description',"
            "'{\"query\":\"legacy\"}',? FROM generate_series(1,?) AS i",
            (owner, row_count),
        )
        conn.execute(
            "INSERT INTO moodboard_notes(moodboard_id,note_id) "
            "SELECT id,'legacy-note-' || lpad("
            "row_number() OVER (ORDER BY id)::text,4,'0') FROM moodboards"
        )
        conn.execute(
            "INSERT INTO note_studio_documents("
            "note_id,payload_json,template_type,handwriting_mode,render_version) "
            "SELECT 'legacy-note-' || lpad(i::text,4,'0'),'{}','lined','off',1 "
            "FROM generate_series(1,?) AS i",
            (row_count,),
        )


def test_postgres_v61_placement_order_is_board_global_across_resumed_pages(
    pg_database_config: DatabaseConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner = "960019"
    first_board_count = 260
    second_board_count = 5
    first_board_rank_multiplier = 137
    backend, db = _open_db(pg_database_config, owner=owner)
    try:
        _restore_large_postgres_v62_fixture(db, owner=owner, row_count=2)
        with db.transaction() as conn:
            conn.execute("DELETE FROM moodboard_notes")
            conn.execute("DELETE FROM note_studio_documents")
            conn.execute("DELETE FROM notes")
            board_ids = [
                row["id"]
                for row in conn.execute(
                    "SELECT id FROM moodboards ORDER BY id"
                ).fetchall()
            ]
            conn.execute(
                "INSERT INTO notes(id,title,content,client_id) "
                "SELECT '00000000-0000-4000-8000-' || lpad(i::text,12,'0'),"
                "'Board one note ' || i::text,'Body',? "
                "FROM generate_series(1,?) AS i",
                (owner, first_board_count),
            )
            conn.execute(
                "INSERT INTO moodboard_notes(moodboard_id,note_id,created_at) "
                "SELECT ?,'00000000-0000-4000-8000-' || lpad(i::text,12,'0'),"
                "TIMESTAMPTZ '2026-01-01 00:00:00+00' + "
                "(mod(i * ?, ?) * INTERVAL '1 second') "
                "FROM generate_series(1,?) AS i",
                (
                    board_ids[0],
                    first_board_rank_multiplier,
                    first_board_count,
                    first_board_count,
                ),
            )
            conn.execute(
                "INSERT INTO notes(id,title,content,client_id) "
                "SELECT '00000000-0000-4000-8000-' || lpad(i::text,12,'0'),"
                "'Board two note ' || i::text,'Body',? "
                "FROM generate_series(1001,?) AS i",
                (owner, 1000 + second_board_count),
            )
            conn.execute(
                "INSERT INTO moodboard_notes(moodboard_id,note_id,created_at) "
                "SELECT ?,'00000000-0000-4000-8000-' || lpad(i::text,12,'0'),"
                "TIMESTAMPTZ '2026-02-01 00:00:00+00' + "
                "((? - i) * INTERVAL '1 second') "
                "FROM generate_series(1001,?) AS i",
                (
                    board_ids[1],
                    1000 + second_board_count,
                    1000 + second_board_count,
                ),
            )

        failed = False

        def interrupt_after_first_placement_copy_page(label: str) -> None:
            nonlocal failed
            if label == "copy:moodboard_notes:128" and not failed:
                failed = True
                raise RuntimeError("injected placement-page interruption")

        monkeypatch.setattr(
            db,
            "_notes_moodboard_studio_v61_postgres_checkpoint",
            interrupt_after_first_placement_copy_page,
        )
        with pytest.raises(RuntimeError, match="placement-page interruption"):
            with db.transaction() as conn:
                db._migrate_from_v62_to_v63_postgres(conn)
        assert failed
        with backend.transaction() as conn:
            version = db._get_schema_version_postgres(conn)
            partial = backend.execute(
                "SELECT copied_count,status FROM chacha_schema_migration_progress "
                "WHERE migration_id=? AND phase='moodboard_notes'",
                (db._NOTES_MOODBOARD_STUDIO_V61_MIGRATION_ID,),
                connection=conn,
            ).rows
        assert version == 62
        assert partial == [{"copied_count": 128, "status": "running"}]

        monkeypatch.setattr(
            db,
            "_notes_moodboard_studio_v61_postgres_checkpoint",
            lambda _label: None,
        )
        with db.transaction() as conn:
            db._migrate_from_v62_to_v63_postgres(conn)

        with backend.transaction() as conn:
            placements = backend.execute(
                "SELECT moodboard_id,note_id,created_at,order_index FROM moodboard_notes "
                "ORDER BY moodboard_id,created_at,note_id",
                connection=conn,
            ).rows
            fingerprints = backend.execute(
                "SELECT phase,copied_count,aggregate_fingerprint,status "
                "FROM chacha_schema_migration_progress WHERE migration_id=? "
                "AND phase IN ('source_prediction:moodboard_notes',"
                "'aggregate_verification:moodboard_notes') ORDER BY phase",
                (db._NOTES_MOODBOARD_STUDIO_V61_MIGRATION_ID,),
                connection=conn,
            ).rows

        first_board = [
            row for row in placements if row["moodboard_id"] == board_ids[0]
        ]
        second_board = [
            row for row in placements if row["moodboard_id"] == board_ids[1]
        ]
        assert [row["order_index"] for row in first_board] == list(
            range(first_board_count)
        )
        assert [row["order_index"] for row in second_board] == list(
            range(second_board_count)
        )
        expected_first_board_note_ids = [
            f"00000000-0000-4000-8000-{note_number:012d}"
            for note_number in sorted(
                range(1, first_board_count + 1),
                key=lambda value: (
                    (value * first_board_rank_multiplier) % first_board_count,
                    f"00000000-0000-4000-8000-{value:012d}",
                ),
            )
        ]
        assert [row["note_id"] for row in first_board] == expected_first_board_note_ids
        assert [
            (row["note_id"], row["order_index"]) for row in first_board
        ] == [
            (note_id, order_index)
            for order_index, note_id in enumerate(expected_first_board_note_ids)
        ]
        assert [row["copied_count"] for row in fingerprints] == [
            first_board_count + second_board_count,
            first_board_count + second_board_count,
        ]
        assert len({row["aggregate_fingerprint"] for row in fingerprints}) == 1
        assert {row["status"] for row in fingerprints} == {"complete"}
    finally:
        db.close_all_connections()
        backend.get_pool().close_all()


def test_postgres_v61_upgrade_matches_sqlite_rule_and_studio_conversion(
    pg_database_config: DatabaseConfig,
) -> None:
    owner = "960010"
    other_owner = "960011"
    collection_sync_id = str(uuid4())
    valid_note_id = str(uuid4())
    source_note_id = str(uuid4())
    backend, db = _open_db(pg_database_config, owner=owner)
    try:
        _restore_large_postgres_v62_fixture(db, owner=owner, row_count=4)
        with db.transaction() as conn:
            conn.execute(
                "UPDATE notes SET id=? WHERE id='legacy-note-0001'", (valid_note_id,)
            )
            conn.execute(
                "UPDATE notes SET id=? WHERE id='legacy-note-0002'", (source_note_id,)
            )
            collection = conn.execute(
                "INSERT INTO keyword_collections(sync_id,name,client_id) "
                "VALUES (?,?,?) RETURNING id",
                (collection_sync_id, "Legacy portable collection", owner),
            ).fetchone()
            board_ids = [
                row["id"]
                for row in conn.execute("SELECT id FROM moodboards ORDER BY id").fetchall()
            ]
            conn.execute(
                "UPDATE moodboards SET smart_rule_json=? WHERE id=?",
                (json.dumps({"collection_ids": [collection["id"]]}), board_ids[0]),
            )
            conn.execute(
                "UPDATE moodboards SET smart_rule_json=? WHERE id=?",
                (json.dumps({"query": "legacy", "future_key": True}), board_ids[1]),
            )
            companion_hash = "sha256:" + hashlib.sha256(b"Legacy body 1").hexdigest()
            excerpt = "Legacy body"
            excerpt_hash = "sha256:" + hashlib.sha256(excerpt.encode()).hexdigest()
            valid_payload = {
                "meta": {
                    "title": "Legacy note 1",
                    "source_note_id": source_note_id,
                },
                "layout": {"render_version": 1},
                "sections": [
                    {
                        "id": "section-1",
                        "kind": "notes",
                        "title": "Summary",
                        "content": "Accepted content",
                    }
                ],
            }
            source_graph = valid_payload["sections"]
            diagram = "graph TD; A-->B"
            manifest = {
                "diagram_type": "flowchart",
                "source_section_ids": ["section-1"],
                "source_graph": source_graph,
                "diagram": diagram,
                "format": "mermaid",
                "status": "ready",
                "render_hash": diagram_render_hash(
                    diagram_type="flowchart",
                    context="Summary\nAccepted content",
                    diagram=diagram,
                ),
                "canonical_source": source_graph,
                "generation_status": "ready",
                "cached_svg": "<svg>derived cache</svg>",
            }
            conn.execute(
                "UPDATE note_studio_documents SET payload_json=?,source_note_id=?,"
                "excerpt_snapshot=?,excerpt_hash=?,companion_content_hash=?,"
                "diagram_manifest_json=? "
                "WHERE note_id=?",
                (
                    json.dumps(valid_payload),
                    source_note_id,
                    excerpt,
                    excerpt_hash,
                    companion_hash,
                    json.dumps(manifest),
                    valid_note_id,
                ),
            )
            conn.execute(
                "UPDATE note_studio_documents SET payload_json=?,companion_content_hash=? "
                "WHERE note_id=?",
                (
                    json.dumps({"meta": {"title": "wrong"}, "sections": []}),
                    "sha256:" + hashlib.sha256(b"Legacy body 2").hexdigest(),
                    source_note_id,
                ),
            )
            conn.execute(
                "INSERT INTO notes(id,title,content,client_id) VALUES (?,?,?,?)",
                ("cross-owner-source", "Cross owner", "Source", other_owner),
            )
            conn.execute(
                "UPDATE note_studio_documents SET payload_json=?,source_note_id=?,"
                "companion_content_hash=? WHERE note_id='legacy-note-0003'",
                (
                    json.dumps({"sections": []}),
                    "cross-owner-source",
                    "sha256:" + hashlib.sha256(b"Legacy body 3").hexdigest(),
                ),
            )
            source_fk = conn.execute(
                "SELECT k.conname FROM pg_constraint k JOIN pg_class c ON c.oid=k.conrelid "
                "JOIN pg_class parent ON parent.oid=k.confrelid "
                "WHERE c.relname='note_studio_documents' AND parent.relname='notes' "
                "AND k.contype='f' AND array_length(k.conkey,1)=1 "
                "AND (SELECT a.attname FROM pg_attribute a "
                "WHERE a.attrelid=c.oid AND a.attnum=k.conkey[1])='source_note_id'"
            ).fetchone()
            if source_fk is not None:
                conn.execute(
                    f"ALTER TABLE note_studio_documents DROP CONSTRAINT {source_fk['conname']}"  # nosec B608 - server catalog identifier.
                )
            conn.execute(
                "UPDATE note_studio_documents SET payload_json=?,source_note_id=?,"
                "companion_content_hash=? WHERE note_id='legacy-note-0004'",
                (
                    json.dumps({"sections": []}),
                    "unknown-source",
                    "sha256:" + hashlib.sha256(b"Legacy body 4").hexdigest(),
                ),
            )

        with db.transaction() as conn:
            db._migrate_from_v62_to_v63_postgres(conn)

        with backend.transaction() as conn:
            boards = backend.execute(
                "SELECT id,smart_rule_json,source_diagnostic_code FROM moodboards "
                "ORDER BY id",
                connection=conn,
            ).rows
            studios = {
                row["note_id"]: row
                for row in backend.execute(
                    "SELECT * FROM note_studio_documents ORDER BY note_id",
                    connection=conn,
                ).rows
            }
        assert json.loads(boards[0]["smart_rule_json"])["collection_sync_ids"] == [
            collection_sync_id
        ]
        assert boards[0]["source_diagnostic_code"] is None
        assert boards[1]["smart_rule_json"] is None
        assert boards[1]["source_diagnostic_code"] == "legacy_moodboard_rule_invalid"

        valid = studios[valid_note_id]
        assert valid["source_diagnostic_code"] is None
        assert set(json.loads(valid["payload_json"])) == {"sections"}
        assert not {
            "canonical_source", "generation_status", "cached_svg"
        }.intersection(json.loads(valid["diagram_manifest_json"]))
        provenance = json.loads(valid["accepted_provenance_json"])
        parsed = parse_notes_studio_document_v1(
            {
                "note_id": valid["note_id"],
                "source_note_id": valid["source_note_id"],
                "payload_json": json.loads(valid["payload_json"]),
                "template_type": valid["template_type"],
                "handwriting_mode": valid["handwriting_mode"],
                "excerpt_snapshot": valid["excerpt_snapshot"],
                "excerpt_hash": valid["excerpt_hash"],
                "diagram_manifest_json": json.loads(valid["diagram_manifest_json"]),
                "companion_content_hash": valid["companion_content_hash"],
                "render_version": valid["render_version"],
                "note_revision": valid["note_revision"],
                "note_hash": valid["note_hash"],
                "accepted_provenance": provenance,
            },
            bound_attestation="trusted_bootstrap_v1",
            bound_accepted_at=provenance["accepted_at"],
        )
        content_state = parsed.model_dump(mode="json")
        content_state.pop("accepted_provenance")
        assert provenance["result_hash"] == studio_result_hash(content_state)
        assert valid["canonical_hash"] == notes_studio_document_object_hash(
            parsed, revision=1, deleted=False
        )
        assert studios[source_note_id]["source_diagnostic_code"] == (
            "legacy_studio_payload_invalid"
        )
        for note_id in ("legacy-note-0003", "legacy-note-0004"):
            assert studios[note_id]["source_note_id"] is not None
            assert studios[note_id]["source_diagnostic_code"] == (
                "legacy_studio_lineage_unproven"
            )
    finally:
        db.close_all_connections()
        backend.get_pool().close_all()


def test_postgres_v61_large_upgrade_resumes_after_every_durable_boundary(
    pg_database_config: DatabaseConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner = "960004"
    row_count = 257
    backend, db = _open_db(pg_database_config, owner=owner)
    injected: list[str] = []

    class InjectedFailure(RuntimeError):
        pass

    def fail_each_stage_once(stage: str) -> None:
        if stage not in injected:
            injected.append(stage)
            raise InjectedFailure(stage)

    try:
        _restore_large_postgres_v62_fixture(db, owner=owner, row_count=row_count)
        monkeypatch.setattr(
            db,
            "_notes_moodboard_studio_v61_postgres_checkpoint",
            fail_each_stage_once,
        )

        for _attempt in range(100):
            try:
                with db.transaction() as conn:
                    db._migrate_from_v62_to_v63_postgres(conn)
            except InjectedFailure:
                pass
            with backend.transaction() as conn:
                version = db._get_schema_version_postgres(conn)
                progress_exists = backend.table_exists(
                    "chacha_schema_migration_progress", connection=conn
                )
                progress = (
                    backend.execute(
                        "SELECT phase,keyset_cursor,copied_count,"
                        "aggregate_fingerprint,status "
                        "FROM chacha_schema_migration_progress "
                        "WHERE migration_id=? ORDER BY phase",
                        (CharactersRAGDB._NOTES_MOODBOARD_STUDIO_V61_MIGRATION_ID,),
                        connection=conn,
                    ).rows
                    if progress_exists
                    else []
                )
            serialized = json.dumps(progress, default=str)
            assert "Legacy board" not in serialized
            assert "Legacy body" not in serialized
            for row in progress:
                assert re.fullmatch(
                    r"sha256:[0-9a-f]{64}", row["aggregate_fingerprint"]
                )
                if row["keyset_cursor"] is not None:
                    cursor = json.loads(row["keyset_cursor"])
                    assert len(cursor) == (
                        4 if row["phase"].endswith("moodboard_notes") else 2
                    )
            if version == 63:
                break
            assert version == 62
            assert not any(
                row["phase"] == "migration" and row["status"] == "complete"
                for row in progress
            )
        else:
            pytest.fail("PostgreSQL v62-to-v63 migration did not converge after fault injection")

        with db.transaction() as conn:
            db._verify_notes_moodboard_studio_schema_postgres(conn)
            final_progress = {
                row["phase"]: row
                for row in conn.execute(
                    "SELECT phase,copied_count,aggregate_fingerprint,status "
                    "FROM chacha_schema_migration_progress WHERE migration_id=?",
                    (CharactersRAGDB._NOTES_MOODBOARD_STUDIO_V61_MIGRATION_ID,),
                ).fetchall()
            }
        for phase in ("moodboards", "moodboard_notes", "note_studio_documents"):
            verified = final_progress[f"aggregate_verification:{phase}"]
            copied = final_progress[phase]
            predicted = final_progress[f"source_prediction:{phase}"]
            assert (
                predicted["copied_count"]
                == copied["copied_count"]
                == verified["copied_count"]
                == row_count
            )
            assert (
                predicted["aggregate_fingerprint"]
                == copied["aggregate_fingerprint"]
                == verified["aggregate_fingerprint"]
            )
            assert predicted["status"] == copied["status"] == verified["status"] == "complete"
            assert len([stage for stage in injected if stage.startswith(f"copy:{phase}:")]) == 3
            assert len(
                [
                    stage
                    for stage in injected
                    if stage.startswith(f"aggregate_verification:{phase}:")
                ]
            ) == 3
            assert len(
                [
                    stage
                    for stage in injected
                    if stage.startswith(f"source_prediction:{phase}:")
                ]
            ) == 3
        assert len(
            [stage for stage in injected if stage.startswith("identity:moodboards:")]
        ) == 3
        assert {"schema", "constraint", "rls", "aggregate_verification", "version"}.issubset(
            injected
        )
        assert {f"index:{index}" for index in range(len(V61_INDEXES))}.issubset(
            injected
        )
        assert len(
            [stage for stage in injected if stage.startswith("constraint_validation:")]
        ) >= 20
    finally:
        db.close_all_connections()
        backend.get_pool().close_all()


def test_postgres_v61_aggregate_rejects_valid_looking_target_tampering(
    pg_database_config: DatabaseConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner = "960008"
    backend, db = _open_db(pg_database_config, owner=owner)

    class CopyFinished(RuntimeError):
        pass

    def stop_after_copy(stage: str) -> None:
        if stage == "copy:note_studio_documents:1":
            raise CopyFinished

    try:
        _restore_large_postgres_v62_fixture(db, owner=owner, row_count=1)
        monkeypatch.setattr(
            db,
            "_notes_moodboard_studio_v61_postgres_checkpoint",
            stop_after_copy,
        )
        with pytest.raises(CopyFinished):
            with db.transaction() as conn:
                db._migrate_from_v62_to_v63_postgres(conn)

        with backend.transaction() as conn:
            backend.execute(
                "UPDATE note_studio_documents SET canonical_hash=?",
                ("sha256:" + "f" * 64,),
                connection=conn,
            )
        monkeypatch.setattr(
            db,
            "_notes_moodboard_studio_v61_postgres_checkpoint",
            lambda _stage: None,
        )
        with pytest.raises(SchemaError, match="aggregate verification failed"):
            with db.transaction() as conn:
                db._migrate_from_v62_to_v63_postgres(conn)

        with backend.transaction() as conn:
            assert db._get_schema_version_postgres(conn) == 62
            migration = backend.execute(
                "SELECT status FROM chacha_schema_migration_progress "
                "WHERE migration_id=? AND phase='migration'",
                (CharactersRAGDB._NOTES_MOODBOARD_STUDIO_V61_MIGRATION_ID,),
                connection=conn,
            ).rows
        assert migration == []
    finally:
        db.close_all_connections()
        backend.get_pool().close_all()


def test_postgres_v61_source_prediction_rejects_consistently_wrong_conversion(
    pg_database_config: DatabaseConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner = "960014"
    backend, db = _open_db(pg_database_config, owner=owner)

    class PredictionFinished(RuntimeError):
        pass

    def stop_after_prediction(stage: str) -> None:
        if stage == "source_prediction:note_studio_documents:1":
            raise PredictionFinished

    try:
        _restore_large_postgres_v62_fixture(db, owner=owner, row_count=1)
        monkeypatch.setattr(
            db,
            "_notes_moodboard_studio_v61_postgres_checkpoint",
            stop_after_prediction,
        )
        with pytest.raises(PredictionFinished):
            with db.transaction() as conn:
                db._migrate_from_v62_to_v63_postgres(conn)

        original_expected_row = db._postgres_v61_expected_row

        def consistently_wrong_expected_row(
            conn: Any, *, phase: str, row: dict[str, Any]
        ) -> dict[str, Any]:
            expected = original_expected_row(conn, phase=phase, row=row)
            if phase == "moodboards":
                expected["canonical_hash"] = "sha256:" + "e" * 64
            return expected

        monkeypatch.setattr(
            db, "_postgres_v61_expected_row", consistently_wrong_expected_row
        )
        monkeypatch.setattr(
            db,
            "_notes_moodboard_studio_v61_postgres_checkpoint",
            lambda _stage: None,
        )
        with pytest.raises(SchemaError, match="aggregate verification failed"):
            with db.transaction() as conn:
                db._migrate_from_v62_to_v63_postgres(conn)
        with backend.transaction() as conn:
            assert db._get_schema_version_postgres(conn) == 62
            predictions = backend.execute(
                "SELECT phase,status FROM chacha_schema_migration_progress "
                "WHERE migration_id=? AND phase LIKE 'source_prediction:%%' "
                "ORDER BY phase",
                (CharactersRAGDB._NOTES_MOODBOARD_STUDIO_V61_MIGRATION_ID,),
                connection=conn,
            ).rows
        assert len(predictions) == 3
        assert all(row["status"] == "complete" for row in predictions)
    finally:
        db.close_all_connections()
        backend.get_pool().close_all()


def test_postgres_v61_nonowner_rls_relationships_same_id_and_keyset_plans(
    pg_database_config: DatabaseConfig,
) -> None:
    owner_a = "960005"
    owner_b = "960006"
    dataset = "local-unbound"
    note_a = str(uuid4())
    note_b = str(uuid4())
    shared_sync_id = str(uuid4())
    backend_a, db_a = _open_db(pg_database_config, owner=owner_a)
    backend_b, db_b = _open_db(pg_database_config, owner=owner_b)
    ident = backend_a.escape_identifier  # type: ignore[attr-defined]
    role_name = f"moodboard_studio_rls_{uuid4().hex[:8]}"
    role_created = False
    try:
        db_a.add_note("Owner A", "Body A", note_id=note_a)
        db_b.add_note("Owner B", "Body B", note_id=note_b)
        board_a = db_a.add_moodboard("Board A")
        board_b = db_b.add_moodboard("Board B")
        assert board_a is not None and board_b is not None
        assert db_a.link_note_to_moodboard(board_a, note_a)
        assert db_b.link_note_to_moodboard(board_b, note_b)
        for db, owner, board in (
            (db_a, owner_a, board_a),
            (db_b, owner_b, board_b),
        ):
            with db.transaction() as conn:
                conn.execute(
                    "SELECT set_config('app.current_dataset_id', ?, true)", (dataset,)
                )
                conn.execute(
                    "UPDATE moodboards SET sync_id=? "
                    "WHERE owner_user_id=? AND dataset_id=? AND id=?",
                    (shared_sync_id, owner, dataset, board),
                )

        with backend_a.transaction() as conn:
            backend_a.execute(
                f"CREATE ROLE {ident(role_name)} NOLOGIN NOSUPERUSER NOBYPASSRLS",
                connection=conn,
            )
            backend_a.execute(
                f"GRANT USAGE ON SCHEMA public TO {ident(role_name)}",
                connection=conn,
            )
            backend_a.execute(
                f"GRANT SELECT ON notes TO {ident(role_name)}", connection=conn
            )
            for table in V61_TABLES:
                backend_a.execute(
                    f"GRANT SELECT,INSERT,UPDATE ON {ident(table)} TO {ident(role_name)}",
                    connection=conn,
                )
            backend_a.execute(
                f"GRANT {ident(role_name)} TO CURRENT_USER", connection=conn
            )
        role_created = True

        with backend_a.transaction() as conn:
            backend_a.execute(f"SET LOCAL ROLE {ident(role_name)}", connection=conn)
            backend_a.execute(
                "SELECT set_config('app.current_user_id', ?, true)",
                (owner_a,),
                connection=conn,
            )
            backend_a.execute(
                "SELECT set_config('app.current_dataset_id', ?, true)",
                (dataset,),
                connection=conn,
            )
            principal = backend_a.execute(
                "SELECT rolsuper,rolbypassrls FROM pg_roles WHERE rolname=current_user",
                connection=conn,
            ).rows[0]
            assert principal == {"rolsuper": False, "rolbypassrls": False}
            assert backend_a.execute(
                "SELECT owner_user_id,sync_id FROM moodboards WHERE sync_id=?",
                (shared_sync_id,),
                connection=conn,
            ).rows == [{"owner_user_id": owner_a, "sync_id": shared_sync_id}]
            assert backend_a.execute(
                "SELECT owner_user_id,note_id FROM moodboard_notes",
                connection=conn,
            ).rows == [{"owner_user_id": owner_a, "note_id": note_a}]
            hidden = backend_a.execute(
                "UPDATE moodboards SET name='hidden overwrite' "
                "WHERE owner_user_id=? AND dataset_id=? AND id=?",
                (owner_b, dataset, board_b),
                connection=conn,
            )
            assert hidden.rowcount == 0

        with backend_a.transaction() as conn:
            backend_a.execute(f"SET LOCAL ROLE {ident(role_name)}", connection=conn)
            backend_a.execute(
                "SELECT set_config('app.current_user_id', ?, true)",
                (owner_a,),
                connection=conn,
            )
            backend_a.execute(
                "SELECT set_config('app.current_dataset_id', 'wrong-dataset', true)",
                connection=conn,
            )
            assert backend_a.execute(
                "SELECT id FROM moodboards", connection=conn
            ).rows == []

        with pytest.raises(DatabaseError):
            with backend_a.transaction() as conn:
                backend_a.execute(f"SET LOCAL ROLE {ident(role_name)}", connection=conn)
                backend_a.execute(
                    "SELECT set_config('app.current_user_id', ?, true)",
                    (owner_a,),
                    connection=conn,
                )
                backend_a.execute(
                    "SELECT set_config('app.current_dataset_id', 'wrong-dataset', true)",
                    connection=conn,
                )
                backend_a.execute(
                    "INSERT INTO moodboards(id,name,client_id,owner_user_id,dataset_id,"
                    "sync_id,canvas_json,canonical_revision,canonical_hash) "
                    "VALUES (999999,'wrong dataset',?,?,?,?,?,1,?)",
                    (
                        owner_a,
                        owner_a,
                        dataset,
                        str(uuid4()),
                        "{}",
                        "sha256:" + "9" * 64,
                    ),
                    connection=conn,
                )

        with pytest.raises(DatabaseError):
            with backend_a.transaction() as conn:
                backend_a.execute(f"SET LOCAL ROLE {ident(role_name)}", connection=conn)
                backend_a.execute(
                    "SELECT set_config('app.current_user_id', ?, true)",
                    (owner_a,),
                    connection=conn,
                )
                backend_a.execute(
                    "SELECT set_config('app.current_dataset_id', ?, true)",
                    (dataset,),
                    connection=conn,
                )
                backend_a.execute(
                    "INSERT INTO moodboard_notes("
                    "moodboard_id,note_id,owner_user_id,dataset_id,placement_id,"
                    "x,y,width,height,order_index,display_json,last_modified,deleted,"
                    "version,canonical_revision,canonical_hash) "
                    "VALUES (?,?,?,?,?,0,0,320,220,1,'{}',CURRENT_TIMESTAMP,FALSE,1,1,?)",
                    (
                        board_a,
                        note_b,
                        owner_a,
                        dataset,
                        "notes.moodboard_note:sha256:" + "a" * 64,
                        "sha256:" + "b" * 64,
                    ),
                    connection=conn,
                )

        with pytest.raises(DatabaseError):
            with backend_a.transaction() as conn:
                backend_a.execute(f"SET LOCAL ROLE {ident(role_name)}", connection=conn)
                backend_a.execute(
                    "SELECT set_config('app.current_user_id', ?, true)",
                    (owner_a,),
                    connection=conn,
                )
                backend_a.execute(
                    "SELECT set_config('app.current_dataset_id', ?, true)",
                    (dataset,),
                    connection=conn,
                )
                backend_a.execute(
                    "INSERT INTO note_studio_documents("
                    "note_id,payload_json,template_type,handwriting_mode,source_note_id,"
                    "render_version,owner_user_id,dataset_id,note_revision,note_hash,"
                    "accepted_provenance_json,deleted,version,canonical_revision,canonical_hash) "
                    "VALUES (?,'{}','lined','off',?,1,?,?,1,?,'{}',FALSE,1,1,?)",
                    (
                        note_a,
                        note_b,
                        owner_a,
                        dataset,
                        "sha256:" + "c" * 64,
                        "sha256:" + "d" * 64,
                    ),
                    connection=conn,
                )

        with db_a.transaction() as conn:
            conn.execute("SELECT set_config('app.current_dataset_id', ?, true)", (dataset,))
            conn.execute("SET LOCAL enable_seqscan=off")
            plans = []
            for query, params in (
                (
                    "EXPLAIN SELECT id FROM moodboards WHERE owner_user_id=? "
                    "AND dataset_id=? AND deleted=FALSE "
                    "ORDER BY last_modified,id LIMIT 50",
                    (owner_a, dataset),
                ),
                (
                    "EXPLAIN SELECT placement_id FROM moodboard_notes "
                    "WHERE owner_user_id=? AND dataset_id=? AND moodboard_id=? "
                    "AND deleted=FALSE ORDER BY order_index,placement_id LIMIT 50",
                    (owner_a, dataset, board_a),
                ),
                (
                    "EXPLAIN SELECT note_id FROM note_studio_documents "
                    "WHERE owner_user_id=? AND dataset_id=? AND deleted=FALSE "
                    "ORDER BY last_modified,note_id LIMIT 50",
                    (owner_a, dataset),
                ),
            ):
                plans.append(
                    " ".join(
                        str(next(iter(dict(row).values())))
                        for row in conn.execute(query, params).fetchall()
                    )
                )
        assert "idx_moodboards_scope_page" in plans[0]
        assert "idx_moodboard_notes_scope_board_page" in plans[1]
        assert "idx_note_studio_documents_scope_page" in plans[2]
    finally:
        if role_created:
            with backend_a.transaction() as conn:
                backend_a.execute(
                    f"REVOKE {ident(role_name)} FROM CURRENT_USER", connection=conn
                )
                backend_a.execute(
                    f"DROP OWNED BY {ident(role_name)}", connection=conn
                )
                backend_a.execute(f"DROP ROLE {ident(role_name)}", connection=conn)
        db_a.close_all_connections()
        db_b.close_all_connections()
        backend_a.get_pool().close_all()
        backend_b.get_pool().close_all()
