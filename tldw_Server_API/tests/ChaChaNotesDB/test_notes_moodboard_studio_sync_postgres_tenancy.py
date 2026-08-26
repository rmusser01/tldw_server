"""Required-live PostgreSQL v61 catalog and tenancy proofs."""

from __future__ import annotations

import inspect
import json
import re
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
    SchemaError,
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


def _open_db(config: DatabaseConfig, *, owner: str = "960001") -> tuple[Any, CharactersRAGDB]:
    backend = DatabaseBackendFactory.create_backend(config)
    return backend, CharactersRAGDB(":memory:", client_id=owner, backend=backend)


def test_postgres_v61_migration_contract_is_bounded_and_version_last() -> None:
    source = inspect.getsource(CharactersRAGDB._migrate_from_v60_to_v61_postgres)
    begin_source = inspect.getsource(
        CharactersRAGDB._begin_notes_moodboard_studio_v61_postgres_transaction
    )
    schema_source = inspect.getsource(
        CharactersRAGDB._notes_moodboard_studio_v61_postgres_schema_sql
    )

    assert CharactersRAGDB._POSTGRES_SCHEMA_VERSION == 61
    assert "lock_timeout" in begin_source
    assert "statement_timeout" in begin_source
    assert "lock=True" in begin_source
    assert "chacha_schema_migration_progress" in schema_source
    assert "_NOTES_MOODBOARD_STUDIO_V61_MIGRATION_PAGE_SIZE" in source
    assert source.index(
        "_begin_notes_moodboard_studio_v61_postgres_transaction"
    ) < source.index("LOCK TABLE")
    assert source.rindex("_set_schema_version_postgres") > source.rindex(
        "_verify_notes_moodboard_studio_schema_postgres"
    )
    assert source.index(
        'verification_phase = f"aggregate_verification:{source_phase}"'
    ) < source.index('if not begin_phase("RLS phase")')


def test_fresh_postgres_schema_is_exact_v61_with_forced_rls(
    pg_database_config: DatabaseConfig,
) -> None:
    backend, db = _open_db(pg_database_config)
    try:
        with db.transaction() as conn:
            assert db._get_schema_version_postgres(conn) == 61
            db._verify_notes_moodboard_studio_schema_postgres(conn)
            rows = conn.execute(
                "SELECT c.relname,c.relrowsecurity,c.relforcerowsecurity,"
                "c.relowner=current_user::regrole AS is_owner "
                "FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace "
                "WHERE n.nspname=current_schema() AND c.relname=ANY(?) "
                "ORDER BY c.relname",
                (list(V61_TABLES),),
            ).fetchall()
        assert [row["relname"] for row in rows] == sorted(V61_TABLES)
        assert all(row["relrowsecurity"] and row["relforcerowsecurity"] for row in rows)
        assert all(row["is_owner"] for row in rows)
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
                "c.relowner=current_user::regrole AS is_owner,"
                "pg_has_role(current_user,n.nspowner,'USAGE') AS is_schema_owner "
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
            "is_owner": True,
            "is_schema_owner": True,
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


def _restore_large_postgres_v60_fixture(
    db: CharactersRAGDB,
    *,
    owner: str,
    row_count: int,
) -> None:
    """Replace an empty fresh v61 product graph with a large exact-shape v60 fixture."""
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
        db._set_schema_version_postgres(conn, 60)

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
        _restore_large_postgres_v60_fixture(db, owner=owner, row_count=row_count)
        monkeypatch.setattr(
            db,
            "_notes_moodboard_studio_v61_postgres_checkpoint",
            fail_each_stage_once,
        )

        for _attempt in range(100):
            try:
                with db.transaction() as conn:
                    db._migrate_from_v60_to_v61_postgres(conn)
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
                    assert len(json.loads(row["keyset_cursor"])) <= 3
            if version == 61:
                break
            assert version == 60
            assert not any(
                row["phase"] == "migration" and row["status"] == "complete"
                for row in progress
            )
        else:
            pytest.fail("PostgreSQL v61 migration did not converge after fault injection")

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
            assert copied["copied_count"] == verified["copied_count"] == row_count
            assert copied["aggregate_fingerprint"] == verified["aggregate_fingerprint"]
            assert copied["status"] == verified["status"] == "complete"
            assert len([stage for stage in injected if stage.startswith(f"copy:{phase}:")]) == 3
            assert len(
                [
                    stage
                    for stage in injected
                    if stage.startswith(f"aggregate_verification:{phase}:")
                ]
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
        _restore_large_postgres_v60_fixture(db, owner=owner, row_count=1)
        monkeypatch.setattr(
            db,
            "_notes_moodboard_studio_v61_postgres_checkpoint",
            stop_after_copy,
        )
        with pytest.raises(CopyFinished):
            with db.transaction() as conn:
                db._migrate_from_v60_to_v61_postgres(conn)

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
                db._migrate_from_v60_to_v61_postgres(conn)

        with backend.transaction() as conn:
            assert db._get_schema_version_postgres(conn) == 60
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
