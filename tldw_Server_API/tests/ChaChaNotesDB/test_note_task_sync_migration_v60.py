"""SQLite schema-v60 contracts for owner/dataset-scoped Notes tasks."""

from __future__ import annotations

import inspect
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path

import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDB,
    CharactersRAGDBError,
    SchemaError,
)

pytestmark = pytest.mark.unit

OWNER = "task-v60-owner"
LOCAL_UNBOUND = "local-unbound"
NOTE_ID = "11111111-1111-4111-8111-111111111111"
TASK_ID = "22222222-2222-4222-8222-222222222222"
EVENT_ID = "33333333-3333-4333-8333-333333333333"


def test_postgres_initializer_preserves_v60_and_v61_steps_before_current_v62() -> None:
    assert CharactersRAGDB._POSTGRES_SCHEMA_VERSION == 62
    source = inspect.getsource(CharactersRAGDB._initialize_schema_postgres)
    assert "target_version = self._POSTGRES_SCHEMA_VERSION" in source
    assert "_migrate_from_v59_to_v60_postgres" in source
    assert "_migrate_from_v60_to_v61_postgres" in source
    assert "_migrate_from_v61_to_v62_postgres" in source
    assert "_verify_note_task_schema_postgres" in source


def test_postgres_v60_ddl_is_fixed_and_scopes_graph_plus_authority() -> None:
    statements = CharactersRAGDB._note_task_v60_postgres_ddl()
    sql = " ".join("\n".join(statements).split()).lower()

    assert len(statements) == 8
    assert "create unique index uq_notes_owner_id on notes(client_id,id)" in sql
    for table in CharactersRAGDB._NOTE_TASK_V60_TABLES:
        assert f"create table {table}_v60" in sql
        assert "owner_user_id text not null" in sql.split(
            f"create table {table}_v60", 1
        )[1].split("create table", 1)[0]
        assert "dataset_id text not null" in sql.split(
            f"create table {table}_v60", 1
        )[1].split("create table", 1)[0]
    assert "create table note_task_scope_authority_v60" in sql
    assert "note_task_scope_authority_pkey primary key(owner_user_id)" in sql
    assert "dataset_id<>'local-unbound'" in sql
    assert "primary key(owner_user_id,dataset_id,id)" in sql
    assert "references note_tasks_v60(owner_user_id,dataset_id,id)" in sql
    assert "references task_events_v60(owner_user_id,dataset_id,id)" in sql
    assert "on update cascade" in sql
    assert "boolean not null default false" in sql
    assert "bigint not null default 1" in sql
    assert "timestamptz not null" in sql

    indexes = CharactersRAGDB._note_task_v60_postgres_indexes()
    assert (
        "CREATE INDEX idx_task_events_scope_task_created ON "
        "task_events(owner_user_id,dataset_id,task_id,created_at,id)"
    ) in indexes
    assert (
        "CREATE INDEX idx_task_events_scope_note_created ON "
        "task_events(owner_user_id,dataset_id,note_id,created_at,id)"
    ) in indexes
    assert (
        "CREATE INDEX idx_task_events_scope_cursor ON "
        "task_events(owner_user_id,dataset_id,sync_server_cursor,id)"
    ) in indexes


def test_postgres_v60_migration_uses_fixed_lock_and_version_last_order() -> None:
    source = inspect.getsource(CharactersRAGDB._migrate_from_v59_to_v60_postgres)

    assert "LOCK TABLE notes, note_tasks, task_note_projections, task_events" in source
    assert "task_event_read_state, note_task_reconciliation_state IN ACCESS EXCLUSIVE MODE" in source
    assert source.index("LOCK TABLE") < source.index("_validate_note_task_source_postgres")
    assert source.index("_validate_note_task_source_postgres") < source.index(
        "_create_note_task_schema_v60_postgres"
    )
    verify_pos = source.index("_verify_note_task_schema_postgres")
    version_pos = source.index("UPDATE db_schema_version SET version=60")
    assert verify_pos < version_pos
    assert source.index("_validate_note_task_source_postgres") < source.index(
        "_rename_note_task_v59_postgres_constraints"
    ) < source.index("_create_note_task_schema_v60_postgres")
    assert "WHERE schema_name=%s AND version=59" in source


def test_postgres_v60_frees_only_fixed_v59_constraint_collisions() -> None:
    assert CharactersRAGDB._note_task_v59_postgres_constraint_collisions() == (
        ("note_tasks", "note_tasks_pkey"),
        ("note_tasks", "note_tasks_projection_status_check"),
        ("note_tasks", "note_tasks_status_check"),
        ("task_note_projections", "task_note_projections_pkey"),
        ("task_note_projections", "task_note_projections_projection_status_check"),
        ("task_events", "task_events_pkey"),
        ("task_event_read_state", "task_event_read_state_pkey"),
        ("note_task_reconciliation_state", "note_task_reconciliation_state_pkey"),
    )
    source = inspect.getsource(CharactersRAGDB._rename_note_task_v59_postgres_constraints)
    assert "ALTER TABLE {table_name} RENAME CONSTRAINT {constraint_name}" in source
    assert "TO {constraint_name}_v59" in source


def test_postgres_v60_uses_effective_schema_owner_and_exact_table_owner() -> None:
    for method in (
        CharactersRAGDB._migrate_from_v59_to_v60_postgres,
        CharactersRAGDB._verify_note_task_schema_postgres,
    ):
        source = inspect.getsource(method).replace(" ", "")
        assert (
            "pg_has_role(current_user,namespace_row.nspowner,'USAGE')"
            in source
        )
        assert "table_row.relowner=current_user::regroleASis_table_owner" in source


def _canonical_postgres_v59_source_authority() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    relation_rows = [
        {
            "table_name": table,
            "rls_enabled": table == "notes",
            "rls_forced": table == "notes",
            "is_table_owner": True,
            "is_schema_owner": True,
        }
        for table in ("notes", *CharactersRAGDB._NOTE_TASK_V60_TABLES[:-1])
    ]
    policy_rows = [
        {
            "table_name": "notes",
            "policy_name": "notes_tenant_isolation",
            "permissive": "PERMISSIVE",
            "roles": "{public}",
            "command": "ALL",
            "using_expression": "(client_id = current_setting('app.current_user_id'::text, true))",
            "check_expression": "(client_id = current_setting('app.current_user_id'::text, true))",
        }
    ]
    return relation_rows, policy_rows


def test_postgres_v59_source_authority_accepts_only_reviewed_catalog() -> None:
    relation_rows, policy_rows = _canonical_postgres_v59_source_authority()

    assert CharactersRAGDB._verify_note_task_v59_authority_catalog(
        relation_rows, policy_rows
    ) == ("notes",)

    source = inspect.getsource(CharactersRAGDB._migrate_from_v59_to_v60_postgres)
    authority_pos = source.index("_verify_note_task_v59_authority_catalog")
    assert authority_pos < source.index("collision_names")
    assert source.index("for table in forced_relations") < source.index(
        "_validate_note_task_source_postgres"
    )


@pytest.mark.parametrize(
    "drift",
    (
        "missing_relation",
        "extra_relation",
        "table_owner",
        "schema_owner",
        "notes_rls_disabled",
        "notes_unforced",
        "task_rls_enabled",
        "task_forced",
        "missing_policy",
        "extra_task_policy",
        "policy_name",
        "policy_permissiveness",
        "policy_roles",
        "policy_command",
        "policy_using",
        "policy_check",
    ),
)
def test_postgres_v59_source_authority_drift_matrix_fails_closed(drift: str) -> None:
    relation_rows, policy_rows = _canonical_postgres_v59_source_authority()
    relation_rows = deepcopy(relation_rows)
    policy_rows = deepcopy(policy_rows)
    relations = {str(row["table_name"]): row for row in relation_rows}

    if drift == "missing_relation":
        relation_rows.pop()
    elif drift == "extra_relation":
        relation_rows.append({**relation_rows[-1], "table_name": "unexpected_tasks"})
    elif drift == "table_owner":
        relations["note_tasks"]["is_table_owner"] = False
    elif drift == "schema_owner":
        relations["notes"]["is_schema_owner"] = False
    elif drift == "notes_rls_disabled":
        relations["notes"]["rls_enabled"] = False
    elif drift == "notes_unforced":
        relations["notes"]["rls_forced"] = False
    elif drift == "task_rls_enabled":
        relations["note_tasks"]["rls_enabled"] = True
    elif drift == "task_forced":
        relations["note_tasks"]["rls_forced"] = True
    elif drift == "missing_policy":
        policy_rows.clear()
    elif drift == "extra_task_policy":
        policy_rows.append({**policy_rows[0], "table_name": "note_tasks"})
    elif drift == "policy_name":
        policy_rows[0]["policy_name"] = "notes_open"
    elif drift == "policy_permissiveness":
        policy_rows[0]["permissive"] = "RESTRICTIVE"
    elif drift == "policy_roles":
        policy_rows[0]["roles"] = "{postgres}"
    elif drift == "policy_command":
        policy_rows[0]["command"] = "SELECT"
    elif drift == "policy_using":
        policy_rows[0]["using_expression"] = "true"
    elif drift == "policy_check":
        policy_rows[0]["check_expression"] = "true"

    with pytest.raises(SchemaError, match="v59 PostgreSQL source authority"):
        CharactersRAGDB._verify_note_task_v59_authority_catalog(relation_rows, policy_rows)


def test_postgres_v60_verifier_enumerates_full_catalog_and_pg18_not_null_rows() -> None:
    source = inspect.getsource(CharactersRAGDB._verify_note_task_schema_postgres)

    for catalog in ("pg_class", "pg_attribute", "pg_constraint", "pg_index", "pg_policies"):
        assert catalog in source
    for exact_index_field in (
        "indnatts", "indnkeyatts", "indclass", "indcollation", "indoption", "attnum",
    ):
        assert exact_index_field in source
    for exact_index_catalog in ("pg_am", "pg_opclass", "pg_collation", "pg_get_indexdef"):
        assert exact_index_catalog in source
    assert "constraint_row.contype <> 'n'" in source
    assert "constraint_row.convalidated" in source
    assert "referenced_namespace.nspname=current_schema()" in source.replace(" ", "")
    assert "index_row.indisvalid" in source
    assert "index_row.indisready" in source
    assert "relrowsecurity" in source
    assert "relforcerowsecurity" in source
    assert "permissive" in source
    assert "with_check" in source


def test_postgres_v60_catalog_comparisons_fail_closed_on_security_drift() -> None:
    source = inspect.getsource(CharactersRAGDB._verify_note_task_schema_postgres)

    assert "set(constraints) != expected_constraint_names" in source
    assert "not bool(row.get(\"constraint_validated\"))" in source
    assert "_catalog_names(row.get(\"local_columns\")) != local_columns" in source
    assert "_catalog_names(row.get(\"referenced_columns\")) != referenced_columns" in source
    assert "str(row.get(\"delete_action\")) != delete_action" in source
    assert "str(row.get(\"update_action\")) != update_action" in source
    assert "set(indexes) != set(expected_indexes)" in source
    assert "row.get(\"predicate\") is not None" in source
    assert "len(policy_rows) != len(expected_policies)" in source
    assert "normalized_using != normalized_expected" in source
    assert "normalized_check != normalized_expected" in source

    normalize = CharactersRAGDB._normalize_postgres_catalog_expression
    canonical = "owner_user_id=current_setting('app.current_user_id',true)"
    assert normalize(canonical) != normalize(f"({canonical}) OR TRUE")


def _prepare_v59_database(db_path: Path) -> None:
    """Create the preceding real schema and one complete legacy task graph."""

    original_version = CharactersRAGDB._CURRENT_SCHEMA_VERSION
    CharactersRAGDB._CURRENT_SCHEMA_VERSION = 59
    try:
        db = CharactersRAGDB(str(db_path), client_id=OWNER)
        note_id = db.add_note("Task parent", "- [ ] Review source\n", note_id=NOTE_ID)
        assert note_id == NOTE_ID  # nosec B101
        now = "2026-08-13T00:00:00+00:00"
        with db.transaction() as conn:
            conn.execute(
                """
                INSERT INTO note_tasks(
                    id, note_id, text, status, metadata_json, projection_status, deleted,
                    created_at, updated_at, completed_at, client_id, version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    TASK_ID,
                    NOTE_ID,
                    "Review source",
                    "open",
                    '{"due_date":"2026-08-15","estimate":"30m","priority":"high"}',
                    "live",
                    0,
                    now,
                    now,
                    None,
                    OWNER,
                    1,
                ),
            )
            conn.execute(
                """
                INSERT INTO task_note_projections(
                    task_id, note_id, note_version, line_number, start_offset, end_offset,
                    normalized_text_hash, occurrence_index, block_fingerprint, raw_line,
                    has_child_content, projection_status, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    TASK_ID,
                    NOTE_ID,
                    1,
                    1,
                    0,
                    19,
                    "sha256:" + "a" * 64,
                    0,
                    "sha256:" + "b" * 64,
                    "- [ ] Review source",
                    0,
                    "live",
                    now,
                ),
            )
            conn.execute(
                """
                INSERT INTO task_events(
                    id, task_id, note_id, event_type, actor_type, actor_id, tool_name,
                    policy_mode, approval_id, old_value_json, new_value_json, created_at,
                    client_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    EVENT_ID,
                    TASK_ID,
                    NOTE_ID,
                    "created",
                    "user",
                    OWNER,
                    None,
                    None,
                    None,
                    None,
                    '{"status":"open","text":"Review source"}',
                    now,
                    OWNER,
                ),
            )
            conn.execute(
                "INSERT INTO task_event_read_state(event_id, user_id, read_at) VALUES (?, ?, ?)",
                (EVENT_ID, OWNER, now),
            )
            conn.execute(
                """
                INSERT INTO note_task_reconciliation_state(
                    note_id, note_version, status, reconciled_at, item_count, warning_count,
                    cursor
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (NOTE_ID, 1, "clean", now, 1, 0, None),
            )
        db.close_all_connections()
    finally:
        CharactersRAGDB._CURRENT_SCHEMA_VERSION = original_version


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})")}  # nosec B608


def _rewrite_sqlite_catalog(db_path: Path, *, object_name: str, old: str, new: str) -> None:
    with sqlite3.connect(db_path) as conn:
        current = conn.execute(
            "SELECT sql FROM sqlite_master WHERE name=?", (object_name,)
        ).fetchone()[0]
        assert old in current  # nosec B101 - proves the adversarial mutation was applied.
        conn.execute("PRAGMA writable_schema=ON")
        conn.execute(
            "UPDATE sqlite_master SET sql=? WHERE name=?",
            (current.replace(old, new, 1), object_name),
        )
        schema_version = int(conn.execute("PRAGMA schema_version").fetchone()[0])
        conn.execute(f"PRAGMA schema_version={schema_version + 1}")  # nosec B608
        conn.execute("PRAGMA writable_schema=OFF")


@pytest.mark.parametrize(
    "drift",
    ("weakened_constraint", "extra_index", "extra_trigger", "extra_view", "extra_table"),
)
def test_sqlite_v60_upgrade_rejects_exact_v59_source_catalog_drift_before_ddl(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    drift: str,
) -> None:
    db_path = tmp_path / f"v59-source-{drift}.sqlite"
    _prepare_v59_database(db_path)
    drift_object = "note_tasks"
    if drift == "weakened_constraint":
        _rewrite_sqlite_catalog(
            db_path,
            object_name=drift_object,
            old="CHECK(status IN ('open','done'))",
            new="CHECK(status IN ('open','done','paused'))",
        )
    else:
        statements = {
            "extra_index": (
                "CREATE INDEX unexpected_task_index ON note_tasks(text)",
                "unexpected_task_index",
            ),
            "extra_trigger": (
                "CREATE TRIGGER unexpected_task_trigger AFTER UPDATE ON note_tasks "
                "BEGIN SELECT NEW.id; END",
                "unexpected_task_trigger",
            ),
            "extra_view": (
                "CREATE VIEW unexpected_task_view AS SELECT id FROM note_tasks",
                "unexpected_task_view",
            ),
            "extra_table": (
                "CREATE TABLE unexpected_task_table("
                "id TEXT PRIMARY KEY, task_id TEXT REFERENCES note_tasks(id))",
                "unexpected_task_table",
            ),
        }
        statement, drift_object = statements[drift]
        with sqlite3.connect(db_path) as conn:
            conn.execute(statement)

    checkpoints: list[str] = []
    monkeypatch.setattr(
        CharactersRAGDB,
        "_note_task_v60_migration_checkpoint",
        lambda _db, stage: checkpoints.append(stage),
    )
    with pytest.raises(CharactersRAGDBError, match="v59 SQLite source catalog drifted"):
        CharactersRAGDB(str(db_path), client_id=OWNER)

    assert checkpoints == []  # nosec B101 - validation must precede v60 DDL.
    with sqlite3.connect(db_path) as conn:
        assert conn.execute(  # nosec B101
            "SELECT version FROM db_schema_version WHERE schema_name=?",
            (CharactersRAGDB._SCHEMA_NAME,),
        ).fetchone()[0] == 59
        assert conn.execute("SELECT id FROM note_tasks").fetchone()[0] == TASK_ID  # nosec B101
        assert conn.execute("SELECT id FROM task_events").fetchone()[0] == EVENT_ID  # nosec B101
        catalog_sql = conn.execute(
            "SELECT sql FROM sqlite_master WHERE name=?", (drift_object,)
        ).fetchone()[0]
        assert catalog_sql is not None  # nosec B101
        if drift == "weakened_constraint":
            assert "'paused'" in catalog_sql  # nosec B101


def test_sqlite_v60_upgrade_preserves_graph_under_local_unbound_scope(tmp_path: Path) -> None:
    db_path = tmp_path / "task-v59.sqlite"
    _prepare_v59_database(db_path)

    upgraded = CharactersRAGDB(str(db_path), client_id=OWNER)
    try:
        with upgraded.transaction() as conn:
            assert upgraded._get_db_version(conn) == CharactersRAGDB._CURRENT_SCHEMA_VERSION  # nosec B101
            assert conn.execute("PRAGMA foreign_key_check").fetchall() == []  # nosec B101
            counts = {
                table: int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])  # nosec B608
                for table in (
                    "note_tasks",
                    "task_note_projections",
                    "task_events",
                    "task_event_read_state",
                    "note_task_reconciliation_state",
                    "task_projection_drifts",
                    "note_task_scope_authority",
                )
            }
            assert counts == {  # nosec B101
                "note_tasks": 1,
                "task_note_projections": 1,
                "task_events": 1,
                "task_event_read_state": 1,
                "note_task_reconciliation_state": 1,
                "task_projection_drifts": 0,
                "note_task_scope_authority": 0,
            }
        task = upgraded.get_task(
            owner_user_id=OWNER,
            dataset_id=LOCAL_UNBOUND,
            task_id=TASK_ID,
            include_deleted=True,
        )
        assert task is not None  # nosec B101
        assert task["id"] == TASK_ID  # nosec B101
        assert task["canonical_revision"] == task["version"] == 1  # nosec B101
        assert task["canonical_hash"].startswith("sha256:")  # nosec B101
    finally:
        upgraded.close_all_connections()


def test_sqlite_v60_upgrade_derives_missing_event_note_from_task(tmp_path: Path) -> None:
    db_path = tmp_path / "task-v59-null-event-note.sqlite"
    _prepare_v59_database(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute("UPDATE task_events SET note_id=NULL WHERE id=?", (EVENT_ID,))

    upgraded = CharactersRAGDB(str(db_path), client_id=OWNER)
    try:
        with upgraded.transaction() as conn:
            row = conn.execute("SELECT note_id FROM task_events WHERE id=?", (EVENT_ID,)).fetchone()
            assert row[0] == NOTE_ID  # nosec B101
            note_column = next(
                column for column in conn.execute("PRAGMA table_xinfo(task_events)")
                if column[1] == "note_id"
            )
            assert note_column[3] == 1  # nosec B101 - NOT NULL is catalog authority.
    finally:
        upgraded.close_all_connections()


def test_sqlite_v60_upgrade_rejects_mismatched_event_task_and_note(tmp_path: Path) -> None:
    db_path = tmp_path / "task-v59-mismatched-event-note.sqlite"
    _prepare_v59_database(db_path)
    other_note_id = "44444444-4444-4444-8444-444444444444"
    original_version = CharactersRAGDB._CURRENT_SCHEMA_VERSION
    CharactersRAGDB._CURRENT_SCHEMA_VERSION = 59
    try:
        legacy = CharactersRAGDB(str(db_path), client_id=OWNER)
        assert legacy.add_note("Other", "Body", note_id=other_note_id) == other_note_id  # nosec B101
        with legacy.transaction() as conn:
            conn.execute("UPDATE task_events SET note_id=? WHERE id=?", (other_note_id, EVENT_ID))
        legacy.close_all_connections()
    finally:
        CharactersRAGDB._CURRENT_SCHEMA_VERSION = original_version

    with pytest.raises(CharactersRAGDBError, match="mismatched event parents"):
        CharactersRAGDB(str(db_path), client_id=OWNER)
    with sqlite3.connect(db_path) as conn:
        assert conn.execute(  # nosec B101
            "SELECT version FROM db_schema_version WHERE schema_name=?",
            (CharactersRAGDB._SCHEMA_NAME,),
        ).fetchone()[0] == 59


def test_sqlite_v60_has_exact_scoped_task_graph_columns(tmp_path: Path) -> None:
    db = CharactersRAGDB(str(tmp_path / "fresh.sqlite"), client_id=OWNER)
    try:
        with db.transaction() as conn:
            assert {"owner_user_id", "dataset_id", "canonical_revision", "canonical_hash"} <= _table_columns(  # nosec B101
                conn, "note_tasks"
            )
            assert {
                "owner_user_id",
                "dataset_id",
                "sync_revision",
                "sync_object_hash",
                "sync_server_cursor",
                "source_device_id",
                "client_occurred_at",
                "source_kind",
                "corrects_activity_id",
                "deleted",
                "deleted_at",
                "delete_reason",
            } <= _table_columns(conn, "task_events")  # nosec B101
            for table in (
                "task_note_projections",
                "task_event_read_state",
                "note_task_reconciliation_state",
                "task_projection_drifts",
            ):
                assert {"owner_user_id", "dataset_id"} <= _table_columns(conn, table)  # nosec B101
            assert _table_columns(conn, "note_task_scope_authority") == {  # nosec B101
                "owner_user_id",
                "dataset_id",
                "task_graph_bound",
                "moodboard_graph_bound",
                "studio_graph_bound",
            }
            assert conn.execute(  # nosec B101
                "SELECT COUNT(*) FROM note_task_scope_authority"
            ).fetchone()[0] == 0
            foreign_keys = conn.execute("PRAGMA foreign_key_list(task_note_projections)").fetchall()
            assert any(str(row[5]).upper() == "CASCADE" for row in foreign_keys)  # nosec B101
    finally:
        db.close_all_connections()


def test_sqlite_v60_verifier_rejects_same_name_weak_index(tmp_path: Path) -> None:
    db_path = tmp_path / "weak-index.sqlite"
    db = CharactersRAGDB(str(db_path), client_id=OWNER)
    db.close_all_connections()
    with sqlite3.connect(db_path) as conn:
        conn.execute("DROP INDEX idx_task_events_scope_task_page")
        conn.execute(
            "CREATE INDEX idx_task_events_scope_task_page ON task_events(owner_user_id)"
        )

    with pytest.raises(CharactersRAGDBError, match="index catalog drifted"):
        CharactersRAGDB(str(db_path), client_id=OWNER)


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("note_id TEXT NOT NULL", "note_id BLOB NOT NULL"),
        ("CHECK(length(trim(note_id)) > 0)", "CHECK(1)"),
        ("ON UPDATE CASCADE ON DELETE RESTRICT", "ON UPDATE NO ACTION ON DELETE RESTRICT"),
    ],
    ids=("type", "check", "foreign-key-action"),
)
def test_sqlite_v60_verifier_rejects_table_catalog_drift(
    tmp_path: Path,
    old: str,
    new: str,
) -> None:
    db_path = tmp_path / f"table-drift-{old[:4]}.sqlite"
    db = CharactersRAGDB(str(db_path), client_id=OWNER)
    db.close_all_connections()
    _rewrite_sqlite_catalog(db_path, object_name="task_events", old=old, new=new)

    with pytest.raises(CharactersRAGDBError, match="table catalog drifted"):
        CharactersRAGDB(str(db_path), client_id=OWNER)


@pytest.mark.parametrize(
    "ddl",
    [
        (
            "CREATE TRIGGER unexpected_task_delete AFTER INSERT ON note_tasks "
            "BEGIN DELETE FROM note_tasks WHERE id = NEW.id; END"
        ),
        "CREATE VIEW unexpected_task_view AS SELECT id,note_id FROM note_tasks",
    ],
    ids=("trigger", "view"),
)
def test_sqlite_v60_verifier_rejects_related_trigger_or_view(
    tmp_path: Path,
    ddl: str,
) -> None:
    db_path = tmp_path / "related-catalog-drift.sqlite"
    db = CharactersRAGDB(str(db_path), client_id=OWNER)
    db.close_all_connections()
    with sqlite3.connect(db_path) as conn:
        conn.execute(ddl)

    with pytest.raises(CharactersRAGDBError, match="related catalog drifted"):
        CharactersRAGDB(str(db_path), client_id=OWNER)


@pytest.mark.parametrize("migrated", [False, True], ids=("fresh", "migrated"))
@pytest.mark.parametrize(
    ("filter_column", "filter_value", "expected_index"),
    [
        ("task_id", TASK_ID, "idx_task_events_scope_task_created"),
        ("note_id", NOTE_ID, "idx_task_events_scope_note_created"),
    ],
    ids=("task-page", "note-page"),
)
def test_sqlite_v60_activity_pages_use_scoped_created_indexes(
    tmp_path: Path,
    migrated: bool,
    filter_column: str,
    filter_value: str,
    expected_index: str,
) -> None:
    db_path = tmp_path / f"activity-plan-{migrated}-{filter_column}.sqlite"
    if migrated:
        _prepare_v59_database(db_path)
    db = CharactersRAGDB(str(db_path), client_id=OWNER)
    try:
        with db.transaction() as conn:
            plan = [
                str(row[3])
                for row in conn.execute(
                    "EXPLAIN QUERY PLAN SELECT * FROM task_events "
                    f"WHERE owner_user_id=? AND dataset_id=? AND {filter_column}=? "  # nosec B608
                    "ORDER BY created_at ASC,rowid ASC LIMIT ?",
                    (OWNER, LOCAL_UNBOUND, filter_value, 100),
                )
            ]
        assert any(expected_index in detail for detail in plan), plan  # nosec B101
        assert not any("TEMP B-TREE" in detail.upper() for detail in plan), plan  # nosec B101
    finally:
        db.close_all_connections()


@pytest.mark.parametrize("migrated", [False, True], ids=("fresh", "migrated"))
def test_sqlite_v60_dataset_cursor_page_uses_exact_index(
    tmp_path: Path,
    migrated: bool,
) -> None:
    db_path = tmp_path / f"dataset-cursor-plan-{migrated}.sqlite"
    if migrated:
        _prepare_v59_database(db_path)
    db = CharactersRAGDB(str(db_path), client_id=OWNER)
    note_id = NOTE_ID if migrated else db.add_note("Cursor parent", "Body")

    try:
        with db.transaction() as conn:
            for cursor in range(1, 65):
                target_event = db.record_task_event(
                    owner_user_id=OWNER,
                    dataset_id=LOCAL_UNBOUND,
                    note_id=note_id,
                    event_type="updated",
                    actor_type="user",
                    conn=conn,
                )
                conn.execute(
                    "UPDATE task_events SET sync_server_cursor=? "
                    "WHERE owner_user_id=? AND dataset_id=? AND id=?",
                    (cursor, OWNER, LOCAL_UNBOUND, target_event["id"]),
                )
            conn.execute(
                "INSERT INTO note_task_scope_authority(owner_user_id,dataset_id) VALUES (?,?)",
                (OWNER, "other-dataset"),
            )
            for cursor in range(1, 65):
                other_event = db.record_task_event(
                    owner_user_id=OWNER,
                    dataset_id="other-dataset",
                    note_id=note_id,
                    event_type="updated",
                    actor_type="user",
                    conn=conn,
                )
                conn.execute(
                    "UPDATE task_events SET sync_server_cursor=? "
                    "WHERE owner_user_id=? AND dataset_id=? AND id=?",
                    (cursor, OWNER, "other-dataset", other_event["id"]),
                )
            conn.execute(
                "DELETE FROM note_task_scope_authority WHERE owner_user_id=?",
                (OWNER,),
            )
            conn.execute("ANALYZE task_events")
            plan = [
                str(row[3])
                for row in conn.execute(
                    "EXPLAIN QUERY PLAN SELECT id FROM task_events "
                    "WHERE owner_user_id=? AND dataset_id=? AND sync_server_cursor>? "
                    "ORDER BY sync_server_cursor,id LIMIT ?",
                    (OWNER, LOCAL_UNBOUND, 0, 10),
                )
            ]
        assert any("idx_task_events_scope_cursor" in detail for detail in plan), plan  # nosec B101
        assert not any("TEMP B-TREE" in detail.upper() for detail in plan), plan  # nosec B101
    finally:
        db.close_all_connections()


@pytest.mark.parametrize("stage", ["create", "copy", "index", "verify"])
def test_sqlite_v60_injected_failure_rolls_back_original_graph_and_version(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
) -> None:
    db_path = tmp_path / f"rollback-{stage}.sqlite"
    _prepare_v59_database(db_path)

    def fail_at_stage(self: CharactersRAGDB, current_stage: str) -> None:
        if current_stage == stage:
            raise SchemaError(f"injected v60 {stage} failure")

    monkeypatch.setattr(
        CharactersRAGDB,
        "_note_task_v60_migration_checkpoint",
        fail_at_stage,
        raising=False,
    )
    with pytest.raises(CharactersRAGDBError, match=f"injected v60 {stage} failure"):
        CharactersRAGDB(str(db_path), client_id=OWNER)

    with sqlite3.connect(db_path) as conn:
        assert conn.execute(  # nosec B101
            "SELECT version FROM db_schema_version WHERE schema_name = ?",
            (CharactersRAGDB._SCHEMA_NAME,),
        ).fetchone()[0] == 59
        assert "owner_user_id" not in _table_columns(conn, "note_tasks")  # nosec B101
        assert conn.execute("SELECT id FROM note_tasks").fetchone()[0] == TASK_ID  # nosec B101
        assert conn.execute("SELECT id FROM task_events").fetchone()[0] == EVENT_ID  # nosec B101


def test_sqlite_v60_same_path_concurrent_openers_complete_one_migration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "concurrent-openers.sqlite"
    _prepare_v59_database(db_path)
    start = threading.Barrier(2)
    checkpoint_lock = threading.Lock()
    checkpoints: list[str] = []
    original_checkpoint = CharactersRAGDB._note_task_v60_migration_checkpoint

    def track_checkpoint(_db: CharactersRAGDB, stage: str) -> None:
        with checkpoint_lock:
            checkpoints.append(stage)
        original_checkpoint(stage)

    monkeypatch.setattr(
        CharactersRAGDB,
        "_note_task_v60_migration_checkpoint",
        track_checkpoint,
    )

    def open_same_path() -> int:
        start.wait(timeout=10)
        db = CharactersRAGDB(str(db_path), client_id=OWNER)
        try:
            with db.transaction() as conn:
                return db._get_db_version(conn)
        finally:
            db.close_all_connections()

    with ThreadPoolExecutor(max_workers=2) as executor:
        versions = list(executor.map(lambda _index: open_same_path(), range(2)))

    assert versions == [CharactersRAGDB._CURRENT_SCHEMA_VERSION] * 2  # nosec B101
    assert checkpoints == ["create", "copy", "index", "verify"]  # nosec B101
    with sqlite3.connect(db_path) as conn:
        tables = {str(row[0]) for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )}
        assert conn.execute(  # nosec B101
            "SELECT version FROM db_schema_version WHERE schema_name=?",
            (CharactersRAGDB._SCHEMA_NAME,),
        ).fetchone()[0] == CharactersRAGDB._CURRENT_SCHEMA_VERSION
        assert not any(name.endswith(("_v60", "_v61")) for name in tables)  # nosec B101
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []  # nosec B101


def test_sqlite_v60_rejects_target_table_collision_without_version_change(tmp_path: Path) -> None:
    db_path = tmp_path / "collision.sqlite"
    _prepare_v59_database(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE task_projection_drifts(id TEXT PRIMARY KEY)")

    with pytest.raises(CharactersRAGDBError, match="collision"):
        CharactersRAGDB(str(db_path), client_id=OWNER)

    with sqlite3.connect(db_path) as conn:
        assert conn.execute(  # nosec B101
            "SELECT version FROM db_schema_version WHERE schema_name = ?",
            (CharactersRAGDB._SCHEMA_NAME,),
        ).fetchone()[0] == 59
        assert "owner_user_id" not in _table_columns(conn, "note_tasks")  # nosec B101


@pytest.mark.parametrize(
    ("column", "value"),
    [("canonical_revision", 1.5), ("version", 1.5), ("deleted", 1.5)],
)
def test_sqlite_v60_rejects_noninteger_task_storage_classes(
    tmp_path: Path,
    column: str,
    value: object,
) -> None:
    db = CharactersRAGDB(str(tmp_path / f"integer-{column}.sqlite"), client_id=OWNER)
    try:
        note_id = db.add_note("Task parent", "Body", note_id=NOTE_ID)
        assert note_id == NOTE_ID  # nosec B101
        with pytest.raises(sqlite3.IntegrityError), db.transaction() as conn:
            conn.execute(
                """
                INSERT INTO note_tasks(
                    owner_user_id, dataset_id, id, note_id, text, status, metadata_json,
                    projection_status, deleted, created_at, updated_at, completed_at,
                    client_id, version, canonical_revision, canonical_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,  # nosec B608 - column is test-parametrized from a fixed set.
                (
                    OWNER,
                    LOCAL_UNBOUND,
                    TASK_ID,
                    NOTE_ID,
                    "Review source",
                    "open",
                    "{}",
                    "live",
                    value if column == "deleted" else 0,
                    "2026-08-13T00:00:00+00:00",
                    "2026-08-13T00:00:00+00:00",
                    None,
                    OWNER,
                    value if column == "version" else 1,
                    value if column == "canonical_revision" else 1,
                    "sha256:" + "a" * 64,
                ),
            )
    finally:
        db.close_all_connections()
