"""Real registered PostgreSQL Skill migration and authority-drift contracts."""

from uuid import uuid4

import pytest

from tldw_Server_API.app.core.DB_Management.backends.base import AuthorizationDeniedError
from tldw_Server_API.app.core.DB_Management.chacha.operation_scope import chacha_operation
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB, CharactersRAGDBError, SchemaError

pytestmark = [pytest.mark.integration, pytest.mark.postgres]

# Published PG90 lazy registry, before any owner column or policy existed.
LEGACY_DDL = """CREATE TABLE IF NOT EXISTS skill_registry (
    id SERIAL PRIMARY KEY, name TEXT NOT NULL UNIQUE, description TEXT, argument_hint TEXT,
    disable_model_invocation BOOLEAN DEFAULT FALSE, user_invocable BOOLEAN DEFAULT TRUE,
    allowed_tools TEXT, model TEXT, context TEXT DEFAULT 'inline', directory_path TEXT NOT NULL,
    last_modified TIMESTAMP NOT NULL DEFAULT NOW(), file_hash TEXT, uuid TEXT UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(), deleted BOOLEAN DEFAULT FALSE, version INTEGER NOT NULL DEFAULT 1
)"""


@pytest.fixture
def registry_db(pg_restricted_backend, tmp_path):
    with chacha_operation(independent=True):
        db = CharactersRAGDB(tmp_path / "owner.db", client_id="1", backend=pg_restricted_backend)
    db._discard_legacy_checkout()
    try:
        yield db
    finally:
        db._discard_legacy_checkout()
        db.close_all_connections()


def test_registered_upgrade_preserves_unowned_rows_and_never_assigns_first_opener(pg_restricted_backend, tmp_path):
    class Legacy(CharactersRAGDB):
        _POSTGRES_SCHEMA_VERSION = 90

    handles = []
    try:
        with chacha_operation(independent=True):
            old = Legacy(tmp_path / "old.db", client_id="1", backend=pg_restricted_backend)
            handles.append(old)
            with old.transaction() as conn:
                conn.execute(LEGACY_DDL)
                for deleted, version in ((False, 7), (True, 12)):
                    conn.execute(
                        "INSERT INTO skill_registry(name,directory_path,file_hash,uuid,deleted,version) "
                        "VALUES (?,?,?,?,?,?)",
                        (
                            "legacy-" + str(version),
                            "/unreviewed/legacy/" + str(version),
                            "exact-old-hash",
                            str(uuid4()),
                            deleted,
                            version,
                        ),
                    )
                before = [dict(row) for row in conn.execute("SELECT * FROM skill_registry ORDER BY id").fetchall()]
        old._discard_legacy_checkout()
        old.close_connection()
        for owner in ("2", "3"):
            with chacha_operation(independent=True):
                db = CharactersRAGDB(tmp_path / (owner + ".db"), client_id=owner, backend=pg_restricted_backend)
                handles.append(db)
                with db.transaction() as conn:
                    assert conn.execute("SELECT * FROM skill_registry").fetchall() == []  # nosec B101 # No startup-owner adoption.
                    conn.execute("ALTER TABLE skill_registry NO FORCE ROW LEVEL SECURITY")
                    after = [dict(row) for row in conn.execute("SELECT * FROM skill_registry ORDER BY id").fetchall()]
                    conn.execute("ALTER TABLE skill_registry FORCE ROW LEVEL SECURITY")
                    assert all(row["client_id"] is None for row in after)  # nosec B101 # Legacy identity remains unknown.
                    assert [{k: v for k, v in row.items() if k != "client_id"} for row in after] == before  # nosec B101 # Every old value, UUID and revision survives.
    finally:
        for db in reversed(handles):
            db._discard_legacy_checkout()
            db.close_all_connections()


@pytest.mark.parametrize("owner", ["", "2"])
def test_missing_or_forged_owner_write_is_refused(registry_db, owner):
    db = registry_db
    with pytest.raises(AuthorizationDeniedError):
        with db.backend.transaction() as conn:
            db.backend.execute("SELECT set_config('app.current_user_id', %s, true)", (owner,), connection=conn)
            db.backend.execute(
                "INSERT INTO skill_registry(name,directory_path,uuid,client_id) VALUES (%s,%s,%s,%s)",
                ("refused", "/private/test/refused", str(uuid4()), None if not owner else "1"),
                connection=conn,
            )


DRIFT = {
    "no-force": ("ALTER TABLE skill_registry NO FORCE ROW LEVEL SECURITY",),
    "disabled-rls": ("ALTER TABLE skill_registry DISABLE ROW LEVEL SECURITY",),
    "broad-policy": ("ALTER POLICY skill_registry_tenant_isolation ON skill_registry USING (true)",),
    "owner-default": ("ALTER TABLE skill_registry ALTER COLUMN client_id DROP DEFAULT",),
    "missing-unique": ("DROP INDEX uq_skill_registry_owner_name",),
    "global-normalized-name": ("CREATE UNIQUE INDEX forbidden_skill_normalized_name ON skill_registry(lower(name))",),
    "global-name-version": ("CREATE UNIQUE INDEX forbidden_skill_name_version ON skill_registry(name,version)",),
    "global-unique": ("CREATE UNIQUE INDEX forbidden_skill_global_name ON skill_registry(name)",),
    "expression-unique": (
        "DROP INDEX uq_skill_registry_owner_name",
        "CREATE UNIQUE INDEX wrong_skill_owner_name ON skill_registry(client_id,name,lower(description))",
    ),
}


@pytest.mark.parametrize("drift", DRIFT)
def test_request_and_reopened_schema_refuse_authority_drift(registry_db, tmp_path, drift):
    db = registry_db
    with chacha_operation(independent=True):
        with db.transaction() as conn:
            for statement in DRIFT[drift]:
                conn.execute(statement)
        with pytest.raises(SchemaError):
            db.get_skill_registry("never-read")
        with pytest.raises(CharactersRAGDBError) as reopened:
            CharactersRAGDB(tmp_path / "reopened.db", client_id="2", backend=db.backend)
        assert isinstance(reopened.value.__cause__, SchemaError)  # nosec B101 # Preserve the actual constructor wrapper and schema-error cause.


def test_chat_preflight_refuses_unsafe_registry_before_read(registry_db):
    db = registry_db
    with chacha_operation(independent=True):
        with db.transaction() as conn:
            conn.execute("ALTER TABLE skill_registry DISABLE ROW LEVEL SECURITY")
        with pytest.raises(SchemaError):
            db.history_skills_may_be_visible()


def test_late_registered_migration_failure_rolls_back_catalog_and_legacy_rows(pg_restricted_backend, tmp_path, monkeypatch):
    class Legacy(CharactersRAGDB):
        _POSTGRES_SCHEMA_VERSION = 90

    from tldw_Server_API.app.core.DB_Management.chacha import skill_registry_schema

    verify = skill_registry_schema.verify_postgres_skill_registry

    def reject_after_install(*args):
        verify(*args)
        raise SchemaError("Controlled late Skill verifier failure")

    with chacha_operation(independent=True):
        old = Legacy(tmp_path / "old.db", client_id="1", backend=pg_restricted_backend)
        try:
            with old.transaction() as conn:
                conn.execute(LEGACY_DDL)
                conn.execute(
                    "INSERT INTO skill_registry(name,directory_path,uuid,version) VALUES (?,?,?,?)",
                    ("legacy-rollback", "/unreviewed/legacy/rollback", str(uuid4()), 7),
                )
                before = [dict(row) for row in conn.execute("SELECT * FROM skill_registry").fetchall()]
            with monkeypatch.context() as fault:
                fault.setattr(skill_registry_schema, "verify_postgres_skill_registry", reject_after_install)
                with pytest.raises(CharactersRAGDBError) as failed:
                    CharactersRAGDB(tmp_path / "failed.db", client_id="2", backend=pg_restricted_backend)
            assert isinstance(failed.value.__cause__, SchemaError)  # nosec B101 # Actual late verifier refusal is wrapped by the constructor.
            with old.transaction() as conn:
                assert old._get_schema_version_postgres(getattr(conn, "_connection", conn)) == 90  # nosec B101 # No premature version publication.
                assert [dict(row) for row in conn.execute("SELECT * FROM skill_registry").fetchall()] == before  # nosec B101 # Full legacy rows survive rollback.
                columns = old.backend.get_table_info("skill_registry", connection=getattr(conn, "_connection", conn))
                assert "client_id" not in {row["name"] for row in columns}  # nosec B101 # Owner/default installation rolled back atomically.
                row = conn.execute(
                    "SELECT relrowsecurity,relforcerowsecurity FROM pg_class WHERE oid='skill_registry'::regclass"
                ).fetchone()
                assert not row["relrowsecurity"] and not row["relforcerowsecurity"]  # nosec B101 # Policy state rolls back with the schema.
        finally:
            old._discard_legacy_checkout()
            old.close_all_connections()
