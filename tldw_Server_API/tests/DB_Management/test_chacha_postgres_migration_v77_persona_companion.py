"""Tests for the ChaChaNotes PostgreSQL v77 companion migration."""

import re
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
from psycopg import sql as pg_sql

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.backends.base import (
    BackendType,
    DatabaseConfig,
    DatabaseError as BackendDatabaseError,
)
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory


pytestmark = pytest.mark.unit


@pytest.mark.parametrize("source_version", [72, 73, 76])
def test_postgres_companion_upgrade_finishes_fast_path_and_installs_rls(
    monkeypatch: pytest.MonkeyPatch, tmp_path, source_version: int,
) -> None:
    """Exercise dispatch with a recorded SQL boundary, including old companion73."""
    from tldw_Server_API.app.core.DB_Management.chacha import schema_bootstrap

    conn = object()
    state = {"version": source_version, "companion_sql": "", "rls_version": None}
    legacy = source_version == 73
    backend = SimpleNamespace(
        backend_type=BackendType.SQLITE,
        transaction=lambda: nullcontext(conn),
        table_exists=lambda table, **_kwargs: table == "db_schema_version" or (
            legacy and table in {"persona_buddy_preferences", "persona_visual_pack_reviews"}
        ),
        get_table_info=lambda table, **_kwargs: (
            [{"name": "companion_behavior_json"}] if table == "persona_visual_packs" else []
        ),
    )
    db = CharactersRAGDB(tmp_path / "dispatch.sqlite", "dispatch-owner")
    original_backend = db.backend
    monkeypatch.setattr(db, "_backend", backend)
    monkeypatch.setattr(schema_bootstrap, "postgres_schema_migration", lambda *_args: nullcontext(conn))
    monkeypatch.setattr(db, "_configure_notes_moodboard_studio_v61_postgres_transaction", lambda _conn: None)
    monkeypatch.setattr(db, "_postgres_schema_is_current", lambda _conn: state["version"] == 77)
    monkeypatch.setattr(db, "_get_schema_version_postgres", lambda _conn, **_kwargs: state["version"])
    monkeypatch.setattr(db, "_set_schema_version_postgres", lambda _conn, version: state.update(version=version))
    monkeypatch.setattr(db, "_repair_conversation_assistant_identity", lambda _conn: None)
    for version in range(72, 76):
        monkeypatch.setattr(
            db, f"_migrate_from_v{version}_to_v{version + 1}_postgres",
            lambda _conn, target=version + 1: state.update(version=target),
        )

    def apply_sql(sql: str, migration_conn: object, *, expected_version: int) -> None:
        assert migration_conn is conn
        assert state["version"] == 76
        state.update(version=expected_version, companion_sql=sql)

    def install_rls(migration_conn: object) -> None:
        assert migration_conn is conn
        state["rls_version"] = state["version"]

    monkeypatch.setattr(db, "_apply_postgres_migration_script", apply_sql)
    monkeypatch.setattr(db, "_ensure_chacha_rls_postgres", install_rls)
    try:
        db._initialize_schema_postgres()
        assert state["version"] == 77
        assert "CREATE TABLE IF NOT EXISTS persona_buddy_preferences" in state["companion_sql"]
        assert state["rls_version"] == 77
    finally:
        db._backend = original_backend
        db.close_connection()


def test_postgres_v77_migration_declares_companion_constraints(tmp_path) -> None:
    """PostgreSQL v77 DDL keeps the SQLite companion constraints."""
    db = CharactersRAGDB(tmp_path / "persona_companion_v73_postgres.sqlite", "persona-companion-v73-postgres")
    try:
        statements = db._convert_sqlite_schema_to_postgres_statements(
            db._MIGRATION_SQL_V76_TO_V77_PERSONA_COMPANION_POSTGRES
        )
    finally:
        db.close_connection()

    sql = "\n".join(statements)
    assert "ALTER TABLE persona_visual_packs ADD COLUMN IF NOT EXISTS companion_behavior_json TEXT" in sql
    assert "CREATE TABLE IF NOT EXISTS persona_buddy_preferences" in sql
    assert "ambient_mode IN ('off', 'expressive', 'roaming')" in sql
    assert "CREATE TABLE IF NOT EXISTS persona_visual_pack_reviews" in sql
    assert "UNIQUE(pack_id, fingerprint)" in sql
    assert re.search(r"SET\s+version\s*=\s*77", sql, flags=re.IGNORECASE)


@pytest.mark.integration
@pytest.mark.parametrize("source_version", [72, 73, 76])
def test_postgres_v77_migrates_prior_dev_and_enforces_companion_constraints(
    pg_database_config: DatabaseConfig,
    monkeypatch: pytest.MonkeyPatch,
    source_version: int,
) -> None:
    """The live PostgreSQL migration preserves dev and adds companion constraints."""
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    db: CharactersRAGDB | None = None
    try:
        with monkeypatch.context() as version_patch:
            bootstrap_version = 72 if source_version == 73 else source_version
            version_patch.setattr(CharactersRAGDB, "_CURRENT_SCHEMA_VERSION", bootstrap_version - 4)
            version_patch.setattr(CharactersRAGDB, "_POSTGRES_SCHEMA_VERSION", bootstrap_version)
            db = CharactersRAGDB(db_path=":memory:", client_id="persona-companion-v72-postgres", backend=backend)
            with backend.transaction() as conn:
                assert db._get_schema_version_postgres(conn) == bootstrap_version
                if source_version == 73:
                    db._apply_postgres_migration_script(
                        db._MIGRATION_SQL_V76_TO_V77_PERSONA_COMPANION_POSTGRES.replace(
                            "SET version = 77", "SET version = 73",
                        ).replace("AND version = 76", "AND version = 72"),
                        conn, expected_version=73,
                    )
                    backend.execute(
                        "INSERT INTO persona_buddy_preferences "
                        "(user_id, ambient_mode, version, created_at, updated_at) "
                        "VALUES ('legacy-owner', 'roaming', 4, '2026-08-23', '2026-08-23')",
                        connection=conn,
                    )
            if source_version == 73:
                legacy_persona = db.create_persona_profile({"user_id": "legacy-owner", "name": "Preserved Persona"})
                legacy_pack = db.create_persona_visual_pack(
                    persona_id=legacy_persona, user_id="legacy-owner", title="Preserved Pack",
                    manifest={"manifest_version": 1, "renderer_type": "sprite_frames"},
                )
                # Its read-back must not leave the checkout idle in transaction: those
                # locks would make the migration DDL below time out.
                assert db._get_thread_connection().info.transaction_status.name == "IDLE"
                with backend.transaction() as conn:
                    backend.execute(
                        "UPDATE persona_visual_packs SET companion_behavior_json = %s WHERE id = %s",
                        ('{"ambient_mode":"roaming"}', legacy_pack["id"]), connection=conn,
                    )
                    backend.execute(
                        "INSERT INTO persona_visual_pack_reviews "
                        "(id, pack_id, user_id, reviewer_user_id, fingerprint, pack_version, reviewed_at, created_at) "
                        "VALUES (%s, %s, 'legacy-owner', 'reviewer', %s, 1, '2026-08-23', '2026-08-23')",
                        ("legacy-review", legacy_pack["id"], "b" * 64), connection=conn,
                    )

        for _ in range(2):
            db._initialize_schema_postgres()
            with backend.transaction() as conn:
                assert db._get_schema_version_postgres(conn) == 77
                for table in ("persona_buddy_preferences", "persona_visual_pack_reviews"):
                    flags = backend.execute(
                        "SELECT relrowsecurity, relforcerowsecurity FROM pg_class WHERE oid = %s::regclass",
                        (table,), connection=conn,
                    ).rows[0]
                    assert flags["relrowsecurity"] and flags["relforcerowsecurity"]
                if source_version == 73:
                    preference = backend.execute(
                        "SELECT ambient_mode, version FROM persona_buddy_preferences WHERE user_id = 'legacy-owner'",
                        connection=conn,
                    ).rows[0]
                    assert (preference["ambient_mode"], preference["version"]) == ("roaming", 4)
                    assert backend.execute(
                        "SELECT fingerprint FROM persona_visual_pack_reviews WHERE id = 'legacy-review'", connection=conn,
                    ).scalar == "b" * 64
                    assert backend.execute(
                        "SELECT companion_behavior_json FROM persona_visual_packs WHERE id = %s",
                        (legacy_pack["id"],), connection=conn,
                    ).scalar == '{"ambient_mode":"roaming"}'

        timestamp = "2026-08-23T00:00:00Z"
        preference_insert = (
            "INSERT INTO persona_buddy_preferences "
            "(user_id, ambient_mode, version, created_at, updated_at) VALUES (%s, %s, 1, %s, %s)"
        )
        with pytest.raises(BackendDatabaseError):
            with backend.transaction() as conn:
                backend.execute(
                    preference_insert,
                    ("user-1", "chaotic", timestamp, timestamp),
                    connection=conn,
                )
        with backend.transaction() as conn:
            backend.execute(
                preference_insert,
                ("user-1", "off", timestamp, timestamp),
                connection=conn,
            )
        with pytest.raises(BackendDatabaseError):
            with backend.transaction() as conn:
                backend.execute(
                    preference_insert,
                    ("user-1", "expressive", timestamp, timestamp),
                    connection=conn,
                )

        persona_id = db.create_persona_profile({"user_id": "user-1", "name": "Migrated PostgreSQL Persona"})
        pack = db.create_persona_visual_pack(
            persona_id=persona_id,
            user_id="user-1",
            title="Migrated PostgreSQL Pack",
            manifest={"manifest_version": 1, "renderer_type": "sprite_frames"},
        )
        review_insert = (
            "INSERT INTO persona_visual_pack_reviews "
            "(id, pack_id, user_id, reviewer_user_id, fingerprint, pack_version, reviewed_at, created_at) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"
        )
        review_params = (
            "review-1",
            pack["id"],
            "user-1",
            "reviewer-1",
            "a" * 64,
            1,
            timestamp,
            timestamp,
        )
        with backend.transaction() as conn:
            backend.execute(review_insert, review_params, connection=conn)
        with pytest.raises(BackendDatabaseError):
            with backend.transaction() as conn:
                backend.execute(review_insert, ("review-2", *review_params[1:]), connection=conn)
    finally:
        if db is not None:
            db.close_connection()
        if backend.backend_type == BackendType.POSTGRESQL:
            backend.get_pool().close_all()


@pytest.mark.integration
def test_postgres_companion_rls_failure_rolls_back_schema_and_version(
    pg_database_config: DatabaseConfig, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed policy install cannot leave a77 marker or unprotected new tables."""
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    db: CharactersRAGDB | None = None
    try:
        with monkeypatch.context() as patch:
            patch.setattr(CharactersRAGDB, "_CURRENT_SCHEMA_VERSION", 72)
            patch.setattr(CharactersRAGDB, "_POSTGRES_SCHEMA_VERSION", 76)
            db = CharactersRAGDB(db_path=":memory:", client_id="rollback-owner", backend=backend)
        install_rls = db._ensure_chacha_rls_postgres

        def interrupted(conn: object) -> None:
            install_rls(conn)
            raise RuntimeError("companion policy install interrupted")

        monkeypatch.setattr(db, "_ensure_chacha_rls_postgres", interrupted)
        with pytest.raises(RuntimeError, match="companion policy install interrupted"):
            db._initialize_schema_postgres()
        with backend.transaction() as conn:
            assert db._get_schema_version_postgres(conn) == 76
            assert not backend.table_exists("persona_buddy_preferences", connection=conn)
            assert not backend.table_exists("persona_visual_pack_reviews", connection=conn)
            assert backend.table_exists("native_chat_operations", connection=conn)
    finally:
        if db is not None:
            db.close_connection()
        backend.get_pool().close_all()


@pytest.mark.integration
def test_postgres_companion_policies_isolate_non_superuser_reads_and_writes(
    pg_database_config: DatabaseConfig,
) -> None:
    """The installed policies constrain an application role, not just catalog flags."""
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    db: CharactersRAGDB | None = None

    class RollbackProbe(Exception):
        pass

    try:
        db = CharactersRAGDB(db_path=":memory:", client_id="companion-probe", backend=backend)
        packs = {}
        for owner in ("alice", "bob"):
            persona_id = db.create_persona_profile({"user_id": owner, "name": f"{owner} Persona"})
            packs[owner] = db.create_persona_visual_pack(
                persona_id=persona_id, user_id=owner, title=f"{owner} Pack",
                manifest={"manifest_version": 1, "renderer_type": "sprite_frames"},
            )["id"]
        with pytest.raises(RollbackProbe):
            with db.transaction() as conn:
                for owner in packs:
                    conn.execute(
                        "INSERT INTO persona_buddy_preferences "
                        "(user_id, ambient_mode, version, created_at, updated_at) "
                        "VALUES (?, 'off', 1, '2026-08-23', '2026-08-23')", (owner,),
                    )
                    conn.execute(
                        "INSERT INTO persona_visual_pack_reviews "
                        "(id, pack_id, user_id, reviewer_user_id, fingerprint, pack_version, reviewed_at, created_at) "
                        "VALUES (?, ?, ?, 'reviewer', ?, 1, '2026-08-23', '2026-08-23')",
                        (f"review-{owner}", packs[owner], owner, "a" * 64),
                    )
                role_sql = pg_sql.Identifier("companion_probe_owner_rls").as_string(conn._connection)
                conn.execute(f"CREATE ROLE {role_sql} NOLOGIN NOSUPERUSER")  # nosec B608 - quoted identifier
                conn.execute(f"GRANT USAGE ON SCHEMA public TO {role_sql}")  # nosec B608 - quoted identifier
                conn.execute(
                    f"GRANT SELECT, INSERT ON persona_buddy_preferences, persona_visual_pack_reviews TO {role_sql}"
                )  # nosec B608 - quoted identifier
                conn.execute(f"SET LOCAL ROLE {role_sql}")  # nosec B608 - quoted identifier
                for owner in packs:
                    conn.execute("SELECT set_config('app.current_user_id', ?, true)", (owner,))
                    for table in ("persona_buddy_preferences", "persona_visual_pack_reviews"):
                        table_sql = pg_sql.Identifier(table).as_string(conn._connection)
                        rows = conn.execute(f"SELECT user_id FROM {table_sql}").fetchall()  # nosec B608 - quoted identifier
                        assert [row["user_id"] for row in rows] == [owner]
                    with pytest.raises(Exception, match="row-level security"):
                        with conn._connection.transaction():
                            conn.execute(
                                "INSERT INTO persona_buddy_preferences "
                                "(user_id, ambient_mode, version, created_at, updated_at) "
                                "VALUES ('not-current-owner', 'off', 1, '2026-08-23', '2026-08-23')",
                            )
                raise RollbackProbe()
    finally:
        if db is not None:
            db.close_connection()
        backend.get_pool().close_all()
