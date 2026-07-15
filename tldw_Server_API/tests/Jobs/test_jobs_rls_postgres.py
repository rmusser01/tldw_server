import os

import pytest

psycopg = pytest.importorskip("psycopg")

from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Jobs.pg_migrations import (
    JobsRLSInstallationError,
    ensure_jobs_rls_policies_pg,
    ensure_jobs_tables_pg,
)

PLAYLIST_RLS_TABLES = (
    "playlist_preflights",
    "playlist_preflight_items",
    "playlist_materializations",
    "playlist_materialization_items",
    "media_ingest_runs",
    "media_ingest_run_items",
    "media_ingest_run_events",
)
JOBS_RLS_INSERT_TABLES = (
    "jobs",
    "job_events",
    "job_counters",
    "job_queue_controls",
    "job_sla_policies",
    "job_attachments",
    "job_dependencies",
    "jobs_archive",
)
JOBS_RLS_SEQUENCES = (
    "jobs_id_seq",
    "job_events_id_seq",
    "job_attachments_id_seq",
)


def _render_sql(statement) -> str:
    return statement.as_string() if hasattr(statement, "as_string") else str(statement)


PLAYLIST_RLS_CHILD_PARENTS = {
    "playlist_preflight_items": "playlist_preflights",
    "playlist_materializations": "playlist_preflights",
    "playlist_materialization_items": "playlist_materializations",
    "media_ingest_run_items": "media_ingest_runs",
    "media_ingest_run_events": "media_ingest_runs",
}


def test_playlist_rls_installer_covers_every_authority_table(monkeypatch):
    statements: list[str] = []

    class RecordingCursor:
        last_statement = ""

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def execute(self, statement, _params=None):
            self.last_statement = _render_sql(statement)
            statements.append(self.last_statement)

        def fetchone(self):
            if "relrowsecurity" in self.last_statement:
                return (True, True)
            return None

    class RecordingConnection:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def cursor(self):
            return RecordingCursor()

    monkeypatch.delenv("JOBS_PG_RLS_ROLE", raising=False)
    monkeypatch.setattr(psycopg, "connect", lambda *_args, **_kwargs: RecordingConnection())

    ensure_jobs_rls_policies_pg("postgresql://example/jobs")

    installed_sql = "\n".join(statements)
    for table in PLAYLIST_RLS_TABLES:
        assert f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY" in installed_sql
        assert f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY" in installed_sql
        select_prefix = f"CREATE POLICY {table}_owner_select ON {table} FOR SELECT"
        assert select_prefix in installed_sql
        select_sql = next(statement for statement in statements if select_prefix in statement)
        assert "USING" in select_sql
        assert "IS NOT NULL" in select_sql
        modify_prefix = f"CREATE POLICY {table}_owner_modify ON {table} FOR ALL"
        assert modify_prefix in installed_sql
        modify_sql = next(statement for statement in statements if modify_prefix in statement)
        assert "USING" in modify_sql
        assert "WITH CHECK" in modify_sql
        parent_table = PLAYLIST_RLS_CHILD_PARENTS.get(table)
        if parent_table is not None:
            assert f"FROM {parent_table}" in select_sql
            assert f"FROM {parent_table}" in modify_sql


def test_playlist_rls_installation_propagates_security_critical_alter_failure(monkeypatch):
    class FailingCursor:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def execute(self, statement, _params=None):
            if str(statement) == "ALTER TABLE playlist_preflights FORCE ROW LEVEL SECURITY":
                raise RuntimeError("force denied")

        def fetchone(self):
            return (True, True)

    class FailingConnection:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def cursor(self):
            return FailingCursor()

    monkeypatch.delenv("JOBS_PG_RLS_ROLE", raising=False)
    monkeypatch.setattr(psycopg, "connect", lambda *_args, **_kwargs: FailingConnection())

    with pytest.raises(RuntimeError, match="playlist.*RLS|force denied"):
        ensure_jobs_rls_policies_pg("postgresql://example/jobs")


def test_playlist_rls_installer_keeps_legacy_tables_best_effort_for_psycopg_errors(monkeypatch):
    class MissingLegacyCursor:
        last_statement = ""

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def execute(self, statement, _params=None):
            self.last_statement = _render_sql(statement)
            if self.last_statement == "ALTER TABLE jobs ENABLE ROW LEVEL SECURITY":
                raise psycopg.Error("legacy table unavailable")
            if "CREATE POLICY jobs_domain_select" in self.last_statement:
                raise psycopg.Error("legacy policy unavailable")

        def fetchone(self):
            if "relrowsecurity" in self.last_statement:
                return (True, True)
            return None

    class MissingLegacyConnection:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def cursor(self):
            return MissingLegacyCursor()

    monkeypatch.delenv("JOBS_PG_RLS_ROLE", raising=False)
    monkeypatch.setattr(psycopg, "connect", lambda *_args, **_kwargs: MissingLegacyConnection())

    ensure_jobs_rls_policies_pg("postgresql://example/jobs")


def test_jobs_schema_bootstrap_propagates_security_critical_rls_failure(monkeypatch):
    from tldw_Server_API.app.core.Jobs import pg_migrations

    class BootstrapCursor:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def execute(self, _statement, _params=None):
            if str(_statement) == "ALTER TABLE jobs ENABLE ROW LEVEL SECURITY":
                raise psycopg.Error("legacy RLS bootstrap must be delegated")
            return None

        def fetchone(self):
            return None

    class BootstrapConnection:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def cursor(self):
            return BootstrapCursor()

        def commit(self):
            return None

    def fail_rls(_db_url):
        raise JobsRLSInstallationError("playlist RLS installation failed")

    monkeypatch.setenv("JOBS_PG_RLS_ENABLE", "true")
    monkeypatch.setattr(psycopg, "connect", lambda *_args, **_kwargs: BootstrapConnection())
    monkeypatch.setattr(pg_migrations, "ensure_job_events_pg", lambda _db_url: None)
    monkeypatch.setattr(pg_migrations, "ensure_job_counters_pg", lambda _db_url: None)
    monkeypatch.setattr(pg_migrations, "ensure_jobs_rls_policies_pg", fail_rls)

    with pytest.raises(JobsRLSInstallationError, match="playlist RLS installation failed"):
        ensure_jobs_tables_pg("postgresql://example/jobs")


def test_playlist_rls_role_grants_are_scoped_and_role_must_be_nologin(monkeypatch):
    statements: list[str] = []

    class RoleCursor:
        last_statement = ""

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def execute(self, statement, _params=None):
            self.last_statement = _render_sql(statement)
            statements.append(self.last_statement)

        def fetchone(self):
            if "current_schema" in self.last_statement:
                return ("public",)
            if "FROM pg_roles" in self.last_statement:
                return (False, False, False)
            if "current_user" in self.last_statement:
                return ("app_user",)
            if "relrowsecurity" in self.last_statement:
                return (True, True)
            return None

    class RoleConnection:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def cursor(self):
            return RoleCursor()

    monkeypatch.setenv("JOBS_PG_RLS_ROLE", "jobs_rls")
    monkeypatch.setattr(psycopg, "connect", lambda *_args, **_kwargs: RoleConnection())

    ensure_jobs_rls_policies_pg("postgresql://example/jobs")

    installed_sql = "\n".join(statements)
    expected_tables = (*JOBS_RLS_INSERT_TABLES, *PLAYLIST_RLS_TABLES)
    assert (
        'REVOKE SELECT, UPDATE, DELETE ON ALL TABLES IN SCHEMA "public" FROM "jobs_rls"'
        in installed_sql
    )
    scoped_grant = next(
        statement
        for statement in statements
        if statement.startswith("GRANT SELECT, UPDATE, DELETE ON ")
    )
    assert scoped_grant == (
        "GRANT SELECT, UPDATE, DELETE ON "
        + ", ".join(f'"public"."{table}"' for table in expected_tables)
        + ' TO "jobs_rls"'
    )
    assert "GRANT SELECT, UPDATE, DELETE ON ALL TABLES IN SCHEMA" not in installed_sql
    assert "GRANT INSERT ON" in installed_sql
    for table in expected_tables:
        assert table in installed_sql
    assert "GRANT USAGE, SELECT ON SEQUENCE" in installed_sql
    for sequence in (
        *JOBS_RLS_SEQUENCES,
        "playlist_preflight_items_id_seq",
        "playlist_materialization_items_id_seq",
        "media_ingest_run_items_id_seq",
        "media_ingest_run_events_event_id_seq",
    ):
        assert sequence in installed_sql


def test_playlist_rls_role_quotes_database_identifiers(monkeypatch):
    statements: list[str] = []

    class QuotedIdentifierCursor:
        last_statement = ""

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def execute(self, statement, _params=None):
            self.last_statement = _render_sql(statement)
            statements.append(self.last_statement)

        def fetchone(self):
            if "current_schema" in self.last_statement:
                return ("Tenant-Schema",)
            if "FROM pg_roles" in self.last_statement:
                return (False, False, False)
            if "current_user" in self.last_statement:
                return ("App-Login",)
            if "relrowsecurity" in self.last_statement:
                return (True, True)
            return None

    class QuotedIdentifierConnection:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def cursor(self):
            return QuotedIdentifierCursor()

    monkeypatch.setenv("JOBS_PG_RLS_ROLE", "Jobs-RLS")
    monkeypatch.setattr(psycopg, "connect", lambda *_args, **_kwargs: QuotedIdentifierConnection())

    ensure_jobs_rls_policies_pg("postgresql://example/jobs")

    installed_sql = "\n".join(statements)
    assert 'GRANT "Jobs-RLS" TO "App-Login"' in installed_sql
    assert 'GRANT USAGE ON SCHEMA "Tenant-Schema" TO "Jobs-RLS"' in installed_sql
    assert (
        'REVOKE SELECT, UPDATE, DELETE ON ALL TABLES IN SCHEMA "Tenant-Schema" '
        'FROM "Jobs-RLS"'
    ) in installed_sql
    assert '"Tenant-Schema"."jobs"' in installed_sql


@pytest.mark.parametrize("role", ["", "x" * 64, "contains-nul"])
def test_playlist_rls_rejects_invalid_configured_role(monkeypatch, role):
    from tldw_Server_API.app.core.Jobs import pg_migrations

    if role == "contains-nul":
        real_getenv = os.getenv

        def fake_getenv(name, default=None):
            if name == "JOBS_PG_RLS_ROLE":
                return "bad\x00role"
            return real_getenv(name, default)

        monkeypatch.setattr(pg_migrations.os, "getenv", fake_getenv)
    else:
        monkeypatch.setenv("JOBS_PG_RLS_ROLE", role)

    with pytest.raises(JobsRLSInstallationError, match="1 to 63 bytes|NUL"):
        ensure_jobs_rls_policies_pg("postgresql://example/jobs")


@pytest.mark.parametrize(
    ("role_flags", "expected_error"),
    [
        ((True, False, False), "NOLOGIN"),
        ((False, True, False), "superuser"),
        ((False, False, True), "BYPASSRLS"),
    ],
)
def test_playlist_rls_rejects_configured_privileged_role(monkeypatch, role_flags, expected_error):
    class LoginRoleCursor:
        last_statement = ""

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def execute(self, statement, _params=None):
            self.last_statement = _render_sql(statement)

        def fetchone(self):
            if "current_schema" in self.last_statement:
                return ("public",)
            if "FROM pg_roles" in self.last_statement:
                return role_flags
            return None

    class LoginRoleConnection:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def cursor(self):
            return LoginRoleCursor()

    monkeypatch.setenv("JOBS_PG_RLS_ROLE", "jobs_rls")
    monkeypatch.setattr(psycopg, "connect", lambda *_args, **_kwargs: LoginRoleConnection())

    with pytest.raises(RuntimeError, match=expected_error):
        ensure_jobs_rls_policies_pg("postgresql://example/jobs")


def _dsn_or_skip(monkeypatch):
    base_dsn = os.getenv("JOBS_DB_URL")
    if not base_dsn:
        pytest.skip("JOBS_DB_URL not configured for Postgres RLS tests")
    # Enable single-update acquire path for consistency (not strictly needed here)
    monkeypatch.setenv("JOBS_PG_SINGLE_UPDATE_ACQUIRE", "true")
    monkeypatch.setenv("JOBS_PG_RLS_ENABLE", "true")
    role = "jobs_rls"
    monkeypatch.setenv("JOBS_PG_RLS_ROLE", role)
    monkeypatch.setenv("JOBS_PG_SKIP_SCHEMA_INIT", "true")
    # The application login assumes a dedicated NOLOGIN role for RLS enforcement.
    import psycopg
    from psycopg import sql as _sql
    with psycopg.connect(base_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT rolcanlogin, rolsuper, rolbypassrls FROM pg_roles WHERE rolname = %s",
                (role,),
            )
            role_ident = _sql.Identifier(role)
            if not cur.fetchone():
                cur.execute(_sql.SQL("CREATE ROLE {} NOLOGIN").format(role_ident))
            else:
                cur.execute(
                    _sql.SQL("ALTER ROLE {} NOLOGIN NOSUPERUSER NOBYPASSRLS").format(role_ident)
                )
            cur.execute("SELECT current_schema()")
            schema_row = cur.fetchone()
            schema_name = (schema_row[0] if schema_row else None) or "public"
            cur.execute(
                _sql.SQL("GRANT USAGE ON SCHEMA {} TO {}").format(
                    _sql.Identifier(schema_name),
                    role_ident,
                )
            )
            cur.execute(
                _sql.SQL("GRANT SELECT, UPDATE, DELETE ON ALL TABLES IN SCHEMA {} TO {}").format(
                    _sql.Identifier(schema_name),
                    role_ident,
                )
            )

    monkeypatch.setenv("JOBS_DB_URL", base_dsn)
    return base_dsn, base_dsn


def _row_val(row, key, idx):
    if isinstance(row, dict):
        return row.get(key)
    return row[idx] if row is not None else None


def _set_raw_rls_context(cur, *, owner_user_id: str | None) -> None:
    from psycopg import sql as _sql

    cur.execute(_sql.SQL("SET ROLE {}").format(_sql.Identifier("jobs_rls")))
    cur.execute("SELECT set_config('app.is_admin', 'false', false)")
    if owner_user_id is None:
        cur.execute("RESET app.owner_user_id")
    else:
        cur.execute("SELECT set_config('app.owner_user_id', %s, false)", (owner_user_id,))


def _seed(dsn):
    import psycopg

    with psycopg.connect(dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            # Minimal cleanup to keep test deterministic
            cur.execute("DELETE FROM job_events")
            cur.execute("DELETE FROM jobs")
            cur.execute("DELETE FROM job_counters")
            cur.execute("DELETE FROM job_queue_controls")
            cur.execute("DELETE FROM job_sla_policies")
            # Seed jobs across domains/owners
            cur.execute(
                "INSERT INTO jobs(domain,queue,job_type,owner_user_id,status,priority,created_at) VALUES"
                "('chatbooks','default','export','u1','queued',5,NOW()),"
                "('chatbooks','default','export','u2','queued',5,NOW()),"
                "('web','crawler','fetch','u1','queued',5,NOW()),"
                "('web','crawler','fetch','u2','queued',5,NOW())"
            )
            cur.execute(
                "INSERT INTO job_queue_controls(domain,queue,paused,drain) VALUES"
                "('chatbooks','default',false,false) ON CONFLICT (domain,queue) DO NOTHING"
            )
            cur.execute(
                "INSERT INTO job_counters(domain,queue,job_type,ready_count,scheduled_count,processing_count,quarantined_count) VALUES"
                "('chatbooks','default','export',2,0,0,0) ON CONFLICT (domain,queue,job_type) DO NOTHING"
            )
            cur.execute(
                "INSERT INTO job_sla_policies(domain,queue,job_type,max_queue_latency_seconds,max_duration_seconds,enabled) VALUES"
                "('chatbooks','default','export', 60, 300, true) ON CONFLICT (domain,queue,job_type) DO NOTHING"
            )
            cur.execute(
                "INSERT INTO job_events(job_id,domain,queue,job_type,event_type,attrs_json,owner_user_id,created_at) VALUES"
                "(NULL,'chatbooks','default','export','jobs.seed','{}'::jsonb,'u1',NOW()),"
                "(NULL,'web','crawler','fetch','jobs.seed','{}'::jsonb,'u2',NOW())"
            )


def _seed_playlist_authority(dsn):
    with psycopg.connect(dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT set_config('app.is_admin', 'true', false)")
            for table in (
                "media_ingest_run_events",
                "media_ingest_run_items",
                "media_ingest_runs",
                "playlist_materialization_items",
                "playlist_materializations",
                "playlist_preflight_items",
                "playlist_preflights",
            ):
                cur.execute(f"DELETE FROM {table}")

            cur.execute(
                """
                INSERT INTO playlist_preflights(
                  preflight_id, owner_user_id, status, source_url, source_kind, expires_at
                ) VALUES
                  ('pf-u1', 'u1', 'completed', 'https://example.test/pf-u1', 'playlist', NOW() + INTERVAL '1 hour'),
                  ('pf-u2', 'u2', 'completed', 'https://example.test/pf-u2', 'playlist', NOW() + INTERVAL '1 hour')
                """
            )
            cur.execute(
                """
                INSERT INTO playlist_preflight_items(
                  preflight_id, owner_user_id, occurrence_id, ordinal,
                  occurrence_index_for_source, source_kind, availability, duplicate_status
                ) VALUES
                  ('pf-u1', 'u1', 'pfi-u1', 1, 1, 'video', 'available', 'new'),
                  ('pf-u2', 'u2', 'pfi-u2', 2, 1, 'video', 'available', 'new'),
                  ('pf-u2', 'u1', 'pfi-invalid-parent', 3, 1, 'video', 'available', 'new')
                """
            )
            cur.execute(
                """
                INSERT INTO playlist_materializations(
                  materialization_id, preflight_id, owner_user_id, status, expires_at
                ) VALUES
                  ('mat-u1', 'pf-u1', 'u1', 'ready', NOW() + INTERVAL '1 hour'),
                  ('mat-u2', 'pf-u2', 'u2', 'ready', NOW() + INTERVAL '1 hour'),
                  ('mat-invalid-parent', 'pf-u2', 'u1', 'ready', NOW() + INTERVAL '1 hour')
                """
            )
            cur.execute(
                """
                INSERT INTO playlist_materialization_items(
                  materialization_id, owner_user_id, occurrence_id, ordinal,
                  source_url, source_kind
                ) VALUES
                  ('mat-u1', 'u1', 'mi-u1', 1, 'https://example.test/mi-u1', 'video'),
                  ('mat-u2', 'u2', 'mi-u2', 2, 'https://example.test/mi-u2', 'video'),
                  ('mat-u2', 'u1', 'mi-invalid-parent', 3, 'https://example.test/mi-invalid', 'video')
                """
            )
            cur.execute(
                """
                INSERT INTO media_ingest_runs(run_id, owner_user_id, status, expires_at) VALUES
                  ('run-u1', 'u1', 'ready', NOW() + INTERVAL '1 hour'),
                  ('run-u2', 'u2', 'ready', NOW() + INTERVAL '1 hour')
                """
            )
            cur.execute(
                """
                INSERT INTO media_ingest_run_items(
                  run_id, owner_user_id, occurrence_id, ordinal, input_kind, state
                ) VALUES
                  ('run-u1', 'u1', 'ri-u1', 1, 'url', 'staged'),
                  ('run-u2', 'u2', 'ri-u2', 2, 'url', 'staged'),
                  ('run-u2', 'u1', 'ri-invalid-parent', 3, 'url', 'staged')
                """
            )
            cur.execute(
                """
                INSERT INTO media_ingest_run_events(
                  run_id, owner_user_id, event_type, state
                ) VALUES
                  ('run-u1', 'u1', 'event-u1', 'staged'),
                  ('run-u2', 'u2', 'event-u2', 'staged'),
                  ('run-u2', 'u1', 'event-invalid-parent', 'staged')
                """
            )


@pytest.mark.pg_jobs
def test_rls_context_filters_results(monkeypatch):
    admin_dsn, rls_dsn = _dsn_or_skip(monkeypatch)
    ensure_jobs_tables_pg(admin_dsn)
    ensure_jobs_rls_policies_pg(admin_dsn)
    _seed(admin_dsn)

    jm = JobManager(backend="postgres", db_url=rls_dsn)

    # Admin: see all rows (bypass)
    JobManager.set_rls_context(is_admin=True, domain_allowlist=None, owner_user_id=None)
    all_rows = jm.list_jobs()
    assert len(all_rows) >= 4

    # chatbooks:u1: see exactly one job (domain + owner)
    JobManager.set_rls_context(is_admin=False, domain_allowlist="chatbooks", owner_user_id="u1")
    cb_u1 = jm.list_jobs()
    assert len(cb_u1) == 1
    assert cb_u1[0]["domain"] == "chatbooks" and cb_u1[0]["owner_user_id"] == "u1"

    # web:u2: see exactly one
    JobManager.set_rls_context(is_admin=False, domain_allowlist="web", owner_user_id="u2")
    web_u2 = jm.list_jobs()
    assert len(web_u2) == 1
    assert web_u2[0]["domain"] == "web" and web_u2[0]["owner_user_id"] == "u2"


@pytest.mark.pg_jobs
def test_rls_applies_to_events_and_controls(monkeypatch):
    admin_dsn, rls_dsn = _dsn_or_skip(monkeypatch)
    ensure_jobs_tables_pg(admin_dsn)
    ensure_jobs_rls_policies_pg(admin_dsn)
    _seed(admin_dsn)

    jm = JobManager(backend="postgres", db_url=rls_dsn)

    # chatbooks:u1 context
    JobManager.set_rls_context(is_admin=False, domain_allowlist="chatbooks", owner_user_id="u1")
    conn = jm._connect()
    try:
        with jm._pg_cursor(conn) as cur:
            # job_events should only show chatbooks/u1 rows
            cur.execute("SELECT COUNT(*) FROM job_events")
            ev_count = int(_row_val(cur.fetchone(), "count", 0) or 0)
            assert ev_count == 1
            # job_queue_controls should only show chatbooks rows
            cur.execute("SELECT COUNT(*) FROM job_queue_controls")
            qc_count = int(_row_val(cur.fetchone(), "count", 0) or 0)
            assert qc_count == 1
            # job_sla_policies should only show chatbooks rows
            cur.execute("SELECT COUNT(*) FROM job_sla_policies")
            sla_count = int(_row_val(cur.fetchone(), "count", 0) or 0)
            assert sla_count >= 1
    finally:
        conn.close()


@pytest.mark.pg_jobs
def test_playlist_authority_rls_isolates_owners_and_fences_children(monkeypatch):
    admin_dsn, rls_dsn = _dsn_or_skip(monkeypatch)
    ensure_jobs_tables_pg(admin_dsn)
    ensure_jobs_rls_policies_pg(admin_dsn)
    _seed_playlist_authority(admin_dsn)

    visible_rows = (
        ("playlist_preflights", "preflight_id", "pf-u1"),
        ("playlist_preflight_items", "occurrence_id", "pfi-u1"),
        ("playlist_materializations", "materialization_id", "mat-u1"),
        ("playlist_materialization_items", "occurrence_id", "mi-u1"),
        ("media_ingest_runs", "run_id", "run-u1"),
        ("media_ingest_run_items", "occurrence_id", "ri-u1"),
        ("media_ingest_run_events", "event_type", "event-u1"),
    )
    child_rows = (
        ("playlist_preflight_items", "occurrence_id", "pfi-u1", "preflight_id", "pf-u2"),
        (
            "playlist_materializations",
            "materialization_id",
            "mat-u1",
            "preflight_id",
            "pf-u2",
        ),
        (
            "playlist_materialization_items",
            "occurrence_id",
            "mi-u1",
            "materialization_id",
            "mat-u2",
        ),
        ("media_ingest_run_items", "occurrence_id", "ri-u1", "run_id", "run-u2"),
        ("media_ingest_run_events", "event_type", "event-u1", "run_id", "run-u2"),
    )

    with psycopg.connect(rls_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            _set_raw_rls_context(cur, owner_user_id="u1")

            for table, identity_column, expected_identity in visible_rows:
                cur.execute(f"SELECT {identity_column} FROM {table} ORDER BY {identity_column}")
                assert [row[0] for row in cur.fetchall()] == [expected_identity]

                cur.execute(
                    f"UPDATE {table} SET owner_user_id = owner_user_id WHERE owner_user_id = 'u2'"
                )
                assert cur.rowcount == 0

            for table, identity_column, identity, parent_column, other_parent in child_rows:
                with pytest.raises(psycopg.errors.InsufficientPrivilege):
                    cur.execute(
                        f"UPDATE {table} SET owner_user_id = 'u2' "
                        f"WHERE {identity_column} = %s",
                        (identity,),
                    )
                with pytest.raises(psycopg.errors.InsufficientPrivilege):
                    cur.execute(
                        f"UPDATE {table} SET {parent_column} = %s "
                        f"WHERE {identity_column} = %s",
                        (other_parent, identity),
                    )


@pytest.mark.pg_jobs
def test_playlist_rls_role_and_table_flags_are_hardened(monkeypatch):
    admin_dsn, _rls_dsn = _dsn_or_skip(monkeypatch)
    ensure_jobs_tables_pg(admin_dsn)
    ensure_jobs_rls_policies_pg(admin_dsn)

    with psycopg.connect(admin_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT rolcanlogin, rolsuper, rolbypassrls "
                "FROM pg_roles WHERE rolname = 'jobs_rls'"
            )
            assert cur.fetchone() == (False, False, False)
            cur.execute(
                """
                SELECT relname, relrowsecurity, relforcerowsecurity
                FROM pg_class
                WHERE relname = ANY(%s)
                ORDER BY relname
                """,
                (list(PLAYLIST_RLS_TABLES),),
            )
            flags = {row[0]: (bool(row[1]), bool(row[2])) for row in cur.fetchall()}
    assert flags == dict.fromkeys(sorted(PLAYLIST_RLS_TABLES), (True, True))


@pytest.mark.pg_jobs
def test_playlist_rls_role_cannot_access_unrelated_tables(monkeypatch):
    admin_dsn, _rls_dsn = _dsn_or_skip(monkeypatch)
    ensure_jobs_tables_pg(admin_dsn)
    probe_table = "jobs_rls_unrelated_privilege_probe"

    from psycopg import sql as _sql

    with psycopg.connect(admin_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT current_schema()")
            schema_name = cur.fetchone()[0]
            probe_ident = _sql.Identifier(schema_name, probe_table)
            cur.execute(_sql.SQL("DROP TABLE IF EXISTS {}").format(probe_ident))
            cur.execute(_sql.SQL("CREATE TABLE {} (id INTEGER)").format(probe_ident))
            cur.execute(
                _sql.SQL("GRANT SELECT, UPDATE, DELETE ON {} TO jobs_rls").format(probe_ident)
            )

    try:
        ensure_jobs_rls_policies_pg(admin_dsn)
        qualified_probe = probe_ident.as_string()
        with psycopg.connect(admin_dsn, autocommit=True) as conn:
            with conn.cursor() as cur:
                for privilege in ("SELECT", "UPDATE", "DELETE"):
                    cur.execute(
                        "SELECT has_table_privilege(%s, %s, %s)",
                        ("jobs_rls", qualified_probe, privilege),
                    )
                    assert cur.fetchone()[0] is False
                cur.execute("SET ROLE jobs_rls")
                with pytest.raises(psycopg.errors.InsufficientPrivilege):
                    cur.execute(_sql.SQL("SELECT * FROM {}").format(probe_ident))
    finally:
        with psycopg.connect(admin_dsn, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(_sql.SQL("DROP TABLE IF EXISTS {}").format(probe_ident))


@pytest.mark.pg_jobs
def test_playlist_rls_unset_and_blank_owner_context_fail_closed(monkeypatch):
    admin_dsn, rls_dsn = _dsn_or_skip(monkeypatch)
    ensure_jobs_tables_pg(admin_dsn)
    ensure_jobs_rls_policies_pg(admin_dsn)
    _seed_playlist_authority(admin_dsn)

    with psycopg.connect(rls_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            _set_raw_rls_context(cur, owner_user_id=None)
            for context_name, owner_value in (("unset", None), ("blank", "")):
                if owner_value is None:
                    cur.execute("RESET app.owner_user_id")
                else:
                    cur.execute("SELECT set_config('app.owner_user_id', %s, false)", (owner_value,))
                for table in PLAYLIST_RLS_TABLES:
                    cur.execute(f"SELECT COUNT(*) FROM {table}")
                    assert cur.fetchone()[0] == 0, context_name
                    cur.execute(f"UPDATE {table} SET owner_user_id = owner_user_id")
                    assert cur.rowcount == 0, context_name
                with pytest.raises(psycopg.errors.InsufficientPrivilege):
                    cur.execute(
                        """
                        INSERT INTO playlist_preflights(
                          preflight_id, owner_user_id, status, source_url, source_kind, expires_at
                        ) VALUES (%s, 'u1', 'completed', 'https://example.test/denied',
                                  'playlist', NOW() + INTERVAL '1 hour')
                        """,
                        (f"pf-denied-{context_name}",),
                    )


@pytest.mark.pg_jobs
def test_playlist_rls_role_can_insert_owned_graph_but_not_cross_tenant(monkeypatch):
    admin_dsn, rls_dsn = _dsn_or_skip(monkeypatch)
    ensure_jobs_tables_pg(admin_dsn)
    ensure_jobs_rls_policies_pg(admin_dsn)
    _seed_playlist_authority(admin_dsn)

    with psycopg.connect(rls_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            _set_raw_rls_context(cur, owner_user_id="u1")
            cur.execute(
                """
                INSERT INTO playlist_preflights(
                  preflight_id, owner_user_id, status, source_url, source_kind, expires_at
                ) VALUES ('pf-insert-u1', 'u1', 'completed', 'https://example.test/pf-insert',
                          'playlist', NOW() + INTERVAL '1 hour')
                """
            )
            cur.execute(
                """
                INSERT INTO playlist_preflight_items(
                  preflight_id, owner_user_id, occurrence_id, ordinal,
                  occurrence_index_for_source, source_kind, availability, duplicate_status
                ) VALUES ('pf-insert-u1', 'u1', 'pfi-insert-u1', 10, 1,
                          'video', 'available', 'new')
                """
            )
            cur.execute(
                """
                INSERT INTO playlist_materializations(
                  materialization_id, preflight_id, owner_user_id, status, expires_at
                ) VALUES ('mat-insert-u1', 'pf-insert-u1', 'u1', 'ready',
                          NOW() + INTERVAL '1 hour')
                """
            )
            cur.execute(
                """
                INSERT INTO playlist_materialization_items(
                  materialization_id, owner_user_id, occurrence_id, ordinal,
                  source_url, source_kind
                ) VALUES ('mat-insert-u1', 'u1', 'mi-insert-u1', 10,
                          'https://example.test/mi-insert', 'video')
                """
            )
            cur.execute(
                """
                INSERT INTO media_ingest_runs(run_id, owner_user_id, status, expires_at)
                VALUES ('run-insert-u1', 'u1', 'ready', NOW() + INTERVAL '1 hour')
                """
            )
            cur.execute(
                """
                INSERT INTO media_ingest_run_items(
                  run_id, owner_user_id, occurrence_id, ordinal, input_kind, state
                ) VALUES ('run-insert-u1', 'u1', 'ri-insert-u1', 10, 'url', 'staged')
                """
            )
            cur.execute(
                """
                INSERT INTO media_ingest_run_events(run_id, owner_user_id, event_type, state)
                VALUES ('run-insert-u1', 'u1', 'event-insert-u1', 'staged')
                """
            )

            for table in PLAYLIST_RLS_TABLES:
                cur.execute(f"SELECT COUNT(*) FROM {table} WHERE owner_user_id = 'u1'")
                assert cur.fetchone()[0] >= 1

            with pytest.raises(psycopg.errors.InsufficientPrivilege):
                cur.execute(
                    """
                    INSERT INTO playlist_preflights(
                      preflight_id, owner_user_id, status, source_url, source_kind, expires_at
                    ) VALUES ('pf-cross-owner', 'u2', 'completed',
                              'https://example.test/cross-owner', 'playlist',
                              NOW() + INTERVAL '1 hour')
                    """
                )

            cross_parent_inserts = (
                """
                INSERT INTO playlist_preflight_items(
                  preflight_id, owner_user_id, occurrence_id, ordinal,
                  occurrence_index_for_source, source_kind, availability, duplicate_status
                ) VALUES ('pf-u2', 'u1', 'pfi-cross-parent', 20, 1,
                          'video', 'available', 'new')
                """,
                """
                INSERT INTO playlist_materializations(
                  materialization_id, preflight_id, owner_user_id, status, expires_at
                ) VALUES ('mat-cross-parent', 'pf-u2', 'u1', 'ready',
                          NOW() + INTERVAL '1 hour')
                """,
                """
                INSERT INTO playlist_materialization_items(
                  materialization_id, owner_user_id, occurrence_id, ordinal,
                  source_url, source_kind
                ) VALUES ('mat-u2', 'u1', 'mi-cross-parent', 20,
                          'https://example.test/cross-parent', 'video')
                """,
                """
                INSERT INTO media_ingest_run_items(
                  run_id, owner_user_id, occurrence_id, ordinal, input_kind, state
                ) VALUES ('run-u2', 'u1', 'ri-cross-parent', 20, 'url', 'staged')
                """,
                """
                INSERT INTO media_ingest_run_events(run_id, owner_user_id, event_type, state)
                VALUES ('run-u2', 'u1', 'event-cross-parent', 'staged')
                """,
            )
            for statement in cross_parent_inserts:
                with pytest.raises(psycopg.errors.InsufficientPrivilege):
                    cur.execute(statement)
