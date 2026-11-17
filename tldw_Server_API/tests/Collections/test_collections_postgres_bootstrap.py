"""Real PostgreSQL regressions for idempotent Collections schema setup."""

import pytest

from tldw_Server_API.app.core.DB_Management.backends.base import DatabaseError
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.Collections_DB import CollectionsDatabase


@pytest.fixture
def collections_backend(pg_database_config):
    """Use the repository's isolated PostgreSQL database, without schema seeds."""
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    try:
        yield backend
    finally:
        CollectionsDatabase._bootstrapped_backend_targets.discard(CollectionsDatabase._backend_target_key(backend))
        backend.get_pool().close_all()


def test_fresh_and_repeated_postgres_collections_schema_preserve_content(collections_backend):
    """Fresh and explicit repeated bootstrap must keep usable canonical tables."""
    db = CollectionsDatabase.from_backend("1", collections_backend)
    template = db.create_output_template(
        name="PostgreSQL bootstrap control",
        type_="summary",
        format_="markdown",
        body="Retain this template across schema initialization.",
        description="Fresh PostgreSQL fixture",
        is_default=False,
    )
    db.ensure_schema()
    rows, total = db.list_output_templates(
        q="PostgreSQL bootstrap control",
        limit=10,
        offset=0,
    )
    assert total == 1
    assert [(row.id, row.body) for row in rows] == [(template.id, template.body)]


def test_postgres_collections_backfills_a_missing_legacy_column(collections_backend):
    """Introspection must still repair an actually missing legacy column."""
    db = CollectionsDatabase.from_backend("1", collections_backend)
    collections_backend.execute("ALTER TABLE output_templates DROP COLUMN metadata_json")
    db.ensure_schema()
    assert "metadata_json" in db._table_columns("output_templates")


def test_postgres_collections_does_not_hide_column_inspection_failure(
    collections_backend,
    monkeypatch,
):
    """An unreadable schema is an error, not an empty column inventory."""
    db = CollectionsDatabase.from_backend("1", collections_backend)
    original = collections_backend.get_table_info

    def fail_output_inspection(table):
        if table == "outputs":
            raise DatabaseError("Cannot inspect output columns")
        return original(table)

    monkeypatch.setattr(collections_backend, "get_table_info", fail_output_inspection)
    with pytest.raises(DatabaseError, match="Cannot inspect output columns"):
        db.ensure_schema()


@pytest.fixture
def migrated_collections_serving(collections_backend, pg_restricted_backend, pg_database_config, monkeypatch):
    """A real owner deployment plus a distinct DML-only serving role."""
    from psycopg import sql

    monkeypatch.setattr(CollectionsDatabase, "_bootstrapped_backend_targets", set())
    role = sql.Identifier(pg_restricted_backend.config.pg_user)
    with collections_backend.transaction() as conn:
        conn.execute(sql.SQL("ALTER SCHEMA public OWNER TO {}").format(sql.Identifier(pg_database_config.pg_user)))
        conn.execute(sql.SQL("REVOKE CREATE ON SCHEMA public FROM {}").format(role))
        conn.execute(
            sql.SQL("REVOKE CREATE ON DATABASE {} FROM {}").format(sql.Identifier(pg_database_config.pg_database), role)
        )
    owner = CollectionsDatabase.from_backend("1", collections_backend)
    retained = owner.create_output_template(
        name="X01 retained Collections schema control",
        type_="summary",
        format_="markdown",
        body="Retain this real owner-provisioned content.",
        description="Historical serving boundary",
        is_default=False,
    )
    with collections_backend.transaction() as conn:
        conn.execute(sql.SQL("GRANT USAGE ON SCHEMA public TO {}").format(role))
        conn.execute(sql.SQL("GRANT SELECT,INSERT,UPDATE,DELETE ON ALL TABLES IN SCHEMA public TO {}").format(role))
        conn.execute(sql.SQL("GRANT USAGE,SELECT ON ALL SEQUENCES IN SCHEMA public TO {}").format(role))
    flags = pg_restricted_backend.execute(
        "SELECT rolsuper,rolbypassrls,rolcreatedb,rolcreaterole, "
        "has_schema_privilege(current_user,'public','CREATE') AS can_create, "
        "EXISTS(SELECT 1 FROM pg_class WHERE relnamespace='public'::regnamespace "
        "AND pg_has_role(current_user,relowner,'USAGE')) AS owns_table "
        "FROM pg_roles WHERE rolname=current_user"
    ).first
    assert flags and not any(flags.values())  # nosec B101 -- actual ordinary nonowner, never the migration role.
    return collections_backend, pg_restricted_backend, retained


def test_nonowner_collections_preserves_migrated_content_and_user_boundaries(migrated_collections_serving):
    """An already migrated serving role must read/write without replaying owner DDL."""
    collections_backend, pg_restricted_backend, retained = migrated_collections_serving
    queries = (
        "SELECT table_name,column_name,data_type,column_default FROM information_schema.columns "
        "WHERE table_schema='public' ORDER BY table_name,ordinal_position",
        "SELECT tablename,indexname,indexdef FROM pg_indexes WHERE schemaname='public' ORDER BY tablename,indexname",
    )
    before = [collections_backend.execute(query).rows for query in queries]
    # A fresh serving process has no owner's in-memory bootstrap receipt.
    CollectionsDatabase._bootstrapped_backend_targets.clear()
    try:
        serving = CollectionsDatabase.from_backend("1", pg_restricted_backend)
    except DatabaseError as error:
        raise AssertionError(
            {
                "phase": "ordinary-constructor",
                "error": type(error).__name__,
                "cause": type(error.__cause__).__name__,
            }
        ) from None
    rows, count = serving.list_output_templates(q="X01 retained Collections schema control", limit=10, offset=0)
    assert count == 1 and [(row.id, row.body) for row in rows] == [(retained.id, retained.body)]  # nosec B101 -- retain real owner data.
    new = serving.create_output_template(
        name="X01 new Collections schema control",
        type_="summary",
        format_="markdown",
        body="New content through the ordinary serving role.",
        description=None,
        is_default=False,
    )
    other = CollectionsDatabase.from_backend("2", pg_restricted_backend)
    foreign, count = other.list_output_templates(q="X01 retained Collections schema control", limit=10, offset=0)
    assert count == 0 and foreign == []  # nosec B101 -- actual foreign-user filtering after migration.
    rows, count = serving.list_output_templates(q="X01 new Collections schema control", limit=10, offset=0)
    assert count == 1 and rows[0].id == new.id  # nosec B101 -- ordinary write/read remains usable.
    assert [collections_backend.execute(query).rows for query in queries] == before  # nosec B101 -- serving path never changes owner schema.


def _collections_schema_metadata(backend):
    """Keep fixed public schema metadata in memory while exercising refusal."""
    queries = (
        "SELECT table_name,column_name,data_type,is_nullable,column_default FROM information_schema.columns "
        "WHERE table_schema='public' ORDER BY table_name,ordinal_position",
        "SELECT tablename,indexname,indexdef FROM pg_indexes WHERE schemaname='public' ORDER BY tablename,indexname",
        "SELECT conrelid,conname,pg_get_constraintdef(oid) AS definition FROM pg_constraint "
        "WHERE connamespace='public'::regnamespace ORDER BY conrelid,conname",
    )
    return [backend.execute(query).rows for query in queries]


def test_nonowner_collections_explicit_ensure_never_replays_ddl(migrated_collections_serving):
    """Explicit schema callers retain the same ordinary-role boundary as constructors."""
    owner, serving_backend, _retained = migrated_collections_serving
    before = _collections_schema_metadata(owner)
    db = CollectionsDatabase.from_backend("1", serving_backend)
    db.ensure_schema()
    assert _collections_schema_metadata(owner) == before  # nosec B101 -- direct runtime ensure leaves owner schema intact.


@pytest.mark.parametrize(
    "drift",
    [
        ("DROP TABLE public.audio_studio_idempotency_keys",),
        ("ALTER TABLE public.user_notifications DROP COLUMN delivery_status",),
        ("ALTER TABLE public.audiobook_projects DROP COLUMN project_id",),
        ("DROP INDEX public.ux_outputs_user_idempotency_active",),
        (
            "DROP INDEX public.ux_outputs_user_idempotency_active",
            "CREATE UNIQUE INDEX ux_outputs_user_idempotency_active ON public.outputs(user_id,idempotency_key) "
            "WHERE idempotency_key IS NOT NULL",
        ),
        (
            "DROP INDEX public.ux_outputs_user_idempotency_active",
            "CREATE UNIQUE INDEX ux_outputs_user_idempotency_active ON public.outputs(user_id,idempotency_key) "
            "WHERE (idempotency_key IS NOT NULL AND deleted) = FALSE",
        ),
        (
            "DROP INDEX public.ux_output_templates_user_name",
            "ALTER TABLE public.output_templates ADD CONSTRAINT ux_output_templates_user_name "
            "UNIQUE(user_id,name) DEFERRABLE INITIALLY IMMEDIATE",
        ),
        (
            "ALTER TABLE public.audio_studio_idempotency_keys DROP CONSTRAINT audio_studio_idempotency_keys_pkey",
            "ALTER TABLE public.audio_studio_idempotency_keys ADD PRIMARY KEY(namespace,key)",
        ),
        ("CREATE UNIQUE INDEX ux_outputs_user_idempotency ON public.outputs(user_id,idempotency_key)",),
    ],
    ids=[
        "late-table",
        "backfill-column",
        "project-column",
        "unique-index",
        "predicate",
        "predicate-grouping",
        "deferrable",
        "composite-key",
        "obsolete-index",
    ],
)
def test_nonowner_collections_refuses_schema_drift_despite_owner_cache(migrated_collections_serving, drift):
    """An owner receipt cannot admit another role to an incomplete runtime schema."""
    owner, serving_backend, _retained = migrated_collections_serving
    for statement in drift:
        owner.execute(statement)
    before = _collections_schema_metadata(owner)
    with pytest.raises(DatabaseError):
        CollectionsDatabase.from_backend("1", serving_backend)
    assert _collections_schema_metadata(owner) == before  # nosec B101 -- readiness refusal performs no owner repair.


def test_collections_same_database_role_refresh_checks_runtime_schema(migrated_collections_serving, monkeypatch):
    """A changed serving role on one DB must pass readiness before any public operation."""
    owner, serving_backend, _retained = migrated_collections_serving
    adapter = CollectionsDatabase.from_backend("1", owner)
    owner.execute("ALTER TABLE public.user_notifications DROP COLUMN delivery_status")
    before = _collections_schema_metadata(owner)
    adapter._uses_shared_content_backend = True
    monkeypatch.setattr(adapter, "_resolve_backend", lambda: (serving_backend, True))
    with pytest.raises(DatabaseError):
        adapter.list_output_templates(q="X01", limit=10, offset=0)
    assert _collections_schema_metadata(owner) == before  # nosec B101 -- role refresh cannot bypass or repair schema admission.


def test_nonowner_collections_reconciles_templates_and_projects_per_user(migrated_collections_serving, monkeypatch):
    """Schema caching must not skip existing tenant-specific DML for the next user."""
    import json
    from types import SimpleNamespace

    from tldw_Server_API.app.core.DB_Management import Collections_DB as module
    from tldw_Server_API.app.core.Watchlists import template_store

    owner, serving_backend, _retained = migrated_collections_serving
    first = CollectionsDatabase.from_backend("1", owner)
    second = CollectionsDatabase.from_backend("2", owner)
    old = []
    for user, db in (("1", first), ("2", second)):
        old.append(
            db.create_audiobook_project(
                project_id=None,
                title="Retained audiobook " + user,
                source_ref=None,
                status="draft",
                settings_json=json.dumps({"project_id": "x01-project-" + user}),
            )
        )
    record = SimpleNamespace(
        name="X01 catalog seed",
        format="markdown",
        content="Per-user template.",
        description="Actual tenant reconciliation",
        updated_at=1700000000.0,
    )
    monkeypatch.setitem(module.settings, "WATCHLISTS_SEED_OUTPUT_TEMPLATES", True)
    monkeypatch.setattr(template_store, "list_templates", lambda: [record])
    monkeypatch.setattr(template_store, "load_template", lambda name: record)
    foreign_settings = first.get_audiobook_project(old[0].id).settings_json
    parsed_foreign = []
    original_loads = module.json.loads

    def observe_json(value, *args, **kwargs):
        if value == foreign_settings:
            parsed_foreign.append(True)
        return original_loads(value, *args, **kwargs)

    monkeypatch.setattr(module.json, "loads", observe_json)
    bob = CollectionsDatabase.from_backend("2", serving_backend)
    assert not parsed_foreign  # nosec B101 -- tenant reconciliation must not inspect a foreign user's settings.
    rows, count = bob.list_output_templates(q=record.name, limit=10, offset=0)
    assert count == 1 and rows[0].body == record.content  # nosec B101 -- normal seed DML must run for each actual user.
    assert bob.get_audiobook_project(old[1].id).project_id == "x01-project-2"  # nosec B101 -- retained user2 project backfill.
    assert first.get_audiobook_project(old[0].id).project_id is None  # nosec B101 -- another user's pending reconciliation stays untouched.
    alice = CollectionsDatabase.from_backend("1", serving_backend)
    rows, count = alice.list_output_templates(q=record.name, limit=10, offset=0)
    assert count == 1 and rows[0].user_id == "1"  # nosec B101 -- second tenant also receives its own template.
    assert alice.get_audiobook_project(old[0].id).project_id == "x01-project-1"  # nosec B101 -- per-user backfill survives schema separation.
    assert bob.get_audiobook_project(old[1].id).project_id == "x01-project-2"  # nosec B101 -- later reconciliation preserves the other user.
