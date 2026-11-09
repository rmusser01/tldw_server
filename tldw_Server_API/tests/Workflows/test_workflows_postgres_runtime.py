"""Owner-migrated Workflows must serve with ordinary PostgreSQL privileges."""

import pytest
from psycopg import sql

from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.Workflows_DB import WorkflowsDatabase, WorkflowsSchemaError

pytestmark = [pytest.mark.integration, pytest.mark.postgres]


def snapshot(backend):
    """Keep schema metadata and the migration row in RAM for no-repair checks."""
    return {
        "columns": backend.execute(
            "SELECT table_name,column_name,data_type,is_nullable,column_default FROM information_schema.columns WHERE table_schema='public' ORDER BY table_name,ordinal_position"
        ).rows,
        "indexes": backend.execute(
            "SELECT tablename,indexname,indexdef FROM pg_indexes WHERE schemaname='public' ORDER BY tablename,indexname"
        ).rows,
        "versions": backend.execute("SELECT version FROM workflow_schema_version ORDER BY version").rows,
    }


@pytest.fixture
def migrated_workflows(pg_database_config, pg_restricted_backend):
    """Use official isolation, a real migration owner and a distinct DML-only role."""
    owner = DatabaseBackendFactory.create_backend(pg_database_config)
    role = sql.Identifier(pg_restricted_backend.config.pg_user)
    try:
        WorkflowsDatabase(backend=owner)
        with owner.transaction() as conn:
            conn.execute(sql.SQL("ALTER SCHEMA public OWNER TO {}").format(sql.Identifier(pg_database_config.pg_user)))
            conn.execute(sql.SQL("REVOKE CREATE ON SCHEMA public FROM {}").format(role))
            conn.execute(
                sql.SQL("REVOKE CREATE ON DATABASE {} FROM {}").format(
                    sql.Identifier(pg_database_config.pg_database), role
                )
            )
            conn.execute(sql.SQL("GRANT USAGE ON SCHEMA public TO {}").format(role))
            conn.execute(sql.SQL("GRANT SELECT,INSERT,UPDATE,DELETE ON ALL TABLES IN SCHEMA public TO {}").format(role))
            conn.execute(sql.SQL("GRANT USAGE,SELECT ON ALL SEQUENCES IN SCHEMA public TO {}").format(role))
            conn.execute(sql.SQL("REVOKE INSERT,UPDATE,DELETE ON workflow_schema_version FROM {}").format(role))
        flags = pg_restricted_backend.execute(
            "SELECT rolsuper,rolbypassrls,rolcreatedb,rolcreaterole,has_schema_privilege(current_user,'public','CREATE') AS can_create,EXISTS(SELECT 1 FROM pg_class WHERE relnamespace='public'::regnamespace AND pg_has_role(current_user,relowner,'USAGE')) AS owns FROM pg_roles WHERE rolname=current_user"
        ).first
        assert flags and not any(flags.values())  # nosec B101 -- actual ordinary serving boundary.
        yield owner, pg_restricted_backend
    finally:
        pg_restricted_backend.get_pool().close_all()
        owner.get_pool().close_all()


def test_owner_migrated_workflows_reopen_write_and_probe_without_ddl(migrated_workflows):
    """The real shared adapter supports readiness and tenant data without schema rights."""
    owner, runtime = migrated_workflows
    before = snapshot(owner)
    db = WorkflowsDatabase(backend=runtime)
    assert db.health_probe() == {"schema_version": 9, "expected_version": 9}  # nosec B101 -- actual read-only health.
    workflow_id = db.create_definition(
        tenant_id="x01-tenant",
        name="x01-owned",
        version=1,
        owner_id="341",
        visibility="private",
        description=None,
        tags=[],
        definition={"steps": []},
    )
    assert db.get_definition(workflow_id).owner_id == "341"  # nosec B101 -- real new-data round trip.
    assert not db.list_definitions(tenant_id="foreign", owner_id="341")  # nosec B101 -- tenant boundary remains.
    assert not db.list_definitions(tenant_id="x01-tenant", owner_id="foreign")  # nosec B101 -- owner boundary remains.
    assert snapshot(owner) == before  # nosec B101 -- no migration or metadata writes while serving.


@pytest.mark.parametrize(
    "drift",
    [
        "UPDATE workflow_schema_version SET version=8",
        "UPDATE workflow_schema_version SET version=10",
        "DROP TABLE workflow_webhook_dlq",
        "ALTER TABLE workflow_runs DROP COLUMN metadata_json",
        "DROP INDEX ux_events_run_seq",
        "DELETE FROM workflow_schema_version",
        "INSERT INTO workflow_schema_version(version) VALUES (9)",
        "ALTER TABLE workflow_schema_version ALTER COLUMN version TYPE TEXT USING version::text",
        "DROP INDEX ux_events_run_seq; CREATE UNIQUE INDEX ux_events_run_seq ON workflow_events(event_id)",
        "DROP INDEX ux_events_run_seq; CREATE UNIQUE INDEX ux_events_run_seq ON workflow_events(run_id,event_seq) WHERE event_seq > 0",
        "DROP INDEX ux_workflow_research_wait_run_step; CREATE UNIQUE INDEX ux_workflow_research_wait_run_step ON workflow_research_waits(lower(workflow_run_id),step_id)",
        "DROP INDEX ux_events_run_seq; ALTER TABLE workflow_events ADD CONSTRAINT ux_events_run_seq UNIQUE(run_id,event_seq) DEFERRABLE INITIALLY IMMEDIATE",
        "DROP INDEX ux_workflow_research_wait_run_step; ALTER TABLE workflow_research_waits ADD CONSTRAINT ux_workflow_research_wait_run_step UNIQUE(workflow_run_id,step_id) DEFERRABLE INITIALLY IMMEDIATE",
    ],
    ids=[
        "older-version",
        "newer-version",
        "table",
        "column",
        "unique-index",
        "missing-version",
        "duplicate-version",
        "text-version",
        "unique-wrong-keys",
        "unique-partial",
        "unique-expression",
        "events-deferrable",
        "research-wait-deferrable",
    ],
)
def test_nonowner_workflows_refuses_drift_without_repair(migrated_workflows, drift):
    """Restricted serving cannot upgrade, repair or accept an incompatible store."""
    owner, runtime = migrated_workflows
    owner.execute(drift)
    before = snapshot(owner)
    with pytest.raises(WorkflowsSchemaError):
        WorkflowsDatabase(backend=runtime)
    assert snapshot(owner) == before  # nosec B101 -- rejected schema remains unchanged.


@pytest.mark.parametrize("privilege", ["UPDATE", "UPDATE(version)", "INSERT(version)", "TRUNCATE"])
def test_nonowner_workflows_refuses_writable_version_metadata(migrated_workflows, privilege):
    """Data writers must not acquire the ability to certify migration completion."""
    owner, runtime = migrated_workflows
    with owner.transaction() as conn:
        conn.execute(
            sql.SQL("GRANT {} ON workflow_schema_version TO {}").format(
                sql.SQL(privilege), sql.Identifier(runtime.config.pg_user)
            )
        )
    before = snapshot(owner)
    with pytest.raises(WorkflowsSchemaError):
        WorkflowsDatabase(backend=runtime)
    assert snapshot(owner) == before  # nosec B101 -- unsafe metadata authority is not used.


def test_nonowner_workflows_research_wait_upsert_uses_canonical_arbiter(migrated_workflows):
    """The admitted restricted store can update the same real research wait."""
    owner, runtime = migrated_workflows
    before = snapshot(owner)
    db = WorkflowsDatabase(backend=runtime)
    db.create_run(
        run_id="x01-wait-run",
        tenant_id="x01-tenant",
        user_id="341",
        inputs={},
        workflow_id=None,
        definition_version=1,
        definition_snapshot={"steps": []},
    )
    for checkpoint in ("first", "second"):
        db.upsert_research_wait_link(
            wait_id="x01-wait",
            tenant_id="x01-tenant",
            workflow_run_id="x01-wait-run",
            step_id="research",
            research_run_id="research-1",
            checkpoint_id=checkpoint,
            checkpoint_type="sources_review",
            wait_status="waiting",
            wait_payload={},
            active_poll_seconds=1.0,
        )
    row = db.get_research_wait_link(workflow_run_id="x01-wait-run", step_id="research")
    assert row["checkpoint_id"] == "second"  # nosec B101 -- real ON CONFLICT update must work under runtime privileges.
    assert snapshot(owner) == before  # nosec B101 -- serving must not change schema/version metadata.
