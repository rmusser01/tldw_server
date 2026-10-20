"""Explicit workflow migrations are atomic, bounded, inert, and fail closed."""

import sqlite3

import psycopg
import pytest

from tldw_Server_API.app.core.DB_Management.jobs_workflow_migrations import (
    inspect_workflow_support_in_transaction,
    install_workflow_support_in_transaction,
)
from tldw_Server_API.app.core.Jobs.operations.workflow_contracts import WorkflowOutcome
from tldw_Server_API.app.core.Jobs.workflow_capability import inspect_workflow_support
from tldw_Server_API.tests.Jobs.workflow_test_support import ROOT_REFERENCE, ROOT_UUID

pytestmark = pytest.mark.integration
DATABASE_ERRORS = (sqlite3.IntegrityError, psycopg.IntegrityError, psycopg.errors.RaiseException)


def test_support_requires_explicit_install(uninstalled_workflow_case):
    case = uninstalled_workflow_case
    assert inspect_workflow_support(case.manager) is WorkflowOutcome.UNAVAILABLE
    assert "job_workflow_roots" not in case.table_names()
    case.install()
    case.install()
    assert case.support_version() == 1
    assert inspect_workflow_support(case.manager) is WorkflowOutcome.APPLIED


def test_migration_rolls_back_partial_install(uninstalled_workflow_case):
    case = uninstalled_workflow_case
    with case.connection() as conn:
        if case.manager.backend == "sqlite":
            conn.execute("BEGIN IMMEDIATE")
        install_workflow_support_in_transaction(conn, backend=case.manager.backend)
        assert inspect_workflow_support_in_transaction(conn, backend=case.manager.backend) is WorkflowOutcome.APPLIED
        conn.rollback()
    assert not {"job_workflow_schema", "job_workflow_roots", "job_workflow_effect_receipts"} & case.table_names()
    assert inspect_workflow_support(case.manager) is WorkflowOutcome.UNAVAILABLE
    with case.connection() as conn:
        if case.manager.backend == "sqlite":
            columns = [row[1] for row in conn.execute("PRAGMA table_info(jobs)")]
        else:
            columns = [
                row[0]
                for row in conn.execute(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_schema=current_schema() AND table_name='jobs'"
                )
            ]
        assert "workflow_capability_version" not in columns


def test_missing_constraint_is_unavailable_not_repaired(workflow_case):
    case = workflow_case

    def receipt_shape():
        with case.connection() as conn:
            if case.manager.backend == "sqlite":
                rows = conn.execute(
                    "SELECT type,name,sql FROM sqlite_schema "
                    "WHERE tbl_name='job_workflow_effect_receipts' ORDER BY type,name"
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT conname,pg_get_constraintdef(oid) FROM pg_constraint "
                    "WHERE conrelid='job_workflow_effect_receipts'::regclass ORDER BY conname"
                ).fetchall()
            return [tuple(row) for row in rows]

    with case.connection() as conn:
        if case.manager.backend == "sqlite":
            conn.execute("DROP TABLE job_workflow_effect_receipts")
            conn.execute("CREATE TABLE job_workflow_effect_receipts (effect_key TEXT)")
        else:
            conn.execute("ALTER TABLE job_workflow_effect_receipts DROP CONSTRAINT wf_receipt_slot")
        conn.commit()
    before = case.table_names()
    malformed_shape = receipt_shape()
    assert inspect_workflow_support(case.manager) is WorkflowOutcome.UNAVAILABLE
    assert case.table_names() == before
    with pytest.raises(ValueError):
        case.install()
    assert inspect_workflow_support(case.manager) is WorkflowOutcome.UNAVAILABLE
    assert receipt_shape() == malformed_shape


@pytest.mark.parametrize(
    "changes",
    [
        {"owner_user_id": "\u00e9" * 257},
        {"idempotency_key": "\u00e9" * 101},
        {"max_retries": -1},
        {"quarantine_threshold": 0},
        {"no_progress_limit": 6},
        {"domain": "bad/code"},
        {"expired_lease_policy": "consume_retry"},
        {"job_uuid": "not-a-uuid"},
        {"reference_id": "not-a-uuid"},
    ],
)
def test_schema_rejects_invalid_root_facts(workflow_case, changes):
    case = workflow_case
    with case.connection() as conn:
        with pytest.raises(DATABASE_ERRORS):
            case.insert_root(conn, **changes)


@pytest.mark.parametrize(
    "column,value",
    [
        ("domain", "embeddings"),
        ("queue", "workflow"),
        ("job_type", "embedding_workflow"),
        ("payload_field", "workflow_id"),
        ("job_uuid", ROOT_UUID),
        ("reference_id", ROOT_REFERENCE),
        ("lease_digest", "a" * 64),
    ],
)
@pytest.mark.parametrize("suffix", ["\x00suffix", "\x00/" + "\u00e9" * 64], ids=["nul", "nul-oversize"])
def test_root_shape_rejects_embedded_nul(workflow_case, column, value, suffix):
    case = workflow_case
    assert inspect_workflow_support(case.manager) is WorkflowOutcome.APPLIED
    errors = (sqlite3.IntegrityError,) if case.manager.backend == "sqlite" else (ValueError, psycopg.DataError)
    match = "CHECK constraint failed" if case.manager.backend == "sqlite" else "NUL|0x00"
    with case.connection() as conn:
        with pytest.raises(errors, match=match):
            case.insert_root(conn, **{column: value + suffix})


@pytest.mark.parametrize("column", ["domain", "queue", "job_type", "payload_field"])
def test_root_fixed_codes_accept_64_reject_65_bytes(workflow_case, column):
    case = workflow_case
    with case.connection() as conn:
        case.insert_root(conn, **{column: "a" * 64})
    with case.connection() as conn:
        with pytest.raises(DATABASE_ERRORS):
            case.insert_root(conn, **{column: "a" * 65})


@pytest.mark.parametrize(
    "column,value", [("job_uuid", ROOT_UUID), ("effect_key", "a" * 64), ("facts_fingerprint", "b" * 64)]
)
@pytest.mark.parametrize("suffix", ["\x00suffix", "\x00/" + "\u00e9" * 64], ids=["nul", "nul-oversize"])
def test_receipt_shape_rejects_embedded_nul(workflow_case, column, value, suffix):
    case = workflow_case
    invalid = value + suffix
    assert inspect_workflow_support(case.manager) is WorkflowOutcome.APPLIED
    errors = (sqlite3.IntegrityError,) if case.manager.backend == "sqlite" else (ValueError, psycopg.DataError)
    match = "CHECK constraint failed" if case.manager.backend == "sqlite" else "NUL|0x00"
    with case.connection() as conn:
        if column == "job_uuid" and case.manager.backend == "sqlite":
            # Isolate the receipt UUID CHECK from its FK to the same malformed parent.
            conn.execute("PRAGMA ignore_check_constraints=ON")
            try:
                case.insert_root(conn, job_uuid=invalid)
            finally:
                conn.execute("PRAGMA ignore_check_constraints=OFF")
            assert conn.execute("PRAGMA ignore_check_constraints").fetchone()[0] == 0
        else:
            case.insert_root(conn)
        with pytest.raises(errors, match=match):
            case.insert(conn, "job_workflow_effect_receipts", case.receipt_values(**{column: invalid}))


def test_roots_are_not_tied_to_active_jobs_and_snapshot_is_immutable(workflow_case):
    case = workflow_case
    with case.connection() as conn:
        case.insert_root(conn)
        conn.commit()
    with case.connection() as conn:
        with pytest.raises(DATABASE_ERRORS):
            conn.execute("UPDATE job_workflow_roots SET max_retries=4")
    with case.connection() as conn:
        assert conn.execute("SELECT max_retries FROM job_workflow_roots").fetchone()[0] == 3


@pytest.mark.parametrize(
    "changes",
    [
        {"slot": 0},
        {"slot": 32769},
        {"purpose": "unknown"},
        {"action": "unknown"},
        {"facts": '"' + "\u00e9" * 2048 + '"'},
        {"effect_key": "A" * 64},
        {"owner_user_id": "owner-b"},
        {"released": 2},
    ],
)
def test_receipt_constraints_bound_facts_namespace_and_owner(workflow_case, changes):
    case = workflow_case
    with case.connection() as conn:
        case.insert_root(conn)
        conn.commit()
    with case.connection() as conn:
        with pytest.raises(DATABASE_ERRORS):
            case.insert(conn, "job_workflow_effect_receipts", case.receipt_values(**changes))


def test_progress_and_action_share_keys_and_released_slots(workflow_case):
    case = workflow_case
    with case.connection() as conn:
        case.insert_root(conn)
        case.insert(conn, "job_workflow_effect_receipts", case.receipt_values(slot=32768, released=1))
        conn.commit()
    with case.connection() as conn:
        with pytest.raises(DATABASE_ERRORS):
            case.insert(
                conn,
                "job_workflow_effect_receipts",
                case.receipt_values(
                    purpose="progress", action=None, progress_kind="cursor", resulting_status=None, slot=32768
                ),
            )
    with case.connection() as conn:
        with pytest.raises(DATABASE_ERRORS):
            case.insert(conn, "job_workflow_effect_receipts", case.receipt_values(effect_key="c" * 64, slot=32768))


@pytest.mark.pg_jobs
def test_runtime_role_has_forced_owner_rls_without_admin_bypass(workflow_pg_case):
    case = workflow_pg_case
    with case.connection() as conn:
        case.insert_root(conn)
        case.insert(conn, "job_workflow_effect_receipts", case.receipt_values())
        conn.commit()
    with case.connection(runtime=True) as conn:
        assert conn.execute("SELECT rolsuper,rolbypassrls FROM pg_roles WHERE rolname=current_user").fetchone() == (
            False,
            False,
        )
        conn.execute("SELECT set_config('app.is_admin','true',true)")
        conn.execute("SELECT set_config('app.owner_user_id','owner-b',true)")
        assert conn.execute("SELECT COUNT(*) FROM job_workflow_roots").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM job_workflow_effect_receipts").fetchone()[0] == 0
        conn.execute("SELECT set_config('app.owner_user_id','owner-a',true)")
        assert conn.execute("SELECT COUNT(*) FROM job_workflow_roots").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM job_workflow_effect_receipts").fetchone()[0] == 1
        conn.execute("SELECT set_config('app.owner_user_id','',true)")
        assert conn.execute("SELECT COUNT(*) FROM job_workflow_roots").fetchone()[0] == 0


def test_install_does_not_enroll_legacy_rows(uninstalled_workflow_case):
    case = uninstalled_workflow_case
    with case.connection() as conn:
        conn.execute(
            "INSERT INTO jobs(domain,queue,job_type,payload,status) VALUES('embeddings','default','legacy','{}','queued')"
        )
        conn.commit()
    case.install()
    with case.connection() as conn:
        assert conn.execute("SELECT workflow_capability_version FROM jobs").fetchone()[0] is None
        assert conn.execute("SELECT COUNT(*) FROM job_workflow_roots").fetchone()[0] == 0


def test_failed_partial_install_has_no_tables_or_version(uninstalled_workflow_case):
    case = uninstalled_workflow_case

    class InterruptedConnection:
        def __init__(self, connection):
            self.connection = connection

        def execute(self, statement, *args):
            if isinstance(statement, str) and statement.startswith("CREATE TABLE job_workflow_effect_receipts"):
                raise RuntimeError("injected schema interruption")
            return self.connection.execute(statement, *args)

    with case.connection() as conn:
        if case.manager.backend == "sqlite":
            conn.execute("BEGIN IMMEDIATE")
        with pytest.raises(RuntimeError, match="injected schema interruption"):
            install_workflow_support_in_transaction(InterruptedConnection(conn), backend=case.manager.backend)
        conn.rollback()
    assert not {"job_workflow_schema", "job_workflow_roots", "job_workflow_effect_receipts"} & case.table_names()
    assert inspect_workflow_support(case.manager) is WorkflowOutcome.UNAVAILABLE


@pytest.mark.pg_jobs
def test_changed_constraint_literal_is_unavailable(workflow_pg_case):
    case = workflow_pg_case
    with case.connection() as conn:
        conn.execute("ALTER TABLE job_workflow_roots DROP CONSTRAINT wf_root_reserved")
        conn.execute(
            "ALTER TABLE job_workflow_roots ADD CONSTRAINT wf_root_reserved "
            "CHECK (NOT (domain='admin_ webhooks' AND queue='delivery'))"
        )
        conn.commit()
    assert inspect_workflow_support(case.manager) is WorkflowOutcome.UNAVAILABLE


@pytest.mark.pg_jobs
def test_extra_admin_rls_policy_is_unavailable(workflow_pg_case):
    case = workflow_pg_case
    with case.connection() as conn:
        conn.execute("CREATE POLICY bypass ON job_workflow_roots USING (true) WITH CHECK (true)")
        conn.commit()
    assert inspect_workflow_support(case.manager) is WorkflowOutcome.UNAVAILABLE


@pytest.mark.pg_jobs
def test_runtime_cannot_insert_or_cleanup_another_owner(workflow_pg_case):
    case = workflow_pg_case
    with case.connection() as conn:
        case.insert_root(conn)
        case.insert(conn, "job_workflow_effect_receipts", case.receipt_values())
        conn.commit()
    with case.connection(runtime=True) as conn:
        conn.execute("SELECT set_config('app.owner_user_id','owner-b',true)")
        conn.execute("SELECT set_config('app.is_admin','true',true)")
        assert conn.execute("DELETE FROM job_workflow_effect_receipts").rowcount == 0
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            case.insert_root(conn, idempotency_key="another-key")


def test_ordinary_constructor_preserves_support_without_enrollment(workflow_case):
    from tldw_Server_API.app.core.Jobs.manager import JobManager

    case = workflow_case
    # Runtime PostgreSQL construction is explicitly schema-dormant, like workers.
    manager = JobManager(backend=case.manager.backend, db_path=case.manager.db_path, db_url=case.manager.db_url)
    assert inspect_workflow_support(manager) is WorkflowOutcome.APPLIED
    with case.connection() as conn:
        assert conn.execute("SELECT COUNT(*) FROM job_workflow_roots").fetchone()[0] == 0


@pytest.mark.parametrize(
    "changes",
    [
        {"job_uuid": "00000000-0000-4000-8000-000000000003"},
        {"owner_user_id": "owner-b", "idempotency_key": "other"},
    ],
)
def test_admission_identity_and_root_uuid_are_unique(workflow_case, changes):
    case = workflow_case
    with case.connection() as conn:
        case.insert_root(conn)
        conn.commit()
    with case.connection() as conn:
        with pytest.raises(DATABASE_ERRORS):
            case.insert_root(conn, **changes)


def test_owner_scoped_admission_and_separate_proof_capacity(workflow_case):
    case = workflow_case
    with case.connection() as conn:
        case.insert_root(conn)
        case.insert_root(conn, owner_user_id="owner-b", job_uuid="00000000-0000-4000-8000-000000000003")
        case.insert(conn, "job_workflow_effect_receipts", case.receipt_values(slot=32768))
        case.insert(
            conn,
            "job_workflow_effect_receipts",
            case.receipt_values(
                effect_key="c" * 64,
                purpose="progress",
                action=None,
                progress_kind="cursor",
                resulting_status=None,
                slot=32768,
            ),
        )
        assert conn.execute("SELECT COUNT(*) FROM job_workflow_roots").fetchone()[0] == 2
        assert conn.execute("SELECT COUNT(*) FROM job_workflow_effect_receipts").fetchone()[0] == 2


def test_receipt_limits_use_encoded_bytes(workflow_case):
    case = workflow_case
    facts = '"' + "\u00e9" * 2047 + '"'
    assert len(facts.encode("utf-8")) == 4096
    with case.connection() as conn:
        case.insert_root(conn)
        case.insert(conn, "job_workflow_effect_receipts", case.receipt_values(facts=facts))
        conn.commit()
    with case.connection() as conn:
        with pytest.raises(DATABASE_ERRORS):
            case.insert(
                conn,
                "job_workflow_effect_receipts",
                case.receipt_values(facts=facts[:-1] + 'a"', slot=2, effect_key="c" * 64),
            )


def test_released_receipts_retain_immutable_slot_and_fingerprint(workflow_case):
    case = workflow_case
    with case.connection() as conn:
        case.insert_root(conn)
        case.insert(conn, "job_workflow_effect_receipts", case.receipt_values())
        conn.execute("UPDATE job_workflow_effect_receipts SET released=1")
        conn.commit()
    with case.connection() as conn:
        with pytest.raises(DATABASE_ERRORS):
            if case.manager.backend == "postgres":
                conn.execute("UPDATE job_workflow_effect_receipts SET facts_fingerprint=%s", ("c" * 64,))
            else:
                conn.execute("UPDATE job_workflow_effect_receipts SET facts_fingerprint=?", ("c" * 64,))
    with case.connection() as conn:
        assert tuple(
            conn.execute("SELECT slot,released,facts_fingerprint FROM job_workflow_effect_receipts").fetchone()
        ) == (1, 1, "b" * 64)
