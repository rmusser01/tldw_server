"""PostgreSQL workflow schema SQL, including forced, owner-only RLS."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from contextlib import contextmanager
from datetime import datetime
from typing import Any

import sqlglot
from sqlglot import exp

from tldw_Server_API.app.core.Jobs.operations.workflow_contracts import (
    WorkflowAdmissionCommand,
    WorkflowAdmissionInputs,
    WorkflowEffect,
    WorkflowEffectResult,
    WorkflowFenceError,
    WorkflowLease,
    WorkflowOutcome,
    WorkflowPolicy,
    WorkflowReceipt,
    WorkflowRoot,
    encode_workflow_receipt,
)

from .jobs_workflow_migrations import (
    RECEIPT_IDENTITY_COLUMNS,
    ROOT_IDENTITY_COLUMNS,
    enrollment_constraint,
    schema_definitions,
    table_statements,
)

_OWNER = "owner_user_id=NULLIF(current_setting('app.owner_user_id',true),'')"


def begin_runtime(connection: Any, owner_user_id: str) -> int:
    """Fail closed on role/setup errors, with transaction-local tenant scope."""
    from psycopg import IsolationLevel, sql

    # psycopg begins once on the first statement; an explicit BEGIN here
    # would issue a second BEGIN inside its implicit transaction.
    connection.isolation_level = IsolationLevel.READ_COMMITTED
    role = os.getenv("JOBS_PG_RLS_ROLE", "").strip()
    if role:
        connection.execute(sql.SQL("SET LOCAL ROLE {}").format(sql.Identifier(role)))
    flags = connection.execute("SELECT rolsuper,rolbypassrls FROM pg_roles WHERE rolname=current_user").fetchone()
    if flags is None or any(flags):
        raise WorkflowFenceError(WorkflowOutcome.UNAVAILABLE)
    connection.execute(
        "SELECT set_config('app.owner_user_id',%s,true),set_config('app.is_admin','false',true),set_config('app.domain_allowlist','',true),set_config('row_security','on',true)",
        (owner_user_id,),
    )
    actual = connection.execute(
        "SELECT current_setting('app.owner_user_id'),current_setting('app.is_admin'),current_setting('app.domain_allowlist'),current_setting('row_security')"
    ).fetchone()
    if actual != (owner_user_id, "false", "", "on"):
        raise WorkflowFenceError(WorkflowOutcome.UNAVAILABLE)
    return connection.execute("SELECT txid_current()").fetchone()[0]


def rows(connection: Any, statement: str, parameters: tuple = ()) -> list[dict]:
    """Only internal SQL is accepted; values always use driver parameters."""
    from psycopg.rows import dict_row

    with connection.cursor(row_factory=dict_row) as cursor:
        cursor.execute(statement.replace("?", "%s"), parameters)
        return cursor.fetchall() if cursor.description else []


def jobs_by_uuid(connection: Any, job_uuid: str, *, lock: bool = False) -> list[dict]:
    """Lock only UUID-selected Jobs rows, before capability metadata."""
    statement = (
        "SELECT * FROM jobs WHERE uuid=? LIMIT 2 FOR UPDATE" if lock else "SELECT * FROM jobs WHERE uuid=? LIMIT 2"
    )
    return rows(connection, statement, (job_uuid,))


def support_by_uuid(connection: Any, owner: str, job_uuid: str, *, lock: bool = False) -> list[dict]:
    """Read exact retained authority under forced owner RLS."""
    statement = (
        "SELECT * FROM job_workflow_roots WHERE owner_user_id=? AND job_uuid=? FOR UPDATE"
        if lock
        else "SELECT * FROM job_workflow_roots WHERE owner_user_id=? AND job_uuid=?"
    )
    return rows(connection, statement, (owner, job_uuid))


def lock_correlation(connection: Any, manager: Any, command: WorkflowAdmissionCommand, policy: WorkflowPolicy) -> None:
    """Serialize publication by a stable owner/scope/key transaction lock."""
    scope = policy.scope
    key = manager._pg_advisory_key(
        "workflow-admission-v1",
        command.owner_user_id,
        scope.domain,
        scope.queue,
        scope.job_type,
        command.idempotency_key,
    )
    connection.execute("SELECT pg_advisory_xact_lock(%s)", (key,))


def insert_root(connection: Any, root: WorkflowRoot, policy: WorkflowPolicy) -> None:
    """Publish root metadata and enrollment without a nested transaction."""
    rows(
        connection,
        "INSERT INTO job_workflow_roots(owner_user_id,job_uuid,job_id_hint,domain,queue,job_type,payload_field,reference_id,idempotency_key,max_retries,quarantine_threshold,expired_lease_policy,no_progress_limit) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            root.owner_user_id,
            root.job_uuid,
            root.job_id_hint,
            root.scope.domain,
            root.scope.queue,
            root.scope.job_type,
            policy.payload_field,
            root.reference_id,
            root.idempotency_key,
            root.controls.max_retries,
            root.controls.quarantine_threshold,
            root.controls.expired_lease_policy.value,
            root.controls.no_progress_limit,
        ),
    )
    connection.execute("UPDATE jobs SET workflow_capability_version=1 WHERE uuid=%s", (root.job_uuid,))


def create_root_job(
    connection: Any, inputs: WorkflowAdmissionInputs, uuid_value: str
) -> tuple[dict, dict] | bool | None:
    """Use the ordinary quota, insert, counter and event leaves atomically."""
    from psycopg.rows import dict_row

    from tldw_Server_API.app.core.Jobs.operations.postgres import admission

    command = inputs.job
    with connection.cursor(row_factory=dict_row) as cursor:
        if inputs.max_queued_quota or inputs.submits_per_minute_quota:
            cursor.execute("SELECT pg_advisory_xact_lock(%s)", (admission._quota_lock_key(command),))
        now = connection.execute("SELECT clock_timestamp()").fetchone()[0]
        rejection = admission._quota_rejection(
            cursor,
            command=command,
            now=now,
            max_queued_quota=inputs.max_queued_quota,
            submits_per_minute_quota=inputs.submits_per_minute_quota,
        )
        if rejection is not None:
            return None
        row = admission._insert_job(
            cursor,
            command=command,
            uuid_value=uuid_value,
            payload_json=json.dumps(command.payload),
            available_at=None,
            idempotent_insert=True,
        )
        if row is None:
            return False
        if inputs.counters_enabled:
            admission._bump_counters(cursor, command=command, available_at=None)
        event = admission._insert_created_event(
            cursor, row=row, idempotent=False, request_id=command.request_id, trace_id=command.trace_id
        )
        return row, event


def lease_is_live(connection: Any, job_uuid: str) -> bool:
    """clock_timestamp deliberately does not freeze at transaction start."""
    return bool(
        connection.execute(
            "SELECT 1 FROM jobs WHERE uuid=%s AND leased_until>clock_timestamp()", (job_uuid,)
        ).fetchone()
    )


def extend_horizon(connection: Any, job_uuid: str, minimum_seconds: int) -> datetime:
    """Extend an unexpired locked lease, rejecting update-boundary expiry."""
    row = connection.execute(
        "UPDATE jobs SET leased_until=GREATEST(leased_until,clock_timestamp()+%s*interval '1 second'),updated_at=clock_timestamp() WHERE uuid=%s AND leased_until>clock_timestamp() RETURNING leased_until",
        (minimum_seconds, job_uuid),
    ).fetchone()
    if row is None:
        raise WorkflowFenceError(WorkflowOutcome.STALE_LEASE)
    return row[0]


def database_now(connection: Any) -> datetime:
    """Use the current database clock, not frozen transaction time."""
    return connection.execute("SELECT clock_timestamp()").fetchone()[0]


def insert_effect_receipt(connection: Any, root: WorkflowRoot, receipt: WorkflowReceipt, slot: int) -> None:
    """Retain compact proof under forced transaction-local owner RLS."""
    connection.execute(
        "INSERT INTO job_workflow_effect_receipts(owner_user_id,job_uuid,effect_key,purpose,action,facts_fingerprint,facts,resulting_status,applied_at,slot) VALUES(%s,%s,%s,'action',%s,%s,%s,%s,%s,%s)",
        (
            root.owner_user_id,
            root.job_uuid,
            receipt.effect_key,
            receipt.action.value,
            receipt.facts_fingerprint,
            encode_workflow_receipt(receipt).decode("utf-8"),
            receipt.resulting_status,
            receipt.applied_at,
            slot,
        ),
    )


def transition_effect(
    connection: Any,
    root: WorkflowRoot,
    lease: WorkflowLease,
    effect: WorkflowEffect,
    receipt: WorkflowReceipt,
    row: dict,
    *,
    counters_enabled: bool,
    outbox_enabled: bool,
    expired: bool = False,
) -> dict:
    """Apply on the caller's locked connection without a committing wrapper."""
    from psycopg.rows import dict_row

    from tldw_Server_API.app.core.Jobs.operations.postgres.lifecycle import (
        _insert_prepared_event,
        _prepared_counter_transition,
    )

    from .jobs_workflow_store import effect_event_type, effect_transition_parameters

    expiry_guard = "leased_until<=clock_timestamp()" if expired else "leased_until>clock_timestamp()"
    parameters = effect_transition_parameters(root, lease, effect, receipt, row)
    if expired and receipt.resulting_status == "queued":
        parameters = parameters[:8] + (None,) + parameters[9:]
    changed = connection.execute(
        "UPDATE jobs SET status=%s,result=%s::jsonb,retry_count=%s,failure_streak_code=%s,failure_streak_count=%s,error_code=%s,error_message=%s,last_error=%s,available_at=%s,quarantined_at=%s,completed_at=%s,cancelled_at=%s,completion_token=%s,acquired_at=%s,started_at=%s,leased_until=NULL,worker_id=NULL,lease_id=NULL,updated_at=clock_timestamp() WHERE uuid=%s AND owner_user_id=%s AND status='processing' AND worker_id=%s AND lease_id=%s AND (completion_token IS NULL OR completion_token=%s) AND "
        + expiry_guard,  # nosec B608
        parameters,
    )
    if changed.rowcount != 1:
        raise WorkflowFenceError(WorkflowOutcome.STALE_LEASE)
    with connection.cursor(row_factory=dict_row) as cursor:
        if counters_enabled:
            _prepared_counter_transition(
                cursor, row=row, new_status=receipt.resulting_status, queued_scheduled=not expired
            )
        updated = jobs_by_uuid(connection, root.job_uuid)[0]
        if outbox_enabled:
            _insert_prepared_event(
                cursor,
                row=updated,
                event_type=effect_event_type(receipt),
                marker={"kind": effect.action.value, "origin": "workflow"},
                reason_code=effect.reason_code,
            )
    return updated


def enrollment_projection(connection: Any) -> str:
    """Project one bounded marker without assuming explicit schema installation."""
    return (
        "workflow_capability_version"
        if "workflow_capability_version" in _columns(connection, "jobs")
        else "NULL AS workflow_capability_version"
    )


@contextmanager
def _maintenance_owner_scope(connection: Any, owner: str):
    """Explicit non-bypass owner scope, restored on success and by rollback on SQL error."""
    from psycopg import sql
    from psycopg.pq import TransactionStatus

    from tldw_Server_API.app.core.Jobs.operations.workflow_contracts import _text

    _text(owner, 512)
    old = connection.execute(
        "SELECT current_setting('role'),current_setting('app.owner_user_id',true),current_setting('app.is_admin',true),current_setting('app.domain_allowlist',true),current_setting('row_security')"
    ).fetchone()
    try:
        role = os.getenv("JOBS_PG_RLS_ROLE", "").strip()
        if role:
            connection.execute(sql.SQL("SET LOCAL ROLE {}").format(sql.Identifier(role)))
        flags = connection.execute("SELECT rolsuper,rolbypassrls FROM pg_roles WHERE rolname=current_user").fetchone()
        if flags is None or any(flags):
            raise WorkflowFenceError(WorkflowOutcome.UNAVAILABLE)
        connection.execute(
            "SELECT set_config('app.owner_user_id',%s,true),set_config('app.is_admin','false',true),set_config('app.domain_allowlist','',true),set_config('row_security','on',true)",
            (owner,),
        )
        actual = connection.execute(
            "SELECT current_setting('app.owner_user_id'),current_setting('app.is_admin'),current_setting('app.domain_allowlist'),current_setting('row_security')"
        ).fetchone()
        if actual != (owner, "false", "", "on"):
            raise WorkflowFenceError(WorkflowOutcome.UNAVAILABLE)
        yield
    finally:
        if connection.info.transaction_status is TransactionStatus.INTRANS:
            connection.execute(
                "SELECT set_config('app.owner_user_id',%s,true),set_config('app.is_admin',%s,true),set_config('app.domain_allowlist',%s,true),set_config('row_security',%s,true),set_config('role',%s,true)",
                (*(value or "" for value in old[1:4]), old[4], old[0]),
            )


def lease_is_expired(connection: Any, job_uuid: str) -> bool:
    """Use a non-frozen database clock and the exact concrete old deadline."""
    return bool(
        connection.execute(
            "SELECT 1 FROM jobs WHERE uuid=%s AND leased_until<=clock_timestamp()", (job_uuid,)
        ).fetchone()
    )


def record_workflow_acquisition_in_transaction(
    connection: Any, *, backend: str, acquired: Mapping[str, object]
) -> None:
    """Publish epoch/digest atomically under explicit runtime authority."""
    if acquired.get("workflow_capability_version") is None:
        return
    from psycopg import Error

    from .jobs_workflow_store import _record_workflow_acquisition

    try:
        with _maintenance_owner_scope(connection, acquired.get("owner_user_id")):
            _record_workflow_acquisition(connection, backend=backend, acquired=acquired)
    except (Error, ValueError, TypeError, KeyError):
        raise WorkflowFenceError(WorkflowOutcome.UNAVAILABLE) from None


def recover_workflow_expiry_in_transaction(
    connection: Any, *, backend: str, locked_root: Mapping[str, object]
) -> WorkflowEffectResult:
    """Lock ownership is external; support access never uses an admin bypass."""
    if locked_root.get("workflow_capability_version") is None:
        return WorkflowEffectResult(WorkflowOutcome.MISSING)
    from psycopg import Error

    from .jobs_workflow_store import _recover_workflow_expiry

    try:
        with _maintenance_owner_scope(connection, locked_root.get("owner_user_id")):
            return _recover_workflow_expiry(connection, backend=backend, locked_root=locked_root)
    except (Error, ValueError, TypeError, KeyError):
        raise WorkflowFenceError(WorkflowOutcome.UNAVAILABLE) from None


def _normalized(definition: str) -> exp.Expression:
    """Compare parsed structure without discarding grouping or literal bytes."""
    if definition.startswith("CHECK "):
        definition = definition[6:]
    elif definition.startswith(("PRIMARY KEY", "UNIQUE", "FOREIGN KEY")):
        definition = f"CREATE TABLE workflow_shape ({definition})"
    expression = sqlglot.parse_one(definition, read="postgres")

    def canonical(node: exp.Expression) -> exp.Expression:
        if isinstance(node, exp.Paren):
            return node.unnest()
        if (
            isinstance(node, exp.Cast)
            and isinstance(node.this, exp.Literal)
            and node.to.this in {exp.DataType.Type.TEXT, exp.DataType.Type.BIGINT}
        ):
            return node.this
        if isinstance(node, exp.Between):
            return exp.And(
                this=exp.GTE(this=node.this.copy(), expression=node.args["low"]),
                expression=exp.LTE(this=node.this.copy(), expression=node.args["high"]),
            )
        if isinstance(node, exp.In) and not node.args.get("query"):
            return exp.EQ(this=node.this, expression=exp.Any(this=exp.Array(expressions=node.expressions)))
        if isinstance(node, exp.Trim) and not node.args.get("position"):
            node.set("position", "BOTH")
        return node

    # Bottom-up traversal also visits children of removed wrapper nodes.
    for node in reversed(list(expression.walk())):
        replacement = canonical(node)
        if node is expression:
            expression = replacement
        elif replacement is not node:
            node.replace(replacement)
    return expression


def _trigger_body(suffix: str) -> str:
    columns = ROOT_IDENTITY_COLUMNS if suffix == "root" else RECEIPT_IDENTITY_COLUMNS
    new = ",".join(f"NEW.{column}" for column in columns)
    old = ",".join(f"OLD.{column}" for column in columns)
    predicate = f"ROW({new}) IS DISTINCT FROM ROW({old})"
    if suffix == "receipt":
        predicate += " OR NEW.released<OLD.released"
    return f"BEGIN IF {predicate} THEN RAISE EXCEPTION 'immutable workflow facts'; END IF; RETURN NEW; END"


def begin_install(connection: Any) -> None:
    """Start the facade-owned transaction; leaf migrations never do this."""
    connection.execute("BEGIN")


def _columns(connection: Any, table: str) -> dict[str, tuple]:
    return {
        row[0]: tuple(row[1:])
        for row in connection.execute(
            "SELECT a.attname,format_type(a.atttypid,a.atttypmod),a.attnotnull, "
            "pg_get_expr(d.adbin,d.adrelid) FROM pg_attribute a "
            "JOIN pg_class t ON t.oid=a.attrelid JOIN pg_namespace n ON n.oid=t.relnamespace "
            "LEFT JOIN pg_attrdef d ON d.adrelid=t.oid AND d.adnum=a.attnum "
            "WHERE n.nspname=current_schema() AND t.relname=%s AND a.attnum>0 AND NOT a.attisdropped",
            (table,),
        )
    }


def _constraints(connection: Any, table: str) -> dict[str, exp.Expression] | None:
    rows = connection.execute(
        "SELECT c.conname,pg_get_constraintdef(c.oid),c.convalidated,c.condeferrable "
        "FROM pg_constraint c JOIN pg_class t ON t.oid=c.conrelid "
        "JOIN pg_namespace n ON n.oid=t.relnamespace "
        "WHERE n.nspname=current_schema() AND t.relname=%s "
        "AND (t.relname<>'jobs' OR c.conname='wf_jobs_enrollment')",
        (table,),
    ).fetchall()
    if any(not row[2] or row[3] for row in rows):
        return None
    # PG 18 adds catalog NOT NULL constraints; attnotnull is checked separately.
    return {row[0]: _normalized(row[1]) for row in rows if not row[1].startswith("NOT NULL")}


def inspect_support(connection: Any) -> WorkflowOutcome:
    """Read all required structural/security invariants without DDL or repair."""
    tables = {
        row[0]: tuple(row[1:])
        for row in connection.execute(
            "SELECT t.relname,t.relrowsecurity,t.relforcerowsecurity FROM pg_class t "
            "JOIN pg_namespace n ON n.oid=t.relnamespace WHERE n.nspname=current_schema() "
            "AND t.relkind='r'"
        )
    }
    definitions = schema_definitions("postgres")
    if not definitions.keys() <= tables.keys():
        return WorkflowOutcome.UNAVAILABLE
    for table, (columns, constraints) in definitions.items():
        actual = _columns(connection, table)
        expected = {}
        for column, declaration in columns.items():
            kind = "bigint" if declaration.startswith("BIGINT") else "text"
            if declaration.startswith("TIMESTAMP"):
                kind = "timestamp with time zone"
            expected[column] = (kind, "NOT NULL" in declaration, "0" if "DEFAULT 0" in declaration else None)
        if actual != expected or _constraints(connection, table) != {
            name: _normalized(definition) for name, definition in constraints.items()
        }:
            return WorkflowOutcome.UNAVAILABLE
    if _columns(connection, "jobs").get("workflow_capability_version") != ("bigint", False, None):
        return WorkflowOutcome.UNAVAILABLE
    constraints = _constraints(connection, "jobs")
    if constraints is None or constraints.get("wf_jobs_enrollment") != _normalized(enrollment_constraint("postgres")):
        return WorkflowOutcome.UNAVAILABLE
    for suffix, table in (("root", "job_workflow_roots"), ("receipt", "job_workflow_effect_receipts")):
        if tables[table] != (True, True):
            return WorkflowOutcome.UNAVAILABLE
        policies = connection.execute(
            "SELECT p.polname,p.polcmd,p.polpermissive,p.polroles, "
            "pg_get_expr(p.polqual,p.polrelid),pg_get_expr(p.polwithcheck,p.polrelid) "
            "FROM pg_policy p JOIN pg_class t ON t.oid=p.polrelid "
            "JOIN pg_namespace n ON n.oid=t.relnamespace "
            "WHERE n.nspname=current_schema() AND t.relname=%s",
            (table,),
        ).fetchall()
        if len(policies) != 1:
            return WorkflowOutcome.UNAVAILABLE
        policy = policies[0]
        if (
            tuple(policy[:4]) != (f"wf_{suffix}_owner", "*", True, [0])
            or _normalized(policy[4]) != _normalized(_OWNER)
            or _normalized(policy[5]) != _normalized(_OWNER)
        ):
            return WorkflowOutcome.UNAVAILABLE
        triggers = connection.execute(
            "SELECT g.tgname,g.tgenabled,g.tgtype,p.prosrc,p.prosecdef,l.lanname "
            "FROM pg_trigger g JOIN pg_class t ON t.oid=g.tgrelid "
            "JOIN pg_namespace n ON n.oid=t.relnamespace "
            "JOIN pg_proc p ON p.oid=g.tgfoid JOIN pg_language l ON l.oid=p.prolang "
            "WHERE n.nspname=current_schema() AND t.relname=%s AND NOT g.tgisinternal",
            (table,),
        ).fetchall()
        if [tuple(row) for row in triggers] != [
            (f"wf_immutable_{suffix}", "O", 19, _trigger_body(suffix), False, "plpgsql")
        ]:
            return WorkflowOutcome.UNAVAILABLE
    version = connection.execute("SELECT singleton,version FROM job_workflow_schema").fetchall()
    return WorkflowOutcome.APPLIED if version == [(1, 1)] else WorkflowOutcome.UNAVAILABLE


def install_support(connection: Any) -> None:
    """Create complete owner-only support, or reject an unsupported shape."""
    from psycopg import sql

    connection.execute("SELECT pg_advisory_xact_lock(hashtextextended('jobs.workflow.schema.v1',0))")
    if inspect_support(connection) is WorkflowOutcome.APPLIED:
        return
    tables = {row[0] for row in connection.execute("SELECT tablename FROM pg_tables WHERE schemaname=current_schema()")}
    if tables & table_statements("postgres").keys() or "workflow_capability_version" in _columns(connection, "jobs"):
        raise ValueError("workflow support schema is partial or unsupported")
    if "jobs" not in tables:
        raise ValueError("ordinary Jobs schema is required")
    for statement in table_statements("postgres").values():
        connection.execute(statement)
    connection.execute(
        "ALTER TABLE jobs ADD COLUMN workflow_capability_version BIGINT "
        "CONSTRAINT wf_jobs_enrollment " + enrollment_constraint("postgres")
    )
    for suffix, table in (("root", "job_workflow_roots"), ("receipt", "job_workflow_effect_receipts")):
        connection.execute(sql.SQL("ALTER TABLE {} ENABLE ROW LEVEL SECURITY").format(sql.Identifier(table)))
        connection.execute(sql.SQL("ALTER TABLE {} FORCE ROW LEVEL SECURITY").format(sql.Identifier(table)))
        connection.execute(
            sql.SQL("CREATE POLICY {} ON {} USING (" + _OWNER + ") WITH CHECK (" + _OWNER + ")").format(
                sql.Identifier(f"wf_{suffix}_owner"), sql.Identifier(table)
            )
        )
        connection.execute(
            sql.SQL("CREATE FUNCTION {}() RETURNS trigger LANGUAGE plpgsql AS {}").format(
                sql.Identifier(f"wf_immutable_{suffix}"), sql.Literal(_trigger_body(suffix))
            )
        )
        connection.execute(
            sql.SQL("CREATE TRIGGER {} BEFORE UPDATE ON {} FOR EACH ROW EXECUTE FUNCTION {}()").format(
                sql.Identifier(f"wf_immutable_{suffix}"),
                sql.Identifier(table),
                sql.Identifier(f"wf_immutable_{suffix}"),
            )
        )
    connection.execute("INSERT INTO job_workflow_schema(singleton,version) VALUES(1,1)")
    if inspect_support(connection) is not WorkflowOutcome.APPLIED:
        raise ValueError("workflow support installation verification failed")
