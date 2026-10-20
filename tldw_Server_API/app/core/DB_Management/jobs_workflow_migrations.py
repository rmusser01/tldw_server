"""Explicit, transaction-neutral version-one workflow schema definitions.

Only the caller owns transactions. No legacy Jobs initializer imports this
module or installs these tables. Runtime inspection never repairs a schema.
"""

from __future__ import annotations

from typing import Any

from tldw_Server_API.app.core.Jobs.operations.workflow_contracts import WorkflowOutcome

ROOT_IDENTITY_COLUMNS = (
    "owner_user_id",
    "job_uuid",
    "job_id_hint",
    "domain",
    "queue",
    "job_type",
    "payload_field",
    "reference_id",
    "idempotency_key",
    "max_retries",
    "quarantine_threshold",
    "expired_lease_policy",
    "no_progress_limit",
)
RECEIPT_IDENTITY_COLUMNS = (
    "owner_user_id",
    "job_uuid",
    "effect_key",
    "purpose",
    "action",
    "progress_kind",
    "facts_fingerprint",
    "facts",
    "resulting_status",
    "applied_at",
    "slot",
)


def schema_definitions(backend: str) -> dict[str, tuple[dict[str, str], dict[str, str]]]:
    """Return closed SQL declarations used by the two DB_Management leaves."""
    pg = backend == "postgres"
    integer = "BIGINT" if pg else "INTEGER"

    def bounded(column: str, maximum: int) -> str:
        size = f"octet_length({column})" if pg else f"length(CAST({column} AS BLOB))"
        typed = "" if pg else f"typeof({column})='text' AND "
        return f"{typed}{size} BETWEEN 1 AND {maximum} AND trim({column})<>''"

    def code(column: str) -> str:
        if pg:
            return f"{column} COLLATE \"C\" ~ '^[a-z][a-z0-9_.-]{{0,63}}$'"
        return f"length({column}) BETWEEN 1 AND 64 AND {column} GLOB '[a-z]*' AND {column} NOT GLOB '*[^a-z0-9_.-]*'"

    def uuid(column: str) -> str:
        if pg:
            return f"{column} COLLATE \"C\" ~ '^[0-9a-f]{{8}}-[0-9a-f]{{4}}-4[0-9a-f]{{3}}-[89ab][0-9a-f]{{3}}-[0-9a-f]{{12}}$'"
        return (
            f"length({column})=36 AND substr({column},9,1)='-' "
            f"AND substr({column},14,1)='-' AND substr({column},19,1)='-' "
            f"AND substr({column},24,1)='-' AND substr({column},15,1)='4' "
            f"AND substr({column},20,1) IN ('8','9','a','b') "
            f"AND length(replace({column},'-',''))=32 "
            f"AND replace({column},'-','') NOT GLOB '*[^0-9a-f]*'"
        )

    def digest(column: str) -> str:
        if pg:
            return f"{column} COLLATE \"C\" ~ '^[0-9a-f]{{64}}$'"
        return f"length({column})=64 AND {column} NOT GLOB '*[^0-9a-f]*'"

    def number(column: str, expression: str) -> str:
        return expression if pg else f"typeof({column})='integer' AND {expression}"

    def choice(column: str, values: tuple[str, ...]) -> str:
        literals = ",".join(f"'{value}'" for value in values)
        return f"{column}=ANY(ARRAY[{literals}])" if pg else f"{column} IN ({literals})"

    def check(expression: str) -> str:
        return f"CHECK ({expression})"

    roots = {
        "owner_user_id": "TEXT NOT NULL",
        "job_uuid": "TEXT NOT NULL",
        "job_id_hint": integer,
        "domain": "TEXT NOT NULL",
        "queue": "TEXT NOT NULL",
        "job_type": "TEXT NOT NULL",
        "payload_field": "TEXT NOT NULL",
        "reference_id": "TEXT NOT NULL",
        "idempotency_key": "TEXT NOT NULL",
        "max_retries": f"{integer} NOT NULL",
        "quarantine_threshold": integer,
        "expired_lease_policy": "TEXT NOT NULL",
        "no_progress_limit": f"{integer} NOT NULL",
        "current_epoch": f"{integer} NOT NULL DEFAULT 0",
        "lease_digest": "TEXT",
        "progress_flag": f"{integer} NOT NULL DEFAULT 0",
        "last_closed_epoch": f"{integer} NOT NULL DEFAULT 0",
        "no_progress_count": f"{integer} NOT NULL DEFAULT 0",
    }
    root_constraints = {
        "wf_root_pk": "PRIMARY KEY (owner_user_id,job_uuid)",
        "wf_root_uuid_unique": "UNIQUE (job_uuid)",
        "wf_root_admission_unique": "UNIQUE (owner_user_id,domain,queue,job_type,idempotency_key)",
        "wf_root_owner": check(bounded("owner_user_id", 512)),
        "wf_root_uuid": check(uuid("job_uuid")),
        "wf_root_reference": check(uuid("reference_id")),
        "wf_root_key": check(bounded("idempotency_key", 200)),
        "wf_root_hint": check("job_id_hint IS NULL OR (" + number("job_id_hint", "job_id_hint>0") + ")"),
        "wf_root_max_retries": check(number("max_retries", "max_retries>=0")),
        "wf_root_quarantine": check(
            "quarantine_threshold IS NULL OR (" + number("quarantine_threshold", "quarantine_threshold>0") + ")"
        ),
        "wf_root_expiry": check("expired_lease_policy='requeue_no_attempt'"),
        "wf_root_no_progress_limit": check(number("no_progress_limit", "no_progress_limit BETWEEN 1 AND 5")),
        "wf_root_current_epoch": check(number("current_epoch", "current_epoch>=0")),
        "wf_root_lease_digest": check("lease_digest IS NULL OR (" + digest("lease_digest") + ")"),
        "wf_root_progress_flag": check(number("progress_flag", "progress_flag IN (0,1)")),
        "wf_root_closed_epoch": check(
            number("last_closed_epoch", "last_closed_epoch>=0 AND last_closed_epoch<=current_epoch")
        ),
        "wf_root_no_progress_count": check(number("no_progress_count", "no_progress_count>=0")),
        "wf_root_reserved": check("NOT (domain='admin_webhooks' AND queue='delivery')"),
    }
    for column in ("domain", "queue", "job_type", "payload_field"):
        root_constraints[f"wf_root_{column}"] = check(code(column))
    receipts = {
        "owner_user_id": "TEXT NOT NULL",
        "job_uuid": "TEXT NOT NULL",
        "effect_key": "TEXT NOT NULL",
        "purpose": "TEXT NOT NULL",
        "action": "TEXT",
        "progress_kind": "TEXT",
        "facts_fingerprint": "TEXT NOT NULL",
        "facts": "TEXT NOT NULL",
        "resulting_status": "TEXT",
        "applied_at": "TIMESTAMP WITH TIME ZONE NOT NULL" if pg else "TEXT NOT NULL",
        "released": f"{integer} NOT NULL DEFAULT 0",
        "slot": f"{integer} NOT NULL",
    }
    action = choice("action", ("continue", "retry", "complete", "fail", "cancel"))
    progress = choice("progress_kind", ("root_binding", "output_receipt", "cursor", "accounting", "sink_lifecycle"))
    status = (
        "(action='continue' AND resulting_status='queued') OR "
        "(action='retry' AND " + choice("resulting_status", ("queued", "failed", "quarantined")) + ") OR "
        "(action='complete' AND resulting_status='completed') OR "
        "(action='fail' AND " + choice("resulting_status", ("failed", "quarantined")) + ") OR "
        "(action='cancel' AND resulting_status='cancelled')"
    )
    receipt_constraints = {
        "wf_receipt_pk": "PRIMARY KEY (owner_user_id,job_uuid,effect_key)",
        "wf_receipt_slot_unique": "UNIQUE (owner_user_id,job_uuid,purpose,slot)",
        "wf_receipt_root_fk": "FOREIGN KEY (owner_user_id,job_uuid) REFERENCES job_workflow_roots(owner_user_id,job_uuid)",
        "wf_receipt_owner": check(bounded("owner_user_id", 512)),
        "wf_receipt_uuid": check(uuid("job_uuid")),
        "wf_receipt_key": check(digest("effect_key")),
        "wf_receipt_digest": check(digest("facts_fingerprint")),
        "wf_receipt_facts": check(bounded("facts", 4096)),
        "wf_receipt_slot": check(number("slot", "slot BETWEEN 1 AND 32768")),
        "wf_receipt_released": check(number("released", "released IN (0,1)")),
        "wf_receipt_purpose": check(
            f"(purpose='action' AND action IS NOT NULL AND {action} AND progress_kind IS NULL "
            f"AND resulting_status IS NOT NULL AND ({status})) OR "
            f"(purpose='progress' AND action IS NULL AND progress_kind IS NOT NULL AND {progress} "
            "AND resulting_status IS NULL AND released=0)"
        ),
    }
    if not pg:
        receipt_constraints["wf_receipt_time"] = check("julianday(applied_at) IS NOT NULL")
    return {
        "job_workflow_schema": (
            {"singleton": f"{integer} NOT NULL", "version": f"{integer} NOT NULL"},
            {
                "wf_schema_pk": "PRIMARY KEY (singleton)",
                "wf_schema_singleton": check(number("singleton", "singleton=1")),
                "wf_schema_version": check(number("version", "version=1")),
            },
        ),
        "job_workflow_roots": (roots, root_constraints),
        "job_workflow_effect_receipts": (receipts, receipt_constraints),
    }


def table_statements(backend: str) -> dict[str, str]:
    """Build DDL solely from the closed, trusted schema inventory."""
    return {
        table: f"CREATE TABLE {table} ("
        + ",".join(
            [f"{column} {declaration}" for column, declaration in columns.items()]
            + [f"CONSTRAINT {name} {definition}" for name, definition in constraints.items()]
        )
        + ")"
        for table, (columns, constraints) in schema_definitions(backend).items()
    }


def enrollment_constraint(backend: str) -> str:
    """Return the explicit-install-only active Jobs enrollment marker check."""
    expression = "workflow_capability_version=1"
    if backend == "sqlite":
        expression = "typeof(workflow_capability_version)='integer' AND " + expression
    return f"CHECK (workflow_capability_version IS NULL OR ({expression}))"


def install_workflow_support_in_transaction(connection: Any, *, backend: str) -> None:
    """Install support in the existing transaction; never commit or reconnect."""
    from .jobs_workflow_store import workflow_store

    workflow_store(backend).install_support(connection)


def inspect_workflow_support_in_transaction(connection: Any, *, backend: str) -> WorkflowOutcome:
    """Read readiness without schema writes or repairs in the caller transaction."""
    from .jobs_workflow_store import workflow_store

    return workflow_store(backend).inspect_support(connection)
