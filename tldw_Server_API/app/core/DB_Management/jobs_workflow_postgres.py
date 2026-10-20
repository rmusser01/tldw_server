"""PostgreSQL workflow schema SQL, including forced, owner-only RLS."""

from __future__ import annotations

from typing import Any

import sqlglot
from sqlglot import exp

from tldw_Server_API.app.core.Jobs.operations.workflow_contracts import WorkflowOutcome

from .jobs_workflow_migrations import (
    RECEIPT_IDENTITY_COLUMNS,
    ROOT_IDENTITY_COLUMNS,
    enrollment_constraint,
    schema_definitions,
    table_statements,
)

_OWNER = "owner_user_id=NULLIF(current_setting('app.owner_user_id',true),'')"


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
