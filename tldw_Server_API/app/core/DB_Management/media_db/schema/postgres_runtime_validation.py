"""Validate an owner-migrated Media schema without granting runtime DDL."""

from __future__ import annotations

from typing import Any

from sqlglot import exp, parse_one
from sqlglot.errors import ParseError

from tldw_Server_API.app.core.DB_Management.media_db.errors import SchemaError
from tldw_Server_API.app.core.DB_Management.media_db.schema.features.core_media import (
    _EXPECTED_MEDIA_COLUMNS,
    _POSTGRES_REQUIRED_TABLES,
    ensure_postgres_media_owner_identity,
    ensure_postgres_sync_log_contract,
)
from tldw_Server_API.app.core.DB_Management.media_db.schema.features.postgres_rls import postgres_policy_sets

_REQUIRED_TABLES = set(_POSTGRES_REQUIRED_TABLES) | {
    "schema_version",
    "mediafiles",
    "visualdocuments",
    "tts_history",
    "data_tables",
    "data_table_columns",
    "data_table_rows",
    "data_table_sources",
    "output_templates",
    "reading_highlights",
    "collection_tags",
    "content_items",
    "content_item_tags",
    "document_reading_progress",
    "document_annotations",
    "document_parsed_references_cache",
    "claims_review_log",
    "claims_review_extractor_metrics_daily",
    "claims_review_rules",
    "claims_monitoring_settings",
    "claims_monitoring_alerts",
    "claims_monitoring_config",
    "claims_monitoring_events",
    "claims_monitoring_health",
    "claims_analytics_exports",
    "claims_notifications",
    "claim_clusters",
    "claim_cluster_membership",
    "claim_cluster_links",
    "email_sources",
    "email_messages",
    "email_attachments",
    "email_participants",
    "email_message_participants",
    "email_labels",
    "email_message_labels",
    "email_sync_state",
    "email_backfill_state",
}
_ADDITIVE_COLUMNS = {
    "media": _EXPECTED_MEDIA_COLUMNS,
    "document_reading_progress": {"cfi", "percentage"},
    "document_annotations": {"chapter_title", "percentage"},
    "claims": {
        "review_status",
        "reviewer_id",
        "review_group",
        "reviewed_at",
        "review_notes",
        "review_version",
        "review_reason_code",
        "claim_cluster_id",
    },
    "claims_monitoring_events": {"delivered_at"},
    "claims_analytics_exports": {"job_id", "error_code", "snapshot_at", "snapshot_event_id"},
    "data_tables": {
        "workspace_tag",
        "column_hints_json",
        "last_modified",
        "version",
        "client_id",
        "deleted",
        "prev_version",
        "merge_parent_uuid",
    },
    "data_table_columns": {
        "format",
        "last_modified",
        "version",
        "client_id",
        "deleted",
        "prev_version",
        "merge_parent_uuid",
    },
    "data_table_rows": {
        "row_hash",
        "last_modified",
        "version",
        "client_id",
        "deleted",
        "prev_version",
        "merge_parent_uuid",
    },
    "data_table_sources": {"last_modified", "version", "client_id", "deleted", "prev_version", "merge_parent_uuid"},
}
_FTS = (
    ("media_fts", "media", ("title", "content")),
    ("keyword_fts", "keywords", ("keyword",)),
    ("claims_fts", "claims", ("claim_text",)),
    ("unvectorized_chunks_fts", "unvectorizedmediachunks", ("chunk_text",)),
)


def canonical_policy_expression(expression: str, table: str) -> exp.Expression:
    """Normalize deparser decoration without changing Boolean or correlated scope."""

    def normalize(node: exp.Expression) -> exp.Expression:
        for key, value in list(node.args.items()):
            if isinstance(value, exp.Expression):
                node.set(key, normalize(value))
            elif isinstance(value, list):
                node.set(key, [normalize(item) if isinstance(item, exp.Expression) else item for item in value])
        if isinstance(node, exp.Paren):
            return node.this
        if isinstance(node, exp.Alias) and isinstance(node.parent, exp.Select):
            select = node.parent
            if (
                len(select.expressions) == 1
                and isinstance(node.this, exp.Anonymous)
                and node.this.name.lower() == "current_setting"
                and node.alias == "current_setting"
                and not any(
                    select.args.get(key)
                    for key in (
                        "from",
                        "from_",
                        "joins",
                        "where",
                        "group",
                        "having",
                        "order",
                        "limit",
                        "offset",
                        "distinct",
                    )
                )
            ):
                return node.this
        if isinstance(node, exp.Cast) and isinstance(node.this, exp.Literal) and node.this.is_string:
            if node.to.is_type(exp.DataType.Type.TEXT):
                return node.this
            if node.to.is_type(exp.DataType.Type.ARRAY) and node.this.this == "{}":
                node.set("this", parse_one("ARRAY[]", read="postgres"))
        if isinstance(node, exp.Column) and node.table == table and node.find_ancestor(exp.Select) is None:
            node.set("table", None)
        if isinstance(node, exp.Identifier) and node.name.isascii() and node.name == node.name.lower():
            node.set("quoted", False)
        if isinstance(node, exp.Between) and not node.args.get("symmetric"):
            node = exp.And(
                this=exp.GTE(this=node.this.copy(), expression=node.args["low"].copy()),
                expression=exp.LTE(this=node.this.copy(), expression=node.args["high"].copy()),
            )
        if isinstance(node, (exp.And, exp.Or)):
            kind = type(node)
            parts = []

            def flatten(child: exp.Expression) -> None:
                if isinstance(child, kind):
                    flatten(child.this)
                    flatten(child.expression)
                else:
                    parts.append(child)

            flatten(node)
            node = parts[0]
            for part in parts[1:]:
                node = kind(this=node, expression=part)
        return node

    try:
        return normalize(parse_one(expression, read="postgres"))
    except ParseError as exc:
        raise SchemaError("PostgreSQL runtime RLS policy cannot be parsed") from exc


def validate_postgres_runtime_schema(db: Any, conn: Any) -> None:
    """Refuse incomplete catalogs; leave all schema repair to the owner."""
    backend = db.backend
    writable_version = backend.execute(
        "SELECT has_table_privilege(current_user,'schema_version','INSERT,UPDATE,DELETE,TRUNCATE') "
        "OR has_any_column_privilege(current_user,'schema_version','INSERT,UPDATE')",
        connection=conn,
    ).scalar
    if writable_version:
        raise SchemaError("PostgreSQL runtime migration-version authority is unsafe")
    tables = backend.execute(
        "SELECT c.relname FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace "
        "WHERE n.nspname=current_schema() AND c.relkind IN ('r','p') AND c.relname=ANY(%s)",
        (sorted(_REQUIRED_TABLES),),
        connection=conn,
    ).rows
    if {row["relname"] for row in tables} != _REQUIRED_TABLES:
        raise SchemaError("PostgreSQL runtime schema is incomplete; run schema-owner migration")
    names = ", ".join(backend.escape_identifier(name) for name in sorted(_REQUIRED_TABLES))
    # Fixed package table names, individually escaped; locks prevent concurrent catalog replacement.
    backend.execute(f"LOCK TABLE {names} IN ACCESS SHARE MODE", connection=conn)  # nosec B608
    for table, expected_columns in _ADDITIVE_COLUMNS.items():
        columns = {row["name"] for row in backend.get_table_info(table, connection=conn)}
        if not expected_columns <= columns:
            raise SchemaError(f"PostgreSQL runtime columns are incomplete for {table}")
    ensure_postgres_sync_log_contract(db, conn, validate_only=True)
    ensure_postgres_media_owner_identity(db, conn, validate_only=True)

    for alias, table, columns in _FTS:
        field = alias + "_tsv"
        row = backend.execute(
            "SELECT a.atttypid='tsvector'::regtype AS vector_type, "
            "i.indisvalid AND i.indisready AND i.indpred IS NULL AND i.indnkeyatts=1 "
            "AND i.indkey[0]=a.attnum AND am.amname='gin' AS valid_index, "
            "t.tgtype=23 AND t.tgenabled IN ('O','A') AND NOT t.tgisinternal "
            "AND t.tgqual IS NULL AND t.tgattr::text='' AND t.tgnargs=0 AND octet_length(t.tgargs)=0 "
            "AND p.pronargs=0 AND p.prorettype='trigger'::regtype AND NOT p.prosecdef "
            "AND NOT p.proretset AND p.proconfig IS NULL AND p.provolatile='v' "
            "AND (SELECT lanname FROM pg_language WHERE oid=p.prolang)='plpgsql' "
            "AS valid_trigger, p.prosrc "
            "FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace "
            "JOIN pg_attribute a ON a.attrelid=c.oid AND a.attname=%s AND NOT a.attisdropped "
            "JOIN pg_index i ON i.indrelid=c.oid "
            "JOIN pg_class ix ON ix.oid=i.indexrelid AND ix.relname=%s "
            "JOIN pg_am am ON am.oid=ix.relam "
            "JOIN pg_trigger t ON t.tgrelid=c.oid AND t.tgname=%s "
            "JOIN pg_proc p ON p.oid=t.tgfoid AND p.proname=%s AND p.pronamespace=n.oid "
            "WHERE n.nspname=current_schema() AND c.relname=%s",
            (field, f"idx_{table}_{field}", f"update_{field}_trigger", f"update_{field}_function", table),
            connection=conn,
        ).first
        joined = " || ' ' || ".join(f"coalesce(NEW.{backend.escape_identifier(column)}, '')" for column in columns)
        body = f"""
                BEGIN
                    NEW.{backend.escape_identifier(field)} :=
                        to_tsvector('english', {joined});
                    RETURN NEW;
                END;
        """
        if (
            not row
            or not all(row[key] for key in ("vector_type", "valid_index", "valid_trigger"))
            or row["prosrc"].strip() != body.strip()
        ):
            raise SchemaError(f"PostgreSQL runtime FTS catalog drifted for {table}")
        backend.register_fts_table(alias, table)

    policies = postgres_policy_sets(backend)
    states = backend.execute(
        "SELECT c.relname,c.relrowsecurity,c.relforcerowsecurity "
        "FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace "
        "WHERE n.nspname=current_schema() AND c.relname=ANY(%s)",
        (list(policies),),
        connection=conn,
    ).rows
    if len(states) != len(policies) or any(
        not row["relrowsecurity"] or not row["relforcerowsecurity"] for row in states
    ):
        raise SchemaError("PostgreSQL runtime RLS catalog drifted")
    rows = backend.execute(
        "SELECT tablename,policyname,permissive,roles::text,cmd,qual,with_check FROM pg_policies "
        "WHERE schemaname=current_schema() AND tablename=ANY(%s)",
        (list(policies),),
        connection=conn,
    ).rows
    expected = {(table, name): predicate for table, definitions in policies.items() for name, predicate in definitions}
    actual = {(row["tablename"], row["policyname"]) for row in rows}
    # Conversation storage shares sync_log and installs this additional owner-only policy.
    conversation_policy = ("sync_log", "sync_log_tenant_isolation")
    if conversation_policy in actual:
        expected[conversation_policy] = "client_id = current_setting('app.current_user_id', true)"
    if actual != set(expected):
        raise SchemaError("PostgreSQL runtime RLS policy catalog drifted")
    for row in rows:
        table = row["tablename"]
        predicate = canonical_policy_expression(expected[table, row["policyname"]], table)
        if (
            row["permissive"] != "PERMISSIVE"
            or row["roles"] != "{public}"
            or row["cmd"] != "ALL"
            or not row["qual"]
            or not row["with_check"]
            or canonical_policy_expression(row["qual"], table) != predicate
            or canonical_policy_expression(row["with_check"], table) != predicate
        ):
            raise SchemaError(f"PostgreSQL runtime RLS policy authority drifted for {table}/{row['policyname']}")
    from tldw_Server_API.app.core.DB_Management.media_db.schema.backends.postgres_helpers import (
        _SAFE_TRANSCRIPT_TEXT_FUNCTION_SQL,
    )
    from tldw_Server_API.app.core.DB_Management.media_db.schema.postgres_claims_json_helpers import (
        POSTGRES_CLAIMS_JSON_HELPER_DDL,
    )

    helpers = (
        (
            "public.tldw_try_extract_normalized_transcript_text(text)",
            _SAFE_TRANSCRIPT_TEXT_FUNCTION_SQL,
            "text",
            True,
            "u",
            ["search_path=pg_catalog"],
        ),
        ("tldw_claims_safe_json(text)", POSTGRES_CLAIMS_JSON_HELPER_DDL[0], "json", False, "s", None),
        ("tldw_claims_compact_json(json)", POSTGRES_CLAIMS_JSON_HELPER_DDL[1], "text", False, "s", None),
    )
    for signature, ddl, returns, strict, parallel, settings in helpers:
        row = backend.execute(
            "SELECT p.prosrc, p.provolatile, p.proisstrict, p.prosecdef, p.proparallel, p.proconfig, "
            "p.proretset, p.pronargdefaults, p.provariadic, p.prorettype=%s::regtype AS return_type, "
            "l.lanname FROM pg_proc p JOIN pg_language l ON l.oid=p.prolang "
            "WHERE p.oid=to_regprocedure(%s)",
            (returns, signature),
            connection=conn,
        ).first
        body = ddl.split("AS $function$", 1)[1].split("$function$", 1)[0].strip()
        if (
            not row
            or row["prosrc"].strip() != body
            or row["provolatile"] != "i"
            or row["proisstrict"] != strict
            or row["prosecdef"]
            or row["proparallel"] != parallel
            or row["proconfig"] != settings
            or row["proretset"]
            or row["pronargdefaults"]
            or row["provariadic"]
            or not row["return_type"]
            or row["lanname"] != "plpgsql"
        ):
            raise SchemaError(f"PostgreSQL runtime content helper catalog drifted for {signature}")
