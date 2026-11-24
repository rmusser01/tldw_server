"""PostgreSQL schema migration steps moved out of CharactersRAGDB (TASK-13950.22)."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from tldw_Server_API.app.core.DB_Management.backends.pg_rls_policies import (
    build_chacha_rls_sql,
    build_shared_workspace_chat_rls_sql,
)
from tldw_Server_API.app.core.DB_Management.chacha import postgres_schema, schema_sql
from tldw_Server_API.app.core.DB_Management.chacha.migrations import support as migration_support
from tldw_Server_API.app.core.DB_Management.chacha.postgres_note_schema_contracts import (
    MOODBOARD_STUDIO_V61_INDEX_STATEMENTS,
    require_moodboard_migration_complete,
)
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import SchemaError
from tldw_Server_API.app.core.DB_Management.sql_utils import split_sql_statements


def _migrate_from_v54_to_v55_postgres(ctx, conn: Any) -> None:
    """Backfill stable organization identities inside the caller's transaction."""
    from tldw_Server_API.app.core.Sync_Contracts.notes_organization import new_organization_sync_id

    postgres_schema._ensure_note_folder_sync_suppression_schema_postgres(ctx, conn)
    table_indexes = (
        ("chacha_keywords", "idx_keywords_sync_id_unique"),
        ("keyword_collections", "idx_keyword_collections_sync_id_unique"),
        ("note_folders", "idx_note_folders_sync_id_unique"),
    )
    for table_name, index_name in table_indexes:
        ctx.backend.execute(
            f"ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS sync_id TEXT",  # nosec B608
            connection=conn,
        )
        rows = ctx.backend.execute(
            f"SELECT id FROM {table_name} WHERE sync_id IS NULL OR sync_id = %s ORDER BY id",  # nosec B608
            ("",),
            connection=conn,
        ).rows
        for row in rows:
            ctx.backend.execute(
                f"UPDATE {table_name} SET sync_id = %s WHERE id = %s",  # nosec B608
                (new_organization_sync_id(), int(row["id"])),
                connection=conn,
            )

        invalid = ctx.backend.execute(
            f"SELECT COUNT(*) FROM {table_name} WHERE sync_id IS NULL OR sync_id = %s",  # nosec B608
            ("",),
            connection=conn,
        ).scalar
        duplicates = ctx.backend.execute(
            f"SELECT COUNT(*) FROM ("  # nosec B608
            f"SELECT sync_id FROM {table_name} GROUP BY sync_id HAVING COUNT(*) > 1"
            ") duplicate_sync_ids",
            connection=conn,
        ).scalar
        if int(invalid or 0) or int(duplicates or 0):
            raise SchemaError(  # noqa: TRY003
                f"Cannot migrate {table_name} stable identities: "
                f"invalid={int(invalid or 0)}, duplicates={int(duplicates or 0)}."
            )

        ctx.backend.execute(
            f"ALTER TABLE {table_name} ALTER COLUMN sync_id SET NOT NULL",  # nosec B608
            connection=conn,
        )
        ctx.backend.execute(
            f"CREATE UNIQUE INDEX IF NOT EXISTS {index_name} ON {table_name}(sync_id)",  # nosec B608
            connection=conn,
        )

    postgres_schema._set_schema_version_postgres(ctx, conn, 55)


def _migrate_from_v55_to_v56_postgres(ctx, conn: Any) -> None:
    """Owner-scope and rekey WebClipper rows in one serialized transaction."""
    backend = ctx.backend
    if not backend.table_exists("note_clipper_documents", connection=conn):
        postgres_schema._ensure_web_clipper_schema_postgres(ctx, conn)
        postgres_schema._set_schema_version_postgres(ctx, conn, 56)
        return

    has_placements = backend.table_exists(
        "note_clipper_workspace_placements", connection=conn
    )
    sync_tables = tuple(
        table_name
        for table_name in ("sync_envelopes", "sync_object_state")
        if backend.table_exists(table_name, connection=conn)
    )
    lock_tables = ["notes", "workspaces", "sync_log", "note_clipper_documents"]
    if has_placements:
        lock_tables.append("note_clipper_workspace_placements")
    lock_tables.extend(sync_tables)
    backend.execute(
        f"LOCK TABLE {', '.join(lock_tables)} IN ACCESS EXCLUSIVE MODE",  # nosec B608
        connection=conn,
    )

    rls_state = backend.execute(
        """
            SELECT notes_table.relrowsecurity,
                   notes_table.relforcerowsecurity,
                   notes_table.relowner = current_user::regrole AS is_schema_owner
              FROM pg_class notes_table
             WHERE notes_table.oid = 'notes'::regclass
            """,
        connection=conn,
    ).first
    if rls_state is None or not bool(rls_state.get("is_schema_owner")):
        raise SchemaError(  # noqa: TRY003
            "WebClipper v56 requires the verified PostgreSQL schema-owner migration path."
        )
    rls_enabled = bool(rls_state.get("relrowsecurity"))
    rls_forced = bool(rls_state.get("relforcerowsecurity"))
    if rls_forced and not rls_enabled:
        raise SchemaError("PostgreSQL notes RLS state cannot be verified.")  # noqa: TRY003
    if rls_forced:
        backend.execute(
            "ALTER TABLE notes NO FORCE ROW LEVEL SECURITY",
            connection=conn,
        )

    backend.execute(
        "ALTER TABLE note_clipper_documents ADD COLUMN IF NOT EXISTS client_id TEXT",
        connection=conn,
    )
    if has_placements:
        backend.execute(
            "ALTER TABLE note_clipper_workspace_placements "
            "ADD COLUMN IF NOT EXISTS client_id TEXT",
            connection=conn,
        )

    invalid_owners = backend.execute(
        """
            SELECT COUNT(*) AS invalid_owner_count
              FROM note_clipper_documents document
              LEFT JOIN notes note ON note.id = document.note_id
             WHERE note.id IS NULL
                OR note.client_id IS NULL
                OR BTRIM(note.client_id) = ''
                OR note.client_id <> BTRIM(note.client_id)
                OR BTRIM(note.client_id) !~ '^[1-9][0-9]*$'
                OR (
                     document.client_id IS NOT NULL
                     AND document.client_id IS DISTINCT FROM note.client_id
                   )
            """,
        connection=conn,
    ).scalar
    if int(invalid_owners or 0):
        raise SchemaError(  # noqa: TRY003
            "Cannot migrate WebClipper mappings: an authoritative note owner is missing or invalid."
        )

    if has_placements:
        invalid_placement_owners = backend.execute(
            """
                SELECT COUNT(*) AS invalid_owner_count
                  FROM note_clipper_workspace_placements placement
                  LEFT JOIN note_clipper_documents document
                    ON document.clip_id = placement.clip_id
                  LEFT JOIN notes owner_note
                    ON owner_note.id = document.note_id
                  LEFT JOIN notes source_note
                    ON source_note.id = placement.source_note_id
                  LEFT JOIN workspaces workspace
                    ON workspace.id = placement.workspace_id
                 WHERE document.clip_id IS NULL
                    OR owner_note.id IS NULL
                    OR placement.source_note_id IS DISTINCT FROM document.note_id
                    OR source_note.client_id IS DISTINCT FROM owner_note.client_id
                    OR workspace.client_id IS DISTINCT FROM owner_note.client_id
                    OR (
                         document.client_id IS NOT NULL
                         AND document.client_id IS DISTINCT FROM owner_note.client_id
                       )
                    OR (
                         placement.client_id IS NOT NULL
                         AND placement.client_id IS DISTINCT FROM owner_note.client_id
                       )
                """,
            connection=conn,
        ).scalar
        if int(invalid_placement_owners or 0):
            raise SchemaError(  # noqa: TRY003
                "Cannot migrate WebClipper placements: owner relationships are invalid."
            )

    rows = backend.execute(
        """
            SELECT document.clip_id,
                   document.note_id,
                   note.id AS existing_note_id,
                   note.client_id AS owner_client_id
              FROM note_clipper_documents document
              LEFT JOIN notes note ON note.id = document.note_id
             ORDER BY note.client_id, document.clip_id
            """,
        connection=conn,
    ).rows
    plan = migration_support._web_clipper_v56_rekey_plan(rows)
    owner_by_clip_id = {
        str(row["clip_id"]): str(row["owner_client_id"])
        for row in rows
    }
    old_ids = [old_id for _, old_id, _ in plan]
    for clip_id, _, replacement in plan:
        collision = backend.execute(
            "SELECT 1 FROM notes WHERE id = %s LIMIT 1",
            (replacement,),
            connection=conn,
        ).first
        if collision:
            raise SchemaError(f"WebClipper note rekey target '{replacement}' already exists.")  # noqa: TRY003
        owner_client_id = owner_by_clip_id[clip_id]
        sync_log_collision = backend.execute(
            "SELECT 1 FROM sync_log "
            "WHERE client_id = %s AND entity = %s AND entity_id = %s LIMIT 1",
            (owner_client_id, "notes", replacement),
            connection=conn,
        ).first
        if sync_log_collision:
            raise SchemaError(  # noqa: TRY003
                f"WebClipper note rekey target '{replacement}' already has Sync history."
            )

    for table_name, id_column in (
        ("sync_envelopes", "entity_id"),
        ("sync_object_state", "object_id"),
    ):
        if not old_ids or table_name not in sync_tables:
            continue
        for old_id in old_ids:
            found = backend.execute(
                f"SELECT 1 FROM {table_name} "  # nosec B608
                f"WHERE domain = %s AND {id_column} = %s LIMIT 1",
                ("notes.note", old_id),
                connection=conn,
            ).first
            if found:
                raise SchemaError(  # noqa: TRY003
                    f"WebClipper note '{old_id}' has canonical Sync history; explicit migration is required."
                )
        for _, _, replacement in plan:
            found = backend.execute(
                f"SELECT 1 FROM {table_name} "  # nosec B608
                f"WHERE domain = %s AND {id_column} = %s LIMIT 1",
                ("notes.note", replacement),
                connection=conn,
            ).first
            if found:
                raise SchemaError(  # noqa: TRY003
                    f"WebClipper note rekey target '{replacement}' already has canonical Sync history."
                )

    if old_ids:
        foreign_keys = backend.execute(
            """
                SELECT c.conname,
                       ns.nspname AS table_schema,
                       rel.relname AS table_name,
                       att.attname AS column_name,
                       c.confupdtype AS on_update
                  FROM pg_constraint c
                  JOIN pg_class rel ON rel.oid = c.conrelid
                  JOIN pg_namespace ns ON ns.oid = rel.relnamespace
                  JOIN LATERAL unnest(c.conkey) AS key(attnum) ON TRUE
                  JOIN pg_attribute att
                    ON att.attrelid = c.conrelid AND att.attnum = key.attnum
                 WHERE c.contype = 'f' AND c.confrelid = 'notes'::regclass
                """,
            connection=conn,
        ).rows
        for foreign_key in foreign_keys:
            if str(foreign_key["on_update"]) == "c":
                continue
            schema_name = str(foreign_key["table_schema"])
            table_name = str(foreign_key["table_name"])
            column_name = str(foreign_key["column_name"])
            qualified_table = (
                f"{postgres_schema._quoted_postgres_identifier(schema_name)}."
                f"{postgres_schema._quoted_postgres_identifier(table_name)}"
            )
            quoted_column = postgres_schema._quoted_postgres_identifier(column_name)
            for old_id in old_ids:
                found = backend.execute(
                    f"SELECT 1 FROM {qualified_table} WHERE {quoted_column} = %s LIMIT 1",  # nosec B608
                    (old_id,),
                    connection=conn,
                ).first
                if found:
                    raise SchemaError(  # noqa: TRY003
                        f"Cannot rekey WebClipper note '{old_id}': live non-cascading reference "
                        f"from {table_name}.{column_name}."
                    )

    constraints = backend.execute(
        """
            SELECT c.conname,
                   c.contype,
                   pg_get_constraintdef(c.oid) AS definition,
                   ARRAY(
                     SELECT a.attname
                       FROM unnest(c.conkey) WITH ORDINALITY AS key(attnum, ord)
                       JOIN pg_attribute a
                         ON a.attrelid = c.conrelid AND a.attnum = key.attnum
                      ORDER BY key.ord
                   ) AS columns
              FROM pg_constraint c
             WHERE c.conrelid = 'note_clipper_documents'::regclass
            """,
        connection=conn,
    ).rows
    has_clip_primary_key = any(
        str(constraint["contype"]) == "p"
        and [str(column) for column in (constraint.get("columns") or [])] == ["clip_id"]
        for constraint in constraints
    )
    has_unique_note_id = any(
        str(constraint["contype"]) == "u"
        and [str(column) for column in (constraint.get("columns") or [])] == ["note_id"]
        for constraint in constraints
    )
    has_cascading_note_fk = any(
        str(constraint["contype"]) == "f"
        and [str(column) for column in (constraint.get("columns") or [])] == ["note_id"]
        and re.search(
            r"REFERENCES\s+(?:\w+\.)?notes\s*\(",
            str(constraint.get("definition") or "").replace('"', ""),
            re.IGNORECASE,
        )
        and "ON UPDATE CASCADE" in str(constraint.get("definition") or "").upper()
        for constraint in constraints
    )
    if not (has_clip_primary_key and has_unique_note_id and has_cascading_note_fk):
        raise SchemaError("WebClipper durable note mapping constraints cannot be verified.")  # noqa: TRY003

    placement_constraints: list[Any] = []
    if has_placements:
        placement_constraints = backend.execute(
            """
                SELECT c.conname,
                       c.contype,
                       pg_get_constraintdef(c.oid) AS definition,
                       ARRAY(
                         SELECT a.attname
                           FROM unnest(c.conkey) WITH ORDINALITY AS key(attnum, ord)
                           JOIN pg_attribute a
                             ON a.attrelid = c.conrelid AND a.attnum = key.attnum
                          ORDER BY key.ord
                       ) AS columns
                  FROM pg_constraint c
                 WHERE c.conrelid = 'note_clipper_workspace_placements'::regclass
                """,
            connection=conn,
        ).rows
        has_placement_primary_key = any(
            str(constraint["contype"]) == "p"
            and [str(column) for column in (constraint.get("columns") or [])]
            == ["clip_id", "workspace_id"]
            for constraint in placement_constraints
        )
        has_document_fk = any(
            str(constraint["contype"]) == "f"
            and [str(column) for column in (constraint.get("columns") or [])] == ["clip_id"]
            and "REFERENCES note_clipper_documents" in str(
                constraint.get("definition") or ""
            ).replace('"', "")
            for constraint in placement_constraints
        )
        if not (has_placement_primary_key and has_document_fk):
            raise SchemaError("WebClipper placement constraints cannot be verified.")  # noqa: TRY003

    for constraint in placement_constraints:
        constraint_type = str(constraint["contype"])
        columns = [str(column) for column in (constraint.get("columns") or [])]
        definition = str(constraint.get("definition") or "").replace('"', "")
        if not (
            (constraint_type == "p" and columns == ["clip_id", "workspace_id"])
            or (
                constraint_type == "f"
                and columns == ["clip_id"]
                and "REFERENCES note_clipper_documents" in definition
            )
        ):
            continue
        backend.execute(
            "ALTER TABLE note_clipper_workspace_placements DROP CONSTRAINT "
            f"{postgres_schema._quoted_postgres_identifier(str(constraint['conname']))}",  # nosec B608
            connection=conn,
        )

    for constraint in constraints:
        constraint_name = str(constraint["conname"])
        constraint_type = str(constraint["contype"])
        definition = str(constraint.get("definition") or "")
        columns = [str(column) for column in (constraint.get("columns") or [])]
        normalized_definition = definition.replace('"', "")
        is_clip_fk = constraint_type == "f" and columns == ["clip_id"]
        is_legacy_clip_fk = is_clip_fk and bool(
            re.search(r"REFERENCES\s+(?:\w+\.)?notes\s*\(", normalized_definition, re.IGNORECASE)
        )
        if is_clip_fk and not is_legacy_clip_fk:
            raise SchemaError("WebClipper clip_id has an unverifiable PostgreSQL reference.")  # noqa: TRY003
        is_legacy_identity_check = constraint_type == "c" and bool(
            re.search(
                r"(?:clip_id\s*=\s*note_id|note_id\s*=\s*clip_id)",
                normalized_definition,
            )
        )
        if (
            constraint_type == "c"
            and "clip_id" in normalized_definition
            and "note_id" in normalized_definition
            and not is_legacy_identity_check
        ):
            raise SchemaError("WebClipper identity check constraint cannot be verified.")  # noqa: TRY003
        should_drop = (
            is_legacy_clip_fk
            or is_legacy_identity_check
            or (constraint_type == "p" and columns == ["clip_id"])
            or (constraint_type == "u" and columns == ["note_id"])
        )
        if should_drop:
            backend.execute(
                "ALTER TABLE note_clipper_documents DROP CONSTRAINT "
                f"{postgres_schema._quoted_postgres_identifier(constraint_name)}",  # nosec B608
                connection=conn,
            )

    backend.execute(
        """
            UPDATE note_clipper_documents document
               SET client_id = note.client_id
              FROM notes note
             WHERE note.id = document.note_id
            """,
        connection=conn,
    )
    backend.execute(
        "ALTER TABLE note_clipper_documents ALTER COLUMN client_id SET NOT NULL",
        connection=conn,
    )
    backend.execute(
        "ALTER TABLE note_clipper_documents ADD CONSTRAINT "
        "note_clipper_documents_pkey PRIMARY KEY (client_id, clip_id)",
        connection=conn,
    )
    backend.execute(
        "ALTER TABLE note_clipper_documents ADD CONSTRAINT "
        "note_clipper_documents_client_note_key UNIQUE (client_id, note_id)",
        connection=conn,
    )

    if has_placements:
        backend.execute(
            """
                UPDATE note_clipper_workspace_placements placement
                   SET client_id = document.client_id
                  FROM note_clipper_documents document
                 WHERE document.clip_id = placement.clip_id
                """,
            connection=conn,
        )
        backend.execute(
            "ALTER TABLE note_clipper_workspace_placements "
            "ALTER COLUMN client_id SET NOT NULL",
            connection=conn,
        )
        backend.execute(
            "ALTER TABLE note_clipper_workspace_placements ADD CONSTRAINT "
            "note_clipper_workspace_placements_pkey "
            "PRIMARY KEY (client_id, clip_id, workspace_id)",
            connection=conn,
        )
        backend.execute(
            "ALTER TABLE note_clipper_workspace_placements ADD CONSTRAINT "
            "note_clipper_workspace_placements_document_fkey "
            "FOREIGN KEY (client_id, clip_id) "
            "REFERENCES note_clipper_documents(client_id, clip_id) "
            "ON DELETE CASCADE ON UPDATE CASCADE",
            connection=conn,
        )

    for index_name in (
        "idx_note_clipper_documents_note_id",
        "idx_note_clipper_workspace_placements_workspace",
        "idx_note_clipper_workspace_placements_source_note",
    ):
        backend.execute(
            f"DROP INDEX IF EXISTS {postgres_schema._quoted_postgres_identifier(index_name)}",  # nosec B608
            connection=conn,
        )

    for clip_id, old_id, replacement in plan:
        owner_client_id = owner_by_clip_id[clip_id]
        invalid_payloads = backend.execute(
            "SELECT COUNT(*) FROM sync_log "
            "WHERE client_id = %s AND entity = %s AND entity_id = %s "
            "AND (jsonb_typeof(payload::jsonb) IS DISTINCT FROM 'object' "
            "OR jsonb_typeof(payload::jsonb -> 'id') IS DISTINCT FROM 'string' "
            "OR (payload::jsonb ->> 'id') IS DISTINCT FROM %s)",
            (owner_client_id, "notes", old_id, old_id),
            connection=conn,
        ).scalar
        if int(invalid_payloads or 0):
            raise SchemaError("WebClipper legacy sync_log payload cannot be verified.")  # noqa: TRY003
        backend.execute(
            "UPDATE notes SET id = %s WHERE client_id = %s AND id = %s",
            (replacement, owner_client_id, old_id),
            connection=conn,
        )
        backend.execute(
            "UPDATE sync_log SET entity_id = %s, "
            "payload = jsonb_set(payload::jsonb, '{id}', to_jsonb(%s::text), true)::text "
            "WHERE client_id = %s AND entity = %s AND entity_id = %s",
            (replacement, replacement, owner_client_id, "notes", old_id),
            connection=conn,
        )
        invalid_rewrites = backend.execute(
            "SELECT COUNT(*) FROM sync_log "
            "WHERE client_id = %s AND entity = %s AND entity_id = %s "
            "AND (jsonb_typeof(payload::jsonb) IS DISTINCT FROM 'object' "
            "OR jsonb_typeof(payload::jsonb -> 'id') IS DISTINCT FROM 'string' "
            "OR (payload::jsonb ->> 'id') IS DISTINCT FROM %s)",
            (owner_client_id, "notes", replacement, replacement),
            connection=conn,
        ).scalar
        if int(invalid_rewrites or 0):
            raise SchemaError("WebClipper sync_log payload rewrite verification failed.")  # noqa: TRY003
        mapping = backend.execute(
            "SELECT note_id FROM note_clipper_documents "
            "WHERE client_id = %s AND clip_id = %s",
            (owner_client_id, clip_id),
            connection=conn,
        ).first
        if mapping is None or str(mapping["note_id"]) != replacement:
            raise SchemaError("WebClipper v56 mapping verification failed.")  # noqa: TRY003

    if has_placements:
        mismatched = backend.execute(
            """
                SELECT COUNT(*)
                  FROM note_clipper_workspace_placements placement
                  JOIN note_clipper_documents document
                    ON document.client_id = placement.client_id
                   AND document.clip_id = placement.clip_id
                 WHERE placement.source_note_id <> document.note_id
                """,
            connection=conn,
        ).scalar
        if int(mismatched or 0):
            raise SchemaError("WebClipper v56 placement verification failed.")  # noqa: TRY003

    if rls_forced:
        backend.execute(
            "ALTER TABLE notes FORCE ROW LEVEL SECURITY",
            connection=conn,
        )
    postgres_schema._ensure_web_clipper_schema_postgres(ctx, conn)
    if has_placements:
        backend.execute(
            "ALTER TABLE note_clipper_workspace_placements VALIDATE CONSTRAINT "
            "note_clipper_workspace_placements_document_fkey",
            connection=conn,
        )
    postgres_schema._set_schema_version_postgres(ctx, conn, 56)


def _migrate_from_v56_to_v57_postgres(ctx, conn: Any) -> None:
    """Owner-scope shared Notes organization uniqueness without guessing owners."""
    backend = ctx.backend
    locked_tables = (
        "chacha_keywords",
        "collection_keywords",
        "conversation_keywords",
        "conversations",
        "keyword_collections",
        "note_folder_memberships",
        "note_folder_source_memberships",
        "note_folder_sync_suppressions",
        "note_folders",
        "note_keywords",
        "notes",
    )
    backend.execute(
        """
            LOCK TABLE
              chacha_keywords,
              collection_keywords,
              conversation_keywords,
              conversations,
              keyword_collections,
              note_folder_memberships,
              note_folder_source_memberships,
              note_folder_sync_suppressions,
              note_folders,
              note_keywords,
              notes
            IN SHARE ROW EXCLUSIVE MODE
            """,
        connection=conn,
    )
    rls_states = backend.execute(
        """
            SELECT table_row.relname AS table_name,
                   table_row.relrowsecurity,
                   table_row.relforcerowsecurity,
                   table_row.relowner = current_user::regrole AS is_schema_owner
              FROM pg_class AS table_row
              JOIN pg_namespace AS namespace_row
                ON namespace_row.oid = table_row.relnamespace
             WHERE namespace_row.nspname = current_schema()
               AND table_row.relkind IN ('r', 'p')
               AND table_row.relname IN (
                 'chacha_keywords',
                 'collection_keywords',
                 'conversation_keywords',
                 'conversations',
                 'keyword_collections',
                 'note_folder_memberships',
                 'note_folder_source_memberships',
                 'note_folder_sync_suppressions',
                 'note_folders',
                 'note_keywords',
                 'notes'
               )
            """,
        connection=conn,
    ).rows
    state_by_table = {
        str(row["table_name"]): row
        for row in rls_states
    }
    if set(state_by_table) != set(locked_tables) or any(
        not bool(state_by_table[table_name].get("is_schema_owner"))
        for table_name in locked_tables
    ):
        raise SchemaError(  # noqa: TRY003
            "Notes organization v57 requires the verified PostgreSQL "
            "schema-owner migration path for every locked table."
        )
    forced_rls_tables: list[str] = []
    for table_name in locked_tables:
        state = state_by_table[table_name]
        rls_enabled = bool(state.get("relrowsecurity"))
        rls_forced = bool(state.get("relforcerowsecurity"))
        if rls_forced and not rls_enabled:
            raise SchemaError(  # noqa: TRY003
                f"PostgreSQL RLS state cannot be verified for {table_name}."
            )
        if not rls_forced:
            continue
        backend.execute(
            f"ALTER TABLE {postgres_schema._quoted_postgres_identifier(table_name)} "  # nosec B608
            "NO FORCE ROW LEVEL SECURITY",
            connection=conn,
        )
        forced_rls_tables.append(table_name)

    resource_tables = (
        "chacha_keywords",
        "keyword_collections",
        "note_folders",
    )
    for table_name in resource_tables:
        invalid_owners = backend.execute(
            f"""
                SELECT COUNT(*)
                  FROM {table_name}
                 WHERE client_id IS NULL
                    OR BTRIM(client_id) = ''
                    OR client_id <> BTRIM(client_id)
                    OR BTRIM(client_id) !~ '^[1-9][0-9]*$'
                """,  # nosec B608 -- table names are fixed migration constants.
            connection=conn,
        ).scalar
        if int(invalid_owners or 0):
            raise SchemaError(  # noqa: TRY003
                f"Cannot migrate {table_name}: {int(invalid_owners)} row(s) lack "
                "a canonical authenticated owner. Refusing to infer ownership."
            )

    collision_checks = (
        ("chacha_keywords", "sync_id", None),
        ("chacha_keywords", "LOWER(keyword)", None),
        ("keyword_collections", "sync_id", None),
        ("keyword_collections", "LOWER(name)", None),
        ("note_folders", "sync_id", None),
        ("note_folders", "LOWER(path)", "deleted = FALSE"),
    )
    for table_name, key_expression, predicate in collision_checks:
        where_clause = f" WHERE {predicate}" if predicate else ""
        collisions = backend.execute(
            f"""
                SELECT COUNT(*)
                  FROM (
                    SELECT client_id, {key_expression}
                      FROM {table_name}{where_clause}
                     GROUP BY client_id, {key_expression}
                    HAVING COUNT(*) > 1
                  ) duplicate_owner_keys
                """,  # nosec B608 -- all SQL fragments are fixed migration constants.
            connection=conn,
        ).scalar
        if int(collisions or 0):
            raise SchemaError(  # noqa: TRY003
                f"Cannot migrate {table_name} owner-scoped uniqueness: "
                f"{int(collisions)} colliding owner/key group(s)."
            )

    ownership_checks = (
        (
            "keyword collection hierarchy",
            """
                SELECT COUNT(*)
                  FROM keyword_collections AS child
                  JOIN keyword_collections AS parent ON parent.id = child.parent_id
                 WHERE child.client_id IS DISTINCT FROM parent.client_id
                """,
        ),
        (
            "folder hierarchy",
            """
                SELECT COUNT(*)
                  FROM note_folders AS child
                  JOIN note_folders AS parent ON parent.id = child.parent_id
                 WHERE child.client_id IS DISTINCT FROM parent.client_id
                """,
        ),
        (
            "note keyword link",
            """
                SELECT COUNT(*)
                  FROM note_keywords AS link
                  JOIN notes AS note ON note.id = link.note_id
                  JOIN chacha_keywords AS keyword ON keyword.id = link.keyword_id
                 WHERE note.client_id IS DISTINCT FROM keyword.client_id
                """,
        ),
        (
            "conversation keyword link",
            """
                SELECT COUNT(*)
                  FROM conversation_keywords AS link
                  JOIN conversations AS conversation ON conversation.id = link.conversation_id
                  JOIN chacha_keywords AS keyword ON keyword.id = link.keyword_id
                 WHERE conversation.client_id IS DISTINCT FROM keyword.client_id
                """,
        ),
        (
            "collection keyword link",
            """
                SELECT COUNT(*)
                  FROM collection_keywords AS link
                  JOIN keyword_collections AS collection ON collection.id = link.collection_id
                  JOIN chacha_keywords AS keyword ON keyword.id = link.keyword_id
                 WHERE collection.client_id IS DISTINCT FROM keyword.client_id
                """,
        ),
        (
            "note folder membership",
            """
                SELECT COUNT(*)
                  FROM note_folder_memberships AS link
                  JOIN notes AS note ON note.id = link.note_id
                  JOIN note_folders AS folder ON folder.id = link.folder_id
                 WHERE note.client_id IS DISTINCT FROM folder.client_id
                """,
        ),
        (
            "note folder source membership",
            """
                SELECT COUNT(*)
                  FROM note_folder_source_memberships AS link
                  JOIN notes AS note ON note.id = link.note_id
                  JOIN note_folders AS folder ON folder.id = link.folder_id
                 WHERE note.client_id IS DISTINCT FROM folder.client_id
                """,
        ),
        (
            "note folder suppression",
            """
                SELECT COUNT(*)
                  FROM note_folder_sync_suppressions AS link
                  JOIN notes AS note ON note.id = link.note_id
                  JOIN note_folders AS folder ON folder.id = link.folder_id
                 WHERE note.client_id IS DISTINCT FROM folder.client_id
                """,
        ),
    )
    for relationship_name, statement in ownership_checks:
        cross_owner_rows = backend.execute(statement, connection=conn).scalar
        if int(cross_owner_rows or 0):
            raise SchemaError(  # noqa: TRY003
                f"Cannot migrate {relationship_name}: {int(cross_owner_rows)} "
                "cross-owner relation(s) require authoritative repair."
            )

    constraints = backend.execute(
        """
            SELECT rel.relname AS table_name,
                   constraint_row.conname,
                   ARRAY(
                     SELECT attribute_row.attname
                       FROM unnest(constraint_row.conkey) WITH ORDINALITY AS key(attnum, ord)
                       JOIN pg_attribute attribute_row
                         ON attribute_row.attrelid = constraint_row.conrelid
                        AND attribute_row.attnum = key.attnum
                      ORDER BY key.ord
                   ) AS columns
              FROM pg_constraint constraint_row
              JOIN pg_class rel ON rel.oid = constraint_row.conrelid
              JOIN pg_namespace namespace_row ON namespace_row.oid = rel.relnamespace
             WHERE constraint_row.contype = 'u'
               AND namespace_row.nspname = current_schema()
               AND rel.relname IN ('chacha_keywords', 'keyword_collections', 'note_folders')
            """,
        connection=conn,
    ).rows
    natural_key_columns = {
        "chacha_keywords": ["keyword"],
        "keyword_collections": ["name"],
        "note_folders": ["path"],
    }
    for constraint in constraints:
        table_name = str(constraint["table_name"])
        columns = [str(column) for column in (constraint.get("columns") or [])]
        if columns != natural_key_columns.get(table_name):
            continue
        quoted_table = postgres_schema._quoted_postgres_identifier(table_name)
        quoted_constraint = postgres_schema._quoted_postgres_identifier(str(constraint["conname"]))
        backend.execute(
            f"ALTER TABLE {quoted_table} DROP CONSTRAINT IF EXISTS {quoted_constraint}",  # nosec B608
            connection=conn,
        )

    for index_name in (
        "idx_keywords_sync_id_unique",
        "idx_keyword_collections_sync_id_unique",
        "idx_note_folders_sync_id_unique",
        "idx_note_folders_path_lower",
    ):
        backend.execute(
            f"DROP INDEX IF EXISTS {postgres_schema._quoted_postgres_identifier(index_name)}",  # nosec B608
            connection=conn,
        )

    owner_indexes = (
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_keywords_sync_id_unique "
        "ON chacha_keywords(client_id, sync_id)",
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_keywords_keyword_lower_unique "
        "ON chacha_keywords(client_id, LOWER(keyword))",
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_keyword_collections_sync_id_unique "
        "ON keyword_collections(client_id, sync_id)",
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_keyword_collections_name_lower_unique "
        "ON keyword_collections(client_id, LOWER(name))",
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_note_folders_sync_id_unique "
        "ON note_folders(client_id, sync_id)",
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_note_folders_path_lower "
        "ON note_folders(client_id, LOWER(path)) WHERE deleted = FALSE",
    )
    for statement in owner_indexes:
        backend.execute(statement, connection=conn)

    for table_name in forced_rls_tables:
        backend.execute(
            f"ALTER TABLE {postgres_schema._quoted_postgres_identifier(table_name)} "  # nosec B608
            "FORCE ROW LEVEL SECURITY",
            connection=conn,
        )
    postgres_schema._set_schema_version_postgres(ctx, conn, 57)


def _migrate_from_v57_to_v58_postgres(ctx, conn: Any) -> None:
    """Migrate shared PostgreSQL Notes links under verified schema authority."""

    backend = ctx.backend
    locked_tables = (
        "notes",
        "note_edges",
        "chacha_keywords",
        "note_keywords",
        "conversations",
    )
    backend.execute(
        "LOCK TABLE notes, note_edges, chacha_keywords, note_keywords, conversations "
        "IN ACCESS EXCLUSIVE MODE",
        connection=conn,
    )
    states = backend.execute(
        """
            SELECT table_row.relname AS table_name,
                   table_row.relrowsecurity,
                   table_row.relforcerowsecurity,
                   table_row.relowner = current_user::regrole AS is_schema_owner
              FROM pg_class AS table_row
              JOIN pg_namespace AS namespace_row ON namespace_row.oid = table_row.relnamespace
             WHERE namespace_row.nspname = current_schema()
               AND table_row.relkind IN ('r', 'p')
               AND table_row.relname IN (
                 'notes', 'note_edges', 'chacha_keywords', 'note_keywords', 'conversations'
               )
            """,
        connection=conn,
    ).rows
    state_by_table = {str(row["table_name"]): row for row in states}
    if set(state_by_table) != set(locked_tables) or any(
        not bool(state_by_table[table]["is_schema_owner"]) for table in locked_tables
    ):
        raise SchemaError(  # noqa: TRY003
            "Notes link v58 requires the verified PostgreSQL schema-owner migration path."
        )
    notes_state = state_by_table["notes"]
    notes_forced = bool(notes_state["relforcerowsecurity"])
    if notes_forced and not bool(notes_state["relrowsecurity"]):
        raise SchemaError("Notes link v58 cannot verify the notes RLS state.")  # noqa: TRY003
    if notes_forced:
        backend.execute("ALTER TABLE notes NO FORCE ROW LEVEL SECURITY", connection=conn)

    raw_rows = backend.execute(
        """
            SELECT edge.*,
                   source_note.client_id AS source_owner,
                   target_note.client_id AS target_owner
              FROM note_edges AS edge
              LEFT JOIN notes AS source_note ON source_note.id = edge.from_note_id
              LEFT JOIN notes AS target_note ON target_note.id = edge.to_note_id
             ORDER BY edge.edge_id
            """,
        connection=conn,
    ).rows
    canonical_rows = [
        migration_support._notes_link_v58_transform_row(
            row,
            source_owner=row.get("source_owner"),
            target_owner=row.get("target_owner"),
        )
        for row in raw_rows
    ]
    migration_support._validate_notes_link_v58_unique(canonical_rows)

    backend.execute(
        """
            ALTER TABLE note_edges
              ADD COLUMN IF NOT EXISTS label TEXT,
              ADD COLUMN IF NOT EXISTS properties JSONB,
              ADD COLUMN IF NOT EXISTS last_modified TIMESTAMPTZ,
              ADD COLUMN IF NOT EXISTS version INTEGER,
              ADD COLUMN IF NOT EXISTS deleted BOOLEAN,
              ADD COLUMN IF NOT EXISTS deleted_at TIMESTAMPTZ
            """,
        connection=conn,
    )
    canonical_payload = json.dumps(
        [
            {
                "edge_id": row["edge_id"],
                "weight": row["weight"],
                "label": row["label"],
                "properties": row["properties"],
            }
            for row in canonical_rows
        ],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    backend.execute(
        """
            UPDATE note_edges AS edge
               SET weight = staged.weight,
                   label = staged.label,
                   properties = staged.properties,
                   last_modified = edge.created_at,
                   version = 1,
                   deleted = FALSE,
                   deleted_at = NULL
              FROM jsonb_to_recordset(%s::jsonb) AS staged(
                edge_id TEXT, weight DOUBLE PRECISION, label TEXT, properties JSONB
              )
             WHERE edge.edge_id = staged.edge_id
            """,
        (canonical_payload,),
        connection=conn,
    )
    backend.execute(
        """
            ALTER TABLE note_edges
              ALTER COLUMN properties SET DEFAULT '{}'::jsonb,
              ALTER COLUMN properties SET NOT NULL,
              ALTER COLUMN weight SET DEFAULT 1.0,
              ALTER COLUMN weight SET NOT NULL,
              ALTER COLUMN last_modified SET DEFAULT CURRENT_TIMESTAMP,
              ALTER COLUMN last_modified SET NOT NULL,
              ALTER COLUMN version SET DEFAULT 1,
              ALTER COLUMN version SET NOT NULL,
              ALTER COLUMN deleted SET DEFAULT FALSE,
              ALTER COLUMN deleted SET NOT NULL
            """,
        connection=conn,
    )
    backend.execute(
        "ALTER TABLE note_edges DROP CONSTRAINT IF EXISTS note_edges_source_note_fkey",
        connection=conn,
    )
    backend.execute(
        "ALTER TABLE note_edges DROP CONSTRAINT IF EXISTS note_edges_target_note_fkey",
        connection=conn,
    )
    constraints = (
        (
            "note_edges_source_note_fkey",
            "FOREIGN KEY (from_note_id) REFERENCES notes(id) ON DELETE CASCADE ON UPDATE CASCADE",
        ),
        (
            "note_edges_target_note_fkey",
            "FOREIGN KEY (to_note_id) REFERENCES notes(id) ON DELETE CASCADE ON UPDATE CASCADE",
        ),
        ("note_edges_manual_type_check", "CHECK (type = 'manual')"),
        ("note_edges_directed_check", "CHECK (directed IN (0, 1))"),
        (
            "note_edges_weight_check",
            "CHECK (weight >= 0 AND weight <= 1000000 AND weight NOT IN "
            "('NaN'::float8, 'Infinity'::float8, '-Infinity'::float8))",
        ),
        ("note_edges_label_check", "CHECK (label IS NULL OR char_length(label) <= 256)"),
        ("note_edges_self_link_check", "CHECK (from_note_id <> to_note_id)"),
        ("note_edges_version_check", "CHECK (version >= 1)"),
        (
            "note_edges_id_check",
            "CHECK (edge_id ~ '^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$')",
        ),
        (
            "note_edges_source_id_check",
            "CHECK (from_note_id ~ '^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$')",
        ),
        (
            "note_edges_target_id_check",
            "CHECK (to_note_id ~ '^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$')",
        ),
        (
            "note_edges_created_by_check",
            "CHECK (created_by IS NOT NULL AND char_length(created_by) > 0)",
        ),
        (
            "note_edges_logical_identity_key",
            "UNIQUE (user_id, type, directed, from_note_id, to_note_id)",
        ),
    )
    for constraint_name, definition in constraints:
        backend.execute(
            f"""
                DO $migration$
                BEGIN
                  IF NOT EXISTS (
                    SELECT 1 FROM pg_constraint
                     WHERE conrelid = 'note_edges'::regclass
                       AND conname = '{constraint_name}'
                  ) THEN
                    ALTER TABLE note_edges ADD CONSTRAINT {constraint_name} {definition};
                  END IF;
                END
                $migration$
                """,  # nosec B608 -- names and definitions are fixed migration constants.
            connection=conn,
        )
    backend.execute(
        "CREATE INDEX IF NOT EXISTS idx_note_edges_owner_live "
        "ON note_edges(user_id, deleted, type, edge_id)",
        connection=conn,
    )
    backend.execute(
        "CREATE INDEX IF NOT EXISTS idx_note_edges_from_live "
        "ON note_edges(user_id, from_note_id, edge_id) WHERE deleted = FALSE",
        connection=conn,
    )
    backend.execute(
        "CREATE INDEX IF NOT EXISTS idx_note_edges_to_live "
        "ON note_edges(user_id, to_note_id, edge_id) WHERE deleted = FALSE",
        connection=conn,
    )
    postgres_schema._notes_graph_schema_postgres(ctx, conn)
    backend.execute(
        "INSERT INTO note_graph_dirty(owner_user_id, note_id, generation, last_modified) "
        "SELECT client_id, id, 1, CURRENT_TIMESTAMP FROM notes "
        "ON CONFLICT(owner_user_id, note_id) DO UPDATE SET "
        "generation = note_graph_dirty.generation + 1, last_modified = CURRENT_TIMESTAMP",
        connection=conn,
    )
    backend.execute(
        "INSERT INTO note_graph_projection_state(owner_user_id, parser_version, rebuild_state, "
        "rebuild_cursor, updated_at) "
        "SELECT DISTINCT client_id, 1, 'pending', NULL, CURRENT_TIMESTAMP FROM notes "
        "ON CONFLICT(owner_user_id) DO UPDATE SET parser_version = 1, "
        "rebuild_state = 'pending', rebuild_cursor = NULL, updated_at = CURRENT_TIMESTAMP",
        connection=conn,
    )
    if notes_forced:
        backend.execute("ALTER TABLE notes FORCE ROW LEVEL SECURITY", connection=conn)
    postgres_schema._set_schema_version_postgres(ctx, conn, 58)


def _migrate_from_v89_to_v90_postgres(ctx, conn: Any) -> None:
    """Install transcripts and forced owner/parent RLS before publishing the version."""
    from tldw_Server_API.app.core.DB_Management.chacha.persona_transcript_store import (
        persona_transcript_rls_sql,
        persona_transcript_schema_sql,
    )

    if postgres_schema._get_schema_version_postgres(ctx, conn) != 89:
        raise SchemaError("Persona transcript migration requires PostgreSQL version 89.")
    for statement in (
        persona_transcript_schema_sql(),
        "CREATE INDEX persona_transcript_owner_activity ON persona_session_transcripts(user_id,last_activity) WHERE turn_count>0",
        *persona_transcript_rls_sql(),
    ):
        ctx.backend.execute(statement, connection=conn)
    from tldw_Server_API.app.core.DB_Management.chacha.persona_transcript_store import (
        verify_persona_transcript_postgres,
    )

    verify_persona_transcript_postgres(
        ctx.backend, getattr(conn, "_connection", conn), ctx.normalize_postgres_catalog_expression
    )
    ctx.backend.execute(
        "UPDATE db_schema_version SET version=90 WHERE schema_name=%s AND version=89",
        (ctx.schema_name,),
        connection=conn,
    )
    if postgres_schema._get_schema_version_postgres(ctx, conn) != 90:
        raise SchemaError("Persona transcript PostgreSQL version transition failed.")


def _migrate_from_v62_to_v63_postgres(ctx, conn: Any) -> None:
    """Run the durable bounded PostgreSQL v63 migration, version last."""
    def begin_phase(label: str) -> bool:
        version = postgres_schema._begin_notes_moodboard_studio_v61_postgres_transaction(ctx,
            conn
        )
        if version == 63:
            postgres_schema._verify_notes_moodboard_studio_schema_postgres(ctx, conn)
            require_moodboard_migration_complete(postgres_schema._postgres_v61_progress_row(ctx, conn, "migration"))
            conn.commit()
            return False
        if version != 62:
            raise SchemaError(  # noqa: TRY003
                f"Notes moodboard/Studio v63 {label} observed schema version {version}."
            )
        return True

    if not begin_phase("schema phase"):
        return
    progress_exists = ctx.backend.table_exists(
        "chacha_schema_migration_progress", connection=conn
    )
    if progress_exists and postgres_schema._postgres_v61_phase_is_complete(ctx, conn, "schema"):
        conn.commit()
    else:
        ctx.backend.execute(
            "LOCK TABLE note_task_scope_authority, moodboards, moodboard_notes, "
            "note_studio_documents IN SHARE ROW EXCLUSIVE MODE",
            connection=conn,
        )
        for statement in postgres_schema._notes_moodboard_studio_v61_postgres_schema_sql():
            ctx.backend.execute(statement, connection=conn)
        postgres_schema._complete_postgres_v61_phase(ctx, conn, "schema")
        conn.commit()
        postgres_schema._notes_moodboard_studio_v61_postgres_checkpoint("schema")

    while True:
        if not begin_phase("moodboard identity allocation phase"):
            return
        progress = postgres_schema._postgres_v61_progress_row(ctx, conn, "moodboard_identities")
        if progress["status"] == "complete":
            conn.commit()
            break
        rows, cursor, source_exhausted = (
            postgres_schema._postgres_v61_allocate_board_identity_page(ctx,
                conn, progress=progress
            )
        )
        added, fingerprint = postgres_schema._postgres_v61_progress_fingerprint(
            str(progress["aggregate_fingerprint"]), rows
        )
        total = int(progress["copied_count"]) + added
        status = "complete" if source_exhausted else "running"
        postgres_schema._upsert_postgres_v61_progress(ctx,
            conn,
            phase="moodboard_identities",
            cursor=cursor,
            count=total,
            fingerprint=fingerprint,
            status=status,
        )
        conn.commit()
        postgres_schema._notes_moodboard_studio_v61_postgres_checkpoint(
            f"identity:moodboards:{total}"
        )
        if status == "complete":
            break

    for source_phase in (
        "moodboards", "moodboard_notes", "note_studio_documents",
    ):
        prediction_phase = f"source_prediction:{source_phase}"
        while True:
            if not begin_phase(f"{source_phase} source prediction phase"):
                return
            progress = postgres_schema._postgres_v61_progress_row(ctx, conn, prediction_phase)
            if progress["status"] == "complete":
                conn.commit()
                break
            page = postgres_schema._postgres_v61_fingerprint_phase_page(ctx,
                conn,
                source_phase=source_phase,
                progress=progress,
                predict=True,
            )
            postgres_schema._upsert_postgres_v61_progress(ctx,
                conn,
                phase=prediction_phase,
                cursor=page["cursor"],
                count=page["count"],
                fingerprint=page["fingerprint"],
                status=page["status"],
            )
            conn.commit()
            postgres_schema._notes_moodboard_studio_v61_postgres_checkpoint(
                f"{prediction_phase}:{page['count']}"
            )
            if page["status"] == "complete":
                break

    for phase in ("moodboards", "moodboard_notes", "note_studio_documents"):
        while True:
            if not begin_phase(f"{phase} copy phase"):
                return
            progress = postgres_schema._postgres_v61_progress_row(ctx, conn, phase)
            if progress["status"] == "complete":
                conn.commit()
                break
            rows, cursor, source_exhausted = postgres_schema._postgres_v61_copy_page(ctx,
                conn, phase=phase, progress=progress
            )
            added, fingerprint = postgres_schema._postgres_v61_progress_fingerprint(
                str(progress["aggregate_fingerprint"]), rows
            )
            total = int(progress["copied_count"]) + added
            status = "complete" if source_exhausted else "running"
            postgres_schema._upsert_postgres_v61_progress(ctx,
                conn, phase=phase, cursor=cursor, count=total,
                fingerprint=fingerprint, status=status,
            )
            conn.commit()
            postgres_schema._notes_moodboard_studio_v61_postgres_checkpoint(
                f"copy:{phase}:{total}"
            )
            if status == "complete":
                break

    # Prove every durable target field while the migration owner can still
    # read all legacy tenants; forced RLS is installed only after this
    # global aggregate succeeds.
    for source_phase in ("moodboards", "moodboard_notes", "note_studio_documents"):
        verification_phase = f"aggregate_verification:{source_phase}"
        while True:
            if not begin_phase(f"{source_phase} aggregate verification phase"):
                return
            progress = postgres_schema._postgres_v61_progress_row(ctx, conn, verification_phase)
            if progress["status"] == "complete":
                conn.commit()
                break
            page = postgres_schema._postgres_v61_fingerprint_phase_page(ctx,
                conn,
                source_phase=source_phase,
                progress=progress,
                predict=False,
            )
            postgres_schema._upsert_postgres_v61_progress(ctx,
                conn,
                phase=verification_phase,
                cursor=page["cursor"],
                count=page["count"],
                fingerprint=page["fingerprint"],
                status=page["status"],
            )
            conn.commit()
            postgres_schema._notes_moodboard_studio_v61_postgres_checkpoint(
                f"{verification_phase}:{page['count']}"
            )
            if page["status"] == "complete":
                break

    if not begin_phase("aggregate verification phase"):
        return
    if postgres_schema._postgres_v61_phase_is_complete(ctx, conn, "aggregate_verification"):
        conn.commit()
    else:
        for source_phase in (
            "moodboards",
            "moodboard_notes",
            "note_studio_documents",
        ):
            predicted = postgres_schema._postgres_v61_progress_row(ctx,
                conn, f"source_prediction:{source_phase}"
            )
            copied = postgres_schema._postgres_v61_progress_row(ctx, conn, source_phase)
            verified = postgres_schema._postgres_v61_progress_row(ctx,
                conn, f"aggregate_verification:{source_phase}"
            )
            if (
                predicted["status"] != "complete"
                or copied["status"] != "complete"
                or verified["status"] != "complete"
                or int(predicted["copied_count"])
                != int(copied["copied_count"])
                or int(copied["copied_count"])
                != int(verified["copied_count"])
                or predicted["aggregate_fingerprint"]
                != copied["aggregate_fingerprint"]
                or copied["aggregate_fingerprint"]
                != verified["aggregate_fingerprint"]
            ):
                raise SchemaError(
                    "Notes moodboard/Studio v61 aggregate verification failed."
                )  # noqa: TRY003
        postgres_schema._complete_postgres_v61_phase(ctx, conn, "aggregate_verification")
        conn.commit()
        postgres_schema._notes_moodboard_studio_v61_postgres_checkpoint(
            "aggregate_verification"
        )

    if not begin_phase("constraint phase"):
        return
    run_constraints = not postgres_schema._postgres_v61_phase_is_complete(ctx,
        conn, "constraints"
    )
    if run_constraints:
        ctx.backend.execute(
            "LOCK TABLE note_task_scope_authority, moodboards, moodboard_notes, "
            "note_studio_documents IN SHARE ROW EXCLUSIVE MODE",
            connection=conn,
        )
    constraint_statements = (
        "ALTER TABLE moodboards ALTER COLUMN owner_user_id SET NOT NULL",
        "ALTER TABLE moodboards ALTER COLUMN dataset_id SET NOT NULL",
        "ALTER TABLE moodboards ALTER COLUMN sync_id SET NOT NULL",
        "ALTER TABLE moodboards ALTER COLUMN canvas_json SET NOT NULL",
        "ALTER TABLE moodboards ALTER COLUMN canonical_revision SET NOT NULL",
        "ALTER TABLE moodboards ALTER COLUMN canonical_hash SET NOT NULL",
        "ALTER TABLE moodboard_notes ALTER COLUMN owner_user_id SET NOT NULL",
        "ALTER TABLE moodboard_notes ALTER COLUMN dataset_id SET NOT NULL",
        "ALTER TABLE moodboard_notes ALTER COLUMN placement_id SET NOT NULL",
        "ALTER TABLE moodboard_notes ALTER COLUMN x SET NOT NULL",
        "ALTER TABLE moodboard_notes ALTER COLUMN y SET NOT NULL",
        "ALTER TABLE moodboard_notes ALTER COLUMN width SET NOT NULL",
        "ALTER TABLE moodboard_notes ALTER COLUMN height SET NOT NULL",
        "ALTER TABLE moodboard_notes ALTER COLUMN order_index SET NOT NULL",
        "ALTER TABLE moodboard_notes ALTER COLUMN display_json SET NOT NULL",
        "ALTER TABLE moodboard_notes ALTER COLUMN last_modified SET NOT NULL",
        "ALTER TABLE moodboard_notes ALTER COLUMN deleted SET NOT NULL",
        "ALTER TABLE moodboard_notes ALTER COLUMN version SET NOT NULL",
        "ALTER TABLE moodboard_notes ALTER COLUMN canonical_revision SET NOT NULL",
        "ALTER TABLE moodboard_notes ALTER COLUMN canonical_hash SET NOT NULL",
        "ALTER TABLE note_studio_documents ALTER COLUMN owner_user_id SET NOT NULL",
        "ALTER TABLE note_studio_documents ALTER COLUMN dataset_id SET NOT NULL",
        "ALTER TABLE note_studio_documents ALTER COLUMN note_revision SET NOT NULL",
        "ALTER TABLE note_studio_documents ALTER COLUMN note_hash SET NOT NULL",
        "ALTER TABLE note_studio_documents ALTER COLUMN accepted_provenance_json SET NOT NULL",
        "ALTER TABLE note_studio_documents ALTER COLUMN deleted SET NOT NULL",
        "ALTER TABLE note_studio_documents ALTER COLUMN version SET NOT NULL",
        "ALTER TABLE note_studio_documents ALTER COLUMN canonical_revision SET NOT NULL",
        "ALTER TABLE note_studio_documents ALTER COLUMN canonical_hash SET NOT NULL",
        "ALTER TABLE moodboards ALTER COLUMN canonical_revision SET DEFAULT 1",
        "ALTER TABLE moodboard_notes ALTER COLUMN x SET DEFAULT 0",
        "ALTER TABLE moodboard_notes ALTER COLUMN y SET DEFAULT 0",
        "ALTER TABLE moodboard_notes ALTER COLUMN width SET DEFAULT 320",
        "ALTER TABLE moodboard_notes ALTER COLUMN height SET DEFAULT 220",
        "ALTER TABLE moodboard_notes ALTER COLUMN order_index SET DEFAULT 0",
        "ALTER TABLE moodboard_notes ALTER COLUMN display_json SET DEFAULT '{}'",
        "ALTER TABLE moodboard_notes ALTER COLUMN deleted SET DEFAULT FALSE",
        "ALTER TABLE moodboard_notes ALTER COLUMN version SET DEFAULT 1",
        "ALTER TABLE moodboard_notes ALTER COLUMN canonical_revision SET DEFAULT 1",
        "ALTER TABLE note_studio_documents ALTER COLUMN note_revision SET DEFAULT 1",
        "ALTER TABLE note_studio_documents ALTER COLUMN deleted SET DEFAULT FALSE",
        "ALTER TABLE note_studio_documents ALTER COLUMN version SET DEFAULT 1",
        "ALTER TABLE note_studio_documents ALTER COLUMN canonical_revision SET DEFAULT 1",
        "ALTER TABLE moodboards ADD CONSTRAINT moodboards_v61_scope_sync_unique UNIQUE(owner_user_id,dataset_id,sync_id)",
        "ALTER TABLE moodboards ADD CONSTRAINT moodboards_v61_scope_id_unique UNIQUE(owner_user_id,dataset_id,id)",
        "ALTER TABLE moodboards ADD CONSTRAINT moodboards_v61_owner_check CHECK(char_length(btrim(owner_user_id))>0) NOT VALID",
        "ALTER TABLE moodboards ADD CONSTRAINT moodboards_v61_dataset_check CHECK(char_length(btrim(dataset_id))>0) NOT VALID",
        "ALTER TABLE moodboards ADD CONSTRAINT moodboards_v61_sync_id_check CHECK(sync_id ~ '^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$') NOT VALID",
        "ALTER TABLE moodboards ADD CONSTRAINT moodboards_v61_canonical_revision_check CHECK(canonical_revision>=1) NOT VALID",
        "ALTER TABLE moodboards ADD CONSTRAINT moodboards_v61_canonical_hash_check CHECK(canonical_hash ~ '^sha256:[0-9a-f]{64}$') NOT VALID",
        "ALTER TABLE moodboards ADD CONSTRAINT moodboards_v61_diagnostic_code_check CHECK(source_diagnostic_code IS NULL OR char_length(source_diagnostic_code) BETWEEN 1 AND 64) NOT VALID",
        "ALTER TABLE moodboards ADD CONSTRAINT moodboards_v61_diagnostic_hash_check CHECK(source_diagnostic_hash IS NULL OR source_diagnostic_hash ~ '^sha256:[0-9a-f]{64}$') NOT VALID",
        "ALTER TABLE moodboard_notes ADD CONSTRAINT moodboard_notes_v61_scope_placement_unique UNIQUE(owner_user_id,dataset_id,placement_id)",
        "ALTER TABLE moodboard_notes ADD CONSTRAINT moodboard_notes_v61_owner_check CHECK(char_length(btrim(owner_user_id))>0) NOT VALID",
        "ALTER TABLE moodboard_notes ADD CONSTRAINT moodboard_notes_v61_dataset_check CHECK(char_length(btrim(dataset_id))>0) NOT VALID",
        "ALTER TABLE moodboard_notes ADD CONSTRAINT moodboard_notes_v61_note_id_check CHECK(char_length(btrim(note_id))>0) NOT VALID",
        "ALTER TABLE moodboard_notes ADD CONSTRAINT moodboard_notes_v61_placement_id_check CHECK(placement_id ~ '^notes[.]moodboard_note:sha256:[0-9a-f]{64}$') NOT VALID",
        "ALTER TABLE moodboard_notes ADD CONSTRAINT moodboard_notes_v61_width_check CHECK(width BETWEEN 1 AND 1000000) NOT VALID",
        "ALTER TABLE moodboard_notes ADD CONSTRAINT moodboard_notes_v61_height_check CHECK(height BETWEEN 1 AND 1000000) NOT VALID",
        "ALTER TABLE moodboard_notes ADD CONSTRAINT moodboard_notes_v61_version_check CHECK(version>=1) NOT VALID",
        "ALTER TABLE moodboard_notes ADD CONSTRAINT moodboard_notes_v61_canonical_revision_check CHECK(canonical_revision>=1) NOT VALID",
        "ALTER TABLE moodboard_notes ADD CONSTRAINT moodboard_notes_v61_canonical_hash_check CHECK(canonical_hash ~ '^sha256:[0-9a-f]{64}$') NOT VALID",
        "ALTER TABLE moodboard_notes ADD CONSTRAINT moodboard_notes_v61_diagnostic_code_check CHECK(source_diagnostic_code IS NULL OR char_length(source_diagnostic_code) BETWEEN 1 AND 64) NOT VALID",
        "ALTER TABLE moodboard_notes ADD CONSTRAINT moodboard_notes_v61_diagnostic_hash_check CHECK(source_diagnostic_hash IS NULL OR source_diagnostic_hash ~ '^sha256:[0-9a-f]{64}$') NOT VALID",
        "ALTER TABLE note_studio_documents ADD CONSTRAINT note_studio_documents_v61_owner_check CHECK(char_length(btrim(owner_user_id))>0) NOT VALID",
        "ALTER TABLE note_studio_documents ADD CONSTRAINT note_studio_documents_v61_dataset_check CHECK(char_length(btrim(dataset_id))>0) NOT VALID",
        "ALTER TABLE note_studio_documents ADD CONSTRAINT note_studio_documents_v61_note_id_check CHECK(char_length(btrim(note_id))>0) NOT VALID",
        "ALTER TABLE note_studio_documents ADD CONSTRAINT note_studio_documents_v61_note_revision_check CHECK(note_revision>=1) NOT VALID",
        "ALTER TABLE note_studio_documents ADD CONSTRAINT note_studio_documents_v61_note_hash_check CHECK(note_hash ~ '^sha256:[0-9a-f]{64}$') NOT VALID",
        "ALTER TABLE note_studio_documents ADD CONSTRAINT note_studio_documents_v61_version_check CHECK(version>=1) NOT VALID",
        "ALTER TABLE note_studio_documents ADD CONSTRAINT note_studio_documents_v61_canonical_revision_check CHECK(canonical_revision>=1) NOT VALID",
        "ALTER TABLE note_studio_documents ADD CONSTRAINT note_studio_documents_v61_canonical_hash_check CHECK(canonical_hash ~ '^sha256:[0-9a-f]{64}$') NOT VALID",
        "ALTER TABLE note_studio_documents ADD CONSTRAINT note_studio_documents_v61_diagnostic_code_check CHECK(source_diagnostic_code IS NULL OR char_length(source_diagnostic_code) BETWEEN 1 AND 64) NOT VALID",
        "ALTER TABLE note_studio_documents ADD CONSTRAINT note_studio_documents_v61_diagnostic_hash_check CHECK(source_diagnostic_hash IS NULL OR source_diagnostic_hash ~ '^sha256:[0-9a-f]{64}$') NOT VALID",
        "ALTER TABLE moodboard_notes ADD CONSTRAINT moodboard_notes_v61_board_fk FOREIGN KEY(owner_user_id,dataset_id,moodboard_id) REFERENCES moodboards(owner_user_id,dataset_id,id) ON UPDATE CASCADE ON DELETE CASCADE NOT VALID",
        "ALTER TABLE moodboard_notes ADD CONSTRAINT moodboard_notes_v61_note_fk FOREIGN KEY(owner_user_id,note_id) REFERENCES notes(client_id,id) ON UPDATE CASCADE ON DELETE CASCADE NOT VALID",
        "ALTER TABLE note_studio_documents ADD CONSTRAINT note_studio_documents_v61_note_fk FOREIGN KEY(owner_user_id,note_id) REFERENCES notes(client_id,id) ON UPDATE CASCADE ON DELETE CASCADE NOT VALID",
    )
    if run_constraints:
        existing_constraints = {
            str(row["conname"])
            for row in ctx.backend.execute(
                "SELECT conname FROM pg_constraint k JOIN pg_namespace n ON n.oid=k.connamespace "
                "WHERE n.nspname=current_schema()", connection=conn,
            ).rows
        }
        for statement in constraint_statements:
            match = re.search(r"ADD CONSTRAINT ([a-z0-9_]+)", statement)
            if match and match.group(1) in existing_constraints:
                continue
            ctx.backend.execute(statement, connection=conn)
        postgres_schema._complete_postgres_v61_phase(ctx, conn, "constraints")
        conn.commit()
        postgres_schema._notes_moodboard_studio_v61_postgres_checkpoint("constraint")
    else:
        conn.commit()

    validation_constraints = tuple(
        statement.partition("ADD CONSTRAINT ")[2].partition(" ")[0]
        for statement in constraint_statements
        if "ADD CONSTRAINT " in statement and " NOT VALID" in statement
    )
    for constraint in validation_constraints:
        if not begin_phase(f"{constraint} validation phase"):
            return
        progress_phase = f"constraint_validation:{constraint}"
        if postgres_schema._postgres_v61_phase_is_complete(ctx, conn, progress_phase):
            conn.commit()
            continue
        table = (
            "note_studio_documents"
            if constraint.startswith("note_studio")
            else "moodboards"
            if constraint.startswith("moodboards_")
            else "moodboard_notes"
        )
        ctx.backend.execute(
            f"ALTER TABLE {table} VALIDATE CONSTRAINT {constraint}",  # nosec B608 - fixed names.
            connection=conn,
        )
        postgres_schema._complete_postgres_v61_phase(ctx, conn, progress_phase)
        conn.commit()
        postgres_schema._notes_moodboard_studio_v61_postgres_checkpoint(progress_phase)

    indexes = MOODBOARD_STUDIO_V61_INDEX_STATEMENTS
    for index, statement in enumerate(indexes):
        if not begin_phase(f"index {index} phase"):
            return
        progress_phase = f"index:{index}"
        if postgres_schema._postgres_v61_phase_is_complete(ctx, conn, progress_phase):
            conn.commit()
            continue
        ctx.backend.execute(statement, connection=conn)
        postgres_schema._complete_postgres_v61_phase(ctx, conn, progress_phase)
        conn.commit()
        postgres_schema._notes_moodboard_studio_v61_postgres_checkpoint(progress_phase)

    if not begin_phase("RLS phase"):
        return
    if postgres_schema._postgres_v61_phase_is_complete(ctx, conn, "rls"):
        conn.commit()
    else:
        ctx.backend.execute(
            "LOCK TABLE note_task_scope_authority, moodboards, moodboard_notes, "
            "note_studio_documents IN SHARE ROW EXCLUSIVE MODE",
            connection=conn,
        )
        for statement in build_chacha_rls_sql():
            if any(table in statement for table in schema_sql._NOTES_MOODBOARD_STUDIO_V61_RELATIONS):
                ctx.backend.execute(statement, connection=conn)
        postgres_schema._complete_postgres_v61_phase(ctx, conn, "rls")
        conn.commit()
        postgres_schema._notes_moodboard_studio_v61_postgres_checkpoint("rls")

    if not begin_phase("version phase"):
        return
    ctx.backend.execute(
        "LOCK TABLE note_task_scope_authority, moodboards, moodboard_notes, "
        "note_studio_documents IN SHARE MODE",
        connection=conn,
    )
    postgres_schema._verify_notes_moodboard_studio_schema_postgres(ctx, conn)
    postgres_schema._postgres_v61_progress_row(ctx, conn, "migration")
    postgres_schema._upsert_postgres_v61_progress(ctx,
        conn, phase="migration", cursor=None, count=0,
        fingerprint=f"sha256:{hashlib.sha256().hexdigest()}", status="complete",
    )
    postgres_schema._set_schema_version_postgres(ctx, conn, 63)
    conn.commit()
    postgres_schema._notes_moodboard_studio_v61_postgres_checkpoint("version")


def _migrate_from_v58_to_v59_postgres(ctx, conn: Any) -> None:
    """Install the empty registry under verified PostgreSQL schema authority."""

    backend = ctx.backend
    backend.execute("LOCK TABLE notes IN ACCESS EXCLUSIVE MODE", connection=conn)
    states = backend.execute(
        """
            SELECT table_row.relname AS table_name,
                   table_row.relrowsecurity,
                   table_row.relforcerowsecurity,
                   table_row.relowner = current_user::regrole AS is_schema_owner,
                   pg_has_role(current_user, namespace_row.nspowner, 'USAGE')
                     AS is_current_schema_owner
              FROM pg_class AS table_row
              JOIN pg_namespace AS namespace_row ON namespace_row.oid = table_row.relnamespace
             WHERE namespace_row.nspname = current_schema()
               AND table_row.relkind IN ('r', 'p')
               AND table_row.relname = 'notes'
            """,
        connection=conn,
    ).rows
    if len(states) != 1 or str(states[0].get("table_name")) != "notes":
        raise SchemaError("Notes attachment v59 cannot verify the notes relation.")  # noqa: TRY003
    notes_state = states[0]
    if not bool(notes_state.get("is_schema_owner")) or not bool(
        notes_state.get("is_current_schema_owner")
    ):
        raise SchemaError(  # noqa: TRY003
            "Notes attachment v59 requires the verified PostgreSQL schema-owner migration path."
        )
    notes_rls_enabled = bool(notes_state.get("relrowsecurity"))
    notes_rls_forced = bool(notes_state.get("relforcerowsecurity"))
    if notes_rls_forced and not notes_rls_enabled:
        raise SchemaError("Notes attachment v59 cannot verify the notes RLS state.")  # noqa: TRY003
    if backend.table_exists("note_attachments", connection=conn):
        raise SchemaError(  # noqa: TRY003
            "Notes attachment v59 registry collision requires explicit repair."
        )
    if notes_rls_forced:
        backend.execute("ALTER TABLE notes NO FORCE ROW LEVEL SECURITY", connection=conn)

    postgres_schema._create_note_attachment_schema_postgres(ctx, conn)

    if notes_rls_forced:
        backend.execute("ALTER TABLE notes FORCE ROW LEVEL SECURITY", connection=conn)
    chacha_rls = build_chacha_rls_sql()
    notes_rls = [
        statement
        for statement in chacha_rls
        if "ALTER TABLE IF EXISTS notes " in statement
        or "DROP POLICY IF EXISTS notes_tenant_isolation ON notes" in statement
        or "CREATE POLICY notes_tenant_isolation ON notes" in statement
    ]
    attachment_rls = [
        statement
        for statement in chacha_rls
        if "note_attachments" in statement
    ]
    if len(notes_rls) != 4 or len(attachment_rls) != 4:
        raise SchemaError("Notes attachment v59 RLS definition is not canonical.")  # noqa: TRY003
    for statement in (*notes_rls, *attachment_rls):
        backend.execute(statement, connection=conn)
    postgres_schema._verify_note_attachment_schema_postgres(ctx, conn)
    postgres_schema._set_schema_version_postgres(ctx, conn, 59)


def _migrate_from_v59_to_v60_postgres(ctx, conn: Any) -> None:
    """Atomically rebuild the PostgreSQL task graph under owner/dataset tenancy."""
    backend = ctx.backend
    backend.execute(
        "LOCK TABLE notes, note_tasks, task_note_projections, task_events, "
        "task_event_read_state, note_task_reconciliation_state IN ACCESS EXCLUSIVE MODE",
        connection=conn,
    )
    source_relations = ("notes", *schema_sql._NOTE_TASK_V60_TABLES[:-1])
    relation_rows = backend.execute(
        """
            SELECT table_row.relname AS table_name, table_row.relrowsecurity AS rls_enabled,
                   table_row.relforcerowsecurity AS rls_forced,
                   table_row.relowner=current_user::regrole AS is_table_owner,
                   pg_has_role(current_user, namespace_row.nspowner, 'USAGE')
                     AS is_schema_owner
              FROM pg_class AS table_row
              JOIN pg_namespace AS namespace_row ON namespace_row.oid=table_row.relnamespace
             WHERE namespace_row.nspname=current_schema() AND table_row.relkind IN ('r','p')
               AND table_row.relname = ANY(%s)
            """,
        (list(source_relations),),
        connection=conn,
    ).rows
    policy_rows = backend.execute(
        """
            SELECT tablename AS table_name, policyname AS policy_name, permissive,
                   roles::text AS roles, cmd AS command, qual AS using_expression,
                   with_check AS check_expression
              FROM pg_policies
             WHERE schemaname=current_schema() AND tablename = ANY(%s)
             ORDER BY tablename,policyname
            """,
        (list(source_relations),),
        connection=conn,
    ).rows
    forced_relations = postgres_schema._verify_note_task_v59_authority_catalog(
        relation_rows, policy_rows
    )
    collision_names = (
        "task_projection_drifts",
        schema_sql._NOTE_TASK_SCOPE_AUTHORITY_TABLE,
        *(f"{table}_v60" for table in schema_sql._NOTE_TASK_V60_RELATIONS),
    )
    collisions = backend.execute(
        """
            SELECT table_row.relname AS relation_name
              FROM pg_class AS table_row
              JOIN pg_namespace AS namespace_row ON namespace_row.oid=table_row.relnamespace
             WHERE namespace_row.nspname=current_schema() AND table_row.relname = ANY(%s)
            """,
        (list(collision_names),),
        connection=conn,
    ).rows
    if collisions:
        raise SchemaError("Notes task v60 target collision requires explicit repair.")  # noqa: TRY003
    note_index_collision = backend.execute(
        "SELECT 1 FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace "
        "WHERE n.nspname=current_schema() AND c.relname='uq_notes_owner_id'",
        connection=conn,
    ).rows
    if note_index_collision:
        raise SchemaError("Notes task v60 notes-owner index collision requires explicit repair.")  # noqa: TRY003

    for table in forced_relations:
        backend.execute(f"ALTER TABLE {table} NO FORCE ROW LEVEL SECURITY", connection=conn)  # nosec B608
    source = postgres_schema._validate_note_task_source_postgres(ctx, conn)
    migration_support._note_task_v60_migration_checkpoint("validate")

    postgres_schema._rename_note_task_v59_postgres_constraints(ctx, conn)
    postgres_schema._create_note_task_schema_v60_postgres(ctx, conn)
    migration_support._note_task_v60_migration_checkpoint("create")

    dataset = schema_sql._LOCAL_UNBOUND_TASK_DATASET_ID
    for row in source["tasks"]:
        owner, _note_id = source["task_scopes"][str(row["id"])]
        canonical_source = ctx.note_task_v60_json_safe(row)
        canonical_hash, diagnostic_code, diagnostic_hash = ctx.canonicalize_legacy_task_v60(
            canonical_source, owner_user_id=owner
        )
        backend.execute(
            """
                INSERT INTO note_tasks_v60(
                  owner_user_id,dataset_id,id,note_id,text,status,metadata_json,projection_status,
                  deleted,created_at,updated_at,completed_at,client_id,version,canonical_revision,
                  canonical_hash,source_diagnostic_code,source_diagnostic_hash
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                """,
            (
                owner, dataset, row["id"], row["note_id"], row["text"], row["status"],
                row["metadata_json"], row["projection_status"], row["deleted"], row["created_at"],
                row["updated_at"], row["completed_at"], row["client_id"], row["version"],
                row["version"], canonical_hash, diagnostic_code, diagnostic_hash,
            ),
            connection=conn,
        )
    for row in source["projections"]:
        owner, _note_id = source["task_scopes"][str(row["task_id"])]
        backend.execute(
            """
                INSERT INTO task_note_projections_v60(
                  owner_user_id,dataset_id,task_id,note_id,note_version,line_number,start_offset,
                  end_offset,normalized_text_hash,occurrence_index,block_fingerprint,raw_line,
                  has_child_content,projection_status,updated_at
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                """,
            (owner, dataset, *tuple(row.values())),
            connection=conn,
        )
    for row, owner, note_id, source_hash in source["events"]:
        backend.execute(
            """
                INSERT INTO task_events_v60(
                  owner_user_id,dataset_id,id,task_id,note_id,event_type,actor_type,actor_id,tool_name,
                  policy_mode,approval_id,old_value_json,new_value_json,created_at,client_id,sync_revision,
                  sync_object_hash,sync_server_cursor,source_device_id,client_occurred_at,source_kind,
                  corrects_activity_id,deleted,deleted_at,delete_reason,source_diagnostic_code,
                  source_diagnostic_hash
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                """,
            (
                owner, dataset, row["id"], row["task_id"], note_id, row["event_type"],
                row["actor_type"], row["actor_id"], row["tool_name"], row["policy_mode"],
                row["approval_id"], row["old_value_json"], row["new_value_json"],
                row["created_at"], row["client_id"], 1, source_hash, None, None,
                row["created_at"], "trusted_bootstrap_v1", None, False, None, None,
                "legacy_task_activity_unverified", source_hash,
            ),
            connection=conn,
        )
    event_owners = {str(row[0]["id"]): row[1] for row in source["events"]}
    for row in source["read_states"]:
        backend.execute(
            "INSERT INTO task_event_read_state_v60 VALUES (%s,%s,%s,%s,%s,%s)",
            (event_owners[str(row["event_id"])], dataset, *tuple(row.values())),
            connection=conn,
        )
    for row in source["reconciliation"]:
        backend.execute(
            "INSERT INTO note_task_reconciliation_state_v60 VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)",
            (source["notes"][str(row["note_id"])], dataset, *tuple(row.values())),
            connection=conn,
        )
    migration_support._note_task_v60_migration_checkpoint("copy")

    verification_queries = {
        "note_tasks": (source["tasks"], "id"),
        "task_note_projections": (source["projections"], "task_id"),
        "task_events": ([row[0] for row in source["events"]], "id"),
        "task_event_read_state": (source["read_states"], "event_id,user_id"),
        "note_task_reconciliation_state": (source["reconciliation"], "note_id"),
    }
    for table, (source_rows, ordering) in verification_queries.items():
        source_columns = tuple(source_rows[0]) if source_rows else {
            "note_tasks": ("id","note_id","text","status","metadata_json","projection_status","deleted","created_at","updated_at","completed_at","client_id","version"),
            "task_note_projections": ("task_id","note_id","note_version","line_number","start_offset","end_offset","normalized_text_hash","occurrence_index","block_fingerprint","raw_line","has_child_content","projection_status","updated_at"),
            "task_events": ("id","task_id","note_id","event_type","actor_type","actor_id","tool_name","policy_mode","approval_id","old_value_json","new_value_json","created_at","client_id"),
            "task_event_read_state": ("event_id","user_id","read_at","dismissed_at"),
            "note_task_reconciliation_state": ("note_id","note_version","status","reconciled_at","item_count","warning_count","cursor"),
        }[table]
        target_rows = [
            dict(row) for row in backend.execute(
                f"SELECT {','.join(source_columns)} FROM {table}_v60 ORDER BY {ordering}",  # nosec B608
                connection=conn,
            ).rows
        ]
        if len(source_rows) != len(target_rows) or ctx.note_task_v60_hash(
            ctx.note_task_v60_json_safe(source_rows)
        ) != ctx.note_task_v60_hash(ctx.note_task_v60_json_safe(target_rows)):
            raise SchemaError(f"Notes task v60 source verification failed for {table}.")  # noqa: TRY003

    for table in (
        "task_event_read_state", "task_note_projections", "note_task_reconciliation_state",
        "task_events", "note_tasks",
    ):
        backend.execute(f"DROP TABLE {table}", connection=conn)  # nosec B608
    for table in schema_sql._NOTE_TASK_V60_RELATIONS:
        backend.execute(f"ALTER TABLE {table}_v60 RENAME TO {table}", connection=conn)  # nosec B608
    for statement in postgres_schema._note_task_v60_postgres_indexes():
        backend.execute(statement, connection=conn)
    migration_support._note_task_v60_migration_checkpoint("index")

    for table in forced_relations:
        backend.execute(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY", connection=conn)  # nosec B608
    task_rls: list[str] = []
    for statement in build_chacha_rls_sql():
        if any(
            f"ALTER TABLE IF EXISTS {table} " in statement
            or f"DROP POLICY IF EXISTS {table}_tenant_isolation ON {table}" in statement
            or f"CREATE POLICY {table}_tenant_isolation ON {table}" in statement
            for table in schema_sql._NOTE_TASK_V60_RELATIONS
        ):
            task_rls.append(statement)
    if len(task_rls) != 4 * len(schema_sql._NOTE_TASK_V60_RELATIONS):
        raise SchemaError("Notes task v60 RLS definition is not canonical.")  # noqa: TRY003
    for statement in task_rls:
        backend.execute(statement, connection=conn)
    postgres_schema._verify_note_task_schema_postgres(ctx, conn)
    migration_support._note_task_v60_migration_checkpoint("verify")
    version_cursor = backend.execute(
        "UPDATE db_schema_version SET version=60 WHERE schema_name=%s AND version=59",
        (ctx.schema_name,),
        connection=conn,
    )
    if version_cursor.rowcount != 1 or postgres_schema._get_schema_version_postgres(ctx, conn) != 60:
        raise SchemaError("Notes task v60 PostgreSQL migration failed version verification.")  # noqa: TRY003


def _migrate_from_v60_to_v61_postgres(ctx, conn: Any) -> None:
    """Add recipient-owned shared chat state with verified forced RLS."""
    if postgres_schema._get_schema_version_postgres(ctx, conn) != 60:
        raise SchemaError("Shared workspace chat v61 migration requires schema version 60.")  # noqa: TRY003
    migration_script = "\n".join(
        (
            schema_sql._MIGRATION_SQL_V60_TO_V61_POSTGRES,
            *build_shared_workspace_chat_rls_sql(),
            schema_sql._SHARED_WORKSPACE_CHAT_V61_POSTGRES_VERIFY_SQL,
        )
    )
    postgres_schema._apply_postgres_migration_script(ctx,
        migration_script,
        conn,
        expected_version=61,
    )
    if postgres_schema._get_schema_version_postgres(ctx, conn) != 61:
        raise SchemaError("Shared workspace chat v61 PostgreSQL version verification failed.")  # noqa: TRY003


def _migrate_from_v61_to_v62_postgres(ctx, conn: Any) -> None:
    """Add operation-fenced staged Workspace clone lifecycle markers."""
    if postgres_schema._get_schema_version_postgres(ctx, conn) != 61:
        raise SchemaError("Workspace clone lifecycle v62 migration requires schema version 61.")  # noqa: TRY003
    postgres_schema._apply_postgres_migration_script(ctx,
        schema_sql._MIGRATION_SQL_V61_TO_V62_POSTGRES,
        conn,
        expected_version=62,
    )
    if postgres_schema._get_schema_version_postgres(ctx, conn) != 62:
        raise SchemaError("Workspace clone lifecycle v62 PostgreSQL version verification failed.")  # noqa: TRY003


def _migrate_from_v63_to_v64_postgres(ctx, conn: Any) -> None:
    """Add durable, forced-RLS Notes graph suggestion persistence."""
    if postgres_schema._get_schema_version_postgres(ctx, conn) != 63:
        raise SchemaError("Notes graph suggestion v64 PostgreSQL migration requires schema version 63.")  # noqa: TRY003
    postgres_schema._apply_postgres_migration_script(ctx,
        schema_sql._MIGRATION_SQL_V63_TO_V64_POSTGRES,
        conn,
        expected_version=64,
    )
    postgres_schema._configure_note_graph_suggestion_receipt_delete_trigger_postgres(ctx, conn)
    if postgres_schema._get_schema_version_postgres(ctx, conn) != 64:
        raise SchemaError("Notes graph suggestion v64 PostgreSQL version verification failed.")  # noqa: TRY003


def _migrate_from_v85_to_v86_postgres(ctx, conn: Any) -> None:
    """Add durable, forced-RLS Notes semantic-index persistence."""
    if postgres_schema._get_schema_version_postgres(ctx, conn) != 85:
        raise SchemaError("Notes semantic v86 PostgreSQL migration requires schema version 85.")  # noqa: TRY003
    postgres_schema._apply_postgres_migration_script(ctx,
        schema_sql._MIGRATION_SQL_V85_TO_V86_POSTGRES,
        conn,
        expected_version=86,
    )
    if postgres_schema._get_schema_version_postgres(ctx, conn) != 86:
        raise SchemaError("Notes semantic v86 PostgreSQL version verification failed.")  # noqa: TRY003


def _migrate_from_v86_to_v87_postgres(ctx, conn: Any) -> None:
    """Add forced-RLS crash-durable semantic cleanup authority."""
    if postgres_schema._get_schema_version_postgres(ctx, conn) != 86:
        raise SchemaError("Notes semantic v87 PostgreSQL migration requires schema version 86.")  # noqa: TRY003
    postgres_schema._apply_postgres_migration_script(ctx,
        schema_sql._MIGRATION_SQL_V86_TO_V87_POSTGRES,
        conn,
        expected_version=87,
    )
    if postgres_schema._get_schema_version_postgres(ctx, conn) != 87:
        raise SchemaError("Notes semantic v87 PostgreSQL version verification failed.")  # noqa: TRY003


def _migrate_from_v87_to_v88_postgres(ctx, conn: Any) -> None:
    """Add forced-RLS operation receipts and resolved-model authority."""
    if postgres_schema._get_schema_version_postgres(ctx, conn) != 87:
        raise SchemaError("Notes semantic v88 PostgreSQL migration requires schema version 87.")
    postgres_schema._apply_postgres_migration_script(ctx,
        schema_sql._MIGRATION_SQL_V87_TO_V88_POSTGRES,
        conn,
        expected_version=88,
    )
    if postgres_schema._get_schema_version_postgres(ctx, conn) != 88:
        raise SchemaError("Notes semantic v88 PostgreSQL version verification failed.")


def _migrate_from_v65_to_v66_postgres(ctx, conn: Any) -> None:
    """Create Buddy storage and its tenant policies in the caller transaction."""
    from tldw_Server_API.app.core.DB_Management.Buddy_DB import BUDDY_SCHEMA_SQL
    from tldw_Server_API.app.core.DB_Management.Buddy_Turns_DB import BUDDY_TURNS_SCHEMA_SQL

    for statement in split_sql_statements(BUDDY_SCHEMA_SQL + BUDDY_TURNS_SCHEMA_SQL):
        ctx.backend.execute(statement, connection=conn)
    for table in (
        "buddy_profiles",
        "buddy_assets",
        "buddy_attachments",
        "buddy_result_acknowledgements",
        "buddy_turn_owners",
        "buddy_turns",
    ):
        ctx.backend.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY", connection=conn)
        ctx.backend.execute(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY", connection=conn)
        ctx.backend.execute(
            f"CREATE POLICY {table}_tenant_isolation ON {table} "
            "USING (user_id = current_setting('app.current_user_id', true)) "
            "WITH CHECK (user_id = current_setting('app.current_user_id', true))",
            connection=conn,
        )
    ctx.backend.execute(
        "UPDATE db_schema_version SET version = %s WHERE schema_name = %s AND version = %s",
        (66, ctx.schema_name, 65),
        connection=conn,
    )


def _migrate_from_v75_to_v76_postgres(ctx, conn: Any) -> None:
    """Install native fork storage and direct-owner RLS with the version bump."""
    from tldw_Server_API.app.core.DB_Management.chacha.native_fork_schema import native_fork_schema_statements

    if not migration_support._native_fork_schema_is_installed(ctx, conn):
        for statement in native_fork_schema_statements(postgres=True):
            ctx.backend.execute(statement, connection=conn)
    postgres_schema._ensure_chacha_rls_postgres(ctx, conn)
    ctx.backend.execute(
        "UPDATE db_schema_version SET version = %s WHERE schema_name = %s AND version = 75",
        (76, ctx.schema_name), connection=conn,
    )
    if postgres_schema._get_schema_version_postgres(ctx, conn) != 76:
        raise SchemaError("Native fork PostgreSQL migration V75->V76 failed version verification.")  # noqa: TRY003


def _migrate_from_v84_to_v85_postgres(ctx, conn: Any) -> None:
    """Use server UUID issuance on all INSERT paths, preserving forced-owner RLS."""
    ctx.backend.execute(
        "ALTER TABLE conversations ADD COLUMN source_incarnation_id TEXT NOT NULL DEFAULT gen_random_uuid()::text",
        connection=conn,
    )
    ctx.backend.execute(
        """CREATE FUNCTION conversation_source_incarnation_guard() RETURNS trigger LANGUAGE plpgsql AS $$
            BEGIN
                IF TG_OP = 'INSERT' THEN
                    NEW.source_incarnation_id := gen_random_uuid()::text;
                ELSIF NEW.source_incarnation_id IS DISTINCT FROM OLD.source_incarnation_id THEN
                    RAISE EXCEPTION 'conversation_incarnation_immutable';
                END IF;
                RETURN NEW;
            END; $$""",
        connection=conn,
    )
    ctx.backend.execute(
        "CREATE TRIGGER conversation_incarnation_guard BEFORE INSERT OR UPDATE OF source_incarnation_id ON conversations FOR EACH ROW EXECUTE FUNCTION conversation_source_incarnation_guard()",
        connection=conn,
    )
    ctx.backend.execute(
        "UPDATE db_schema_version SET version = 85 WHERE schema_name = %s AND version = 84",
        (ctx.schema_name,),
        connection=conn,
    )
    if postgres_schema._get_schema_version_postgres(ctx, conn) != 85:
        raise SchemaError("Conversation incarnation migration failed version verification.")  # noqa: TRY003


def _migrate_from_v83_to_v84_postgres(ctx, conn: Any) -> None:
    """Add explicit roles without changing existing forced-owner RLS or rows."""
    ctx.backend.execute(
        "ALTER TABLE native_chat_asset_references ADD COLUMN role TEXT "
        "CHECK (role IN ('message_image','document_context','generated_image','reference_image'))",
        connection=conn,
    )
    ctx.backend.execute(
        "UPDATE db_schema_version SET version = 84 WHERE schema_name = %s AND version = 83",
        (ctx.schema_name,),
        connection=conn,
    )
    if postgres_schema._get_schema_version_postgres(ctx, conn) != 84:
        raise SchemaError("Native reference role migration failed version verification.")  # noqa: TRY003


def _migrate_from_v82_to_v83_postgres(ctx, conn: Any) -> None:
    """Retain the existing forced-owner RLS while adding frozen quota identity."""
    ctx.backend.execute('ALTER TABLE native_chat_quota_intents ADD COLUMN quota_owner_json TEXT', connection=conn)
    ctx.backend.execute('UPDATE db_schema_version SET version = 83 WHERE schema_name = %s AND version = 82', (ctx.schema_name,), connection=conn)
    if postgres_schema._get_schema_version_postgres(ctx, conn) != 83:
        raise SchemaError('Native quota descriptor migration failed version verification.')


def _migrate_from_v90_to_v91_postgres(ctx, conn: Any) -> None:
    """Preserve unowned legacy rows before publishing forced registry isolation."""
    if postgres_schema._get_schema_version_postgres(ctx, conn) != 90:
        raise SchemaError("Skill registry PostgreSQL owner migration requires schema version90.")
    from tldw_Server_API.app.core.DB_Management.chacha.skill_registry_schema import (
        postgres_skill_registry_statements,
    )

    for statement in postgres_skill_registry_statements():
        ctx.backend.execute(statement, connection=conn)
    from tldw_Server_API.app.core.DB_Management.chacha.skill_registry_schema import verify_postgres_skill_registry

    verify_postgres_skill_registry(
        ctx.backend, getattr(conn, "_connection", conn), ctx.normalize_postgres_catalog_expression
    )
    ctx.backend.execute(
        "UPDATE db_schema_version SET version = %s WHERE schema_name = %s AND version = 90",
        (91, ctx.schema_name),
        connection=conn,
    )
    if postgres_schema._get_schema_version_postgres(ctx, conn) != 91:
        raise SchemaError("Skill registry PostgreSQL owner migration failed version verification.")


def _migrate_from_v88_to_v89_postgres(ctx, conn: Any) -> None:
    """Quarantine unowned legacy dictionaries before publishing forced RLS."""
    if postgres_schema._get_schema_version_postgres(ctx, conn) != 88:
        raise SchemaError("Dictionary PostgreSQL owner migration requires schema version 88.")
    from tldw_Server_API.app.core.DB_Management.backends.pg_rls_policies import build_dictionary_rls_sql
    from tldw_Server_API.app.core.DB_Management.chacha.dictionary_schema import (
        postgres_dictionary_schema_statements,
    )

    for statement in (*postgres_dictionary_schema_statements(), *build_dictionary_rls_sql()):
        ctx.backend.execute(statement, connection=conn)
    postgres_schema._verify_dictionary_schema_postgres(ctx, conn)
    ctx.backend.execute(
        "UPDATE db_schema_version SET version = %s WHERE schema_name = %s AND version = 88",
        (89, ctx.schema_name), connection=conn,
    )
    if postgres_schema._get_schema_version_postgres(ctx, conn) != 89:
        raise SchemaError("Dictionary PostgreSQL owner migration failed version verification.")


def _migrate_from_v81_to_v82_postgres(ctx, conn: Any) -> None:
    """Install permanent local receipts with forced owner RLS before the version bump."""
    from tldw_Server_API.app.core.DB_Management.backends.pg_rls_policies import (
        build_notes_provenance_receipts_rls_sql,
    )
    from tldw_Server_API.app.core.DB_Management.chacha.note_provenance_store import (
        notes_provenance_receipts_schema_sql,
    )

    for statement in (notes_provenance_receipts_schema_sql(), *build_notes_provenance_receipts_rls_sql()):
        ctx.backend.execute(statement, connection=conn)
    ctx.backend.execute(
        "UPDATE db_schema_version SET version = %s WHERE schema_name = %s AND version = 81",
        (82, ctx.schema_name),
        connection=conn,
    )
    if postgres_schema._get_schema_version_postgres(ctx, conn) != 82:
        raise SchemaError("Knowledge provenance receipt PostgreSQL migration failed version verification.")


def _migrate_from_v80_to_v81_postgres(ctx, conn: Any) -> None:
    """Install owner-bound Knowledge evidence and forced RLS before version bump."""
    from tldw_Server_API.app.core.DB_Management.backends.pg_rls_policies import build_notes_provenance_rls_sql
    from tldw_Server_API.app.core.DB_Management.chacha.note_provenance_store import notes_provenance_schema_sql

    for statement in (notes_provenance_schema_sql(postgres=True), *build_notes_provenance_rls_sql()):
        ctx.backend.execute(statement, connection=conn)
    ctx.backend.execute("UPDATE db_schema_version SET version = %s WHERE schema_name = %s AND version = 80", (81, ctx.schema_name), connection=conn)
    if postgres_schema._get_schema_version_postgres(ctx, conn) != 81:
        raise SchemaError("Knowledge provenance PostgreSQL migration failed version verification.")


def _migrate_from_v78_to_v79_postgres(ctx, conn: Any) -> None:
    """Add PostgreSQL client-id create fingerprints in the caller's upgrade transaction."""
    for statement in split_sql_statements(schema_sql._MIGRATION_SQL_V78_TO_V79_POSTGRES):
        ctx.backend.execute(statement, connection=conn)
    if postgres_schema._get_schema_version_postgres(ctx, conn) != 79:
        raise SchemaError(  # noqa: TRY003
            "Conversation create fingerprint PostgreSQL migration V78->V79 failed version verification."
        )


def _migrate_from_v79_to_v80_postgres(ctx, conn: Any) -> None:
    """Clear the retired Sync placeholder persona from every owner's chats.

        One PostgreSQL database holds every owner's chats, and `conversations`
        has a forced tenant policy. Startup refuses a SUPERUSER or BYPASSRLS role
        for the content backend, but this migration can still run under an admin
        DSN, and such a role sees all rows. A table owner that does not
        bypass it would repair only the rows its session happens to see and then
        record the version, leaving the other owners' chats broken for good. So
        the policy is lifted for the owner inside this transaction, behind an
        ACCESS EXCLUSIVE lock that keeps every other session out of the table,
        and put back before the version bump.
        """
    backend = ctx.backend
    backend.execute("LOCK TABLE conversations IN ACCESS EXCLUSIVE MODE", connection=conn)
    states = backend.execute(
        """
            SELECT table_row.relrowsecurity,
                   table_row.relforcerowsecurity,
                   pg_has_role(current_user, table_row.relowner, 'USAGE') AS is_table_owner,
                   (SELECT role_row.rolsuper OR role_row.rolbypassrls
                      FROM pg_roles AS role_row
                     WHERE role_row.rolname = current_user) AS bypasses_rls
              FROM pg_class AS table_row
              JOIN pg_namespace AS namespace_row ON namespace_row.oid = table_row.relnamespace
             WHERE namespace_row.nspname = current_schema()
               AND table_row.relkind IN ('r', 'p')
               AND table_row.relname = 'conversations'
            """,
        connection=conn,
    ).rows
    if len(states) != 1:
        raise SchemaError("Sync placeholder repair v80 cannot verify the conversations relation.")  # noqa: TRY003
    state = states[0]
    policy_applies = bool(state.get("relrowsecurity")) and not bool(state.get("bypasses_rls"))
    is_table_owner = bool(state.get("is_table_owner"))
    if policy_applies and not is_table_owner:
        raise SchemaError(  # noqa: TRY003
            "Sync placeholder repair v80 requires the PostgreSQL schema-owner migration path."
        )
    lift_forced_policy = policy_applies and bool(state.get("relforcerowsecurity"))
    if lift_forced_policy:
        backend.execute("ALTER TABLE conversations NO FORCE ROW LEVEL SECURITY", connection=conn)
    backend.execute(schema_sql._RETIRED_SYNC_PLACEHOLDER_REPAIR_SQL, connection=conn)
    if lift_forced_policy:
        backend.execute("ALTER TABLE conversations FORCE ROW LEVEL SECURITY", connection=conn)
    backend.execute(
        "UPDATE db_schema_version SET version = %s WHERE schema_name = %s AND version = 79",
        (80, ctx.schema_name), connection=conn,
    )
    if postgres_schema._get_schema_version_postgres(ctx, conn) != 80:
        raise SchemaError(  # noqa: TRY003
            "Sync placeholder repair PostgreSQL migration V79->V80 failed version verification."
        )


def _migrate_from_v77_to_v78_postgres(ctx, conn: Any) -> None:
    """Install PostgreSQL receipts with forced owner RLS before the version bump."""
    from tldw_Server_API.app.core.DB_Management.backends.pg_rls_policies import build_workspace_chat_startup_rls_sql
    from tldw_Server_API.app.core.DB_Management.chacha.workspace_chat_startup_schema import (
        workspace_chat_startup_schema_statements,
    )

    for statement in (*workspace_chat_startup_schema_statements(postgres=True), *build_workspace_chat_startup_rls_sql()):
        ctx.backend.execute(statement, connection=conn)
    ctx.backend.execute("UPDATE db_schema_version SET version = %s WHERE schema_name = %s AND version = 77", (78, ctx.schema_name), connection=conn)
    if postgres_schema._get_schema_version_postgres(ctx, conn) != 78:
        raise SchemaError("Workspace startup PostgreSQL migration V77->V78 failed version verification.")


def _migrate_from_v74_to_v75_postgres(ctx, conn: Any) -> None:
    """Create the PostgreSQL projection store and protected nullable provenance."""
    statements = (
        schema_sql._HISTORY_PROJECTIONS_SCHEMA_SQL,
        "ALTER TABLE messages ADD COLUMN IF NOT EXISTS history_admission_json TEXT",
        """CREATE OR REPLACE FUNCTION history_projection_immutable() RETURNS TRIGGER AS $$
               BEGIN RAISE EXCEPTION 'History projections are immutable'; END;
               $$ LANGUAGE plpgsql""",
        "DROP TRIGGER IF EXISTS history_projection_immutable ON conversation_history_projections",
        "CREATE TRIGGER history_projection_immutable BEFORE UPDATE ON conversation_history_projections "
        "FOR EACH ROW EXECUTE FUNCTION history_projection_immutable()",
    )
    for statement in statements:
        ctx.backend.execute(statement, connection=conn)
    ctx.backend.execute(
        "UPDATE db_schema_version SET version = %s WHERE schema_name = %s AND version = 74",
        (75, ctx.schema_name), connection=conn,
    )
    if postgres_schema._get_schema_version_postgres(ctx, conn) != 75:
        raise SchemaError("History projection PostgreSQL migration V74->V75 failed version verification.")  # noqa: TRY003
    postgres_schema._ensure_chacha_rls_postgres(ctx, conn)


def _migrate_from_v66_to_v67_postgres(ctx, conn: Any) -> None:
    """Add PostgreSQL OSCE quiz storage in the caller transaction."""
    for statement in split_sql_statements(schema_sql._MIGRATION_SQL_V66_TO_V67_POSTGRES):
        ctx.backend.execute(statement, connection=conn)
    if postgres_schema._get_schema_version_postgres(ctx, conn) != 67:
        raise SchemaError("OSCE PostgreSQL migration V66->V67 failed version verification.")  # noqa: TRY003
    postgres_schema._sync_postgres_sequences(ctx, conn)


def _migrate_from_v76_to_v77_persona_companion_postgres(ctx, conn: Any) -> None:
    """Add Persona companion storage in the PostgreSQL migration transaction."""
    if postgres_schema._get_schema_version_postgres(ctx, conn) != 76:
        raise SchemaError("Persona companion migration requires PostgreSQL schema V76.")  # noqa: TRY003
    postgres_schema._apply_postgres_migration_script(ctx,
        schema_sql._MIGRATION_SQL_V76_TO_V77_PERSONA_COMPANION_POSTGRES,
        conn,
        expected_version=77,
    )
    if postgres_schema._get_schema_version_postgres(ctx, conn) != 77:
        raise SchemaError("Persona companion PostgreSQL migration failed version verification.")  # noqa: TRY003
    postgres_schema._ensure_chacha_rls_postgres(ctx, conn)


def _migrate_from_v69_to_v70_postgres(ctx, conn: Any) -> None:
    """Add local tombstone metadata in the existing PostgreSQL migration transaction."""
    if postgres_schema._get_schema_version_postgres(ctx, conn) != 69:
        raise SchemaError("Keyword survivor migration requires PostgreSQL schema V69.")  # noqa: TRY003
    ctx.backend.execute(
        "ALTER TABLE chacha_keywords ADD COLUMN merged_into_sync_id TEXT "
        "CONSTRAINT keyword_merge_tombstone CHECK (merged_into_sync_id IS NULL "
        "OR (deleted = TRUE AND merged_into_sync_id <> sync_id))",
        connection=conn,
    )
    ctx.backend.execute(
        "UPDATE db_schema_version SET version = %s WHERE schema_name = %s AND version = %s",
        (70, ctx.schema_name, 69), connection=conn,
    )
    if postgres_schema._get_schema_version_postgres(ctx, conn) != 70:
        raise SchemaError("Keyword survivor PostgreSQL migration failed version verification.")  # noqa: TRY003


def _migrate_from_v67_to_v68_postgres(ctx, conn: Any) -> None:
    """Reserve character names per owner without changing existing rows or IDs."""
    if postgres_schema._get_schema_version_postgres(ctx, conn) != 67:
        raise SchemaError("Character name migration requires PostgreSQL schema V67.")  # noqa: TRY003
    catalog_query = (
        "SELECT c.conname, array_agg(a.attname ORDER BY k.ordinality) AS columns "
        "FROM pg_constraint c "
        "CROSS JOIN LATERAL unnest(c.conkey) WITH ORDINALITY AS k(attnum, ordinality) "
        "JOIN pg_attribute a ON a.attrelid = c.conrelid AND a.attnum = k.attnum "
        "WHERE c.conrelid = 'character_cards'::regclass AND c.contype = 'u' "
        "GROUP BY c.oid, c.conname"
    )
    constraints = ctx.backend.execute(catalog_query, connection=conn).rows
    old_keys = [row for row in constraints if row["columns"] == ["name"]]
    new_key = "character_cards_client_id_name_key"
    if len(old_keys) != 1 or any(row["conname"] == new_key for row in constraints):
        raise SchemaError("Unexpected character name constraint catalog at PostgreSQL V67.")  # noqa: TRY003
    ctx.backend.execute(
        "ALTER TABLE character_cards ADD CONSTRAINT character_cards_client_id_name_key "
        "UNIQUE (client_id, name)",
        connection=conn,
    )
    old_key = ctx.backend.escape_identifier(old_keys[0]["conname"])
    ctx.backend.execute(
        f"ALTER TABLE character_cards DROP CONSTRAINT {old_key}",  # nosec B608 # Catalog-validated, quoted identifier
        connection=conn,
    )
    final_constraints = ctx.backend.execute(catalog_query, connection=conn).rows
    remaining_global_name_key = ctx.backend.execute(
        "SELECT EXISTS (SELECT 1 FROM pg_index i "
        "JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = i.indkey[0] "
        "WHERE i.indrelid = 'character_cards'::regclass AND i.indisunique "
        "AND i.indnkeyatts = 1 AND a.attname = 'name')",
        connection=conn,
    ).scalar
    if remaining_global_name_key or not any(
        row["conname"] == new_key and row["columns"] == ["client_id", "name"]
        for row in final_constraints
    ):
        raise SchemaError("Character owner/name constraint verification failed.")  # noqa: TRY003
    result = ctx.backend.execute(
        "UPDATE db_schema_version SET version = %s WHERE schema_name = %s AND version = %s RETURNING version",
        (68, ctx.schema_name, 67),
        connection=conn,
    )
    if result.rowcount != 1 or postgres_schema._get_schema_version_postgres(ctx, conn) != 68:
        raise SchemaError("Character name PostgreSQL migration V67->V68 failed version verification.")  # noqa: TRY003


def _migrate_from_v68_to_v69_postgres(ctx, conn: Any) -> None:
    """Reserve deck names per owner without changing records or tombstones."""
    if postgres_schema._get_schema_version_postgres(ctx, conn) != 68:
        raise SchemaError("Deck name migration requires PostgreSQL schema V68.")  # noqa: TRY003
    catalog_query = (
        "SELECT c.conname, array_agg(a.attname ORDER BY k.ordinality) AS columns "
        "FROM pg_constraint c "
        "CROSS JOIN LATERAL unnest(c.conkey) WITH ORDINALITY AS k(attnum, ordinality) "
        "JOIN pg_attribute a ON a.attrelid = c.conrelid AND a.attnum = k.attnum "
        "WHERE c.conrelid = 'decks'::regclass AND c.contype = 'u' "
        "GROUP BY c.oid, c.conname"
    )
    constraints = ctx.backend.execute(catalog_query, connection=conn).rows
    old_keys = [row for row in constraints if row["columns"] == ["name"]]
    new_key = "decks_client_id_name_key"
    if len(old_keys) != 1 or any(row["conname"] == new_key for row in constraints):
        raise SchemaError("Unexpected deck name constraint catalog at PostgreSQL V68.")  # noqa: TRY003
    ctx.backend.execute(
        "ALTER TABLE decks ADD CONSTRAINT decks_client_id_name_key UNIQUE (client_id, name)",
        connection=conn,
    )
    old_key = ctx.backend.escape_identifier(old_keys[0]["conname"])
    ctx.backend.execute(
        f"ALTER TABLE decks DROP CONSTRAINT {old_key}",  # nosec B608 # Catalog-validated, quoted identifier
        connection=conn,
    )
    final_constraints = ctx.backend.execute(catalog_query, connection=conn).rows
    remaining_global_name_key = ctx.backend.execute(
        "SELECT EXISTS (SELECT 1 FROM pg_index i "
        "JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = i.indkey[0] "
        "WHERE i.indrelid = 'decks'::regclass AND i.indisunique "
        "AND i.indnkeyatts = 1 AND a.attname = 'name')",
        connection=conn,
    ).scalar
    if remaining_global_name_key or not any(
        row["conname"] == new_key and row["columns"] == ["client_id", "name"]
        for row in final_constraints
    ):
        raise SchemaError("Deck owner/name constraint verification failed.")  # noqa: TRY003
    result = ctx.backend.execute(
        "UPDATE db_schema_version SET version = %s WHERE schema_name = %s AND version = %s RETURNING version",
        (69, ctx.schema_name, 68),
        connection=conn,
    )
    if result.rowcount != 1 or postgres_schema._get_schema_version_postgres(ctx, conn) != 69:
        raise SchemaError("Deck name PostgreSQL migration V68->V69 failed version verification.")  # noqa: TRY003


def _migrate_from_v72_to_v73_postgres(ctx, conn: Any) -> None:
    """Apply PostgreSQL opt-out storage and legacy backfill in the caller transaction."""
    for statement in split_sql_statements(schema_sql._MIGRATION_SQL_V72_TO_V73_POSTGRES):
        ctx.backend.execute(statement, connection=conn)
    if postgres_schema._get_schema_version_postgres(ctx, conn) != 73:
        raise SchemaError("Workspace Persona PostgreSQL migration V72->V73 failed version verification.")  # noqa: TRY003


def _migrate_from_v73_to_v74_postgres(ctx, conn: Any) -> None:
    """Add PostgreSQL startup storage in the caller's offline upgrade transaction."""
    for statement in split_sql_statements(schema_sql._MIGRATION_SQL_V73_TO_V74_POSTGRES):
        ctx.backend.execute(statement, connection=conn)
    if postgres_schema._get_schema_version_postgres(ctx, conn) != 74:
        raise SchemaError("Conversation startup PostgreSQL migration V73->V74 failed version verification.")  # noqa: TRY003


def _migrate_from_v64_to_v65_postgres(ctx, conn: Any) -> None:
    """Migrate PostgreSQL schema from V64 to V65 in the caller transaction."""
    ddl, _, _version_update = schema_sql._MIGRATION_SQL_V64_TO_V65_POSTGRES.partition(
        "UPDATE db_schema_version"
    )
    for statement in split_sql_statements(ddl):
        ctx.backend.execute(statement, connection=conn)
    migration_support._migration_v65_checkpoint(ctx, "schema-created")
    result = ctx.backend.execute(
        "UPDATE db_schema_version SET version = %s "
        "WHERE schema_name = %s AND version = %s RETURNING version",
        (65, ctx.schema_name, 64),
        connection=conn,
    )
    if result is None or result.rowcount != 1:
        raise SchemaError(  # noqa: TRY003
            f"[{ctx.schema_name}] Migration V64->V65 did not advance exactly one V64 schema row."
        )


def _install_document_chat_links_postgres(ctx, conn: Any) -> None:
    for statement in (
        *schema_sql.DOCUMENT_CHAT_LINK_SCHEMA,
        "ALTER TABLE document_chat_links ENABLE ROW LEVEL SECURITY",
        "ALTER TABLE document_chat_links FORCE ROW LEVEL SECURITY",
        """CREATE POLICY document_chat_links_owner ON document_chat_links
           USING (owner_id = current_setting('app.current_user_id', true))
           WITH CHECK (owner_id = current_setting('app.current_user_id', true))""",
    ):
        ctx.backend.execute(statement, connection=conn)


def _migrate_from_v91_to_v92_postgres(ctx, conn: Any) -> None:
    """Add Document Workspace chat links and force owner-only row access."""
    _install_document_chat_links_postgres(ctx, conn)
    ctx.backend.execute(
        "UPDATE db_schema_version SET version = 92 WHERE schema_name = %s AND version = 91",
        (ctx.schema_name,), connection=conn,
    )
    if postgres_schema._get_schema_version_postgres(ctx, conn) != 92:
        raise SchemaError("Document chat links PostgreSQL migration failed version verification.")


def _migrate_from_v92_to_v93_postgres(ctx, conn: Any) -> None:
    """Converge the two v92 schemas and enforce owner RLS before the version bump."""
    from tldw_Server_API.app.core.DB_Management.backends.pg_rls_policies import build_research_study_rls_sql
    from tldw_Server_API.app.core.DB_Management.chacha.research_study_schema import RESEARCH_STUDY_SCHEMA

    if not ctx.backend.table_exists("document_chat_links", connection=conn):
        _install_document_chat_links_postgres(ctx, conn)
    if not ctx.backend.table_exists("research_study_outputs", connection=conn):
        for statement in RESEARCH_STUDY_SCHEMA:
            ctx.backend.execute(statement, connection=conn)
    for statement in build_research_study_rls_sql():
        ctx.backend.execute(statement, connection=conn)
    ctx.backend.execute(
        "UPDATE db_schema_version SET version = %s WHERE schema_name = %s AND version = 92",
        (93, ctx.schema_name),
        connection=conn,
    )
    if postgres_schema._get_schema_version_postgres(ctx, conn) != 93:
        raise SchemaError("Research Study publication migration V92->V93 failed version verification.")


# Source schema version -> step name for the v72 to v93 upgrades, in order (R3, R12). The v5 to v71 ladder stays inline in
# `_initialize_schema_postgres`. Names are resolved on this module at call time. Pinned by test_chacha_structure_pins.py
# (FROZEN_POSTGRES). To add a migration: write `_migrate_from_v<N>_to_v<N+1>_postgres(ctx, conn)` here, add its key below, raise
# `_POSTGRES_SCHEMA_VERSION` on CharactersRAGDB, and extend FROZEN_POSTGRES in the pins file.
POSTGRES_STEPS: dict[int, str] = {
    72: "_migrate_from_v72_to_v73_postgres",
    73: "_migrate_from_v73_to_v74_postgres",
    74: "_migrate_from_v74_to_v75_postgres",
    75: "_migrate_from_v75_to_v76_postgres",
    76: "_migrate_from_v76_to_v77_persona_companion_postgres",
    77: "_migrate_from_v77_to_v78_postgres",
    78: "_migrate_from_v78_to_v79_postgres",
    79: "_migrate_from_v79_to_v80_postgres",
    80: "_migrate_from_v80_to_v81_postgres",
    81: "_migrate_from_v81_to_v82_postgres",
    82: "_migrate_from_v82_to_v83_postgres",
    83: "_migrate_from_v83_to_v84_postgres",
    84: "_migrate_from_v84_to_v85_postgres",
    85: "_migrate_from_v85_to_v86_postgres",
    86: "_migrate_from_v86_to_v87_postgres",
    87: "_migrate_from_v87_to_v88_postgres",
    88: "_migrate_from_v88_to_v89_postgres",
    89: "_migrate_from_v89_to_v90_postgres",
    90: "_migrate_from_v90_to_v91_postgres",
    91: "_migrate_from_v91_to_v92_postgres",
    92: "_migrate_from_v92_to_v93_postgres",
}
