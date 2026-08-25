"""PostgreSQL migration body for schema v26 staged clone persistence."""

from __future__ import annotations

from typing import Any, Protocol

from tldw_Server_API.app.core.DB_Management.media_db.errors import SchemaError


class _StagedClonePersistenceBackend(Protocol):
    def escape_identifier(self, name: str) -> str: ...

    def execute(
        self,
        query: str,
        params: tuple[object, ...] | None = None,
        *,
        connection: Any,
    ) -> Any: ...


class PostgresStagedClonePersistenceBody(Protocol):
    @property
    def backend(self) -> _StagedClonePersistenceBackend: ...


_CORE_RELATIONS = ("media", "keywords", "mediakeywords")
_PENDING_RELATION = "operationownedclonekeywords"
_LEGACY_RELATION = "operationownedclonekeywords_v25"
_LEGACY_COLUMNS = {
    "media_id",
    "keyword_id",
    "operation_id",
    "source_identity",
    "created_by_clone",
}
_FINAL_COLUMNS = {
    "media_id",
    "keyword",
    "operation_id",
    "source_identity",
    "client_id",
}


def _result_rows(result: Any) -> list[dict[str, Any]]:
    return list(getattr(result, "rows", None) or [])


def _count_result(result: Any, *, failure: str) -> int:
    rows = _result_rows(result)
    if len(rows) != 1 or "count" not in rows[0]:
        raise SchemaError(failure)
    return int(rows[0]["count"])


def _complete_marker_sql(alias: str, ident: Any) -> str:
    return f"""
        {alias}.{ident('system_operation_id')} IS NOT NULL
        AND length({alias}.{ident('system_operation_id')}) BETWEEN 1 AND 255
        AND {alias}.{ident('system_operation_kind')} = 'shared_workspace_clone'
        AND {alias}.{ident('system_source_identity')} IS NOT NULL
        AND length({alias}.{ident('system_source_identity')}) BETWEEN 1 AND 255
        AND {alias}.{ident('system_content_hash')} IS NOT NULL
        AND {alias}.{ident('system_content_hash')} ~ '^[0-9a-f]{{64}}$'
    """


def _create_pending_table(
    backend: _StagedClonePersistenceBackend,
    conn: Any,
) -> None:
    ident = backend.escape_identifier
    backend.execute(
        f"""
        CREATE TABLE {ident(_PENDING_RELATION)} (
            {ident('media_id')} BIGINT NOT NULL,
            {ident('keyword')} TEXT NOT NULL
                CHECK (length({ident('keyword')}) BETWEEN 1 AND 255
                       AND {ident('keyword')} = lower(btrim({ident('keyword')}))),
            {ident('operation_id')} TEXT NOT NULL
                CHECK (length({ident('operation_id')}) BETWEEN 1 AND 255),
            {ident('source_identity')} TEXT NOT NULL
                CHECK (length({ident('source_identity')}) BETWEEN 1 AND 255),
            {ident('client_id')} TEXT NOT NULL
                CHECK (length({ident('client_id')}) BETWEEN 1 AND 255),
            PRIMARY KEY ({ident('media_id')}, {ident('keyword')}),
            FOREIGN KEY ({ident('media_id')}) REFERENCES {ident('media')} ({ident('id')})
                ON DELETE CASCADE
        )
        """,  # nosec B608
        connection=conn,
    )


def _inspect_and_unlock_relations(
    backend: _StagedClonePersistenceBackend,
    conn: Any,
) -> tuple[bool, set[str]]:
    ident = backend.escape_identifier
    initial_rows = _result_rows(
        backend.execute(
            """
            SELECT table_row.relname
              FROM pg_class AS table_row
              JOIN pg_namespace AS namespace_row
                ON namespace_row.oid = table_row.relnamespace
             WHERE namespace_row.nspname = current_schema()
               AND table_row.relkind IN ('r', 'p')
               AND table_row.relname = ANY(%s)
            """,
            ([*_CORE_RELATIONS, _PENDING_RELATION],),
            connection=conn,
        )
    )
    initial_names = {str(row["relname"]) for row in initial_rows}
    pending_exists = _PENDING_RELATION in initial_names
    lock_names = [*_CORE_RELATIONS]
    if pending_exists:
        lock_names.append(_PENDING_RELATION)
    backend.execute(
        "LOCK TABLE "
        + ", ".join(ident(name) for name in lock_names)
        + " IN ACCESS EXCLUSIVE MODE",
        connection=conn,
    )

    relation_rows = _result_rows(
        backend.execute(
            """
            SELECT table_row.relname AS table_name,
                   table_row.relrowsecurity AS rls_enabled,
                   table_row.relforcerowsecurity AS rls_forced,
                   table_row.relowner = current_user::regrole AS is_table_owner,
                   pg_has_role(current_user, namespace_row.nspowner, 'USAGE')
                     AS is_schema_owner
              FROM pg_class AS table_row
              JOIN pg_namespace AS namespace_row
                ON namespace_row.oid = table_row.relnamespace
             WHERE namespace_row.nspname = current_schema()
               AND table_row.relkind IN ('r', 'p')
               AND table_row.relname = ANY(%s)
            """,
            (
                [
                    *_CORE_RELATIONS,
                    _PENDING_RELATION,
                    _LEGACY_RELATION,
                ],
            ),
            connection=conn,
        )
    )
    relation_by_name = {str(row["table_name"]): row for row in relation_rows}
    expected = {*_CORE_RELATIONS}
    if pending_exists:
        expected.add(_PENDING_RELATION)
    if not expected.issubset(relation_by_name):
        raise SchemaError("Media v26 migration relation set changed while acquiring locks.")
    if _LEGACY_RELATION in relation_by_name:
        raise SchemaError("Media v26 migration legacy relation collision requires repair.")

    forced_relations: set[str] = set()
    for name in sorted(expected):
        row = relation_by_name[name]
        if not bool(row["is_table_owner"]) or not bool(row["is_schema_owner"]):
            raise SchemaError("Media v26 migration requires affected relation ownership.")
        rls_enabled = bool(row["rls_enabled"])
        rls_forced = bool(row["rls_forced"])
        if rls_forced and not rls_enabled:
            raise SchemaError("Media v26 migration found inconsistent row security state.")
        if rls_forced:
            forced_relations.add(name)

    for name in sorted(forced_relations):
        backend.execute(
            f"ALTER TABLE {ident(name)} NO FORCE ROW LEVEL SECURITY",  # nosec B608
            connection=conn,
        )
    return pending_exists, forced_relations


def _pending_shape(
    backend: _StagedClonePersistenceBackend,
    conn: Any,
    *,
    pending_exists: bool,
) -> str:
    if not pending_exists:
        return "absent"
    columns = {
        str(row["column_name"])
        for row in _result_rows(
            backend.execute(
                """
                SELECT column_name
                  FROM information_schema.columns
                 WHERE table_schema = current_schema()
                   AND table_name = %s
                """,
                (_PENDING_RELATION,),
                connection=conn,
            )
        )
    }
    if columns == _LEGACY_COLUMNS:
        return "legacy"
    if columns == _FINAL_COLUMNS:
        return "final"
    raise SchemaError("Media v26 migration found an unsupported pending keyword shape.")


def run_postgres_migrate_to_v26(
    db: PostgresStagedClonePersistenceBody,
    conn: Any,
) -> None:
    """Migrate staged clone keywords across the complete owner-controlled dataset."""

    backend = db.backend
    ident = backend.escape_identifier
    media = ident("media")
    pending = ident(_PENDING_RELATION)
    legacy = ident(_LEGACY_RELATION)
    keywords = ident("keywords")
    links = ident("mediakeywords")
    marker = _complete_marker_sql("owned_media", ident)

    pending_exists, forced_relations = _inspect_and_unlock_relations(backend, conn)
    shape = _pending_shape(backend, conn, pending_exists=pending_exists)

    backend.execute(
        f"""
        UPDATE {media}
           SET {ident('system_operation_id')} = NULL,
               {ident('system_operation_kind')} = NULL,
               {ident('system_source_identity')} = NULL,
               {ident('system_content_hash')} = NULL
         WHERE (
            (
                {ident('system_operation_id')} IS NULL
                AND {ident('system_operation_kind')} IS NULL
                AND {ident('system_source_identity')} IS NULL
                AND {ident('system_content_hash')} IS NULL
            )
            OR
            (
                length({ident('system_operation_id')}) BETWEEN 1 AND 255
                AND {ident('system_operation_kind')} = 'shared_workspace_clone'
                AND length({ident('system_source_identity')}) BETWEEN 1 AND 255
                AND {ident('system_content_hash')} ~ '^[0-9a-f]{{64}}$'
            )
         ) IS NOT TRUE
        """,  # nosec B608
        connection=conn,
    )
    backend.execute(
        f"ALTER TABLE {media} DROP CONSTRAINT IF EXISTS "
        f"{ident('ck_media_system_operation_ownership')}",
        connection=conn,
    )
    backend.execute(
        f"""
        ALTER TABLE {media}
            ADD CONSTRAINT {ident('ck_media_system_operation_ownership')}
            CHECK (
                (
                    {ident('system_operation_id')} IS NULL
                    AND {ident('system_operation_kind')} IS NULL
                    AND {ident('system_source_identity')} IS NULL
                    AND {ident('system_content_hash')} IS NULL
                )
                OR
                (
                    {ident('system_operation_id')} IS NOT NULL
                    AND length({ident('system_operation_id')}) BETWEEN 1 AND 255
                    AND {ident('system_operation_kind')} IS NOT NULL
                    AND {ident('system_operation_kind')} = 'shared_workspace_clone'
                    AND {ident('system_source_identity')} IS NOT NULL
                    AND length({ident('system_source_identity')}) BETWEEN 1 AND 255
                    AND {ident('system_content_hash')} IS NOT NULL
                    AND {ident('system_content_hash')} ~ '^[0-9a-f]{{64}}$'
                )
            )
        """,  # nosec B608
        connection=conn,
    )

    backend.execute(
        f"DROP INDEX IF EXISTS {ident('idx_owned_clone_keywords_keyword')}",
        connection=conn,
    )
    backend.execute(
        f"DROP INDEX IF EXISTS {ident('idx_owned_clone_keywords_operation')}",
        connection=conn,
    )
    if shape == "legacy":
        backend.execute(
            f"ALTER TABLE {pending} RENAME TO {legacy}",
            connection=conn,
        )
        _create_pending_table(backend, conn)
    elif shape == "absent":
        _create_pending_table(backend, conn)

    backend.execute(
        f"""
        INSERT INTO {pending} (
            {ident('media_id')}, {ident('keyword')}, {ident('operation_id')},
            {ident('source_identity')}, {ident('client_id')}
        )
        SELECT staged_links.{ident('media_id')},
               lower(btrim(source_keyword.{ident('keyword')})),
               owned_media.{ident('system_operation_id')},
               owned_media.{ident('system_source_identity')},
               owned_media.{ident('client_id')}
          FROM {links} AS staged_links
          JOIN {keywords} AS source_keyword
            ON source_keyword.{ident('id')} = staged_links.{ident('keyword_id')}
          JOIN {media} AS owned_media
            ON owned_media.{ident('id')} = staged_links.{ident('media_id')}
         WHERE {marker}
           AND length(btrim(source_keyword.{ident('keyword')})) BETWEEN 1 AND 255
        ON CONFLICT DO NOTHING
        """,  # nosec B608
        connection=conn,
    )
    if shape == "legacy":
        backend.execute(
            f"""
            INSERT INTO {pending} (
                {ident('media_id')}, {ident('keyword')}, {ident('operation_id')},
                {ident('source_identity')}, {ident('client_id')}
            )
            SELECT holds.{ident('media_id')},
                   lower(btrim(source_keyword.{ident('keyword')})),
                   owned_media.{ident('system_operation_id')},
                   owned_media.{ident('system_source_identity')},
                   owned_media.{ident('client_id')}
              FROM {legacy} AS holds
              JOIN {keywords} AS source_keyword
                ON source_keyword.{ident('id')} = holds.{ident('keyword_id')}
              JOIN {media} AS owned_media
                ON owned_media.{ident('id')} = holds.{ident('media_id')}
             WHERE {marker}
               AND holds.{ident('operation_id')} = owned_media.{ident('system_operation_id')}
               AND holds.{ident('source_identity')} = owned_media.{ident('system_source_identity')}
               AND length(btrim(source_keyword.{ident('keyword')})) BETWEEN 1 AND 255
            ON CONFLICT DO NOTHING
            """,  # nosec B608
            connection=conn,
        )

    missing_links = _count_result(
        backend.execute(
            f"""
            SELECT COUNT(*) AS count
              FROM {links} AS staged_links
              JOIN {media} AS owned_media
                ON owned_media.{ident('id')} = staged_links.{ident('media_id')}
              LEFT JOIN {keywords} AS source_keyword
                ON source_keyword.{ident('id')} = staged_links.{ident('keyword_id')}
              LEFT JOIN {pending} AS copied
                ON copied.{ident('media_id')} = staged_links.{ident('media_id')}
               AND copied.{ident('keyword')} = lower(btrim(source_keyword.{ident('keyword')}))
               AND copied.{ident('operation_id')} = owned_media.{ident('system_operation_id')}
               AND copied.{ident('source_identity')} = owned_media.{ident('system_source_identity')}
               AND copied.{ident('client_id')} = owned_media.{ident('client_id')}
             WHERE {marker}
               AND (
                    source_keyword.{ident('id')} IS NULL
                    OR length(btrim(source_keyword.{ident('keyword')})) NOT BETWEEN 1 AND 255
                    OR copied.{ident('media_id')} IS NULL
               )
            """,  # nosec B608
            connection=conn,
        ),
        failure="Media v26 migration could not verify staged keyword links.",
    )
    if missing_links:
        raise SchemaError("Media v26 migration could not copy every staged keyword link.")

    if shape == "legacy":
        missing_holds = _count_result(
            backend.execute(
                f"""
                SELECT COUNT(*) AS count
                  FROM {legacy} AS holds
                  JOIN {media} AS owned_media
                    ON owned_media.{ident('id')} = holds.{ident('media_id')}
                  LEFT JOIN {keywords} AS source_keyword
                    ON source_keyword.{ident('id')} = holds.{ident('keyword_id')}
                  LEFT JOIN {pending} AS copied
                    ON copied.{ident('media_id')} = holds.{ident('media_id')}
                   AND copied.{ident('keyword')} = lower(btrim(source_keyword.{ident('keyword')}))
                   AND copied.{ident('operation_id')} = owned_media.{ident('system_operation_id')}
                   AND copied.{ident('source_identity')} = owned_media.{ident('system_source_identity')}
                   AND copied.{ident('client_id')} = owned_media.{ident('client_id')}
                 WHERE {marker}
                   AND holds.{ident('operation_id')} = owned_media.{ident('system_operation_id')}
                   AND holds.{ident('source_identity')} = owned_media.{ident('system_source_identity')}
                   AND (
                        source_keyword.{ident('id')} IS NULL
                        OR length(btrim(source_keyword.{ident('keyword')})) NOT BETWEEN 1 AND 255
                        OR copied.{ident('media_id')} IS NULL
                   )
                """,  # nosec B608
                connection=conn,
            ),
            failure="Media v26 migration could not verify legacy keyword holds.",
        )
        if missing_holds:
            raise SchemaError("Media v26 migration could not copy every exact legacy hold.")

    backend.execute(
        f"""
        DELETE FROM {links} AS staged_links
         USING {media} AS owned_media
         WHERE staged_links.{ident('media_id')} = owned_media.{ident('id')}
           AND {marker}
        """,  # nosec B608
        connection=conn,
    )
    if shape == "legacy":
        backend.execute(
            f"""
            DELETE FROM {keywords} AS doomed
             USING {legacy} AS holds, {media} AS owned_media
             WHERE doomed.{ident('id')} = holds.{ident('keyword_id')}
               AND owned_media.{ident('id')} = holds.{ident('media_id')}
               AND {marker}
               AND holds.{ident('operation_id')} = owned_media.{ident('system_operation_id')}
               AND holds.{ident('source_identity')} = owned_media.{ident('system_source_identity')}
               AND holds.{ident('created_by_clone')} IS TRUE
               AND NOT EXISTS (
                    SELECT 1 FROM {links} AS retained_link
                     WHERE retained_link.{ident('keyword_id')} = doomed.{ident('id')}
               )
            """,  # nosec B608
            connection=conn,
        )

    remaining_links = _count_result(
        backend.execute(
            f"""
            SELECT COUNT(*) AS count
              FROM {links} AS staged_links
              JOIN {media} AS owned_media
                ON owned_media.{ident('id')} = staged_links.{ident('media_id')}
             WHERE {marker}
            """,  # nosec B608
            connection=conn,
        ),
        failure="Media v26 migration could not verify staged link deletion.",
    )
    if remaining_links:
        raise SchemaError("Media v26 migration left direct staged keyword links.")
    if shape == "legacy":
        backend.execute(f"DROP TABLE {legacy}", connection=conn)

    backend.execute(
        f"CREATE INDEX {ident('idx_owned_clone_keywords_keyword')} "
        f"ON {pending} ({ident('keyword')})",
        connection=conn,
    )
    backend.execute(
        f"CREATE INDEX {ident('idx_owned_clone_keywords_operation')} "
        f"ON {pending} ({ident('operation_id')}, {ident('source_identity')})",
        connection=conn,
    )

    for name in sorted(forced_relations - {"media", _PENDING_RELATION}):
        backend.execute(
            f"ALTER TABLE {ident(name)} FORCE ROW LEVEL SECURITY",  # nosec B608
            connection=conn,
        )
    backend.execute(
        f"ALTER TABLE {media} ENABLE ROW LEVEL SECURITY",
        connection=conn,
    )
    backend.execute(
        f"ALTER TABLE {media} FORCE ROW LEVEL SECURITY",
        connection=conn,
    )
    backend.execute(
        f"ALTER TABLE {pending} ENABLE ROW LEVEL SECURITY",
        connection=conn,
    )
    backend.execute(
        f"ALTER TABLE {pending} FORCE ROW LEVEL SECURITY",
        connection=conn,
    )


__all__ = ["PostgresStagedClonePersistenceBody", "run_postgres_migrate_to_v26"]
