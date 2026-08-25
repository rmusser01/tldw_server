"""PostgreSQL migration body for schema v26 staged clone persistence."""

from __future__ import annotations

from typing import Any, Protocol


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


def run_postgres_migrate_to_v26(
    db: PostgresStagedClonePersistenceBody,
    conn: Any,
) -> None:
    """Replace v25 keyword holds and repair exact Media marker enforcement."""

    backend = db.backend
    ident = backend.escape_identifier
    media = ident("media")
    pending = ident("operationownedclonekeywords")
    legacy = ident("operationownedclonekeywords_v25")
    keywords = ident("keywords")

    backend.execute(
        f"""
        UPDATE {media}
           SET {ident('system_operation_id')} = NULL,
               {ident('system_operation_kind')} = NULL,
               {ident('system_source_identity')} = NULL,
               {ident('system_content_hash')} = NULL
         WHERE NOT (
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
         )
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
    backend.execute(
        f"""
        DO $staged_clone_v26$
        BEGIN
            IF to_regclass(current_schema() || '.operationownedclonekeywords') IS NULL THEN
                CREATE TABLE {pending} (
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
                    FOREIGN KEY ({ident('media_id')}) REFERENCES {media} ({ident('id')})
                        ON DELETE CASCADE
                );
            ELSIF EXISTS (
                SELECT 1 FROM information_schema.columns
                 WHERE table_schema = current_schema()
                   AND table_name = 'operationownedclonekeywords'
                   AND column_name = 'keyword_id'
            ) THEN
                ALTER TABLE {pending} RENAME TO {legacy};
                CREATE TABLE {pending} (
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
                    FOREIGN KEY ({ident('media_id')}) REFERENCES {media} ({ident('id')})
                        ON DELETE CASCADE
                );
                INSERT INTO {pending} (
                    {ident('media_id')}, {ident('keyword')}, {ident('operation_id')},
                    {ident('source_identity')}, {ident('client_id')}
                )
                SELECT holds.{ident('media_id')}, lower(btrim(source_keyword.{ident('keyword')})),
                       owned_media.{ident('system_operation_id')},
                       owned_media.{ident('system_source_identity')},
                       owned_media.{ident('client_id')}
                  FROM {legacy} AS holds
                  JOIN {keywords} AS source_keyword
                    ON source_keyword.{ident('id')} = holds.{ident('keyword_id')}
                  JOIN {media} AS owned_media
                    ON owned_media.{ident('id')} = holds.{ident('media_id')}
                 WHERE owned_media.{ident('system_operation_kind')} = 'shared_workspace_clone'
                   AND length(owned_media.{ident('system_operation_id')}) BETWEEN 1 AND 255
                   AND length(owned_media.{ident('system_source_identity')}) BETWEEN 1 AND 255
                   AND length(btrim(source_keyword.{ident('keyword')})) BETWEEN 1 AND 255
                ON CONFLICT DO NOTHING;
                DROP TABLE {legacy};
            END IF;
        END
        $staged_clone_v26$
        """,  # nosec B608
        connection=conn,
    )
    backend.execute(
        f"CREATE INDEX IF NOT EXISTS {ident('idx_owned_clone_keywords_keyword')} "
        f"ON {pending} ({ident('keyword')})",
        connection=conn,
    )
    backend.execute(
        f"CREATE INDEX IF NOT EXISTS {ident('idx_owned_clone_keywords_operation')} "
        f"ON {pending} ({ident('operation_id')}, {ident('source_identity')})",
        connection=conn,
    )


__all__ = ["PostgresStagedClonePersistenceBody", "run_postgres_migrate_to_v26"]
