"""PostgreSQL migration body for schema v25 operation-owned clone Media."""

from __future__ import annotations

from typing import Any, Protocol


class _OperationOwnedCloneMediaBackend(Protocol):
    """Backend surface required by the v25 migration helper."""

    def escape_identifier(self, name: str) -> str: ...

    def execute(
        self,
        query: str,
        params: tuple[object, ...] | None = None,
        *,
        connection: Any,
    ) -> Any: ...


class PostgresOperationOwnedCloneMediaBody(Protocol):
    """DB surface required by the v25 migration helper."""

    @property
    def backend(self) -> _OperationOwnedCloneMediaBackend: ...


def run_postgres_migrate_to_v25(
    db: PostgresOperationOwnedCloneMediaBody,
    conn: Any,
) -> None:
    """Add complete operation ownership markers and their uniqueness fence."""

    backend = db.backend
    ident = backend.escape_identifier
    table = ident("media")
    columns = (
        "system_operation_id",
        "system_operation_kind",
        "system_source_identity",
        "system_content_hash",
    )
    for column in columns:
        backend.execute(
            f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {ident(column)} TEXT",
            connection=conn,
        )

    # Every interpolated identifier below is produced by backend escaping.
    backend.execute(
        f"""
        DO $operation_owned_media_v25$
        BEGIN
            IF NOT EXISTS (
                SELECT 1
                  FROM pg_constraint
                 WHERE conname = 'ck_media_system_operation_ownership'
                   AND conrelid = '{table}'::regclass
            ) THEN
                ALTER TABLE {table}
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
                    );
            END IF;
        END
        $operation_owned_media_v25$
        """,  # nosec B608
        connection=conn,
    )
    backend.execute(
        f"CREATE UNIQUE INDEX IF NOT EXISTS {ident('ux_media_system_operation_source')} "
        f"ON {table} ({ident('system_operation_kind')}, "
        f"{ident('system_operation_id')}, {ident('system_source_identity')}) "
        f"WHERE {ident('system_operation_id')} IS NOT NULL",
        connection=conn,
    )
    holds_table = ident("operationownedclonekeywords")
    backend.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {holds_table} (
            {ident('media_id')} BIGINT NOT NULL,
            {ident('keyword_id')} BIGINT NOT NULL,
            {ident('operation_id')} TEXT NOT NULL
                CHECK (length({ident('operation_id')}) BETWEEN 1 AND 255),
            {ident('source_identity')} TEXT NOT NULL
                CHECK (length({ident('source_identity')}) BETWEEN 1 AND 255),
            {ident('created_by_clone')} BOOLEAN NOT NULL,
            PRIMARY KEY ({ident('media_id')}, {ident('keyword_id')}),
            FOREIGN KEY ({ident('media_id')}) REFERENCES {table} ({ident('id')})
                ON DELETE CASCADE,
            FOREIGN KEY ({ident('keyword_id')}) REFERENCES {ident('keywords')} ({ident('id')})
                ON DELETE CASCADE
        )
        """,  # nosec B608
        connection=conn,
    )
    backend.execute(
        f"CREATE INDEX IF NOT EXISTS {ident('idx_owned_clone_keywords_keyword')} "
        f"ON {holds_table} ({ident('keyword_id')})",
        connection=conn,
    )
    backend.execute(
        f"CREATE INDEX IF NOT EXISTS {ident('idx_owned_clone_keywords_operation')} "
        f"ON {holds_table} ({ident('operation_id')}, {ident('source_identity')})",
        connection=conn,
    )


__all__ = [
    "PostgresOperationOwnedCloneMediaBody",
    "run_postgres_migrate_to_v25",
]
