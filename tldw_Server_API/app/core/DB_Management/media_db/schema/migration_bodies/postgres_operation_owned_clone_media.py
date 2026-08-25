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
                            AND {ident('system_operation_kind')} = 'shared_workspace_clone'
                            AND {ident('system_source_identity')} IS NOT NULL
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


__all__ = [
    "PostgresOperationOwnedCloneMediaBody",
    "run_postgres_migrate_to_v25",
]
