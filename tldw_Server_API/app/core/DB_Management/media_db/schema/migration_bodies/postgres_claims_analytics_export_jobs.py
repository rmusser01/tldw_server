"""PostgreSQL migration body for schema v24 Claims analytics export Jobs fields."""

from __future__ import annotations

from typing import Any, Protocol

from tldw_Server_API.app.core.DB_Management.media_db.schema.postgres_claims_json_helpers import (
    POSTGRES_CLAIMS_JSON_HELPER_DDL,
)


class _ClaimsAnalyticsExportJobsBackend(Protocol):
    """Backend surface required by the v24 migration helper."""

    def escape_identifier(self, name: str) -> str: ...

    def execute(
        self,
        query: str,
        params: tuple[object, ...] | None = None,
        *,
        connection: Any,
    ) -> Any: ...


class PostgresClaimsAnalyticsExportJobsBody(Protocol):
    """DB surface required by the v24 migration helper."""

    @property
    def backend(self) -> _ClaimsAnalyticsExportJobsBackend: ...


def run_postgres_migrate_to_v24(
    db: PostgresClaimsAnalyticsExportJobsBody,
    conn: Any,
) -> None:
    """Add nullable Jobs linkage and snapshot fields to Claims analytics exports."""

    backend = db.backend
    ident = backend.escape_identifier
    table = ident("claims_analytics_exports")

    statements = [
        *POSTGRES_CLAIMS_JSON_HELPER_DDL,
        (
            f"ALTER TABLE {table} "
            f"ADD COLUMN IF NOT EXISTS {ident('job_id')} BIGINT"
        ),
        (
            f"ALTER TABLE {table} "
            f"ADD COLUMN IF NOT EXISTS {ident('error_code')} TEXT"
        ),
        (
            f"ALTER TABLE {table} "
            f"ADD COLUMN IF NOT EXISTS {ident('snapshot_at')} TIMESTAMPTZ"
        ),
        (
            f"ALTER TABLE {table} "
            f"ADD COLUMN IF NOT EXISTS {ident('snapshot_event_id')} BIGINT"
        ),
        (
            f"CREATE INDEX IF NOT EXISTS {ident('idx_claims_analytics_exports_job_id')} "
            f"ON {table} ({ident('job_id')})"
        ),
    ]

    for statement in statements:
        backend.execute(statement, connection=conn)


__all__ = [
    "PostgresClaimsAnalyticsExportJobsBody",
    "run_postgres_migrate_to_v24",
]
