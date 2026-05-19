"""PostgreSQL migration body for schema v23 transcript run-history scaffolding."""

from __future__ import annotations

from typing import Any, Protocol


class _TranscriptRunHistoryBackend(Protocol):
    """Backend surface required by the v23 migration helper."""

    def escape_identifier(self, name: str) -> str: ...

    def execute(
        self,
        query: str,
        params: tuple[object, ...] | None = None,
        *,
        connection: Any,
    ) -> Any: ...


class PostgresTranscriptRunHistoryBody(Protocol):
    """DB surface required by the v23 migration helper."""

    @property
    def backend(self) -> _TranscriptRunHistoryBackend: ...


def run_postgres_migrate_to_v23(
    db: PostgresTranscriptRunHistoryBody,
    conn: Any,
) -> None:
    """Add transcript run-history columns, indexes, and backfill state on PostgreSQL."""

    backend = db.backend
    ident = backend.escape_identifier

    statements = [
        (
            f"ALTER TABLE {ident('media')} "
            f"ADD COLUMN IF NOT EXISTS {ident('latest_transcription_run_id')} BIGINT"
        ),
        (
            f"ALTER TABLE {ident('media')} "
            f"ADD COLUMN IF NOT EXISTS {ident('next_transcription_run_id')} BIGINT NOT NULL DEFAULT 1"
        ),
        (
            f"ALTER TABLE {ident('transcripts')} "
            f"ADD COLUMN IF NOT EXISTS {ident('transcription_run_id')} BIGINT"
        ),
        (
            f"ALTER TABLE {ident('transcripts')} "
            f"ADD COLUMN IF NOT EXISTS {ident('supersedes_run_id')} BIGINT"
        ),
        (
            f"ALTER TABLE {ident('transcripts')} "
            f"ADD COLUMN IF NOT EXISTS {ident('idempotency_key')} TEXT"
        ),
        (
            f"ALTER TABLE {ident('transcripts')} "
            "DROP CONSTRAINT IF EXISTS transcripts_media_id_whisper_model_key"
        ),
        (
            f"CREATE INDEX IF NOT EXISTS {ident('idx_media_latest_transcription_run_id')} "
            f"ON {ident('media')} ({ident('latest_transcription_run_id')})"
        ),
        (
            f"CREATE INDEX IF NOT EXISTS {ident('idx_media_next_transcription_run_id')} "
            f"ON {ident('media')} ({ident('next_transcription_run_id')})"
        ),
        (
            f"CREATE UNIQUE INDEX IF NOT EXISTS {ident('idx_transcripts_media_run_id')} "
            f"ON {ident('transcripts')} ({ident('media_id')}, {ident('transcription_run_id')} DESC) "
            f"WHERE {ident('transcription_run_id')} IS NOT NULL"
        ),
        (
            f"CREATE INDEX IF NOT EXISTS {ident('idx_transcripts_supersedes_run_id')} "
            f"ON {ident('transcripts')} ({ident('supersedes_run_id')})"
        ),
        (
            f"CREATE UNIQUE INDEX IF NOT EXISTS {ident('idx_transcripts_media_idempotency_key')} "
            f"ON {ident('transcripts')} ({ident('media_id')}, {ident('idempotency_key')}) "
            f"WHERE {ident('idempotency_key')} IS NOT NULL"
        ),
        (
            """
            WITH transcript_runs AS (
                SELECT
                    id AS transcript_id,
                    ROW_NUMBER() OVER (
                        PARTITION BY media_id
                        ORDER BY created_at NULLS FIRST, id ASC
                    ) AS assigned_run_id
                FROM transcripts
            )
            UPDATE transcripts AS t
            SET transcription_run_id = transcript_runs.assigned_run_id
            FROM transcript_runs
            WHERE t.id = transcript_runs.transcript_id
              AND t.transcription_run_id IS NULL
            """
        ),
        (
            """
            WITH media_run_bounds AS (
                SELECT
                    media_id AS media_id,
                    MAX(transcription_run_id) FILTER (WHERE deleted = FALSE) AS latest_run_id,
                    COALESCE(MAX(transcription_run_id), 0) + 1 AS next_run_id
                FROM transcripts
                GROUP BY media_id
            )
            UPDATE media AS m
            SET
                latest_transcription_run_id = media_run_bounds.latest_run_id,
                next_transcription_run_id = media_run_bounds.next_run_id
            FROM media_run_bounds
            WHERE m.id = media_run_bounds.media_id
            """
        ),
        (
            """
            UPDATE media
            SET next_transcription_run_id = 1
            WHERE next_transcription_run_id IS NULL
               OR next_transcription_run_id < 1
            """
        ),
    ]

    for statement in statements:
        backend.execute(statement, connection=conn)


__all__ = ["PostgresTranscriptRunHistoryBody", "run_postgres_migrate_to_v23"]
