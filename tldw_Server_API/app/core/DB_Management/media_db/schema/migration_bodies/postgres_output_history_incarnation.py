"""Install original-instance TTS history disposal authority without guessing links."""

from typing import Any

from tldw_Server_API.app.core.DB_Management.media_db.schema.postgres_sqlite_conversion import (
    _convert_sqlite_sql_to_postgres_statements,
)


def run_postgres_migrate_to_v27(db: Any, conn: Any) -> None:
    """Upgrade existing history and create its receiver; propagate every failure."""
    statements = _convert_sqlite_sql_to_postgres_statements(db, db._TTS_HISTORY_TABLE_SQL)
    # Optional history may be absent; install its table before upgrading/indexing.
    db.backend.execute(statements[0], connection=conn)
    db.backend.execute(
        "ALTER TABLE tts_history ADD COLUMN IF NOT EXISTS output_incarnation TEXT",
        connection=conn,
    )
    for statement in statements[1:]:
        db.backend.execute(statement, connection=conn)
