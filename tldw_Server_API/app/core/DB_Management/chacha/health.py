"""Connection probes using the ChaCha database transaction contract."""

from tldw_Server_API.app.core.DB_Management import sqlite_policy
from tldw_Server_API.app.core.DB_Management.backends.base import BackendType
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB


def probe_chacha_connection(db: CharactersRAGDB) -> None:
    """Probe a connection in the caller's operation scope, raising on failure."""
    connection = db.get_connection()
    if db.backend_type == BackendType.POSTGRESQL:
        with db.transaction() as transaction:
            transaction.execute("SELECT 1")
    else:
        sqlite_policy.configure_sqlite_connection(
            connection,
            use_wal=False,
            synchronous=None,
            foreign_keys=True,
            busy_timeout_ms=1000,
            temp_store=None,
        )
        connection.execute("SELECT 1")
