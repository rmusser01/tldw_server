"""Test-only safe PostgreSQL failure classification; no product behavior substitution."""
import json
import pytest
from psycopg import Cursor
from tldw_Server_API.app.core.DB_Management.backends.postgresql_backend import PostgreSQLConnectionPool

class ObservedCursor(Cursor):
    def execute(self, query, params=None, **kwargs):
        try:
            return super().execute(query, params, **kwargs)
        except Exception as exc:
            if getattr(exc, 'sqlstate', None) == '40P01':
                print('SEQUENCE_DEADLOCK_PROOF', json.dumps({'sqlstate': exc.sqlstate, 'detail': exc.diag.message_detail}))
            raise

@pytest.fixture(autouse=True)
def observed_connections(monkeypatch):
    original=PostgreSQLConnectionPool.get_connection
    def get(self):
        conn=original(self)
        conn.cursor_factory=ObservedCursor
        return conn
    monkeypatch.setattr(PostgreSQLConnectionPool, 'get_connection', get)
