"""Pooled connections must not carry one request's tenant identity to the next.

The content backend writes its tenant GUCs with set_config(..., false), which
is session-scoped: it survives the transaction and rides the connection back
into the pool. A rollback does not clear it, and the pool was created without a
reset hook, so a borrower whose scope re-application failed could inherit the
previous request's app.current_user_id or app.is_admin.

Driven against a stub connection, so no database is needed.
"""

import pytest

from tldw_Server_API.app.core.DB_Management.backends.postgresql_backend import (
    _reset_pooled_connection,
)

pytestmark = pytest.mark.unit


class _Cursor:
    def __init__(self, sink, failing):
        self._sink = sink
        self._failing = failing

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql):
        if sql in self._failing:
            raise RuntimeError(f"cannot run {sql}")
        self._sink.append(sql)


class _Conn:
    def __init__(self, failing=()):
        self.executed: list[str] = []
        self._failing = set(failing)

    def cursor(self):
        return _Cursor(self.executed, self._failing)


def test_reset_clears_session_gucs_and_role():
    conn = _Conn()

    _reset_pooled_connection(conn)

    assert "RESET ALL" in conn.executed, "session GUCs must be cleared"
    assert "RESET ROLE" in conn.executed
    assert "RESET SESSION AUTHORIZATION" in conn.executed


def test_reset_continues_when_one_statement_is_rejected():
    """A role the server will not let us reset must not skip clearing GUCs."""
    conn = _Conn(failing={"RESET SESSION AUTHORIZATION"})

    _reset_pooled_connection(conn)

    assert "RESET ALL" in conn.executed


def test_reset_never_raises_into_the_pool():
    """The pool discards a connection it cannot reset; it must not see an error."""
    conn = _Conn(failing={"RESET ROLE", "RESET SESSION AUTHORIZATION", "RESET ALL"})

    _reset_pooled_connection(conn)

    assert conn.executed == []
