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
    def __init__(self, sink: list[str], failing: set[str]) -> None:
        self._sink = sink
        self._failing = failing

    def __enter__(self) -> "_Cursor":
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def execute(self, sql: str) -> None:
        if sql in self._failing:
            raise RuntimeError(f"cannot run {sql}")
        self._sink.append(sql)


class _Conn:
    """Tracks transaction state the way psycopg does.

    Executing a statement opens an implicit transaction. psycopg_pool discards
    any connection its reset callback hands back while still INTRANS, so the
    callback has to close it -- and with a commit, since SET and RESET are
    transactional in PostgreSQL and a rollback would undo the reset.
    """

    def __init__(
        self, failing: tuple[str, ...] | set[str] = (), commit_fails: bool = False
    ) -> None:
        self.executed: list[str] = []
        self._failing = set(failing)
        self.in_transaction = False
        self.committed = False
        self.rolled_back = False
        self._commit_fails = commit_fails

    def cursor(self) -> _Cursor:
        self.in_transaction = True
        return _Cursor(self.executed, self._failing)

    def commit(self) -> None:
        if self._commit_fails:
            raise RuntimeError("commit refused")
        self.in_transaction = False
        self.committed = True

    def rollback(self) -> None:
        self.in_transaction = False
        self.rolled_back = True


def test_reset_clears_session_gucs_and_role():
    conn = _Conn()

    _reset_pooled_connection(conn)

    assert "RESET ALL" in conn.executed, "session GUCs must be cleared"
    assert "RESET ROLE" in conn.executed
    assert "RESET SESSION AUTHORIZATION" in conn.executed


def test_a_partly_scrubbed_connection_is_rejected():
    """The remaining statements still run, but the connection is not reused.

    Returning quietly would put a connection that may still carry another
    account's tenant settings back into rotation.
    """
    conn = _Conn(failing={"RESET SESSION AUTHORIZATION"})

    with pytest.raises(RuntimeError, match="another account"):
        _reset_pooled_connection(conn)

    assert "RESET ALL" in conn.executed, "must still attempt the remaining resets"


def test_a_connection_that_cannot_be_scrubbed_at_all_is_rejected():
    conn = _Conn(failing={"RESET ROLE", "RESET SESSION AUTHORIZATION", "RESET ALL"})

    with pytest.raises(RuntimeError, match="Discarding"):
        _reset_pooled_connection(conn)


def test_reset_leaves_no_open_transaction():
    """The regression: psycopg_pool discarded every connection we reset.

    "connection left in status INTRANS by reset function ...: discarded".
    A stub without transaction state cannot catch this, which is why it only
    surfaced once psycopg was actually installed and the pool ran for real.
    """
    conn = _Conn()

    _reset_pooled_connection(conn)

    assert conn.in_transaction is False
    assert conn.committed is True, "must commit: rollback would undo the RESET"


def test_a_failed_commit_rejects_the_connection_and_leaves_no_transaction():
    conn = _Conn(commit_fails=True)

    with pytest.raises(RuntimeError, match="Could not clear session state"):
        _reset_pooled_connection(conn)

    assert conn.in_transaction is False
    assert conn.rolled_back is True
