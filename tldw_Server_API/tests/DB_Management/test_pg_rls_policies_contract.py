import contextlib
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.DB_Management.backends.base import DatabaseError
from tldw_Server_API.app.core.DB_Management.backends.pg_rls_policies import (
    ensure_prompt_studio_rls,
)


class _FailingCursor:
    def __init__(self) -> None:
        self.calls = 0

    def execute(self, _sql: str) -> None:
        self.calls += 1
        if self.calls == 2:
            raise RuntimeError("boom")


class _TxnConn:
    def __init__(self) -> None:
        self.cursor_obj = _FailingCursor()
        self.committed = False
        self.rolled_back = False

    def cursor(self):
        return self.cursor_obj

    def commit(self) -> None:
        self.committed = True

    def rollback(self) -> None:
        self.rolled_back = True


class _Backend:
    backend_type = SimpleNamespace(name="POSTGRESQL")

    def __init__(self, conn: _TxnConn) -> None:
        self._conn = conn

    def transaction(self):
        @contextlib.contextmanager
        def _ctx():
            yield self._conn

        return _ctx()


@pytest.mark.parametrize(
    "backend",
    [
        SimpleNamespace(backend_type=SimpleNamespace(name="SQLITE")),
        object(),
    ],
)
def test_ensure_prompt_studio_rls_returns_false_for_non_postgres_backends(backend):
    if ensure_prompt_studio_rls(backend) is not False:
        pytest.fail("expected non-PostgreSQL backends to be ignored")


def test_ensure_prompt_studio_rls_raises_on_partial_failure():
    conn = _TxnConn()

    with pytest.raises(DatabaseError, match="prompt_studio"):
        ensure_prompt_studio_rls(_Backend(conn))

    if conn.committed is not False:
        pytest.fail("transaction should not commit after a partial failure")
    if conn.rolled_back is not True:
        pytest.fail("transaction should roll back after a partial failure")


def test_run_pg_rls_auto_ensure_logs_success_only_after_both_installers_pass(monkeypatch):
    import tldw_Server_API.app.main as main_mod

    monkeypatch.setattr(main_mod, "ensure_prompt_studio_rls", lambda _backend: True, raising=False)
    monkeypatch.setattr(main_mod, "ensure_chacha_rls", lambda _backend: True, raising=False)

    logged_messages: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    class _LoggerStub:
        def info(self, message: str, *args: object, **kwargs: object) -> None:
            logged_messages.append((message, args, kwargs))

    monkeypatch.setattr(main_mod, "logger", _LoggerStub(), raising=False)

    main_mod._run_pg_rls_auto_ensure(object())

    if not logged_messages:
        pytest.fail("expected startup helper to log the combined RLS result")
    if "PG RLS ensure invoked" not in logged_messages[0][0]:
        pytest.fail("expected startup helper to log the combined RLS result")
