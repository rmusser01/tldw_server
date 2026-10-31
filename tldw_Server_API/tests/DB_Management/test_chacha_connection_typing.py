import builtins
import sys
from typing import get_type_hints

import pytest

from tldw_Server_API.app.core.DB_Management import ChaChaNotes_DB as chacha_module
from tldw_Server_API.app.core.DB_Management.backends.sqlite_backend import SQLiteBackend


def test_connection_lifecycle_helpers_use_named_raw_connection_alias() -> None:
    alias = getattr(chacha_module, "RawBackendConnection", None)

    assert alias is not None

    release_hints = get_type_hints(
        chacha_module.CharactersRAGDB._release_connection,
        globalns=vars(chacha_module),
    )
    open_hints = get_type_hints(
        chacha_module.CharactersRAGDB._open_new_connection,
        globalns=vars(chacha_module),
    )

    assert release_hints["connection"] == alias
    assert open_hints["return"] == alias


@pytest.mark.parametrize("grouped", [False, True])
def test_close_all_connections_suppresses_noncritical_local_reset(monkeypatch, grouped):
    if grouped and sys.version_info < (3, 12):
        pytest.skip("contextlib.suppress supports exception groups on Python 3.12+")
    from types import SimpleNamespace

    events: list[str] = []

    def fail_reset():
        events.append("thread_local_reset")
        error = ValueError("local reset failed")
        raise builtins.ExceptionGroup("cleanup", [error]) if grouped else error

    database = chacha_module.CharactersRAGDB.__new__(chacha_module.CharactersRAGDB)
    database._backend = SQLiteBackend.__new__(SQLiteBackend)  # PostgreSQL keeps its thread-local
    database._uses_shared_content_backend = False
    database._local = chacha_module.threading.local()
    monkeypatch.setattr(database, "close_connection", lambda: events.append("close_connection"))
    monkeypatch.setattr(
        database.backend,
        "get_pool",
        lambda: SimpleNamespace(close_all=lambda: events.append("pool_close_all")),
    )
    monkeypatch.setattr(chacha_module, "threading", SimpleNamespace(local=fail_reset))
    database.close_all_connections()
    assert events == ["close_connection", "pool_close_all", "thread_local_reset"]


def test_close_all_connections_propagates_unexpected_local_reset(monkeypatch):
    from types import SimpleNamespace

    class UnexpectedCleanupError(Exception):
        pass

    events: list[str] = []
    reset_error = UnexpectedCleanupError("unexpected local reset failure")

    def fail_reset():
        events.append("thread_local_reset")
        raise reset_error

    database = chacha_module.CharactersRAGDB.__new__(chacha_module.CharactersRAGDB)
    database._backend = SQLiteBackend.__new__(SQLiteBackend)  # PostgreSQL keeps its thread-local
    database._uses_shared_content_backend = False
    database._local = chacha_module.threading.local()
    monkeypatch.setattr(database, "close_connection", lambda: events.append("close_connection"))
    monkeypatch.setattr(
        database.backend,
        "get_pool",
        lambda: SimpleNamespace(close_all=lambda: events.append("pool_close_all")),
    )
    monkeypatch.setattr(chacha_module, "threading", SimpleNamespace(local=fail_reset))
    with pytest.raises(UnexpectedCleanupError, match="unexpected local reset failure") as exc_info:
        database.close_all_connections()
    assert exc_info.value is reset_error
    assert events == ["close_connection", "pool_close_all", "thread_local_reset"]
