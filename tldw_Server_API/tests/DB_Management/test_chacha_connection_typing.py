import builtins
import sys
from typing import get_type_hints

import pytest

from tldw_Server_API.app.core.DB_Management import ChaChaNotes_DB as chacha_module


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

    from tldw_Server_API.app.core.DB_Management.backends.base import BackendType

    def fail_reset():
        error = ValueError("local reset failed")
        raise builtins.ExceptionGroup("cleanup", [error]) if grouped else error

    database = SimpleNamespace(
        close_connection=lambda: None,
        backend_type=BackendType.POSTGRESQL,
        backend=SimpleNamespace(
            get_pool=lambda: SimpleNamespace(close_all=lambda: None),
        ),
    )
    monkeypatch.setattr(chacha_module, "threading", SimpleNamespace(local=fail_reset))
    assert chacha_module.CharactersRAGDB.close_all_connections(database) is None


def test_close_all_connections_propagates_unexpected_local_reset(monkeypatch):
    from types import SimpleNamespace

    from tldw_Server_API.app.core.DB_Management.backends.base import BackendType

    class UnexpectedCleanupError(Exception):
        pass

    def fail_reset():
        raise UnexpectedCleanupError("unexpected local reset failure")

    database = SimpleNamespace(
        close_connection=lambda: None,
        backend_type=BackendType.POSTGRESQL,
        backend=SimpleNamespace(
            get_pool=lambda: SimpleNamespace(close_all=lambda: None),
        ),
    )
    monkeypatch.setattr(chacha_module, "threading", SimpleNamespace(local=fail_reset))
    with pytest.raises(UnexpectedCleanupError, match="unexpected local reset failure"):
        chacha_module.CharactersRAGDB.close_all_connections(database)
