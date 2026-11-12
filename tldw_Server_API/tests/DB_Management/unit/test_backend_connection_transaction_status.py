"""Connection status must reflect PostgreSQL activity without guessing idle."""

from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.DB_Management.backends.base import BackendType
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import BackendConnectionWrapper


@pytest.mark.parametrize("status", ["IDLE", "ACTIVE", "INTRANS", "INERROR", "UNKNOWN", None])
@pytest.mark.parametrize("operation_depth,backend_depth", [(0, 0), (1, 0), (0, 1)])
def test_postgres_transaction_status_includes_raw_and_managed_activity(status, operation_depth, backend_depth):
    wrapper = object.__new__(BackendConnectionWrapper)
    wrapper._connection = SimpleNamespace(info=SimpleNamespace(transaction_status=SimpleNamespace(name=status)))
    wrapper._operation_state = SimpleNamespace(tx_depth=operation_depth)
    wrapper._backend = SimpleNamespace(backend_type=BackendType.POSTGRESQL, _tx_depth=lambda connection: backend_depth)
    assert wrapper.in_transaction is bool(operation_depth or backend_depth or status != "IDLE")


def test_postgres_missing_driver_status_is_not_proven_idle():
    wrapper = object.__new__(BackendConnectionWrapper)
    wrapper._connection = SimpleNamespace()
    wrapper._operation_state = None
    wrapper._operation_owned = False
    wrapper._backend = SimpleNamespace(backend_type=BackendType.POSTGRESQL, _tx_depth=lambda connection: 0)
    assert wrapper.in_transaction is True


@pytest.mark.parametrize("active", [False, True])
def test_sqlite_passes_through_connection_status(active):
    wrapper = object.__new__(BackendConnectionWrapper)
    wrapper._connection = SimpleNamespace(in_transaction=active)
    wrapper._backend = SimpleNamespace(backend_type=BackendType.SQLITE)
    assert wrapper.in_transaction is active


@pytest.mark.parametrize(
    "operation_owned,legacy_depth,expected", [(True, 1, True), (True, 0, False), (False, 1, False)]
)
def test_legacy_managed_depth_applies_only_to_operation_owned_wrappers(operation_owned, legacy_depth, expected):
    wrapper = object.__new__(BackendConnectionWrapper)
    wrapper._connection = SimpleNamespace(info=SimpleNamespace(transaction_status=SimpleNamespace(name="IDLE")))
    wrapper._operation_state = None
    wrapper._operation_owned = operation_owned
    wrapper._db = SimpleNamespace(_connection_state=lambda: SimpleNamespace(tx_depth=legacy_depth))
    wrapper._backend = SimpleNamespace(backend_type=BackendType.POSTGRESQL, _tx_depth=lambda connection: 0)
    assert wrapper.in_transaction is expected
