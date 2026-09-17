"""Independent actual-wrapper check; no database or runtime I/O."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tldw_Server_API.app.core.DB_Management.backends.base import BackendType, QueryResult
from tldw_Server_API.app.core.DB_Management.chacha.operation_scope import (
    ClosedChaChaOperationError,
    chacha_operation,
    current_connection_state,
)
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import BackendConnectionWrapper


class FixtureDatabase:
    def _prepare_backend_statement(self, query, params):
        return query, params


@pytest.mark.parametrize("later_owner", [False, True], ids=["outside-owner", "pool-reused-in-later-owner"])
def test_escaped_connection_cannot_create_an_unowned_or_relabelled_cursor(later_owner):
    db = FixtureDatabase()
    raw = object()
    pool = Mock()
    execute = Mock(return_value=QueryResult(rows=[{"value": 1}], rowcount=1))
    backend = SimpleNamespace(backend_type=BackendType.POSTGRESQL, get_pool=lambda: pool, execute=execute)
    with chacha_operation(independent=True):
        state = current_connection_state(db)
        state.conn, state.backend_ref = raw, backend
        escaped = BackendConnectionWrapper(db, raw, backend)
    pool.return_connection.assert_called_once_with(raw)

    with chacha_operation(independent=True) if later_owner else nullcontext():
        if later_owner:
            new_state = current_connection_state(db)
            new_state.conn, new_state.backend_ref = raw, backend
        with pytest.raises(ClosedChaChaOperationError):
            escaped.cursor().execute("SELECT 1")
        execute.assert_not_called()
