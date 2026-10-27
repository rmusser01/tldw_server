"""The moves that cut the foundation layer's upward imports (TASK-13950.12)."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


def test_database_error_is_defined_in_core_exceptions_and_reexported_by_authnz() -> None:
    from tldw_Server_API.app.core import exceptions as core_exc
    from tldw_Server_API.app.core.AuthNZ import exceptions as authnz_exc

    assert core_exc.DatabaseError.__module__ == "tldw_Server_API.app.core.exceptions"
    assert authnz_exc.DatabaseError is core_exc.DatabaseError
    assert authnz_exc.UserRegistrationException is core_exc.UserRegistrationException
    assert issubclass(core_exc.DatabaseError, core_exc.UserRegistrationException)
    assert issubclass(authnz_exc.TransactionError, core_exc.DatabaseError)
    assert issubclass(core_exc.AudioQuotaStoreUnavailable, authnz_exc.DatabaseError)
