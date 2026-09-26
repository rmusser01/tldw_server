"""Startup must refuse a Postgres role that is exempt from RLS.

ENABLE ROW LEVEL SECURITY exempts the table owner; FORCE closes that gap, but
neither binds a SUPERUSER or BYPASSRLS role -- Postgres skips RLS for those
entirely. A deployment pointed at such a role enforces no policies at all, no
matter how many are installed, so the server must not start.

Exercised against a stub backend, so no database is required.
"""

from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.DB_Management.media_db.runtime.factory import (
    assert_postgres_role_cannot_bypass_rls,
)

pytestmark = pytest.mark.unit


class _Backend:
    """Minimal stand-in exposing just the execute/first shape the check uses."""

    def __init__(self, row):
        self._row = row
        self.queries: list[str] = []

    def execute(self, sql, connection=None):
        self.queries.append(sql)
        return SimpleNamespace(first=self._row)


def test_ordinary_role_passes():
    backend = _Backend({"rolsuper": False, "rolbypassrls": False})

    assert_postgres_role_cannot_bypass_rls(backend)

    assert "pg_roles" in backend.queries[0]


@pytest.mark.parametrize(
    ("row", "expected"),
    [
        ({"rolsuper": True, "rolbypassrls": False}, "SUPERUSER"),
        ({"rolsuper": False, "rolbypassrls": True}, "BYPASSRLS"),
        ({"rolsuper": True, "rolbypassrls": True}, "SUPERUSER"),
    ],
)
def test_exempt_roles_are_rejected_by_name(row, expected):
    """The regression: these roles make every installed policy decoration."""
    backend = _Backend(row)

    with pytest.raises(RuntimeError, match=expected):
        assert_postgres_role_cannot_bypass_rls(backend)


def test_rejection_names_the_remedy():
    backend = _Backend({"rolsuper": True, "rolbypassrls": False})

    with pytest.raises(RuntimeError, match="NOBYPASSRLS"):
        assert_postgres_role_cannot_bypass_rls(backend)


def test_unreadable_role_fails_closed():
    """An unverifiable role is treated as unsafe, not assumed fine."""
    backend = _Backend(None)

    with pytest.raises(RuntimeError, match="could not read the current role"):
        assert_postgres_role_cannot_bypass_rls(backend)


def test_tuple_rows_are_handled():
    """Some backends return positional rows rather than dicts."""
    with pytest.raises(RuntimeError, match="SUPERUSER"):
        assert_postgres_role_cannot_bypass_rls(_Backend((True, False)))

    assert_postgres_role_cannot_bypass_rls(_Backend((False, False)))
