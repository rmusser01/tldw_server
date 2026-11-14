"""pg_restricted_backend's teardown fails fast on a leaked session (TASK-13861, optional item)."""

import os
from pathlib import Path

import pytest

pytest_plugins = ["pytester"]
pytestmark = pytest.mark.integration

_LEAKING_TEST = '''
_kept = []


def test_leaks_a_transaction(pg_restricted_backend):
    conn = pg_restricted_backend.get_pool().get_connection()
    conn.execute("CREATE TABLE leaked (id int)")
    conn.commit()
    conn.execute("SELECT * FROM leaked")  # an open transaction holding a lock on the role's table
    _kept.append(conn)  # never returned, as a leaked checkout is not
'''

# The blocker is the admin role's session, so terminating the role's sessions frees nothing.
_LEAKING_THROUGH_ANOTHER_ROLE = '''
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory

_kept = []


def test_leaks_a_transaction_of_another_role(pg_restricted_backend, pg_database_config):
    with pg_restricted_backend.transaction() as conn:
        conn.execute("CREATE TABLE leaked (id int)")
    admin = DatabaseBackendFactory.create_backend(pg_database_config)
    conn = admin.get_pool().get_connection()
    conn.execute("SELECT * FROM leaked")  # the admin session's open transaction locks the role's table
    _kept.append((admin, conn))
'''


@pytest.mark.parametrize(
    "leaking_test",
    [_LEAKING_TEST, _LEAKING_THROUGH_ANOTHER_ROLE],
    ids=["the-roles-session", "another-roles-session"],
)
def test_teardown_names_the_leaked_session_instead_of_hanging(pytester, monkeypatch, pg_server, leaking_test):
    # pg_server skips this test when PostgreSQL is unreachable, as its siblings do;
    # the subprocess is then forced to require the server it just found.
    root = str(Path(__file__).resolve().parents[3])
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(filter(None, [root, os.getenv("PYTHONPATH")])))
    monkeypatch.setenv("TLDW_TEST_POSTGRES_REQUIRED", "1")
    monkeypatch.delenv("PYTEST_ADDOPTS", raising=False)
    pytester.makeconftest('pytest_plugins = ["tldw_Server_API.tests._plugins.postgres"]\n')
    pytester.makepyfile(test_leak=leaking_test)

    result = pytester.runpytest_subprocess("-q", "-p", "no:cacheprovider", timeout=90)

    result.assert_outcomes(passed=1, errors=1)
    result.stdout.fnmatch_lines(["*left 1 session(s) of tldw_rls_*idle in transaction*SELECT * FROM leaked*"])
