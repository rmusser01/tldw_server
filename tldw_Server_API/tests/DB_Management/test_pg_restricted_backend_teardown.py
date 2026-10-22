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


def test_teardown_names_the_leaked_session_instead_of_hanging(pytester, monkeypatch):
    root = str(Path(__file__).resolve().parents[3])
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(filter(None, [root, os.getenv("PYTHONPATH")])))
    monkeypatch.setenv("TLDW_TEST_POSTGRES_REQUIRED", "1")
    monkeypatch.delenv("PYTEST_ADDOPTS", raising=False)
    pytester.makeconftest('pytest_plugins = ["tldw_Server_API.tests._plugins.postgres"]\n')
    pytester.makepyfile(test_leak=_LEAKING_TEST)

    result = pytester.runpytest_subprocess("-q", "-p", "no:cacheprovider", timeout=90)

    result.assert_outcomes(passed=1, errors=1)
    result.stdout.fnmatch_lines(["*left 1 session(s) of tldw_rls_*idle in transaction*"])
