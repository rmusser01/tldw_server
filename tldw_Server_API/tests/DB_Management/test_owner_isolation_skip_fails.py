"""A skipped ``owner_isolation`` test fails when TLDW_TEST_POSTGRES_REQUIRED is set (TASK-13950.23)."""

import os
from pathlib import Path

import pytest

pytest_plugins = ["pytester"]
pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[3]

_INNER = '''
import pytest


@pytest.mark.owner_isolation
def test_marked_skip():
    pytest.skip("no pg")


def test_unmarked_skip():
    pytest.skip("no pg")


@pytest.mark.owner_isolation
@pytest.mark.xfail(reason="known")
def test_marked_xfail():
    assert False


@pytest.fixture
def skipping_fixture():
    pytest.skip("fixture has no pg")


@pytest.mark.owner_isolation
def test_marked_fixture_skip(skipping_fixture):
    pass
'''


def _run(pytester, monkeypatch, required: bool, *selector: str):
    monkeypatch.setenv("PYTHONPATH", str(_REPO_ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    monkeypatch.delenv("TLDW_TEST_POSTGRES_REQUIRED", raising=False)
    if required:
        monkeypatch.setenv("TLDW_TEST_POSTGRES_REQUIRED", "1")
    pytester.makepyfile(test_inner=_INNER)
    pytester.makeini("[pytest]\nmarkers =\n    owner_isolation: x\n")
    return pytester.runpytest_subprocess(
        "-p", "tldw_Server_API.tests._plugins.postgres", "-p", "no:cacheprovider", "-rfsx", *selector
    )


def test_owner_isolation_skip_fails_when_postgres_required(pytester, monkeypatch):
    result = _run(pytester, monkeypatch, True, "-k", "test_marked_skip")
    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(
        ["*owner_isolation test skipped while TLDW_TEST_POSTGRES_REQUIRED is set: no pg*"]
    )


def test_owner_isolation_skip_stays_a_skip_when_postgres_not_required(pytester, monkeypatch):
    _run(pytester, monkeypatch, False, "-k", "test_marked_skip").assert_outcomes(skipped=1)


def test_unmarked_skip_still_skips_when_postgres_required(pytester, monkeypatch):
    _run(pytester, monkeypatch, True, "-k", "test_unmarked_skip").assert_outcomes(skipped=1)


def test_marked_xfail_stays_xfailed_when_postgres_required(pytester, monkeypatch):
    _run(pytester, monkeypatch, True, "-k", "test_marked_xfail").assert_outcomes(xfailed=1)


def test_marked_fixture_skip_in_setup_fails_when_postgres_required(pytester, monkeypatch):
    _run(pytester, monkeypatch, True, "-k", "test_marked_fixture_skip").assert_outcomes(errors=1)
