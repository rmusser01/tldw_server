"""AuthNZ fixtures preserve URI credentials and their existing selection policy."""

from urllib.parse import quote

import pytest

from tldw_Server_API.tests.AuthNZ import conftest as fixtures


@pytest.mark.parametrize(
    "user,password,database",
    [("simple", "simple-password", "tldw_test_uri"), ("user@tenant", "literal%2F:@/?# +", "tldw_test_uri /%")],
)
def test_authnz_fixture_decodes_uri_components_once(user, password, database):
    """Encoded separators and literal percent sequences must reach asyncpg unchanged."""
    uri = f"postgresql://{quote(user, safe='')}:{quote(password, safe='')}@127.0.0.1:15492/{quote(database, safe='')}"
    assert fixtures._parse_pg_dsn(uri) == {
        "host": "127.0.0.1",
        "port": 15492,
        "user": user,
        "password": password,
        "db": database,
    }  # nosec B101 -- synthetic fixture codec regression.


@pytest.mark.parametrize(
    "user,password,database",
    [("simple", "simple-password", "tldw_test_uri"), ("user@tenant", "literal%2F:@/?# +", "tldw_test_uri /%")],
)
def test_authnz_fixture_rebuilds_component_safe_uri(monkeypatch, user, password, database):
    """Every fixture URI uses the same component quoting without changing precedence."""
    for name, value in {
        "TEST_DB_USER": user,
        "TEST_DB_PASSWORD": password,
        "TEST_DB_HOST": "127.0.0.1",
        "TEST_DB_PORT": 15492,
    }.items():
        monkeypatch.setattr(fixtures, name, value)
    expected = (
        f"postgresql://{quote(user, safe='')}:{quote(password, safe='')}@127.0.0.1:15492/{quote(database, safe='')}"
    )
    assert fixtures._pg_test_dsn(database) == expected  # nosec B101 -- exact synthetic URI construction.
