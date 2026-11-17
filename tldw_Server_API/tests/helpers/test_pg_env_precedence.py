from urllib.parse import quote

import pytest

from tldw_Server_API.tests.helpers.pg_env import _parse_dsn, get_pg_env


def test_get_pg_env_prefers_jobs_over_test_database_url(monkeypatch):
    monkeypatch.setenv("JOBS_DB_URL", "postgresql://jobs_user:jobs_pwd@host1:5555/jobs_db")
    monkeypatch.setenv("TEST_DATABASE_URL", "postgresql://test_user:test_pwd@host2:6666/test_db")
    pg = get_pg_env()
    assert pg.dsn.startswith("postgresql://jobs_user:jobs_pwd@host1:5555/jobs_db")


def test_get_pg_env_builds_from_container_style(monkeypatch):
    for key in [
        "JOBS_DB_URL",
        "POSTGRES_TEST_DSN",
        "TEST_DATABASE_URL",
        "DATABASE_URL",
    ]:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("POSTGRES_TEST_HOST", "127.0.0.1")
    monkeypatch.setenv("POSTGRES_TEST_PORT", "55432")
    monkeypatch.setenv("POSTGRES_TEST_USER", "tldw")
    monkeypatch.setenv("POSTGRES_TEST_PASSWORD", "tldw")
    monkeypatch.setenv("POSTGRES_TEST_DB", "tldw_content")
    pg = get_pg_env()
    assert pg.dsn == "postgresql://tldw:tldw@127.0.0.1:55432/tldw_content"


@pytest.mark.parametrize(
    "user,password,database",
    [
        ("fixture@local", "synthetic:a/b?#%", "fixture/name with space"),
        ("lecteur", "synthetic-é", "données"),
        ("fixture", "synthetic%21", "literal%20name"),
    ],
)
def test_get_pg_env_decodes_uri_components_once(monkeypatch, user, password, database):
    for key in ("JOBS_DB_URL", "POSTGRES_TEST_DSN", "DATABASE_URL"):
        monkeypatch.delenv(key, raising=False)
    dsn = "postgresql://{}:{}@127.0.0.1:15492/{}?connect_timeout=2".format(
        quote(user, safe=""), quote(password, safe=""), quote(database, safe="")
    )
    monkeypatch.setenv("TEST_DATABASE_URL", dsn)
    pg = get_pg_env()
    assert (pg.user, pg.password, pg.database) == (user, password, database)  # nosec B101 -- real fixture URI credential/database boundary.
    assert pg.dsn == dsn  # nosec B101 -- selected URI and query parameters remain unchanged.


def test_get_pg_env_quotes_container_values_for_uri_consumers(monkeypatch):
    for key in ("JOBS_DB_URL", "POSTGRES_TEST_DSN", "TEST_DATABASE_URL", "DATABASE_URL"):
        monkeypatch.delenv(key, raising=False)
    for key, value in {
        "HOST": "127.0.0.1",
        "PORT": "15492",
        "USER": "fixture@local",
        "PASSWORD": "synthetic:a/b?#%",  # nosec B105 -- synthetic encoding sample, never used to connect.
        "DB": "fixture/name with space",
    }.items():
        monkeypatch.setenv("POSTGRES_TEST_" + key, value)
    pg = get_pg_env()
    assert _parse_dsn(pg.dsn) == (pg.host, pg.port, pg.user, pg.password, pg.database)  # nosec B101 -- both fixture consumers must resolve the identical connection.
