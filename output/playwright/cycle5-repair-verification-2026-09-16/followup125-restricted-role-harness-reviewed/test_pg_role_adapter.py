"""Role qualification/lifecycle guards use synthetic connections only."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


def module():
    spec = importlib.util.spec_from_file_location("matrix_pg_role", Path(__file__).with_name("pg_role_adapter.py"))
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def metadata():
    return {
        "name": "tldw_matrix_1234567890abcdef",
        "session_user": "tldw_matrix_1234567890abcdef",
        "login": True,
        "superuser": False,
        "bypassrls": False,
        "inherit": False,
        "createdb": False,
        "createrole": False,
        "replication": False,
        "memberships": 0,
        "row_security": "on",
        "database_owner": "tldw_matrix_1234567890abcdef",
    }


@pytest.mark.parametrize(
    "field,value",
    [
        ("superuser", True),
        ("bypassrls", True),
        ("createdb", True),
        ("createrole", True),
        ("inherit", True),
        ("replication", True),
        ("login", False),
        ("memberships", 1),
        ("row_security", "off"),
        ("session_user", "fixture_admin"),
        ("database_owner", "fixture_admin"),
    ],
)
def test_role_catalog_guard_rejects_privileged_or_mismatched_session(field, value):
    m = module()
    row = metadata()
    expected = row["name"]
    row[field] = value
    with pytest.raises(RuntimeError, match="role"):
        m.require_runtime_role(row, expected)


def test_role_catalog_guard_accepts_complete_direct_non_bypass_identity():
    m = module()
    row = metadata()
    assert m.require_runtime_role(row, row["name"]) == row


@pytest.mark.parametrize("fail_create,fail_body", [(False, False), (True, False), (False, True)])
def test_generated_role_cleanup_is_exact_and_only_after_successful_creation(monkeypatch, fail_create, fail_body):
    m = module()
    events = []

    class Connection:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def execute(self, query, params=None):
            text = query.as_string() if hasattr(query, "as_string") else query
            events.append(text)
            if fail_create and text.startswith("CREATE ROLE"):
                raise RuntimeError("controlled creation failure")
            return SimpleNamespace(fetchone=lambda: None)

    monkeypatch.setattr(m, "connect", lambda *_args, **_kwargs: Connection())
    admin = {"host": "127.0.0.1", "port": 55475, "user": "fixture_admin", "password": "synthetic-admin"}  # nosec B105 - fake connection only
    try:
        with m.runtime_server(admin) as runtime:
            assert runtime["user"] != admin["user"]
            assert runtime["password"] != admin["password"]
            assert events[0].startswith("CREATE ROLE")
            if fail_body:
                raise RuntimeError("controlled body failure")
    except RuntimeError:
        assert fail_create or fail_body
    drops = [event for event in events if event.startswith("DROP ROLE")]
    assert len(drops) == (0 if fail_create else 1)
    assert not any("DROP OWNED" in event or "DROP DATABASE" in event or "CREATE DATABASE" in event for event in events)
    if drops:
        assert runtime["user"] in drops[0]


def test_qualification_revokes_createdb_before_two_direct_login_checks(monkeypatch):
    m = module()
    events = []
    server = {
        "host": "127.0.0.1",
        "port": 55475,
        "user": metadata()["name"],
        "password": "synthetic",  # nosec B105 - fake connection only
        "_matrix_provisioner": {"user": "admin"},
    }

    class Connection:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def execute(self, query):
            events.append(query.as_string())

    monkeypatch.setattr(m, "connect", lambda *_: Connection())
    monkeypatch.setattr(m, "inspect_runtime", lambda database: events.append(database["database"]) or metadata())
    auth = {**server, "database": "tldw_test_aaaaaaaa"}
    content = {**server, "database": "tldw_test_bbbbbbbb"}
    assert m.qualify_runtime(server, auth, content)["createdb"] is False
    assert "NOCREATEDB" in events[0]
    assert events[1:] == [auth["database"], content["database"]]


@pytest.mark.parametrize("wrong", ["user", "password", "host", "port", "database"])
def test_qualification_rejects_foreign_fixture_before_role_change(monkeypatch, wrong):
    m = module()
    server = {
        "host": "127.0.0.1",
        "port": 55475,
        "user": metadata()["name"],
        "password": "synthetic",  # nosec B105 - fake connection only
        "_matrix_provisioner": {"user": "admin"},
    }
    auth = {**server, "database": "tldw_test_aaaaaaaa"}
    content = {**server, "database": "tldw_test_bbbbbbbb", wrong: "foreign"}
    monkeypatch.setattr(m, "connect", lambda *_: pytest.fail("No connection expected"))
    with pytest.raises(RuntimeError, match="target"):
        m.qualify_runtime(server, auth, content)
