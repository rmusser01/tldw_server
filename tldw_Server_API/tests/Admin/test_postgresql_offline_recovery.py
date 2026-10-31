"""Existing offline helper contract, with only database/tool boundaries mocked."""

from pathlib import Path
from types import SimpleNamespace

import psycopg
import pytest
from Helper_Scripts import pg_backup_restore as cli

from tldw_Server_API.app.core.DB_Management import DB_Backups
from tldw_Server_API.app.services import admin_data_ops_service as data_ops
from tldw_Server_API.tests.Admin.test_postgresql_database_archive import archive
from tldw_Server_API.tests.Admin.test_postgresql_database_bundle import database_scope as database_scope

pytestmark = pytest.mark.unit


def arguments(path, **changed):
    result = {
        "bundle_file": str(path),
        "confirm_database": "c09_database",
        "confirm_offline": True,
        "timeout": 30,
        "no_drop": False,
    }
    result.update(changed)
    return SimpleNamespace(**result)


@pytest.mark.parametrize(
    "changed", [{"confirm_offline": False}, {"confirm_database": "other"}, {"no_drop": True}, {"timeout": 0}]
)
def test_offline_helper_requires_exact_acknowledgements_before_connecting(database_scope, monkeypatch, changed):
    monkeypatch.setattr(
        DB_Backups,
        "assert_postgres_database_offline",
        lambda *a, **kw: pytest.fail("unacknowledged connection"),
        raising=False,
    )
    assert cli._restore_bundle(arguments(database_scope / "not-opened.zip", **changed)) == 2  # nosec B101 -- refuse invalid scope/ack before opening any client.


@pytest.mark.parametrize("others", [0, 1, 2])
def test_offline_session_check_counts_idle_and_active_clients(database_scope, monkeypatch, others):
    commands = []

    class Connection:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            commands.append("closed")

        def execute(self, query, *params):
            commands.append(query)
            return SimpleNamespace(fetchone=lambda: (others,))

    monkeypatch.setattr(psycopg, "connect", lambda *a, **kw: Connection())
    config = data_ops._postgresql_database_config()
    if others:
        with pytest.raises(RuntimeError, match="other_database_clients"):
            DB_Backups.assert_postgres_database_offline(config, timeout_seconds=3)
    else:
        DB_Backups.assert_postgres_database_offline(config, timeout_seconds=3)
    query = next(command for command in commands if "pg_stat_activity" in command)
    assert "pg_stat_activity" in query and "pg_backend_pid()" in query and "state" not in query  # nosec B101 -- do not hide idle or another role's sessions.
    assert commands[-1] == "closed"  # nosec B101 -- no incidental application pool is left behind.


def test_offline_session_check_fails_closed_on_query_error(database_scope, monkeypatch):
    monkeypatch.setattr(
        psycopg, "connect", lambda *a, **kw: (_ for _ in ()).throw(psycopg.OperationalError("fixture denial"))
    )
    with pytest.raises(RuntimeError, match="offline_session_check_failed"):
        DB_Backups.assert_postgres_database_offline(data_ops._postgresql_database_config(), timeout_seconds=3)


def test_offline_helper_safety_then_rechecks_and_restores_restricted_target(database_scope, monkeypatch):
    path = archive(database_scope)
    events = []
    monkeypatch.setattr(DB_Backups.shutil, "which", lambda _: "/fixture/pg_restore")
    monkeypatch.setattr(DB_Backups, "_run_postgres_tool", lambda *a, **kw: SimpleNamespace(returncode=0))
    monkeypatch.setattr(
        DB_Backups,
        "assert_postgres_database_offline",
        lambda config, **kw: events.append(("offline", config.pg_user)),
        raising=False,
    )

    def backup(backend, backup_dir, **kwargs):
        events.append(("safety", backend.config.pg_user))
        output = Path(backup_dir) / "safety.dump"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"PGDMP safety")
        return str(output)

    def restore(backend, dump_file, **kwargs):
        events.append(("restore", backend.config.pg_user, kwargs["drop_first"], Path(dump_file).is_file()))
        return "ok"

    monkeypatch.setattr(DB_Backups, "create_postgres_backup", backup)
    monkeypatch.setattr(DB_Backups, "restore_postgres_backup", restore)
    monkeypatch.setenv("TLDW_POSTGRES_BACKUP_DSN", "postgresql://c09_backup@127.0.0.1:15492/c09_database")
    assert cli._restore_bundle(arguments(path)) == 0  # nosec B101 -- success requires the actual restore boundary to finish.
    assert events == [
        ("offline", "c09_database"),
        ("safety", "c09_backup"),
        ("offline", "c09_database"),
        ("restore", "c09_database", True, True),
    ]  # nosec B101 -- backup privilege never reaches restore; check clients again immediately before overwrite.
    assert not list((database_scope / "backups").glob("tldw_pg_bundle_*"))  # nosec B101 -- completed transient extraction is removed.


def test_offline_helper_refuses_clients_without_safety_or_restore(database_scope, monkeypatch):
    def busy(*args, **kwargs):
        raise RuntimeError("other_database_clients")

    monkeypatch.setattr(DB_Backups, "assert_postgres_database_offline", busy, raising=False)
    monkeypatch.setattr(DB_Backups, "create_postgres_backup", lambda *a, **kw: pytest.fail("busy database dumped"))
    monkeypatch.setattr(
        DB_Backups, "restore_postgres_backup", lambda *a, **kw: pytest.fail("busy database overwritten")
    )
    assert cli._restore_bundle(arguments(archive(database_scope))) == 1  # nosec B101 -- refuse before any safety or overwrite tool.


def test_offline_probe_does_not_claim_success_after_its_deadline(database_scope, monkeypatch):
    now = [0]

    class Connection:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def execute(self, *args):
            now[0] = 4
            return SimpleNamespace(fetchone=lambda: (0,))

    monkeypatch.setattr(DB_Backups.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(psycopg, "connect", lambda *a, **kw: Connection())
    with pytest.raises(RuntimeError, match="offline_session_check_timeout"):
        DB_Backups.assert_postgres_database_offline(data_ops._postgresql_database_config(), timeout_seconds=3)


def test_offline_helper_rejects_late_natural_restore_completion(database_scope, monkeypatch):
    now = [0]
    monkeypatch.setattr(DB_Backups.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(DB_Backups.shutil, "which", lambda _: "/fixture/pg_restore")
    monkeypatch.setattr(DB_Backups, "_run_postgres_tool", lambda *a, **kw: SimpleNamespace(returncode=0))
    monkeypatch.setattr(DB_Backups, "assert_postgres_database_offline", lambda *a, **kw: None)

    def backup(backend, backup_dir, **kwargs):
        output = Path(backup_dir) / "safety.dump"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"PGDMP safety")
        return str(output)

    def restore(*args, **kwargs):
        now[0] = 31
        return "ok"

    monkeypatch.setattr(DB_Backups, "create_postgres_backup", backup)
    monkeypatch.setattr(DB_Backups, "restore_postgres_backup", restore)
    assert cli._restore_bundle(arguments(archive(database_scope))) == 1  # nosec B101 -- no terminal success beyond the whole-operation deadline.
