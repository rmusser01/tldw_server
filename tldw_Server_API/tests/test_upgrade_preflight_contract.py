"""Upgrade preflight must cover the supported runtime and per-user stores."""

import sqlite3
from types import SimpleNamespace

import pytest
from Helper_Scripts import upgrade_helpers


@pytest.mark.parametrize("version,status", [((3, 11, 9), "fail"), ((3, 12, 0), "ok"), ((3, 13, 0), "ok")])
def test_default_python_floor_matches_supported_runtime(monkeypatch, version, status):
    monkeypatch.setattr(upgrade_helpers, "sys", SimpleNamespace(version_info=version))
    result = upgrade_helpers.check_python_version()
    assert result["status"] == status  # nosec B101 - pytest runtime contract


def test_explicit_python_floor_remains_supported(monkeypatch):
    monkeypatch.setattr(upgrade_helpers, "sys", SimpleNamespace(version_info=(3, 11, 9)))
    result = upgrade_helpers.check_python_version(min_version=(3, 11))
    assert result["status"] == "ok"  # nosec B101 - pytest compatibility contract


def test_unsupported_python_blocks_aggregated_preflight(monkeypatch, tmp_path):
    monkeypatch.setattr(upgrade_helpers, "sys", SimpleNamespace(version_info=(3, 11, 9)))
    result = upgrade_helpers.run_pre_upgrade_checks(db_dir=str(tmp_path), min_disk_mb=1)
    assert not result.can_proceed  # nosec B101 - pytest refusal contract
    assert result.failures[0].startswith("python_version:")  # nosec B101 - pytest failure attribution


@pytest.mark.parametrize("nested", [False, True], ids=["root", "per-user"])
def test_corrupt_database_blocks_preflight(tmp_path, nested):
    parent = tmp_path / "user_databases" / "alice" if nested else tmp_path
    parent.mkdir(parents=True, exist_ok=True)
    (parent / "Media_DB_v2.db").write_bytes(b"not a SQLite database" * 20)
    result = upgrade_helpers.run_pre_upgrade_checks(db_dir=str(tmp_path), min_disk_mb=1)
    assert not result.can_proceed  # nosec B101 - pytest data-integrity refusal
    assert len(result.failures) == 1  # nosec B101 - pytest complete failure attribution
    assert result.failures[0].startswith("db_integrity:")  # nosec B101 - pytest failure attribution


def test_all_nested_stores_are_checked_without_changing_records(tmp_path):
    stores = [
        tmp_path / "users.db",
        tmp_path / "user_databases" / "alice" / "Media_DB_v2.db",
        tmp_path / "user_databases" / "bob" / "ChaChaNotes.db",
    ]
    for path in stores:
        path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(path) as connection:
            connection.execute("CREATE TABLE sentinel (value TEXT)")
            connection.execute("INSERT INTO sentinel VALUES (?)", (path.name,))
    before = {path: path.read_bytes() for path in stores}
    result = upgrade_helpers.run_pre_upgrade_checks(db_dir=str(tmp_path), min_disk_mb=1)
    integrity_checks = [check for check in result.checks if check["name"] == "db_integrity"]
    assert len(integrity_checks) == len(stores)  # nosec B101 - pytest all-store coverage
    assert result.can_proceed  # nosec B101 - pytest valid-store positive control
    assert {path: path.read_bytes() for path in stores} == before  # nosec B101 - pytest record preservation


@pytest.mark.parametrize("filename", ["user_prompts_v2.sqlite", "legacy.sqlite3"])
@pytest.mark.parametrize("corrupt", [True, False], ids=["corrupt", "valid"])
def test_non_db_sqlite_stores_receive_integrity_checks(tmp_path, filename, corrupt):
    path = tmp_path / "user_databases" / "alice" / "prompts_user_dbs" / filename
    path.parent.mkdir(parents=True)
    if corrupt:
        path.write_bytes(b"not a SQLite database" * 20)
    else:
        with sqlite3.connect(path) as connection:
            connection.execute("CREATE TABLE sentinel (value TEXT)")
            connection.execute("INSERT INTO sentinel VALUES (?)", ("retained Prompt",))
    before = path.read_bytes()
    result = upgrade_helpers.run_pre_upgrade_checks(db_dir=str(tmp_path), min_disk_mb=1)
    checks = [check for check in result.checks if check["name"] == "db_integrity"]
    assert len(checks) == 1  # nosec B101 - one real store must be inspected
    assert checks[0]["message"].startswith(filename + ":")  # nosec B101 - a fresh-install placeholder is not an inspected store
    assert checks[0]["status"] == ("fail" if corrupt else "ok")  # nosec B101 - integrity determines upgrade admission
    assert result.can_proceed is not corrupt  # nosec B101 - corrupt stores block the upgrade
    assert path.read_bytes() == before  # nosec B101 - preflight never changes user records


@pytest.mark.parametrize("blocked_kind", ["database", "directory", "wal", "shm", "journal"])
def test_upgrade_refuses_unwritable_database_and_sqlite_sidecars(monkeypatch, tmp_path, blocked_kind):
    path = tmp_path / "user_databases" / "alice" / "Media_DB_v2.db"
    path.parent.mkdir(parents=True)
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE sentinel (value TEXT)")
        connection.execute("INSERT INTO sentinel VALUES (?)", ("retained source",))
    sidecars = {suffix: path.with_name(path.name + "-" + suffix) for suffix in ("wal", "shm", "journal")}
    for sidecar in sidecars.values():
        sidecar.write_bytes(b"")
    blocked = (
        path if blocked_kind == "database" else path.parent if blocked_kind == "directory" else sidecars[blocked_kind]
    )
    before = {item: item.read_bytes() for item in (path, *sidecars.values())}
    real_access = upgrade_helpers.os.access
    monkeypatch.setattr(
        upgrade_helpers.os,
        "access",
        lambda item, mode: False if type(path)(item) == blocked else real_access(item, mode),
    )
    result = upgrade_helpers.run_pre_upgrade_checks(db_dir=str(tmp_path), min_disk_mb=1)
    access = [check for check in result.checks if check["name"] == "db_write_access"]
    assert len(access) == 1  # nosec B101 - upgrade admission must test write access
    assert access[0]["status"] == "fail"  # nosec B101 - migrations cannot use the denied file/directory
    assert blocked.name in access[0]["message"]  # nosec B101 - identify the permission requiring operator repair
    assert not result.can_proceed  # nosec B101 - integrity alone cannot authorize migration
    assert {item: item.read_bytes() for item in before} == before  # nosec B101 - refusal never modifies database or sidecars


def test_writable_database_keeps_upgrade_positive_control(tmp_path):
    path = tmp_path / "Media_DB_v2.db"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE sentinel (value TEXT)")
    before = path.read_bytes()
    result = upgrade_helpers.run_pre_upgrade_checks(db_dir=str(tmp_path), min_disk_mb=1)
    assert result.can_proceed  # nosec B101 - normal writable deployments still upgrade
    assert path.read_bytes() == before  # nosec B101 - permission checks never write user data


def test_integrity_only_check_keeps_readable_database_contract(monkeypatch, tmp_path):
    path = tmp_path / "Media_DB_v2.db"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE sentinel (value TEXT)")
    monkeypatch.setattr(upgrade_helpers.os, "access", lambda *args: False)
    result = upgrade_helpers.check_database_integrity(str(path))
    assert result["status"] == "ok"  # nosec B101 - standalone integrity remains independent of upgrade write admission
