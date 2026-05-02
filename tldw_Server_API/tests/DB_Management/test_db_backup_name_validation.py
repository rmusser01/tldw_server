import pytest
from types import SimpleNamespace

from tldw_Server_API.app.core.DB_Management.DB_Backups import (
    _POSTGRES_BACKUP_EXTS,
    _SQLITE_BACKUP_EXTS,
    create_postgres_backup,
    _sanitize_backup_label,
    _validate_backup_name,
    restore_postgres_backup,
)
from tldw_Server_API.app.core.DB_Management.backends.base import BackendType


@pytest.mark.parametrize(
    ("label", "fallback", "expected"),
    [
        (" My DB!@# ", "db", "My_DB"),
        ("__leading__", "db", "leading"),
        ("--", "fallback", "fallback"),
        ("", "fallback", "fallback"),
    ],
)
def test_sanitize_backup_label(label: str, fallback: str, expected: str) -> None:
    assert _sanitize_backup_label(label, fallback) == expected


def test_sanitize_backup_label_truncates_long_names() -> None:
    label = "a" * 200
    assert _sanitize_backup_label(label, "db") == "a" * 100


@pytest.mark.parametrize("name", ["backup.db", "data.sqlib"])
def test_validate_backup_name_accepts_sqlite_names(name: str) -> None:
    assert _validate_backup_name(name, _SQLITE_BACKUP_EXTS) == name


def test_validate_backup_name_trims_whitespace() -> None:
    assert _validate_backup_name(" backup.db ", _SQLITE_BACKUP_EXTS) == "backup.db"


def test_validate_backup_name_accepts_postgres_dumps() -> None:
    assert _validate_backup_name("content.dump", _POSTGRES_BACKUP_EXTS) == "content.dump"


@pytest.mark.parametrize(
    "name",
    [
        "../evil.db",
        "subdir/evil.db",
        "/abs.db",
        "-flag.db",
        "backup.txt",
        "",
        "   ",
    ],
)
def test_validate_backup_name_rejects_invalid_names(name: str) -> None:
    assert _validate_backup_name(name, _SQLITE_BACKUP_EXTS) is None


def _postgres_backend_stub(*, database: str = "tldw") -> SimpleNamespace:
    return SimpleNamespace(
        backend_type=BackendType.POSTGRESQL,
        config=SimpleNamespace(
            pg_host="localhost",
            pg_port=5432,
            pg_database=database,
            pg_user="postgres",
            pg_password=None,
        ),
    )


def test_restore_postgres_backup_accepts_explicit_dump_path_within_backup_base(
    monkeypatch,
    tmp_path,
) -> None:
    backup_base = tmp_path / "backups"
    dump_dir = backup_base / "authnz" / "1"
    dump_dir.mkdir(parents=True, exist_ok=True)
    dump_path = dump_dir / "authnz_pgdump_20260101_010101.dump"
    dump_path.write_text("dump-bytes", encoding="utf-8")

    monkeypatch.setenv("TLDW_DB_BACKUP_PATH", str(backup_base))
    monkeypatch.setattr(
        "tldw_Server_API.app.core.DB_Management.DB_Backups.shutil.which",
        lambda _name: "/usr/bin/pg_restore",
    )

    observed: dict[str, list[str]] = {}

    def _fake_run(cmd, env, capture_output, text):
        observed["cmd"] = cmd
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(
        "tldw_Server_API.app.core.DB_Management.DB_Backups.subprocess.run",
        _fake_run,
    )

    result = restore_postgres_backup(
        _postgres_backend_stub(database="authdb"),
        str(dump_path),
        drop_first=False,
    )

    assert result == "ok"
    assert observed["cmd"][-1] == str(dump_path)


def test_restore_postgres_backup_rejects_path_outside_backup_base(
    monkeypatch,
    tmp_path,
) -> None:
    backup_base = tmp_path / "backups"
    backup_base.mkdir(parents=True, exist_ok=True)
    outside_dump = tmp_path / "outside.dump"
    outside_dump.write_text("dump", encoding="utf-8")

    monkeypatch.setenv("TLDW_DB_BACKUP_PATH", str(backup_base))
    monkeypatch.setattr(
        "tldw_Server_API.app.core.DB_Management.DB_Backups.shutil.which",
        lambda _name: "/usr/bin/pg_restore",
    )

    result = restore_postgres_backup(_postgres_backend_stub(), str(outside_dump))
    assert result.startswith("dump not found:")


def test_create_postgres_backup_sanitizes_runtime_failures(
    monkeypatch,
    tmp_path,
) -> None:
    backup_base = tmp_path / "backups"
    backup_dir = backup_base / "postgres"
    monkeypatch.setenv("TLDW_DB_BACKUP_PATH", str(backup_base))
    monkeypatch.setattr(
        "tldw_Server_API.app.core.DB_Management.DB_Backups.shutil.which",
        lambda _name: "/usr/bin/pg_dump",
    )

    def _fail_run(*_args, **_kwargs):
        raise RuntimeError("pg_dump leaked password at /private/.pgpass")

    monkeypatch.setattr(
        "tldw_Server_API.app.core.DB_Management.DB_Backups.subprocess.run",
        _fail_run,
    )

    result = create_postgres_backup(_postgres_backend_stub(), str(backup_dir))

    assert result == "pg_dump error"
    assert "leaked password" not in result
    assert "/private/.pgpass" not in result


def test_restore_postgres_backup_sanitizes_runtime_failures(
    monkeypatch,
    tmp_path,
) -> None:
    backup_base = tmp_path / "backups"
    dump_path = backup_base / "authnz_pgdump_20260101_010101.dump"
    backup_base.mkdir(parents=True, exist_ok=True)
    dump_path.write_text("dump-bytes", encoding="utf-8")
    monkeypatch.setenv("TLDW_DB_BACKUP_PATH", str(backup_base))
    monkeypatch.setattr(
        "tldw_Server_API.app.core.DB_Management.DB_Backups.shutil.which",
        lambda _name: "/usr/bin/pg_restore",
    )

    def _fail_run(*_args, **_kwargs):
        raise RuntimeError("pg_restore leaked password at /private/.pgpass")

    monkeypatch.setattr(
        "tldw_Server_API.app.core.DB_Management.DB_Backups.subprocess.run",
        _fail_run,
    )

    result = restore_postgres_backup(_postgres_backend_stub(), str(dump_path))

    assert result == "pg_restore error"
    assert "leaked password" not in result
    assert "/private/.pgpass" not in result
