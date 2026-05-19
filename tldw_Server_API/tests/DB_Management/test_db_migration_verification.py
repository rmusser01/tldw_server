from pathlib import Path
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.DB_Management.db_migration import DatabaseMigrator


def test_verify_migrations_reports_noncontiguous_available_versions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "app.db"
    db_path.touch()
    migrator = DatabaseMigrator(str(db_path), str(tmp_path / "unused"))

    monkeypatch.setattr(
        migrator,
        "get_applied_migrations",
        lambda: [{"version": 1, "checksum": "a"}],
    )
    monkeypatch.setattr(
        migrator,
        "load_migrations",
        lambda: [
            SimpleNamespace(version=1, checksum="a"),
            SimpleNamespace(version=3, checksum="c"),
        ],
    )

    issues = migrator.verify_migrations()

    assert any(issue["issue"] == "migration_version_gap" for issue in issues)
    assert any(issue["version"] == 2 for issue in issues)


def test_verify_migrations_returns_empty_list_when_no_migration_files_exist(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "app.db"
    db_path.touch()
    migrator = DatabaseMigrator(str(db_path), str(tmp_path / "missing"))

    assert migrator.verify_migrations() == []


def test_verify_migrations_reports_missing_files_when_applied_migrations_exist_and_no_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "app.db"
    db_path.touch()
    migrator = DatabaseMigrator(str(db_path), str(tmp_path / "missing"))

    monkeypatch.setattr(
        migrator,
        "get_applied_migrations",
        lambda: [{"version": 1, "checksum": "a"}],
    )
    monkeypatch.setattr(migrator, "load_migrations", list)

    issues = migrator.verify_migrations()

    assert any(issue["issue"] == "migration_file_missing" for issue in issues)
    assert not any(issue["issue"] == "migration_version_gap" for issue in issues)


def test_verify_migrations_does_not_duplicate_gap_and_missing_file_for_applied_versions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "app.db"
    db_path.touch()
    migrator = DatabaseMigrator(str(db_path), str(tmp_path / "unused"))

    monkeypatch.setattr(
        migrator,
        "get_applied_migrations",
        lambda: [{"version": 2, "checksum": "abc"}],
    )
    monkeypatch.setattr(
        migrator,
        "load_migrations",
        lambda: [
            SimpleNamespace(version=1, checksum="aaa"),
            SimpleNamespace(version=3, checksum="ccc"),
        ],
    )

    issues = migrator.verify_migrations()

    assert any(issue["issue"] == "migration_file_missing" and issue["version"] == 2 for issue in issues)
    assert not any(issue["issue"] == "migration_version_gap" and issue["version"] == 2 for issue in issues)


def test_verify_migrations_preserves_existing_missing_file_checks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "app.db"
    db_path.touch()
    migrator = DatabaseMigrator(str(db_path), str(tmp_path / "unused"))

    monkeypatch.setattr(
        migrator,
        "get_applied_migrations",
        lambda: [{"version": 2, "checksum": "abc"}],
    )
    monkeypatch.setattr(
        migrator,
        "load_migrations",
        lambda: [SimpleNamespace(version=1, checksum="aaa")],
    )

    issues = migrator.verify_migrations()

    assert any(issue["issue"] == "migration_file_missing" for issue in issues)


def test_verify_migrations_preserves_existing_checksum_mismatch_checks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "app.db"
    db_path.touch()
    migrator = DatabaseMigrator(str(db_path), str(tmp_path / "unused"))

    monkeypatch.setattr(
        migrator,
        "get_applied_migrations",
        lambda: [{"version": 1, "checksum": "abc"}],
    )
    monkeypatch.setattr(
        migrator,
        "load_migrations",
        lambda: [SimpleNamespace(version=1, checksum="def")],
    )

    issues = migrator.verify_migrations()

    assert any(issue["issue"] == "checksum_mismatch" for issue in issues)
