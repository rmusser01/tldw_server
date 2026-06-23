from pathlib import Path

import pytest

from tldw_Server_API.app.core.DB_Management import DB_Backups
from tldw_Server_API.app.core.DB_Management.media_db import legacy_backup


pytestmark = pytest.mark.unit


def test_create_automated_backup_sanitizes_helper_failures(monkeypatch):
    def fail_create_backup(*_args):
        raise RuntimeError("backup failed at /private/media.db")

    monkeypatch.setattr(DB_Backups, "create_backup", fail_create_backup)

    message = legacy_backup.create_automated_backup("media.db", "backups")

    assert message == "Failed to create backup."
    assert "backup failed" not in message
    assert "/private/media.db" not in message


def test_create_incremental_backup_sanitizes_helper_failures(monkeypatch):
    def fail_incremental_backup(*_args):
        raise RuntimeError("incremental backup failed at /private/media-wal.db")

    monkeypatch.setattr(DB_Backups, "create_incremental_backup", fail_incremental_backup)

    message = legacy_backup.create_incremental_backup("media.db", "backups")

    assert message == "Failed to create incremental backup."
    assert "incremental backup failed" not in message
    assert "/private/media-wal.db" not in message


def test_rotate_backups_sanitizes_filesystem_failures(monkeypatch, tmp_path):
    def fail_exists(_path):
        raise OSError("cannot stat /private/backups")

    with monkeypatch.context() as scoped_patch:
        scoped_patch.setattr(Path, "exists", fail_exists)
        message = legacy_backup.rotate_backups(tmp_path)

    assert message == "Failed to rotate backups."
    assert "cannot stat" not in message
    assert "/private/backups" not in message
