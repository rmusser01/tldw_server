from __future__ import annotations

import os
import stat

from tldw_Server_API.cli.wizard.utils import env as env_utils


def test_ensure_env_creates_file_with_defaults_and_permissions(tmp_path):
    env_path = tmp_path / ".env"

    result = env_utils.ensure_env(env_path, defaults={"AUTH_MODE": "single_user"})

    assert env_path.exists()
    assert "AUTH_MODE=single_user" in env_path.read_text(encoding="utf-8")
    assert result.created is True
    if os.name != "nt":
        mode = stat.S_IMODE(env_path.stat().st_mode)
        assert mode == 0o600


def test_ensure_env_updates_and_backs_up(tmp_path):
    env_path = tmp_path / ".env"
    original = "AUTH_MODE=multi_user\nAUTH_MODE=single_user\nSINGLE_USER_API_KEY=oldkey\n"
    env_path.write_text(original, encoding="utf-8")

    result = env_utils.ensure_env(
        env_path,
        updates={"AUTH_MODE": "single_user", "SINGLE_USER_API_KEY": "newkey"},
    )

    updated = env_path.read_text(encoding="utf-8")
    assert updated.count("AUTH_MODE=") == 1
    assert "AUTH_MODE=single_user" in updated
    assert "SINGLE_USER_API_KEY=newkey" in updated
    assert result.backup_path is not None
    assert result.backup_path.exists()
    assert result.backup_path.read_text(encoding="utf-8") == original
    if os.name != "nt":
        backup_mode = stat.S_IMODE(result.backup_path.stat().st_mode)
        assert backup_mode == 0o600


def test_mask_env_values():
    values = {
        "SINGLE_USER_API_KEY": "tldw_abcdef123456",
        "AUTH_MODE": "single_user",
    }

    masked = env_utils.mask_env_values(values)

    assert masked["AUTH_MODE"] == "single_user"
    assert masked["SINGLE_USER_API_KEY"] != values["SINGLE_USER_API_KEY"]
    assert masked["SINGLE_USER_API_KEY"].endswith("3456")


def test_mask_env_values_masks_url_credentials():
    values = {
        "DATABASE_URL": "postgresql://tldw_user:TestPassword123!@postgres:5432/tldw_users",
    }

    masked = env_utils.mask_env_values(values)

    assert masked["DATABASE_URL"] == "postgresql://tldw_user:********@postgres:5432/tldw_users"
    assert "TestPassword123!" not in masked["DATABASE_URL"]


def test_mask_env_values_masks_url_credentials_with_malformed_port():
    values = {
        "DATABASE_URL": "postgresql://tldw_user:TestPassword123!@postgres:bad/tldw_users",
    }

    masked = env_utils.mask_env_values(values)

    assert masked["DATABASE_URL"] == "postgresql://tldw_user:********@postgres:bad/tldw_users"
    assert "TestPassword123!" not in masked["DATABASE_URL"]


def test_mask_env_values_fallback_masks_credentials_when_urlsplit_raises(monkeypatch):
    values = {
        "DATABASE_URL": "postgresql://tldw_user:TestPassword123!@postgres:5432/tldw_users",
    }

    def fail_urlsplit(_value):
        raise ValueError("malformed URL")

    monkeypatch.setattr(env_utils, "urlsplit", fail_urlsplit)

    masked = env_utils.mask_env_values(values)

    assert masked["DATABASE_URL"] == "postgresql://tldw_user:********@postgres:5432/tldw_users"
    assert "TestPassword123!" not in masked["DATABASE_URL"]


def test_ensure_env_dry_run_does_not_write_or_backup(tmp_path):
    env_path = tmp_path / ".env"
    original = "AUTH_MODE=single_user\n"
    env_path.write_text(original, encoding="utf-8")

    if os.name != "nt":
        os.chmod(env_path, 0o644)
        original_mode = stat.S_IMODE(env_path.stat().st_mode)
    else:
        original_mode = None

    result = env_utils.ensure_env(env_path, updates={"AUTH_MODE": "multi_user"}, dry_run=True)

    assert env_path.read_text(encoding="utf-8") == original
    assert result.backup_path is None
    assert list(tmp_path.glob(".env.*.bak")) == []
    if original_mode is not None:
        assert stat.S_IMODE(env_path.stat().st_mode) == original_mode


def test_ensure_env_dry_run_does_not_create_file(tmp_path):
    env_path = tmp_path / ".env"
    assert not env_path.exists()

    result = env_utils.ensure_env(env_path, defaults={"AUTH_MODE": "single_user"}, dry_run=True)

    assert not env_path.exists()
    assert result.created is True
    assert result.changed is True
