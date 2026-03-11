from __future__ import annotations

import os
import platform
import sys
from pathlib import Path

import pytest

from tldw_Server_API.app.core.config import clear_config_cache, settings as app_settings
from tldw_Server_API.app.core.testing import is_truthy


def _configure_sqlite_store(monkeypatch, tmp_path: Path) -> None:
    db_path = str(tmp_path / "sandbox_store.db")
    root_dir = str(tmp_path / "sandbox_root")
    snapshot_dir = str(tmp_path / "snapshots")
    monkeypatch.setenv("SANDBOX_STORE_BACKEND", "sqlite")
    monkeypatch.setenv("SANDBOX_STORE_DB_PATH", db_path)
    monkeypatch.setenv("SANDBOX_ROOT_DIR", root_dir)
    monkeypatch.setenv("SANDBOX_SNAPSHOT_PATH", snapshot_dir)
    if hasattr(app_settings, "SANDBOX_STORE_BACKEND"):
        monkeypatch.setattr(app_settings, "SANDBOX_STORE_BACKEND", "sqlite")
    if hasattr(app_settings, "SANDBOX_STORE_DB_PATH"):
        monkeypatch.setattr(app_settings, "SANDBOX_STORE_DB_PATH", db_path)
    if hasattr(app_settings, "SANDBOX_ROOT_DIR"):
        monkeypatch.setattr(app_settings, "SANDBOX_ROOT_DIR", root_dir)
    if hasattr(app_settings, "SANDBOX_SNAPSHOT_PATH"):
        monkeypatch.setattr(app_settings, "SANDBOX_SNAPSHOT_PATH", snapshot_dir)
    clear_config_cache()


def _require_vz_linux_real_host_e2e(monkeypatch, tmp_path: Path) -> str:
    _configure_sqlite_store(monkeypatch, tmp_path)
    if sys.platform != "darwin":
        pytest.skip("macOS host only")
    if platform.machine() != "arm64":
        pytest.skip("Apple silicon host only")
    if not is_truthy(os.getenv("TLDW_SANDBOX_VZ_LINUX_E2E")):
        pytest.skip("Set TLDW_SANDBOX_VZ_LINUX_E2E=1 to enable this test")
    base_image = str(os.getenv("TLDW_SANDBOX_VZ_LINUX_E2E_BASE_IMAGE") or "").strip()
    if not base_image:
        pytest.skip("TLDW_SANDBOX_VZ_LINUX_E2E_BASE_IMAGE is required")
    monkeypatch.setenv("SANDBOX_ENABLE_EXECUTION", "1")
    monkeypatch.setenv("SANDBOX_BACKGROUND_EXECUTION", "0")
    return base_image


def test_vz_linux_real_host_e2e_requires_opt_in(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(platform, "machine", lambda: "arm64")
    monkeypatch.delenv("TLDW_SANDBOX_VZ_LINUX_E2E", raising=False)

    with pytest.raises(pytest.skip.Exception, match="TLDW_SANDBOX_VZ_LINUX_E2E"):
        _require_vz_linux_real_host_e2e(monkeypatch, tmp_path)
