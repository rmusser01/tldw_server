from __future__ import annotations

from pathlib import Path

from tldw_Server_API.app.core.config import clear_config_cache, settings as app_settings
from tldw_Server_API.app.core.Sandbox.store import get_store


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


def test_store_persists_vz_linux_session_control_metadata(monkeypatch, tmp_path: Path) -> None:
    _configure_sqlite_store(monkeypatch, tmp_path)

    store_a = get_store()
    store_a.put_vz_session_control(
        session_id="sess-1",
        runtime="vz_linux",
        vm_id="vm-session-1",
        template_id="vz_linux:ubuntu-24.04",
        workspace_mount="/tmp/ws",
        agent_ready=True,
        helper_instance_id="helper-a",
        helper_started_at="2026-05-09T00:00:00Z",
    )

    store_b = get_store()
    row = store_b.get_vz_session_control("sess-1")

    assert row is not None
    assert row["vm_id"] == "vm-session-1"
    assert row["template_id"] == "vz_linux:ubuntu-24.04"
    assert row["workspace_mount"] == "/tmp/ws"
    assert row["agent_ready"] is True
    assert row["helper_instance_id"] == "helper-a"
    assert row["helper_started_at"] == "2026-05-09T00:00:00Z"
    assert store_b.delete_vz_session_control("sess-1") is True
    assert store_b.get_vz_session_control("sess-1") is None
