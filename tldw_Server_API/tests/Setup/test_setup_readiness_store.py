from __future__ import annotations

import threading
import time

import pytest

from tldw_Server_API.app.core.Setup import readiness_store
from tldw_Server_API.app.core.Setup.readiness_store import SetupReadinessStore


def test_readiness_store_defaults_to_not_started(tmp_path):
    store = SetupReadinessStore(tmp_path / "setup_readiness.json")

    readiness = store.load()

    assert readiness["status"] == "not_started"
    assert readiness["operation_id"] is None
    assert readiness["lanes"] == []
    assert readiness["overlays"] == []


def test_readiness_store_round_trips_lane_status(tmp_path):
    store = SetupReadinessStore(tmp_path / "setup_readiness.json")

    saved = store.save(
        {
            "status": "previewed",
            "selected_profile_id": "advanced_custom",
            "lanes": [{"lane_id": "chat", "status": "skipped"}],
            "overlays": ["restart_required"],
            "last_preview": {"preview_id": "preview-1"},
        }
    )

    assert saved["lanes"][0]["status"] == "skipped"
    assert saved["selected_profile_id"] == "advanced_custom"
    assert store.load()["overlays"] == ["restart_required"]


def test_readiness_store_removes_temp_file_on_atomic_replace_failure(tmp_path, monkeypatch):
    readiness_path = tmp_path / "setup_readiness.json"
    store = SetupReadinessStore(readiness_path)
    store.update(status="previewed", lanes=[{"lane_id": "chat", "status": "skipped"}])
    original_contents = readiness_path.read_text(encoding="utf-8")

    def _raise_replace(_src, _dst):
        raise OSError("replace failed")

    monkeypatch.setattr(readiness_store.os, "replace", _raise_replace)

    with pytest.raises(OSError, match="replace failed"):
        store.update(status="failed")

    assert readiness_path.read_text(encoding="utf-8") == original_contents
    assert list(tmp_path.glob("setup_readiness.json.*.tmp")) == []


def test_resolve_readiness_file_does_not_follow_fixed_probe_symlink(tmp_path, monkeypatch):
    readiness_path = tmp_path / "setup_readiness.json"
    symlink_target = tmp_path / "should_not_change.txt"
    symlink_target.write_text("keep", encoding="utf-8")
    (tmp_path / ".write_test").symlink_to(symlink_target)
    monkeypatch.setenv("TLDW_SETUP_READINESS_FILE", str(readiness_path))

    resolved = readiness_store._resolve_readiness_file()

    assert resolved == readiness_path
    assert symlink_target.read_text(encoding="utf-8") == "keep"


def test_readiness_store_update_preserves_concurrent_fields(tmp_path, monkeypatch):
    readiness_path = tmp_path / "setup_readiness.json"
    store = SetupReadinessStore(readiness_path)
    first_load_started = threading.Event()
    original_load = store.load
    load_calls = 0

    def slow_first_load():
        nonlocal load_calls
        load_calls += 1
        data = original_load()
        if load_calls == 1:
            first_load_started.set()
            time.sleep(0.05)
        return data

    monkeypatch.setattr(store, "load", slow_first_load)

    first = threading.Thread(target=lambda: store.update(selected_profile_id="local_performance"))
    second = threading.Thread(target=lambda: store.update(operation_id="op-1"))

    first.start()
    assert first_load_started.wait(timeout=1)
    second.start()
    first.join(timeout=1)
    second.join(timeout=1)

    assert not first.is_alive()
    assert not second.is_alive()
    readiness = store.load()
    assert readiness["selected_profile_id"] == "local_performance"
    assert readiness["operation_id"] == "op-1"
