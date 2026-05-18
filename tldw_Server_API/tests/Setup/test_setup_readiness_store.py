from __future__ import annotations

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
