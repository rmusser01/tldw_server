import json
import os
import tempfile

import pytest

import tldw_Server_API.app.core.Moderation.moderation_service as moderation_service_module
from tldw_Server_API.app.core.Moderation.moderation_service import ModerationService


@pytest.mark.unit
def test_runtime_overrides_parse_false_string():
    svc = ModerationService()
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as tmp:
        json.dump({"pii_enabled": "false"}, tmp)
        tmp_path = tmp.name
    try:
        svc._runtime_override = {}
        svc._runtime_overrides_path = tmp_path
        svc._load_runtime_overrides_file()
        assert svc._runtime_override.get("pii_enabled") is False
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            _ = None


@pytest.mark.unit
def test_update_settings_can_clear_runtime_overrides():
    svc = ModerationService()
    # Set an override, then clear it explicitly
    svc.update_settings(pii_enabled=True)
    assert svc.get_settings()["pii_enabled"] is True
    svc.update_settings(pii_enabled=None, clear_pii=True)
    assert "pii_enabled" not in svc._runtime_override
    assert svc.get_settings()["pii_enabled"] is None


@pytest.mark.unit
def test_update_settings_persist_failure_does_not_mutate_runtime_override(monkeypatch, tmp_path):
    """Persistence failures are noncritical: the in-memory override is still applied."""
    svc = ModerationService()
    svc._runtime_overrides_path = str(tmp_path / "runtime_overrides.json")
    svc._runtime_override = {"pii_enabled": False}

    def _raise_disk_full(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Moderation.moderation_service.os.fsync",
        _raise_disk_full,
    )

    # Should NOT raise; persistence failure is caught and logged.
    result = svc.update_settings(pii_enabled=True, persist=True)
    assert result is not None
    # The in-memory override should still be applied despite persistence failure.
    assert svc._runtime_override.get("pii_enabled") is True


@pytest.mark.unit
def test_update_settings_persist_failure_sanitizes_warning_log(monkeypatch, tmp_path):
    svc = ModerationService()
    svc._runtime_overrides_path = str(tmp_path / "runtime_overrides.json")
    svc._runtime_override = {"pii_enabled": False}

    def _raise_disk_full(*_args, **_kwargs):
        raise OSError("disk full at /private/moderation-runtime.json")

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Moderation.moderation_service.os.fsync",
        _raise_disk_full,
    )

    messages: list[str] = []
    sink_id = moderation_service_module.logger.add(lambda message: messages.append(str(message)), level="WARNING")
    try:
        result = svc.update_settings(pii_enabled=True, persist=True)
    finally:
        moderation_service_module.logger.remove(sink_id)

    joined = "\n".join(messages)
    assert result is not None
    assert svc._runtime_override.get("pii_enabled") is True
    assert "Failed to persist moderation overrides" in joined
    assert "disk full" not in joined
    assert "moderation-runtime.json" not in joined


@pytest.mark.unit
def test_load_runtime_overrides_file_sanitizes_warning_log(monkeypatch, tmp_path):
    svc = ModerationService()
    overrides_path = tmp_path / "runtime_overrides.json"
    overrides_path.write_text("{}", encoding="utf-8")
    svc._runtime_overrides_path = str(overrides_path)

    def _raise_load_failure(_file_obj):
        raise ValueError("runtime override parse failed at /private/runtime-overrides.json")

    monkeypatch.setattr(moderation_service_module.json, "load", _raise_load_failure)

    messages: list[str] = []
    sink_id = moderation_service_module.logger.add(lambda message: messages.append(str(message)), level="WARNING")
    try:
        svc._load_runtime_overrides_file()
    finally:
        moderation_service_module.logger.remove(sink_id)

    joined = "\n".join(messages)
    assert "Failed to load runtime overrides file" in joined
    assert "runtime override parse failed" not in joined
    assert "runtime-overrides.json" not in joined


@pytest.mark.unit
def test_save_runtime_overrides_file_sanitizes_warning_log(monkeypatch, tmp_path):
    svc = ModerationService()
    svc._runtime_overrides_path = str(tmp_path / "runtime_overrides.json")
    svc._runtime_override = {"pii_enabled": True}

    def _raise_save_failure(_overrides):
        raise OSError("runtime override save failed at /private/runtime-overrides.json")

    monkeypatch.setattr(svc, "_persist_runtime_overrides", _raise_save_failure)

    messages: list[str] = []
    sink_id = moderation_service_module.logger.add(lambda message: messages.append(str(message)), level="WARNING")
    try:
        svc._save_runtime_overrides_file()
    finally:
        moderation_service_module.logger.remove(sink_id)

    joined = "\n".join(messages)
    assert "Failed to save runtime overrides file" in joined
    assert "runtime override save failed" not in joined
    assert "runtime-overrides.json" not in joined


@pytest.mark.unit
def test_load_runtime_overrides_file_sanitizes_invalid_pii_value_log(tmp_path):
    svc = ModerationService()
    overrides_path = tmp_path / "runtime_overrides.json"
    overrides_path.write_text(
        json.dumps({"pii_enabled": "definitely /private/runtime-overrides.json"}),
        encoding="utf-8",
    )
    svc._runtime_overrides_path = str(overrides_path)

    messages: list[str] = []
    sink_id = moderation_service_module.logger.add(lambda message: messages.append(str(message)), level="WARNING")
    try:
        svc._load_runtime_overrides_file()
    finally:
        moderation_service_module.logger.remove(sink_id)

    joined = "\n".join(messages)
    assert "Invalid pii_enabled override value" in joined
    assert "definitely" not in joined
    assert "runtime-overrides.json" not in joined
    assert "pii_enabled" not in svc._runtime_override


@pytest.mark.unit
def test_runtime_overrides_ignore_invalid_string():
    svc = ModerationService()
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as tmp:
        json.dump({"pii_enabled": "nope"}, tmp)
        tmp_path = tmp.name
    try:
        svc._runtime_override = {}
        svc._runtime_overrides_path = tmp_path
        svc._load_runtime_overrides_file()
        assert "pii_enabled" not in svc._runtime_override
        assert svc.get_settings()["pii_enabled"] is None
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            _ = None
