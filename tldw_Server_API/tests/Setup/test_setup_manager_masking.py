import pytest

from tldw_Server_API.app.core.Setup import setup_manager
from tldw_Server_API.app.core.Setup.setup_manager import get_config_snapshot, SENSITIVE_KEY_MARKERS


def test_secret_values_are_masked_in_config_snapshot():


    snapshot = get_config_snapshot()
    sections = snapshot.get("sections", [])

    # Collect any entries that should be treated as secret per server rules
    secret_entries = []
    for section in sections:
        for field in section.get("fields", []):
            key = str(field.get("key", "")).lower()
            if any(marker in key for marker in SENSITIVE_KEY_MARKERS):
                secret_entries.append(field)

    # Sanity: there should be at least one secret-like field in the shipped config
    assert secret_entries, "Expected at least one secret-like field in config snapshot"

    # All secret entries must be masked (empty value) but flagged as secret
    for entry in secret_entries:
        assert entry.get("is_secret") is True
        # The server must not expose the raw value for secrets
        assert entry.get("value") == ""
        # is_set helps clients know if a masked secret exists (optional but desirable)
        assert "is_set" in entry


def test_update_config_preserves_existing_config_when_atomic_replace_fails(tmp_path, monkeypatch):
    config_path = tmp_path / "config.txt"
    original = (
        "[Setup]\n"
        "enable_first_time_setup = true\n"
        "setup_completed = false\n"
        "allow_remote_setup_access = false\n"
    )
    config_path.write_text(original, encoding="utf-8")
    monkeypatch.setattr(setup_manager, "get_config_file_path", lambda: config_path)

    def _raise_replace(_src, _dst):
        raise OSError("replace failed")

    monkeypatch.setattr(setup_manager.os, "replace", _raise_replace)

    with pytest.raises(OSError, match="replace failed"):
        setup_manager.update_config({"Setup": {"setup_completed": True}}, create_backup=False)

    assert config_path.read_text(encoding="utf-8") == original
