from tldw_Server_API.app.core import startup_logging


def test_startup_api_key_log_value_masks_by_default(monkeypatch) -> None:
    monkeypatch.delenv("SHOW_API_KEY_ON_STARTUP", raising=False)

    api_key = "sk-abcdefghijklmnopqrstuvwxyz123456"
    display = startup_logging.startup_api_key_log_value(api_key)

    assert display != api_key
    assert display == startup_logging.mask_api_key_for_startup_logs(api_key)


def test_startup_api_key_log_value_shows_full_key_only_when_explicit(monkeypatch) -> None:
    monkeypatch.setenv("SHOW_API_KEY_ON_STARTUP", "true")

    api_key = "sk-abcdefghijklmnopqrstuvwxyz123456"
    display = startup_logging.startup_api_key_log_value(api_key)

    assert display == api_key


def test_normalize_startup_log_level_accepts_case_insensitive_loguru_level() -> None:
    """Normalize a supported Loguru level without case sensitivity."""
    assert startup_logging.normalize_startup_log_level("warning") == "WARNING"


def test_normalize_startup_log_level_falls_back_to_info() -> None:
    """Fall back to INFO when the requested level is unsupported."""
    assert startup_logging.normalize_startup_log_level("not-a-level") == "INFO"


def test_normalize_startup_log_level_defaults_to_info_when_missing() -> None:
    """Default startup logging to INFO when no level is configured."""
    assert startup_logging.normalize_startup_log_level(None) == "INFO"
