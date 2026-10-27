"""DEFAULT_STORAGE_QUOTA_MB is deprecated: values below 100 are accepted and an explicit value warns (spec 2 §7)."""

import pytest
from loguru import logger

from tldw_Server_API.app.core.AuthNZ import settings as settings_mod
from tldw_Server_API.app.core.AuthNZ.settings import Settings

pytestmark = pytest.mark.unit


def test_values_below_100_no_longer_fail_validation() -> None:
    """A deprecated, ignored setting can't stop startup."""
    assert Settings(DEFAULT_STORAGE_QUOTA_MB=0).DEFAULT_STORAGE_QUOTA_MB == 0


def test_env_var_logs_one_deprecation_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    """Loading settings with the env var set logs exactly one deprecation warning."""
    monkeypatch.setenv("DEFAULT_STORAGE_QUOTA_MB", "50")
    settings_mod.reset_settings()
    messages: list[str] = []
    sink_id = logger.add(lambda m: messages.append(str(m)), level="WARNING")
    try:
        settings_mod.get_settings()
        settings_mod.get_settings()
    finally:
        logger.remove(sink_id)
        settings_mod.reset_settings()
    assert len([m for m in messages if "DEFAULT_STORAGE_QUOTA_MB" in m and "deprecated" in m]) == 1
