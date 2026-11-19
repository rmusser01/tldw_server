"""Security pins for config keys that move into ``config_sections`` (TASK-13950.21, Task 7).

Run by security-required's TEST_MODE-unset step (C4), which runs this directory.
"""

from __future__ import annotations

import configparser

import pydantic
import pytest

from tldw_Server_API.app.core.AuthNZ import settings as authnz_settings

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("source", ["env", "config"])
@pytest.mark.parametrize("value", ["multi-user", "MULTI_USER", ""])
def test_invalid_auth_mode_fails_closed_in_every_source(monkeypatch, source, value):
    """A bad AUTH_MODE, in the environment or in config.txt, must stop AuthNZ rather than fall back."""
    parser = configparser.ConfigParser(interpolation=None)
    parser.add_section("AuthNZ")
    if source == "env":
        monkeypatch.setenv("AUTH_MODE", value)
    else:
        monkeypatch.delenv("AUTH_MODE", raising=False)
        parser.set("AuthNZ", "auth_mode", value)
    monkeypatch.setattr(authnz_settings, "load_comprehensive_config", lambda: parser)
    authnz_settings.reset_settings()
    try:
        with pytest.raises(pydantic.ValidationError):
            authnz_settings.get_settings()
    finally:
        authnz_settings.reset_settings()
