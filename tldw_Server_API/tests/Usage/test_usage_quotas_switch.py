"""The usage-quota master switch (spec 2, Docs/Design/2026-10-02-usage-quota-posture-design.md §1)."""

import configparser

import pytest
from loguru import logger

from tldw_Server_API.app.core import config as cfg

pytestmark = pytest.mark.unit


def _config(enabled: str | None) -> configparser.ConfigParser:
    """A config.txt stand-in, optionally with [Usage-Quotas] enabled set."""
    cp = configparser.ConfigParser()
    if enabled is not None:
        cp.read_dict({"Usage-Quotas": {"enabled": enabled}})
    return cp


@pytest.fixture(autouse=True)
def _clean(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start every test from a stock install: no env, no config section, no warning issued yet."""
    monkeypatch.delenv("USAGE_QUOTAS_ENABLED", raising=False)
    monkeypatch.delenv("LIMIT_ENFORCEMENT_ENABLED", raising=False)
    monkeypatch.setattr(cfg, "_USAGE_QUOTAS_LEGACY_WARNED", False)
    monkeypatch.setattr(cfg, "load_comprehensive_config", lambda: _config(None))


def test_off_by_default() -> None:
    """A stock install (no env, no config section) has usage quotas off."""
    assert cfg.usage_quotas_enabled() is False


@pytest.mark.parametrize("spelling", ["true", "1", "yes", "on"])
def test_config_txt_truthy_spellings(monkeypatch: pytest.MonkeyPatch, spelling: str) -> None:
    """Each common truthy spelling in config.txt's [Usage-Quotas] enabled turns quotas on."""
    monkeypatch.setattr(cfg, "load_comprehensive_config", lambda: _config(spelling))
    assert cfg.usage_quotas_enabled() is True


def test_new_env_beats_legacy_env_and_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """USAGE_QUOTAS_ENABLED overrides both a truthy config.txt setting and the legacy env var."""
    monkeypatch.setattr(cfg, "load_comprehensive_config", lambda: _config("true"))
    monkeypatch.setenv("LIMIT_ENFORCEMENT_ENABLED", "true")
    monkeypatch.setenv("USAGE_QUOTAS_ENABLED", "0")
    assert cfg.usage_quotas_enabled() is False


def test_legacy_env_beats_config_and_warns_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no new env var set, a truthy legacy LIMIT_ENFORCEMENT_ENABLED wins over config.txt and logs its deprecation warning exactly once across repeated calls."""
    monkeypatch.setattr(cfg, "load_comprehensive_config", lambda: _config("false"))
    monkeypatch.setenv("LIMIT_ENFORCEMENT_ENABLED", "true")
    messages: list[str] = []
    handler_id = logger.add(lambda m: messages.append(str(m)), level="WARNING")
    try:
        assert cfg.usage_quotas_enabled() is True
        assert cfg.usage_quotas_enabled() is True
    finally:
        logger.remove(handler_id)
    assert sum("LIMIT_ENFORCEMENT_ENABLED is deprecated" in m for m in messages) == 1


def test_legacy_env_still_turns_quotas_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """A falsy legacy LIMIT_ENFORCEMENT_ENABLED turns quotas off even when config.txt says true."""
    monkeypatch.setattr(cfg, "load_comprehensive_config", lambda: _config("true"))
    monkeypatch.setenv("LIMIT_ENFORCEMENT_ENABLED", "false")
    assert cfg.usage_quotas_enabled() is False


def test_unreadable_config_means_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """A config.txt that fails to parse is treated as quotas off rather than raising."""

    def _broken() -> configparser.ConfigParser:
        """A config loader stand-in that always raises, simulating an unreadable config.txt."""
        raise configparser.Error("bad file")

    monkeypatch.setattr(cfg, "load_comprehensive_config", _broken)
    assert cfg.usage_quotas_enabled() is False
