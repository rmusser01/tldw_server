from __future__ import annotations

from configparser import ConfigParser

import pytest

from tldw_Server_API.app.core import config
from tldw_Server_API.app.core.config_sections import load_config_sections
from tldw_Server_API.app.core.config_sections.stt import load_stt_config


@pytest.fixture(autouse=True)
def _clear_stt_env(monkeypatch) -> None:
    for key in (
        "STT_WS_CONTROL_V2_ENABLED",
        "STT_PAUSED_AUDIO_QUEUE_CAP_SECONDS",
        "STT_OVERFLOW_WARNING_INTERVAL_SECONDS",
        "STT_TRANSCRIPT_DIAGNOSTICS_ENABLED",
        "STT_DELETE_AUDIO_AFTER_SUCCESS",
        "STT_DELETE_AUDIO_AFTER",
        "STT_AUDIO_RETENTION_HOURS",
        "STT_REDACT_PII",
        "STT_ALLOW_UNREDACTED_PARTIALS",
        "STT_REDACT_CATEGORIES",
    ):
        monkeypatch.delenv(key, raising=False)


def test_stt_vnext_defaults_are_bounded() -> None:
    parser = ConfigParser()
    parser.add_section("STT-Settings")

    cfg = load_stt_config(parser, env={})

    assert cfg.ws_control_v2_enabled is False
    assert cfg.paused_audio_queue_cap_seconds == 2.0
    assert cfg.overflow_warning_interval_seconds == 5.0
    assert cfg.transcript_diagnostics_enabled is False
    assert cfg.delete_audio_after_success is True
    assert cfg.audio_retention_hours == 0.0
    assert cfg.redact_pii is False
    assert cfg.allow_unredacted_partials is False
    assert cfg.redact_categories == []


def test_stt_vnext_env_overrides_config_parser() -> None:
    parser = ConfigParser()
    parser.add_section("STT-Settings")
    parser.set("STT-Settings", "ws_control_v2_enabled", "false")
    parser.set("STT-Settings", "paused_audio_queue_cap_seconds", "8")
    parser.set("STT-Settings", "overflow_warning_interval_seconds", "9")
    parser.set("STT-Settings", "transcript_diagnostics_enabled", "false")
    parser.set("STT-Settings", "delete_audio_after_success", "true")
    parser.set("STT-Settings", "audio_retention_hours", "0")
    parser.set("STT-Settings", "redact_pii", "false")
    parser.set("STT-Settings", "allow_unredacted_partials", "false")
    parser.set("STT-Settings", "redact_categories", "email")

    cfg = load_stt_config(
        parser,
        env={
            "STT_WS_CONTROL_V2_ENABLED": "true",
            "STT_PAUSED_AUDIO_QUEUE_CAP_SECONDS": "12.5",
            "STT_OVERFLOW_WARNING_INTERVAL_SECONDS": "7.5",
            "STT_TRANSCRIPT_DIAGNOSTICS_ENABLED": "yes",
            "STT_DELETE_AUDIO_AFTER_SUCCESS": "0",
            "STT_AUDIO_RETENTION_HOURS": "24",
            "STT_REDACT_PII": "1",
            "STT_ALLOW_UNREDACTED_PARTIALS": "true",
            "STT_REDACT_CATEGORIES": " phone, EMAIL ,phone ",
        },
    )

    assert cfg.ws_control_v2_enabled is True
    assert cfg.paused_audio_queue_cap_seconds == 12.5
    assert cfg.overflow_warning_interval_seconds == 7.5
    assert cfg.transcript_diagnostics_enabled is True
    assert cfg.delete_audio_after_success is False
    assert cfg.audio_retention_hours == 24.0
    assert cfg.redact_pii is True
    assert cfg.allow_unredacted_partials is True
    assert cfg.redact_categories == ["phone", "email"]


def test_stt_vnext_accepts_prd_delete_audio_after_config_alias() -> None:
    parser = ConfigParser()
    parser.add_section("STT-Settings")
    parser.set("STT-Settings", "delete_audio_after", "false")

    cfg = load_stt_config(parser, env={})

    assert cfg.delete_audio_after_success is False


def test_stt_vnext_invalid_values_fall_back_or_clamp() -> None:
    parser = ConfigParser()
    parser.add_section("STT-Settings")
    parser.set("STT-Settings", "ws_control_v2_enabled", "maybe")
    parser.set("STT-Settings", "paused_audio_queue_cap_seconds", "-4")
    parser.set("STT-Settings", "overflow_warning_interval_seconds", "-3")
    parser.set("STT-Settings", "transcript_diagnostics_enabled", "unexpected")
    parser.set("STT-Settings", "delete_audio_after_success", "???")
    parser.set("STT-Settings", "audio_retention_hours", "-1")
    parser.set("STT-Settings", "redact_pii", "not-really")
    parser.set("STT-Settings", "allow_unredacted_partials", "hmm")
    parser.set("STT-Settings", "redact_categories", "[]")

    cfg = load_stt_config(parser, env={})

    assert cfg.ws_control_v2_enabled is False
    assert cfg.paused_audio_queue_cap_seconds == 2.0
    assert cfg.overflow_warning_interval_seconds == 5.0
    assert cfg.transcript_diagnostics_enabled is False
    assert cfg.delete_audio_after_success is True
    assert cfg.audio_retention_hours == 0.0
    assert cfg.redact_pii is False
    assert cfg.allow_unredacted_partials is False
    assert cfg.redact_categories == []


def test_config_sections_expose_stt_section() -> None:
    parser = ConfigParser()
    parser.add_section("STT-Settings")

    sections = load_config_sections(parser)

    assert hasattr(sections, "stt")
    assert sections.stt.delete_audio_after_success is True


def test_config_py_exports_canonical_stt_vnext_section(monkeypatch) -> None:
    parser = ConfigParser()
    parser.add_section("STT-Settings")
    parser.set("STT-Settings", "ws_control_v2_enabled", "true")
    parser.set("STT-Settings", "paused_audio_queue_cap_seconds", "9")
    parser.set("STT-Settings", "overflow_warning_interval_seconds", "11")
    parser.set("STT-Settings", "transcript_diagnostics_enabled", "true")
    parser.set("STT-Settings", "delete_audio_after_success", "false")
    parser.set("STT-Settings", "audio_retention_hours", "24")
    parser.set("STT-Settings", "redact_pii", "true")
    parser.set("STT-Settings", "allow_unredacted_partials", "true")
    parser.set("STT-Settings", "redact_categories", "email, ssn")

    def _fake_load_comprehensive_config():
        return parser

    _fake_load_comprehensive_config.cache_clear = lambda: None
    monkeypatch.setattr(config, "load_comprehensive_config", _fake_load_comprehensive_config, raising=True)
    config.clear_config_cache()
    try:
        data = config.load_and_log_configs()
        stt_settings = data["STT_Settings"]
        legacy_stt_settings = data["STT-Settings"]
        exported_stt = config.get_stt_config()
    finally:
        config.clear_config_cache()

    assert stt_settings["ws_control_v2_enabled"] is True
    assert stt_settings["paused_audio_queue_cap_seconds"] == 9.0
    assert stt_settings["overflow_warning_interval_seconds"] == 11.0
    assert stt_settings["transcript_diagnostics_enabled"] is True
    assert stt_settings["delete_audio_after_success"] is False
    assert stt_settings["audio_retention_hours"] == 24.0
    assert stt_settings["redact_pii"] is True
    assert stt_settings["allow_unredacted_partials"] is True
    assert stt_settings["redact_categories"] == ["email", "ssn"]

    assert legacy_stt_settings["ws_control_v2_enabled"] is True
    assert legacy_stt_settings["delete_audio_after_success"] is False
    assert legacy_stt_settings["allow_unredacted_partials"] is True
    assert legacy_stt_settings["redact_categories"] == ["email", "ssn"]

    assert exported_stt["ws_control_v2_enabled"] is True
    assert exported_stt["delete_audio_after_success"] is False
    assert exported_stt["allow_unredacted_partials"] is True
    assert exported_stt["redact_categories"] == ["email", "ssn"]


def test_get_stt_config_falls_back_to_legacy_section_name(monkeypatch) -> None:
    legacy_stt = {
        "ws_control_v2_enabled": True,
        "delete_audio_after_success": False,
        "paused_audio_queue_cap_seconds": 6.5,
        "overflow_warning_interval_seconds": 8.0,
        "redact_pii": True,
    }

    config.clear_config_cache()
    monkeypatch.setattr(config.loaded_config_data, "_data", {"STT-Settings": legacy_stt}, raising=False)
    try:
        exported_stt = config.get_stt_config()
    finally:
        config.clear_config_cache()

    assert exported_stt["ws_control_v2_enabled"] is True
    assert exported_stt["delete_audio_after_success"] is False
    assert exported_stt["paused_audio_queue_cap_seconds"] == 6.5
    assert exported_stt["overflow_warning_interval_seconds"] == 8.0
    assert exported_stt["redact_pii"] is True
