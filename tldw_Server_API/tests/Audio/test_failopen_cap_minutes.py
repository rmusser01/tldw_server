import os
import types

import pytest


pytestmark = pytest.mark.unit


class _LoggerStub:
    def __init__(self):
        self.debug_calls = []

    def debug(self, *args, **kwargs):
        self.debug_calls.append((args, kwargs))


_SENSITIVE_LOG_MARKERS = (
    "audio config backend exploded",
    "/private/tmp/audio-failopen.ini",
)


def _assert_sanitized_debug_log(logger_stub: _LoggerStub, expected_message: str) -> None:
    assert logger_stub.debug_calls
    assert [args[0] for args, _kwargs in logger_stub.debug_calls if args] == [expected_message]
    assert all(not kwargs.get("exc_info") for _args, kwargs in logger_stub.debug_calls)

    rendered_calls = repr(logger_stub.debug_calls)
    for marker in _SENSITIVE_LOG_MARKERS:
        assert marker not in rendered_calls


def _fake_cfg(sections):


    cfg = types.SimpleNamespace()
    def has_section(name: str) -> bool:
        return name in sections
    def get(section: str, key: str, fallback: str = ""):
        try:
            return sections[section][key]
        except Exception:
            return fallback
    cfg.has_section = has_section
    cfg.get = get
    return cfg


def _import_audio_module(monkeypatch, cfg=None):


    # Ensure clean import state
    import importlib
    mod = importlib.import_module("tldw_Server_API.app.api.v1.endpoints.audio")
    # Patch load_comprehensive_config to return our fake cfg
    if cfg is not None:
        monkeypatch.setattr(mod, "load_comprehensive_config", lambda: cfg, raising=True)
    return mod


def test_failopen_default_when_no_env_or_config(monkeypatch):


    # No env, no config
    monkeypatch.delenv("AUDIO_FAILOPEN_CAP_MINUTES", raising=False)
    mod = _import_audio_module(monkeypatch, cfg=None)
    assert abs(mod._get_failopen_cap_minutes() - 5.0) < 1e-6


def test_failopen_env_overrides(monkeypatch):


    monkeypatch.setenv("AUDIO_FAILOPEN_CAP_MINUTES", "7.5")
    mod = _import_audio_module(monkeypatch, cfg=_fake_cfg({}))
    assert abs(mod._get_failopen_cap_minutes() - 7.5) < 1e-6


def test_failopen_audio_quota_overrides_when_no_env(monkeypatch):


    monkeypatch.delenv("AUDIO_FAILOPEN_CAP_MINUTES", raising=False)
    cfg = _fake_cfg({
        "Audio-Quota": {"failopen_cap_minutes": "9.0"}
    })
    mod = _import_audio_module(monkeypatch, cfg=cfg)
    assert abs(mod._get_failopen_cap_minutes() - 9.0) < 1e-6


def test_failopen_audio_section_used_when_no_env_or_audio_quota(monkeypatch):


    monkeypatch.delenv("AUDIO_FAILOPEN_CAP_MINUTES", raising=False)
    cfg = _fake_cfg({
        "Audio": {"failopen_cap_minutes": "6.0"}
    })
    mod = _import_audio_module(monkeypatch, cfg=cfg)
    assert abs(mod._get_failopen_cap_minutes() - 6.0) < 1e-6


def test_failopen_non_positive_env_ignored(monkeypatch):


    monkeypatch.setenv("AUDIO_FAILOPEN_CAP_MINUTES", "0")
    # Provide a config fallback to verify env is ignored and config wins
    cfg = _fake_cfg({
        "Audio-Quota": {"failopen_cap_minutes": "3.5"}
    })
    mod = _import_audio_module(monkeypatch, cfg=cfg)
    assert abs(mod._get_failopen_cap_minutes() - 3.5) < 1e-6


def test_failopen_env_parse_log_is_sanitized(monkeypatch):
    monkeypatch.setenv("AUDIO_FAILOPEN_CAP_MINUTES", "audio config backend exploded /private/tmp/audio-failopen.ini")
    mod = _import_audio_module(monkeypatch, cfg=_fake_cfg({}))
    logger_stub = _LoggerStub()
    monkeypatch.setattr(mod, "logger", logger_stub, raising=True)

    assert abs(mod._get_failopen_cap_minutes() - 5.0) < 1e-6

    _assert_sanitized_debug_log(logger_stub, "AUDIO_FAILOPEN_CAP_MINUTES parse failed")


def test_failopen_audio_quota_parse_log_is_sanitized(monkeypatch):
    monkeypatch.delenv("AUDIO_FAILOPEN_CAP_MINUTES", raising=False)
    cfg = _fake_cfg({
        "Audio-Quota": {"failopen_cap_minutes": "audio config backend exploded /private/tmp/audio-failopen.ini"}
    })
    mod = _import_audio_module(monkeypatch, cfg=cfg)
    logger_stub = _LoggerStub()
    monkeypatch.setattr(mod, "logger", logger_stub, raising=True)

    assert abs(mod._get_failopen_cap_minutes() - 5.0) < 1e-6

    _assert_sanitized_debug_log(logger_stub, "[Audio-Quota].failopen_cap_minutes parse failed")


def test_failopen_audio_section_parse_log_is_sanitized(monkeypatch):
    monkeypatch.delenv("AUDIO_FAILOPEN_CAP_MINUTES", raising=False)
    cfg = _fake_cfg({
        "Audio": {"failopen_cap_minutes": "audio config backend exploded /private/tmp/audio-failopen.ini"}
    })
    mod = _import_audio_module(monkeypatch, cfg=cfg)
    logger_stub = _LoggerStub()
    monkeypatch.setattr(mod, "logger", logger_stub, raising=True)

    assert abs(mod._get_failopen_cap_minutes() - 5.0) < 1e-6

    _assert_sanitized_debug_log(logger_stub, "[Audio].failopen_cap_minutes parse failed")


def test_failopen_config_read_log_is_sanitized(monkeypatch):
    monkeypatch.delenv("AUDIO_FAILOPEN_CAP_MINUTES", raising=False)
    mod = _import_audio_module(monkeypatch, cfg=_fake_cfg({}))
    logger_stub = _LoggerStub()
    monkeypatch.setattr(mod, "logger", logger_stub, raising=True)

    def _raise_config_error():
        raise RuntimeError("audio config backend exploded /private/tmp/audio-failopen.ini")

    monkeypatch.setattr(mod, "load_comprehensive_config", _raise_config_error, raising=True)

    assert abs(mod._get_failopen_cap_minutes() - 5.0) < 1e-6

    _assert_sanitized_debug_log(logger_stub, "Config read for failopen cap failed")
