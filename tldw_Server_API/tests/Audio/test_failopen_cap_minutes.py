import importlib
import os
import types

import loguru
import pytest
from fastapi import APIRouter


pytestmark = pytest.mark.unit


class _LoggerStub:
    def __init__(self):
        self.debug_calls = []
        self.warning_calls = []

    def debug(self, *args, **kwargs):
        self.debug_calls.append((args, kwargs))

    def warning(self, *args, **kwargs):
        self.warning_calls.append((args, kwargs))


_SENSITIVE_LOG_MARKERS = (
    "audio config backend exploded",
    "streaming import leaked",
    "user id leaked",
    "cannot import name",
    "/private/",
)


def _assert_sanitized_debug_log(logger_stub: _LoggerStub, expected_message: str) -> None:
    assert logger_stub.debug_calls
    assert [args[0] for args, _kwargs in logger_stub.debug_calls if args] == [expected_message]
    assert all(not kwargs.get("exc_info") for _args, kwargs in logger_stub.debug_calls)

    rendered_calls = repr(logger_stub.debug_calls)
    for marker in _SENSITIVE_LOG_MARKERS:
        assert marker not in rendered_calls


def _assert_sanitized_warning_log(logger_stub: _LoggerStub, expected_message: str) -> None:
    assert logger_stub.warning_calls
    assert [args[0] for args, _kwargs in logger_stub.warning_calls if args] == [expected_message]
    assert all(not kwargs.get("exc_info") for _args, kwargs in logger_stub.warning_calls)

    rendered_calls = repr(logger_stub.warning_calls)
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


def _import_audio_aggregate_module(monkeypatch, cfg=None):


    import importlib
    package_mod = importlib.import_module("tldw_Server_API.app.api.v1.endpoints.audio")
    aggregate_mod = importlib.import_module("tldw_Server_API.app.api.v1.endpoints.audio.audio")
    if cfg is not None:
        monkeypatch.setattr(package_mod, "load_comprehensive_config", lambda: cfg, raising=True)
        monkeypatch.setattr(aggregate_mod, "load_comprehensive_config", lambda: cfg, raising=True)
    return aggregate_mod


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


def test_aggregate_failopen_env_parse_log_is_sanitized(monkeypatch):
    monkeypatch.setenv("AUDIO_FAILOPEN_CAP_MINUTES", "audio config backend exploded /private/tmp/audio-failopen.ini")
    mod = _import_audio_aggregate_module(monkeypatch, cfg=_fake_cfg({}))
    logger_stub = _LoggerStub()
    monkeypatch.setattr(mod, "logger", logger_stub, raising=True)

    assert abs(mod._get_failopen_cap_minutes() - 5.0) < 1e-6

    _assert_sanitized_debug_log(logger_stub, "AUDIO_FAILOPEN_CAP_MINUTES parse failed")


def test_aggregate_failopen_audio_quota_parse_log_is_sanitized(monkeypatch):
    monkeypatch.delenv("AUDIO_FAILOPEN_CAP_MINUTES", raising=False)
    cfg = _fake_cfg({
        "Audio-Quota": {"failopen_cap_minutes": "audio config backend exploded /private/tmp/audio-failopen.ini"}
    })
    mod = _import_audio_aggregate_module(monkeypatch, cfg=cfg)
    logger_stub = _LoggerStub()
    monkeypatch.setattr(mod, "logger", logger_stub, raising=True)

    assert abs(mod._get_failopen_cap_minutes() - 5.0) < 1e-6

    _assert_sanitized_debug_log(logger_stub, "[Audio-Quota].failopen_cap_minutes parse failed")


def test_aggregate_failopen_audio_section_parse_log_is_sanitized(monkeypatch):
    monkeypatch.delenv("AUDIO_FAILOPEN_CAP_MINUTES", raising=False)
    cfg = _fake_cfg({
        "Audio": {"failopen_cap_minutes": "audio config backend exploded /private/tmp/audio-failopen.ini"}
    })
    mod = _import_audio_aggregate_module(monkeypatch, cfg=cfg)
    logger_stub = _LoggerStub()
    monkeypatch.setattr(mod, "logger", logger_stub, raising=True)

    assert abs(mod._get_failopen_cap_minutes() - 5.0) < 1e-6

    _assert_sanitized_debug_log(logger_stub, "[Audio].failopen_cap_minutes parse failed")


def test_aggregate_failopen_config_read_log_is_sanitized(monkeypatch):
    monkeypatch.delenv("AUDIO_FAILOPEN_CAP_MINUTES", raising=False)
    mod = _import_audio_aggregate_module(monkeypatch, cfg=_fake_cfg({}))
    logger_stub = _LoggerStub()
    monkeypatch.setattr(mod, "logger", logger_stub, raising=True)

    def _raise_config_error():
        raise RuntimeError("audio config backend exploded /private/tmp/audio-failopen.ini")

    import importlib
    package_mod = importlib.import_module("tldw_Server_API.app.api.v1.endpoints.audio")
    monkeypatch.setattr(package_mod, "load_comprehensive_config", _raise_config_error, raising=True)
    monkeypatch.setattr(mod, "load_comprehensive_config", _raise_config_error, raising=True)

    assert abs(mod._get_failopen_cap_minutes() - 5.0) < 1e-6

    _assert_sanitized_debug_log(logger_stub, "Config read for failopen cap failed")


def test_aggregate_streaming_route_import_failure_log_is_sanitized(monkeypatch):
    mod = _import_audio_aggregate_module(monkeypatch, cfg=_fake_cfg({}))
    logger_stub = _LoggerStub()
    monkeypatch.setattr(mod, "logger", logger_stub, raising=True)

    def _raise_import_error():
        raise RuntimeError("streaming import leaked /private/audio-streaming.py")

    monkeypatch.setattr(mod, "_load_audio_streaming", _raise_import_error, raising=True)

    router = mod._mount_streaming_routes()

    assert isinstance(router, APIRouter)
    _assert_sanitized_warning_log(logger_stub, "Audio streaming routes unavailable; skipping import")


@pytest.mark.asyncio
async def test_aggregate_resolve_tts_byok_user_id_failure_log_is_sanitized(monkeypatch):
    mod = _import_audio_aggregate_module(monkeypatch, cfg=_fake_cfg({}))
    logger_stub = _LoggerStub()
    monkeypatch.setattr(mod, "logger", logger_stub, raising=True)

    user = types.SimpleNamespace(id="user id leaked /private/user-id.txt")

    result = await mod._resolve_tts_byok(
        provider_hint=None,
        current_user=user,
        request=object(),
    )

    assert result == (None, None, None)
    _assert_sanitized_debug_log(logger_stub, "Failed to extract user_id from current_user")


def test_aggregate_quota_helper_import_logs_are_sanitized(monkeypatch):
    from tldw_Server_API.app.core.Usage import audio_quota

    mod = _import_audio_aggregate_module(monkeypatch, cfg=_fake_cfg({}))
    logger_stub = _LoggerStub()

    with monkeypatch.context() as ctx:
        ctx.setattr(loguru, "logger", logger_stub, raising=True)
        ctx.delattr(audio_quota, "active_streams_count", raising=True)
        ctx.delattr(audio_quota, "can_start_job", raising=True)
        importlib.reload(mod)

    importlib.reload(mod)

    assert [args[0] for args, _kwargs in logger_stub.debug_calls if args] == [
        "audio_quota optional helpers not available",
        "audio_quota job helpers not available",
    ]
    assert all(not kwargs.get("exc_info") for _args, kwargs in logger_stub.debug_calls)

    rendered_calls = repr(logger_stub.debug_calls)
    for marker in _SENSITIVE_LOG_MARKERS:
        assert marker not in rendered_calls
