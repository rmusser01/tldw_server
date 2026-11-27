"""Surface-aware effective policy and [Moderation] surface keys (TASK-13950.27 PR B)."""

from __future__ import annotations

import inspect

import pytest
from loguru import logger

import tldw_Server_API.app.core.Moderation.moderation_service as ms_mod
from tldw_Server_API.app.core.config_sections.moderation import load_moderation_surface_config
from tldw_Server_API.app.core.Moderation.models import ModerationPolicy
from tldw_Server_API.app.core.Moderation.moderation_service import ModerationService
from tldw_Server_API.app.core.Moderation.supervised_policy import GuardianModerationProxy

pytestmark = pytest.mark.unit

SURFACE = "rag.generation"


def _service(monkeypatch, tmp_path, **mod_cfg) -> ModerationService:
    cfg = {
        "blocklist_file": str(tmp_path / "bl.txt"),
        "user_overrides_file": str(tmp_path / "uo.json"),
        "runtime_overrides_file": str(tmp_path / "ro.json"),
        **{k: str(v).lower() if isinstance(v, bool) else v for k, v in mod_cfg.items()},
    }
    monkeypatch.setattr(ms_mod, "load_and_log_configs", lambda: {"moderation": cfg})
    return ModerationService()


def test_global_off_surface_on_stays_disabled(monkeypatch, tmp_path):
    svc = _service(monkeypatch, tmp_path, enabled=False, **{f"surface.{SURFACE}": True})
    assert svc.get_effective_policy(None, surface=SURFACE).enabled is False


def test_global_on_surface_off_disables_default(monkeypatch, tmp_path):
    svc = _service(monkeypatch, tmp_path, enabled=True, **{f"surface.{SURFACE}": False})
    assert svc.get_effective_policy(None, surface=SURFACE).enabled is False
    assert svc.get_effective_policy(None, surface="rag.auxiliary").enabled is True
    assert svc._global_policy.enabled is True  # the shared policy is never mutated


def test_global_on_surface_unset_inherits(monkeypatch, tmp_path):
    svc = _service(monkeypatch, tmp_path, enabled=True)
    assert svc.get_effective_policy(None, surface=SURFACE).enabled is True


def test_user_override_enables_when_global_off_and_surface_off(monkeypatch, tmp_path):
    svc = _service(monkeypatch, tmp_path, enabled=True, **{f"surface.{SURFACE}": False})
    svc._user_overrides = {"u1": {"enabled": True}}
    assert svc.get_effective_policy("u1", surface=SURFACE).enabled is True
    assert svc.get_effective_policy("u2", surface=SURFACE).enabled is False


def test_guardian_overlay_enables_when_global_off_and_surface_off(monkeypatch, tmp_path):
    svc = _service(monkeypatch, tmp_path, enabled=True, **{f"surface.{SURFACE}": False})

    class Engine:
        def build_moderation_policy_overlay(self, _uid, base, chat_type=None):  # noqa: ANN001, ANN202
            return ModerationPolicy(enabled=True, input_action=base.input_action)

    proxy = GuardianModerationProxy(svc, Engine(), "child1")
    assert proxy.get_effective_policy("child1", surface=SURFACE).enabled is True


def test_surface_none_means_no_narrowing(monkeypatch, tmp_path):
    svc = _service(monkeypatch, tmp_path, enabled=True, **{f"surface.{SURFACE}": False})
    assert svc.get_effective_policy(None, surface=None).enabled is True


def test_no_narrowing_returns_the_same_object(monkeypatch, tmp_path):
    svc = _service(monkeypatch, tmp_path, enabled=True, **{f"surface.{SURFACE}": True})
    assert svc.get_effective_policy(None, surface=SURFACE) is svc._global_policy
    assert svc.get_effective_policy(None, surface=None) is svc._global_policy
    off = _service(monkeypatch, tmp_path, enabled=False, **{f"surface.{SURFACE}": False})
    assert off.get_effective_policy(None, surface=SURFACE) is off._global_policy


def test_unknown_surface_key_is_ignored_with_one_warning(monkeypatch, tmp_path):
    messages: list[str] = []
    sink = logger.add(lambda m: messages.append(str(m)), level="WARNING")
    try:
        svc = _service(monkeypatch, tmp_path, enabled=True, **{"surface.no.such": False})
    finally:
        logger.remove(sink)
    assert svc.get_effective_policy(None, surface="no.such").enabled is True
    assert len([m for m in messages if "surface.no.such" in m]) == 1


def test_surface_is_a_required_keyword():
    for cls in (ModerationService, GuardianModerationProxy):
        param = inspect.signature(cls.get_effective_policy).parameters["surface"]
        assert param.kind is inspect.Parameter.KEYWORD_ONLY
        assert param.default is inspect.Parameter.empty
    with pytest.raises(TypeError):
        ModerationService.get_effective_policy(object(), "u")  # type: ignore[call-arg]


class _Parser:
    def __init__(self, options: dict[str, str]) -> None:
        self._o = options

    def has_section(self, section: str) -> bool:
        return section == "Moderation"

    def options(self, section: str) -> list[str]:
        return list(self._o)

    def get(self, section, option, *, fallback):  # noqa: ANN001, ANN201
        return self._o.get(option, fallback)


def test_stream_buffer_chars_reads_config_and_env():
    assert load_moderation_surface_config(_Parser({}), {}).stream_buffer_chars == 1024
    assert load_moderation_surface_config(_Parser({"stream_buffer_chars": "2048"}), {}).stream_buffer_chars == 2048
    cfg = load_moderation_surface_config(_Parser({"stream_buffer_chars": "2048"}), {"MODERATION_STREAM_BUFFER_CHARS": "64"})
    assert cfg.stream_buffer_chars == 64
    with pytest.raises(ValueError):
        load_moderation_surface_config(_Parser({"stream_buffer_chars": "0"}), {})
    cfg = load_moderation_surface_config(_Parser({f"surface.{SURFACE}": "false"}), {})
    assert cfg.surface_enabled == {SURFACE: False}
