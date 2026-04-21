from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from tldw_Server_API.app.core.TTS.adapter_registry import TTSAdapterFactory, TTSProvider
from tldw_Server_API.app.core.TTS.tts_request_resolution import resolve_tts_request_defaults


pytestmark = pytest.mark.unit


def test_omnivoice_provider_enum_exposes_member() -> None:
    if TTSProvider.OMNIVOICE.value != "omnivoice":
        raise AssertionError("Expected TTSProvider.OMNIVOICE.value to be 'omnivoice'")


def test_omnivoice_model_alias_resolves_to_provider() -> None:
    factory = TTSAdapterFactory({})

    if factory.get_provider_for_model("omnivoice") != TTSProvider.OMNIVOICE:
        raise AssertionError("Expected 'omnivoice' model alias to resolve to TTSProvider.OMNIVOICE")


def test_omnivoice_defaults_use_auto_voice(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "tldw_Server_API.app.core.TTS.tts_request_resolution.get_tts_config",
        lambda: SimpleNamespace(default_provider=None, default_voice=None),
    )

    resolved = resolve_tts_request_defaults(
        provider="omnivoice",
        model=None,
        voice=None,
    )

    if resolved.provider != "omnivoice":
        raise AssertionError("Expected resolved provider to be 'omnivoice'")
    if resolved.model != "omnivoice":
        raise AssertionError("Expected resolved model to be 'omnivoice'")
    if resolved.voice != "auto":
        raise AssertionError("Expected resolved voice to be 'auto'")


def test_omnivoice_provider_is_disabled_by_default_in_config() -> None:
    config_path = Path(__file__).resolve().parents[3] / "Config_Files" / "tts_providers_config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    providers = config["providers"]
    if providers["omnivoice"]["enabled"] is not False:
        raise AssertionError("Expected OmniVoice to be disabled by default")
    if providers["omnivoice"]["runtime"] != "sidecar":
        raise AssertionError("Expected OmniVoice runtime to be 'sidecar'")
    if providers["omnivoice"]["model"] != "omnivoice":
        raise AssertionError("Expected OmniVoice model to be 'omnivoice'")
