from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase

import pytest
import yaml

from tldw_Server_API.app.core.TTS.adapter_registry import (
    TTSAdapterFactory,
    TTSAdapterRegistry,
    TTSProvider,
)
from tldw_Server_API.app.core.TTS.tts_request_resolution import resolve_tts_request_defaults


pytestmark = pytest.mark.unit


def test_omnivoice_provider_enum_exposes_member() -> None:
    TestCase().assertEqual(TTSProvider.OMNIVOICE.value, "omnivoice")


@pytest.mark.parametrize("model_alias", ["omnivoice", "omni-voice", "omni_voice"])
def test_omnivoice_model_alias_resolves_to_provider(model_alias: str) -> None:
    factory = TTSAdapterFactory({})

    TestCase().assertEqual(factory.get_provider_for_model(model_alias), TTSProvider.OMNIVOICE)


@pytest.mark.parametrize("provider_alias", ["omnivoice", "omni-voice", "omni_voice"])
def test_omnivoice_provider_aliases_use_auto_voice(
    monkeypatch: pytest.MonkeyPatch,
    provider_alias: str,
) -> None:
    monkeypatch.setattr(
        "tldw_Server_API.app.core.TTS.tts_request_resolution.get_tts_config",
        lambda: SimpleNamespace(default_provider=None, default_voice=None),
    )

    resolved = resolve_tts_request_defaults(
        provider=provider_alias,
        model=None,
        voice=None,
    )

    TestCase().assertEqual(resolved.provider, "omnivoice")
    TestCase().assertEqual(resolved.model, "omnivoice")
    TestCase().assertEqual(resolved.voice, "auto")


@pytest.mark.parametrize("model_alias", ["omnivoice", "omni-voice", "omni_voice"])
def test_omnivoice_model_aliases_use_canonical_defaults(
    monkeypatch: pytest.MonkeyPatch,
    model_alias: str,
) -> None:
    monkeypatch.setattr(
        "tldw_Server_API.app.core.TTS.tts_request_resolution.get_tts_config",
        lambda: SimpleNamespace(default_provider=None, default_voice=None),
    )

    resolved = resolve_tts_request_defaults(
        provider=None,
        model=model_alias,
        voice=None,
    )

    TestCase().assertEqual(resolved.provider, "omnivoice")
    TestCase().assertEqual(resolved.model, "omnivoice")
    TestCase().assertEqual(resolved.voice, "auto")


@pytest.mark.parametrize("model_alias", ["OmniVoice", "OMNIVOICE", "Omni-Voice", "Omni_Voice"])
def test_omnivoice_mixed_case_model_aliases_use_canonical_defaults(
    monkeypatch: pytest.MonkeyPatch,
    model_alias: str,
) -> None:
    monkeypatch.setattr(
        "tldw_Server_API.app.core.TTS.tts_request_resolution.get_tts_config",
        lambda: SimpleNamespace(default_provider=None, default_voice=None),
    )

    resolved = resolve_tts_request_defaults(
        provider=None,
        model=model_alias,
        voice=None,
    )

    TestCase().assertEqual(resolved.provider, "omnivoice")
    TestCase().assertEqual(resolved.model, "omnivoice")
    TestCase().assertEqual(resolved.voice, "auto")


def test_omnivoice_default_adapter_path_is_registered() -> None:
    TestCase().assertEqual(
        TTSAdapterRegistry.DEFAULT_ADAPTERS[TTSProvider.OMNIVOICE],
        "tldw_Server_API.app.core.TTS.adapters.omnivoice_adapter.OmniVoiceAdapter",
    )


@pytest.mark.asyncio
async def test_enabled_omnivoice_provider_returns_unavailable_instead_of_import_error() -> None:
    factory = TTSAdapterFactory(
        {
            "providers": {
                "omnivoice": {
                    "enabled": True,
                }
            }
        }
    )

    adapter = await factory.registry.get_adapter(TTSProvider.OMNIVOICE)

    TestCase().assertIsNone(adapter)


def test_omnivoice_provider_is_disabled_by_default_in_config() -> None:
    config_path = Path(__file__).resolve().parents[3] / "Config_Files" / "tts_providers_config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    providers = config["providers"]
    omnivoice = providers["omnivoice"]

    TestCase().assertIs(omnivoice["enabled"], False)
    TestCase().assertEqual(omnivoice["runtime"], "sidecar")
    TestCase().assertEqual(omnivoice["model"], "omnivoice")
    TestCase().assertEqual(omnivoice["sample_rate"], 24000)
    TestCase().assertEqual(omnivoice["max_concurrent_generations"], 1)
    TestCase().assertEqual(omnivoice["extra_params"]["repo_path"], "../OmniVoice")
    TestCase().assertEqual(omnivoice["extra_params"]["host"], "127.0.0.1")
    TestCase().assertEqual(omnivoice["extra_params"]["port"], 8039)
    TestCase().assertIs(omnivoice["extra_params"]["autoselect_port"], True)
    TestCase().assertIs(omnivoice["extra_params"]["warmup_on_startup"], False)
    TestCase().assertEqual(omnivoice["extra_params"]["idle_shutdown_seconds"], 900)
    TestCase().assertIs(omnivoice["extra_params"]["resident_mode"], False)
