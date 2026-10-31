from __future__ import annotations

from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.TTS import tts_request_resolution


pytestmark = pytest.mark.unit


def test_resolve_tts_request_defaults_uses_kitten_defaults_when_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        tts_request_resolution,
        "get_tts_config",
        lambda: SimpleNamespace(default_provider=None, default_voice=None),
    )

    resolved = tts_request_resolution.resolve_tts_request_defaults(
        provider=None,
        model=None,
        voice=None,
    )

    assert resolved.provider == "kitten_tts"
    assert resolved.model == "KittenML/kitten-tts-nano-0.8"
    assert resolved.voice == "Bella"


def test_resolve_tts_request_defaults_prefers_configured_default_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        tts_request_resolution,
        "get_tts_config",
        lambda: SimpleNamespace(default_provider="openai", default_voice="shimmer"),
    )

    resolved = tts_request_resolution.resolve_tts_request_defaults(
        provider=None,
        model=None,
        voice=None,
    )

    assert resolved.provider == "openai"
    assert resolved.model == "tts-1"
    assert resolved.voice == "shimmer"


def test_resolve_tts_request_defaults_maps_openai_provider_to_valid_model() -> None:
    resolved = tts_request_resolution.resolve_tts_request_defaults(
        provider="openai",
        model=None,
        voice=None,
    )

    assert resolved.provider == "openai"
    assert resolved.model == "tts-1"
    assert resolved.voice == "alloy"


def test_resolve_tts_request_defaults_infers_openai_gpt_4o_mini_tts() -> None:
    resolved = tts_request_resolution.resolve_tts_request_defaults(
        provider=None,
        model="gpt-4o-mini-tts",
        voice=None,
    )

    assert resolved.provider == "openai"
    assert resolved.model == "gpt-4o-mini-tts"
    assert resolved.voice == "alloy"


def test_resolve_tts_request_defaults_infers_provider_from_model_alias() -> None:
    resolved = tts_request_resolution.resolve_tts_request_defaults(
        provider=None,
        model="KittenML/kitten-tts-nano-0.8",
        voice=None,
    )

    assert resolved.provider == "kitten_tts"
    assert resolved.model == "KittenML/kitten-tts-nano-0.8"
    assert resolved.voice == "Bella"
