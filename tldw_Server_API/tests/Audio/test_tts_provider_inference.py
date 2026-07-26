import sys
from types import SimpleNamespace

import pytest

if sys.version_info < (3, 10):
    pytest.skip("TTS provider inference tests require Python 3.10+", allow_module_level=True)

from tldw_Server_API.app.core.Audio import tts_service
from tldw_Server_API.app.core.Audio.tts_service import (
    _infer_tts_provider_from_model,
    _sanitize_speech_request,
)


@pytest.mark.parametrize(
    "model_name",
    ["supertonic2", "supertonic2-v1", "supertonic-2", "supertonic-2-v1", "tts-supertonic2-1"],
)
def test_infer_tts_provider_supertonic2_aliases(model_name: str) -> None:
    assert _infer_tts_provider_from_model(model_name) == "supertonic2"


@pytest.mark.parametrize(
    "model_name",
    [
        "kitten_tts",
        "kitten-tts",
        "kittentts",
        "KittenML/kitten-tts-nano-0.8",
        "KittenML/kitten-tts-nano-0.8-fp32",
    ],
)
def test_infer_tts_provider_kitten_aliases(model_name: str) -> None:
    assert _infer_tts_provider_from_model(model_name) == "kitten_tts"


@pytest.mark.parametrize(
    "model_name",
    ["qwen3_tts", "qwen3-tts", "Qwen/Qwen3-TTS-12Hz-0.6B-Base"],
)
def test_infer_tts_provider_qwen3_aliases(model_name: str) -> None:
    assert _infer_tts_provider_from_model(model_name) == "qwen3_tts"


@pytest.mark.parametrize(
    ("model_name", "expected_provider"),
    [
        ("gpt-4o-mini-tts", "openai"),
        ("pocket-tts-onnx", "pocket_tts"),
        ("pocket-tts-cpp", "pocket_tts_cpp"),
        ("vibevoice-realtime-0.5b", "vibevoice_realtime"),
        ("s2-pro", "fish_s2"),
        ("lux_tts", "lux_tts"),
    ],
)
def test_infer_tts_provider_uses_authoritative_factory_routes(
    model_name: str,
    expected_provider: str,
) -> None:
    assert _infer_tts_provider_from_model(model_name) == expected_provider


def test_sanitize_speech_request_passes_kitten_provider_hint(monkeypatch) -> None:
    captured: dict[str, str | None] = {}

    class FakeValidator:
        def __init__(self, _config):
            return

        def sanitize_text(self, text: str, provider: str | None = None) -> str:
            captured["provider"] = provider
            return text

    monkeypatch.setattr(tts_service, "get_tts_config", lambda: SimpleNamespace(strict_validation=False))
    monkeypatch.setattr(tts_service, "TTSInputValidator", FakeValidator)

    request = SimpleNamespace(
        model="KittenML/kitten-tts-nano-0.8-fp32",
        input="Hello from KittenTTS",
    )

    provider_hint = _sanitize_speech_request(request, request_id="req-test")

    assert provider_hint == "kitten_tts"
    assert captured["provider"] == "kitten_tts"
