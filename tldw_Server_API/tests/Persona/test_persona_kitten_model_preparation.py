"""Kitten readiness loads the selected model, without synthesizing speech."""

from contextlib import asynccontextmanager
from types import SimpleNamespace

import numpy as np
import pytest

from tldw_Server_API.app.core.Persona import live_tts
from tldw_Server_API.app.core.TTS.adapter_registry import TTSAdapterRegistry
from tldw_Server_API.app.core.TTS.adapters import kitten_tts_adapter as kitten
from tldw_Server_API.app.core.TTS.adapters.base import AudioFormat, TTSRequest
from tldw_Server_API.app.core.TTS.tts_exceptions import TTSProviderInitializationError

pytestmark = pytest.mark.unit
DEFAULT_MODEL = "KittenML/kitten-tts-nano-0.8-fp32"
SELECTED_MODEL = "KittenML/kitten-tts-mini-0.8"


@pytest.fixture
def kitten_assets(monkeypatch):
    available = {SELECTED_MODEL}
    loaded = []
    synthesized = []

    def load(model_name, **kwargs):
        loaded.append(model_name)
        if model_name not in available:
            raise FileNotFoundError("Selected model assets are unavailable")
        return SimpleNamespace(repo_id=model_name, revision="fixture-revision")

    real_runtime = kitten.KittenRuntime

    class Runtime:
        sample_rate = 24000
        resolve_voice = real_runtime.resolve_voice

        def __init__(self, assets):
            self.assets = assets
            self.voice_aliases = dict(kitten.DEFAULT_VOICE_ALIASES)
            self._lower_aliases = {name.lower(): value for name, value in self.voice_aliases.items()}
            self.voices = dict.fromkeys(self.voice_aliases.values())

        def generate(self, text, **kwargs):
            synthesized.append((self.assets.repo_id, text))
            return np.zeros(240, dtype=np.float32)

    monkeypatch.setattr(kitten, "download_model_assets", load)
    monkeypatch.setattr(kitten, "KittenRuntime", Runtime)
    return SimpleNamespace(available=available, loaded=loaded, synthesized=synthesized)


@pytest.fixture
def kitten_preparation(monkeypatch, kitten_assets):
    registry = TTSAdapterRegistry(
        config={
            "providers": {
                "kitten_tts": {
                    "enabled": True,
                    "model": DEFAULT_MODEL,
                    "auto_download": False,
                }
            }
        }
    )

    class Service:
        def _convert_request(self, request):
            return TTSRequest(text=request.input, model=request.model, voice=request.voice, format=AudioFormat.MP3)

        async def _prepare_generate_speech_request(self, **kwargs):
            adapter = await registry.get_adapter("kitten_tts")
            return adapter, "kitten_tts", kwargs["tts_request"]

        def _cleanup_transient_pocket_tts_cpp_voice_path(self, request):
            pass

        async def _close_request_adapter(self, adapter, overrides):
            assert overrides is None

    @asynccontextmanager
    async def route(**kwargs):
        from tldw_Server_API.app.api.v1.schemas.audio_schemas import OpenAISpeechRequest

        request = OpenAISpeechRequest(model=kwargs["model"], input=kwargs["text"], voice=kwargs["voice"] or "")
        yield Service(), "kitten_tts", request, None

    monkeypatch.setattr(live_tts, "resolve_persona_speech", route)
    return registry


@pytest.mark.asyncio
async def test_kitten_initialization_does_not_require_default_model_assets(kitten_assets):
    adapter = kitten.KittenTTSAdapter({"model": DEFAULT_MODEL, "auto_download": False})
    assert await adapter.ensure_initialized()
    assert adapter._runtime is None
    assert kitten_assets.loaded == []


@pytest.mark.asyncio
async def test_preparation_loads_selected_kitten_model_and_reuses_it_for_speech(
    kitten_preparation,
    kitten_assets,
):
    config = {"tts_provider": "kitten_tts", "tts_model": SELECTED_MODEL, "tts_voice": "Bella"}
    await live_tts.prepare_persona_speech(config)
    await live_tts.prepare_persona_speech(config)
    assert kitten_assets.loaded == [SELECTED_MODEL]
    assert kitten_assets.synthesized == []
    adapter = await kitten_preparation.get_adapter("kitten_tts")
    response = await adapter.generate(
        TTSRequest(
            text="Selected model speech",
            model=SELECTED_MODEL,
            voice="Bella",
            format=AudioFormat.PCM,
            stream=False,
        )
    )
    assert response.audio_data
    assert kitten_assets.loaded == [SELECTED_MODEL]
    assert kitten_assets.synthesized == [(SELECTED_MODEL, "Selected model speech")]


@pytest.mark.asyncio
@pytest.mark.parametrize("selected_model", [SELECTED_MODEL, "Unsupported/kitten-model"])
async def test_preparation_rejects_unavailable_selected_kitten_model(
    kitten_preparation,
    kitten_assets,
    selected_model,
):
    # Mutate the set used by the asset boundary, keeping the default available.
    kitten_assets.available.clear()
    kitten_assets.available.add(DEFAULT_MODEL)
    with pytest.raises((TTSProviderInitializationError, ValueError)):
        await live_tts.prepare_persona_speech(
            {
                "tts_provider": "kitten_tts",
                "tts_model": selected_model,
                "tts_voice": "Bella",
            }
        )
    assert kitten_assets.synthesized == []


@pytest.mark.asyncio
@pytest.mark.parametrize("voice", ["af_heart", "not-a-kitten-voice"])
async def test_preparation_rejects_voice_missing_from_loaded_kitten_runtime(kitten_preparation, kitten_assets, voice):
    with pytest.raises(ValueError, match="not available"):
        await live_tts.prepare_persona_speech(
            {"tts_provider": "kitten_tts", "tts_model": SELECTED_MODEL, "tts_voice": voice}
        )
    assert kitten_assets.synthesized == []


@pytest.mark.asyncio
@pytest.mark.parametrize("voice", ["", "bella", "Bella"])
async def test_preparation_accepts_kitten_default_and_supported_aliases(kitten_preparation, kitten_assets, voice):
    await live_tts.prepare_persona_speech(
        {"tts_provider": "kitten_tts", "tts_model": SELECTED_MODEL, "tts_voice": voice}
    )
    assert kitten_assets.loaded == [SELECTED_MODEL]
    assert kitten_assets.synthesized == []
