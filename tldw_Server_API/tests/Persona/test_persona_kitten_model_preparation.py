"""Kitten readiness loads the selected model, without synthesizing speech."""

from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from tldw_Server_API.app.api.v1.schemas.audio_schemas import OpenAISpeechRequest
from tldw_Server_API.app.core.Persona import live_tts
from tldw_Server_API.app.core.TTS.adapter_registry import TTSAdapterRegistry
from tldw_Server_API.app.core.TTS.adapters import kitten_tts_adapter as kitten
from tldw_Server_API.app.core.TTS.adapters.base import AudioFormat, TTSRequest
from tldw_Server_API.app.core.TTS.tts_exceptions import TTSProviderInitializationError

pytestmark = pytest.mark.unit
DEFAULT_MODEL = "KittenML/kitten-tts-nano-0.8-fp32"
SELECTED_MODEL = "KittenML/kitten-tts-mini-0.8"


@pytest.fixture
def kitten_assets(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    available = {SELECTED_MODEL}
    loaded = []
    synthesized = []

    def load(model_name: str, **kwargs: Any) -> SimpleNamespace:
        loaded.append(model_name)
        if model_name not in available:
            raise FileNotFoundError("Selected model assets are unavailable")
        return SimpleNamespace(repo_id=model_name, revision="fixture-revision")

    real_runtime = kitten.KittenRuntime

    class Runtime:
        sample_rate = 24000
        resolve_voice = real_runtime.resolve_voice

        def __init__(self, assets: SimpleNamespace) -> None:
            self.assets = assets
            self.voice_aliases = dict(kitten.DEFAULT_VOICE_ALIASES)
            self._lower_aliases = {name.lower(): value for name, value in self.voice_aliases.items()}
            self.voices = dict.fromkeys(self.voice_aliases.values())

        def generate(self, text: str, **kwargs: Any) -> np.ndarray:
            synthesized.append((self.assets.repo_id, text))
            return np.zeros(240, dtype=np.float32)

    monkeypatch.setattr(kitten, "download_model_assets", load)
    monkeypatch.setattr(kitten, "KittenRuntime", Runtime)
    return SimpleNamespace(available=available, loaded=loaded, synthesized=synthesized)


@pytest.fixture
def kitten_preparation(monkeypatch: pytest.MonkeyPatch, kitten_assets: SimpleNamespace) -> TTSAdapterRegistry:
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
        def _convert_request(self, request: OpenAISpeechRequest) -> TTSRequest:
            return TTSRequest(text=request.input, model=request.model, voice=request.voice, format=AudioFormat.MP3)

        async def _prepare_generate_speech_request(
            self, **kwargs: Any
        ) -> tuple[kitten.KittenTTSAdapter | None, str, TTSRequest]:
            adapter = await registry.get_adapter("kitten_tts")
            return adapter, "kitten_tts", kwargs["tts_request"]

        def _cleanup_transient_pocket_tts_cpp_voice_path(self, request: TTSRequest) -> None:
            pass

        async def _close_request_adapter(
            self, adapter: kitten.KittenTTSAdapter, overrides: dict[str, Any] | None
        ) -> None:
            assert overrides is None

    @asynccontextmanager
    async def route(**kwargs: Any) -> AsyncIterator[tuple[Service, str, OpenAISpeechRequest, None]]:
        request = OpenAISpeechRequest(model=kwargs["model"], input=kwargs["text"], voice=kwargs["voice"] or "")
        yield Service(), "kitten_tts", request, None

    monkeypatch.setattr(live_tts, "resolve_persona_speech", route)
    return registry


@pytest.mark.asyncio
async def test_kitten_initialization_does_not_require_default_model_assets(kitten_assets: SimpleNamespace) -> None:
    adapter = kitten.KittenTTSAdapter({"model": DEFAULT_MODEL, "auto_download": False})
    assert await adapter.ensure_initialized()
    assert adapter._runtime is None
    assert kitten_assets.loaded == []


@pytest.mark.asyncio
async def test_preparation_loads_selected_kitten_model_and_reuses_it_for_speech(
    kitten_preparation: TTSAdapterRegistry,
    kitten_assets: SimpleNamespace,
) -> None:
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
    kitten_preparation: TTSAdapterRegistry,
    kitten_assets: SimpleNamespace,
    selected_model: str,
) -> None:
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
async def test_preparation_rejects_voice_missing_from_loaded_kitten_runtime(
    kitten_preparation: TTSAdapterRegistry, kitten_assets: SimpleNamespace, voice: str
) -> None:
    with pytest.raises(ValueError, match="not available"):
        await live_tts.prepare_persona_speech(
            {"tts_provider": "kitten_tts", "tts_model": SELECTED_MODEL, "tts_voice": voice}
        )
    assert kitten_assets.synthesized == []


@pytest.mark.asyncio
@pytest.mark.parametrize("voice", ["", "bella", "Bella"])
async def test_preparation_accepts_kitten_default_and_supported_aliases(
    kitten_preparation: TTSAdapterRegistry, kitten_assets: SimpleNamespace, voice: str
) -> None:
    await live_tts.prepare_persona_speech(
        {"tts_provider": "kitten_tts", "tts_model": SELECTED_MODEL, "tts_voice": voice}
    )
    assert kitten_assets.loaded == [SELECTED_MODEL]
    assert kitten_assets.synthesized == []


@pytest.mark.asyncio
async def test_persona_override_does_not_change_cached_default_model(
    kitten_preparation: TTSAdapterRegistry, kitten_assets: SimpleNamespace
) -> None:
    kitten_assets.available.add(DEFAULT_MODEL)
    await live_tts.prepare_persona_speech(
        {"tts_provider": "kitten_tts", "tts_model": SELECTED_MODEL, "tts_voice": "Bella"}
    )
    adapter = await kitten_preparation.get_adapter("kitten_tts")
    response = await adapter.generate(
        TTSRequest(text="Default model speech", voice="Bella", format=AudioFormat.PCM, stream=False)
    )
    assert response.model == DEFAULT_MODEL
    assert kitten_assets.synthesized == [(DEFAULT_MODEL, "Default model speech")]


@pytest.fixture
def kitten_health(
    monkeypatch: pytest.MonkeyPatch, kitten_preparation: TTSAdapterRegistry
) -> Callable[[], Awaitable[dict[str, Any]]]:
    from tldw_Server_API.app.api.v1.endpoints.audio import audio_health
    from tldw_Server_API.app.core.TTS import adapter_registry

    class Service:
        def get_status(self) -> dict[str, Any]:
            return kitten_preparation.get_status_summary()

        async def get_capabilities(self) -> dict[str, Any]:
            return {}

    async def list_capabilities(**kwargs: Any) -> list[dict[str, Any]]:
        # Limit health discovery to the provider under test; no other models load.
        return [{"provider": "kitten_tts", "availability": "enabled", "capabilities": None}]

    async def factory() -> SimpleNamespace:
        return SimpleNamespace(registry=kitten_preparation)

    monkeypatch.setattr(kitten_preparation, "list_capabilities", list_capabilities)
    monkeypatch.setattr(adapter_registry, "get_tts_factory", factory)

    async def health() -> dict[str, Any]:
        return await audio_health.get_tts_health(
            audio_health._build_internal_health_request("/api/v1/audio/health"), Service()
        )

    return health


@pytest.mark.asyncio
async def test_kitten_health_does_not_claim_unloaded_runtime_ready(
    kitten_preparation: TTSAdapterRegistry,
    kitten_health: Callable[[], Awaitable[dict[str, Any]]],
    kitten_assets: SimpleNamespace,
) -> None:
    await kitten_preparation.get_adapter("kitten_tts")
    health = await kitten_health()
    assert health["status"] == "unhealthy"
    assert health["providers"]["available"] == 0
    detail = health["providers"]["details"]["kitten_tts"]
    assert detail["runtime_ready"] is False
    assert detail["runtime_reason"] == "model_not_loaded"
    assert health["capabilities_envelope"][0]["availability"] == "unprepared"
    assert kitten_assets.loaded == []


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [FileNotFoundError, ImportError])
async def test_kitten_health_reports_model_or_dependency_load_failure(
    monkeypatch: pytest.MonkeyPatch,
    kitten_preparation: TTSAdapterRegistry,
    kitten_health: Callable[[], Awaitable[dict[str, Any]]],
    failure: type[Exception],
) -> None:
    adapter = await kitten_preparation.get_adapter("kitten_tts")

    def unavailable(*args: Any, **kwargs: Any) -> None:
        raise failure("private dependency diagnostic")

    monkeypatch.setattr(kitten, "KittenRuntime", unavailable)
    with pytest.raises(TTSProviderInitializationError):
        await adapter._load_runtime_for_model(SELECTED_MODEL)
    health = await kitten_health()
    assert health["status"] == "unhealthy"
    detail = health["providers"]["details"]["kitten_tts"]
    assert detail["runtime_reason"] == "model_load_failed"
    assert detail["failed"] is True
    assert "private dependency diagnostic" not in str(health)


@pytest.mark.asyncio
async def test_kitten_health_recovers_after_selected_model_load(
    kitten_preparation: TTSAdapterRegistry,
    kitten_health: Callable[[], Awaitable[dict[str, Any]]],
    kitten_assets: SimpleNamespace,
) -> None:
    adapter = await kitten_preparation.get_adapter("kitten_tts")
    with pytest.raises(TTSProviderInitializationError):
        await adapter._load_runtime_for_model(DEFAULT_MODEL)
    await live_tts.prepare_persona_speech(
        {"tts_provider": "kitten_tts", "tts_model": SELECTED_MODEL, "tts_voice": "Bella"}
    )
    health = await kitten_health()
    assert health["status"] == "healthy"
    detail = health["providers"]["details"]["kitten_tts"]
    assert detail["runtime_ready"] is True
    assert detail["runtime_reason"] is None
    assert detail["runtime_model"] == SELECTED_MODEL
    assert kitten_assets.synthesized == []
