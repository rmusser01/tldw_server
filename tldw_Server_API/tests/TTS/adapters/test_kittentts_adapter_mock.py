from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from tldw_Server_API.app.core.TTS.adapters.base import AudioFormat, TTSRequest

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_kittentts_capabilities_report_static_voices_without_runtime_init(monkeypatch):
    from tldw_Server_API.app.core.TTS.adapters import kitten_tts_adapter as mod

    def _unexpected(*_args, **_kwargs):
        raise AssertionError("capability discovery should not initialize the Kitten runtime")

    monkeypatch.setattr(mod, "download_model_assets", _unexpected)
    monkeypatch.setattr(mod, "KittenRuntime", _unexpected)

    adapter = mod.KittenTTSAdapter({"model": "KittenML/kitten-tts-nano-0.8"})
    caps = await adapter.get_capabilities()

    assert caps.provider_name == "KittenTTS"
    assert "Bella" in [voice.name for voice in caps.supported_voices]
    assert AudioFormat.WAV in caps.supported_formats
    assert AudioFormat.MP3 in caps.supported_formats
    assert AudioFormat.PCM in caps.supported_formats
    assert adapter._runtime is None


@pytest.mark.asyncio
async def test_kittentts_generate_non_stream(monkeypatch, tmp_path):
    from tldw_Server_API.app.core.TTS.adapters import kitten_tts_adapter as mod

    fake_assets = SimpleNamespace(
        repo_id="KittenML/kitten-tts-nano-0.8-fp32",
        revision="deadbeef1",
    )
    download_calls = []

    def fake_download(model_name, *, cache_dir=None, auto_download=True, revision=None):
        download_calls.append((model_name, cache_dir, auto_download, revision))
        return fake_assets

    class FakeRuntime:
        instances = []

        def __init__(self, assets):
            self.assets = assets
            self.calls = []
            FakeRuntime.instances.append(self)

        def generate(self, text, *, voice=None, speed=1.0, clean_text=False):
            self.calls.append((text, voice, speed, clean_text))
            return np.linspace(-0.25, 0.25, 2400, dtype=np.float32)

    monkeypatch.setattr(mod, "download_model_assets", fake_download)
    monkeypatch.setattr(mod, "KittenRuntime", FakeRuntime)

    adapter = mod.KittenTTSAdapter(
        {
            "model": "KittenML/kitten-tts-nano-0.8",
            "model_revision": "deadbeef1",
            "cache_dir": str(tmp_path / "cache"),
            "auto_download": False,
            "clean_text": False,
        }
    )

    request = TTSRequest(
        text="Hello from Kitten",
        voice="Bella",
        format=AudioFormat.WAV,
        stream=False,
        speed=1.1,
        extra_params={"clean_text": True},
    )

    response = await adapter.generate(request)

    assert response.audio_data is not None
    assert response.audio_data[:4] == b"RIFF"
    assert response.provider == "kitten_tts"
    assert response.voice_used == "Bella"
    assert response.model == "KittenML/kitten-tts-nano-0.8-fp32"
    assert download_calls == [("KittenML/kitten-tts-nano-0.8-fp32", str(tmp_path / "cache"), False, "deadbeef1")]
    assert FakeRuntime.instances[0].calls == [("Hello from Kitten", "Bella", 1.1, True)]


@pytest.mark.asyncio
async def test_kittentts_generate_stream_returns_pcm_chunks(monkeypatch):
    from tldw_Server_API.app.core.TTS.adapters import kitten_tts_adapter as mod

    monkeypatch.setattr(
        mod,
        "download_model_assets",
        lambda *_args, **_kwargs: SimpleNamespace(
            repo_id="KittenML/kitten-tts-nano-0.8-fp32",
            revision="deadbeef1",
        ),
    )

    class FakeRuntime:
        def __init__(self, _assets):
            return

        def generate(self, text, *, voice=None, speed=1.0, clean_text=False):
            _ = (text, voice, speed, clean_text)
            return np.linspace(-0.5, 0.5, 512, dtype=np.float32)

    monkeypatch.setattr(mod, "KittenRuntime", FakeRuntime)

    adapter = mod.KittenTTSAdapter({"model": "KittenML/kitten-tts-nano-0.8"})
    response = await adapter.generate(
        TTSRequest(
            text="stream please",
            voice="Leo",
            format=AudioFormat.PCM,
            stream=True,
        )
    )

    assert response.audio_stream is not None
    chunks = [chunk async for chunk in response.audio_stream]
    assert chunks
    assert all(isinstance(chunk, (bytes, bytearray)) for chunk in chunks)
    assert len(b"".join(chunks)) > 0


@pytest.mark.asyncio
async def test_kittentts_runtime_load_uses_revision_for_requested_model(monkeypatch):
    from tldw_Server_API.app.core.TTS.adapters import kitten_tts_adapter as mod

    fake_assets = SimpleNamespace(
        repo_id="KittenML/kitten-tts-mini-0.8",
        revision="ff7d47695d85548bcb0ac9378e063682e1cf0548",
    )
    download_calls = []

    def fake_download(model_name, *, cache_dir=None, auto_download=True, revision=None):
        download_calls.append((model_name, cache_dir, auto_download, revision))
        return fake_assets

    class FakeRuntime:
        def __init__(self, _assets):
            return

        def generate(self, text, *, voice=None, speed=1.0, clean_text=False):
            _ = (text, voice, speed, clean_text)
            return np.linspace(-0.5, 0.5, 512, dtype=np.float32)

    monkeypatch.setattr(mod, "download_model_assets", fake_download)
    monkeypatch.setattr(mod, "KittenRuntime", FakeRuntime)

    adapter = mod.KittenTTSAdapter(
        {
            "model": "KittenML/kitten-tts-nano-0.8",
            "model_revision": "deadbeef1",
        }
    )

    await adapter.generate(
        TTSRequest(
            text="switch models",
            model="KittenML/kitten-tts-mini-0.8",
            voice="Leo",
            format=AudioFormat.WAV,
            stream=False,
        )
    )

    assert download_calls == [
        ("KittenML/kitten-tts-mini-0.8", None, True, mod.resolve_model_revision("KittenML/kitten-tts-mini-0.8"))
    ]


@pytest.mark.asyncio
async def test_kittentts_override_preserves_configured_default_revision(monkeypatch: pytest.MonkeyPatch) -> None:
    from tldw_Server_API.app.core.TTS.adapters import kitten_tts_adapter as mod

    downloads: list[tuple[str, str | None]] = []

    def load(model_name: str, *, revision: str | None = None, **kwargs: Any) -> SimpleNamespace:
        downloads.append((model_name, revision))
        return SimpleNamespace(repo_id=model_name, revision=revision)

    monkeypatch.setattr(mod, "download_model_assets", load)
    monkeypatch.setattr(mod, "KittenRuntime", lambda assets: SimpleNamespace(assets=assets))
    adapter = mod.KittenTTSAdapter({"model": "KittenML/kitten-tts-nano-0.8-fp32", "model_revision": "deadbeef12345678"})
    await adapter._load_runtime_for_model("KittenML/kitten-tts-mini-0.8")
    await adapter._load_runtime_for_model("KittenML/kitten-tts-nano-0.8-fp32")

    assert downloads[-1] == ("KittenML/kitten-tts-nano-0.8-fp32", "deadbeef12345678")
