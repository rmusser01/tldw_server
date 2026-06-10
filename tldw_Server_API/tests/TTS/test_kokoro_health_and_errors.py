import types

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import MagicMock

from tldw_Server_API.app.core.TTS.tts_exceptions import TTSGenerationError


@pytest.mark.asyncio
async def test_audio_health_endpoint_smoke():
    # Import router lazily to avoid heavy imports at module load time
    from tldw_Server_API.app.api.v1.endpoints.audio.audio import router

    app = FastAPI()
    app.include_router(router, prefix="")
    with TestClient(app) as client:
        resp = client.get("/health")
    assert resp.status_code in (200, 500)
    if resp.status_code == 200:
        data = resp.json()
        assert "status" in data
        assert "providers" in data


@pytest.mark.asyncio
async def test_kokoro_pytorch_requires_pkg(monkeypatch):
    # Skip if torch is not installed in the environment
    try:
        import torch  # noqa
    except Exception:
        pytest.skip("torch not available")

    # Force use_onnx=False and point to a fake model path; mock os.path.exists to True
    from tldw_Server_API.app.core.TTS.adapters.kokoro_adapter import KokoroAdapter

    adapter = KokoroAdapter({"kokoro_use_onnx": False, "kokoro_model_path": "/fake/model.pth", "kokoro_device": "cpu"})

    monkeypatch.setattr("os.path.exists", lambda p: True)

    mock_resource_manager = MagicMock()
    mock_resource_manager.memory_monitor.is_memory_critical.return_value = False

    async def fake_get_resource_manager():
        return mock_resource_manager

    monkeypatch.setattr(
        "tldw_Server_API.app.core.TTS.adapters.kokoro_adapter.get_resource_manager", fake_get_resource_manager
    )

    class DummyTorchModule:
        def eval(self):
            return self

        def to(self, *args, **kwargs):
            return self

    if not hasattr(torch, "jit"):
        monkeypatch.setattr(torch, "jit", types.SimpleNamespace(), raising=False)
    monkeypatch.setattr(torch.jit, "load", lambda *args, **kwargs: DummyTorchModule(), raising=False)
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: DummyTorchModule(), raising=False)

    # Ensure 'kokoro' module import fails to trigger guidance error
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "kokoro" or name.startswith("kokoro."):
            raise ImportError("kokoro package not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    ok = await adapter.initialize()
    assert ok is True  # generic torch model path loads

    async def fake_stream(self, text, voice, lang, request):
        raise TTSGenerationError("kokoro package not installed", provider="Kokoro")
        if False:
            yield b""

    monkeypatch.setattr(
        "tldw_Server_API.app.core.TTS.adapters.kokoro_adapter.KokoroAdapter._stream_audio_kokoro",
        fake_stream,
        raising=False,
    )

    monkeypatch.setattr(
        "tldw_Server_API.app.core.TTS.adapters.kokoro_adapter.KokoroAdapter.preprocess_text",
        lambda self, text: text,
        raising=False,
    )

    from tldw_Server_API.app.core.TTS.adapters.base import TTSRequest, AudioFormat

    req = TTSRequest(text="hello", voice="af_bella", format=AudioFormat.WAV, stream=True)
    # Expect a generation error indicating kokoro package requirement
    with pytest.raises(TTSGenerationError):
        gen = await adapter.generate(req)
        # Exhaust the generator to trigger code paths
        async for _ in gen.audio_stream:
            pass
