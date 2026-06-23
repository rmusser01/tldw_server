import base64
import sys
import types

import numpy as np
import pytest

from tldw_Server_API.app.core.TTS.adapters.base import AudioFormat, TTSRequest
from tldw_Server_API.app.core.TTS.adapters.qwen3_tts_adapter import Qwen3TTSAdapter
from tldw_Server_API.app.core.TTS.tts_exceptions import TTSValidationError, TTSInvalidVoiceReferenceError


class _FakeBackend:
    def __init__(self) -> None:
        self.calls = []

    def generate_custom_voice(self, text, language=None, speaker=None, instruct=None, **_kwargs):
        self.calls.append(
            {
                "mode": "custom",
                "text": text,
                "language": language,
                "speaker": speaker,
                "instruct": instruct,
            }
        )
        return np.zeros(160, dtype=np.float32)

    def generate_voice_design(self, text, language=None, instruct=None, **_kwargs):
        self.calls.append(
            {
                "mode": "design",
                "text": text,
                "language": language,
                "instruct": instruct,
            }
        )
        return np.zeros(160, dtype=np.float32)

    def generate_voice_clone(
        self,
        text,
        language=None,
        ref_audio=None,
        ref_text=None,
        x_vector_only_mode=False,
        voice_clone_prompt=None,
        **_kwargs,
    ):
        self.calls.append(
            {
                "mode": "clone",
                "text": text,
                "language": language,
                "ref_audio": ref_audio,
                "ref_text": ref_text,
                "x_vector_only_mode": x_vector_only_mode,
                "voice_clone_prompt": voice_clone_prompt,
            }
        )
        return np.zeros(160, dtype=np.float32)


@pytest.fixture
def fake_qwen_module(monkeypatch):
    holder = {"backend": None, "last_model_id": None}
    module = types.ModuleType("qwen_tts")

    class FakeQwen3TTS:
        @classmethod
        def from_pretrained(cls, model_id, **_kwargs):
            holder["last_model_id"] = model_id
            backend = _FakeBackend()
            holder["backend"] = backend
            return backend

    module.Qwen3TTS = FakeQwen3TTS
    monkeypatch.setitem(sys.modules, "qwen_tts", module)
    return holder


@pytest.fixture
def fake_qwen_model_module(monkeypatch):
    holder = {"backend": None, "last_model_id": None}
    module = types.ModuleType("qwen_tts")

    class FakeQwen3TTSModel:
        @classmethod
        def from_pretrained(cls, model_id, **_kwargs):
            holder["last_model_id"] = model_id
            backend = _FakeBackend()
            holder["backend"] = backend
            return backend

    module.Qwen3TTSModel = FakeQwen3TTSModel
    monkeypatch.setitem(sys.modules, "qwen_tts", module)
    return holder


@pytest.mark.asyncio
async def test_auto_model_selects_customvoice_cpu(fake_qwen_module):
    adapter = Qwen3TTSAdapter({"runtime": "upstream", "device": "cpu", "model": "auto"})
    await adapter.ensure_initialized()

    request = TTSRequest(
        text="Hello",
        voice="Vivian",
        language="en",
        format=AudioFormat.PCM,
        stream=False,
    )
    request.model = "auto"

    response = await adapter.generate(request)
    assert response.audio_content
    assert fake_qwen_module["last_model_id"] == adapter.MODEL_CUSTOMVOICE_06B


def test_resolve_torch_dtype_falls_back_for_partial_torch_module(monkeypatch):
    module = types.ModuleType("torch")
    monkeypatch.setitem(sys.modules, "torch", module)
    adapter = Qwen3TTSAdapter({"device": "cpu", "model": "auto", "dtype": "float16"})

    assert adapter._resolve_torch_dtype() == "float16"


@pytest.mark.parametrize("model_alias", ["qwen3_tts", "qwen3-tts"])
def test_provider_style_model_aliases_resolve_to_canonical_model(model_alias):
    adapter = Qwen3TTSAdapter({"device": "cpu", "model": "auto"})

    request = TTSRequest(
        text="Hello",
        voice="Vivian",
        language="en",
        format=AudioFormat.PCM,
        stream=False,
    )
    request.model = model_alias

    assert adapter._resolve_model(request) == adapter.MODEL_CUSTOMVOICE_06B


@pytest.mark.asyncio
async def test_voice_design_requires_instruct(fake_qwen_module):
    adapter = Qwen3TTSAdapter({"runtime": "upstream", "device": "cpu", "model": "auto"})
    await adapter.ensure_initialized()

    request = TTSRequest(
        text="Hello",
        voice=None,
        language="en",
        format=AudioFormat.PCM,
        stream=False,
        extra_params={},
    )
    request.model = Qwen3TTSAdapter.MODEL_VOICEDESIGN_17B.lower()

    with pytest.raises(TTSValidationError):
        await adapter.generate(request)


@pytest.mark.asyncio
async def test_voice_clone_maps_reference_and_prompt(fake_qwen_module):
    adapter = Qwen3TTSAdapter({"runtime": "upstream", "device": "cpu", "model": "auto"})
    await adapter.ensure_initialized()

    voice_bytes = b"VOICE_BYTES"
    prompt_bytes = b"PROMPT_BYTES"
    prompt_b64 = base64.b64encode(prompt_bytes).decode("utf-8")

    request = TTSRequest(
        text="Hello",
        voice=None,
        language=None,
        format=AudioFormat.PCM,
        stream=False,
        voice_reference=voice_bytes,
        extra_params={
            "reference_text": "ref transcript",
            "voice_clone_prompt": prompt_b64,
            "language": "ja",
        },
    )
    request.model = Qwen3TTSAdapter.MODEL_BASE_06B.lower()

    response = await adapter.generate(request)
    assert response.audio_content

    backend = fake_qwen_module["backend"]
    assert backend is not None
    assert backend.calls
    call = backend.calls[-1]
    assert call["mode"] == "clone"
    assert call["language"] == "ja"
    assert call["ref_audio"] == base64.b64encode(voice_bytes).decode("ascii")
    assert call["ref_text"] == "ref transcript"
    assert call["voice_clone_prompt"] == prompt_bytes


@pytest.mark.asyncio
async def test_streaming_transcode_fallback_buffers_output(fake_qwen_module, monkeypatch):
    adapter = Qwen3TTSAdapter({"runtime": "upstream", "device": "cpu", "model": "auto"})
    await adapter.ensure_initialized()

    async def fake_generate_pcm(_request, _model_id):
        return np.zeros(480, dtype=np.int16)

    async def fake_convert(_audio, source_format, target_format, sample_rate):
        return b"converted-bytes"

    monkeypatch.setattr(adapter, "_can_stream_transcode", lambda _fmt: (False, "no-writer"))
    monkeypatch.setattr(adapter, "_generate_pcm", fake_generate_pcm)
    monkeypatch.setattr(adapter, "convert_audio_format", fake_convert)

    request = TTSRequest(
        text="Hello",
        voice="Vivian",
        language="en",
        format=AudioFormat.MP3,
        stream=True,
    )
    request.model = "auto"

    response = await adapter.generate(request)
    assert response.audio_stream is not None

    chunks = []
    async for chunk in response.audio_stream:
        chunks.append(chunk)
    assert b"".join(chunks) == b"converted-bytes"
    assert response.metadata.get("streaming_fallback") == "buffered"


@pytest.mark.asyncio
async def test_voice_clone_requires_reference_even_with_x_vector_only(fake_qwen_module):
    adapter = Qwen3TTSAdapter({"runtime": "upstream", "device": "cpu", "model": "auto"})
    adapter._resolve_torch_dtype = lambda: "float32"
    await adapter.ensure_initialized()

    request = TTSRequest(
        text="Hello",
        voice=None,
        language="en",
        format=AudioFormat.PCM,
        stream=False,
        extra_params={"x_vector_only_mode": True},
    )
    request.model = Qwen3TTSAdapter.MODEL_BASE_06B.lower()

    with pytest.raises(TTSInvalidVoiceReferenceError):
        await adapter.generate(request)


@pytest.mark.asyncio
async def test_voice_clone_requires_reference_text_unless_x_vector_only(fake_qwen_module):
    adapter = Qwen3TTSAdapter({"runtime": "upstream", "device": "cpu", "model": "auto"})
    adapter._resolve_torch_dtype = lambda: "float32"
    await adapter.ensure_initialized()

    request = TTSRequest(
        text="Hello",
        voice=None,
        language="en",
        format=AudioFormat.PCM,
        stream=False,
        voice_reference=b"RIFF" + b"\x00" * 1000,
        extra_params={},
    )
    request.model = Qwen3TTSAdapter.MODEL_BASE_06B.lower()

    with pytest.raises(TTSValidationError):
        await adapter.generate(request)


@pytest.mark.asyncio
async def test_builder_discovers_qwen3_tts_model(fake_qwen_model_module):
    adapter = Qwen3TTSAdapter({"runtime": "upstream", "device": "cpu", "model": "auto"})
    adapter._resolve_torch_dtype = lambda: "float32"
    await adapter.ensure_initialized()

    request = TTSRequest(
        text="Hello",
        voice="Vivian",
        language="en",
        format=AudioFormat.PCM,
        stream=False,
    )
    request.model = "auto"

    response = await adapter.generate(request)
    assert response.audio_content
    assert fake_qwen_model_module["last_model_id"] == adapter.MODEL_CUSTOMVOICE_06B
