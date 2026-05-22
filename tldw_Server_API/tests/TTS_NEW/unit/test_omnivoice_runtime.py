import io
import sys
import types
import wave
from pathlib import Path

import numpy as np
import pytest

from tldw_Server_API.app.core.TTS.adapters.omnivoice_runtime import (
    OmniVoiceRuntime,
    OmniVoiceRuntimeError,
)
from tldw_Server_API.app.core.TTS.adapters.omnivoice_sidecar_protocol import (
    OmniVoiceSynthesizeRequest,
)


class _FakeOmniVoice:
    model_path = None
    load_kwargs = None
    generate_calls = []
    generated_audio = np.zeros(2400, dtype=np.float32)
    load_error = None

    @classmethod
    def reset(cls, generated_audio=None, load_error=None):
        cls.model_path = None
        cls.load_kwargs = None
        cls.generate_calls = []
        cls.generated_audio = (
            np.zeros(2400, dtype=np.float32) if generated_audio is None else generated_audio
        )
        cls.load_error = load_error

    @classmethod
    def from_pretrained(cls, model_path, **kwargs):
        if cls.load_error is not None:
            raise cls.load_error
        cls.model_path = model_path
        cls.load_kwargs = kwargs
        return cls()

    def generate(self, **kwargs):
        self.generate_calls.append(kwargs)
        return self.generated_audio


@pytest.fixture
def fake_omnivoice(monkeypatch):
    _FakeOmniVoice.reset()
    module = types.ModuleType("omnivoice")
    module.OmniVoice = _FakeOmniVoice
    monkeypatch.setitem(sys.modules, "omnivoice", module)
    return _FakeOmniVoice


def _runtime_config(tmp_path: Path) -> dict[str, str]:
    model_dir = tmp_path / "model"
    scratch_dir = tmp_path / "scratch"
    model_dir.mkdir()
    scratch_dir.mkdir()
    return {"model_path": str(model_dir), "scratch_dir": str(scratch_dir)}


def _assert_wav_24000_mono(audio_bytes: bytes) -> None:
    with wave.open(io.BytesIO(audio_bytes), "rb") as wav_file:
        assert wav_file.getframerate() == 24000
        assert wav_file.getnchannels() == 1
        assert wav_file.getnframes() > 0


@pytest.mark.asyncio
async def test_runtime_rejects_missing_local_model_path(tmp_path):
    runtime = OmniVoiceRuntime({"model_path": str(tmp_path / "missing")})

    with pytest.raises(OmniVoiceRuntimeError) as exc:
        await runtime.load()

    assert exc.value.code == "MODEL_NOT_AVAILABLE"


@pytest.mark.asyncio
async def test_runtime_loads_from_local_directory_and_passes_safe_kwargs(tmp_path, fake_omnivoice):
    config = _runtime_config(tmp_path)
    config.update(
        {
            "device_map": "cpu",
            "dtype": "float16",
            "load_asr": False,
            "asr_model_name": "fake-asr",
            "ignored": "not-forwarded",
        }
    )
    runtime = OmniVoiceRuntime(config)

    await runtime.load()

    assert Path(fake_omnivoice.model_path).is_dir()
    assert Path(fake_omnivoice.model_path) == Path(config["model_path"])
    assert fake_omnivoice.load_kwargs == {
        "device_map": "cpu",
        "dtype": "float16",
        "load_asr": False,
        "asr_model_name": "fake-asr",
    }


@pytest.mark.asyncio
async def test_runtime_auto_mode_generates_language_and_parseable_wav(tmp_path, fake_omnivoice):
    runtime = OmniVoiceRuntime(_runtime_config(tmp_path))
    request = OmniVoiceSynthesizeRequest(text="hello", mode="auto", language_id="en")

    result = await runtime.synthesize(request)

    assert fake_omnivoice.generate_calls == [{"text": "hello", "language": "en"}]
    assert result.audio_format == "wav"
    assert result.sample_rate == 24000
    assert result.channels == 1
    assert result.cold_start is True
    _assert_wav_24000_mono(result.audio_bytes)


@pytest.mark.asyncio
async def test_runtime_design_mode_includes_instruct(tmp_path, fake_omnivoice):
    runtime = OmniVoiceRuntime(_runtime_config(tmp_path))
    request = OmniVoiceSynthesizeRequest(
        text="hello",
        mode="design",
        instruct="warm narrator",
        generation={"num_step": 8},
    )

    await runtime.synthesize(request)

    assert fake_omnivoice.generate_calls == [
        {"text": "hello", "instruct": "warm narrator", "num_step": 8}
    ]


@pytest.mark.asyncio
async def test_runtime_clone_mode_includes_reference_audio_and_text(tmp_path, fake_omnivoice):
    config = _runtime_config(tmp_path)
    reference_path = Path(config["scratch_dir"]) / "ref.wav"
    reference_path.write_bytes(b"fake wav")
    runtime = OmniVoiceRuntime(config)
    request = OmniVoiceSynthesizeRequest(
        text="hello",
        mode="clone",
        reference_audio_path=str(reference_path),
        reference_text="reference transcript",
    )

    await runtime.synthesize(request)

    assert fake_omnivoice.generate_calls == [
        {
            "text": "hello",
            "ref_audio": str(reference_path),
            "ref_text": "reference transcript",
        }
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("reference_kind", ["missing", "directory"])
async def test_runtime_rejects_invalid_reference_inside_managed_directory(
    tmp_path,
    fake_omnivoice,
    reference_kind,
):
    config = _runtime_config(tmp_path)
    reference_path = Path(config["scratch_dir"]) / "ref.wav"
    if reference_kind == "directory":
        reference_path.mkdir()
    runtime = OmniVoiceRuntime(config)
    await runtime.load()
    request = OmniVoiceSynthesizeRequest(
        text="hello",
        mode="clone",
        reference_audio_path=str(reference_path),
        reference_text="reference transcript",
    )

    with pytest.raises(OmniVoiceRuntimeError) as exc:
        await runtime.synthesize(request)

    assert exc.value.code == "INVALID_REFERENCE_AUDIO"
    assert fake_omnivoice.generate_calls == []


@pytest.mark.asyncio
async def test_runtime_empty_audio_output_raises(tmp_path, fake_omnivoice):
    fake_omnivoice.reset(generated_audio=np.array([], dtype=np.float32))
    runtime = OmniVoiceRuntime(_runtime_config(tmp_path))
    request = OmniVoiceSynthesizeRequest(text="hello", mode="auto")

    with pytest.raises(OmniVoiceRuntimeError) as exc:
        await runtime.synthesize(request)

    assert exc.value.code == "EMPTY_AUDIO_OUTPUT"


@pytest.mark.asyncio
async def test_runtime_success_after_failure_restores_ready_status(tmp_path, fake_omnivoice):
    fake_omnivoice.reset(generated_audio=np.array([], dtype=np.float32))
    runtime = OmniVoiceRuntime(_runtime_config(tmp_path))

    with pytest.raises(OmniVoiceRuntimeError) as exc:
        await runtime.synthesize(OmniVoiceSynthesizeRequest(text="hello", mode="auto"))

    assert exc.value.code == "EMPTY_AUDIO_OUTPUT"
    assert runtime.status == "error"
    assert runtime.last_error_code == "EMPTY_AUDIO_OUTPUT"

    fake_omnivoice.generated_audio = np.zeros(2400, dtype=np.float32)
    result = await runtime.synthesize(OmniVoiceSynthesizeRequest(text="hello again", mode="auto"))

    assert result.audio_bytes
    assert runtime.status == "ready"
    assert runtime.last_error_code is None


@pytest.mark.asyncio
async def test_runtime_rejects_clone_reference_outside_managed_directories(tmp_path, fake_omnivoice):
    config = _runtime_config(tmp_path)
    outside_reference = tmp_path / "outside.wav"
    outside_reference.write_bytes(b"fake wav")
    runtime = OmniVoiceRuntime(config)
    request = OmniVoiceSynthesizeRequest(
        text="hello",
        mode="clone",
        reference_audio_path=str(outside_reference),
        reference_text="reference transcript",
    )

    with pytest.raises(OmniVoiceRuntimeError) as exc:
        await runtime.synthesize(request)

    assert exc.value.code == "REFERENCE_PATH_NOT_ALLOWED"
    assert fake_omnivoice.generate_calls == []


@pytest.mark.asyncio
async def test_model_load_failure_message_does_not_include_local_model_path(tmp_path, fake_omnivoice):
    config = _runtime_config(tmp_path)
    fake_omnivoice.reset(load_error=RuntimeError("boom"))
    runtime = OmniVoiceRuntime(config)

    with pytest.raises(OmniVoiceRuntimeError) as exc:
        await runtime.load()

    assert exc.value.code == "MODEL_LOAD_FAILED"
    assert config["model_path"] not in str(exc.value)
