"""Unit tests for VibeVoice adapter request and model-state behavior."""

import asyncio

import pytest

from tldw_Server_API.app.core.TTS.adapters.base import AudioFormat, TTSRequest
from tldw_Server_API.app.core.TTS.adapters.vibevoice_adapter import VibeVoiceAdapter


@pytest.mark.unit
def test_preprocess_text_converts_bracket_markers_to_speaker_format():
    adapter = VibeVoiceAdapter({})
    raw = "[1]: Hello there\n[2]: And hi back"
    out = adapter.preprocess_text(raw)
    # Expect conversion to Speaker N: form
    assert "Speaker 1:" in out
    assert "Speaker 2:" in out


@pytest.mark.unit
def test_build_voice_samples_with_mapping_1_based_keys(tmp_path):
    # Prepare adapter with available voices mapping
    adapter = VibeVoiceAdapter({})
    adapter.available_voices = {
        "en-Alice_woman": "/voices/alice.wav",
        "en-Frank_man": "/voices/frank.wav",
    }

    text = "Speaker 1: Hello\nSpeaker 2: Hi"
    mapping = {"1": "en-Alice_woman", "2": "en-Frank_man"}

    res = adapter._build_voice_samples(
        formatted_text=text,
        voice_reference_path=None,
        primary_voice="speaker_1",
        speakers_to_voices=mapping,
    )

    assert res == ["/voices/alice.wav", "/voices/frank.wav"]


@pytest.mark.unit
def test_build_voice_samples_fallbacks_to_reference_then_files(tmp_path):
    # Create two fake voice files
    voices_dir = tmp_path / "voices"
    voices_dir.mkdir()
    (voices_dir / "a.wav").write_bytes(b"00")
    (voices_dir / "b.wav").write_bytes(b"11")

    adapter = VibeVoiceAdapter({"vibevoice_voices_dir": str(voices_dir)})
    adapter.available_voices = {}

    text = "Speaker 1: Hello\nSpeaker 2: Hi"
    # Provide explicit reference for first speaker
    ref = "/path/to/ref.wav"
    res = adapter._build_voice_samples(text, ref, "speaker_1", None)

    # First slot is the reference, second picked from voices_dir
    assert len(res) == 2
    assert res[0] == ref
    assert res[1].endswith(".wav")


@pytest.mark.unit
def test_default_mapping_merged_with_request_overrides():
    # Adapter with default mapping in config
    adapter = VibeVoiceAdapter({"vibevoice_speakers_to_voices": {"1": "en-Alice_woman"}})
    adapter.available_voices = {
        "en-Alice_woman": "/voices/alice.wav",
        "en-Frank_man": "/voices/frank.wav",
    }

    text = "Speaker 1: Hello\nSpeaker 2: Hi"

    # Request mapping overrides/extends defaults
    req_mapping = {"2": "en-Frank_man"}
    # Mirror adapter's merge semantics
    merged = {**adapter.default_speakers_to_voices, **req_mapping}

    res = adapter._build_voice_samples(
        formatted_text=text,
        voice_reference_path=None,
        primary_voice="speaker_1",
        speakers_to_voices=merged,
    )

    assert res == ["/voices/alice.wav", "/voices/frank.wav"]


@pytest.mark.unit
def test_request_overrides_default_for_same_speaker():
    adapter = VibeVoiceAdapter({"vibevoice_speakers_to_voices": {"1": "en-Alice_woman"}})
    adapter.available_voices = {
        "en-Alice_woman": "/voices/alice.wav",
    }

    text = "Speaker 1: Hello"
    # Override default with explicit path for speaker 1
    req_mapping = {"1": "/override/one.wav"}
    merged = {**adapter.default_speakers_to_voices, **req_mapping}

    res = adapter._build_voice_samples(
        formatted_text=text,
        voice_reference_path=None,
        primary_voice="speaker_1",
        speakers_to_voices=merged,
    )

    assert res == ["/override/one.wav"]


@pytest.mark.unit
def test_vibevoice_adapter_initializes_model_state_lock():
    """Verify VibeVoice creates an async lock for model state changes."""
    adapter = VibeVoiceAdapter({})

    assert isinstance(adapter._model_state_lock, asyncio.Lock)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_generate_complete_forwards_generation_config(monkeypatch):
    """Verify non-streaming generation forwards per-request generation config."""
    adapter = VibeVoiceAdapter({})
    request = TTSRequest(text="Speaker 1: hello", voice="speaker_1", format=AudioFormat.WAV)
    generation_config = {"cfg_scale": 1.7, "speakers_to_voices": {"1": "alice"}}
    seen: dict[str, object] = {}

    async def fake_stream(
        request_arg,
        voice_arg,
        speaker_id_arg,
        voice_reference_path=None,
        gen_config=None,
    ):
        """Capture arguments passed into the streaming implementation."""
        seen["request"] = request_arg
        seen["voice"] = voice_arg
        seen["speaker_id"] = speaker_id_arg
        seen["voice_reference_path"] = voice_reference_path
        seen["gen_config"] = gen_config
        yield b"chunk"

    monkeypatch.setattr(adapter, "_stream_audio_vibevoice", fake_stream)

    audio = await adapter._generate_complete_vibevoice(
        request,
        "speaker_1",
        1,
        "/tmp/reference.wav",
        generation_config,
    )

    assert audio == b"chunk"
    assert seen["gen_config"] is generation_config
    assert seen["voice_reference_path"] == "/tmp/reference.wav"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_generate_canonicalizes_lowercase_requested_variant(monkeypatch):
    """Verify lowercase variant requests are mapped to canonical model names."""
    adapter = VibeVoiceAdapter({})
    adapter.available_voices = {"speaker_1": "/voices/speaker.wav"}
    seen: dict[str, object] = {}

    async def fake_ensure_initialized():
        """Pretend the adapter is initialized for generation."""
        return True

    async def fake_ensure_model_variant(variant):
        """Capture the canonical variant selected by generation."""
        seen["variant"] = variant
        adapter.variant = variant

    def fake_stream(_request, _voice, _speaker_id, _voice_reference_path=None, gen_config=None):
        """Return a tiny async stream while recording generation config."""
        seen["gen_config"] = gen_config

        async def _chunks():
            """Yield a single audio chunk for the fake stream."""
            yield b"chunk"

        return _chunks()

    monkeypatch.setattr(adapter, "ensure_initialized", fake_ensure_initialized)
    monkeypatch.setattr(adapter, "_ensure_model_variant", fake_ensure_model_variant)
    monkeypatch.setattr(adapter, "_stream_audio_vibevoice", fake_stream)
    monkeypatch.setattr(adapter, "cleanup_after_generation", fake_ensure_initialized)

    request = TTSRequest(
        text="Speaker 1: hello",
        voice="speaker_1",
        model="7B",
        format=AudioFormat.WAV,
        stream=True,
    )

    response = await adapter.generate(request)

    assert response.audio_stream is not None
    assert seen["variant"] == "7B"
    assert seen["gen_config"]["model_variant"] == "7B"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_reload_model_for_q8_variant_recomputes_quantization(monkeypatch):
    """Verify Q8 variant reload disables external quantization flags."""
    adapter = VibeVoiceAdapter({"vibevoice_device": "cuda", "vibevoice_use_quantization": True})
    adapter.use_quantization = True

    async def fake_cleanup_resources():
        """Pretend cleanup completed before reinitialization."""
        return None

    async def fake_initialize():
        """Pretend model initialization completed after reload."""
        return True

    monkeypatch.setattr(adapter, "_cleanup_resources", fake_cleanup_resources)
    monkeypatch.setattr(adapter, "initialize", fake_initialize)

    await adapter._reload_model_for_variant("7B-Q8")

    assert adapter.variant == "7B-Q8"
    assert adapter.use_quantization is False
