from __future__ import annotations

import os
import wave
from io import BytesIO
from pathlib import Path

import pytest

from tldw_Server_API.app.core.TTS.adapters.omnivoice_runtime import OmniVoiceRuntime
from tldw_Server_API.app.core.TTS.adapters.omnivoice_sidecar_protocol import (
    OmniVoiceSynthesizeRequest,
)

pytestmark = [pytest.mark.asyncio, pytest.mark.integration]


def _require_real_omnivoice() -> Path:
    if os.environ.get("TLDW_TEST_OMNIVOICE_REAL") != "1":
        pytest.skip("Set TLDW_TEST_OMNIVOICE_REAL=1 to run real OmniVoice tests")

    model_path = os.environ.get("TLDW_OMNIVOICE_MODEL_PATH")
    if not model_path or not Path(model_path).is_dir():
        pytest.skip("TLDW_OMNIVOICE_MODEL_PATH must point to a local model directory")

    return Path(model_path)


def _assert_non_empty_wav_24000(audio_bytes: bytes) -> None:
    assert audio_bytes
    with wave.open(BytesIO(audio_bytes), "rb") as wav_file:
        assert wav_file.getframerate() == 24000
        assert wav_file.getnframes() > 0


def _write_tiny_reference_wav(reference_path: Path) -> None:
    reference_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(reference_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(24000)
        wav_file.writeframes(b"\x00\x00" * 240)


async def test_real_omnivoice_runtime_auto_voice_smoke(tmp_path):
    model_path = _require_real_omnivoice()
    runtime = OmniVoiceRuntime({"model_path": str(model_path), "scratch_dir": str(tmp_path)})

    result = await runtime.synthesize(
        OmniVoiceSynthesizeRequest(
            text="Hello from the OmniVoice real runtime smoke test.",
            mode="auto",
            voice="auto",
        )
    )

    _assert_non_empty_wav_24000(result.audio_bytes)


async def test_real_omnivoice_runtime_design_smoke(tmp_path):
    model_path = _require_real_omnivoice()
    runtime = OmniVoiceRuntime({"model_path": str(model_path), "scratch_dir": str(tmp_path)})

    result = await runtime.synthesize(
        OmniVoiceSynthesizeRequest(
            text="This sentence should sound calm and clear.",
            mode="design",
            instruct="calm clear narrator",
        )
    )

    _assert_non_empty_wav_24000(result.audio_bytes)


async def test_real_omnivoice_runtime_clone_smoke(tmp_path):
    model_path = _require_real_omnivoice()
    scratch_dir = tmp_path / "scratch"
    reference_path = scratch_dir / "reference.wav"
    _write_tiny_reference_wav(reference_path)
    runtime = OmniVoiceRuntime({"model_path": str(model_path), "scratch_dir": str(scratch_dir)})

    result = await runtime.synthesize(
        OmniVoiceSynthesizeRequest(
            text="Clone smoke test output.",
            mode="clone",
            reference_audio_path=str(reference_path),
            reference_text="This is a short reference recording.",
        )
    )

    _assert_non_empty_wav_24000(result.audio_bytes)
