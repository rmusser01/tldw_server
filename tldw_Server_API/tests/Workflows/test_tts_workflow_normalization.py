import asyncio
from pathlib import Path

import pytest


@pytest.mark.asyncio
async def test_run_tts_adapter_post_process_normalization(monkeypatch, tmp_path):
    """
    Ensure the TTS workflow adapter honors post_process.normalize
    and sets the normalized flag when ffmpeg-style normalization succeeds.
    """
    # Run in an isolated working directory so artifacts are local to the test
    monkeypatch.chdir(tmp_path)

    # Stub TTS service to avoid real model initialization
    from tldw_Server_API.app.core.TTS import tts_service_v2 as tts_mod

    class _FakeTTSService:
        async def generate_speech(self, request, provider=None, fallback=True, voice_to_voice_start=None, voice_to_voice_route="audio.speech"):
            # Yield a small, deterministic "audio" payload
            yield b"fake-audio-bytes"

    async def _fake_get_tts_service_v2(config=None):
        return _FakeTTSService()

    monkeypatch.setattr(tts_mod, "get_tts_service_v2", _fake_get_tts_service_v2, raising=True)
    import tldw_Server_API.app.core.Workflows.adapters.audio.tts as tts_adapter_mod

    monkeypatch.setattr(tts_adapter_mod, "get_tts_service_v2", _fake_get_tts_service_v2, raising=True)

    # Stub ffmpeg detection and subprocess execution inside the workflows module
    import tldw_Server_API.app.core.Workflows.adapters as wf_adapters

    # Pretend ffmpeg is available by patching the stdlib shutil.which used in the adapter
    import shutil

    monkeypatch.setattr(
        shutil,
        "which",
        lambda name: "/usr/bin/ffmpeg" if name == "ffmpeg" else None,
        raising=False,
    )

    async def _fake_create_subprocess_exec(*cmd, **kwargs):
        # Simulate ffmpeg writing a normalized output file (last arg in cmd)
        if cmd:
            out_path = Path(cmd[-1])
            try:
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_bytes(b"normalized-audio")
            except Exception:
                _ = None

        class _FakeProc:
            def __init__(self):
                self.returncode = 0

            async def communicate(self):
                # Simulate successful execution with no output
                return b"", b""

        return _FakeProc()

    monkeypatch.setattr(
        wf_adapters.asyncio,
        "create_subprocess_exec",
        _fake_create_subprocess_exec,
        raising=True,
    )

    # Minimal config that enables normalization
    config = {
        "input": "Hello from workflow TTS",
        "model": "kokoro",
        "voice": "af_heart",
        "response_format": "mp3",
        "post_process": {
            "normalize": True,
            "target_lufs": -16.0,
            "true_peak_dbfs": -1.5,
            "lra": 11.0,
        },
    }
    context = {}

    result = await wf_adapters.run_tts_adapter(config, context)

    # Should not return a top-level error
    assert "error" not in result

    # Normalization should be marked as successful
    assert result.get("normalized") is True

    # Audio URI should point at the normalized file
    audio_uri = result.get("audio_uri")
    assert isinstance(audio_uri, str)
    assert audio_uri.startswith("file://")
    normalized_path = Path(audio_uri[len("file://") :])
    assert normalized_path.name.startswith("normalized.")
    assert normalized_path.exists()
    # Ensure some bytes were written
    assert normalized_path.stat().st_size > 0
