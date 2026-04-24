from __future__ import annotations

import contextlib
import os
import wave
from io import BytesIO
from pathlib import Path

import pytest

from tldw_Server_API.app.core.TTS.adapters.base import AudioFormat, TTSRequest
from tldw_Server_API.app.core.TTS.adapters.omnivoice_adapter import OmniVoiceAdapter
from tldw_Server_API.app.core.TTS.adapters.omnivoice_sidecar_supervisor import OmniVoiceSidecarSupervisor
from tldw_Server_API.app.core.config import load_tts_config

pytestmark = [pytest.mark.asyncio, pytest.mark.integration, pytest.mark.local_llm_service]


def _require_omnivoice_runtime() -> tuple[dict[str, object], Path]:
    if os.getenv("TLDW_RUN_OMNIVOICE_INTEGRATION") != "1":
        pytest.skip("TLDW_RUN_OMNIVOICE_INTEGRATION not set")

    config = load_tts_config() or {}
    provider_cfg = config.get("omnivoice_config")
    if not isinstance(provider_cfg, dict):
        pytest.skip("OmniVoice provider config is unavailable")

    extra_params = provider_cfg.get("extra_params")
    if not isinstance(extra_params, dict):
        pytest.skip("OmniVoice provider extra_params are unavailable")

    python_path = extra_params.get("python_path")
    if not python_path:
        pytest.skip("OmniVoice python_path is not configured")

    interpreter_path = Path(str(python_path)).expanduser()
    if not interpreter_path.is_absolute():
        interpreter_path = (Path.cwd() / interpreter_path).resolve()
    if not interpreter_path.exists():
        pytest.skip("OmniVoice sidecar runtime is not installed")

    model_id = str(extra_params.get("model_id") or "k2-fsa/OmniVoice")
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{model_id.replace('/', '--')}"
    if not cache_dir.exists():
        pytest.skip("OmniVoice weights are not cached locally")

    return provider_cfg, Path.cwd().resolve()


@pytest.mark.requires_model
async def test_omnivoice_sidecar_real_smoke():
    provider_cfg, repo_root = _require_omnivoice_runtime()
    supervisor = OmniVoiceSidecarSupervisor(provider_cfg, repo_root=repo_root)
    adapter = OmniVoiceAdapter({**provider_cfg, "_supervisor": supervisor})

    try:
        await adapter.ensure_initialized()

        request = TTSRequest(
            text="Hello from the OmniVoice integration smoke test.",
            voice="auto",
            model="omnivoice",
            format=AudioFormat.WAV,
            stream=False,
        )

        response = await adapter.generate(request)

        assert response.provider == "omnivoice"
        assert response.sample_rate == 24000
        assert response.audio_data

        with wave.open(BytesIO(response.audio_data), "rb") as wav_file:
            assert wav_file.getframerate() == 24000
            assert wav_file.getnchannels() == 1
            assert wav_file.getnframes() > 0
    finally:
        supervisor.mark_closing()
        with contextlib.suppress(Exception):
            await supervisor._stop_process()  # noqa: SLF001
