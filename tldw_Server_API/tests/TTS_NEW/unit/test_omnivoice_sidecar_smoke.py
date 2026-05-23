from __future__ import annotations

import struct
import wave
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from tldw_Server_API.app.core.TTS.adapters.base import AudioFormat, TTSResponse

if TYPE_CHECKING:
    from Helper_Scripts.TTS_Installers.smoke_test_omnivoice_sidecar import (
        OmniVoiceSmokeConfig,
    )


def _valid_smoke_config(tmp_path: Path) -> OmniVoiceSmokeConfig:
    from Helper_Scripts.TTS_Installers import smoke_test_omnivoice_sidecar as smoke

    model_path = tmp_path / "models" / "OmniVoice"
    model_path.mkdir(parents=True)
    sidecar_python = tmp_path / "models" / "omnivoice_sidecar" / ".venv" / "bin" / "python"
    sidecar_python.parent.mkdir(parents=True)
    sidecar_python.write_text("#!/usr/bin/env python\n", encoding="utf-8")
    sidecar_python.chmod(0o755)

    return smoke.OmniVoiceSmokeConfig(
        repo_root=tmp_path,
        model_path=model_path,
        sidecar_python=sidecar_python,
        runtime_path=tmp_path / "models" / "omnivoice_sidecar" / "runtime",
        scratch_dir=tmp_path / "models" / "omnivoice_sidecar" / "runtime" / "scratch",
        output_path=tmp_path / "smoke.wav",
        text="Hello from the managed sidecar.",
        port=8844,
        num_step=8,
        speed=1.0,
        timeout=123.0,
    )


def _wav_bytes(samples: list[int], *, sample_rate: int = 24000, channels: int = 1) -> bytes:
    buffer = BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        frames = b"".join(struct.pack("<h", sample) for sample in samples)
        if channels > 1:
            mono_frames = [
                frames[index : index + 2]
                for index in range(0, len(frames), 2)
            ]
            frames = b"".join(frame * channels for frame in mono_frames)
        wav_file.writeframes(frames)
    return buffer.getvalue()


@pytest.mark.unit
def test_smoke_helper_parse_args_accepts_operator_inputs(tmp_path):
    from Helper_Scripts.TTS_Installers.smoke_test_omnivoice_sidecar import parse_args

    output_path = tmp_path / "omnivoice.wav"

    args = parse_args(
        [
            "--model-path",
            "/models/OmniVoice",
            "--sidecar-python",
            "/runtime/.venv/bin/python",
            "--output",
            str(output_path),
            "--port",
            "8844",
            "--num-step",
            "8",
            "--speed",
            "1.25",
        ]
    )

    assert args.model_path == "/models/OmniVoice"
    assert args.sidecar_python == "/runtime/.venv/bin/python"
    assert args.output == str(output_path)
    assert args.port == 8844
    assert args.num_step == 8
    assert args.speed == pytest.approx(1.25)


@pytest.mark.unit
def test_smoke_helper_builds_provider_config_with_managed_sidecar_paths(tmp_path):
    from Helper_Scripts.TTS_Installers.smoke_test_omnivoice_sidecar import (
        build_sidecar_provider_config,
    )

    model_path = tmp_path / "models" / "OmniVoice"
    sidecar_python = tmp_path / "models" / "omnivoice_sidecar" / ".venv" / "bin" / "python"
    runtime_path = tmp_path / "models" / "omnivoice_sidecar" / "runtime"
    scratch_dir = runtime_path / "scratch"

    config = build_sidecar_provider_config(
        model_path=model_path,
        sidecar_python=sidecar_python,
        runtime_path=runtime_path,
        scratch_dir=scratch_dir,
        port=8844,
        timeout=123.0,
    )

    assert config["enabled"] is True
    assert config["runtime"] == "sidecar"
    assert config["model"] == "omnivoice"
    assert config["sample_rate"] == 24000
    assert config["timeout"] == pytest.approx(123.0)
    assert config["max_concurrent_generations"] == 1

    extra_params = config["extra_params"]
    assert extra_params["model_path"] == str(model_path)
    assert extra_params["python_path"] == str(sidecar_python)
    assert extra_params["runtime_path"] == str(runtime_path)
    assert extra_params["scratch_dir"] == str(scratch_dir)
    assert extra_params["host"] == "127.0.0.1"
    assert extra_params["port"] == 8844
    assert extra_params["autoselect_port"] is True
    assert extra_params["port_probe_max"] >= 1


@pytest.mark.unit
def test_smoke_helper_preserves_sidecar_python_symlink_when_resolving_config(tmp_path):
    from Helper_Scripts.TTS_Installers.smoke_test_omnivoice_sidecar import (
        build_smoke_config,
        parse_args,
    )

    repo_root = tmp_path / "repo"
    (repo_root / "tldw_Server_API").mkdir(parents=True)
    (repo_root / "pyproject.toml").write_text("[project]\nname = \"test\"\n", encoding="utf-8")
    base_python = tmp_path / "base-python"
    base_python.write_text("# fake base python\n", encoding="utf-8")
    sidecar_python = repo_root / "models" / "omnivoice_sidecar" / ".venv" / "bin" / "python"
    sidecar_python.parent.mkdir(parents=True)
    sidecar_python.symlink_to(base_python)
    model_path = repo_root / "models" / "OmniVoice"
    model_path.mkdir(parents=True)

    args = parse_args(
        [
            "--repo-root",
            str(repo_root),
            "--model-path",
            "models/OmniVoice",
            "--sidecar-python",
            "models/omnivoice_sidecar/.venv/bin/python",
        ]
    )

    config = build_smoke_config(args)

    assert config.sidecar_python == sidecar_python
    assert config.sidecar_python != base_python.resolve()


@pytest.mark.unit
@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("speed", 0.0, "OmniVoice sidecar smoke speed"),
        ("num_step", 0, "OmniVoice sidecar smoke num_step"),
    ],
)
def test_smoke_helper_rejects_invalid_generation_controls(tmp_path, field, value, message):
    from dataclasses import replace

    from Helper_Scripts.TTS_Installers.smoke_test_omnivoice_sidecar import validate_smoke_config

    config = replace(_valid_smoke_config(tmp_path), **{field: value})

    with pytest.raises(ValueError, match=message):
        validate_smoke_config(config)


@pytest.mark.unit
def test_smoke_helper_rejects_non_executable_sidecar_python(tmp_path):
    from dataclasses import replace

    from Helper_Scripts.TTS_Installers.smoke_test_omnivoice_sidecar import validate_smoke_config

    config = _valid_smoke_config(tmp_path)
    config.sidecar_python.chmod(0o644)

    with pytest.raises(ValueError, match="executable"):
        validate_smoke_config(replace(config, sidecar_python=config.sidecar_python))


@pytest.mark.unit
def test_smoke_helper_builds_non_streaming_wav_request():
    from Helper_Scripts.TTS_Installers.smoke_test_omnivoice_sidecar import build_tts_request

    request = build_tts_request(
        text="Hello from the smoke test.",
        num_step=8,
        speed=1.25,
    )

    assert request.provider == "omnivoice"
    assert request.model == "omnivoice"
    assert request.voice == "auto"
    assert request.format is AudioFormat.WAV
    assert request.stream is False
    assert request.speed == pytest.approx(1.25)
    assert request.extra_params["language_id"] == "en"
    assert request.extra_params["num_step"] == 8


@pytest.mark.unit
def test_smoke_helper_validates_non_silent_24000_mono_wav():
    from Helper_Scripts.TTS_Installers.smoke_test_omnivoice_sidecar import validate_wav_audio

    summary = validate_wav_audio(_wav_bytes([0, 500, -500, 1000, -1000]))

    assert summary.sample_rate == 24000
    assert summary.channels == 1
    assert summary.frame_count == 5
    assert summary.rms > 0
    assert summary.peak == 1000


@pytest.mark.unit
def test_smoke_helper_limits_wav_sample_analysis_window():
    from Helper_Scripts.TTS_Installers.smoke_test_omnivoice_sidecar import validate_wav_audio

    summary = validate_wav_audio(
        _wav_bytes([1000, 0, 3000, 0], sample_rate=4),
        expected_sample_rate=4,
        max_analysis_seconds=0.25,
    )

    assert summary.frame_count == 4
    assert summary.peak == 1000


@pytest.mark.unit
def test_smoke_helper_rejects_silent_wav():
    from Helper_Scripts.TTS_Installers.smoke_test_omnivoice_sidecar import validate_wav_audio

    with pytest.raises(ValueError, match="silent"):
        validate_wav_audio(_wav_bytes([0, 0, 0, 0]))


@pytest.mark.unit
@pytest.mark.parametrize(
    ("audio_bytes", "message"),
    [
        (b"not a wav", "parseable WAV"),
        (_wav_bytes([1, 2, 3], sample_rate=16000), "24000"),
        (_wav_bytes([1, 2, 3], channels=2), "mono"),
    ],
)
def test_smoke_helper_rejects_invalid_wav_shape(audio_bytes, message):
    from Helper_Scripts.TTS_Installers.smoke_test_omnivoice_sidecar import validate_wav_audio

    with pytest.raises(ValueError, match=message):
        validate_wav_audio(audio_bytes)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_smoke_helper_runs_existing_sidecar_adapter_path_and_shuts_down(
    tmp_path,
    monkeypatch,
):
    from Helper_Scripts.TTS_Installers import smoke_test_omnivoice_sidecar as smoke

    audio_bytes = _wav_bytes([0, 600, -600, 1200, -1200])
    supervisor_instances = []
    adapter_instances = []

    class _FakeSupervisor:
        def __init__(self, provider_config, repo_root):
            self.provider_config = provider_config
            self.repo_root = repo_root
            self.shutdown_called = False
            supervisor_instances.append(self)

        async def shutdown(self):
            self.shutdown_called = True

    class _FakeAdapter:
        def __init__(self, config):
            self.config = config
            self.supervisor = None
            self.request = None
            adapter_instances.append(self)

        def set_supervisor(self, supervisor):
            self.supervisor = supervisor

        async def initialize(self):
            return True

        async def generate(self, request):
            self.request = request
            return TTSResponse(
                audio_data=audio_bytes,
                format=AudioFormat.WAV,
                sample_rate=24000,
                channels=1,
                provider="omnivoice",
                model="omnivoice",
            )

    monkeypatch.setattr(smoke, "OmniVoiceSidecarSupervisor", _FakeSupervisor, raising=True)
    monkeypatch.setattr(smoke, "OmniVoiceAdapter", _FakeAdapter, raising=True)

    config = smoke.OmniVoiceSmokeConfig(
        repo_root=tmp_path,
        model_path=tmp_path / "models" / "OmniVoice",
        sidecar_python=tmp_path / "models" / "omnivoice_sidecar" / ".venv" / "bin" / "python",
        runtime_path=tmp_path / "models" / "omnivoice_sidecar" / "runtime",
        scratch_dir=tmp_path / "models" / "omnivoice_sidecar" / "runtime" / "scratch",
        output_path=tmp_path / "smoke.wav",
        text="Hello from the managed sidecar.",
        port=8844,
        num_step=8,
        speed=1.0,
        timeout=123.0,
    )

    summary = await smoke.run_smoke(config)

    assert config.output_path.read_bytes() == audio_bytes
    assert summary.frame_count == 5
    assert supervisor_instances[0].shutdown_called is True
    assert supervisor_instances[0].repo_root == tmp_path
    assert supervisor_instances[0].provider_config["extra_params"]["python_path"] == str(
        config.sidecar_python,
    )
    assert adapter_instances[0].supervisor is supervisor_instances[0]
    assert adapter_instances[0].request.stream is False
    assert adapter_instances[0].request.format is AudioFormat.WAV


@pytest.mark.unit
@pytest.mark.asyncio
async def test_smoke_helper_preserves_primary_error_when_shutdown_fails(
    tmp_path,
    monkeypatch,
    capsys,
):
    from Helper_Scripts.TTS_Installers import smoke_test_omnivoice_sidecar as smoke

    class _FailingShutdownSupervisor:
        def __init__(self, provider_config, repo_root):
            pass

        async def shutdown(self):
            raise RuntimeError("shutdown failed")

    class _FailingAdapter:
        def __init__(self, config):
            pass

        def set_supervisor(self, supervisor):
            pass

        async def initialize(self):
            raise RuntimeError("primary failed")

    monkeypatch.setattr(
        smoke,
        "OmniVoiceSidecarSupervisor",
        _FailingShutdownSupervisor,
        raising=True,
    )
    monkeypatch.setattr(smoke, "OmniVoiceAdapter", _FailingAdapter, raising=True)

    with pytest.raises(RuntimeError, match="primary failed"):
        await smoke.run_smoke(_valid_smoke_config(tmp_path))

    assert "shutdown failed" in capsys.readouterr().err


@pytest.mark.unit
@pytest.mark.asyncio
async def test_smoke_helper_reports_shutdown_error_after_success(tmp_path, monkeypatch):
    from Helper_Scripts.TTS_Installers import smoke_test_omnivoice_sidecar as smoke

    class _FailingShutdownSupervisor:
        def __init__(self, provider_config, repo_root):
            pass

        async def shutdown(self):
            raise RuntimeError("shutdown failed")

    class _SuccessfulAdapter:
        def __init__(self, config):
            pass

        def set_supervisor(self, supervisor):
            pass

        async def initialize(self):
            return True

        async def generate(self, request):
            return TTSResponse(
                audio_data=_wav_bytes([0, 600, -600, 1200, -1200]),
                format=AudioFormat.WAV,
                sample_rate=24000,
                channels=1,
                provider="omnivoice",
                model="omnivoice",
            )

    monkeypatch.setattr(
        smoke,
        "OmniVoiceSidecarSupervisor",
        _FailingShutdownSupervisor,
        raising=True,
    )
    monkeypatch.setattr(smoke, "OmniVoiceAdapter", _SuccessfulAdapter, raising=True)

    with pytest.raises(RuntimeError, match="Failed to shut down OmniVoice sidecar supervisor"):
        await smoke.run_smoke(_valid_smoke_config(tmp_path))
