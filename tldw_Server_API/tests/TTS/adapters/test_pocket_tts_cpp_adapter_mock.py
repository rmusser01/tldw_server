# test_pocket_tts_cpp_adapter_mock.py
# Description: Mock/unit tests for PocketTTS.cpp CLI adapter
#
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock

import numpy as np
import pytest

from tldw_Server_API.app.core.TTS.adapters import pocket_tts_cpp_adapter as pocket_tts_cpp_mod
from tldw_Server_API.app.core.TTS.adapters.base import AudioFormat, ProviderStatus, TTSRequest
from tldw_Server_API.app.core.TTS.adapters.pocket_tts_cpp_adapter import PocketTTSCppAdapter
from tldw_Server_API.app.core.TTS.adapters.pocket_tts_cpp_runtime import (
    PROVIDER_MANAGED_VOICE_TOKEN_KEY,
    register_provider_managed_voice_path,
)
from tldw_Server_API.app.core.TTS.tts_exceptions import (
    TTSGenerationError,
    TTSInvalidVoiceReferenceError,
    TTSModelNotFoundError,
)


pytestmark = pytest.mark.unit


def _make_wav_bytes(*, sample_rate: int = 24000, channels: int = 1) -> bytes:
    import io
    import wave

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"\x00\x00" * sample_rate)
    return buffer.getvalue()


def _write_cpp_assets(root: Path) -> tuple[Path, Path, Path]:
    binary_path = root / "bin" / "pocket-tts"
    model_path = root / "models" / "pocket_tts_cpp" / "onnx"
    tokenizer_path = root / "models" / "pocket_tts_cpp" / "tokenizer.model"

    binary_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.mkdir(parents=True, exist_ok=True)
    tokenizer_path.parent.mkdir(parents=True, exist_ok=True)

    binary_path.write_text("#!/bin/sh\n", encoding="utf-8")
    binary_path.chmod(0o755)
    tokenizer_path.write_bytes(b"tokenizer")
    for name in [
        "flow_lm_main_int8.onnx",
        "flow_lm_flow_int8.onnx",
        "mimi_decoder_int8.onnx",
        "mimi_encoder.onnx",
        "text_conditioner.onnx",
    ]:
        (model_path / name).write_bytes(b"")

    return binary_path, model_path, tokenizer_path


def _build_adapter(
    root: Path,
    *,
    prefer_stdout: bool = True,
    streaming_transport: str = "auto",
) -> PocketTTSCppAdapter:
    binary_path, model_path, tokenizer_path = _write_cpp_assets(root)
    return PocketTTSCppAdapter(
        {
            "binary_path": str(binary_path),
            "model_path": str(model_path),
            "tokenizer_path": str(tokenizer_path),
            "timeout": 30,
            "prefer_stdout": prefer_stdout,
            "streaming_transport": streaming_transport,
        }
    )


def _provider_managed_extras(voice_path: Path) -> dict[str, str]:
    token = register_provider_managed_voice_path(voice_path)
    return {
        "pocket_tts_cpp_voice_path": str(voice_path),
        PROVIDER_MANAGED_VOICE_TOKEN_KEY: token,
    }


def _capture_pocket_tts_cpp_logs(level: str = "DEBUG") -> tuple[list[str], int]:
    messages: list[str] = []
    sink_id = pocket_tts_cpp_mod.logger.add(
        lambda message: messages.append(
            f"{message.record['message']}\n"
            f"{message.record.get('extra', {})}\n"
            f"{message.record.get('exception')}"
        ),
        level=level,
    )
    return messages, sink_id


@pytest.mark.asyncio
async def test_capabilities_do_not_advertise_streaming_for_provider_selection(tmp_path):
    adapter = _build_adapter(tmp_path)

    capabilities = await adapter.get_capabilities()

    assert capabilities.supports_streaming is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "missing_path",
    ["binary", "tokenizer", "onnx"],
)
async def test_initialize_requires_binary_tokenizer_and_onnx_assets(tmp_path, missing_path):
    adapter = _build_adapter(tmp_path)

    if missing_path == "binary":
        Path(adapter.binary_path).unlink()
    elif missing_path == "tokenizer":
        Path(adapter.tokenizer_path).unlink()
    else:
        (Path(adapter.model_path) / "mimi_decoder_int8.onnx").unlink()

    with pytest.raises(TTSModelNotFoundError):
        await adapter.initialize()


@pytest.mark.asyncio
async def test_non_streaming_generation_uses_provider_voice_path_and_stdout_for_pcm(tmp_path, monkeypatch):
    adapter = _build_adapter(tmp_path, prefer_stdout=True)
    adapter._initialized = True
    adapter._status = ProviderStatus.AVAILABLE

    voice_path = tmp_path / "voices" / "providers" / "pocket_tts_cpp" / "custom_voice-1.wav"
    voice_path.parent.mkdir(parents=True, exist_ok=True)
    voice_path.write_bytes(_make_wav_bytes())

    stdout_pcm = np.array([0.0, 0.5, -0.5], dtype=np.float32).tobytes()
    captured: dict[str, tuple[str, ...]] = {}

    class _FakeProcess:
        returncode = 0

        async def communicate(self):
            return stdout_pcm, b""

    async def _fake_create_subprocess_exec(*cmd, **kwargs):
        captured["cmd"] = tuple(str(part) for part in cmd)
        return _FakeProcess()

    monkeypatch.setattr(
        "tldw_Server_API.app.core.TTS.adapters.pocket_tts_cpp_adapter.asyncio.create_subprocess_exec",
        _fake_create_subprocess_exec,
        raising=True,
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.TTS.adapters.pocket_tts_cpp_adapter.tempfile.NamedTemporaryFile",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("stdout transport should not create a temp output file")),
        raising=True,
    )

    request = TTSRequest(
        text="hello world",
        voice="custom:voice-1",
        format=AudioFormat.PCM,
        stream=False,
        extra_params=_provider_managed_extras(voice_path),
    )

    response = await adapter.generate(request)

    assert response.audio_data is not None
    assert response.audio_data != stdout_pcm
    assert response.metadata["transport"] == "stdout"
    assert "--stdout" in captured["cmd"]
    assert str(voice_path) in captured["cmd"]


@pytest.mark.asyncio
async def test_non_streaming_generation_failure_log_sanitizes_exception_text(tmp_path, monkeypatch):
    adapter = _build_adapter(tmp_path, prefer_stdout=True)
    adapter._initialized = True
    adapter._status = ProviderStatus.AVAILABLE

    voice_path = tmp_path / "voices" / "providers" / "pocket_tts_cpp" / "custom_voice-secret.wav"
    voice_path.parent.mkdir(parents=True, exist_ok=True)
    voice_path.write_bytes(_make_wav_bytes())
    raw_marker = "RAW_POCKET_TTS_CPP_GENERATION_SECRET_MARKER token=secret"

    class _FakeProcess:
        returncode = 0

        async def communicate(self):
            return b"pcm", b""

    async def _fake_create_subprocess_exec(*cmd, **kwargs):
        return _FakeProcess()

    async def _fake_convert_stdout_audio(stdout: bytes, target_format: AudioFormat) -> bytes:
        raise RuntimeError(raw_marker)

    monkeypatch.setattr(
        "tldw_Server_API.app.core.TTS.adapters.pocket_tts_cpp_adapter.asyncio.create_subprocess_exec",
        _fake_create_subprocess_exec,
        raising=True,
    )
    monkeypatch.setattr(adapter, "_convert_stdout_audio", _fake_convert_stdout_audio)

    request = TTSRequest(
        text="hello world",
        voice="custom:voice-secret",
        format=AudioFormat.PCM,
        stream=False,
        extra_params=_provider_managed_extras(voice_path),
    )
    messages, sink_id = _capture_pocket_tts_cpp_logs(level="ERROR")

    try:
        with pytest.raises(TTSGenerationError) as exc_info:
            await adapter.generate(request)
    finally:
        pocket_tts_cpp_mod.logger.remove(sink_id)

    assert raw_marker in exc_info.value.details["error"]
    rendered_logs = "\n".join(messages)
    assert "PocketTTS.cpp generation failed" in rendered_logs
    assert "RAW_POCKET_TTS_CPP_GENERATION_SECRET_MARKER" not in rendered_logs
    assert "token=secret" not in rendered_logs
    assert "exception_type=RuntimeError" in rendered_logs


@pytest.mark.asyncio
async def test_non_streaming_generation_requires_provider_managed_voice_path_for_custom_voice(tmp_path):
    adapter = _build_adapter(tmp_path)
    adapter._initialized = True
    adapter._status = ProviderStatus.AVAILABLE

    request = TTSRequest(
        text="hello world",
        voice="custom:voice-1",
        format=AudioFormat.WAV,
        stream=False,
        extra_params={},
    )

    with pytest.raises(TTSInvalidVoiceReferenceError):
        await adapter.generate(request)


@pytest.mark.asyncio
async def test_non_streaming_generation_requires_provider_managed_voice_path_for_direct_reference(tmp_path):
    adapter = _build_adapter(tmp_path)
    adapter._initialized = True
    adapter._status = ProviderStatus.AVAILABLE

    request = TTSRequest(
        text="hello world",
        format=AudioFormat.WAV,
        stream=False,
        voice_reference=_make_wav_bytes(),
        extra_params={},
    )

    with pytest.raises(TTSInvalidVoiceReferenceError):
        await adapter.generate(request)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "target_format",
    [AudioFormat.WAV, AudioFormat.MP3, AudioFormat.OPUS, AudioFormat.FLAC, AudioFormat.PCM, AudioFormat.AAC],
)
async def test_non_streaming_generation_normalizes_file_output_to_requested_format(tmp_path, monkeypatch, target_format):
    adapter = _build_adapter(tmp_path, prefer_stdout=False)
    adapter._initialized = True
    adapter._status = ProviderStatus.AVAILABLE

    voice_path = tmp_path / "voices" / "providers" / "pocket_tts_cpp" / "custom_voice-2.wav"
    voice_path.parent.mkdir(parents=True, exist_ok=True)
    voice_path.write_bytes(_make_wav_bytes())

    recorded: dict[str, object] = {}
    target_bytes = b"normalized-" + target_format.value.encode("utf-8")
    wav_bytes = _make_wav_bytes()

    class _FakeProcess:
        returncode = 0

        async def communicate(self):
            output_path = Path(str(recorded["output_path"]))
            output_path.write_bytes(wav_bytes)
            return b"", b""

    async def _fake_create_subprocess_exec(*cmd, **kwargs):
        recorded["cmd"] = tuple(str(part) for part in cmd)
        recorded["output_path"] = cmd[-1]
        return _FakeProcess()

    async def _fake_convert_format(input_path: Path, output_path: Path, target_format_arg: str, **kwargs):
        recorded["convert"] = (Path(input_path), Path(output_path), target_format_arg, kwargs)
        Path(output_path).write_bytes(target_bytes)
        return True

    monkeypatch.setattr(
        "tldw_Server_API.app.core.TTS.adapters.pocket_tts_cpp_adapter.asyncio.create_subprocess_exec",
        _fake_create_subprocess_exec,
        raising=True,
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.TTS.adapters.pocket_tts_cpp_adapter.AudioConverter.convert_format",
        _fake_convert_format,
        raising=True,
    )

    request = TTSRequest(
        text="hello world",
        voice="custom:voice-2",
        format=target_format,
        stream=False,
        extra_params=_provider_managed_extras(voice_path),
    )

    response = await adapter.generate(request)

    if target_format == AudioFormat.WAV:
        expected_bytes = wav_bytes
    elif target_format == AudioFormat.PCM:
        import io
        import wave

        with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
            expected_bytes = wav_file.readframes(wav_file.getnframes())
    else:
        expected_bytes = target_bytes
    assert response.audio_data == expected_bytes
    assert response.metadata["transport"] == "file"
    assert "--stdout" not in recorded["cmd"]
    assert str(voice_path) in recorded["cmd"]
    if target_format in {AudioFormat.WAV, AudioFormat.PCM}:
        assert "convert" not in recorded
    else:
        assert recorded["convert"][2] == target_format.value


@pytest.mark.asyncio
async def test_streaming_generation_probes_cli_once_and_returns_stdout_stream(tmp_path, monkeypatch):
    adapter = _build_adapter(tmp_path, prefer_stdout=True)
    adapter._initialized = True
    adapter._status = ProviderStatus.AVAILABLE

    voice_path = tmp_path / "voices" / "providers" / "pocket_tts_cpp" / "custom_voice-stream.wav"
    voice_path.parent.mkdir(parents=True, exist_ok=True)
    voice_path.write_bytes(_make_wav_bytes())

    probe_calls = 0

    async def _fake_probe() -> bool:
        nonlocal probe_calls
        probe_calls += 1
        return True

    async def _fake_cli_stream(request: TTSRequest, resolved_voice_path: Path):
        assert request.stream is True
        assert resolved_voice_path == voice_path
        for chunk in [b"cli-chunk-1", b"cli-chunk-2"]:
            yield chunk

    monkeypatch.setattr(adapter, "_probe_cli_streaming_support", _fake_probe)
    monkeypatch.setattr(adapter, "_stream_via_cli_stdout", _fake_cli_stream)

    request = TTSRequest(
        text="stream this",
        voice="custom:voice-stream",
        format=AudioFormat.PCM,
        stream=True,
        extra_params=_provider_managed_extras(voice_path),
    )

    response_one = await adapter.generate(request)
    assert response_one.metadata["transport"] == "stdout_stream"
    assert [chunk async for chunk in response_one.audio_stream] == [b"cli-chunk-1", b"cli-chunk-2"]

    response_two = await adapter.generate(request)
    assert [chunk async for chunk in response_two.audio_stream] == [b"cli-chunk-1", b"cli-chunk-2"]
    assert probe_calls == 1


@pytest.mark.asyncio
async def test_streaming_generation_retries_probe_after_non_incremental_result(tmp_path, monkeypatch):
    adapter = _build_adapter(tmp_path, prefer_stdout=True)
    adapter._initialized = True
    adapter._status = ProviderStatus.AVAILABLE

    voice_path = tmp_path / "voices" / "providers" / "pocket_tts_cpp" / "custom_voice-retry.wav"
    voice_path.parent.mkdir(parents=True, exist_ok=True)
    voice_path.write_bytes(_make_wav_bytes())

    probe_calls = 0

    async def _fake_probe() -> bool:
        nonlocal probe_calls
        probe_calls += 1
        return False

    async def _fake_cli_stream(request: TTSRequest, resolved_voice_path: Path):
        assert resolved_voice_path == voice_path
        assert request.stream is True
        for chunk in [b"cli-retry-1", b"cli-retry-2"]:
            yield chunk

    monkeypatch.setattr(adapter, "_probe_cli_streaming_support", _fake_probe)
    monkeypatch.setattr(adapter, "_stream_via_cli_stdout", _fake_cli_stream)

    request = TTSRequest(
        text="retry stream",
        voice="custom:voice-retry",
        format=AudioFormat.PCM,
        stream=True,
        extra_params=_provider_managed_extras(voice_path),
    )

    with pytest.raises(TTSGenerationError):
        await adapter.generate(request)

    with pytest.raises(TTSGenerationError):
        await adapter.generate(request)
    assert probe_calls == 1


@pytest.mark.asyncio
async def test_convert_stdout_audio_trims_partial_float_frame_before_decoding(tmp_path, monkeypatch):
    adapter = _build_adapter(tmp_path)

    captured: dict[str, object] = {}

    async def _fake_convert(audio_data, source_format, target_format, sample_rate):
        captured["audio_data"] = audio_data
        captured["source_format"] = source_format
        captured["target_format"] = target_format
        captured["sample_rate"] = sample_rate
        return b"converted"

    monkeypatch.setattr(adapter, "convert_audio_format", _fake_convert)

    stdout = np.array([0.0, 0.5, -0.5], dtype=np.dtype("<f4")).tobytes() + b"\xff"
    output = await adapter._convert_stdout_audio(stdout, AudioFormat.WAV)

    assert output == b"converted"
    assert captured["source_format"] == AudioFormat.PCM
    assert captured["target_format"] == AudioFormat.WAV
    assert captured["sample_rate"] == adapter.DEFAULT_SAMPLE_RATE
    assert list(captured["audio_data"]) == [0, 16383, -16383]


@pytest.mark.asyncio
async def test_streaming_generation_forced_cli_still_probes_feasibility(tmp_path, monkeypatch):
    adapter = _build_adapter(tmp_path, prefer_stdout=True, streaming_transport="cli")
    adapter._initialized = True
    adapter._status = ProviderStatus.AVAILABLE

    voice_path = tmp_path / "voices" / "providers" / "pocket_tts_cpp" / "custom_voice-forced-cli.wav"
    voice_path.parent.mkdir(parents=True, exist_ok=True)
    voice_path.write_bytes(_make_wav_bytes())

    probe_calls = 0

    async def _fake_probe() -> bool:
        nonlocal probe_calls
        probe_calls += 1
        return False

    monkeypatch.setattr(adapter, "_probe_cli_streaming_support", _fake_probe)

    request = TTSRequest(
        text="forced cli probe",
        voice="custom:voice-forced-cli",
        format=AudioFormat.PCM,
        stream=True,
        extra_params=_provider_managed_extras(voice_path),
    )

    with pytest.raises(TTSGenerationError):
        await adapter.generate(request)

    assert probe_calls == 1


@pytest.mark.asyncio
async def test_cli_streaming_probe_fallback_log_sanitizes_exception_text(tmp_path, monkeypatch):
    adapter = _build_adapter(tmp_path)
    adapter._initialized = True
    adapter._status = ProviderStatus.AVAILABLE

    voice_path = tmp_path / "voices" / "providers" / "pocket_tts_cpp" / "custom_voice-probe-secret.wav"
    voice_path.parent.mkdir(parents=True, exist_ok=True)
    voice_path.write_bytes(_make_wav_bytes())
    raw_marker = "RAW_POCKET_TTS_CPP_PROBE_SECRET_MARKER token=secret"

    async def _fake_probe() -> bool:
        raise RuntimeError(raw_marker)

    monkeypatch.setattr(adapter, "_probe_cli_streaming_support", _fake_probe)
    messages, sink_id = _capture_pocket_tts_cpp_logs(level="WARNING")

    try:
        supported = await adapter._get_cli_streaming_support(voice_path)
    finally:
        pocket_tts_cpp_mod.logger.remove(sink_id)

    assert supported is False
    assert adapter._cli_streaming_supported is False
    rendered_logs = "\n".join(messages)
    assert "PocketTTS.cpp CLI streaming probe failed" in rendered_logs
    assert "RAW_POCKET_TTS_CPP_PROBE_SECRET_MARKER" not in rendered_logs
    assert "token=secret" not in rendered_logs
    assert "exception_type=RuntimeError" in rendered_logs


@pytest.mark.asyncio
async def test_non_streaming_generation_rejects_arbitrary_existing_voice_path_without_provider_root(
    tmp_path,
):
    adapter = _build_adapter(tmp_path)
    adapter._initialized = True
    adapter._status = ProviderStatus.AVAILABLE

    unmanaged_voice_path = tmp_path / "unmanaged" / "voice.wav"
    unmanaged_voice_path.parent.mkdir(parents=True, exist_ok=True)
    unmanaged_voice_path.write_bytes(_make_wav_bytes())

    request = TTSRequest(
        text="hello world",
        voice="custom:voice-unmanaged",
        format=AudioFormat.WAV,
        stream=False,
        extra_params={
            "pocket_tts_cpp_voice_path": str(unmanaged_voice_path),
            PROVIDER_MANAGED_VOICE_TOKEN_KEY: "forged-token",
        },
    )

    with pytest.raises(TTSInvalidVoiceReferenceError):
        await adapter.generate(request)


@pytest.mark.asyncio
async def test_non_streaming_generation_rejects_voice_path_outside_provider_managed_root(
    tmp_path,
):
    adapter = _build_adapter(tmp_path)
    adapter._initialized = True
    adapter._status = ProviderStatus.AVAILABLE

    provider_root = tmp_path / "voices" / "providers" / "pocket_tts_cpp"
    provider_root.mkdir(parents=True, exist_ok=True)
    voice_path = tmp_path / "shared" / "voice.wav"
    voice_path.parent.mkdir(parents=True, exist_ok=True)
    voice_path.write_bytes(_make_wav_bytes())
    managed_voice_path = provider_root / "managed.wav"
    managed_voice_path.write_bytes(_make_wav_bytes())

    request = TTSRequest(
        text="hello world",
        voice="custom:voice-outside-root",
        format=AudioFormat.WAV,
        stream=False,
        extra_params={
            "pocket_tts_cpp_voice_path": str(voice_path),
            PROVIDER_MANAGED_VOICE_TOKEN_KEY: register_provider_managed_voice_path(managed_voice_path),
        },
    )

    with pytest.raises(TTSInvalidVoiceReferenceError):
        await adapter.generate(request)
