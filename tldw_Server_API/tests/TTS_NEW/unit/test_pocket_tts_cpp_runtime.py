import asyncio
import hashlib
import os
import time
import wave
from io import BytesIO
from pathlib import Path

import pytest

from tldw_Server_API.app.core.TTS.voice_manager import VoiceManager, VOICE_RATE_LIMITS


class _RuntimeVoiceManager:
    def __init__(self, root: Path, voice_bytes: bytes = b"RIFF" + b"\x00" * 64):
        self.root = root
        self.voice_bytes = voice_bytes

    def get_user_voices_path(self, user_id: int) -> Path:
        assert user_id == 7
        self.root.mkdir(parents=True, exist_ok=True)
        return self.root

    async def load_voice_reference_audio(self, user_id: int, voice_id: str) -> bytes:
        assert user_id == 7
        assert voice_id == "voice-123"
        return self.voice_bytes


def _make_wav_bytes(
    payload: bytes = b"\x00\x01" * 8,
    *,
    sample_rate: int = 24000,
    channels: int = 1,
    sample_width: int = 2,
) -> bytes:
    buffer = BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(payload)
    return buffer.getvalue()


class _TimedProbeStdout:
    def __init__(self, chunks: list[tuple[float, bytes]]):
        self._chunks = chunks
        self._read_calls = 0

    async def read(self, _n: int) -> bytes:
        if self._read_calls >= len(self._chunks):
            return b""

        delay, chunk = self._chunks[self._read_calls]
        if delay:
            await asyncio.sleep(delay)
        self._read_calls += 1
        return chunk


class _TimedProbeProcess:
    def __init__(
        self,
        *,
        chunks: list[tuple[float, bytes]],
        returncode: int = 0,
        stderr_bytes: bytes = b"",
    ):
        self.stdout = _TimedProbeStdout(chunks)
        self.stderr_bytes = stderr_bytes
        self.returncode = None
        self._returncode = returncode

    async def communicate(self):
        self.returncode = self._returncode
        return b"", self.stderr_bytes

    def kill(self):
        self.returncode = -9


@pytest.mark.asyncio
async def test_pocket_tts_cpp_probe_accepts_only_when_later_stdout_progress_arrives(
    monkeypatch,
):
    from tldw_Server_API.app.core.TTS.adapters import pocket_tts_cpp_runtime as runtime_module
    from tldw_Server_API.app.core.TTS.adapters.pocket_tts_cpp_runtime import probe_cli_streaming_incremental

    fake_process = _TimedProbeProcess(
        chunks=[
            (0.0, b"probe-bytes"),
            (0.03, b"later-bytes"),
        ]
    )

    async def _fake_create_subprocess_exec(*args, **kwargs):  # noqa: ARG001
        return fake_process

    monkeypatch.setattr(runtime_module.asyncio, "create_subprocess_exec", _fake_create_subprocess_exec, raising=True)

    result = await probe_cli_streaming_incremental(
        binary_path=Path("/tmp/pocket-tts"),
        voice_path=Path("/tmp/voice.wav"),
        model_path=Path("/tmp/models"),
        tokenizer_path=Path("/tmp/tokenizer.model"),
        precision="int8",
        timeout=0.1,
        enable_voice_cache=True,
    )

    assert result is True


@pytest.mark.asyncio
async def test_pocket_tts_cpp_probe_rejects_stdout_bytes_when_process_exits_immediately_after_first_output(
    monkeypatch,
):
    from tldw_Server_API.app.core.TTS.adapters import pocket_tts_cpp_runtime as runtime_module
    from tldw_Server_API.app.core.TTS.adapters.pocket_tts_cpp_runtime import probe_cli_streaming_incremental

    fake_process = _TimedProbeProcess(
        chunks=[
            (0.0, b"probe-bytes"),
        ]
    )

    async def _fake_create_subprocess_exec(*args, **kwargs):  # noqa: ARG001
        return fake_process

    monkeypatch.setattr(runtime_module.asyncio, "create_subprocess_exec", _fake_create_subprocess_exec, raising=True)

    result = await probe_cli_streaming_incremental(
        binary_path=Path("/tmp/pocket-tts"),
        voice_path=Path("/tmp/voice.wav"),
        model_path=Path("/tmp/models"),
        tokenizer_path=Path("/tmp/tokenizer.model"),
        precision="int8",
        timeout=0.1,
        enable_voice_cache=True,
    )

    assert result is False


@pytest.mark.asyncio
async def test_pocket_tts_cpp_probe_rejects_early_burst_without_later_stdout_progress(
    monkeypatch,
):
    from tldw_Server_API.app.core.TTS.adapters import pocket_tts_cpp_runtime as runtime_module
    from tldw_Server_API.app.core.TTS.adapters.pocket_tts_cpp_runtime import probe_cli_streaming_incremental

    fake_process = _TimedProbeProcess(
        chunks=[
            (0.0, b"probe-bytes"),
            (0.0, b"burst-bytes"),
        ],
    )

    async def _fake_create_subprocess_exec(*args, **kwargs):  # noqa: ARG001
        return fake_process

    monkeypatch.setattr(runtime_module.asyncio, "create_subprocess_exec", _fake_create_subprocess_exec, raising=True)

    result = await probe_cli_streaming_incremental(
        binary_path=Path("/tmp/pocket-tts"),
        voice_path=Path("/tmp/voice.wav"),
        model_path=Path("/tmp/models"),
        tokenizer_path=Path("/tmp/tokenizer.model"),
        precision="int8",
        timeout=0.1,
        enable_voice_cache=True,
    )

    assert result is False


def test_pocket_tts_cpp_trust_token_binds_to_registered_voice_path(tmp_path):
    from tldw_Server_API.app.core.TTS.adapters.pocket_tts_cpp_runtime import (
        PROVIDER_MANAGED_VOICE_TOKEN_KEY,
        register_provider_managed_voice_path,
        resolve_provider_managed_voice_path,
        revoke_provider_managed_voice_token,
    )

    voice_path = tmp_path / "voices" / "providers" / "pocket_tts_cpp" / "voice.wav"
    voice_path.parent.mkdir(parents=True, exist_ok=True)
    voice_path.write_bytes(b"RIFF" + b"\x00" * 32)

    token = register_provider_managed_voice_path(voice_path)
    assert token
    assert isinstance(token, str)
    assert token != str(voice_path)
    assert len(token) > 16
    assert PROVIDER_MANAGED_VOICE_TOKEN_KEY.startswith("_")

    resolved = resolve_provider_managed_voice_path(token, voice_path)
    assert resolved == voice_path.resolve()

    revoke_provider_managed_voice_token(token)

    with pytest.raises(ValueError):
        resolve_provider_managed_voice_path(token, voice_path)


def test_pocket_tts_cpp_trust_token_rejects_forged_voice_path(tmp_path):
    from tldw_Server_API.app.core.TTS.adapters.pocket_tts_cpp_runtime import (
        register_provider_managed_voice_path,
        resolve_provider_managed_voice_path,
        revoke_provider_managed_voice_token,
    )

    managed_path = tmp_path / "voices" / "providers" / "pocket_tts_cpp" / "managed.wav"
    forged_path = tmp_path / "voices" / "providers" / "pocket_tts_cpp" / "forged.wav"
    managed_path.parent.mkdir(parents=True, exist_ok=True)
    managed_path.write_bytes(b"RIFF" + b"\x00" * 32)
    forged_path.write_bytes(b"RIFF" + b"\x01" * 32)

    token = register_provider_managed_voice_path(managed_path)

    with pytest.raises(ValueError):
        resolve_provider_managed_voice_path(token, forged_path)

    revoke_provider_managed_voice_token(token)


def test_pocket_tts_cpp_trust_token_rejects_paths_outside_provider_runtime_dir(tmp_path):
    from tldw_Server_API.app.core.TTS.adapters.pocket_tts_cpp_runtime import (
        register_provider_managed_voice_path,
    )

    unmanaged_path = tmp_path / "voices" / "shared" / "voice.wav"
    unmanaged_path.parent.mkdir(parents=True, exist_ok=True)
    unmanaged_path.write_bytes(b"RIFF" + b"\x00" * 32)

    with pytest.raises(ValueError):
        register_provider_managed_voice_path(unmanaged_path)


def test_pocket_tts_cpp_validate_runtime_assets_requires_tokenizer_file_and_model_directory(tmp_path):
    from tldw_Server_API.app.core.TTS.adapters.pocket_tts_cpp_runtime import validate_runtime_assets
    from tldw_Server_API.app.core.TTS.tts_exceptions import TTSModelNotFoundError

    binary_path = tmp_path / "bin" / "pocket-tts"
    tokenizer_path = tmp_path / "models" / "pocket_tts_cpp" / "tokenizer.model"
    model_path = tmp_path / "models" / "pocket_tts_cpp" / "onnx"

    binary_path.parent.mkdir(parents=True, exist_ok=True)
    binary_path.write_text("#!/bin/sh\n", encoding="utf-8")
    binary_path.chmod(0o755)
    tokenizer_path.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_bytes(b"not-a-directory")

    with pytest.raises(TTSModelNotFoundError, match="tokenizer"):
        validate_runtime_assets(
            binary_path=binary_path,
            model_path=model_path,
            tokenizer_path=tokenizer_path,
            precision="int8",
        )


@pytest.mark.asyncio
async def test_write_runtime_file_writes_via_atomic_replace(tmp_path, monkeypatch):
    from tldw_Server_API.app.core.TTS.adapters import pocket_tts_cpp_runtime as runtime_module

    target_path = tmp_path / "voices" / "providers" / "pocket_tts_cpp" / "voice.wav"
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_bytes(b"old-bytes")
    replace_calls: list[tuple[Path, Path, bytes, bytes]] = []

    def _fake_replace(src: str, dst: str) -> None:
        src_path = Path(src)
        dst_path = Path(dst)
        replace_calls.append((src_path, dst_path, src_path.read_bytes(), dst_path.read_bytes()))
        os.unlink(dst_path)
        os.rename(src_path, dst_path)

    monkeypatch.setattr(runtime_module.os, "replace", _fake_replace, raising=True)

    await runtime_module._write_runtime_file(target_path, b"new-bytes")

    assert replace_calls
    src_path, dst_path, temp_bytes, old_bytes = replace_calls[0]
    assert src_path != target_path
    assert dst_path == target_path
    assert temp_bytes == b"new-bytes"
    assert old_bytes == b"old-bytes"
    assert target_path.read_bytes() == b"new-bytes"


@pytest.mark.asyncio
async def test_pocket_tts_cpp_materializes_stored_voice_to_stable_custom_path(tmp_path):
    from tldw_Server_API.app.core.TTS.adapters import pocket_tts_cpp_runtime as runtime_module
    from tldw_Server_API.app.core.TTS.adapters.pocket_tts_cpp_runtime import (
        materialize_custom_voice_reference,
    )

    runtime_root = tmp_path / "voices"
    manager = _RuntimeVoiceManager(runtime_root)
    normalized = _make_wav_bytes(b"\x00\x01" * 8)

    async def _fake_convert(input_path: Path, output_path: Path, sample_rate: int, channels: int, bit_depth: int) -> bool:
        assert input_path.read_bytes() == manager.voice_bytes
        assert sample_rate == 24000
        assert channels == 1
        assert bit_depth == 16
        output_path.write_bytes(normalized)
        return True

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(runtime_module.AudioConverter, "convert_to_wav", _fake_convert)
    try:
        materialized = await materialize_custom_voice_reference(
            voice_manager=manager,
            user_id=7,
            voice_id="voice-123",
        )
    finally:
        monkeypatch.undo()

    assert materialized == runtime_root / "providers" / "pocket_tts_cpp" / "custom_voice-123.wav"
    assert materialized.exists()
    assert materialized.read_bytes() == normalized


@pytest.mark.asyncio
async def test_pocket_tts_cpp_normalizes_stored_custom_voice_before_materialization(tmp_path):
    from tldw_Server_API.app.core.TTS.adapters import pocket_tts_cpp_runtime as runtime_module
    from tldw_Server_API.app.core.TTS.adapters.pocket_tts_cpp_runtime import (
        materialize_custom_voice_reference,
    )

    runtime_root = tmp_path / "voices"
    stored_bytes = b"ID3" + b"\x06" * 20
    manager = _RuntimeVoiceManager(runtime_root, voice_bytes=stored_bytes)
    normalized = _make_wav_bytes(b"\x06\x07" * 8)

    async def _fake_convert(input_path: Path, output_path: Path, sample_rate: int, channels: int, bit_depth: int) -> bool:
        assert input_path.read_bytes() == stored_bytes
        assert sample_rate == 24000
        assert channels == 1
        assert bit_depth == 16
        output_path.write_bytes(normalized)
        return True

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(runtime_module.AudioConverter, "convert_to_wav", _fake_convert)
    try:
        materialized = await materialize_custom_voice_reference(
            voice_manager=manager,
            user_id=7,
            voice_id="voice-123",
        )
    finally:
        monkeypatch.undo()

    assert materialized.exists()
    assert materialized.read_bytes() == normalized


@pytest.mark.asyncio
async def test_pocket_tts_cpp_materializes_direct_reference_to_deterministic_ref_path(tmp_path):
    from tldw_Server_API.app.core.TTS.adapters import pocket_tts_cpp_runtime as runtime_module
    from tldw_Server_API.app.core.TTS.adapters.pocket_tts_cpp_runtime import (
        materialize_direct_voice_reference,
    )

    voice_reference = b"RIFF" + b"\x01" * 32
    runtime_root = tmp_path / "voices"
    manager = _RuntimeVoiceManager(runtime_root)
    normalized = _make_wav_bytes(b"\x01\x02" * 8)

    async def _fake_convert(input_path: Path, output_path: Path, sample_rate: int, channels: int, bit_depth: int) -> bool:
        assert input_path.read_bytes() == voice_reference
        assert sample_rate == 24000
        assert channels == 1
        assert bit_depth == 16
        output_path.write_bytes(normalized)
        return True

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(runtime_module.AudioConverter, "convert_to_wav", _fake_convert)
    try:
        materialized, is_transient = await materialize_direct_voice_reference(
            voice_manager=manager,
            user_id=7,
            voice_reference=voice_reference,
            persist_direct_voice_references=True,
        )
    finally:
        monkeypatch.undo()

    expected_name = f"ref_{hashlib.sha256(normalized).hexdigest()}.wav"
    assert materialized == runtime_root / "providers" / "pocket_tts_cpp" / expected_name
    assert materialized.exists()
    assert materialized.read_bytes() == normalized
    assert is_transient is False


@pytest.mark.asyncio
async def test_pocket_tts_cpp_normalizes_non_wav_direct_reference_before_materialization(tmp_path):
    from tldw_Server_API.app.core.TTS.adapters import pocket_tts_cpp_runtime as runtime_module
    from tldw_Server_API.app.core.TTS.adapters.pocket_tts_cpp_runtime import (
        materialize_direct_voice_reference,
    )

    runtime_root = tmp_path / "voices"
    manager = _RuntimeVoiceManager(runtime_root)
    mp3_like_bytes = b"ID3" + b"\x04" * 17
    normalized = _make_wav_bytes(b"\x04\x05" * 12)
    converter_calls: list[tuple[Path, Path]] = []

    async def _fake_convert(input_path: Path, output_path: Path, sample_rate: int, channels: int, bit_depth: int) -> bool:
        converter_calls.append((input_path, output_path))
        assert input_path.read_bytes() == mp3_like_bytes
        assert sample_rate == 24000
        assert channels == 1
        assert bit_depth == 16
        output_path.write_bytes(normalized)
        return True

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(runtime_module.AudioConverter, "convert_to_wav", _fake_convert)
    try:
        materialized, is_transient = await materialize_direct_voice_reference(
            voice_manager=manager,
            user_id=7,
            voice_reference=mp3_like_bytes,
            persist_direct_voice_references=True,
        )
    finally:
        monkeypatch.undo()

    expected_name = f"ref_{hashlib.sha256(normalized).hexdigest()}.wav"
    assert materialized == runtime_root / "providers" / "pocket_tts_cpp" / expected_name
    assert is_transient is False
    assert len(converter_calls) == 1
    assert materialized.read_bytes() == normalized
    with wave.open(str(materialized), "rb") as wav_file:
        assert wav_file.getnchannels() == 1
        assert wav_file.getsampwidth() == 2
        assert wav_file.getframerate() == 24000


@pytest.mark.asyncio
async def test_pocket_tts_cpp_transient_direct_references_use_unique_request_scoped_paths(tmp_path):
    from tldw_Server_API.app.core.TTS.adapters import pocket_tts_cpp_runtime as runtime_module
    from tldw_Server_API.app.core.TTS.adapters.pocket_tts_cpp_runtime import (
        materialize_direct_voice_reference,
    )

    runtime_root = tmp_path / "voices"
    manager = _RuntimeVoiceManager(runtime_root)
    voice_reference = b"RIFF" + b"\x07" * 32
    normalized = _make_wav_bytes(b"\x07\x08" * 10)

    async def _fake_convert(input_path: Path, output_path: Path, sample_rate: int, channels: int, bit_depth: int) -> bool:
        output_path.write_bytes(normalized)
        return True

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(runtime_module.AudioConverter, "convert_to_wav", _fake_convert)
    try:
        first_path, first_transient = await materialize_direct_voice_reference(
            voice_manager=manager,
            user_id=7,
            voice_reference=voice_reference,
            persist_direct_voice_references=False,
        )
        second_path, second_transient = await materialize_direct_voice_reference(
            voice_manager=manager,
            user_id=7,
            voice_reference=voice_reference,
            persist_direct_voice_references=False,
        )
    finally:
        monkeypatch.undo()

    assert first_transient is True
    assert second_transient is True
    assert first_path != second_path
    assert first_path.exists()
    assert second_path.exists()


@pytest.mark.asyncio
async def test_pocket_tts_cpp_materialized_direct_reference_enforces_max_bytes_immediately(tmp_path):
    from tldw_Server_API.app.core.TTS.adapters import pocket_tts_cpp_runtime as runtime_module
    from tldw_Server_API.app.core.TTS.adapters.pocket_tts_cpp_runtime import (
        materialize_direct_voice_reference,
    )

    runtime_root = tmp_path / "voices"
    manager = _RuntimeVoiceManager(runtime_root)
    runtime_dir = runtime_root / "providers" / "pocket_tts_cpp"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    old_file = runtime_dir / "ref_old.wav"
    old_file.write_bytes(b"old-old")
    normalized = _make_wav_bytes(b"\x03\x03")

    async def _fake_convert(input_path: Path, output_path: Path, sample_rate: int, channels: int, bit_depth: int) -> bool:
        output_path.write_bytes(normalized)
        return True

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(runtime_module.AudioConverter, "convert_to_wav", _fake_convert)
    try:
        materialized, is_transient = await materialize_direct_voice_reference(
            voice_manager=manager,
            user_id=7,
            voice_reference=b"RIFF" + b"\x03" * 8,
            persist_direct_voice_references=True,
            cache_max_bytes=12,
        )
    finally:
        monkeypatch.undo()

    assert is_transient is False
    assert materialized.exists()
    assert old_file.exists()
    remaining_files = sorted(runtime_dir.glob("*.wav"))
    assert remaining_files == sorted([materialized, old_file])
    assert sum(path.stat().st_size for path in remaining_files) == (
        materialized.stat().st_size + old_file.stat().st_size
    )


@pytest.mark.asyncio
async def test_pocket_tts_cpp_post_write_pruning_never_returns_deleted_active_path(tmp_path):
    from tldw_Server_API.app.core.TTS.adapters import pocket_tts_cpp_runtime as runtime_module
    from tldw_Server_API.app.core.TTS.adapters.pocket_tts_cpp_runtime import (
        materialize_direct_voice_reference,
    )

    runtime_root = tmp_path / "voices"
    manager = _RuntimeVoiceManager(runtime_root)
    runtime_dir = runtime_root / "providers" / "pocket_tts_cpp"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    stale_file = runtime_dir / "ref_stale.wav"
    stale_file.write_bytes(b"old-old-old-old")
    normalized = _make_wav_bytes(b"\x09\x09" * 2)

    async def _fake_convert(input_path: Path, output_path: Path, sample_rate: int, channels: int, bit_depth: int) -> bool:
        output_path.write_bytes(normalized)
        return True

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(runtime_module.AudioConverter, "convert_to_wav", _fake_convert)
    try:
        materialized, is_transient = await materialize_direct_voice_reference(
            voice_manager=manager,
            user_id=7,
            voice_reference=b"RIFF" + b"\x09" * 20,
            persist_direct_voice_references=True,
            cache_max_bytes=8,
        )
    finally:
        monkeypatch.undo()

    assert is_transient is False
    assert materialized.exists()
    assert stale_file.exists()
    assert materialized in runtime_dir.glob("*.wav")


@pytest.mark.asyncio
async def test_pocket_tts_cpp_custom_voice_materialization_enforces_max_bytes_immediately(tmp_path):
    from tldw_Server_API.app.core.TTS.adapters import pocket_tts_cpp_runtime as runtime_module
    from tldw_Server_API.app.core.TTS.adapters.pocket_tts_cpp_runtime import (
        materialize_custom_voice_reference,
    )

    runtime_root = tmp_path / "voices"
    manager = _RuntimeVoiceManager(runtime_root, voice_bytes=b"RIFF" + b"\x05" * 12)
    runtime_dir = runtime_root / "providers" / "pocket_tts_cpp"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    old_file = runtime_dir / "ref_old.wav"
    old_file.write_bytes(b"old-old")
    normalized = _make_wav_bytes(b"\x05\x06" * 8)

    async def _fake_convert(input_path: Path, output_path: Path, sample_rate: int, channels: int, bit_depth: int) -> bool:
        assert input_path.read_bytes() == manager.voice_bytes
        output_path.write_bytes(normalized)
        return True

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(runtime_module.AudioConverter, "convert_to_wav", _fake_convert)
    try:
        materialized = await materialize_custom_voice_reference(
            voice_manager=manager,
            user_id=7,
            voice_id="voice-123",
            cache_max_bytes=20,
        )
    finally:
        monkeypatch.undo()

    assert materialized.exists()
    assert old_file.exists()
    assert materialized in runtime_dir.glob("*.wav")
    assert materialized.read_bytes() == normalized


@pytest.mark.asyncio
async def test_pocket_tts_cpp_deletes_transient_direct_reference_when_persistence_disabled(tmp_path):
    from tldw_Server_API.app.core.TTS.adapters import pocket_tts_cpp_runtime as runtime_module
    from tldw_Server_API.app.core.TTS.adapters.pocket_tts_cpp_runtime import (
        cleanup_transient_voice_reference,
        materialize_direct_voice_reference,
    )

    runtime_root = tmp_path / "voices"
    manager = _RuntimeVoiceManager(runtime_root)
    normalized = _make_wav_bytes(b"\x02\x02" * 8)

    async def _fake_convert(input_path: Path, output_path: Path, sample_rate: int, channels: int, bit_depth: int) -> bool:
        output_path.write_bytes(normalized)
        return True

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(runtime_module.AudioConverter, "convert_to_wav", _fake_convert)
    materialized, is_transient = await materialize_direct_voice_reference(
        voice_manager=manager,
        user_id=7,
        voice_reference=b"RIFF" + b"\x02" * 16,
        persist_direct_voice_references=False,
    )
    monkeypatch.undo()

    assert is_transient is True
    assert materialized.exists()

    cleanup_transient_voice_reference(materialized, is_transient)

    assert not materialized.exists()


def test_pocket_tts_cpp_prunes_provider_cache_for_ttl_and_oversize_files(tmp_path):
    from tldw_Server_API.app.core.TTS.adapters.pocket_tts_cpp_runtime import (
        prune_materialized_voice_cache,
    )

    runtime_dir = tmp_path / "voices" / "providers" / "pocket_tts_cpp"
    runtime_dir.mkdir(parents=True, exist_ok=True)

    expired = runtime_dir / "expired.wav"
    oldest = runtime_dir / "oldest.wav"
    newest = runtime_dir / "newest.wav"
    expired.write_bytes(b"x" * 4)
    oldest.write_bytes(b"y" * 7)
    newest.write_bytes(b"z" * 7)

    now = time.time()
    os.utime(expired, (now - 7200, now - 7200))
    os.utime(oldest, (now - 60, now - 60))
    os.utime(newest, (now - 5, now - 5))

    removed = prune_materialized_voice_cache(
        runtime_dir,
        cache_ttl_hours=1,
        cache_max_bytes=10,
    )

    removed_names = {path.name for path in removed}
    assert removed_names == {"expired.wav", "oldest.wav"}
    assert not expired.exists()
    assert not oldest.exists()
    assert newest.exists()


def test_pocket_tts_cpp_expired_cache_prune_failure_log_is_sanitized(tmp_path, monkeypatch):
    from tldw_Server_API.app.core.TTS.adapters import pocket_tts_cpp_runtime as runtime_module
    from tldw_Server_API.app.core.TTS.adapters.pocket_tts_cpp_runtime import (
        prune_materialized_voice_cache,
    )

    runtime_dir = tmp_path / "voices" / "providers" / "pocket_tts_cpp"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    raw_marker = "POCKETTTS_EXPIRED_SECRET_TOKEN"
    raw_exception_text = f"expired prune leaked /private/cache/{raw_marker}.wav"
    expired = runtime_dir / f"expired_{raw_marker}.wav"
    expired.write_bytes(b"x" * 4)
    old_time = time.time() - 7200
    os.utime(expired, (old_time, old_time))

    def _fail_unlink(self, *args, **kwargs):  # noqa: ARG001
        if self == expired:
            raise PermissionError(raw_exception_text)
        return original_unlink(self, *args, **kwargs)

    original_unlink = Path.unlink
    monkeypatch.setattr(Path, "unlink", _fail_unlink)
    logged_messages: list[str] = []
    sink_id = runtime_module.logger.add(
        lambda message: logged_messages.append(str(message.record["message"])),
        level="WARNING",
    )
    try:
        removed = prune_materialized_voice_cache(
            runtime_dir,
            cache_ttl_hours=1,
            cache_max_bytes=None,
        )
    finally:
        runtime_module.logger.remove(sink_id)

    rendered = "\n".join(logged_messages)
    assert removed == []
    assert expired.exists()
    assert "Failed pruning expired PocketTTS.cpp cache file" in rendered
    assert raw_marker not in rendered
    assert raw_exception_text not in rendered
    assert "PermissionError" in rendered


def test_pocket_tts_cpp_oversized_cache_prune_failure_log_is_sanitized(tmp_path, monkeypatch):
    from tldw_Server_API.app.core.TTS.adapters import pocket_tts_cpp_runtime as runtime_module
    from tldw_Server_API.app.core.TTS.adapters.pocket_tts_cpp_runtime import (
        prune_materialized_voice_cache,
    )

    runtime_dir = tmp_path / "voices" / "providers" / "pocket_tts_cpp"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    raw_marker = "POCKETTTS_OVERSIZED_SECRET_TOKEN"
    raw_exception_text = f"oversized prune leaked /private/cache/{raw_marker}.wav"
    oversized = runtime_dir / f"oversized_{raw_marker}.wav"
    oversized.write_bytes(b"x" * 16)

    def _fail_unlink(self, *args, **kwargs):  # noqa: ARG001
        if self == oversized:
            raise PermissionError(raw_exception_text)
        return original_unlink(self, *args, **kwargs)

    original_unlink = Path.unlink
    monkeypatch.setattr(Path, "unlink", _fail_unlink)
    logged_messages: list[str] = []
    sink_id = runtime_module.logger.add(
        lambda message: logged_messages.append(str(message.record["message"])),
        level="WARNING",
    )
    try:
        removed = prune_materialized_voice_cache(
            runtime_dir,
            cache_ttl_hours=None,
            cache_max_bytes=8,
        )
    finally:
        runtime_module.logger.remove(sink_id)

    rendered = "\n".join(logged_messages)
    assert removed == []
    assert oversized.exists()
    assert "Failed pruning oversized PocketTTS.cpp cache file" in rendered
    assert raw_marker not in rendered
    assert raw_exception_text not in rendered
    assert "PermissionError" in rendered


def test_pocket_tts_cpp_transient_cache_cleanup_failure_log_is_sanitized(tmp_path, monkeypatch):
    from tldw_Server_API.app.core.TTS.adapters import pocket_tts_cpp_runtime as runtime_module
    from tldw_Server_API.app.core.TTS.adapters.pocket_tts_cpp_runtime import (
        cleanup_transient_voice_reference,
    )

    runtime_dir = tmp_path / "voices" / "providers" / "pocket_tts_cpp"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    raw_marker = "POCKETTTS_TRANSIENT_SECRET_TOKEN"
    raw_exception_text = f"transient cleanup leaked /private/cache/{raw_marker}.wav"
    transient = runtime_dir / f"transient_{raw_marker}.wav"
    transient.write_bytes(b"x" * 4)

    def _fail_unlink(self, *args, **kwargs):  # noqa: ARG001
        if self == transient:
            raise PermissionError(raw_exception_text)
        return original_unlink(self, *args, **kwargs)

    original_unlink = Path.unlink
    monkeypatch.setattr(Path, "unlink", _fail_unlink)
    logged_messages: list[str] = []
    sink_id = runtime_module.logger.add(
        lambda message: logged_messages.append(str(message.record["message"])),
        level="WARNING",
    )
    try:
        cleanup_transient_voice_reference(transient, is_transient=True)
    finally:
        runtime_module.logger.remove(sink_id)

    rendered = "\n".join(logged_messages)
    assert transient.exists()
    assert "Failed cleaning transient PocketTTS.cpp cache file" in rendered
    assert raw_marker not in rendered
    assert raw_exception_text not in rendered
    assert "PermissionError" in rendered


def test_pocket_tts_cpp_prune_keeps_active_registered_voice_path(tmp_path):
    from tldw_Server_API.app.core.TTS.adapters.pocket_tts_cpp_runtime import (
        PROVIDER_MANAGED_VOICE_LEASE_DIRNAME,
        prune_materialized_voice_cache,
        resolve_provider_managed_voice_path,
        register_provider_managed_voice_path,
    )
    from tldw_Server_API.app.core.TTS.adapters import pocket_tts_cpp_runtime as runtime_module

    runtime_dir = tmp_path / "voices" / "providers" / "pocket_tts_cpp"
    runtime_dir.mkdir(parents=True, exist_ok=True)

    active = runtime_dir / "active.wav"
    stale = runtime_dir / "stale.wav"
    active.write_bytes(b"a" * 7)
    stale.write_bytes(b"b" * 7)

    now = time.time()
    os.utime(active, (now - 120, now - 120))
    os.utime(stale, (now - 60, now - 60))

    token = register_provider_managed_voice_path(active)
    runtime_module._PROVIDER_MANAGED_VOICE_TOKENS.clear()

    assert resolve_provider_managed_voice_path(token, active) == active.resolve()

    runtime_module._PROVIDER_MANAGED_VOICE_TOKENS.clear()
    removed = prune_materialized_voice_cache(
        runtime_dir,
        cache_ttl_hours=None,
        cache_max_bytes=8,
    )

    removed_names = {path.name for path in removed}
    assert removed_names == {"stale.wav"}
    assert active.exists()
    assert not stale.exists()
    lease_files = list((runtime_dir / PROVIDER_MANAGED_VOICE_LEASE_DIRNAME).glob("*.json"))
    assert len(lease_files) == 1
    assert lease_files[0].stem == token


@pytest.mark.asyncio
async def test_pocket_tts_cpp_provider_cache_is_excluded_from_uploaded_voice_quota(tmp_path, monkeypatch):
    manager = VoiceManager()
    voices_root = tmp_path / "voices"
    processed_dir = voices_root / "processed"
    provider_dir = voices_root / "providers" / "pocket_tts_cpp"
    processed_dir.mkdir(parents=True, exist_ok=True)
    provider_dir.mkdir(parents=True, exist_ok=True)

    processed_file = processed_dir / "user_voice.wav"
    provider_file = provider_dir / "ref_cache.wav"
    processed_file.write_bytes(b"a" * 512)
    provider_file.write_bytes(b"b" * 4096)

    monkeypatch.setattr(manager, "get_user_voices_path", lambda user_id: voices_root, raising=False)
    monkeypatch.setitem(VOICE_RATE_LIMITS, "total_storage_mb", 0.001)
    monkeypatch.setitem(VOICE_RATE_LIMITS, "max_voices_per_user", 10)

    async def _fake_duration(_path: Path) -> float:
        return 1.0

    monkeypatch.setattr(manager, "_get_audio_duration", _fake_duration, raising=False)

    allowed, message = await manager.check_rate_limits(99)

    assert allowed is True
    assert message == ""
