import asyncio
import base64
import wave
from io import BytesIO
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from tldw_Server_API.app.api.v1.schemas.audio_schemas import OpenAISpeechRequest
from tldw_Server_API.app.core.TTS.adapters.base import AudioFormat, TTSRequest, TTSResponse
from tldw_Server_API.app.core.TTS.adapters.pocket_tts_cpp_runtime import (
    PROVIDER_MANAGED_VOICE_LEASE_DIRNAME,
    PROVIDER_MANAGED_VOICE_TOKEN_KEY,
    register_provider_managed_voice_path,
    resolve_provider_managed_voice_path,
)
from tldw_Server_API.app.core.TTS.tts_exceptions import TTSTimeoutError, TTSGenerationError, TTSValidationError
from tldw_Server_API.app.core.TTS.tts_service_v2 import TTSServiceV2
from tldw_Server_API.app.core.TTS.voice_manager import VoiceReferenceMetadata


def _make_wav_bytes(
    payload: bytes = b"\x00\x01" * 8,
    *,
    sample_rate: int = 24000,
    channels: int = 1,
    sample_width: int = 2,
) -> bytes:
    frame_width = max(channels * sample_width, 1)
    min_payload_len = sample_rate * frame_width
    if len(payload) < min_payload_len:
        repeats = (min_payload_len + len(payload) - 1) // len(payload)
        payload = (payload * repeats)[:min_payload_len]
    buffer = BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(payload)
    return buffer.getvalue()


def _make_wav_base64(payload: bytes = b"\x00\x01" * 8) -> str:
    return base64.b64encode(_make_wav_bytes(payload)).decode("ascii")


class _ServiceVoiceManager:
    def __init__(self, root: Path):
        self.root = root

    def get_user_voices_path(self, user_id: int) -> Path:
        assert user_id == 1
        self.root.mkdir(parents=True, exist_ok=True)
        return self.root

    async def load_voice_reference_audio(self, user_id: int, voice_id: str) -> bytes:
        assert user_id == 1
        assert voice_id == "voice-1"
        return _make_wav_bytes()

    async def load_reference_metadata(self, user_id: int, voice_id: str):
        assert user_id == 1
        assert voice_id == "voice-1"
        return VoiceReferenceMetadata(
            voice_id=voice_id,
            reference_text="stored reference text",
        )


@pytest.mark.asyncio
async def test_pocket_tts_cpp_service_injects_stable_path_for_custom_voice(tmp_path, monkeypatch):
    service = TTSServiceV2()
    manager = _ServiceVoiceManager(tmp_path / "voices")
    from tldw_Server_API.app.core.TTS.adapters import pocket_tts_cpp_runtime as runtime_module

    source_wav = _make_wav_bytes()
    converted_wav = _make_wav_bytes(b"\x00\x01" * 8)

    monkeypatch.setattr(
        "tldw_Server_API.app.core.TTS.voice_manager.get_voice_manager",
        lambda: manager,
        raising=True,
    )

    async def _fake_convert(input_path: Path, output_path: Path, sample_rate: int, channels: int, bit_depth: int) -> bool:
        assert input_path.read_bytes() == source_wav
        output_path.write_bytes(converted_wav)
        return True

    monkeypatch.setattr(
        runtime_module.AudioConverter,
        "convert_to_wav",
        _fake_convert,
        raising=True,
    )

    request = TTSRequest(
        text="hello",
        voice="custom:voice-1",
        format=AudioFormat.WAV,
        extra_params={},
    )

    await service._apply_custom_voice_reference(request, user_id=1, provider_hint="pocket_tts_cpp")

    expected = tmp_path / "voices" / "providers" / "pocket_tts_cpp" / "custom_voice-1.wav"
    assert request.voice_reference == source_wav
    assert request.extra_params["pocket_tts_cpp_voice_path"] == str(expected)
    assert request.extra_params[PROVIDER_MANAGED_VOICE_TOKEN_KEY]
    assert request.extra_params["pocket_tts_cpp_reference_text"] == "stored reference text"
    assert expected.exists()
    assert expected.read_bytes() == converted_wav


@pytest.mark.asyncio
async def test_pocket_tts_cpp_service_injects_stable_path_for_direct_voice_reference(tmp_path):
    service = TTSServiceV2()
    request = TTSRequest(
        text="hello",
        voice="alloy",
        format=AudioFormat.WAV,
        voice_reference=_make_wav_bytes(b"\x01\x02" * 8),
        extra_params={"reference_text": "direct reference text"},
    )

    class _DirectReferenceVoiceManager:
        def get_user_voices_path(self, user_id: int) -> Path:
            assert user_id == 1
            root = tmp_path / "voices"
            root.mkdir(parents=True, exist_ok=True)
            return root

        async def load_reference_metadata(self, user_id: int, voice_id: str):
            return None

    from tldw_Server_API.app.core.TTS import voice_manager as voice_manager_module
    from tldw_Server_API.app.core.TTS.adapters import pocket_tts_cpp_runtime as runtime_module

    converted_wav = _make_wav_bytes(b"\x01\x02" * 8)

    async def _fake_convert(input_path: Path, output_path: Path, sample_rate: int, channels: int, bit_depth: int) -> bool:
        output_path.write_bytes(converted_wav)
        return True

    original_get_voice_manager = voice_manager_module.get_voice_manager
    voice_manager_module.get_voice_manager = lambda: _DirectReferenceVoiceManager()
    original_convert = runtime_module.AudioConverter.convert_to_wav
    runtime_module.AudioConverter.convert_to_wav = _fake_convert
    try:
        await service._apply_custom_voice_reference(request, user_id=1, provider_hint="pocket_tts_cpp")
    finally:
        voice_manager_module.get_voice_manager = original_get_voice_manager
        runtime_module.AudioConverter.convert_to_wav = original_convert

    voice_path = Path(request.extra_params["pocket_tts_cpp_voice_path"])
    assert voice_path.parent == tmp_path / "voices" / "providers" / "pocket_tts_cpp"
    assert voice_path.name.startswith("ref_")
    assert voice_path.suffix == ".wav"
    assert request.extra_params[PROVIDER_MANAGED_VOICE_TOKEN_KEY]
    assert request.extra_params["pocket_tts_cpp_reference_text"] == "direct reference text"
    assert voice_path.exists()


@pytest.mark.asyncio
async def test_pocket_tts_cpp_invalid_direct_voice_reference_is_rejected_before_materialization(monkeypatch):
    service = TTSServiceV2()
    request = OpenAISpeechRequest(
        model="pocket_tts_cpp",
        input="hello",
        voice="alloy",
        response_format="wav",
        stream=False,
    )
    tts_request = TTSRequest(
        text="hello",
        voice="alloy",
        format=AudioFormat.WAV,
        voice_reference=b"not-audio",
        extra_params={},
    )
    materialization_calls = 0

    async def _unexpected_materialization(*args, **kwargs):
        nonlocal materialization_calls
        materialization_calls += 1
        raise AssertionError("invalid voice_reference should be rejected before materialization")

    monkeypatch.setattr(service, "_apply_custom_voice_reference", _unexpected_materialization, raising=True)

    with pytest.raises(TTSValidationError, match="Voice reference file is not a valid audio format"):
        await service._prepare_generate_speech_request(
            request=request,
            tts_request=tts_request,
            provider="pocket_tts_cpp",
            provider_hint="pocket_tts_cpp",
            provider_overrides=None,
            fallback=False,
            user_id=1,
        )

    assert materialization_calls == 0


@pytest.mark.asyncio
async def test_pocket_tts_cpp_service_cleans_trust_token_after_generation(tmp_path):
    service = TTSServiceV2()
    seen_tokens: list[str] = []
    from tldw_Server_API.app.core.TTS.adapters import pocket_tts_cpp_runtime as runtime_module

    converted_wav = _make_wav_bytes(b"\x00\x01" * 8)

    class _FakeVoiceManager:
        def get_user_voices_path(self, user_id: int) -> Path:
            assert user_id == 1
            root = tmp_path / "voices"
            root.mkdir(parents=True, exist_ok=True)
            return root

        async def load_voice_reference_audio(self, user_id: int, voice_id: str) -> bytes:
            assert user_id == 1
            assert voice_id == "voice-1"
            return _make_wav_bytes()

        async def load_reference_metadata(self, user_id: int, voice_id: str):
            return None

    class _FakeAdapter:
        provider_name = "pocket_tts_cpp"
        provider_key = "pocket_tts_cpp"

        async def generate(self, request):
            token = request.extra_params[PROVIDER_MANAGED_VOICE_TOKEN_KEY]
            seen_tokens.append(token)
            voice_path = Path(request.extra_params["pocket_tts_cpp_voice_path"])
            assert resolve_provider_managed_voice_path(token, voice_path) == voice_path.resolve()
            return TTSResponse(audio_data=b"ok", format=request.format, sample_rate=24000)

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        "tldw_Server_API.app.core.TTS.voice_manager.get_voice_manager",
        lambda: _FakeVoiceManager(),
        raising=True,
    )

    async def _fake_convert(input_path: Path, output_path: Path, sample_rate: int, channels: int, bit_depth: int) -> bool:
        output_path.write_bytes(converted_wav)
        return True

    monkeypatch.setattr(
        runtime_module.AudioConverter,
        "convert_to_wav",
        _fake_convert,
        raising=True,
    )

    class _Factory:
        def get_provider_for_model(self, _model):
            return "pocket_tts_cpp"

    service._ensure_factory = AsyncMock(return_value=_Factory())
    service._get_adapter = AsyncMock(return_value=_FakeAdapter())

    request = OpenAISpeechRequest(
        model="pocket_tts_cpp",
        input="hello",
        voice="custom:voice-1",
        response_format="wav",
        stream=False,
    )

    try:
        chunks = []
        async for chunk in service.generate_speech(request, user_id=1, fallback=False):
            chunks.append(chunk)
    finally:
        monkeypatch.undo()

    assert b"".join(chunks) == b"ok"
    assert seen_tokens
    with pytest.raises(ValueError):
        resolve_provider_managed_voice_path(
            seen_tokens[0],
            Path(tmp_path / "voices" / "providers" / "pocket_tts_cpp" / "custom_voice-1.wav"),
        )


@pytest.mark.asyncio
async def test_generate_speech_closes_stream_before_pocket_tts_cpp_cleanup(tmp_path, monkeypatch):
    service = TTSServiceV2()
    events: list[tuple[str, bool] | str] = []
    stream_closed = False

    voice_path = tmp_path / "voices" / "providers" / "pocket_tts_cpp" / "custom_voice-stream.wav"
    voice_path.parent.mkdir(parents=True, exist_ok=True)
    voice_path.write_bytes(_make_wav_bytes())
    token = register_provider_managed_voice_path(voice_path)
    provider_request = TTSRequest(
        text="hello",
        voice="custom:voice-stream",
        format=AudioFormat.WAV,
        stream=True,
        extra_params={
            "pocket_tts_cpp_voice_path": str(voice_path),
            PROVIDER_MANAGED_VOICE_TOKEN_KEY: token,
            "_pocket_tts_cpp_transient_voice_path": str(voice_path),
        },
    )

    class _StreamingAdapter:
        provider_name = "pocket_tts_cpp"
        provider_key = "pocket_tts_cpp"

        async def generate(self, request):  # noqa: ARG002
            async def _stream():
                nonlocal stream_closed
                try:
                    yield b"chunk-1"
                    yield b"chunk-2"
                finally:
                    stream_closed = True
                    events.append("stream_closed")

            return TTSResponse(audio_stream=_stream(), format=AudioFormat.WAV, sample_rate=24000)

    service._prepare_generate_speech_request = AsyncMock(
        return_value=(_StreamingAdapter(), "pocket_tts_cpp", provider_request)
    )

    def _record_cleanup(request):
        assert request is provider_request
        events.append(("cleanup", stream_closed))

    monkeypatch.setattr(
        service,
        "_cleanup_transient_pocket_tts_cpp_voice_path",
        _record_cleanup,
        raising=True,
    )

    request = OpenAISpeechRequest(
        model="pocket_tts_cpp",
        input="hello",
        voice="custom:voice-stream",
        response_format="wav",
        stream=True,
    )

    generator = service.generate_speech(request, user_id=1, fallback=False)

    assert await anext(generator) == b"chunk-1"
    await generator.aclose()

    assert events == ["stream_closed", ("cleanup", True)]


@pytest.mark.asyncio
async def test_generate_with_adapter_closes_stream_before_pocket_tts_cpp_cleanup(tmp_path, monkeypatch):
    service = TTSServiceV2()
    events: list[tuple[str, bool] | str] = []
    stream_closed = False

    voice_path = tmp_path / "voices" / "providers" / "pocket_tts_cpp" / "custom_voice-helper.wav"
    voice_path.parent.mkdir(parents=True, exist_ok=True)
    voice_path.write_bytes(_make_wav_bytes())
    token = register_provider_managed_voice_path(voice_path)
    request = TTSRequest(
        text="hello",
        voice="custom:voice-helper",
        format=AudioFormat.WAV,
        stream=True,
        extra_params={
            "pocket_tts_cpp_voice_path": str(voice_path),
            PROVIDER_MANAGED_VOICE_TOKEN_KEY: token,
            "_pocket_tts_cpp_transient_voice_path": str(voice_path),
        },
    )

    class _StreamingAdapter:
        provider_name = "pocket_tts_cpp"
        provider_key = "pocket_tts_cpp"

        async def generate(self, stream_request):  # noqa: ARG002
            async def _stream():
                nonlocal stream_closed
                try:
                    yield b"chunk-1"
                    yield b"chunk-2"
                finally:
                    stream_closed = True
                    events.append("stream_closed")

            return TTSResponse(audio_stream=_stream(), format=AudioFormat.WAV, sample_rate=24000)

    def _record_cleanup(cleanup_request):
        assert cleanup_request is request
        events.append(("cleanup", stream_closed))

    monkeypatch.setattr(
        service,
        "_cleanup_transient_pocket_tts_cpp_voice_path",
        _record_cleanup,
        raising=True,
    )

    generator = service._generate_with_adapter(_StreamingAdapter(), request, user_id=1)

    assert await anext(generator) == b"chunk-1"
    await generator.aclose()

    assert events == ["stream_closed", ("cleanup", True)]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "voice,voice_reference,expected_fragment",
    [
        ("custom:voice-1", None, "custom_voice-1.wav"),
        ("alloy", _make_wav_base64(b"\x01\x02" * 8), "ref_"),
    ],
)
async def test_pocket_tts_cpp_fallback_materializes_voice_for_fallback_adapter(
    tmp_path,
    monkeypatch,
    voice,
    voice_reference,
    expected_fragment,
):
    service = TTSServiceV2()
    seen: dict[str, str] = {}
    converted_wav = _make_wav_bytes(b"\x01\x02" * 8)

    class _FailingAdapter:
        provider_name = "openai"
        provider_key = "openai"

        async def generate(self, request):
            raise TTSTimeoutError("primary provider failed", provider="openai")

    class _FallbackFactory:
        def get_provider_for_model(self, _model):
            return "openai"

    class _FallbackAdapter:
        provider_name = "pocket_tts_cpp"
        provider_key = "pocket_tts_cpp"

        async def generate(self, request):
            token = request.extra_params[PROVIDER_MANAGED_VOICE_TOKEN_KEY]
            voice_path = Path(request.extra_params["pocket_tts_cpp_voice_path"])
            seen["token"] = token
            seen["voice_path"] = str(voice_path)
            assert resolve_provider_managed_voice_path(token, voice_path) == voice_path.resolve()
            return TTSResponse(audio_data=b"fallback", format=request.format, sample_rate=24000)

    monkeypatch.setattr(
        "tldw_Server_API.app.core.TTS.voice_manager.get_voice_manager",
        lambda: _ServiceVoiceManager(tmp_path / "voices"),
        raising=True,
    )

    async def _fake_convert(input_path: Path, output_path: Path, sample_rate: int, channels: int, bit_depth: int) -> bool:
        output_path.write_bytes(converted_wav)
        return True

    monkeypatch.setattr(
        "tldw_Server_API.app.core.TTS.adapters.pocket_tts_cpp_runtime.AudioConverter.convert_to_wav",
        _fake_convert,
        raising=True,
    )

    service._ensure_factory = AsyncMock(return_value=_FallbackFactory())
    service._get_adapter = AsyncMock(return_value=_FailingAdapter())
    service._get_fallback_adapter = AsyncMock(return_value=_FallbackAdapter())

    request = OpenAISpeechRequest(
        model="openai",
        input="hello",
        voice=voice,
        response_format="wav",
        stream=False,
        voice_reference=voice_reference,
        extra_params={"reference_text": "fallback reference text"},
    )

    chunks = []
    async for chunk in service.generate_speech(request, user_id=1, fallback=True):
        chunks.append(chunk)

    assert b"".join(chunks) == b"fallback"
    voice_path = Path(seen["voice_path"])
    assert voice_path.parent.parts[-3:] == ("voices", "providers", "pocket_tts_cpp")
    assert expected_fragment in voice_path.name
    with pytest.raises(ValueError):
        resolve_provider_managed_voice_path(seen["token"], voice_path)


@pytest.mark.asyncio
async def test_pocket_tts_cpp_fallback_validation_failure_revokes_lease_but_keeps_stable_materialized_voice(
    tmp_path,
    monkeypatch,
):
    service = TTSServiceV2()
    seen: dict[str, str] = {}
    converted_wav = _make_wav_bytes(b"\x05\x06" * 8)

    class _PocketAdapter:
        provider_name = "pocket_tts_cpp"
        provider_key = "pocket_tts_cpp"

        async def generate(self, request):  # noqa: ARG002
            raise AssertionError("adapter.generate should not be reached on validation failure")

    async def _fake_convert(input_path: Path, output_path: Path, sample_rate: int, channels: int, bit_depth: int) -> bool:
        output_path.write_bytes(converted_wav)
        return True

    def _raise_validation(request, provider_key):  # noqa: ARG001
        extras = request.extra_params if isinstance(request.extra_params, dict) else {}
        seen["token"] = extras.get(PROVIDER_MANAGED_VOICE_TOKEN_KEY)
        seen["voice_path"] = extras.get("pocket_tts_cpp_voice_path")
        raise TTSValidationError("synthetic validation failure", provider="pocket_tts_cpp")

    monkeypatch.setattr(
        "tldw_Server_API.app.core.TTS.voice_manager.get_voice_manager",
        lambda: _ServiceVoiceManager(tmp_path / "voices"),
        raising=True,
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.TTS.adapters.pocket_tts_cpp_runtime.AudioConverter.convert_to_wav",
        _fake_convert,
        raising=True,
    )
    monkeypatch.setattr(service, "_maybe_sanitize_request", _raise_validation, raising=True)

    request = TTSRequest(
        text="hello",
        voice="custom:voice-1",
        format=AudioFormat.WAV,
        stream=False,
        extra_params={"reference_text": "fallback reference text"},
    )

    with pytest.raises(TTSValidationError):
        async for _ in service._generate_with_adapter(_PocketAdapter(), request, user_id=1):
            pass

    runtime_dir = tmp_path / "voices" / "providers" / "pocket_tts_cpp"
    assert seen["token"]
    assert seen["voice_path"].endswith("custom_voice-1.wav")
    assert (runtime_dir / "custom_voice-1.wav").exists()
    assert not list((runtime_dir / PROVIDER_MANAGED_VOICE_LEASE_DIRNAME).glob("*.json"))
    with pytest.raises(ValueError):
        resolve_provider_managed_voice_path(seen["token"], Path(seen["voice_path"]))


@pytest.mark.asyncio
async def test_pocket_tts_cpp_validation_failure_cleans_transient_direct_reference(tmp_path, monkeypatch):
    service = TTSServiceV2()

    class _DirectReferenceVoiceManager:
        def get_user_voices_path(self, user_id: int) -> Path:
            assert user_id == 1
            root = tmp_path / "voices"
            root.mkdir(parents=True, exist_ok=True)
            return root

        async def load_reference_metadata(self, user_id: int, voice_id: str):
            return None

    monkeypatch.setattr(
        "tldw_Server_API.app.core.TTS.voice_manager.get_voice_manager",
        lambda: _DirectReferenceVoiceManager(),
        raising=True,
    )

    class _FactoryWithPocketRuntimeConfig:
        registry = type(
            "_Registry",
            (),
            {
                "config": {
                    "providers": {
                        "pocket_tts_cpp": {
                            "persist_direct_voice_references": False,
                            "cache_ttl_hours": None,
                            "cache_max_bytes_per_user": None,
                        }
                    }
                }
            },
        )()

        def get_provider_for_model(self, _model):
            return "pocket_tts_cpp"

    service._ensure_factory = AsyncMock(return_value=_FactoryWithPocketRuntimeConfig())
    service._get_adapter = AsyncMock()
    converted_wav = _make_wav_bytes(b"\x02\x03" * 8)

    async def _fake_convert(input_path: Path, output_path: Path, sample_rate: int, channels: int, bit_depth: int) -> bool:
        output_path.write_bytes(converted_wav)
        return True

    def _raise_validation(*args, **kwargs):
        raise TTSValidationError("synthetic validation failure", provider="pocket_tts_cpp")

    monkeypatch.setattr(
        "tldw_Server_API.app.core.TTS.tts_service_v2.validate_tts_request",
        _raise_validation,
        raising=True,
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.TTS.adapters.pocket_tts_cpp_runtime.AudioConverter.convert_to_wav",
        _fake_convert,
        raising=True,
    )

    request = OpenAISpeechRequest(
        model="pocket_tts_cpp",
        input="hello",
        voice="alloy",
        response_format="wav",
        stream=False,
        voice_reference=_make_wav_base64(b"\x02\x03" * 8),
        extra_params={"reference_text": "direct reference text"},
    )

    with pytest.raises(TTSValidationError):
        async for _ in service.generate_speech(request, user_id=1, fallback=False):
            pass

    runtime_dir = tmp_path / "voices" / "providers" / "pocket_tts_cpp"
    assert list(runtime_dir.glob("ref_*.wav")) == []
    service._get_adapter.assert_not_called()


@pytest.mark.asyncio
async def test_pocket_tts_cpp_cancellation_during_adapter_acquisition_cleans_transient_direct_reference(
    tmp_path, monkeypatch
):
    service = TTSServiceV2()

    class _DirectReferenceVoiceManager:
        def get_user_voices_path(self, user_id: int) -> Path:
            assert user_id == 1
            root = tmp_path / "voices"
            root.mkdir(parents=True, exist_ok=True)
            return root

        async def load_reference_metadata(self, user_id: int, voice_id: str):
            return None

    monkeypatch.setattr(
        "tldw_Server_API.app.core.TTS.voice_manager.get_voice_manager",
        lambda: _DirectReferenceVoiceManager(),
        raising=True,
    )

    class _FactoryWithPocketRuntimeConfig:
        registry = type(
            "_Registry",
            (),
            {
                "config": {
                    "providers": {
                        "pocket_tts_cpp": {
                            "persist_direct_voice_references": False,
                            "cache_ttl_hours": None,
                            "cache_max_bytes_per_user": None,
                        }
                    }
                }
            },
        )()

        def get_provider_for_model(self, _model):
            return "pocket_tts_cpp"

    service._ensure_factory = AsyncMock(return_value=_FactoryWithPocketRuntimeConfig())
    converted_wav = _make_wav_bytes(b"\x04\x05" * 8)

    async def _fake_convert(input_path: Path, output_path: Path, sample_rate: int, channels: int, bit_depth: int) -> bool:
        output_path.write_bytes(converted_wav)
        return True

    async def _cancel_adapter(*args, **kwargs):
        raise asyncio.CancelledError()

    service._get_adapter = AsyncMock(side_effect=_cancel_adapter)
    monkeypatch.setattr(
        "tldw_Server_API.app.core.TTS.adapters.pocket_tts_cpp_runtime.AudioConverter.convert_to_wav",
        _fake_convert,
        raising=True,
    )

    request = OpenAISpeechRequest(
        model="pocket_tts_cpp",
        input="hello",
        voice="alloy",
        response_format="wav",
        stream=False,
        voice_reference=_make_wav_base64(b"\x04\x05" * 8),
        extra_params={"reference_text": "direct reference text"},
    )

    with pytest.raises(asyncio.CancelledError):
        async for _ in service.generate_speech(request, user_id=1, fallback=False):
            pass

    runtime_dir = tmp_path / "voices" / "providers" / "pocket_tts_cpp"
    assert list(runtime_dir.glob("ref_*.wav")) == []


@pytest.mark.asyncio
async def test_pocket_tts_cpp_materialization_failure_surfaces_explicit_error(tmp_path, monkeypatch):
    service = TTSServiceV2()

    class _DirectReferenceVoiceManager:
        def get_user_voices_path(self, user_id: int) -> Path:
            assert user_id == 1
            root = tmp_path / "voices"
            root.mkdir(parents=True, exist_ok=True)
            return root

        async def load_reference_metadata(self, user_id: int, voice_id: str):
            return None

    monkeypatch.setattr(
        "tldw_Server_API.app.core.TTS.voice_manager.get_voice_manager",
        lambda: _DirectReferenceVoiceManager(),
        raising=True,
    )

    class _FactoryWithPocketRuntimeConfig:
        registry = type(
            "_Registry",
            (),
            {
                "config": {
                    "providers": {
                        "pocket_tts_cpp": {
                            "persist_direct_voice_references": False,
                            "cache_ttl_hours": None,
                            "cache_max_bytes_per_user": None,
                        }
                    }
                }
            },
        )()

        def get_provider_for_model(self, _model):
            return "pocket_tts_cpp"

    service._ensure_factory = AsyncMock(return_value=_FactoryWithPocketRuntimeConfig())
    service._get_adapter = AsyncMock()

    async def _raise_convert(input_path: Path, output_path: Path, sample_rate: int, channels: int, bit_depth: int) -> bool:
        raise RuntimeError("ffmpeg missing")

    monkeypatch.setattr(
        "tldw_Server_API.app.core.TTS.adapters.pocket_tts_cpp_runtime.AudioConverter.convert_to_wav",
        _raise_convert,
        raising=True,
    )

    request = OpenAISpeechRequest(
        model="pocket_tts_cpp",
        input="hello",
        voice="alloy",
        response_format="wav",
        stream=False,
        voice_reference=_make_wav_base64(b"\x06\x07" * 8),
        extra_params={"reference_text": "direct reference text"},
    )

    with pytest.raises(TTSGenerationError) as exc_info:
        async for _ in service.generate_speech(request, user_id=1, fallback=False):
            pass

    assert exc_info.value.provider == "pocket_tts_cpp"
    assert exc_info.value.error_code == "pocket_tts_cpp_voice_materialization_failed"
    assert "ffmpeg missing" in exc_info.value.details["reason"]
    service._get_adapter.assert_not_called()


@pytest.mark.asyncio
async def test_prepare_generate_speech_request_logs_noncritical_touch_model_failure(monkeypatch):
    service = TTSServiceV2()
    warnings: list[str] = []

    class _Adapter:
        provider_name = "pocket_tts_cpp"
        provider_key = "pocket_tts_cpp"

    class _ResourceManager:
        def touch_model(self, provider_key, model_name):
            raise RuntimeError(f"cache miss for {provider_key}:{model_name}")

    request = OpenAISpeechRequest(
        model="pocket_tts_cpp",
        input="hello",
        voice="custom:voice-1",
        response_format="wav",
        stream=False,
    )
    tts_request = TTSRequest(
        text="hello",
        voice="custom:voice-1",
        format=AudioFormat.WAV,
        extra_params={},
    )

    monkeypatch.setattr(service, "_apply_custom_voice_reference", AsyncMock(), raising=True)
    monkeypatch.setattr(service, "_get_adapter", AsyncMock(return_value=_Adapter()), raising=True)
    monkeypatch.setattr(
        "tldw_Server_API.app.core.TTS.tts_service_v2.get_resource_manager",
        AsyncMock(return_value=_ResourceManager()),
        raising=True,
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.TTS.tts_service_v2.logger.warning",
        lambda message, *args: warnings.append(message.format(*args)),
        raising=True,
    )

    await service._prepare_generate_speech_request(
        request=request,
        tts_request=tts_request,
        provider="pocket_tts_cpp",
        provider_hint="pocket_tts_cpp",
        provider_overrides=None,
        fallback=False,
        user_id=1,
    )

    assert warnings
    assert "touch_model" in warnings[0]
