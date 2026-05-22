import asyncio
import inspect
import wave
from io import BytesIO
from pathlib import Path
from typing import get_type_hints

import httpx
import pytest

from tldw_Server_API.app.core.TTS.adapters.base import AudioFormat, ProviderStatus, TTSRequest
from tldw_Server_API.app.core.TTS.adapters import omnivoice_adapter as omnivoice_adapter_module
from tldw_Server_API.app.core.TTS.adapters.omnivoice_adapter import OmniVoiceAdapter
from tldw_Server_API.app.core.TTS.adapters.omnivoice_sidecar_protocol import OmniVoiceSynthesizeRequest
from tldw_Server_API.app.core.TTS.tts_exceptions import TTSGenerationError


pytestmark = pytest.mark.unit


def _make_wav_bytes(
    payload: bytes = b"\x00\x01" * 16,
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


def _make_reference_wav(duration_seconds: float, *, sample_rate: int = 24000) -> bytes:
    frame_count = max(1, int(duration_seconds * sample_rate))
    return _make_wav_bytes(payload=b"\x00\x01" * frame_count, sample_rate=sample_rate)


class _FakeSupervisor:
    def __init__(self, base_url: str = "http://127.0.0.1:8039") -> None:
        self.base_url = base_url
        self.ensure_started_calls = 0

    async def ensure_started(self) -> str:
        self.ensure_started_calls += 1
        return self.base_url


class _FakeClient:
    def __init__(self, recorded: dict[str, object], response: httpx.Response) -> None:
        self.recorded = recorded
        self.response = response

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, url, *, json=None, headers=None):
        self.recorded["url"] = url
        self.recorded["json"] = json
        self.recorded["headers"] = dict(headers or {})
        return self.response


def test_omnivoice_adapter_documents_module_and_init_contract() -> None:
    assert omnivoice_adapter_module.__doc__
    assert inspect.signature(OmniVoiceAdapter.__init__).return_annotation != inspect.Signature.empty
    assert get_type_hints(OmniVoiceAdapter.__init__)["return"] is type(None)


@pytest.mark.asyncio
async def test_omnivoice_capabilities_do_not_advertise_streaming(tmp_path):
    adapter = OmniVoiceAdapter({"sample_rate": 24000, "timeout": 5})

    capabilities = await adapter.get_capabilities()

    assert capabilities.supports_streaming is False
    assert capabilities.supports_voice_cloning is True
    assert AudioFormat.WAV in capabilities.supported_formats
    assert AudioFormat.PCM in capabilities.supported_formats


@pytest.mark.asyncio
async def test_omnivoice_initialize_requires_attached_supervisor():
    adapter = OmniVoiceAdapter({"sample_rate": 24000, "timeout": 5})

    success = await adapter.initialize()

    assert success is False
    assert adapter.status == ProviderStatus.NOT_CONFIGURED


@pytest.mark.asyncio
async def test_omnivoice_generate_posts_narrow_internal_payload_and_returns_wav(monkeypatch):
    adapter = OmniVoiceAdapter({"sample_rate": 24000, "timeout": 5})
    adapter._initialized = True
    adapter._status = ProviderStatus.AVAILABLE
    adapter.set_supervisor(_FakeSupervisor())

    wav_bytes = _make_reference_wav(3.5)
    recorded: dict[str, object] = {}

    def _fake_client_factory(*, timeout: float):
        assert timeout == pytest.approx(5.0)  # nosec B101
        return _FakeClient(
            recorded,
            httpx.Response(
                200,
                content=wav_bytes,
                headers={
                    "X-OmniVoice-Audio-Format": "wav",
                    "X-OmniVoice-Sample-Rate": "24000",
                    "X-OmniVoice-Channels": "1",
                },
            ),
        )

    monkeypatch.setattr(
        "tldw_Server_API.app.core.TTS.adapters.omnivoice_adapter.create_sidecar_async_client",
        _fake_client_factory,
        raising=True,
    )

    request = TTSRequest(
        text="hello world",
        voice="auto",
        format=AudioFormat.WAV,
        stream=False,
        extra_params={"reference_text": "ignored for auto mode", "temperature": 0.3},
    )

    response = await adapter.generate(request)

    assert response.audio_data == wav_bytes
    assert response.format == AudioFormat.WAV
    assert recorded["url"] == "http://127.0.0.1:8039/v1/synthesize"
    parsed_payload = OmniVoiceSynthesizeRequest(**recorded["json"])
    assert recorded["json"] == {
        "text": "hello world",
        "mode": "auto",
        "voice": "auto",
        "requested_sample_rate": 24000,
    }
    assert parsed_payload.requested_sample_rate == 24000  # nosec B101


@pytest.mark.asyncio
async def test_omnivoice_clone_request_materializes_reference_audio_but_sends_narrow_payload(
    tmp_path, monkeypatch
):
    adapter = OmniVoiceAdapter({"sample_rate": 24000, "timeout": 5, "temp_dir": str(tmp_path)})
    adapter._initialized = True
    adapter._status = ProviderStatus.AVAILABLE
    adapter.set_supervisor(_FakeSupervisor())

    wav_bytes = _make_reference_wav(3.5)
    recorded: dict[str, object] = {}

    def _fake_client_factory(*, timeout: float):
        return _FakeClient(
            recorded,
            httpx.Response(
                200,
                content=wav_bytes,
                headers={
                    "X-OmniVoice-Audio-Format": "wav",
                    "X-OmniVoice-Sample-Rate": "24000",
                    "X-OmniVoice-Channels": "1",
                },
            ),
        )

    monkeypatch.setattr(
        "tldw_Server_API.app.core.TTS.adapters.omnivoice_adapter.create_sidecar_async_client",
        _fake_client_factory,
        raising=True,
    )

    request = TTSRequest(
        text="clone me",
        voice="clone",
        format=AudioFormat.PCM,
        stream=False,
        voice_reference=wav_bytes,
        extra_params={"reference_text": "reference transcript"},
    )

    response = await adapter.generate(request)

    transient_path = Path(recorded["json"]["reference_audio_path"])
    assert response.audio_data is not None
    assert response.format == AudioFormat.PCM
    assert response.metadata["used_reference_audio"] is True
    assert "reference_audio_path" not in response.metadata
    parsed_payload = OmniVoiceSynthesizeRequest(**recorded["json"])
    assert recorded["json"] == {
        "text": "clone me",
        "mode": "clone",
        "reference_audio_path": str(transient_path),
        "reference_text": "reference transcript",
        "requested_sample_rate": 24000,
    }
    assert parsed_payload.reference_audio_path == str(transient_path)  # nosec B101
    assert parsed_payload.requested_sample_rate == 24000  # nosec B101
    assert transient_path.parent == tmp_path
    assert transient_path.exists() is False


@pytest.mark.asyncio
async def test_omnivoice_clone_request_offloads_reference_audio_materialization_to_thread(
    tmp_path, monkeypatch
):
    adapter = OmniVoiceAdapter({"sample_rate": 24000, "timeout": 5, "temp_dir": str(tmp_path)})
    adapter._initialized = True
    adapter._status = ProviderStatus.AVAILABLE
    adapter.set_supervisor(_FakeSupervisor())

    wav_bytes = _make_reference_wav(3.5)
    recorded: dict[str, object] = {}
    to_thread_calls: list[str] = []

    async def _fake_to_thread(func, *args, **kwargs):
        to_thread_calls.append(getattr(func, "__name__", repr(func)))
        return func(*args, **kwargs)

    def _fake_client_factory(*, timeout: float):
        return _FakeClient(
            recorded,
            httpx.Response(
                200,
                content=wav_bytes,
                headers={
                    "X-OmniVoice-Audio-Format": "wav",
                    "X-OmniVoice-Sample-Rate": "24000",
                    "X-OmniVoice-Channels": "1",
                },
            ),
        )

    monkeypatch.setattr(asyncio, "to_thread", _fake_to_thread)
    monkeypatch.setattr(
        "tldw_Server_API.app.core.TTS.adapters.omnivoice_adapter.create_sidecar_async_client",
        _fake_client_factory,
        raising=True,
    )

    request = TTSRequest(
        text="clone me",
        voice="clone",
        format=AudioFormat.WAV,
        stream=False,
        voice_reference=wav_bytes,
        extra_params={"reference_text": "reference transcript"},
    )

    await adapter.generate(request)

    assert to_thread_calls  # nosec B101


@pytest.mark.asyncio
async def test_omnivoice_pcm_sidecar_response_preserves_reported_channel_count_when_wrapped_as_wav(
    monkeypatch,
):
    adapter = OmniVoiceAdapter({"sample_rate": 24000, "timeout": 5})
    adapter._initialized = True
    adapter._status = ProviderStatus.AVAILABLE
    adapter.set_supervisor(_FakeSupervisor())

    stereo_pcm = b"\x00\x01\x02\x03" * 32
    recorded: dict[str, object] = {}

    def _fake_client_factory(*, timeout: float):
        return _FakeClient(
            recorded,
            httpx.Response(
                200,
                content=stereo_pcm,
                headers={
                    "X-OmniVoice-Audio-Format": "pcm",
                    "X-OmniVoice-Sample-Rate": "24000",
                    "X-OmniVoice-Channels": "2",
                },
            ),
        )

    monkeypatch.setattr(
        "tldw_Server_API.app.core.TTS.adapters.omnivoice_adapter.create_sidecar_async_client",
        _fake_client_factory,
        raising=True,
    )

    request = TTSRequest(
        text="stereo please",
        voice="auto",
        format=AudioFormat.WAV,
        stream=False,
    )

    response = await adapter.generate(request)

    with wave.open(BytesIO(response.audio_data), "rb") as wav_file:
        assert wav_file.getnchannels() == 2  # nosec B101


@pytest.mark.asyncio
async def test_omnivoice_generate_transcodes_wav_to_requested_mp3(monkeypatch, tmp_path):
    adapter = OmniVoiceAdapter({"sample_rate": 24000, "timeout": 5, "temp_dir": str(tmp_path)})
    adapter._initialized = True
    adapter._status = ProviderStatus.AVAILABLE
    adapter.set_supervisor(_FakeSupervisor())

    wav_bytes = _make_reference_wav(3.5)
    converted_bytes = b"ID3-mock-mp3"

    async def _fake_to_thread(func, *args, **kwargs):
        return func(*args, **kwargs)

    async def _fake_convert_format(input_path, output_path, target_format, **kwargs):  # noqa: ARG001
        output_path.with_suffix(f".{target_format}").write_bytes(converted_bytes)
        return True

    def _fake_client_factory(*, timeout: float):
        return _FakeClient(
            {},
            httpx.Response(
                200,
                content=wav_bytes,
                headers={
                    "X-OmniVoice-Audio-Format": "wav",
                    "X-OmniVoice-Sample-Rate": "24000",
                    "X-OmniVoice-Channels": "1",
                },
            ),
        )

    monkeypatch.setattr(asyncio, "to_thread", _fake_to_thread)
    monkeypatch.setattr(
        "tldw_Server_API.app.core.TTS.adapters.omnivoice_adapter.AudioConverter.convert_format",
        _fake_convert_format,
        raising=True,
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.TTS.adapters.omnivoice_adapter.create_sidecar_async_client",
        _fake_client_factory,
        raising=True,
    )

    request = TTSRequest(
        text="convert me",
        voice="auto",
        format=AudioFormat.MP3,
        stream=False,
    )

    response = await adapter.generate(request)

    assert response.format == AudioFormat.MP3
    assert response.audio_data == converted_bytes


@pytest.mark.asyncio
async def test_omnivoice_sidecar_error_sanitizes_response_text(monkeypatch):
    adapter = OmniVoiceAdapter({"sample_rate": 24000, "timeout": 5})
    adapter._initialized = True
    adapter._status = ProviderStatus.AVAILABLE
    adapter.set_supervisor(_FakeSupervisor())

    warnings: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def _fake_warning(*args, **kwargs):
        warnings.append((args, kwargs))

    def _fake_client_factory(*, timeout: float):
        return _FakeClient(
            {},
            httpx.Response(
                500,
                text="Traceback: /Users/private/omnivoice_sidecar.py\nRuntimeError: boom",
            ),
        )

    monkeypatch.setattr(
        "tldw_Server_API.app.core.TTS.adapters.omnivoice_adapter.create_sidecar_async_client",
        _fake_client_factory,
        raising=True,
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.TTS.adapters.omnivoice_adapter.logger.warning",
        _fake_warning,
        raising=True,
    )

    request = TTSRequest(
        text="fail please",
        voice="auto",
        format=AudioFormat.WAV,
        stream=False,
    )

    with pytest.raises(TTSGenerationError) as exc_info:
        await adapter.generate(request)

    assert exc_info.value.details["response_text"] == (
        "OmniVoice sidecar reported an internal error; see server logs."
    )
    assert warnings
