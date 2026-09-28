from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.services import audiobook_jobs_worker, outputs_service


pytestmark = pytest.mark.unit

DEFAULT_KITTEN_TTS_MODEL = "KittenML/kitten-tts-nano-0.8"
DEFAULT_KITTEN_TTS_VOICE = "Bella"


class _RecordingTTSService:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.status: dict[str, object] = {
            "providers": {"pocket_tts_cpp": {"status": "not_initialized"}}
        }

    async def generate_speech(self, request, provider=None, fallback=None, user_id=None):
        self.calls.append(
            {
                "request": request,
                "provider": provider,
                "fallback": fallback,
                "user_id": user_id,
            }
        )
        yield b"tts-audio"

    def get_status(self) -> dict[str, object]:
        return self.status


def _patch_tts_service(monkeypatch: pytest.MonkeyPatch, service: _RecordingTTSService) -> None:
    async def _fake_get_tts_service_v2():
        return service

    monkeypatch.setattr(outputs_service, "get_tts_service_v2", _fake_get_tts_service_v2, raising=False)
    monkeypatch.setattr(audiobook_jobs_worker, "get_tts_service_v2", _fake_get_tts_service_v2, raising=False)


@pytest.mark.asyncio
async def test_outputs_service_defaults_to_kitten_model_and_voice_when_unset(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    service = _RecordingTTSService()
    _patch_tts_service(monkeypatch, service)

    output_path = tmp_path / "output.mp3"

    await outputs_service._write_tts_audio_file(
        rendered="Hello world",
        path=output_path,
        tts_model=None,
        tts_voice=None,
        tts_speed=None,
        template_row=None,
    )

    assert output_path.read_bytes() == b"tts-audio"
    assert len(service.calls) == 1
    request = service.calls[0]["request"]
    assert getattr(request, "model", None) == DEFAULT_KITTEN_TTS_MODEL
    assert getattr(request, "voice", None) == DEFAULT_KITTEN_TTS_VOICE


@pytest.mark.asyncio
async def test_outputs_service_falls_back_to_kitten_when_template_pocket_tts_cpp_is_unready(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    service = _RecordingTTSService()
    service.status = {"providers": {"pocket_tts_cpp": {"status": "not_initialized"}}}
    _patch_tts_service(monkeypatch, service)

    output_path = tmp_path / "template-pocket.mp3"
    template_row = SimpleNamespace(
        metadata_json=json.dumps(
            {
                "tts_default_model": "pocket_tts_cpp",
                "tts_default_voice": "stored_voice",
            }
        )
    )

    await outputs_service._write_tts_audio_file(
        rendered="Hello world",
        path=output_path,
        tts_model=None,
        tts_voice=None,
        tts_speed=None,
        template_row=template_row,
    )

    request = service.calls[0]["request"]
    assert getattr(request, "model", None) == DEFAULT_KITTEN_TTS_MODEL
    assert getattr(request, "voice", None) == DEFAULT_KITTEN_TTS_VOICE


def test_outputs_service_infers_kitten_provider_from_kittenml_alias() -> None:
    assert outputs_service._infer_output_tts_provider_from_model("KittenML/kitten-tts-nano-0.8") == "kitten_tts"


@pytest.mark.asyncio
async def test_outputs_service_preserves_explicit_template_and_request_overrides(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    service = _RecordingTTSService()
    _patch_tts_service(monkeypatch, service)

    output_path = tmp_path / "explicit.mp3"
    template_row = SimpleNamespace(
        metadata_json=json.dumps(
            {
                "tts_default_model": DEFAULT_KITTEN_TTS_MODEL,
                "tts_default_voice": DEFAULT_KITTEN_TTS_VOICE,
                "tts_default_speed": 1.25,
            }
        )
    )

    await outputs_service._write_tts_audio_file(
        rendered="Hello world",
        path=output_path,
        tts_model="pocket_tts_cpp",
        tts_voice="custom_voice",
        tts_speed=0.9,
        template_row=template_row,
    )

    assert output_path.read_bytes() == b"tts-audio"
    request = service.calls[0]["request"]
    assert getattr(request, "model", None) == "pocket_tts_cpp"
    assert getattr(request, "voice", None) == "custom_voice"
    assert getattr(request, "speed", None) == 0.9


def test_validate_text_defaults_provider_hint_to_kitten_tts_when_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, str | None] = {}

    class _FakeValidator:
        def __init__(self, _config: dict[str, object]) -> None:
            return

        def sanitize_text(self, text: str, provider: str | None = None) -> str:
            captured["provider"] = provider
            return text

    monkeypatch.setattr(audiobook_jobs_worker, "TTSInputValidator", _FakeValidator)

    result = audiobook_jobs_worker._validate_text("Hello world", provider=None, model=None)

    assert result == "Hello world"
    assert captured["provider"] == "kitten_tts"


def test_resolve_item_requests_without_provider_or_model_does_not_require_subtitles() -> None:
    payload = {
        "source": {"input_type": "txt", "raw_text": "Hello world."},
        "output": {"formats": ["mp3"], "merge": False, "per_chapter": True},
    }

    resolved = audiobook_jobs_worker._resolve_item_requests(payload)

    assert len(resolved) == 1
    assert resolved[0]["tts_provider"] is None
    assert resolved[0]["tts_model"] is None


def test_is_kokoro_request_requires_explicit_signal() -> None:
    assert audiobook_jobs_worker._is_kokoro_request(None, None) is False
    assert audiobook_jobs_worker._is_kokoro_request("kokoro", None) is True
    assert audiobook_jobs_worker._is_kokoro_request(None, "kokoro") is True


@pytest.mark.asyncio
async def test_generate_tts_audio_defaults_to_kitten_provider_model_and_voice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _RecordingTTSService()
    _patch_tts_service(monkeypatch, service)

    audio_bytes, alignment, _route = await audiobook_jobs_worker._generate_tts_audio(
        text="Hello world",
        backend=None,
        allow_fallback=None,
        model=None,
        provider=None,
        voice=None,
        speed=None,
        response_format="mp3",
        user_id=17,
    )

    assert audio_bytes == b"tts-audio"
    assert alignment is None
    assert len(service.calls) == 1
    call = service.calls[0]
    request = call["request"]
    assert call["provider"] == "kitten_tts"
    assert getattr(request, "model", None) == DEFAULT_KITTEN_TTS_MODEL
    assert getattr(request, "voice", None) == DEFAULT_KITTEN_TTS_VOICE


@pytest.mark.asyncio
async def test_generate_tts_audio_preserves_explicit_provider_model_and_voice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _RecordingTTSService()
    _patch_tts_service(monkeypatch, service)

    audio_bytes, alignment, _route = await audiobook_jobs_worker._generate_tts_audio(
        text="Hello world",
        backend=None,
        allow_fallback=None,
        model="pocket_tts_cpp",
        provider="pocket_tts_cpp",
        voice="custom_voice",
        speed=1.1,
        response_format="mp3",
        user_id=17,
    )

    assert audio_bytes == b"tts-audio"
    assert alignment is None
    call = service.calls[0]
    request = call["request"]
    assert call["provider"] == "pocket_tts_cpp"
    assert getattr(request, "model", None) == "pocket_tts_cpp"
    assert getattr(request, "voice", None) == "custom_voice"
    assert getattr(request, "speed", None) == 1.1


@pytest.mark.asyncio
async def test_generate_tts_audio_infers_pocket_tts_cpp_from_model_only_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _RecordingTTSService()
    _patch_tts_service(monkeypatch, service)

    audio_bytes, alignment, _route = await audiobook_jobs_worker._generate_tts_audio(
        text="Hello world",
        backend=None,
        allow_fallback=None,
        model="pocket_tts_cpp",
        provider=None,
        voice=None,
        speed=None,
        response_format="mp3",
        user_id=17,
    )

    assert audio_bytes == b"tts-audio"
    assert alignment is None
    call = service.calls[0]
    request = call["request"]
    assert call["provider"] == "pocket_tts_cpp"
    assert getattr(request, "model", None) == "pocket_tts_cpp"


@pytest.mark.asyncio
async def test_generate_tts_audio_uses_openai_default_voice_for_explicit_openai_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _RecordingTTSService()
    _patch_tts_service(monkeypatch, service)

    audio_bytes, alignment, _route = await audiobook_jobs_worker._generate_tts_audio(
        text="Hello world",
        backend=None,
        allow_fallback=None,
        model=None,
        provider="openai",
        voice=None,
        speed=None,
        response_format="mp3",
        user_id=17,
    )

    assert audio_bytes == b"tts-audio"
    assert alignment is None
    call = service.calls[0]
    request = call["request"]
    assert call["provider"] == "openai"
    assert getattr(request, "model", None) == "tts-1"
    assert getattr(request, "voice", None) == "alloy"


@pytest.mark.asyncio
async def test_generate_tts_audio_preserves_pocket_tts_cpp_provider_defaults_without_kitten_voice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _RecordingTTSService()
    _patch_tts_service(monkeypatch, service)

    audio_bytes, alignment, _route = await audiobook_jobs_worker._generate_tts_audio(
        text="Hello world",
        backend=None,
        allow_fallback=None,
        model=None,
        provider="pocket_tts_cpp",
        voice=None,
        speed=None,
        response_format="mp3",
        user_id=17,
    )

    assert audio_bytes == b"tts-audio"
    assert alignment is None
    call = service.calls[0]
    request = call["request"]
    assert call["provider"] == "pocket_tts_cpp"
    assert getattr(request, "model", None) == "pocket_tts_cpp"
    assert getattr(request, "voice", None) == "clone_required"
