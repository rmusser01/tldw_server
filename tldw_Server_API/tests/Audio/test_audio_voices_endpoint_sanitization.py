import sys
import types
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from tldw_Server_API.app.api.v1.endpoints.audio import audio_voices
from tldw_Server_API.app.api.v1.schemas.audio_schemas import VoiceEncodeRequest


class _LoggerStub:
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.warning_kwargs = []

    def error(self, message, *args, **kwargs):
        self.errors.append(str(message))

    def warning(self, message, *args, **kwargs):
        self.warnings.append(str(message))
        self.warning_kwargs.append(kwargs)


class _UploadFileStub:
    filename = "sample.wav"

    async def read(self):
        return b"audio-bytes"


class _VoiceUploadRequestStub:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _VoiceProcessingError(Exception):
    pass


class _VoiceQuotaExceededError(Exception):
    pass


class _VoiceManagerStub:
    async def upload_voice(self, **kwargs):
        _ = kwargs
        raise RuntimeError("voice backend exploded at /private/voices.db")

    async def encode_voice_reference(self, **kwargs):
        _ = kwargs
        raise RuntimeError("voice backend exploded at /private/voices.db")

    async def list_user_voices(self, *_args, **_kwargs):
        raise RuntimeError("voice backend exploded at /private/voices.db")

    async def get_voice(self, *_args, **_kwargs):
        raise RuntimeError("voice backend exploded at /private/voices.db")

    async def delete_voice(self, *_args, **_kwargs):
        raise RuntimeError("voice backend exploded at /private/voices.db")


class _QuotaVoiceManagerStub(_VoiceManagerStub):
    async def upload_voice(self, **kwargs):
        _ = kwargs
        raise _VoiceQuotaExceededError("voice quota exploded at /private/voices.db")


class _UploadProcessingVoiceManagerStub(_VoiceManagerStub):
    async def upload_voice(self, **kwargs):
        _ = kwargs
        raise _VoiceProcessingError("voice processing exploded at /private/voices.db")


class _EncodeProcessingVoiceManagerStub(_VoiceManagerStub):
    async def encode_voice_reference(self, **kwargs):
        _ = kwargs
        raise _VoiceProcessingError("voice encoding exploded at /private/voices.db")


def _install_voice_manager(monkeypatch, voice_manager=None):
    fake_module = types.ModuleType("voice_manager")
    fake_module.VoiceProcessingError = _VoiceProcessingError
    fake_module.VoiceQuotaExceededError = _VoiceQuotaExceededError
    fake_module.VoiceUploadRequest = _VoiceUploadRequestStub
    fake_module.get_voice_manager = lambda: voice_manager or _VoiceManagerStub()
    monkeypatch.setitem(
        sys.modules,
        "tldw_Server_API.app.core.TTS.voice_manager",
        fake_module,
    )


async def _call_endpoint(route_name: str):
    user = SimpleNamespace(id=1)
    request = SimpleNamespace(headers={}, state=SimpleNamespace())
    if route_name == "upload":
        return await audio_voices.upload_voice(
            request=request,
            file=_UploadFileStub(),
            name="sample",
            current_user=user,
        )
    if route_name == "encode":
        return await audio_voices.encode_voice_reference(
            VoiceEncodeRequest(voice_id="voice-1", provider="neutts"),
            current_user=user,
        )
    if route_name == "list":
        return await audio_voices.list_voices(request=request, current_user=user)
    if route_name == "get":
        return await audio_voices.get_voice_details(request=request, voice_id="voice-1", current_user=user)
    if route_name == "delete":
        return await audio_voices.delete_voice(request=request, voice_id="voice-1", current_user=user)
    if route_name == "preview":
        return await audio_voices.preview_voice(
            request=request,
            voice_id="voice-1",
            current_user=user,
            tts_service=object(),
        )
    raise AssertionError(f"unknown route {route_name}")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("route_name", "expected_detail", "expected_log"),
    [
        ("upload", "Failed to upload voice sample", "Voice upload error"),
        ("encode", "Failed to encode voice reference", "Voice encode error"),
        ("list", "Failed to list voices", "Error listing voices"),
        ("get", "Failed to get voice details", "Error getting voice details"),
        ("delete", "Failed to delete voice", "Error deleting voice"),
        ("preview", "Failed to generate voice preview", "Voice preview error"),
    ],
)
async def test_audio_voice_generic_failure_logs_are_sanitized(
    monkeypatch,
    route_name: str,
    expected_detail: str,
    expected_log: str,
):
    logger_stub = _LoggerStub()
    monkeypatch.setattr(audio_voices, "logger", logger_stub)
    monkeypatch.setattr(audio_voices, "ensure_request_id", lambda _request: "req-test")
    _install_voice_manager(monkeypatch)

    with pytest.raises(HTTPException) as exc_info:
        await _call_endpoint(route_name)

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == expected_detail
    assert logger_stub.errors == [expected_log]
    assert "/private/" not in logger_stub.errors[0]
    assert "exploded" not in logger_stub.errors[0]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("route_name", "voice_manager", "expected_status", "expected_log"),
    [
        ("upload", _QuotaVoiceManagerStub(), 429, "Voice quota exceeded"),
        ("upload", _UploadProcessingVoiceManagerStub(), 400, "Voice processing failed"),
        ("encode", _EncodeProcessingVoiceManagerStub(), 400, "Voice encoding failed"),
    ],
)
async def test_audio_voice_validation_failure_logs_are_sanitized(
    monkeypatch,
    route_name: str,
    voice_manager,
    expected_status: int,
    expected_log: str,
):
    logger_stub = _LoggerStub()
    monkeypatch.setattr(audio_voices, "logger", logger_stub)
    monkeypatch.setattr(audio_voices, "ensure_request_id", lambda _request: "req-test")
    _install_voice_manager(monkeypatch, voice_manager)

    with pytest.raises(HTTPException) as exc_info:
        await _call_endpoint(route_name)

    assert exc_info.value.status_code == expected_status
    assert logger_stub.warnings == [expected_log]
    assert logger_stub.warning_kwargs == [{}]
    assert "/private/" not in logger_stub.warnings[0]
    assert "exploded" not in logger_stub.warnings[0]
