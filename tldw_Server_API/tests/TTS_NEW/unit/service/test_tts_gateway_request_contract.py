from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException
from pydantic import ValidationError
from starlette.requests import Request

from tldw_Server_API.app.api.v1.endpoints.audio import audio_tts
from tldw_Server_API.app.api.v1.schemas.audio_schemas import OpenAISpeechRequest
from tldw_Server_API.app.core.TTS.adapters.base import AudioFormat, TTSRequest
from tldw_Server_API.app.core.TTS.tts_exceptions import TTSValidationError
from tldw_Server_API.app.core.TTS.tts_service_v2 import TTSServiceV2
from tldw_Server_API.app.core.Utils.pydantic_compat import model_dump_compat

pytestmark = pytest.mark.unit


def _http_request(backend: str | None = None) -> Request:
    headers = [] if backend is None else [(b"x-tldw-tts-backend", backend.encode("latin-1"))]
    return Request({"type": "http", "method": "POST", "path": "/", "headers": headers})


def _convert(**values) -> TTSRequest:
    payload = {
        "model": "Vendor/Expressive-TTS",
        "input": "Read this exactly.",
    }
    payload.update(values)
    return TTSServiceV2(MagicMock())._convert_request(OpenAISpeechRequest(**payload))


class _CapturingJobManager:
    def __init__(self) -> None:
        self.payload = None

    def create_job(self, **kwargs):
        self.payload = kwargs["payload"]
        return {"id": 41, "status": "queued"}


async def _submit_job_and_convert(
    monkeypatch: pytest.MonkeyPatch,
    request_data: OpenAISpeechRequest,
    *,
    header_backend: str | None = None,
) -> tuple[dict, OpenAISpeechRequest, TTSRequest]:
    async def _resolve_tts_byok(**_kwargs):
        return 1, {}, None

    def _preflight_gateway_speech(**kwargs):
        return SimpleNamespace(
            backend=kwargs["backend"],
            model=kwargs["model"],
            voice=kwargs["voice"] if kwargs["voice_supplied"] else "af_heart",
            response_format=kwargs["response_format"],
            allow_fallback=bool(kwargs["allow_fallback"]),
            conversion_required=False,
        )

    shim_map = {
        "_sanitize_speech_request": lambda *_args, **_kwargs: None,
        "_resolve_tts_byok": _resolve_tts_byok,
    }
    monkeypatch.setattr(audio_tts, "_audio_shim_attr", shim_map.__getitem__)
    monkeypatch.setattr(audio_tts, "preflight_gateway_speech", _preflight_gateway_speech)
    job_manager = _CapturingJobManager()

    await audio_tts.create_speech_job(
        request_data=request_data,
        request=_http_request(header_backend),
        current_user=SimpleNamespace(id="1"),
        jm=job_manager,
    )

    assert job_manager.payload is not None
    queued = job_manager.payload["speech_request"]
    reconstructed = OpenAISpeechRequest(**queued)
    converted = TTSServiceV2(MagicMock())._convert_request(reconstructed)
    return queued, reconstructed, converted


def test_speech_schema_keeps_model_required() -> None:
    with pytest.raises(ValidationError):
        OpenAISpeechRequest(input="Missing model")


def test_speech_schema_serializes_backend_and_fallback_without_changing_defaults() -> None:
    request = OpenAISpeechRequest(
        model="Vendor/Expressive-TTS",
        input="Hello",
        backend="gateway:company-proxy",
        allow_fallback=False,
    )

    payload = model_dump_compat(request)

    assert payload["backend"] == "gateway:company-proxy"
    assert payload["allow_fallback"] is False
    assert request.voice == "af_heart"
    assert request.response_format == "mp3"


@pytest.mark.parametrize(
    ("body", "header", "expected"),
    [
        ("openrouter", None, "openrouter"),
        (None, "gateway:company-proxy", "gateway:company-proxy"),
        ("OPENAI", "open-ai", "openai"),
        ("company-proxy", "gateway:company-proxy", "gateway:company-proxy"),
    ],
)
def test_backend_body_header_mirror_accepts_equivalent_canonical_ids(
    body: str | None,
    header: str | None,
    expected: str,
) -> None:
    request_data = OpenAISpeechRequest(
        model="Vendor/Expressive-TTS",
        input="Hello",
        backend=body,
    )

    resolved = audio_tts._resolve_tts_backend_mirror(request_data, _http_request(header))

    assert resolved == expected
    assert request_data.backend == expected


@pytest.mark.parametrize(
    ("body", "header"),
    [
        ("openrouter", "gateway:company-proxy"),
        (None, "gateway:Bad_Slug"),
        ("", None),
    ],
)
def test_backend_body_header_mirror_rejects_conflict_or_malformed_identity(
    body: str | None,
    header: str | None,
) -> None:
    request_data = OpenAISpeechRequest(
        model="Vendor/Expressive-TTS",
        input="Hello",
        backend=body,
    )

    with pytest.raises(HTTPException) as exc_info:
        audio_tts._resolve_tts_backend_mirror(request_data, _http_request(header))

    assert exc_info.value.status_code == 400
    assert isinstance(exc_info.value.detail, dict)
    assert exc_info.value.detail["error_code"] == "invalid_tts_backend"


@pytest.mark.asyncio
async def test_malformed_backend_header_fails_before_sanitization_or_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_data = OpenAISpeechRequest(model="Vendor/Expressive-TTS", input="Hello")

    def fail_if_called(_name: str):
        raise AssertionError("backend validation must run before provider or credential resolution")

    monkeypatch.setattr(audio_tts, "_audio_shim_attr", fail_if_called)

    with pytest.raises(HTTPException) as exc_info:
        await audio_tts.create_speech_job(
            request_data=request_data,
            request=_http_request("gateway:Bad_Slug"),
            current_user=SimpleNamespace(id="1"),
            jm=MagicMock(),
        )

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_explicit_speech_rejects_authority_extra_params_before_synthesis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        audio_tts,
        "_audio_shim_attr",
        lambda name: (lambda *_args, **_kwargs: None)
        if name == "_sanitize_speech_request"
        else MagicMock(),
    )
    tts_service = MagicMock()

    with pytest.raises(HTTPException) as exc_info:
        await audio_tts.create_speech(
            OpenAISpeechRequest(
                backend="gateway:company-proxy",
                model="Vendor/Expressive-TTS",
                input="Hello",
                stream=False,
                extra_params={
                    "provider_overrides": {"apiKey": "distinctive-private-secret"}
                },
            ),
            _http_request(),
            tts_service=tts_service,
            current_user=SimpleNamespace(id="1"),
            media_db=None,
            usage_log=MagicMock(),
        )

    assert exc_info.value.status_code == 422
    tts_service.generate_speech.assert_not_called()


@pytest.mark.asyncio
async def test_speech_job_minimal_gateway_payload_persists_resolved_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queued, reconstructed, converted = await _submit_job_and_convert(
        monkeypatch,
        OpenAISpeechRequest(
            backend="gateway:company-proxy",
            model="Vendor/Expressive-TTS",
            input="Hello",
        ),
    )

    assert queued == {
        "backend": "gateway:company-proxy",
        "model": "Vendor/Expressive-TTS",
        "input": "Hello",
        "voice": "af_heart",
        "allow_fallback": True,
        "stream": False,
    }
    assert reconstructed.model_fields_set == {
        "backend",
        "model",
        "input",
        "voice",
        "allow_fallback",
        "stream",
    }
    assert converted.supplied_fields.isdisjoint(
        {"speed", "language", "lang_code", "target_sample_rate", "format", "extra_params"}
    )
    assert "voice" in converted.supplied_fields
    assert converted.voice == "af_heart"
    assert converted.speed == 1.0
    assert converted.format is AudioFormat.MP3


@pytest.mark.asyncio
async def test_speech_job_header_backend_becomes_explicit_worker_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queued, reconstructed, converted = await _submit_job_and_convert(
        monkeypatch,
        OpenAISpeechRequest(model="Vendor/Expressive-TTS", input="Hello"),
        header_backend="gateway:company-proxy",
    )

    assert queued["backend"] == "gateway:company-proxy"
    assert "backend" in reconstructed.model_fields_set
    assert converted.backend == "gateway:company-proxy"


@pytest.mark.asyncio
async def test_speech_job_explicit_defaults_and_fallback_survive_worker_roundtrip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queued, reconstructed, converted = await _submit_job_and_convert(
        monkeypatch,
        OpenAISpeechRequest(
            backend="openrouter",
            model="Vendor/Expressive-TTS",
            input="Hello",
            voice="af_heart",
            speed=1.0,
            language=None,
            lang_code=None,
            target_sample_rate=None,
            response_format="mp3",
            extra_params={"style": "warm"},
            allow_fallback=False,
        ),
    )

    assert queued == {
        "backend": "openrouter",
        "model": "Vendor/Expressive-TTS",
        "input": "Hello",
        "voice": "af_heart",
        "response_format": "mp3",
        "speed": 1.0,
        "stream": False,
        "allow_fallback": False,
        "target_sample_rate": None,
        "lang_code": None,
        "language": None,
        "extra_params": {"style": "warm"},
    }
    assert reconstructed.model_fields_set == set(queued)
    assert converted.allow_fallback is False
    assert converted.extra_params == {"style": "warm"}
    assert converted.supplied_fields == {
        "voice",
        "speed",
        "language",
        "lang_code",
        "target_sample_rate",
        "format",
        "extra_params",
    }
    assert converted.supplied_field_values == {
        "voice": "af_heart",
        "speed": 1.0,
        "language": None,
        "lang_code": None,
        "target_sample_rate": None,
        "format": "mp3",
        "extra_params": {"style": "warm"},
    }


@pytest.mark.asyncio
async def test_speech_job_legacy_payload_keeps_backend_absent_and_default_parity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queued, reconstructed, converted = await _submit_job_and_convert(
        monkeypatch,
        OpenAISpeechRequest(model="tts-1", input="Hello"),
    )

    direct = TTSServiceV2(MagicMock())._convert_request(
        OpenAISpeechRequest(model="tts-1", input="Hello", stream=False)
    )
    assert queued == {"model": "tts-1", "input": "Hello", "stream": False}
    assert "backend" not in reconstructed.model_fields_set
    assert converted.backend is None
    assert converted.allow_fallback is True
    assert converted.dict() == direct.dict()


def test_gateway_conversion_tracks_all_explicit_common_fields_and_raw_values() -> None:
    omitted = _convert(backend="gateway:company-proxy")
    explicit = _convert(
        backend="gateway:company-proxy",
        voice="af_heart",
        speed=1.0,
        lang_code="pt-BR",
        target_sample_rate=24000,
        response_format="mp3",
        extra_params={},
    )

    tracked = {
        "voice",
        "speed",
        "language",
        "lang_code",
        "target_sample_rate",
        "format",
        "extra_params",
    }
    assert omitted.supplied_fields.isdisjoint(tracked)
    assert explicit.supplied_fields & tracked == tracked - {"language"}
    assert explicit.supplied_field_values == {
        "voice": "af_heart",
        "speed": 1.0,
        "lang_code": "pt-BR",
        "target_sample_rate": 24000,
        "format": "mp3",
        "extra_params": {},
    }


def test_gateway_conversion_preserves_exact_locale_and_request_contract() -> None:
    converted = _convert(
        backend="gateway:company-proxy",
        allow_fallback=False,
        lang_code="pt-BR",
    )

    assert converted.backend == "gateway:company-proxy"
    assert converted.allow_fallback is False
    assert converted.language == "pt-BR"
    assert converted.lang_code == "pt-BR"
    assert converted.supplied_field_values["lang_code"] == "pt-BR"


def test_gateway_target_sample_rate_stays_out_of_extra_params() -> None:
    converted = _convert(
        backend="gateway:company-proxy",
        target_sample_rate=24000,
        extra_params={"style": "warm"},
    )

    assert converted.target_sample_rate == 24000
    assert converted.extra_params == {"style": "warm"}


def test_gateway_conversion_uses_explicit_language_when_lang_code_is_omitted() -> None:
    converted = _convert(
        backend="openrouter",
        language="en-GB",
    )

    assert converted.language == "en-GB"
    assert converted.lang_code is None


def test_gateway_conversion_rejects_conflicting_explicit_languages() -> None:
    request = OpenAISpeechRequest(
        backend="openrouter",
        model="Vendor/Expressive-TTS",
        input="Hello",
        lang_code="pt-BR",
        language="en-GB",
    )

    with pytest.raises(TTSValidationError, match="lang_code and language"):
        TTSServiceV2(MagicMock())._convert_request(request)


def test_expanded_supplied_fields_survive_dict_roundtrip() -> None:
    converted = _convert(
        backend="openrouter",
        voice="NarratorVoice",
        response_format="wav",
        target_sample_rate=48000,
        extra_params={"style": "warm"},
    )

    restored = TTSRequest(**converted.dict())

    assert restored.dict() == converted.dict()
    assert restored.supplied_fields == converted.supplied_fields
    assert restored.supplied_field_values == converted.supplied_field_values


def test_gateway_conversion_uses_pydantic_v1_fields_set_for_expanded_markers() -> None:
    pydantic_request = OpenAISpeechRequest(
        backend="openrouter",
        model="Vendor/Expressive-TTS",
        input="Hello",
        voice="NarratorVoice",
        response_format="wav",
        target_sample_rate=48000,
        extra_params={"style": "warm"},
    )
    request = SimpleNamespace(**model_dump_compat(pydantic_request))
    request.__fields_set__ = {
        "backend",
        "model",
        "input",
        "voice",
        "response_format",
        "target_sample_rate",
        "extra_params",
    }

    converted = TTSServiceV2(MagicMock())._convert_request(request)

    assert converted.supplied_fields & {
        "voice",
        "format",
        "target_sample_rate",
        "extra_params",
    } == {"voice", "format", "target_sample_rate", "extra_params"}


def test_legacy_conversion_keeps_existing_normalization_and_payload_structure() -> None:
    converted = _convert(
        model="chatterbox-multilingual",
        voice="NarratorVoice",
        lang_code="pt-BR",
        language="en-GB",
        response_format="wav",
        target_sample_rate=24000,
        extra_params={"temperature": 0.7},
    )

    assert converted.backend is None
    assert converted.allow_fallback is True
    assert converted.model == "chatterbox-multilingual"
    assert converted.voice == "NarratorVoice"
    assert converted.language == "pt"
    assert converted.lang_code == "pt"
    assert converted.format is AudioFormat.WAV
    assert converted.target_sample_rate == 24000
    assert converted.extra_params == {
        "temperature": 0.7,
        "target_sample_rate": 24000,
        "sample_rate": 24000,
    }
