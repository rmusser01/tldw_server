"""Tests for the generic OpenAI-compatible speech gateway adapter."""

from __future__ import annotations

import asyncio
import math
from collections.abc import AsyncIterator, Mapping
from copy import deepcopy
from typing import Any

import pytest

from tldw_Server_API.app.core.exceptions import NetworkError
from tldw_Server_API.app.core.TTS.adapters import (
    openai_compatible_speech_adapter as adapter_module,
)
from tldw_Server_API.app.core.TTS.adapters.base import (
    AudioFormat,
    TTSAdapter,
    TTSRequest,
)
from tldw_Server_API.app.core.TTS.adapters.openai_compatible_speech_adapter import (
    OpenAICompatibleSpeechAdapter,
)
from tldw_Server_API.app.core.TTS.gateway_config import (
    copy_gateway_extra_params,
    normalize_gateway_specs,
)
from tldw_Server_API.app.core.TTS.tts_exceptions import (
    TTSAudioQualityError,
    TTSAuthenticationError,
    TTSModelNotFoundError,
    TTSNetworkError,
    TTSProviderError,
    TTSProviderNotConfiguredError,
    TTSQuotaExceededError,
    TTSRateLimitError,
    TTSTextTooLongError,
    TTSTimeoutError,
    TTSValidationError,
)

MP3 = b"ID3\x04\x00\x00\x00\x00\x00\x00" + b"\x01" * 32
WAV = b"RIFF\x24\x00\x00\x00WAVEfmt " + b"\x00" * 24
FLAC = b"fLaC" + b"\x00" * 32
OGG = b"OggS" + b"\x00" * 32
OPUS = b"OggS" + b"\x00" * 24 + b"OpusHead" + b"\x01" * 24
_OMIT = object()


def _adapter_config(
    *,
    model: str = "Vendor/Expressive-TTS",
    source_format: str = "mp3",
    allowed_request_options: tuple[str, ...] = (),
    capability_overrides: Mapping[str, Any] | None = None,
    config_overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the attempt-local adapter dictionary from an immutable GatewaySpec."""
    capabilities: dict[str, Any] = {
        "formats": [source_format],
        "supports_speed": True,
        "supports_language": True,
        "supports_target_sample_rate": True,
        "allow_octet_stream": False,
        "max_input_characters": 100,
        "max_response_bytes": 1024,
        "pcm": {"sample_rate": 24000, "channels": 1, "sample_width_bits": 16},
    }
    capabilities.update(dict(capability_overrides or {}))
    spec = normalize_gateway_specs(
        {},
        {
            "company": {
                "enabled": True,
                "display_name": "Company Speech",
                "base_url": "https://speech.example/v1/",
                "speech_path": "audio/speech",
                "headers": {"X-Route": "admin-route"},
                "api_key": "admin-secret",
                "default_model": model,
                "default_voice": "Narrator",
                "allowed_models": [model],
                "allowed_request_options": list(allowed_request_options),
                "capability_defaults": capabilities,
            }
        },
    )["gateway:company"]
    effective = spec.capabilities_for_model(model)
    config: dict[str, Any] = {
        "backend_id": spec.backend_id,
        "base_url": spec.base_url,
        "speech_path": spec.speech_path,
        "headers": spec.headers,
        "api_key": spec.api_key,
        "default_voice": spec.default_voice_for_model(model),
        "allowed_request_options": spec.allowed_request_options,
        "capabilities": effective.model_dump(mode="python"),
        "source_format": source_format,
        "conversion_needed": False,
        "timeout_seconds": 30.0,
    }
    config.update(dict(config_overrides or {}))
    return config


def _request(**overrides: Any) -> TTSRequest:
    values: dict[str, Any] = {
        "text": "Read this exactly.",
        "voice": "GuideVoice",
        "language": "en-US",
        "format": AudioFormat.MP3,
        "speed": 1.25,
        "target_sample_rate": 48000,
        "stream": True,
        "provider": "gateway:company",
        "model": "Vendor/Expressive-TTS",
    }
    for key, value in overrides.items():
        if value is _OMIT:
            values.pop(key, None)
        else:
            values[key] = value
    return TTSRequest(**values)


def _stream_stub(
    calls: list[dict[str, Any]],
    *,
    chunks: tuple[bytes, ...] = (MP3,),
    status: int = 200,
    headers: Mapping[str, str] | None = None,
):
    async def stream(**kwargs: Any) -> AsyncIterator[bytes]:
        calls.append(kwargs)
        callback = kwargs["on_response"]
        result = callback(
            status,
            dict({"Content-Type": "audio/mpeg"} if headers is None else headers),
        )
        if asyncio.iscoroutine(result):
            await result
        for chunk in chunks:
            yield chunk

    return stream


async def _collect(response) -> bytes:
    assert response.audio_stream is not None
    return b"".join([chunk async for chunk in response.audio_stream])


@pytest.mark.unit
@pytest.mark.asyncio
async def test_one_post_uses_server_url_fixed_auth_exact_body_and_one_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(adapter_module, "astream_bytes", _stream_stub(calls))
    original = _adapter_config(
        allowed_request_options=("/provider/order",),
        config_overrides={"conversion_needed": True},
    )
    adapter = OpenAICompatibleSpeechAdapter(original)
    original["base_url"] = "https://attacker.invalid/"
    request = _request(extra_params={"provider": {"order": ["Acme/Primary"]}})

    response = await adapter.generate(request)

    assert isinstance(adapter, TTSAdapter)
    assert await _collect(response) == MP3
    assert len(calls) == 1
    call = calls[0]
    assert call["method"] == "POST"
    assert str(call["url"]) == "https://speech.example/v1/audio/speech"
    assert call["headers"] == {
        "X-Route": "admin-route",
        "Authorization": "Bearer admin-secret",
        "Content-Type": "application/json",
    }
    assert call["json"] == {
        "model": "Vendor/Expressive-TTS",
        "input": "Read this exactly.",
        "voice": "GuideVoice",
        "response_format": "mp3",
        "speed": 1.25,
        "language": "en-US",
        "target_sample_rate": 48000,
        "provider": {"order": ["Acme/Primary"]},
    }
    assert call["retry"].attempts == 1
    assert response.metadata == {
        "backend_id": "gateway:company",
        "model": "Vendor/Expressive-TTS",
        "voice": "GuideVoice",
        "source_format": "mp3",
        "declared_content_type": "audio/mpeg",
        "conversion_needed": True,
    }


@pytest.mark.unit
@pytest.mark.asyncio
async def test_model_and_voice_casing_are_preserved(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(adapter_module, "astream_bytes", _stream_stub(calls))
    model = "Vendor/MiXeD-Case-TTS"
    adapter = OpenAICompatibleSpeechAdapter(_adapter_config(model=model))

    await _collect(await adapter.generate(_request(model=model, voice="Voice/ExactCASE")))

    assert calls[0]["json"]["model"] == model
    assert calls[0]["json"]["voice"] == "Voice/ExactCASE"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_lang_code_wins_and_equal_aliases_are_allowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(adapter_module, "astream_bytes", _stream_stub(calls))
    request = _request(language="fr-FR", lang_code="fr-FR")

    await _collect(await OpenAICompatibleSpeechAdapter(_adapter_config()).generate(request))

    assert calls[0]["json"]["language"] == "fr-FR"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_conflicting_language_aliases_fail_before_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(adapter_module, "astream_bytes", _stream_stub(calls))
    request = _request(language="en-US", lang_code="fr-FR")

    with pytest.raises(TTSValidationError, match="language"):
        await OpenAICompatibleSpeechAdapter(_adapter_config()).generate(request)

    assert calls == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_omitted_unsupported_common_field_defaults_are_not_injected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(adapter_module, "astream_bytes", _stream_stub(calls))
    adapter = OpenAICompatibleSpeechAdapter(
        _adapter_config(
            capability_overrides={
                "supports_speed": False,
                "supports_language": False,
                "supports_target_sample_rate": False,
            }
        )
    )

    response = await adapter.generate(
        _request(speed=_OMIT, language=_OMIT, target_sample_rate=None)
    )
    await _collect(response)
    assert calls[0]["json"] == {
        "model": "Vendor/Expressive-TTS",
        "input": "Read this exactly.",
        "voice": "GuideVoice",
        "response_format": "mp3",
    }


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("speed", 1.0),
        ("speed", 1.5),
        ("language", "en"),
        ("lang_code", "fr-FR"),
    ],
)
async def test_explicit_unsupported_common_field_fails_before_network(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: Any,
) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(adapter_module, "astream_bytes", _stream_stub(calls))
    adapter = OpenAICompatibleSpeechAdapter(
        _adapter_config(
            capability_overrides={
                "supports_speed": False,
                "supports_language": False,
                "supports_target_sample_rate": False,
            }
        )
    )
    overrides = {"speed": _OMIT, "language": _OMIT, "target_sample_rate": None}
    overrides[field] = value

    with pytest.raises(TTSValidationError, match=field):
        await adapter.generate(_request(**overrides))

    assert calls == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_explicit_unsupported_target_sample_rate_still_fails_before_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(adapter_module, "astream_bytes", _stream_stub(calls))
    adapter = OpenAICompatibleSpeechAdapter(
        _adapter_config(
            capability_overrides={"supports_target_sample_rate": False}
        )
    )

    with pytest.raises(TTSValidationError, match="target_sample_rate"):
        await adapter.generate(_request(target_sample_rate=44100))

    assert calls == []


@pytest.mark.unit
def test_json_pointer_copy_supports_nested_and_escaped_tokens() -> None:
    supplied = {
        "provider": {
            "order": ["Acme/Primary", "Acme/Backup"],
            "options": {"quality/tier": "high", "literal~name": True},
        }
    }

    copied = copy_gateway_extra_params(
        supplied,
        frozenset(
            {
                "/provider/order",
                "/provider/options/quality~1tier",
                "/provider/options/literal~0name",
            }
        ),
    )

    assert copied == supplied
    assert copied is not supplied
    assert copied["provider"] is not supplied["provider"]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("supplied", "allowed"),
    [
        ({"provider": {"order": ["a"], "unknown": True}}, {"/provider/order"}),
        ({"provider": {"order": ["a"]}}, {"/provider"}),
        ({"provider": {"order": ["a"]}}, {"/provide/order"}),
        ({"provider": {}}, {"/provider/order"}),
        ({"provider": {"order": ["a"]}}, {"/provider/ordering"}),
    ],
)
def test_every_supplied_leaf_requires_an_exact_pointer(
    supplied: dict[str, Any],
    allowed: set[str],
) -> None:
    with pytest.raises(ValueError, match="allow|pointer|container|leaf"):
        copy_gateway_extra_params(supplied, frozenset(allowed))


@pytest.mark.unit
@pytest.mark.parametrize(
    "reserved",
    [
        "url",
        "speech_path",
        "headers",
        "authorization",
        "api_key",
        "credential",
        "model",
        "input",
        "voice",
        "response_format",
        "speed",
        "lang_code",
        "language",
        "target_sample_rate",
    ],
)
def test_reserved_override_is_impossible_even_with_runtime_allowlist(
    reserved: str,
) -> None:
    with pytest.raises(ValueError, match="reserved"):
        copy_gateway_extra_params(
            {"extension": {reserved: "attacker-controlled"}},
            frozenset({f"/extension/{reserved}"}),
        )


def _nested(depth: int) -> dict[str, Any]:
    value: Any = "leaf"
    for index in range(depth):
        value = {f"k{index}": value}
    return value


@pytest.mark.unit
@pytest.mark.parametrize(
    "supplied",
    [
        _nested(9),
        {f"k{index}": index for index in range(65)},
        {"k" * 4097: "value"},
        {"option": "v" * 4097},
        {f"k{index}": "v" * 4096 for index in range(17)},
        {1: "non-string-key"},
        {"option": {"not-json"}},
        {"option": math.nan},
        {"option": math.inf},
        {"option": [*range(65)]},
        {"option": [[[[[[[[[1]]]]]]]]]},
    ],
)
def test_extra_params_limits_and_json_types_are_enforced(supplied: dict[Any, Any]) -> None:
    with pytest.raises(ValueError, match="depth|leav|4096|65536|JSON|finite|key|string"):
        copy_gateway_extra_params(
            supplied,
            frozenset({"/option", *(f"/k{index}" for index in range(65))}),
        )


@pytest.mark.unit
def test_array_is_one_authorized_field_but_contents_remain_bounded() -> None:
    value = {"provider": {"order": [{"only": "A"}, ["B", "C"]]}}

    assert copy_gateway_extra_params(
        value,
        frozenset({"/provider/order"}),
    ) == value

    with pytest.raises(ValueError, match="reserved"):
        copy_gateway_extra_params(
            {"provider": {"order": [{"url": "https://attacker.invalid"}]}},
            frozenset({"/provider/order"}),
        )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_extra_param_failure_makes_no_network_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(adapter_module, "astream_bytes", _stream_stub(calls))
    adapter = OpenAICompatibleSpeechAdapter(
        _adapter_config(allowed_request_options=("/provider/order",))
    )

    with pytest.raises(TTSValidationError, match="extra_params"):
        await adapter.generate(
            _request(extra_params={"provider": {"order": [*range(65)]}})
        )

    assert calls == []


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (401, TTSAuthenticationError),
        (403, TTSAuthenticationError),
        (402, TTSQuotaExceededError),
        (429, TTSRateLimitError),
        (408, TTSTimeoutError),
        (504, TTSTimeoutError),
        (404, TTSModelNotFoundError),
        (422, TTSValidationError),
        (500, TTSProviderError),
    ],
)
async def test_status_taxonomy_is_typed_and_sanitized(
    monkeypatch: pytest.MonkeyPatch,
    status: int,
    expected: type[Exception],
) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        adapter_module,
        "astream_bytes",
        _stream_stub(
            calls,
            status=status,
            headers={"Content-Type": "application/json", "X-Private": "secret"},
        ),
    )

    response = await OpenAICompatibleSpeechAdapter(_adapter_config()).generate(_request())
    with pytest.raises(expected) as exc_info:
        await _collect(response)

    assert "secret" not in str(exc_info.value)
    assert "speech.example" not in str(exc_info.value)
    assert "admin-secret" not in str(exc_info.value)


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("transport_error", "expected"),
    [
        (TimeoutError("read timed out with private detail"), TTSTimeoutError),
        (NetworkError("dns failed with private detail"), TTSNetworkError),
        (OSError("socket failed with private detail"), TTSNetworkError),
    ],
)
async def test_transport_taxonomy_is_typed_and_sanitized(
    monkeypatch: pytest.MonkeyPatch,
    transport_error: Exception,
    expected: type[Exception],
) -> None:
    async def fail(**_kwargs: Any) -> AsyncIterator[bytes]:
        raise transport_error
        yield b""  # pragma: no cover

    monkeypatch.setattr(adapter_module, "astream_bytes", fail)
    response = await OpenAICompatibleSpeechAdapter(_adapter_config()).generate(_request())

    with pytest.raises(expected) as exc_info:
        await _collect(response)

    assert "private detail" not in str(exc_info.value)


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("source_format", "content_type", "audio"),
    [
        ("mp3", "audio/mp3; charset=binary", MP3),
        ("wav", "audio/x-wav", WAV),
        ("flac", "audio/x-flac", FLAC),
        ("ogg", "application/ogg", OGG),
        ("opus", "audio/opus", OPUS),
    ],
)
async def test_mime_aliases_and_signatures_are_validated(
    monkeypatch: pytest.MonkeyPatch,
    source_format: str,
    content_type: str,
    audio: bytes,
) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        adapter_module,
        "astream_bytes",
        _stream_stub(calls, chunks=(audio,), headers={"Content-Type": content_type}),
    )
    adapter = OpenAICompatibleSpeechAdapter(_adapter_config(source_format=source_format))

    assert await _collect(await adapter.generate(_request())) == audio


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "audio",
    [
        OGG,
        b"OggS" + b"\x00" * 24 + b"\x01vorbis" + b"\x00" * 24,
        b"OggS" + b"\x00" * 128,
        b"\x00" * 128,
        b"OggS" + b"\x00" * 65532 + b"OpusHead",
    ],
)
async def test_opus_requires_ogg_container_and_bounded_opus_head(
    monkeypatch: pytest.MonkeyPatch,
    audio: bytes,
) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        adapter_module,
        "astream_bytes",
        _stream_stub(
            calls,
            chunks=(audio,),
            headers={"Content-Type": "audio/opus"},
        ),
    )
    response = await OpenAICompatibleSpeechAdapter(
        _adapter_config(
            source_format="opus",
            capability_overrides={"max_response_bytes": 70000},
        )
    ).generate(_request())

    with pytest.raises(TTSAudioQualityError, match="signature"):
        await _collect(response)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_opus_head_may_span_chunks_before_first_validated_yield(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        adapter_module,
        "astream_bytes",
        _stream_stub(
            calls,
            chunks=(OPUS[:29], OPUS[29:33], OPUS[33:]),
            headers={"Content-Type": "audio/opus"},
        ),
    )

    assert await _collect(
        await OpenAICompatibleSpeechAdapter(
            _adapter_config(source_format="opus")
        ).generate(_request())
    ) == OPUS


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"Content-Type": "application/json"},
        {"Content-Type": "audio/mpeg, application/json"},
        {"Content-Type": "application/octet-stream"},
    ],
)
async def test_missing_mismatched_ambiguous_and_unapproved_octet_mime_fail(
    monkeypatch: pytest.MonkeyPatch,
    headers: Mapping[str, str],
) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        adapter_module,
        "astream_bytes",
        _stream_stub(calls, chunks=(MP3,), headers=headers),
    )
    response = await OpenAICompatibleSpeechAdapter(_adapter_config()).generate(_request())

    with pytest.raises(TTSAudioQualityError, match="content type"):
        await _collect(response)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_octet_stream_requires_capability_and_still_checks_signature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        adapter_module,
        "astream_bytes",
        _stream_stub(
            calls,
            chunks=(MP3,),
            headers={"Content-Type": "application/octet-stream"},
        ),
    )
    adapter = OpenAICompatibleSpeechAdapter(
        _adapter_config(capability_overrides={"allow_octet_stream": True})
    )

    assert await _collect(await adapter.generate(_request())) == MP3


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("chunks", [(), (b"",), (b"not audio",)])
async def test_empty_or_invalid_audio_fails(
    monkeypatch: pytest.MonkeyPatch,
    chunks: tuple[bytes, ...],
) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(adapter_module, "astream_bytes", _stream_stub(calls, chunks=chunks))
    response = await OpenAICompatibleSpeechAdapter(_adapter_config()).generate(_request())

    with pytest.raises(TTSAudioQualityError, match="empty|signature"):
        await _collect(response)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_signature_sniff_is_bounded_to_64_kib(monkeypatch: pytest.MonkeyPatch) -> None:
    pulls = 0

    async def invalid(**kwargs: Any) -> AsyncIterator[bytes]:
        nonlocal pulls
        kwargs["on_response"](200, {"Content-Type": "audio/mpeg"})
        pulls += 1
        yield b"x" * 65536
        pulls += 1
        yield MP3

    monkeypatch.setattr(adapter_module, "astream_bytes", invalid)
    response = await OpenAICompatibleSpeechAdapter(
        _adapter_config(capability_overrides={"max_response_bytes": 131072})
    ).generate(_request())

    with pytest.raises(TTSAudioQualityError, match="signature"):
        await _collect(response)
    assert pulls == 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_declared_and_streamed_response_size_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []
    adapter = OpenAICompatibleSpeechAdapter(
        _adapter_config(capability_overrides={"max_response_bytes": len(MP3)})
    )
    monkeypatch.setattr(
        adapter_module,
        "astream_bytes",
        _stream_stub(
            calls,
            headers={"Content-Type": "audio/mpeg", "Content-Length": str(len(MP3) + 1)},
        ),
    )
    with pytest.raises(TTSAudioQualityError, match="size"):
        await _collect(await adapter.generate(_request()))

    calls.clear()
    monkeypatch.setattr(
        adapter_module,
        "astream_bytes",
        _stream_stub(calls, chunks=(MP3, b"x")),
    )
    with pytest.raises(TTSAudioQualityError, match="size"):
        await _collect(await adapter.generate(_request()))


@pytest.mark.unit
@pytest.mark.asyncio
async def test_pcm_stream_emits_complete_frames_across_chunk_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        adapter_module,
        "astream_bytes",
        _stream_stub(
            calls,
            chunks=(b"\x01", b"\x02\x03\x04"),
            headers={"Content-Type": "audio/pcm"},
        ),
    )
    adapter = OpenAICompatibleSpeechAdapter(_adapter_config(source_format="pcm"))
    response = await adapter.generate(_request(format=AudioFormat.PCM))

    assert response.sample_rate == 24000
    assert response.channels == 1
    assert await _collect(response) == b"\x01\x02\x03\x04"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_pcm_rejects_unaligned_tail(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        adapter_module,
        "astream_bytes",
        _stream_stub(
            calls,
            chunks=(b"\x01\x02\x03",),
            headers={"Content-Type": "audio/pcm"},
        ),
    )
    response = await OpenAICompatibleSpeechAdapter(
        _adapter_config(source_format="pcm")
    ).generate(_request(format=AudioFormat.PCM))

    with pytest.raises(TTSAudioQualityError, match="frame"):
        await _collect(response)


class _TrackingStream:
    def __init__(self, outcomes: list[bytes | BaseException]) -> None:
        self.outcomes = outcomes
        self.closed = 0
        self.started = False

    def __aiter__(self) -> _TrackingStream:
        return self

    async def __anext__(self) -> bytes:
        if not self.started:
            self.started = True
        if not self.outcomes:
            raise StopAsyncIteration
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    async def aclose(self) -> None:
        self.closed += 1


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "outcomes",
    [
        [MP3],
        [b"invalid"],
        [asyncio.CancelledError()],
    ],
)
async def test_upstream_iterator_closes_exactly_once_on_every_path(
    monkeypatch: pytest.MonkeyPatch,
    outcomes: list[bytes | BaseException],
) -> None:
    upstream = _TrackingStream(deepcopy(outcomes))

    def stream(**kwargs: Any) -> _TrackingStream:
        kwargs["on_response"](200, {"Content-Type": "audio/mpeg"})
        return upstream

    monkeypatch.setattr(adapter_module, "astream_bytes", stream)
    response = await OpenAICompatibleSpeechAdapter(_adapter_config()).generate(_request())
    if outcomes and isinstance(outcomes[0], asyncio.CancelledError):
        with pytest.raises(asyncio.CancelledError):
            await _collect(response)
    elif outcomes == [b"invalid"]:
        with pytest.raises(TTSAudioQualityError):
            await _collect(response)
    else:
        assert await _collect(response) == MP3
    assert upstream.closed == 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_pre_body_status_failure_releases_upstream_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    closed = 0

    async def stream(**kwargs: Any) -> AsyncIterator[bytes]:
        nonlocal closed
        try:
            kwargs["on_response"](500, {"Content-Type": "application/json"})
            yield MP3  # pragma: no cover
        finally:
            closed += 1

    monkeypatch.setattr(adapter_module, "astream_bytes", stream)
    response = await OpenAICompatibleSpeechAdapter(_adapter_config()).generate(_request())

    with pytest.raises(TTSProviderError):
        await _collect(response)
    assert closed == 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_response_size_termination_releases_upstream_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    upstream = _TrackingStream([MP3, b"x"])

    def stream(**kwargs: Any) -> _TrackingStream:
        kwargs["on_response"](200, {"Content-Type": "audio/mpeg"})
        return upstream

    monkeypatch.setattr(adapter_module, "astream_bytes", stream)
    adapter = OpenAICompatibleSpeechAdapter(
        _adapter_config(capability_overrides={"max_response_bytes": len(MP3)})
    )

    with pytest.raises(TTSAudioQualityError, match="size"):
        await _collect(await adapter.generate(_request()))
    assert upstream.closed == 1


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [True, False])
async def test_streaming_and_non_streaming_both_return_async_iterator_contract(
    monkeypatch: pytest.MonkeyPatch,
    streaming: bool,
) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(adapter_module, "astream_bytes", _stream_stub(calls))
    response = await OpenAICompatibleSpeechAdapter(_adapter_config()).generate(
        _request(stream=streaming)
    )

    assert response.audio_data is None
    assert response.audio_stream is not None
    assert hasattr(response.audio_stream, "__aiter__")
    assert await _collect(response) == MP3


@pytest.mark.unit
@pytest.mark.asyncio
async def test_preflight_failures_do_not_call_network(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(adapter_module, "astream_bytes", _stream_stub(calls))

    with pytest.raises(TTSTextTooLongError):
        await OpenAICompatibleSpeechAdapter(
            _adapter_config(capability_overrides={"max_input_characters": 3})
        ).generate(_request(text="four"))
    with pytest.raises(TTSProviderNotConfiguredError):
        await OpenAICompatibleSpeechAdapter(
            _adapter_config(config_overrides={"api_key": None})
        ).generate(_request())
    with pytest.raises(TTSValidationError, match="header"):
        OpenAICompatibleSpeechAdapter(
            _adapter_config(
                config_overrides={
                    "headers": (("Authorization", "attacker"),),
                }
            )
        )

    assert calls == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_metadata_never_discloses_transport_or_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(adapter_module, "astream_bytes", _stream_stub(calls))
    response = await OpenAICompatibleSpeechAdapter(_adapter_config()).generate(_request())
    await _collect(response)

    serialized = repr(response.metadata).lower()
    for forbidden in (
        "speech.example",
        "admin-secret",
        "authorization",
        "x-route",
        "credential",
        "raw",
        "header",
        "url",
    ):
        assert forbidden not in serialized
