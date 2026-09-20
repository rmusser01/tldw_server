from __future__ import annotations

import asyncio
import gc
import json
import os
import struct
import threading
import wave
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any, BinaryIO

import pytest

from tldw_Server_API.app.core.exceptions import (
    STTExecutionPlanError,
    STTExecutionUnsupportedError,
    STTTranscriptionError,
)
from tldw_Server_API.app.core.http_client import (
    RetryPolicy,
    resolve_afetch_transport,
)
from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio import (
    Audio_Transcription_AudioCpp as audio_cpp,
)
from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.stt_execution_contract import (
    SttAudioEgress,
    SttExecutionRoute,
    SttTranscriptionOutcome,
    _normalize_audio_endpoint,
)
from tldw_Server_API.app.core.Security.egress import ConfiguredEndpointScope

_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "audio_cpp_http_v1.json"


def _audio_cpp_http_fixture() -> dict[str, object]:
    return json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, separators=(",", ":")).encode()


def _write_audio_cpp_pcm_wav(
    path: Path,
    *,
    frames: int = 16,
    channels: int = 1,
    sample_width: int = 2,
    sample_rate: int = 16_000,
) -> bytes:
    payload = bytes(index % 251 for index in range(frames * channels * sample_width))
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(payload)
    return payload


def _write_audio_cpp_non_pcm_wav(path: Path) -> None:
    fmt = struct.pack("<HHIIHH", 3, 1, 16_000, 64_000, 4, 32)
    data = b"\x00\x00\x00\x00"
    riff_size = 4 + 8 + len(fmt) + 8 + len(data)
    path.write_bytes(
        b"RIFF"
        + struct.pack("<I", riff_size)
        + b"WAVE"
        + b"fmt "
        + struct.pack("<I", len(fmt))
        + fmt
        + b"data"
        + struct.pack("<I", len(data))
        + data
    )


class _ClosableAudioCppResponse:
    def __init__(self, content: bytes, *, status_code: int = 200) -> None:
        self._content = content
        self.status_code = status_code
        self.closed = False

    @property
    def content(self) -> bytes:
        return self._content

    async def aclose(self) -> None:
        self.closed = True


class _SyncClosableAudioCppResponse:
    def __init__(self, content: bytes, *, status_code: int = 200) -> None:
        self.content = content
        self.status_code = status_code
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _ExplodingContentAudioCppResponse(_ClosableAudioCppResponse):
    @property
    def content(self) -> bytes:
        raise TimeoutError("server-secret-response-read")


class _BlockingCloseAudioCppResponse(_ClosableAudioCppResponse):
    def __init__(
        self,
        content: bytes,
        *,
        close_entered: asyncio.Event,
        release_close: asyncio.Event,
    ) -> None:
        super().__init__(content)
        self._close_entered = close_entered
        self._release_close = release_close

    async def aclose(self) -> None:
        self._close_entered.set()
        await self._release_close.wait()
        self.closed = True


class _SelfCancellingCloseAudioCppResponse(_ClosableAudioCppResponse):
    async def aclose(self) -> None:
        task = asyncio.current_task()
        assert task is not None
        task.cancel()
        await asyncio.sleep(0)


class _FailingCloseAudioCppResponse(_ClosableAudioCppResponse):
    async def aclose(self) -> None:
        raise RuntimeError("server-secret-close")


class _TrackedAudioCppClient:
    def __init__(self) -> None:
        self.close_calls = 0

    async def aclose(self) -> None:
        self.close_calls += 1


def _audio_cpp_catalog(*model_ids: str) -> bytes:
    return _json_bytes(
        {
            "object": "list",
            "data": [
                {
                    "id": model_id,
                    "family": "whisper",
                    "task": "asr",
                    "mode": "offline",
                }
                for model_id in model_ids
            ],
        }
    )


def _audio_cpp_route(
    *,
    origin: str = "http://127.0.0.1:18080",
    model_id: str = "whisper-small",
    transport: str | None = None,
) -> SttExecutionRoute:
    selected_transport = transport or resolve_afetch_transport(None)
    transcription_url = f"{origin}/v1/audio/transcriptions"
    _normalized, egress, endpoint_id = _normalize_audio_endpoint(transcription_url)
    return SttExecutionRoute(
        route_id="audio-cpp-0",
        provider="audio-cpp",
        model_label=model_id,
        artifact_id=None,
        identity_resolved=False,
        backend="audio_cpp_http",
        source="audio_cpp_http",
        audio_egress=egress,
        endpoint_id=endpoint_id,
        device=None,
        compute_type=None,
        dtype=None,
        decoding_ids=(),
        local_model_available=False,
        would_download=False,
        transport=selected_transport,
    )


def _audio_cpp_request_kwargs(
    tmp_path: Path,
    *,
    origin: str = "http://127.0.0.1:18080",
    model_id: str = "whisper-small",
    language: str | None = "en",
    transport: str | None = None,
) -> dict[str, object]:
    selected_transport = transport or resolve_afetch_transport(None)
    audio_path = tmp_path / "sample.wav"
    if not audio_path.exists():
        _write_audio_cpp_pcm_wav(audio_path)
    return {
        "audio_path": audio_path,
        "base_dir": tmp_path,
        "route": _audio_cpp_route(
            origin=origin,
            model_id=model_id,
            transport=selected_transport,
        ),
        "origin": origin,
        "model_id": model_id,
        "timeout_seconds": 17.25,
        "transport": selected_transport,
        "language": language,
    }


def _successful_audio_cpp_response(url: str) -> _ClosableAudioCppResponse:
    if url.endswith("/health"):
        body = _json_bytes({"status": "ok", "backend": "cpu", "models": 2})
    elif url.endswith("/v1/models"):
        body = _audio_cpp_catalog("whisper-small", "Whisper-Small")
    else:
        body = _json_bytes({"text": "hello"})
    return _ClosableAudioCppResponse(body)


@pytest.fixture(autouse=True)
def _reset_audio_cpp_process_cache() -> None:
    reset = getattr(audio_cpp, "reset_audio_cpp_discovery_cache", None)
    if reset is not None:
        reset()
    yield
    if reset is not None:
        reset()


def test_audio_cpp_config_defaults_are_disabled_and_local() -> None:
    cfg = audio_cpp.load_audio_cpp_config({}, env={})

    assert cfg == audio_cpp.AudioCppConfig(
        enabled=False,
        origin="http://127.0.0.1:8080",
        default_model=None,
        timeout_seconds=600.0,
    )


def test_audio_cpp_config_environment_overrides_supplied_mapping() -> None:
    cfg = audio_cpp.load_audio_cpp_config(
        {
            "audio_cpp_enabled": "false",
            "audio_cpp_base_url": "http://127.0.0.1:9000",
            "audio_cpp_default_model": "config-model",
            "audio_cpp_timeout_seconds": "10",
        },
        env={
            "STT_AUDIO_CPP_ENABLED": "true",
            "STT_AUDIO_CPP_BASE_URL": "https://EXAMPLE.com:443/",
            "STT_AUDIO_CPP_DEFAULT_MODEL": "environment-model",
            "STT_AUDIO_CPP_TIMEOUT_SECONDS": "30.5",
        },
    )

    assert cfg == audio_cpp.AudioCppConfig(
        enabled=True,
        origin="https://example.com",
        default_model="environment-model",
        timeout_seconds=30.5,
    )


@pytest.mark.parametrize(
    ("token", "expected"),
    [
        ("1", True),
        ("true", True),
        ("YES", True),
        (" y ", True),
        ("On", True),
        ("0", False),
        ("false", False),
        ("NO", False),
        (" n ", False),
        ("Off", False),
    ],
)
def test_audio_cpp_config_boolean_tokens_are_explicit_and_case_insensitive(
    token: str,
    expected: bool,
) -> None:
    cfg = audio_cpp.load_audio_cpp_config(
        {"audio_cpp_enabled": token},
        env={},
    )

    assert cfg.enabled is expected


@pytest.mark.parametrize("raw", ["", " ", "maybe", "2", "enabled", True, 0])
def test_audio_cpp_config_rejects_invalid_explicit_boolean_values(
    raw: object,
) -> None:
    with pytest.raises(
        STTExecutionUnsupportedError,
        match="audio.cpp enabled setting is invalid",
    ):
        audio_cpp.load_audio_cpp_config(
            {"audio_cpp_enabled": raw},
            env={},
        )


def test_audio_cpp_config_rejects_invalid_boolean_environment_override() -> None:
    with pytest.raises(
        STTExecutionUnsupportedError,
        match="audio.cpp enabled setting is invalid",
    ):
        audio_cpp.load_audio_cpp_config(
            {"audio_cpp_enabled": "true"},
            env={"STT_AUDIO_CPP_ENABLED": ""},
        )


@pytest.mark.parametrize(
    "raw",
    [
        True,
        False,
        "",
        " ",
        "not-a-number",
        "nan",
        "NaN",
        "inf",
        "-inf",
        float("nan"),
        float("inf"),
        float("-inf"),
        pytest.param(10**10000, id="overflowing-integer"),
        0,
        0.0,
        -1,
        "-0.5",
    ],
)
def test_audio_cpp_config_timeout_must_be_finite_and_positive(
    raw: object,
) -> None:
    with pytest.raises(
        STTExecutionUnsupportedError,
        match="audio.cpp timeout setting is invalid",
    ):
        audio_cpp.load_audio_cpp_config(
            {"audio_cpp_timeout_seconds": raw},
            env={},
        )


def test_audio_cpp_config_rejects_invalid_timeout_environment_override() -> None:
    with pytest.raises(
        STTExecutionUnsupportedError,
        match="audio.cpp timeout setting is invalid",
    ):
        audio_cpp.load_audio_cpp_config(
            {"audio_cpp_timeout_seconds": "30"},
            env={"STT_AUDIO_CPP_TIMEOUT_SECONDS": "NaN"},
        )


def test_audio_cpp_config_rejects_timeout_object_with_exploding_float() -> None:
    class ExplodingFloat:
        def __float__(self) -> float:
            raise RuntimeError("untrusted conversion ran")

    with pytest.raises(
        STTExecutionUnsupportedError,
        match="audio.cpp timeout setting is invalid",
    ):
        audio_cpp.load_audio_cpp_config(
            {"audio_cpp_timeout_seconds": ExplodingFloat()},
            env={},
        )


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("http://localhost:80", "http://localhost"),
        ("HTTP://LOCALHOST:8080/", "http://localhost:8080"),
        ("https://Example.COM:443/", "https://example.com"),
        ("https://example.com:8443", "https://example.com:8443"),
        ("http://[0:0:0:0:0:0:0:1]:8080/", "http://[::1]:8080"),
    ],
)
def test_audio_cpp_origin_is_canonicalized(
    raw: str,
    expected: str,
) -> None:
    cfg = audio_cpp.load_audio_cpp_config(
        {"audio_cpp_base_url": raw},
        env={},
    )

    assert cfg.origin == expected


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "localhost:8080",
        "ftp://localhost:8080",
        "http:///missing-authority",
        "http://user@localhost:8080",
        "http://user:password@localhost:8080",
        "http://localhost:8080/api",
        "http://localhost:8080/.",
        "http://localhost:8080/./",
        "http://localhost:8080/a/..",
        "http://localhost:8080?query=yes",
        "http://localhost:8080#fragment",
        "http://localhost:",
        "http://localhost:0",
        "http://localhost:65536",
        "http://localhost:not-a-port",
        "http://local%68ost:8080",
        "http://2130706433:8080",
        "http://127.1:8080",
        "http://0x7f000001:8080",
        "http://0177.0.0.1:8080",
        "http://localhost:8080\\evil",
        "http:\\\\localhost:8080",
        "http://localhost:8080/\n",
        "http://localhost:8080/\x7f",
    ],
)
def test_audio_cpp_origin_rejects_noncanonical_or_ambiguous_values(
    raw: str,
) -> None:
    with pytest.raises(
        STTExecutionUnsupportedError,
        match="audio.cpp origin is invalid",
    ):
        audio_cpp.load_audio_cpp_config(
            {"audio_cpp_base_url": raw},
            env={},
        )


def test_audio_cpp_origin_environment_override_is_validated() -> None:
    with pytest.raises(
        STTExecutionUnsupportedError,
        match="audio.cpp origin is invalid",
    ):
        audio_cpp.load_audio_cpp_config(
            {"audio_cpp_base_url": "http://127.0.0.1:8080"},
            env={"STT_AUDIO_CPP_BASE_URL": "http://127.0.0.1:8080/api"},
        )


@pytest.mark.parametrize(
    "raw",
    [
        " http://localhost:8080",
        "http://localhost:8080 ",
        "http://localhost:8080/ ",
    ],
)
def test_audio_cpp_origin_rejects_boundary_whitespace_in_mapping(
    raw: str,
) -> None:
    with pytest.raises(
        STTExecutionUnsupportedError,
        match="audio.cpp origin is invalid",
    ):
        audio_cpp.load_audio_cpp_config(
            {"audio_cpp_base_url": raw},
            env={},
        )


@pytest.mark.parametrize(
    "raw",
    [
        " http://localhost:8080",
        "http://localhost:8080 ",
        "http://localhost:8080/ ",
    ],
)
def test_audio_cpp_origin_rejects_boundary_whitespace_in_environment(
    raw: str,
) -> None:
    with pytest.raises(
        STTExecutionUnsupportedError,
        match="audio.cpp origin is invalid",
    ):
        audio_cpp.load_audio_cpp_config(
            {"audio_cpp_base_url": "http://127.0.0.1:8080"},
            env={"STT_AUDIO_CPP_BASE_URL": raw},
        )


def test_audio_cpp_origin_rejects_bracketed_ipvfuture_authority() -> None:
    with pytest.raises(
        STTExecutionUnsupportedError,
        match="audio.cpp origin is invalid",
    ):
        audio_cpp.load_audio_cpp_config(
            {"audio_cpp_base_url": "http://[v1.localhost]:8080"},
            env={},
        )


def test_audio_cpp_origin_accepts_maximum_length_dns_hostname() -> None:
    hostname = ".".join(("a" * 63, "b" * 63, "c" * 63, "d" * 61))
    assert len(hostname) == 253

    cfg = audio_cpp.load_audio_cpp_config(
        {"audio_cpp_base_url": f"https://{hostname}"},
        env={},
    )

    assert cfg.origin == f"https://{hostname}"


def test_audio_cpp_origin_rejects_dns_hostname_over_253_characters() -> None:
    hostname = ".".join(("a" * 63, "b" * 63, "c" * 63, "d" * 62))
    assert len(hostname) == 254

    with pytest.raises(
        STTExecutionUnsupportedError,
        match="audio.cpp origin is invalid",
    ):
        audio_cpp.load_audio_cpp_config(
            {"audio_cpp_base_url": f"https://{hostname}"},
            env={},
        )


def test_audio_cpp_canonical_origin_routes_have_single_slashes() -> None:
    cfg = audio_cpp.load_audio_cpp_config(
        {"audio_cpp_base_url": "HTTP://LOCALHOST:80/"},
        env={},
    )

    for path in ("/health", "/v1/models", "/v1/audio/transcriptions"):
        route = f"{cfg.origin}{path}"
        assert "//" not in route.partition("://")[2]


@pytest.mark.parametrize("default_model", ["", " ", "\t", "\n"])
def test_audio_cpp_config_allows_empty_default_model_until_requested(
    default_model: str,
) -> None:
    cfg = audio_cpp.load_audio_cpp_config(
        {"audio_cpp_default_model": default_model},
        env={},
    )

    assert cfg.default_model is None
    with pytest.raises(
        STTExecutionUnsupportedError,
        match="audio.cpp model is required",
    ):
        audio_cpp.normalize_audio_cpp_model(None, default_model=cfg.default_model)


@pytest.mark.parametrize(
    ("selector", "expected_model"),
    [
        ("audio-cpp:whisper-small", "whisper-small"),
        ("audiocpp:whisper-small", "whisper-small"),
        ("audio_cpp:whisper-small", "whisper-small"),
        ("AUDIO-CPP:whisper-small", "whisper-small"),
        ("AUDIOCPP:whisper-small", "whisper-small"),
        ("AUDIO_CPP:whisper-small", "whisper-small"),
        ("AUDIO-CPP:Whisper-Small", "Whisper-Small"),
    ],
)
def test_audio_cpp_selector_prefixes_strip_to_exact_server_model(
    selector: str,
    expected_model: str,
) -> None:
    assert (
        audio_cpp.normalize_audio_cpp_model(
            selector,
            default_model="configured-model",
        )
        == expected_model
    )


@pytest.mark.parametrize(
    "selector",
    [
        "audio-cpp",
        "audiocpp",
        "audio_cpp",
        "AUDIO-CPP",
        "AUDIOCPP",
        "AUDIO_CPP",
        None,
    ],
)
def test_audio_cpp_exact_selector_uses_configured_default(
    selector: str | None,
) -> None:
    assert (
        audio_cpp.normalize_audio_cpp_model(
            selector,
            default_model="configured-model",
        )
        == "configured-model"
    )


def test_audio_cpp_selector_preserves_safe_unprefixed_model() -> None:
    assert (
        audio_cpp.normalize_audio_cpp_model(
            "org/whisper-small+q5",
            default_model="configured-model",
        )
        == "org/whisper-small+q5"
    )


@pytest.mark.parametrize(
    "selector",
    [" audio-cpp", "audio-cpp ", "\taudiocpp", "audio_cpp\n"],
)
def test_audio_cpp_selector_rejects_boundary_altered_exact_selectors(
    selector: str,
) -> None:
    with pytest.raises(
        STTExecutionUnsupportedError,
        match="audio.cpp model is invalid",
    ):
        audio_cpp.normalize_audio_cpp_model(
            selector,
            default_model="configured-model",
        )


@pytest.mark.parametrize(
    "model",
    [
        "",
        " ",
        "audio-cpp:",
        "audiocpp:",
        "audio_cpp:",
        "http://example.com/model",
        "https:example.com/model",
        "//example.com/model",
        "/absolute/model",
        "./relative-model",
        "../relative-model",
        "model?query",
        "model#fragment",
        "user@model",
        "model name",
        "model\\name",
        "\tmodel",
        "model\n",
        " model ",
        " audio-cpp",
        "audio-cpp ",
        "audio-cpp:model\n",
        "audiocpp:\tmodel",
        "audio_cpp:model ",
        "model\nname",
        "model\x7fname",
        "org/nested/model",
    ],
)
def test_audio_cpp_selector_rejects_unsafe_or_empty_required_model_ids(
    model: str,
) -> None:
    with pytest.raises(
        STTExecutionUnsupportedError,
        match="audio.cpp model is invalid|audio.cpp model is required",
    ):
        audio_cpp.normalize_audio_cpp_model(model, default_model=None)


@pytest.mark.parametrize(
    "default_model",
    [
        "https://example.com/model",
        "\tmodel",
        "model\n",
        " model ",
    ],
)
def test_audio_cpp_config_rejects_unsafe_nonempty_default_model(
    default_model: str,
) -> None:
    with pytest.raises(
        STTExecutionUnsupportedError,
        match="audio.cpp model is invalid",
    ):
        audio_cpp.load_audio_cpp_config(
            {"audio_cpp_default_model": default_model},
            env={},
        )


def test_audio_cpp_fixture_provenance_is_exact() -> None:
    fixture = _audio_cpp_http_fixture()

    assert fixture["_provenance"] == {
        "repository": "https://github.com/0xShug0/audio.cpp",
        "commit": "10287cb60e71c12177b6bbbc70726950a9c7e29a",
        "contract": "audio_cpp_http_v1",
    }


def test_audio_cpp_fixture_health_is_valid() -> None:
    fixture = _audio_cpp_http_fixture()

    backend = audio_cpp.parse_audio_cpp_health(_json_bytes(fixture["health"]))

    assert backend == "cpu"


def test_audio_cpp_fixture_catalog_is_valid() -> None:
    fixture = _audio_cpp_http_fixture()

    discovery = audio_cpp.parse_audio_cpp_catalog(
        _json_bytes(fixture["models"]),
        backend="cpu",
        model_id="whisper-small",
    )

    assert discovery == audio_cpp.AudioCppDiscovery(
        backend="cpu",
        model_id="whisper-small",
        family="whisper",
        mode="offline",
    )


def test_audio_cpp_fixture_transcription_contract_is_valid() -> None:
    fixture = _audio_cpp_http_fixture()

    assert audio_cpp.parse_audio_cpp_transcription(_json_bytes(fixture["transcription"])) == "fixture transcript"


def test_audio_cpp_health_allows_unknown_fields() -> None:
    body = _json_bytes(
        {
            "status": "ok",
            "backend": "cpu",
            "models": 1,
            "future": {"nested": True},
        }
    )

    assert audio_cpp.parse_audio_cpp_health(body) == "cpu"


def test_audio_cpp_catalog_allows_unknown_fields() -> None:
    body = _json_bytes(
        {
            "object": "list",
            "future": True,
            "data": [
                {
                    "id": "whisper-small",
                    "family": "whisper",
                    "task": "asr",
                    "mode": "offline",
                    "future": {"nested": True},
                }
            ],
        }
    )

    discovery = audio_cpp.parse_audio_cpp_catalog(
        body,
        backend="cpu",
        model_id="whisper-small",
    )

    assert discovery.model_id == "whisper-small"


def test_audio_cpp_transcription_contract_allows_unknown_fields() -> None:
    body = _json_bytes(
        {
            "text": "accepted",
            "timing": {"total_ms": 1, "future": True},
            "future": ["value"],
        }
    )

    assert audio_cpp.parse_audio_cpp_transcription(body) == "accepted"


def test_audio_cpp_catalog_model_matching_is_exact() -> None:
    body = _json_bytes(
        {
            "object": "list",
            "data": [
                {
                    "id": "Whisper-Small",
                    "family": "whisper",
                    "task": "asr",
                    "mode": "offline",
                },
                {
                    "id": "whisper-small",
                    "family": "whisper",
                    "task": "asr",
                    "mode": "streaming",
                },
            ],
        }
    )

    discovery = audio_cpp.parse_audio_cpp_catalog(
        body,
        backend="cpu",
        model_id="whisper-small",
    )

    assert discovery.mode == "streaming"


@pytest.mark.parametrize("mode", ["offline", "streaming"])
def test_audio_cpp_catalog_accepts_supported_asr_modes(mode: str) -> None:
    body = _json_bytes(
        {
            "object": "list",
            "data": [
                {
                    "id": "whisper-small",
                    "family": "whisper",
                    "task": "asr",
                    "mode": mode,
                }
            ],
        }
    )

    discovery = audio_cpp.parse_audio_cpp_catalog(
        body,
        backend="cpu",
        model_id="whisper-small",
    )

    assert discovery.mode == mode


@pytest.mark.parametrize(
    "entry",
    [
        {
            "id": "whisper-small",
            "family": "whisper",
            "task": "tts",
            "mode": "offline",
        },
        {"family": "whisper", "task": "asr", "mode": "offline"},
    ],
)
def test_audio_cpp_catalog_rejects_non_asr_or_missing_model_ids(
    entry: dict[str, str],
) -> None:
    body = _json_bytes({"object": "list", "data": [entry]})

    with pytest.raises(STTExecutionUnsupportedError):
        audio_cpp.parse_audio_cpp_catalog(
            body,
            backend="cpu",
            model_id="whisper-small",
        )


def test_audio_cpp_catalog_rejects_duplicate_model_ids() -> None:
    entry = {
        "id": "whisper-small",
        "family": "whisper",
        "task": "asr",
        "mode": "offline",
    }
    body = _json_bytes({"object": "list", "data": [entry, entry]})

    with pytest.raises(STTExecutionUnsupportedError):
        audio_cpp.parse_audio_cpp_catalog(
            body,
            backend="cpu",
            model_id="whisper-small",
        )


@pytest.mark.parametrize(
    ("parser", "body"),
    [
        pytest.param(
            lambda body: audio_cpp.parse_audio_cpp_health(body),
            b'{"status":"warming","status":"ok","backend":"cpu","models":1}',
            id="health",
        ),
        pytest.param(
            lambda body: audio_cpp.parse_audio_cpp_catalog(body, backend="cpu", model_id="whisper-small"),
            (
                b'{"object":"invalid","object":"list","data":['
                b'{"id":"whisper-small","family":"whisper",'
                b'"task":"asr","mode":"offline"}]}'
            ),
            id="catalog",
        ),
        pytest.param(
            lambda body: audio_cpp.parse_audio_cpp_transcription(body),
            b'{"text":{"server":"raw"},"text":"accepted"}',
            id="transcription",
        ),
    ],
)
def test_audio_cpp_health_catalog_and_transcription_contract_reject_duplicate_json_keys(
    parser: Callable[[bytes], object],
    body: bytes,
) -> None:
    with pytest.raises(STTExecutionUnsupportedError):
        parser(body)


def test_audio_cpp_transcription_contract_rejects_nested_duplicate_json_keys() -> None:
    body = b'{"text":"accepted","future":{"duplicate":1,"duplicate":2}}'

    with pytest.raises(STTExecutionUnsupportedError):
        audio_cpp.parse_audio_cpp_transcription(body)


@pytest.mark.parametrize(
    "parser",
    [
        pytest.param(lambda body: audio_cpp.parse_audio_cpp_health(body), id="health"),
        pytest.param(
            lambda body: audio_cpp.parse_audio_cpp_catalog(
                body,
                backend="cpu",
                model_id="whisper-small",
            ),
            id="catalog",
        ),
        pytest.param(
            lambda body: audio_cpp.parse_audio_cpp_transcription(body),
            id="transcription",
        ),
    ],
)
@pytest.mark.parametrize("body", [b"\xff", b"[]", b'"value"', b"null"])
def test_audio_cpp_health_catalog_and_transcription_contract_reject_invalid_input(
    parser: Callable[[bytes], object],
    body: bytes,
) -> None:
    with pytest.raises(STTExecutionUnsupportedError):
        parser(body)


@pytest.mark.parametrize(
    "parser",
    [
        pytest.param(lambda body: audio_cpp.parse_audio_cpp_health(body), id="health"),
        pytest.param(
            lambda body: audio_cpp.parse_audio_cpp_catalog(
                body,
                backend="cpu",
                model_id="whisper-small",
            ),
            id="catalog",
        ),
        pytest.param(
            lambda body: audio_cpp.parse_audio_cpp_transcription(body),
            id="transcription",
        ),
    ],
)
def test_audio_cpp_health_catalog_and_transcription_contract_reject_oversized_bodies(
    parser: Callable[[bytes], object],
) -> None:
    body = b"x" * (audio_cpp.MAX_AUDIO_CPP_RESPONSE_BYTES + 1)

    with pytest.raises(STTExecutionUnsupportedError):
        parser(body)


@pytest.mark.parametrize(
    "body",
    [
        {"status": "warming", "backend": "cpu", "models": 1},
        {"status": "ok", "backend": "../cpu", "models": 1},
        {"status": "ok", "backend": "cpu", "models": -1},
        {"status": "ok", "backend": "cpu", "models": True},
    ],
)
def test_audio_cpp_health_rejects_unsafe_values(body: dict[str, object]) -> None:
    with pytest.raises(STTExecutionUnsupportedError):
        audio_cpp.parse_audio_cpp_health(_json_bytes(body))


def test_audio_cpp_health_model_count_accepts_explicit_upper_bound() -> None:
    body = {
        "status": "ok",
        "backend": "cpu",
        "models": audio_cpp.MAX_AUDIO_CPP_HEALTH_MODELS,
    }

    assert audio_cpp.parse_audio_cpp_health(_json_bytes(body)) == "cpu"


def test_audio_cpp_health_model_count_rejects_above_upper_bound() -> None:
    body = {
        "status": "ok",
        "backend": "cpu",
        "models": audio_cpp.MAX_AUDIO_CPP_HEALTH_MODELS + 1,
    }

    with pytest.raises(STTExecutionUnsupportedError):
        audio_cpp.parse_audio_cpp_health(_json_bytes(body))


def test_audio_cpp_health_pathological_integer_has_bounded_domain_error() -> None:
    body = b'{"status":"ok","backend":"cpu","models":' + (b"9" * 5000) + b"}"

    with pytest.raises(
        STTExecutionUnsupportedError,
        match="^audio.cpp health response is invalid$",
    ):
        audio_cpp.parse_audio_cpp_health(body)


def test_audio_cpp_health_accepts_json_integer_at_digit_limit() -> None:
    body = (
        b'{"status":"ok","backend":"cpu","models":1,"future":'
        + (b"9" * audio_cpp.MAX_AUDIO_CPP_JSON_INTEGER_DIGITS)
        + b"}"
    )

    assert audio_cpp.parse_audio_cpp_health(body) == "cpu"


def test_audio_cpp_health_rejects_json_integer_above_digit_limit() -> None:
    body = (
        b'{"status":"ok","backend":"cpu","models":1,"future":'
        + (b"9" * (audio_cpp.MAX_AUDIO_CPP_JSON_INTEGER_DIGITS + 1))
        + b"}"
    )

    with pytest.raises(
        STTExecutionUnsupportedError,
        match="^audio.cpp health response is invalid$",
    ):
        audio_cpp.parse_audio_cpp_health(body)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("id", "../whisper"),
        ("family", "whisper family"),
        ("mode", "batch"),
    ],
)
def test_audio_cpp_catalog_rejects_unsafe_model_values(
    field: str,
    value: str,
) -> None:
    entry = {
        "id": "whisper-small",
        "family": "whisper",
        "task": "asr",
        "mode": "offline",
    }
    entry[field] = value
    body = _json_bytes({"object": "list", "data": [entry]})

    with pytest.raises(STTExecutionUnsupportedError):
        audio_cpp.parse_audio_cpp_catalog(
            body,
            backend="cpu",
            model_id="whisper-small",
        )


def test_audio_cpp_catalog_rejects_excessive_entries() -> None:
    entries = [
        {
            "id": f"model-{index}",
            "family": "whisper",
            "task": "asr",
            "mode": "offline",
        }
        for index in range(audio_cpp.MAX_AUDIO_CPP_CATALOG_ENTRIES + 1)
    ]

    with pytest.raises(STTExecutionUnsupportedError):
        audio_cpp.parse_audio_cpp_catalog(
            _json_bytes({"object": "list", "data": entries}),
            backend="cpu",
            model_id="model-0",
        )


def test_audio_cpp_catalog_accepts_exact_entry_limit() -> None:
    entries = [
        {
            "id": f"model-{index}",
            "family": "whisper",
            "task": "asr",
            "mode": "offline",
        }
        for index in range(audio_cpp.MAX_AUDIO_CPP_CATALOG_ENTRIES)
    ]
    selected_model = entries[-1]["id"]

    discovery = audio_cpp.parse_audio_cpp_catalog(
        _json_bytes({"object": "list", "data": entries}),
        backend="cpu",
        model_id=selected_model,
    )

    assert discovery.model_id == selected_model


def test_audio_cpp_catalog_accepts_unrelated_non_asr_model() -> None:
    body = {
        "object": "list",
        "data": [
            {
                "id": "speaker",
                "family": "kokoro",
                "task": "tts",
                "mode": "offline",
            },
            {
                "id": "whisper-small",
                "family": "whisper",
                "task": "asr",
                "mode": "streaming",
            },
        ],
    }

    discovery = audio_cpp.parse_audio_cpp_catalog(
        _json_bytes(body),
        backend="cpu",
        model_id="whisper-small",
    )

    assert discovery.model_id == "whisper-small"


@pytest.mark.parametrize("text", ["", " ", "\t\n"])
def test_audio_cpp_transcription_contract_accepts_empty_or_whitespace_text(
    text: str,
) -> None:
    assert audio_cpp.parse_audio_cpp_transcription(_json_bytes({"text": text})) == text


def test_audio_cpp_transcription_contract_rejects_overlong_text() -> None:
    text = "x" * (audio_cpp.MAX_AUDIO_CPP_TRANSCRIPT_CHARS + 1)

    with pytest.raises(STTExecutionUnsupportedError):
        audio_cpp.parse_audio_cpp_transcription(_json_bytes({"text": text}))


def test_audio_cpp_transcription_contract_accepts_exact_text_limit() -> None:
    text = "x" * audio_cpp.MAX_AUDIO_CPP_TRANSCRIPT_CHARS

    assert audio_cpp.parse_audio_cpp_transcription(_json_bytes({"text": text})) == text


def test_audio_cpp_transcription_contract_errors_never_echo_server_text() -> None:
    secret = "server-secret-value"
    body = _json_bytes({"text": {"raw": secret}})

    with pytest.raises(STTExecutionUnsupportedError) as exc_info:
        audio_cpp.parse_audio_cpp_transcription(body)

    assert secret not in str(exc_info.value)


@pytest.mark.parametrize("suffix", [".wav", ".WAV"])
def test_audio_cpp_wav_accepts_valid_pcm_and_rewinds_upload(
    tmp_path: Path,
    suffix: str,
) -> None:
    path = tmp_path / f"sample{suffix}"
    _write_audio_cpp_pcm_wav(path)

    with audio_cpp.open_audio_cpp_wav(path, base_dir=tmp_path) as upload:
        assert upload.tell() == 0
        assert upload.read(4) == b"RIFF"


def test_audio_cpp_wav_validation_uses_standard_wave_reader(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "sample.wav"
    _write_audio_cpp_pcm_wav(path)
    original_wave_open = wave.open
    opened: list[tuple[object, str | None]] = []

    def tracking_wave_open(
        file: object,
        mode: str | None = None,
    ) -> wave.Wave_read:
        opened.append((file, mode))
        return original_wave_open(file, mode)

    monkeypatch.setattr(wave, "open", tracking_wave_open)

    with audio_cpp.open_audio_cpp_wav(path, base_dir=tmp_path) as upload:
        assert opened == [(upload, "rb")]


def test_audio_cpp_wav_accepts_relative_path_under_relative_base_dir(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    base_dir = Path("uploads")
    base_dir.mkdir()
    path = Path("sample.wav")
    _write_audio_cpp_pcm_wav(base_dir / path)

    with audio_cpp.open_audio_cpp_wav(path, base_dir=base_dir) as upload:
        assert upload.read(4) == b"RIFF"


def test_audio_cpp_wav_rejects_renamed_non_wav(tmp_path: Path) -> None:
    path = tmp_path / "renamed.wav"
    path.write_bytes(b"this is not wave audio")

    with pytest.raises(STTExecutionUnsupportedError):
        audio_cpp.open_audio_cpp_wav(path, base_dir=tmp_path)


def test_audio_cpp_wav_rejects_truncated_riff(tmp_path: Path) -> None:
    path = tmp_path / "truncated.wav"
    path.write_bytes(b"RIFF\x20\x00\x00\x00WAVE")

    with pytest.raises(STTExecutionUnsupportedError):
        audio_cpp.open_audio_cpp_wav(path, base_dir=tmp_path)


def test_audio_cpp_wav_rejects_compressed_or_non_pcm(tmp_path: Path) -> None:
    path = tmp_path / "float.wav"
    _write_audio_cpp_non_pcm_wav(path)

    with pytest.raises(STTExecutionUnsupportedError):
        audio_cpp.open_audio_cpp_wav(path, base_dir=tmp_path)


def test_audio_cpp_wav_rejects_wrong_suffix(tmp_path: Path) -> None:
    path = tmp_path / "sample.mp3"
    _write_audio_cpp_pcm_wav(path)

    with pytest.raises(STTExecutionUnsupportedError):
        audio_cpp.open_audio_cpp_wav(path, base_dir=tmp_path)


def test_audio_cpp_wav_rejects_directory(tmp_path: Path) -> None:
    with pytest.raises(STTExecutionUnsupportedError):
        audio_cpp.open_audio_cpp_wav(tmp_path, base_dir=tmp_path)


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFO is unavailable")
def test_audio_cpp_wav_rejects_fifo_before_any_open_attempt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "audio.wav"
    os.mkfifo(path)

    def fail_if_opened(
        _path: Path,
        _base_dir: Path,
        *,
        mode: str,
    ) -> BinaryIO:
        raise AssertionError(f"FIFO reached open boundary with mode {mode}")

    monkeypatch.setattr(audio_cpp, "open_safe_local_path", fail_if_opened)

    with pytest.raises(STTExecutionUnsupportedError):
        audio_cpp.open_audio_cpp_wav(path, base_dir=tmp_path)


def test_audio_cpp_wav_rejects_symlink_path_escape(tmp_path: Path) -> None:
    base_dir = tmp_path / "authorized"
    base_dir.mkdir()
    outside = tmp_path / "outside.pcm"
    _write_audio_cpp_pcm_wav(outside)
    link = base_dir / "escape.wav"
    link.symlink_to(outside)

    with pytest.raises(STTExecutionUnsupportedError):
        audio_cpp.open_audio_cpp_wav(link, base_dir=base_dir)


def test_audio_cpp_wav_rejects_path_outside_base_dir(tmp_path: Path) -> None:
    base_dir = tmp_path / "authorized"
    base_dir.mkdir()
    outside = tmp_path / "outside.wav"
    _write_audio_cpp_pcm_wav(outside)

    with pytest.raises(STTExecutionUnsupportedError):
        audio_cpp.open_audio_cpp_wav(outside, base_dir=base_dir)


def test_audio_cpp_wav_rejects_opened_file_identity_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    requested = tmp_path / "requested.wav"
    replacement = tmp_path / "replacement.wav"
    _write_audio_cpp_pcm_wav(requested)
    _write_audio_cpp_pcm_wav(replacement)
    replacement_handle: BinaryIO | None = None

    def open_replacement(
        _path: Path,
        _base_dir: Path,
        *,
        mode: str,
    ) -> BinaryIO:
        nonlocal replacement_handle
        assert mode == "rb"
        replacement_handle = replacement.open("rb")
        return replacement_handle

    monkeypatch.setattr(audio_cpp, "open_safe_local_path", open_replacement)

    with pytest.raises(STTExecutionUnsupportedError):
        audio_cpp.open_audio_cpp_wav(requested, base_dir=tmp_path)
    assert replacement_handle is not None
    assert replacement_handle.closed


def test_audio_cpp_wav_rejects_missing_path(tmp_path: Path) -> None:
    with pytest.raises(STTExecutionUnsupportedError):
        audio_cpp.open_audio_cpp_wav(
            tmp_path / "missing.wav",
            base_dir=tmp_path,
        )


def test_audio_cpp_wav_rejects_declared_late_truncation(tmp_path: Path) -> None:
    path = tmp_path / "late-truncation.wav"
    _write_audio_cpp_pcm_wav(path, frames=8192)
    original_size = path.stat().st_size
    with path.open("r+b") as wav_file:
        wav_file.truncate(original_size - 1024)
        wav_file.seek(4)
        wav_file.write(struct.pack("<I", original_size - 1024 - 8))
    with wave.open(str(path), "rb") as wav_file:
        assert wav_file.readframes(1)

    with pytest.raises(STTExecutionUnsupportedError):
        audio_cpp.open_audio_cpp_wav(path, base_dir=tmp_path)


def test_audio_cpp_http_routes_append_only_fixed_canonical_paths() -> None:
    assert audio_cpp.audio_cpp_routes("http://127.0.0.1:18080") == (
        "http://127.0.0.1:18080/health",
        "http://127.0.0.1:18080/v1/models",
        "http://127.0.0.1:18080/v1/audio/transcriptions",
    )


@pytest.mark.parametrize(
    "origin",
    [
        "HTTP://127.0.0.1:18080",
        "http://127.0.0.1:80",
        "http://127.0.0.1:18080/",
        "http://127.0.0.1:18080/api",
        "http://user@127.0.0.1:18080",
        "http://127.0.0.1:18080?secret=yes",
    ],
)
def test_audio_cpp_http_routes_fail_closed_for_noncanonical_origin(
    origin: str,
) -> None:
    with pytest.raises(
        STTExecutionUnsupportedError,
        match="^audio.cpp origin is invalid$",
    ):
        audio_cpp.audio_cpp_routes(origin)


@pytest.mark.asyncio
async def test_audio_cpp_http_first_request_discovers_then_warm_cache_skips_discovery(
    tmp_path: Path,
) -> None:
    request = _audio_cpp_request_kwargs(tmp_path)
    calls: list[dict[str, Any]] = []
    responses: list[_ClosableAudioCppResponse] = []

    async def fake_afetch(**kwargs: Any) -> _ClosableAudioCppResponse:
        calls.append(kwargs)
        response = _successful_audio_cpp_response(kwargs["url"])
        responses.append(response)
        return response

    first = await audio_cpp.transcribe_audio_cpp_async(
        **request,
        afetch_fn=fake_afetch,
    )
    second = await audio_cpp.transcribe_audio_cpp_async(
        **request,
        afetch_fn=fake_afetch,
    )

    assert [call["url"] for call in calls] == [
        "http://127.0.0.1:18080/health",
        "http://127.0.0.1:18080/v1/models",
        "http://127.0.0.1:18080/v1/audio/transcriptions",
        "http://127.0.0.1:18080/v1/audio/transcriptions",
    ]
    assert first.artifact["text"] == second.artifact["text"] == "hello"
    assert all(response.closed for response in responses)


@pytest.mark.asyncio
async def test_audio_cpp_discovery_cache_singleflights_same_loop_concurrent_first_use(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    request = _audio_cpp_request_kwargs(tmp_path)
    health_entered = asyncio.Event()
    release_health = asyncio.Event()
    follower_attached = asyncio.Event()
    calls: list[str] = []
    original_wrap_future = asyncio.wrap_future

    def tracking_wrap_future(*args: Any, **kwargs: Any) -> asyncio.Future[Any]:
        follower_attached.set()
        return original_wrap_future(*args, **kwargs)

    monkeypatch.setattr(audio_cpp.asyncio, "wrap_future", tracking_wrap_future)

    async def fake_afetch(**kwargs: Any) -> _ClosableAudioCppResponse:
        url = kwargs["url"]
        calls.append(url)
        if url.endswith("/health"):
            health_entered.set()
            await asyncio.wait_for(release_health.wait(), timeout=2)
        return _successful_audio_cpp_response(url)

    first = asyncio.create_task(
        audio_cpp.transcribe_audio_cpp_async(
            **request,
            afetch_fn=fake_afetch,
        )
    )
    await asyncio.wait_for(health_entered.wait(), timeout=2)
    second = asyncio.create_task(
        audio_cpp.transcribe_audio_cpp_async(
            **request,
            afetch_fn=fake_afetch,
        )
    )
    await asyncio.wait_for(follower_attached.wait(), timeout=2)
    release_health.set()

    results = await asyncio.wait_for(
        asyncio.gather(first, second),
        timeout=2,
    )

    assert len(results) == 2
    assert sum(url.endswith("/health") for url in calls) == 1
    assert sum(url.endswith("/v1/models") for url in calls) == 1
    assert sum(url.endswith("/v1/audio/transcriptions") for url in calls) == 2


def test_audio_cpp_cache_singleflight_is_cross_loop_and_nonblocking(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    request = _audio_cpp_request_kwargs(tmp_path)
    start = threading.Barrier(3)
    release_health = threading.Event()
    health_entered = threading.Event()
    follower_attached = threading.Event()
    call_lock = threading.Lock()
    calls: list[str] = []
    loop_heartbeats = [threading.Event(), threading.Event()]
    results: list[SttTranscriptionOutcome | BaseException | None] = [None, None]
    original_wrap_future = asyncio.wrap_future

    def tracking_wrap_future(*args: Any, **kwargs: Any) -> asyncio.Future[Any]:
        follower_attached.set()
        return original_wrap_future(*args, **kwargs)

    monkeypatch.setattr(audio_cpp.asyncio, "wrap_future", tracking_wrap_future)

    async def fake_afetch(**kwargs: Any) -> _ClosableAudioCppResponse:
        url = kwargs["url"]
        with call_lock:
            calls.append(url)
        if url.endswith("/health"):
            health_entered.set()
            await asyncio.to_thread(release_health.wait)
        return _successful_audio_cpp_response(url)

    async def run_caller(index: int) -> SttTranscriptionOutcome:
        task = asyncio.create_task(
            audio_cpp.transcribe_audio_cpp_async(
                **request,
                afetch_fn=fake_afetch,
            )
        )
        asyncio.get_running_loop().call_soon(loop_heartbeats[index].set)
        return await task

    def worker(index: int) -> None:
        try:
            start.wait(timeout=2)
            results[index] = asyncio.run(run_caller(index))
        except BaseException as exc:  # noqa: BLE001 - surface any thread failure
            results[index] = exc

    threads = [
        threading.Thread(target=worker, args=(index,), daemon=True)
        for index in range(2)
    ]
    try:
        for thread in threads:
            thread.start()
        start.wait(timeout=2)
        assert health_entered.wait(timeout=2)
        assert follower_attached.wait(timeout=2)
        assert all(heartbeat.wait(timeout=2) for heartbeat in loop_heartbeats)
    finally:
        release_health.set()
        for thread in threads:
            thread.join(timeout=3)

    assert all(not thread.is_alive() for thread in threads)
    assert all(isinstance(result, SttTranscriptionOutcome) for result in results)
    assert sum(url.endswith("/health") for url in calls) == 1
    assert sum(url.endswith("/v1/models") for url in calls) == 1
    assert sum(url.endswith("/v1/audio/transcriptions") for url in calls) == 2


@pytest.mark.asyncio
async def test_audio_cpp_discovery_cache_follower_cancellation_is_shielded(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    request = _audio_cpp_request_kwargs(tmp_path)
    health_entered = asyncio.Event()
    release_health = asyncio.Event()
    follower_attached = asyncio.Event()
    calls: list[str] = []
    original_wrap_future = asyncio.wrap_future

    def tracking_wrap_future(*args: Any, **kwargs: Any) -> asyncio.Future[Any]:
        follower_attached.set()
        return original_wrap_future(*args, **kwargs)

    monkeypatch.setattr(audio_cpp.asyncio, "wrap_future", tracking_wrap_future)

    async def fake_afetch(**kwargs: Any) -> _ClosableAudioCppResponse:
        url = kwargs["url"]
        calls.append(url)
        if url.endswith("/health"):
            health_entered.set()
            await asyncio.wait_for(release_health.wait(), timeout=2)
        return _successful_audio_cpp_response(url)

    leader = asyncio.create_task(
        audio_cpp.transcribe_audio_cpp_async(
            **request,
            afetch_fn=fake_afetch,
        )
    )
    await asyncio.wait_for(health_entered.wait(), timeout=2)
    follower = asyncio.create_task(
        audio_cpp.transcribe_audio_cpp_async(
            **request,
            afetch_fn=fake_afetch,
        )
    )
    await asyncio.wait_for(follower_attached.wait(), timeout=2)
    follower.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(follower, timeout=2)
    release_health.set()
    await asyncio.wait_for(leader, timeout=2)
    await asyncio.wait_for(
        audio_cpp.transcribe_audio_cpp_async(
            **request,
            afetch_fn=fake_afetch,
        ),
        timeout=2,
    )

    assert sum(url.endswith("/health") for url in calls) == 1
    assert sum(url.endswith("/v1/models") for url in calls) == 1
    assert sum(url.endswith("/v1/audio/transcriptions") for url in calls) == 2


@pytest.mark.asyncio
async def test_audio_cpp_cancelled_follower_consumes_later_leader_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    request = _audio_cpp_request_kwargs(tmp_path)
    health_entered = asyncio.Event()
    release_health = asyncio.Event()
    follower_attached = asyncio.Event()
    loop_errors: list[dict[str, object]] = []
    loop = asyncio.get_running_loop()
    previous_handler = loop.get_exception_handler()
    original_wrap_future = asyncio.wrap_future

    def tracking_wrap_future(*args: Any, **kwargs: Any) -> asyncio.Future[Any]:
        follower_attached.set()
        return original_wrap_future(*args, **kwargs)

    monkeypatch.setattr(audio_cpp.asyncio, "wrap_future", tracking_wrap_future)
    loop.set_exception_handler(lambda _loop, context: loop_errors.append(context))

    async def fake_afetch(**kwargs: Any) -> _ClosableAudioCppResponse:
        if kwargs["url"].endswith("/health"):
            health_entered.set()
            await release_health.wait()
            raise TimeoutError("server-secret-leader")
        return _successful_audio_cpp_response(kwargs["url"])

    leader = asyncio.create_task(
        audio_cpp.transcribe_audio_cpp_async(**request, afetch_fn=fake_afetch)
    )
    follower: asyncio.Task[SttTranscriptionOutcome] | None = None
    try:
        await asyncio.wait_for(health_entered.wait(), timeout=2)
        follower = asyncio.create_task(
            audio_cpp.transcribe_audio_cpp_async(**request, afetch_fn=fake_afetch)
        )
        await asyncio.wait_for(follower_attached.wait(), timeout=2)
        follower.cancel()
        with pytest.raises(asyncio.CancelledError):
            await follower
        follower = None
        release_health.set()
        with pytest.raises(
            STTTranscriptionError,
            match="^audio.cpp request failed$",
        ):
            await leader
        gc.collect()
        await asyncio.sleep(0)
        await asyncio.sleep(0)
    finally:
        release_health.set()
        if follower is not None and not follower.done():
            follower.cancel()
        if not leader.done():
            leader.cancel()
        await asyncio.gather(
            *(task for task in (leader, follower) if task is not None),
            return_exceptions=True,
        )
        loop.set_exception_handler(previous_handler)

    assert loop_errors == []


@pytest.mark.asyncio
async def test_audio_cpp_discovery_cache_key_uses_endpoint_id_and_exact_model(
    tmp_path: Path,
) -> None:
    calls: list[str] = []

    async def fake_afetch(**kwargs: Any) -> _ClosableAudioCppResponse:
        url = kwargs["url"]
        calls.append(url)
        if url.endswith("/health"):
            return _ClosableAudioCppResponse(_json_bytes({"status": "ok", "backend": "cpu", "models": 2}))
        if url.endswith("/v1/models"):
            return _ClosableAudioCppResponse(_audio_cpp_catalog("whisper-small", "Whisper-Small"))
        return _ClosableAudioCppResponse(_json_bytes({"text": "ok"}))

    first = _audio_cpp_request_kwargs(tmp_path, model_id="whisper-small")
    exact_case_variant = _audio_cpp_request_kwargs(
        tmp_path,
        model_id="Whisper-Small",
    )
    different_endpoint = _audio_cpp_request_kwargs(
        tmp_path,
        origin="http://127.0.0.1:18081",
        model_id="whisper-small",
    )
    for request in (
        first,
        exact_case_variant,
        different_endpoint,
        first,
    ):
        await audio_cpp.transcribe_audio_cpp_async(
            **request,
            afetch_fn=fake_afetch,
        )

    assert sum(url.endswith("/health") for url in calls) == 3
    assert sum(url.endswith("/v1/models") for url in calls) == 3
    assert sum(url.endswith("/v1/audio/transcriptions") for url in calls) == 4


@pytest.mark.asyncio
async def test_audio_cpp_discovery_cache_reset_clears_warm_entry(
    tmp_path: Path,
) -> None:
    request = _audio_cpp_request_kwargs(tmp_path)
    calls: list[str] = []

    async def fake_afetch(**kwargs: Any) -> _ClosableAudioCppResponse:
        calls.append(kwargs["url"])
        return _successful_audio_cpp_response(kwargs["url"])

    await audio_cpp.transcribe_audio_cpp_async(
        **request,
        afetch_fn=fake_afetch,
    )
    audio_cpp.reset_audio_cpp_discovery_cache()
    await audio_cpp.transcribe_audio_cpp_async(
        **request,
        afetch_fn=fake_afetch,
    )

    assert sum(url.endswith("/health") for url in calls) == 2
    assert sum(url.endswith("/v1/models") for url in calls) == 2


@pytest.mark.asyncio
async def test_audio_cpp_discovery_cache_reset_is_generation_safe_during_inflight(
    tmp_path: Path,
) -> None:
    request = _audio_cpp_request_kwargs(tmp_path)
    first_health_entered = asyncio.Event()
    release_first_health = asyncio.Event()
    calls: list[str] = []
    health_calls = 0

    async def fake_afetch(**kwargs: Any) -> _ClosableAudioCppResponse:
        nonlocal health_calls
        url = kwargs["url"]
        calls.append(url)
        if url.endswith("/health"):
            health_calls += 1
            if health_calls == 1:
                first_health_entered.set()
                await asyncio.wait_for(
                    release_first_health.wait(),
                    timeout=2,
                )
        return _successful_audio_cpp_response(url)

    old_leader = asyncio.create_task(
        audio_cpp.transcribe_audio_cpp_async(
            **request,
            afetch_fn=fake_afetch,
        )
    )
    await asyncio.wait_for(first_health_entered.wait(), timeout=2)
    audio_cpp.reset_audio_cpp_discovery_cache()
    new_leader = asyncio.create_task(
        audio_cpp.transcribe_audio_cpp_async(
            **request,
            afetch_fn=fake_afetch,
        )
    )
    await asyncio.wait_for(new_leader, timeout=2)
    release_first_health.set()
    await asyncio.wait_for(old_leader, timeout=2)
    await asyncio.wait_for(
        audio_cpp.transcribe_audio_cpp_async(
            **request,
            afetch_fn=fake_afetch,
        ),
        timeout=2,
    )

    assert sum(url.endswith("/health") for url in calls) == 2
    assert sum(url.endswith("/v1/models") for url in calls) == 2
    assert sum(url.endswith("/v1/audio/transcriptions") for url in calls) == 3


@pytest.mark.asyncio
async def test_audio_cpp_discovery_cache_stale_transcription_failure_preserves_new_generation(
    tmp_path: Path,
) -> None:
    request = _audio_cpp_request_kwargs(tmp_path)
    first_transcription_entered = asyncio.Event()
    release_first_transcription = asyncio.Event()
    calls: list[str] = []
    transcription_calls = 0

    async def fake_afetch(**kwargs: Any) -> _ClosableAudioCppResponse:
        nonlocal transcription_calls
        url = kwargs["url"]
        calls.append(url)
        if url.endswith("/v1/audio/transcriptions"):
            transcription_calls += 1
            if transcription_calls == 1:
                first_transcription_entered.set()
                await asyncio.wait_for(
                    release_first_transcription.wait(),
                    timeout=2,
                )
                raise TimeoutError("stale request failed")
        return _successful_audio_cpp_response(url)

    stale_request = asyncio.create_task(
        audio_cpp.transcribe_audio_cpp_async(
            **request,
            afetch_fn=fake_afetch,
        )
    )
    await asyncio.wait_for(first_transcription_entered.wait(), timeout=2)
    audio_cpp.reset_audio_cpp_discovery_cache()
    await asyncio.wait_for(
        audio_cpp.transcribe_audio_cpp_async(
            **request,
            afetch_fn=fake_afetch,
        ),
        timeout=2,
    )
    release_first_transcription.set()
    with pytest.raises(STTTranscriptionError):
        await asyncio.wait_for(stale_request, timeout=2)
    await asyncio.wait_for(
        audio_cpp.transcribe_audio_cpp_async(
            **request,
            afetch_fn=fake_afetch,
        ),
        timeout=2,
    )

    assert sum(url.endswith("/health") for url in calls) == 2
    assert sum(url.endswith("/v1/models") for url in calls) == 2
    assert transcription_calls == 3


@pytest.mark.asyncio
async def test_audio_cpp_http_requests_use_frozen_secure_no_retry_policy(
    tmp_path: Path,
) -> None:
    request = _audio_cpp_request_kwargs(tmp_path)
    calls: list[dict[str, Any]] = []

    async def fake_afetch(**kwargs: Any) -> _ClosableAudioCppResponse:
        calls.append(kwargs)
        return _successful_audio_cpp_response(kwargs["url"])

    await audio_cpp.transcribe_audio_cpp_async(
        **request,
        afetch_fn=fake_afetch,
    )

    assert [call["method"] for call in calls] == ["GET", "GET", "POST"]
    for call in calls:
        retry = call["retry"]
        assert isinstance(retry, RetryPolicy)
        assert retry.attempts == 1
        assert retry.retry_on_status == ()
        assert retry.retry_on_methods == ()
        assert call["allow_redirects"] is False
        assert call["verify"] is True
        assert call["timeout"] == 17.25
        assert call["transport"] == request["transport"]
        assert call["max_response_bytes"] == audio_cpp.MAX_AUDIO_CPP_RESPONSE_BYTES
        assert call["configured_endpoint"] == ConfiguredEndpointScope.from_url(str(request["origin"]))


@pytest.mark.asyncio
@pytest.mark.parametrize("language", [None, "en-US"])
async def test_audio_cpp_http_multipart_is_minimal_and_exact(
    tmp_path: Path,
    language: str | None,
) -> None:
    request = _audio_cpp_request_kwargs(tmp_path, language=language)
    upload_snapshot: dict[str, Any] = {}

    async def fake_afetch(**kwargs: Any) -> _ClosableAudioCppResponse:
        if kwargs["method"] == "POST":
            files = kwargs["files"]
            data = kwargs["data"]
            assert set(files) == {"file"}
            filename, handle, content_type = files["file"]
            upload_snapshot.update(
                filename=filename,
                content_type=content_type,
                position=handle.tell(),
                header=handle.read(4),
                data=dict(data),
                handle=handle,
            )
        return _successful_audio_cpp_response(kwargs["url"])

    await audio_cpp.transcribe_audio_cpp_async(
        **request,
        afetch_fn=fake_afetch,
    )

    expected_data = {"model": "whisper-small"}
    if language is not None:
        expected_data["language"] = language
    assert upload_snapshot == {
        "filename": "audio.wav",
        "content_type": "audio/wav",
        "position": 0,
        "header": b"RIFF",
        "data": expected_data,
        "handle": upload_snapshot["handle"],
    }
    assert upload_snapshot["handle"].closed
    forbidden = {
        "prompt",
        "hotwords",
        "diarization",
        "timestamps",
        "stream",
        "response_format",
    }
    assert forbidden.isdisjoint(upload_snapshot["data"])


@pytest.mark.asyncio
async def test_audio_cpp_http_response_handles_close_for_async_and_sync_closers(
    tmp_path: Path,
) -> None:
    request = _audio_cpp_request_kwargs(tmp_path)
    responses: list[_ClosableAudioCppResponse | _SyncClosableAudioCppResponse] = []

    async def fake_afetch(
        **kwargs: Any,
    ) -> _ClosableAudioCppResponse | _SyncClosableAudioCppResponse:
        base = _successful_audio_cpp_response(kwargs["url"])
        if len(responses) % 2:
            response = _SyncClosableAudioCppResponse(
                base.content,
                status_code=base.status_code,
            )
        else:
            response = base
        responses.append(response)
        return response

    await audio_cpp.transcribe_audio_cpp_async(
        **request,
        afetch_fn=fake_afetch,
    )

    assert len(responses) == 3
    assert all(response.closed for response in responses)


@pytest.mark.asyncio
@pytest.mark.parametrize("failed_path", ["/health", "/v1/models"])
async def test_audio_cpp_discovery_contract_failure_is_not_cached_or_retried(
    tmp_path: Path,
    failed_path: str,
) -> None:
    request = _audio_cpp_request_kwargs(tmp_path)
    calls: list[str] = []
    failed = False

    async def fake_afetch(**kwargs: Any) -> _ClosableAudioCppResponse:
        nonlocal failed
        url = kwargs["url"]
        calls.append(url)
        if not failed and url.endswith(failed_path):
            failed = True
            return _ClosableAudioCppResponse(b'{"server-secret":"value"}')
        return _successful_audio_cpp_response(url)

    with pytest.raises(STTExecutionUnsupportedError) as exc_info:
        await audio_cpp.transcribe_audio_cpp_async(
            **request,
            afetch_fn=fake_afetch,
        )
    await audio_cpp.transcribe_audio_cpp_async(
        **request,
        afetch_fn=fake_afetch,
    )

    assert "secret" not in str(exc_info.value)
    assert sum(url.endswith(failed_path) for url in calls) == 2


@pytest.mark.asyncio
async def test_audio_cpp_discovery_unknown_model_failure_is_not_cached_or_retried(
    tmp_path: Path,
) -> None:
    request = _audio_cpp_request_kwargs(tmp_path)
    calls: list[str] = []

    async def fake_afetch(**kwargs: Any) -> _ClosableAudioCppResponse:
        url = kwargs["url"]
        calls.append(url)
        if url.endswith("/v1/models"):
            return _ClosableAudioCppResponse(_audio_cpp_catalog("other-model"))
        return _successful_audio_cpp_response(url)

    for _attempt in range(2):
        with pytest.raises(
            STTExecutionUnsupportedError,
            match="^audio.cpp requested model is unavailable$",
        ):
            await audio_cpp.transcribe_audio_cpp_async(
                **request,
                afetch_fn=fake_afetch,
            )

    assert sum(url.endswith("/health") for url in calls) == 2
    assert sum(url.endswith("/v1/models") for url in calls) == 2
    assert not any(url.endswith("/v1/audio/transcriptions") for url in calls)


@pytest.mark.asyncio
async def test_audio_cpp_discovery_transport_failure_is_not_cached_or_retried(
    tmp_path: Path,
) -> None:
    request = _audio_cpp_request_kwargs(tmp_path)
    calls: list[str] = []
    fail_once = True

    async def fake_afetch(**kwargs: Any) -> _ClosableAudioCppResponse:
        nonlocal fail_once
        url = kwargs["url"]
        calls.append(url)
        if fail_once:
            fail_once = False
            raise TimeoutError("server-secret-timeout")
        return _successful_audio_cpp_response(url)

    with pytest.raises(
        STTTranscriptionError,
        match="^audio.cpp request failed$",
    ) as exc_info:
        await audio_cpp.transcribe_audio_cpp_async(
            **request,
            afetch_fn=fake_afetch,
        )
    await audio_cpp.transcribe_audio_cpp_async(
        **request,
        afetch_fn=fake_afetch,
    )

    assert "secret" not in str(exc_info.value)
    assert sum(url.endswith("/health") for url in calls) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code", [404, 422])
async def test_audio_cpp_http_transcription_model_unavailable_invalidates_cache(
    tmp_path: Path,
    status_code: int,
) -> None:
    request = _audio_cpp_request_kwargs(tmp_path)
    calls: list[str] = []
    transcription_calls = 0

    async def fake_afetch(**kwargs: Any) -> _ClosableAudioCppResponse:
        nonlocal transcription_calls
        url = kwargs["url"]
        calls.append(url)
        if url.endswith("/v1/audio/transcriptions"):
            transcription_calls += 1
            if transcription_calls == 2:
                return _ClosableAudioCppResponse(
                    b'{"detail":"server-secret-model"}',
                    status_code=status_code,
                )
        return _successful_audio_cpp_response(url)

    await audio_cpp.transcribe_audio_cpp_async(
        **request,
        afetch_fn=fake_afetch,
    )
    with pytest.raises(
        STTTranscriptionError,
        match="^audio.cpp requested model is unavailable$",
    ):
        await audio_cpp.transcribe_audio_cpp_async(
            **request,
            afetch_fn=fake_afetch,
        )
    await audio_cpp.transcribe_audio_cpp_async(
        **request,
        afetch_fn=fake_afetch,
    )

    assert sum(url.endswith("/health") for url in calls) == 2
    assert sum(url.endswith("/v1/models") for url in calls) == 2
    assert transcription_calls == 3


@pytest.mark.asyncio
async def test_audio_cpp_http_transcription_transport_failure_invalidates_cache(
    tmp_path: Path,
) -> None:
    request = _audio_cpp_request_kwargs(tmp_path)
    calls: list[str] = []
    transcription_calls = 0

    async def fake_afetch(**kwargs: Any) -> _ClosableAudioCppResponse:
        nonlocal transcription_calls
        url = kwargs["url"]
        calls.append(url)
        if url.endswith("/v1/audio/transcriptions"):
            transcription_calls += 1
            if transcription_calls == 2:
                raise TimeoutError("server-secret-timeout")
        return _successful_audio_cpp_response(url)

    await audio_cpp.transcribe_audio_cpp_async(
        **request,
        afetch_fn=fake_afetch,
    )
    with pytest.raises(
        STTTranscriptionError,
        match="^audio.cpp request failed$",
    ):
        await audio_cpp.transcribe_audio_cpp_async(
            **request,
            afetch_fn=fake_afetch,
        )
    await audio_cpp.transcribe_audio_cpp_async(
        **request,
        afetch_fn=fake_afetch,
    )

    assert sum(url.endswith("/health") for url in calls) == 2
    assert sum(url.endswith("/v1/models") for url in calls) == 2
    assert transcription_calls == 3


@pytest.mark.asyncio
async def test_audio_cpp_http_response_read_transport_failure_closes_handle(
    tmp_path: Path,
) -> None:
    request = _audio_cpp_request_kwargs(tmp_path)
    response = _ExplodingContentAudioCppResponse(b"unused")

    async def fake_afetch(**_kwargs: Any) -> _ClosableAudioCppResponse:
        return response

    with pytest.raises(
        STTTranscriptionError,
        match="^audio.cpp request failed$",
    ):
        await audio_cpp.transcribe_audio_cpp_async(
            **request,
            afetch_fn=fake_afetch,
        )

    assert response.closed


@pytest.mark.asyncio
async def test_audio_cpp_http_response_close_completes_before_cancellation_propagates(
    tmp_path: Path,
) -> None:
    request = _audio_cpp_request_kwargs(tmp_path)
    close_entered = asyncio.Event()
    release_close = asyncio.Event()
    response = _BlockingCloseAudioCppResponse(
        _json_bytes({"status": "ok", "backend": "cpu", "models": 1}),
        close_entered=close_entered,
        release_close=release_close,
    )

    async def fake_afetch(**kwargs: Any) -> _ClosableAudioCppResponse:
        if kwargs["url"].endswith("/health"):
            return response
        return _successful_audio_cpp_response(kwargs["url"])

    task = asyncio.create_task(
        audio_cpp.transcribe_audio_cpp_async(
            **request,
            afetch_fn=fake_afetch,
        )
    )
    await asyncio.wait_for(close_entered.wait(), timeout=2)
    task.cancel()
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(asyncio.shield(task), timeout=0.05)
    release_close.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=2)

    assert response.closed


@pytest.mark.asyncio
async def test_audio_cpp_http_response_close_survives_repeated_caller_cancellation(
    tmp_path: Path,
) -> None:
    request = _audio_cpp_request_kwargs(tmp_path)
    close_entered = asyncio.Event()
    release_close = asyncio.Event()
    response = _BlockingCloseAudioCppResponse(
        _json_bytes({"status": "ok", "backend": "cpu", "models": 1}),
        close_entered=close_entered,
        release_close=release_close,
    )

    async def fake_afetch(**kwargs: Any) -> _ClosableAudioCppResponse:
        if kwargs["url"].endswith("/health"):
            return response
        return _successful_audio_cpp_response(kwargs["url"])

    task = asyncio.create_task(
        audio_cpp.transcribe_audio_cpp_async(**request, afetch_fn=fake_afetch)
    )
    try:
        await asyncio.wait_for(close_entered.wait(), timeout=2)
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
    finally:
        release_close.set()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=2)
    assert response.closed


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response_type",
    [_SelfCancellingCloseAudioCppResponse, _FailingCloseAudioCppResponse],
)
async def test_audio_cpp_http_cleanup_cannot_replace_primary_status_error(
    tmp_path: Path,
    response_type: type[_ClosableAudioCppResponse],
) -> None:
    request = _audio_cpp_request_kwargs(tmp_path)

    async def fake_afetch(**kwargs: Any) -> _ClosableAudioCppResponse:
        if kwargs["url"].endswith("/health"):
            return response_type(b'{"server-secret":"hidden"}', status_code=503)
        return _successful_audio_cpp_response(kwargs["url"])

    with pytest.raises(
        STTTranscriptionError,
        match="^audio.cpp server is busy$",
    ):
        await audio_cpp.transcribe_audio_cpp_async(**request, afetch_fn=fake_afetch)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status_code", "message"),
    [
        (503, "audio.cpp server is busy"),
        (500, "audio.cpp request failed"),
    ],
)
async def test_audio_cpp_http_status_failures_are_static_and_not_retried(
    tmp_path: Path,
    status_code: int,
    message: str,
) -> None:
    request = _audio_cpp_request_kwargs(tmp_path)
    calls: list[str] = []
    failed_response: _ClosableAudioCppResponse | None = None

    async def fake_afetch(**kwargs: Any) -> _ClosableAudioCppResponse:
        nonlocal failed_response
        url = kwargs["url"]
        calls.append(url)
        if url.endswith("/v1/audio/transcriptions"):
            failed_response = _ClosableAudioCppResponse(
                b'{"detail":"server-secret-body"}',
                status_code=status_code,
            )
            return failed_response
        return _successful_audio_cpp_response(url)

    with pytest.raises(STTTranscriptionError, match=f"^{message}$") as exc_info:
        await audio_cpp.transcribe_audio_cpp_async(
            **request,
            afetch_fn=fake_afetch,
        )

    assert calls.count("http://127.0.0.1:18080/v1/audio/transcriptions") == 1
    assert failed_response is not None and failed_response.closed
    assert "secret" not in str(exc_info.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code", [404, 422])
async def test_audio_cpp_model_unavailable_status_wins_before_body_access(
    tmp_path: Path,
    status_code: int,
) -> None:
    request = _audio_cpp_request_kwargs(tmp_path)

    async def fake_afetch(**kwargs: Any) -> _ClosableAudioCppResponse:
        if kwargs["url"].endswith("/v1/audio/transcriptions"):
            return _ExplodingContentAudioCppResponse(
                b"unused",
                status_code=status_code,
            )
        return _successful_audio_cpp_response(kwargs["url"])

    with pytest.raises(
        STTTranscriptionError,
        match="^audio.cpp requested model is unavailable$",
    ):
        await audio_cpp.transcribe_audio_cpp_async(**request, afetch_fn=fake_afetch)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body",
    [
        pytest.param(b"{", id="malformed"),
        pytest.param(
            b"x" * (audio_cpp.MAX_AUDIO_CPP_RESPONSE_BYTES + 1),
            id="oversized",
        ),
        pytest.param(
            b'{"timing":{"total_ms":1},"server-secret":"hidden"}',
            id="missing-text",
        ),
    ],
)
async def test_audio_cpp_http_transcription_json_failures_are_bounded_static_and_closed(
    tmp_path: Path,
    body: bytes,
) -> None:
    request = _audio_cpp_request_kwargs(tmp_path)
    response: _ClosableAudioCppResponse | None = None

    async def fake_afetch(**kwargs: Any) -> _ClosableAudioCppResponse:
        nonlocal response
        if kwargs["url"].endswith("/v1/audio/transcriptions"):
            response = _ClosableAudioCppResponse(body)
            return response
        return _successful_audio_cpp_response(kwargs["url"])

    with pytest.raises(
        STTExecutionUnsupportedError,
        match="^audio.cpp transcription response is invalid$",
    ) as exc_info:
        await audio_cpp.transcribe_audio_cpp_async(
            **request,
            afetch_fn=fake_afetch,
        )

    assert response is not None and response.closed
    assert "secret" not in str(exc_info.value)
    assert str(request["audio_path"]) not in str(exc_info.value)
    assert str(request["origin"]) not in str(exc_info.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("text", ["", " ", "\t\n"])
async def test_audio_cpp_http_empty_transcript_is_successful_and_timing_is_ignored(
    tmp_path: Path,
    text: str,
) -> None:
    request = _audio_cpp_request_kwargs(tmp_path, language=None)

    async def fake_afetch(**kwargs: Any) -> _ClosableAudioCppResponse:
        if kwargs["url"].endswith("/v1/audio/transcriptions"):
            return _ClosableAudioCppResponse(
                _json_bytes(
                    {
                        "text": text,
                        "timing": {
                            "total_ms": 123,
                            "duration_ms": 456,
                        },
                    }
                )
            )
        return _successful_audio_cpp_response(kwargs["url"])

    outcome = await audio_cpp.transcribe_audio_cpp_async(
        **request,
        afetch_fn=fake_afetch,
    )

    assert isinstance(outcome, SttTranscriptionOutcome)
    assert outcome.artifact == {
        "text": text,
        "segments": [],
        "language": None,
        "diarization": {"enabled": False, "speakers": None},
        "usage": {"duration_ms": None, "tokens": None},
        "metadata": {
            "provider": "audio-cpp",
            "contract": "audio_cpp_http_v1",
            "model_id": "whisper-small",
            "model_family": "whisper",
            "model_mode": "offline",
            "server_backend": "cpu",
        },
    }
    assert outcome.actual_execution == audio_cpp.actual_execution_from_route(
        request["route"],
        device=None,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(
            lambda route: replace(route, provider="external"),
            id="provider",
        ),
        pytest.param(
            lambda route: replace(route, backend="external_http"),
            id="backend",
        ),
        pytest.param(
            lambda route: replace(route, source="external_http"),
            id="source",
        ),
        pytest.param(
            lambda route: replace(route, model_label="Whisper-Small"),
            id="model",
        ),
        pytest.param(
            lambda route: replace(
                route,
                endpoint_id="sha256:" + ("f" * 64),
            ),
            id="endpoint",
        ),
        pytest.param(
            lambda route: replace(
                route,
                audio_egress=(
                    SttAudioEgress.REMOTE if route.audio_egress is SttAudioEgress.LOOPBACK else SttAudioEgress.LOOPBACK
                ),
            ),
            id="egress",
        ),
        pytest.param(
            lambda route: replace(
                route,
                transport=("httpx" if route.transport == "aiohttp" else "aiohttp"),
            ),
            id="transport",
        ),
        pytest.param(
            lambda route: replace(
                route,
                artifact_id="sha256:" + ("a" * 64),
            ),
            id="artifact-id",
        ),
        pytest.param(
            lambda route: replace(
                route,
                artifact_id="sha256:" + ("a" * 64),
                identity_resolved=True,
            ),
            id="resolved-identity",
        ),
        pytest.param(
            lambda route: replace(route, device="cpu"),
            id="device",
        ),
        pytest.param(
            lambda route: replace(route, compute_type="float32"),
            id="compute-type",
        ),
        pytest.param(
            lambda route: replace(route, dtype="float32"),
            id="dtype",
        ),
        pytest.param(
            lambda route: replace(route, decoding_ids=("language_contract",)),
            id="decoding-ids",
        ),
        pytest.param(
            lambda route: replace(route, local_model_available=True),
            id="local-model-available",
        ),
        pytest.param(
            lambda route: replace(route, would_download=True),
            id="would-download",
        ),
    ],
)
async def test_audio_cpp_http_route_mismatch_fails_before_wav_open_or_afetch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mutate: Callable[[SttExecutionRoute], SttExecutionRoute],
) -> None:
    request = _audio_cpp_request_kwargs(tmp_path)
    request["route"] = mutate(request["route"])

    def fail_open(*_args: Any, **_kwargs: Any) -> BinaryIO:
        raise AssertionError("WAV opened before route validation")

    async def fail_afetch(**_kwargs: Any) -> _ClosableAudioCppResponse:
        raise AssertionError("network used before route validation")

    monkeypatch.setattr(audio_cpp, "open_audio_cpp_wav", fail_open)
    with pytest.raises(
        STTExecutionPlanError,
        match="^Invalid audio.cpp execution route$",
    ):
        await audio_cpp.transcribe_audio_cpp_async(
            **request,
            afetch_fn=fail_afetch,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("origin", "HTTP://127.0.0.1:18080"),
        ("model_id", " whisper-small"),
        ("timeout_seconds", float("nan")),
        ("timeout_seconds", 0.0),
        ("transport", "invalid"),
    ],
)
async def test_audio_cpp_http_frozen_input_mismatch_fails_before_wav_open_or_afetch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    request = _audio_cpp_request_kwargs(tmp_path)
    request[field] = value

    def fail_open(*_args: Any, **_kwargs: Any) -> BinaryIO:
        raise AssertionError("WAV opened before frozen input validation")

    async def fail_afetch(**_kwargs: Any) -> _ClosableAudioCppResponse:
        raise AssertionError("network used before frozen input validation")

    monkeypatch.setattr(audio_cpp, "open_audio_cpp_wav", fail_open)
    with pytest.raises(
        STTExecutionPlanError,
        match="^Invalid audio.cpp execution route$",
    ):
        await audio_cpp.transcribe_audio_cpp_async(
            **request,
            afetch_fn=fail_afetch,
        )


@pytest.mark.asyncio
async def test_audio_cpp_http_runtime_uses_only_frozen_inputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    request = _audio_cpp_request_kwargs(tmp_path)

    def fail_config_read(
        *_args: Any,
        **_kwargs: Any,
    ) -> audio_cpp.AudioCppConfig:
        raise AssertionError("runtime reread audio.cpp configuration")

    monkeypatch.setattr(audio_cpp, "load_audio_cpp_config", fail_config_read)

    outcome = await audio_cpp.transcribe_audio_cpp_async(
        **request,
        afetch_fn=lambda **kwargs: asyncio.sleep(
            0,
            result=_successful_audio_cpp_response(kwargs["url"]),
        ),
    )

    assert outcome.artifact["text"] == "hello"


@pytest.mark.asyncio
async def test_audio_cpp_wav_validation_does_not_block_event_loop(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    request = _audio_cpp_request_kwargs(tmp_path)
    original_open = audio_cpp.open_audio_cpp_wav
    validation_started = threading.Event()
    validation_finished = threading.Event()
    heartbeat_seen = threading.Event()

    def slow_open(*args: Any, **kwargs: Any) -> BinaryIO:
        validation_started.set()
        assert heartbeat_seen.wait(timeout=2)
        handle = original_open(*args, **kwargs)
        validation_finished.set()
        return handle

    monkeypatch.setattr(audio_cpp, "open_audio_cpp_wav", slow_open)

    async def fake_afetch(**kwargs: Any) -> _ClosableAudioCppResponse:
        return _successful_audio_cpp_response(kwargs["url"])

    task = asyncio.create_task(
        audio_cpp.transcribe_audio_cpp_async(**request, afetch_fn=fake_afetch)
    )
    await asyncio.to_thread(validation_started.wait, 2)
    asyncio.get_running_loop().call_soon(heartbeat_seen.set)
    await asyncio.wait_for(task, timeout=2)

    assert validation_finished.is_set()


@pytest.mark.asyncio
async def test_audio_cpp_wav_open_result_is_closed_after_cancellation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    request = _audio_cpp_request_kwargs(tmp_path)
    open_started = threading.Event()
    release_open = threading.Event()
    open_returned = threading.Event()
    opened_handle = (tmp_path / "sample.wav").open("rb")

    def blocking_open(*_args: Any, **_kwargs: Any) -> BinaryIO:
        open_started.set()
        assert release_open.wait(timeout=2)
        open_returned.set()
        return opened_handle

    monkeypatch.setattr(audio_cpp, "open_audio_cpp_wav", blocking_open)
    task = asyncio.create_task(
        audio_cpp.transcribe_audio_cpp_async(
            **request,
            afetch_fn=lambda **_kwargs: asyncio.sleep(0),
        )
    )
    try:
        await asyncio.to_thread(open_started.wait, 2)
        task.cancel()
        release_open.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.to_thread(open_returned.wait, 2)

        assert opened_handle.closed
    finally:
        release_open.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        opened_handle.close()


def test_audio_cpp_default_sync_calls_own_and_close_one_client_per_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    request = _audio_cpp_request_kwargs(tmp_path)
    clients: list[_TrackedAudioCppClient] = []
    request_clients: list[object] = []

    def create_client(*_args: Any, **_kwargs: Any) -> _TrackedAudioCppClient:
        client = _TrackedAudioCppClient()
        clients.append(client)
        return client

    async def fake_afetch(**kwargs: Any) -> _ClosableAudioCppResponse:
        request_clients.append(kwargs.get("client"))
        return _successful_audio_cpp_response(kwargs["url"])

    monkeypatch.setattr(
        audio_cpp,
        "_create_audio_cpp_client",
        create_client,
        raising=False,
    )
    monkeypatch.setattr(audio_cpp, "afetch", fake_afetch)

    audio_cpp.transcribe_audio_cpp(**request)
    audio_cpp.reset_audio_cpp_discovery_cache()
    audio_cpp.transcribe_audio_cpp(**request)

    assert len(clients) == 2
    assert request_clients == [clients[0]] * 3 + [clients[1]] * 3
    assert [client.close_calls for client in clients] == [1, 1]


def test_audio_cpp_http_sync_wrapper_works_without_running_loop(
    tmp_path: Path,
) -> None:
    request = _audio_cpp_request_kwargs(tmp_path)

    async def fake_afetch(**kwargs: Any) -> _ClosableAudioCppResponse:
        return _successful_audio_cpp_response(kwargs["url"])

    outcome = audio_cpp.transcribe_audio_cpp(
        **request,
        afetch_fn=fake_afetch,
    )

    assert isinstance(outcome, SttTranscriptionOutcome)
    assert outcome.artifact["text"] == "hello"


@pytest.mark.asyncio
async def test_audio_cpp_http_sync_wrapper_uses_worker_from_running_loop(
    tmp_path: Path,
) -> None:
    request = _audio_cpp_request_kwargs(tmp_path)

    async def fake_afetch(**kwargs: Any) -> _ClosableAudioCppResponse:
        return _successful_audio_cpp_response(kwargs["url"])

    outcome = audio_cpp.transcribe_audio_cpp(
        **request,
        afetch_fn=fake_afetch,
    )

    assert isinstance(outcome, SttTranscriptionOutcome)
    assert outcome.artifact["text"] == "hello"


@pytest.mark.asyncio
async def test_audio_cpp_sync_wrapper_rejects_same_loop_discovery_leader(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    request = _audio_cpp_request_kwargs(tmp_path)
    _routes, key = audio_cpp._validate_audio_cpp_execution(
        route=request["route"],
        origin=request["origin"],
        model_id=request["model_id"],
        timeout_seconds=request["timeout_seconds"],
        transport=request["transport"],
    )
    monkeypatch.setattr(
        audio_cpp,
        "_audio_cpp_discovery_leader_loops",
        {key: asyncio.get_running_loop()},
        raising=False,
    )

    def fail_open(*_args: Any, **_kwargs: Any) -> BinaryIO:
        raise AssertionError("WAV opened despite same-loop discovery leader")

    async def fail_afetch(**_kwargs: Any) -> _ClosableAudioCppResponse:
        raise AssertionError("network used despite same-loop discovery leader")

    monkeypatch.setattr(audio_cpp, "open_audio_cpp_wav", fail_open)
    with pytest.raises(
        STTExecutionPlanError,
        match="^Invalid audio.cpp execution route$",
    ):
        audio_cpp.transcribe_audio_cpp(**request, afetch_fn=fail_afetch)


def test_audio_cpp_registry_reset_clears_discovery_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio import (
        stt_provider_adapter as spa,
    )

    calls: list[str] = []
    monkeypatch.setattr(
        audio_cpp,
        "reset_audio_cpp_discovery_cache",
        lambda: calls.append("reset"),
    )
    spa._REGISTRY = spa.SttProviderRegistry()

    spa.reset_stt_provider_registry()

    assert calls == ["reset"]
    assert spa._REGISTRY is None
