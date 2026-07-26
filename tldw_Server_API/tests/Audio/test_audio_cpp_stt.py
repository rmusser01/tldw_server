from __future__ import annotations

import json
import os
import struct
import wave
from collections.abc import Callable
from pathlib import Path
from typing import BinaryIO

import pytest

from tldw_Server_API.app.core.exceptions import STTExecutionUnsupportedError
from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio import (
    Audio_Transcription_AudioCpp as audio_cpp,
)

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
    "selector",
    [
        "audio-cpp:whisper-small",
        "audiocpp:whisper-small",
        "audio_cpp:whisper-small",
    ],
)
def test_audio_cpp_selector_prefixes_strip_to_exact_server_model(
    selector: str,
) -> None:
    assert (
        audio_cpp.normalize_audio_cpp_model(
            selector,
            default_model="configured-model",
        )
        == "whisper-small"
    )


@pytest.mark.parametrize("selector", ["audio-cpp", "audiocpp", "audio_cpp", None])
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
