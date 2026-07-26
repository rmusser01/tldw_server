from __future__ import annotations

import pytest

from tldw_Server_API.app.core.exceptions import STTExecutionUnsupportedError
from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio import (
    Audio_Transcription_AudioCpp as audio_cpp,
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
