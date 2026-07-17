from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
from types import MappingProxyType
from urllib.parse import quote

import pytest
from pydantic import ValidationError

from tldw_Server_API.app.core.TTS.gateway_config import (
    GatewayConfig,
    GatewayPCMCapabilities,
    GatewaySpec,
    build_gateway_url,
    canonicalize_gateway_id,
    decode_json_pointer,
    normalize_gateway_specs,
)
from tldw_Server_API.app.core.TTS.tts_config import (
    ProviderConfig,
    TTSConfig,
    TTSConfigManager,
)


def _gateway(**overrides):
    config = {
        "enabled": True,
        "display_name": "Company Speech",
        "base_url": "https://speech.example.com/v1/",
        "speech_path": "audio/speech",
        "api_key": "admin-secret",
        "default_model": "Vendor/Expressive-TTS",
        "default_voice": "narrator",
        "allowed_models": ["Vendor/Expressive-TTS"],
        "capability_defaults": {"formats": ["mp3", "pcm"]},
    }
    config.update(overrides)
    return config


@pytest.mark.unit
def test_canonical_gateway_ids_and_collisions():
    assert canonicalize_gateway_id("openrouter", builtin=True) == "openrouter"
    assert canonicalize_gateway_id("company-proxy") == "gateway:company-proxy"

    with pytest.raises(ValueError, match="reserved"):
        normalize_gateway_specs({}, {"openrouter": _gateway()})
    with pytest.raises(ValueError, match="collision"):
        normalize_gateway_specs(
            {},
            {"company-proxy": _gateway(), "Company-Proxy": _gateway()},
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    "slug",
    [
        "",
        "Upper",
        "-edge",
        "edge_underscore",
        "a" * 64,
        " company",
        "company ",
        " gateway:company",
        "gateway:company ",
        " openrouter",
        "openrouter ",
    ],
)
def test_gateway_slug_uses_strict_lowercase_pattern(slug):
    with pytest.raises(ValueError, match="slug"):
        canonicalize_gateway_id(slug)


@pytest.mark.unit
@pytest.mark.parametrize(
    "base_url",
    [
        "speech.example.com/v1",
        "ftp://speech.example.com/v1",
        "https://user:pass@speech.example.com/v1",
        "https://speech.example.com/v1?tenant=x",
        "https://speech.example.com/v1?",
        "https://speech.example.com/v1#fragment",
        "https://speech.example.com/v1#",
        "https://speech.example.com/v1?#",
        "https://@speech.example.com/v1",
        "http://speech.example.com/v1",
        "http://8.8.8.8/v1",
    ],
)
def test_base_url_requires_safe_absolute_https(base_url):
    with pytest.raises(ValueError, match="base_url|HTTP"):
        normalize_gateway_specs({}, {"company": _gateway(base_url=base_url)})


@pytest.mark.unit
@pytest.mark.parametrize(
    "base_url",
    [
        "http://localhost:8080/v1/",
        "http://127.0.0.1:8080/v1/",
        "http://10.12.0.8/v1/",
        "http://[::1]:8080/v1/",
        "http://169.254.2.4/v1/",
    ],
)
def test_insecure_http_requires_opt_in_and_local_or_private_literal(base_url):
    with pytest.raises(ValueError, match="allow_insecure_http"):
        normalize_gateway_specs({}, {"company": _gateway(base_url=base_url)})

    specs = normalize_gateway_specs(
        {},
        {"company": _gateway(base_url=base_url, allow_insecure_http=True)},
    )
    assert specs["gateway:company"].enabled is True


@pytest.mark.unit
@pytest.mark.parametrize(
    "path",
    [
        "/audio/speech",
        "https://evil.example/audio/speech",
        "//evil.example/audio/speech",
        r"audio\speech",
        "audio/../speech",
        "audio/./speech",
        "audio/speech?x=1",
        "audio/speech#x",
        "audio/%2e%2e/speech",
        "audio/%252e%252e/speech",
        "audio/%255c..%255cspeech",
        "%252f%252fevil.example/audio/speech",
    ],
)
def test_gateway_paths_are_strict_relative_paths(path):
    with pytest.raises(ValueError, match="speech_path"):
        normalize_gateway_specs({}, {"company": _gateway(speech_path=path)})


@pytest.mark.unit
@pytest.mark.parametrize(
    "overrides",
    [
        {"speech_path": "   "},
        {"models_path": "\t"},
        {"discovery": {"models_path": " \t "}},
    ],
)
def test_gateway_paths_reject_whitespace_only_values(overrides):
    with pytest.raises(ValueError, match="speech_path|models_path"):
        normalize_gateway_specs({}, {"company": _gateway(**overrides)})


@pytest.mark.unit
def test_build_gateway_url_preserves_base_authority_and_path():
    assert (
        str(build_gateway_url("https://speech.example/v1/", "audio/speech"))
        == "https://speech.example/v1/audio/speech"
    )
    with pytest.raises(ValueError):
        build_gateway_url("https://speech.example/v1/", "https://evil.example/")


@pytest.mark.unit
def test_gateway_path_rejects_excessive_encoding_layers():
    path = "audio/speech"
    for _ in range(10):
        path = quote(path, safe="")

    with pytest.raises(ValueError, match="encoding layers"):
        normalize_gateway_specs({}, {"company": _gateway(speech_path=path)})


@pytest.mark.unit
def test_openrouter_is_normalized_as_ordinary_spec_data(monkeypatch):
    monkeypatch.setenv("OPENROUTER_SITE_URL", "https://app.example")
    monkeypatch.setenv("OPENROUTER_SITE_NAME", "Research App")
    specs = normalize_gateway_specs(
        {
            "openrouter": _gateway(
                display_name="OpenRouter",
                base_url=None,
                speech_path=None,
                discovery={"enabled": True},
                allowed_models=None,
                allow_discovered_models=True,
            )
        },
        {},
    )
    spec = specs["openrouter"]
    assert spec.base_url == "https://openrouter.ai/api/v1/"
    assert spec.speech_path == "audio/speech"
    assert spec.models_path == "models"
    assert spec.discovery_query == (("output_modalities", "speech"),)
    assert dict(spec.headers) == {
        "HTTP-Referer": "https://app.example",
        "X-Title": "Research App",
    }


@pytest.mark.unit
def test_discovery_models_path_is_single_validated_effective_value():
    same = normalize_gateway_specs(
        {},
        {
            "company": _gateway(
                models_path="models",
                discovery={"enabled": True, "models_path": "models"},
            )
        },
    )["gateway:company"]
    assert same.models_path == "models"
    assert same.discovery.models_path == "models"

    with pytest.raises(ValueError, match="conflicting.*models_path"):
        normalize_gateway_specs(
            {},
            {
                "company": _gateway(
                    models_path="models-a",
                    discovery={"enabled": True, "models_path": "models-b"},
                )
            },
        )

    with pytest.raises(ValueError, match="models_path"):
        normalize_gateway_specs(
            {},
            {
                "company": _gateway(
                    models_path="models",
                    discovery={
                        "enabled": True,
                        "models_path": "safe/%252e%252e/escape",
                    },
                )
            },
        )


@pytest.mark.unit
def test_model_authorization_capability_and_voice_overlay_precedence():
    spec = normalize_gateway_specs(
        {},
        {
            "company": _gateway(
                allowed_models=["Vendor/Expressive-TTS", "Vendor/Other"],
                capability_defaults={
                    "formats": ["mp3"],
                    "supports_speed": False,
                },
                model_overrides={
                    "Vendor/Other": {
                        "default_voice": "guide",
                        "formats": ["wav"],
                        "supports_speed": True,
                    }
                },
            )
        },
    )["gateway:company"]

    assert spec.allows_model("Vendor/Expressive-TTS", {"Unknown/Discovered"})
    assert not spec.allows_model("Unknown/Discovered", {"Unknown/Discovered"})
    assert spec.capabilities_for_model("Vendor/Other").formats == ("wav",)
    assert spec.capabilities_for_model("Vendor/Other").supports_speed is True
    assert spec.default_voice_for_model("Vendor/Expressive-TTS") == "narrator"
    assert spec.default_voice_for_model("Vendor/Other") == "guide"
    assert spec.default_voice_for_model("Unknown/Discovered") is None


@pytest.mark.unit
def test_pcm_model_overlay_remains_typed_and_deeply_immutable():
    spec = normalize_gateway_specs(
        {},
        {
            "company": _gateway(
                model_overrides={
                    "Vendor/Expressive-TTS": {
                        "pcm": {
                            "sample_rate": 48000,
                            "channels": 2,
                            "sample_width_bits": 24,
                        }
                    }
                }
            )
        },
    )["gateway:company"]

    capabilities = spec.capabilities_for_model("Vendor/Expressive-TTS")

    assert isinstance(capabilities.pcm, GatewayPCMCapabilities)
    assert capabilities.pcm.sample_rate == 48000
    with pytest.raises(ValidationError):
        capabilities.pcm.sample_rate = 16000


@pytest.mark.unit
def test_discovered_models_are_authorized_only_when_enabled():
    discovered = {"Vendor/Discovered"}
    disabled = normalize_gateway_specs(
        {}, {"a": _gateway(allowed_models=None)}
    )["gateway:a"]
    enabled = normalize_gateway_specs(
        {},
        {"b": _gateway(allowed_models=None, allow_discovered_models=True)},
    )["gateway:b"]
    assert not disabled.allows_model("Vendor/Discovered", discovered)
    assert enabled.allows_model("Vendor/Discovered", discovered)


@pytest.mark.unit
@pytest.mark.parametrize(
    "pointer",
    ["provider/options", "/provider/~2bad", "/headers/x", "/api_key"],
)
def test_request_option_pointers_reject_malformed_and_reserved_fields(pointer):
    with pytest.raises(ValueError, match="allowed_request_options|JSON Pointer|reserved"):
        normalize_gateway_specs(
            {}, {"company": _gateway(allowed_request_options=[pointer])}
        )


@pytest.mark.unit
def test_request_option_pointer_unescaping():
    assert decode_json_pointer("/provider/a~1b/~0name") == (
        "provider",
        "a/b",
        "~name",
    )


@pytest.mark.unit
def test_json_pointer_supports_root_and_empty_reference_tokens():
    assert decode_json_pointer("") == ()
    assert decode_json_pointer("/") == ("",)
    assert decode_json_pointer("/valid/") == ("valid", "")
    assert decode_json_pointer("//double") == ("", "double")


@pytest.mark.unit
def test_request_option_allows_empty_leaf_name_but_rejects_whole_document():
    spec = normalize_gateway_specs(
        {},
        {
            "company": _gateway(
                allowed_request_options=["/provider/options/", "//extension"]
            )
        },
    )["gateway:company"]
    assert spec.allowed_request_options == frozenset(
        {"/provider/options/", "//extension"}
    )

    with pytest.raises(ValueError, match="whole document"):
        normalize_gateway_specs(
            {}, {"company": _gateway(allowed_request_options=[""])}
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    "fallback,match",
    [
        ({"max_attempts": 0}, "max_attempts"),
        ({"max_attempts": 5}, "max_attempts"),
        (
            {
                "targets": [
                    {
                        "backend": "gateway:primary",
                        "model": "Vendor/Expressive-TTS",
                    }
                ]
            },
            "itself",
        ),
        (
            {
                "targets": [
                    {
                        "backend": "gateway:secondary",
                        "model": "Vendor/Expressive-TTS",
                    }
                ]
                * 2
            },
            "duplicate",
        ),
        (
            {
                "targets": [
                    {"backend": "gateway:a", "model": "Vendor/Expressive-TTS"},
                    {"backend": "gateway:b", "model": "Vendor/Expressive-TTS"},
                    {"backend": "gateway:c", "model": "Vendor/Expressive-TTS"},
                    {"backend": "gateway:d", "model": "Vendor/Expressive-TTS"},
                ]
            },
            "at most 3",
        ),
        (
            {
                "targets": [
                    {
                        "backend": "gateway:missing",
                        "model": "Vendor/Expressive-TTS",
                    }
                ]
            },
            "unknown",
        ),
    ],
)
def test_fallback_policy_bounds_and_targets(fallback, match):
    gateways = {
        "primary": _gateway(fallback=fallback),
        "secondary": _gateway(),
        "a": _gateway(),
        "b": _gateway(),
        "c": _gateway(),
        "d": _gateway(),
    }
    with pytest.raises(ValueError, match=match):
        normalize_gateway_specs({}, gateways)


@pytest.mark.unit
def test_fallback_cycle_is_rejected():
    with pytest.raises(ValueError, match="cycle"):
        normalize_gateway_specs(
            {},
            {
                "a": _gateway(
                    fallback={
                        "targets": [
                            {
                                "backend": "gateway:b",
                                "model": "Vendor/Expressive-TTS",
                            }
                        ]
                    }
                ),
                "b": _gateway(
                    fallback={
                        "targets": [
                            {
                                "backend": "gateway:a",
                                "model": "Vendor/Expressive-TTS",
                            }
                        ]
                    }
                ),
            },
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    "categories",
    [
        [""],
        ["unknown_error"],
        ["timeout", "timeout"],
    ],
)
def test_fallback_categories_reject_blank_unknown_and_duplicates(categories):
    with pytest.raises(ValueError, match="fallback|categor|duplicate"):
        normalize_gateway_specs(
            {}, {"company": _gateway(fallback={"on": categories})}
        )


@pytest.mark.unit
def test_fallback_accepts_every_stable_category():
    categories = [
        "timeout",
        "network_error",
        "upstream_5xx",
        "circuit_open",
        "rate_limited",
        "quota_exceeded",
        "authentication_failed",
        "model_not_found",
        "invalid_audio",
    ]
    spec = normalize_gateway_specs(
        {}, {"company": _gateway(fallback={"on": categories})}
    )["gateway:company"]
    assert spec.fallback.on == tuple(categories)


@pytest.mark.unit
def test_fallback_target_requires_model():
    with pytest.raises(ValueError, match="model"):
        normalize_gateway_specs(
            {},
            {
                "primary": _gateway(
                    fallback={"targets": [{"backend": "gateway:secondary"}]}
                ),
                "secondary": _gateway(),
            },
        )


@pytest.mark.unit
def test_fallback_target_without_voice_requires_effective_model_default():
    target = _gateway(
        allowed_models=["Vendor/Expressive-TTS", "Vendor/NoDefaultVoice"]
    )
    with pytest.raises(ValueError, match="default voice"):
        normalize_gateway_specs(
            {},
            {
                "primary": _gateway(
                    fallback={
                        "targets": [
                            {
                                "backend": "gateway:secondary",
                                "model": "Vendor/NoDefaultVoice",
                            }
                        ]
                    }
                ),
                "secondary": target,
            },
        )

    specs = normalize_gateway_specs(
        {},
        {
            "primary": _gateway(
                fallback={
                    "targets": [
                        {
                            "backend": "gateway:secondary",
                            "model": "Vendor/NoDefaultVoice",
                        }
                    ]
                }
            ),
            "secondary": _gateway(
                allowed_models=[
                    "Vendor/Expressive-TTS",
                    "Vendor/NoDefaultVoice",
                ],
                model_overrides={
                    "Vendor/NoDefaultVoice": {"default_voice": "guide"}
                },
            ),
        },
    )
    assert specs["gateway:primary"].fallback.targets[0].voice is None


@pytest.mark.unit
def test_long_acyclic_fallback_graph_does_not_recurse():
    definitions = {}
    length = 1100
    for index in range(length):
        fallback = {}
        if index + 1 < length:
            fallback = {
                "targets": [
                    {
                        "backend": f"gateway:g{index + 1}",
                        "model": "Vendor/Expressive-TTS",
                    }
                ]
            }
        definitions[f"g{index}"] = _gateway(fallback=fallback)

    specs = normalize_gateway_specs({}, definitions)

    assert len(specs) == length


@pytest.mark.unit
def test_normalization_is_local_only(monkeypatch):
    from tldw_Server_API.app.core import http_client

    def fail_network(*_args, **_kwargs):
        raise AssertionError("startup validation must not access the network")

    monkeypatch.setattr(http_client, "afetch_json", fail_network)
    monkeypatch.setattr(http_client, "astream_bytes", fail_network)
    assert normalize_gateway_specs({}, {"company": _gateway()})


@pytest.mark.unit
def test_recursive_environment_interpolation(monkeypatch):
    monkeypatch.setenv("DISCOVERY_VALUE", "speech")
    monkeypatch.setenv("MODEL_ID", "Vendor/Model")
    monkeypatch.setenv("VOICE_ID", "narrator")
    monkeypatch.setenv("TARGET_MODEL", "Vendor/Fallback")
    monkeypatch.setenv("TARGET_VOICE", "guide")
    monkeypatch.setenv("CONVERSION_BYTES", "2048")
    spec = normalize_gateway_specs(
        {},
        {
            "primary": _gateway(
                default_model="${MODEL_ID}",
                default_voice="${VOICE_ID}",
                allowed_models=["${MODEL_ID}"],
                discovery={"query": {"output_modalities": "${DISCOVERY_VALUE}"}},
                conversion={"max_input_bytes": "${CONVERSION_BYTES}"},
                fallback={
                    "targets": [
                        {
                            "backend": "gateway:secondary",
                            "model": "${TARGET_MODEL}",
                            "voice": "${TARGET_VOICE}",
                        }
                    ]
                },
            ),
            "secondary": _gateway(),
        },
    )["gateway:primary"]
    assert spec.default_model == "Vendor/Model"
    assert spec.discovery_query == (("output_modalities", "speech"),)
    assert spec.conversion.max_input_bytes == 2048
    assert spec.fallback.targets[0].model == "Vendor/Fallback"
    assert spec.fallback.targets[0].voice == "guide"


@pytest.mark.unit
def test_unresolved_required_environment_reports_path_not_secret(monkeypatch):
    monkeypatch.delenv("MISSING_TTS_MODEL", raising=False)
    with pytest.raises(ValueError) as exc_info:
        normalize_gateway_specs(
            {}, {"company": _gateway(default_model="${MISSING_TTS_MODEL}")}
        )
    message = str(exc_info.value)
    assert "gateways.company.default_model" in message
    assert "MISSING_TTS_MODEL" not in message


@pytest.mark.unit
def test_unresolved_environment_on_disabled_gateway_is_allowed(monkeypatch):
    monkeypatch.delenv("MISSING_OPTIONAL_KEY", raising=False)
    spec = normalize_gateway_specs(
        {},
        {
            "company": _gateway(
                enabled=False,
                api_key="${MISSING_OPTIONAL_KEY}",
            )
        },
    )["gateway:company"]
    assert spec.enabled is False
    assert spec.api_key is None


@pytest.mark.unit
def test_unresolved_optional_admin_key_is_allowed_for_byok_gateway(monkeypatch):
    monkeypatch.delenv("MISSING_OPTIONAL_ADMIN_KEY", raising=False)
    spec = normalize_gateway_specs(
        {},
        {
            "company": _gateway(
                api_key="${MISSING_OPTIONAL_ADMIN_KEY}",
                allow_user_api_key=True,
            )
        },
    )["gateway:company"]
    assert spec.enabled is True
    assert spec.api_key is None
    assert spec.allow_user_api_key is True


@pytest.mark.unit
def test_disabled_gateway_ignores_unresolved_nested_optional_scalars(monkeypatch):
    monkeypatch.delenv("MISSING_OPTIONAL_LIMIT", raising=False)
    spec = normalize_gateway_specs(
        {},
        {
            "company": _gateway(
                enabled=False,
                conversion={"max_input_bytes": "${MISSING_OPTIONAL_LIMIT}"},
            )
        },
    )["gateway:company"]
    assert spec.conversion.max_input_bytes == 26214400


@pytest.mark.unit
def test_openrouter_provider_is_explicit_gateway_config_and_serializes_fields():
    config = TTSConfig(
        providers={
            "openrouter": _gateway(
                allow_user_api_key=True,
                allowed_request_options=["/provider/options/style"],
            ),
            "openai": {"enabled": True, "model": "tts-1"},
        },
        gateways={"company": _gateway()},
    )
    assert isinstance(config.providers["openrouter"], GatewayConfig)
    assert isinstance(config.providers["openai"], ProviderConfig)
    dumped = config.model_dump()
    assert dumped["providers"]["openrouter"]["allow_user_api_key"] is True
    assert dumped["providers"]["openrouter"]["allowed_request_options"] == (
        "/provider/options/style",
    )
    assert "openrouter" not in dumped["gateways"]


@pytest.mark.unit
def test_missing_ffmpeg_only_removes_conversion_formats():
    spec = normalize_gateway_specs(
        {},
        {
            "company": _gateway(
                capability_defaults={"formats": ["mp3", "pcm"]},
                conversion={
                    "enabled": True,
                    "source_format": "mp3",
                    "target_formats": ["wav", "flac"],
                },
            )
        },
        ffmpeg_available=False,
    )["gateway:company"]
    assert spec.enabled is True
    assert spec.capabilities_for_model(spec.default_model).formats == ("mp3", "pcm")
    assert spec.conversion.target_formats == ()
    assert spec.ffmpeg_path is None


@pytest.mark.unit
def test_gateway_normalization_pins_ffmpeg_identity_and_generation(monkeypatch, tmp_path):
    first = tmp_path / "first" / "ffmpeg"
    second = tmp_path / "second" / "ffmpeg"
    for executable in (first, second):
        executable.parent.mkdir()
        executable.write_text("#!/bin/sh\nexit 0\n")
        executable.chmod(0o755)

    config = _gateway(
        conversion={"enabled": True, "target_formats": ["wav"]},
    )
    monkeypatch.setenv("PATH", str(first.parent))
    first_spec = normalize_gateway_specs({}, {"company": config})["gateway:company"]

    monkeypatch.setenv("PATH", str(second.parent))
    second_spec = normalize_gateway_specs({}, {"company": config})["gateway:company"]

    assert first_spec.ffmpeg_path == str(first.resolve())
    assert first_spec.ffmpeg_path != str(second.resolve())
    assert first_spec.conversion.target_formats == ("wav",)
    assert second_spec.ffmpeg_path == str(second.resolve())
    assert first_spec.config_generation != second_spec.config_generation


@pytest.mark.unit
def test_gateway_normalization_clears_unusable_injected_ffmpeg(tmp_path):
    missing = Path(tmp_path, "missing-ffmpeg")
    spec = normalize_gateway_specs(
        {},
        {
            "company": _gateway(
                conversion={"enabled": True, "target_formats": ["wav"]},
            )
        },
        ffmpeg_path=str(missing),
    )["gateway:company"]

    assert spec.ffmpeg_path is None
    assert spec.conversion.target_formats == ()


@pytest.mark.unit
def test_gateway_spec_is_deeply_immutable_and_generation_is_secret_free():
    config = _gateway(
        api_key="first-secret",
        model_overrides={"Vendor/Expressive-TTS": {"voices": ["narrator"]}},
    )
    first = normalize_gateway_specs({}, {"company": config})["gateway:company"]
    config["model_overrides"]["Vendor/Expressive-TTS"]["voices"].append("mutated")
    second = normalize_gateway_specs(
        {},
        {
            "company": _gateway(
                api_key="second-secret",
                model_overrides={
                    "Vendor/Expressive-TTS": {"voices": ["narrator"]}
                },
            )
        },
    )["gateway:company"]

    assert isinstance(first, GatewaySpec)
    assert isinstance(first.model_overrides, MappingProxyType)
    assert "first-secret" not in repr(first)
    assert first.model_overrides["Vendor/Expressive-TTS"].voices == ("narrator",)
    assert first.config_generation == second.config_generation
    with pytest.raises(FrozenInstanceError):
        first.enabled = False
    with pytest.raises(TypeError):
        first.model_overrides["x"] = None


@pytest.mark.unit
def test_display_name_changes_config_generation():
    first = normalize_gateway_specs(
        {}, {"company": _gateway(display_name="First label")}
    )["gateway:company"]
    second = normalize_gateway_specs(
        {}, {"company": _gateway(display_name="Second label")}
    )["gateway:company"]

    assert first.config_generation != second.config_generation


@pytest.mark.unit
def test_config_generation_canonicalizes_query_and_fallback_category_order():
    first = normalize_gateway_specs(
        {},
        {
            "primary": _gateway(
                discovery={"query": {"b": "2", "a": "1"}},
                fallback={
                    "on": ["timeout", "network_error"],
                    "targets": [
                        {
                            "backend": "gateway:secondary",
                            "model": "Vendor/Expressive-TTS",
                        },
                        {
                            "backend": "gateway:tertiary",
                            "model": "Vendor/Expressive-TTS",
                        },
                    ],
                },
            ),
            "secondary": _gateway(),
            "tertiary": _gateway(),
        },
    )["gateway:primary"]
    second = normalize_gateway_specs(
        {},
        {
            "primary": _gateway(
                discovery={"query": {"a": "1", "b": "2"}},
                fallback={
                    "on": ["network_error", "timeout"],
                    "targets": [
                        {
                            "backend": "gateway:secondary",
                            "model": "Vendor/Expressive-TTS",
                        },
                        {
                            "backend": "gateway:tertiary",
                            "model": "Vendor/Expressive-TTS",
                        },
                    ],
                },
            ),
            "secondary": _gateway(),
            "tertiary": _gateway(),
        },
    )["gateway:primary"]
    changed_target_order = normalize_gateway_specs(
        {},
        {
            "primary": _gateway(
                discovery={"query": {"a": "1", "b": "2"}},
                fallback={
                    "on": ["network_error", "timeout"],
                    "targets": [
                        {
                            "backend": "gateway:tertiary",
                            "model": "Vendor/Expressive-TTS",
                        },
                        {
                            "backend": "gateway:secondary",
                            "model": "Vendor/Expressive-TTS",
                        },
                    ],
                },
            ),
            "secondary": _gateway(),
            "tertiary": _gateway(),
        },
    )["gateway:primary"]

    assert first.config_generation == second.config_generation
    assert first.config_generation != changed_target_order.config_generation


@pytest.mark.unit
@pytest.mark.parametrize(
    "overrides",
    [
        {"default_model": "   ", "allowed_models": None},
        {"default_voice": "\t"},
        {"api_key": "   ", "allow_user_api_key": True},
        {
            "default_voice": None,
            "model_overrides": {
                "Vendor/Expressive-TTS": {"default_voice": "   "}
            },
        },
    ],
)
def test_enabled_gateway_rejects_blank_required_values(overrides):
    with pytest.raises(ValueError, match="blank|requires"):
        normalize_gateway_specs({}, {"company": _gateway(**overrides)})


@pytest.mark.unit
def test_required_model_voice_and_key_preserve_nonblank_exact_spacing():
    model = " Vendor/Model "
    voice = " narrator "
    spec = normalize_gateway_specs(
        {},
        {
            "company": _gateway(
                api_key=" secret ",
                default_model=model,
                default_voice=voice,
                allowed_models=[model],
            )
        },
    )["gateway:company"]

    assert spec.default_model == model
    assert spec.default_voice == voice
    assert spec.api_key == " secret "


@pytest.mark.unit
def test_tts_config_resolves_nested_gateway_environment_before_validation(monkeypatch):
    monkeypatch.setenv("GW_SUPPORTS_SPEED", "true")
    monkeypatch.setenv("GW_MAX_RESPONSE", "1048576")
    monkeypatch.setenv("GW_DISCOVERY_TTL", "321")
    monkeypatch.setenv("GW_DISCOVERY_MODE", "speech")
    monkeypatch.setenv("GW_CONVERSION_ENABLED", "true")
    monkeypatch.setenv("GW_CONVERSION_LIMIT", "2048")
    monkeypatch.setenv("GW_FALLBACK_ATTEMPTS", "2")
    monkeypatch.setenv("GW_FALLBACK_MODEL", "Vendor/Fallback")
    monkeypatch.setenv("GW_FALLBACK_VOICE", "guide")
    config = TTSConfig(
        gateways={
            "primary": _gateway(
                capability_defaults={
                    "formats": ["mp3"],
                    "supports_speed": "${GW_SUPPORTS_SPEED}",
                    "max_response_bytes": "${GW_MAX_RESPONSE}",
                },
                discovery={
                    "enabled": True,
                    "models_path": "models",
                    "query": {"output_modalities": "${GW_DISCOVERY_MODE}"},
                    "ttl_seconds": "${GW_DISCOVERY_TTL}",
                },
                conversion={
                    "enabled": "${GW_CONVERSION_ENABLED}",
                    "max_input_bytes": "${GW_CONVERSION_LIMIT}",
                },
                fallback={
                    "max_attempts": "${GW_FALLBACK_ATTEMPTS}",
                    "targets": [
                        {
                            "backend": "gateway:secondary",
                            "model": "${GW_FALLBACK_MODEL}",
                            "voice": "${GW_FALLBACK_VOICE}",
                        }
                    ],
                },
            ),
            "secondary": _gateway(),
        }
    )

    primary = config.gateways["primary"]
    assert isinstance(primary, GatewayConfig)
    assert primary.capability_defaults.supports_speed is True
    assert primary.capability_defaults.max_response_bytes == 1048576
    assert primary.discovery.ttl_seconds == 321
    assert primary.discovery.query == (("output_modalities", "speech"),)
    assert primary.conversion.enabled is True
    assert primary.conversion.max_input_bytes == 2048
    assert primary.fallback.max_attempts == 2
    assert primary.fallback.targets[0].model == "Vendor/Fallback"
    assert primary.fallback.targets[0].voice == "guide"


@pytest.mark.unit
def test_openrouter_environment_is_materialized_by_key_before_validation(monkeypatch):
    monkeypatch.setenv("OPENROUTER_GATEWAY_LIMIT", "4096")
    config = TTSConfig(
        providers={
            "openrouter": _gateway(
                base_url=None,
                speech_path=None,
                capability_defaults={
                    "formats": ["mp3"],
                    "max_response_bytes": "${OPENROUTER_GATEWAY_LIMIT}",
                },
            )
        }
    )

    openrouter = config.providers["openrouter"]
    assert isinstance(openrouter, GatewayConfig)
    assert openrouter.base_url == "https://openrouter.ai/api/v1/"
    assert openrouter.capability_defaults.max_response_bytes == 4096


@pytest.mark.unit
def test_tts_config_required_placeholder_error_has_exact_secret_free_path(monkeypatch):
    monkeypatch.delenv("VERY_SECRET_MISSING_MODEL_NAME", raising=False)

    with pytest.raises(ValueError) as exc_info:
        TTSConfig(
            gateways={
                "company": _gateway(
                    default_model="${VERY_SECRET_MISSING_MODEL_NAME}"
                )
            }
        )

    message = str(exc_info.value)
    assert "gateways.company.default_model" in message
    assert "VERY_SECRET_MISSING_MODEL_NAME" not in message


@pytest.mark.unit
def test_false_enabled_placeholder_keeps_incomplete_gateway_disabled(monkeypatch):
    monkeypatch.setenv("GW_DISABLED_FLAG", "false")
    config = TTSConfig(
        gateways={
            "disabled": {
                "enabled": "${GW_DISABLED_FLAG}",
                "conversion": {"max_input_bytes": 2048},
            }
        }
    )

    assert config.gateways["disabled"].enabled is False


@pytest.mark.unit
def test_config_manager_reload_materializes_gateway_environment_first(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("GW_MANAGER_RESPONSE_LIMIT", "8192")
    yaml_path = tmp_path / "tts.yaml"
    yaml_path.write_text(
        """
gateways:
  managed:
    enabled: true
    display_name: Managed Gateway
    base_url: https://speech.example.com/v1/
    speech_path: audio/speech
    api_key: manager-secret
    default_model: Vendor/Managed
    default_voice: narrator
    capability_defaults:
      formats: [mp3]
      max_response_bytes: ${GW_MANAGER_RESPONSE_LIMIT}
""",
        encoding="utf-8",
    )
    manager = TTSConfigManager(
        yaml_path=yaml_path,
        config_txt_path=tmp_path / "missing-config.txt",
    )

    assert manager.get_config().gateways["managed"].capability_defaults.max_response_bytes == 8192
    assert manager.get_gateway_spec("managed").backend_id == "gateway:managed"


@pytest.mark.unit
def test_config_manager_exposes_specs_without_discovery():
    manager = TTSConfigManager.__new__(TTSConfigManager)
    manager._config = TTSConfig(gateways={"company": _gateway()})
    manager._gateway_specs = None
    manager._sources = {}
    specs = manager.get_gateway_specs()
    assert specs["gateway:company"].backend_id == "gateway:company"
    assert manager.get_gateway_spec("company") is specs["gateway:company"]
    assert manager.get_gateway_spec("missing") is None
