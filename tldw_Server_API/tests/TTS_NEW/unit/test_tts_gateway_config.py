from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import MappingProxyType

import pytest

from tldw_Server_API.app.core.TTS.gateway_config import (
    GatewayConfig,
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
    ["", "Upper", "-edge", "edge_underscore", "a" * 64],
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
        "https://speech.example.com/v1#fragment",
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
    ],
)
def test_gateway_paths_are_strict_relative_paths(path):
    with pytest.raises(ValueError, match="speech_path"):
        normalize_gateway_specs({}, {"company": _gateway(speech_path=path)})


@pytest.mark.unit
def test_build_gateway_url_preserves_base_authority_and_path():
    assert (
        str(build_gateway_url("https://speech.example/v1/", "audio/speech"))
        == "https://speech.example/v1/audio/speech"
    )
    with pytest.raises(ValueError):
        build_gateway_url("https://speech.example/v1/", "https://evil.example/")


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
    ["provider/options", "//double", "/provider/~2bad", "/headers/x", "/api_key"],
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
@pytest.mark.parametrize(
    "fallback,match",
    [
        ({"max_attempts": 0}, "max_attempts"),
        ({"max_attempts": 5}, "max_attempts"),
        ({"targets": [{"backend": "gateway:primary"}]}, "itself"),
        (
            {"targets": [{"backend": "gateway:secondary"}] * 2},
            "duplicate",
        ),
        (
            {
                "targets": [
                    {"backend": "gateway:a"},
                    {"backend": "gateway:b"},
                    {"backend": "gateway:c"},
                    {"backend": "gateway:d"},
                ]
            },
            "at most 3",
        ),
        ({"targets": [{"backend": "gateway:missing"}]}, "unknown"),
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
                "a": _gateway(fallback={"targets": [{"backend": "gateway:b"}]}),
                "b": _gateway(fallback={"targets": [{"backend": "gateway:a"}]}),
            },
        )


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
    assert first.model_overrides["Vendor/Expressive-TTS"].voices == ("narrator",)
    assert first.config_generation == second.config_generation
    with pytest.raises(FrozenInstanceError):
        first.enabled = False
    with pytest.raises(TypeError):
        first.model_overrides["x"] = None


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
