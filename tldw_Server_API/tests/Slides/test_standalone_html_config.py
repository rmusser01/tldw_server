from __future__ import annotations

import dataclasses
import hashlib
import json
from configparser import ConfigParser
from pathlib import Path
from urllib.parse import urlsplit

import pytest

import tldw_Server_API.app.core.Slides.standalone_html_config as config_mod
from tldw_Server_API.app.core.Slides.standalone_html_config import (
    ALLOWED_TARGETS_JSON_MAX_BYTES,
    ALLOWED_TARGETS_MAX_ENTRIES,
    CLOSED_ADAPTER_CATALOG,
    PROMPT_CONTRACT_VERSION,
    PROMPT_MAX_BYTES,
    StandaloneHtmlConfigError,
    StandaloneHtmlGenerationAvailability,
    load_standalone_html_config,
)
from tldw_Server_API.app.core.Utils.prompt_loader import (
    PromptAssetUnavailableError,
    load_prompt,
    load_prompt_strict,
)

pytestmark = pytest.mark.unit


_ALL_AVAILABLE = StandaloneHtmlGenerationAvailability(
    digest_key_available=True,
    worker_handler_registered=True,
    reconciler_admission_ready=True,
    validator_available=True,
)


def _parser(**overrides: object) -> ConfigParser:
    values: dict[str, str] = {
        "enabled": "true",
        "egress_enabled": "true",
        "default_provider": "openai",
        "default_model": "gpt-4o-mini",
        "default_adapter_id": "openai_official_chat_v1",
        "allowed_targets_json": json.dumps(
            [
                {
                    "provider": "openai",
                    "model": "gpt-4o-mini",
                    "adapter_id": "openai_official_chat_v1",
                }
            ]
        ),
        "connect_timeout_seconds": "10",
        "read_timeout_seconds": "120",
        "overall_timeout_seconds": "180",
        "max_output_tokens": "16384",
        "max_source_chars": "200000",
        "max_source_tokens": "50000",
        "max_provider_response_bytes": "8388608",
    }
    values.update({key: str(value) for key, value in overrides.items()})
    parser = ConfigParser(interpolation=None)
    parser.read_dict({"SlidesStandaloneHtml": values})
    return parser


def _prompt_loader(_module: str, _key: str, _max_bytes: int) -> str:
    return "maintained standalone prompt"


def _load(parser: ConfigParser | None = None, **kwargs: object):
    return load_standalone_html_config(
        parser or _parser(),
        env={},
        availability=_ALL_AVAILABLE,
        prompt_loader=_prompt_loader,
        **kwargs,
    )


def test_closed_catalog_has_exact_six_fixed_adapter_identities() -> None:
    assert [adapter.adapter_id for adapter in CLOSED_ADAPTER_CATALOG] == [
        "openai_official_chat_v1",
        "anthropic_official_messages_v1",
        "llamacpp_loopback_chat_v1_ipv4",
        "llamacpp_loopback_chat_v1_ipv6",
        "ollama_loopback_chat_v1_ipv4",
        "ollama_loopback_chat_v1_ipv6",
    ]
    assert [
        (adapter.provider, adapter.endpoint_identity, adapter.verified_https) for adapter in CLOSED_ADAPTER_CATALOG
    ] == [
        ("openai", "https://api.openai.com:443/v1/chat/completions", True),
        ("anthropic", "https://api.anthropic.com:443/v1/messages", True),
        ("llama.cpp", "http://127.0.0.1:8080/v1/chat/completions", False),
        ("llama.cpp", "http://[::1]:8080/v1/chat/completions", False),
        ("ollama", "http://127.0.0.1:11434/v1/chat/completions", False),
        ("ollama", "http://[::1]:11434/v1/chat/completions", False),
    ]
    assert all(adapter.fixed_endpoint for adapter in CLOSED_ADAPTER_CATALOG)


def test_catalog_has_only_verified_remote_https_or_literal_loopback_http() -> None:
    for adapter in CLOSED_ADAPTER_CATALOG:
        endpoint = urlsplit(adapter.endpoint_identity)
        assert endpoint.username is None
        assert endpoint.password is None
        assert endpoint.query == ""
        assert endpoint.fragment == ""
        if endpoint.scheme == "https":
            assert adapter.verified_https is True
            assert endpoint.hostname in {"api.openai.com", "api.anthropic.com"}
            assert endpoint.port == 443
        else:
            assert endpoint.scheme == "http"
            assert endpoint.hostname in {"127.0.0.1", "::1"}
            assert adapter.verified_https is False


@pytest.mark.parametrize(
    ("adapter_id", "provider"),
    [
        ("openai_official_chat_v1", "openai"),
        ("anthropic_official_messages_v1", "anthropic"),
        ("llamacpp_loopback_chat_v1_ipv4", "llama.cpp"),
        ("llamacpp_loopback_chat_v1_ipv6", "llama.cpp"),
        ("ollama_loopback_chat_v1_ipv4", "ollama"),
        ("ollama_loopback_chat_v1_ipv6", "ollama"),
    ],
)
def test_each_catalog_adapter_can_be_exactly_allowlisted(adapter_id: str, provider: str) -> None:
    allowed = json.dumps([{"provider": provider, "model": "CaseSensitive-Model", "adapter_id": adapter_id}])
    cfg = _load(
        _parser(
            default_provider=provider,
            default_model="CaseSensitive-Model",
            default_adapter_id=adapter_id,
            allowed_targets_json=allowed,
        )
    )

    assert cfg.enabled is True
    assert cfg.target is not None
    assert cfg.target.model == "CaseSensitive-Model"
    assert cfg.target.endpoint_identity == next(
        item.endpoint_identity for item in CLOSED_ADAPTER_CATALOG if item.adapter_id == adapter_id
    )


def test_generation_is_default_off_and_egress_is_an_independent_kill() -> None:
    default_parser = ConfigParser(interpolation=None)
    default_parser.add_section("SlidesStandaloneHtml")
    default_cfg = load_standalone_html_config(
        default_parser,
        env={},
        availability=_ALL_AVAILABLE,
        prompt_loader=_prompt_loader,
    )
    killed_cfg = _load(_parser(egress_enabled="false"))

    assert default_cfg.feature_enabled is False
    assert default_cfg.egress_enabled is False
    assert default_cfg.enabled is False
    assert default_cfg.disabled_reason == "feature_disabled"
    assert default_cfg.target is None
    assert default_cfg.generation_config_revision is None
    assert killed_cfg.feature_enabled is True
    assert killed_cfg.egress_enabled is False
    assert killed_cfg.enabled is False
    assert killed_cfg.disabled_reason == "egress_disabled"
    assert killed_cfg.generation_config_revision is None


def test_valid_snapshot_is_frozen_bounded_and_has_deterministic_revision() -> None:
    first = _load()
    second = _load()

    assert first == second
    assert dataclasses.is_dataclass(first)
    with pytest.raises(dataclasses.FrozenInstanceError):
        first.enabled = False  # type: ignore[misc]
    assert first.enabled is True
    assert first.disabled_reason is None
    assert first.target is not None
    assert first.target.provider == "openai"
    assert first.target.model == "gpt-4o-mini"
    assert first.target.adapter_id == "openai_official_chat_v1"
    assert first.target.endpoint_identity == "https://api.openai.com:443/v1/chat/completions"
    assert first.prompt is not None
    assert first.prompt.text == "maintained standalone prompt"
    assert first.prompt.byte_count == len(first.prompt.text.encode("utf-8"))
    assert first.prompt.sha256 == hashlib.sha256(first.prompt.text.encode("utf-8")).hexdigest()
    assert first.prompt.contract_version == "slides.standalone_html.v1"
    assert first.input_limits.max_request_bytes == 4_194_304
    assert first.input_limits.max_source_chars == 200_000
    assert first.input_limits.max_source_tokens == 50_000
    assert first.output_limits.max_provider_response_bytes == 8_388_608
    assert first.output_limits.max_document_bytes == 1_048_576
    assert first.provider_limits.max_output_tokens == 16_384
    assert first.generation_config_revision is not None
    assert first.generation_config_revision.startswith("sha256:")
    assert len(first.generation_config_revision) == 71


def test_default_environment_is_used_when_no_explicit_mapping_is_supplied(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = {
        "SLIDES_STANDALONE_ENABLED": "true",
        "SLIDES_STANDALONE_EGRESS_ENABLED": "true",
        "SLIDES_STANDALONE_DEFAULT_PROVIDER": "anthropic",
        "SLIDES_STANDALONE_DEFAULT_MODEL": "Env-Model",
        "SLIDES_STANDALONE_DEFAULT_ADAPTER_ID": "anthropic_official_messages_v1",
        "SLIDES_STANDALONE_ALLOWED_TARGETS_JSON": json.dumps(
            [
                {
                    "provider": "anthropic",
                    "model": "Env-Model",
                    "adapter_id": "anthropic_official_messages_v1",
                }
            ]
        ),
    }
    for key, value in values.items():
        monkeypatch.setenv(key, value)

    cfg = load_standalone_html_config(
        ConfigParser(interpolation=None),
        availability=_ALL_AVAILABLE,
        prompt_loader=_prompt_loader,
    )

    assert cfg.enabled is True
    assert cfg.target is not None
    assert cfg.target.model == "Env-Model"
    assert cfg.target.adapter_id == "anthropic_official_messages_v1"


def test_effective_limits_only_clamp_downward() -> None:
    lowered = _load(
        _parser(
            connect_timeout_seconds="2.5",
            read_timeout_seconds="30",
            overall_timeout_seconds="60",
            max_output_tokens="2048",
            max_source_chars="1234",
            max_source_tokens="567",
            max_provider_response_bytes="999999",
        )
    )
    raised = _load(
        _parser(
            connect_timeout_seconds="999",
            read_timeout_seconds="999",
            overall_timeout_seconds="999",
            max_output_tokens="999999",
            max_source_chars="999999999",
            max_source_tokens="999999999",
            max_provider_response_bytes="999999999",
        )
    )

    assert dataclasses.asdict(lowered.provider_limits) == {
        "connect_timeout_seconds": 2.5,
        "read_timeout_seconds": 30.0,
        "overall_timeout_seconds": 60.0,
        "max_output_tokens": 2048,
    }
    assert lowered.input_limits.max_source_chars == 1234
    assert lowered.input_limits.max_source_tokens == 567
    assert lowered.output_limits.max_provider_response_bytes == 999999
    assert dataclasses.asdict(raised.provider_limits) == {
        "connect_timeout_seconds": 10.0,
        "read_timeout_seconds": 120.0,
        "overall_timeout_seconds": 180.0,
        "max_output_tokens": 32768,
    }
    assert raised.input_limits.max_source_chars == 200_000
    assert raised.input_limits.max_source_tokens == 50_000
    assert raised.output_limits.max_provider_response_bytes == 8_388_608


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("enabled", "maybe"),
        ("egress_enabled", "maybe"),
        ("connect_timeout_seconds", "nan"),
        ("read_timeout_seconds", "0"),
        ("overall_timeout_seconds", "-1"),
        ("max_output_tokens", "0"),
        ("max_source_chars", "0"),
        ("max_source_tokens", "not-an-int"),
        ("max_provider_response_bytes", "-1"),
    ],
)
def test_security_configuration_rejects_malformed_scalars_without_echo(field: str, value: str) -> None:
    with pytest.raises(StandaloneHtmlConfigError) as caught:
        _load(_parser(**{field: value}))

    assert str(caught.value) == "standalone_html_config_invalid"
    assert value not in repr(caught.value)


def test_configparser_interpolation_errors_are_normalized_without_a_chain() -> None:
    source = _parser()
    parser = ConfigParser()
    parser.read_dict({"SlidesStandaloneHtml": dict(source.items("SlidesStandaloneHtml", raw=True))})
    parser.set("SlidesStandaloneHtml", "enabled", "%(missing_option)s")

    with pytest.raises(StandaloneHtmlConfigError, match="^standalone_html_config_invalid$") as caught:
        _load(parser)

    assert caught.value.__cause__ is None
    assert caught.value.__suppress_context__ is True


@pytest.mark.parametrize(
    "allowed_targets_json",
    [
        "{}",
        "null",
        "[NaN]",
        '[{"provider":"openai","provider":"anthropic","model":"m","adapter_id":"openai_official_chat_v1"}]',
        '[{"provider":"openai","model":"m","adapter_id":"openai_official_chat_v1","endpoint":"https://evil.example"}]',
        '[{"provider":"*","model":"m","adapter_id":"openai_official_chat_v1"}]',
        '[{"provider":"openai","model":"*","adapter_id":"openai_official_chat_v1"}]',
        '[{"provider":"openai","model":"m","adapter_id":"custom_openai"}]',
        '[{"provider":"openai","model":"m","adapter_id":"openai_official_chat_v1"},{"provider":"openai","model":"m","adapter_id":"openai_official_chat_v1"}]',
    ],
)
def test_allowed_target_json_rejects_malformed_or_open_ended_entries(
    allowed_targets_json: str,
) -> None:
    with pytest.raises(StandaloneHtmlConfigError, match="^standalone_html_config_invalid$"):
        _load(_parser(allowed_targets_json=allowed_targets_json))


@pytest.mark.parametrize("provider", [" openai", "openai ", "\topenai"])
def test_allowed_target_provider_rejects_surrounding_whitespace(provider: str) -> None:
    allowed = json.dumps(
        [
            {
                "provider": provider,
                "model": "gpt-4o-mini",
                "adapter_id": "openai_official_chat_v1",
            }
        ]
    )

    with pytest.raises(StandaloneHtmlConfigError, match="^standalone_html_config_invalid$"):
        _load(_parser(allowed_targets_json=allowed))


def test_allowed_target_json_is_byte_bounded_before_decode(monkeypatch: pytest.MonkeyPatch) -> None:
    decode_calls = 0

    def fail_decode(*_args: object, **_kwargs: object) -> object:
        nonlocal decode_calls
        decode_calls += 1
        raise AssertionError("oversized JSON must be rejected before decoding")

    monkeypatch.setattr(config_mod.json, "loads", fail_decode)
    oversized = "[" + (" " * ALLOWED_TARGETS_JSON_MAX_BYTES) + "]"

    with pytest.raises(StandaloneHtmlConfigError, match="^standalone_html_config_invalid$"):
        _load(_parser(allowed_targets_json=oversized))

    assert decode_calls == 0


def test_allowed_target_json_counts_outer_whitespace_before_trimming(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    decode_calls = 0

    def fail_decode(*_args: object, **_kwargs: object) -> object:
        nonlocal decode_calls
        decode_calls += 1
        raise AssertionError("oversized outer whitespace must be rejected before decoding")

    monkeypatch.setattr(config_mod.json, "loads", fail_decode)
    oversized = (" " * ALLOWED_TARGETS_JSON_MAX_BYTES) + "[]"

    with pytest.raises(StandaloneHtmlConfigError, match="^standalone_html_config_invalid$"):
        load_standalone_html_config(
            _parser(),
            env={"SLIDES_STANDALONE_ALLOWED_TARGETS_JSON": oversized},
            availability=_ALL_AVAILABLE,
            prompt_loader=_prompt_loader,
        )

    assert decode_calls == 0


def test_oversized_allowed_target_text_is_rejected_before_encoding() -> None:
    class EncodingBomb(str):
        def encode(self, *_args: object, **_kwargs: object) -> bytes:
            raise AssertionError("oversized text must be rejected before encoding")

    oversized = EncodingBomb("x" * (ALLOWED_TARGETS_JSON_MAX_BYTES + 1))

    with pytest.raises(StandaloneHtmlConfigError, match="^standalone_html_config_invalid$"):
        config_mod._parse_allowed_targets(oversized)


def test_allowed_target_catalog_is_bounded_to_64_exact_tuples() -> None:
    at_limit = [
        {
            "provider": "openai",
            "model": f"model-{index}",
            "adapter_id": "openai_official_chat_v1",
        }
        for index in range(ALLOWED_TARGETS_MAX_ENTRIES)
    ]
    at_limit[0]["model"] = "gpt-4o-mini"

    assert _load(_parser(allowed_targets_json=json.dumps(at_limit))).enabled is True
    with pytest.raises(StandaloneHtmlConfigError, match="^standalone_html_config_invalid$"):
        _load(
            _parser(
                allowed_targets_json=json.dumps(
                    [
                        *at_limit,
                        {
                            "provider": "anthropic",
                            "model": "overflow-model",
                            "adapter_id": "anthropic_official_messages_v1",
                        },
                    ]
                )
            )
        )


@pytest.mark.parametrize(
    "forbidden_option",
    [
        "base_url",
        "endpoint",
        "endpoint_override",
        "proxy",
        "router",
        "fallback_provider",
        "fallback_model",
        "custom_adapter",
        "verify_tls",
    ],
)
def test_endpoint_router_proxy_fallback_and_tls_overrides_are_rejected(
    forbidden_option: str,
) -> None:
    parser = _parser()
    parser.set("SlidesStandaloneHtml", forbidden_option, "configured")

    with pytest.raises(StandaloneHtmlConfigError, match="^standalone_html_config_invalid$"):
        _load(parser)


@pytest.mark.parametrize("unknown_option", ["hmac_keys_json", "enable_fallback", "future_endpoint_mode"])
def test_closed_section_rejects_every_unknown_option(unknown_option: str) -> None:
    parser = _parser()
    parser.set("SlidesStandaloneHtml", unknown_option, "configured")

    with pytest.raises(StandaloneHtmlConfigError, match="^standalone_html_config_invalid$"):
        _load(parser)


@pytest.mark.parametrize(
    "env_key",
    [
        "SLIDES_STANDALONE_BASE_URL",
        "SLIDES_STANDALONE_ENDPOINT",
        "SLIDES_STANDALONE_PROXY",
        "SLIDES_STANDALONE_ROUTER",
        "SLIDES_STANDALONE_FALLBACK_PROVIDER",
        "SLIDES_STANDALONE_VERIFY_TLS",
    ],
)
def test_environment_endpoint_and_routing_overrides_are_rejected(env_key: str) -> None:
    with pytest.raises(StandaloneHtmlConfigError, match="^standalone_html_config_invalid$"):
        load_standalone_html_config(
            _parser(),
            env={env_key: "configured"},
            availability=_ALL_AVAILABLE,
            prompt_loader=_prompt_loader,
        )


def test_default_target_requires_exact_allowlist_membership_and_model_case() -> None:
    wrong_case = _load(_parser(default_model="GPT-4O-MINI"))
    missing_model = _load(_parser(default_model=""))
    wrong_adapter = _load(_parser(default_adapter_id="anthropic_official_messages_v1"))

    assert wrong_case.enabled is False
    assert wrong_case.disabled_reason == "default_model_not_allowed"
    assert missing_model.enabled is False
    assert missing_model.disabled_reason == "default_model_not_configured"
    assert wrong_adapter.enabled is False
    assert wrong_adapter.disabled_reason == "default_endpoint_not_allowed"


@pytest.mark.parametrize(
    "model",
    [
        "bad\nmodel",
        "bad\x00model",
        "bad\x7fmodel",
        "bad\ud800model",
        "é" * 129,
    ],
)
def test_model_identifiers_are_bounded_unicode_scalars_without_controls(model: str) -> None:
    allowed = json.dumps(
        [
            {
                "provider": "openai",
                "model": model,
                "adapter_id": "openai_official_chat_v1",
            }
        ]
    )

    with pytest.raises(StandaloneHtmlConfigError, match="^standalone_html_config_invalid$"):
        _load(_parser(default_model=model, allowed_targets_json=allowed))


def test_oversized_model_is_rejected_before_normalization_or_encoding() -> None:
    class ModelBomb(str):
        def strip(self, *_args: object, **_kwargs: object) -> str:
            raise AssertionError("oversized model must be rejected before normalization")

    with pytest.raises(StandaloneHtmlConfigError, match="^standalone_html_config_invalid$"):
        config_mod._validate_model(ModelBomb("x" * 257))


@pytest.mark.parametrize(
    ("availability_field", "reason"),
    [
        ("digest_key_available", "digest_key_unavailable"),
        ("worker_handler_registered", "generation_worker_unavailable"),
        ("reconciler_admission_ready", "generation_reconciler_overloaded"),
        ("validator_available", "validator_unavailable"),
    ],
)
def test_dynamic_availability_fails_closed_with_safe_reason(
    availability_field: str,
    reason: str,
) -> None:
    availability = dataclasses.replace(_ALL_AVAILABLE, **{availability_field: False})
    cfg = load_standalone_html_config(
        _parser(),
        env={},
        availability=availability,
        prompt_loader=_prompt_loader,
    )

    assert cfg.enabled is False
    assert cfg.disabled_reason == reason
    assert cfg.generation_config_revision is None


def test_validator_unavailable_takes_precedence_across_dynamic_failures() -> None:
    cfg = load_standalone_html_config(
        _parser(),
        env={},
        availability=StandaloneHtmlGenerationAvailability(
            digest_key_available=False,
            worker_handler_registered=False,
            reconciler_admission_ready=False,
            validator_available=False,
        ),
        prompt_loader=_prompt_loader,
    )

    assert cfg.enabled is False
    assert cfg.disabled_reason == "validator_unavailable"


@pytest.mark.parametrize(
    ("availability_field", "invalid_value"),
    [
        ("digest_key_available", "true"),
        ("worker_handler_registered", 1),
        ("reconciler_admission_ready", None),
        ("validator_available", 0),
    ],
)
def test_dynamic_availability_requires_actual_booleans(
    availability_field: str,
    invalid_value: object,
) -> None:
    values: dict[str, object] = {
        "digest_key_available": True,
        "worker_handler_registered": True,
        "reconciler_admission_ready": True,
        "validator_available": True,
    }
    values[availability_field] = invalid_value

    with pytest.raises(StandaloneHtmlConfigError, match="^standalone_html_config_invalid$"):
        StandaloneHtmlGenerationAvailability(**values)  # type: ignore[arg-type]


def test_config_loader_rejects_wrong_availability_dto_without_attribute_errors() -> None:
    with pytest.raises(StandaloneHtmlConfigError, match="^standalone_html_config_invalid$") as caught:
        load_standalone_html_config(
            _parser(),
            env={},
            availability=object(),  # type: ignore[arg-type]
            prompt_loader=_prompt_loader,
        )

    assert caught.value.__cause__ is None
    assert caught.value.__suppress_context__ is True


@pytest.mark.parametrize(
    ("change", "value"),
    [
        ("default_model", "gpt-4o"),
        ("connect_timeout_seconds", "9"),
        ("read_timeout_seconds", "119"),
        ("overall_timeout_seconds", "179"),
        ("max_output_tokens", "16000"),
        ("max_source_chars", "199999"),
        ("max_source_tokens", "49999"),
        ("max_provider_response_bytes", "8388607"),
    ],
)
def test_revision_changes_for_every_effective_static_component(change: str, value: str) -> None:
    baseline = _load()
    overrides: dict[str, object] = {change: value}
    if change == "default_model":
        overrides["allowed_targets_json"] = json.dumps(
            [
                {
                    "provider": "openai",
                    "model": value,
                    "adapter_id": "openai_official_chat_v1",
                }
            ]
        )
    changed = _load(_parser(**overrides))

    assert changed.generation_config_revision != baseline.generation_config_revision


def test_revision_uses_prompt_digest_and_contract_but_not_raw_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    first = _load()

    def changed_prompt(_module: str, _key: str, _max_bytes: int) -> str:
        return "changed maintained prompt"

    second = load_standalone_html_config(
        _parser(),
        env={},
        availability=_ALL_AVAILABLE,
        prompt_loader=changed_prompt,
    )
    assert second.generation_config_revision != first.generation_config_revision
    assert second.prompt is not None
    revision_manifest = second.revision_manifest
    assert second.prompt.text not in revision_manifest
    assert second.prompt.sha256 in revision_manifest
    assert PROMPT_CONTRACT_VERSION in revision_manifest


@pytest.mark.parametrize("prompt", ["unsafe\x00prompt", "unsafe\ud800prompt"])
def test_injected_prompt_loader_is_revalidated(prompt: str) -> None:
    def injected(_module: str, _key: str, _max_bytes: int) -> str:
        return prompt

    cfg = load_standalone_html_config(
        _parser(),
        env={},
        availability=_ALL_AVAILABLE,
        prompt_loader=injected,
    )

    assert cfg.enabled is False
    assert cfg.disabled_reason == "prompt_asset_unavailable"
    assert cfg.prompt is None


def test_oversized_injected_prompt_is_rejected_before_encoding() -> None:
    class PromptBomb(str):
        def strip(self, *_args: object, **_kwargs: object) -> PromptBomb:
            return self

        def encode(self, *_args: object, **_kwargs: object) -> bytes:
            raise AssertionError("oversized prompt must be rejected before encoding")

    def injected(_module: str, _key: str, _max_bytes: int) -> str:
        return PromptBomb("x" * (PROMPT_MAX_BYTES + 1))

    cfg = load_standalone_html_config(
        _parser(),
        env={},
        availability=_ALL_AVAILABLE,
        prompt_loader=injected,
    )

    assert cfg.enabled is False
    assert cfg.disabled_reason == "prompt_asset_unavailable"


def test_revision_manifest_is_canonical_utf8_without_ascii_escaping() -> None:
    model = "模型-Ä"
    cfg = _load(
        _parser(
            default_model=model,
            allowed_targets_json=json.dumps(
                [
                    {
                        "provider": "openai",
                        "model": model,
                        "adapter_id": "openai_official_chat_v1",
                    }
                ]
            ),
        )
    )

    assert model in cfg.revision_manifest
    assert "\\u6a21" not in cfg.revision_manifest


def test_strict_prompt_loads_packaged_asset_and_enforces_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TLDW_PROMPT_FILE_SLIDES__STANDALONE_HTML_SYSTEM", raising=False)

    prompt = load_prompt_strict("slides", "standalone_html_system", PROMPT_MAX_BYTES)

    assert len(prompt.encode("utf-8")) <= 131_072
    required_fragments = (
        "one complete self-contained HTML document",
        '<section class="slide">',
        "inline CSS",
        "inline JavaScript",
        "prefers-reduced-motion",
        "speaker-led",
        "self-guided",
        "Do not wrap",
        "must implement bounded in-document slide selection",
        "ArrowLeft, ArrowRight, Home, and End",
        "N key",
        "current slide",
        "Concept 1, Concept 2",
        "Story 1, Story 2",
        "Punchy Point 1, Punchy Point 2",
        "Option A, Option B",
        "Phase 1, Phase 2, Phase 3",
        "Preserve supplied citations",
        "nonlinked visible text",
        "@font-face",
        "source-map directives",
        "analytics or telemetry",
        "citation URLs only as inert nonlinked text",
        "URL-bearing attributes, CSS values, or script construction",
        "accessible document landmarks",
        "Do not autoplay or auto-advance",
    )
    assert all(fragment in prompt for fragment in required_fragments)
    forbidden_fragments = (
        "html-ppt-skill",
        "https://",
        "http://",
        "GSAP",
        "ScrollTrigger",
        "BroadcastChannel",
        "postMessage",
        "<iframe",
    )
    assert all(fragment not in prompt for fragment in forbidden_fragments)


def test_strict_prompt_override_is_exact_and_bounded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    override = tmp_path / "standalone.txt"
    override.write_text("override prompt", encoding="utf-8")
    monkeypatch.setenv("TLDW_PROMPT_FILE_SLIDES__STANDALONE_HTML_SYSTEM", str(override))

    assert load_prompt_strict("slides", "standalone_html_system", 15) == "override prompt"
    with pytest.raises(PromptAssetUnavailableError, match="^prompt_asset_unavailable$"):
        load_prompt_strict("slides", "standalone_html_system", 14)


@pytest.mark.parametrize("override_kind", ["missing", "directory", "symlink", "invalid_utf8", "blank"])
def test_configured_bad_prompt_override_fails_without_packaged_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    override_kind: str,
) -> None:
    override = tmp_path / "override.txt"
    if override_kind == "directory":
        override.mkdir()
    elif override_kind == "symlink":
        target = tmp_path / "target.txt"
        target.write_text("valid override", encoding="utf-8")
        override.symlink_to(target)
    elif override_kind == "invalid_utf8":
        override.write_bytes(b"\xff")
    elif override_kind == "blank":
        override.write_text(" \n\t", encoding="utf-8")
    monkeypatch.setenv("TLDW_PROMPT_FILE_SLIDES__STANDALONE_HTML_SYSTEM", str(override))

    with pytest.raises(PromptAssetUnavailableError, match="^prompt_asset_unavailable$"):
        load_prompt_strict("slides", "standalone_html_system", PROMPT_MAX_BYTES)


def test_present_but_blank_prompt_override_env_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TLDW_PROMPT_FILE_SLIDES__STANDALONE_HTML_SYSTEM", " \t ")

    with pytest.raises(PromptAssetUnavailableError, match="^prompt_asset_unavailable$"):
        load_prompt_strict("slides", "standalone_html_system", PROMPT_MAX_BYTES)


def test_strict_prompt_normalizes_pathological_parser_depth(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_dir = tmp_path / "config"
    prompts_dir = config_dir / "Prompts"
    prompts_dir.mkdir(parents=True)
    (prompts_dir / "slides.prompts.json").write_text(
        "[" * 2_000 + "0" + "]" * 2_000,
        encoding="utf-8",
    )
    monkeypatch.setenv("TLDW_CONFIG_DIR", str(config_dir))
    monkeypatch.delenv("TLDW_PROMPT_FILE_SLIDES__STANDALONE_HTML_SYSTEM", raising=False)

    with pytest.raises(PromptAssetUnavailableError, match="^prompt_asset_unavailable$") as caught:
        load_prompt_strict("slides", "standalone_html_system", PROMPT_MAX_BYTES)

    assert caught.value.__cause__ is None
    assert caught.value.__suppress_context__ is True


def test_strict_json_prompt_rejects_duplicate_members(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_dir = tmp_path / "config"
    prompts_dir = config_dir / "Prompts"
    prompts_dir.mkdir(parents=True)
    (prompts_dir / "slides.prompts.json").write_text(
        '{"standalone_html_system":"first","standalone_html_system":"second"}',
        encoding="utf-8",
    )
    monkeypatch.setenv("TLDW_CONFIG_DIR", str(config_dir))
    monkeypatch.delenv("TLDW_PROMPT_FILE_SLIDES__STANDALONE_HTML_SYSTEM", raising=False)

    with pytest.raises(PromptAssetUnavailableError, match="^prompt_asset_unavailable$"):
        load_prompt_strict("slides", "standalone_html_system", PROMPT_MAX_BYTES)


def test_strict_prompt_mapping_rejects_normalized_key_aliases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_dir = tmp_path / "config"
    prompts_dir = config_dir / "Prompts"
    prompts_dir.mkdir(parents=True)
    (prompts_dir / "slides.prompts.json").write_text(
        json.dumps(
            {
                "standalone_html_system": "first",
                "Standalone HTML System": "second",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("TLDW_CONFIG_DIR", str(config_dir))
    monkeypatch.delenv("TLDW_PROMPT_FILE_SLIDES__STANDALONE_HTML_SYSTEM", raising=False)

    with pytest.raises(PromptAssetUnavailableError, match="^prompt_asset_unavailable$"):
        load_prompt_strict("slides", "standalone_html_system", PROMPT_MAX_BYTES)


def test_strict_markdown_prompt_rejects_duplicate_matching_headings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_dir = tmp_path / "config"
    prompts_dir = config_dir / "Prompts"
    prompts_dir.mkdir(parents=True)
    (prompts_dir / "slides.prompts.md").write_text(
        "# standalone_html_system\n```\nfirst\n```\n" "# standalone_html_system\n```\nsecond\n```\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("TLDW_CONFIG_DIR", str(config_dir))
    monkeypatch.delenv("TLDW_PROMPT_FILE_SLIDES__STANDALONE_HTML_SYSTEM", raising=False)

    with pytest.raises(PromptAssetUnavailableError, match="^prompt_asset_unavailable$"):
        load_prompt_strict("slides", "standalone_html_system", PROMPT_MAX_BYTES)


def test_strict_prompt_context_integrity_failure_does_not_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core.Context_Integrity.models import (
        ContextIntegrityBootState,
        ContextIntegrityFinding,
    )
    from tldw_Server_API.app.core.Context_Integrity.resolver import (
        ContextIntegrityResolver,
        clear_global_context_integrity_resolver,
        set_global_context_integrity_resolver,
    )

    override = tmp_path / "override.txt"
    override.write_text("blocked override", encoding="utf-8")
    env_name = "TLDW_PROMPT_FILE_SLIDES__STANDALONE_HTML_SYSTEM"
    monkeypatch.setenv(env_name, str(override))
    resolver = ContextIntegrityResolver(
        ContextIntegrityBootState(
            mode="enforce",
            degraded=False,
            manifest_sequence=1,
            manifest_digest="sha256:manifest",
            findings=(
                ContextIntegrityFinding(
                    asset_id=f"prompt_file:env:{env_name}:{override.name}",
                    state="changed_approved_non_executable",
                    severity="warning",
                    summary="changed",
                    remediation="review",
                    source_type="prompt_file",
                ),
            ),
        )
    )
    set_global_context_integrity_resolver(resolver)
    try:
        with pytest.raises(PromptAssetUnavailableError, match="^prompt_asset_unavailable$"):
            load_prompt_strict("slides", "standalone_html_system", PROMPT_MAX_BYTES)
    finally:
        clear_global_context_integrity_resolver()


def test_legacy_prompt_loader_keeps_existing_unreadable_override_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_dir = tmp_path / "config"
    prompts_dir = config_dir / "Prompts"
    prompts_dir.mkdir(parents=True)
    (prompts_dir / "demo.prompts.md").write_text(
        "# Existing Key\n```\nlegacy fallback\n```\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("TLDW_CONFIG_DIR", str(config_dir))
    monkeypatch.setenv("TLDW_PROMPT_FILE_DEMO__EXISTING_KEY", str(tmp_path / "missing.txt"))

    assert load_prompt("demo", "Existing Key") == "legacy fallback"
