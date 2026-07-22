from __future__ import annotations

import pytest


def test_config_loader_honors_explicit_environment_absence(monkeypatch):
    """Provider env values absent from a supplied snapshot cannot appear later."""
    from tldw_Server_API.app.core import config

    monkeypatch.setenv("OPENAI_API_KEY", "late-openai-key-canary")
    monkeypatch.setenv(
        "CUSTOM_OPENAI_API_URL",
        "https://late-custom-openai.example/v1",
    )
    monkeypatch.setenv("BEDROCK_REGION", "late-bedrock-region-canary")
    monkeypatch.setenv("RAG_DEFAULT_LLM_PROVIDER", "late-rag-provider-canary")
    monkeypatch.setenv("RAG_DEFAULT_LLM_MODEL", "late-rag-model-canary")

    loaded = config.load_and_log_configs(environment={})

    assert loaded["openai_api"]["api_key"] != "late-openai-key-canary"
    assert (
        loaded["custom_openai_api"]["api_ip"]
        != "https://late-custom-openai.example/v1"
    )
    assert loaded["bedrock_api"]["region"] != "late-bedrock-region-canary"
    assert loaded["RAG_DEFAULT_LLM_PROVIDER"] != "late-rag-provider-canary"
    assert loaded["RAG_DEFAULT_LLM_MODEL"] != "late-rag-model-canary"


def test_provider_identity_contract_is_canonical_first_and_deterministic():
    from tldw_Server_API.app.core.LLM_Calls.provider_identity import (
        canonical_provider_name,
        provider_lookup_names,
    )

    assert canonical_provider_name("OAI") == "openai"
    assert provider_lookup_names("oai") == ("openai", "oai")
    assert provider_lookup_names("openai-compatible") == (
        "custom-openai-api",
        "custom-openai_api",
        "custom_openai-api",
        "custom_openai_api",
        "custom-openai",
        "custom_openai",
        "openai-compatible",
        "openai_compatible",
        "customopenai",
    )
    assert canonical_provider_name("aws_bedrock") == "bedrock"
    assert provider_lookup_names("aws_bedrock") == (
        "bedrock",
        "aws-bedrock",
        "aws_bedrock",
        "amazon-bedrock",
        "amazon_bedrock",
    )


def test_unknown_provider_identity_preserves_underscore_spelling():
    from tldw_Server_API.app.core.LLM_Calls.provider_identity import (
        canonical_provider_name,
        provider_lookup_names,
    )

    assert canonical_provider_name(" Foo_Bar ") == "foo_bar"
    assert provider_lookup_names("foo_bar") == ("foo_bar",)


def test_legacy_storage_normalizer_remains_noncanonical():
    from tldw_Server_API.app.core.AuthNZ.user_provider_secrets import normalize_provider_name

    assert normalize_provider_name(" OAI ") == "oai"
    assert normalize_provider_name(" Foo_Bar ") == "foo_bar"


def test_default_allowlist_accepts_registered_alias(monkeypatch):
    from types import SimpleNamespace

    from tldw_Server_API.app.core.AuthNZ import byok_helpers

    monkeypatch.setattr(
        byok_helpers,
        "get_settings",
        lambda: SimpleNamespace(BYOK_ALLOWED_PROVIDERS=[]),
    )

    assert byok_helpers.is_provider_allowlisted("oai") is True


def test_validate_credential_fields_default_allowlist():
    from tldw_Server_API.app.core.AuthNZ.byok_helpers import validate_credential_fields

    fields = {"org_id": "org-123", "project_id": "proj-456"}
    cleaned = validate_credential_fields("unknown-provider", fields)
    assert cleaned == fields

    with pytest.raises(ValueError):
        validate_credential_fields("unknown-provider", {"api_key": "nope"})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("org_id", 123),
        ("project_id", True),
        ("org_id", "org-good\r\nInjected: yes"),
        ("project_id", "project\x00hidden"),
        ("org_id", "é"),
        ("project_id", "p" * 513),
    ],
)
def test_validate_credential_header_fields_reject_unsafe_values(field, value):
    """Credential-derived HTTP header values must be bounded printable ASCII strings."""
    from tldw_Server_API.app.core.AuthNZ.byok_helpers import validate_credential_fields

    with pytest.raises(ValueError):
        validate_credential_fields("openai", {field: value})


def test_validate_credential_fields_required_policy(monkeypatch):
    from tldw_Server_API.app.core.AuthNZ.byok_helpers import validate_credential_fields
    from tldw_Server_API.app.core.LLM_Calls import provider_metadata

    monkeypatch.setitem(
        provider_metadata.BYOK_CREDENTIAL_FIELDS,
        "test-provider",
        {"allowed": {"org_id"}, "required": {"org_id"}},
    )

    cleaned = validate_credential_fields("test-provider", {"org_id": "org-789"})
    assert cleaned == {"org_id": "org-789"}

    with pytest.raises(ValueError):
        validate_credential_fields("test-provider", {})


def test_server_default_key_does_not_bypass_unhealthy_override_store(monkeypatch):
    """A configured env key cannot hide an unavailable canonical credential store."""
    from tldw_Server_API.app.core.AuthNZ import byok_helpers
    from tldw_Server_API.app.core.AuthNZ.byok_runtime import ByokResolutionError
    from tldw_Server_API.app.core.AuthNZ.llm_provider_overrides import (
        LLMProviderOverride,
        set_llm_provider_overrides_cache_for_tests,
    )

    monkeypatch.setenv("OPENAI_API_KEY", "env-key-must-not-be-used")
    set_llm_provider_overrides_cache_for_tests(
        {"openai": LLMProviderOverride(provider="openai", api_key="stale-key")},
        healthy=False,
    )
    try:
        with pytest.raises(ByokResolutionError) as exc_info:
            byok_helpers.resolve_server_default_key("openai")
    finally:
        set_llm_provider_overrides_cache_for_tests({})

    assert exc_info.value.code == "credential_store_unavailable"


def test_static_server_default_key_can_exclude_override_after_atomic_lookup(monkeypatch):
    """An authoritative caller can continue with static env/config only."""
    from tldw_Server_API.app.core.AuthNZ import byok_helpers
    from tldw_Server_API.app.core.AuthNZ.llm_provider_overrides import (
        LLMProviderOverride,
        set_llm_provider_overrides_cache_for_tests,
    )

    monkeypatch.setenv("OPENAI_API_KEY", "configured-static-key")
    set_llm_provider_overrides_cache_for_tests(
        {"openai": LLMProviderOverride(provider="openai", api_key="late-override-key")}
    )
    try:
        resolved = byok_helpers.resolve_server_default_key(
            "openai",
            include_override=False,
        )
    finally:
        set_llm_provider_overrides_cache_for_tests({})

    assert resolved == "configured-static-key"


def test_server_snapshot_skips_placeholder_custom_aliases(monkeypatch):
    """A placeholder alias cannot mask a later key or endpoint in one slot."""
    from tldw_Server_API.app.core.AuthNZ import byok_helpers

    monkeypatch.setattr(byok_helpers, "load_and_log_configs", lambda **_kwargs: {})
    monkeypatch.setenv("CUSTOM_OPENAI2_API_KEY", "CHANGE_ME")
    monkeypatch.setenv("CUSTOM_OPENAI_API_KEY_2", "slot-2-real-key")
    monkeypatch.setenv("CUSTOM_OPENAI2_API_IP", "CHANGE_ME")
    monkeypatch.setenv(
        "CUSTOM_OPENAI2_API_BASE",
        "https://slot-2.example/v1",
    )

    snapshot = byok_helpers.load_server_config_snapshot()

    assert snapshot["custom_openai_api_2"]["api_key"] == "slot-2-real-key"
    assert snapshot["custom_openai_api_2"]["api_ip"] == "https://slot-2.example/v1"


def test_server_snapshot_removes_provider_placeholders_loaded_from_config(monkeypatch):
    """Config-file placeholders must not become an authoritative runtime snapshot."""
    from tldw_Server_API.app.core.AuthNZ import byok_helpers

    monkeypatch.setattr(
        byok_helpers,
        "load_and_log_configs",
        lambda **_kwargs: {
            "custom_openai_api_2": {
                "api_key": "CHANGE_ME",
                "api_ip": "REPLACE-ME",
                "model": "change_me_for_this_deployment",
                "api_timeout": "30",
            }
        },
    )
    for env_name in (
        "CUSTOM_OPENAI2_API_KEY",
        "CUSTOM_OPENAI_API_KEY_2",
        "CUSTOM_OPENAI_API_2_API_KEY",
        "CUSTOM_OPENAI2_API_IP",
        "CUSTOM_OPENAI2_API_BASE",
        "CUSTOM_OPENAI2_API_URL",
        "CUSTOM_OPENAI2_API_BASE_URL",
        "CUSTOM_OPENAI2_BASE_URL",
        "CUSTOM_OPENAI_API_2_IP",
        "CUSTOM_OPENAI_API_2_BASE",
        "CUSTOM_OPENAI_API_2_URL",
        "CUSTOM_OPENAI_API_2_BASE_URL",
        "CUSTOM_OPENAI_API_IP_2",
        "CUSTOM_OPENAI_API_BASE_2",
        "CUSTOM_OPENAI_API_URL_2",
        "CUSTOM_OPENAI_API_BASE_URL_2",
    ):
        monkeypatch.delenv(env_name, raising=False)

    snapshot = byok_helpers.load_server_config_snapshot()

    assert snapshot["custom_openai_api_2"] == {"api_timeout": "30"}


@pytest.mark.parametrize(
    "placeholder",
    ("CHANGE_ME", "change_me_for_this_deployment", "REPLACE-ME"),
)
def test_server_default_key_rejects_placeholders(placeholder):
    from tldw_Server_API.app.core.AuthNZ import byok_helpers

    snapshot = {"openai_api": {"api_key": placeholder}}

    assert byok_helpers.resolve_server_default_key_from_snapshot("openai", snapshot) is None


def test_trusted_base_url_request_rejects_legacy_boolean_only_user_shape():
    from tldw_Server_API.app.core.AuthNZ.byok_helpers import is_trusted_base_url_request

    assert (
        is_trusted_base_url_request(
            None,
            principal=None,
            user={
                "role": "user",
                "roles": ["user"],
                "permissions": [],
                "is_admin": True,
                "is_superuser": True,
            },
        )
        is False
    )


def test_trusted_base_url_request_accepts_permission_claims_from_legacy_user_shape():
    from tldw_Server_API.app.core.AuthNZ.byok_helpers import is_trusted_base_url_request

    assert (
        is_trusted_base_url_request(
            None,
            principal=None,
            user={
                "role": "user",
                "roles": ["user"],
                "permissions": ["system.configure"],
                "is_admin": False,
                "is_superuser": False,
            },
        )
        is True
    )
