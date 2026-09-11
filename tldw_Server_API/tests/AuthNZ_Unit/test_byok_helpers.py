from __future__ import annotations

from types import SimpleNamespace

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


def test_gateway_specs_augment_byok_allowlist_without_discovery(monkeypatch):
    from tldw_Server_API.app.core.AuthNZ import byok_helpers

    monkeypatch.setattr(
        byok_helpers,
        "get_settings",
        lambda: SimpleNamespace(
            BYOK_ALLOWED_PROVIDERS=[
                "openai",
                "gateway:admin-only",
                "gateway:disabled",
                "gateway:orphaned",
            ]
        ),
    )
    monkeypatch.setattr(
        byok_helpers,
        "get_byok_gateway_specs",
        lambda: {
            "openrouter": SimpleNamespace(enabled=True, allow_user_api_key=True),
            "gateway:voice-lab": SimpleNamespace(enabled=True, allow_user_api_key=True),
            "gateway:admin-only": SimpleNamespace(enabled=True, allow_user_api_key=False),
            "gateway:disabled": SimpleNamespace(enabled=False, allow_user_api_key=True),
        },
        raising=False,
    )

    assert byok_helpers.resolve_byok_allowlist() == {
        "openai",
        "openrouter",
        "gateway:voice-lab",
    }


def test_gateway_config_failure_preserves_static_allowlist_and_fails_dynamic_closed(
    monkeypatch,
):
    from tldw_Server_API.app.core.AuthNZ import byok_helpers

    log_calls: list[tuple[str, tuple[object, ...]]] = []

    class _Logger:
        def warning(self, message: str, *args: object) -> None:
            log_calls.append((message, args))

    def _fail_gateway_specs():
        raise RuntimeError("sensitive gateway config detail")

    monkeypatch.setattr(
        byok_helpers,
        "get_settings",
        lambda: SimpleNamespace(
            BYOK_ALLOWED_PROVIDERS=[
                "openai",
                "anthropic",
                "gateway:voice-lab",
            ]
        ),
    )
    monkeypatch.setattr(byok_helpers, "logger", _Logger(), raising=False)
    monkeypatch.setattr(byok_helpers, "get_byok_gateway_specs", _fail_gateway_specs)

    assert byok_helpers.resolve_byok_allowlist() == {"openai", "anthropic"}
    assert byok_helpers.is_provider_allowlisted("openai") is True
    assert byok_helpers.is_provider_allowlisted("gateway:voice-lab") is False
    assert log_calls
    assert all(call[1] == ("RuntimeError",) for call in log_calls)
    assert "sensitive gateway config detail" not in repr(log_calls)


@pytest.mark.parametrize(
    "credential_fields",
    [
        {"base_url": "https://evil.example/v1"},
        {"url": "https://evil.example/v1"},
        {"headers": {"Authorization": "Bearer attacker"}},
        {"auth_scheme": "basic"},
        {"org_id": "metadata-authority"},
    ],
)
def test_gateway_credential_fields_reject_endpoint_authority_for_every_principal(
    monkeypatch,
    credential_fields,
):
    from tldw_Server_API.app.core.AuthNZ import byok_helpers

    monkeypatch.setattr(
        byok_helpers,
        "get_byok_gateway_specs",
        lambda: {
            "gateway:voice-lab": SimpleNamespace(
                enabled=True,
                allow_user_api_key=True,
            )
        },
        raising=False,
    )

    with pytest.raises(ValueError, match="Unsupported credential field"):
        byok_helpers.validate_credential_fields(
            "gateway:voice-lab",
            credential_fields,
            allow_base_url=True,
        )

    assert byok_helpers.validate_credential_fields(
        "gateway:voice-lab",
        {},
        allow_base_url=True,
    ) == {}


def test_openrouter_keeps_general_credential_policy_when_used_by_tts(monkeypatch):
    from tldw_Server_API.app.core.AuthNZ import byok_helpers

    monkeypatch.setattr(
        byok_helpers,
        "get_byok_gateway_specs",
        lambda: {
            "openrouter": SimpleNamespace(enabled=True, allow_user_api_key=True)
        },
        raising=False,
    )

    assert byok_helpers.validate_credential_fields(
        "openrouter",
        {"org_id": "org-general", "project_id": "project-general"},
    ) == {"org_id": "org-general", "project_id": "project-general"}


def _probe_spec(*, enabled=True, discovery_enabled=True, models_path="models"):
    return SimpleNamespace(
        backend_id="gateway:voice-lab",
        enabled=enabled,
        base_url="https://voice.example/v1/",
        models_path=models_path,
        headers=(("X-Admin", "configured"),),
        discovery_query=(("output_modalities", "speech"),),
        discovery=SimpleNamespace(
            enabled=discovery_enabled,
            timeout_seconds=2.5,
        ),
    )


@pytest.mark.asyncio
async def test_gateway_credential_probe_verified_uses_bounded_configured_discovery(
    monkeypatch,
):
    from tldw_Server_API.app.core.AuthNZ import byok_testing

    seen = {}

    async def _fake_afetch_json(**kwargs):
        seen.update(kwargs)
        await kwargs["on_response"](200, {"Content-Type": "application/json"})
        return {"data": [{"id": "Vendor/Voice"}]}

    monkeypatch.setattr(byok_testing, "afetch_json", _fake_afetch_json, raising=False)
    result = await byok_testing.probe_gateway_credentials(
        spec=_probe_spec(),
        api_key="user-key",
    )

    assert result == "verified"
    assert seen["url"] == "https://voice.example/v1/models"
    assert seen["headers"] == {
        "X-Admin": "configured",
        "Authorization": "Bearer user-key",
    }
    assert seen["allow_redirects"] is False
    assert seen["max_bytes"] == 1_048_576


@pytest.mark.asyncio
async def test_gateway_credential_probe_rejects_only_definitive_auth_status(monkeypatch):
    from tldw_Server_API.app.core.AuthNZ import byok_testing

    async def _fake_afetch_json(**kwargs):
        await kwargs["on_response"](401, {"Content-Type": "application/json"})
        return {"error": {"message": "secret upstream detail"}}

    monkeypatch.setattr(byok_testing, "afetch_json", _fake_afetch_json, raising=False)

    assert (
        await byok_testing.probe_gateway_credentials(
            spec=_probe_spec(),
            api_key="rejected-key",
        )
        == "rejected"
    )


@pytest.mark.asyncio
async def test_gateway_credential_probe_unavailable_is_stored_unverified(monkeypatch):
    from tldw_Server_API.app.core.AuthNZ import byok_testing
    from tldw_Server_API.app.core.exceptions import NetworkError

    async def _fake_afetch_json(**_kwargs):
        raise NetworkError("upstream details must not escape")

    monkeypatch.setattr(byok_testing, "afetch_json", _fake_afetch_json, raising=False)

    assert (
        await byok_testing.probe_gateway_credentials(
            spec=_probe_spec(),
            api_key="unverified-key",
        )
        == "stored-unverified"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("declared_length", [None, "2"])
async def test_gateway_credential_probe_stops_oversized_chunked_discovery_early(
    monkeypatch,
    declared_length,
):
    import httpx

    from tldw_Server_API.app.core.AuthNZ import byok_testing
    from tldw_Server_API.app.core.http_client import afetch_json, create_async_client

    chunks = [b'{"data":["', b"x" * 700_000, b"y" * 700_000, b'"]}']

    class _CountingStream(httpx.AsyncByteStream):
        def __init__(self) -> None:
            self.yielded = 0
            self.close_count = 0

        async def __aiter__(self):
            for chunk in chunks:
                self.yielded += 1
                yield chunk

        async def aclose(self) -> None:
            self.close_count += 1

    stream = _CountingStream()
    headers = {"Content-Type": "application/json"}
    if declared_length is not None:
        headers["Content-Length"] = declared_length

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            request=request,
            headers=headers,
            stream=stream,
        )

    client = create_async_client(transport=httpx.MockTransport(handler))

    async def _central_fetch(**kwargs):
        return await afetch_json(client=client, **kwargs)

    monkeypatch.setattr(byok_testing, "afetch_json", _central_fetch)
    try:
        result = await byok_testing.probe_gateway_credentials(
            spec=_probe_spec(),
            api_key="user-key",
        )
    finally:
        await client.aclose()

    assert result == "stored-unverified"
    assert stream.yielded < len(chunks)
    assert stream.close_count == 1


@pytest.mark.asyncio
async def test_gateway_credential_probe_classifies_rejection_before_reading_body(
    monkeypatch,
):
    import httpx

    from tldw_Server_API.app.core.AuthNZ import byok_testing
    from tldw_Server_API.app.core.http_client import afetch_json, create_async_client

    class _CountingStream(httpx.AsyncByteStream):
        def __init__(self) -> None:
            self.yielded = 0
            self.close_count = 0

        async def __aiter__(self):
            self.yielded += 1
            yield b'{"error":"sensitive upstream body"}'

        async def aclose(self) -> None:
            self.close_count += 1

    stream = _CountingStream()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            401,
            request=request,
            headers={"Content-Type": "application/json"},
            stream=stream,
        )

    client = create_async_client(transport=httpx.MockTransport(handler))

    async def _central_fetch(**kwargs):
        return await afetch_json(client=client, **kwargs)

    monkeypatch.setattr(byok_testing, "afetch_json", _central_fetch)
    try:
        result = await byok_testing.probe_gateway_credentials(
            spec=_probe_spec(),
            api_key="rejected-key",
        )
    finally:
        await client.aclose()

    assert result == "rejected"
    assert stream.yielded == 0
    assert stream.close_count == 1
    assert (
        await byok_testing.probe_gateway_credentials(
            spec=_probe_spec(discovery_enabled=False),
            api_key="unverified-key",
        )
        == "stored-unverified"
    )
