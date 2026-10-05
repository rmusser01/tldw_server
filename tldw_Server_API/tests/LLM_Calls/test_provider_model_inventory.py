"""Cloud discovery contracts exercised without provider requests or credentials."""

from __future__ import annotations

import importlib
from typing import Any
from urllib.parse import parse_qs, urlsplit

import pytest

from tldw_Server_API.app.core.LLM_Calls.provider_readiness import ModelDiscoveryResult

pytestmark = pytest.mark.unit


class Response:
    def __init__(self, payload: Any = None, status: int = 200):
        self.payload = payload
        self.status_code = status
        self.closed = False

    def json(self):
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload

    def close(self):
        self.closed = True


class Fetcher:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if not self.responses:
            raise AssertionError("Unexpected discovery request")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


@pytest.fixture
def inventory(monkeypatch):
    module = importlib.import_module("tldw_Server_API.app.core.LLM_Calls.provider_model_inventory")
    adapter_utils = importlib.import_module("tldw_Server_API.app.core.LLM_Calls.adapter_utils")
    # Chat fixture collection injects a localhost OpenAI endpoint. Defaults here
    # must exercise adapter resolution without process-global config or mock URLs.
    for name in (
        "OPENAI_API_BASE_URL",
        "OPENAI_API_BASE",
        "OPENAI_BASE_URL",
        "MOCK_OPENAI_BASE_URL",
        "ANTHROPIC_BASE_URL",
        "DEEPSEEK_BASE_URL",
        "GOOGLE_GEMINI_BASE_URL",
        "GROQ_BASE_URL",
        "MISTRAL_API_BASE",
        "OPENROUTER_BASE_URL",
        "QWEN_BASE_URL",
        "QWEN_REGION",
        "NOVITA_BASE_URL",
        "NOVITA_API_BASE_URL",
        "POE_BASE_URL",
        "POE_API_BASE_URL",
        "TOGETHER_BASE_URL",
        "TOGETHER_API_BASE_URL",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(adapter_utils, "ensure_app_config", lambda config=None: {} if config is None else config)
    module._CACHE.clear()

    def no_network(**kwargs):
        raise AssertionError("Unit discovery must not make a real request")

    monkeypatch.setattr(module, "_http_fetch", no_network)
    yield module
    module._CACHE.clear()


@pytest.mark.parametrize(
    ("provider", "expected"),
    [
        ("openai", "https://api.openai.com/v1"),
        ("anthropic", "https://api.anthropic.com/v1"),
        ("deepseek", "https://api.deepseek.com"),
        ("google", "https://generativelanguage.googleapis.com/v1beta"),
        ("groq", "https://api.groq.com/openai/v1"),
        ("mistral", "https://api.mistral.ai/v1"),
        ("openrouter", "https://openrouter.ai/api/v1"),
        ("novita", "https://api.novita.ai/openai"),
        ("poe", "https://api.poe.com/v1"),
        ("together", "https://api.together.xyz/v1"),
        ("cohere", "https://api.cohere.ai"),
        ("moonshot", "https://api.moonshot.cn/v1"),
    ],
)
def test_endpoint_resolution_is_independent_of_generation_registry(inventory, monkeypatch, provider, expected):
    registry_module = importlib.import_module("tldw_Server_API.app.core.LLM_Calls.adapter_registry")

    def unavailable_registry():
        raise RuntimeError("Generation registry disabled without SDK or startup credentials")

    monkeypatch.setattr(registry_module, "get_registry", unavailable_registry)
    assert inventory.resolve_provider_models_base_url(provider, {}, credentials_resolved=True) == expected


def test_admin_key_discovers_openai_when_generation_registry_is_unavailable(inventory, monkeypatch):
    registry_module = importlib.import_module("tldw_Server_API.app.core.LLM_Calls.adapter_registry")
    monkeypatch.setattr(registry_module, "get_registry", lambda: None)
    response = Response({"data": [{"id": "current-model"}]})
    fetcher = Fetcher(response)
    assert inventory.discover_provider_models(
        "openai", "synthetic-admin-key", fetch_fn=fetcher
    ) == ModelDiscoveryResult("ready", ("current-model",))
    assert fetcher.calls[0]["url"] == "https://api.openai.com/v1/models"
    assert response.closed


def test_cloud_names_cover_all_commercial_catalog_providers(inventory):
    expected = frozenset(
        {
            "openai",
            "anthropic",
            "bedrock",
            "cohere",
            "deepseek",
            "google",
            "groq",
            "huggingface",
            "mistral",
            "openrouter",
            "qwen",
            "moonshot",
            "zai",
            "novita",
            "poe",
            "together",
            "minimax",
        }
    )
    assert expected == inventory.CLOUD_MODEL_PROVIDERS


@pytest.mark.parametrize(
    ("provider", "section", "env_key", "config_field", "env_wins"),
    [
        ("openai", "openai_api", "OPENAI_BASE_URL", "api_base_url", False),
        ("anthropic", "anthropic_api", "ANTHROPIC_BASE_URL", "api_base_url", False),
        ("deepseek", "deepseek_api", "DEEPSEEK_BASE_URL", "api_base_url", True),
        ("google", "google_api", "GOOGLE_GEMINI_BASE_URL", "api_base_url", False),
        ("groq", "groq_api", "GROQ_BASE_URL", "api_base_url", False),
        ("mistral", "mistral_api", "MISTRAL_API_BASE", "api_base_url", False),
        ("openrouter", "openrouter_api", "OPENROUTER_BASE_URL", "api_base_url", False),
        ("novita", "novita_api", "NOVITA_BASE_URL", "api_ip", False),
        ("poe", "poe_api", "POE_BASE_URL", "api_base_url", False),
        ("together", "together_api", "TOGETHER_BASE_URL", "api_base_url", False),
        ("qwen", "qwen_api", "QWEN_BASE_URL", "api_base_url", True),
    ],
)
def test_base_resolver_matches_adapter_config_and_environment_precedence(
    inventory,
    monkeypatch,
    provider,
    section,
    env_key,
    config_field,
    env_wins,
):
    monkeypatch.setenv(env_key, "https://environment.example/v1")
    config = {section: {config_field: "https://configured.example/v1"}}
    resolver = inventory.resolve_provider_models_base_url
    assert resolver(provider, config) == (
        "https://environment.example/v1" if env_wins else "https://configured.example/v1"
    )
    assert resolver(provider, config, credentials_resolved=True) == "https://configured.example/v1"
    if provider == "qwen":
        assert (
            resolver(provider, {section: {"region": "us"}}, credentials_resolved=True)
            == "https://dashscope-us.aliyuncs.com/compatible-mode/v1"
        )


@pytest.mark.parametrize(
    ("provider", "env_key", "default"),
    [
        ("openai", "OPENAI_BASE_URL", "https://api.openai.com/v1"),
        ("anthropic", "ANTHROPIC_BASE_URL", "https://api.anthropic.com/v1"),
        ("deepseek", "DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        ("google", "GOOGLE_GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta"),
        ("groq", "GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
        ("mistral", "MISTRAL_API_BASE", "https://api.mistral.ai/v1"),
        ("openrouter", "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        ("poe", "POE_BASE_URL", "https://api.poe.com/v1"),
        ("together", "TOGETHER_BASE_URL", "https://api.together.xyz/v1"),
    ],
)
def test_resolved_credentials_ignore_endpoint_environment(inventory, monkeypatch, provider, env_key, default):
    monkeypatch.setenv(env_key, "https://environment.example/v1")
    assert inventory.resolve_provider_models_base_url(provider, {}, credentials_resolved=True) == default


@pytest.mark.parametrize("provider", ["cohere", "moonshot", "zai"])
def test_function_adapters_preserve_configured_base_without_guessing_api_url(inventory, provider):
    assert (
        inventory.resolve_provider_models_base_url(
            provider,
            {f"{provider}_api": {"api_base_url": "https://configured.example/base"}},
            credentials_resolved=True,
        )
        == "https://configured.example/base"
    )


def test_resolver_rejects_unsafe_config_without_default_or_environment_fallback(inventory):
    assert (
        inventory.resolve_provider_models_base_url(
            "openai", {"openai_api": {"api_base_url": "https://user:secret@example.com/v1"}}, credentials_resolved=True
        )
        is None
    )


@pytest.mark.parametrize(
    ("provider", "endpoint"),
    [
        ("openai", "https://api.openai.com/v1/models"),
        ("deepseek", "https://api.deepseek.com/models"),
        ("groq", "https://api.groq.com/openai/v1/models"),
        ("mistral", "https://api.mistral.ai/v1/models"),
        ("openrouter", "https://openrouter.ai/api/v1/models/user"),
        ("poe", "https://api.poe.com/v1/models"),
        ("novita", "https://api.novita.ai/openai/v1/models"),
    ],
)
def test_compatible_discovery_preserves_only_exact_ids(inventory, provider, endpoint):
    response = Response(
        {
            "data": [
                {"id": "Vendor/Exact-ID", "name": "Not-an-ID", "display_name": "Pretty Name", "aliases": ["alias"]},
                {"id": "Vendor/Exact-ID"},
                {"id": "vendor/exact-id"},
            ]
        }
    )
    fetcher = Fetcher(response)
    result = inventory.discover_provider_models(provider, "synthetic-key", fetch_fn=fetcher)
    assert result == ModelDiscoveryResult("ready", ("Vendor/Exact-ID", "vendor/exact-id"))
    assert fetcher.calls[0]["url"] == endpoint
    assert fetcher.calls[0]["headers"]["Authorization"] == "Bearer synthetic-key"
    assert response.closed


def test_together_uses_authoritative_array_and_adapter_default(inventory):
    fetcher = Fetcher(Response([{"id": "org/model", "display_name": "Do not select this"}]))
    assert inventory.discover_provider_models("together", "synthetic-key", fetch_fn=fetcher) == ModelDiscoveryResult(
        "ready", ("org/model",)
    )
    assert fetcher.calls[0]["url"] == "https://api.together.xyz/v1/models"


def test_anthropic_auth_and_pagination_use_ids_not_display_names(inventory):
    pages = [
        Response(
            {
                "data": [{"id": "claude-exact-1", "display_name": "Claude Latest"}],
                "has_more": True,
                "last_id": "claude-exact-1",
            }
        ),
        Response({"data": [{"id": "claude-exact-2"}], "has_more": False, "last_id": "claude-exact-2"}),
    ]
    fetcher = Fetcher(*pages)
    assert inventory.discover_provider_models("anthropic", "synthetic-key", fetch_fn=fetcher) == ModelDiscoveryResult(
        "ready", ("claude-exact-1", "claude-exact-2")
    )
    assert urlsplit(fetcher.calls[0]["url"]).path == "/v1/models"
    assert parse_qs(urlsplit(fetcher.calls[1]["url"]).query)["after_id"] == ["claude-exact-1"]
    headers = fetcher.calls[0]["headers"]
    assert headers["x-api-key"] == "synthetic-key"
    assert headers["anthropic-version"] == "2023-06-01"
    assert "Authorization" not in headers
    assert all(page.closed for page in pages)


def test_google_filters_generation_and_strips_only_resource_prefix(inventory):
    fetcher = Fetcher(
        Response(
            {
                "models": [
                    {
                        "name": "models/gemini-exact",
                        "displayName": "Gemini Alias",
                        "baseModelId": "different-id",
                        "supportedGenerationMethods": ["generateContent"],
                    },
                    {"name": "models/embed-exact", "supportedGenerationMethods": ["embedContent"]},
                ],
                "nextPageToken": "opaque +/=",
            }
        ),
        Response(
            {
                "models": [
                    {"name": "models/ExactCase", "supportedGenerationMethods": ["generateContent"]},
                ]
            }
        ),
    )
    assert inventory.discover_provider_models("google", "synthetic-key", fetch_fn=fetcher) == ModelDiscoveryResult(
        "ready", ("gemini-exact", "ExactCase")
    )
    assert urlsplit(fetcher.calls[0]["url"]).path == "/v1beta/models"
    assert parse_qs(urlsplit(fetcher.calls[1]["url"]).query)["pageToken"] == ["opaque +/="]
    assert fetcher.calls[0]["headers"]["x-goog-api-key"] == "synthetic-key"
    assert all("synthetic-key" not in call["url"] for call in fetcher.calls)


def test_cohere_requires_chat_and_excludes_deprecated_models(inventory):
    fetcher = Fetcher(
        Response(
            {
                "models": [
                    {"name": "command-exact", "endpoints": ["chat"], "is_deprecated": False},
                    {"name": "command-retired", "endpoints": ["chat"], "is_deprecated": True},
                    {"name": "embed-only", "endpoints": ["embed"]},
                    {"name": "missing-chat"},
                ],
                "next_page_token": "next",
            }
        ),
        Response({"models": [{"name": "command-second", "endpoints": ["chat"]}]}),
    )
    assert inventory.discover_provider_models("cohere", "synthetic-key", fetch_fn=fetcher) == ModelDiscoveryResult(
        "ready", ("command-exact", "command-second")
    )
    assert urlsplit(fetcher.calls[0]["url"]).path == "/v1/models"
    assert parse_qs(urlsplit(fetcher.calls[1]["url"]).query)["page_token"] == ["next"]


@pytest.mark.parametrize(
    ("provider", "payload"),
    [
        ("deepseek", {"data": []}),
        ("anthropic", {"data": [], "has_more": False}),
        ("google", {"models": []}),
        ("cohere", {"models": []}),
        ("together", []),
    ],
)
def test_authoritative_empty_inventory_is_ready_and_cached(inventory, provider, payload):
    fetcher = Fetcher(Response(payload))
    assert inventory.discover_provider_models(provider, "synthetic-key", fetch_fn=fetcher) == ModelDiscoveryResult(
        "ready"
    )
    assert inventory.discover_provider_models(provider, "synthetic-key", fetch_fn=fetcher) == ModelDiscoveryResult(
        "ready"
    )
    assert len(fetcher.calls) == 1


@pytest.mark.parametrize("provider", ["bedrock", "zai", "minimax", "unknown", "ollama"])
def test_unverified_or_local_providers_are_unsupported_without_requests(inventory, provider):
    fetcher = Fetcher()
    assert inventory.discover_provider_models(provider, "synthetic-key", fetch_fn=fetcher) == ModelDiscoveryResult(
        "unsupported"
    )
    assert fetcher.calls == []


@pytest.mark.parametrize("key", [None, "", "   ", "key\r\ninjected: header"])
def test_missing_or_invalid_credentials_never_request(inventory, key):
    fetcher = Fetcher()
    assert inventory.discover_provider_models("deepseek", key, fetch_fn=fetcher) == ModelDiscoveryResult("auth_failed")
    assert fetcher.calls == []


@pytest.mark.parametrize(
    "base_url",
    [
        "https://user:secret@example.com/v1",
        "https://example.com/v1?key=secret",
        "https://example.com/v1#secret",
        "http://example.com/v1",
        "file:///etc/passwd",
        "https:///v1",
        "https://example.com:bad/v1",
        "https://example.com/\nsecret",
        "https://example.com\\@other.example/v1",
        "",
    ],
)
def test_unsafe_urls_are_rejected_without_fetch_or_secret_output(inventory, base_url, capsys):
    fetcher = Fetcher()
    assert inventory.discover_provider_models(
        "openai", "synthetic-key", base_url=base_url, fetch_fn=fetcher
    ) == ModelDiscoveryResult("unsupported")
    assert fetcher.calls == []
    assert capsys.readouterr() == ("", "")


@pytest.mark.parametrize(
    ("provider", "base_url", "endpoint"),
    [
        ("openai", "https://validated.example/custom/v1/", "https://validated.example/custom/v1/models"),
        ("deepseek", "https://validated.example/models", "https://validated.example/models"),
        ("cohere", "https://validated.example/v1", "https://validated.example/v1/models"),
        ("google", "https://validated.example/v1beta", "https://validated.example/v1beta/models"),
        ("anthropic", "https://validated.example", "https://validated.example/v1/models"),
    ],
)
def test_validated_base_url_is_honored_without_scope_widening(inventory, provider, base_url, endpoint):
    payload = {"data": [], "has_more": False} if provider in {"openai", "deepseek", "anthropic"} else {"models": []}
    fetcher = Fetcher(Response(payload))
    assert (
        inventory.discover_provider_models(provider, "synthetic-key", base_url=base_url, fetch_fn=fetcher).status
        == "ready"
    )
    call = fetcher.calls[0]
    assert urlsplit(call["url"])._replace(query="").geturl() == endpoint
    assert call["allow_redirects"] is False
    assert call["retry"].attempts == 1
    assert call.get("configured_endpoint") is None
    assert call["sensitive_observability"] is True
    assert 0 < call["timeout"] <= 5.0


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (401, "auth_failed"),
        (403, "auth_failed"),
        (498, "auth_failed"),
        (429, "server_error"),
        (500, "server_error"),
        (503, "server_error"),
        (404, "unsupported"),
        (302, "unsupported"),
        (307, "unsupported"),
    ],
)
def test_failures_never_fall_back_to_a_cached_inventory(inventory, status, expected):
    responses = [Response({"data": [{"id": "current"}]}), Response({"private_error": "synthetic-key"}, status)]
    fetcher = Fetcher(*responses, Response({"data": [{"id": "new"}]}))
    assert inventory.discover_provider_models("deepseek", "synthetic-key", fetch_fn=fetcher).models == ("current",)
    assert inventory.discover_provider_models(
        "deepseek", "synthetic-key", force_refresh=True, fetch_fn=fetcher
    ) == ModelDiscoveryResult(expected)
    assert inventory.discover_provider_models("deepseek", "synthetic-key", fetch_fn=fetcher).models == ("new",)
    assert all(response.closed for response in responses)


@pytest.mark.parametrize(
    "payload",
    [
        None,
        {},
        [],
        {"data": {}},
        {"data": ["alias"]},
        {"data": [{"display_name": "Pretty"}]},
        {"data": [{"id": " padded "}]},
        {"data": [{"id": "bad\nID"}]},
        ValueError("private secret synthetic-key"),
    ],
)
def test_malformed_inventory_is_not_empty_success_or_alias_fallback(inventory, payload):
    response = Response(payload)
    assert inventory.discover_provider_models(
        "openai", "synthetic-key", fetch_fn=Fetcher(response)
    ) == ModelDiscoveryResult("unsupported")
    assert response.closed


def test_fetch_exception_is_sanitized_and_not_cached(inventory, capsys):
    fetcher = Fetcher(TimeoutError("https://user:synthetic-key@example.com"), Response({"data": []}))
    assert inventory.discover_provider_models("deepseek", "synthetic-key", fetch_fn=fetcher) == ModelDiscoveryResult(
        "unreachable"
    )
    assert inventory.discover_provider_models("deepseek", "synthetic-key", fetch_fn=fetcher) == ModelDiscoveryResult(
        "ready"
    )
    assert capsys.readouterr() == ("", "")


def test_success_cache_scopes_provider_endpoint_and_credential_digest(inventory):
    fetcher = Fetcher(*(Response({"data": [{"id": name}]}) for name in ("first", "second", "third", "fourth")))
    assert inventory.discover_provider_models("openai", "credential-A", fetch_fn=fetcher).models == ("first",)
    assert inventory.discover_provider_models("openai", "credential-A", fetch_fn=fetcher).models == ("first",)
    assert inventory.discover_provider_models("openai", "credential-B", fetch_fn=fetcher).models == ("second",)
    assert inventory.discover_provider_models(
        "openai", "credential-A", base_url="https://validated.example/v1", fetch_fn=fetcher
    ).models == ("third",)
    assert inventory.discover_provider_models("deepseek", "credential-A", fetch_fn=fetcher).models == ("fourth",)
    assert "credential-A" not in repr(inventory._CACHE)
    assert "credential-B" not in repr(inventory._CACHE)


def test_ttl_is_300_seconds_and_expired_failure_cannot_reuse_old_cache(inventory, monkeypatch):
    now = [0.0]
    monkeypatch.setattr(inventory.time, "monotonic", lambda: now[0])
    fetcher = Fetcher(Response({"data": [{"id": "old"}]}), Response({}, 503), Response({"data": [{"id": "new"}]}))
    assert inventory.discover_provider_models("deepseek", "synthetic-key", fetch_fn=fetcher).models == ("old",)
    now[0] = 299.99
    assert inventory.discover_provider_models("deepseek", "synthetic-key", fetch_fn=fetcher).models == ("old",)
    now[0] = 300.0
    assert inventory.discover_provider_models("deepseek", "synthetic-key", fetch_fn=fetcher) == ModelDiscoveryResult(
        "server_error"
    )
    assert inventory.discover_provider_models("deepseek", "synthetic-key", fetch_fn=fetcher).models == ("new",)


def test_cache_is_bounded_to_128_entries_and_oldest_is_evicted(inventory):
    fetcher = Fetcher(*(Response({"data": []}) for _ in range(130)))
    for number in range(129):
        assert inventory.discover_provider_models("deepseek", f"synthetic-{number}", fetch_fn=fetcher).status == "ready"
    assert len(inventory._CACHE) == 128
    assert inventory.discover_provider_models("deepseek", "synthetic-0", fetch_fn=fetcher).status == "ready"
    assert len(fetcher.calls) == 130


@pytest.mark.parametrize(
    "payload",
    [
        {"data": [{"id": "a"}], "has_more": True},
        {"data": [{"id": "a"}], "has_more": "false", "last_id": "a"},
    ],
)
def test_anthropic_malformed_pagination_is_not_partial_success(inventory, payload):
    assert inventory.discover_provider_models(
        "anthropic", "synthetic-key", fetch_fn=Fetcher(Response(payload))
    ) == ModelDiscoveryResult("unsupported")


def test_repeated_cursor_and_second_page_failure_never_return_partial_models(inventory):
    page = {
        "models": [{"name": "models/a", "supportedGenerationMethods": ["generateContent"]}],
        "nextPageToken": "repeat",
    }
    fetcher = Fetcher(Response(page), Response(page))
    assert inventory.discover_provider_models("google", "synthetic-key", fetch_fn=fetcher) == ModelDiscoveryResult(
        "unsupported"
    )
    assert len(fetcher.calls) == 2
    fetcher = Fetcher(Response(page), Response({}, 401))
    assert inventory.discover_provider_models("google", "synthetic-key", fetch_fn=fetcher) == ModelDiscoveryResult(
        "auth_failed"
    )


def test_ten_page_limit_fails_without_publishing_partial_inventory(inventory):
    fetcher = Fetcher(
        *(Response({"data": [{"id": f"model-{i}"}], "has_more": True, "last_id": f"model-{i}"}) for i in range(10))
    )
    assert inventory.discover_provider_models("anthropic", "synthetic-key", fetch_fn=fetcher) == ModelDiscoveryResult(
        "unsupported"
    )
    assert len(fetcher.calls) == 10
    assert inventory._CACHE == {}


def test_model_limit_never_exposes_a_truncated_authoritative_inventory(inventory):
    fetcher = Fetcher(Response({"data": [{"id": f"model-{i}"} for i in range(1001)]}))
    assert inventory.discover_provider_models("openai", "synthetic-key", fetch_fn=fetcher) == ModelDiscoveryResult(
        "unsupported"
    )


def test_pagination_shares_one_five_second_budget(inventory, monkeypatch):
    now = [0.0]
    calls = []
    monkeypatch.setattr(inventory.time, "monotonic", lambda: now[0])

    def fetcher(**kwargs):
        calls.append(kwargs)
        now[0] += 1.5
        return Response({"data": [{"id": str(len(calls))}], "has_more": True, "last_id": str(len(calls))})

    assert inventory.discover_provider_models("anthropic", "synthetic-key", fetch_fn=fetcher) == ModelDiscoveryResult(
        "unreachable"
    )
    assert [call["timeout"] for call in calls] == [5.0, 3.5, 2.0, 0.5]
    assert [call["deadline"] for call in calls] == [5.0] * 4
    assert inventory._CACHE == {}


def test_default_fetch_keeps_central_egress_guard_without_configured_endpoint_scope(inventory, monkeypatch):
    from tldw_Server_API.app.core import http_client
    from tldw_Server_API.app.core.Security import egress

    calls = []

    def blocked(url, **kwargs):
        calls.append((url, kwargs))
        return egress.URLPolicyResult(False, "Test denial", reason_code="blocked")

    monkeypatch.setattr(egress, "evaluate_url_policy", blocked)
    monkeypatch.setattr(inventory, "_http_fetch", http_client.fetch)
    assert inventory.discover_provider_models(
        "deepseek", "synthetic-key", base_url="https://validated.example"
    ) == ModelDiscoveryResult("unreachable")
    assert calls
    assert all(kwargs.get("configured_endpoint") is None for _url, kwargs in calls)


def test_response_bytes_are_bounded(inventory):
    fetcher = Fetcher(Response({"data": []}))
    assert inventory.discover_provider_models("deepseek", "synthetic-key", fetch_fn=fetcher).status == "ready"
    assert fetcher.calls[0]["max_response_bytes"] == 2 * 1024 * 1024


def test_openrouter_uses_credential_filtered_inventory(inventory):
    fetcher = Fetcher(Response({"data": [{"id": "account/current-model"}]}))
    result = inventory.discover_provider_models(
        "openrouter", "synthetic-key", base_url="https://openrouter.ai/api/v1", fetch_fn=fetcher,
    )
    assert result.models == ("account/current-model",)
    assert fetcher.calls[0]["url"] == "https://openrouter.ai/api/v1/models/user"
    assert fetcher.calls[0]["headers"]["Authorization"] == "Bearer synthetic-key"


def test_older_inflight_success_cannot_recache_after_newer_failed_refresh(inventory):
    def older_fetch(**kwargs):
        failure = inventory.discover_provider_models(
            "deepseek",
            "synthetic-key",
            force_refresh=True,
            fetch_fn=Fetcher(Response({}, 401)),
        )
        assert failure.status == "auth_failed"
        return Response({"data": [{"id": "older"}]})

    assert inventory.discover_provider_models("deepseek", "synthetic-key", fetch_fn=older_fetch).status == "ready"
    assert inventory._CACHE == {}
    fresh = Fetcher(Response({"data": [{"id": "current"}]}))
    assert inventory.discover_provider_models("deepseek", "synthetic-key", fetch_fn=fresh).models == ("current",)
    assert len(fresh.calls) == 1


def test_refresh_on_one_scope_does_not_lock_network_for_other_scopes(inventory):
    def fetcher(**kwargs):
        other = inventory.discover_provider_models(
            "deepseek",
            "different-key",
            fetch_fn=Fetcher(Response({"data": []})),
        )
        assert other.status == "ready"
        return Response({"data": []})

    assert inventory.discover_provider_models("deepseek", "synthetic-key", fetch_fn=fetcher).status == "ready"


@pytest.mark.parametrize("base", ["https://trusted.example/custom/v1", "https://user:key@trusted.example/v1", ""])
def test_resolver_explicit_base_override_never_falls_back(inventory, base):
    expected = base if base == "https://trusted.example/custom/v1" else None
    assert (
        inventory.resolve_provider_models_base_url(
            "openai",
            {"openai_api": {"api_base_url": "https://fallback.example/v1"}},
            credentials_resolved=True,
            base_url=base,
        )
        == expected
    )


def test_one_attempt_policy_allows_scoped_discovery_and_caps_timeout(inventory):
    from tldw_Server_API.app.core.LLM_Calls.capability_registry import ProviderCallPolicy
    from tldw_Server_API.app.core.Security.egress import ConfiguredEndpointScope

    scope = ConfiguredEndpointScope.from_url("https://trusted.example/v1")
    policy = ProviderCallPolicy(max_transport_attempts=1, maximum_timeout_seconds=0.75, required_endpoint_scope=scope)
    fetcher = Fetcher(Response({"data": [{"id": "current"}]}))
    assert inventory.discover_provider_models(
        "openai",
        "synthetic-key",
        base_url="https://trusted.example/v1",
        call_policy=policy,
        fetch_fn=fetcher,
    ) == ModelDiscoveryResult("ready", ("current",))
    call = fetcher.calls[0]
    assert 0 < call["timeout"] <= 0.75
    assert call["configured_endpoint"] is scope
    assert call["retry"].attempts == 1


def test_policy_scope_mismatch_is_rejected_even_with_warm_cache(inventory):
    from tldw_Server_API.app.core.LLM_Calls.capability_registry import ProviderCallPolicy
    from tldw_Server_API.app.core.Security.egress import ConfiguredEndpointScope

    fetcher = Fetcher(Response({"data": [{"id": "current"}]}))
    base = "https://trusted.example/v1"
    assert (
        inventory.discover_provider_models("openai", "synthetic-key", base_url=base, fetch_fn=fetcher).status == "ready"
    )
    policy = ProviderCallPolicy(required_endpoint_scope=ConfiguredEndpointScope.from_url("https://other.example"))
    assert inventory.discover_provider_models(
        "openai",
        "synthetic-key",
        base_url=base,
        call_policy=policy,
        fetch_fn=fetcher,
    ) == ModelDiscoveryResult("unsupported")
    assert len(fetcher.calls) == 1


def test_one_attempt_policy_does_not_disable_readonly_pagination(inventory):
    from tldw_Server_API.app.core.LLM_Calls.capability_registry import ProviderCallPolicy

    fetcher = Fetcher(
        Response({"data": [{"id": "a"}], "has_more": True, "last_id": "a"}),
        Response({"data": [{"id": "b"}], "has_more": False}),
    )
    assert inventory.discover_provider_models(
        "anthropic",
        "synthetic-key",
        call_policy=ProviderCallPolicy(max_transport_attempts=1),
        fetch_fn=fetcher,
    ).models == ("a", "b")
    assert all(call["retry"].attempts == 1 for call in fetcher.calls)


@pytest.mark.parametrize(
    ("provider", "payload"),
    [
        ("together", [{"id": "chat", "type": "chat"}, {"id": "embedding", "type": "embedding"}]),
        (
            "mistral",
            {
                "data": [
                    {"id": "chat", "capabilities": {"completion_chat": True}},
                    {"id": "embedding", "capabilities": {"completion_chat": False}},
                ]
            },
        ),
        (
            "openrouter",
            {
                "data": [
                    {"id": "chat", "architecture": {"output_modalities": ["text"]}},
                    {"id": "image", "architecture": {"output_modalities": ["image"]}},
                ]
            },
        ),
    ],
)
def test_documented_capabilities_filter_nonchat_without_name_heuristics(inventory, provider, payload):
    assert inventory.discover_provider_models(
        provider, "synthetic-key", fetch_fn=Fetcher(Response(payload))
    ).models == ("chat",)


@pytest.mark.parametrize("base", ["https://api.moonshot.cn/v1", "https://api.moonshot.ai/v1"])
def test_moonshot_official_models_use_exact_ids_and_matching_region(inventory, base):
    fetcher = Fetcher(Response({"data": [{"id": "kimi-current", "display_name": "Never an ID"}]}))
    assert inventory.discover_provider_models(
        "moonshot",
        "synthetic-key",
        base_url=base,
        fetch_fn=fetcher,
    ) == ModelDiscoveryResult("ready", ("kimi-current",))
    assert fetcher.calls[0]["url"] == base + "/models"
    assert fetcher.calls[0]["headers"]["Authorization"] == "Bearer synthetic-key"


def test_huggingface_official_router_inventory_is_not_the_public_hub(inventory):
    base = "https://router.huggingface.co/v1"
    config = {"huggingface_api": {"api_base_url": base}}
    assert inventory.resolve_provider_models_base_url("huggingface", config, credentials_resolved=True) == base
    fetcher = Fetcher(Response({"data": [{"id": "Vendor/Callable", "name": "Alias"}]}))
    assert inventory.discover_provider_models(
        "huggingface",
        "synthetic-key",
        base_url=base,
        fetch_fn=fetcher,
    ) == ModelDiscoveryResult("ready", ("Vendor/Callable",))
    assert fetcher.calls[0]["url"] == base + "/models"
    assert fetcher.calls[0]["headers"]["Authorization"] == "Bearer synthetic-key"


@pytest.mark.parametrize(
    "base",
    [
        "https://huggingface.co/api",
        "https://api-inference.huggingface.co/v1",
        "https://router.huggingface.co/hf-inference",
        "https://router.huggingface.co:8443/v1",
        "https://router.huggingface.co/v1/../hf-inference",
    ],
)
def test_huggingface_nonrouter_inventory_never_dispatches(inventory, base):
    fetcher = Fetcher()
    assert inventory.discover_provider_models(
        "huggingface",
        "synthetic-key",
        base_url=base,
        fetch_fn=fetcher,
    ) == ModelDiscoveryResult("unsupported")
    assert fetcher.calls == []


def test_huggingface_model_specific_router_generation_is_not_global_router_inventory(inventory):
    config = {
        "huggingface_api": {
            "use_router_url_format": "true",
            "router_base_url": "https://router.huggingface.co/hf-inference",
        }
    }
    assert inventory.resolve_provider_models_base_url("huggingface", config, credentials_resolved=True) is None
    assert inventory.resolve_provider_models_base_url("huggingface", {}, credentials_resolved=True) is None


def qwen_page(rows, *, total, page=1, size=100):
    return {"success": True, "output": {"models": rows, "total": total, "page_no": page, "page_size": size}}


def test_qwen_documented_native_list_paginates_and_filters_reported_modalities(inventory):
    fetcher = Fetcher(
        Response(
            qwen_page(
                [
                    {
                        "model": "qwen-current",
                        "name": "Not an ID",
                        "inference_metadata": {"response_modality": ["Text"]},
                    },
                    {"model": "image-only", "inference_metadata": {"response_modality": ["Image"]}},
                ],
                total=3,
                size=2,
            )
        ),
        Response(qwen_page([{"model": "new-provider-id", "capabilities": ["TG"]}], total=3, page=2, size=2)),
    )
    assert inventory.discover_provider_models(
        "qwen",
        "synthetic-key",
        base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        fetch_fn=fetcher,
    ) == ModelDiscoveryResult("ready", ("qwen-current", "new-provider-id"))
    first = urlsplit(fetcher.calls[0]["url"])
    assert first.netloc == "dashscope-intl.aliyuncs.com"
    assert first.path == "/api/v1/models"
    assert parse_qs(first.query) == {"page_no": ["1"], "page_size": ["100"]}
    assert parse_qs(urlsplit(fetcher.calls[1]["url"]).query)["page_no"] == ["2"]


def test_qwen_authoritative_empty_is_cached_success(inventory):
    fetcher = Fetcher(Response(qwen_page([], total=0)))
    assert inventory.discover_provider_models("qwen", "synthetic-key", fetch_fn=fetcher) == ModelDiscoveryResult(
        "ready"
    )
    assert inventory.discover_provider_models("qwen", "synthetic-key", fetch_fn=fetcher) == ModelDiscoveryResult(
        "ready"
    )
    assert len(fetcher.calls) == 1


@pytest.mark.parametrize(
    "payload",
    [
        {"success": False, "output": {"models": [], "total": 0, "page_no": 1, "page_size": 100}},
        qwen_page([], total=1),
        qwen_page([], total=True),
        qwen_page([], total=1001),
        qwen_page([], total=0, page=2),
        qwen_page([], total=0, size=0),
        qwen_page([{"name": "display-only"}], total=1),
    ],
)
def test_qwen_malformed_or_incomplete_inventory_is_not_success(inventory, payload):
    assert inventory.discover_provider_models(
        "qwen", "synthetic-key", fetch_fn=Fetcher(Response(payload))
    ) == ModelDiscoveryResult("unsupported")


def test_qwen_wrong_second_page_never_publishes_partial_success(inventory):
    fetcher = Fetcher(
        Response(qwen_page([{"model": "a"}], total=2, size=1)),
        Response(qwen_page([{"model": "b"}], total=2, page=1, size=1)),
    )
    assert inventory.discover_provider_models("qwen", "synthetic-key", fetch_fn=fetcher) == ModelDiscoveryResult(
        "unsupported"
    )
    assert len(fetcher.calls) == 2
    assert inventory._CACHE == {}


def test_qwen_changed_total_never_publishes_a_mixed_inventory(inventory):
    fetcher = Fetcher(
        Response(qwen_page([{"model": "a"}], total=2, size=1)),
        Response(qwen_page([{"model": "b"}], total=3, page=2, size=1)),
    )
    assert inventory.discover_provider_models("qwen", "synthetic-key", fetch_fn=fetcher) == ModelDiscoveryResult(
        "unsupported"
    )
    assert len(fetcher.calls) == 2
    assert inventory._CACHE == {}


def test_qwen_unknown_base_path_never_guesses_an_inventory_route(inventory):
    fetcher = Fetcher()
    assert inventory.discover_provider_models(
        "qwen",
        "synthetic-key",
        base_url="https://validated.example/custom/v1",
        fetch_fn=fetcher,
    ) == ModelDiscoveryResult("unsupported")
    assert fetcher.calls == []
