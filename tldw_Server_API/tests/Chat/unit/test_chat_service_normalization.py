"""Cloud availability uses exact credential-scoped IDs; local behavior is retained."""

from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.Chat import chat_service
from tldw_Server_API.app.core.Chat.Chat_Deps import ChatConfigurationError

pytestmark = pytest.mark.unit


@pytest.fixture
def scoped_inventory(monkeypatch):
    result = SimpleNamespace(status="ready", models=())
    calls = []
    resolutions = []

    def resolve(provider, app_config=None, *, credentials_resolved=False, base_url=None):
        resolutions.append((provider, app_config, credentials_resolved, base_url))
        return "https://tenant.invalid/v1"

    def discover(provider, api_key, *, base_url):
        calls.append((provider, api_key, base_url))
        return result

    monkeypatch.setattr(chat_service, "resolve_provider_models_base_url", resolve)
    monkeypatch.setattr(chat_service, "discover_provider_models", discover)
    monkeypatch.setattr(chat_service, "list_provider_models", lambda _provider: ["retired-model"])
    monkeypatch.setattr(chat_service, "_configured_models_for_provider", lambda _provider: ("retired-model",))
    return SimpleNamespace(result=result, calls=calls, resolutions=resolutions)


def _scoped_availability(provider, model):
    return chat_service.is_model_known_for_provider(
        provider,
        model,
        api_key="fake-tenant-key",
        app_config={},
        credentials_resolved=True,
    )


def test_openrouter_preserves_namespaced_model():
    request = SimpleNamespace(api_provider="openrouter", model="z-ai/glm-4.6")
    assert chat_service.normalize_request_provider_and_model(request, "openrouter") == "openrouter"
    assert request.model == "z-ai/glm-4.6"


def test_openrouter_strips_redundant_openrouter_prefix():
    request = SimpleNamespace(api_provider="openrouter", model="openrouter/gpt-4o-mini")
    assert chat_service.normalize_request_provider_and_model(request, "openrouter") == "openrouter"
    assert request.model == "gpt-4o-mini"


def test_openrouter_does_not_have_implicit_pytest_dummy_alias(monkeypatch):
    monkeypatch.setattr(chat_service, "_load_alias_overrides_cached", lambda: {})
    request = SimpleNamespace(api_provider="openrouter", model="dummy")
    assert chat_service.normalize_request_provider_and_model(request, "openrouter") == "openrouter"
    assert request.model == "dummy"


@pytest.mark.parametrize(
    "provider, current_id, selected_id",
    [
        ("openrouter", "glm-4.6", "z-ai/glm-4.6"),
        ("openrouter", "z-ai/glm-4.6", "glm-4.6"),
        ("openrouter", "moonshotai/kimi-k2.5-0127", "moonshotai/kimi-k2.5"),
        ("openrouter", "moonshotai/kimi-k2.5", "moonshotai/kimi-k2.5-0127"),
        ("together", "Llama-3.3-70B-Instruct-Turbo", "meta-llama/Llama-3.3-70B-Instruct-Turbo"),
        ("openai", "current-model", "CURRENT-MODEL"),
        ("openrouter", "current-model", "retired-model"),
    ],
)
def test_scoped_cloud_availability_rejects_aliases_and_static_only_ids(
    scoped_inventory,
    provider,
    current_id,
    selected_id,
):
    scoped_inventory.result.models = (current_id,)
    assert _scoped_availability(provider, selected_id) is False
    assert scoped_inventory.calls == [(provider, "fake-tenant-key", "https://tenant.invalid/v1")]


@pytest.mark.parametrize(
    "provider, model",
    [
        ("openrouter", "z-ai/glm-4.6"),
        ("openrouter", "moonshotai/kimi-k2.5-0127"),
        ("together", "meta-llama/Llama-3.3-70B-Instruct-Turbo"),
        ("openai", "current-model"),
    ],
)
def test_scoped_cloud_availability_accepts_exact_current_inventory(scoped_inventory, provider, model):
    scoped_inventory.result.models = (model,)
    assert _scoped_availability(provider, model) is True
    assert scoped_inventory.calls == [(provider, "fake-tenant-key", "https://tenant.invalid/v1")]
    assert scoped_inventory.resolutions == [(provider, {}, True, None)]


def test_cloud_prevalidation_without_credential_context_defers(scoped_inventory):
    assert chat_service.is_model_known_for_provider("openrouter", "tenant/model") is None
    assert scoped_inventory.calls == []
    assert scoped_inventory.resolutions == []


def test_resolved_missing_key_does_not_discover_with_global_credentials(scoped_inventory):
    assert (
        chat_service.is_model_known_for_provider(
            "openrouter",
            "tenant/model",
            api_key=None,
            app_config={},
            credentials_resolved=True,
        )
        is False
    )
    assert scoped_inventory.calls == []
    assert scoped_inventory.resolutions == []


def test_scoped_failed_discovery_is_unavailable_not_static_fallback(scoped_inventory):
    scoped_inventory.result.status = "unreachable"
    with pytest.raises(ChatConfigurationError):
        _scoped_availability("openrouter", "retired-model")


def test_scoped_empty_inventory_rejects_every_selection(scoped_inventory):
    assert _scoped_availability("openrouter", "retired-model") is False


def test_together_preserves_namespaced_model():
    request = SimpleNamespace(api_provider="together", model="meta-llama/Llama-3.3-70B-Instruct-Turbo")
    assert chat_service.normalize_request_provider_and_model(request, "together") == "together"
    assert request.model == "meta-llama/Llama-3.3-70B-Instruct-Turbo"


@pytest.mark.parametrize("model", ["vendor/model", "anthropic/local-model", "../../models/local.gguf"])
def test_local_normalization_preserves_opaque_model_paths(monkeypatch, model):
    monkeypatch.setattr(chat_service, "_load_alias_overrides_cached", lambda: {})
    monkeypatch.setattr(chat_service, "_load_models_with_case_cached", lambda _provider: ())
    request = SimpleNamespace(api_provider="ollama", model=model)
    assert chat_service.normalize_request_provider_and_model(request, "ollama") == "ollama"
    assert request.model == model


def test_local_normalization_preserves_configured_aliases(monkeypatch):
    monkeypatch.setattr(
        chat_service,
        "_load_alias_overrides_cached",
        lambda: {"ollama": {"friendly": "local-model"}},
    )
    request = SimpleNamespace(api_provider="ollama", model="friendly")
    assert chat_service.normalize_request_provider_and_model(request, "ollama") == "ollama"
    assert request.model == "local-model"


def test_local_normalization_preserves_case_insensitive_catalog_match(monkeypatch):
    monkeypatch.setattr(chat_service, "_load_alias_overrides_cached", lambda: {})
    monkeypatch.setattr(chat_service, "_load_models_with_case_cached", lambda _provider: ("Local-Canonical",))
    request = SimpleNamespace(api_provider="ollama", model="local-canonical")
    assert chat_service.normalize_request_provider_and_model(request, "ollama") == "ollama"
    assert request.model == "Local-Canonical"


def test_local_availability_preserves_configured_and_catalog_models_without_cloud_discovery(
    scoped_inventory,
    monkeypatch,
):
    monkeypatch.setattr(chat_service, "list_provider_models", lambda _provider: ["Catalog-Local"])
    monkeypatch.setattr(chat_service, "_configured_models_for_provider", lambda _provider: ("Configured-Local",))
    assert chat_service.known_models_for_provider_cached("ollama") == ("Catalog-Local", "Configured-Local")
    assert chat_service.is_model_known_for_provider("ollama", "catalog-local") is True
    assert chat_service.is_model_known_for_provider("ollama", "configured-local") is True
    assert chat_service.is_model_known_for_provider("ollama", "unknown-local") is False
    assert scoped_inventory.calls == []
    assert scoped_inventory.resolutions == []


def test_local_availability_without_inventory_still_defers(scoped_inventory, monkeypatch):
    monkeypatch.setattr(chat_service, "list_provider_models", lambda _provider: [])
    monkeypatch.setattr(chat_service, "_configured_models_for_provider", lambda _provider: ())
    assert chat_service.is_model_known_for_provider("ollama", "local-model") is None
    assert scoped_inventory.calls == []
