"""Provider setup changes must reach strict Chat validation without a restart."""

from types import SimpleNamespace

import pytest

from tldw_Server_API.app.api.v1.endpoints import setup as setup_endpoint
from tldw_Server_API.app.api.v1.schemas.setup_schemas import SetupProviderSaveRequest
from tldw_Server_API.app.core import config
from tldw_Server_API.app.core.Chat import chat_service
from tldw_Server_API.app.core.custom_openai_providers import custom_openai_model_env_keys
from tldw_Server_API.app.core.Setup import setup_manager


@pytest.fixture
def isolated_model_config(tmp_path, monkeypatch):
    """Use actual config writes/loaders, isolated from the developer's settings."""
    path = tmp_path / "config.txt"
    path.write_text(
        "[API]\ncustom_openai_api_model = startup-custom\n"
        "custom_openai2_api_model = startup-second\n"
        "[Local-API]\nollama_model = startup-local\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(config, "resolve_config_file", lambda: path)
    monkeypatch.setattr(config, "_candidate_env_paths", lambda *_args: [])
    monkeypatch.setattr(setup_manager, "get_config_file_path", lambda: path)
    # Catalog transport is unrelated; retain a catalog-only positive control.
    monkeypatch.setattr(chat_service, "list_provider_models", lambda _provider: ["catalog-model"])
    for number in (1, 2):
        for name in custom_openai_model_env_keys(number):
            monkeypatch.delenv(name, raising=False)
    config.clear_config_cache()
    chat_service.invalidate_model_alias_caches()
    # Reproduce a worker that imported Chat before first-run setup saved a model.
    monkeypatch.setattr(chat_service, "_config", config.load_comprehensive_config())
    yield path
    config.clear_config_cache()
    chat_service.invalidate_model_alias_caches()


@pytest.mark.asyncio
@pytest.mark.parametrize("warm_inventory", [False, True])
@pytest.mark.parametrize(
    ("setup_provider", "chat_provider", "startup_model"),
    [
        ("custom_openai", "custom-openai-api", "startup-custom"),
        ("ollama", "ollama", "startup-local"),
    ],
)
async def test_setup_save_refreshes_cold_and_warm_chat_inventory_without_restart(
    isolated_model_config, warm_inventory, setup_provider, chat_provider, startup_model
):
    if warm_inventory:
        assert chat_service.is_model_known_for_provider(chat_provider, startup_model) is True

    for model in ("first-saved-model", "second-saved-model"):
        result = await setup_endpoint.save_first_run_provider(
            SetupProviderSaveRequest(provider_key=setup_provider, model=model), _guard=None
        )
        assert result.status.value == "saved"
        assert result.requires_restart is False
        request = SimpleNamespace(api_provider=chat_provider, model=model)
        provider = chat_service.normalize_request_provider_and_model(request, default_provider=chat_provider)
        assert provider == chat_provider
        assert chat_service.is_model_known_for_provider(provider, request.model) is True
        assert chat_service.is_model_known_for_provider(provider, startup_model) is False
        assert chat_service.is_model_known_for_provider(provider, "unrelated-model") is False
        assert chat_service.is_model_known_for_provider(provider, "catalog-model") is True


def test_numbered_custom_provider_refresh_remains_scoped(isolated_model_config):
    assert chat_service.is_model_known_for_provider("custom-openai-api-2", "startup-second") is True
    setup_manager.update_config({"API": {"custom_openai2_api_model": "new-second"}})
    config.clear_config_cache()

    assert chat_service.is_model_known_for_provider("custom-openai-api-2", "new-second") is True
    assert chat_service.is_model_known_for_provider("custom-openai-api", "new-second") is False
    assert chat_service.is_model_known_for_provider("custom-openai-api-2", "startup-second") is False


@pytest.mark.asyncio
async def test_setup_save_keeps_effective_environment_model_precedence(isolated_model_config, monkeypatch):
    monkeypatch.setenv("CUSTOM_OPENAI_API_MODEL", "environment-model")
    result = await setup_endpoint.save_first_run_provider(
        SetupProviderSaveRequest(provider_key="custom_openai", model="saved-file-model"),
        _guard=None,
    )

    assert result.status.value == "saved"
    assert chat_service.is_model_known_for_provider("custom-openai-api", "environment-model") is True
    assert chat_service.is_model_known_for_provider("custom-openai-api", "saved-file-model") is False
