from tldw_Server_API.app.api.v1.schemas.setup_schemas import (
    SetupProviderSaveResponse,
    SetupProviderSaveStatus,
    SetupProviderType,
)
from tldw_Server_API.app.core.Setup.provider_catalog import (
    REQUIRED_SETUP_PROVIDER_KEYS,
    get_setup_provider_catalog,
    mask_secret,
)


def test_catalog_covers_required_prd_provider_keys():
    catalog = get_setup_provider_catalog()
    keys = {provider.provider_key for provider in catalog.providers}

    assert set(REQUIRED_SETUP_PROVIDER_KEYS) <= keys


def test_catalog_marks_local_providers_as_endpoint_based():
    catalog = get_setup_provider_catalog()
    providers = {provider.provider_key: provider for provider in catalog.providers}

    assert providers["ollama"].provider_type is SetupProviderType.LOCAL_ENDPOINT
    assert providers["llamacpp"].provider_type is SetupProviderType.LOCAL_ENDPOINT
    assert providers["custom_openai"].provider_type is SetupProviderType.LOCAL_ENDPOINT


def test_catalog_defaults_align_with_runtime_adapter_defaults():
    catalog = get_setup_provider_catalog()
    providers = {provider.provider_key: provider for provider in catalog.providers}

    assert providers["moonshot"].base_url_field == "moonshot_api_base_url"
    assert providers["moonshot"].default_base_url == "https://api.moonshot.cn/v1"
    assert providers["zai"].base_url_field == "zai_api_base_url"
    assert providers["zai"].default_base_url == "https://api.z.ai/api/paas/v4"


def test_mask_secret_never_returns_raw_value():
    assert mask_secret("sk-abcdefghijklmnopqrstuvwxyz") == "sk-...wxyz"
    assert mask_secret("tiny") == "****ny"
    assert mask_secret("") == ""


def test_mask_secret_never_returns_full_short_value():
    assert mask_secret("a") == "****"
    assert mask_secret("ab") != "****ab"
    assert "ab" not in mask_secret("ab")
    assert mask_secret("tiny") == "****ny"


def test_provider_save_response_contract_masks_secret_and_uses_saved_status():
    raw_key = "sk-abcdefghijklmnopqrstuvwxyz"

    response = SetupProviderSaveResponse(
        provider_key="openai",
        status=SetupProviderSaveStatus.SAVED,
        masked_api_key=mask_secret(raw_key),
        make_default=True,
    )

    assert response.provider_key == "openai"
    assert response.status is SetupProviderSaveStatus.SAVED
    assert response.masked_api_key == "sk-...wxyz"
    assert raw_key not in response.model_dump_json()
