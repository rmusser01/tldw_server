"""Backend-generated provider catalog for first-run setup."""

from __future__ import annotations

from tldw_Server_API.app.api.v1.schemas.setup_schemas import (
    SetupProviderCatalogEntry,
    SetupProviderCatalogResponse,
    SetupProviderType,
)

REQUIRED_SETUP_PROVIDER_KEYS = (
    "openai",
    "anthropic",
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
    "ollama",
    "llamacpp",
    "koboldcpp",
    "oobabooga",
    "tabbyapi",
    "vllm",
    "aphrodite",
    "custom_openai",
)


def mask_secret(value: str) -> str:
    """Return a non-reversible display hint for a secret value."""
    if not value:
        return ""
    if len(value) <= 2:
        return "****"
    if len(value) <= 8:
        return f"****{value[-2:]}"
    return f"{value[:3]}...{value[-4:]}"


_HOSTED_PROVIDERS: tuple[SetupProviderCatalogEntry, ...] = (
    SetupProviderCatalogEntry(
        provider_key="openai",
        label="OpenAI",
        provider_type=SetupProviderType.HOSTED_API_KEY,
        config_section="API",
        api_key_field="openai_api_key",
        model_field="openai_model",
        supports_preflight=True,
        recommended_for_first_chat=True,
    ),
    SetupProviderCatalogEntry(
        provider_key="anthropic",
        label="Anthropic",
        provider_type=SetupProviderType.HOSTED_API_KEY,
        config_section="API",
        api_key_field="anthropic_api_key",
        model_field="anthropic_model",
        supports_preflight=True,
        recommended_for_first_chat=True,
    ),
    SetupProviderCatalogEntry(
        provider_key="cohere",
        label="Cohere",
        provider_type=SetupProviderType.HOSTED_API_KEY,
        config_section="API",
        api_key_field="cohere_api_key",
        model_field="cohere_model",
        supports_preflight=True,
    ),
    SetupProviderCatalogEntry(
        provider_key="deepseek",
        label="DeepSeek",
        provider_type=SetupProviderType.HOSTED_API_KEY,
        config_section="API",
        api_key_field="deepseek_api_key",
        model_field="deepseek_model",
        supports_preflight=True,
    ),
    SetupProviderCatalogEntry(
        provider_key="google",
        label="Google",
        provider_type=SetupProviderType.HOSTED_API_KEY,
        config_section="API",
        api_key_field="google_api_key",
        model_field="google_model",
        supports_preflight=True,
    ),
    SetupProviderCatalogEntry(
        provider_key="groq",
        label="Groq",
        provider_type=SetupProviderType.HOSTED_API_KEY,
        config_section="API",
        api_key_field="groq_api_key",
        model_field="groq_model",
        supports_preflight=True,
    ),
    SetupProviderCatalogEntry(
        provider_key="huggingface",
        label="HuggingFace",
        provider_type=SetupProviderType.HOSTED_API_KEY,
        config_section="API",
        api_key_field="huggingface_api_key",
        base_url_field="huggingface_api_base_url",
        model_field="huggingface_model",
        default_base_url="https://router.huggingface.co/v1",
    ),
    SetupProviderCatalogEntry(
        provider_key="mistral",
        label="Mistral",
        provider_type=SetupProviderType.HOSTED_API_KEY,
        config_section="API",
        api_key_field="mistral_api_key",
        model_field="mistral_model",
        supports_preflight=True,
    ),
    SetupProviderCatalogEntry(
        provider_key="openrouter",
        label="OpenRouter",
        provider_type=SetupProviderType.HOSTED_API_KEY,
        config_section="API",
        api_key_field="openrouter_api_key",
        model_field="openrouter_model",
        default_base_url="https://openrouter.ai/api/v1",
        supports_preflight=True,
    ),
    SetupProviderCatalogEntry(
        provider_key="qwen",
        label="Qwen",
        provider_type=SetupProviderType.HOSTED_API_KEY,
        config_section="API",
        api_key_field="qwen_api_key",
        base_url_field="qwen_api_base_url",
        model_field="qwen_model",
        default_base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    ),
    SetupProviderCatalogEntry(
        provider_key="moonshot",
        label="Moonshot",
        provider_type=SetupProviderType.HOSTED_API_KEY,
        config_section="API",
        api_key_field="moonshot_api_key",
        base_url_field="moonshot_api_base_url",
        model_field="moonshot_model",
        default_base_url="https://api.moonshot.cn/v1",
    ),
    SetupProviderCatalogEntry(
        provider_key="zai",
        label="Z.AI",
        provider_type=SetupProviderType.HOSTED_API_KEY,
        config_section="API",
        api_key_field="zai_api_key",
        base_url_field="zai_api_base_url",
        model_field="zai_model",
        default_base_url="https://api.z.ai/api/paas/v4",
    ),
)

_LOCAL_PROVIDERS: tuple[SetupProviderCatalogEntry, ...] = (
    SetupProviderCatalogEntry(
        provider_key="ollama",
        label="Ollama",
        provider_type=SetupProviderType.LOCAL_ENDPOINT,
        config_section="Local-API",
        api_key_field="ollama_api_key",
        base_url_field="ollama_api_IP",
        model_field="ollama_model",
        default_base_url="http://127.0.0.1:11434/v1",
        supports_preflight=True,
        recommended_for_first_chat=True,
    ),
    SetupProviderCatalogEntry(
        provider_key="llamacpp",
        label="llama.cpp",
        provider_type=SetupProviderType.LOCAL_ENDPOINT,
        config_section="Local-API",
        api_key_field="llama_api_key",
        base_url_field="llama_api_IP",
        model_field="llama_model",
        default_base_url="http://127.0.0.1:8080/v1",
        supports_preflight=True,
    ),
    SetupProviderCatalogEntry(
        provider_key="koboldcpp",
        label="Kobold.cpp",
        provider_type=SetupProviderType.LOCAL_ENDPOINT,
        config_section="Local-API",
        api_key_field="kobold_api_key",
        base_url_field="kobold_api_IP",
        model_field="kobold_model",
        default_base_url="http://127.0.0.1:5001/api/v1/generate",
        supports_preflight=True,
    ),
    SetupProviderCatalogEntry(
        provider_key="oobabooga",
        label="Oobabooga",
        provider_type=SetupProviderType.LOCAL_ENDPOINT,
        config_section="Local-API",
        api_key_field="ooba_api_key",
        base_url_field="ooba_api_IP",
        model_field="ooba_model",
        default_base_url="http://127.0.0.1:5000/v1",
        supports_preflight=True,
    ),
    SetupProviderCatalogEntry(
        provider_key="tabbyapi",
        label="TabbyAPI",
        provider_type=SetupProviderType.LOCAL_ENDPOINT,
        config_section="Local-API",
        api_key_field="tabby_api_key",
        base_url_field="tabby_api_IP",
        model_field="tabby_model",
        default_base_url="http://127.0.0.1:5000/v1",
        supports_preflight=True,
    ),
    SetupProviderCatalogEntry(
        provider_key="vllm",
        label="vLLM",
        provider_type=SetupProviderType.LOCAL_ENDPOINT,
        config_section="Local-API",
        api_key_field="vllm_api_key",
        base_url_field="vllm_api_IP",
        model_field="vllm_model",
        default_base_url="http://127.0.0.1:8000/v1",
        supports_preflight=True,
    ),
    SetupProviderCatalogEntry(
        provider_key="aphrodite",
        label="Aphrodite",
        provider_type=SetupProviderType.LOCAL_ENDPOINT,
        config_section="Local-API",
        api_key_field="aphrodite_api_key",
        base_url_field="aphrodite_api_IP",
        model_field="aphrodite_model",
        default_base_url="http://127.0.0.1:8080/v1",
        supports_preflight=True,
    ),
    SetupProviderCatalogEntry(
        provider_key="custom_openai",
        label="Custom OpenAI-compatible",
        provider_type=SetupProviderType.LOCAL_ENDPOINT,
        config_section="API",
        api_key_field="custom_openai_api_key",
        base_url_field="custom_openai_api_ip",
        model_field="custom_openai_api_model",
        default_base_url="http://127.0.0.1:8000/v1",
        supports_preflight=True,
    ),
)

_SETUP_PROVIDER_CATALOG = (*_HOSTED_PROVIDERS, *_LOCAL_PROVIDERS)
_CATALOG_BY_KEY = {entry.provider_key: entry for entry in _SETUP_PROVIDER_CATALOG}


def get_setup_provider_catalog() -> SetupProviderCatalogResponse:
    """Return the deterministic first-run provider catalog."""
    return SetupProviderCatalogResponse(providers=list(_SETUP_PROVIDER_CATALOG))


def get_setup_provider_entry(provider_key: str) -> SetupProviderCatalogEntry | None:
    """Return a catalog entry by setup provider key."""
    return _CATALOG_BY_KEY.get(provider_key.strip().lower())
