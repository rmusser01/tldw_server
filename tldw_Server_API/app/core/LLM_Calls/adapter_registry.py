from __future__ import annotations

"""
Chat provider adapter registry (LLM).

Mirrors the TTS adapter pattern with a lightweight registry that:
- Lazily resolves adapters from dotted paths or classes
- Caches initialized adapters
- Exposes capability discovery for endpoints/clients

Initial version ships without default adapters; providers can be registered
by initialization code or tests. Future phases may add defaults.
"""

from typing import Any

from loguru import logger

from tldw_Server_API.app.core.custom_openai_providers import (
    custom_openai_aliases,
    custom_openai_provider_name,
    iter_custom_openai_provider_numbers,
)
from tldw_Server_API.app.core.Infrastructure.provider_registry import ProviderRegistryBase

from .providers.base import ChatProvider
from .providers.custom_openai_adapter import make_custom_openai_adapter_class


class ChatProviderRegistry:
    """Registry for Chat (LLM) providers and their adapters."""

    # Default adapter mappings (lazy via dotted paths)
    DEFAULT_ADAPTERS: dict[str, Any] = {
        "openai": "tldw_Server_API.app.core.LLM_Calls.providers.openai_adapter.OpenAIAdapter",
        "anthropic": "tldw_Server_API.app.core.LLM_Calls.providers.anthropic_adapter.AnthropicAdapter",
        "groq": "tldw_Server_API.app.core.LLM_Calls.providers.groq_adapter.GroqAdapter",
        "openrouter": "tldw_Server_API.app.core.LLM_Calls.providers.openrouter_adapter.OpenRouterAdapter",
        "google": "tldw_Server_API.app.core.LLM_Calls.providers.google_adapter.GoogleAdapter",
        "mistral": "tldw_Server_API.app.core.LLM_Calls.providers.mistral_adapter.MistralAdapter",
        "qwen": "tldw_Server_API.app.core.LLM_Calls.providers.qwen_adapter.QwenAdapter",
        "deepseek": "tldw_Server_API.app.core.LLM_Calls.providers.deepseek_adapter.DeepSeekAdapter",
        "huggingface": "tldw_Server_API.app.core.LLM_Calls.providers.huggingface_adapter.HuggingFaceAdapter",
        "bedrock": "tldw_Server_API.app.core.LLM_Calls.providers.bedrock_adapter.BedrockAdapter",
        "custom-openai-api": "tldw_Server_API.app.core.LLM_Calls.providers.custom_openai_adapter.CustomOpenAIAdapter",
        "custom-openai-api-2": "tldw_Server_API.app.core.LLM_Calls.providers.custom_openai_adapter.CustomOpenAIAdapter2",
        "novita": "tldw_Server_API.app.core.LLM_Calls.providers.custom_openai_adapter.NovitaAdapter",
        "poe": "tldw_Server_API.app.core.LLM_Calls.providers.custom_openai_adapter.PoeAdapter",
        "together": "tldw_Server_API.app.core.LLM_Calls.providers.custom_openai_adapter.TogetherAdapter",
        "mlx": "tldw_Server_API.app.core.LLM_Calls.providers.mlx_provider.MLXChatAdapter",
        "cohere": "tldw_Server_API.app.core.LLM_Calls.providers.cohere_adapter.CohereAdapter",
        "moonshot": "tldw_Server_API.app.core.LLM_Calls.providers.moonshot_adapter.MoonshotAdapter",
        "zai": "tldw_Server_API.app.core.LLM_Calls.providers.zai_adapter.ZaiAdapter",
        "llama.cpp": "tldw_Server_API.app.core.LLM_Calls.providers.local_adapters.LlamaCppAdapter",
        "kobold": "tldw_Server_API.app.core.LLM_Calls.providers.local_adapters.KoboldAdapter",
        "ooba": "tldw_Server_API.app.core.LLM_Calls.providers.local_adapters.OobaAdapter",
        "tabbyapi": "tldw_Server_API.app.core.LLM_Calls.providers.local_adapters.TabbyAPIAdapter",
        "vllm": "tldw_Server_API.app.core.LLM_Calls.providers.local_adapters.VLLMAdapter",
        "local-llm": "tldw_Server_API.app.core.LLM_Calls.providers.local_adapters.LocalLLMAdapter",
        "ollama": "tldw_Server_API.app.core.LLM_Calls.providers.local_adapters.OllamaAdapter",
        "aphrodite": "tldw_Server_API.app.core.LLM_Calls.providers.local_adapters.AphroditeAdapter",
    }
    DEFAULT_ADAPTERS.update(
        {
            custom_openai_provider_name(number): make_custom_openai_adapter_class(number)
            for number in iter_custom_openai_provider_numbers(start=3)
        }
    )

    DEFAULT_ALIASES: dict[str, tuple[str, ...]] = {
        "openai": ("oai",),
        "bedrock": ("aws-bedrock", "amazon-bedrock"),
        "custom-openai-api": (
            "custom_openai_api",
            "custom-openai",
            "openai-compatible",
            "customopenai",
        ),
        "custom-openai-api-2": (
            "custom_openai_api_2",
            "custom-openai-2",
            "openai-compatible-2",
            "customopenai2",
        ),
        "novita": ("novita-ai",),
        "poe": ("poe-api",),
        "together": ("together-ai", "togetherai"),
        "llama.cpp": ("llama-cpp", "llama_cpp", "llamacpp"),
        "kobold": ("kobold-cpp", "kobold_cpp", "koboldcpp"),
        "ooba": ("oobabooga", "text-generation-webui", "text_generation_webui"),
        "tabbyapi": ("tabby-api", "tabby_api", "tabby"),
        "local-llm": ("local_llm",),
        "zai": ("z-ai", "z.ai"),
    }
    DEFAULT_ALIASES.update(
        {
            custom_openai_provider_name(number): custom_openai_aliases(number)
            for number in iter_custom_openai_provider_numbers(start=3)
        }
    )

    @staticmethod
    def _parse_optional_bool(value: Any) -> bool | None:
        if value is None:
            return None
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        lowered = str(value).strip().lower()
        if lowered in {"1", "true", "yes", "on", "enabled"}:
            return True
        if lowered in {"0", "false", "no", "off", "disabled"}:
            return False
        return None

    def _is_provider_enabled_by_config(self, provider_name: str) -> bool | None:
        if not isinstance(self._config, dict):
            return None

        provider_key = str(provider_name or "").strip()
        if not provider_key:
            return None

        providers_cfg = self._config.get("providers")
        if isinstance(providers_cfg, dict):
            cfg_entry = providers_cfg.get(provider_key)
            if cfg_entry is None:
                cfg_entry = providers_cfg.get(provider_key.replace("-", "_"))
            if isinstance(cfg_entry, dict) and "enabled" in cfg_entry:
                return self._parse_optional_bool(cfg_entry.get("enabled"))

        enabled_keys = [
            f"{provider_key}_enabled",
            f"{provider_key.replace('-', '_')}_enabled",
        ]
        for enabled_key in enabled_keys:
            if enabled_key in self._config:
                return self._parse_optional_bool(self._config.get(enabled_key))
        return None

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        *,
        include_defaults: bool = True,
    ):
        # Keep config available for future adapter initialization needs
        self._config = config or {}
        self._adapter_specs: dict[str, Any] = (
            self.DEFAULT_ADAPTERS.copy() if include_defaults else {}
        )
        self._base: ProviderRegistryBase[ChatProvider] = ProviderRegistryBase(
            adapter_validator=lambda adapter: isinstance(adapter, ChatProvider),
            provider_enabled_callback=self._is_provider_enabled_by_config,
        )
        for provider_name, adapter_spec in self._adapter_specs.items():
            aliases = self.DEFAULT_ALIASES.get(provider_name)
            self._base.register_adapter(provider_name, adapter_spec, aliases=aliases)

    def register_adapter(
        self,
        name: str,
        adapter: Any,
        *,
        aliases: list[str] | tuple[str, ...] | set[str] | None = None,
    ) -> None:
        """Register an adapter class or dotted path for a provider name."""
        provider_key = self._base.normalize_provider_name(name)
        all_aliases = set(self.DEFAULT_ALIASES.get(provider_key, ()))
        if aliases:
            all_aliases.update(str(alias) for alias in aliases)
        self._base.register_adapter(name, adapter, aliases=sorted(all_aliases) if all_aliases else None)
        self._adapter_specs[provider_key] = adapter
        try:
            n = adapter.__name__  # type: ignore[attr-defined]
        except Exception:
            n = str(adapter)
        logger.info(f"Registered LLM adapter {n} for provider '{name}'")

    def get_adapter(self, name: str) -> ChatProvider | None:
        """Return an initialized adapter instance for a provider name, if any."""
        adapter = self._base.get_adapter(name)
        if adapter is None:
            provider_key = self._base.resolve_provider_name(name)
            if provider_key not in self._adapter_specs:
                logger.debug(f"No adapter spec registered for provider '{name}'")
            else:
                logger.debug(f"Adapter for provider '{name}' is currently unavailable")
            return None
        return adapter

    def get_all_capabilities(self) -> dict[str, dict[str, Any]]:
        """Return capabilities for all registered providers, initializing as needed."""
        out: dict[str, dict[str, Any]] = {}
        entries = self._base.list_capabilities(
            capability_getter=lambda adapter: adapter.capabilities() or {}
        )
        for entry in entries:
            provider_name = str(entry.get("provider") or "")
            capabilities = entry.get("capabilities")
            if provider_name and isinstance(capabilities, dict):
                out[provider_name] = capabilities
        return out

    def list_capabilities(self, *, include_disabled: bool = True) -> list[dict[str, Any]]:
        """Return standardized capability envelopes for registered providers."""
        return self._base.list_capabilities(
            capability_getter=lambda adapter: adapter.capabilities() or {},
            include_disabled=include_disabled,
        )

    def list_providers(self) -> list[str]:
        """Return a sorted list of registered provider names."""
        return self._base.list_providers(include_disabled=True)


_registry: ChatProviderRegistry | None = None


def get_registry() -> ChatProviderRegistry:
    global _registry
    if _registry is None:
        _registry = ChatProviderRegistry()
    return _registry
