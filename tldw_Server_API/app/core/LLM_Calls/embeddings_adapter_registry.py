from __future__ import annotations

"""
Embeddings provider adapter registry.

Lightweight registry to lazily construct embeddings adapters and expose
capability discovery for diagnostics.
"""

import importlib
from typing import Any

from loguru import logger

from .providers.base import EmbeddingsProvider


class EmbeddingsProviderRegistry:
    """Registry for embeddings providers and their adapters."""

    DEFAULT_ADAPTERS: dict[str, str] = {
        # Seed with OpenAI; extended with HF/Google
        "openai": "tldw_Server_API.app.core.LLM_Calls.providers.openai_embeddings_adapter.OpenAIEmbeddingsAdapter",
        "huggingface": "tldw_Server_API.app.core.LLM_Calls.providers.huggingface_embeddings_adapter.HuggingFaceEmbeddingsAdapter",
        "google": "tldw_Server_API.app.core.LLM_Calls.providers.google_embeddings_adapter.GoogleEmbeddingsAdapter",
        "mlx": "tldw_Server_API.app.core.LLM_Calls.providers.mlx_provider.MLXEmbeddingsAdapter",
    }

    def __init__(self, config: dict[str, Any] | None = None):
        self._config = config or {}
        self._adapters: dict[str, EmbeddingsProvider] = {}
        self._adapter_specs: dict[str, Any] = self.DEFAULT_ADAPTERS.copy()

    def register_adapter(self, name: str, adapter: Any) -> None:
        self._adapter_specs[name] = adapter
        try:
            n = adapter.__name__  # type: ignore[attr-defined]
        except Exception:
            n = str(adapter)
        logger.info(f"Registered Embeddings adapter {n} for provider '{name}'")

    def has_adapter(self, name: str) -> bool:
        """Return whether an adapter is registered without initializing it."""

        return name in self._adapter_specs

    def _resolve_adapter_class(self, spec: Any) -> type[EmbeddingsProvider]:
        if isinstance(spec, str):
            module_path, _, class_name = spec.rpartition(".")
            if not module_path:
                raise ImportError(f"Invalid adapter spec '{spec}'")
            module = importlib.import_module(module_path)
            cls = getattr(module, class_name)
            return cls
        return spec

    def get_adapter(self, name: str) -> EmbeddingsProvider | None:
        if name in self._adapters:
            return self._adapters[name]
        spec = self._adapter_specs.get(name)
        if not spec:
            logger.debug(f"No embeddings adapter spec for provider '{name}'")
            return None
        try:
            adapter_cls = self._resolve_adapter_class(spec)
            adapter = adapter_cls()  # type: ignore[call-arg]
            if not isinstance(adapter, EmbeddingsProvider):
                logger.error(f"Embeddings adapter for '{name}' does not implement EmbeddingsProvider")
                return None
            self._adapters[name] = adapter
            return adapter
        except Exception as e:
            logger.error(f"Failed to initialize embeddings adapter for '{name}': {e}")
            return None

    def get_all_capabilities(self) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for name in list(self._adapter_specs.keys()):
            adapter = self.get_adapter(name)
            if not adapter:
                continue
            try:
                out[name] = adapter.capabilities() or {}
            except Exception as e:
                logger.warning(f"Embeddings capability discovery failed for '{name}': {e}")
        return out


_emb_registry: EmbeddingsProviderRegistry | None = None


def get_embeddings_registry() -> EmbeddingsProviderRegistry:
    global _emb_registry
    if _emb_registry is None:
        _emb_registry = EmbeddingsProviderRegistry()
    return _emb_registry
