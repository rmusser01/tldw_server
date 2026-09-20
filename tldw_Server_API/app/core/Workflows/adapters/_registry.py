"""Adapter registry for workflow step types.

This module provides a decorator-based registration system for workflow adapters.
Each adapter is registered with metadata including category, description, and
parallelizability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from pydantic import BaseModel


@dataclass
class AdapterSpec:
    """Metadata for a registered adapter."""

    name: str
    func: Callable
    category: str
    description: str
    parallelizable: bool = True
    config_model: type[BaseModel] | None = None
    tags: list[str] = field(default_factory=list)


class AdapterRegistry:
    """Central registry for workflow adapters.

    This registry uses a singleton pattern to ensure all adapters are registered
    to a single global instance. Adapters are registered using the @register
    decorator.

    Example:
        from tldw_Server_API.app.core.Workflows.adapters._registry import registry

        @registry.register(
            "llm",
            category="llm",
            description="Invoke an LLM chat completion",
            parallelizable=True,
        )
        async def run_llm_adapter(config, context):
            ...
    """

    _instance: AdapterRegistry | None = None

    def __init__(self) -> None:
        self._adapters: dict[str, AdapterSpec] = {}
        # Adapter category modules are imported on first registry access rather
        # than at import time; see set_loader().
        self._loader: Callable[[], None] | None = None
        self._loading = False
        self._loaded = False

    def set_loader(self, loader: Callable[[], None]) -> None:
        """Install the callback that imports every adapter category module.

        The adapters package registers its categories through this hook instead
        of importing them at module scope. Importing them eagerly pulled the RAG
        pipeline -- and through it nltk, scipy, sklearn and pandas -- into every
        process that touched workflows, including CLI runs and test sessions.
        """
        self._loader = loader

    def _ensure_loaded(self) -> None:
        """Import all adapter categories once, before answering a query."""
        if self._loaded or self._loading or self._loader is None:
            return
        self._loading = True
        try:
            self._loader()
            self._loaded = True
        finally:
            self._loading = False

    @classmethod
    def get(cls) -> AdapterRegistry:
        """Get the singleton registry instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the registry (primarily for testing)."""
        cls._instance = None

    def register(
        self,
        name: str,
        *,
        category: str = "misc",
        description: str = "",
        parallelizable: bool = True,
        config_model: type[BaseModel] | None = None,
        tags: list[str] | None = None,
    ) -> Callable[[Callable], Callable]:
        """Decorator to register an adapter with metadata.

        Args:
            name: The step type name (e.g., "llm", "prompt")
            category: Category for grouping (e.g., "llm", "audio", "control")
            description: Human-readable description of what the adapter does
            parallelizable: Whether this adapter can run in parallel (for map steps)
            config_model: Optional Pydantic model for config validation
            tags: Optional list of tags for filtering/search

        Returns:
            Decorator function that registers the adapter
        """

        def decorator(func: Callable) -> Callable:
            self._adapters[name] = AdapterSpec(
                name=name,
                func=func,
                category=category,
                description=description,
                parallelizable=parallelizable,
                config_model=config_model,
                tags=tags or [],
            )
            return func

        return decorator

    def get_adapter(self, name: str) -> Callable | None:
        """Get an adapter function by name.

        Args:
            name: The step type name

        Returns:
            The adapter function, or None if not found
        """
        self._ensure_loaded()
        spec = self._adapters.get(name)
        return spec.func if spec else None

    def get_spec(self, name: str) -> AdapterSpec | None:
        """Get the full adapter spec by name.

        Args:
            name: The step type name

        Returns:
            The AdapterSpec, or None if not found
        """
        self._ensure_loaded()
        return self._adapters.get(name)

    def list_adapters(self) -> list[str]:
        """List all registered adapter names.

        Returns:
            List of adapter names
        """
        self._ensure_loaded()
        return list(self._adapters.keys())

    def get_parallelizable(self) -> set[str]:
        """Get the set of parallelizable adapter names.

        This replaces the MAP_SUBSTEP_TYPES constant from constants.py.

        Returns:
            Set of adapter names that can run in parallel
        """
        self._ensure_loaded()
        return {name for name, spec in self._adapters.items() if spec.parallelizable}

    def get_by_category(self, category: str) -> list[str]:
        """Get adapter names by category.

        Args:
            category: The category to filter by

        Returns:
            List of adapter names in that category
        """
        self._ensure_loaded()
        return [name for name, spec in self._adapters.items() if spec.category == category]

    def get_categories(self) -> list[str]:
        """Get all unique categories.

        Returns:
            List of unique category names
        """
        self._ensure_loaded()
        return list({spec.category for spec in self._adapters.values()})

    def get_catalog(self) -> dict[str, list[dict[str, Any]]]:
        """Get full catalog grouped by category.

        Returns:
            Dict mapping category names to adapter metadata entries
        """
        self._ensure_loaded()
        catalog: dict[str, list[dict[str, Any]]] = {}
        for name, spec in self._adapters.items():
            catalog.setdefault(spec.category, []).append(
                {
                    "name": name,
                    "description": spec.description,
                    "parallelizable": spec.parallelizable,
                    "config_schema": spec.config_model.model_json_schema() if spec.config_model else None,
                    "tags": spec.tags,
                }
            )
        for entries in catalog.values():
            entries.sort(key=lambda item: str(item.get("name") or ""))
        return catalog

    def __len__(self) -> int:
        self._ensure_loaded()
        return len(self._adapters)

    def __contains__(self, name: str) -> bool:
        self._ensure_loaded()
        return name in self._adapters


# Global singleton instance
registry = AdapterRegistry.get()


def get_adapter(name: str) -> Callable | None:
    """Convenience function to get an adapter by name.

    Args:
        name: The step type name

    Returns:
        The adapter function, or None if not found
    """
    return registry.get_adapter(name)


def get_parallelizable() -> set[str]:
    """Convenience function to get parallelizable adapter names.

    Returns:
        Set of adapter names that can run in parallel
    """
    return registry.get_parallelizable()
