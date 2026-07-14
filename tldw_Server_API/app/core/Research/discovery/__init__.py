"""Research discovery source catalog primitives."""

from importlib import import_module

__all__ = [
    "CATALOG_VERSION",
    "DiscoveryExecutionPolicy",
    "DiscoveryMetrics",
    "DiscoveryProviderRouter",
    "DiscoverySearchResponse",
    "DiscoverySourceStatus",
    "ResearchSourceCatalog",
    "ResearchSourceCatalogEntry",
    "ResearchDiscoveryService",
    "ResearchSourceRouter",
    "SourceCapabilities",
    "SourceSelectionError",
    "SourceStatus",
    "default_discovery_adapters",
    "default_source_catalog",
]

_EXPORTS = {
    "CATALOG_VERSION": (".catalog", "CATALOG_VERSION"),
    "DiscoveryExecutionPolicy": (".models", "DiscoveryExecutionPolicy"),
    "DiscoveryMetrics": (".models", "DiscoveryMetrics"),
    "DiscoveryProviderRouter": (".service", "DiscoveryProviderRouter"),
    "DiscoverySearchResponse": (".models", "DiscoverySearchResponse"),
    "DiscoverySourceStatus": (".models", "DiscoverySourceStatus"),
    "ResearchSourceCatalog": (".catalog", "ResearchSourceCatalog"),
    "ResearchSourceCatalogEntry": (".models", "ResearchSourceCatalogEntry"),
    "ResearchDiscoveryService": (".service", "ResearchDiscoveryService"),
    "ResearchSourceRouter": (".router", "ResearchSourceRouter"),
    "SourceCapabilities": (".models", "SourceCapabilities"),
    "SourceSelectionError": (".models", "SourceSelectionError"),
    "SourceStatus": (".models", "SourceStatus"),
    "default_discovery_adapters": (".adapters", "default_discovery_adapters"),
    "default_source_catalog": (".catalog", "default_source_catalog"),
}
_SUBMODULES = frozenset({"adapters", "catalog", "models", "router", "service"})


def __getattr__(name: str) -> object:
    """Resolve legacy public exports without importing unrelated modules."""
    if name in _SUBMODULES:
        return import_module(f".{name}", __name__)
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Include lazily exported names in interactive discovery."""
    return sorted({*globals(), *__all__, *_SUBMODULES})
