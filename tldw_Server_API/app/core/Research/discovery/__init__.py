"""Research discovery source catalog primitives."""

from .adapters import default_discovery_adapters
from .catalog import CATALOG_VERSION, ResearchSourceCatalog, default_source_catalog
from .models import (
    DiscoveryExecutionPolicy,
    ResearchSourceCatalogEntry,
    SourceCapabilities,
    SourceSelectionError,
    SourceStatus,
)
from .router import ResearchSourceRouter

__all__ = [
    "CATALOG_VERSION",
    "DiscoveryExecutionPolicy",
    "ResearchSourceCatalog",
    "ResearchSourceCatalogEntry",
    "ResearchSourceRouter",
    "SourceCapabilities",
    "SourceSelectionError",
    "SourceStatus",
    "default_discovery_adapters",
    "default_source_catalog",
]
