"""Research discovery source catalog primitives."""

from .adapters import default_discovery_adapters
from .catalog import CATALOG_VERSION, ResearchSourceCatalog, default_source_catalog
from .models import (
    DiscoveryExecutionPolicy,
    DiscoveryMetrics,
    ResearchSourceCatalogEntry,
    SourceCapabilities,
    SourceSelectionError,
    SourceStatus,
    DiscoverySearchResponse,
    DiscoverySourceStatus,
)
from .router import ResearchSourceRouter
from .service import DiscoveryProviderRouter, ResearchDiscoveryService

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
