"""Research discovery source catalog primitives."""

from .catalog import CATALOG_VERSION, ResearchSourceCatalog, default_source_catalog
from .models import ResearchSourceCatalogEntry, SourceCapabilities, SourceSelectionError

__all__ = [
    "CATALOG_VERSION",
    "ResearchSourceCatalog",
    "ResearchSourceCatalogEntry",
    "SourceCapabilities",
    "SourceSelectionError",
    "default_source_catalog",
]
