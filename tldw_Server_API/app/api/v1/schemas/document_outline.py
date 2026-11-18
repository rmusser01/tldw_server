"""Compatibility exports for core-owned Media contracts."""

from tldw_Server_API.app.core.Media.document_outline_schemas import (
    DocumentOutlineResponse as DocumentOutlineResponse,
)
from tldw_Server_API.app.core.Media.document_outline_schemas import (
    OutlineEntry as OutlineEntry,
)

__all__ = ["OutlineEntry", "DocumentOutlineResponse"]
