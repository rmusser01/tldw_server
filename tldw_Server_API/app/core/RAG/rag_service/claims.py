"""
Backward-compatible wrapper that re-exports claim extraction/verification
from the new ingestion-time claims engine module.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tldw_Server_API.app.core.Claims_Extraction.claims_engine import ClaimsEngine


def __getattr__(name: str) -> Any:
    """Resolve the claims engine only after circular package imports finish."""
    if name != "ClaimsEngine":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from tldw_Server_API.app.core.Claims_Extraction.claims_engine import (
        ClaimsEngine as _ResolvedClaimsEngine,
    )

    globals()[name] = _ResolvedClaimsEngine
    return _ResolvedClaimsEngine


__all__ = ["ClaimsEngine"]
