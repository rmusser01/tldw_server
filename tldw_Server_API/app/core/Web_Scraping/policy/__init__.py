"""Concrete policy adapters for Web_Scraping."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .probe import DefaultProbeEgressGuard

if TYPE_CHECKING:
    from .adapters import DefaultWebOutboundPolicyChecker

__all__ = ["DefaultProbeEgressGuard", "DefaultWebOutboundPolicyChecker"]


def __getattr__(name: str) -> Any:
    if name == "DefaultWebOutboundPolicyChecker":
        from .adapters import DefaultWebOutboundPolicyChecker

        globals()[name] = DefaultWebOutboundPolicyChecker
        return DefaultWebOutboundPolicyChecker
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
