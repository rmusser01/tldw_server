"""Concrete policy adapters for Web_Scraping."""

from __future__ import annotations

from .adapters import DefaultProbeEgressGuard, DefaultWebOutboundPolicyChecker

__all__ = ["DefaultProbeEgressGuard", "DefaultWebOutboundPolicyChecker"]
