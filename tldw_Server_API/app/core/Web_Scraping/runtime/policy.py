"""Protocol-only policy boundary for Web_Scraping runtime callers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

from .requests import RuntimeRequestContext
from .responses import PolicyDecision


class OutboundPolicyChecker(Protocol):
    """Async scrape-level outbound policy checker."""

    async def decide(
        self,
        url: str,
        *,
        respect_robots: bool,
        user_agent: str | None,
        context: RuntimeRequestContext,
        config: Mapping[str, Any] | None = None,
    ) -> PolicyDecision:
        """Return the outbound policy decision for a scrape request."""
