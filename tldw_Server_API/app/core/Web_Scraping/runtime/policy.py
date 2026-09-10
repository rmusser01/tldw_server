"""Protocol-only policy boundary for Web_Scraping runtime callers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from .requests import RuntimeRequestContext
from .responses import PolicyDecision


@dataclass(frozen=True, slots=True)
class ProbeEgressDecision:
    """Fresh egress decision for one concrete probe dispatch."""

    allowed: bool
    reason: str
    resolved_ips: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "resolved_ips", tuple(self.resolved_ips))


class ProbeEgressGuard(Protocol):
    """Check fresh egress policy immediately before a probe dispatch."""

    async def decide(
        self,
        url: str,
        *,
        context: RuntimeRequestContext,
    ) -> ProbeEgressDecision:
        """Return the probe-level egress decision for one URL."""
        raise NotImplementedError


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
