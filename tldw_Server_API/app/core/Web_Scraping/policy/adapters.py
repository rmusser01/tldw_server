"""Concrete outbound policy adapters."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from tldw_Server_API.app.core.Web_Scraping.outbound_policy import decide_web_outbound_policy
from tldw_Server_API.app.core.Web_Scraping.runtime import (
    PolicyDecision,
    RuntimeRequestContext,
)

from ..runtime.controls import ParentControls
from .probe import DefaultProbeEgressGuard as DefaultProbeEgressGuard


class DefaultWebOutboundPolicyChecker:
    def __init__(self, *, credential_free: bool = False, parent: ParentControls | None = None) -> None:
        self.credential_free = credential_free
        self.parent = parent

    async def decide(
        self,
        url: str,
        *,
        respect_robots: bool,
        user_agent: str | None,
        context: RuntimeRequestContext,
        config: Mapping[str, Any] | None = None,
    ) -> PolicyDecision:
        robots_kwargs: dict[str, Any] = {}
        if self.parent is not None:
            from ..filters import RobotsFilter
            from ..preflight.adapters.http import GuardedHttpProbe
            from ..preflight.context import PreflightRuntimeControls
            from ..preflight.probes import ProbeHttpRequest

            self.parent.check()
            controls = PreflightRuntimeControls(context, parent=self.parent)
            probe = GuardedHttpProbe(controls=controls, egress_guard=DefaultProbeEgressGuard())

            async def fetch_robots(robots_url: str) -> dict[str, Any]:
                response = await probe.get(
                    ProbeHttpRequest(
                        url=robots_url, headers={"User-Agent": user_agent or "*"}, timeout_s=5, allow_redirects=True
                    )
                )
                return {"text": response.text, "status": response.status}

            robots_kwargs["robots_filter"] = RobotsFilter(user_agent=user_agent or "*", fetch_robots=fetch_robots)
        raw = await decide_web_outbound_policy(
            url,
            respect_robots=respect_robots,
            user_agent=user_agent,
            source=context.source,
            stage=context.stage,
            config=dict(config or {}),
            **({"credential_free": True} if self.credential_free else {}),
            **robots_kwargs,
        )
        return PolicyDecision(
            allowed=bool(raw.allowed),
            mode=str(raw.mode),
            reason=str(raw.reason),
            stage=str(raw.stage),
            source=str(raw.source),
            details=getattr(raw, "details", None),
        )
