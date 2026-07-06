"""Concrete outbound policy adapters."""

from __future__ import annotations

from typing import Any, Mapping

from tldw_Server_API.app.core.Web_Scraping.outbound_policy import decide_web_outbound_policy
from tldw_Server_API.app.core.Web_Scraping.runtime import PolicyDecision, RuntimeRequestContext


class DefaultWebOutboundPolicyChecker:
    async def decide(
        self,
        url: str,
        *,
        respect_robots: bool,
        user_agent: str | None,
        context: RuntimeRequestContext,
        config: Mapping[str, Any] | None = None,
    ) -> PolicyDecision:
        raw = await decide_web_outbound_policy(
            url,
            respect_robots=respect_robots,
            user_agent=user_agent,
            source=context.source,
            stage=context.stage,
            config=dict(config or {}),
        )
        return PolicyDecision(
            allowed=bool(raw.allowed),
            mode=str(raw.mode),
            reason=str(raw.reason),
            stage=str(raw.stage),
            source=str(raw.source),
            details=getattr(raw, "details", None),
        )
