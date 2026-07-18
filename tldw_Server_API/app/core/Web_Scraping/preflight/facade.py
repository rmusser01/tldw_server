"""Scrape-level policy admission boundary for governed preflight targets."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..runtime.policy import OutboundPolicyChecker
from ..runtime.requests import RuntimeRequestContext
from .target import PreflightTarget


async def evaluate_target(
    url: str,
    *,
    respect_robots: bool,
    user_agent: str | None,
    request_context: RuntimeRequestContext,
    config: Mapping[str, Any] | None,
    policy_checker: OutboundPolicyChecker,
) -> PreflightTarget:
    """Bind a target to its single scrape-level outbound policy decision."""
    decision = await policy_checker.decide(
        url,
        respect_robots=respect_robots,
        user_agent=user_agent,
        context=request_context,
        config=config,
    )
    return PreflightTarget(
        url=url,
        decision=decision,
        request_context=request_context,
    )
