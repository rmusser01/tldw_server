"""Policy admission and compatibility construction for preflight targets."""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from time import monotonic
from typing import Any

from loguru import logger

from ..policy import DefaultProbeEgressGuard, DefaultWebOutboundPolicyChecker
from ..runtime.policy import OutboundPolicyChecker, ProbeEgressGuard
from ..runtime.requests import RuntimeRequestContext
from .adapters.browser import GuardedPlaywrightBrowserProbe
from .adapters.external_tools import GuardedExternalToolProbe
from .adapters.http import GuardedHttpProbe
from .context import PreflightExecutionContext, PreflightLimits, PreflightRuntimeControls
from .options import PreflightOptions
from .probes import BrowserProbe, ExternalToolProbe, HttpProbe
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


@dataclass(frozen=True, slots=True)
class PreflightAdapterOverrides:
    """Optional replacements for individual context dependencies."""

    http: HttpProbe | None = None
    browser: BrowserProbe | None = None
    external_tools: ExternalToolProbe | None = None
    egress_guard: ProbeEgressGuard | None = None
    clock: Callable[[], float] | None = None
    sleep: Callable[[float], Awaitable[None]] | None = None


def _default_identity_selector() -> Mapping[str, str]:
    from ..scraper_analyzers.utils.browser_identities import MODERN_BROWSER_IDENTITIES

    # Browser profile selection is not security-sensitive.
    return random.choice(MODERN_BROWSER_IDENTITIES)  # nosec B311


def _default_policy_checker() -> OutboundPolicyChecker:
    return DefaultWebOutboundPolicyChecker()


def _default_egress_guard() -> ProbeEgressGuard:
    return DefaultProbeEgressGuard()


def build_execution_context(
    target: PreflightTarget,
    options: PreflightOptions,
    *,
    policy_checker: OutboundPolicyChecker | None = None,
    limits: PreflightLimits | None = None,
    identity_selector: Callable[[], Mapping[str, str]] | None = None,
    injected_adapters: PreflightAdapterOverrides | None = None,
) -> PreflightExecutionContext:
    """Build one governed dependency graph for a preflight execution."""
    overrides = injected_adapters or PreflightAdapterOverrides()
    selected_clock = overrides.clock if overrides.clock is not None else monotonic
    selected_sleep = overrides.sleep if overrides.sleep is not None else asyncio.sleep
    selected_limits = limits if limits is not None else PreflightLimits()
    selected_checker = policy_checker if policy_checker is not None else _default_policy_checker()
    selected_guard = overrides.egress_guard if overrides.egress_guard is not None else _default_egress_guard()

    deadline = None
    if options.timeout_s is not None and options.timeout_s > 0:
        deadline = selected_clock() + options.timeout_s
    controls = PreflightRuntimeControls(
        request_context=target.request_context,
        limits=selected_limits,
        deadline=deadline,
        clock=selected_clock,
        sleep=selected_sleep,
    )

    http = overrides.http
    if http is None:
        http = GuardedHttpProbe(controls=controls, egress_guard=selected_guard)
    browser = overrides.browser
    if browser is None:
        browser = GuardedPlaywrightBrowserProbe(
            controls=controls,
            egress_guard=selected_guard,
            no_sandbox=options.playwright_no_sandbox,
        )
    external_tools = overrides.external_tools
    if external_tools is None:
        external_tools = GuardedExternalToolProbe(
            controls=controls,
            egress_guard=selected_guard,
        )

    return PreflightExecutionContext(
        request_context=target.request_context,
        policy_checker=selected_checker,
        egress_guard=selected_guard,
        controls=controls,
        http=http,
        browser=browser,
        external_tools=external_tools,
        identity_selector=(identity_selector if identity_selector is not None else _default_identity_selector),
    )


_POLICY_DENIED = {
    "status": "error",
    "message": "Probe destination was denied.",
    "error_code": "policy_denied",
}
_POLICY_ERROR = {**_POLICY_DENIED, "error_code": "policy_error"}


async def run_legacy_analyzer(
    url: str,
    analyzer: Callable[..., Awaitable[Any]],
    *args: Any,
    policy_checker_factory: Callable[[], OutboundPolicyChecker] = _default_policy_checker,
    context_factory: Callable[..., PreflightExecutionContext] = build_execution_context,
    **kwargs: Any,
) -> Any:
    """Admit and invoke one supplied legacy analyzer under governed context."""
    request_context = RuntimeRequestContext(source="preflight", stage="preflight")
    try:
        policy_checker = policy_checker_factory()
        target = await evaluate_target(
            url,
            respect_robots=False,
            user_agent=None,
            request_context=request_context,
            config=None,
            policy_checker=policy_checker,
        )
        allowed = bool(target.decision.allowed)
    except asyncio.CancelledError:
        raise
    except Exception:  # noqa: BLE001 - policy failures must be sanitized
        return dict(_POLICY_ERROR)
    if not allowed:
        return dict(_POLICY_DENIED)

    context = context_factory(
        target,
        PreflightOptions(),
        policy_checker=policy_checker,
    )
    try:
        return await analyzer(url, context, *args, **kwargs)
    finally:
        try:
            await context.close()
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - preserve the analyzer outcome
            logger.warning("Legacy analyzer context cleanup failed.")
