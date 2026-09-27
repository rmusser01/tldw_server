"""Policy admission and compatibility construction for preflight targets."""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, replace
from time import monotonic
from typing import Any

from loguru import logger

from ..contracts import (
    PreflightAdvice,
    PreflightResult,
    RuntimeFailure,
    WebScrapingStatus,
)
from ..contracts.conversion import preflight_result_to_public_dict
from ..runtime.policy import OutboundPolicyChecker, ProbeEgressGuard
from ..runtime.requests import RuntimeRequestContext
from .context import (
    PreflightDeadlineExceeded,
    PreflightExecutionContext,
    PreflightLimits,
    PreflightRuntimeControls,
)
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
    from .utils.browser_identities import MODERN_BROWSER_IDENTITIES

    # Browser profile selection is not security-sensitive.
    return random.choice(MODERN_BROWSER_IDENTITIES)  # nosec B311


def _default_policy_checker() -> OutboundPolicyChecker:
    from ..policy import DefaultWebOutboundPolicyChecker

    return DefaultWebOutboundPolicyChecker()


def _default_egress_guard() -> ProbeEgressGuard:
    from ..policy import DefaultProbeEgressGuard

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
        from .adapters.http import GuardedHttpProbe

        http = GuardedHttpProbe(controls=controls, egress_guard=selected_guard)
    browser = overrides.browser
    if browser is None:
        from .adapters.browser import GuardedPlaywrightBrowserProbe

        browser = GuardedPlaywrightBrowserProbe(
            controls=controls,
            egress_guard=selected_guard,
            no_sandbox=options.playwright_no_sandbox,
        )
    external_tools = overrides.external_tools
    if external_tools is None:
        from .adapters.external_tools import GuardedExternalToolProbe

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


def _advice_signals(analysis: Mapping[str, Any]) -> tuple[bool, bool]:
    results = analysis.get("results")
    if not isinstance(results, Mapping):
        return False, False

    js_result = results.get("js")
    js_required = bool(
        isinstance(js_result, Mapping)
        and js_result.get("status") == "success"
        and (js_result.get("js_required") or js_result.get("is_spa"))
    )
    tls_result = results.get("tls")
    tls_active = bool(isinstance(tls_result, Mapping) and tls_result.get("status") == "active")
    return js_required, tls_active


def _derive_advice(analysis: Mapping[str, Any]) -> PreflightAdvice:
    js_required, tls_active = _advice_signals(analysis)
    notes: list[str] = []
    if js_required:
        notes.append("js_required")
    if tls_active:
        notes.append("tls_active")
    return PreflightAdvice(
        backend="curl" if tls_active else None,
        method="playwright" if js_required else None,
        notes=tuple(notes),
    )


def _failed_preflight(
    status: WebScrapingStatus,
    public_message: str,
) -> PreflightResult:
    return PreflightResult(
        status=status,
        failure=RuntimeFailure(
            status=status,
            public_message=public_message,
        ),
    )


def _consume_runner_task(task: asyncio.Task[Any]) -> None:
    try:
        task.exception()
    except asyncio.CancelledError:
        pass


async def _retire_runner_task(task: asyncio.Task[Any]) -> None:
    if not task.done():
        task.cancel()

    current = asyncio.current_task()
    observed_cancellations = current.cancelling() if current is not None else 0
    pending_cancellation: asyncio.CancelledError | None = None
    while not task.done():
        try:
            await asyncio.wait({task})
        except asyncio.CancelledError as exc:
            current_cancellations = current.cancelling() if current is not None else 0
            if current_cancellations > observed_cancellations:
                observed_cancellations = current_cancellations
                if pending_cancellation is None:
                    pending_cancellation = exc
                if not task.done():
                    task.cancel()

    _consume_runner_task(task)
    if pending_cancellation is not None:
        raise pending_cancellation


async def _run_before_deadline(
    target: PreflightTarget,
    options: PreflightOptions,
    context: PreflightExecutionContext,
) -> Mapping[str, Any]:
    from .runner import gather_analysis_with_context

    remaining_s = context.controls.remaining_seconds()
    if remaining_s is not None and remaining_s <= 0:
        raise PreflightDeadlineExceeded

    runner_task = asyncio.create_task(
        gather_analysis_with_context(target, options, context),
        name="preflight-runner",
    )
    if remaining_s is None:
        runner_waiter = asyncio.create_task(
            asyncio.wait({runner_task}),
            name="preflight-runner-waiter",
        )
        try:
            await asyncio.shield(runner_waiter)
        except asyncio.CancelledError:
            try:
                await _retire_runner_task(runner_task)
            except asyncio.CancelledError:
                # Retirement may surface the child cancellation; the outer one wins.
                pass
            await runner_waiter
            raise
        return runner_task.result()

    deadline_task = asyncio.create_task(
        asyncio.sleep(remaining_s),
        name="preflight-deadline",
    )
    try:
        try:
            done, _pending = await asyncio.wait(
                {runner_task, deadline_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
        except asyncio.CancelledError:
            await _retire_runner_task(runner_task)
            raise

        current = asyncio.current_task()
        if current is not None and current.cancelling():
            await _retire_runner_task(runner_task)
            raise asyncio.CancelledError from None
        if runner_task in done:
            return runner_task.result()

        await _retire_runner_task(runner_task)
        current = asyncio.current_task()
        if current is not None and current.cancelling():
            raise asyncio.CancelledError from None
        raise PreflightDeadlineExceeded from None
    finally:
        await _retire_runner_task(deadline_task)


async def run_preflight(
    target: PreflightTarget,
    options: PreflightOptions,
    context: PreflightExecutionContext,
) -> PreflightResult | None:
    """Run governed preflight analysis and normalize only overall failures."""
    if not options.enabled:
        return None
    if not target.decision.allowed:
        raise ValueError("run_preflight requires an allowed target")

    try:
        analysis = await _run_before_deadline(target, options, context)
        return PreflightResult(
            analysis=analysis,
            advice=_derive_advice(analysis),
        )
    except asyncio.CancelledError:
        raise
    except PreflightDeadlineExceeded:
        return _failed_preflight(
            WebScrapingStatus.TIMEOUT,
            "Preflight analysis timed out.",
        )
    except Exception:  # noqa: BLE001 - overall preflight failure is sanitized
        return _failed_preflight(
            WebScrapingStatus.ERROR,
            "Preflight analysis failed.",
        )
    finally:
        try:
            await context.close()
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - cleanup cannot replace an outcome
            logger.warning("Preflight context cleanup failed.")


def apply_preflight_advice(
    result: PreflightResult | None,
    *,
    backend: str,
    method: str,
    backend_setting: str,
) -> tuple[str, str, PreflightResult | None]:
    """Apply successful analyzer signals to current routing selections."""
    if result is None:
        return backend, method, None

    notes: list[str] = []
    if result.status is WebScrapingStatus.OK:
        js_required, tls_active = _advice_signals(result.analysis)
        if method == "auto" and js_required:
            method = "playwright"
            notes.append("js_required")
        if backend_setting == "auto" and tls_active:
            backend = "curl"
            notes.append("tls_active")

    updated = replace(
        result,
        advice=PreflightAdvice(
            backend=backend,
            method=method,
            notes=tuple(notes),
        ),
    )
    return backend, method, updated


def public_preflight_payload(
    result: PreflightResult | None,
    include_results: bool,
) -> dict[str, Any] | None:
    """Return the legacy payload only for included successful overall runs."""
    if not include_results or result is None or result.status is not WebScrapingStatus.OK:
        return None
    return preflight_result_to_public_dict(result)
