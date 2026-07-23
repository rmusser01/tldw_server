"""Deterministic governed preflight orchestration and legacy aggregate APIs."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any, TypedDict

from loguru import logger

from ..runtime.requests import RuntimeRequestContext
from .analyzers.behavioral_detector import (
    _detect_honeypots,
    detect_honeypots,
)
from .analyzers.captcha_detector import _detect_captcha, detect_captcha
from .analyzers.fingerprint_analyzer import (
    _analyze_fingerprinting,
    analyze_fingerprinting,
)
from .analyzers.integrity_analyzer import (
    _analyze_function_integrity,
    analyze_function_integrity,
)
from .analyzers.js_detector import _analyze_js_rendering, analyze_js_rendering
from .analyzers.rate_limit_profiler import (
    _profile_rate_limits,
    profile_rate_limits,
)
from .analyzers.robots_checker import _check_robots_txt, check_robots_txt
from .analyzers.tls_analyzer import (
    _analyze_tls_fingerprint,
    analyze_tls_fingerprint,
)
from .analyzers.waf_detector import _detect_waf, detect_waf
from .context import (
    PreflightDeadlineExceeded,
    PreflightExecutionContext,
)
from .facade import (
    _default_policy_checker,
    build_execution_context,
    evaluate_target,
)
from .options import PreflightOptions, ScanDepth
from .probes import ProbeError
from .recommendations.recommender import generate_recommendations
from .scoring.scoring_engine import calculate_difficulty_score
from .target import PreflightTarget

ANALYZER_KEYS = (
    "robots",
    "tls",
    "js",
    "behavioral",
    "captcha",
    "fingerprint",
    "integrity",
    "rate_limit",
    "waf",
)

_ANALYZER_FAILURE_MESSAGES = {
    "robots": "Robots.txt check failed.",
    "tls": "TLS fingerprint analysis failed.",
    "js": "JavaScript rendering analysis failed.",
    "behavioral": "Honeypot detection failed.",
    "captcha": "Captcha detection failed.",
    "fingerprint": "Fingerprint analysis failed.",
    "integrity": "Function integrity analysis failed.",
    "rate_limit": "Rate limit profiling failed.",
    "waf": "WAF detection failed.",
}

_POLICY_DENIED = {
    "status": "error",
    "message": "Probe destination was denied.",
    "error_code": "policy_denied",
}
_POLICY_ERROR = {**_POLICY_DENIED, "error_code": "policy_error"}
_TIMEOUT_ERROR = {
    "status": "error",
    "message": "Probe timed out.",
    "error_code": "timeout",
}


class AnalysisOutput(TypedDict):
    """Historical aggregate analyzer result shape."""

    results: dict[str, Any]
    score: dict[str, Any]
    recommendations: dict[str, Any]


async def _isolated(
    name: str,
    call: Callable[[], Awaitable[dict[str, Any]]],
) -> dict[str, Any]:
    """Run one analyzer while preserving cancellation and safe failures."""
    try:
        return await call()
    except asyncio.CancelledError:
        raise
    except PreflightDeadlineExceeded:
        raise
    except ProbeError as exc:
        return {
            "status": "error",
            "message": exc.public_message,
            "error_code": exc.error_code,
        }
    except Exception as exc:  # noqa: BLE001 - analyzer failures are isolated and sanitized
        logger.warning(
            "Preflight analyzer failure: analyzer={} exception={}",
            name,
            type(exc).__name__,
        )
        return {
            "status": "error",
            "message": _ANALYZER_FAILURE_MESSAGES[name],
            "error_code": "analyzer_error",
        }


async def gather_analysis_with_context(
    target: PreflightTarget,
    options: PreflightOptions,
    context: PreflightExecutionContext,
) -> AnalysisOutput:
    """Run all private analyzers sequentially in the approved stable order."""
    results: dict[str, Any] = {}
    results["robots"] = await _isolated(
        "robots",
        lambda: _check_robots_txt(target.url, context),
    )
    crawl_delay = results["robots"].get("crawl_delay")
    results["tls"] = await _isolated(
        "tls",
        lambda: _analyze_tls_fingerprint(target.url, context),
    )
    results["js"] = await _isolated(
        "js",
        lambda: _analyze_js_rendering(target.url, context),
    )
    results["behavioral"] = await _isolated(
        "behavioral",
        lambda: _detect_honeypots(
            target.url,
            context,
            options.scan_depth,
        ),
    )
    results["captcha"] = await _isolated(
        "captcha",
        lambda: _detect_captcha(target.url, context),
    )
    results["fingerprint"] = await _isolated(
        "fingerprint",
        lambda: _analyze_fingerprinting(target.url, context),
    )
    results["integrity"] = await _isolated(
        "integrity",
        lambda: _analyze_function_integrity(target.url, context),
    )
    results["rate_limit"] = await _isolated(
        "rate_limit",
        lambda: _profile_rate_limits(
            target.url,
            context,
            crawl_delay,
            options.impersonate,
        ),
    )
    results["waf"] = await _isolated(
        "waf",
        lambda: _detect_waf(
            target.url,
            context,
            options.find_all_waf,
            options.external_tools_enabled,
        ),
    )
    return {
        "results": results,
        "score": calculate_difficulty_score(results),
        "recommendations": generate_recommendations(results),
    }


def _policy_failure_analysis(error_code: str) -> AnalysisOutput:
    entry = _POLICY_DENIED if error_code == "policy_denied" else _POLICY_ERROR
    results = {key: dict(entry) for key in ANALYZER_KEYS}
    return {
        "results": results,
        "score": calculate_difficulty_score(results),
        "recommendations": generate_recommendations(results),
    }


def _timeout_failure_analysis() -> AnalysisOutput:
    results = {key: dict(_TIMEOUT_ERROR) for key in ANALYZER_KEYS}
    return {
        "results": results,
        "score": calculate_difficulty_score(results),
        "recommendations": generate_recommendations(results),
    }


async def gather_analysis(
    url: str,
    *,
    find_all: bool = False,
    impersonate: bool = False,
    scan_depth: ScanDepth | None = None,
) -> AnalysisOutput:
    """Run the historical aggregate API through one governed context."""
    request_context = RuntimeRequestContext(source="preflight", stage="preflight")
    try:
        policy_checker = _default_policy_checker()
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
    except Exception as exc:  # noqa: BLE001 - direct legacy policy failures are sanitized
        logger.warning("Legacy preflight policy failure: exception={}", type(exc).__name__)
        return _policy_failure_analysis("policy_error")

    if not allowed:
        return _policy_failure_analysis("policy_denied")

    options = PreflightOptions(
        enabled=True,
        scan_depth=scan_depth or "default",
        find_all_waf=find_all,
        impersonate=impersonate,
    )
    context = build_execution_context(
        target,
        options,
        policy_checker=policy_checker,
    )
    try:
        try:
            return await gather_analysis_with_context(target, options, context)
        except PreflightDeadlineExceeded:
            return _timeout_failure_analysis()
    finally:
        try:
            await context.close()
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - preserve the established runner outcome
            logger.warning("Legacy aggregate context cleanup failed.")


def run_analysis(
    url: str,
    *,
    find_all: bool = False,
    impersonate: bool = False,
    scan_depth: ScanDepth | None = None,
) -> AnalysisOutput:
    """Run the legacy aggregate API when no event loop is active."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(
            gather_analysis(
                url,
                find_all=find_all,
                impersonate=impersonate,
                scan_depth=scan_depth,
            )
        )

    raise RuntimeError(
        "run_analysis cannot be used inside an active event loop; " "use 'await gather_analysis' instead."
    )


__all__ = [
    "AnalysisOutput",
    "ScanDepth",
    "analyze_fingerprinting",
    "analyze_function_integrity",
    "analyze_js_rendering",
    "analyze_tls_fingerprint",
    "check_robots_txt",
    "detect_captcha",
    "detect_honeypots",
    "detect_waf",
    "gather_analysis",
    "gather_analysis_with_context",
    "profile_rate_limits",
    "run_analysis",
]
