from __future__ import annotations

import asyncio
import inspect
from collections.abc import Mapping
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, get_type_hints
from unittest.mock import AsyncMock, Mock

import pytest

from tldw_Server_API.app.core.Web_Scraping import preflight as preflight_facade
from tldw_Server_API.app.core.Web_Scraping.browser_transport import decide_browser_transport
from tldw_Server_API.app.core.Web_Scraping.contracts import (
    PreflightAdvice,
    PreflightResult,
    RuntimeFailure,
    WebScrapingStatus,
)
from tldw_Server_API.app.core.Web_Scraping.policy import DefaultWebOutboundPolicyChecker
from tldw_Server_API.app.core.Web_Scraping.preflight import PreflightTarget
from tldw_Server_API.app.core.Web_Scraping.runtime import PolicyDecision, RuntimeRequestContext

URL = "https://example.com/article"
ANALYSIS = {"results": {"headers": {"status": "success"}}}


def allowed_decision() -> PolicyDecision:
    return PolicyDecision(
        allowed=True,
        mode="compat",
        reason="allowed",
        stage="pre_fetch",
        source="enhanced_scrape",
    )


def denied_decision() -> PolicyDecision:
    return PolicyDecision(
        allowed=False,
        mode="strict",
        reason="robots_disallowed",
        stage="pre_fetch",
        source="enhanced_scrape",
    )


def target(decision: PolicyDecision | None = None, *, url: str = URL) -> PreflightTarget:
    return PreflightTarget(
        url=url,
        decision=decision or allowed_decision(),
        request_context=RuntimeRequestContext(source="enhanced_scrape", stage="pre_fetch"),
    )


def successful_preflight(analysis: Mapping[str, Any] | None = None) -> PreflightResult:
    return PreflightResult(analysis=analysis or ANALYSIS)


def successful_article(url: str, *, method: str) -> dict[str, Any]:
    return {
        "url": url,
        "content": f"{method} content",
        "method": method,
        "extraction_successful": True,
    }


@dataclass
class EnhancedHarness:
    enhanced: Any
    scraper: Any
    policy_checker: DefaultWebOutboundPolicyChecker
    evaluate_target: AsyncMock
    build_context: Mock
    run_preflight: AsyncMock
    apply_advice: Mock
    public_payload: Mock
    trafilatura: AsyncMock
    playwright: AsyncMock
    beautifulsoup: AsyncMock


def install_enhanced_defaults(
    monkeypatch: pytest.MonkeyPatch,
    *,
    backend: str = "auto",
    preflight: bool = True,
    include_results: bool = True,
    decision: PolicyDecision | None = None,
    preflight_result: PreflightResult | None = None,
) -> EnhancedHarness:
    from tldw_Server_API.app.core.Web_Scraping import enhanced_web_scraping as enhanced

    selected_decision = decision or allowed_decision()
    config = {
        "web_scraper_preflight_analyzers": preflight,
        "web_scraper_preflight_include_results": include_results,
        "web_scraper_preflight_scan_depth": "default",
        "web_scraper_preflight_timeout_s": 0,
        "web_scraper_respect_robots": True,
    }
    plan = SimpleNamespace(
        backend=backend,
        handler="",
        ua_profile="test-profile",
        extra_headers={},
        cookies={},
        respect_robots=True,
        impersonate=None,
        proxies=None,
        strategy_order=None,
        schema_rules=None,
        llm_settings=None,
        regex_settings=None,
        cluster_settings=None,
    )
    scraper = enhanced.EnhancedWebScraper(config=config)
    policy_checker = enhanced._ENHANCED_POLICY_CHECKER
    evaluate = AsyncMock(return_value=target(selected_decision))
    build_context = Mock(return_value=object())
    run = AsyncMock(return_value=preflight_result or successful_preflight())
    apply_advice = Mock(wraps=preflight_facade.apply_preflight_advice)
    public_payload = Mock(wraps=preflight_facade.public_preflight_payload)
    trafilatura = AsyncMock(return_value=successful_article(URL, method="trafilatura"))
    playwright = AsyncMock(return_value=successful_article(URL, method="playwright"))
    beautifulsoup = AsyncMock(return_value=successful_article(URL, method="beautifulsoup"))

    async def deny_legacy_policy(*_args: Any, **_kwargs: Any) -> Any:
        return enhanced.WebOutboundPolicyDecision(
            allowed=False,
            mode="strict",
            reason="deny_legacy_path",
            stage="pre_fetch",
            source="enhanced_scrape",
        )

    monkeypatch.setattr(enhanced, "preflight_facade", preflight_facade, raising=False)
    monkeypatch.setattr(preflight_facade, "evaluate_target", evaluate)
    monkeypatch.setattr(preflight_facade, "build_execution_context", build_context)
    monkeypatch.setattr(preflight_facade, "run_preflight", run)
    monkeypatch.setattr(preflight_facade, "apply_preflight_advice", apply_advice)
    monkeypatch.setattr(preflight_facade, "public_preflight_payload", public_payload)
    monkeypatch.setattr(enhanced, "decide_web_outbound_policy", deny_legacy_policy)
    monkeypatch.setattr(scraper.rate_limiter, "acquire", AsyncMock(return_value=None))
    monkeypatch.setattr(scraper, "_resolve_scrape_plan", lambda _url: (plan, backend, ""))
    monkeypatch.setattr(scraper, "_scrape_with_trafilatura", trafilatura)
    monkeypatch.setattr(scraper, "_scrape_with_playwright", playwright)
    monkeypatch.setattr(scraper, "_scrape_with_beautifulsoup", beautifulsoup)
    monkeypatch.setattr(enhanced, "increment_counter", lambda *_args, **_kwargs: None)
    return EnhancedHarness(
        enhanced=enhanced,
        scraper=scraper,
        policy_checker=policy_checker,
        evaluate_target=evaluate,
        build_context=build_context,
        run_preflight=run,
        apply_advice=apply_advice,
        public_payload=public_payload,
        trafilatura=trafilatura,
        playwright=playwright,
        beautifulsoup=beautifulsoup,
    )


@pytest.mark.unit
async def test_enhanced_start_skips_playwright_when_transport_is_denied(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A denied transport must prevent Playwright startup entirely."""
    from tldw_Server_API.app.core.Web_Scraping import enhanced_web_scraping as enhanced

    scraper = enhanced.EnhancedWebScraper(config={})
    denied = decide_browser_transport(
        configured_mode="disabled",
        auth_mode="single_user",
        outbound_policy_mode="compat",
    )
    playwright_factory = Mock(side_effect=AssertionError("Playwright must not start"))
    monkeypatch.setattr(scraper.job_queue, "start", AsyncMock(return_value=None))
    monkeypatch.setattr(enhanced, "default_browser_transport_decision", lambda: denied, raising=False)
    monkeypatch.setattr(enhanced, "async_playwright", playwright_factory)

    await scraper.start()

    playwright_factory.assert_not_called()
    assert scraper._playwright is None
    assert scraper._browser is None


@pytest.mark.unit
async def test_enhanced_public_scrape_denies_playwright_before_context_creation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The public scrape contract must gate a pre-initialized browser too."""
    harness = install_enhanced_defaults(monkeypatch, backend="playwright")
    denied = decide_browser_transport(
        configured_mode="auto",
        auth_mode="multi_user",
        outbound_policy_mode="strict",
    )
    browser = SimpleNamespace(
        new_context=AsyncMock(side_effect=AssertionError("context must not be created"))
    )
    harness.scraper._browser = browser
    real_playwright_scrape = harness.enhanced.EnhancedWebScraper._scrape_with_playwright.__get__(
        harness.scraper
    )
    monkeypatch.setattr(harness.scraper, "_scrape_with_playwright", real_playwright_scrape)
    monkeypatch.setattr(
        harness.enhanced,
        "default_browser_transport_decision",
        lambda: denied,
        raising=False,
    )

    result = await harness.scraper.scrape_article(URL)

    assert result["error"] == "browser_transport_unavailable"
    assert result["extraction_successful"] is False
    assert result["capability"] == denied.to_capability_metadata()
    assert result["preflight_analysis"]["analysis"] == ANALYSIS
    browser.new_context.assert_not_awaited()


@pytest.mark.unit
async def test_enhanced_denied_target_stops_before_context_preflight_or_extraction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = install_enhanced_defaults(monkeypatch, decision=denied_decision())

    result = await harness.scraper.scrape_article(URL)

    assert result == {
        "url": URL,
        "error": "Blocked by outbound policy",
        "extraction_successful": False,
        "policy_reason": "robots_disallowed",
        "policy_mode": "strict",
        "policy_stage": "pre_fetch",
        "policy_source": "enhanced_scrape",
    }
    harness.evaluate_target.assert_awaited_once()
    harness.build_context.assert_not_called()
    harness.run_preflight.assert_not_awaited()
    harness.trafilatura.assert_not_awaited()
    harness.playwright.assert_not_awaited()
    harness.beautifulsoup.assert_not_awaited()


@pytest.mark.unit
async def test_enhanced_policy_failure_is_structural_and_sanitized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sensitive_url = "https://user:password@example.com/article?token=secret-value"
    harness = install_enhanced_defaults(monkeypatch)
    error = RuntimeError(f"credential leaked for {sensitive_url}")
    harness.evaluate_target.side_effect = error

    async def fail_legacy_policy(*_args: Any, **_kwargs: Any) -> Any:
        raise error

    messages: list[str] = []
    monkeypatch.setattr(harness.enhanced, "decide_web_outbound_policy", fail_legacy_policy)
    monkeypatch.setattr(harness.enhanced.logger, "error", lambda message: messages.append(str(message)))

    result = await harness.scraper.scrape_article(sensitive_url)

    assert result == {
        "url": sensitive_url,
        "error": "Outbound policy evaluation failed. Please contact system administrator.",
        "extraction_successful": False,
    }
    combined_logs = " ".join(messages)
    assert "secret-value" not in combined_logs
    assert "password" not in combined_logs
    assert "credential leaked" not in combined_logs
    assert "?token=" not in combined_logs
    harness.build_context.assert_not_called()
    harness.run_preflight.assert_not_awaited()


@pytest.mark.unit
async def test_enhanced_disabled_still_admits_without_preflight_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = install_enhanced_defaults(monkeypatch, preflight=False)

    result = await harness.scraper.scrape_article(URL)

    assert result == successful_article(URL, method="trafilatura")
    harness.evaluate_target.assert_awaited_once()
    harness.build_context.assert_not_called()
    harness.run_preflight.assert_not_awaited()
    harness.apply_advice.assert_called_once_with(
        None,
        backend="auto",
        method="auto",
        backend_setting="auto",
    )
    harness.public_payload.assert_called_once_with(None, True)


@pytest.mark.unit
async def test_enhanced_preflight_advice_log_excludes_target_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sensitive_url = "https://user:password@example.com/article?token=secret-value"
    contract = PreflightResult(
        analysis={"results": {"tls": {"status": "active"}}},
    )
    harness = install_enhanced_defaults(monkeypatch, preflight_result=contract)
    messages: list[str] = []
    monkeypatch.setattr(harness.enhanced.logger, "debug", lambda message: messages.append(str(message)))

    await harness.scraper.scrape_article(sensitive_url)

    combined_logs = " ".join(messages)
    assert "tls_active" in combined_logs
    assert "secret-value" not in combined_logs
    assert "password" not in combined_logs
    assert "?token=" not in combined_logs
    assert sensitive_url not in combined_logs


@pytest.mark.unit
async def test_enhanced_evaluates_target_once_with_runtime_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = install_enhanced_defaults(monkeypatch)

    await harness.scraper.scrape_article(URL, user_agent="caller-agent")

    harness.evaluate_target.assert_awaited_once()
    call = harness.evaluate_target.await_args
    assert call.args == (URL,)
    assert call.kwargs["respect_robots"] is True
    assert call.kwargs["user_agent"] == "caller-agent"
    assert call.kwargs["policy_checker"] is harness.policy_checker
    assert type(harness.policy_checker) is DefaultWebOutboundPolicyChecker
    assert call.kwargs["config"] == {"web_scraper": harness.scraper.config}
    context = call.kwargs["request_context"]
    assert context.source == "enhanced_scrape"
    assert context.stage == "pre_fetch"
    harness.build_context.assert_called_once()
    assert harness.build_context.call_args.kwargs["policy_checker"] is harness.policy_checker
    harness.run_preflight.assert_awaited_once()


@pytest.mark.unit
@pytest.mark.parametrize(
    ("analysis", "expected_method"),
    [
        ({"results": {"js": {"status": "success", "js_required": True}}}, "playwright"),
        ({"results": {"js": {"status": "error", "js_required": True}}}, "trafilatura"),
        ({"results": {"js": {"status": "success", "js_required": False}}}, "trafilatura"),
    ],
)
async def test_enhanced_js_routing_uses_only_successful_facade_signals(
    monkeypatch: pytest.MonkeyPatch,
    analysis: dict[str, Any],
    expected_method: str,
) -> None:
    harness = install_enhanced_defaults(
        monkeypatch,
        preflight_result=successful_preflight(analysis),
    )

    result = await harness.scraper.scrape_article(URL)

    assert result["method"] == expected_method
    assert result["preflight_analysis"]["advice"]["method"] == (
        "playwright" if expected_method == "playwright" else "auto"
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("backend_setting", "expected_backend"),
    [("auto", "curl"), ("httpx", "httpx")],
)
async def test_enhanced_tls_routing_changes_only_automatic_backend(
    monkeypatch: pytest.MonkeyPatch,
    backend_setting: str,
    expected_backend: str,
) -> None:
    harness = install_enhanced_defaults(
        monkeypatch,
        backend=backend_setting,
        preflight_result=successful_preflight(
            {"results": {"tls": {"status": "active"}}},
        ),
    )

    result = await harness.scraper.scrape_article(URL)

    assert harness.trafilatura.await_args.kwargs["backend"] == expected_backend
    assert result["preflight_analysis"]["advice"]["backend"] == expected_backend


@pytest.mark.unit
async def test_enhanced_preserves_caller_method_through_facade_advice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = install_enhanced_defaults(
        monkeypatch,
        preflight_result=successful_preflight(
            {"results": {"js": {"status": "success", "js_required": True}}},
        ),
    )

    result = await harness.scraper.scrape_article(URL, method="beautifulsoup")

    assert result["method"] == "beautifulsoup"
    assert result["preflight_analysis"]["advice"]["method"] == "beautifulsoup"
    harness.beautifulsoup.assert_awaited_once()
    harness.playwright.assert_not_awaited()


@pytest.mark.unit
@pytest.mark.parametrize("status", [WebScrapingStatus.TIMEOUT, WebScrapingStatus.ERROR])
async def test_enhanced_non_ok_preflight_is_advisory_and_payload_ineligible(
    monkeypatch: pytest.MonkeyPatch,
    status: WebScrapingStatus,
) -> None:
    contract = PreflightResult(
        analysis={"results": {"js": {"status": "success", "js_required": True}}},
        advice=PreflightAdvice(backend="curl", method="playwright"),
        status=status,
        failure=RuntimeFailure(status=status, public_message="safe preflight failure"),
    )
    harness = install_enhanced_defaults(
        monkeypatch,
        backend="httpx",
        preflight_result=contract,
    )

    result = await harness.scraper.scrape_article(URL, method="beautifulsoup")

    assert result["method"] == "beautifulsoup"
    assert "preflight_analysis" not in result
    assert harness.beautifulsoup.await_args.kwargs["backend"] == "httpx"


@pytest.mark.unit
async def test_enhanced_analyzer_scoped_error_retains_ok_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    analysis = {
        "results": {
            "headers": {"status": "error", "message": "probe failed"},
            "js": {"status": "success", "js_required": False},
        }
    }
    harness = install_enhanced_defaults(
        monkeypatch,
        preflight_result=successful_preflight(analysis),
    )

    result = await harness.scraper.scrape_article(URL)

    assert result["preflight_analysis"] == {
        "analysis": analysis,
        "advice": {"backend": "auto", "method": "auto", "notes": []},
    }


@pytest.mark.unit
@pytest.mark.parametrize("method", ["trafilatura", "playwright", "beautifulsoup"])
async def test_enhanced_payload_attaches_to_each_method_success(
    monkeypatch: pytest.MonkeyPatch,
    method: str,
) -> None:
    harness = install_enhanced_defaults(monkeypatch)

    result = await harness.scraper.scrape_article(URL, method=method)

    assert result["method"] == method
    assert result["preflight_analysis"] == {
        "analysis": ANALYSIS,
        "advice": {"backend": "auto", "method": method, "notes": []},
    }


@pytest.mark.unit
async def test_enhanced_payload_attaches_to_unknown_method_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = install_enhanced_defaults(monkeypatch)

    result = await harness.scraper.scrape_article(URL, method="unknown")

    assert result == {
        "url": URL,
        "error": "Unknown scraping method: unknown",
        "extraction_successful": False,
        "preflight_analysis": {
            "analysis": ANALYSIS,
            "advice": {"backend": "auto", "method": "unknown", "notes": []},
        },
    }


@pytest.mark.unit
async def test_enhanced_evaluate_target_cancellation_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = install_enhanced_defaults(monkeypatch)
    harness.evaluate_target.side_effect = asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await harness.scraper.scrape_article(URL)

    harness.build_context.assert_not_called()
    harness.run_preflight.assert_not_awaited()


@pytest.mark.unit
async def test_enhanced_outer_handler_does_not_swallow_preflight_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = install_enhanced_defaults(monkeypatch)
    harness.run_preflight.side_effect = asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await harness.scraper.scrape_article(URL)


def test_enhanced_public_signature_is_unchanged() -> None:
    from tldw_Server_API.app.core.Web_Scraping import enhanced_web_scraping as enhanced

    signature = inspect.signature(enhanced.EnhancedWebScraper.scrape_article)
    assert list(signature.parameters) == [
        "self",
        "url",
        "method",
        "custom_cookies",
        "user_agent",
        "custom_headers",
        "allow_llm_extraction",
    ]
    assert signature.parameters["method"].default == "auto"
    assert signature.parameters["custom_cookies"].default is None
    assert signature.parameters["user_agent"].default is None
    assert signature.parameters["custom_headers"].default is None
    assert signature.parameters["allow_llm_extraction"].default is True
    assert get_type_hints(enhanced.EnhancedWebScraper.scrape_article) == {
        "url": str,
        "method": str,
        "custom_cookies": list[dict[str, Any]] | None,
        "user_agent": str | None,
        "custom_headers": dict[str, str] | None,
        "allow_llm_extraction": bool,
        "return": dict[str, Any],
    }


def test_enhanced_consumer_uses_package_facade_without_duplicate_helpers() -> None:
    from tldw_Server_API.app.core.Web_Scraping import enhanced_web_scraping as enhanced

    module_source = inspect.getsource(enhanced)
    scrape_source = inspect.getsource(enhanced.EnhancedWebScraper.scrape_article)
    assert not hasattr(enhanced.EnhancedWebScraper, "_run_preflight_analysis")
    assert not hasattr(enhanced.EnhancedWebScraper, "_apply_preflight_advice")
    assert module_source.count("Web_Scraping import preflight as preflight_facade") == 1
    assert "decide_web_outbound_policy(" not in scrape_source


def test_enhanced_policy_checker_is_exact_default_adapter() -> None:
    from tldw_Server_API.app.core.Web_Scraping import enhanced_web_scraping as enhanced

    assert type(enhanced._ENHANCED_POLICY_CHECKER) is DefaultWebOutboundPolicyChecker
