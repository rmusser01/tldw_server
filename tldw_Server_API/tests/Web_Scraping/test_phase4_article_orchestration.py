from __future__ import annotations

import ast
import asyncio
import dataclasses
import inspect
import threading
from collections.abc import Mapping, Sequence
from dataclasses import FrozenInstanceError, dataclass
from typing import Any
from unittest.mock import AsyncMock, Mock, call

import pytest

from tldw_Server_API.app.core.Web_Scraping import preflight as preflight_facade
from tldw_Server_API.app.core.Web_Scraping.contracts import PreflightResult
from tldw_Server_API.app.core.Web_Scraping.orchestration.article_models import (
    ArticleFailure,
    ArticleLimits,
    ArticlePlan,
    DirectBrowserProfile,
)
from tldw_Server_API.app.core.Web_Scraping.preflight import PreflightOptions, PreflightTarget
from tldw_Server_API.app.core.Web_Scraping.runtime import (
    FetchRequest,
    FetchResponse,
    PolicyDecision,
    RuntimeRequestContext,
)

URL = "https://example.com/article"


class FakeFetchClient:
    def __init__(self, outcomes: Sequence[FetchResponse | BaseException]) -> None:
        self.outcomes = list(outcomes)
        self.requests: list[FetchRequest] = []

    def fetch(self, request: FetchRequest) -> FetchResponse:
        self.requests.append(request)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class FakeBrowser:
    def __init__(self, outcomes: Sequence[str | BaseException]) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[tuple[str, DirectBrowserProfile, ArticleLimits]] = []

    async def acquire(
        self,
        url: str,
        profile: DirectBrowserProfile,
        limits: ArticleLimits,
    ) -> str:
        self.calls.append((url, profile, limits))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class FakeExecutor:
    def __init__(self) -> None:
        self.calls: list[tuple[Any, tuple[Any, ...], dict[str, Any]]] = []

    async def run(self, func: Any, /, *args: Any, **kwargs: Any) -> Any:
        self.calls.append((func, args, kwargs))
        return func(*args, **kwargs)


def _allowed_target(url: str = URL) -> PreflightTarget:
    return PreflightTarget(
        url=url,
        decision=PolicyDecision(
            allowed=True,
            mode="test",
            reason="allowed",
            stage="pre_fetch",
            source="article_extract",
        ),
        request_context=RuntimeRequestContext(source="article_extract", stage="pre_fetch"),
    )


def _plan(*, backend: str = "httpx", handler: str = "") -> ArticlePlan:
    return ArticlePlan(
        url=URL,
        domain="example.com",
        backend=backend,
        handler=handler,
        browser=DirectBrowserProfile(
            user_agent="test-agent",
            custom_cookies=(),
            retries=1,
            timeout_ms=1000,
            stealth_enabled=False,
            stealth_wait_ms=0,
        ),
        limits=ArticleLimits(max_article_bytes=123, max_browser_transfer_bytes=456),
    )


def _article(*, successful: bool = True, content: str = "article body") -> dict[str, Any]:
    return {
        "url": URL,
        "title": "Article",
        "author": "Author",
        "date": "2026-08-12",
        "content": content if successful else "",
        "extraction_successful": successful,
        **({} if successful else {"error": "No content extracted"}),
    }


@dataclass
class Harness:
    dependencies: Any
    fetch: FakeFetchClient
    browser: FakeBrowser
    executor: FakeExecutor
    evaluate_target: AsyncMock
    run_preflight: AsyncMock
    apply_advice: Mock
    payload: Mock
    extract: Mock
    metrics: list[tuple[str, dict[str, str]]]
    logs: list[dict[str, str]]


def _harness(
    *,
    backend: str = "httpx",
    config: Mapping[str, Any] | None = None,
    fetch_outcomes: Sequence[FetchResponse | BaseException] | None = None,
    browser_outcomes: Sequence[str | BaseException] | None = None,
    extract_results: Sequence[dict[str, Any] | BaseException] | None = None,
    target: PreflightTarget | BaseException | None = None,
    preflight: PreflightResult | BaseException | None = None,
) -> Harness:
    from tldw_Server_API.app.core.Web_Scraping.orchestration.article import ArticleDependencies

    selected_config = config or {
        "web_scraper": {
            "web_scraper_preflight_analyzers": True,
            "web_scraper_preflight_include_results": True,
        }
    }
    fetch = FakeFetchClient(
        fetch_outcomes
        or [
            FetchResponse(
                url=URL,
                status=200,
                text="<html><body>article</body></html>",
                headers={},
                backend="httpx",
            )
        ]
    )
    browser = FakeBrowser(browser_outcomes or ["<html><body>rendered</body></html>"])
    executor = FakeExecutor()
    evaluate_target = AsyncMock(return_value=target or _allowed_target())
    if isinstance(target, BaseException):
        evaluate_target.side_effect = target
    run_preflight = AsyncMock(return_value=preflight or PreflightResult(analysis={"results": {}}))
    if isinstance(preflight, BaseException):
        run_preflight.side_effect = preflight
    apply_advice = Mock(side_effect=lambda result, **kwargs: (kwargs["backend"], kwargs["method"], result))
    payload = Mock(side_effect=lambda result, include: {"analysis": result.analysis} if include and result else None)
    extract = Mock(side_effect=list(extract_results or [_article()]))
    metrics: list[tuple[str, dict[str, str]]] = []
    logs: list[dict[str, str]] = []

    def record_metric(
        name: str,
        *_args: Any,
        labels: Mapping[str, str] | None = None,
        **_kwargs: Any,
    ) -> None:
        metrics.append((name, dict(labels or {})))

    dependencies = ArticleDependencies(
        load_config=lambda: selected_config,
        resolve_plan=lambda _url, _config: _plan(backend=backend),
        evaluate_target=evaluate_target,
        run_preflight=run_preflight,
        apply_preflight_advice=apply_advice,
        fetch_client=fetch,
        browser=browser,
        executor=executor,
        extract=extract,
        build_preflight_context=lambda *_args, **_kwargs: object(),
        preflight_options=lambda values: PreflightOptions.from_mapping(values),
        public_preflight_payload=payload,
        resolve_handler=lambda _path: None,
        js_required=lambda *_args, **_kwargs: False,
        convert_content=lambda content: f"converted:{content}",
        increment_counter=record_metric,
        observe_histogram=record_metric,
        clock=lambda: 0.0,
        log=lambda _message, **fields: logs.append(fields),
    )
    return Harness(
        dependencies=dependencies,
        fetch=fetch,
        browser=browser,
        executor=executor,
        evaluate_target=evaluate_target,
        run_preflight=run_preflight,
        apply_advice=apply_advice,
        payload=payload,
        extract=extract,
        metrics=metrics,
        logs=logs,
    )


@pytest.mark.asyncio
async def test_runner_preserves_config_snapshot_when_plan_resolution_falls_back() -> None:
    from tldw_Server_API.app.core.Web_Scraping.orchestration.article import _run_article

    harness = _harness(
        config={
            "web_scraper": {
                "web_scraper_preflight_analyzers": True,
                "web_scraper_preflight_include_results": True,
                "web_scraper_max_article_bytes": 321,
                "web_scraper_max_browser_transfer_bytes": 654,
                "web_scraper_retry_count": 2,
                "web_scraper_retry_timeout": 3,
                "web_scraper_stealth_playwright": True,
                "web_scraper_stealth_wait_ms": 7,
            }
        },
        fetch_outcomes=[FetchResponse(URL, 204, {}, "", "httpx")],
    )
    plan_calls = 0

    def resolve_plan(_url: str, _config: Mapping[str, Any]) -> ArticlePlan:
        nonlocal plan_calls
        plan_calls += 1
        raise RuntimeError("rules unavailable")

    harness.dependencies = dataclasses.replace(harness.dependencies, resolve_plan=resolve_plan)
    result = await _run_article(URL, None, True, dependencies=harness.dependencies)

    assert result["extraction_successful"] is True
    assert plan_calls == 1
    harness.run_preflight.assert_awaited_once()
    assert harness.fetch.requests[0].max_response_bytes == 321
    assert harness.fetch.requests[0].context == RuntimeRequestContext(source="article_extract", stage="fetch")
    profile = harness.browser.calls[0][1]
    assert profile.retries == 2
    assert profile.timeout_ms == 3_000
    assert profile.stealth_enabled is True
    assert profile.stealth_wait_ms == 7
    assert harness.browser.calls[0][2] == ArticleLimits(321, 654)
    assert result["preflight_analysis"] == {"analysis": {"results": {}}}


@pytest.mark.asyncio
async def test_runner_handles_malformed_url_when_plan_falls_back_before_policy() -> None:
    from tldw_Server_API.app.core.Web_Scraping.orchestration.article import _run_article

    malformed_url = "http://[::1"
    denied = PreflightTarget(
        url=malformed_url,
        decision=PolicyDecision(False, "strict", "robots_disallowed", "pre_fetch", "article_extract"),
        request_context=RuntimeRequestContext(source="article_extract", stage="pre_fetch"),
    )
    harness = _harness(target=denied)

    def resolve_plan(_url: str, _config: Mapping[str, Any]) -> ArticlePlan:
        raise RuntimeError("rules unavailable")

    harness.dependencies = dataclasses.replace(harness.dependencies, resolve_plan=resolve_plan)
    result = await _run_article(malformed_url, None, True, dependencies=harness.dependencies)

    assert result["policy_reason"] == "robots_disallowed"
    assert result["error"] == "Blocked by outbound policy"
    assert harness.fetch.requests == []


@pytest.mark.asyncio
async def test_runner_denies_policy_before_preflight_fetch_browser_or_extraction() -> None:
    from tldw_Server_API.app.core.Web_Scraping.orchestration.article import _run_article

    denied = PreflightTarget(
        url=URL,
        decision=PolicyDecision(False, "strict", "robots_disallowed", "pre_fetch", "article_extract"),
        request_context=RuntimeRequestContext(source="article_extract", stage="pre_fetch"),
    )
    harness = _harness(target=denied)
    result = await _run_article(URL, None, True, dependencies=harness.dependencies)

    assert result["policy_reason"] == "robots_disallowed"
    harness.run_preflight.assert_not_awaited()
    assert harness.fetch.requests == []
    assert harness.browser.calls == []
    assert harness.executor.calls == []
    assert ("scrape_blocked_by_robots_total", {}) in harness.metrics


@pytest.mark.asyncio
async def test_runner_maps_policy_failure_without_sensitive_error_text() -> None:
    from tldw_Server_API.app.core.Web_Scraping.orchestration.article import _run_article

    harness = _harness(target=RuntimeError("https://user:secret@example.com/?token=bad"))
    result = await _run_article(URL, None, True, dependencies=harness.dependencies)

    assert result["error"] == "policy_error"
    assert "secret" not in str(result)
    assert harness.fetch.requests == []


@pytest.mark.asyncio
async def test_runner_skips_disabled_preflight_and_preserves_success_payload_rules() -> None:
    from tldw_Server_API.app.core.Web_Scraping.orchestration.article import _run_article

    harness = _harness(config={"web_scraper": {"web_scraper_preflight_analyzers": False}})
    result = await _run_article(URL, None, True, dependencies=harness.dependencies)

    harness.run_preflight.assert_not_awaited()
    assert "preflight_analysis" not in result


@pytest.mark.asyncio
async def test_runner_uses_facade_advice_for_automatic_route() -> None:
    from tldw_Server_API.app.core.Web_Scraping.orchestration.article import _run_article

    successful = PreflightResult(
        analysis={"results": {"js": {"status": "success", "js_required": True}, "tls": {"status": "active"}}}
    )
    harness = _harness(backend="auto", preflight=successful)
    harness.dependencies = dataclasses.replace(
        harness.dependencies,
        apply_preflight_advice=preflight_facade.apply_preflight_advice,
        public_preflight_payload=preflight_facade.public_preflight_payload,
        backend_setting=lambda _plan: "auto",
    )
    result = await _run_article(URL, None, True, dependencies=harness.dependencies)

    assert harness.fetch.requests == []
    assert len(harness.browser.calls) == 1
    assert result["preflight_analysis"]["advice"] == {
        "backend": "curl",
        "method": "playwright",
        "notes": ["js_required", "tls_active"],
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("route_backend", "configured_backend"),
    [("httpx", "curl"), ("auto", "httpx")],
)
async def test_runner_preserves_explicit_route_or_config_backend_over_tls_advice(
    route_backend: str,
    configured_backend: str,
) -> None:
    from tldw_Server_API.app.core.Web_Scraping.orchestration.article import _run_article

    harness = _harness(
        backend="httpx",
        config={
            "web_scraper": {
                "web_scraper_preflight_analyzers": True,
                "web_scraper_preflight_include_results": True,
                "web_scraper_default_backend": configured_backend,
            }
        },
        preflight=PreflightResult(analysis={"results": {"tls": {"status": "active"}}}),
    )
    harness.dependencies = dataclasses.replace(
        harness.dependencies,
        apply_preflight_advice=preflight_facade.apply_preflight_advice,
        public_preflight_payload=preflight_facade.public_preflight_payload,
        backend_setting=lambda _plan: route_backend,
    )
    result = await _run_article(URL, None, True, dependencies=harness.dependencies)

    assert harness.fetch.requests[0].backend == "httpx"
    assert result["preflight_analysis"]["advice"]["backend"] == "httpx"


@pytest.mark.asyncio
async def test_runner_snapshots_nested_preflight_config_before_policy_admission() -> None:
    from tldw_Server_API.app.core.Web_Scraping.orchestration.article import _run_article

    source = {"web_scraper": {"web_scraper_preflight_analyzers": False, "nested": {"value": "stable"}}}
    harness = _harness(config=source)

    async def evaluate_target(*_args: Any, **kwargs: Any) -> PreflightTarget:
        source["web_scraper"]["nested"]["value"] = "mutated"
        assert kwargs["config"]["web_scraper"]["nested"]["value"] == "stable"
        return _allowed_target()

    harness.dependencies = dataclasses.replace(harness.dependencies, evaluate_target=evaluate_target)
    result = await _run_article(URL, None, True, dependencies=harness.dependencies)

    assert result["extraction_successful"] is True


@pytest.mark.asyncio
async def test_runner_fails_open_after_preflight_operational_error() -> None:
    from tldw_Server_API.app.core.Web_Scraping.orchestration.article import _run_article

    harness = _harness(preflight=RuntimeError("preflight unavailable"))
    result = await _run_article(URL, None, True, dependencies=harness.dependencies)

    assert result["extraction_successful"] is True
    assert harness.fetch.requests[0].backend == "httpx"
    assert "preflight_analysis" not in result


@pytest.mark.asyncio
async def test_runner_attaches_copied_preflight_payload_to_final_extraction_failure() -> None:
    from tldw_Server_API.app.core.Web_Scraping.orchestration.article import _run_article

    shared = _article(successful=False)
    harness = _harness(extract_results=[shared, shared])
    result = await _run_article(URL, None, True, dependencies=harness.dependencies)

    assert result["extraction_successful"] is False
    assert result["preflight_analysis"] == {"analysis": {"results": {}}}
    assert result is not shared
    assert "preflight_analysis" not in shared


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fetch_outcome",
    [
        FetchResponse(url=URL, status=204, text="", headers={}, backend="httpx"),
        RuntimeError("http failure"),
    ],
)
async def test_runner_falls_back_to_browser_after_nonextractable_or_failed_http(fetch_outcome: Any) -> None:
    from tldw_Server_API.app.core.Web_Scraping.orchestration.article import _run_article

    harness = _harness(fetch_outcomes=[fetch_outcome])
    result = await _run_article(URL, None, True, dependencies=harness.dependencies)

    assert result["extraction_successful"] is True
    assert len(harness.browser.calls) == 1
    assert ("scrape_fetch_latency_seconds", {"backend": "playwright"}) in harness.metrics
    assert ("scrape_fetch_total", {"backend": "playwright", "outcome": "success"}) in harness.metrics


@pytest.mark.asyncio
async def test_runner_maps_lightweight_extraction_error_before_browser_recovery() -> None:
    from tldw_Server_API.app.core.Web_Scraping.orchestration.article import _run_article

    harness = _harness(extract_results=[RuntimeError("private extraction detail"), _article()])
    result = await _run_article(URL, None, True, dependencies=harness.dependencies)

    assert result["extraction_successful"] is True
    assert harness.logs[0] == {
        "exception_type": "RuntimeError",
        "code": "extraction_error",
        "stage": "extract",
        "host": "example.com",
    }
    assert ("scrape_fetch_total", {"backend": "httpx", "outcome": "error"}) in harness.metrics
    assert ("scrape_playwright_fallback_total", {"reason": "error"}) in harness.metrics


@pytest.mark.asyncio
async def test_runner_logs_sanitized_fetch_failure_before_browser_recovery() -> None:
    from tldw_Server_API.app.core.Web_Scraping.orchestration.article import _run_article

    harness = _harness(fetch_outcomes=[RuntimeError("https://user:secret@example.com/?token=bad")])
    result = await _run_article(URL, None, True, dependencies=harness.dependencies)

    assert result["extraction_successful"] is True
    assert harness.logs == [
        {
            "exception_type": "RuntimeError",
            "code": "fetch_error",
            "stage": "fetch",
            "host": "example.com",
        }
    ]


@pytest.mark.asyncio
async def test_runner_maps_browser_and_extraction_boundaries_to_stable_errors() -> None:
    from tldw_Server_API.app.core.Web_Scraping.orchestration.article import _run_article

    browser_harness = _harness(backend="playwright", browser_outcomes=[RuntimeError("token=secret")])
    browser_result = await _run_article(URL, None, True, dependencies=browser_harness.dependencies)

    assert browser_result["error"] == "browser_error"
    assert "capability" not in browser_result
    assert browser_harness.logs == [
        {
            "exception_type": "RuntimeError",
            "code": "browser_error",
            "stage": "acquire",
            "host": "example.com",
        }
    ]
    assert ("scrape_fetch_latency_seconds", {"backend": "playwright"}) in browser_harness.metrics
    assert ("scrape_fetch_total", {"backend": "playwright", "outcome": "error"}) in browser_harness.metrics

    extraction_harness = _harness(extract_results=[RuntimeError("/private/secret"), RuntimeError("/private/secret")])
    extraction_result = await _run_article(URL, None, True, dependencies=extraction_harness.dependencies)

    assert extraction_result["error"] == "extraction_error"
    assert extraction_harness.logs[-1] == {
        "exception_type": "RuntimeError",
        "code": "extraction_error",
        "stage": "extract",
        "host": "example.com",
    }


@pytest.mark.asyncio
@pytest.mark.unit
async def test_runner_preserves_exact_browser_transport_denial_capability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Expose only bounded transport-denial capability through the public API."""
    from tldw_Server_API.app.core.Web_Scraping.orchestration import article as canonical

    failure = ArticleFailure(
        "browser_transport_unavailable",
        "browser_transport_unattested",
        capability={
            "name": "safe_browser_transport",
            "available": False,
            "configured_mode": "auto",
            "effective_mode": "disabled",
            "dns_peer_attested": False,
            "reason": "browser_transport_unattested",
        },
    )
    harness = _harness(backend="playwright", browser_outcomes=[failure])
    monkeypatch.setattr(
        canonical,
        "_build_default_dependencies",
        lambda _cookies: harness.dependencies,
    )

    result = await canonical.scrape_article(URL)

    assert result == {
        "url": URL,
        "title": "N/A",
        "author": "N/A",
        "date": "N/A",
        "content": "",
        "extraction_successful": False,
        "error": "browser_transport_unavailable",
        "capability": {
            "name": "safe_browser_transport",
            "available": False,
            "configured_mode": "auto",
            "effective_mode": "disabled",
            "dns_peer_attested": False,
            "reason": "browser_transport_unattested",
        },
        "preflight_analysis": {"analysis": {"results": {}}},
    }


@pytest.mark.asyncio
@pytest.mark.unit
async def test_strict_profile_http_success_never_consults_browser(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep successful governed HTTP retrieval independent of browser policy."""
    from tldw_Server_API.app.core.Web_Scraping.orchestration import article as canonical

    harness = _harness(
        config={
            "web_scraper": {
                "web_outbound_policy_mode": "strict",
                "web_browser_transport_mode": "disabled",
            }
        },
        browser_outcomes=[AssertionError("browser must not be called")],
    )
    monkeypatch.setattr(
        canonical,
        "_build_default_dependencies",
        lambda _cookies: harness.dependencies,
    )

    result = await canonical.scrape_article(URL)

    assert result["extraction_successful"] is True
    assert harness.browser.calls == []


@pytest.mark.unit
def test_raw_browser_transport_denial_retains_only_bounded_capability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bound raw browser denial metadata at the public synchronous contract."""
    from tldw_Server_API.app.core.Web_Scraping.orchestration import article as canonical

    failure = ArticleFailure(
        "browser_transport_unavailable",
        "browser_transport_unattested",
        capability={
            "name": "safe_browser_transport",
            "available": False,
            "configured_mode": "auto",
            "effective_mode": "disabled",
            "dns_peer_attested": False,
            "reason": "browser_transport_unattested",
        },
    )
    harness = _harness(backend="playwright", browser_outcomes=[failure])
    monkeypatch.setattr(
        canonical,
        "_build_default_dependencies",
        lambda _cookies: harness.dependencies,
    )

    assert canonical.scrape_article_sync(URL) == {
        "url": URL,
        "extraction_successful": False,
        "error": "browser_transport_unavailable",
        "capability": {
            "name": "safe_browser_transport",
            "available": False,
            "configured_mode": "auto",
            "effective_mode": "disabled",
            "dns_peer_attested": False,
            "reason": "browser_transport_unattested",
        },
    }


@pytest.mark.asyncio
@pytest.mark.unit
async def test_runner_falls_back_after_js_required_and_records_bounded_metric(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Record one bounded reason when the public API escalates for JavaScript."""
    from tldw_Server_API.app.core.Web_Scraping.orchestration import article as canonical

    harness = _harness()
    harness.dependencies = dataclasses.replace(harness.dependencies, js_required=lambda *_args, **_kwargs: True)
    monkeypatch.setattr(
        canonical,
        "_build_default_dependencies",
        lambda _cookies: harness.dependencies,
    )
    result = await canonical.scrape_article(URL)

    assert result["extraction_successful"] is True
    assert ("scrape_playwright_fallback_total", {"reason": "js_required"}) in harness.metrics
    assert ("scrape_playwright_fallback_total", {"reason": "no_extract"}) not in harness.metrics
    assert len(harness.browser.calls) == 1


def test_log_failure_normalizes_untrusted_article_fields() -> None:
    from tldw_Server_API.app.core.Web_Scraping.orchestration.article import _log_failure

    harness = _harness()
    _log_failure(
        harness.dependencies,
        RuntimeError("token=secret"),
        code="../../hostile-code",
        stage="untrusted-stage?token=secret",
        url="https://user:secret@example.com/private",
    )

    assert harness.logs == [
        {
            "exception_type": "RuntimeError",
            "code": "extraction_error",
            "stage": "article",
            "host": "example.com",
        }
    ]


def test_log_failure_preserves_complete_guarded_browser_stage_contract() -> None:
    from tldw_Server_API.app.core.Web_Scraping.orchestration import article as canonical
    from tldw_Server_API.app.core.Web_Scraping.orchestration import article_browser

    expected_stages = {
        "browser_transfer",
        "callback",
        "callback_drain",
        "capability",
        "capacity",
        "cleanup",
        "content",
        "context",
        "egress",
        "http_route",
        "launch",
        "navigation",
        "page",
        "rendered_html",
        "routing",
        "stealth",
        "wait",
        "websocket_route",
    }
    tree = ast.parse(inspect.getsource(article_browser))
    observed_stages: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in {"stage", "failure_stage"} for target in node.targets
        ):
            observed_stages.update(
                value.value
                for value in ast.walk(node.value)
                if isinstance(value, ast.Constant) and isinstance(value.value, str)
            )
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id == "ArticleFailure" and len(node.args) >= 2:
                stage = node.args[1]
                if isinstance(stage, ast.Constant) and isinstance(stage.value, str):
                    observed_stages.add(stage.value)
            if isinstance(node.func, ast.Attribute) and node.func.attr == "fail" and node.args:
                stage = node.args[0]
                if isinstance(stage, ast.Constant) and isinstance(stage.value, str):
                    observed_stages.add(stage.value)
            for keyword in node.keywords:
                if (
                    keyword.arg == "failure_stage"
                    and isinstance(keyword.value, ast.Constant)
                    and isinstance(keyword.value.value, str)
                ):
                    observed_stages.add(keyword.value.value)

    assert observed_stages == expected_stages
    assert expected_stages <= canonical._ARTICLE_LOG_STAGES

    harness = _harness()
    for stage in sorted(expected_stages):
        canonical._log_failure(
            harness.dependencies,
            RuntimeError("private browser detail"),
            code="browser_error",
            stage=stage,
            url=URL,
        )

    assert [entry["stage"] for entry in harness.logs] == sorted(expected_stages)


@pytest.mark.unit
def test_observability_boundary_failures_emit_sanitized_fallback_warnings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep best-effort metrics and logging failures diagnosable without leaking details."""
    from tldw_Server_API.app.core.Web_Scraping.orchestration import article as canonical

    fallback_logger = Mock()
    harness = _harness()

    def fail_observability(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("token=private")

    dependencies = dataclasses.replace(
        harness.dependencies,
        increment_counter=fail_observability,
        observe_histogram=fail_observability,
        log=fail_observability,
    )
    monkeypatch.setattr(canonical, "logger", fallback_logger)

    canonical._record_counter(dependencies, "counter", {"private": "secret"})
    canonical._record_histogram(dependencies, "histogram", 1.0, {"private": "secret"})
    canonical._log_failure(
        dependencies,
        RuntimeError("token=private"),
        code="browser_error",
        stage="acquire",
        url="https://user:secret@example.com/private",
    )

    assert fallback_logger.warning.call_args_list == [
        call(
            "Article metric recording failed.",
            metric_type="counter",
            exception_type="RuntimeError",
        ),
        call(
            "Article metric recording failed.",
            metric_type="histogram",
            exception_type="RuntimeError",
        ),
        call(
            "Article failure logging failed.",
            code="browser_error",
            stage="acquire",
            host="example.com",
            exception_type="RuntimeError",
        ),
    ]


@pytest.mark.unit
def test_observability_fallback_logger_cannot_change_article_outcomes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep metrics best-effort even when both observability layers fail."""
    from tldw_Server_API.app.core.Web_Scraping.orchestration import article as canonical

    fallback_logger = Mock()
    fallback_logger.warning.side_effect = RuntimeError("logger unavailable")

    def fail_metric(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("metrics unavailable")

    harness = _harness()
    dependencies = dataclasses.replace(harness.dependencies, increment_counter=fail_metric)
    monkeypatch.setattr(canonical, "logger", fallback_logger)

    canonical._record_counter(dependencies, "counter", {})


@pytest.mark.asyncio
async def test_runner_preserves_curl_to_httpx_fallback() -> None:
    from tldw_Server_API.app.core.Web_Scraping.orchestration.article import _run_article

    harness = _harness(
        backend="curl", fetch_outcomes=[RuntimeError("curl unavailable"), FetchResponse(URL, 200, {}, "ok", "httpx")]
    )
    result = await _run_article(URL, None, True, dependencies=harness.dependencies)

    assert result["extraction_successful"] is True
    assert [request.backend for request in harness.fetch.requests] == ["curl", "httpx"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_lightweight_redirect_rechecks_policy_before_following() -> None:
    """Reject a redirect destination before issuing a request or falling back to a browser."""
    from tldw_Server_API.app.core.Web_Scraping.orchestration.article import _run_article

    redirect_url = "https://blocked.example/private"
    denied_target = PreflightTarget(
        url=redirect_url,
        decision=PolicyDecision(
            allowed=False,
            mode="enforce",
            reason="robots_disallowed",
            stage="redirect",
            source="article_extract",
        ),
        request_context=RuntimeRequestContext(source="article_extract", stage="redirect"),
    )
    harness = _harness(
        fetch_outcomes=[
            FetchResponse(
                url=URL,
                status=302,
                headers={"Location": redirect_url},
                text="",
                backend="httpx",
            )
        ]
    )
    harness.evaluate_target.side_effect = [_allowed_target(), denied_target]

    result = await _run_article(URL, None, True, dependencies=harness.dependencies)

    assert result["extraction_successful"] is False
    assert result["policy_reason"] == "robots_disallowed"
    assert len(harness.fetch.requests) == 1
    assert harness.fetch.requests[0].allow_redirects is False
    assert harness.evaluate_target.await_args_list[1].args[0] == redirect_url
    assert harness.evaluate_target.await_args_list[1].kwargs["request_context"].stage == "redirect"
    assert harness.browser.calls == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_curl_redirect_policy_denial_is_not_retried_with_httpx() -> None:
    """Treat policy denial as terminal instead of a curl transport failure."""
    from tldw_Server_API.app.core.Web_Scraping.orchestration.article import _run_article

    redirect_url = "https://blocked.example/private"
    denied_target = PreflightTarget(
        url=redirect_url,
        decision=PolicyDecision(
            allowed=False,
            mode="enforce",
            reason="robots_disallowed",
            stage="redirect",
            source="article_extract",
        ),
        request_context=RuntimeRequestContext(source="article_extract", stage="redirect"),
    )
    harness = _harness(
        backend="curl",
        fetch_outcomes=[FetchResponse(URL, 302, {"Location": redirect_url}, "", "curl")],
    )
    harness.evaluate_target.side_effect = [_allowed_target(), denied_target]

    result = await _run_article(URL, None, True, dependencies=harness.dependencies)

    assert result["policy_reason"] == "robots_disallowed"
    assert [request.backend for request in harness.fetch.requests] == ["curl"]
    assert harness.browser.calls == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_lightweight_redirect_loop_is_terminal_before_browser_fallback() -> None:
    """Do not bypass per-hop redirect governance by switching acquisition modes."""
    from tldw_Server_API.app.core.Web_Scraping.orchestration.article import _run_article

    harness = _harness(
        fetch_outcomes=[FetchResponse(URL, 302, {"Location": URL}, "", "httpx")],
    )

    result = await _run_article(URL, None, True, dependencies=harness.dependencies)

    assert result["error"] == "fetch_error"
    assert len(harness.fetch.requests) == 1
    assert harness.browser.calls == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_lightweight_cross_origin_redirect_strips_credentials() -> None:
    """Follow admitted redirects without forwarding origin-bound headers or cookies."""
    from tldw_Server_API.app.core.Web_Scraping.orchestration.article import _run_article

    redirect_url = "https://other.example/final"
    harness = _harness(
        fetch_outcomes=[
            FetchResponse(
                url=URL,
                status=302,
                headers={"location": redirect_url},
                text="",
                backend="httpx",
            ),
            FetchResponse(
                url=redirect_url,
                status=200,
                headers={},
                text="<html><body>redirected article</body></html>",
                backend="httpx",
            ),
        ]
    )
    redirect_plan = dataclasses.replace(
        _plan(),
        headers={
            "Authorization": "Bearer private",
            "User-Agent": "redirect-test-agent",
            "Accept-Encoding": "identity",
            "X-Private": "secret",
        },
    )
    harness.dependencies = dataclasses.replace(
        harness.dependencies,
        resolve_plan=lambda _url, _config: redirect_plan,
    )
    harness.evaluate_target.side_effect = [_allowed_target(), _allowed_target(redirect_url)]

    result = await _run_article(
        URL,
        [{"name": "session", "value": "private"}],
        True,
        dependencies=harness.dependencies,
    )

    assert result["extraction_successful"] is True
    assert [request.url for request in harness.fetch.requests] == [URL, redirect_url]
    assert [request.allow_redirects for request in harness.fetch.requests] == [False, False]
    assert dict(harness.fetch.requests[1].headers) == {
        "User-Agent": "redirect-test-agent",
        "Accept-Encoding": "identity",
    }
    assert dict(harness.fetch.requests[1].cookies) == {}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("backend", "outcomes", "expected_backends"),
    [
        ("httpx", [ValueError("Response exceeds max_response_bytes limit")], ["httpx"]),
        ("curl", [ValueError("Response exceeds max_response_bytes limit")], ["curl"]),
        (
            "curl",
            [RuntimeError("curl unavailable"), ValueError("Response exceeds max_response_bytes limit")],
            ["curl", "httpx"],
        ),
    ],
)
async def test_runner_translates_trusted_http_overflow_without_fallback(
    backend: str,
    outcomes: Sequence[BaseException],
    expected_backends: list[str],
) -> None:
    from tldw_Server_API.app.core.Web_Scraping.orchestration.article import _run_article

    harness = _harness(backend=backend, fetch_outcomes=outcomes)
    result = await _run_article(URL, None, True, dependencies=harness.dependencies)

    assert result["error"] == "response_too_large"
    assert [request.backend for request in harness.fetch.requests] == expected_backends
    assert harness.browser.calls == []
    assert harness.executor.calls == []


@pytest.mark.asyncio
async def test_runner_does_not_classify_untrusted_value_error_as_oversize() -> None:
    from tldw_Server_API.app.core.Web_Scraping.orchestration.article import _run_article

    harness = _harness(fetch_outcomes=[ValueError("bad user input")])
    result = await _run_article(URL, None, True, dependencies=harness.dependencies)

    assert result["extraction_successful"] is True
    assert len(harness.browser.calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    [
        ArticleFailure("response_too_large", "fetch"),
        ArticleFailure("response_too_large", "browser_transfer"),
        ArticleFailure("response_too_large", "rendered_html"),
    ],
)
async def test_runner_never_falls_back_after_an_oversized_acquisition(failure: ArticleFailure) -> None:
    from tldw_Server_API.app.core.Web_Scraping.orchestration.article import _run_article

    if failure.stage == "fetch":
        harness = _harness(fetch_outcomes=[failure])
    else:
        harness = _harness(fetch_outcomes=[FetchResponse(URL, 204, {}, "", "httpx")], browser_outcomes=[failure])
    result = await _run_article(URL, None, True, dependencies=harness.dependencies)

    assert result["error"] == "response_too_large"
    assert len(harness.browser.calls) == (0 if failure.stage == "fetch" else 1)
    assert harness.executor.calls == []


@pytest.mark.asyncio
async def test_runner_copies_successful_generic_extraction_before_conversion() -> None:
    from tldw_Server_API.app.core.Web_Scraping.orchestration.article import _run_article

    shared = _article(content="raw")
    harness = _harness(extract_results=[shared])
    result = await _run_article(URL, None, True, dependencies=harness.dependencies)

    assert result["content"] == "converted:raw"
    assert shared["content"] == "raw"
    assert result is not shared


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["policy", "preflight", "fetch", "browser", "executor"])
async def test_runner_reraises_cancellation_at_each_await(stage: str) -> None:
    from tldw_Server_API.app.core.Web_Scraping.orchestration.article import _run_article

    harness = _harness(fetch_outcomes=[FetchResponse(URL, 204, {}, "", "httpx")])
    if stage == "policy":
        harness.evaluate_target.side_effect = asyncio.CancelledError()
    elif stage == "preflight":
        harness.run_preflight.side_effect = asyncio.CancelledError()
    elif stage == "fetch":
        harness.fetch.outcomes = [asyncio.CancelledError()]
    elif stage == "browser":
        harness.browser.outcomes = [asyncio.CancelledError()]
    else:
        harness.executor.run = AsyncMock(side_effect=asyncio.CancelledError())

    with pytest.raises(asyncio.CancelledError):
        await _run_article(URL, None, True, dependencies=harness.dependencies)


@pytest.mark.asyncio
async def test_runner_keeps_event_loop_live_while_real_executor_runs() -> None:
    from tldw_Server_API.app.core.Web_Scraping.orchestration.article import _run_article
    from tldw_Server_API.app.core.Web_Scraping.orchestration.executor import ExtractionExecutorManager

    harness = _harness()
    started = threading.Event()
    release = threading.Event()

    def blocking_extract(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        started.set()
        assert release.wait(timeout=1.0)
        return _article()

    executor = ExtractionExecutorManager(worker_count_loader=lambda: 1)
    harness.dependencies = dataclasses.replace(harness.dependencies, executor=executor, extract=blocking_extract)
    task = asyncio.create_task(_run_article(URL, None, True, dependencies=harness.dependencies))
    await asyncio.to_thread(started.wait, 1.0)
    heartbeat = asyncio.create_task(asyncio.sleep(0))
    await heartbeat
    release.set()
    result = await task
    await executor.shutdown()

    assert result["extraction_successful"] is True


@pytest.mark.asyncio
async def test_runner_keeps_event_loop_live_while_config_load_blocks() -> None:
    from tldw_Server_API.app.core.Web_Scraping.orchestration.article import _run_article

    harness = _harness()
    loop = asyncio.get_running_loop()
    heartbeat = threading.Event()
    release = threading.Event()
    observed_heartbeat: list[bool] = []

    def release_config() -> None:
        observed_heartbeat.append(heartbeat.is_set())
        release.set()

    def blocking_config() -> Mapping[str, Any]:
        loop.call_soon_threadsafe(heartbeat.set)
        timer = threading.Timer(0.05, release_config)
        timer.start()
        try:
            assert release.wait(timeout=1.0)
        finally:
            timer.join()
        return {"web_scraper": {"web_scraper_preflight_analyzers": False}}

    harness.dependencies = dataclasses.replace(harness.dependencies, load_config=blocking_config)
    result = await _run_article(URL, None, True, dependencies=harness.dependencies)

    assert result["extraction_successful"] is True
    assert observed_heartbeat == [True]


@pytest.mark.asyncio
async def test_runner_forwards_resolved_strategy_settings_exactly() -> None:
    from tldw_Server_API.app.core.Web_Scraping.orchestration.article import _run_article

    def handler(_html: str, _url: str) -> dict[str, Any]:
        return _article()

    plan = dataclasses.replace(
        _plan(handler="package.module:handler"),
        strategy_order=("schema", "regex"),
        schema_rules={"baseSelector": "//article"},
        llm_settings={"model": "local"},
        regex_settings={"patterns": ["ID"]},
        cluster_settings={"threshold": 0.8},
    )
    harness = _harness()
    harness.dependencies = dataclasses.replace(
        harness.dependencies,
        resolve_plan=lambda _url, _config: plan,
        resolve_handler=lambda _path: handler,
    )
    result = await _run_article(URL, None, False, dependencies=harness.dependencies)
    _func, _args, kwargs = harness.executor.calls[0]

    assert result["extraction_successful"] is True
    assert kwargs == {
        "strategy_order": ["schema", "regex"],
        "handler": handler,
        "schema_rules": {"baseSelector": "//article"},
        "llm_settings": {"model": "local"},
        "regex_settings": {"patterns": ("ID",)},
        "cluster_settings": {"threshold": 0.8},
        "allow_llm_extraction": False,
    }


@pytest.mark.asyncio
async def test_public_entry_snapshots_cookies_before_policy_await(monkeypatch: pytest.MonkeyPatch) -> None:
    from tldw_Server_API.app.core.Web_Scraping.orchestration import article as canonical

    caller_cookies = [{"name": "session", "value": "before", "domain": "example.com"}]
    harness = _harness(
        fetch_outcomes=[FetchResponse(URL, 204, {}, "", "httpx")],
        browser_outcomes=["<html><body>rendered</body></html>"],
    )
    captured: dict[str, Any] = {}

    async def evaluate_target(*_args: Any, **_kwargs: Any) -> PreflightTarget:
        caller_cookies[0]["value"] = "after"
        return _allowed_target()

    def build_dependencies(cookie_snapshot: Sequence[Mapping[str, Any]] | None) -> Any:
        captured["cookies"] = cookie_snapshot
        plan = dataclasses.replace(
            _plan(),
            browser=dataclasses.replace(_plan().browser, custom_cookies=tuple(cookie_snapshot or ())),
        )
        return dataclasses.replace(
            harness.dependencies,
            evaluate_target=evaluate_target,
            resolve_plan=lambda _url, _config: plan,
        )

    monkeypatch.setattr(canonical, "_build_default_dependencies", build_dependencies)
    result = await canonical.scrape_article(URL, caller_cookies)

    assert result["extraction_successful"] is True
    assert captured["cookies"] is not caller_cookies
    assert captured["cookies"][0]["value"] == "before"
    assert harness.fetch.requests[0].cookies == {"session": "before"}
    assert harness.browser.calls[0][1].custom_cookies[0]["value"] == "before"


def test_article_dependencies_are_frozen_and_slotted() -> None:
    from tldw_Server_API.app.core.Web_Scraping.orchestration.article import ArticleDependencies

    harness = _harness()
    assert dataclasses.is_dataclass(harness.dependencies)
    assert hasattr(ArticleDependencies, "__slots__")
    with pytest.raises(FrozenInstanceError):
        harness.dependencies.fetch_client = object()


def test_public_coroutine_is_a_direct_canonical_export_with_exact_signature() -> None:
    from tldw_Server_API.app.core.Web_Scraping import Article_Extractor_Lib as legacy
    from tldw_Server_API.app.core.Web_Scraping.orchestration import scrape_article
    from tldw_Server_API.app.core.Web_Scraping.orchestration.article import (
        _js_required,
    )
    from tldw_Server_API.app.core.Web_Scraping.orchestration.article import (
        scrape_article as canonical,
    )

    assert legacy.scrape_article is canonical is scrape_article
    assert legacy._js_required is _js_required
    assert inspect.iscoroutinefunction(canonical)
    signature = inspect.signature(canonical)
    parameters = tuple(signature.parameters.values())
    assert [parameter.name for parameter in parameters] == [
        "url",
        "custom_cookies",
        "allow_llm_extraction",
    ]
    assert [parameter.kind for parameter in parameters] == [
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    ]
    assert parameters[0].default is inspect.Parameter.empty
    assert parameters[1].default is None
    assert parameters[2].default is True
    resolved = inspect.get_annotations(canonical, eval_str=True)
    assert resolved["url"] is str
    assert resolved["custom_cookies"] == list[dict[str, Any]] | None
    assert resolved["allow_llm_extraction"] is bool
    assert resolved["return"] == dict[str, Any]


def test_orchestration_never_recovers_cancelled_error_in_exception_tuples() -> None:
    from pathlib import Path

    root = Path(__file__).parents[2] / "app/core/Web_Scraping/orchestration"
    for path in root.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for handler in (node for node in ast.walk(tree) if isinstance(node, ast.ExceptHandler)):
            assert not (
                isinstance(handler.type, ast.Tuple)
                and any(
                    isinstance(element, ast.Attribute) and element.attr == "CancelledError"
                    for element in handler.type.elts
                )
            ), path
