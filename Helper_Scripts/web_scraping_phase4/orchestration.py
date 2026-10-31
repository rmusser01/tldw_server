"""Capture article orchestration behavior with deterministic local fakes."""

from __future__ import annotations

import os
from collections.abc import Mapping
from contextlib import ExitStack
from typing import Any
from unittest.mock import patch

from Helper_Scripts.web_scraping_phase4.shared import FIXED_ENV, case, metric_patches


def _serialize_request(request: Any) -> dict[str, Any]:
    return {
        "allow_redirects": request.allow_redirects,
        "backend": request.backend,
        "cookies": dict(sorted(request.cookies.items())),
        "headers": dict(sorted(request.headers.items())),
        "method": request.method,
        "timeout": request.timeout,
        "url": request.url,
    }


class _FakePolicyChecker:
    def __init__(self, decision: Any, *, error: bool = False) -> None:
        self.decision = decision
        self.error = error
        self.calls: list[dict[str, Any]] = []

    async def decide(
        self,
        url: str,
        *,
        respect_robots: bool,
        user_agent: str | None,
        context: Any,
        config: Mapping[str, Any] | None,
    ) -> Any:
        self.calls.append(
            {
                "config": dict((config or {}).get("web_scraper", {})),
                "context_source": context.source,
                "context_stage": context.stage,
                "respect_robots": respect_robots,
                "url": url,
                "user_agent": user_agent,
            }
        )
        if self.error:
            raise RuntimeError("fixture policy failure")
        return self.decision


class _FakeFetchClient:
    def __init__(self, responses: list[Any]) -> None:
        self.responses = list(responses)
        self.requests: list[Any] = []

    def fetch(self, request: Any) -> Any:
        self.requests.append(request)
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


async def _run_article_case(
    article: Any,
    FetchResponse: Any,
    PolicyDecision: Any,
    fixture_case: Mapping[str, Any],
) -> dict[str, Any]:
    with ExitStack() as stack:
        recorder = metric_patches(stack, article)
        stack.enter_context(patch.object(article.random, "uniform", lambda *_args, **_kwargs: 0.0))
        stack.enter_context(patch.dict(os.environ, FIXED_ENV, clear=False))
        article.clear_extraction_caches()

        web_scraper_config = {
            "web_scraper_preflight_analyzers": fixture_case.get(
                "preflight_enabled",
                False,
            ),
            "web_scraper_respect_robots": True,
        }
        if "preflight_include_results" in fixture_case:
            web_scraper_config["web_scraper_preflight_include_results"] = fixture_case["preflight_include_results"]
        config = {"web_scraper": web_scraper_config}
        rules = {
            "domains": {
                "example.com": {
                    "backend": fixture_case.get("backend", "httpx"),
                    "cookies": {"mode": "fixture", "session": "plan"},
                    "extra_headers": {"X-Fixture": "phase4"},
                    "handler": "fixture:handler",
                    "respect_robots": True,
                    "ua_profile": "chrome_120_win",
                }
            }
        }
        stack.enter_context(patch.object(article, "load_and_log_configs", lambda: config))
        stack.enter_context(patch.object(article.ScraperRouter, "load_rules_from_yaml", lambda _path: rules))
        stack.enter_context(patch.object(article, "_js_required", lambda *_args, **_kwargs: False))

        handler_result = dict(fixture_case.get("handler_result", {}))

        def _handler(_html: str, url: str) -> dict[str, Any]:
            return {"url": url, **handler_result}

        stack.enter_context(patch.object(article, "resolve_handler", lambda _path: _handler))

        preflight_calls = {
            "build_execution_context": 0,
            "run_preflight": 0,
        }

        def _build_execution_context(*_args: Any, **_kwargs: Any) -> object:
            preflight_calls["build_execution_context"] += 1
            return object()

        async def _run_preflight(*_args: Any, **_kwargs: Any) -> Any:
            preflight_calls["run_preflight"] += 1
            return article.preflight_facade.PreflightResult(
                analysis=fixture_case.get("preflight_analysis", {}),
            )

        stack.enter_context(
            patch.object(
                article.preflight_facade,
                "build_execution_context",
                _build_execution_context,
            )
        )
        stack.enter_context(
            patch.object(
                article.preflight_facade,
                "run_preflight",
                _run_preflight,
            )
        )

        scenario = fixture_case["scenario"]
        decision = None
        if scenario != "policy_error":
            allowed = scenario != "policy_denied"
            decision = PolicyDecision(
                allowed=allowed,
                mode="compat" if allowed else "strict",
                reason="allowed" if allowed else "robots_disallowed",
                stage="pre_fetch",
                source="article_extract",
            )
        policy_checker = _FakePolicyChecker(decision, error=scenario == "policy_error")
        responses: list[Any] = []
        if scenario in {"lightweight_success", "curl_fallback", "preflight_success"}:
            if scenario == "curl_fallback":
                responses.append(RuntimeError("fixture curl failure"))
            responses.append(
                FetchResponse(
                    url=fixture_case["url"],
                    status=200,
                    headers={"Content-Type": "text/html"},
                    text=fixture_case["html"],
                    backend="httpx",
                )
            )
        fetch_client = _FakeFetchClient(responses)
        stack.enter_context(patch.object(article, "_ARTICLE_POLICY_CHECKER", policy_checker))
        stack.enter_context(patch.object(article, "_ARTICLE_FETCH_CLIENT", fetch_client))

        result = await article.scrape_article(
            fixture_case["url"],
            custom_cookies=fixture_case.get("custom_cookies"),
            allow_llm_extraction=False,
        )
        actual: dict[str, Any] = {
            "cache_stats": article.get_extraction_cache_stats(),
            "fetch_requests": [_serialize_request(request) for request in fetch_client.requests],
            "metrics": recorder.events,
            "policy_calls": policy_checker.calls,
            "result": result,
        }
        if fixture_case.get("preflight_enabled"):
            actual["preflight_calls"] = preflight_calls
        article.clear_extraction_caches()
        return actual


async def build_article_cases(
    article: Any,
    FetchResponse: Any,
    PolicyDecision: Any,
) -> list[dict[str, Any]]:
    success_result = {
        "author": "Fixture Author",
        "content": "Fixture body",
        "date": "2026-07-27",
        "extraction_successful": True,
        "title": "Fixture title",
    }
    cases = [
        case(
            {
                "backend": "httpx",
                "name": "policy_denial_short_circuits_fetch",
                "scenario": "policy_denied",
                "url": "https://example.com/blocked",
            }
        ),
        case(
            {
                "backend": "httpx",
                "behavior_change": 7,
                "difference_contract": "change_7_policy_error",
                "name": "policy_error_is_publicly_bounded",
                "scenario": "policy_error",
                "url": "https://example.com/policy-error",
            }
        ),
        case(
            {
                "backend": "httpx",
                "custom_cookies": [{"name": "session", "value": "custom"}],
                "handler_result": success_result,
                "html": "<html><body><article>Fixture source</article></body></html>",
                "name": "lightweight_http_success",
                "scenario": "lightweight_success",
                "url": "https://example.com/article",
            }
        ),
        case(
            {
                "backend": "curl",
                "custom_cookies": [{"name": "session", "value": "custom"}],
                "handler_result": success_result,
                "html": "<html><body><article>Fixture source</article></body></html>",
                "name": "curl_falls_back_to_httpx",
                "scenario": "curl_fallback",
                "url": "https://example.com/fallback",
            }
        ),
        case(
            {
                "backend": "auto",
                "handler_result": success_result,
                "html": "<html><body><article>Fixture source</article></body></html>",
                "name": "preflight_success_applies_advice_and_attaches_payload",
                "preflight_analysis": {"results": {"tls": {"status": "active"}}},
                "preflight_enabled": True,
                "preflight_include_results": True,
                "scenario": "preflight_success",
                "url": "https://example.com/preflight",
            }
        ),
    ]
    for fixture_case in cases:
        fixture_case["expected"] = await _run_article_case(
            article,
            FetchResponse,
            PolicyDecision,
            fixture_case,
        )
    return cases
