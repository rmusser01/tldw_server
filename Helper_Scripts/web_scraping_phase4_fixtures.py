#!/usr/bin/env python3
"""Generate immutable Phase 4 predecessor behavior fixtures."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
import subprocess  # nosec B404
import sys
from collections.abc import Mapping
from contextlib import ExitStack
from pathlib import Path
from typing import Any
from unittest.mock import patch

SCHEMA_VERSION = 1
REPO_ROOT = Path(__file__).resolve().parents[1]
CASE_NAMES = (
    "article_orchestration_fakes",
    "content",
    "extraction",
    "metadata",
    "selectors",
)

_FIXED_ENV = {
    "CLUSTER_LINKAGE": "",
    "EXTRACTOR_CLEAR_CACHES": "",
    "EXTRACTOR_MAX_RETRIES": "0",
    "EXTRACTOR_MAX_WORKERS": "",
    "EXTRACTOR_REGEX_MASK_PII": "false",
    "EXTRACTOR_RETRY_BASE_MS": "0",
    "EXTRACTOR_RETRY_JITTER_MS": "0",
    "SIM_THRESHOLD": "",
    "WATCHLIST_SELECTOR_MAX_EXPR_LEN": "512",
    "WATCHLIST_SELECTOR_MAX_XPATH_DESCENDANT_STEPS": "12",
    "WATCHLIST_SELECTOR_MAX_XPATH_FUNCTION_CALLS": "8",
    "WATCHLIST_SELECTOR_MAX_XPATH_PREDICATES": "10",
    "WORD_COUNT_THRESHOLD": "",
}


def build_manifest(predecessor_commit: str, case_files: dict[str, str]) -> dict[str, object]:
    if re.fullmatch(r"[0-9a-f]{40}", predecessor_commit) is None:
        raise ValueError("predecessor_commit must be a full 40-character lowercase commit id")
    return {
        "schema_version": SCHEMA_VERSION,
        "predecessor_commit": predecessor_commit,
        "cases": dict(sorted(case_files.items())),
    }


def _git_head() -> str:
    git_executable = shutil.which("git")
    if git_executable is None:
        raise OSError("git executable not found")
    completed = subprocess.run(  # nosec B603
        [git_executable, "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _validate_provenance(predecessor_commit: str) -> None:
    build_manifest(predecessor_commit, {})
    head = _git_head()
    if predecessor_commit != head:
        raise ValueError(f"predecessor_commit {predecessor_commit} does not match workspace HEAD {head}")


def _canonical_data(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=True, sort_keys=True))


def _write_json(path: Path, payload: object) -> None:
    encoded = json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    path.write_text(encoded, encoding="utf-8", newline="\n")


def _normalize_formatted_metadata(value: str) -> str:
    return re.sub(
        r'("ingestion_date":\s*)"[^"]+"',
        r'\1"<TIMESTAMP>"',
        value,
        count=1,
    )


class _MetricRecorder:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def counter(self, emitter: str):
        def _record(name: str, labels: Mapping[str, Any] | None = None, **_kwargs: Any) -> None:
            self.events.append(
                {
                    "emitter": emitter,
                    "kind": "counter",
                    "labels": dict(sorted((labels or {}).items())),
                    "name": name,
                }
            )

        return _record

    def histogram(self, emitter: str):
        def _record(
            name: str,
            value: int | float,
            labels: Mapping[str, Any] | None = None,
            **_kwargs: Any,
        ) -> None:
            normalized_value: int | float | str = value
            if "duration" in name or "latency" in name:
                normalized_value = "<TIMING>"
            self.events.append(
                {
                    "emitter": emitter,
                    "kind": "histogram",
                    "labels": dict(sorted((labels or {}).items())),
                    "name": name,
                    "value": normalized_value,
                }
            )

        return _record


def _metric_patches(stack: ExitStack, article: Any) -> _MetricRecorder:
    recorder = _MetricRecorder()
    stack.enter_context(patch.object(article, "increment_counter", recorder.counter("increment_counter")))
    stack.enter_context(patch.object(article, "log_counter", recorder.counter("log_counter")))
    stack.enter_context(patch.object(article, "observe_histogram", recorder.histogram("observe_histogram")))
    stack.enter_context(patch.object(article, "log_histogram", recorder.histogram("log_histogram")))
    return recorder


def _case(payload: dict[str, Any]) -> dict[str, Any]:
    return _canonical_data(payload)


def _load_predecessor_modules() -> tuple[Any, Any, Any, Any, Any, Any]:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    from tldw_Server_API.app.core.Watchlists import fetchers
    from tldw_Server_API.app.core.Web_Scraping import Article_Extractor_Lib as article
    from tldw_Server_API.app.core.Web_Scraping.runtime import (
        FetchRequest,
        FetchResponse,
        PolicyDecision,
        RuntimeRequestContext,
    )

    return article, fetchers, FetchRequest, FetchResponse, PolicyDecision, RuntimeRequestContext


def _build_content_cases(article: Any) -> list[dict[str, Any]]:
    cases = [
        _case(
            {
                "html": (
                    "<html><body><h1>Fixture &amp; Title</h1>"
                    "<p>First paragraph.</p>"
                    "<p>Second <strong>bold</strong> paragraph.</p></body></html>"
                ),
                "name": "paragraph_and_inline_formatting",
                "operation": "convert_html_to_markdown",
            }
        ),
        _case(
            {
                "html": "<div>Lead<span>inline</span></div><p>Tail paragraph.</p>",
                "name": "mixed_block_and_paragraph_formatting",
                "operation": "convert_html_to_markdown",
            }
        ),
    ]
    for case in cases:
        case["expected"] = article.convert_html_to_markdown(case["html"])
    return cases


def _metadata_inspection(handler: Any, content: str) -> dict[str, Any]:
    metadata, clean_content = handler.extract_metadata(content)
    return {
        "clean_content": clean_content,
        "content_hash": handler.get_content_hash(content),
        "has_metadata": handler.has_metadata(content),
        "metadata": metadata,
        "stripped": handler.strip_metadata(content),
    }


def _build_metadata_cases(article: Any) -> list[dict[str, Any]]:
    handler = article.ContentMetadataHandler
    canonical_envelope = (
        "  [METADATA]\n"
        '{"url":"https://example.com/article","literal":"brackets [{]} and \\"quotes\\""}\n'
        "[/METADATA]\n\nArticle body"
    )
    accepted_nested = '[METADATA]{"value":' + "[" * 63 + "0" + "]" * 63 + "}[/METADATA]\nArticle body"
    rejected_nested = '[METADATA]{"value":' + "[" * 64 + "0" + "]" * 64 + "}[/METADATA]\nArticle body"
    cases = [
        _case(
            {
                "additional_metadata": {"author": "Ada", "language": "en"},
                "content": "Fixture body with caf\u00e9.",
                "name": "canonical_formatted_envelope",
                "operation": "format",
                "pipeline": "FixturePipeline",
                "url": "https://example.com/article",
            }
        ),
        _case(
            {
                "content": canonical_envelope,
                "name": "canonical_envelope_inspection",
                "operation": "inspect",
            }
        ),
        _case(
            {
                "content": '[METADATA]{"url":"https://example.com"}\nArticle body',
                "name": "malformed_envelope_passes_through",
                "operation": "inspect",
            }
        ),
        _case(
            {
                "content": accepted_nested,
                "name": "nesting_boundary_is_accepted",
                "operation": "inspect",
            }
        ),
        _case(
            {
                "content": rejected_nested,
                "name": "nesting_over_boundary_is_rejected",
                "operation": "inspect",
            }
        ),
        _case(
            {
                "name": "metadata_only_changes_do_not_change_body_hash",
                "new_content": '[METADATA]{"version":2}[/METADATA]\nSame body',
                "old_content": '[METADATA]{"version":1}[/METADATA]\nSame body',
                "operation": "content_changed",
            }
        ),
        _case(
            {
                "name": "body_changes_are_detected",
                "new_content": '[METADATA]{"version":2}[/METADATA]\nNew body',
                "old_content": '[METADATA]{"version":1}[/METADATA]\nOld body',
                "operation": "content_changed",
            }
        ),
    ]
    for case in cases:
        if case["operation"] == "format":
            case["expected"] = _normalize_formatted_metadata(
                handler.format_content_with_metadata(
                    case["url"],
                    case["content"],
                    pipeline=case["pipeline"],
                    additional_metadata=case["additional_metadata"],
                )
            )
        elif case["operation"] == "inspect":
            case["expected"] = _metadata_inspection(handler, case["content"])
        else:
            case["expected"] = handler.content_changed(case["old_content"], case["new_content"])
    return cases


def _build_selector_cases(fetchers: Any) -> list[dict[str, Any]]:
    schema_html = (
        "<html><body><article><h1> Example Title </h1>"
        '<time>2025-01-15</time><span class="views">1,234</span>'
        '<p class="content">First paragraph.</p><p class="content">Second paragraph.</p>'
        '<a class="more" href="/read-more">Read more</a>'
        '<span class="tag">News</span><span class="tag">Tech</span>'
        "</article></body></html>"
    )
    cases = [
        _case(
            {
                "html": ("<html><body><h1>One</h1><h1>Two</h1>" '<div class="body">Body</div></body></html>'),
                "include_counts": True,
                "name": "validation_counts_and_warnings",
                "operation": "validate",
                "rules": {
                    "content_xpath": "//div[@class='body']",
                    "title_xpath": "//h1",
                },
            }
        ),
        _case(
            {
                "behavior_change": 7,
                "include_counts": False,
                "name": "invalid_xpath_error",
                "operation": "validate",
                "rules": {"content_xpath": "//article["},
            }
        ),
        _case(
            {
                "include_counts": False,
                "name": "selector_complexity_guard",
                "operation": "validate",
                "rules": {"content_xpath": "//div" + "/span" * 200},
            }
        ),
        _case(
            {
                "base_url": "https://example.com/post",
                "html": schema_html,
                "name": "schema_dsl_transforms_and_lists",
                "operation": "extract_schema_fields",
                "rules": {
                    "baseFields": [
                        {
                            "name": "title",
                            "selector": ".//h1",
                            "transforms": ["strip"],
                            "type": "text",
                        },
                        {
                            "join_with": "\n",
                            "name": "content",
                            "selector": ".//p[@class='content']",
                            "type": "text",
                        },
                        {
                            "name": "published",
                            "selector": ".//time",
                            "transforms": [{"format": "%Y-%m-%d", "name": "date_normalize"}],
                            "type": "text",
                        },
                        {
                            "name": "views",
                            "selector": ".//span[@class='views']",
                            "transforms": [{"name": "number_normalize"}],
                            "type": "text",
                        },
                    ],
                    "baseSelector": "//article",
                    "fields": [
                        {
                            "attribute": "href",
                            "name": "link",
                            "selector": ".//a[@class='more']",
                            "transforms": [{"name": "urljoin"}],
                            "type": "attribute",
                        },
                        {
                            "itemType": "text",
                            "name": "tags",
                            "selector": ".//span[@class='tag']",
                            "transforms": ["lowercase"],
                            "type": "list",
                        },
                        {
                            "from": "title",
                            "name": "slug",
                            "transforms": [
                                "lowercase",
                                {"name": "regex_replace", "pattern": r"\s+", "repl": "-"},
                            ],
                            "type": "computed",
                        },
                    ],
                    "name": "article",
                },
            }
        ),
        _case(
            {
                "base_url": "https://example.com/article",
                "html": (
                    "<html><body><main><h1>Legacy title</h1>"
                    '<div class="body"><p>One</p><p>Two</p></div>'
                    '<a rel="author">Grace Hopper</a></main></body></html>'
                ),
                "name": "legacy_selector_field_extraction",
                "operation": "extract_schema_fields",
                "rules": {
                    "author_xpath": ".//a[@rel='author']",
                    "base_xpath": "//main",
                    "content_xpath": ".//div[@class='body']/p",
                    "title_xpath": ".//h1",
                },
            }
        ),
    ]

    for case in cases:
        fetchers.clear_selector_caches()
        if case["operation"] == "validate":
            result = fetchers.validate_selector_rules(
                case["rules"],
                html_text=case.get("html"),
                include_counts=case["include_counts"],
            )
        else:
            result = fetchers.extract_schema_fields(case["html"], case["base_url"], case["rules"])
        case["expected"] = {
            "cache_stats": fetchers.get_selector_cache_stats(),
            "result": result,
        }
        fetchers.clear_selector_caches()
    return cases


def _run_extraction_case(article: Any, case: Mapping[str, Any]) -> dict[str, Any]:
    with ExitStack() as stack:
        recorder = _metric_patches(stack, article)
        stack.enter_context(patch.object(article.random, "uniform", lambda *_args, **_kwargs: 0.0))
        stack.enter_context(patch.dict(os.environ, _FIXED_ENV, clear=False))
        article.clear_extraction_caches()
        operation = case["operation"]
        if operation == "regex":
            result = article.extract_regex_entities(case["html"], case["url"], mask_pii=case["mask_pii"])
        elif operation == "jsonld":
            result = article.extract_jsonld_entities(case["html"], case["url"])
        elif operation == "cluster":
            result = article.extract_cluster_entities(
                case["html"], case["url"], cluster_settings=case["cluster_settings"]
            )
        elif operation == "pipeline":
            fallback_result = case.get("fallback_result")

            def _fallback(_html: str, url: str) -> dict[str, Any]:
                return {"url": url, **dict(fallback_result or {})}

            result = article.extract_article_with_pipeline(
                case["html"],
                case["url"],
                strategy_order=case.get("strategy_order"),
                fallback_extractor=_fallback if fallback_result is not None else None,
                allow_llm_extraction=case["allow_llm_extraction"],
            )
        else:
            raise ValueError(f"Unknown extraction operation: {operation}")

        actual = {
            "cache_stats": article.get_extraction_cache_stats(),
            "metrics": recorder.events,
            "result": result,
        }
        article.clear_extraction_caches()
        return actual


def _build_extraction_cases(article: Any) -> list[dict[str, Any]]:
    description_only_jsonld = (
        '<html><head><script type="application/ld+json">'
        '{"@context":"https://schema.org","@type":"Article",'
        '"headline":"Structured title","description":"Structured summary"}'
        "</script></head><body></body></html>"
    )
    cases = [
        _case(
            {
                "html": (
                    "<html><head><title>Contacts</title></head><body>"
                    "Email demo@example.com or call +1 (415) 555-2671."
                    "</body></html>"
                ),
                "mask_pii": False,
                "name": "regex_catalog_matches",
                "operation": "regex",
                "url": "https://example.com/contacts",
            }
        ),
        _case(
            {
                "html": (
                    '<html><head><script type="application/ld+json">'
                    '{"@context":"https://schema.org","@type":"NewsArticle",'
                    '"headline":"JSON-LD Title","author":{"@type":"Person","name":"Jane Doe"},'
                    '"datePublished":"2024-05-01","articleBody":"JSON-LD body text."}'
                    "</script></head><body></body></html>"
                ),
                "name": "jsonld_article",
                "operation": "jsonld",
                "url": "https://example.com/jsonld",
            }
        ),
        _case(
            {
                "cluster_settings": {
                    "cluster_threshold": 0.1,
                    "embed_dims": 32,
                    "max_blocks": 10,
                    "method": "greedy",
                    "min_block_chars": 20,
                    "min_word_count": 4,
                    "prefilter_threshold": 0.0,
                    "tag_keywords": {
                        "research": ["research", "dataset"],
                        "security": ["security", "encryption"],
                    },
                    "tag_top_k": 2,
                },
                "html": (
                    "<html><head><title>Cluster Fixture</title></head><body><article>"
                    "<p>Security research explains encryption controls for a stable local fixture.</p>"
                    "<p>The research dataset contains deterministic examples for repeatable extraction.</p>"
                    "</article></body></html>"
                ),
                "name": "cluster_extraction",
                "operation": "cluster",
                "url": "https://example.com/cluster",
            }
        ),
        _case(
            {
                "allow_llm_extraction": False,
                "fallback_result": {
                    "author": "Fixture Author",
                    "content": "Fallback body",
                    "date": "2026-07-27",
                    "extraction_successful": True,
                    "summary": "   ",
                    "title": "Fallback title",
                },
                "html": description_only_jsonld,
                "name": "jsonld_summary_carries_to_fallback",
                "operation": "pipeline",
                "strategy_order": ["jsonld", "trafilatura"],
                "url": "https://example.com/summary",
            }
        ),
        _case(
            {
                "allow_llm_extraction": False,
                "fallback_result": {
                    "author": "Fixture Author",
                    "content": "Fallback body",
                    "date": "2026-07-27",
                    "extraction_successful": True,
                    "title": "Fallback title",
                },
                "html": "<html><body><p>Fallback input.</p></body></html>",
                "name": "unknown_strategy_is_traced",
                "operation": "pipeline",
                "strategy_order": ["mystery", "trafilatura"],
                "url": "https://example.com/unknown",
            }
        ),
        _case(
            {
                "allow_llm_extraction": False,
                "behavior_change": 1,
                "html": "<html><body>Contact predecessor@example.com</body></html>",
                "name": "default_regex_is_terminal_in_predecessor",
                "operation": "pipeline",
                "strategy_order": None,
                "url": "https://example.com/regex-default",
            }
        ),
    ]
    for case in cases:
        case["expected"] = _run_extraction_case(article, case)
    return cases


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
    case: Mapping[str, Any],
) -> dict[str, Any]:
    with ExitStack() as stack:
        recorder = _metric_patches(stack, article)
        stack.enter_context(patch.object(article.random, "uniform", lambda *_args, **_kwargs: 0.0))
        stack.enter_context(patch.dict(os.environ, _FIXED_ENV, clear=False))
        article.clear_extraction_caches()

        config = {
            "web_scraper": {
                "web_scraper_preflight_analyzers": False,
                "web_scraper_respect_robots": True,
            }
        }
        rules = {
            "domains": {
                "example.com": {
                    "backend": case.get("backend", "httpx"),
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

        handler_result = dict(case.get("handler_result", {}))

        def _handler(_html: str, url: str) -> dict[str, Any]:
            return {"url": url, **handler_result}

        stack.enter_context(patch.object(article, "resolve_handler", lambda _path: _handler))

        scenario = case["scenario"]
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
        if scenario in {"lightweight_success", "curl_fallback"}:
            if scenario == "curl_fallback":
                responses.append(RuntimeError("fixture curl failure"))
            responses.append(
                FetchResponse(
                    url=case["url"],
                    status=200,
                    headers={"Content-Type": "text/html"},
                    text=case["html"],
                    backend="httpx",
                )
            )
        fetch_client = _FakeFetchClient(responses)
        stack.enter_context(patch.object(article, "_ARTICLE_POLICY_CHECKER", policy_checker))
        stack.enter_context(patch.object(article, "_ARTICLE_FETCH_CLIENT", fetch_client))

        result = await article.scrape_article(
            case["url"],
            custom_cookies=case.get("custom_cookies"),
            allow_llm_extraction=False,
        )
        actual = {
            "cache_stats": article.get_extraction_cache_stats(),
            "fetch_requests": [_serialize_request(request) for request in fetch_client.requests],
            "metrics": recorder.events,
            "policy_calls": policy_checker.calls,
            "result": result,
        }
        article.clear_extraction_caches()
        return actual


async def _build_article_cases(
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
        _case(
            {
                "backend": "httpx",
                "name": "policy_denial_short_circuits_fetch",
                "scenario": "policy_denied",
                "url": "https://example.com/blocked",
            }
        ),
        _case(
            {
                "backend": "httpx",
                "behavior_change": 7,
                "name": "policy_error_is_publicly_bounded",
                "scenario": "policy_error",
                "url": "https://example.com/policy-error",
            }
        ),
        _case(
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
        _case(
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
    ]
    for case in cases:
        case["expected"] = await _run_article_case(article, FetchResponse, PolicyDecision, case)
    return cases


def build_case_payloads() -> dict[str, dict[str, Any]]:
    article, fetchers, _FetchRequest, FetchResponse, PolicyDecision, _RuntimeContext = _load_predecessor_modules()
    with patch.dict(os.environ, _FIXED_ENV, clear=False):
        fetchers.reload_selector_guardrails_from_env()
        payloads = {
            "article_orchestration_fakes": {
                "category": "article_orchestration_fakes",
                "cases": asyncio.run(_build_article_cases(article, FetchResponse, PolicyDecision)),
            },
            "content": {"category": "content", "cases": _build_content_cases(article)},
            "extraction": {"category": "extraction", "cases": _build_extraction_cases(article)},
            "metadata": {"category": "metadata", "cases": _build_metadata_cases(article)},
            "selectors": {"category": "selectors", "cases": _build_selector_cases(fetchers)},
        }
    fetchers.reload_selector_guardrails_from_env()
    if set(payloads) != set(CASE_NAMES):
        raise RuntimeError("Fixture category set is incomplete")
    return payloads


def generate_fixtures(predecessor_commit: str, output: Path) -> None:
    _validate_provenance(predecessor_commit)
    payloads = build_case_payloads()
    output.mkdir(parents=True, exist_ok=True)

    case_files: dict[str, str] = {}
    for category in sorted(payloads):
        filename = f"{category}.json"
        _write_json(output / filename, payloads[category])
        case_files[category] = filename

    _write_json(output / "manifest.json", build_manifest(predecessor_commit, case_files))


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predecessor-commit", required=True)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        generate_fixtures(args.predecessor_commit, args.output)
    except (OSError, subprocess.CalledProcessError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
