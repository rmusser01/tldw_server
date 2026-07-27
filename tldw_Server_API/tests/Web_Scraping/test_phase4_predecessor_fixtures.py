from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from tldw_Server_API.app.core.Watchlists import fetchers
from tldw_Server_API.app.core.Web_Scraping import Article_Extractor_Lib as article
from tldw_Server_API.app.core.Web_Scraping.runtime import (
    FetchRequest,
    FetchResponse,
    PolicyDecision,
    RuntimeRequestContext,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
GENERATOR = REPO_ROOT / "Helper_Scripts" / "web_scraping_phase4_fixtures.py"
FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "phase4"
MANIFEST = FIXTURE_ROOT / "manifest.json"

CASE_KEYS = {
    "article_orchestration_fakes",
    "content",
    "extraction",
    "metadata",
    "selectors",
}

APPROVED_BEHAVIOR_CHANGES = {
    1: "default regex becomes non-terminal enrichment",
    2: "caller cancellation is re-raised",
    3: "sync entry points reject active event loops before side effects",
    4: "regex validation and execution failures are bounded and sanitized",
    5: "the individual URL service passes system_message",
    6: "raw-browser sync performs governed admission",
    7: "public exception text is replaced by stable sanitized codes",
    8: "extraction work submission is bounded",
    9: "direct Playwright routing installs egress controls",
    10: "moved observability removes sensitive and high-cardinality fields",
    11: "direct article acquisition enforces response size limits",
}

_FIXED_EXTRACTION_ENV = {
    "CLUSTER_LINKAGE": "",
    "EXTRACTOR_CLEAR_CACHES": "",
    "EXTRACTOR_MAX_RETRIES": "0",
    "EXTRACTOR_MAX_WORKERS": "",
    "EXTRACTOR_REGEX_MASK_PII": "false",
    "EXTRACTOR_RETRY_BASE_MS": "0",
    "EXTRACTOR_RETRY_JITTER_MS": "0",
    "SIM_THRESHOLD": "",
    "WORD_COUNT_THRESHOLD": "",
}

_FIXED_SELECTOR_ENV = {
    "WATCHLIST_SELECTOR_MAX_EXPR_LEN": "512",
    "WATCHLIST_SELECTOR_MAX_XPATH_DESCENDANT_STEPS": "12",
    "WATCHLIST_SELECTOR_MAX_XPATH_FUNCTION_CALLS": "8",
    "WATCHLIST_SELECTOR_MAX_XPATH_PREDICATES": "10",
}


def assert_predecessor_behavior(
    actual: object,
    expected: object,
    *,
    behavior_change: int | None = None,
) -> None:
    """Assert predecessor parity or identify one approved Phase 4 difference."""
    if behavior_change is not None and (
        type(behavior_change) is not int or behavior_change not in APPROVED_BEHAVIOR_CHANGES
    ):
        raise ValueError("behavior_change must be one integer in range 1..11")
    if actual == expected:
        return
    if behavior_change is None:
        assert actual == expected


def _load_manifest_or_skip() -> dict[str, Any]:
    if not MANIFEST.is_file():
        pytest.skip("Phase 4 predecessor manifest has not been generated")
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def _load_cases(category: str) -> list[dict[str, Any]]:
    manifest = _load_manifest_or_skip()
    case_path = FIXTURE_ROOT / manifest["cases"][category]
    if not case_path.is_file():
        pytest.fail(f"Missing predecessor fixture: {case_path.name}")
    payload = json.loads(case_path.read_text(encoding="utf-8"))
    assert payload["category"] == category
    assert isinstance(payload["cases"], list)
    assert payload["cases"]
    return payload["cases"]


def _assert_case(case: Mapping[str, Any], actual: object) -> None:
    try:
        assert_predecessor_behavior(
            actual,
            case["expected"],
            behavior_change=case.get("behavior_change"),
        )
    except AssertionError as exc:
        raise AssertionError(f"Predecessor case failed: {case['name']}") from exc


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


def _install_metric_recorder(monkeypatch: pytest.MonkeyPatch) -> _MetricRecorder:
    recorder = _MetricRecorder()
    monkeypatch.setattr(article, "increment_counter", recorder.counter("increment_counter"))
    monkeypatch.setattr(article, "log_counter", recorder.counter("log_counter"))
    monkeypatch.setattr(article, "observe_histogram", recorder.histogram("observe_histogram"))
    monkeypatch.setattr(article, "log_histogram", recorder.histogram("log_histogram"))
    return recorder


def _set_environment(monkeypatch: pytest.MonkeyPatch, values: Mapping[str, str]) -> None:
    for name, value in values.items():
        monkeypatch.setenv(name, value)


def _serialize_request(request: FetchRequest) -> dict[str, Any]:
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
    def __init__(self, decision: PolicyDecision | None, *, error: bool = False) -> None:
        self.decision = decision
        self.error = error
        self.calls: list[dict[str, Any]] = []

    async def decide(
        self,
        url: str,
        *,
        respect_robots: bool,
        user_agent: str | None,
        context: RuntimeRequestContext,
        config: Mapping[str, Any] | None,
    ) -> PolicyDecision:
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
        assert self.decision is not None
        return self.decision


class _FakeFetchClient:
    def __init__(self, responses: list[FetchResponse | BaseException]) -> None:
        self.responses = list(responses)
        self.requests: list[FetchRequest] = []

    def fetch(self, request: FetchRequest) -> FetchResponse:
        self.requests.append(request)
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


def _policy_decision(case: Mapping[str, Any]) -> PolicyDecision | None:
    if case["scenario"] == "policy_error":
        return None
    allowed = case["scenario"] != "policy_denied"
    return PolicyDecision(
        allowed=allowed,
        mode="compat" if allowed else "strict",
        reason="allowed" if allowed else "robots_disallowed",
        stage="pre_fetch",
        source="article_extract",
    )


async def _run_article_case(case: Mapping[str, Any], monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    recorder = _install_metric_recorder(monkeypatch)
    _set_environment(monkeypatch, _FIXED_EXTRACTION_ENV)
    monkeypatch.setattr(article.random, "uniform", lambda *_args, **_kwargs: 0.0)
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
    monkeypatch.setattr(article, "load_and_log_configs", lambda: config)
    monkeypatch.setattr(article.ScraperRouter, "load_rules_from_yaml", lambda _path: rules)
    monkeypatch.setattr(article, "_js_required", lambda *_args, **_kwargs: False)

    handler_result = dict(case.get("handler_result", {}))

    def _handler(_html: str, url: str) -> dict[str, Any]:
        return {"url": url, **handler_result}

    monkeypatch.setattr(article, "resolve_handler", lambda _path: _handler)

    decision = _policy_decision(case)
    policy_checker = _FakePolicyChecker(decision, error=case["scenario"] == "policy_error")
    responses: list[FetchResponse | BaseException] = []
    if case["scenario"] in {"lightweight_success", "curl_fallback"}:
        if case["scenario"] == "curl_fallback":
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
    monkeypatch.setattr(article, "_ARTICLE_POLICY_CHECKER", policy_checker)
    monkeypatch.setattr(article, "_ARTICLE_FETCH_CLIENT", fetch_client)

    result = await article.scrape_article(
        case["url"],
        custom_cookies=case.get("custom_cookies"),
        allow_llm_extraction=False,
    )
    cache_stats = article.get_extraction_cache_stats()
    article.clear_extraction_caches()
    return {
        "cache_stats": cache_stats,
        "fetch_requests": [_serialize_request(request) for request in fetch_client.requests],
        "metrics": recorder.events,
        "policy_calls": policy_checker.calls,
        "result": result,
    }


def test_phase4_fixture_generator_is_explicit_and_checked_in() -> None:
    assert GENERATOR.is_file(), f"Missing explicit fixture generator: {GENERATOR}"


def test_phase4_fixture_manifest_is_pinned() -> None:
    assert MANIFEST.is_file(), f"Missing fixture manifest: {MANIFEST}"
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    assert manifest["schema_version"] == 1
    assert re.fullmatch(r"[0-9a-f]{40}", manifest["predecessor_commit"])
    assert set(manifest["cases"]) == CASE_KEYS
    assert list(manifest["cases"]) == sorted(manifest["cases"])


def test_phase4_fixture_json_is_canonical_and_complete() -> None:
    manifest = _load_manifest_or_skip()
    expected_names = {"manifest.json", *manifest["cases"].values()}
    assert {path.name for path in FIXTURE_ROOT.glob("*.json")} == expected_names

    for path in sorted(FIXTURE_ROOT.glob("*.json")):
        raw = path.read_bytes()
        decoded = raw.decode("ascii")
        payload = json.loads(decoded)
        canonical = json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
        assert decoded == canonical, path.name


def test_differential_helper_requires_a_tag_for_a_difference() -> None:
    with pytest.raises(AssertionError):
        assert_predecessor_behavior({"value": "current"}, {"value": "predecessor"})


def test_differential_helper_accepts_none_for_equal_values() -> None:
    assert_predecessor_behavior({"value": "same"}, {"value": "same"}, behavior_change=None)


@pytest.mark.parametrize("behavior_change", [1, 11])
def test_differential_helper_accepts_boundary_behavior_changes(behavior_change: int) -> None:
    assert_predecessor_behavior(
        {"value": "current"},
        {"value": "predecessor"},
        behavior_change=behavior_change,
    )


@pytest.mark.parametrize("behavior_change", [0, 12, "1", 1.0, True])
def test_differential_helper_rejects_invalid_behavior_changes(behavior_change: object) -> None:
    with pytest.raises(ValueError, match="range 1..11"):
        assert_predecessor_behavior(
            {"value": "current"},
            {"value": "predecessor"},
            behavior_change=behavior_change,  # type: ignore[arg-type]
        )


def test_content_formatting_matches_predecessor() -> None:
    for case in _load_cases("content"):
        assert case["operation"] == "convert_html_to_markdown"
        actual = article.convert_html_to_markdown(case["html"])
        _assert_case(case, actual)


def test_metadata_envelopes_hashing_and_guards_match_predecessor() -> None:
    handler = article.ContentMetadataHandler
    for case in _load_cases("metadata"):
        operation = case["operation"]
        if operation == "format":
            actual: object = _normalize_formatted_metadata(
                handler.format_content_with_metadata(
                    case["url"],
                    case["content"],
                    pipeline=case["pipeline"],
                    additional_metadata=case.get("additional_metadata"),
                )
            )
        elif operation == "inspect":
            metadata, clean_content = handler.extract_metadata(case["content"])
            actual = {
                "clean_content": clean_content,
                "content_hash": handler.get_content_hash(case["content"]),
                "has_metadata": handler.has_metadata(case["content"]),
                "metadata": metadata,
                "stripped": handler.strip_metadata(case["content"]),
            }
        elif operation == "content_changed":
            actual = handler.content_changed(case["old_content"], case["new_content"])
        else:
            raise AssertionError(f"Unknown metadata fixture operation: {operation}")
        _assert_case(case, actual)


def test_selector_validation_and_extraction_match_predecessor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with monkeypatch.context() as environment:
        _set_environment(environment, _FIXED_SELECTOR_ENV)
        fetchers.reload_selector_guardrails_from_env()
        for case in _load_cases("selectors"):
            fetchers.clear_selector_caches()
            operation = case["operation"]
            if operation == "validate":
                result = fetchers.validate_selector_rules(
                    case["rules"],
                    html_text=case.get("html"),
                    include_counts=case.get("include_counts", False),
                )
            elif operation == "extract_schema_fields":
                result = fetchers.extract_schema_fields(case["html"], case["base_url"], case["rules"])
            else:
                raise AssertionError(f"Unknown selector fixture operation: {operation}")
            actual = {
                "cache_stats": fetchers.get_selector_cache_stats(),
                "result": result,
            }
            fetchers.clear_selector_caches()
            _assert_case(case, actual)
    fetchers.reload_selector_guardrails_from_env()


def test_extraction_behavior_matches_predecessor(monkeypatch: pytest.MonkeyPatch) -> None:
    for case in _load_cases("extraction"):
        recorder = _install_metric_recorder(monkeypatch)
        _set_environment(monkeypatch, _FIXED_EXTRACTION_ENV)
        monkeypatch.setattr(article.random, "uniform", lambda *_args, **_kwargs: 0.0)
        article.clear_extraction_caches()
        operation = case["operation"]
        if operation == "regex":
            result = article.extract_regex_entities(
                case["html"],
                case["url"],
                mask_pii=case["mask_pii"],
            )
        elif operation == "jsonld":
            result = article.extract_jsonld_entities(case["html"], case["url"])
        elif operation == "cluster":
            result = article.extract_cluster_entities(
                case["html"],
                case["url"],
                cluster_settings=case["cluster_settings"],
            )
        elif operation == "pipeline":
            fallback_result = case.get("fallback_result")

            def _fallback(
                _html: str,
                url: str,
                _fallback_result: object = fallback_result,
            ) -> dict[str, Any]:
                return {"url": url, **dict(_fallback_result or {})}

            result = article.extract_article_with_pipeline(
                case["html"],
                case["url"],
                strategy_order=case.get("strategy_order"),
                fallback_extractor=_fallback if fallback_result is not None else None,
                allow_llm_extraction=case.get("allow_llm_extraction", False),
            )
        else:
            raise AssertionError(f"Unknown extraction fixture operation: {operation}")
        actual = {
            "cache_stats": article.get_extraction_cache_stats(),
            "metrics": recorder.events,
            "result": result,
        }
        article.clear_extraction_caches()
        _assert_case(case, actual)


@pytest.mark.asyncio
async def test_article_orchestration_fakes_match_predecessor(monkeypatch: pytest.MonkeyPatch) -> None:
    for case in _load_cases("article_orchestration_fakes"):
        actual = await _run_article_case(case, monkeypatch)
        _assert_case(case, actual)
