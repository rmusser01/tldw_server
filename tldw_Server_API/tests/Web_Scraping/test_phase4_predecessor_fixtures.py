from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
from Helper_Scripts import web_scraping_phase4_fixtures as fixture_generator

from tldw_Server_API.app.core.Watchlists import fetchers
from tldw_Server_API.app.core.Web_Scraping import Article_Extractor_Lib as article
from tldw_Server_API.app.core.Web_Scraping.runtime import (
    FetchRequest,
    FetchResponse,
    PolicyDecision,
    RuntimeRequestContext,
)
from tldw_Server_API.tests.Web_Scraping.phase4_fixture_contracts import (
    assert_predecessor_behavior,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
GENERATOR = REPO_ROOT / "Helper_Scripts" / "web_scraping_phase4_fixtures.py"
FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "phase4"

CASE_KEYS = {
    "article_orchestration_fakes",
    "content",
    "extraction",
    "metadata",
    "router",
    "selectors",
}
PINNED_PREDECESSOR_COMMIT = "c2a1695744032245acdb1cd115dd888586dc9623"

_FIXED_EXTRACTION_ENV = {
    "CLUSTER_LINKAGE": "",
    "EXTRACTOR_CLEAR_CACHES": "",
    "EXTRACTOR_MAX_RETRIES": "0",
    "EXTRACTOR_MAX_WORKERS": "",
    "EXTRACTOR_RETRY_BASE_MS": "0",
    "EXTRACTOR_RETRY_JITTER_MS": "0",
    "REGEX_PII_MASK": "false",
    "SIM_THRESHOLD": "",
    "WORD_COUNT_THRESHOLD": "",
}

_FIXED_SELECTOR_ENV = {
    "WATCHLIST_SELECTOR_MAX_EXPR_LEN": "512",
    "WATCHLIST_SELECTOR_MAX_XPATH_DESCENDANT_STEPS": "12",
    "WATCHLIST_SELECTOR_MAX_XPATH_FUNCTION_CALLS": "8",
    "WATCHLIST_SELECTOR_MAX_XPATH_PREDICATES": "10",
}


def _load_fixture_set(
    fixture_root: Path = FIXTURE_ROOT,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]], dict[str, bytes]]:
    with fixture_generator.fixture_publication_reader(
        fixture_root,
        source_root=REPO_ROOT,
    ) as locked_root:
        manifest_path = locked_root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Missing phase 4 predecessor manifest: {manifest_path}")
        manifest_raw = manifest_path.read_bytes()
        manifest = json.loads(manifest_raw.decode("utf-8"))
        cases: dict[str, list[dict[str, Any]]] = {}
        raw_files = {"manifest.json": manifest_raw}
        for category in sorted(CASE_KEYS):
            case_path = locked_root / manifest["cases"][category]
            if not case_path.is_file():
                pytest.fail(f"Missing predecessor fixture: {case_path.name}")
            raw = case_path.read_bytes()
            payload = json.loads(raw.decode("utf-8"))
            assert payload["category"] == category
            assert isinstance(payload["cases"], list)
            assert payload["cases"]
            cases[category] = payload["cases"]
            raw_files[case_path.name] = raw
    return manifest, cases, raw_files


_FIXTURE_MANIFEST, _FIXTURE_CASES, _FIXTURE_RAW_FILES = _load_fixture_set()


def _load_manifest() -> dict[str, Any]:
    return _FIXTURE_MANIFEST


def _load_cases(category: str) -> list[dict[str, Any]]:
    return _FIXTURE_CASES[category]


def _assert_case(case: Mapping[str, Any], actual: object) -> None:
    try:
        assert_predecessor_behavior(
            actual,
            case["expected"],
            behavior_change=case.get("behavior_change"),
            difference_contract=case.get("difference_contract"),
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

    web_scraper_config = {
        "web_scraper_preflight_analyzers": case.get("preflight_enabled", False),
        "web_scraper_respect_robots": True,
    }
    if "preflight_include_results" in case:
        web_scraper_config["web_scraper_preflight_include_results"] = case["preflight_include_results"]
    config = {"web_scraper": web_scraper_config}
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
            analysis=case.get("preflight_analysis", {}),
        )

    monkeypatch.setattr(
        article.preflight_facade,
        "build_execution_context",
        _build_execution_context,
    )
    monkeypatch.setattr(article.preflight_facade, "run_preflight", _run_preflight)

    decision = _policy_decision(case)
    policy_checker = _FakePolicyChecker(decision, error=case["scenario"] == "policy_error")
    responses: list[FetchResponse | BaseException] = []
    if case["scenario"] in {"lightweight_success", "curl_fallback", "preflight_success"}:
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
    actual: dict[str, Any] = {
        "cache_stats": cache_stats,
        "fetch_requests": [_serialize_request(request) for request in fetch_client.requests],
        "metrics": recorder.events,
        "policy_calls": policy_checker.calls,
        "result": result,
    }
    if case.get("preflight_enabled"):
        actual["preflight_calls"] = preflight_calls
    return actual


def test_phase4_fixture_generator_is_explicit_and_checked_in() -> None:
    assert GENERATOR.is_file(), f"Missing explicit fixture generator: {GENERATOR}"


def test_phase4_fixture_manifest_is_pinned() -> None:
    manifest = _load_manifest()

    assert manifest["schema_version"] == 1
    assert manifest["predecessor_commit"] == PINNED_PREDECESSOR_COMMIT
    assert set(manifest["cases"]) == CASE_KEYS
    assert list(manifest["cases"]) == sorted(manifest["cases"])


def test_phase4_fixture_json_is_canonical_and_complete() -> None:
    manifest = _load_manifest()
    expected_names = {"manifest.json", *manifest["cases"].values()}
    assert set(_FIXTURE_RAW_FILES) == expected_names

    for name, raw in sorted(_FIXTURE_RAW_FILES.items()):
        decoded = raw.decode("ascii")
        payload = json.loads(decoded)
        canonical = json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
        assert decoded == canonical, name


def test_article_orchestration_fixture_pins_enabled_preflight() -> None:
    cases = {fixture_case["name"]: fixture_case for fixture_case in _load_cases("article_orchestration_fakes")}

    legacy_case_names = {
        "curl_falls_back_to_httpx",
        "lightweight_http_success",
        "policy_denial_short_circuits_fetch",
        "policy_error_is_publicly_bounded",
    }
    for name in legacy_case_names:
        policy_config = cases[name]["expected"]["policy_calls"][0]["config"]
        assert "web_scraper_preflight_include_results" not in policy_config

    fixture_case = cases["preflight_success_applies_advice_and_attaches_payload"]
    assert fixture_case["preflight_enabled"] is True
    assert fixture_case["preflight_include_results"] is True
    assert fixture_case["expected"]["preflight_calls"] == {
        "build_execution_context": 1,
        "run_preflight": 1,
    }
    assert fixture_case["expected"]["fetch_requests"][0]["backend"] == "curl"
    assert fixture_case["expected"]["result"]["preflight_analysis"] == {
        "advice": {"backend": "curl", "method": "auto", "notes": ["tls_active"]},
        "analysis": {"results": {"tls": {"status": "active"}}},
    }


def test_missing_manifest_is_a_hard_failure(
    tmp_path: Path,
) -> None:
    def _reject_skip(reason: str) -> None:
        pytest.fail(f"Missing immutable fixtures must fail, not skip: {reason}")

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(pytest, "skip", _reject_skip)
        with pytest.raises(FileNotFoundError, match="Missing phase 4 predecessor manifest"):
            _load_fixture_set(tmp_path)


def test_tagged_fixture_cases_select_explicit_difference_contracts() -> None:
    tagged_cases = {
        case["name"]: (case["behavior_change"], case.get("difference_contract"))
        for category in sorted(CASE_KEYS)
        for case in _load_cases(category)
        if "behavior_change" in case
    }

    assert tagged_cases == {
        "default_regex_is_terminal_in_predecessor": (
            1,
            "change_1_default_regex_non_terminal",
        ),
        "invalid_xpath_error": (7, "change_7_selector_invalid"),
        "policy_error_is_publicly_bounded": (7, "change_7_policy_error"),
        "regex_replace_stdlib_invalid_lookbehind_fallback": (
            4,
            "change_4_selector_regex_failure_returns_original",
        ),
    }


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
    try:
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
                    if case.get("capture_regex_error"):
                        try:
                            result = fetchers.extract_schema_fields(
                                case["html"],
                                case["base_url"],
                                case["rules"],
                            )
                        except re.error:
                            actual = {"outcome": "regex_error", "value": None}
                        else:
                            actual = {
                                "outcome": "returned",
                                "value": {
                                    "cache_stats": fetchers.get_selector_cache_stats(),
                                    "result": result,
                                },
                            }
                        fetchers.clear_selector_caches()
                        _assert_case(case, actual)
                        continue
                    result = fetchers.extract_schema_fields(case["html"], case["base_url"], case["rules"])
                else:
                    raise AssertionError(f"Unknown selector fixture operation: {operation}")
                actual = {
                    "cache_stats": fetchers.get_selector_cache_stats(),
                    "result": result,
                }
                fetchers.clear_selector_caches()
                _assert_case(case, actual)
    finally:
        fetchers.reload_selector_guardrails_from_env()


def test_selector_replay_rejects_regression_to_escaped_regex_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_extract = fetchers.extract_schema_fields
    original_reload = fetchers.reload_selector_guardrails_from_env
    reload_calls = 0

    def _track_reload() -> None:
        nonlocal reload_calls
        reload_calls += 1
        original_reload()

    def _raise_for_regex_failure_case(
        html_text: str,
        base_url: str,
        rules: dict[str, Any],
    ) -> dict[str, Any]:
        if rules.get("name") == "regex_fallback":
            raise re.error("private engine detail")
        return original_extract(html_text, base_url, rules)

    monkeypatch.setattr(fetchers, "reload_selector_guardrails_from_env", _track_reload)
    monkeypatch.setattr(fetchers, "extract_schema_fields", _raise_for_regex_failure_case)

    with pytest.raises(AssertionError, match="Predecessor case failed"):
        test_selector_validation_and_extraction_match_predecessor(monkeypatch)
    assert reload_calls == 2


def test_extraction_replay_overrides_and_restores_regex_pii_mask(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = next(
        case for case in _load_cases("extraction") if case["name"] == "default_regex_is_terminal_in_predecessor"
    )
    monkeypatch.setenv("REGEX_PII_MASK", "true")

    with monkeypatch.context() as environment:
        recorder = _install_metric_recorder(environment)
        _set_environment(environment, _FIXED_EXTRACTION_ENV)
        environment.setattr(article.random, "uniform", lambda *_args, **_kwargs: 0.0)
        article.clear_extraction_caches()
        result = article.extract_article_with_pipeline(
            case["html"],
            case["url"],
            strategy_order=case.get("strategy_order"),
            allow_llm_extraction=case.get("allow_llm_extraction", False),
        )
        actual = {
            "cache_stats": article.get_extraction_cache_stats(),
            "metrics": recorder.events,
            "result": result,
        }
        article.clear_extraction_caches()

    assert os.environ["REGEX_PII_MASK"] == "true"
    _assert_case(case, actual)


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
