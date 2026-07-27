from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
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


class _DifferenceToken(Enum):
    ANY_PATH = "<ANY_PATH>"
    MISSING = "<MISSING>"


_ANY_PATH = _DifferenceToken.ANY_PATH
_MISSING = _DifferenceToken.MISSING
_PathPart = str | int
_PathPatternPart = _PathPart | _DifferenceToken


@dataclass(frozen=True)
class _Difference:
    path: tuple[_PathPart, ...]
    actual: object
    expected: object


@dataclass(frozen=True)
class _DifferenceRule:
    path: tuple[_PathPatternPart, ...]
    description: str
    validator: Callable[[_Difference], bool]


@dataclass(frozen=True)
class _DifferenceContract:
    behavior_change: int
    rules: tuple[_DifferenceRule, ...]


def _changed_from_to(expected: str, actual: str) -> Callable[[_Difference], bool]:
    return lambda difference: difference.expected == expected and difference.actual == actual


def _regex_becomes_non_terminal(difference: _Difference) -> bool:
    return (
        difference.expected == "regex"
        and isinstance(difference.actual, str)
        and difference.actual in {"cluster", "llm", "trafilatura"}
    )


def _regex_trace_reason_becomes_non_terminal(difference: _Difference) -> bool:
    return (
        difference.expected == "regex_extracted"
        and isinstance(difference.actual, str)
        and difference.actual in {"regex_enriched", "regex_non_terminal"}
    )


def _regex_trace_status_becomes_non_terminal(difference: _Difference) -> bool:
    return (
        difference.expected == "success"
        and isinstance(difference.actual, str)
        and difference.actual in {"continued", "enriched"}
    )


def _is_new_downstream_trace_step(difference: _Difference) -> bool:
    if difference.expected is not _MISSING or not isinstance(difference.actual, Mapping):
        return False
    strategy = difference.actual.get("strategy")
    status = difference.actual.get("status")
    return (
        isinstance(strategy, str)
        and strategy in {"cluster", "llm", "trafilatura"}
        and isinstance(status, str)
        and status in {"failed", "skipped", "success"}
    )


def _is_new_regex_or_downstream_metric(difference: _Difference) -> bool:
    if difference.expected is not _MISSING or not isinstance(difference.actual, Mapping):
        return False
    labels = difference.actual.get("labels")
    strategy = labels.get("strategy") if isinstance(labels, Mapping) else None
    name = difference.actual.get("name")
    return (
        isinstance(strategy, str)
        and strategy in {"cluster", "llm", "regex", "trafilatura"}
        and isinstance(name, str)
        and name.startswith("extraction_")
    )


DIFFERENCE_CONTRACTS = {
    "change_1_default_regex_non_terminal": _DifferenceContract(
        behavior_change=1,
        rules=(
            _DifferenceRule(
                path=("result", "extraction_strategy"),
                description="regex no longer terminates the default extraction pipeline",
                validator=_regex_becomes_non_terminal,
            ),
            _DifferenceRule(
                path=("result", "extraction_trace", _ANY_PATH, "reason"),
                description="the regex trace records enrichment rather than terminal extraction",
                validator=_regex_trace_reason_becomes_non_terminal,
            ),
            _DifferenceRule(
                path=("result", "extraction_trace", _ANY_PATH, "status"),
                description="the regex trace records continued extraction",
                validator=_regex_trace_status_becomes_non_terminal,
            ),
            _DifferenceRule(
                path=("result", "extraction_trace", _ANY_PATH),
                description="a downstream strategy runs after regex enrichment",
                validator=_is_new_downstream_trace_step,
            ),
            _DifferenceRule(
                path=("metrics", _ANY_PATH, "labels", "status"),
                description="regex metrics record continued extraction",
                validator=_regex_trace_status_becomes_non_terminal,
            ),
            _DifferenceRule(
                path=("metrics", _ANY_PATH),
                description="regex enrichment emits downstream extraction metrics",
                validator=_is_new_regex_or_downstream_metric,
            ),
        ),
    ),
    "change_7_policy_error": _DifferenceContract(
        behavior_change=7,
        rules=(
            _DifferenceRule(
                path=("result", "error"),
                description="policy exception text is replaced by a stable public code",
                validator=_changed_from_to(
                    "Outbound policy evaluation failed. Please contact system administrator.",
                    "policy_error",
                ),
            ),
        ),
    ),
    "change_7_selector_invalid": _DifferenceContract(
        behavior_change=7,
        rules=(
            _DifferenceRule(
                path=("result", "errors", _ANY_PATH, "error"),
                description="selector parser text is replaced by a stable public code",
                validator=_changed_from_to("Invalid expression", "selector_invalid"),
            ),
        ),
    ),
    "change_11_response_too_large": _DifferenceContract(
        behavior_change=11,
        rules=(
            _DifferenceRule(
                path=("result", "error"),
                description="oversized direct responses return the approved stable code",
                validator=_changed_from_to("Upstream response accepted", "response_too_large"),
            ),
        ),
    ),
}


def _collect_differences(
    actual: object,
    expected: object,
    path: tuple[_PathPart, ...] = (),
) -> list[_Difference]:
    if actual is _MISSING or expected is _MISSING:
        return [_Difference(path=path, actual=actual, expected=expected)]
    if isinstance(actual, Mapping) and isinstance(expected, Mapping):
        differences: list[_Difference] = []
        for key in sorted(set(actual) | set(expected), key=str):
            differences.extend(
                _collect_differences(
                    actual.get(key, _MISSING),
                    expected.get(key, _MISSING),
                    (*path, key),
                )
            )
        return differences
    if isinstance(actual, (list, tuple)) and isinstance(expected, (list, tuple)):
        differences = []
        for index in range(max(len(actual), len(expected))):
            differences.extend(
                _collect_differences(
                    actual[index] if index < len(actual) else _MISSING,
                    expected[index] if index < len(expected) else _MISSING,
                    (*path, index),
                )
            )
        return differences
    if actual != expected:
        return [_Difference(path=path, actual=actual, expected=expected)]
    return []


def _path_matches(pattern: tuple[_PathPatternPart, ...], path: tuple[_PathPart, ...]) -> bool:
    return len(pattern) == len(path) and all(
        expected_part is _ANY_PATH or expected_part == actual_part for expected_part, actual_part in zip(pattern, path)
    )


def _format_path(path: tuple[_PathPart, ...]) -> str:
    formatted = "$"
    for part in path:
        formatted += f"[{part}]" if isinstance(part, int) else f".{part}"
    return formatted


def assert_predecessor_behavior(
    actual: object,
    expected: object,
    *,
    behavior_change: int | None = None,
    difference_contract: str | None = None,
) -> None:
    """Assert parity or validate every difference against one approved change contract."""
    if behavior_change is not None and (
        type(behavior_change) is not int or behavior_change not in APPROVED_BEHAVIOR_CHANGES
    ):
        raise ValueError("behavior_change must be one integer in range 1..11")
    if behavior_change is None:
        if difference_contract is not None:
            raise ValueError("difference_contract requires behavior_change")
        assert actual == expected
        return
    assert difference_contract is not None, "A difference contract is required for an approved change"
    try:
        contract = DIFFERENCE_CONTRACTS[difference_contract]
    except KeyError as exc:
        raise ValueError(f"Unknown difference contract: {difference_contract}") from exc
    if contract.behavior_change != behavior_change:
        raise ValueError(
            f"Difference contract {difference_contract!r} belongs to behavior change "
            f"{contract.behavior_change}, not {behavior_change}"
        )

    for difference in _collect_differences(actual, expected):
        matching_rules = [rule for rule in contract.rules if _path_matches(rule.path, difference.path)]
        path = _format_path(difference.path)
        assert matching_rules, f"Difference at {path} is not covered by difference contract {difference_contract!r}"
        assert (
            len(matching_rules) == 1
        ), f"Difference at {path} is covered by more than one rule in {difference_contract!r}"
        rule = matching_rules[0]
        assert rule.validator(difference), f"Difference at {path} violates contract rule: {rule.description}"


def _load_manifest() -> dict[str, Any]:
    if not MANIFEST.is_file():
        raise FileNotFoundError(f"Missing phase 4 predecessor manifest: {MANIFEST}")
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def _load_cases(category: str) -> list[dict[str, Any]]:
    manifest = _load_manifest()
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
    manifest = _load_manifest()
    expected_names = {"manifest.json", *manifest["cases"].values()}
    assert {path.name for path in FIXTURE_ROOT.glob("*.json")} == expected_names

    for path in sorted(FIXTURE_ROOT.glob("*.json")):
        raw = path.read_bytes()
        decoded = raw.decode("ascii")
        payload = json.loads(decoded)
        canonical = json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
        assert decoded == canonical, path.name


def test_missing_manifest_is_a_hard_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    missing_manifest = tmp_path / "manifest.json"
    monkeypatch.setitem(globals(), "MANIFEST", missing_manifest)

    def _reject_skip(reason: str) -> None:
        pytest.fail(f"Missing immutable fixtures must fail, not skip: {reason}")

    monkeypatch.setattr(pytest, "skip", _reject_skip)
    with pytest.raises(FileNotFoundError, match="Missing phase 4 predecessor manifest"):
        _load_manifest()


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
    }


def test_differential_helper_requires_a_tag_for_a_difference() -> None:
    with pytest.raises(AssertionError):
        assert_predecessor_behavior({"value": "current"}, {"value": "predecessor"})


def test_differential_helper_accepts_none_for_equal_values() -> None:
    assert_predecessor_behavior({"value": "same"}, {"value": "same"}, behavior_change=None)


def test_differential_helper_rejects_valid_change_without_a_contract() -> None:
    with pytest.raises(AssertionError, match="difference contract"):
        assert_predecessor_behavior(
            {"result": {"url": "https://current.example"}},
            {"result": {"url": "https://predecessor.example"}},
            behavior_change=1,
        )


@pytest.mark.parametrize(
    ("behavior_change", "difference_contract", "actual", "expected"),
    [
        (
            1,
            "change_1_default_regex_non_terminal",
            {"result": {"extraction_strategy": "trafilatura"}},
            {"result": {"extraction_strategy": "regex"}},
        ),
        (
            11,
            "change_11_response_too_large",
            {"result": {"error": "response_too_large"}},
            {"result": {"error": "Upstream response accepted"}},
        ),
    ],
)
def test_differential_helper_accepts_semantically_constrained_boundary_changes(
    behavior_change: int,
    difference_contract: str,
    actual: object,
    expected: object,
) -> None:
    assert_predecessor_behavior(
        actual,
        expected,
        behavior_change=behavior_change,
        difference_contract=difference_contract,
    )


def test_differential_helper_accepts_a_covered_planned_difference() -> None:
    assert_predecessor_behavior(
        {
            "result": {
                "error": "policy_error",
                "url": "https://example.com/policy-error",
            }
        },
        {
            "result": {
                "error": "Outbound policy evaluation failed. Please contact system administrator.",
                "url": "https://example.com/policy-error",
            }
        },
        behavior_change=7,
        difference_contract="change_7_policy_error",
    )


def test_differential_helper_rejects_an_extra_unrelated_difference() -> None:
    with pytest.raises(AssertionError, match="not covered"):
        assert_predecessor_behavior(
            {
                "result": {
                    "error": "policy_error",
                    "url": "https://unrelated.example/policy-error",
                }
            },
            {
                "result": {
                    "error": ("Outbound policy evaluation failed. " "Please contact system administrator."),
                    "url": "https://example.com/policy-error",
                }
            },
            behavior_change=7,
            difference_contract="change_7_policy_error",
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
