"""Pure, deterministic scoring for the offline web-retrieval quality fixture."""

from __future__ import annotations

import json
import math
import re
import statistics
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

from tldw_Server_API.app.core.Evaluations.article_extraction_benchmark import (
    get_accuracy,
    precision_score,
    recall_score,
    string_shingle_matching,
)

FIXTURE_SCHEMA_VERSION = "web-retrieval-quality-fixture-v1"
REPORT_SCHEMA_VERSION = "web-retrieval-quality-report-v1"
_CHARACTER_ESTIMATE_ALGORITHM = "characters-ceil-div4-v1"
ALGORITHM_VERSIONS: Mapping[str, str] = MappingProxyType(
    {
        "budget": "char-utf8-budget-v1",
        "crawl": "ordered-visit-stop-v1",
        "extraction": "token-shingle-f1-v1",
        "provenance": "required-field-recall-v1",
        "search_order": "position-match-v1",
        "token_estimate": _CHARACTER_ESTIMATE_ALGORITHM,
    }
)

__all__ = [
    "ALGORITHM_VERSIONS",
    "FIXTURE_SCHEMA_VERSION",
    "REPORT_SCHEMA_VERSION",
    "FixtureValidationError",
    "evaluate_fixture_suite",
    "load_fixture_suite",
    "render_human_summary",
    "serialize_report",
    "validate_fixture_suite",
]

_TOP_LEVEL_FIELDS = frozenset(
    {"schema_version", "suite_id", "baseline_revision", "cases"}
)
_CASE_FIELDS = frozenset({"id", "kind", "input", "expected", "observed"})
_SUPPORTED_KINDS = frozenset(
    {"extraction", "search_order", "crawl_graph", "provenance"}
)
_REVISION_PATTERN = re.compile(r"[0-9a-f]{40}")


class FixtureValidationError(ValueError):
    """Raised when an offline retrieval-quality fixture violates its schema."""


def load_fixture_suite(path: Path) -> dict[str, Any]:
    """Load and validate one UTF-8 JSON fixture suite."""
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise FixtureValidationError("fixture root must be an object")
    return validate_fixture_suite(value)


def validate_fixture_suite(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a fixture and return a defensive plain-container copy."""
    suite = _require_mapping(value, "fixture")
    _require_exact_fields(suite, _TOP_LEVEL_FIELDS, "fixture")

    if suite["schema_version"] != FIXTURE_SCHEMA_VERSION:
        raise FixtureValidationError(
            f"schema_version must be {FIXTURE_SCHEMA_VERSION!r}"
        )
    suite_id = _require_nonempty_string(suite["suite_id"], "suite_id")
    baseline_revision = _require_nonempty_string(
        suite["baseline_revision"], "baseline_revision"
    )
    if _REVISION_PATTERN.fullmatch(baseline_revision) is None:
        raise FixtureValidationError(
            "baseline_revision must be a 40-character lowercase hexadecimal revision"
        )

    raw_cases = suite["cases"]
    if not isinstance(raw_cases, list):
        raise FixtureValidationError("cases must be a list")

    cases: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, raw_case in enumerate(raw_cases):
        case = _validate_case(raw_case, index)
        case_id = case["id"]
        if case_id in seen_ids:
            raise FixtureValidationError(f"duplicate case id: {case_id}")
        seen_ids.add(case_id)
        cases.append(case)

    return {
        "schema_version": FIXTURE_SCHEMA_VERSION,
        "suite_id": suite_id,
        "baseline_revision": baseline_revision,
        "cases": cases,
    }


def evaluate_fixture_suite(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and score a fixture using only deterministic local operations."""
    suite = validate_fixture_suite(value)
    case_reports = sorted(
        (_evaluate_case(case) for case in suite["cases"]),
        key=lambda case: case["id"],
    )
    total_characters = sum(case["budget"]["characters"] for case in case_reports)
    total_utf8_bytes = sum(case["budget"]["utf8_bytes"] for case in case_reports)
    mean_case_score = (
        statistics.mean(case["score"] for case in case_reports)
        if case_reports
        else 0.0
    )

    return {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "fixture_schema_version": FIXTURE_SCHEMA_VERSION,
        "suite_id": suite["suite_id"],
        "baseline_revision": suite["baseline_revision"],
        "algorithm_versions": dict(ALGORITHM_VERSIONS),
        "cases": case_reports,
        "summary": {
            "case_count": len(case_reports),
            "mean_case_score": _round_metric(mean_case_score),
            "total_characters": total_characters,
            "total_utf8_bytes": total_utf8_bytes,
            "estimated_tokens": _token_estimate(total_characters),
        },
    }


def serialize_report(report: Mapping[str, Any]) -> str:
    """Serialize a report to byte-stable UTF-8 JSON text."""
    return (
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )


def render_human_summary(report: Mapping[str, Any]) -> str:
    """Render the stable, timestamp-free human report grammar."""
    cases = report["cases"]
    summary = report["summary"]
    lines = [
        (
            f"suite={report['suite_id']} baseline={report['baseline_revision']} "
            f"cases={summary['case_count']}"
        )
    ]
    for case in sorted(cases, key=lambda item: item["id"]):
        budget = case["budget"]
        lines.append(
            f"case={case['id']} kind={case['kind']} score={case['score']:.6f} "
            f"characters={budget['characters']} utf8_bytes={budget['utf8_bytes']}"
        )
    lines.append(
        f"total mean_case_score={summary['mean_case_score']:.6f} "
        f"characters={summary['total_characters']} "
        f"utf8_bytes={summary['total_utf8_bytes']} "
        f"estimated_tokens={summary['estimated_tokens']['value']} authoritative=false"
    )
    return "\n".join(lines)


def _validate_case(raw_case: object, index: int) -> dict[str, Any]:
    """Validate one fixture case and return its normalized copy."""
    context = f"cases[{index}]"
    case = _require_mapping(raw_case, context)
    _require_exact_fields(case, _CASE_FIELDS, context)
    case_id = _require_nonempty_string(case["id"], f"{context}.id")
    kind = _require_nonempty_string(case["kind"], f"{context}.kind")
    if kind not in _SUPPORTED_KINDS:
        raise FixtureValidationError(f"{context} has unsupported kind: {kind}")

    validators = {
        "extraction": _validate_extraction,
        "search_order": _validate_search_order,
        "crawl_graph": _validate_crawl_graph,
        "provenance": _validate_provenance,
    }
    input_value, expected, observed = validators[kind](case, context)
    return {
        "id": case_id,
        "kind": kind,
        "input": input_value,
        "expected": expected,
        "observed": observed,
    }


def _validate_extraction(
    case: Mapping[str, Any], context: str
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Validate the input, expected, and observed extraction sections."""
    input_value = _section(case, "input", {"url", "html"}, context)
    expected = _section(case, "expected", {"text"}, context)
    observed = _section(case, "observed", {"text", "output_text"}, context)
    return (
        {
            "url": _require_url(input_value["url"], f"{context}.input.url"),
            "html": _require_string(input_value["html"], f"{context}.input.html"),
        },
        {"text": _require_string(expected["text"], f"{context}.expected.text")},
        {
            "text": _require_string(observed["text"], f"{context}.observed.text"),
            "output_text": _require_string(
                observed["output_text"], f"{context}.observed.output_text"
            ),
        },
    )


def _validate_search_order(
    case: Mapping[str, Any], context: str
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Validate search-provider inputs and ordered URL observations."""
    input_value = _section(case, "input", {"provider_results"}, context)
    expected = _section(case, "expected", {"ordered_urls"}, context)
    observed = _section(case, "observed", {"ordered_urls", "output_text"}, context)
    raw_results = input_value["provider_results"]
    if not isinstance(raw_results, list) or not raw_results:
        raise FixtureValidationError(
            f"{context}.input.provider_results must be a non-empty list"
        )
    provider_results = []
    for index, raw_result in enumerate(raw_results):
        result_context = f"{context}.input.provider_results[{index}]"
        result = _require_mapping(raw_result, result_context)
        _require_exact_fields(result, {"provider", "url", "title"}, result_context)
        provider_results.append(
            {
                "provider": _require_nonempty_string(
                    result["provider"], f"{result_context}.provider"
                ),
                "url": _require_url(result["url"], f"{result_context}.url"),
                "title": _require_string(result["title"], f"{result_context}.title"),
            }
        )
    return (
        {"provider_results": provider_results},
        {
            "ordered_urls": _require_url_list(
                expected["ordered_urls"], f"{context}.expected.ordered_urls"
            )
        },
        {
            "ordered_urls": _require_url_list(
                observed["ordered_urls"], f"{context}.observed.ordered_urls"
            ),
            "output_text": _require_string(
                observed["output_text"], f"{context}.observed.output_text"
            ),
        },
    )


def _validate_crawl_graph(
    case: Mapping[str, Any], context: str
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Validate a bounded crawl graph and its observed traversal."""
    input_value = _section(case, "input", {"start_url", "links", "page_limit"}, context)
    expected = _section(case, "expected", {"visited_urls", "stop_reason"}, context)
    observed = _section(
        case, "observed", {"visited_urls", "stop_reason", "output_text"}, context
    )
    page_limit = input_value["page_limit"]
    if type(page_limit) is not int or page_limit <= 0:
        raise FixtureValidationError(f"{context}.input.page_limit must be a positive integer")
    raw_links = _require_mapping(input_value["links"], f"{context}.input.links")
    links: dict[str, list[str]] = {}
    for raw_url, raw_destinations in raw_links.items():
        url = _require_url(raw_url, f"{context}.input.links key")
        links[url] = _require_url_list(
            raw_destinations,
            f"{context}.input.links[{url!r}]",
            allow_empty=True,
        )
    return (
        {
            "start_url": _require_url(
                input_value["start_url"], f"{context}.input.start_url"
            ),
            "links": links,
            "page_limit": page_limit,
        },
        {
            "visited_urls": _require_url_list(
                expected["visited_urls"], f"{context}.expected.visited_urls"
            ),
            "stop_reason": _require_nonempty_string(
                expected["stop_reason"], f"{context}.expected.stop_reason"
            ),
        },
        {
            "visited_urls": _require_url_list(
                observed["visited_urls"], f"{context}.observed.visited_urls"
            ),
            "stop_reason": _require_nonempty_string(
                observed["stop_reason"], f"{context}.observed.stop_reason"
            ),
            "output_text": _require_string(
                observed["output_text"], f"{context}.observed.output_text"
            ),
        },
    )


def _validate_provenance(
    case: Mapping[str, Any], context: str
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Validate provenance requirements and a JSON-compatible record."""
    input_value = _section(case, "input", {"required_fields"}, context)
    expected = _section(case, "expected", set(), context)
    observed = _section(case, "observed", {"record", "output_text"}, context)
    required_fields = _require_string_list(
        input_value["required_fields"],
        f"{context}.input.required_fields",
    )
    if len(set(required_fields)) != len(required_fields):
        raise FixtureValidationError(f"{context}.input.required_fields must be unique")
    record = _require_mapping(observed["record"], f"{context}.observed.record")
    copied_record = _copy_json_value(record, f"{context}.observed.record")
    if not isinstance(copied_record, dict):
        raise FixtureValidationError(f"{context}.observed.record must be an object")
    return (
        {"required_fields": required_fields},
        dict(expected),
        {
            "record": copied_record,
            "output_text": _require_string(
                observed["output_text"], f"{context}.observed.output_text"
            ),
        },
    )


def _evaluate_case(case: Mapping[str, Any]) -> dict[str, Any]:
    """Score one normalized case and attach its output budget."""
    evaluators = {
        "extraction": _score_extraction,
        "search_order": _score_search_order,
        "crawl_graph": _score_crawl_graph,
        "provenance": _score_provenance,
    }
    metrics, score = evaluators[case["kind"]](case)
    output_text = case["observed"]["output_text"]
    return {
        "id": case["id"],
        "kind": case["kind"],
        "score": _round_metric(score),
        "metrics": metrics,
        "budget": _output_budget(output_text),
    }


def _score_extraction(case: Mapping[str, Any]) -> tuple[dict[str, Any], float]:
    """Score extraction text with the established shingle metrics."""
    true = case["expected"]["text"]
    predicted = case["observed"]["text"]
    tp, fp, fn = string_shingle_matching(true=true, pred=predicted)
    precision = precision_score(tp, fp, fn)
    recall = recall_score(tp, fp, fn)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return (
        {
            "precision": _round_metric(precision),
            "recall": _round_metric(recall),
            "f1": _round_metric(f1),
            "accuracy": _round_metric(get_accuracy(true=true, pred=predicted)),
        },
        f1,
    )


def _score_search_order(case: Mapping[str, Any]) -> tuple[dict[str, Any], float]:
    """Score position agreement between expected and observed URLs."""
    expected = case["expected"]["ordered_urls"]
    observed = case["observed"]["ordered_urls"]
    ratio = sum(left == right for left, right in zip(expected, observed)) / max(
        len(expected), len(observed)
    )
    return (
        {
            "position_match_ratio": _round_metric(ratio),
            "exact_order_match": expected == observed,
        },
        ratio,
    )


def _score_crawl_graph(case: Mapping[str, Any]) -> tuple[dict[str, Any], float]:
    """Score crawl coverage, visit order, and terminal reason."""
    expected = case["expected"]["visited_urls"]
    observed = case["observed"]["visited_urls"]
    recall = len(set(expected) & set(observed)) / len(set(expected))
    position_ratio = sum(left == right for left, right in zip(expected, observed)) / max(
        len(expected), len(observed)
    )
    stop_match = case["expected"]["stop_reason"] == case["observed"]["stop_reason"]
    score = statistics.mean((recall, position_ratio, float(stop_match)))
    return (
        {
            "visited_url_recall": _round_metric(recall),
            "position_match_ratio": _round_metric(position_ratio),
            "stop_reason_match": stop_match,
        },
        score,
    )


def _score_provenance(case: Mapping[str, Any]) -> tuple[dict[str, Any], float]:
    """Score recall of required non-empty provenance fields."""
    required_fields = case["input"]["required_fields"]
    record = case["observed"]["record"]
    present = sum(
        field in record and _is_nonempty(record[field]) for field in required_fields
    )
    recall = present / len(required_fields)
    return ({"required_field_recall": _round_metric(recall)}, recall)


def _output_budget(output_text: str) -> dict[str, Any]:
    """Measure characters, UTF-8 bytes, and estimated tokens."""
    characters = len(output_text)
    return {
        "characters": characters,
        "utf8_bytes": len(output_text.encode("utf-8")),
        "estimated_tokens": _token_estimate(characters),
    }


def _token_estimate(characters: int) -> dict[str, Any]:
    """Estimate tokens with the declared non-authoritative algorithm."""
    return {
        "value": (characters + 3) // 4,
        "algorithm": ALGORITHM_VERSIONS["token_estimate"],
        "authoritative": False,
    }


def _section(
    case: Mapping[str, Any],
    name: str,
    fields: set[str],
    context: str,
) -> Mapping[str, Any]:
    """Return a case section after enforcing its exact field set."""
    section = _require_mapping(case[name], f"{context}.{name}")
    _require_exact_fields(section, fields, f"{context}.{name}")
    return section


def _require_mapping(value: object, context: str) -> Mapping[str, Any]:
    """Require an object-like mapping with string keys."""
    if not isinstance(value, Mapping):
        raise FixtureValidationError(f"{context} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise FixtureValidationError(f"{context} keys must be strings")
    return value


def _require_exact_fields(
    value: Mapping[str, Any], required: set[str] | frozenset[str], context: str
) -> None:
    """Reject missing or unknown fields in a fixture object."""
    fields = set(value)
    missing = required - fields
    unknown = fields - required
    if missing:
        raise FixtureValidationError(
            f"{context} missing fields: {', '.join(sorted(missing))}"
        )
    if unknown:
        raise FixtureValidationError(
            f"{context} unknown fields: {', '.join(sorted(unknown))}"
        )


def _require_string(value: object, context: str) -> str:
    """Require and return a string value."""
    if not isinstance(value, str):
        raise FixtureValidationError(f"{context} must be a string")
    return value


def _require_nonempty_string(value: object, context: str) -> str:
    """Require and return a non-empty string value."""
    string = _require_string(value, context)
    if not string:
        raise FixtureValidationError(f"{context} must be non-empty")
    return string


def _require_url(value: object, context: str) -> str:
    """Require the fixture's non-empty URL string representation."""
    return _require_nonempty_string(value, context)


def _require_string_list(value: object, context: str) -> list[str]:
    """Require a non-empty list of non-empty strings."""
    if not isinstance(value, list) or not value:
        raise FixtureValidationError(f"{context} must be a non-empty list")
    return [
        _require_nonempty_string(item, f"{context}[{index}]")
        for index, item in enumerate(value)
    ]


def _require_url_list(
    value: object, context: str, *, allow_empty: bool = False
) -> list[str]:
    """Require a URL-string list, optionally permitting no entries."""
    if not isinstance(value, list) or (not value and not allow_empty):
        qualifier = "a list" if allow_empty else "a non-empty list"
        raise FixtureValidationError(f"{context} must be {qualifier}")
    return [
        _require_url(item, f"{context}[{index}]")
        for index, item in enumerate(value)
    ]


def _copy_json_value(value: object, context: str) -> Any:
    """Recursively copy a finite JSON-compatible value."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise FixtureValidationError(f"{context} must not contain non-finite numbers")
        return value
    if isinstance(value, list):
        return [
            _copy_json_value(item, f"{context}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, Mapping):
        copied: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise FixtureValidationError(f"{context} keys must be strings")
            copied[key] = _copy_json_value(item, f"{context}.{key}")
        return copied
    raise FixtureValidationError(f"{context} contains a non-JSON value")


def _is_nonempty(value: object) -> bool:
    """Return whether a provenance value counts as present."""
    return value is not None and value != "" and value != [] and value != {}


def _round_metric(value: float) -> float:
    """Round a finite metric to the report's fixed precision."""
    if not math.isfinite(value):
        raise FixtureValidationError("reported metrics must be finite")
    return round(value, 6)
