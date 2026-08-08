from __future__ import annotations

import json
from copy import deepcopy
from enum import Enum
from pathlib import Path

import pytest
from Helper_Scripts import web_scraping_phase4_fixtures as fixture_generator

from tldw_Server_API.tests.Web_Scraping.phase4_fixture_change1_contract import (
    _validate_change_1_profile,
)
from tldw_Server_API.tests.Web_Scraping.phase4_fixture_contracts import (
    assert_predecessor_behavior,
)

_CHANGE_1_CONTRACT = "change_1_default_regex_non_terminal"
_CHANGE_4_CONTRACT = "change_4_selector_regex_failure_returns_original"
_REPO_ROOT = Path(__file__).resolve().parents[3]
_FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "phase4"

pytestmark = pytest.mark.unit


def _load_selector_fixture() -> dict[str, object]:
    with fixture_generator.fixture_publication_reader(
        _FIXTURE_ROOT,
        source_root=_REPO_ROOT,
    ) as locked_root:
        manifest = json.loads((locked_root / "manifest.json").read_text(encoding="ascii"))
        return json.loads((locked_root / manifest["cases"]["selectors"]).read_text(encoding="ascii"))


class _StringSubclass(str):
    pass


class _StrategyLabel(str, Enum):
    TRAFILATURA = "trafilatura"


def _counter(strategy: str, status: str) -> dict[str, object]:
    return {
        "emitter": "log_counter",
        "kind": "counter",
        "labels": {"status": status, "strategy": strategy},
        "name": "extraction_strategy_total",
    }


def _duration(strategy: str, status: str) -> dict[str, object]:
    return {
        "emitter": "observe_histogram",
        "kind": "histogram",
        "labels": {"status": status, "strategy": strategy},
        "name": "extraction_strategy_duration_seconds",
        "value": "<TIMING>",
    }


def _content_length(strategy: str, value: int = 31) -> dict[str, object]:
    return {
        "emitter": "observe_histogram",
        "kind": "histogram",
        "labels": {"strategy": strategy},
        "name": "extraction_content_length_bytes",
        "value": value,
    }


def _cluster_counter(name: str, labels: dict[str, str]) -> dict[str, object]:
    return {
        "emitter": "increment_counter",
        "kind": "counter",
        "labels": labels,
        "name": name,
    }


def _predecessor_change_1_profile() -> dict[str, object]:
    return {
        "metrics": [
            _counter("jsonld", "failed"),
            _duration("jsonld", "failed"),
            _counter("schema", "skipped"),
            _duration("schema", "skipped"),
            _counter("regex", "success"),
            _duration("regex", "success"),
            _content_length("regex"),
        ],
        "result": {
            "extraction_strategy": "regex",
            "extraction_trace": [
                {"reason": "jsonld_no_content", "status": "failed", "strategy": "jsonld"},
                {
                    "reason": "no_schema_rules_or_handler",
                    "status": "skipped",
                    "strategy": "schema",
                },
                {"reason": "regex_extracted", "status": "success", "strategy": "regex"},
            ],
        },
    }


def _coherent_change_1_profile() -> tuple[dict[str, object], dict[str, object]]:
    expected = _predecessor_change_1_profile()
    actual = deepcopy(expected)
    result = actual["result"]
    assert isinstance(result, dict)
    trace = result["extraction_trace"]
    assert isinstance(trace, list)
    metrics = actual["metrics"]
    assert isinstance(metrics, list)

    result["extraction_strategy"] = "trafilatura"
    trace[2] = {"reason": "regex_enriched", "status": "enriched", "strategy": "regex"}
    trace.extend(
        [
            {
                "detail": "cluster_no_blocks",
                "reason": "cluster_no_content",
                "status": "failed",
                "strategy": "cluster",
            },
            {"reason": "extracted", "status": "success", "strategy": "trafilatura"},
        ]
    )
    metrics[4] = _counter("regex", "enriched")
    metrics[5] = _duration("regex", "enriched")
    metrics.extend(
        [
            _counter("cluster", "failed"),
            _duration("cluster", "failed"),
            _counter("trafilatura", "success"),
            _duration("trafilatura", "success"),
            _content_length("trafilatura", 33),
        ]
    )
    return actual, expected


def _coherent_cluster_success_change_1_profile() -> tuple[dict[str, object], dict[str, object]]:
    expected = _predecessor_change_1_profile()
    expected["cache_stats"] = {"cluster_embedding_cache_size": 0}
    actual = deepcopy(expected)
    cache_stats = actual["cache_stats"]
    result = actual["result"]
    assert isinstance(cache_stats, dict)
    assert isinstance(result, dict)
    trace = result["extraction_trace"]
    metrics = actual["metrics"]
    assert isinstance(trace, list)
    assert isinstance(metrics, list)

    cache_stats["cluster_embedding_cache_size"] = 1
    result.update(
        {
            "cluster_blocks": ["Contact predecessor@example.com"],
            "cluster_block_count": 1,
            "cluster_prefiltered_count": 1,
            "cluster_total_blocks": 1,
            "cluster_cluster_count": 1,
            "cluster_method": "greedy",
            "cluster_similarity_threshold": 0.4,
            "cluster_word_threshold": 8,
            "extraction_strategy": "cluster",
        }
    )
    trace[2] = {"reason": "regex_enriched", "status": "enriched", "strategy": "regex"}
    trace.append(
        {
            "detail": "cluster_blocks=1",
            "reason": "cluster_extracted",
            "status": "success",
            "strategy": "cluster",
        }
    )
    metrics[4] = _counter("regex", "enriched")
    metrics[5] = _duration("regex", "enriched")
    metrics.extend(
        [
            _cluster_counter("extraction_cluster_total", {"status": "started"}),
            _cluster_counter(
                "extraction_cluster_cache_total",
                {"cache": "embedding", "result": "miss"},
            ),
            _cluster_counter(
                "extraction_cluster_cache_total",
                {"cache": "embedding", "result": "hit"},
            ),
            _cluster_counter("extraction_cluster_total", {"status": "success"}),
            _counter("cluster", "success"),
            _duration("cluster", "success"),
            _content_length("cluster"),
        ]
    )
    return actual, expected


def _coherent_fresh_multi_block_cluster_success_change_1_profile() -> tuple[dict[str, object], dict[str, object]]:
    actual, expected = _coherent_cluster_success_change_1_profile()
    cache_stats = actual["cache_stats"]
    result = actual["result"]
    metrics = actual["metrics"]
    assert isinstance(cache_stats, dict)
    assert isinstance(result, dict)
    assert isinstance(metrics, list)

    blocks = [
        "First distinct article block with enough words for canonical clustering.",
        "Second distinct article block with enough words for canonical clustering.",
    ]
    cache_stats["cluster_embedding_cache_size"] = 3
    result.update(
        {
            "cluster_blocks": blocks,
            "cluster_block_count": 2,
            "cluster_prefiltered_count": 2,
            "cluster_total_blocks": 2,
            "cluster_cluster_count": 1,
        }
    )
    trace = result["extraction_trace"]
    assert isinstance(trace, list)
    terminal = trace[-1]
    assert isinstance(terminal, dict)
    terminal["detail"] = "cluster_blocks=2"
    metrics[7:11] = [
        _cluster_counter("extraction_cluster_total", {"status": "started"}),
        _cluster_counter(
            "extraction_cluster_cache_total",
            {"cache": "embedding", "result": "miss"},
        ),
        _cluster_counter(
            "extraction_cluster_cache_total",
            {"cache": "embedding", "result": "miss"},
        ),
        _cluster_counter(
            "extraction_cluster_cache_total",
            {"cache": "embedding", "result": "miss"},
        ),
        _cluster_counter("extraction_cluster_total", {"status": "success"}),
    ]
    return actual, expected


def _assert_change_1(actual: object, expected: object) -> None:
    assert_predecessor_behavior(
        actual,
        expected,
        behavior_change=1,
        difference_contract=_CHANGE_1_CONTRACT,
    )


def _change_4_profiles() -> tuple[dict[str, object], dict[str, object]]:
    actual = {
        "outcome": "returned",
        "value": {
            "cache_stats": {
                "selector_css_cache_size": 0,
                "selector_xpath_cache_size": 2,
            },
            "result": {
                "content": "AB123",
                "extraction_successful": True,
                "schema_fields": {"content": "AB123"},
                "schema_name": "regex_fallback",
                "url": "https://example.com/regex-fallback",
            },
        },
    }
    expected = {"outcome": "regex_error", "value": None}
    return actual, expected


def _assert_change_4(actual: object, expected: object) -> None:
    assert_predecessor_behavior(
        actual,
        expected,
        behavior_change=4,
        difference_contract=_CHANGE_4_CONTRACT,
    )


def test_selector_fixture_pins_format_map_and_unknown_transform_parity() -> None:
    payload = _load_selector_fixture()
    cases = {fixture_case["name"]: fixture_case for fixture_case in payload["cases"]}

    assert cases["computed_template_mapping_and_attribute_compatibility"]["expected"]["result"]["schema_fields"] == {
        "author": {"name": "Ada"},
        "byline": "By Ada|type=<class 'str'>",
        "title": "Headline",
    }
    assert cases["unknown_prepend_append_transform_noops"]["expected"]["result"]["schema_fields"] == {"value": "x"}


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


def test_differential_helper_accepts_change_11_boundary_contract() -> None:
    assert_predecessor_behavior(
        {"result": {"error": "response_too_large"}},
        {"result": {"error": "Upstream response accepted"}},
        behavior_change=11,
        difference_contract="change_11_response_too_large",
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


def test_tagged_contract_rejects_str_subclass_at_approved_path() -> None:
    with pytest.raises(AssertionError, match="Strict JSON validation failed"):
        assert_predecessor_behavior(
            {"result": {"error": _StringSubclass("policy_error")}},
            {"result": {"error": ("Outbound policy evaluation failed. " "Please contact system administrator.")}},
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


def test_change_4_selector_regex_failure_contract_accepts_exact_transition() -> None:
    actual, expected = _change_4_profiles()

    _assert_change_4(actual, expected)


def test_change_4_selector_regex_failure_contract_rejects_wrong_outcome() -> None:
    actual, expected = _change_4_profiles()
    actual["outcome"] = "regex_invalid"

    with pytest.raises(AssertionError, match="violates contract rule"):
        _assert_change_4(actual, expected)


def test_change_4_selector_regex_failure_contract_rejects_cache_profile_mutation() -> None:
    actual, expected = _change_4_profiles()
    value = actual["value"]
    assert isinstance(value, dict)
    cache_stats = value["cache_stats"]
    assert isinstance(cache_stats, dict)
    cache_stats["selector_xpath_cache_size"] = 3

    with pytest.raises(AssertionError, match="profile"):
        _assert_change_4(actual, expected)


def test_change_4_selector_regex_failure_contract_rejects_bool_cache_mutation() -> None:
    actual, expected = _change_4_profiles()
    value = actual["value"]
    assert isinstance(value, dict)
    cache_stats = value["cache_stats"]
    assert isinstance(cache_stats, dict)
    cache_stats["selector_css_cache_size"] = False

    with pytest.raises(AssertionError, match="profile"):
        _assert_change_4(actual, expected)


def test_change_4_selector_regex_failure_contract_rejects_result_profile_mutation() -> None:
    actual, expected = _change_4_profiles()
    value = actual["value"]
    assert isinstance(value, dict)
    result = value["result"]
    assert isinstance(result, dict)
    result["content"] = "changed"

    with pytest.raises(AssertionError, match="profile"):
        _assert_change_4(actual, expected)


def test_change_4_selector_regex_failure_contract_rejects_uncovered_difference() -> None:
    actual, expected = _change_4_profiles()
    actual["detail"] = "raw engine text"

    with pytest.raises(AssertionError, match="not covered"):
        _assert_change_4(actual, expected)


def test_change_1_contract_accepts_explicit_predecessor_equality_profile() -> None:
    expected = _predecessor_change_1_profile()
    _assert_change_1(deepcopy(expected), expected)


def test_change_1_contract_accepts_complete_coherent_transition() -> None:
    actual, expected = _coherent_change_1_profile()
    _assert_change_1(actual, expected)


def test_change_1_contract_accepts_exact_cluster_success_profile() -> None:
    actual, expected = _coherent_cluster_success_change_1_profile()

    _assert_change_1(actual, expected)


def test_change_1_contract_accepts_fresh_multi_block_cluster_success_profile() -> None:
    actual, expected = _coherent_fresh_multi_block_cluster_success_change_1_profile()

    _assert_change_1(actual, expected)


def test_change_1_contract_rejects_orphan_cluster_fields_on_trafilatura_success() -> None:
    actual, expected = _coherent_change_1_profile()
    result = actual["result"]
    assert isinstance(result, dict)
    result.update(
        {
            "cluster_blocks": ["Valid-looking orphan cluster block."],
            "cluster_block_count": 1,
            "cluster_prefiltered_count": 1,
            "cluster_total_blocks": 1,
            "cluster_cluster_count": 1,
            "cluster_method": "greedy",
            "cluster_similarity_threshold": 0.4,
            "cluster_word_threshold": 8,
        }
    )

    with pytest.raises(AssertionError, match="cluster result fields"):
        _assert_change_1(actual, expected)


@pytest.mark.parametrize("label_key", ["url", "base_url", "error", "host", "payload", "pattern"])
def test_change_1_contract_rejects_sensitive_or_high_cardinality_cluster_metric_labels(
    label_key: str,
) -> None:
    actual, expected = _coherent_cluster_success_change_1_profile()
    metrics = actual["metrics"]
    assert isinstance(metrics, list)
    metric = metrics[7]
    assert isinstance(metric, dict)
    labels = metric["labels"]
    assert isinstance(labels, dict)
    labels[label_key] = "https://sensitive.example/path?token=secret"

    with pytest.raises(AssertionError, match="violates contract rule"):
        _assert_change_1(actual, expected)


def test_change_1_contract_rejects_extra_cluster_metric_event_key() -> None:
    actual, expected = _coherent_cluster_success_change_1_profile()
    metrics = actual["metrics"]
    assert isinstance(metrics, list)
    metric = metrics[7]
    assert isinstance(metric, dict)
    metric["raw_error"] = "https://sensitive.example/path?token=secret"

    with pytest.raises(AssertionError, match="violates contract rule"):
        _assert_change_1(actual, expected)


@pytest.mark.parametrize(
    ("metric_index", "labels"),
    [
        (7, {"status": "https://sensitive.example/raw-status"}),
        (8, {"cache": "embedding", "result": "customer-12345"}),
        (8, {"cache": "provider-payload", "result": "miss"}),
    ],
)
def test_change_1_contract_rejects_unbounded_cluster_metric_label_values(
    metric_index: int,
    labels: dict[str, str],
) -> None:
    actual, expected = _coherent_cluster_success_change_1_profile()
    metrics = actual["metrics"]
    assert isinstance(metrics, list)
    metric = metrics[metric_index]
    assert isinstance(metric, dict)
    metric["labels"] = labels

    with pytest.raises(AssertionError, match="violates contract rule"):
        _assert_change_1(actual, expected)


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("cluster_blocks", ["valid", 3]),
        ("cluster_block_count", True),
        ("cluster_prefiltered_count", 0),
        ("cluster_total_blocks", -1),
        ("cluster_cluster_count", 0),
        ("cluster_method", "provider_payload"),
        ("cluster_similarity_threshold", 2.0),
        ("cluster_word_threshold", 0),
    ],
)
def test_change_1_contract_rejects_invalid_cluster_result_fields(
    field: str,
    invalid_value: object,
) -> None:
    actual, expected = _coherent_cluster_success_change_1_profile()
    result = actual["result"]
    assert isinstance(result, dict)
    result[field] = invalid_value

    with pytest.raises(AssertionError, match="violates contract rule|profile"):
        _assert_change_1(actual, expected)


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("cluster_block_count", 2),
        ("cluster_prefiltered_count", 2),
        ("cluster_cluster_count", 2),
    ],
)
def test_change_1_contract_rejects_incoherent_cluster_counts(
    field: str,
    invalid_value: int,
) -> None:
    actual, expected = _coherent_cluster_success_change_1_profile()
    result = actual["result"]
    assert isinstance(result, dict)
    result[field] = invalid_value

    with pytest.raises(AssertionError, match="profile"):
        _assert_change_1(actual, expected)


def test_change_1_contract_rejects_incoherent_cluster_cache_size() -> None:
    actual, expected = _coherent_cluster_success_change_1_profile()
    cache_stats = actual["cache_stats"]
    assert isinstance(cache_stats, dict)
    cache_stats["cluster_embedding_cache_size"] = 2

    with pytest.raises(AssertionError, match="violates contract rule|profile"):
        _assert_change_1(actual, expected)


def test_change_1_contract_rejects_missing_cluster_internal_metrics() -> None:
    actual, expected = _coherent_cluster_success_change_1_profile()
    metrics = actual["metrics"]
    assert isinstance(metrics, list)
    del metrics[7:11]

    with pytest.raises(AssertionError, match="profile"):
        _assert_change_1(actual, expected)


def test_change_1_contract_rejects_missing_cluster_cache_profile() -> None:
    actual, expected = _coherent_cluster_success_change_1_profile()
    actual.pop("cache_stats")
    expected.pop("cache_stats")

    with pytest.raises(AssertionError, match="profile"):
        _assert_change_1(actual, expected)


def test_change_1_contract_rejects_duplicate_metric() -> None:
    actual, expected = _coherent_change_1_profile()
    metrics = actual["metrics"]
    assert isinstance(metrics, list)
    metrics.append(deepcopy(metrics[7]))

    with pytest.raises(AssertionError, match="profile"):
        _assert_change_1(actual, expected)


def test_change_1_contract_rejects_standalone_metric() -> None:
    expected = _predecessor_change_1_profile()
    actual = deepcopy(expected)
    metrics = actual["metrics"]
    assert isinstance(metrics, list)
    metrics.append(_counter("trafilatura", "success"))

    with pytest.raises(AssertionError, match="profile"):
        _assert_change_1(actual, expected)


def test_change_1_contract_rejects_standalone_trace() -> None:
    expected = _predecessor_change_1_profile()
    actual = deepcopy(expected)
    result = actual["result"]
    assert isinstance(result, dict)
    trace = result["extraction_trace"]
    assert isinstance(trace, list)
    trace.append({"reason": "extracted", "status": "success", "strategy": "trafilatura"})

    with pytest.raises(AssertionError, match="profile"):
        _assert_change_1(actual, expected)


def test_change_1_contract_rejects_mismatched_terminal_strategy() -> None:
    actual, expected = _coherent_change_1_profile()
    result = actual["result"]
    assert isinstance(result, dict)
    result["extraction_strategy"] = "cluster"

    with pytest.raises(AssertionError, match="profile"):
        _assert_change_1(actual, expected)


def test_change_1_contract_rejects_unknown_downstream_strategy_with_profile_diagnostic() -> None:
    actual, expected = _coherent_change_1_profile()
    result = actual["result"]
    assert isinstance(result, dict)
    trace = result["extraction_trace"]
    assert isinstance(trace, list)
    trace[3]["strategy"] = "unknown"

    with pytest.raises(AssertionError, match="Change 1 profile"):
        _validate_change_1_profile(actual, expected)


def test_change_1_contract_rejects_missing_required_paired_metric() -> None:
    actual, expected = _coherent_change_1_profile()
    metrics = actual["metrics"]
    assert isinstance(metrics, list)
    del metrics[8]

    with pytest.raises(AssertionError, match="profile"):
        _assert_change_1(actual, expected)


def test_change_1_contract_rejects_incoherent_trace_metric_ordering() -> None:
    actual, expected = _coherent_change_1_profile()
    metrics = actual["metrics"]
    assert isinstance(metrics, list)
    metrics[7:] = [*metrics[9:12], *metrics[7:9]]

    with pytest.raises(AssertionError, match="profile"):
        _assert_change_1(actual, expected)


def test_change_1_contract_rejects_added_trace_with_extra_key() -> None:
    actual, expected = _coherent_change_1_profile()
    result = actual["result"]
    assert isinstance(result, dict)
    trace = result["extraction_trace"]
    assert isinstance(trace, list)
    trace_entry = trace[3]
    assert isinstance(trace_entry, dict)
    trace_entry["secret"] = "must-not-be-approved"

    with pytest.raises(AssertionError, match="violates contract rule"):
        _assert_change_1(actual, expected)


def test_change_1_contract_rejects_added_metric_with_extra_top_level_key() -> None:
    actual, expected = _coherent_change_1_profile()
    metrics = actual["metrics"]
    assert isinstance(metrics, list)
    metric = metrics[7]
    assert isinstance(metric, dict)
    metric["secret"] = "must-not-be-approved"

    with pytest.raises(AssertionError, match="violates contract rule"):
        _assert_change_1(actual, expected)


def test_change_1_contract_rejects_added_metric_with_extra_label() -> None:
    actual, expected = _coherent_change_1_profile()
    metrics = actual["metrics"]
    assert isinstance(metrics, list)
    metric = metrics[7]
    assert isinstance(metric, dict)
    labels = metric["labels"]
    assert isinstance(labels, dict)
    labels["request_url"] = "https://sensitive.example/path"

    with pytest.raises(AssertionError, match="violates contract rule"):
        _assert_change_1(actual, expected)


def test_change_1_contract_rejects_enum_string_subclass_in_added_metric() -> None:
    actual, expected = _coherent_change_1_profile()
    metrics = actual["metrics"]
    assert isinstance(metrics, list)
    metric = metrics[9]
    assert isinstance(metric, dict)
    labels = metric["labels"]
    assert isinstance(labels, dict)
    labels["strategy"] = _StrategyLabel.TRAFILATURA

    with pytest.raises(AssertionError, match="Strict JSON validation failed"):
        _assert_change_1(actual, expected)


def test_change_1_contract_rejects_non_string_key_in_added_metric() -> None:
    actual, expected = _coherent_change_1_profile()
    metrics = actual["metrics"]
    assert isinstance(metrics, list)
    metric = metrics[9]
    assert isinstance(metric, dict)
    labels = metric["labels"]
    assert isinstance(labels, dict)
    labels[1] = "not-json"

    with pytest.raises(AssertionError, match="Strict JSON validation failed"):
        _assert_change_1(actual, expected)


def test_change_1_contract_rejects_non_finite_float_in_added_metric() -> None:
    actual, expected = _coherent_change_1_profile()
    metrics = actual["metrics"]
    assert isinstance(metrics, list)
    metric = metrics[11]
    assert isinstance(metric, dict)
    metric["value"] = float("inf")

    with pytest.raises(AssertionError, match="Strict JSON validation failed"):
        _assert_change_1(actual, expected)


def test_change_1_contract_rejects_detail_for_trace_reason_without_detail() -> None:
    actual, expected = _coherent_change_1_profile()
    result = actual["result"]
    assert isinstance(result, dict)
    trace = result["extraction_trace"]
    assert isinstance(trace, list)
    trace_entry = trace[4]
    assert isinstance(trace_entry, dict)
    trace_entry["detail"] = "must-not-be-approved"

    with pytest.raises(AssertionError, match="violates contract rule"):
        _assert_change_1(actual, expected)


@pytest.mark.parametrize(
    "replacement",
    [
        {
            "reason": "extracted",
            "status": "skipped",
            "strategy": "trafilatura",
        },
        {
            "detail": "cluster_blocks=unknown",
            "reason": "cluster_extracted",
            "status": "success",
            "strategy": "cluster",
        },
    ],
)
def test_change_1_contract_rejects_noncanonical_added_trace(
    replacement: dict[str, object],
) -> None:
    actual, expected = _coherent_change_1_profile()
    result = actual["result"]
    assert isinstance(result, dict)
    trace = result["extraction_trace"]
    assert isinstance(trace, list)
    trace[3] = replacement

    with pytest.raises(AssertionError, match="violates contract rule"):
        _assert_change_1(actual, expected)


@pytest.mark.parametrize(
    ("index", "replacement"),
    [
        (
            7,
            {
                "emitter": "increment_counter",
                "kind": "counter",
                "labels": {"status": "failed", "strategy": "cluster"},
                "name": "extraction_strategy_total",
            },
        ),
        (
            8,
            {
                "emitter": "observe_histogram",
                "kind": "counter",
                "labels": {"status": "failed", "strategy": "cluster"},
                "name": "extraction_strategy_duration_seconds",
                "value": "<TIMING>",
            },
        ),
        (
            11,
            {
                "emitter": "observe_histogram",
                "kind": "histogram",
                "labels": {"strategy": "trafilatura"},
                "name": "extraction_unapproved_value",
                "value": 33,
            },
        ),
        (
            11,
            {
                "emitter": "observe_histogram",
                "kind": "histogram",
                "labels": {"strategy": "trafilatura"},
                "name": "extraction_content_length_bytes",
                "value": -1,
            },
        ),
    ],
)
def test_change_1_contract_rejects_noncanonical_added_metric(
    index: int,
    replacement: dict[str, object],
) -> None:
    actual, expected = _coherent_change_1_profile()
    metrics = actual["metrics"]
    assert isinstance(metrics, list)
    metrics[index] = replacement

    with pytest.raises(AssertionError, match="violates contract rule"):
        _assert_change_1(actual, expected)


@pytest.mark.parametrize(
    ("actual", "expected"),
    [
        (True, 1),
        (1, 1.0),
        ([1], (1,)),
        ({"value": []}, {"value": "[]"}),
    ],
)
@pytest.mark.parametrize("tagged", [False, True])
def test_comparison_is_strict_about_json_types(
    actual: object,
    expected: object,
    tagged: bool,
) -> None:
    kwargs = {"behavior_change": 11, "difference_contract": "change_11_response_too_large"} if tagged else {}
    with pytest.raises(AssertionError):
        assert_predecessor_behavior(actual, expected, **kwargs)


@pytest.mark.parametrize("tagged", [False, True])
def test_comparison_rejects_equal_non_json_containers(tagged: bool) -> None:
    kwargs = {"behavior_change": 11, "difference_contract": "change_11_response_too_large"} if tagged else {}
    with pytest.raises(AssertionError, match="JSON"):
        assert_predecessor_behavior(("same",), ("same",), **kwargs)
