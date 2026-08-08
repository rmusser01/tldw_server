"""Semantic whole-profile contract for approved Phase 4 behavior change 1."""

from __future__ import annotations

import re

from tldw_Server_API.tests.Web_Scraping.phase4_fixture_difference_engine import (
    ANY_PATH,
    MISSING,
    Difference,
    DifferenceContract,
    DifferenceRule,
    collect_differences,
)

_DOWNSTREAM_STRATEGIES = frozenset({"cluster", "trafilatura"})
_DOWNSTREAM_STRATEGIES_ORDER = ("cluster", "trafilatura")
_TRACE_KEYS = frozenset({"reason", "status", "strategy"})
_TRACE_DETAIL_KEYS = _TRACE_KEYS | {"detail"}
_TRACE_COMBINATIONS_WITHOUT_DETAIL = frozenset(
    {
        ("trafilatura", "failed", "extractor_error"),
        ("trafilatura", "failed", "no_content"),
        ("trafilatura", "success", "extracted"),
    }
)
_CLUSTER_FAILURE_DETAILS = frozenset(
    {
        "cluster_empty_content",
        "cluster_empty_html",
        "cluster_no_blocks",
        "cluster_no_clusters",
    }
)
_COUNTER_METRIC_KEYS = frozenset({"emitter", "kind", "labels", "name"})
_HISTOGRAM_METRIC_KEYS = _COUNTER_METRIC_KEYS | {"value"}
_STRATEGY_STATUS_LABEL_KEYS = frozenset({"status", "strategy"})
_STRATEGY_LABEL_KEYS = frozenset({"strategy"})


def _regex_becomes_non_terminal(difference: Difference) -> bool:
    return (
        difference.expected == "regex"
        and isinstance(difference.actual, str)
        and difference.actual in _DOWNSTREAM_STRATEGIES
    )


def _regex_trace_reason_becomes_non_terminal(difference: Difference) -> bool:
    return (
        difference.expected == "regex_extracted"
        and isinstance(difference.actual, str)
        and difference.actual in {"regex_enriched", "regex_non_terminal"}
    )


def _regex_trace_status_becomes_non_terminal(difference: Difference) -> bool:
    return (
        difference.expected == "success"
        and isinstance(difference.actual, str)
        and difference.actual in {"continued", "enriched"}
    )


def _is_new_downstream_trace_step(difference: Difference) -> bool:
    if difference.expected is not MISSING or type(difference.actual) is not dict:
        return False
    record = difference.actual
    keys = frozenset(record)
    strategy = record.get("strategy")
    status = record.get("status")
    reason = record.get("reason")
    if not all(isinstance(value, str) for value in (strategy, status, reason)):
        return False

    combination = (strategy, status, reason)
    if combination in _TRACE_COMBINATIONS_WITHOUT_DETAIL:
        return keys == _TRACE_KEYS
    if combination == ("cluster", "success", "cluster_extracted"):
        detail = record.get("detail")
        return (
            keys == _TRACE_DETAIL_KEYS
            and isinstance(detail, str)
            and re.fullmatch(r"cluster_blocks=[1-9][0-9]*", detail) is not None
        )
    if combination == ("cluster", "failed", "cluster_no_content"):
        detail = record.get("detail")
        return keys == _TRACE_DETAIL_KEYS and isinstance(detail, str) and detail in _CLUSTER_FAILURE_DETAILS
    return False


def _has_strategy_status_labels(value: object) -> bool:
    if type(value) is not dict or frozenset(value) != _STRATEGY_STATUS_LABEL_KEYS:
        return False
    strategy = value.get("strategy")
    status = value.get("status")
    return (
        isinstance(strategy, str)
        and strategy in _DOWNSTREAM_STRATEGIES
        and isinstance(status, str)
        and status in {"failed", "success"}
    )


def _has_strategy_labels(value: object) -> bool:
    if type(value) is not dict or frozenset(value) != _STRATEGY_LABEL_KEYS:
        return False
    strategy = value.get("strategy")
    return isinstance(strategy, str) and strategy in _DOWNSTREAM_STRATEGIES


def _is_new_regex_or_downstream_metric(difference: Difference) -> bool:
    if difference.expected is not MISSING or type(difference.actual) is not dict:
        return False
    record = difference.actual
    keys = frozenset(record)
    emitter = record.get("emitter")
    kind = record.get("kind")
    name = record.get("name")
    labels = record.get("labels")
    value = record.get("value")

    if name == "extraction_strategy_total":
        return (
            keys == _COUNTER_METRIC_KEYS
            and emitter == "log_counter"
            and kind == "counter"
            and _has_strategy_status_labels(labels)
        )
    if name == "extraction_strategy_duration_seconds":
        return (
            keys == _HISTOGRAM_METRIC_KEYS
            and emitter == "observe_histogram"
            and kind == "histogram"
            and _has_strategy_status_labels(labels)
            and value == "<TIMING>"
        )
    if name == "extraction_content_length_bytes":
        return (
            keys == _HISTOGRAM_METRIC_KEYS
            and emitter == "observe_histogram"
            and kind == "histogram"
            and _has_strategy_labels(labels)
            and type(value) is int
            and value > 0
        )
    return False


def _profile_dict(value: object, path: str) -> dict[str, object]:
    assert type(value) is dict, f"Change 1 profile requires an object at {path}"
    return value


def _profile_list(value: object, path: str) -> list[object]:
    assert type(value) is list, f"Change 1 profile requires an array at {path}"
    return value


def _metric_profile(metric: object) -> tuple[str, str, str | None]:
    record = _profile_dict(metric, "$.metrics[]")
    labels = _profile_dict(record.get("labels"), "$.metrics[].labels")
    name = record.get("name")
    strategy = labels.get("strategy")
    status = labels.get("status")
    assert isinstance(name, str), "Change 1 profile requires a metric name"
    assert isinstance(strategy, str), "Change 1 profile requires a metric strategy"
    assert status is None or isinstance(status, str), "Change 1 profile requires a string metric status"
    return name, strategy, status


def _validate_change_1_profile(actual: object, expected: object) -> None:
    if not collect_differences(actual, expected):
        return

    actual_root = _profile_dict(actual, "$")
    expected_root = _profile_dict(expected, "$")
    actual_result = _profile_dict(actual_root.get("result"), "$.result")
    expected_result = _profile_dict(expected_root.get("result"), "$.result")
    actual_trace = _profile_list(
        actual_result.get("extraction_trace"),
        "$.result.extraction_trace",
    )
    expected_trace = _profile_list(
        expected_result.get("extraction_trace"),
        "$.result.extraction_trace",
    )
    actual_metrics = _profile_list(actual_root.get("metrics"), "$.metrics")
    expected_metrics = _profile_list(expected_root.get("metrics"), "$.metrics")

    assert len(expected_trace) == 3, "Change 1 profile requires the pinned three-step predecessor trace"
    assert len(expected_metrics) == 7, "Change 1 profile requires the pinned predecessor metric sequence"
    assert actual_trace[:2] == expected_trace[:2], "Change 1 profile changed a pre-regex trace step"
    assert actual_metrics[:4] == expected_metrics[:4], "Change 1 profile changed a pre-regex metric"
    assert actual_metrics[6] == expected_metrics[6], "Change 1 profile changed regex enrichment length"

    predecessor_regex = _profile_dict(
        expected_trace[2],
        "$.result.extraction_trace[2]",
    )
    actual_regex = _profile_dict(actual_trace[2], "$.result.extraction_trace[2]")
    assert predecessor_regex == {
        "reason": "regex_extracted",
        "status": "success",
        "strategy": "regex",
    }, "Change 1 profile does not match the pinned regex predecessor step"
    regex_transition = (actual_regex.get("status"), actual_regex.get("reason"))
    assert regex_transition in {
        ("continued", "regex_non_terminal"),
        ("enriched", "regex_enriched"),
    }, "Change 1 profile requires one coherent regex status/reason transition"
    regex_status = actual_regex["status"]
    assert actual_regex.get("strategy") == "regex", "Change 1 profile changed the regex trace strategy"

    assert _metric_profile(actual_metrics[4]) == (
        "extraction_strategy_total",
        "regex",
        regex_status,
    ), "Change 1 profile requires one matching regex counter transition"
    assert _metric_profile(actual_metrics[5]) == (
        "extraction_strategy_duration_seconds",
        "regex",
        regex_status,
    ), "Change 1 profile requires one matching regex timing transition"

    downstream_trace = actual_trace[len(expected_trace) :]
    assert downstream_trace, "Change 1 profile cannot contain standalone metric additions"
    assert 1 <= len(downstream_trace) <= 2, "Change 1 profile has invalid downstream trace cardinality"
    trace_records = [_profile_dict(entry, "$.result.extraction_trace[]") for entry in downstream_trace]
    strategies = [entry.get("strategy") for entry in trace_records]
    assert all(
        isinstance(strategy, str) for strategy in strategies
    ), "Change 1 profile requires string downstream strategies"
    assert all(
        strategy in _DOWNSTREAM_STRATEGIES_ORDER for strategy in strategies
    ), "Change 1 profile requires known downstream strategies"
    positions = [_DOWNSTREAM_STRATEGIES_ORDER.index(strategy) for strategy in strategies]
    assert positions == sorted(set(positions)), "Change 1 profile has incoherent downstream trace ordering"

    successful_steps = [entry for entry in trace_records if entry.get("status") == "success"]
    assert len(successful_steps) == 1, "Change 1 profile requires exactly one successful terminal strategy"
    terminal = successful_steps[0]
    assert terminal is trace_records[-1], "Change 1 profile cannot continue after terminal success"
    terminal_strategy = terminal.get("strategy")
    assert (
        actual_result.get("extraction_strategy") == terminal_strategy
    ), "Change 1 profile terminal strategy does not match extraction_strategy"

    expected_added_metrics: list[tuple[str, str, str | None]] = []
    for entry in trace_records:
        strategy = entry["strategy"]
        status = entry.get("status")
        assert isinstance(strategy, str) and isinstance(status, str)
        expected_added_metrics.extend(
            [
                ("extraction_strategy_total", strategy, status),
                ("extraction_strategy_duration_seconds", strategy, status),
            ]
        )
        if status == "success":
            expected_added_metrics.append(("extraction_content_length_bytes", strategy, None))

    added_metrics = actual_metrics[len(expected_metrics) :]
    actual_added_metrics = [_metric_profile(metric) for metric in added_metrics]
    assert actual_added_metrics == expected_added_metrics, (
        "Change 1 profile requires ordered counter/timing pairs and one terminal " "content-length metric"
    )


CHANGE_1_CONTRACT = DifferenceContract(
    behavior_change=1,
    rules=(
        DifferenceRule(
            identifier="terminal_strategy",
            path=("result", "extraction_strategy"),
            description="regex no longer terminates the default extraction pipeline",
            validator=_regex_becomes_non_terminal,
            minimum_count=1,
        ),
        DifferenceRule(
            identifier="regex_trace_reason",
            path=("result", "extraction_trace", 2, "reason"),
            description="the regex trace records enrichment rather than terminal extraction",
            validator=_regex_trace_reason_becomes_non_terminal,
            minimum_count=1,
        ),
        DifferenceRule(
            identifier="regex_trace_status",
            path=("result", "extraction_trace", 2, "status"),
            description="the regex trace records continued extraction",
            validator=_regex_trace_status_becomes_non_terminal,
            minimum_count=1,
        ),
        DifferenceRule(
            identifier="downstream_trace",
            path=("result", "extraction_trace", ANY_PATH),
            description="a downstream strategy runs after regex enrichment",
            validator=_is_new_downstream_trace_step,
            minimum_count=1,
            maximum_count=2,
        ),
        DifferenceRule(
            identifier="regex_counter_status",
            path=("metrics", 4, "labels", "status"),
            description="regex metrics record continued extraction",
            validator=_regex_trace_status_becomes_non_terminal,
            minimum_count=1,
        ),
        DifferenceRule(
            identifier="regex_timing_status",
            path=("metrics", 5, "labels", "status"),
            description="regex timing metrics record continued extraction",
            validator=_regex_trace_status_becomes_non_terminal,
            minimum_count=1,
        ),
        DifferenceRule(
            identifier="downstream_metric",
            path=("metrics", ANY_PATH),
            description="regex enrichment emits downstream extraction metrics",
            validator=_is_new_regex_or_downstream_metric,
            minimum_count=3,
            maximum_count=5,
        ),
    ),
    allow_predecessor_equality=True,
    profile_validator=_validate_change_1_profile,
)
