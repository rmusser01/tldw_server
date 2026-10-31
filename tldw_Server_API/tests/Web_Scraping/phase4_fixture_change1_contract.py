"""Semantic whole-profile contract for approved Phase 4 behavior change 1."""

from __future__ import annotations

import math
import re

from tldw_Server_API.app.core.Web_Scraping.extraction import caches as extraction_caches
from tldw_Server_API.app.core.Web_Scraping.extraction.strategies import cluster as cluster_strategy
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
_CLUSTER_TOTAL_LABEL_KEYS = frozenset({"status"})
_CLUSTER_CACHE_LABEL_KEYS = frozenset({"cache", "result"})
_CLUSTER_TOTAL_STATUSES = frozenset({"started", "no_blocks", "no_clusters", "empty", "success"})
_CLUSTER_CACHE_RESULTS = frozenset({"hit", "miss"})
_CLUSTER_METHODS = frozenset({"greedy", "greedy_fallback", "hierarchical"})
_CLUSTER_RESULT_FIELDS = (
    "cluster_blocks",
    "cluster_block_count",
    "cluster_prefiltered_count",
    "cluster_total_blocks",
    "cluster_cluster_count",
    "cluster_method",
    "cluster_similarity_threshold",
    "cluster_word_threshold",
)


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


def _cluster_cache_created_by_downstream_extraction(difference: Difference) -> bool:
    maximum_fresh_entries = min(
        extraction_caches._CLUSTER_EMBED_CACHE_MAX,
        cluster_strategy._CLUSTER_MAX_BLOCKS + 1,
    )
    return (
        difference.expected == 0 and type(difference.actual) is int and 0 < difference.actual <= maximum_fresh_entries
    )


def _is_cluster_internal_metric_record(record: object) -> bool:
    if type(record) is not dict or frozenset(record) != _COUNTER_METRIC_KEYS:
        return False
    if record.get("emitter") != "increment_counter" or record.get("kind") != "counter":
        return False
    labels = record.get("labels")
    if type(labels) is not dict:
        return False
    if record.get("name") == "extraction_cluster_total":
        return (
            frozenset(labels) == _CLUSTER_TOTAL_LABEL_KEYS
            and type(labels.get("status")) is str
            and labels["status"] in _CLUSTER_TOTAL_STATUSES
        )
    if record.get("name") == "extraction_cluster_cache_total":
        return (
            frozenset(labels) == _CLUSTER_CACHE_LABEL_KEYS
            and labels.get("cache") == "embedding"
            and type(labels.get("result")) is str
            and labels["result"] in _CLUSTER_CACHE_RESULTS
        )
    return False


def _is_cluster_internal_metric(difference: Difference) -> bool:
    return difference.expected is MISSING and _is_cluster_internal_metric_record(difference.actual)


def _is_nonempty_cluster_blocks(value: object) -> bool:
    return type(value) is list and bool(value) and all(type(block) is str and bool(block.strip()) for block in value)


def _is_positive_int(value: object) -> bool:
    return type(value) is int and value > 0


def _is_default_cluster_total_blocks(value: object) -> bool:
    return _is_positive_int(value) and value <= cluster_strategy._CLUSTER_MAX_BLOCKS


def _is_cluster_method(value: object) -> bool:
    return type(value) is str and value in _CLUSTER_METHODS


def _is_cluster_similarity_threshold(value: object) -> bool:
    return type(value) is float and math.isfinite(value) and 0.0 <= value <= 1.0


_CLUSTER_RESULT_VALIDATORS = {
    "cluster_blocks": _is_nonempty_cluster_blocks,
    "cluster_block_count": _is_positive_int,
    "cluster_prefiltered_count": _is_positive_int,
    "cluster_total_blocks": _is_default_cluster_total_blocks,
    "cluster_cluster_count": _is_positive_int,
    "cluster_method": _is_cluster_method,
    "cluster_similarity_threshold": _is_cluster_similarity_threshold,
    "cluster_word_threshold": _is_positive_int,
}


def _new_cluster_result_field_validator(field: str):
    validator = _CLUSTER_RESULT_VALIDATORS[field]
    return lambda difference: difference.expected is MISSING and validator(difference.actual)


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

    present_cluster_fields = [field for field in _CLUSTER_RESULT_FIELDS if field in actual_result]
    if terminal_strategy != "cluster":
        assert not present_cluster_fields, (
            "Change 1 profile cluster result fields require a successful " "terminal cluster strategy"
        )

    if terminal_strategy == "cluster":
        missing_fields = [field for field in _CLUSTER_RESULT_FIELDS if field not in actual_result]
        assert not missing_fields, f"Change 1 profile is missing cluster result fields: {missing_fields}"
        blocks = actual_result["cluster_blocks"]
        block_count = actual_result["cluster_block_count"]
        prefiltered_count = actual_result["cluster_prefiltered_count"]
        total_blocks = actual_result["cluster_total_blocks"]
        cluster_count = actual_result["cluster_cluster_count"]
        assert _is_nonempty_cluster_blocks(blocks), "Change 1 profile requires non-empty cluster blocks"
        assert all(
            _CLUSTER_RESULT_VALIDATORS[field](actual_result[field]) for field in _CLUSTER_RESULT_FIELDS
        ), "Change 1 profile contains an invalid cluster result field"
        assert isinstance(blocks, list)
        assert block_count == len(blocks), "Change 1 profile cluster block count must match cluster_blocks"
        assert (
            block_count <= prefiltered_count <= total_blocks
        ), "Change 1 profile requires block_count <= prefiltered_count <= total_blocks"
        assert (
            cluster_count <= prefiltered_count
        ), "Change 1 profile cluster count cannot exceed the prefiltered block count"
        assert (
            terminal.get("detail") == f"cluster_blocks={block_count}"
        ), "Change 1 profile cluster trace detail must match cluster_block_count"

        actual_cache = _profile_dict(actual_root.get("cache_stats"), "$.cache_stats")
        expected_cache = _profile_dict(expected_root.get("cache_stats"), "$.cache_stats")
        assert (
            expected_cache.get("cluster_embedding_cache_size") == 0
        ), "Change 1 profile requires an empty predecessor cluster cache"
        assert (
            type(actual_cache.get("cluster_embedding_cache_size")) is int
        ), "Change 1 profile requires an integer cluster embedding cache size"

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
    cluster_internal_metrics = [
        _profile_dict(metric, "$.metrics[]")
        for metric in added_metrics
        if _profile_dict(metric, "$.metrics[]").get("name")
        in {"extraction_cluster_total", "extraction_cluster_cache_total"}
    ]
    assert all(
        _is_cluster_internal_metric_record(metric) for metric in cluster_internal_metrics
    ), "Change 1 profile contains a noncanonical cluster metric"
    cluster_trace = next(
        (entry for entry in trace_records if entry.get("strategy") == "cluster"),
        None,
    )
    if terminal_strategy == "cluster":
        total_blocks = actual_result["cluster_total_blocks"]
        assert type(total_blocks) is int
        embedding_operations = total_blocks + 1
        assert (
            embedding_operations <= extraction_caches._CLUSTER_EMBED_CACHE_MAX
        ), "Change 1 profile default cluster embeddings must fit the canonical cache"
        internal_metric_count = embedding_operations + 2
        assert (
            len(added_metrics) == internal_metric_count + 3
        ), "Change 1 profile requires one exact contiguous cluster metric sequence"
        started_metric = _profile_dict(added_metrics[0], "$.metrics[]")
        cache_metrics = [_profile_dict(metric, "$.metrics[]") for metric in added_metrics[1 : 1 + embedding_operations]]
        success_metric = _profile_dict(added_metrics[1 + embedding_operations], "$.metrics[]")
        pipeline_metrics = added_metrics[internal_metric_count:]
        assert (
            _is_cluster_internal_metric_record(started_metric)
            and started_metric.get("name") == "extraction_cluster_total"
            and started_metric.get("labels") == {"status": "started"}
        ), "Change 1 profile requires a contiguous cluster lifecycle starting with started"
        assert all(
            _is_cluster_internal_metric_record(metric) and metric.get("name") == "extraction_cluster_cache_total"
            for metric in cache_metrics
        ), "Change 1 profile requires contiguous cache lookups between started and success"
        assert (
            _is_cluster_internal_metric_record(success_metric)
            and success_metric.get("name") == "extraction_cluster_total"
            and success_metric.get("labels") == {"status": "success"}
        ), "Change 1 profile requires contiguous cluster success after the final cache lookup"
        assert [_metric_profile(metric) for metric in pipeline_metrics] == expected_added_metrics, (
            "Change 1 profile requires contiguous pipeline cluster counter, " "duration, and content-length metrics"
        )
        cache_results = [_profile_dict(metric["labels"], "$.metrics[].labels")["result"] for metric in cache_metrics]
        assert cache_results[0] == "miss", "Change 1 profile requires a fresh document-embedding cache miss"
        miss_count = cache_results.count("miss")
        actual_cache = _profile_dict(actual_root.get("cache_stats"), "$.cache_stats")
        assert actual_cache.get("cluster_embedding_cache_size") == miss_count, (
            "Change 1 profile fresh cache growth must equal the canonical " "embedding miss count"
        )
    else:
        cluster_cache_metrics = [
            metric for metric in cluster_internal_metrics if metric.get("name") == "extraction_cluster_cache_total"
        ]
        assert not cluster_cache_metrics, (
            "Change 1 profile cluster cache metrics require successful " "terminal cluster extraction"
        )
        if cluster_trace is not None:
            assert cluster_trace == {
                "detail": "cluster_no_blocks",
                "reason": "cluster_no_content",
                "status": "failed",
                "strategy": "cluster",
            }, "Change 1 profile only approves the trace-proven cluster_no_blocks fallback"
            assert (
                len(added_metrics) == len(expected_added_metrics) + 2
            ), "Change 1 profile requires one exact cluster_no_blocks metric segment"
            started_metric = _profile_dict(added_metrics[0], "$.metrics[]")
            no_blocks_metric = _profile_dict(added_metrics[1], "$.metrics[]")
            assert (
                _is_cluster_internal_metric_record(started_metric)
                and started_metric.get("name") == "extraction_cluster_total"
                and started_metric.get("labels") == {"status": "started"}
            ), "Change 1 profile requires cluster started before cluster_no_blocks"
            assert (
                _is_cluster_internal_metric_record(no_blocks_metric)
                and no_blocks_metric.get("name") == "extraction_cluster_total"
                and no_blocks_metric.get("labels") == {"status": "no_blocks"}
            ), "Change 1 profile requires cluster no_blocks immediately after started"
            pipeline_metrics = added_metrics[2:]
        else:
            assert (
                not cluster_internal_metrics
            ), "Change 1 profile cluster lifecycle metrics require a matching cluster trace"
            pipeline_metrics = added_metrics
        assert [
            _metric_profile(metric) for metric in pipeline_metrics
        ] == expected_added_metrics, (
            "Change 1 profile requires lifecycle and ordered pipeline metrics to match the trace"
        )
        expected_cache_size = (
            _profile_dict(expected_root["cache_stats"], "$.cache_stats").get("cluster_embedding_cache_size")
            if "cache_stats" in expected_root
            else MISSING
        )
        actual_cache_size = (
            _profile_dict(actual_root["cache_stats"], "$.cache_stats").get("cluster_embedding_cache_size")
            if "cache_stats" in actual_root
            else MISSING
        )
        assert actual_cache_size == expected_cache_size, (
            "Change 1 profile rejects cache growth without successful " "terminal cluster extraction"
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
            validator=lambda difference: _is_new_regex_or_downstream_metric(difference)
            or _is_cluster_internal_metric(difference),
            minimum_count=3,
            maximum_count=cluster_strategy._CLUSTER_MAX_BLOCKS + 6,
        ),
        DifferenceRule(
            identifier="downstream_cluster_cache",
            path=("cache_stats", "cluster_embedding_cache_size"),
            description="the newly reached cluster strategy may populate the shared embedding cache",
            validator=_cluster_cache_created_by_downstream_extraction,
            minimum_count=0,
            maximum_count=1,
        ),
        *(
            DifferenceRule(
                identifier=f"cluster_result_{field}",
                path=("result", field),
                description="a successful downstream cluster result may include its established metadata",
                validator=_new_cluster_result_field_validator(field),
                minimum_count=0,
                maximum_count=1,
            )
            for field in _CLUSTER_RESULT_FIELDS
        ),
    ),
    allow_predecessor_equality=True,
    profile_validator=_validate_change_1_profile,
)
