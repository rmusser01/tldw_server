"""Shared deterministic normalization and metric capture utilities."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from contextlib import ExitStack
from typing import Any
from unittest.mock import patch

FIXED_ENV = {
    "CLUSTER_LINKAGE": "",
    "EXTRACTOR_CLEAR_CACHES": "",
    "EXTRACTOR_MAX_RETRIES": "0",
    "EXTRACTOR_MAX_WORKERS": "",
    "EXTRACTOR_RETRY_BASE_MS": "0",
    "EXTRACTOR_RETRY_JITTER_MS": "0",
    "REGEX_PII_MASK": "false",
    "SIM_THRESHOLD": "",
    "WATCHLIST_SELECTOR_MAX_EXPR_LEN": "512",
    "WATCHLIST_SELECTOR_MAX_XPATH_DESCENDANT_STEPS": "12",
    "WATCHLIST_SELECTOR_MAX_XPATH_FUNCTION_CALLS": "8",
    "WATCHLIST_SELECTOR_MAX_XPATH_PREDICATES": "10",
    "WORD_COUNT_THRESHOLD": "",
}


def canonical_data(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=True, sort_keys=True))


def case(payload: dict[str, Any]) -> dict[str, Any]:
    return canonical_data(payload)


def normalize_formatted_metadata(value: str) -> str:
    return re.sub(
        r'("ingestion_date":\s*)"[^"]+"',
        r'\1"<TIMESTAMP>"',
        value,
        count=1,
    )


class MetricRecorder:
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


def metric_patches(stack: ExitStack, article: Any) -> MetricRecorder:
    recorder = MetricRecorder()
    stack.enter_context(patch.object(article, "increment_counter", recorder.counter("increment_counter")))
    stack.enter_context(patch.object(article, "log_counter", recorder.counter("log_counter")))
    stack.enter_context(patch.object(article, "observe_histogram", recorder.histogram("observe_histogram")))
    stack.enter_context(patch.object(article, "log_histogram", recorder.histogram("log_histogram")))
    return recorder
