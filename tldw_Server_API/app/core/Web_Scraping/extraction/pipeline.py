"""Canonical orchestration for article extraction strategies."""

import hashlib
import inspect
import json
import os
import random
from collections.abc import Iterator
from contextlib import contextmanager
from copy import deepcopy
from typing import Any, Callable, Optional

from tldw_Server_API.app.core.Web_Scraping.extraction.caches import (
    _schema_cache_get,
    _schema_cache_put,
)
from tldw_Server_API.app.core.Web_Scraping.extraction.dependencies import (
    ExtractionDependencies,
    build_default_dependencies,
)
from tldw_Server_API.app.core.Web_Scraping.extraction.enrichment import (
    enrich_with_regex_matches,
    regex_mask_override,
)
from tldw_Server_API.app.core.Web_Scraping.extraction.metrics import (
    emit_counter,
    emit_histogram,
    emit_log_counter,
)
from tldw_Server_API.app.core.Web_Scraping.extraction.retry import cap_retry_delay
from tldw_Server_API.app.core.Web_Scraping.extraction.strategies import (
    extract_jsonld_entities,
    extract_regex_entities,
)
from tldw_Server_API.app.core.Web_Scraping.extraction.strategies.cluster import (
    _extract_cluster_entities_with_dependencies,
)
from tldw_Server_API.app.core.Web_Scraping.extraction.strategies.llm import (
    _extract_llm_entities_with_dependencies,
    schema_rules_to_field_specs,
)
from tldw_Server_API.app.core.Web_Scraping.extraction.strategies.trafilatura import extract_with_trafilatura
from tldw_Server_API.app.core.Web_Scraping.extraction.throttles import (
    cancellable_semaphore,
    get_strategy_semaphore,
)

DEFAULT_EXTRACTION_STRATEGY_ORDER = [
    "jsonld",
    "schema",
    "regex",
    "llm",
    "cluster",
    "trafilatura",
]
_STRATEGY_ALIASES = {
    "json-ld": "jsonld",
    "json_ld": "jsonld",
    "microdata": "jsonld",
    "schema_css": "schema",
    "schema_xpath": "schema",
    "clustering": "cluster",
}
_KNOWN_STRATEGIES = set(DEFAULT_EXTRACTION_STRATEGY_ORDER)
_METRIC_UNKNOWN_STRATEGY = "unknown"
_PIPELINE_NONCRITICAL_EXCEPTIONS = (
    AssertionError,
    AttributeError,
    ConnectionError,
    FileNotFoundError,
    ImportError,
    IndexError,
    KeyError,
    LookupError,
    OSError,
    PermissionError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
    UnicodeDecodeError,
    json.JSONDecodeError,
)


def _copy_result(result: dict[str, Any]) -> dict[str, Any]:
    return deepcopy(result)


def _metric_strategy(strategy: str) -> str:
    return strategy if strategy in _KNOWN_STRATEGIES else _METRIC_UNKNOWN_STRATEGY


def _metric_attempt(attempt: int) -> str:
    return str(attempt) if attempt <= 3 else "4_plus"


def _ignore_metric_failure() -> None:
    """Keep extraction behavior independent from optional metric sinks."""


def _schema_cache_key(html_text: str, url: str, schema_rules: dict[str, Any]) -> str:
    html_hash = hashlib.sha1(
        html_text.encode("utf-8", errors="ignore"),
        usedforsecurity=False,
    ).hexdigest()
    try:
        rules_repr = json.dumps(schema_rules, sort_keys=True, ensure_ascii=True)
    except _PIPELINE_NONCRITICAL_EXCEPTIONS:
        rules_repr = str(schema_rules)
    raw = f"{url}|{rules_repr}|{html_hash}"
    return hashlib.sha1(raw.encode("utf-8", errors="ignore"), usedforsecurity=False).hexdigest()


def _normalize_strategy_order(strategy_order: Optional[list[str]]) -> tuple[list[str], list[str], bool]:
    default_regex_enrichment = strategy_order is None
    raw = DEFAULT_EXTRACTION_STRATEGY_ORDER if strategy_order is None else strategy_order
    normalized: list[str] = []
    unknown: list[str] = []
    for item in raw:
        if not isinstance(item, str):
            continue
        key = item.strip().lower()
        if not key:
            continue
        key = _STRATEGY_ALIASES.get(key, key)
        if key in _KNOWN_STRATEGIES:
            if key not in normalized:
                normalized.append(key)
        else:
            unknown.append(key)
    if not normalized:
        normalized = list(DEFAULT_EXTRACTION_STRATEGY_ORDER)
    return normalized, unknown, default_regex_enrichment


def _trace_entry(
    dependencies: ExtractionDependencies,
    strategy: str,
    status: str,
    reason: str,
    detail: Optional[str] = None,
) -> dict[str, Any]:
    try:
        emit_log_counter(
            dependencies,
            "extraction_strategy_total",
            labels={"strategy": _metric_strategy(strategy), "status": status},
        )
    except _PIPELINE_NONCRITICAL_EXCEPTIONS:
        _ignore_metric_failure()
    entry = {"strategy": strategy, "status": status, "reason": reason}
    if detail:
        entry["detail"] = detail
    return entry


def _record_strategy_metrics(
    dependencies: ExtractionDependencies,
    strategy: str,
    status: str,
    duration_s: float,
    result: Optional[dict[str, Any]] = None,
) -> None:
    try:
        emit_histogram(
            dependencies,
            "extraction_strategy_duration_seconds",
            duration_s,
            labels={"strategy": _metric_strategy(strategy), "status": status},
        )
    except _PIPELINE_NONCRITICAL_EXCEPTIONS:
        _ignore_metric_failure()
    if status not in {"success", "enriched"} or not result:
        return
    content = result.get("content")
    if isinstance(content, str) and content:
        try:
            emit_histogram(
                dependencies,
                "extraction_content_length_bytes",
                len(content.encode("utf-8", errors="ignore")),
                labels={"strategy": _metric_strategy(strategy)},
            )
        except _PIPELINE_NONCRITICAL_EXCEPTIONS:
            _ignore_metric_failure()


def _attach_trace(
    result: dict[str, Any],
    trace: list[dict[str, Any]],
    strategy: Optional[str],
    strategy_order: list[str],
) -> dict[str, Any]:
    attached = _copy_result(result)
    attached["extraction_trace"] = deepcopy(trace)
    attached["extraction_strategy"] = strategy
    attached["extraction_strategy_order"] = list(strategy_order)
    return attached


def _env_int(name: str) -> Optional[int]:
    raw = os.getenv(name, "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _extractor_max_workers() -> Optional[int]:
    value = _env_int("EXTRACTOR_MAX_WORKERS")
    return value if value and value > 0 else None


@contextmanager
def _strategy_throttle(strategy: str, dependencies: ExtractionDependencies) -> Iterator[None]:
    dependencies.cancellation_checkpoint()
    semaphore = get_strategy_semaphore(strategy, _extractor_max_workers())
    if semaphore is None:
        yield
        return
    with cancellable_semaphore(semaphore, dependencies.cancellation_checkpoint):
        yield


def _coerce_positive_int(value: Any) -> Optional[int]:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _coerce_non_negative_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed >= 0.0 else default


def _extractor_retry_settings() -> tuple[int, float, float]:
    return (
        _coerce_positive_int(os.getenv("EXTRACTOR_MAX_RETRIES")) or 0,
        _coerce_non_negative_float(os.getenv("EXTRACTOR_RETRY_BASE_MS")),
        _coerce_non_negative_float(os.getenv("EXTRACTOR_RETRY_JITTER_MS")),
    )


def _run_with_retries(
    func: Callable[[], dict[str, Any]],
    *,
    strategy: str,
    dependencies: ExtractionDependencies,
) -> tuple[Optional[dict[str, Any]], Optional[Exception], int]:
    max_retries, base_delay_ms, jitter_ms = _extractor_retry_settings()
    attempts = 0
    while True:
        dependencies.cancellation_checkpoint()
        try:
            return func(), None, attempts
        except _PIPELINE_NONCRITICAL_EXCEPTIONS as exc:
            if attempts >= max_retries:
                return None, exc, attempts
            delay_s = (base_delay_ms / 1000.0) * (2**attempts)
            if jitter_ms:
                delay_s += random.uniform(0.0, jitter_ms / 1000.0)  # nosec B311
            delay_s = cap_retry_delay(delay_s)
            attempts += 1
            try:
                emit_counter(
                    dependencies,
                    "extraction_retry_total",
                    labels={"strategy": _metric_strategy(strategy), "attempt": _metric_attempt(attempts)},
                )
            except _PIPELINE_NONCRITICAL_EXCEPTIONS:
                _ignore_metric_failure()
            if delay_s > 0.0:
                dependencies.sleep(delay_s)
            dependencies.cancellation_checkpoint()


def _schema_rule_keys(schema_rules: Optional[dict[str, Any]]) -> list[str]:
    if not isinstance(schema_rules, dict):
        return []
    keys: list[str] = []
    if any(schema_rules.get(key) for key in ("baseSelector", "base_selector", "baseXpath", "base_xpath")):
        keys.append("baseSelector")
    fields = schema_rules_to_field_specs(schema_rules)
    if fields:
        keys.extend(field["name"] for field in fields if isinstance(field.get("name"), str))
    else:
        for key in (
            "title_xpath",
            "title_selector",
            "summary_xpath",
            "summary_selector",
            "description_xpath",
            "content_xpath",
            "content_selector",
            "author_xpath",
            "author_selector",
            "published_xpath",
            "date_xpath",
            "date_selector",
        ):
            if schema_rules.get(key):
                keys.append(key)
    return sorted(set(keys))


def _invoke_handler(
    handler: Callable[[str, str], dict[str, Any]],
    html: str,
    url: str,
    *,
    allow_llm_extraction: bool,
) -> dict[str, Any]:
    """Forward the LLM policy when supported without breaking legacy handlers."""

    try:
        inspect.signature(handler).bind(
            html,
            url,
            allow_llm_extraction=allow_llm_extraction,
        )
    except (TypeError, ValueError):
        return handler(html, url)
    return handler(html, url, allow_llm_extraction=allow_llm_extraction)


def _extract_article_with_pipeline_with_dependencies(
    html: str,
    url: str,
    *,
    dependencies: ExtractionDependencies,
    strategy_order: Optional[list[str]] = None,
    handler: Optional[Callable[[str, str], dict[str, Any]]] = None,
    fallback_extractor: Optional[Callable[[str, str], dict[str, Any]]] = None,
    schema_rules: Optional[dict[str, Any]] = None,
    llm_settings: Optional[dict[str, Any]] = None,
    regex_settings: Optional[dict[str, Any]] = None,
    cluster_settings: Optional[dict[str, Any]] = None,
    allow_llm_extraction: bool = True,
) -> dict[str, Any]:
    dependencies.cancellation_checkpoint()
    trace: list[dict[str, Any]] = []
    jsonld_summary: Optional[str] = None
    regex_result: Optional[dict[str, Any]] = None
    order, unknown, default_regex_enrichment = _normalize_strategy_order(strategy_order)
    if not allow_llm_extraction:
        order = [strategy for strategy in order if strategy != "llm"]

    def finalize(result: dict[str, Any], *, strategy: Optional[str]) -> dict[str, Any]:
        final_result = _copy_result(result)
        summary = final_result.get("summary")
        if (
            final_result.get("extraction_successful")
            and jsonld_summary
            and (not isinstance(summary, str) or not summary.strip())
        ):
            final_result["summary"] = jsonld_summary
        final_result = enrich_with_regex_matches(final_result, regex_result)
        final_result = _attach_trace(final_result, trace, strategy, order)
        return final_result

    for strategy in unknown:
        trace.append(_trace_entry(dependencies, strategy, "skipped", "unknown_strategy"))

    last_result: Optional[dict[str, Any]] = None
    for strategy in order:
        dependencies.cancellation_checkpoint()
        start = dependencies.perf_counter()
        if strategy == "jsonld":
            with _strategy_throttle(strategy, dependencies):
                result = _copy_result(extract_jsonld_entities(html, url))
            summary = result.get("summary")
            if isinstance(summary, str) and summary.strip():
                jsonld_summary = summary
            last_result = result
            if result.get("extraction_successful"):
                trace.append(_trace_entry(dependencies, strategy, "success", "jsonld_extracted"))
                _record_strategy_metrics(dependencies, strategy, "success", dependencies.perf_counter() - start, result)
                return finalize(result, strategy=strategy)
            detail = result.get("jsonld_error")
            trace.append(
                _trace_entry(dependencies, strategy, "failed", "jsonld_no_content", str(detail) if detail else None)
            )
            _record_strategy_metrics(dependencies, strategy, "failed", dependencies.perf_counter() - start, result)
            continue
        if strategy == "schema":
            if isinstance(schema_rules, dict) and schema_rules:
                cache_key = _schema_cache_key(html, url, schema_rules)
                cached = _schema_cache_get(cache_key)
                if cached and cached.get("extraction_successful"):
                    result = _copy_result(cached)
                    result["schema_cache_hit"] = True
                    trace.append(_trace_entry(dependencies, strategy, "success", "schema_cached"))
                    _record_strategy_metrics(
                        dependencies, strategy, "success", dependencies.perf_counter() - start, result
                    )
                    return finalize(result, strategy=strategy)
                try:
                    validation = dependencies.validate_selector_rules(schema_rules, html_text=html, include_counts=True)
                except _PIPELINE_NONCRITICAL_EXCEPTIONS as exc:
                    reason = "schema_import_error" if isinstance(exc, ImportError) else "schema_error"
                    trace.append(_trace_entry(dependencies, strategy, "failed", reason))
                    _record_strategy_metrics(dependencies, strategy, "failed", dependencies.perf_counter() - start)
                    continue
                if validation is None:
                    trace.append(_trace_entry(dependencies, strategy, "failed", "schema_error"))
                    _record_strategy_metrics(dependencies, strategy, "failed", dependencies.perf_counter() - start)
                    continue
                errors = validation.get("errors") if isinstance(validation, dict) else None
                warnings = validation.get("warnings") if isinstance(validation, dict) else None
                selector_counts = validation.get("selector_counts") if isinstance(validation, dict) else None
                warning_detail = (
                    f"{len(warnings)} selector warning(s)" if isinstance(warnings, list) and warnings else None
                )
                if errors:
                    trace.append(
                        _trace_entry(
                            dependencies,
                            strategy,
                            "failed",
                            "schema_invalid_selectors",
                            f"{len(errors)} invalid selector(s)",
                        )
                    )
                    _record_strategy_metrics(dependencies, strategy, "failed", dependencies.perf_counter() - start)
                    continue
                with _strategy_throttle(strategy, dependencies):
                    result, exc, _attempts = _run_with_retries(
                        lambda: dependencies.extract_schema_fields(html, url, schema_rules),
                        strategy=strategy,
                        dependencies=dependencies,
                    )
                if exc or result is None:
                    reason = "schema_import_error" if isinstance(exc, ImportError) else "schema_error"
                    trace.append(_trace_entry(dependencies, strategy, "failed", reason))
                    _record_strategy_metrics(dependencies, strategy, "failed", dependencies.perf_counter() - start)
                    continue
                result = _copy_result(result)
                if warning_detail:
                    result["schema_selector_warnings"] = deepcopy(warnings)
                if isinstance(selector_counts, dict):
                    normalized_counts: dict[str, int] = {}
                    for key, count in selector_counts.items():
                        if not isinstance(key, str):
                            continue
                        normalized_counts[
                            key.split(".", 1)[1] if key.startswith(("fields.", "baseFields.")) else key
                        ] = int(count)
                    result["schema_selector_counts"] = normalized_counts
                result["schema_rule_keys"] = _schema_rule_keys(schema_rules)
                if result.get("extraction_successful"):
                    _schema_cache_put(cache_key, result)
                    trace.append(_trace_entry(dependencies, strategy, "success", "schema_extracted", warning_detail))
                    _record_strategy_metrics(
                        dependencies, strategy, "success", dependencies.perf_counter() - start, result
                    )
                    return finalize(result, strategy=strategy)
                trace.append(_trace_entry(dependencies, strategy, "failed", "schema_no_content", warning_detail))
                _record_strategy_metrics(dependencies, strategy, "failed", dependencies.perf_counter() - start, result)
                last_result = result
                continue
            if handler is None:
                trace.append(_trace_entry(dependencies, strategy, "skipped", "no_schema_rules_or_handler"))
                _record_strategy_metrics(dependencies, strategy, "skipped", dependencies.perf_counter() - start)
                continue
            with _strategy_throttle(strategy, dependencies):
                result, exc, _attempts = _run_with_retries(
                    lambda: _invoke_handler(
                        handler,
                        html,
                        url,
                        allow_llm_extraction=allow_llm_extraction,
                    ),
                    strategy=strategy,
                    dependencies=dependencies,
                )
            if exc or result is None:
                trace.append(_trace_entry(dependencies, strategy, "failed", "handler_error"))
                _record_strategy_metrics(dependencies, strategy, "failed", dependencies.perf_counter() - start)
                continue
            result = _copy_result(result)
            if "extraction_trace" in result:
                result["handler_trace"] = result.pop("extraction_trace")
            if result.get("extraction_successful"):
                trace.append(_trace_entry(dependencies, strategy, "success", "handler_extracted"))
                _record_strategy_metrics(dependencies, strategy, "success", dependencies.perf_counter() - start, result)
                return finalize(result, strategy=strategy)
            trace.append(_trace_entry(dependencies, strategy, "failed", "handler_no_content"))
            _record_strategy_metrics(dependencies, strategy, "failed", dependencies.perf_counter() - start, result)
            last_result = result
            continue
        if strategy == "regex":
            with _strategy_throttle(strategy, dependencies):
                result = _copy_result(extract_regex_entities(html, url, mask_pii=regex_mask_override(regex_settings)))
            if result.get("extraction_successful"):
                regex_result = result
                if default_regex_enrichment:
                    trace.append(_trace_entry(dependencies, strategy, "enriched", "regex_enriched"))
                    _record_strategy_metrics(
                        dependencies, strategy, "enriched", dependencies.perf_counter() - start, result
                    )
                    continue
                trace.append(_trace_entry(dependencies, strategy, "success", "regex_extracted"))
                _record_strategy_metrics(dependencies, strategy, "success", dependencies.perf_counter() - start, result)
                return finalize(result, strategy=strategy)
            last_result = result
            trace.append(_trace_entry(dependencies, strategy, "failed", "regex_no_matches"))
            _record_strategy_metrics(dependencies, strategy, "failed", dependencies.perf_counter() - start, result)
            continue
        if strategy == "llm":
            with _strategy_throttle(strategy, dependencies):
                result = _copy_result(
                    _extract_llm_entities_with_dependencies(
                        html,
                        url,
                        dependencies=dependencies,
                        llm_settings=llm_settings,
                        schema_rules=schema_rules,
                    )
                )
            last_result = result
            if result.get("extraction_successful"):
                trace.append(_trace_entry(dependencies, strategy, "success", "llm_extracted"))
                _record_strategy_metrics(dependencies, strategy, "success", dependencies.perf_counter() - start, result)
                return finalize(result, strategy=strategy)
            detail = result.get("llm_error")
            trace.append(
                _trace_entry(dependencies, strategy, "failed", "llm_no_content", str(detail) if detail else None)
            )
            _record_strategy_metrics(dependencies, strategy, "failed", dependencies.perf_counter() - start, result)
            continue
        if strategy == "cluster":
            with _strategy_throttle(strategy, dependencies):
                result = _copy_result(
                    _extract_cluster_entities_with_dependencies(
                        html,
                        url,
                        dependencies=dependencies,
                        cluster_settings=cluster_settings,
                    )
                )
            last_result = result
            if result.get("extraction_successful"):
                detail = f"cluster_blocks={result.get('cluster_block_count')}"
                trace.append(_trace_entry(dependencies, strategy, "success", "cluster_extracted", detail))
                _record_strategy_metrics(dependencies, strategy, "success", dependencies.perf_counter() - start, result)
                return finalize(result, strategy=strategy)
            detail = result.get("cluster_error")
            trace.append(
                _trace_entry(dependencies, strategy, "failed", "cluster_no_content", str(detail) if detail else None)
            )
            _record_strategy_metrics(dependencies, strategy, "failed", dependencies.perf_counter() - start, result)
            continue
        extractor = fallback_extractor or extract_with_trafilatura
        with _strategy_throttle(strategy, dependencies):
            result, exc, _attempts = _run_with_retries(
                lambda _extractor=extractor: _extractor(html, url),
                strategy=strategy,
                dependencies=dependencies,
            )
        if exc or result is None:
            trace.append(_trace_entry(dependencies, strategy, "failed", "extractor_error"))
            _record_strategy_metrics(dependencies, strategy, "failed", dependencies.perf_counter() - start)
            continue
        result = _copy_result(result)
        last_result = result
        if result.get("extraction_successful"):
            trace.append(_trace_entry(dependencies, strategy, "success", "extracted"))
            _record_strategy_metrics(dependencies, strategy, "success", dependencies.perf_counter() - start, result)
            return finalize(result, strategy=strategy)
        trace.append(_trace_entry(dependencies, strategy, "failed", "no_content"))
        _record_strategy_metrics(dependencies, strategy, "failed", dependencies.perf_counter() - start, result)

    if last_result is None:
        last_result = {
            "title": "N/A",
            "author": "N/A",
            "content": "",
            "date": "N/A",
            "url": url,
            "extraction_successful": False,
        }
    return finalize(last_result, strategy=None)


def extract_article_with_pipeline(
    html: str,
    url: str,
    *,
    strategy_order: Optional[list[str]] = None,
    handler: Optional[Callable[[str, str], dict[str, Any]]] = None,
    fallback_extractor: Optional[Callable[[str, str], dict[str, Any]]] = None,
    schema_rules: Optional[dict[str, Any]] = None,
    llm_settings: Optional[dict[str, Any]] = None,
    regex_settings: Optional[dict[str, Any]] = None,
    cluster_settings: Optional[dict[str, Any]] = None,
    allow_llm_extraction: bool = True,
) -> dict[str, Any]:
    """Run article extraction strategies in deterministic fallback order."""
    return _extract_article_with_pipeline_with_dependencies(
        html,
        url,
        dependencies=build_default_dependencies(),
        strategy_order=strategy_order,
        handler=handler,
        fallback_extractor=fallback_extractor,
        schema_rules=schema_rules,
        llm_settings=llm_settings,
        regex_settings=regex_settings,
        cluster_settings=cluster_settings,
        allow_llm_extraction=allow_llm_extraction,
    )


def extract_article_data_from_html(
    html: str,
    url: str,
    strategy_order: Optional[list[str]] = None,
    handler: Optional[Callable[[str, str], dict[str, Any]]] = None,
    schema_rules: Optional[dict[str, Any]] = None,
    llm_settings: Optional[dict[str, Any]] = None,
    regex_settings: Optional[dict[str, Any]] = None,
    cluster_settings: Optional[dict[str, Any]] = None,
    allow_llm_extraction: bool = True,
) -> dict[str, Any]:
    """Extract article metadata and body from raw HTML."""
    return _extract_article_with_pipeline_with_dependencies(
        html,
        url,
        dependencies=build_default_dependencies(),
        strategy_order=strategy_order,
        handler=handler,
        schema_rules=schema_rules,
        llm_settings=llm_settings,
        regex_settings=regex_settings,
        cluster_settings=cluster_settings,
        allow_llm_extraction=allow_llm_extraction,
    )
