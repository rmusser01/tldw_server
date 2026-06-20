"""Service orchestration for research discovery searches."""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, is_dataclass, replace
from typing import TYPE_CHECKING, Any, Protocol

from tldw_Server_API.app.core.exceptions import (
    ResearchDiscoveryBadRequestError,
    ResearchDiscoveryTimeoutError,
    ResearchDiscoveryUpstreamError,
    ResearchDiscoveryValidationError,
)

from .adapters import default_discovery_adapters
from .catalog import ResearchSourceCatalog, default_source_catalog
from .identity import (
    has_unsafe_url_material,
    normalize_and_merge_records,
    safe_provider_metadata,
)
from .models import (
    DiscoveryExecutionPolicy,
    DiscoveryMetrics,
    DiscoveryResult,
    DiscoverySearchResponse,
    ResearchSourceCatalogEntry,
    SourceStatus,
)
from .router import ResearchSourceRouter

if TYPE_CHECKING:
    from tldw_Server_API.app.core.DB_Management.ResearchSessionsDB import ResearchSessionsDB

    from .oa import ResearchOAResolver


DEFAULT_PER_SOURCE_LIMIT = 5
DEFAULT_TOTAL_LIMIT = 25
DEFAULT_TOTAL_TIMEOUT_SECONDS = 30.0
DEFAULT_PER_SOURCE_TIMEOUT_SECONDS = 10.0
DEFAULT_MAX_CONCURRENCY = 4
DEFAULT_OA_RESOLVER_CONCURRENCY = 4
DEFAULT_SOURCE_CATEGORIES = ("open_research_graph",)
MAX_DISCOVERY_FILTER_KEYS = 50
MAX_DISCOVERY_FILTER_DEPTH = 6
MAX_DISCOVERY_FILTER_BYTES = 8192
NO_RUNNABLE_STATUSES = {
    "policy_blocked",
    "credentials_missing",
    "provider_not_configured",
}
FAILURE_STATUSES = NO_RUNNABLE_STATUSES | {
    "internal_error",
    "provider_error",
    "timeout",
    "rate_limited",
}
UNSAFE_WARNING_TEXT = "warning_redacted"
_URL_IN_TEXT_RE = re.compile(r"https?://[^\s<>)\"']+", re.IGNORECASE)
_UNSAFE_WARNING_PARTS = (
    "/private/",
    "access_token",
    "api_key",
    "apikey",
    "authorization",
    "credential",
    "secret",
    "signature",
    "token",
)
_DIAGNOSTIC_METADATA_KEY_PARTS = (
    "detail",
    "error",
    "exception",
    "message",
    "status",
    "trace",
    "warning",
)


class DiscoveryProviderRouter(Protocol):
    async def search_sources(
        self,
        *,
        query: str,
        sources: Iterable[ResearchSourceCatalogEntry],
        per_source_limit: int,
        filters: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], list[SourceStatus]]: ...


class ResearchDiscoveryService:
    """Coordinate catalog selection, provider routing, merging, and snapshots."""

    def __init__(
        self,
        catalog: ResearchSourceCatalog | None = None,
        router: DiscoveryProviderRouter | None = None,
        snapshot_db: ResearchSessionsDB | None = None,
        oa_resolver: ResearchOAResolver | None = None,
        db_factory: Callable[[str], ResearchSessionsDB] | None = None,
        snapshot_retention_hours: int = 24,
        total_timeout_seconds: float = DEFAULT_TOTAL_TIMEOUT_SECONDS,
    ) -> None:
        if snapshot_db is not None and db_factory is not None:
            raise ValueError("research_discovery_snapshot_db_conflict")
        if total_timeout_seconds <= 0:
            raise ValueError("research_discovery_total_timeout_must_be_positive")

        self._catalog = catalog or default_source_catalog()
        self._router = router or ResearchSourceRouter(
            catalog=self._catalog,
            adapters=default_discovery_adapters(),
            per_source_timeout_seconds=DEFAULT_PER_SOURCE_TIMEOUT_SECONDS,
            max_concurrency=DEFAULT_MAX_CONCURRENCY,
        )
        if oa_resolver is None:
            from .oa import ResearchOAResolver

            oa_resolver = ResearchOAResolver()

        self._snapshot_db = snapshot_db
        self._db_factory = db_factory
        self._oa_resolver = oa_resolver
        self._snapshot_retention_hours = snapshot_retention_hours
        self._total_timeout_seconds = total_timeout_seconds

    @property
    def adapter_names(self) -> tuple[str, ...]:
        """Return configured provider adapter names for diagnostics."""
        return tuple(getattr(self._router, "adapter_names", ()))

    async def search(
        self,
        *,
        owner_user_id: str,
        query: str,
        source_ids: Sequence[str] | str | None = None,
        categories: Sequence[str] | str | None = None,
        max_sources: int | None = None,
        per_source_limit: int = DEFAULT_PER_SOURCE_LIMIT,
        total_limit: int = DEFAULT_TOTAL_LIMIT,
        limit: int | None = None,
        source_policy: str = "balanced",
        fallback_policy: str | Mapping[str, Any] | None = "disabled",
        filters: Mapping[str, Any] | None = None,
        provider_overrides: Mapping[str, Any] | None = None,
        total_timeout_seconds: float | None = None,
    ) -> DiscoverySearchResponse:
        """Run a discovery search and persist a sanitized result snapshot."""
        query_text = query.strip()
        if not query_text:
            raise ResearchDiscoveryValidationError("research_discovery_query_required")
        if _text_contains_unsafe_url_material(query_text):
            raise ResearchDiscoveryValidationError("research_discovery_query_contains_unsafe_url")

        if limit is not None:
            per_source_limit = limit
            total_limit = limit
        if per_source_limit < 1 or total_limit < 1:
            raise ResearchDiscoveryValidationError("research_discovery_limit_must_be_positive")
        if max_sources is not None and max_sources < 1:
            raise ResearchDiscoveryValidationError("research_discovery_max_sources_must_be_positive")
        timeout_budget = self._total_timeout_seconds if total_timeout_seconds is None else total_timeout_seconds
        if timeout_budget <= 0:
            raise ResearchDiscoveryValidationError("research_discovery_total_timeout_must_be_positive")
        if not _fallback_policy_is_disabled(fallback_policy):
            raise ResearchDiscoveryValidationError("research_discovery_fallback_disabled")

        normalized_source_ids = _normalize_string_sequence(source_ids)
        normalized_categories = _normalize_string_sequence(categories)
        defaulted_categories: list[str] = []
        if not normalized_source_ids and not normalized_categories:
            defaulted_categories = list(DEFAULT_SOURCE_CATEGORIES)
            normalized_categories = list(DEFAULT_SOURCE_CATEGORIES)
        safe_filters = _safe_json_mapping(_merge_filter_inputs(provider_overrides, filters))
        if _contains_unsafe_url_text(safe_filters):
            raise ResearchDiscoveryValidationError("research_discovery_filters_contain_unsafe_url")
        selected_sources = self._resolve_sources(
            source_ids=normalized_source_ids,
            categories=normalized_categories,
            max_sources=max_sources,
        )
        execution_policy = _execution_policy(
            selected_sources=selected_sources,
            total_timeout_seconds=timeout_budget,
        )

        started_at = time.perf_counter()
        try:
            raw_records, source_statuses = await asyncio.wait_for(
                self._router.search_sources(
                    query=query_text,
                    sources=selected_sources,
                    per_source_limit=per_source_limit,
                    filters=safe_filters,
                ),
                timeout=execution_policy.total_timeout_seconds,
            )
        except TimeoutError:
            raise ResearchDiscoveryTimeoutError("research_discovery_total_timeout") from None

        remaining_timeout = _remaining_timeout_seconds(
            started_at=started_at,
            total_timeout_seconds=timeout_budget,
        )
        try:
            normalized_results = await asyncio.wait_for(
                _normalize_records(
                    raw_records=raw_records,
                    catalog_version=self._catalog.catalog_version,
                ),
                timeout=remaining_timeout,
            )
        except TimeoutError:
            raise ResearchDiscoveryTimeoutError("research_discovery_total_timeout") from None

        remaining_timeout = _remaining_timeout_seconds(
            started_at=started_at,
            total_timeout_seconds=timeout_budget,
        )
        try:
            results = await asyncio.wait_for(
                _enrich_records(
                    results=normalized_results[:total_limit],
                    oa_resolver=self._oa_resolver,
                ),
                timeout=remaining_timeout,
            )
        except TimeoutError:
            raise ResearchDiscoveryTimeoutError("research_discovery_total_timeout") from None
        source_statuses_tuple = _sanitize_source_statuses(source_statuses)
        if not results:
            _raise_for_empty_terminal_outcome(
                selected_sources=selected_sources,
                source_statuses=source_statuses_tuple,
            )

        warnings = _response_warnings(
            source_statuses=source_statuses_tuple,
            results=results,
        )
        effective_config = _effective_config_snapshot(
            selected_sources=selected_sources,
            source_policy=source_policy,
            fallback_policy=fallback_policy,
            filters=safe_filters,
            execution_policy=execution_policy,
            snapshot_retention_hours=self._snapshot_retention_hours,
            defaulted_categories=defaulted_categories,
        )
        metrics = _metrics(
            selected_sources=selected_sources,
            result_count=len(results),
            deduped_result_count=len(normalized_results),
            oa_candidate_count=sum(len(result.oa_candidates) for result in results),
            started_at=started_at,
        )
        response = DiscoverySearchResponse(
            discovery_id="",
            query=query_text,
            results=results,
            source_statuses=source_statuses_tuple,
            warnings=warnings,
            effective_config=effective_config,
            catalog_version=self._catalog.catalog_version,
            metrics=metrics,
        )

        snapshot = await asyncio.to_thread(
            _persist_discovery_snapshot,
            self._snapshot_db_for_user(owner_user_id),
            owner_user_id=owner_user_id,
            query=query_text,
            request_json=_request_snapshot(
                query=query_text,
                source_ids=normalized_source_ids,
                categories=normalized_categories,
                max_sources=max_sources,
                per_source_limit=per_source_limit,
                total_limit=total_limit,
                source_policy=source_policy,
                fallback_policy=fallback_policy,
                filters=safe_filters,
                total_timeout_seconds=timeout_budget,
            ),
            response_json=_response_snapshot(response),
            effective_config_json=effective_config,
            catalog_version=self._catalog.catalog_version,
            retention_hours=self._snapshot_retention_hours,
        )
        return replace(response, discovery_id=snapshot.id)

    def _resolve_sources(
        self,
        *,
        source_ids: list[str],
        categories: list[str],
        max_sources: int | None,
    ) -> tuple[ResearchSourceCatalogEntry, ...]:
        effective_max_sources = max_sources or self._catalog.max_selected_sources
        if not source_ids and not categories:
            return tuple(self._catalog.list_sources()[:effective_max_sources])

        selected_sources, selection_error = self._catalog.resolve_selection(
            source_ids=source_ids,
            categories=categories,
        )
        if selection_error is not None:
            detail = f"{selection_error.code}:{selection_error.selected_count}:{selection_error.limit}"
            if selection_error.code == "source_selection_over_cap":
                raise ResearchDiscoveryValidationError(detail)
            raise ResearchDiscoveryBadRequestError(detail)
        if len(selected_sources) > effective_max_sources:
            raise ResearchDiscoveryValidationError(
                f"source_selection_over_cap:{len(selected_sources)}:{effective_max_sources}"
            )
        return tuple(selected_sources)

    def _snapshot_db_for_user(self, owner_user_id: str) -> ResearchSessionsDB:
        from tldw_Server_API.app.core.DB_Management.ResearchSessionsDB import (
            ResearchSessionsDB,
        )
        from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths

        if self._snapshot_db is not None:
            return self._snapshot_db
        if self._db_factory is not None:
            return self._db_factory(owner_user_id)
        return ResearchSessionsDB(DatabasePaths.get_research_sessions_db_path(owner_user_id))


def _execution_policy(
    *,
    selected_sources: Sequence[ResearchSourceCatalogEntry],
    total_timeout_seconds: float,
) -> DiscoveryExecutionPolicy:
    """Build bounded execution limits for the selected sources."""
    return DiscoveryExecutionPolicy(
        per_source_timeout_seconds=min(
            DEFAULT_PER_SOURCE_TIMEOUT_SECONDS,
            total_timeout_seconds,
        ),
        total_timeout_seconds=total_timeout_seconds,
        max_concurrency=max(1, min(DEFAULT_MAX_CONCURRENCY, len(selected_sources))),
    )


def _normalize_string_sequence(value: Sequence[str] | str | None) -> list[str]:
    """Coerce optional string or string sequence inputs into clean lists."""
    if value is None:
        return []
    raw_values: Iterable[Any] = (value,) if isinstance(value, str) else value
    return [text for item in raw_values if (text := str(item).strip())]


def _fallback_policy_is_disabled(policy: str | Mapping[str, Any] | None) -> bool:
    """Return whether the Phase 1 fallback policy remains disabled."""
    if policy is None:
        return True
    if isinstance(policy, str):
        return policy == "disabled"
    if not policy:
        return True

    mode = str(policy.get("mode") or policy.get("policy") or "disabled").strip()
    enabled = policy.get("enabled")
    return mode == "disabled" and enabled is not True


def _safe_json_mapping(value: Mapping[str, Any] | None) -> dict[str, Any]:
    """Validate and sanitize filter/provider override mappings for snapshots."""
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ResearchDiscoveryBadRequestError("research_discovery_provider_overrides_must_be_mapping")
    _validate_filter_payload(value)
    safe_value = safe_provider_metadata(dict(value))
    _validate_filter_payload(safe_value)
    return safe_value


def _merge_filter_inputs(
    provider_overrides: Mapping[str, Any] | None,
    filters: Mapping[str, Any] | None,
) -> Mapping[str, Any] | None:
    """Merge provider overrides with explicit filters, with filters winning."""
    if provider_overrides is None:
        return filters
    if filters is None:
        return provider_overrides
    merged = dict(provider_overrides)
    merged.update(filters)
    return merged


def _validate_filter_payload(value: Mapping[str, Any]) -> None:
    """Reject filter payloads that are too large, deep, or key-heavy."""
    key_count = _count_filter_keys(value, depth=1)
    if key_count > MAX_DISCOVERY_FILTER_KEYS:
        raise ResearchDiscoveryValidationError("research_discovery_filters_too_many_keys")

    serialized = json.dumps(_to_jsonable(value), sort_keys=True, separators=(",", ":"))
    if len(serialized.encode("utf-8")) > MAX_DISCOVERY_FILTER_BYTES:
        raise ResearchDiscoveryValidationError("research_discovery_filters_too_large")


def _count_filter_keys(value: Any, *, depth: int) -> int:
    """Count mapping keys recursively while enforcing maximum depth."""
    if depth > MAX_DISCOVERY_FILTER_DEPTH:
        raise ResearchDiscoveryValidationError("research_discovery_filters_too_deep")
    if isinstance(value, Mapping):
        return len(value) + sum(_count_filter_keys(item, depth=depth + 1) for item in value.values())
    if isinstance(value, (tuple, list)):
        return sum(_count_filter_keys(item, depth=depth + 1) for item in value)
    return 0


def _raise_for_empty_terminal_outcome(
    *,
    selected_sources: Sequence[ResearchSourceCatalogEntry],
    source_statuses: Sequence[SourceStatus],
) -> None:
    """Raise a typed terminal error when all selected sources failed."""
    status_by_source_id = {
        status.source_id: status
        for status in source_statuses
        if status.source_id in {source.source_id for source in selected_sources}
    }
    if not selected_sources or len(status_by_source_id) != len(selected_sources):
        return

    selected_statuses = tuple(status_by_source_id[source.source_id] for source in selected_sources)
    if all(status.status in NO_RUNNABLE_STATUSES for status in selected_statuses):
        raise ResearchDiscoveryValidationError("research_discovery_no_runnable_sources")
    if all(status.status in FAILURE_STATUSES for status in selected_statuses):
        raise ResearchDiscoveryUpstreamError("research_discovery_all_sources_failed")


def _response_warnings(
    *,
    source_statuses: Sequence[SourceStatus],
    results: Sequence[DiscoveryResult],
) -> tuple[str, ...]:
    """Collect stable warning strings from source statuses and results."""
    warnings: list[str] = []
    for status in source_statuses:
        if status.status != "ok":
            if status.message:
                warnings.append(f"{status.source_id}:{status.status}:{status.message}")
            else:
                warnings.append(f"{status.source_id}:{status.status}")
        for warning in status.warnings:
            warnings.append(f"{status.source_id}:{warning}")
    for result in results:
        warnings.extend(result.warnings)
        for candidate in result.oa_candidates:
            warnings.extend(candidate.warnings)
    return tuple(_dedupe_strings(warnings))


def _remaining_timeout_seconds(
    *,
    started_at: float,
    total_timeout_seconds: float,
) -> float:
    """Return remaining total search budget or raise a typed timeout."""
    remaining = total_timeout_seconds - (time.perf_counter() - started_at)
    if remaining <= 0:
        raise ResearchDiscoveryTimeoutError("research_discovery_total_timeout")
    return remaining


async def _normalize_records(
    *,
    raw_records: list[dict[str, Any]],
    catalog_version: str,
) -> tuple[DiscoveryResult, ...]:
    """Normalize and dedupe raw provider records without OA resolver work."""
    return tuple(normalize_and_merge_records(raw_records, catalog_version=catalog_version))


async def _enrich_records(
    *,
    results: Sequence[DiscoveryResult],
    oa_resolver: ResearchOAResolver,
) -> tuple[DiscoveryResult, ...]:
    """Attach resolver OA candidates to the already-limited result set."""
    enriched_results = await _attach_oa_resolver_candidates(
        results=results,
        oa_resolver=oa_resolver,
    )
    return _sanitize_results(enriched_results)


async def _attach_oa_resolver_candidates(
    *,
    results: Sequence[DiscoveryResult],
    oa_resolver: ResearchOAResolver,
) -> tuple[DiscoveryResult, ...]:
    """Resolve extra OA candidates with bounded background concurrency."""
    enriched: list[DiscoveryResult] = []
    if not results:
        return ()

    semaphore = asyncio.Semaphore(DEFAULT_OA_RESOLVER_CONCURRENCY)

    async def _resolve_with_limit(result: DiscoveryResult) -> Sequence[Any]:
        async with semaphore:
            return await asyncio.to_thread(
                _resolve_oa_candidates_for_result,
                oa_resolver=oa_resolver,
                result=result,
            )

    resolver_candidate_groups = await asyncio.gather(*(_resolve_with_limit(result) for result in results))
    for result, resolver_candidates in zip(results, resolver_candidate_groups):
        oa_candidates = _merge_oa_candidates(result.oa_candidates, resolver_candidates)
        ranking_signals = dict(result.ranking_signals)
        ranking_signals["has_oa_candidate"] = bool(oa_candidates)
        enriched.append(
            replace(
                result,
                oa_candidates=oa_candidates,
                recommended_candidate_id=(oa_candidates[0].candidate_id if oa_candidates else None),
                ingest_eligible=any(candidate.safe_url for candidate in oa_candidates),
                ranking_signals=ranking_signals,
            )
        )
    return tuple(enriched)


def _resolve_oa_candidates_for_result(
    *,
    oa_resolver: ResearchOAResolver,
    result: DiscoveryResult,
) -> Sequence[Any]:
    """Resolve OA candidates for one result, isolating resolver failures."""
    try:
        return oa_resolver.resolve_for_result(
            result_fingerprint=result.fingerprint,
            source_id=result.primary_source_id,
            provider=result.primary_provider,
            doi=result.doi,
            provider_ids=result.provider_ids,
            raw_urls=(),
        )
    except Exception:
        return ()


def _merge_oa_candidates(
    current_candidates: Sequence[Any],
    resolver_candidates: Sequence[Any],
) -> tuple[Any, ...]:
    """Merge existing and resolver candidates by stable candidate id."""
    merged: list[Any] = []
    seen_candidate_ids: set[str] = set()
    for candidate in (*current_candidates, *resolver_candidates):
        candidate_id = str(getattr(candidate, "candidate_id", ""))
        if candidate_id in seen_candidate_ids:
            continue
        seen_candidate_ids.add(candidate_id)
        merged.append(candidate)
    return tuple(merged)


def _sanitize_source_statuses(statuses: Sequence[SourceStatus]) -> tuple[SourceStatus, ...]:
    """Sanitize source status messages and warnings before returning them."""
    return tuple(
        replace(
            status,
            message=_safe_warning_text(status.message) if status.message else None,
            warnings=_safe_warning_tuple(status.warnings),
        )
        for status in statuses
    )


def _sanitize_results(results: Sequence[DiscoveryResult]) -> tuple[DiscoveryResult, ...]:
    """Sanitize all result warning and diagnostic metadata fields."""
    return tuple(_sanitize_result(result) for result in results)


def _sanitize_result(result: DiscoveryResult) -> DiscoveryResult:
    """Sanitize one discovery result and nested provenance/candidates."""
    return replace(
        result,
        warnings=_safe_warning_tuple(result.warnings),
        oa_candidates=tuple(
            replace(candidate, warnings=_safe_warning_tuple(candidate.warnings)) for candidate in result.oa_candidates
        ),
        merged_provenance=tuple(
            replace(
                provenance,
                status=_safe_warning_text(provenance.status),
                warnings=_safe_warning_tuple(provenance.warnings),
                safe_metadata=_sanitize_warning_metadata(provenance.safe_metadata),
            )
            for provenance in result.merged_provenance
        ),
        safe_metadata=_sanitize_warning_metadata(result.safe_metadata),
    )


def _safe_warning_tuple(warnings: Sequence[str]) -> tuple[str, ...]:
    """Sanitize and deduplicate warning text."""
    return tuple(_dedupe_strings(_safe_warning_text(warning) for warning in warnings))


def _safe_warning_text(value: Any) -> str:
    """Return warning text with unsafe URL or secret material redacted."""
    text = str(value or "").strip()
    if not text:
        return UNSAFE_WARNING_TEXT
    if _warning_text_is_unsafe(text):
        return UNSAFE_WARNING_TEXT
    return text


def _warning_text_is_unsafe(text: str) -> bool:
    """Return whether warning text contains secret-like material."""
    lowered = text.lower()
    if any(part in lowered for part in _UNSAFE_WARNING_PARTS):
        return True
    return _text_contains_unsafe_url_material(text)


def _text_contains_unsafe_url_material(text: str) -> bool:
    """Return whether text embeds an unsafe URL."""
    for raw_url in _URL_IN_TEXT_RE.findall(text):
        if has_unsafe_url_material(raw_url.rstrip(".,;:[]")):
            return True
    return False


def _contains_unsafe_url_text(value: Any) -> bool:
    """Recursively detect unsafe URL text inside JSON-like values."""
    if isinstance(value, str):
        return _text_contains_unsafe_url_material(value)
    if isinstance(value, Mapping):
        return any(_contains_unsafe_url_text(key) or _contains_unsafe_url_text(item) for key, item in value.items())
    if isinstance(value, (tuple, list)):
        return any(_contains_unsafe_url_text(item) for item in value)
    return False


def _sanitize_warning_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Sanitize diagnostic metadata while preserving benign values."""
    cleaned: dict[str, Any] = {}
    for key, value in metadata.items():
        cleaned[str(key)] = _sanitize_warning_value(
            value,
            sanitize_text=_is_diagnostic_metadata_key(key),
        )
    return cleaned


def _sanitize_warning_value(value: Any, *, sanitize_text: bool) -> Any:
    """Sanitize metadata values, always redacting unsafe embedded URLs."""
    if isinstance(value, str):
        if sanitize_text or _text_contains_unsafe_url_material(value):
            return _safe_warning_text(value)
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _sanitize_warning_value(
                item,
                sanitize_text=sanitize_text or _is_diagnostic_metadata_key(key),
            )
            for key, item in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [_sanitize_warning_value(item, sanitize_text=sanitize_text) for item in value]
    return value


def _is_diagnostic_metadata_key(key: Any) -> bool:
    """Return whether a metadata key commonly carries diagnostics."""
    key_text = str(key).strip().lower()
    separator_normalized = re.sub(r"[^a-z0-9]+", "_", key_text).strip("_")
    compact = re.sub(r"[^a-z0-9]+", "", key_text)
    variants = {variant for variant in (key_text, separator_normalized, compact) if variant}
    return any(part in variant for variant in variants for part in _DIAGNOSTIC_METADATA_KEY_PARTS)


def _metrics(
    *,
    selected_sources: Sequence[ResearchSourceCatalogEntry],
    result_count: int,
    deduped_result_count: int,
    oa_candidate_count: int,
    started_at: float,
) -> DiscoveryMetrics:
    """Build aggregate metrics for a completed discovery response."""
    return DiscoveryMetrics(
        selected_source_count=len(selected_sources),
        result_count=result_count,
        deduped_result_count=deduped_result_count,
        oa_candidate_count=oa_candidate_count,
        elapsed_ms=max(0.0, (time.perf_counter() - started_at) * 1000),
    )


def _request_snapshot(
    *,
    query: str,
    source_ids: Sequence[str],
    categories: Sequence[str],
    max_sources: int | None,
    per_source_limit: int,
    total_limit: int,
    source_policy: str,
    fallback_policy: str | Mapping[str, Any] | None,
    filters: Mapping[str, Any],
    total_timeout_seconds: float,
) -> dict[str, Any]:
    """Build the sanitized request payload persisted with a snapshot."""
    return _to_jsonable(
        {
            "query": query,
            "source_ids": list(source_ids),
            "categories": list(categories),
            "max_sources": max_sources,
            "per_source_limit": per_source_limit,
            "total_limit": total_limit,
            "source_policy": source_policy,
            "fallback_policy": "disabled",
            "filters": dict(filters),
            "total_timeout_seconds": total_timeout_seconds,
        }
    )


def _effective_config_snapshot(
    *,
    selected_sources: Sequence[ResearchSourceCatalogEntry],
    source_policy: str,
    fallback_policy: str | Mapping[str, Any] | None,
    filters: Mapping[str, Any],
    execution_policy: DiscoveryExecutionPolicy,
    snapshot_retention_hours: int,
    defaulted_categories: Sequence[str],
) -> dict[str, Any]:
    """Build the sanitized effective execution config for response/snapshot."""
    return _to_jsonable(
        {
            "source_ids": [source.source_id for source in selected_sources],
            "categories": sorted({source.category for source in selected_sources}),
            "defaulted_categories": list(defaulted_categories),
            "source_policy": source_policy,
            "fallback_policy": "disabled",
            "filters": dict(filters),
            "execution_policy": execution_policy,
            "snapshot_retention_hours": snapshot_retention_hours,
        }
    )


def _response_snapshot(response: DiscoverySearchResponse) -> dict[str, Any]:
    """Build a response snapshot without the generated snapshot id."""
    payload = _to_jsonable(response)
    if isinstance(payload, dict):
        payload.pop("discovery_id", None)
        return payload
    return {}


def _persist_discovery_snapshot(
    snapshot_db: ResearchSessionsDB,
    *,
    owner_user_id: str,
    query: str,
    request_json: dict[str, Any],
    response_json: dict[str, Any],
    effective_config_json: dict[str, Any],
    catalog_version: str,
    retention_hours: int,
) -> Any:
    """Delete expired rows and persist one discovery snapshot synchronously."""
    snapshot_db.delete_expired_discovery_snapshots()
    return snapshot_db.create_discovery_snapshot(
        owner_user_id=owner_user_id,
        query=query,
        request_json=request_json,
        response_json=response_json,
        effective_config_json=effective_config_json,
        catalog_version=catalog_version,
        retention_hours=retention_hours,
    )


def _to_jsonable(value: Any) -> Any:
    """Convert dataclasses and JSON-like values into snapshot-safe values."""
    if is_dataclass(value):
        return _to_jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _dedupe_strings(values: Iterable[str]) -> list[str]:
    """Deduplicate strings while preserving input order."""
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped
