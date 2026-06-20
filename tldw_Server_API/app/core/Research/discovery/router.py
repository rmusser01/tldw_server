"""Provider routing for research discovery sources."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Iterable, Mapping
from typing import Any, Protocol

from .catalog import ResearchSourceCatalog
from .models import ResearchSourceCatalogEntry, SourceStatus


DEFAULT_ADAPTER_VERSION = "research-discovery-router-v1"
PROVIDER_REQUEST_FAILED_MESSAGE = "Provider request failed."
INTERNAL_ADAPTER_ERROR_MESSAGE = "Discovery adapter failed unexpectedly."
TIMEOUT_CONTINUATION_WARNING = "provider_call_may_continue_after_timeout"


class DiscoveryProviderError(Exception):
    """Sanitized exception for failures returned by external discovery providers."""

    def __init__(
        self,
        _message: str | None = None,
        *,
        safe_message: str = PROVIDER_REQUEST_FAILED_MESSAGE,
    ) -> None:
        self.safe_message = safe_message
        super().__init__(safe_message)


class DiscoveryProviderAdapter(Protocol):
    async def search(
        self,
        *,
        query: str,
        source: ResearchSourceCatalogEntry,
        limit: int,
        filters: dict[str, Any],
    ) -> list[dict[str, Any]]: ...


class SourceRateLimiter(Protocol):
    def __call__(self, source_id: str) -> bool | Awaitable[bool]: ...


class ResearchSourceRouter:
    """Route resolved catalog sources to configured provider adapters."""

    def __init__(
        self,
        *,
        catalog: ResearchSourceCatalog,
        adapters: Mapping[str, DiscoveryProviderAdapter],
        per_source_timeout_seconds: float = 10.0,
        max_concurrency: int = 4,
        rate_limiter: SourceRateLimiter | None = None,
    ) -> None:
        if per_source_timeout_seconds <= 0:
            raise ValueError("per_source_timeout_seconds must be positive")
        if max_concurrency <= 0:
            raise ValueError("max_concurrency must be positive")

        self._catalog = catalog
        self._adapters = dict(adapters)
        self._per_source_timeout_seconds = per_source_timeout_seconds
        self._max_concurrency = max_concurrency
        self._rate_limiter = rate_limiter
        self._adapter_names = tuple(sorted(self._adapters))

    @property
    def adapter_names(self) -> tuple[str, ...]:
        """Return configured adapter names in deterministic order."""
        return self._adapter_names

    async def search_sources(
        self,
        *,
        query: str,
        sources: Iterable[ResearchSourceCatalogEntry],
        per_source_limit: int,
        filters: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], list[SourceStatus]]:
        """Search selected sources and return raw records plus per-source statuses."""
        selected_sources = tuple(sorted(sources, key=_source_sort_key))
        semaphore = asyncio.Semaphore(self._max_concurrency)

        tasks = [
            asyncio.create_task(
                self._search_one_source(
                    query=query,
                    source=source,
                    per_source_limit=per_source_limit,
                    filters=dict(filters),
                    semaphore=semaphore,
                )
            )
            for source in selected_sources
        ]

        if not tasks:
            return [], []

        results = await asyncio.gather(*tasks)
        records: list[dict[str, Any]] = []
        statuses: list[SourceStatus] = []
        for source_records, status in results:
            records.extend(source_records)
            statuses.append(status)
        return records, statuses

    async def _search_one_source(
        self,
        *,
        query: str,
        source: ResearchSourceCatalogEntry,
        per_source_limit: int,
        filters: dict[str, Any],
        semaphore: asyncio.Semaphore,
    ) -> tuple[list[dict[str, Any]], SourceStatus]:
        provider = source.provider_adapter

        blocked_status = self._preflight_status(source)
        if blocked_status is not None:
            return [], blocked_status

        if provider is None:
            return [], SourceStatus(
                source_id=source.source_id,
                provider=None,
                status="provider_not_configured",
                message="Source provider adapter is not configured.",
                result_count=0,
                elapsed_ms=None,
                warnings=(),
            )
        adapter = self._adapters[provider]

        async with semaphore:
            started_at = time.perf_counter()
            try:
                allowed = await self._rate_limit_allows(source.source_id)
                if not allowed:
                    return [], SourceStatus(
                        source_id=source.source_id,
                        provider=provider,
                        status="rate_limited",
                        message="Source rate limit denied this request.",
                        result_count=0,
                        elapsed_ms=_elapsed_ms(started_at),
                        warnings=(),
                    )

                raw_records = await asyncio.wait_for(
                    adapter.search(
                        query=query,
                        source=source,
                        limit=per_source_limit,
                        filters=filters,
                    ),
                    timeout=self._per_source_timeout_seconds,
                )
                enriched_records = [
                    self._enrich_record(record=record, source=source, provider=provider)
                    for record in _validate_adapter_records(raw_records)
                ]
            except TimeoutError:
                # asyncio.to_thread calls cannot be stopped once dispatched; the
                # timeout only releases this router task back to the caller.
                return [], SourceStatus(
                    source_id=source.source_id,
                    provider=provider,
                    status="timeout",
                    message="Provider request timed out.",
                    result_count=0,
                    elapsed_ms=_elapsed_ms(started_at),
                    warnings=(TIMEOUT_CONTINUATION_WARNING,),
                )
            except DiscoveryProviderError as exc:
                return [], SourceStatus(
                    source_id=source.source_id,
                    provider=provider,
                    status="provider_error",
                    message=exc.safe_message,
                    result_count=0,
                    elapsed_ms=_elapsed_ms(started_at),
                    warnings=(),
                )
            except Exception:
                return [], SourceStatus(
                    source_id=source.source_id,
                    provider=provider,
                    status="internal_error",
                    message=INTERNAL_ADAPTER_ERROR_MESSAGE,
                    result_count=0,
                    elapsed_ms=_elapsed_ms(started_at),
                    warnings=(),
                )

        return enriched_records, SourceStatus(
            source_id=source.source_id,
            provider=provider,
            status="ok",
            message=None,
            result_count=len(enriched_records),
            elapsed_ms=_elapsed_ms(started_at),
            warnings=(),
        )

    def _preflight_status(
        self,
        source: ResearchSourceCatalogEntry,
    ) -> SourceStatus | None:
        provider = source.provider_adapter
        if not source.enabled or source.default_discovery_mode == "disabled":
            return SourceStatus(
                source_id=source.source_id,
                provider=provider,
                status="policy_blocked",
                message="Source discovery is disabled by policy.",
                result_count=0,
                elapsed_ms=None,
                warnings=(),
            )

        if source.capabilities.requires_credentials and not source.configured:
            return SourceStatus(
                source_id=source.source_id,
                provider=provider,
                status="credentials_missing",
                message="Source credentials are not configured.",
                result_count=0,
                elapsed_ms=None,
                warnings=(),
            )

        if provider is None or provider not in self._adapters:
            return SourceStatus(
                source_id=source.source_id,
                provider=provider,
                status="provider_not_configured",
                message="Source provider adapter is not configured.",
                result_count=0,
                elapsed_ms=None,
                warnings=(),
            )

        return None

    async def _rate_limit_allows(self, source_id: str) -> bool:
        if self._rate_limiter is None:
            return True

        decision = self._rate_limiter(source_id)
        if isinstance(decision, Awaitable):
            decision = await decision
        return bool(decision)

    @staticmethod
    def _enrich_record(
        *,
        record: dict[str, Any],
        source: ResearchSourceCatalogEntry,
        provider: str,
    ) -> dict[str, Any]:
        enriched = dict(record)
        enriched["source_id"] = source.source_id
        enriched["source_category"] = source.category
        enriched["provider"] = enriched.get("provider") or provider
        enriched["discovery_mode"] = source.default_discovery_mode
        enriched["adapter_version"] = enriched.get("adapter_version") or (f"{provider}:{DEFAULT_ADAPTER_VERSION}")
        enriched["source_priority"] = source.priority
        return enriched


def _source_sort_key(source: ResearchSourceCatalogEntry) -> tuple[int, str]:
    return source.priority, source.source_id


def _validate_adapter_records(records: object) -> list[dict[str, Any]]:
    if not isinstance(records, list):
        raise TypeError("adapter result must be a list")
    for record in records:
        if not isinstance(record, dict):
            raise TypeError("adapter result records must be mappings")
    return records


def _elapsed_ms(started_at: float) -> float:
    return (time.perf_counter() - started_at) * 1000.0
