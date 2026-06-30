from __future__ import annotations

"""Replay and repair helpers for Sync v2 accepted envelopes."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

from .materializers import MaterializationResult, SyncMaterializer
from .models import SyncDomain, SyncEnvelope
from .store import SyncV2Store


@dataclass(frozen=True, slots=True)
class SyncReplayRepairEnvelopeError:
    """One envelope-level replay failure safe to return in repair responses."""

    server_cursor: int | None
    client_envelope_id: str
    domain: SyncDomain
    object_id: str
    error_code: str | None = None
    message: str | None = None


@dataclass(frozen=True, slots=True)
class SyncReplayRepairDomainResult:
    """Per-domain replay/repair counters."""

    domain: SyncDomain
    scanned_count: int = 0
    attempted_count: int = 0
    applied_count: int = 0
    failed_count: int = 0
    conflict_count: int = 0
    skipped_count: int = 0
    last_cursor: int = 0
    errors: list[SyncReplayRepairEnvelopeError] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class SyncReplayRepairResult:
    """Aggregate replay/repair outcome for one dataset."""

    dataset_id: str
    domains: list[SyncDomain]
    from_cursor: int = 0
    to_cursor: int = 0
    scanned_count: int = 0
    attempted_count: int = 0
    applied_count: int = 0
    failed_count: int = 0
    conflict_count: int = 0
    skipped_count: int = 0
    domain_results: list[SyncReplayRepairDomainResult] = field(default_factory=list)
    repair_status: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class _DomainAccumulator:
    domain: SyncDomain
    scanned_count: int = 0
    attempted_count: int = 0
    applied_count: int = 0
    failed_count: int = 0
    conflict_count: int = 0
    skipped_count: int = 0
    last_cursor: int = 0
    errors: list[SyncReplayRepairEnvelopeError] = field(default_factory=list)

    def result(self) -> SyncReplayRepairDomainResult:
        return SyncReplayRepairDomainResult(
            domain=self.domain,
            scanned_count=self.scanned_count,
            attempted_count=self.attempted_count,
            applied_count=self.applied_count,
            failed_count=self.failed_count,
            conflict_count=self.conflict_count,
            skipped_count=self.skipped_count,
            last_cursor=self.last_cursor,
            errors=list(self.errors),
        )


class SyncReplayRepairer:
    """Re-apply accepted envelopes through the configured materializers."""

    def __init__(
        self,
        *,
        store: SyncV2Store,
        materializers: Mapping[SyncDomain, SyncMaterializer],
        materialize: Callable[[SyncEnvelope], MaterializationResult],
        snapshot: Callable[[SyncEnvelope], SyncEnvelope],
        scan_limit: int,
    ) -> None:
        self.store = store
        self.materializers = materializers
        self.materialize = materialize
        self.snapshot = snapshot
        self.scan_limit = max(1, scan_limit)

    def run(
        self,
        *,
        dataset_id: str,
        domains: Sequence[SyncDomain],
        since_cursor: int = 0,
        failed_only: bool = False,
        limit: int | None = None,
    ) -> SyncReplayRepairResult:
        """Replay accepted envelopes for selected domains in cursor order."""

        selected_domains = list(domains)
        selected = set(selected_domains)
        accumulators: dict[SyncDomain, _DomainAccumulator] = {}
        cursor = since_cursor
        remaining = limit
        scanned_count = 0

        while True:
            if remaining is not None and remaining <= 0:
                break
            page_start_cursor = cursor
            page_limit = self.scan_limit if remaining is None else min(self.scan_limit, remaining)
            page = self.store.list_accepted_envelopes_for_replay(
                dataset_id,
                since_cursor=cursor,
                limit=page_limit,
            )
            if not page:
                break

            for envelope in page:
                cursor = max(cursor, envelope.server_cursor or cursor)
                if envelope.domain not in selected:
                    continue
                scanned_count += 1
                if remaining is not None:
                    remaining -= 1
                domain_result = _accumulator(accumulators, envelope.domain)
                domain_result.scanned_count += 1
                domain_result.last_cursor = max(domain_result.last_cursor, envelope.server_cursor or 0)
                self._repair_envelope(
                    envelope,
                    failed_only=failed_only,
                    result=domain_result,
                )
                if remaining is not None and remaining <= 0:
                    break

            next_cursor = max((item.server_cursor or cursor for item in page), default=cursor)
            if next_cursor <= page_start_cursor:
                break
            cursor = next_cursor

        domain_results = [accumulators[domain].result() for domain in selected_domains if domain in accumulators]
        attempted_count = sum(item.attempted_count for item in domain_results)
        applied_count = sum(item.applied_count for item in domain_results)
        failed_count = sum(item.failed_count for item in domain_results)
        conflict_count = sum(item.conflict_count for item in domain_results)
        skipped_count = sum(item.skipped_count for item in domain_results)
        repair_status = {
            "status": "repair_needed" if failed_count or conflict_count else "healthy",
            "failed_count": failed_count,
            "conflict_count": conflict_count,
            "skipped_count": skipped_count,
        }
        return SyncReplayRepairResult(
            dataset_id=dataset_id,
            domains=selected_domains,
            from_cursor=since_cursor,
            to_cursor=cursor,
            scanned_count=scanned_count,
            attempted_count=attempted_count,
            applied_count=applied_count,
            failed_count=failed_count,
            conflict_count=conflict_count,
            skipped_count=skipped_count,
            domain_results=domain_results,
            repair_status=repair_status,
        )

    def _repair_envelope(
        self,
        envelope: SyncEnvelope,
        *,
        failed_only: bool,
        result: _DomainAccumulator,
    ) -> None:
        if envelope.apply_status == "conflict":
            result.conflict_count += 1
            result.skipped_count += 1
            return
        if failed_only and envelope.apply_status != "failed":
            result.skipped_count += 1
            return
        if envelope.domain not in self.materializers:
            result.skipped_count += 1
            return

        result.attempted_count += 1
        materialization = self.materialize(envelope)
        snapshot = self.snapshot(envelope)
        status = snapshot.apply_status or materialization.status
        if status == "applied":
            result.applied_count += 1
            return
        if status == "conflict":
            result.conflict_count += 1
            return
        if status == "failed":
            result.failed_count += 1
            result.errors.append(_error_from_envelope(snapshot, materialization))
            return
        result.skipped_count += 1


def _accumulator(
    accumulators: dict[SyncDomain, _DomainAccumulator],
    domain: SyncDomain,
) -> _DomainAccumulator:
    result = accumulators.get(domain)
    if result is None:
        result = _DomainAccumulator(domain=domain)
        accumulators[domain] = result
    return result


def _error_from_envelope(
    envelope: SyncEnvelope,
    materialization: MaterializationResult,
) -> SyncReplayRepairEnvelopeError:
    return SyncReplayRepairEnvelopeError(
        server_cursor=envelope.server_cursor,
        client_envelope_id=envelope.client_envelope_id,
        domain=envelope.domain,
        object_id=envelope.object_id,
        error_code=envelope.apply_error_code or materialization.error_code,
        message=envelope.apply_error_message or materialization.message,
    )


__all__ = [
    "SyncReplayRepairDomainResult",
    "SyncReplayRepairEnvelopeError",
    "SyncReplayRepairResult",
    "SyncReplayRepairer",
]
