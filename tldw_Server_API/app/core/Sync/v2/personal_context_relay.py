"""Ordered relay for encrypted Personal Context authority publications."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from time import monotonic_ns
from typing import Any, Literal

from tldw_Server_API.app.core.Personalization.personal_context_publication import (
    PublicationRelayPoisoned,
    PublicationSourceRow,
)


@dataclass(frozen=True, slots=True)
class PersonalContextRelayResult:
    """Content-free progress from one bounded authority relay attempt."""

    staged_rows: int
    source_exhausted: bool
    visible_lookahead: bool
    continuation: Literal[
        "complete", "personal_context_relay_pending", "relay_poisoned"
    ]
    inspected_rows: int = 0


AuthorityStager = Callable[[PublicationSourceRow, str, str], int]
AuthorityFinalizer = Callable[[PublicationSourceRow, int, str, str], None]
AuthorityCanceller = Callable[[PublicationSourceRow, int, str, str], None]


class PersonalContextAuthoritySourceError(RuntimeError):
    """Authenticated source content is malformed and requires durable attention."""


@dataclass(slots=True)
class PersonalContextRelay:
    """Relay the encrypted canonical journal in durable profile order."""

    publications: Any
    stage_authority: AuthorityStager
    finalize_authority: AuthorityFinalizer | None = None
    cancel_authority: AuthorityCanceller | None = None
    clock_ns: Callable[[], int] = monotonic_ns

    def relay_profile(
        self,
        *,
        user_id: str,
        profile_id: str,
        dataset_id: str,
        after_server_cursor: int | None,
        row_budget: int = 100,
        wall_time_ms: int = 100,
        deadline_ns: int | None = None,
    ) -> PersonalContextRelayResult:
        """Stage the first incomplete batch only, bounded by rows and wall time."""

        del after_server_cursor
        if row_budget < 1 or wall_time_ms < 1:
            raise ValueError("relay limits must be positive")
        deadline_ns = deadline_ns or self.clock_ns() + wall_time_ms * 1_000_000
        with self.publications.profile_lease(profile_id) as lease:
            if lease is None:
                return PersonalContextRelayResult(
                    0, False, False, "personal_context_relay_pending"
                )
            staged = 0
            inspected = 0
            while True:
                try:
                    batch = self.publications.earliest_nonterminal_batch(
                        profile_id,
                        row_limit=row_budget - inspected,
                    )
                except PublicationRelayPoisoned:
                    return PersonalContextRelayResult(staged, False, False, "relay_poisoned", inspected)
                if batch is None:
                    return PersonalContextRelayResult(staged, True, False, "complete", inspected)
                acknowledged_ordinals = {
                    row.batch_ordinal for row in batch.rows if row.row_state == "acknowledged"
                }
                for row in batch.rows:
                    if inspected >= row_budget or self.clock_ns() >= deadline_ns:
                        return PersonalContextRelayResult(staged, False, False, "personal_context_relay_pending", inspected)
                    inspected += 1
                    if row.row_state == "acknowledged":
                        if self.finalize_authority is not None:
                            cursor = row.sync_server_cursor
                            if cursor is None:
                                return PersonalContextRelayResult(staged, False, False, "personal_context_relay_pending", inspected)
                            claimed_row = replace(row, relay_owner_token=lease.owner_token)
                            if not self.publications.renew_lease(lease) or not self.publications.row_is_current(claimed_row, lease):
                                return PersonalContextRelayResult(staged, False, False, "personal_context_relay_pending", inspected)
                            try:
                                self.finalize_authority(claimed_row, cursor, dataset_id, user_id)
                            except Exception:  # noqa: BLE001 - durable pending row is retryable.
                                return PersonalContextRelayResult(staged, False, False, "personal_context_relay_pending", inspected)
                        continue
                    if row.role == "manifest" and any(item.role == "semantic" and item.batch_ordinal not in acknowledged_ordinals for item in batch.rows):
                        return PersonalContextRelayResult(staged, False, False, "personal_context_relay_pending", inspected)
                    claimed_row = replace(row, relay_owner_token=lease.owner_token)
                    cursor = row.sync_server_cursor
                    if row.row_state == "staged" and cursor is None:
                        return PersonalContextRelayResult(staged, False, False, "personal_context_relay_pending", inspected)
                    if row.row_state == "pending":
                        if not self.publications.renew_lease(lease) or not self.publications.row_is_current(claimed_row, lease):
                            return PersonalContextRelayResult(staged, False, False, "personal_context_relay_pending", inspected)
                        try:
                            cursor = self.stage_authority(claimed_row, dataset_id, user_id)
                        except PersonalContextAuthoritySourceError:
                            self.publications.mark_attention(batch, lease=lease)
                            return PersonalContextRelayResult(staged, False, False, "relay_poisoned", inspected)
                        except Exception:  # noqa: BLE001 - storage/head/transport failures retry.
                            return PersonalContextRelayResult(staged, False, False, "personal_context_relay_pending", inspected)
                        if isinstance(cursor, bool) or not isinstance(cursor, int) or cursor < 1:
                            return PersonalContextRelayResult(staged, False, False, "personal_context_relay_pending", inspected)
                        if not self.publications.renew_lease(lease) or not self.publications.row_is_current(claimed_row, lease):
                            self._cancel(claimed_row, cursor, dataset_id, user_id)
                            return PersonalContextRelayResult(staged, False, False, "personal_context_relay_pending", inspected)
                        try:
                            self.publications.record_staged_row(
                                claimed_row,
                                server_cursor=cursor,
                                lease=lease,
                            )
                        except Exception:  # noqa: BLE001 - lost source CAS needs compensation.
                            self._cancel(claimed_row, cursor, dataset_id, user_id)
                            return PersonalContextRelayResult(staged, False, False, "personal_context_relay_pending", inspected)
                    if cursor is None:
                        return PersonalContextRelayResult(staged, False, False, "personal_context_relay_pending", inspected)
                    if not self.publications.renew_lease(lease) or not self.publications.row_is_current(claimed_row, lease):
                        self._cancel(claimed_row, cursor, dataset_id, user_id)
                        return PersonalContextRelayResult(staged, False, False, "personal_context_relay_pending", inspected)
                    if self.finalize_authority is not None:
                        try:
                            self.finalize_authority(claimed_row, cursor, dataset_id, user_id)
                        except Exception:  # noqa: BLE001 - staged receipt repairs on retry.
                            return PersonalContextRelayResult(staged, False, False, "personal_context_relay_pending", inspected)
                    if not self.publications.renew_lease(lease) or not self.publications.row_is_current(claimed_row, lease):
                        return PersonalContextRelayResult(staged, False, False, "personal_context_relay_pending", inspected)
                    self.publications.acknowledge_row(claimed_row, server_cursor=cursor, lease=lease)
                    acknowledged_ordinals.add(row.batch_ordinal)
                    staged += 1
                if not self.publications.complete_if_acknowledged(batch, lease=lease):
                    return PersonalContextRelayResult(staged, False, False, "personal_context_relay_pending", inspected)

    def _cancel(
        self,
        row: PublicationSourceRow,
        server_cursor: int,
        dataset_id: str,
        user_id: str,
    ) -> None:
        """Best-effort compensation; only the exact pending row can be removed."""

        if self.cancel_authority is None:
            return
        try:
            self.cancel_authority(row, server_cursor, dataset_id, user_id)
        except Exception:  # noqa: BLE001 - cancellation is best-effort retry cleanup.
            return


__all__ = [
    "PersonalContextAuthoritySourceError",
    "PersonalContextRelay",
    "PersonalContextRelayResult",
]
