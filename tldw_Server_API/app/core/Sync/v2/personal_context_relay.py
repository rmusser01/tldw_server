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


AuthorityStager = Callable[[PublicationSourceRow, str, str], int]


@dataclass(slots=True)
class PersonalContextRelay:
    """Relay the encrypted canonical journal in durable profile order."""

    publications: Any
    stage_authority: AuthorityStager
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
    ) -> PersonalContextRelayResult:
        """Stage the first incomplete batch only, bounded by rows and wall time."""

        del after_server_cursor
        if row_budget < 1 or wall_time_ms < 1:
            raise ValueError("relay limits must be positive")
        deadline_ns = self.clock_ns() + wall_time_ms * 1_000_000
        with self.publications.profile_lease(profile_id) as lease:
            if lease is None:
                return PersonalContextRelayResult(
                    0, False, False, "personal_context_relay_pending"
                )
            staged = 0
            while True:
                try:
                    batch = self.publications.earliest_nonterminal_batch(profile_id)
                except PublicationRelayPoisoned:
                    return PersonalContextRelayResult(staged, False, False, "relay_poisoned")
                if batch is None:
                    return PersonalContextRelayResult(staged, True, False, "complete")
                acknowledged_ordinals = {
                    row.batch_ordinal for row in batch.rows if row.row_state == "acknowledged"
                }
                for row in batch.rows:
                    if row.row_state == "acknowledged":
                        continue
                    if staged >= row_budget or self.clock_ns() >= deadline_ns:
                        return PersonalContextRelayResult(staged, False, False, "personal_context_relay_pending")
                    if row.role == "manifest" and any(item.role == "semantic" and item.batch_ordinal not in acknowledged_ordinals for item in batch.rows):
                        return PersonalContextRelayResult(staged, False, False, "personal_context_relay_pending")
                    if not self.publications.renew_lease(lease):
                        return PersonalContextRelayResult(staged, False, False, "personal_context_relay_pending")
                    claimed_row = replace(row, relay_owner_token=lease.owner_token)
                    if not self.publications.row_is_current(claimed_row, lease):
                        return PersonalContextRelayResult(staged, False, False, "personal_context_relay_pending")
                    try:
                        cursor = self.stage_authority(claimed_row, dataset_id, user_id)
                    except Exception:  # noqa: BLE001 - poison blocks later authority batches.
                        self.publications.mark_attention(batch)
                        return PersonalContextRelayResult(staged, False, False, "relay_poisoned")
                    if isinstance(cursor, bool) or not isinstance(cursor, int) or cursor < 1:
                        raise RuntimeError("authority relay receipt is invalid")
                    if not self.publications.renew_lease(lease) or not self.publications.row_is_current(claimed_row, lease):
                        return PersonalContextRelayResult(staged, False, False, "personal_context_relay_pending")
                    self.publications.acknowledge_row(claimed_row, server_cursor=cursor, lease=lease)
                    acknowledged_ordinals.add(row.batch_ordinal)
                    staged += 1
                if not self.publications.complete_if_acknowledged(batch):
                    return PersonalContextRelayResult(staged, False, False, "personal_context_relay_pending")


__all__ = ["PersonalContextRelay", "PersonalContextRelayResult"]
