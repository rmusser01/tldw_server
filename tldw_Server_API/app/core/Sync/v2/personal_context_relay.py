"""Ordered relay for encrypted Personal Context authority publications."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from time import monotonic_ns
from typing import Any, Literal

from tldw_Server_API.app.core.Personalization.personal_context_publication import (
    AuthorityStageReceipt,
    PublicationRelayPoisoned,
    PublicationSourceRow,
    PublicationStageIdentity,
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


@dataclass(slots=True)
class PersonalContextRecoveryBudget:
    """One consumable row allowance and absolute deadline for a pull."""

    deadline_ns: int
    remaining_rows: int
    clock_ns: Callable[[], int] = monotonic_ns
    relay: PersonalContextRelayResult | None = None

    def deadline_open(self) -> bool:
        """Return whether recovery may still do deadline-bound work."""

        return self.clock_ns() < self.deadline_ns

    def can_inspect(self) -> bool:
        """Return whether one more row may be queried and inspected."""

        return self.remaining_rows > 0 and self.deadline_open()

    def consume(self) -> bool:
        """Spend one row immediately before its decrypt or classification."""

        if not self.can_inspect():
            return False
        self.remaining_rows -= 1
        return True


AuthorityStager = Callable[[PublicationSourceRow, str, str], AuthorityStageReceipt]
AuthorityFinalizer = Callable[
    [PublicationSourceRow, AuthorityStageReceipt, str, str], None
]
AuthorityCanceller = Callable[
    [PublicationSourceRow | PublicationStageIdentity, AuthorityStageReceipt | None, str, str],
    Literal["removed", "absent", "applied"],
]


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
        budget: PersonalContextRecoveryBudget | None = None,
    ) -> PersonalContextRelayResult:
        """Stage the first incomplete batch only, bounded by rows and wall time."""

        del after_server_cursor
        if row_budget < 1 or wall_time_ms < 1:
            raise ValueError("relay limits must be positive")
        if budget is None:
            budget = PersonalContextRecoveryBudget(
                deadline_ns=deadline_ns
                or self.clock_ns() + wall_time_ms * 1_000_000,
                remaining_rows=row_budget,
                clock_ns=self.clock_ns,
            )
        starting_rows = budget.remaining_rows
        try:
            with self.publications.profile_lease(profile_id) as lease:
                if lease is None:
                    result = self._pending()
                else:
                    result = self._relay_owned(
                        lease=lease,
                        user_id=user_id,
                        profile_id=profile_id,
                        dataset_id=dataset_id,
                        budget=budget,
                    )
        except PublicationRelayPoisoned:
            result = PersonalContextRelayResult(0, False, False, "relay_poisoned")
        except Exception:  # noqa: BLE001 - DB/lease/adapter races remain retryable.
            result = self._pending()
        return replace(
            result,
            inspected_rows=starting_rows - budget.remaining_rows,
        )

    def _relay_owned(
        self,
        *,
        lease: Any,
        user_id: str,
        profile_id: str,
        dataset_id: str,
        budget: PersonalContextRecoveryBudget,
    ) -> PersonalContextRelayResult:
        staged = 0
        unfinished_lookup = getattr(self.publications, "unfinished_stage_identities", None)
        if unfinished_lookup is not None and budget.can_inspect():
            for identity in unfinished_lookup(
                profile_id,
                row_limit=budget.remaining_rows,
                budget=budget,
            ):
                if not budget.deadline_open():
                    return self._pending(staged)
                claimed_identity = replace(
                    identity, relay_owner_token=lease.owner_token
                )
                cancellation = self._cancel(
                    claimed_identity, None, dataset_id, user_id
                )
                if cancellation == "failed":
                    return self._pending(staged)
                try:
                    self.publications.retire_terminal_stage_identity(
                        claimed_identity, lease=lease
                    )
                except Exception:  # noqa: BLE001 - exact cleanup retries after races.
                    return self._pending(staged)
            if not budget.can_inspect():
                return self._pending(staged)

        while True:
            if not budget.can_inspect():
                return self._pending(staged)
            batch = self.publications.earliest_nonterminal_batch(
                profile_id,
                row_limit=budget.remaining_rows,
                lease=lease,
                budget=budget,
            )
            if batch is None:
                return PersonalContextRelayResult(
                    staged, True, False, "complete"
                )
            acknowledged_ordinals = {
                row.batch_ordinal
                for row in batch.rows
                if row.row_state == "acknowledged"
            }
            for row in batch.rows:
                if not budget.deadline_open():
                    return self._pending(staged)
                claimed_row = replace(row, relay_owner_token=lease.owner_token)

                if row.row_state == "acknowledged":
                    receipt = self._receipt_for_row(row)
                    if receipt is None or not self._renewed_current(claimed_row, lease):
                        return self._pending(staged)
                    if self.finalize_authority is not None:
                        try:
                            self.finalize_authority(
                                claimed_row, receipt, dataset_id, user_id
                            )
                        except Exception:  # noqa: BLE001 - exact applied replay retries.
                            return self._pending(staged)
                    continue

                if row.role == "manifest" and any(
                    item.role == "semantic"
                    and item.batch_ordinal not in acknowledged_ordinals
                    for item in batch.rows
                ):
                    return self._pending(staged)

                receipt = self._receipt_for_row(row)
                if row.row_state == "pending":
                    if not self._renewed_current(claimed_row, lease):
                        return self._pending(staged)
                    try:
                        receipt = self.stage_authority(
                            claimed_row, dataset_id, user_id
                        )
                    except PersonalContextAuthoritySourceError:
                        self.publications.mark_attention(batch, lease=lease)
                        return PersonalContextRelayResult(
                            staged, False, False, "relay_poisoned"
                        )
                    except Exception:  # noqa: BLE001 - storage/head/adapter failures retry.
                        return self._pending(staged)
                    if not self._receipt_matches(claimed_row, receipt):
                        return self._pending(staged)
                    if not self._renewed_current(claimed_row, lease):
                        return self._pending(staged)
                    try:
                        self.publications.record_staged_row(
                            claimed_row,
                            server_cursor=receipt.server_cursor,
                            lease=lease,
                        )
                    except Exception:  # noqa: BLE001 - classify uncertain commit state.
                        state = self.publications.stage_receipt_state(
                            claimed_row, receipt, lease=lease
                        )
                        if state == "claimable":
                            self._cancel(
                                claimed_row, receipt, dataset_id, user_id
                            )
                        return self._pending(staged)

                if receipt is None or not self._receipt_matches(claimed_row, receipt):
                    return self._pending(staged)
                if not self._renewed_current(claimed_row, lease):
                    return self._pending(staged)
                try:
                    self.publications.acknowledge_row(
                        claimed_row,
                        server_cursor=receipt.server_cursor,
                        lease=lease,
                    )
                except Exception:  # noqa: BLE001 - durable staged source retries.
                    return self._pending(staged)
                acknowledged_ordinals.add(row.batch_ordinal)

                if not self._renewed_current(claimed_row, lease):
                    return self._pending(staged)
                if self.finalize_authority is not None:
                    try:
                        self.finalize_authority(
                            claimed_row, receipt, dataset_id, user_id
                        )
                    except Exception:  # noqa: BLE001 - acknowledged source repairs on retry.
                        return self._pending(staged)
                staged += 1

            if not self.publications.complete_if_acknowledged(batch, lease=lease):
                return self._pending(staged)

    def _renewed_current(self, row: PublicationSourceRow, lease: Any) -> bool:
        return bool(
            self.publications.renew_lease(lease)
            and self.publications.row_is_current(row, lease)
        )

    @staticmethod
    def _receipt_for_row(row: PublicationSourceRow) -> AuthorityStageReceipt | None:
        if row.sync_server_cursor is None:
            return None
        return AuthorityStageReceipt(
            server_cursor=row.sync_server_cursor,
            deterministic_envelope_id=row.deterministic_envelope_id,
            publication_batch_id=row.publication_batch_id,
            profile_publication_sequence=row.profile_publication_sequence,
            batch_ordinal=row.batch_ordinal,
            batch_size=row.batch_size,
            purge_generation=row.purge_generation,
        )

    @staticmethod
    def _receipt_matches(row: PublicationSourceRow, receipt: object) -> bool:
        return bool(
            isinstance(receipt, AuthorityStageReceipt)
            and receipt.server_cursor > 0
            and receipt.deterministic_envelope_id == row.deterministic_envelope_id
            and receipt.publication_batch_id == row.publication_batch_id
            and receipt.profile_publication_sequence == row.profile_publication_sequence
            and receipt.batch_ordinal == row.batch_ordinal
            and receipt.batch_size == row.batch_size
            and receipt.purge_generation == row.purge_generation
        )

    @staticmethod
    def _pending(staged: int = 0, inspected: int = 0) -> PersonalContextRelayResult:
        return PersonalContextRelayResult(
            staged, False, False, "personal_context_relay_pending", inspected
        )

    def _cancel(
        self,
        row: PublicationSourceRow | PublicationStageIdentity,
        receipt: AuthorityStageReceipt | None,
        dataset_id: str,
        user_id: str,
    ) -> Literal["removed", "absent", "applied", "failed"]:
        """Best-effort compensation; only the exact pending row can be removed."""

        if self.cancel_authority is None:
            return "failed"
        try:
            outcome = self.cancel_authority(row, receipt, dataset_id, user_id)
        except Exception:  # noqa: BLE001 - cancellation is best-effort retry cleanup.
            return "failed"
        return outcome if outcome in {"removed", "absent", "applied"} else "failed"


__all__ = [
    "PersonalContextAuthoritySourceError",
    "PersonalContextRecoveryBudget",
    "PersonalContextRelay",
    "PersonalContextRelayResult",
]
