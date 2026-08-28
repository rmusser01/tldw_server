"""Review-safe decisions and reconciliation for Notes graph suggestions."""

from __future__ import annotations

import hashlib
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from tldw_Server_API.app.core.DB_Management.chacha.note_graph_suggestion_models import (
    NoteGraphSuggestion,
    NoteGraphSuggestionKind,
    NoteGraphSuggestionState,
)
from tldw_Server_API.app.core.DB_Management.chacha.note_graph_suggestion_store import (
    MutationResult,
)
from tldw_Server_API.app.core.DB_Management.chacha.organization_sync_store import (
    NotesOrganizationSyncStore,
)
from tldw_Server_API.app.core.Sync.v2.materializers.guarded_product_mutation import (
    GuardedProductMutation,
)
from tldw_Server_API.app.core.Sync.v2.notes_organization import organization_link_id
from tldw_Server_API.app.core.Sync.v2.server_origin_batch import (
    SyncServerOriginBatchMaterializationError,
)

from .suggestion_observability import (
    DecisionOutcome,
    ReconciliationOutcome,
    SuggestionEventName,
    record_acceptance_reconciliation,
    record_decision_outcome,
    record_event,
)

_LINK_SOURCE = "notes.graph.link.create"
_ORGANIZATION_SOURCE = "notes_graph_suggestion"


class SuggestionDecisionService:
    """Coordinate durable decisions with guarded canonical Notes mutations."""

    def __init__(
        self,
        *,
        store: Any,
        link_coordinator: Any,
        organization_coordinator: Any,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.store = store
        self.links = link_coordinator
        self.organization = organization_coordinator
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    @staticmethod
    def _step_key(suggestion_id: str, step: str) -> str:
        return f"notes-graph:{suggestion_id}:{step}"

    @staticmethod
    def _link_identity(dataset_id: str, key: str) -> str:
        digest = hashlib.sha256(f"{dataset_id}:{_LINK_SOURCE}:{key}".encode()).digest()[:16]
        return str(uuid.UUID(bytes=digest, version=4))

    def _before(self, fence: NoteGraphSuggestion) -> Callable[[Any], None]:
        return lambda conn: self.store.guard_acceptance_in_transaction(
            conn=conn,
            fence=fence,
            now=self._clock(),
        )

    def _after(
        self,
        fence: NoteGraphSuggestion,
        finalized: list[MutationResult],
    ) -> Callable[[Any, str], None]:
        def finalize(conn: Any, resource_identity: str) -> None:
            finalized.append(
                self.store.finalize_acceptance_in_transaction(
                    conn=conn,
                    fence=fence,
                    accepted_resource_identity=resource_identity,
                    now=self._clock(),
                )
            )

        return finalize

    def _record_decision(self, result: MutationResult) -> None:
        state = result.envelope.get("state")
        suggestion = result.suggestion
        if state not in {"accepted", "rejected", "stale"} or suggestion is None:
            return
        outcome = DecisionOutcome(state)
        record_decision_outcome(outcome)
        record_event(
            SuggestionEventName(state),
            run_id=suggestion.run_id,
            suggestion_id=suggestion.id,
        )

    def reject(
        self,
        *,
        dataset_id: str,
        suggestion_id: str,
        expected_revision: int,
        expected_source_fingerprint: str,
        expected_target_fingerprint: str | None,
        idempotency_key: str,
    ) -> MutationResult:
        """Reject one exact pending suggestion and record its durable outcome."""

        result = self.store.reject_suggestion(
            dataset_id=dataset_id,
            suggestion_id=suggestion_id,
            expected_revision=expected_revision,
            expected_source_fingerprint=expected_source_fingerprint,
            expected_target_fingerprint=expected_target_fingerprint,
            idempotency_key=idempotency_key,
            now=self._clock(),
        )
        self._record_decision(result)
        return result

    def reset_rejections(
        self,
        *,
        dataset_id: str,
        source_note_id: str,
        source_fingerprint: str,
        expected_revision: int,
        idempotency_key: str,
    ) -> MutationResult:
        """Reset only the exact source-fingerprint rejection-set revision."""

        return self.store.reset_rejections(
            dataset_id=dataset_id,
            source_note_id=source_note_id,
            source_fingerprint=source_fingerprint,
            expected_revision=expected_revision,
            idempotency_key=idempotency_key,
            now=self._clock(),
        )

    def accept(
        self,
        *,
        dataset_id: str,
        suggestion_id: str,
        expected_revision: int,
        expected_source_fingerprint: str,
        expected_target_fingerprint: str | None,
        idempotency_key: str,
    ) -> MutationResult:
        """Accept through the canonical coordinator and its product-transaction guard."""

        claimed = self.store.claim_acceptance(
            dataset_id=dataset_id,
            suggestion_id=suggestion_id,
            expected_revision=expected_revision,
            expected_source_fingerprint=expected_source_fingerprint,
            expected_target_fingerprint=expected_target_fingerprint,
            idempotency_key=idempotency_key,
            now=self._clock(),
        )
        fence = claimed.suggestion
        if fence is None or fence.state != NoteGraphSuggestionState.ACCEPTING:
            self._record_decision(claimed)
            return claimed
        if self._lease_expired(fence):
            fence = self.store.reclaim_expired_acceptance(
                dataset_id=fence.dataset_id,
                suggestion_id=fence.id,
                decision_receipt_id=str(fence.decision_receipt_id),
                expected_state="accepting",
                expected_revision=fence.revision,
                expected_lease_token=str(fence.acceptance_lease_token),
                now=self._clock(),
            )
            result = self._resolve(fence)
            self._record_decision(result)
            return result
        result = (
            self._accept_related(fence)
            if fence.kind == NoteGraphSuggestionKind.RELATED_NOTE
            else self._accept_tag(fence)
        )
        self._record_decision(result)
        return result

    def _lease_expired(self, fence: NoteGraphSuggestion) -> bool:
        value = fence.acceptance_lease_expires_at
        if value is None:
            return True
        expires_at = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return expires_at <= self._clock()

    def _accept_related(self, fence: NoteGraphSuggestion) -> MutationResult:
        existing = self.store.finalize_existing_acceptance(
            fence=fence,
            accepted_resource_identity=None,
            resolved_keyword_sync_id=None,
            now=self._clock(),
        )
        if existing.disposition != "in_progress":
            return existing
        fence = self.store.renew_acceptance(fence=fence, now=self._clock())
        key = self._step_key(fence.id, "link")
        expected_edge_id = self._link_identity(fence.dataset_id, key)
        finalized: list[MutationResult] = []
        guard = GuardedProductMutation(
            expected_domain="notes.link",
            expected_object_id=expected_edge_id,
            before=self._before(fence),
            after=self._after(fence, finalized),
        )
        try:
            self.links.create(
                source_note_id=fence.source_note_id,
                target_note_id=str(fence.target_note_id),
                directed=False,
                weight=1.0,
                label=None,
                properties={},
                idempotency_key=key,
                guarded_mutation=guard,
            )
        except SyncServerOriginBatchMaterializationError as exc:
            try:
                existing = self.store.finalize_existing_acceptance(
                    fence=fence,
                    accepted_resource_identity=None,
                    resolved_keyword_sync_id=None,
                    now=self._clock(),
                )
            except RuntimeError as fence_error:
                if str(fence_error) != "notes_graph_suggestion_conflict":
                    raise
                raise exc from fence_error
            if existing.disposition != "in_progress":
                return existing
            if not exc.retryable:
                return self._release(fence)
            raise
        if len(finalized) != 1:
            raise RuntimeError("notes_graph_acceptance_finalization_missing")
        return finalized[0]

    def _resolve_keyword(
        self,
        fence: NoteGraphSuggestion,
        *,
        conn: Any | None = None,
        for_update: bool = False,
    ) -> tuple[str | None, bool]:
        if fence.keyword_sync_id:
            dataset = self.organization.active_dataset()
            resources = NotesOrganizationSyncStore(self.organization.note_db)
            sync_id = fence.keyword_sync_id
            seen: set[str] = set()
            while sync_id not in seen and len(seen) < 100:
                seen.add(sync_id)
                resource = resources.get_resource(
                    "notes.keyword",
                    sync_id,
                    conn=conn,
                    for_update=for_update,
                )
                if resource is not None and not resource.deleted:
                    return sync_id, False
                head = self.organization.service.store.get_current_head(
                    dataset.dataset_id,
                    "notes.keyword",
                    sync_id,
                )
                merge = (
                    head.routing_metadata.get("notes_keyword_merge_response")
                    if head is not None
                    else None
                )
                target_id = merge.get("target_keyword_id") if isinstance(merge, dict) else None
                if not isinstance(target_id, int) or isinstance(target_id, bool):
                    break
                target = resources.get_resource_row_by_local_id(
                    "notes.keyword",
                    target_id,
                    include_deleted=True,
                    conn=conn,
                    for_update=for_update,
                )
                if target is None:
                    break
                sync_id = str(target["sync_id"])
            return None, True
        display = fence.normalized_tag or fence.display_tag or ""
        resource = NotesOrganizationSyncStore(
            self.organization.note_db
        ).find_keyword_by_normalized_identity(
            display,
            conn=conn,
            for_update=for_update,
        )
        return (resource.sync_id, False) if resource is not None else (None, False)

    def _mark_keyword_acceptance_stale(
        self,
        fence: NoteGraphSuggestion,
        *,
        now: datetime,
    ) -> MutationResult:
        return self.store.mark_acceptance_stale(
            fence=fence,
            reason="canonical_resource_missing",
            verifier=lambda conn: self._resolve_keyword(
                fence,
                conn=conn,
                for_update=True,
            )[1],
            now=now,
        )

    @staticmethod
    def _keyword_link_identity(fence: NoteGraphSuggestion, keyword_sync_id: str) -> str:
        return organization_link_id(
            "notes.keyword_link",
            ["note", fence.source_note_id, keyword_sync_id],
        )

    def _accept_tag(self, fence: NoteGraphSuggestion) -> MutationResult:
        keyword_sync_id, stale = self._resolve_keyword(fence)
        if stale:
            stale_result = self._mark_keyword_acceptance_stale(
                fence,
                now=self._clock(),
            )
            if stale_result.disposition != "in_progress":
                return stale_result
            keyword_sync_id, stale = self._resolve_keyword(fence)
            if stale:
                return self._release(fence)
        relationship_identity = (
            self._keyword_link_identity(fence, keyword_sync_id)
            if keyword_sync_id is not None
            else None
        )
        existing = self.store.finalize_existing_acceptance(
            fence=fence,
            accepted_resource_identity=relationship_identity,
            resolved_keyword_sync_id=keyword_sync_id,
            now=self._clock(),
        )
        if existing.disposition != "in_progress":
            return existing

        if keyword_sync_id is None:
            keyword_key = self._step_key(fence.id, "keyword")
            keyword_plan = self.organization.plan_keyword_create(
                str(fence.display_tag),
                idempotency_key=keyword_key,
            )
            keyword_step = keyword_plan.steps[-1]
            keyword_sync_id = keyword_step.object_id
            fence = self.store.renew_acceptance(fence=fence, now=self._clock())
            keyword_guard = GuardedProductMutation(
                expected_domain="notes.keyword",
                expected_object_id=keyword_sync_id,
                before=self._before(fence),
                after=None,
            )
            try:
                self.organization.capture(
                    steps=keyword_plan.steps,
                    source=_ORGANIZATION_SOURCE,
                    idempotency_key=keyword_key,
                    guarded_mutations=(keyword_guard,),
                )
            except SyncServerOriginBatchMaterializationError:
                collision = NotesOrganizationSyncStore(
                    self.organization.note_db
                ).find_keyword_by_normalized_identity(
                    fence.normalized_tag or str(fence.display_tag)
                )
                if collision is None:
                    return self._release(fence)
                keyword_sync_id = collision.sync_id
            else:
                canonical = NotesOrganizationSyncStore(
                    self.organization.note_db
                ).find_keyword_by_normalized_identity(
                    fence.normalized_tag or str(fence.display_tag)
                )
                if canonical is None:
                    return self._release(fence)
                keyword_sync_id = canonical.sync_id

        relationship_key = self._step_key(fence.id, "tag-membership")
        relationship_plan = self.organization.plan_relationship(
            "notes.keyword_link",
            {
                "subject_type": "note",
                "subject_id": fence.source_note_id,
                "keyword_sync_id": keyword_sync_id,
            },
            True,
            source=_ORGANIZATION_SOURCE,
            idempotency_key=relationship_key,
        )
        relationship_step = relationship_plan.steps[-1]
        fence = self.store.renew_acceptance(fence=fence, now=self._clock())
        finalized: list[MutationResult] = []
        relationship_guard = GuardedProductMutation(
            expected_domain="notes.keyword_link",
            expected_object_id=relationship_step.object_id,
            before=self._before(fence),
            after=self._after(fence, finalized),
        )
        try:
            self.organization.capture(
                steps=relationship_plan.steps,
                source=_ORGANIZATION_SOURCE,
                idempotency_key=relationship_key,
                guarded_mutations=(relationship_guard,),
            )
        except SyncServerOriginBatchMaterializationError:
            existing = self.store.finalize_existing_acceptance(
                fence=fence,
                accepted_resource_identity=relationship_step.object_id,
                resolved_keyword_sync_id=keyword_sync_id,
                now=self._clock(),
            )
            return existing if existing.disposition != "in_progress" else self._release(fence)
        if len(finalized) != 1:
            raise RuntimeError("notes_graph_acceptance_finalization_missing")
        return finalized[0]

    def _release(self, fence: NoteGraphSuggestion) -> MutationResult:
        return self.store.release_acceptance(
            dataset_id=fence.dataset_id,
            suggestion_id=fence.id,
            decision_receipt_id=str(fence.decision_receipt_id),
            expected_state="accepting",
            expected_revision=fence.revision,
            expected_lease_token=str(fence.acceptance_lease_token),
            now=self._clock(),
        )

    def _resolve(
        self,
        fence: NoteGraphSuggestion,
        *,
        now: datetime | None = None,
    ) -> MutationResult:
        decision_time = now or self._clock()
        keyword_sync_id = None
        relationship_identity = None
        if fence.kind == NoteGraphSuggestionKind.TAG:
            keyword_sync_id, stale = self._resolve_keyword(fence)
            if stale:
                stale_result = self._mark_keyword_acceptance_stale(
                    fence,
                    now=decision_time,
                )
                if stale_result.disposition != "in_progress":
                    return stale_result
                keyword_sync_id, stale = self._resolve_keyword(fence)
                if stale:
                    keyword_sync_id = None
            if keyword_sync_id is not None:
                relationship_identity = self._keyword_link_identity(fence, keyword_sync_id)
        return self.store.resolve_expired_acceptance(
            fence=fence,
            accepted_resource_identity=relationship_identity,
            resolved_keyword_sync_id=keyword_sync_id,
            now=decision_time,
        )

    def reconcile_expired(
        self,
        *,
        dataset_id: str,
        limit: int = 100,
        now: datetime | None = None,
        on_claimed: Callable[[int], None] | None = None,
    ) -> tuple[MutationResult, ...]:
        """Resolve a bounded claimed set without creating canonical product state."""

        reconciliation_time = now or self._clock()
        claims = self.store.claim_expired_acceptances(
            dataset_id=dataset_id,
            limit=limit,
            now=reconciliation_time,
        )
        if claims and on_claimed is not None:
            on_claimed(len(claims))
        results: list[MutationResult] = []
        for fence in claims:
            result = self._resolve(fence, now=reconciliation_time)
            results.append(result)
            self._record_decision(result)
            outcome = (
                ReconciliationOutcome.RELEASED
                if result.envelope.get("state") == "pending"
                else ReconciliationOutcome.COMPLETED
            )
            record_acceptance_reconciliation(outcome)
            suggestion = result.suggestion or fence
            record_event(
                SuggestionEventName.RECONCILED,
                run_id=suggestion.run_id,
                suggestion_id=suggestion.id,
            )
        return tuple(results)


__all__ = ["SuggestionDecisionService"]
