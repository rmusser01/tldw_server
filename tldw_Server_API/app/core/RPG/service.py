from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from tldw_Server_API.app.core.DB_Management.RPG_DB import RPGRepository
from tldw_Server_API.app.core.RPG.authority import decide_authority
from tldw_Server_API.app.core.RPG.constants import MAX_RPG_CONTEXT_CHARS
from tldw_Server_API.app.core.RPG.context import SessionContext, SessionContextBuilder
from tldw_Server_API.app.core.RPG.errors import RPGConflictError
from tldw_Server_API.app.core.RPG.events import canonical_request_hash, validate_event_envelope
from tldw_Server_API.app.core.RPG.models import (
    RPGCampaign,
    RPGSession,
    RPGSessionEvent,
    RPGSnapshotState,
)
from tldw_Server_API.app.core.RPG.reducer import reduce_events
from tldw_Server_API.app.core.RPG.rules.adapters import RuleAdapterRegistry, build_default_adapter_registry
from tldw_Server_API.app.core.RPG.rules.content_packs import RuleLookupResult
from tldw_Server_API.app.core.RPG.rules.lookup import RulesLookupService


@dataclass(frozen=True, slots=True)
class RPGServiceProposal:
    id: int
    session_id: int
    status: str
    proposed_events: list[dict[str, Any]]


@dataclass(frozen=True, slots=True)
class RecordEventsResult:
    committed_events: list[RPGSessionEvent]
    proposal: RPGServiceProposal | None


@dataclass(frozen=True, slots=True)
class SnapshotResult:
    snapshot_version: int
    last_event_sequence: int
    snapshot: RPGSnapshotState
    diagnostics: dict[str, Any]


class RPGService:
    def __init__(
        self,
        repo: RPGRepository,
        owner_user_id: int,
        adapter_registry: RuleAdapterRegistry | None = None,
    ) -> None:
        self.repo = repo
        self.owner_user_id = owner_user_id
        self.adapter_registry = adapter_registry or build_default_adapter_registry()

    def create_campaign(
        self,
        title: str,
        description: str | None,
        default_adapter_key: str,
        idempotency_key: str,
    ) -> RPGCampaign:
        self._require_idempotency_key(idempotency_key)
        adapter = self.adapter_registry.get(default_adapter_key)
        request_hash = canonical_request_hash(
            {
                "title": title,
                "description": description,
                "default_adapter_key": adapter.adapter_key,
                "default_adapter_version": adapter.adapter_version,
            }
        )
        return self.repo.create_campaign(
            owner_user_id=self.owner_user_id,
            title=title,
            description=description,
            default_adapter_key=adapter.adapter_key,
            default_adapter_version=adapter.adapter_version,
            settings={},
            linked_rules_pack_refs=[],
            idempotency_key=idempotency_key,
            request_payload_hash=request_hash,
            source_type="user",
        )

    def create_session(
        self,
        campaign_id: int,
        title: str,
        adapter_key: str,
        idempotency_key: str,
    ) -> RPGSession:
        self._require_idempotency_key(idempotency_key)
        adapter = self.adapter_registry.get(adapter_key)
        authority_settings = {"model_auto_commit": False, "mcp_auto_commit": False}
        request_hash = canonical_request_hash(
            {
                "campaign_id": campaign_id,
                "title": title,
                "adapter_key": adapter.adapter_key,
                "adapter_version": adapter.adapter_version,
                "authority_settings": authority_settings,
            }
        )
        return self.repo.create_session(
            owner_user_id=self.owner_user_id,
            campaign_id=campaign_id,
            title=title,
            adapter_key=adapter.adapter_key,
            adapter_version=adapter.adapter_version,
            authority_settings=authority_settings,
            linked_chat_id=None,
            active_rules_pack_refs=[],
            idempotency_key=idempotency_key,
            request_payload_hash=request_hash,
            source_type="user",
        )

    def record_events(
        self,
        session_id: int,
        events: list[dict[str, Any]],
        source_type: str,
        expected_last_event_sequence: int,
        idempotency_key: str,
    ) -> RecordEventsResult:
        self._require_idempotency_key(idempotency_key)
        if not events:
            raise RPGConflictError("events_required")

        session = self.repo.get_session(owner_user_id=self.owner_user_id, session_id=session_id)
        source_actor_id = self._source_actor_id(source_type)
        normalized = [
            validate_event_envelope(
                {
                    **event,
                    "source_type": source_type,
                    "source_actor_id": source_actor_id,
                }
            )
            for event in events
        ]
        decisions = [
            decide_authority(source_type, event["event_type"], session.authority_settings) for event in normalized
        ]
        request_hash = canonical_request_hash(
            {
                "events": normalized,
                "expected_last_event_sequence": expected_last_event_sequence,
                "source_type": source_type,
            }
        )
        if any(decision.action == "proposal" for decision in decisions):
            current = self.repo.get_latest_snapshot(
                owner_user_id=self.owner_user_id,
                session_id=session_id,
            )
            proposal = self.repo.create_proposal(
                owner_user_id=self.owner_user_id,
                session_id=session_id,
                base_event_sequence=expected_last_event_sequence,
                base_snapshot_version=current.snapshot_version,
                proposed_events=normalized,
                source_type=source_type,
                source_actor_id=source_actor_id,
                model_metadata={},
                idempotency_key=idempotency_key,
                request_payload_hash=request_hash,
            )
            return RecordEventsResult(
                committed_events=[],
                proposal=RPGServiceProposal(
                    id=proposal.id,
                    session_id=session_id,
                    status=proposal.status,
                    proposed_events=proposal.proposed_events,
                ),
            )

        return self._commit_validated_events(
            session=session,
            normalized=normalized,
            expected_last_event_sequence=expected_last_event_sequence,
            idempotency_key=idempotency_key,
            request_hash=request_hash,
            proposal_id=None,
        )

    def apply_proposal(
        self,
        session_id: int,
        proposal_id: int,
        expected_last_event_sequence: int,
        idempotency_key: str,
        review_notes: str | None = None,
    ) -> RecordEventsResult:
        self._require_idempotency_key(idempotency_key)
        session = self.repo.get_session(owner_user_id=self.owner_user_id, session_id=session_id)
        proposal = self.repo.get_proposal(owner_user_id=self.owner_user_id, proposal_id=proposal_id)
        if proposal.session_id != session_id:
            raise RPGConflictError("proposal_session_mismatch")
        if expected_last_event_sequence != proposal.base_event_sequence:
            raise RPGConflictError("stale_event_sequence")

        normalized = [validate_event_envelope(event) for event in proposal.proposed_events]
        request_hash = canonical_request_hash(
            {
                "proposal_id": proposal_id,
                "expected_last_event_sequence": expected_last_event_sequence,
                "events": normalized,
                "review_notes": review_notes,
            }
        )
        if proposal.status == "applied":
            return self._commit_validated_events(
                session=session,
                normalized=normalized,
                expected_last_event_sequence=proposal.base_event_sequence,
                idempotency_key=idempotency_key,
                request_hash=request_hash,
                proposal_id=proposal_id,
                proposal_review_notes=review_notes,
            )
        if proposal.status != "pending":
            raise RPGConflictError("proposal_not_pending")
        if proposal.base_event_sequence != session.last_event_sequence:
            self.repo.mark_proposal_conflicted(self.owner_user_id, proposal_id)
            raise RPGConflictError("proposal_base_sequence_conflict")

        return self._commit_validated_events(
            session=session,
            normalized=normalized,
            expected_last_event_sequence=proposal.base_event_sequence,
            idempotency_key=idempotency_key,
            request_hash=request_hash,
            proposal_id=proposal_id,
            proposal_review_notes=review_notes,
        )

    def reject_proposal(
        self,
        session_id: int,
        proposal_id: int,
        idempotency_key: str,
        review_notes: str | None = None,
    ) -> RPGServiceProposal:
        self._require_idempotency_key(idempotency_key)
        proposal = self.repo.get_proposal(owner_user_id=self.owner_user_id, proposal_id=proposal_id)
        if proposal.session_id != session_id:
            raise RPGConflictError("proposal_session_mismatch")
        rejected = self.repo.mark_proposal_rejected(
            owner_user_id=self.owner_user_id,
            proposal_id=proposal_id,
            idempotency_key=idempotency_key,
            review_notes=review_notes,
        )
        return RPGServiceProposal(
            id=rejected.id,
            session_id=session_id,
            status=rejected.status,
            proposed_events=rejected.proposed_events,
        )

    def get_snapshot(self, session_id: int) -> SnapshotResult:
        record = self.repo.get_latest_snapshot(
            owner_user_id=self.owner_user_id,
            session_id=session_id,
        )
        return SnapshotResult(
            snapshot_version=record.snapshot_version,
            last_event_sequence=record.last_event_sequence,
            snapshot=RPGSnapshotState(**record.snapshot_json),
            diagnostics=record.diagnostics_json,
        )

    def lookup_rules(self, session_id: int, query: str) -> RuleLookupResult:
        session = self.repo.get_session(owner_user_id=self.owner_user_id, session_id=session_id)
        return RulesLookupService().lookup(
            adapter_key=session.adapter_key,
            query=query,
            linked_rules_pack_refs=session.active_rules_pack_refs,
        )

    def build_context(
        self,
        session_id: int,
        query: str | None = None,
        max_chars: int = MAX_RPG_CONTEXT_CHARS,
    ) -> SessionContext:
        bounded_max_chars = min(max(int(max_chars), 1000), MAX_RPG_CONTEXT_CHARS)
        session = self.repo.get_session(owner_user_id=self.owner_user_id, session_id=session_id)
        snapshot = self.get_snapshot(session_id).snapshot
        rules_results = self.lookup_rules(session_id, query).results if query and query.strip() else []
        return SessionContextBuilder(max_chars=bounded_max_chars).build(
            adapter_key=session.adapter_key,
            session_title=session.title,
            snapshot=snapshot,
            rules_results=rules_results,
        )

    def _commit_validated_events(
        self,
        session: RPGSession,
        normalized: list[dict[str, Any]],
        expected_last_event_sequence: int,
        idempotency_key: str,
        request_hash: str,
        proposal_id: int | None,
        proposal_review_notes: str | None = None,
    ) -> RecordEventsResult:
        current = self.repo.get_latest_snapshot(
            owner_user_id=self.owner_user_id,
            session_id=session.id,
        )
        next_snapshot = reduce_events(RPGSnapshotState(**current.snapshot_json), normalized)
        committed = self.repo.commit_events_and_snapshot(
            owner_user_id=self.owner_user_id,
            session_id=session.id,
            expected_last_event_sequence=expected_last_event_sequence,
            base_snapshot_version=current.snapshot_version,
            events=normalized,
            snapshot=asdict(next_snapshot),
            diagnostics={"applied_event_count": len(normalized)},
            idempotency_key=idempotency_key,
            request_payload_hash=request_hash,
            adapter_key=session.adapter_key,
            adapter_version=session.adapter_version,
            proposal_id=proposal_id,
            proposal_review_notes=proposal_review_notes,
        )
        return RecordEventsResult(committed_events=committed.events, proposal=None)

    @staticmethod
    def _require_idempotency_key(idempotency_key: str) -> None:
        if not idempotency_key:
            raise RPGConflictError("idempotency_key_required")

    def _source_actor_id(self, source_type: str) -> str | None:
        if source_type in {"user", "mcp"}:
            return f"{source_type}:{self.owner_user_id}"
        return None
