from __future__ import annotations

import base64
from datetime import UTC, datetime, timedelta

from tldw_profile_core import (
    AgentVisibility,
    PreferencePayload,
    ProfileControls,
    ProfileManifest,
    ProfileProposal,
    ProfileProvenance,
    ProfileRecord,
    ProfileScope,
    ProposalOperation,
    ProvenanceSource,
    RecordKind,
    RecordState,
    ScopeKind,
    SemanticKey,
    SyncMode,
)
from tldw_profile_core.models import ActorType

NOW = datetime(2026, 8, 30, 12, 0, tzinfo=UTC)


def encoded_master_key(byte: bytes = b"m") -> str:
    return base64.b64encode(byte * 32).decode("ascii")


def manifest(profile_id: str = "profile-a") -> ProfileManifest:
    return ProfileManifest(
        profile_id=profile_id,
        revision=0,
        purge_generation=0,
        created_at=NOW,
        updated_at=NOW,
        current_version_id="manifest-v1",
    )


def global_scope(profile_id: str = "profile-a") -> ProfileScope:
    return ProfileScope(
        scope_id=f"{profile_id}-global",
        profile_id=profile_id,
        kind=ScopeKind.GLOBAL,
        version_id="scope-v1",
        created_at=NOW,
        updated_at=NOW,
    )


def preference_record(
    profile_id: str = "profile-a",
    *,
    record_id: str = "record-a",
    version_id: str = "record-v1",
    parent_version_id: str | None = None,
    value: str = "concise",
    state: RecordState = RecordState.ACTIVE,
) -> ProfileRecord:
    deleted = state is RecordState.DELETED
    return ProfileRecord(
        profile_id=profile_id,
        record_id=record_id,
        scope_id=f"{profile_id}-global",
        kind=RecordKind.PREFERENCE,
        payload=(
            None
            if deleted
            else PreferencePayload(
                subject="response.detail",
                polarity="like",
                value=value,
            )
        ),
        semantic_key=(
            None
            if deleted
            else SemanticKey(
                namespace="preference",
                subject="response.detail",
            )
        ),
        state=state,
        controls=ProfileControls(
            sync_mode=SyncMode.SYNCABLE,
            agent_visibility=AgentVisibility.AGENT_VISIBLE,
        ),
        provenance=ProfileProvenance(
            source=ProvenanceSource.MANUAL,
            actor=ActorType.USER,
            reason_code="settings_edit",
        ),
        version_id=version_id,
        parent_version_id=parent_version_id,
        created_at=NOW,
        updated_at=NOW,
    )


def proposal(
    profile_id: str = "profile-a",
    *,
    proposal_id: str = "proposal-a",
    value: str = "structured answers",
) -> ProfileProposal:
    return ProfileProposal(
        proposal_id=proposal_id,
        profile_id=profile_id,
        scope_id=f"{profile_id}-global",
        operation=ProposalOperation.CREATE,
        target_record_id=None,
        base_version_id=None,
        proposed_record=preference_record(
            profile_id,
            record_id=f"{proposal_id}-record",
            version_id=f"{proposal_id}-record-v1",
            value=value,
        ),
        provenance=ProfileProvenance(
            source=ProvenanceSource.AGENT,
            actor=ActorType.AGENT,
            reason_code="conversation_learning",
        ),
        confidence=0.8,
        created_at=NOW,
        expires_at=NOW + timedelta(days=90),
    )
