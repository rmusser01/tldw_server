from __future__ import annotations

from collections import defaultdict
from datetime import UTC, datetime, timedelta

import pytest
from tldw_profile_core import (
    AgentVisibility,
    GoalPayload,
    PreferencePayload,
    ProfileControls,
    ProfileProposal,
    ProfileProvenance,
    ProposalOperation,
    ProposalState,
    RecordState,
    SemanticKey,
    SyncMode,
)
from tldw_profile_core.models import ActorType, ProvenanceSource

from tldw_Server_API.app.core.DB_Management.Personalization_DB import PersonalizationDB
from tldw_Server_API.app.core.Personalization import personal_context_repository
from tldw_Server_API.app.core.Personalization.personal_context_repository import (
    PersonalContextRepository,
)
from tldw_Server_API.app.core.Personalization.personal_context_repository_models import (
    ProfileQuotaExceededError,
    ProfileSemanticKeyCollisionError,
    ProfileUnsupportedSchemaError,
)
from tldw_Server_API.app.core.Personalization.personal_context_service import (
    PersonalContextService,
    ProfileConflictError,
    ProfileKeyCollisionError,
    ProfileOperationalState,
    ProfileUnsupportedOperationError,
    RecordMutation,
)
from tldw_Server_API.tests.Personalization.personal_context_test_support import (
    encoded_master_key,
)

pytestmark = pytest.mark.unit

NOW = datetime(2026, 8, 30, 18, 0, tzinfo=UTC)


def _ids():
    counters: defaultdict[str, int] = defaultdict(int)

    def issue(label: str) -> str:
        counters[label] += 1
        return f"{label}-{counters[label]}"

    return issue


@pytest.fixture()
def service(tmp_path, monkeypatch) -> PersonalContextService:
    monkeypatch.setenv("TLDW_PERSONAL_CONTEXT_MASTER_KEY", encoded_master_key())
    repository = PersonalContextRepository(PersonalizationDB(str(tmp_path / "personalization.db")))
    return PersonalContextService(
        repository,
        clock=lambda: NOW,
        id_factory=_ids(),
        workspace_access=lambda workspace_id: workspace_id == "workspace-owned",
    )


def _controls(*, visible: bool = True) -> ProfileControls:
    return ProfileControls(
        sync_mode=SyncMode.SYNCABLE,
        agent_visibility=(AgentVisibility.AGENT_VISIBLE if visible else AgentVisibility.USER_ONLY),
    )


def _payload(value: str = "concise") -> PreferencePayload:
    return PreferencePayload(
        subject="response.detail",
        polarity="like",
        value=value,
    )


def _semantic_key() -> SemanticKey:
    return SemanticKey(namespace="preference", subject="response.detail")


def _create_ready_profile(service: PersonalContextService):
    manifest = service.create_profile()
    scope = service.list_scopes()[0]
    return manifest, scope


def test_profile_creation_is_absent_then_disabled_until_runtime_enabled(
    service: PersonalContextService,
) -> None:
    assert service.status().state is ProfileOperationalState.ABSENT

    manifest, _scope = _create_ready_profile(service)

    assert manifest.revision == 0
    assert service.status().state is ProfileOperationalState.DISABLED
    policy = service.set_runtime_enabled(True, expected_version_id=None)
    assert policy.enabled is True
    assert service.status().state is ProfileOperationalState.AVAILABLE


def test_status_fails_closed_for_residual_state_without_manifest(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("TLDW_PERSONAL_CONTEXT_MASTER_KEY", encoded_master_key())
    repository = PersonalContextRepository(PersonalizationDB(str(tmp_path / "residual-state.db")))
    service = PersonalContextService(repository, clock=lambda: NOW, id_factory=_ids())
    manifest = service.create_profile()
    with repository.database.transaction(immediate=True) as connection:
        connection.execute(
            "DELETE FROM personal_context_object_heads WHERE profile_id = ? AND object_type = 'manifest'",
            (manifest.profile_id,),
        )
        connection.execute(
            "DELETE FROM personal_context_object_versions WHERE profile_id = ? AND object_type = 'manifest'",
            (manifest.profile_id,),
        )

    assert service.status().state is ProfileOperationalState.LOCKED


def test_status_distinguishes_unsupported_schema(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("TLDW_PERSONAL_CONTEXT_MASTER_KEY", encoded_master_key())
    repository = PersonalContextRepository(PersonalizationDB(str(tmp_path / "unsupported-schema.db")))
    service = PersonalContextService(repository, clock=lambda: NOW, id_factory=_ids())
    service.create_profile()

    def unsupported_manifest(_profile_id):
        raise ProfileUnsupportedSchemaError("unsupported")

    monkeypatch.setattr(repository, "get_manifest", unsupported_manifest)

    assert service.status().state is ProfileOperationalState.UNSUPPORTED


def test_workspace_scope_requires_authenticated_workspace_ownership(
    service: PersonalContextService,
) -> None:
    _create_ready_profile(service)

    with pytest.raises(KeyError):
        service.create_workspace_scope("workspace-someone-elses", "Private")

    scope = service.create_workspace_scope("workspace-owned", "Project Atlas")
    assert scope.kind.value == "workspace"
    assert service.workspace_id_for_scope(scope.scope_id) == "workspace-owned"


def test_workspace_scope_and_private_mapping_roll_back_together(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("TLDW_PERSONAL_CONTEXT_MASTER_KEY", encoded_master_key())
    repository = PersonalContextRepository(PersonalizationDB(str(tmp_path / "atomic-workspace.db")))
    service = PersonalContextService(
        repository,
        clock=lambda: NOW,
        id_factory=_ids(),
        workspace_access=lambda _workspace_id: True,
    )
    original = service.create_profile()

    def fail_runtime_head(*_args, **_kwargs):
        raise RuntimeError("simulated runtime mapping failure")

    monkeypatch.setattr(repository, "_set_runtime_head", fail_runtime_head)

    with pytest.raises(RuntimeError, match="simulated runtime mapping failure"):
        service.create_workspace_scope("workspace-owned", "Project Atlas")

    assert service.list_scopes()[0].kind.value == "global"
    assert len(service.list_scopes()) == 1
    assert service.get_manifest() == original


def test_record_mutations_are_optimistic_unique_and_bounded(
    service: PersonalContextService,
) -> None:
    _manifest, scope = _create_ready_profile(service)
    first = service.create_manual_record(
        scope_id=scope.scope_id,
        payload=_payload(),
        semantic_key=_semantic_key(),
        controls=_controls(visible=False),
    )

    assert first.controls.agent_visibility is AgentVisibility.USER_ONLY
    with pytest.raises(ProfileKeyCollisionError):
        service.create_manual_record(
            scope_id=scope.scope_id,
            payload=_payload("detailed"),
            semantic_key=_semantic_key(),
            controls=_controls(),
        )

    updated = service.update_record(
        first.record_id,
        RecordMutation(payload=_payload("structured")),
        expected_version_id=first.version_id,
    )
    assert updated.parent_version_id == first.version_id
    assert updated.payload.value == "structured"

    with pytest.raises(ProfileConflictError):
        service.update_record(
            first.record_id,
            RecordMutation(payload=_payload("stale")),
            expected_version_id=first.version_id,
        )

    archived = service.archive_record(
        first.record_id,
        expected_version_id=updated.version_id,
    )
    assert archived.state is RecordState.ARCHIVED
    with pytest.raises(ValueError, match="only active records can be updated"):
        service.update_record(
            first.record_id,
            RecordMutation(payload=_payload("invalid archived update")),
            expected_version_id=archived.version_id,
        )
    with pytest.raises(ValueError, match="only active records can be archived"):
        service.archive_record(
            first.record_id,
            expected_version_id=archived.version_id,
        )
    restored = service.restore_record(
        first.record_id,
        expected_version_id=archived.version_id,
    )
    with pytest.raises(ValueError, match="only archived records can be restored"):
        service.restore_record(
            first.record_id,
            expected_version_id=restored.version_id,
        )
    with pytest.raises(ValueError, match="record kind is immutable"):
        service.update_record(
            first.record_id,
            RecordMutation(
                payload=GoalPayload(
                    subject="response.detail",
                    outcome="structured",
                )
            ),
            expected_version_id=restored.version_id,
        )
    deleted = service.delete_record(
        first.record_id,
        expected_version_id=restored.version_id,
    )
    assert deleted.state is RecordState.DELETED
    assert deleted.payload is None
    assert deleted.semantic_key is None
    assert service.search_records("structured", limit=5) == ()

    with pytest.raises(ValueError):
        service.search_records("structured", limit=21)


def test_repository_enforces_semantic_collision_inside_write_transaction(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("TLDW_PERSONAL_CONTEXT_MASTER_KEY", encoded_master_key())
    repository = PersonalContextRepository(PersonalizationDB(str(tmp_path / "semantic-race.db")))
    service = PersonalContextService(
        repository,
        clock=lambda: NOW,
        id_factory=_ids(),
        workspace_access=lambda _workspace_id: True,
    )
    _manifest, scope = _create_ready_profile(service)
    first = service.build_manual_record(
        scope_id=scope.scope_id,
        payload=_payload("first"),
        semantic_key=_semantic_key(),
        controls=_controls(),
    )
    racing = service.build_manual_record(
        scope_id=scope.scope_id,
        payload=_payload("racing"),
        semantic_key=_semantic_key(),
        controls=_controls(),
    )
    service.create_record(first)
    current = service.get_manifest()
    next_manifest = current.model_copy(
        update={
            "revision": current.revision + 1,
            "current_version_id": "manifest-racing",
        }
    )

    with pytest.raises(ProfileSemanticKeyCollisionError):
        repository.commit_record_and_manifest(
            racing,
            next_manifest,
            expected_record_version=None,
            expected_manifest_version=current.current_version_id,
        )

    assert repository.get_record(current.profile_id, racing.record_id) is None


def test_proposal_review_applies_content_or_shreds_it(
    service: PersonalContextService,
) -> None:
    manifest, scope = _create_ready_profile(service)
    now = NOW
    proposed = service.build_manual_record(
        scope_id=scope.scope_id,
        payload=_payload("tables"),
        semantic_key=SemanticKey(namespace="preference", subject="answer.layout"),
        controls=_controls(),
    )
    proposal = ProfileProposal(
        proposal_id="proposal-accept",
        profile_id=manifest.profile_id,
        scope_id=scope.scope_id,
        operation=ProposalOperation.CREATE,
        target_record_id=None,
        base_version_id=None,
        proposed_record=proposed,
        provenance=ProfileProvenance(
            source=ProvenanceSource.AGENT,
            actor=ActorType.AGENT,
            reason_code="conversation_learning",
        ),
        confidence=0.8,
        created_at=now,
        expires_at=now + timedelta(days=90),
    )
    service.create_proposal(proposal)

    receipt, accepted = service.review_proposal("proposal-accept", action="accept")
    assert receipt.state.value == "accepted"
    assert receipt.proposed_record is None
    assert accepted is not None
    assert service.get_record(accepted.record_id) is not None

    rejected_proposal = proposal.model_copy(
        update={
            "proposal_id": "proposal-reject",
            "proposed_record": proposed.model_copy(
                update={
                    "record_id": "record-reject",
                    "version_id": "record-reject-v1",
                }
            ),
        }
    )
    service.create_proposal(rejected_proposal)
    rejected, record = service.review_proposal("proposal-reject", action="reject")
    assert rejected.state.value == "rejected"
    assert rejected.proposed_record is None
    assert record is None


def test_proposal_update_cannot_move_record_to_another_scope(
    service: PersonalContextService,
) -> None:
    manifest, global_scope = _create_ready_profile(service)
    workspace_scope = service.create_workspace_scope("workspace-owned", "Project Atlas")
    current = service.create_manual_record(
        scope_id=global_scope.scope_id,
        payload=_payload("concise"),
        semantic_key=None,
        controls=_controls(),
    )
    moved = current.model_copy(
        update={
            "scope_id": workspace_scope.scope_id,
            "payload": _payload("detailed"),
            "version_id": "moved-record-version",
            "parent_version_id": current.version_id,
            "updated_at": NOW + timedelta(seconds=1),
        }
    )
    proposal = ProfileProposal(
        proposal_id="proposal-move-scope",
        profile_id=manifest.profile_id,
        scope_id=workspace_scope.scope_id,
        operation=ProposalOperation.UPDATE,
        target_record_id=current.record_id,
        base_version_id=current.version_id,
        proposed_record=moved,
        provenance=ProfileProvenance(
            source=ProvenanceSource.AGENT,
            actor=ActorType.AGENT,
            reason_code="conversation_learning",
        ),
        confidence=0.8,
        created_at=NOW,
        expires_at=NOW + timedelta(days=90),
    )
    service.create_proposal(proposal)

    with pytest.raises(ProfileConflictError, match="target scope"):
        service.review_proposal(proposal.proposal_id, action="accept")

    assert service.get_record(current.record_id) == current


def test_duplicate_proposal_id_is_a_typed_conflict(
    service: PersonalContextService,
) -> None:
    manifest, scope = _create_ready_profile(service)
    proposed = service.build_manual_record(
        scope_id=scope.scope_id,
        payload=_payload("tables"),
        semantic_key=None,
        controls=_controls(),
    )
    proposal = ProfileProposal(
        proposal_id="proposal-duplicate",
        profile_id=manifest.profile_id,
        scope_id=scope.scope_id,
        operation=ProposalOperation.CREATE,
        target_record_id=None,
        base_version_id=None,
        proposed_record=proposed,
        provenance=ProfileProvenance(
            source=ProvenanceSource.AGENT,
            actor=ActorType.AGENT,
            reason_code="conversation_learning",
        ),
        confidence=0.8,
        created_at=NOW,
        expires_at=NOW + timedelta(days=90),
    )
    service.create_proposal(proposal)

    with pytest.raises(ProfileConflictError, match="proposal changed"):
        service.create_proposal(proposal)


def test_pending_proposal_quota_is_enforced_inside_repository_transaction(
    service: PersonalContextService,
    monkeypatch,
) -> None:
    manifest, scope = _create_ready_profile(service)
    monkeypatch.setattr(personal_context_repository, "_MAX_PENDING_PROPOSALS", 1)
    proposed = service.build_manual_record(
        scope_id=scope.scope_id,
        payload=_payload("first proposal"),
        semantic_key=None,
        controls=_controls(),
    )
    first = ProfileProposal(
        proposal_id="proposal-quota-1",
        profile_id=manifest.profile_id,
        scope_id=scope.scope_id,
        operation=ProposalOperation.CREATE,
        target_record_id=None,
        base_version_id=None,
        proposed_record=proposed,
        provenance=ProfileProvenance(
            source=ProvenanceSource.AGENT,
            actor=ActorType.AGENT,
            reason_code="conversation_learning",
        ),
        confidence=0.8,
        created_at=NOW,
        expires_at=NOW + timedelta(days=90),
    )
    service.create_proposal(first)
    second = first.model_copy(
        update={
            "proposal_id": "proposal-quota-2",
            "proposed_record": proposed.model_copy(
                update={
                    "record_id": "record-quota-2",
                    "version_id": "record-quota-v2",
                }
            ),
        }
    )

    with pytest.raises(ProfileQuotaExceededError):
        service.create_proposal(second)

    assert service.list_proposals() == (first,)


def test_expired_proposal_body_is_replaced_by_content_free_receipt(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("TLDW_PERSONAL_CONTEXT_MASTER_KEY", encoded_master_key())
    repository = PersonalContextRepository(PersonalizationDB(str(tmp_path / "proposal-expiry.db")))
    current_time = [NOW]
    service = PersonalContextService(
        repository,
        clock=lambda: current_time[0],
        id_factory=_ids(),
    )
    manifest, scope = _create_ready_profile(service)
    proposed = service.build_manual_record(
        scope_id=scope.scope_id,
        payload=_payload("temporary proposal"),
        semantic_key=None,
        controls=_controls(),
    )
    proposal = ProfileProposal(
        proposal_id="proposal-expiring",
        profile_id=manifest.profile_id,
        scope_id=scope.scope_id,
        operation=ProposalOperation.CREATE,
        target_record_id=None,
        base_version_id=None,
        proposed_record=proposed,
        provenance=ProfileProvenance(
            source=ProvenanceSource.AGENT,
            actor=ActorType.AGENT,
            reason_code="conversation_learning",
        ),
        confidence=0.8,
        created_at=NOW,
        expires_at=NOW + timedelta(days=90),
    )
    service.create_proposal(proposal)
    current_time[0] = NOW + timedelta(days=91)
    replacement_record = service.build_manual_record(
        scope_id=scope.scope_id,
        payload=_payload("replacement proposal"),
        semantic_key=None,
        controls=_controls(),
    )
    replacement = ProfileProposal(
        proposal_id="proposal-after-expiry",
        profile_id=manifest.profile_id,
        scope_id=scope.scope_id,
        operation=ProposalOperation.CREATE,
        target_record_id=None,
        base_version_id=None,
        proposed_record=replacement_record,
        provenance=ProfileProvenance(
            source=ProvenanceSource.AGENT,
            actor=ActorType.AGENT,
            reason_code="conversation_learning",
        ),
        confidence=0.8,
        created_at=current_time[0],
        expires_at=current_time[0] + timedelta(days=90),
    )
    service.create_proposal(replacement)

    receipt = repository.get_proposal(manifest.profile_id, proposal.proposal_id)
    assert receipt is not None
    assert receipt.state is ProposalState.EXPIRED
    assert receipt.proposed_record is None
    assert service.list_proposals() == (replacement,)
    assert (
        repository.proposal_version_count(
            manifest.profile_id,
            proposal.proposal_id,
        )
        == 1
    )


def test_expired_proposal_rejection_records_expired_receipt(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("TLDW_PERSONAL_CONTEXT_MASTER_KEY", encoded_master_key())
    repository = PersonalContextRepository(PersonalizationDB(str(tmp_path / "expired-reject.db")))
    current_time = [NOW]
    service = PersonalContextService(
        repository,
        clock=lambda: current_time[0],
        id_factory=_ids(),
    )
    manifest, scope = _create_ready_profile(service)
    proposed = service.build_manual_record(
        scope_id=scope.scope_id,
        payload=_payload("temporary"),
        semantic_key=None,
        controls=_controls(),
    )
    pending = ProfileProposal(
        proposal_id="proposal-expired-reject",
        profile_id=manifest.profile_id,
        scope_id=scope.scope_id,
        operation=ProposalOperation.CREATE,
        target_record_id=None,
        base_version_id=None,
        proposed_record=proposed,
        provenance=ProfileProvenance(
            source=ProvenanceSource.AGENT,
            actor=ActorType.AGENT,
            reason_code="conversation_learning",
        ),
        confidence=0.8,
        created_at=NOW,
        expires_at=NOW + timedelta(days=90),
    )
    service.create_proposal(pending)
    current_time[0] = NOW + timedelta(days=91)

    with pytest.raises(ValueError, match="proposal has expired"):
        service.review_proposal(pending.proposal_id, action="reject")

    receipt = repository.get_proposal(manifest.profile_id, pending.proposal_id)
    assert receipt is not None
    assert receipt.state is ProposalState.EXPIRED
    assert receipt.proposed_record is None


def test_acceptance_transaction_rechecks_proposal_expiry(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("TLDW_PERSONAL_CONTEXT_MASTER_KEY", encoded_master_key())
    repository = PersonalContextRepository(PersonalizationDB(str(tmp_path / "expiry-race.db")))
    repository_time = [NOW]

    class RepositoryDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return repository_time[0]

    monkeypatch.setattr(personal_context_repository, "datetime", RepositoryDateTime)
    service = PersonalContextService(
        repository,
        clock=lambda: NOW,
        id_factory=_ids(),
    )
    manifest, scope = _create_ready_profile(service)
    proposed = service.build_manual_record(
        scope_id=scope.scope_id,
        payload=_payload("too late"),
        semantic_key=None,
        controls=_controls(),
    )
    pending = ProfileProposal(
        proposal_id="proposal-expiry-race",
        profile_id=manifest.profile_id,
        scope_id=scope.scope_id,
        operation=ProposalOperation.CREATE,
        target_record_id=None,
        base_version_id=None,
        proposed_record=proposed,
        provenance=ProfileProvenance(
            source=ProvenanceSource.AGENT,
            actor=ActorType.AGENT,
            reason_code="conversation_learning",
        ),
        confidence=0.8,
        created_at=NOW,
        expires_at=NOW + timedelta(days=90),
    )
    service.create_proposal(pending)
    repository_time[0] = NOW + timedelta(days=91)

    with pytest.raises(ValueError, match="proposal has expired"):
        service.review_proposal(pending.proposal_id, action="accept")

    assert repository.get_record(manifest.profile_id, proposed.record_id) is None
    receipt = repository.get_proposal(manifest.profile_id, pending.proposal_id)
    assert receipt is not None
    assert receipt.state is ProposalState.EXPIRED
    assert receipt.proposed_record is None
    assert (
        repository.proposal_version_count(
            manifest.profile_id,
            pending.proposal_id,
        )
        == 1
    )


def test_exports_and_global_purge_are_explicit_server_operations(
    service: PersonalContextService,
) -> None:
    manifest, scope = _create_ready_profile(service)
    service.create_manual_record(
        scope_id=scope.scope_id,
        payload=_payload("markdown"),
        semantic_key=None,
        controls=_controls(visible=False),
    )

    exported = service.export_plaintext(
        confirmation="EXPORT PLAINTEXT",
        scope_ids=(scope.scope_id,),
    )
    assert exported["manifest"]["profile_id"] == manifest.profile_id
    assert exported["records"][0]["payload"]["value"] == "markdown"
    assert "runtime" not in exported

    recovery = service.export_recovery(
        confirmation="EXPORT RECOVERY",
        passphrase="correct horse battery staple",
    )
    assert recovery["algorithm"] == "scrypt-aes-256-gcm"
    assert "ciphertext" in recovery

    with pytest.raises(ProfileUnsupportedOperationError):
        service.purge_profile(
            mode="local_copy",
            confirmation="DELETE EVERYWHERE",
            expected_purge_generation=0,
        )

    barrier = service.purge_profile(
        mode="everywhere",
        confirmation="DELETE EVERYWHERE",
        expected_purge_generation=0,
    )
    assert barrier.purge_generation == 1
    assert service.status().state is ProfileOperationalState.PURGE_PENDING
    assert service.list_records(include_archived=True) == ()
    with pytest.raises(
        ProfileUnsupportedOperationError,
        match="profile_purge_pending",
    ):
        service.create_workspace_scope("workspace-owned", "Resurrection attempt")
    with pytest.raises(
        ProfileUnsupportedOperationError,
        match="profile_purge_pending",
    ):
        service.set_runtime_enabled(True, expected_version_id=None)


def test_repository_caps_encrypted_heads_and_pages(
    service: PersonalContextService,
    monkeypatch,
) -> None:
    manifest, scope = _create_ready_profile(service)
    repository = service._repository
    monkeypatch.setattr(personal_context_repository, "_MAX_RECORD_HEADS", 1)
    service.create_manual_record(
        scope_id=scope.scope_id,
        payload=_payload("first"),
        semantic_key=None,
        controls=_controls(),
    )

    with pytest.raises(ProfileQuotaExceededError, match="record quota"):
        service.create_manual_record(
            scope_id=scope.scope_id,
            payload=_payload("second"),
            semantic_key=None,
            controls=_controls(),
        )
    with pytest.raises(ValueError, match="page is out of bounds"):
        repository.list_records(manifest.profile_id, limit=1_001)

    monkeypatch.setattr(personal_context_repository, "_MAX_SCOPE_HEADS", 1)
    with pytest.raises(ProfileQuotaExceededError, match="scope quota"):
        service.create_workspace_scope("workspace-owned", "Too many scopes")
